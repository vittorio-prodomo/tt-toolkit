"""TorchMetrics mAP wrapper for instance segmentation."""

import logging
from typing import Any

import torch
from training_toolkit.metrics.ddp_map import DDPSegmentationMAP

logger = logging.getLogger(__name__)


class SegmentationMAP:
    """Wrapper around TorchMetrics MeanAveragePrecision for instance segmentation.

    Configures segmentation IoU, per-class breakdown, and provides a
    :meth:`format_results` helper that flattens per-class metrics into
    loggable keys like ``map_crack``, ``map_spall``.

    Args:
        class_names: Mapping ``{class_id: name}`` for per-class logging.
    """

    def __init__(
        self,
        class_names: dict[int, str] | None = None,
        iou_type: str = "segm",
    ) -> None:
        self.class_names = class_names or {}
        # DDPSegmentationMAP: drop-in MeanAveragePrecision subclass. Under multi-GPU
        # DDP + segm it computes EXACT global mAP by gathering each rank's compact
        # RLE state to rank 0 and running faster-coco-eval once (Pattern A) — instead
        # of all_gather'ing ~250 MB of masks per epoch to every rank. Single-GPU /
        # non-segm fall through to stock torchmetrics. Escape hatch: TT_DDP_EXACT_MAP=0.
        self.metric = DDPSegmentationMAP(
            iou_type=iou_type,
            class_metrics=True,
            backend="faster_coco_eval",
            extended_summary=True,
        )

    def update(self, preds: list[dict], targets: list[dict]) -> None:
        self.metric.update(preds, targets)

    def compute(self) -> dict[str, Any]:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()

    def format_results(self, raw: dict[str, Any]) -> dict[str, float]:
        """Flatten raw TorchMetrics output into a clean dict for logging.

        Converts per-class tensor arrays into individual keys like
        ``map_crack``, ``map_50_spall``, etc. Skips classes with -1 (no data).
        """
        formatted = {}

        # Global metrics
        scalar_keys = (
            "map", "map_50", "map_75", "map_small", "map_medium", "map_large",
            "mar_1", "mar_10", "mar_100",
            "mar_small", "mar_medium", "mar_large",
        )
        for key in scalar_keys:
            val = raw.get(key)
            if val is not None and isinstance(val, torch.Tensor):
                formatted[key] = val.item()

        # Per-class metrics
        map_per_class = raw.get("map_per_class")
        classes = raw.get("classes")
        if map_per_class is not None and len(self.class_names) > 0:
            # Ensure tensors are 1-d (scalar when only one class is present)
            if map_per_class.dim() == 0:
                map_per_class = map_per_class.unsqueeze(0)
            if classes is not None and classes.dim() == 0:
                classes = classes.unsqueeze(0)

            # Use the classes tensor to map indices to class IDs
            for i in range(len(map_per_class)):
                val = map_per_class[i].item()
                if val < 0:  # -1 means no data
                    continue
                class_id = self._resolve_class_id(classes, i)
                if class_id is not None and class_id in self.class_names:
                    name = self.class_names[class_id]
                    formatted[f"map_{name}"] = val

        # Per-class AP at fixed IoU (0.50, 0.75) from the extended summary.
        # Named ap50_/ap75_, NOT map_50_<class>: PerClassBestCheckpointCallback
        # tracks every "val/map_*" key and would save a checkpoint per class per IoU.
        for prefix, iou in (("ap50", 0.50), ("ap75", 0.75)):
            for name, ap in self._ap_per_class_at_iou(raw, iou).items():
                formatted[f"{prefix}_{name}"] = ap

        # Best F1 per class @ IoU=0.50
        f1_data = self._best_f1_per_class(raw)
        if f1_data:
            for entry in f1_data:
                formatted[f"f1_best_{entry['name']}"] = entry["f1"]
            formatted["f1_best"] = sum(e["f1"] for e in f1_data) / len(f1_data)

        return formatted

    def format_verbose(self, raw: dict[str, Any]) -> str:
        """Return a multi-line COCO-style summary string."""
        f = self.format_results(raw)
        _v = lambda k: f"{f[k]:.4f}" if f.get(k, -1) >= 0 else "  N/A "  # noqa: E731

        lines = [
            f"  mAP @[.50:.95] = {_v('map')}   "
            f"mAP @.50 = {_v('map_50')}   "
            f"mAP @.75 = {_v('map_75')}",
            f"  mAP  small     = {_v('map_small')}   "
            f"medium   = {_v('map_medium')}   "
            f"large    = {_v('map_large')}",
            f"  mAR  @1        = {_v('mar_1')}   "
            f"@10      = {_v('mar_10')}   "
            f"@100     = {_v('mar_100')}",
            f"  mAR  small     = {_v('mar_small')}   "
            f"medium   = {_v('mar_medium')}   "
            f"large    = {_v('mar_large')}",
        ]

        # Per-class: mAP + F1 in a single table
        per_class = {k: v for k, v in f.items() if k.startswith("map_") and k not in (
            "map_50", "map_75", "map_small", "map_medium", "map_large",
        )}
        f1_data = self._best_f1_per_class(raw)
        f1_by_name = {e["name"]: e for e in f1_data} if f1_data else {}

        if per_class:
            all_names = sorted(k[4:] for k in per_class)
            max_name_len = max(len(n) for n in all_names + ["Mean"])

            def _c(v: float | None) -> str:
                return f"{v:.4f}" if v is not None and v >= 0 else "  N/A "

            header = f"    {'':>{max_name_len}}   mAP    AP50    AP75     F1   @conf"
            lines.append("  Per-class:")
            lines.append(header)
            for name in all_names:
                row = (f"    {name:<{max_name_len}}  {_c(per_class[f'map_{name}'])}  "
                       f"{_c(f.get(f'ap50_{name}'))}  {_c(f.get(f'ap75_{name}'))}")
                f1_entry = f1_by_name.get(name)
                if f1_entry:
                    row += f"  {f1_entry['f1']:.4f}"
                    if f1_entry.get("threshold") is not None:
                        row += f"  {f1_entry['threshold']:.3f}"
                lines.append(row)
            if f1_data:
                mean_f1 = sum(e["f1"] for e in f1_data) / len(f1_data)
                lines.append(
                    f"    {'Mean':<{max_name_len}}  {_c(f.get('map'))}  "
                    f"{_c(f.get('map_50'))}  {_c(f.get('map_75'))}  {mean_f1:.4f}"
                )

        return "\n".join(lines)

    def extract_pr_curves(
        self,
        raw: dict[str, Any],
        iou_idx: int = 0,
    ) -> dict[str, Any] | None:
        """Extract per-class precision/scores at a single IoU threshold.

        Parameters
        ----------
        raw : dict
            Output of ``self.compute()`` (must include ``extended_summary`` arrays).
        iou_idx : int
            Index into the IoU threshold dimension. 0 = IoU 0.50 (default COCO thresholds).

        Returns
        -------
        dict or None
            ``{"precision": ndarray[R,K], "scores": ndarray[R,K],
            "rec_thresholds": ndarray[R], "class_names": list[str]}``
            or *None* if extended summary arrays are absent.
        """
        import numpy as np

        precision_t = raw.get("precision")
        scores_t = raw.get("scores")
        classes_t = raw.get("classes")
        if precision_t is None or scores_t is None:
            return None

        # precision/scores shape: [T, R, K, A, M]
        # Slice: IoU=iou_idx, area=0 (all), maxDet=-1 (highest)
        precision = precision_t[iou_idx, :, :, 0, -1].cpu().numpy()  # [R, K]
        scores = scores_t[iou_idx, :, :, 0, -1].cpu().numpy()        # [R, K]

        # 101 recall thresholds (0.00, 0.01, ..., 1.00)
        rec_thresholds = np.linspace(0, 1, precision.shape[0])

        # Build ordered class name list matching the K dimension
        class_names_list: list[str] = []
        if classes_t is not None:
            if classes_t.dim() == 0:
                classes_t = classes_t.unsqueeze(0)
            for i in range(len(classes_t)):
                cid = int(classes_t[i].item())
                class_names_list.append(self.class_names.get(cid, str(cid)))
        else:
            for i in range(precision.shape[1]):
                class_names_list.append(self.class_names.get(i, str(i)))

        return {
            "precision": precision,
            "scores": scores,
            "rec_thresholds": rec_thresholds,
            "class_names": class_names_list,
        }

    def _best_f1_per_class(
        self,
        raw: dict[str, Any],
        iou_idx: int = 0,
    ) -> list[dict] | None:
        """Compute best F1 and optimal confidence threshold per class.

        Parameters
        ----------
        raw : dict
            Output of ``self.compute()`` (must include ``extended_summary``).
        iou_idx : int
            IoU threshold index (0 = IoU 0.50).

        Returns
        -------
        list[dict] or None
            Each entry: ``{"name": str, "f1": float, "threshold": float}``.
        """
        precision_t = raw.get("precision")
        scores_t = raw.get("scores")
        if precision_t is None:
            return None

        prec = precision_t[iou_idx, :, :, 0, -1].float().cpu()  # [R=101, K]
        has_data = (prec >= 0).any(dim=0)  # [K]
        prec = prec.clamp(min=0)
        rec = torch.linspace(0, 1, prec.shape[0]).unsqueeze(1)  # [R, 1]
        denom = prec + rec
        f1 = torch.where(denom > 0, 2 * prec * rec / denom, torch.zeros_like(prec))
        best_f1, best_idx = f1.max(dim=0)  # [K]

        sc = scores_t[iou_idx, :, :, 0, -1].float().cpu() if scores_t is not None else None

        classes_t = raw.get("classes")
        if classes_t is not None and classes_t.dim() == 0:
            classes_t = classes_t.unsqueeze(0)

        results = []
        for i in range(len(best_f1)):
            if not has_data[i]:
                continue
            class_id = self._resolve_class_id(classes_t, i)
            if class_id is not None and class_id in self.class_names:
                entry: dict[str, Any] = {
                    "name": self.class_names[class_id],
                    "f1": best_f1[i].item(),
                }
                if sc is not None:
                    entry["threshold"] = sc[best_idx[i], i].item()
                results.append(entry)

        return results if results else None

    def _ap_per_class_at_iou(self, raw: dict[str, Any], iou: float) -> dict[str, float]:
        """COCO AP per class at one IoU threshold (area=all, maxDets=last).

        Same reduction as COCOeval.summarize: the mean over recall points of
        ``precision[t, :, k, 0, -1]`` ignoring -1 entries; a class with no
        valid entry (no GT) is omitted. Needs ``extended_summary=True``.
        """
        precision_t = raw.get("precision")
        if precision_t is None:
            return {}
        thresholds = [float(t) for t in getattr(self.metric, "iou_thresholds", None)
                      or torch.linspace(0.5, 0.95, 10).tolist()]
        matches = [i for i, t in enumerate(thresholds) if abs(t - iou) < 1e-6]
        if not matches:
            return {}
        prec = precision_t[matches[0], :, :, 0, -1].float().cpu()  # [R, K]
        classes_t = raw.get("classes")
        if classes_t is not None and classes_t.dim() == 0:
            classes_t = classes_t.unsqueeze(0)
        out: dict[str, float] = {}
        for i in range(prec.shape[1]):
            valid = prec[:, i][prec[:, i] > -1]
            if valid.numel() == 0:
                continue
            class_id = self._resolve_class_id(classes_t, i)
            if class_id is not None and class_id in self.class_names:
                out[self.class_names[class_id]] = valid.mean().item()
        return out

    def _resolve_class_id(
        self, classes: torch.Tensor | None, index: int,
    ) -> int | None:
        if classes is not None and index < len(classes):
            return int(classes[index].item())
        sorted_ids = sorted(self.class_names.keys())
        return sorted_ids[index] if index < len(sorted_ids) else None

    def to(self, device: torch.device) -> "SegmentationMAP":
        self.metric = self.metric.to(device)
        return self
