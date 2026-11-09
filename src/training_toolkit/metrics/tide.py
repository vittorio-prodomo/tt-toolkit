"""TIDE (Toolbox for Identifying Detection Errors) wrapper.

Accumulates predictions + targets across validation batches and runs
``tidecv.TIDE.evaluate`` on compute(). Adds per-class breakdown that tidecv
does not ship out of the box.
"""

import contextlib
import io
import logging
from typing import Any

import numpy as np
import torch
from pycocotools import mask as mask_utils
from tidecv import TIDE, Data
from tidecv.errors.qualifiers import Qualifier

logger = logging.getLogger(__name__)


class TideAnalyzer:
    """Accumulates val predictions/targets, runs TIDE on compute().

    Mirrors :class:`SegmentationMAP`'s update/compute/reset interface.

    Inputs are filtered to the community convention before analysis:
    ``score_thresh=0.05`` (strict >) + ``max_dets=100`` top-K per image —
    matching the COCO results-file cap that published TIDE numbers implicitly
    assume (see :meth:`_filter_pred`; each knob independently disable-able).

    Note on scale: tidecv reports AP and dAP on a **0-100** scale (percentage
    points), not the 0-1 scale TorchMetrics uses. We preserve tidecv's native
    scale so the dict values match the printed ``tide.summarize()`` text. When
    logging to W&B / MLflow, expect ``val/tide_dap/cls`` to be e.g. ``3.40``
    (i.e. "fixing all Cls errors recovers 3.4 mAP percentage points").
    """

    def __init__(
        self,
        class_names: dict[int, str],
        iou_type: str = "mask",
        score_thresh: float = 0.05,
        max_dets: int | None = 100,
    ) -> None:
        self.class_names = dict(class_names)
        self.mode = "mask" if iou_type in ("mask", "segm") else "bbox"
        self.score_thresh = score_thresh
        self.max_dets = max_dets
        self._gt_buf: list[tuple[int, int, list[float], dict]] = []
        self._pred_buf: list[tuple[int, int, float, list[float], dict]] = []

    def _filter_pred(self, pred: dict) -> dict:
        """Community-standard input filter: score floor (strict >, the
        detectron2/mmdet convention) then top-``max_dets`` by score per image
        (the COCO results-file cap). Published TIDE numbers implicitly run on
        such filtered files; an unfiltered DETR-family decode (~200-300
        structural low-score queries/img) drowns the FalsePos/Bkg
        counterfactuals in junk the ranking-based metric never rewards.
        ``score_thresh=0`` and ``max_dets=None`` disable each independently.
        GT is never filtered."""
        scores = pred["scores"]
        keep = scores > self.score_thresh if self.score_thresh else torch.ones(
            len(scores), dtype=torch.bool, device=scores.device
        )
        n_keep = int(keep.sum())
        if self.max_dets is not None and n_keep > self.max_dets:
            kept_idx = torch.nonzero(keep).squeeze(1)
            order = scores[kept_idx].argsort(descending=True)
            kept_idx = kept_idx[order[: self.max_dets]]
            keep = torch.zeros_like(keep)
            keep[kept_idx] = True
        if bool(keep.all()):
            return pred
        return {k: v[keep] for k, v in pred.items()}

    def update(
        self,
        preds: list[dict],
        targets: list[dict],
        batch_idx: int,
    ) -> None:
        """Accumulate one validation batch.

        Each pred/target dict carries ``boxes`` (xyxy float), ``labels``,
        ``masks`` (bool [N,H,W]); preds also carry ``scores``. The pair
        ``(batch_idx, sample_idx)`` generates a per-image unique id.
        Predictions pass through :meth:`_filter_pred` first.
        """
        for sample_idx, (pred, target) in enumerate(zip(preds, targets)):
            image_id = batch_idx * 10_000 + sample_idx
            self._store_gts(image_id, target)
            self._store_preds(image_id, self._filter_pred(pred))

    def _store_gts(self, image_id: int, target: dict) -> None:
        boxes = target["boxes"]
        labels = target["labels"]
        masks = target["masks"]
        for i in range(boxes.shape[0]):
            bbox = self._xyxy_to_xywh(boxes[i])
            rle = self._encode_mask(masks[i])
            self._gt_buf.append((image_id, int(labels[i].item()), bbox, rle))

    def _store_preds(self, image_id: int, pred: dict) -> None:
        boxes = pred["boxes"]
        labels = pred["labels"]
        masks = pred["masks"]
        scores = pred["scores"]
        for i in range(boxes.shape[0]):
            bbox = self._xyxy_to_xywh(boxes[i])
            rle = self._encode_mask(masks[i])
            self._pred_buf.append((
                image_id,
                int(labels[i].item()),
                float(scores[i].item()),
                bbox,
                rle,
            ))

    @staticmethod
    def _xyxy_to_xywh(box: torch.Tensor) -> list[float]:
        x1, y1, x2, y2 = box.tolist()
        return [x1, y1, x2 - x1, y2 - y1]

    @staticmethod
    def _encode_mask(mask: torch.Tensor) -> dict:
        mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        return mask_utils.encode(np.asfortranarray(mask_np.astype(np.uint8)))

    def compute(self) -> dict[str, Any] | None:
        """Run TIDE and return aggregate + per-class breakdown + summary text.

        Returns ``None`` when zero predictions were accumulated (a degenerate
        or fresh-init model): tidecv's ``get_mAP`` divides by zero on a run
        with no detections, and there is nothing to analyze anyway.
        """
        if not self._pred_buf:
            return None
        gt = Data("gt")
        preds_data = Data("preds")

        for cls_id, cls_name in self.class_names.items():
            gt.add_class(cls_id, cls_name)
            preds_data.add_class(cls_id, cls_name)

        for image_id, cls_id, bbox, rle in self._gt_buf:
            gt.add_ground_truth(image_id, cls_id, box=bbox, mask=rle)

        for image_id, cls_id, score, bbox, rle in self._pred_buf:
            preds_data.add_detection(image_id, cls_id, score, box=bbox, mask=rle)

        tide = TIDE(mode=self.mode)
        run_name = "tide_run"
        tide.evaluate(gt, preds_data, name=run_name)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            tide.summarize()
        summary_text = buf.getvalue()

        main_errors = tide.get_main_errors()[run_name]
        special_errors = tide.get_special_errors()[run_name]
        aggregate = {**main_errors, **special_errors}

        per_class = self._extract_per_class(tide, run_name)

        return {
            "aggregate": aggregate,
            "per_class": per_class,
            "summary_text": summary_text,
            "tide": tide,
        }

    def _extract_per_class(
        self, tide: TIDE, run_name: str,
    ) -> dict[str, dict[str, float]]:
        run = tide.runs[run_name]
        qual = Qualifier("", None)
        baseline_aps = {
            cls_id: obj.get_ap() for cls_id, obj in run.ap_data.objs.items()
        }

        per_class: dict[str, dict[str, float]] = {}
        for cls_id in baseline_aps:
            name = self.class_names.get(cls_id, f"class_{cls_id}")
            per_class[name] = {"AP": baseline_aps[cls_id]}

        for error_type in TIDE._error_types:
            fixed = run.fix_errors(
                qual._make_error_func(error_type), ap_data=run.ap_data,
            )
            for cls_id, baseline_ap in baseline_aps.items():
                name = self.class_names.get(cls_id, f"class_{cls_id}")
                fixed_ap = (
                    fixed.objs[cls_id].get_ap() if cls_id in fixed.objs else baseline_ap
                )
                delta = max(fixed_ap - baseline_ap, 0)
                per_class[name][error_type.short_name] = delta

        return per_class

    def reset(self) -> None:
        self._gt_buf.clear()
        self._pred_buf.clear()

    @staticmethod
    def format_per_class_table(result: dict[str, Any]) -> str:
        """Render a fixed-width per-class breakdown table.

        tidecv's built-in ``summarize()`` prints aggregate dAP only — this
        adds the per-class slice we extract in ``_extract_per_class``.
        Columns: Class | AP | Cls | Loc | Both | Dupe | Bkg | Miss.
        """
        per_class = result["per_class"]
        if not per_class:
            return "Per-class TIDE breakdown: (no data)\n"

        err_cols = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
        name_w = max(len("Class"), max(len(n) for n in per_class))

        lines = []
        header = f"{'Class':<{name_w}}  {'AP':>6}  " + "  ".join(
            f"{c:>5}" for c in err_cols
        )
        lines.append(header)
        lines.append("-" * len(header))
        for name in sorted(per_class):
            entry = per_class[name]
            row = f"{name:<{name_w}}  {entry['AP']:>6.2f}  " + "  ".join(
                f"{entry[c]:>5.2f}" for c in err_cols
            )
            lines.append(row)
        return "\n".join(lines) + "\n"
