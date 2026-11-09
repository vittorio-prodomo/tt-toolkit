"""Abstract base class for model adapters."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from torch import Tensor, nn

from training_toolkit.configs import ModelConfig, TrainConfig, AugConfig

logger = logging.getLogger(__name__)


class ModelAdapter(ABC):
    """Bridge between a specific model library and the generic training loop."""

    # Subclasses override to declare their backbone's divisibility constraint.
    # Use get_imgsz_divisor(model_name) for variant-aware divisors.
    IMGSZ_DIVISOR: int = 1

    # Relative path of the exported weights inside a run directory.
    EXPORT_NAME: str = "model"

    @abstractmethod
    def create_model(self, config: ModelConfig) -> nn.Module:
        """Instantiate the model (with pretrained weights if configured)."""

    @abstractmethod
    def create_criterion(self, config: ModelConfig) -> nn.Module | None:
        """Instantiate the loss function, or None if loss is internal."""

    @abstractmethod
    def format_batch(self, batch: dict) -> dict:
        """Translate a collated batch into model-specific inputs.
        Returns dict with "inputs" (kwargs for model.forward()) and "targets"."""

    @abstractmethod
    def extract_predictions(self, outputs: Any) -> list[dict]:
        """Translate raw model outputs into standardized format.
        Returns list of dicts: {"boxes": T, "scores": T, "labels": T, "masks": T}"""

    @abstractmethod
    def compute_loss(self, outputs: Any, targets: Any) -> dict[str, Tensor]:
        """Compute loss. Returns dict with at least "loss_total"."""

    def get_default_imgsz(self, model_name: str) -> int:
        """Return the recommended image size for a given model variant.

        Default: 512.  Adapters override for variant-specific defaults.
        """
        return 512

    def get_imgsz_divisor(self, model_name: str) -> int:
        """Return the image-size divisibility constraint for a given variant.

        Default: returns ``IMGSZ_DIVISOR``.  Adapters override when the
        divisor varies across model sizes (e.g. RF-DETR Base vs Small).
        """
        return self.IMGSZ_DIVISOR

    def get_default_gradient_clip_val(self, model_name: str) -> float | None:
        """Return the recommended gradient clip norm for this model family.

        Default: ``None`` (no clipping).  Adapters override to match the
        clip value used by the architecture's reference implementation.
        """
        return None

    def resolve_precision(self, precision: str) -> str:
        """Return the precision this backend should train with.

        Default: honor the configured value.  Adapters override to enforce a
        backend-specific requirement (e.g. forcing fp16 where bf16 diverges).
        """
        return precision

    # Multi-scale training
    def compute_multi_scale_sizes(self, aug_config: AugConfig, model_name: str = "") -> list[int]:
        """Return valid square sizes for multi-scale training.

        Snaps to multiples of IMGSZ_DIVISOR within [imgsz * min, imgsz * max].
        """
        assert aug_config.imgsz is not None, "imgsz must be resolved before computing multi-scale sizes"
        base = aug_config.imgsz
        divisor = max(self.get_imgsz_divisor(model_name), 1)
        lo = int(base * aug_config.multi_scale_min)
        hi = int(base * aug_config.multi_scale_max)
        # Snap to divisor grid
        lo = max(divisor * 2, (lo // divisor) * divisor)
        hi = (hi // divisor) * divisor
        return list(range(lo, hi + 1, divisor))

    def resize_batch(self, formatted: dict, target_size: int) -> dict:
        """Resize a formatted batch to *target_size* for multi-scale training.

        Default: no-op.  Adapters override to handle their specific tensor
        layout (NestedTensor vs pixel_values, normalized vs pixel boxes, etc.).
        """
        return formatted

    # Optional hooks / properties
    @property
    def produces_masks(self) -> bool:
        """Whether the model produces instance masks. Controls metric iou_type."""
        return True

    def configure_optimizers(
        self, model: nn.Module, config: TrainConfig | None,
        steps_per_epoch: int | None = None,
    ) -> Any | None:
        """Override to customize optimizer/scheduler. Return None for default."""
        return None

    def on_before_backward(self, loss: Tensor) -> Tensor:
        """Hook to modify loss before backward."""
        return loss

    def enable_masked_loss(self) -> None:
        """Switch the criterion into masked-loss mode.

        Override in adapters that support per-sample ``class_mask`` targets.
        The base implementation rejects the request so that flipping
        ``TrainConfig.masked_loss=True`` against an unsupported backend
        produces a clear error rather than silently doing nothing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support masked_loss yet"
        )

    def enable_region_class_mask(self) -> None:
        """Switch the criterion into region-class-mask mode.

        Override in adapters that support region-level class masking. The base
        implementation rejects the request so unsupported backends fail clearly
        rather than silently ignoring the configured feature.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support region_class_mask"
        )

    def enable_size_band_ignore(self, ioa_thresh: float = 0.5) -> None:
        """Switch the criterion into size-band-ignore mode.

        Override in adapters that support per-instance ``ignore`` targets.
        The base implementation rejects the request so that flipping
        ``TrainConfig.size_band_ignore=True`` against an unsupported backend
        produces a clear error rather than silently doing nothing.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support size_band_ignore"
        )

    def export_model(self, model: nn.Module, output_dir: Path) -> None:
        """Export trained weights in the backend's native format.

        Default: no-op.  Adapters override to save in their specific format
        (e.g. HuggingFace ``save_pretrained`` or rfdetr ``.pth``).
        """

    def postprocess_for_inference(
        self, outputs: Any, image_sizes: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Post-process raw model outputs for metric evaluation.

        Args:
            outputs: Raw model forward outputs.
            image_sizes: (H, W) of the input images (after augmentation/padding).
                Adapters use this to rescale predictions (boxes, masks) to the
                same coordinate space as the targets.
        """
        return self.extract_predictions(outputs)

    # --- shared val-time detection filter -----------------------------------
    # Query-based detectors (RF-DETR, Mask2Former) emit a FIXED number of
    # predictions per image regardless of confidence -- RF-DETR takes a bare
    # top-`num_select` (200 for seg-large) with no score threshold at all, and
    # Mask2Former keeps every query via `max_scores > 0.0`. YOLO, by contrast,
    # already bounds itself with NMS + val_conf_thres + val_max_det.
    #
    # That asymmetry is a pure cost: measured 2026-07-22 on a real validation set, RF-DETR
    # emitted 200 dets/image vs YOLO's 4.5 (44x), which made SegmentationMAP's
    # CPU RLE encode 42x slower (9.45s vs 0.23s per 12 batches) and dominated
    # ~90% of eval wall-clock. It buys nothing: COCO mAP caps at
    # max_detection_thresholds=100, so torchmetrics DISCARDS everything past the
    # top 100 anyway (it warns "Encountered more than 100 detections...").
    #
    # Defaults follow the official repos for the thing being measured (COCO mAP):
    #   max_det=100     -- the COCO cap itself. torchmetrics runs
    #                      max_detection_thresholds=[1,10,100], so anything past
    #                      100 is DISCARDED by the metric. Ultralytics' 300 is a
    #                      *prediction* default, not an evaluation one; copying it
    #                      would cost 3x the RLE encode for exactly zero change.
    #                      (RF-DETR has no single answer to copy either -- its
    #                      num_select is 100/200/300 depending on variant.)
    #   conf_thres=0.001 -- Ultralytics' val default (their predict default is
    #                      0.25; RF-DETR's predict() and Mask2Former's
    #                      post_process_* both default to 0.5, but those are
    #                      deployment APIs -- neither thresholds during eval).
    #                      A low floor is also MORE faithful: with a binding cap
    #                      of 100, it lets the top-100 be genuinely the 100
    #                      highest-scored detections rather than pre-truncated.
    #   nms_iou=0.7     -- Ultralytics' NMS default.
    #
    # On NMS: DETR-family models claim not to need it (one-to-one Hungarian
    # matching is meant to suppress duplicates). Measured on a real validation set with a
    # trained RF-DETR ckpt, that does NOT hold: 90.2% of detections overlap another
    # of the SAME class at mask-IoU > 0.5, median max-IoU 0.87, and only ~11% of
    # that is the query x class top-k expansion. The literature agrees -- see
    # "NMS Strikes Back" (2212.06137) and DETRs-with-Hybrid-Matching, where
    # one-to-many + NMS beats pure one-to-one by up to 2.5 mAP.
    VAL_CONF_THRES_DEFAULT: float = 0.001
    VAL_MAX_DET_DEFAULT: int = 100
    VAL_NMS_IOU_DEFAULT: float = 0.7

    # Keys that live in ModelConfig.extra but are TOOLKIT val-time knobs, NOT
    # arguments for the underlying model library. Adapters that splat `extra`
    # into a backend constructor (e.g. Mask2Former -> HF from_pretrained) MUST
    # strip these first, or the backend rejects the unknown kwarg. Single source
    # of truth -- a per-adapter hand-copied set already drifted (val_full_res_masks
    # was missing from every copy, which crashed the M2F full-res eval).
    TOOLKIT_ONLY_EXTRA_KEYS: frozenset = frozenset({
        "val_mask_res", "val_conf_thres", "val_max_det", "val_nms_iou",
        "val_full_res_masks", "val_mask_downsample",
    })

    # ------------------------------------------------------------------
    # Val mask resolution (shared across all three backends)
    # ------------------------------------------------------------------

    def _init_val_mask_resolution(self, extra: dict | None) -> None:
        """Resolve the val-mask-resolution policy from ``ModelConfig.extra``.

        Call from ``create_model``. Raises on conflicting knobs, so a config
        with an ambiguous spec fails at model construction rather than
        producing a quietly mis-scaled metric for a whole run.
        """
        from training_toolkit.models.val_mask_resolution import (
            resolve_val_mask_resolution,
        )

        self._val_mask_spec = resolve_val_mask_resolution(extra)
        self._val_mask_logged = False

    def _val_mask_target_hw(
        self,
        input_hw: tuple[int, int] | None,
        native_hw: tuple[int, int] | None = None,
        masks_already_native: bool = False,
    ) -> tuple[int, int] | None:
        """Target mask (h, w) for this batch, or ``None`` to leave masks alone.

        ``native_hw`` is the head's own mask resolution. ``masks_already_native``
        says whether the adapter has ALREADY left the masks there:

        * YOLO (``upsample=False``) and Mask2Former (interpolate skipped) can
          decline to upsample in the first place -> nothing to do.
        * RF-DETR cannot: its ``PostProcess`` fuses box rescaling and mask
          interpolation in one call, so masks arrive at full resolution and
          ``native`` has to be reached by resizing back down. Costs a GPU
          interpolate, but the expensive parts (CPU RLE-encode + mask D2H in
          ``SegmentationMAP.update``) still shrink with the final mask area,
          which is where the ~3x came from.
        """
        from training_toolkit.models.val_mask_resolution import target_mask_hw

        spec = getattr(self, "_val_mask_spec", None)
        if spec is None:  # adapter never called _init_val_mask_resolution
            return None
        target = target_mask_hw(spec, input_hw)
        if spec.is_native:
            target = None if masks_already_native else native_hw

        if not getattr(self, "_val_mask_logged", False):
            self._val_mask_logged = True
            effective = target if target is not None else native_hw
            logger.info(
                "Val mask resolution: %s%s. The per-epoch val/map is measured "
                "on THIS scale -- the reported headline number should come "
                "from a separate full-resolution eval.",
                spec.describe(),
                f" -> {effective[0]}x{effective[1]} px" if effective else "",
            )
        return target

    @staticmethod
    def _resize_val_masks_hw(masks: Tensor, hw: tuple[int, int]) -> Tensor:
        """Nearest-resize predicted masks to ``hw`` (bool, empty-safe).

        Nearest (not bilinear) because these are already binary: it is the
        resample that cannot invent a fractional occupancy the metric would
        then have to re-threshold. Matches the downsample
        validated by an earlier fast-eval study.
        """
        import torch.nn.functional as F

        h, w = hw
        if masks.shape[0] == 0:
            return masks.new_zeros((0, h, w))
        if tuple(masks.shape[-2:]) == (h, w):
            return masks
        return (
            F.interpolate(masks.float().unsqueeze(1), size=(h, w), mode="nearest")
            .squeeze(1)
            .bool()
        )

    def _apply_val_mask_resolution(
        self,
        preds: list[dict],
        input_hw: tuple[int, int] | None,
        native_hw: tuple[int, int] | None = None,
        masks_already_native: bool = False,
    ) -> list[dict]:
        """Resize every pred's masks to the configured val resolution.

        The matching GT downsample is handled generically by
        ``LitModule._format_targets_for_metrics``, which keys off
        ``preds[i]["masks"].shape[-2:]`` -- so an adapter only has to get the
        PREDICTED masks to the right resolution and the metric stays
        self-consistent.
        """
        target = self._val_mask_target_hw(input_hw, native_hw, masks_already_native)
        if target is None:
            return preds
        for pred in preds:
            if "masks" in pred:
                pred["masks"] = self._resize_val_masks_hw(pred["masks"], target)
        return preds

    @staticmethod
    def _nms_keep(masks, boxes, scores, labels, iou_thres: float):
        """Greedy class-aware NMS -> indices to keep (score-descending).

        Uses MASK IoU when masks are available, falling back to box IoU. That is
        a deliberate departure from Ultralytics, which NMSes on boxes only
        because its NMS runs BEFORE masks exist. We run post-hoc, so masks are
        already materialised -- and for thin diagonal structures (cracks) a box
        is a terrible duplicate proxy: two disjoint diagonal cracks can share
        almost the same bounding box, so box NMS would suppress genuine separate
        instances. Mask IoU is the criterion the metric actually scores on.

        Class-aware (same-label pairs only), matching torchvision batched_nms:
        DETR-style top-k runs over the flattened query x class grid, so ONE query
        can legitimately appear once per class with an identical mask. Those are
        hedged hypotheses COCO scores per class -- suppressing them across
        classes would delete real signal.
        """
        import torch

        n = scores.numel()
        order = torch.argsort(scores, descending=True)
        if masks is not None and masks.numel():
            flat = masks.reshape(n, -1).float()
            area = flat.sum(1)
            inter = flat @ flat.T
            union = area[:, None] + area[None, :] - inter
        elif boxes is not None and boxes.numel():
            x1 = torch.maximum(boxes[:, None, 0], boxes[None, :, 0])
            y1 = torch.maximum(boxes[:, None, 1], boxes[None, :, 1])
            x2 = torch.minimum(boxes[:, None, 2], boxes[None, :, 2])
            y2 = torch.minimum(boxes[:, None, 3], boxes[None, :, 3])
            inter = (x2 - x1).clamp_min(0) * (y2 - y1).clamp_min(0)
            a = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).clamp_min(0)
            union = a[:, None] + a[None, :] - inter
        else:
            return order                      # nothing to compare on
        iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))
        same = (labels[:, None] == labels[None, :]) if labels is not None \
            else torch.ones_like(iou, dtype=torch.bool)

        dead = torch.zeros(n, dtype=torch.bool, device=scores.device)
        keep = []
        for i in order.tolist():
            if dead[i]:
                continue
            keep.append(i)
            dead |= (iou[i] > iou_thres) & same[i]
            dead[i] = False                   # never suppress the survivor itself
        return torch.tensor(keep, dtype=torch.long, device=scores.device)

    def _apply_val_det_filter(
        self, preds: list[dict], conf_thres: float, max_det: int,
        nms_iou: float = 0.0,
    ) -> list[dict]:
        """conf threshold -> class-aware NMS -> top-k cap, per image.

        Ordering mirrors Ultralytics' pipeline (conf, then NMS, then max_det):
        the cap must come LAST, or it fills up with duplicates that NMS would
        have removed.

        Keeps every per-image tensor field (boxes/scores/labels/masks/...) in
        lockstep by indexing them all with the same kept-index tensor, so a
        caller cannot desync them. No-op when all knobs are falsy.
        """
        import torch

        if not conf_thres and not max_det and not nms_iou:
            return preds
        out = []
        for pred in preds:
            scores = pred.get("scores")
            if scores is None or scores.numel() == 0:
                out.append(pred)
                continue
            keep = torch.arange(scores.numel(), device=scores.device)
            if conf_thres:
                keep = keep[scores[keep] > conf_thres]
            if nms_iou and keep.numel() > 1:
                sub = self._nms_keep(
                    pred["masks"][keep] if "masks" in pred else None,
                    pred["boxes"][keep] if "boxes" in pred else None,
                    scores[keep],
                    pred["labels"][keep] if "labels" in pred else None,
                    nms_iou,
                )
                keep = keep[sub]
            if max_det and keep.numel() > max_det:
                # rank the survivors by score, keep the best max_det
                order = torch.argsort(scores[keep], descending=True)[:max_det]
                keep = keep[order]
            if keep.numel() == scores.numel():
                out.append(pred)
                continue
            out.append({
                k: (v[keep] if hasattr(v, "shape") and v.shape[:1] == scores.shape[:1] else v)
                for k, v in pred.items()
            })
        return out
