"""RF-DETR model adapter using the rfdetr package."""

import contextlib
import io
import logging
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from training_toolkit.configs import ModelConfig, TrainConfig
from training_toolkit.models.base import ModelAdapter

logger = logging.getLogger(__name__)

# Maps user-facing size keywords to the rfdetr segmentation wrapper class
# and its model config class.  All imports are deferred to avoid loading
# rfdetr at module import time.
_SEG_VARIANTS: dict[str, tuple[str, str]] = {
    "nano":    ("RFDETRSegNano",    "RFDETRSegNanoConfig"),
    "small":   ("RFDETRSegSmall",   "RFDETRSegSmallConfig"),
    "medium":  ("RFDETRSegMedium",  "RFDETRSegMediumConfig"),
    "large":   ("RFDETRSegLarge",   "RFDETRSegLargeConfig"),
    "preview": ("RFDETRSegPreview", "RFDETRSegPreviewConfig"),
}

_VARIANT_RESOLUTIONS: dict[str, int] = {
    "nano": 312, "small": 384, "medium": 432, "large": 504, "preview": 432,
}

_VARIANT_DIVISORS: dict[str, int] = {
    "nano": 12, "small": 24, "medium": 24, "large": 24, "preview": 24,
}

# ImageNet statistics — required by RF-DETR's DINOv2 backbone and by the
# upstream RF-DETR training pipeline (see rfdetr/datasets/coco.py:153-156,
# rfdetr/detr.py:62-63 for the predict-time counterparts). The values are
# identical to the upstream constants; keeping a local copy so the
# training-toolkit can normalize without importing rfdetr internals.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class RFDETRAdapter(ModelAdapter):
    """Adapter for RF-DETR instance segmentation.

    Uses the dedicated segmentation variants (``RFDETRSegNano``,
    ``RFDETRSegMedium``, …) which ship with COCO-pretrained segmentation
    weights — backbone, detection head, *and* segmentation head are all
    pretrained.

    Supported sizes via ``config.model.name``:
    ``rf-detr-seg-nano``, ``rf-detr-seg-small``, ``rf-detr-seg-medium``,
    ``rf-detr-seg-large``, ``rf-detr-seg-preview``.
    """

    IMGSZ_DIVISOR = 24  # conservative fallback
    EXPORT_NAME = "model.pth"

    def __init__(self) -> None:
        self._postprocessor = None
        self._weight_dict = None
        self._args = None
        self._masked_loss = False
        self._region_class_mask = False
        self._size_band_ignore = False
        self._ignore_ioa_thresh = 0.5
        self._val_conf_thres: float = self.VAL_CONF_THRES_DEFAULT
        self._val_max_det: int = self.VAL_MAX_DET_DEFAULT
        self._val_nms_iou: float = self.VAL_NMS_IOU_DEFAULT

    # ------------------------------------------------------------------
    # Size / resolution helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _size_from_name(model_name: str) -> str:
        """Extract variant size keyword from a model name string."""
        name = model_name.lower().replace("-", "").replace("_", "")
        for key in ("nano", "small", "medium", "large", "preview"):
            if key in name:
                return key
        return "medium"  # sensible default

    def get_default_imgsz(self, model_name: str) -> int:
        return _VARIANT_RESOLUTIONS.get(self._size_from_name(model_name), 432)

    def get_imgsz_divisor(self, model_name: str) -> int:
        return _VARIANT_DIVISORS.get(self._size_from_name(model_name), self.IMGSZ_DIVISOR)

    def get_default_gradient_clip_val(self, model_name: str) -> float | None:
        return 0.1  # matches DETR/Deformable-DETR/RF-DETR reference configs

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def create_model(self, config: ModelConfig) -> nn.Module:
        from rfdetr.models.lwdetr import PostProcess, build_model

        pretrained = config.pretrained or "auto"
        is_custom_path = pretrained not in ("auto", "none") and Path(pretrained).exists()
        size = self._size_from_name(config.name)

        # The val-mask-resolution knobs are toolkit val-speed knobs, not rf-detr
        # args — read them out and keep them off the args namespace dumped into
        # build_model(). Raises on a conflicting spec, at construction time.
        self._init_val_mask_resolution(config.extra)
        self._val_conf_thres = config.extra.get("val_conf_thres", self.VAL_CONF_THRES_DEFAULT)
        self._val_max_det = config.extra.get("val_max_det", self.VAL_MAX_DET_DEFAULT)
        self._val_nms_iou = config.extra.get("val_nms_iou", self.VAL_NMS_IOU_DEFAULT)
        _args_extra = {k: v for k, v in config.extra.items()
                       if k not in self.TOOLKIT_ONLY_EXTRA_KEYS}

        if pretrained == "auto":
            pretrained_state, self._args = self._load_pretrained(size)
        elif is_custom_path:
            checkpoint = torch.load(pretrained, map_location="cpu", weights_only=False)
            if "model" in checkpoint:
                pretrained_state = checkpoint["model"]
            elif "state_dict" in checkpoint:
                # Lightning checkpoint — strip "model." prefix
                pretrained_state = {
                    k.removeprefix("model."): v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("model.")
                }
            else:
                raise KeyError(
                    f"Checkpoint {pretrained} has neither 'model' nor "
                    f"'state_dict' key (found: {list(checkpoint.keys())})"
                )
            self._args = checkpoint.get("args") or self._build_args(size)
        else:
            pretrained_state = None

        if pretrained_state is not None:
            if self._args is None:
                self._args = self._build_args(size)
            self._args.num_classes = config.num_classes
            self._apply_seg_train_defaults(self._args)
            for k, v in _args_extra.items():
                setattr(self._args, k, v)

            model = build_model(self._args)
            model_state = model.state_dict()
            compatible = {
                k: v for k, v in pretrained_state.items()
                if k in model_state and v.shape == model_state[k].shape
            }
            missing, _ = model.load_state_dict(compatible, strict=False)
            source = pretrained if is_custom_path else f"COCO-seg ({size})"
            logger.info(
                "Loaded %s weights (%d/%d params); "
                "%d parameters randomly initialised",
                source, len(compatible), len(model_state), len(missing),
            )
        else:
            self._args = self._build_args(size)
            self._args.num_classes = config.num_classes
            self._apply_seg_train_defaults(self._args)
            for k, v in _args_extra.items():
                setattr(self._args, k, v)
            model = build_model(self._args)

        self._postprocessor = PostProcess(num_select=self._args.num_select)
        return model

    @staticmethod
    def _load_pretrained(size: str) -> tuple[dict, Any]:
        """Download COCO-pretrained segmentation weights for *size*."""
        wrapper_name, _ = _SEG_VARIANTS.get(size, _SEG_VARIANTS["medium"])
        import rfdetr.detr as detr_module
        wrapper_cls = getattr(detr_module, wrapper_name)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            wrapper = wrapper_cls()
        for line in buf.getvalue().splitlines():
            if line.strip():
                logger.debug(line.strip())

        inner = wrapper.model  # rfdetr.main.Model
        return inner.model.state_dict(), inner.args

    @staticmethod
    def _build_args(size: str):
        """Build rfdetr args from a seg config without downloading weights."""
        _, config_name = _SEG_VARIANTS.get(size, _SEG_VARIANTS["medium"])
        import rfdetr.config as cfg_module
        from rfdetr.main import populate_args

        cfg_cls = getattr(cfg_module, config_name)
        cfg_dict = cfg_cls().dict()
        cfg_dict["pretrain_weights"] = None
        return populate_args(**cfg_dict)

    @staticmethod
    def _apply_seg_train_defaults(args: Any) -> None:
        """Ensure mask-loss hyper-parameters are present on *args*.

        These live in ``SegmentationTrainConfig``, not the model config,
        so they may be absent when args was built from the model config alone.
        """
        from rfdetr.config import SegmentationTrainConfig
        defaults = {
            "mask_ce_loss_coef": SegmentationTrainConfig.model_fields["mask_ce_loss_coef"].default,
            "mask_dice_loss_coef": SegmentationTrainConfig.model_fields["mask_dice_loss_coef"].default,
            "mask_point_sample_ratio": SegmentationTrainConfig.model_fields["mask_point_sample_ratio"].default,
            "segmentation_head": True,
        }
        for k, v in defaults.items():
            if not hasattr(args, k) or not getattr(args, k):
                setattr(args, k, v)

    # ------------------------------------------------------------------
    # Criterion
    # ------------------------------------------------------------------

    def create_criterion(self, config: ModelConfig) -> nn.Module:
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors

        if self._args is None:
            raise RuntimeError("create_model() must be called before create_criterion()")

        # Propagate masked_loss into args so build_criterion_and_postprocessors
        # constructs SetCriterion with masked_loss=True. enable_masked_loss()
        # must be called between create_model() and create_criterion().
        self._args.masked_loss = self._masked_loss
        self._args.region_class_mask = self._region_class_mask
        self._args.size_band_ignore = self._size_band_ignore
        self._args.ignore_ioa_thresh = self._ignore_ioa_thresh

        criterion, _ = build_criterion_and_postprocessors(self._args)
        self._weight_dict = criterion.weight_dict
        return criterion

    def enable_masked_loss(self) -> None:
        self._masked_loss = True

    def enable_region_class_mask(self) -> None:
        self._region_class_mask = True

    def enable_size_band_ignore(self, ioa_thresh: float = 0.5) -> None:
        self._size_band_ignore = True
        self._ignore_ioa_thresh = ioa_thresh

    # ------------------------------------------------------------------
    # Optimizer (layer-wise LR decay for DINOv2 backbone)
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, model: nn.Module, config: TrainConfig | None,
        steps_per_epoch: int | None = None,
    ) -> Any | None:
        if self._args is None or config is None:
            return None

        from rfdetr.util.get_param_dicts import get_param_dict

        # Let RF-DETR's param grouping use config.lr as the base LR,
        # but preserve all the relative scaling (layer decay, component decay).
        self._args.lr = config.lr
        self._args.weight_decay = config.weight_decay

        param_dicts = get_param_dict(self._args, model)
        # Filter to trainable params only
        param_dicts = [d for d in param_dicts if d["params"].requires_grad]

        optimizer = torch.optim.AdamW(
            param_dicts, lr=config.lr, weight_decay=config.weight_decay,
        )

        # Summarise param groups by LR bucket
        lr_buckets: dict[float, int] = {}
        for group in optimizer.param_groups:
            lr = group["lr"]
            n = group["params"].numel() if hasattr(group["params"], "numel") else 1
            lr_buckets[lr] = lr_buckets.get(lr, 0) + n
        for lr_val in sorted(lr_buckets, reverse=True):
            logger.info(
                "RF-DETR optimizer: lr=%.2e → %d params", lr_val, lr_buckets[lr_val],
            )

        from training_toolkit.engine.schedulers import build_scheduler
        result = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
        if result is None:
            return optimizer

        scheduler, lr_config = result
        return [optimizer], [{"scheduler": scheduler, **lr_config}]

    # ------------------------------------------------------------------
    # Batch formatting
    # ------------------------------------------------------------------

    def resize_batch(self, formatted: dict, target_size: int) -> dict:
        import torch.nn.functional as F
        from rfdetr.util.misc import NestedTensor

        nested = formatted["inputs"]["samples"]
        tensors = F.interpolate(
            nested.tensors, size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        mask = F.interpolate(
            nested.mask.unsqueeze(1).float(),
            size=(target_size, target_size), mode="nearest",
        ).squeeze(1).bool()

        formatted["inputs"]["samples"] = NestedTensor(tensors, mask)
        return formatted

    def format_batch(self, batch: dict) -> dict:
        from rfdetr.util.misc import NestedTensor

        images = batch["images"]  # (B, 3, H, W), values in [0, 1]
        B, _, H, W = images.shape

        # Apply ImageNet mean/std normalization to match upstream RF-DETR's
        # train/val (rfdetr/datasets/coco.py) and inference (detr.py) paths.
        # The DINOv2 backbone is pretrained under this normalization; feeding
        # raw /255 inputs silently shifts the distribution and produces
        # checkpoints that are mismatched with rfdetr.predict() at inference.
        mean = torch.tensor(
            _IMAGENET_MEAN, device=images.device, dtype=images.dtype,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            _IMAGENET_STD, device=images.device, dtype=images.dtype,
        ).view(1, 3, 1, 1)
        images = (images - mean) / std

        mask = torch.zeros(B, H, W, dtype=torch.bool, device=images.device)
        nested = NestedTensor(images, mask)

        targets = []
        for t in batch["targets"]:
            target = {
                "labels": t["labels"],
                "boxes": self._xyxy_to_cxcywh(t["boxes"], W, H),
                "masks": t["masks"].float(),
            }
            # Forward the optional class_mask to the criterion — only used when
            # SetCriterion was built with masked_loss=True; ignored otherwise.
            if "class_mask" in t:
                target["class_mask"] = t["class_mask"]
            if self._region_class_mask:
                for key in ("region_boxes", "region_class_masks"):
                    if key in t:
                        target[key] = t[key]
            # Forward the optional per-instance ignore flags (size-band SNIP) —
            # only used when SetCriterion was built with size_band_ignore=True.
            if "ignore" in t:
                target["ignore"] = t["ignore"]
            targets.append(target)

        return {"inputs": {"samples": nested}, "targets": targets}

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, outputs: Any, targets: Any) -> dict[str, Tensor]:
        raise NotImplementedError(
            "RF-DETR uses external criterion — use compute_loss_with_criterion"
        )

    def compute_loss_with_criterion(
        self, criterion: nn.Module, outputs: Any, targets: Any,
    ) -> dict[str, Tensor]:
        with torch.amp.autocast("cuda", enabled=False):
            outputs_fp32 = self._to_float32(outputs)
            targets_fp32 = [self._to_float32(t) for t in targets]
            loss_dict = criterion(outputs_fp32, targets_fp32)

        weighted = {}
        total = torch.tensor(0.0, device=next(iter(loss_dict.values())).device)
        for k, v in loss_dict.items():
            if k not in self._weight_dict:
                # Diagnostic metrics (cardinality_error, class_error) —
                # log but do NOT add to backward loss.
                weighted[k] = v.detach()
                continue
            weighted[k] = v * self._weight_dict[k]
            total = total + weighted[k]

        weighted["loss_total"] = total
        return weighted

    # ------------------------------------------------------------------
    # Inference / postprocessing
    # ------------------------------------------------------------------

    def postprocess_for_inference(
        self, outputs: Any, image_sizes: tuple[int, int] | None = None,
    ) -> list[dict]:
        if self._postprocessor is None:
            raise RuntimeError("create_model() must be called first")

        pred_logits = outputs["pred_logits"]
        batch_size = pred_logits.shape[0]

        # The seg head's OWN mask resolution (imgsz // mask_downsample_ratio),
        # read off the real tensor rather than reconstructed from config — the
        # architecture param can be set per-run and a hand-copied constant
        # would drift. Feeds the `native` val-mask policy.
        native_hw = None
        if "pred_masks" in outputs and isinstance(outputs["pred_masks"], torch.Tensor):
            native_hw = tuple(outputs["pred_masks"].shape[-2:])

        # Ask PostProcess to decode masks straight to the val resolution.
        # `mask_size` decouples the mask resolution from `target_sizes` (which
        # still has to be the true image frame, because boxes are rescaled with
        # it). One interpolation instead of an upsample-then-downsample round
        # trip -- cheaper, and more faithful, since the round trip resamples
        # every boundary twice. Under `native` it skips the interpolation
        # entirely, which is the only way to get genuinely native masks.
        #
        # Safe against an rf-detr that predates `mask_size`: that build ignores
        # the attribute and returns full-resolution masks, and the
        # `_apply_val_mask_resolution` call below still corrects them (its
        # resize is a no-op when the shape already matches, so on a current
        # rf-detr it costs nothing).
        spec = getattr(self, "_val_mask_spec", None)
        if spec is not None and spec.is_native:
            self._postprocessor.mask_size = "native"
        else:
            self._postprocessor.mask_size = self._val_mask_target_hw(
                image_sizes, native_hw, masks_already_native=False
            )

        # PostProcess rescales boxes and interpolates masks to target_sizes.
        if image_sizes is not None:
            h, w = image_sizes
            target_sizes = torch.tensor(
                [[h, w]] * batch_size, device=pred_logits.device,
            )
        elif native_hw is not None:
            mh, mw = native_hw
            target_sizes = torch.tensor(
                [[mh, mw]] * batch_size, device=pred_logits.device,
            )
        else:
            target_sizes = torch.tensor(
                [[640, 640]] * batch_size, device=pred_logits.device,
            )

        results = self._postprocessor(outputs, target_sizes)

        preds = []
        for r in results:
            pred = {
                "boxes": r["boxes"],
                "scores": r["scores"],
                "labels": r["labels"],
            }
            if "masks" in r:
                masks = r["masks"]
                if masks.dim() == 4:
                    masks = masks.squeeze(1)
                pred["masks"] = masks.bool()
            preds.append(pred)

        preds = self._apply_val_det_filter(
            preds, self._val_conf_thres, self._val_max_det, self._val_nms_iou)
        # `masks_already_native=False`: PostProcess fuses box rescaling with the
        # mask interpolate, so masks are at full resolution by the time we get
        # them and `native` has to be reached by resizing back down.
        return self._apply_val_mask_resolution(
            preds, input_hw=image_sizes, native_hw=native_hw,
            masks_already_native=False,
        )

    def predict_with_class_vectors(
        self, outputs: Any, image_sizes: tuple[int, int], n_queries: int = 100,
    ) -> list[dict]:
        """Query-native decode that KEEPS the full per-class probability vector.

        ``postprocess_for_inference`` top-Ks over the flattened (Q*K) grid and
        therefore returns only top-1 (label, score) pairs. Confident-learning
        needs the whole K-vector per instance, so here we select QUERIES (by
        max-class probability) and return their full rows.

        Geometry is reproduced verbatim from ``PostProcess.forward``
        (rf-detr lwdetr.py); a parity test pins our kept-query-0 against stock's
        index-0 box and mask. NOTE the probabilities are per-class SIGMOID and
        do not sum to 1.

        ⚠️ The parity holds only while the maximising (query, class) cell is
        UNIQUE. ``max_q(max_k p)`` equals ``max_{q,k} p`` in value, but when two
        queries tie on that maximum, ``torch.topk``'s tie-break is
        implementation-defined and does NOT agree between stock's full (Q*K)
        topk and our length-Q one (measured: 73/200 forced ties select a
        different query). This is reachable in practice — float32 sigmoid
        saturates to exactly 1.0 above logit ~17, which several confidently
        firing redundant queries around one object can all cross.

        Harmless for the intended consumer: a tie only reorders which of two
        EQUALLY maximal queries lands at index 0; no query's own geometry or
        K-vector is affected, and stock's own tie-break is itself arbitrary.
        But do not rely on ``query_idx[0]`` reproducing stock's choice.
        """
        import torch.nn.functional as F
        from rfdetr.util import box_ops

        logits = outputs["pred_logits"]            # (B, Q, K)
        prob = logits.sigmoid()
        bsz, n_q, _ = prob.shape
        keep_n = min(int(n_queries), n_q)

        # Same query the stock path's global argmax selects.
        q_score = prob.max(dim=-1).values          # (B, Q)
        keep = q_score.topk(keep_n, dim=1).indices  # (B, N)

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"])
        h, w = int(image_sizes[0]), int(image_sizes[1])
        scale = torch.tensor([w, h, w, h], dtype=boxes_xyxy.dtype,
                             device=boxes_xyxy.device)

        masks_in = outputs.get("pred_masks", None)
        results: list[dict] = []
        for i in range(bsz):
            idx = keep[i]
            res = {
                "query_idx": idx.detach().cpu(),
                "probs": prob[i, idx].detach().cpu(),
                "boxes": (boxes_xyxy[i, idx] * scale).detach().cpu(),
            }
            if masks_in is not None:
                m = masks_in[i, idx].unsqueeze(1)          # (N,1,Hm,Wm)
                m = F.interpolate(m, size=(h, w), mode="bilinear",
                                  align_corners=False)
                res["masks"] = (m > 0.0).squeeze(1).detach().cpu()
            results.append(res)
        return results

    def extract_predictions(self, outputs: Any) -> list[dict]:
        return self.postprocess_for_inference(outputs)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, model, output_dir):
        path = output_dir / self.EXPORT_NAME
        torch.save({"model": model.state_dict(), "args": self._args}, str(path))
        logger.info("Exported RF-DETR weights to %s", path)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _xyxy_to_cxcywh(boxes: Tensor, img_w: int, img_h: int) -> Tensor:
        if boxes.numel() == 0:
            return boxes
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return torch.stack([cx, cy, w, h], dim=-1)

    @staticmethod
    def _to_float32(data: Any) -> Any:
        if isinstance(data, Tensor):
            return data.float() if data.is_floating_point() else data
        if isinstance(data, dict):
            return {k: RFDETRAdapter._to_float32(v) for k, v in data.items()}
        if isinstance(data, (list, tuple)):
            return type(data)(RFDETRAdapter._to_float32(v) for v in data)
        return data
