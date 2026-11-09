"""Mask2Former model adapter using HuggingFace Transformers."""

import logging
import re
from typing import Any

import torch
from torch import Tensor, nn

from training_toolkit.configs import ModelConfig, TrainConfig
from training_toolkit.models.base import ModelAdapter

logger = logging.getLogger(__name__)

# HuggingFace checkpoint IDs for COCO instance segmentation, keyed by Swin size.
_HF_CHECKPOINTS: dict[str, str] = {
    "tiny": "facebook/mask2former-swin-tiny-coco-instance",
    "small": "facebook/mask2former-swin-small-coco-instance",
    "base": "facebook/mask2former-swin-base-coco-instance",
    "large": "facebook/mask2former-swin-large-coco-instance",
}

# ImageNet normalization (expected by HuggingFace Mask2Former / Swin backbone).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class Mask2FormerAdapter(ModelAdapter):
    """Adapter for HuggingFace Mask2FormerForUniversalSegmentation.

    Loss is computed internally by the model when mask_labels and class_labels
    are provided to forward(). No external criterion needed.
    """

    # Swin backbone encoder stride
    IMGSZ_DIVISOR = 32

    def __init__(self) -> None:
        self._image_processor = None
        self._masked_loss = False
        self._model = None
        self._val_conf_thres: float = self.VAL_CONF_THRES_DEFAULT
        self._val_max_det: int = self.VAL_MAX_DET_DEFAULT
        self._val_nms_iou: float = self.VAL_NMS_IOU_DEFAULT

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def create_model(self, config: ModelConfig) -> nn.Module:
        from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerConfig

        pretrained = config.pretrained or "auto"
        extra = config.extra or {}

        # The val-mask-resolution knobs are toolkit val-speed knobs, not HF
        # model/config args — read them out and keep them off the kwargs
        # splatted into from_pretrained()/Mask2FormerConfig() below.
        self._init_val_mask_resolution(extra)
        self._val_conf_thres = extra.get("val_conf_thres", self.VAL_CONF_THRES_DEFAULT)
        self._val_max_det = extra.get("val_max_det", self.VAL_MAX_DET_DEFAULT)
        self._val_nms_iou = extra.get("val_nms_iou", self.VAL_NMS_IOU_DEFAULT)
        extra = {k: v for k, v in extra.items()
                 if k not in self.TOOLKIT_ONLY_EXTRA_KEYS}

        if pretrained == "auto":
            pretrained = self._resolve_checkpoint(config.name)
            logger.info("Resolved pretrained checkpoint: %s", pretrained)

        if pretrained and pretrained != "none":
            logger.info("Loading Mask2Former from pretrained: %s", pretrained)
            model = Mask2FormerForUniversalSegmentation.from_pretrained(
                pretrained,
                num_labels=config.num_classes,
                ignore_mismatched_sizes=True,
                **extra,
            )
        else:
            logger.info("Creating Mask2Former from config (no pretrained weights)")
            hf_config = Mask2FormerConfig(num_labels=config.num_classes, **extra)
            model = Mask2FormerForUniversalSegmentation(hf_config)

        logger.info(
            "Backbone BatchNorm layers are frozen (eval mode) — this is "
            "intentional for fine-tuning. Ignore Lightning's eval-mode warning."
        )
        self._model = model
        return model

    @staticmethod
    def _resolve_checkpoint(model_name: str) -> str:
        """Map a model name like ``mask2former-swin-large`` to a HF checkpoint."""
        m = re.search(r"swin[_-]?(tiny|small|base|large)", model_name, re.IGNORECASE)
        size = m.group(1).lower() if m else "large"
        checkpoint = _HF_CHECKPOINTS.get(size)
        if checkpoint is None:
            raise ValueError(
                f"No known checkpoint for Swin size {size!r}. "
                f"Set model.pretrained explicitly. Known sizes: {list(_HF_CHECKPOINTS)}"
            )
        return checkpoint

    def get_default_gradient_clip_val(self, model_name: str) -> float | None:
        return 0.01  # matches Mask2Former paper (Cheng et al., 2022)

    def resolve_precision(self, precision: str) -> str:
        """Mask2Former diverges to NaN under bf16 autocast (saturated-logit
        backward underflow); force fp16 (16-mixed + GradScaler)."""
        if precision and "bf16" in precision:
            logger.warning(
                "Mask2Former diverges to NaN under precision=%r; forcing "
                "'16-mixed' (fp16 + GradScaler). Set precision: 16-mixed in your "
                "config to silence this.",
                precision,
            )
            return "16-mixed"
        return precision

    def create_criterion(self, config: ModelConfig) -> None:
        return None

    def enable_masked_loss(self) -> None:
        from training_toolkit.models.mask2former_masked_loss import MaskedMask2FormerLoss
        if self._model is None:
            raise RuntimeError("create_model() must be called before enable_masked_loss()")
        crit = self._model.criterion
        for attr in ("_get_predictions_permutation_indices", "empty_weight", "num_labels"):
            if not hasattr(crit, attr):
                raise RuntimeError(
                    f"MaskedMask2FormerLoss requires Mask2FormerLoss.{attr} "
                    "(transformers internals changed?). Cannot enable masked loss."
                )
        crit.__class__ = MaskedMask2FormerLoss   # preserves matcher, empty_weight, config
        crit._class_mask = None
        self._masked_loss = True

    def resize_batch(self, formatted: dict, target_size: int) -> dict:
        """Resize pixel_values and mask_labels to *target_size*."""
        import torch.nn.functional as F

        inputs = formatted["inputs"]
        inputs["pixel_values"] = F.interpolate(
            inputs["pixel_values"], size=(target_size, target_size),
            mode="bilinear", align_corners=False,
        )
        inputs["mask_labels"] = [
            F.interpolate(
                m.unsqueeze(1), size=(target_size, target_size), mode="nearest",
            ).squeeze(1)
            for m in inputs["mask_labels"]
        ]
        return formatted

    def format_batch(self, batch: dict) -> dict:
        """Convert collated batch to HuggingFace Mask2Former inputs.

        The model expects:
        - pixel_values: (B, 3, H, W) float tensor, ImageNet-normalised
        - mask_labels: list of (N_i, H, W) float tensors
        - class_labels: list of (N_i,) long tensors
        """
        images = batch["images"]  # (B, 3, H, W) float [0, 1] from collate
        images = self._normalize(images)

        mask_labels = []
        class_labels = []

        for target in batch["targets"]:
            mask_labels.append(target["masks"].float())
            class_labels.append(target["labels"])

        inputs = {
            "pixel_values": images,
            "mask_labels": mask_labels,
            "class_labels": class_labels,
        }

        if self._masked_loss and self._model is not None:
            # Stash per-sample class_mask on the criterion so the overridden
            # loss_labels can read it during the upcoming forward (same order
            # as class_labels). Absent -> None -> no masking for that sample.
            self._model.criterion._class_mask = [
                t.get("class_mask") for t in batch["targets"]
            ]

        return {"inputs": inputs, "targets": batch["targets"]}

    @staticmethod
    def _normalize(images: Tensor) -> Tensor:
        """Apply ImageNet normalisation to a (B, 3, H, W) tensor in [0, 1]."""
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
        return (images - mean) / std

    # ------------------------------------------------------------------
    # Optimizer (backbone / non-backbone split)
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, model: nn.Module, config: TrainConfig | None,
        steps_per_epoch: int | None = None,
    ) -> Any | None:
        """Mask2Former uses a lower LR for the Swin backbone.

        Following the original Detectron2 config: backbone gets lr × 0.1,
        everything else gets the full lr. Override via ``model.extra``:
        ``backbone_lr_factor`` (default 0.1).
        """
        if config is None:
            return None

        backbone_lr_factor = config.model.extra.get("backbone_lr_factor", 0.1)

        backbone_params = []
        other_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # Swin backbone lives under model.model.pixel_level_module.encoder
            if "pixel_level_module.encoder" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "lr": config.lr},
            {"params": backbone_params, "lr": config.lr * backbone_lr_factor},
        ]

        optimizer = torch.optim.AdamW(
            param_groups, lr=config.lr, weight_decay=config.weight_decay,
        )

        logger.info(
            "Mask2Former optimizer: %d backbone params (lr=%.2e), "
            "%d other params (lr=%.2e)",
            len(backbone_params), config.lr * backbone_lr_factor,
            len(other_params), config.lr,
        )

        from training_toolkit.engine.schedulers import build_scheduler
        result = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
        if result is None:
            return optimizer

        scheduler, lr_config = result
        return [optimizer], [{"scheduler": scheduler, **lr_config}]

    def export_model(self, model, output_dir):
        path = output_dir / self.EXPORT_NAME
        model.save_pretrained(str(path))
        logger.info("Exported Mask2Former weights to %s", path)

    def compute_loss(self, outputs: Any, targets: Any) -> dict[str, Tensor]:
        """Extract loss from HuggingFace model outputs.

        The model computes total loss internally. We also extract granular
        components from the loss_dict if available.
        """
        loss_dict = {"loss_total": outputs.loss}

        # HuggingFace Mask2Former stores component losses in the output
        if hasattr(outputs, "loss_dict") and outputs.loss_dict:
            for k, v in outputs.loss_dict.items():
                loss_dict[k] = v

        return loss_dict

    def postprocess_for_inference(
        self, outputs: Any, image_sizes: tuple[int, int] | None = None,
    ) -> list[dict]:
        """Convert Mask2Former outputs to standardized prediction format."""
        import torch.nn.functional as F

        class_logits = outputs.class_queries_logits  # (B, Q, C+1)
        mask_logits = outputs.masks_queries_logits    # (B, Q, H, W)

        batch_size = class_logits.shape[0]
        preds = []

        for i in range(batch_size):
            # Remove no-object class (last), apply softmax
            scores_per_query = class_logits[i].softmax(dim=-1)[:, :-1]  # (Q, C)
            max_scores, pred_labels = scores_per_query.max(dim=-1)       # (Q,), (Q,)

            # Keep all queries above a minimal threshold; the mAP metric
            # handles precision-recall internally.  Using a hard 0.5 threshold
            # would discard all predictions early in training and produce mAP -1.
            keep = max_scores > 0.0
            scores = max_scores[keep]
            labels = pred_labels[keep]
            masks = mask_logits[i][keep].sigmoid()  # (K, H, W)

            # Resize masks to input resolution if needed
            if image_sizes is not None and masks.numel() > 0:
                tgt_h, tgt_w = image_sizes
                if masks.shape[-2:] != (tgt_h, tgt_w):
                    masks = F.interpolate(
                        masks.unsqueeze(1), size=(tgt_h, tgt_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(1)

            masks = masks > 0.5  # binarize

            # Compute boxes from masks
            boxes = self._masks_to_boxes(masks)

            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "masks": masks,
            })

        preds = self._apply_val_det_filter(
            preds, self._val_conf_thres, self._val_max_det, self._val_nms_iou)
        # The head's own mask resolution, read off the real tensor (Mask2Former
        # emits mask logits at its pixel-decoder `common_stride`).
        #
        # `masks_already_native` is False exactly when the loop above upsampled
        # to `image_sizes`. Reaching `native` then costs an interpolate
        # round-trip, which is deliberate: doing it in one step would mean
        # deriving `boxes` from already-downsampled masks, silently moving them
        # out of the input coordinate frame every other consumer assumes. The
        # GPU interpolate is not the bottleneck -- the CPU RLE-encode and mask
        # D2H in `SegmentationMAP.update` are, and those shrink either way.
        native_hw = tuple(mask_logits.shape[-2:])
        return self._apply_val_mask_resolution(
            preds, input_hw=image_sizes, native_hw=native_hw,
            masks_already_native=image_sizes is None,
        )

    def extract_predictions(self, outputs: Any) -> list[dict]:
        """Convenience wrapper without image_sizes."""
        return self.postprocess_for_inference(outputs)

    @staticmethod
    def _masks_to_boxes(masks: Tensor) -> Tensor:
        """Compute bounding boxes from binary masks."""
        if masks.numel() == 0:
            return torch.zeros((0, 4), device=masks.device)

        n = masks.shape[0]
        boxes = torch.zeros(n, 4, device=masks.device)

        for i in range(n):
            y, x = torch.where(masks[i])
            if len(x) == 0:
                continue
            boxes[i] = torch.tensor(
                [x.min(), y.min(), x.max(), y.max()],
                device=masks.device,
                dtype=torch.float32,
            )

        return boxes
