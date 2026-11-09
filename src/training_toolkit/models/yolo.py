"""Ultralytics YOLO model adapter for instance segmentation."""

import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from training_toolkit.configs import ModelConfig, TrainConfig
from training_toolkit.models.base import ModelAdapter

logger = logging.getLogger(__name__)


class YOLOAdapter(ModelAdapter):
    """Adapter for Ultralytics YOLO instance segmentation models.

    Supports YOLO11-seg, YOLO26-seg, and YOLOv8-seg variants. Uses the
    prototype-mask architecture (YOLACT-style): 32 learnable prototypes at
    1/4 resolution combined with per-instance coefficients.

    Backend-specific ``config.model.extra`` keys:

    - ``overlap_mask`` (bool, default False): Use overlap-encoded masks.
    - ``mask_ratio`` (int, default 1): Mask downsampling factor.
    - ``box`` (float): Box loss weight (default 7.5).
    - ``cls`` (float): Classification loss weight (default 0.5).
    - ``dfl`` (float): DFL loss weight (default 1.5).
    - ``backbone_lr_factor`` (float): LR multiplier for backbone layers.
    - ``optimizer`` (str): YOLO-native optimizer (``"auto"``, ``"MuSGD"``, etc.).
    - ``val_conf_thres`` (float): Confidence threshold for val NMS. Defaults to
      ``ModelAdapter.VAL_CONF_THRES_DEFAULT`` -- the same bound the query-based
      adapters apply via ``_apply_val_det_filter``.
    - ``val_max_det`` (int): Max detections per image during val. Defaults to
      ``ModelAdapter.VAL_MAX_DET_DEFAULT`` (the COCO max_dets cap).

      NOTE: YOLO deliberately does NOT call ``_apply_val_det_filter``. It feeds
      both bounds straight into ``non_max_suppression``, which runs BEFORE
      ``process_mask`` -- so masks are never built for detections that will be
      dropped. RF-DETR/Mask2Former cannot do this (their PostProcess fuses
      selection and mask generation), which is why they need the post-hoc
      helper. Same semantics and same defaults, applied at a better point.
    - ``val_mask_downsample`` (``"native"`` | number >= 1, default
      ``"native"``): the resolution the val metric scores masks on, as a
      DIVISOR of the input canvas. ``native`` = the proto resolution
      (input/4), which is what this adapter has always defaulted to;
      ``1`` = full input resolution (the old ``val_full_res_masks: true``).
      Shared with the RF-DETR and Mask2Former adapters — see
      ``models/val_mask_resolution.py`` for why it is relative and not
      absolute pixels. The legacy ``val_full_res_masks`` bool is still
      honoured (deprecated).
    """

    IMGSZ_DIVISOR = 32
    EXPORT_NAME = "model.pt"
    # YOLO-seg prototype masks are emitted at 1/4 of the input resolution (the
    # Proto head upsamples P3/stride-8 features 2x → stride-4). This is fixed by
    # the architecture and is INDEPENDENT of `mask_ratio`. Used to reconstruct the
    # input size from the proto size in the val postprocess (see the mask_ratio bug
    # note in postprocess_for_inference).
    _PROTO_DOWNSAMPLE = 4

    def __init__(self) -> None:
        self._model: nn.Module | None = None
        self._nc: int = 0
        self._nm: int = 32
        self._mask_ratio: int = 1
        self._overlap_mask: bool = False
        self._is_end2end: bool = False
        # Validation postprocessing defaults (overridden in create_model from extra)
        self._val_conf_thres: float = self.VAL_CONF_THRES_DEFAULT
        self._val_max_det: int = self.VAL_MAX_DET_DEFAULT
        self._val_nms_iou: float = self.VAL_NMS_IOU_DEFAULT

    # ------------------------------------------------------------------
    # Resolution helpers
    # ------------------------------------------------------------------

    def get_default_imgsz(self, model_name: str) -> int:
        return 640

    def get_default_gradient_clip_val(self, model_name: str) -> float | None:
        return 10.0  # matches Ultralytics' hardcoded max_norm

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def create_model(self, config: ModelConfig) -> nn.Module:
        from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
        from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

        pretrained = config.pretrained or "auto"
        self._mask_ratio = config.extra.get("mask_ratio", 1)
        self._overlap_mask = config.extra.get("overlap_mask", False)
        self._val_conf_thres = config.extra.get("val_conf_thres", self.VAL_CONF_THRES_DEFAULT)
        self._val_max_det = config.extra.get("val_max_det", self.VAL_MAX_DET_DEFAULT)
        self._val_nms_iou = config.extra.get("val_nms_iou", self.VAL_NMS_IOU_DEFAULT)
        self._init_val_mask_resolution(config.extra)

        if pretrained == "none":
            # From-scratch initialisation
            model = SegmentationModel(
                cfg=config.name + ".yaml",
                nc=config.num_classes,
                verbose=True,
            )
        else:
            # Pretrained: auto-download or custom path
            weight = pretrained if pretrained != "auto" else config.name + ".pt"
            model, ckpt = load_checkpoint(weight)

            # Handle num_classes mismatch via shape-filtered loading
            pretrained_nc = model.model[-1].nc  # nc lives on the head, not the model
            if config.num_classes and pretrained_nc != config.num_classes:
                logger.info(
                    "Rebuilding head: pretrained nc=%d → target nc=%d",
                    pretrained_nc, config.num_classes,
                )
                model = self._rebuild_with_new_nc(model, config.num_classes, ckpt)

        # Build model.args — the loss constructor reads these via attribute access
        args_dict = {**DEFAULT_CFG_DICT}
        args_dict.update({
            "task": "segment",
            "overlap_mask": self._overlap_mask,
            "mask_ratio": self._mask_ratio,
        })
        # Apply user overrides from config.extra (box, cls, dfl, etc.)
        for k, v in config.extra.items():
            args_dict[k] = v
        model.args = IterableSimpleNamespace(**args_dict)

        # Cache metadata
        head = model.model[-1]  # Segment head
        self._nc = head.nc
        self._nm = head.nm if hasattr(head, "nm") else 32
        self._is_end2end = getattr(head, "end2end", False)
        self._model = model

        logger.info(
            "YOLO model ready: nc=%d, nm=%d, end2end=%s, mask_ratio=%d, overlap_mask=%s",
            self._nc, self._nm, self._is_end2end, self._mask_ratio, self._overlap_mask,
        )
        return model

    @staticmethod
    def _rebuild_with_new_nc(
        model: nn.Module, nc: int, ckpt: dict,
    ) -> nn.Module:
        """Rebuild model with a different number of classes, loading compatible weights."""
        from ultralytics.nn.tasks import SegmentationModel

        # Extract the YAML config used to build the original model
        cfg = model.yaml
        cfg["nc"] = nc
        new_model = SegmentationModel(cfg=cfg, nc=nc, verbose=False)

        # Shape-filtered state_dict transfer
        old_state = (ckpt.get("ema") or ckpt["model"]).float().state_dict()
        new_state = new_model.state_dict()
        compatible = {
            k: v for k, v in old_state.items()
            if k in new_state and v.shape == new_state[k].shape
        }
        missing, _ = new_model.load_state_dict(compatible, strict=False)
        logger.info(
            "Transferred %d/%d params; %d randomly initialised",
            len(compatible), len(new_state), len(missing),
        )
        return new_model

    # ------------------------------------------------------------------
    # Criterion
    # ------------------------------------------------------------------

    def create_criterion(self, config: ModelConfig) -> nn.Module | None:
        # Return None here — the YOLO criterion reads device from the model
        # at construction time, but Lightning hasn't moved the model to GPU yet.
        # We lazily create it in compute_loss_with_criterion on first call.
        return None

    def _get_or_create_criterion(self, model: nn.Module):
        """Lazily create the criterion after the model is on device."""
        if not hasattr(self, "_criterion") or self._criterion is None:
            from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss

            if self._is_end2end:
                self._criterion = E2ELoss(model, v8SegmentationLoss)
            else:
                self._criterion = v8SegmentationLoss(model)
            logger.info("YOLO criterion: %s", type(self._criterion).__name__)
        return self._criterion

    # ------------------------------------------------------------------
    # Batch formatting
    # ------------------------------------------------------------------

    def format_batch(self, batch: dict) -> dict:
        images = batch["images"]  # (B, 3, H, W) float [0,1]
        B, _, H, W = images.shape
        device = images.device
        mask_h = H // self._mask_ratio
        mask_w = W // self._mask_ratio

        all_batch_idx = []
        all_cls = []
        all_bboxes = []
        all_masks = []

        for i, t in enumerate(batch["targets"]):
            n = t["labels"].shape[0]
            if n == 0:
                continue

            all_batch_idx.append(
                torch.full((n,), i, dtype=torch.long, device=device)
            )
            all_cls.append(t["labels"].float().unsqueeze(-1))  # (N_i, 1)
            all_bboxes.append(
                self._xyxy_to_cxcywh_norm(t["boxes"], W, H)
            )

            # Downsample per-instance masks
            masks_i = t["masks"].float()  # (N_i, H, W)
            if masks_i.shape[-2:] != (mask_h, mask_w):
                masks_i = F.interpolate(
                    masks_i.unsqueeze(1),
                    size=(mask_h, mask_w),
                    mode="nearest",
                ).squeeze(1)
            all_masks.append(masks_i)

        # Concatenate or build empty tensors
        if all_batch_idx:
            batch_idx = torch.cat(all_batch_idx)
            cls = torch.cat(all_cls)
            bboxes = torch.cat(all_bboxes)
            masks = torch.cat(all_masks)
        else:
            batch_idx = torch.zeros(0, dtype=torch.long, device=device)
            cls = torch.zeros(0, 1, device=device)
            bboxes = torch.zeros(0, 4, device=device)
            masks = torch.zeros(0, mask_h, mask_w, device=device)

        # Build semantic segmentation map for YOLO26's Proto26 head.
        # Per-pixel class index: later (smaller) instances overwrite earlier ones.
        sem_masks = torch.zeros(B, mask_h, mask_w, dtype=torch.float32, device=device)
        for idx in range(batch_idx.shape[0]):
            img_i = batch_idx[idx].item()
            label = cls[idx, 0].long().item()
            sem_masks[img_i][masks[idx] > 0.5] = label

        yolo_batch = {
            "img": images,
            "batch_idx": batch_idx,
            "cls": cls,
            "bboxes": bboxes,
            "masks": masks,
            "sem_masks": sem_masks,
        }

        return {
            "inputs": {"x": images},
            "targets": yolo_batch,
        }

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(self, outputs: Any, targets: Any) -> dict[str, Tensor]:
        from ultralytics.utils.loss import E2ELoss

        criterion = self._get_or_create_criterion(self._model)
        # YOLO loss returns (loss_vector, loss_items_detached)
        # loss_vector is shape (5,) — individual components scaled by batch_size
        # We need to .sum() to get the scalar for backward (same as ultralytics trainer)
        loss_vector, loss_items = criterion(outputs, targets)

        loss_names = ["loss_box", "loss_seg", "loss_cls", "loss_dfl", "loss_semseg"]
        loss_dict = {}
        for name, item in zip(loss_names, loss_items):
            loss_dict[name] = item
        loss_dict["loss_total"] = loss_vector.sum()

        # Advance E2ELoss one2many weight decay schedule
        if isinstance(criterion, E2ELoss):
            criterion.update()

        return loss_dict

    # ------------------------------------------------------------------
    # Inference post-processing
    # ------------------------------------------------------------------

    def _input_hw_from_proto(self, proto_h: int, proto_w: int) -> tuple[int, int]:
        """Reconstruct the model-input (H, W) from the prototype-mask (H, W).

        YOLO-seg prototypes are ALWAYS 1/4 of the input resolution (the Proto head
        is stride-4), INDEPENDENT of ``mask_ratio`` (which only controls GT-mask
        downsampling for the loss, in ``format_batch``). ``process_mask`` needs the
        true input size to scale the box-crop coordinates, so it is ``proto * 4``.
        The previous code used ``proto * mask_ratio``, which only equalled the input
        size at ``mask_ratio == 4``; at ``mask_ratio=1`` it under-sized the crop by
        4x → garbage predicted masks → ~0 segm mAP (silently tanked v3 YOLO).
        """
        return proto_h * self._PROTO_DOWNSAMPLE, proto_w * self._PROTO_DOWNSAMPLE

    def postprocess_for_inference(
        self, outputs: Any, image_sizes: tuple[int, int] | None = None,
    ) -> list[dict]:
        from ultralytics.utils.nms import non_max_suppression
        from ultralytics.utils.ops import process_mask

        # Unpack: eval mode returns ((decoded, proto), raw_dict)
        if isinstance(outputs, tuple) and len(outputs) == 2:
            (decoded, proto) = outputs[0]
        else:
            raise ValueError(
                "postprocess_for_inference expects eval-mode outputs "
                "(model must be in eval mode)"
            )

        # process_mask needs shape=(image_h, image_w) to correctly scale the
        # box crop coordinates, but with upsample=False the masks stay at
        # proto resolution — which IS this head's native mask resolution
        # (protos are always input/4), so the `native` val policy is free here:
        # unlike RF-DETR/M2F, YOLO can simply decline to upsample.
        proto_h, proto_w = proto.shape[2], proto.shape[3]
        img_h, img_w = self._input_hw_from_proto(proto_h, proto_w)

        target = self._val_mask_target_hw(
            (img_h, img_w), (proto_h, proto_w), masks_already_native=True,
        )
        # Let process_mask do the upsample itself when full resolution is what
        # was asked for — it crops at full res, which is strictly better than
        # cropping at proto res and scaling up afterwards.
        upsample = target is not None and tuple(target) == (img_h, img_w)

        if self._is_end2end:
            nms_out = non_max_suppression(
                decoded, conf_thres=self._val_conf_thres, iou_thres=self._val_nms_iou,
                nc=self._nc, max_det=self._val_max_det, end2end=True,
            )
        else:
            nms_out = non_max_suppression(
                decoded, conf_thres=self._val_conf_thres, iou_thres=self._val_nms_iou,
                nc=self._nc, max_det=self._val_max_det,
            )

        mask_h, mask_w = target if target is not None else (proto_h, proto_w)

        preds = []
        for i, det in enumerate(nms_out):
            if det.shape[0] == 0:
                preds.append({
                    "boxes": torch.zeros(0, 4, device=det.device),
                    "scores": torch.zeros(0, device=det.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=det.device),
                    "masks": torch.zeros(
                        0, mask_h, mask_w, dtype=torch.bool, device=det.device
                    ),
                })
                continue

            boxes = det[:, :4]
            scores = det[:, 4]
            labels = det[:, 5].long()
            coeffs = det[:, 6:]

            masks = process_mask(
                proto[i], coeffs, boxes,
                shape=(img_h, img_w), upsample=upsample,
            ).bool()

            # Intermediate resolutions (a divisor other than 1): process_mask
            # only offers proto-res or full-res, so bridge the rest here.
            if not upsample and (mask_h, mask_w) != (proto_h, proto_w):
                masks = self._resize_val_masks_hw(masks, (mask_h, mask_w))

            preds.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
                "masks": masks,
            })

        return preds

    def extract_predictions(self, outputs: Any) -> list[dict]:
        return self.postprocess_for_inference(outputs)

    # ------------------------------------------------------------------
    # Multi-scale training
    # ------------------------------------------------------------------

    def resize_batch(self, formatted: dict, target_size: int) -> dict:
        images = F.interpolate(
            formatted["inputs"]["x"],
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        formatted["inputs"]["x"] = images

        targets = formatted["targets"]
        targets["img"] = images

        if targets["masks"].numel() > 0:
            mask_size = target_size // self._mask_ratio
            targets["masks"] = F.interpolate(
                targets["masks"].unsqueeze(1),
                size=(mask_size, mask_size),
                mode="nearest",
            ).squeeze(1)

        return formatted

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(
        self, model: nn.Module, config: TrainConfig | None,
        steps_per_epoch: int | None = None,
    ) -> Any | None:
        if config is None:
            return None

        # Set epochs on model.args for E2ELoss decay schedule
        if hasattr(model, "args"):
            model.args.epochs = config.epochs

        # Check for YOLO-native optimizer (e.g., MuSGD for YOLO26)
        optimizer = config.model.extra.get("optimizer", None)
        if optimizer is not None:
            return self._build_yolo_native_optimizer(
                model, config, optimizer, steps_per_epoch,
            )

        backbone_lr_factor = config.model.extra.get("backbone_lr_factor", None)
        if backbone_lr_factor is None:
            # No special LR split requested — let the default optimizer handle it
            return None

        # Split params: backbone (low-index layers) vs head/neck
        backbone_params = []
        head_params = []
        backbone_cutoff = self._detect_backbone_cutoff(model)

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if self._is_backbone_param(name, backbone_cutoff):
                backbone_params.append(param)
            else:
                head_params.append(param)

        if not backbone_params:
            logger.warning("No backbone params found; using uniform LR")
            return None

        logger.info(
            "YOLO optimizer: %d backbone params (lr=%.2e), %d head params (lr=%.2e)",
            len(backbone_params), config.lr * backbone_lr_factor,
            len(head_params), config.lr,
        )

        optimizer = torch.optim.AdamW(
            [
                {"params": head_params, "lr": config.lr},
                {"params": backbone_params, "lr": config.lr * backbone_lr_factor},
            ],
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        from training_toolkit.engine.schedulers import build_scheduler

        result = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
        if result is None:
            return optimizer

        scheduler, lr_config = result
        return [optimizer], [{"scheduler": scheduler, **lr_config}]

    def _build_yolo_native_optimizer(
        self, model: nn.Module, config: TrainConfig,
        optimizer_name: str, steps_per_epoch: int | None,
    ) -> Any:
        """Delegate to Ultralytics' build_optimizer for native YOLO optimizers.

        Supports all Ultralytics optimizers including MuSGD (hybrid Muon+SGD
        for YOLO26). Set via ``model.extra.optimizer``.

        Values: ``"auto"`` (Ultralytics picks based on model + iterations),
        ``"MuSGD"``, ``"AdamW"``, ``"SGD"``, etc.
        """
        from functools import partial
        from ultralytics.optim import MuSGD
        from ultralytics.utils.torch_utils import unwrap_model

        iterations = (steps_per_epoch or 1000) * config.epochs

        # For 'auto', Ultralytics picks MuSGD for long runs, AdamW for short
        if optimizer_name.lower() == "auto":
            nc = self._nc or 10
            lr_fit = round(0.002 * 5 / (4 + nc), 6)
            if iterations > 10000:
                name, lr, momentum = "MuSGD", 0.01, 0.9
            else:
                name, lr, momentum = "AdamW", lr_fit, 0.9
            logger.info(
                "YOLO auto-optimizer: selected %s (lr=%.4f, momentum=%.1f) "
                "for %d iterations",
                name, lr, momentum, iterations,
            )
        else:
            name = optimizer_name
            lr = config.lr
            momentum = 0.9

        # Build param groups (4 groups: weight+decay, bn, bias, muon)
        import torch.nn as tnn
        bn = tuple(v for k, v in tnn.__dict__.items() if "Norm" in k)
        g = [{}, {}, {}, {}]
        use_muon = name == "MuSGD"

        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if param.ndim >= 2 and use_muon:
                    g[3][fullname] = param
                elif "bias" in fullname:
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:
                    g[1][fullname] = param
                else:
                    g[0][fullname] = param

        if not use_muon:
            g = [list(x.values()) for x in g[:3]]

        # Build optimizer args per type
        import torch.optim as optim
        decay = config.weight_decay
        if name in ("Adam", "Adamax", "AdamW", "NAdam", "RAdam"):
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name in ("SGD", "MuSGD"):
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise ValueError(f"Unknown YOLO optimizer: {name!r}")

        g_groups = [
            {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"},
            {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"},
            {"params": g[2], **optim_args, "param_group": "bias"},
        ]

        if use_muon:
            import re
            g_groups.append(
                {"params": g[3], **optim_args, "weight_decay": decay,
                 "use_muon": True, "param_group": "muon"}
            )
            # Higher LR for certain head parameters (Ultralytics recipe)
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg")
            g_expanded = []
            for group in g_groups:
                params = group.pop("params")
                if isinstance(params, dict):
                    p1 = [v for k, v in params.items() if pattern.search(k)]
                    p2 = [v for k, v in params.items() if not pattern.search(k)]
                    if p1:
                        g_expanded.append({"params": p1, **group, "lr": lr * 3})
                    if p2:
                        g_expanded.append({"params": p2, **group})
                else:
                    g_expanded.append({"params": params, **group})
            g_groups = g_expanded

        muon_ratio, sgd_ratio = (0.2, 1.0)
        optimizer = getattr(
            optim, name, partial(MuSGD, muon=muon_ratio, sgd=sgd_ratio)
        )(params=g_groups)

        logger.info(
            "YOLO native optimizer: %s (lr=%.4f, weight_decay=%.5f)",
            type(optimizer).__name__, lr, decay,
        )

        from training_toolkit.engine.schedulers import build_scheduler

        result = build_scheduler(optimizer, config, steps_per_epoch=steps_per_epoch)
        if result is None:
            return optimizer

        scheduler, lr_config = result
        return [optimizer], [{"scheduler": scheduler, **lr_config}]

    @staticmethod
    def _detect_backbone_cutoff(model: nn.Module) -> int:
        """Find the index of the first non-backbone layer in model.model.

        YOLO architectures define the backbone as the first continuous block
        of layers before the first Concat or Detect module.
        """
        from ultralytics.nn.modules.conv import Concat

        for i, m in enumerate(model.model):
            if isinstance(m, Concat):
                return i
        return 10  # conservative fallback

    @staticmethod
    def _is_backbone_param(name: str, cutoff: int) -> bool:
        """Check if a parameter belongs to a backbone layer by its name."""
        # Parameter names look like "model.0.conv.weight", "model.5.m.0.cv1.weight"
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model" and parts[1].isdigit():
            return int(parts[1]) < cutoff
        return False

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, model: nn.Module, output_dir: Path) -> None:
        path = output_dir / self.EXPORT_NAME
        ckpt = {
            "model": model,
            "train_args": dict(model.args) if hasattr(model, "args") else {},
        }
        torch.save(ckpt, str(path))
        logger.info("Exported YOLO weights to %s", path)

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _xyxy_to_cxcywh_norm(boxes: Tensor, img_w: int, img_h: int) -> Tensor:
        """Convert xyxy absolute boxes to cxcywh normalised [0, 1]."""
        if boxes.numel() == 0:
            return boxes
        x1, y1, x2, y2 = boxes.unbind(-1)
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        return torch.stack([cx, cy, w, h], dim=-1)
