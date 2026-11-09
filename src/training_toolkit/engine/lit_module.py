"""Generic Lightning module for instance segmentation training."""

import logging
import random
from typing import Any

import numpy as np
import pytorch_lightning as L
import torch
from torch import Tensor

from training_toolkit.configs import TrainConfig
from training_toolkit.metrics import SegmentationMAP, TideAnalyzer
from training_toolkit.metrics.confusion import update_confusion_matrix
from training_toolkit.models.base import ModelAdapter

logger = logging.getLogger(__name__)


class SegmentationLitModule(L.LightningModule):
    """Model-agnostic Lightning module for instance segmentation.

    Delegates all model-specific logic to a :class:`ModelAdapter`. The training
    loop, validation, metrics, optimizer setup, and logging are all generic.
    """

    def __init__(
        self,
        adapter: ModelAdapter,
        config: TrainConfig,
        class_names: dict[int, str] | None = None,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.config = config

        self.model = adapter.create_model(config.model)
        if config.masked_loss:
            adapter.enable_masked_loss()  # raises if the adapter doesn't support it
        if config.region_class_mask:
            adapter.enable_region_class_mask()
        if getattr(config, "size_band_ignore", False):
            adapter.enable_size_band_ignore(getattr(config, "ignore_ioa_thresh", 0.5))
        self.criterion = adapter.create_criterion(config.model)

        iou_type = "segm" if adapter.produces_masks else "bbox"
        self.val_map = SegmentationMAP(class_names=class_names or {}, iou_type=iou_type)
        self.val_tide: TideAnalyzer | None = (
            TideAnalyzer(class_names=class_names or {}, iou_type=iou_type)
            if config.val_tide
            else None
        )

        # Multi-scale training
        aug = config.augmentation
        if aug.multi_scale:
            self._ms_sizes = adapter.compute_multi_scale_sizes(aug, config.model.name)
            logger.info("Multi-scale training enabled: %s", self._ms_sizes)
        else:
            self._ms_sizes = []

        # Buffers for DiagnosticsCallback to consume
        self._captured_val_batches: list[dict] = []
        self._captured_val_preds: list[list[dict]] = []
        self._max_capture: int = 3

        # PR curve stash: updated only on metric improvement
        self._pr_curves: dict | None = None
        self._best_monitored: float | None = None

        # Confusion matrix: accumulated per validation epoch
        self._confusion_data: np.ndarray | None = None
        self._confusion_matrix: dict | None = None

        # TIDE error analysis: stashed on metric improvement
        self._tide_results: dict | None = None

    def setup(self, stage: str) -> None:
        """Called by Lightning before device placement."""
        if stage == "fit":
            self.val_map = self.val_map.to(self.device)

    def on_train_start(self) -> None:
        """Called by Lightning after device placement, before the first step."""
        # Eagerly create the criterion now that the model is on device.
        # This avoids lazy creation during the first training step,
        # which would print a log line that collides with the progress bar.
        if hasattr(self.adapter, "_get_or_create_criterion"):
            self.adapter._get_or_create_criterion(self.model)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        formatted = self.adapter.format_batch(batch)

        if self._ms_sizes:
            size = random.choice(self._ms_sizes)
            formatted = self.adapter.resize_batch(formatted, size)

        outputs = self.model(**formatted["inputs"])

        # Handle external criterion (RF-DETR) vs internal loss (Mask2Former)
        if self.criterion is not None and hasattr(self.adapter, 'compute_loss_with_criterion'):
            loss_dict = self.adapter.compute_loss_with_criterion(
                self.criterion, outputs, formatted["targets"]
            )
        else:
            loss_dict = self.adapter.compute_loss(outputs, formatted["targets"])

        loss = self.adapter.on_before_backward(loss_dict["loss_total"])

        for k, v in loss_dict.items():
            self.log(
                f"train/{k}",
                v,
                prog_bar=(k == "loss_total"),
                batch_size=batch["images"].size(0),
            )

        return loss

    def on_validation_epoch_start(self) -> None:
        self._captured_val_batches.clear()
        self._captured_val_preds.clear()
        nc = len(self.val_map.class_names)
        if nc > 0:
            self._confusion_data = np.zeros((nc + 1, nc + 1), dtype=np.int64)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        formatted = self.adapter.format_batch(batch)
        outputs = self.model(**formatted["inputs"])

        # Log validation losses (diagnostic — not used for backprop).
        # Wrapped in try/except so loss failures never break validation.
        try:
            # Save criterion state to prevent side effects (e.g. YOLO's
            # E2ELoss step counter advancing during validation).
            _saved = {}
            for obj in (self.criterion, getattr(self.adapter, '_criterion', None)):
                if obj is not None and hasattr(obj, '__dict__'):
                    _saved[id(obj)] = (obj, obj.__dict__.copy())

            if self.criterion is not None and hasattr(self.adapter, 'compute_loss_with_criterion'):
                loss_dict = self.adapter.compute_loss_with_criterion(
                    self.criterion, outputs, formatted["targets"]
                )
            else:
                loss_dict = self.adapter.compute_loss(outputs, formatted["targets"])
            for k, v in loss_dict.items():
                self.log(f"val/{k}", v, batch_size=batch["images"].size(0), sync_dist=True)

            for obj, state in _saved.values():
                obj.__dict__.update(state)
        except Exception:
            pass  # Don't let loss computation failures break validation

        _, _, img_h, img_w = batch["images"].shape
        preds = self.adapter.postprocess_for_inference(outputs, image_sizes=(img_h, img_w))
        targets = self._format_targets_for_metrics(batch["targets"], preds)

        self.val_map.update(preds, targets)

        if self.val_tide is not None:
            self.val_tide.update(preds, targets, batch_idx=batch_idx)

        # Accumulate confusion matrix
        if self._confusion_data is not None:
            update_confusion_matrix(self._confusion_data, preds, targets)

        # Capture batches for val visualization (detach to free compute graph)
        if len(self._captured_val_batches) < self._max_capture:
            cpu_batch = {
                "images": batch["images"].detach().cpu(),
                "targets": [
                    {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in t.items()}
                    for t in batch["targets"]
                ],
            }
            cpu_preds = [
                {k: v.detach().cpu() for k, v in p.items()} for p in preds
            ]
            self._captured_val_batches.append(cpu_batch)
            self._captured_val_preds.append(cpu_preds)

    def on_validation_epoch_end(self) -> None:
        # Lightning's pre-fit sanity check runs this hook on a couple of
        # batches seconds after a launch/resume, and its metrics block has
        # repeatedly been mistaken for a real validation when monitoring runs.
        # Label it explicitly, skip best-metric/PR-curve tracking, and reset
        # accumulated state so nothing leaks into epoch 0's real val.
        # (``self.trainer`` raises when unattached — read the backing attr.)
        trainer = getattr(self, "_trainer", None)
        if trainer is not None and trainer.sanity_checking:
            logger.info(
                "Sanity-check val complete — startup smoke test only, "
                "NOT a validation (metrics not meaningful)."
            )
            self.val_map.reset()
            if self.val_tide is not None:
                self.val_tide.reset()
            return

        raw = self.val_map.compute()
        formatted = self.val_map.format_results(raw)

        for k, v in formatted.items():
            self.log(f"val/{k}", v, prog_bar=(k == "map"), sync_dist=True)

        if formatted.get("map") is not None:
            # Flush a newline so the log doesn't collide with Rich's progress bar.
            import sys
            sys.stdout.write("\n")
            sys.stdout.flush()
            if self.config.verbose_metrics:
                logger.info(
                    "Validation results:\n%s",
                    self.val_map.format_verbose(raw),
                )
            else:
                logger.info(
                    "Validation — mAP: %.4f | mAP@50: %.4f | mAP@75: %.4f | F1: %.4f",
                    formatted.get("map", 0),
                    formatted.get("map_50", 0),
                    formatted.get("map_75", 0),
                    formatted.get("f1_best", 0),
                )

        tide_result = self._compute_tide()
        if tide_result is not None:
            for err_name, dap in tide_result["aggregate"].items():
                # TIDE values are on a 0-100 scale (percentage points).
                self.log(
                    f"val/tide_dap/{err_name.lower()}", float(dap),
                    sync_dist=True,
                )

        # Stash PR curves on metric improvement
        monitor_key = self.config.monitor_metric.replace("val/", "")
        current_val = formatted.get(monitor_key)
        if current_val is not None:
            higher_is_better = "loss" not in monitor_key
            improved = (
                self._best_monitored is None
                or (higher_is_better and current_val > self._best_monitored)
                or (not higher_is_better and current_val < self._best_monitored)
            )
            if improved:
                self._best_monitored = current_val
                curves = self.val_map.extract_pr_curves(raw)
                if curves is not None:
                    curves["epoch"] = self.current_epoch
                    self._pr_curves = curves
                if self._confusion_data is not None:
                    class_names = list(self.val_map.class_names.values())
                    self._confusion_matrix = {
                        "matrix": self._confusion_data.copy(),
                        "class_names": class_names,
                        "epoch": self.current_epoch,
                    }
                if tide_result is not None:
                    tide_result["epoch"] = self.current_epoch
                    self._tide_results = tide_result

        self.val_map.reset()

    def _compute_tide(self) -> dict | None:
        """Run TIDE error analysis if enabled. Failure here is non-fatal —
        TIDE adds diagnostics, it must not break training.

        Skipped during the sanity-check val (a fresh-init model yields zero
        detections there, on which tidecv divides by zero) and quiet when no
        predictions were accumulated this epoch; accumulated state is reset
        on every path so sanity/failed batches never leak into the next epoch.
        """
        if self.val_tide is None:
            return None
        try:
            # ``self.trainer`` raises when unattached (see configure_optimizers).
            trainer = getattr(self, "_trainer", None)
            if trainer is not None and trainer.sanity_checking:
                return None
            result = self.val_tide.compute()
            if result is None:
                logger.info("TIDE skipped: no predictions accumulated this epoch")
            return result
        except Exception:
            logger.warning("TIDE compute failed", exc_info=True)
            return None
        finally:
            self.val_tide.reset()

    def configure_optimizers(self) -> Any:
        from training_toolkit.engine.schedulers import build_scheduler

        # In Lightning ≥ 2.0 ``self.trainer`` *raises* RuntimeError when no
        # Trainer is attached — read the backing attribute so this method
        # remains callable outside Lightning's fit() lifecycle (smoke tests,
        # one-off instantiation in scripts, etc.).
        trainer = getattr(self, "_trainer", None)
        steps_per_epoch = (
            int(trainer.estimated_stepping_batches) // self.config.epochs
            if trainer is not None else None
        )
        custom = self.adapter.configure_optimizers(
            self.model, self.config, steps_per_epoch=steps_per_epoch,
        )
        if custom is not None:
            return custom

        if self.config.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.lr,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

        result = build_scheduler(
            optimizer, self.config,
            steps_per_epoch=steps_per_epoch,
        )
        if result is None:
            return optimizer

        scheduler, lr_config = result
        return [optimizer], [{"scheduler": scheduler, **lr_config}]

    def _format_targets_for_metrics(
        self, targets: list[dict], preds: list[dict] | None = None,
    ) -> list[dict]:
        """Ensure targets are in TorchMetrics format.

        When predictions have lower-resolution masks (e.g. YOLO 1/4 proto
        resolution), GT masks are downsampled to match so IoU is computed
        at the same resolution.
        """
        import torch.nn.functional as F

        formatted = []
        for i, t in enumerate(targets):
            masks = t["masks"]
            if masks.dtype != torch.bool:
                masks = masks.bool()

            # Downsample GT masks to match prediction mask resolution
            if preds and i < len(preds) and preds[i]["masks"].numel() > 0:
                pred_h, pred_w = preds[i]["masks"].shape[-2:]
                if masks.numel() > 0 and masks.shape[-2:] != (pred_h, pred_w):
                    masks = F.interpolate(
                        masks.float().unsqueeze(1),
                        size=(pred_h, pred_w),
                        mode="nearest",
                    ).squeeze(1).bool()

            out = {
                "boxes": t["boxes"],
                "labels": t["labels"],
                "masks": masks,
            }
            # Propagate optional COCO iscrowd flags so MeanAveragePrecision
            # ignores crowd/sliver GT (not FN-bait; matched preds not FP)
            # instead of treating a missing key as background. Absent input
            # keeps the key absent -> torchmetrics' own all-zeros default.
            if "iscrowd" in t:
                out["iscrowd"] = t["iscrowd"].to(torch.long)
            formatted.append(out)
        return formatted
