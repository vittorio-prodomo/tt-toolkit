"""Callback to detect and stop diverged training runs (NaN loss, invalid mAP)."""

import logging
import math

import pytorch_lightning as L

logger = logging.getLogger(__name__)


class DivergenceDetector(L.Callback):
    """Stop training if the model has diverged (NaN loss or invalid metrics).

    Monitors the training loss and validation metric each epoch. If either
    is NaN/inf or the metric is <= 0 for ``patience`` consecutive epochs,
    training is stopped early.

    Args:
        monitor: Metric key to check (default: ``"val/map"``).
        patience: Number of consecutive bad epochs before stopping.
    """

    def __init__(self, monitor: str = "val/map", patience: int = 3) -> None:
        self.monitor = monitor
        self.patience = patience
        self._consecutive_bad = 0

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        # Check training loss for NaN
        loss = trainer.callback_metrics.get("train/loss_total")
        if loss is not None and (math.isnan(loss) or math.isinf(loss)):
            self._consecutive_bad += 1
            logger.warning(
                "Divergence detector: train/loss_total is %s "
                "(bad epoch %d/%d)",
                loss, self._consecutive_bad, self.patience,
            )
            if self._consecutive_bad >= self.patience:
                logger.error(
                    "Training diverged — NaN/inf loss for %d consecutive "
                    "epochs. Stopping.",
                    self.patience,
                )
                trainer.should_stop = True
            return

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return

        val = float(metric)
        is_bad = math.isnan(val) or math.isinf(val) or val < 0

        if is_bad:
            self._consecutive_bad += 1
            logger.warning(
                "Divergence detector: %s = %s (bad epoch %d/%d)",
                self.monitor, val, self._consecutive_bad, self.patience,
            )
            if self._consecutive_bad >= self.patience:
                logger.error(
                    "Training diverged — %s invalid for %d consecutive "
                    "epochs. Stopping.",
                    self.monitor, self.patience,
                )
                trainer.should_stop = True
        else:
            # Reset counter on any good epoch
            if self._consecutive_bad > 0:
                logger.info(
                    "Divergence detector: %s = %.4f — recovered, "
                    "resetting counter",
                    self.monitor, val,
                )
            self._consecutive_bad = 0
