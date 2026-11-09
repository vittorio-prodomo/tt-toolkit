"""TTY-aware training progress reporting.

A live progress bar (RichProgressBar) is a TTY artifact: it updates in place
with ANSI/carriage-return control codes and renders nothing useful once stdout
is redirected to a file, a pipe, or a CI/nohup log. For those non-interactive
runs we instead emit *throttled, plain-text* progress records through the
standard logging system, which the run's FileHandler already captures — so the
logfile shows readable train/val progress (and you can tell "slow" from "hung").

Use :func:`select_progress_callback` to pick the right reporter for the run.
"""

import logging
import time

import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar

logger = logging.getLogger(__name__)


class FileProgressLogger(pl.Callback):
    """Emit throttled plain-text train/val progress via the logging system.

    Logs at most one line per ``every_n_batches`` batches OR per ``every_s``
    seconds (whichever comes first), on the global-zero rank only. It is a plain
    ``Callback`` (not a ``ProgressBar``), so it coexists with
    ``enable_progress_bar=False`` and never tries to render a live widget.
    """

    def __init__(self, every_n_batches: int = 50, every_s: float = 30.0):
        if every_n_batches < 1:
            raise ValueError(
                f"every_n_batches must be >= 1, got {every_n_batches}"
            )
        self.every_n_batches = every_n_batches
        self.every_s = every_s
        self._last_emit = 0.0

    def _should_emit(self, batch_idx: int, now: float) -> bool:
        """True on every Nth batch (incl. the first) or once ``every_s`` elapses."""
        return (batch_idx % self.every_n_batches == 0) or (
            now - self._last_emit >= self.every_s
        )

    @staticmethod
    def _select_metrics(callback_metrics, stage) -> dict:
        """One representative loss for the current stage. callback_metrics holds ~20
        loss keys plus the *other* stage's stale values, so filter to ``<stage>/`` and
        prefer the aggregate (``loss_total`` / ``loss``), else any stage loss, else none."""
        prefix = f"{stage}/"
        stage_losses = {
            k: float(v) for k, v in callback_metrics.items()
            if k.startswith(prefix) and "loss" in k.lower()
        }
        for key in (f"{stage}/loss_total", f"{stage}/loss"):
            if key in stage_losses:
                return {key: stage_losses[key]}
        if stage_losses:
            k = next(iter(stage_losses))
            return {k: stage_losses[k]}
        return {}

    @staticmethod
    def _format(epoch, stage, batch_idx, total, metrics) -> str:
        total_s = str(total) if isinstance(total, int) else "?"
        line = f"epoch {epoch} | {stage} {batch_idx + 1}/{total_s}"
        if metrics:
            line += " | " + " ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        return line

    def _emit(self, trainer, stage: str, batch_idx: int, total) -> None:
        if not trainer.is_global_zero:
            return
        now = time.monotonic()
        if not self._should_emit(batch_idx, now):
            return
        self._last_emit = now
        metrics = self._select_metrics(trainer.callback_metrics, stage)
        logger.info(self._format(trainer.current_epoch, stage, batch_idx, total, metrics))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._emit(trainer, "train", batch_idx, trainer.num_training_batches)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        total = trainer.num_val_batches
        if isinstance(total, (list, tuple)):
            total = total[dataloader_idx] if dataloader_idx < len(total) else None
        self._emit(trainer, "val", batch_idx, total)


def select_progress_callback(is_tty: bool):
    """Return ``(progress_callback, enable_progress_bar)`` for the run context.

    TTY → ``RichProgressBar`` (live console bar) with Lightning's bar enabled.
    Non-TTY → ``FileProgressLogger`` (throttled plain-text via logging) with
    Lightning's own bar disabled (passing a ``ProgressBar`` while
    ``enable_progress_bar=False`` would raise, hence the plain callback here).
    """
    if is_tty:
        return RichProgressBar(), True
    return FileProgressLogger(), False
