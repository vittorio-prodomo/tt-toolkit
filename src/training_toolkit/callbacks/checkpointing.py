"""Model checkpointing callback configured from TrainConfig."""

from pathlib import Path

from pytorch_lightning.callbacks import ModelCheckpoint

from training_toolkit.configs import TrainConfig


class LatestLastCheckpoint(ModelCheckpoint):
    """``ModelCheckpoint`` whose ``last.ckpt`` always tracks the LATEST epoch.

    Stock PyTorch-Lightning (2.6) writes ``last.ckpt`` only on epochs where a
    top-k checkpoint was *also* saved — the ``on_validation_end`` /
    ``on_train_epoch_end`` guard is ``self._last_global_step_saved ==
    trainer.global_step``. So once a run plateaus and later epochs stop cracking
    the top-k, ``last.ckpt`` freezes at the last top-k epoch rather than the
    newest weights, and a ``--resume last`` silently rewinds to that epoch.

    This subclass forces a ``last.ckpt`` write every epoch, independent of
    top-k, so ``last.ckpt`` == the latest epoch (and a resume continues from it).
    Engages only when ``save_last is True`` (a real file); ``save_last="link"``
    and ``False`` are left to PL. Couples to PL 2.6 private API — the integration
    test (test_latest_last_checkpoint.py) guards against version drift.
    """

    def on_validation_end(self, trainer, pl_module) -> None:
        super().on_validation_end(trainer, pl_module)
        if not self._should_save_on_train_epoch_end(trainer):
            self._ensure_last_is_latest(trainer)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        super().on_train_epoch_end(trainer, pl_module)
        if self._should_save_on_train_epoch_end(trainer):
            self._ensure_last_is_latest(trainer)

    def _ensure_last_is_latest(self, trainer) -> None:
        """Write last.ckpt now if this epoch's save was skipped (not top-k)."""
        if self.save_last is not True:  # respect "link"/False/None → leave to PL
            return
        if self._should_skip_saving_checkpoint(trainer):
            return
        # A checkpoint already saved this step (a top-k epoch) → last.ckpt current.
        if self._last_global_step_saved == trainer.global_step:
            return
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))


def build_checkpointing(config: TrainConfig, train_only: bool = False) -> ModelCheckpoint:
    """Create a ModelCheckpoint callback from config.

    Args:
        config: Training configuration.
        train_only: When True (no validation split), save only the last
            checkpoint each epoch instead of monitoring a val metric.
    """
    dirpath = Path(config.project_dir) / config.run_name / "checkpoints"

    if train_only:
        return LatestLastCheckpoint(
            dirpath=str(dirpath),
            save_last=True,
            every_n_epochs=1,
            filename="epoch={epoch:03d}",
            auto_insert_metric_name=False,
            verbose=False,
        )

    kwargs = {
        "dirpath": str(dirpath),
        "monitor": config.monitor_metric,
        "mode": "max",
        "save_top_k": config.save_top_k,
        "save_last": True,
        "filename": "epoch={epoch:03d}-map={val/map:.4f}",
        "auto_insert_metric_name": False,
        "verbose": True,
    }

    if config.save_period > 0:
        kwargs["every_n_epochs"] = config.save_period

    return LatestLastCheckpoint(**kwargs)
