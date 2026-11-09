"""Optuna pruning callback for Lightning."""

import pytorch_lightning as L


class OptunaPruningCallback(L.Callback):
    """Reports intermediate metrics to Optuna and prunes unpromising trials.

    This callback runs at the end of each validation epoch, reads the monitored
    metric from the trainer, and reports it to the Optuna trial.  If the trial
    should be pruned (e.g. by MedianPruner), it raises ``optuna.TrialPruned``.
    """

    def __init__(self, trial, metric: str = "val/map") -> None:
        super().__init__()
        self._trial = trial
        self._metric = metric

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        import optuna

        # Skip the sanity-check validation that Lightning runs before training.
        # Reporting an untrained-model metric here would cause the pruner to
        # compare it against post-training values from completed trials,
        # resulting in immediate (unfair) pruning.
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        value = trainer.callback_metrics.get(self._metric)
        if value is None:
            return

        self._trial.report(float(value), step=epoch)
        if self._trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")
