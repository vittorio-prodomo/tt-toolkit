"""Early stopping callback configured from TrainConfig."""

from pytorch_lightning.callbacks import EarlyStopping

from training_toolkit.configs import TrainConfig


def build_early_stopping(config: TrainConfig) -> EarlyStopping:
    """Create an EarlyStopping callback from config."""
    return EarlyStopping(
        monitor=config.monitor_metric,
        patience=config.patience,
        mode="max",  # mAP is higher-is-better
        verbose=True,
    )
