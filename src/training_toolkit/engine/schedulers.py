"""Shared LR scheduler construction with warmup support."""

import logging

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
    LRScheduler,
)

from training_toolkit.configs import TrainConfig

logger = logging.getLogger(__name__)


def build_scheduler(
    optimizer: Optimizer,
    config: TrainConfig,
    steps_per_epoch: int | None = None,
) -> tuple[LRScheduler, dict] | None:
    """Build an LR scheduler with optional linear warmup.

    Args:
        optimizer: The optimizer (any number of param groups).
        config: Training configuration (reads ``scheduler``, ``warmup_epochs``,
            ``epochs``).
        steps_per_epoch: Required only for ``one_cycle`` scheduler (total
            batches per epoch).

    Returns:
        A ``(scheduler, lr_scheduler_config)`` tuple for Lightning, or None
        if ``config.scheduler == "none"`` and there is no warmup.
    """
    warmup = config.warmup_epochs
    total = config.epochs
    key = config.scheduler.lower().replace("-", "").replace("_", "")

    if key == "none" and warmup <= 0:
        return None

    # OneCycleLR has its own warmup — skip our warmup wrapper
    if key == "onecycle":
        scheduler = _build_one_cycle(optimizer, config, steps_per_epoch)
        # OneCycleLR steps per batch, not per epoch
        return scheduler, {"interval": "step"}

    # Build main scheduler (operates over post-warmup epochs)
    main_epochs = max(1, total - warmup)
    main_scheduler = _build_main(key, optimizer, main_epochs)

    if warmup <= 0:
        if main_scheduler is None:
            return None
        return main_scheduler, {"interval": "epoch"}

    # Linear warmup: LR ramps from ~0 to base LR over warmup_epochs
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-5,
        end_factor=1.0,
        total_iters=warmup,
    )

    if main_scheduler is None:
        # warmup only, constant after
        return warmup_scheduler, {"interval": "epoch"}

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup],
    )
    return scheduler, {"interval": "epoch"}


def _build_main(
    key: str,
    optimizer: Optimizer,
    main_epochs: int,
) -> LRScheduler | None:
    """Build the main (post-warmup) scheduler."""
    if key == "cosine":
        return CosineAnnealingLR(optimizer, T_max=main_epochs)

    elif key == "linear":
        return LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=main_epochs,
        )

    elif key == "none":
        return None

    else:
        raise ValueError(
            f"Unknown scheduler: {key!r} "
            f"(expected cosine, linear, one_cycle, or none)"
        )


def _build_one_cycle(
    optimizer: Optimizer,
    config: TrainConfig,
    steps_per_epoch: int | None,
) -> OneCycleLR:
    """Build a OneCycleLR scheduler (has its own internal warmup)."""
    if steps_per_epoch is None:
        raise ValueError(
            "one_cycle scheduler requires steps_per_epoch — "
            "this is a bug, please report it"
        )
    max_lrs = [group["lr"] for group in optimizer.param_groups]
    return OneCycleLR(
        optimizer,
        max_lr=max_lrs,
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
    )
