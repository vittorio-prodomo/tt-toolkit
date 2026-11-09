"""Optuna hyperparameter sweep wrapper around the training loop."""

import copy
import logging
from datetime import datetime
from functools import reduce
from pathlib import Path
from typing import Any

import torch

from training_toolkit.callbacks.optuna_pruning import OptunaPruningCallback
from training_toolkit.configs import TrainConfig
from training_toolkit.configs.sweep_config import SweepConfig, ParamSpace
from training_toolkit.engine.trainer import train

logger = logging.getLogger(__name__)


def sweep(
    base_config: TrainConfig,
    sweep_config: SweepConfig,
    resume_from: str | None = None,
) -> None:
    """Run an Optuna hyperparameter sweep.

    Each trial copies *base_config*, applies Optuna-suggested values from the
    search space, then runs a full ``train()`` call.  The monitored metric is
    reported back to Optuna for pruning and study-level optimization.

    Args:
        base_config: Base training configuration (values not in the search
            space remain fixed across trials).
        sweep_config: Sweep-specific settings (search space, n_trials, etc.).
        resume_from: Passed through to ``train()`` for checkpoint resuming.
    """
    import optuna

    # Pruner
    pruner = _build_pruner(sweep_config.pruner, sweep_config.pruner_n_warmup_epochs)

    # Resolve storage and study name
    storage = sweep_config.storage
    study_name = sweep_config.study_name

    if storage is not None:
        # Auto-generate study name if storage is enabled but no name given
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            study_name = f"{base_config.run_name}_{timestamp}"
            logger.info("Auto-generated study name: %s", study_name)

        # Auto-place SQLite DB inside project_dir
        if storage == "auto":
            sweep_dir = Path(base_config.project_dir) / ".optuna"
            sweep_dir.mkdir(parents=True, exist_ok=True)
            db_path = sweep_dir / f"{study_name}.db"
            storage = f"sqlite:///{db_path.resolve()}"

        logger.info("Optuna storage: %s", storage)

    # Create or load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=sweep_config.direction,
        pruner=pruner,
        load_if_exists=True,
    )

    logger.info(
        "[bold]Starting Optuna sweep: %d trials, metric=%s (%s)[/bold]",
        sweep_config.n_trials,
        sweep_config.metric,
        sweep_config.direction,
        extra={"markup": True},
    )

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)

        # Apply Optuna suggestions to config
        for param_path, space in sweep_config.search_space.items():
            value = _suggest(trial, param_path, space)
            _set_nested(config, param_path, value)

        # Give each trial a unique run name
        config.run_name = f"{base_config.run_name}_trial_{trial.number}"

        # Only run dataset analysis on the first trial — stats don't change
        if trial.number > 0:
            config.analysis = False

        logger.info(
            "[bold]Trial %d[/bold] — %s",
            trial.number,
            {p: trial.params[p] for p in trial.params},
            extra={"markup": True},
        )

        pruning_cb = OptunaPruningCallback(trial, metric=sweep_config.metric)

        try:
            result = train(
                config,
                resume_from=resume_from,
                extra_callbacks=[pruning_cb],
            )
        finally:
            # Force cleanup of GPU state before the next trial.
            # Without this, DataLoader workers (forked from this process)
            # inherit stale CUDA tensors whose destructors crash with
            # "CUDA error: initialization error" in the child process.
            import gc

            gc.collect()
            torch.cuda.empty_cache()

            # Close the W&B run so the next trial gets a fresh one.
            # Without this, WandbLogger reuses the existing run and rejects
            # config updates with different hyperparameter values.
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except ImportError:
                pass

        if result is None:
            raise optuna.TrialPruned("No validation metric available")

        return result

    # Optuna → W&B integration: log study-level metrics (parameter importance,
    # optimization history, trial comparison) to a dedicated W&B run.
    callbacks = []
    if "wandb" in base_config.loggers:
        try:
            from optuna.integration.wandb import WeightsAndBiasesCallback

            wandb_cb = WeightsAndBiasesCallback(
                metric_name=sweep_config.metric,
                as_multirun=True,
            )
            callbacks.append(wandb_cb)
            logger.info("Optuna-W&B integration enabled")
        except (ImportError, ModuleNotFoundError):
            logger.debug("optuna wandb integration not available — skipping")

    # Optuna → MLflow integration: log trial params and metrics to the same
    # MLflow tracking server used by the per-trial Lightning MLFlowLogger.
    if "mlflow" in base_config.loggers:
        try:
            from optuna.integration.mlflow import MLflowCallback

            mlflow_dir = Path(base_config.project_dir) / ".mlflow"
            mlflow_dir.mkdir(parents=True, exist_ok=True)
            db_path = mlflow_dir.resolve() / "mlflow.db"
            tracking_uri = f"sqlite:///{db_path}"

            mlflow_cb = MLflowCallback(
                tracking_uri=tracking_uri,
                metric_name=sweep_config.metric,
                create_experiment=True,
            )
            callbacks.append(mlflow_cb)
            logger.info("Optuna-MLflow integration enabled (tracking: %s)", tracking_uri)
        except (ImportError, ModuleNotFoundError):
            logger.debug("optuna mlflow integration not available — skipping")

    # When resuming, treat n_trials as the desired *total* — only run what's left.
    # Only count completed/pruned trials; failed ones don't count toward the goal.
    existing = sum(
        1
        for t in study.trials
        if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    )
    remaining = sweep_config.n_trials - existing
    if remaining <= 0:
        logger.info(
            "Study already has %d/%d trials — nothing to do.",
            existing,
            sweep_config.n_trials,
        )
        return

    if existing > 0:
        logger.info(
            "Resuming study with %d existing trials — running %d more to reach %d.",
            existing,
            remaining,
            sweep_config.n_trials,
        )

    study.optimize(objective, n_trials=remaining, callbacks=callbacks)

    # Summary
    logger.info("[bold green]Sweep complete![/bold green]", extra={"markup": True})
    logger.info("Best trial: #%d", study.best_trial.number)
    logger.info("Best value: %.4f", study.best_trial.value)
    logger.info("Best params: %s", study.best_trial.params)


def _suggest(trial: Any, name: str, space: ParamSpace) -> Any:
    """Map a ParamSpace to an Optuna trial suggestion."""
    if space.type == "float":
        kwargs: dict[str, Any] = {"name": name, "low": space.low, "high": space.high}
        if space.log:
            kwargs["log"] = True
        if space.step is not None:
            kwargs["step"] = space.step
        return trial.suggest_float(**kwargs)
    elif space.type == "int":
        kwargs = {"name": name, "low": int(space.low), "high": int(space.high)}
        if space.log:
            kwargs["log"] = True
        if space.step is not None:
            kwargs["step"] = int(space.step)
        return trial.suggest_int(**kwargs)
    elif space.type == "categorical":
        return trial.suggest_categorical(name, space.choices)
    else:
        raise ValueError(f"Unknown param type: {space.type!r}")


def _set_nested(obj: Any, dotted_path: str, value: Any) -> None:
    """Set a value on a nested object using a dotted path.

    Example: _set_nested(config, "augmentation.mosaic_p", 0.3)
    sets config.augmentation.mosaic_p = 0.3
    """
    parts = dotted_path.split(".")
    target = reduce(getattr, parts[:-1], obj)
    setattr(target, parts[-1], value)


def _build_pruner(name: str, n_warmup_steps: int = 5) -> Any:
    """Build an Optuna pruner by name.

    Args:
        name: Pruner type ("median", "hyperband", or "none").
        n_warmup_steps: Minimum epochs per trial before pruning is allowed.
    """
    import optuna

    key = name.lower().strip()
    if key == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=n_warmup_steps)
    elif key == "hyperband":
        return optuna.pruners.HyperbandPruner()
    elif key == "none":
        return optuna.pruners.NopPruner()
    else:
        raise ValueError(f"Unknown pruner: {name!r} (expected median/hyperband/none)")
