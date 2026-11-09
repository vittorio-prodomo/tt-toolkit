"""Hyperparameter sweep configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ParamSpace:
    """Defines the search space for a single hyperparameter.

    Attributes:
        type: One of "float", "int", "categorical".
        low: Lower bound (float/int types).
        high: Upper bound (float/int types).
        log: Use log-uniform sampling (float/int types).
        step: Discretization step (float/int types).
        choices: List of choices (categorical type).
    """

    type: str = "float"
    low: float | None = None
    high: float | None = None
    log: bool = False
    step: float | None = None
    choices: list[Any] | None = None

    def __post_init__(self) -> None:
        if self.type in ("float", "int"):
            if self.low is None or self.high is None:
                raise ValueError(f"{self.type} param requires 'low' and 'high'")
        elif self.type == "categorical":
            if not self.choices:
                raise ValueError("categorical param requires 'choices'")
        else:
            raise ValueError(f"Unknown param type: {self.type!r} (expected float/int/categorical)")


@dataclass
class SweepConfig:
    """Configuration for an Optuna hyperparameter sweep.

    Attributes:
        n_trials: Number of Optuna trials to run.
        pruner: Pruning strategy ("median", "hyperband", or "none").
        pruner_n_warmup_epochs: Number of epochs each trial runs before the
            pruner is allowed to stop it (default 5).
        direction: Optimization direction ("maximize" or "minimize").
        metric: Metric to optimize (should match a logged Lightning metric).
        search_space: Mapping of dotted config paths to their search ranges.
        storage: Optuna storage URL (None = in-memory, or e.g. "sqlite:///sweep.db").
        study_name: Name for the Optuna study (for resumable sweeps).
    """

    n_trials: int = 20
    pruner: str = "median"
    pruner_n_warmup_epochs: int = 5
    direction: str = "maximize"
    metric: str = "val/map"
    search_space: dict[str, ParamSpace] = field(default_factory=dict)
    storage: str | None = "auto"  # "auto" = SQLite in project_dir/.optuna/, None = in-memory
    study_name: str | None = None  # None = auto-generated from run_name + timestamp

    def __post_init__(self) -> None:
        if self.n_trials <= 0:
            raise ValueError(f"n_trials must be > 0, got {self.n_trials}")
        if self.direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize', got {self.direction!r}")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh)

        sweep_data = data.get("sweep", data)

        raw_space = sweep_data.pop("search_space", {})
        parsed_space = {}
        for param_path, spec in raw_space.items():
            if isinstance(spec, dict):
                parsed_space[param_path] = ParamSpace(**spec)
            else:
                raise ValueError(
                    f"search_space[{param_path!r}] must be a dict, got {type(spec).__name__}"
                )
        sweep_data["search_space"] = parsed_space

        return cls(**sweep_data)
