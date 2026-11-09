"""W&B logger resume must tolerate changed config values.

Resuming a run after ANY config edit (e.g. a corrected tiling.cache_size, or a
rebuilt dataset changing dataset_dvc_hash) crashed in _build_loggers with
wandb ConfigError: "Attempted to change value of key ..." — W&B rejects value
changes on a resumed run unless allow_val_change=True is passed to
config.update(). Regression: every config.update call in the wandb branch
carries allow_val_change=True. (Observed live 2026-07-02 on a real run.)
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from training_toolkit.configs.train_config import (
    DataConfig,
    ModelConfig,
    TrainConfig,
)
from training_toolkit.engine.trainer import _build_loggers


class _RecordingConfig:
    def __init__(self):
        self.calls = []

    def update(self, d, *args, **kwargs):
        self.calls.append((d, args, kwargs))


class _StubWandbLogger:
    """Stands in for pytorch_lightning.loggers.WandbLogger — no network, no run."""

    last_instance = None

    def __init__(self, **kwargs):
        self.ctor_kwargs = kwargs
        self.experiment = SimpleNamespace(config=_RecordingConfig())
        _StubWandbLogger.last_instance = self


@pytest.fixture
def wandb_stub(monkeypatch):
    import pytorch_lightning.loggers as pl_loggers

    monkeypatch.setattr(pl_loggers, "WandbLogger", _StubWandbLogger)
    monkeypatch.setenv("LOCAL_RANK", "0")
    return _StubWandbLogger


def test_wandb_config_updates_allow_val_change_on_resume(tmp_path, wandb_stub):
    config = TrainConfig(
        model=ModelConfig(name="mock"),
        data=DataConfig(dataset_path=str(tmp_path)),  # no .dvc/curation -> 1 update call
        loggers=["wandb"],
    )
    _build_loggers(config, tmp_path, resume_meta={"wandb_run_id": "abc123"})

    inst = wandb_stub.last_instance
    assert inst is not None, "WandbLogger was never constructed"
    assert inst.ctor_kwargs.get("id") == "abc123"
    assert inst.ctor_kwargs.get("resume") == "must"
    assert inst.experiment.config.calls, "config.update was never called"
    for d, args, kwargs in inst.experiment.config.calls:
        assert kwargs.get("allow_val_change") is True, (
            f"config.update({list(d)[:3]}...) missing allow_val_change=True"
        )


def test_wandb_group_passed_to_logger_when_set(tmp_path, wandb_stub):
    config = TrainConfig(
        model=ModelConfig(name="mock"),
        data=DataConfig(dataset_path=str(tmp_path)),
        loggers=["wandb"],
        wandb_group="ablation_v3",
    )
    _build_loggers(config, tmp_path, resume_meta={})
    assert wandb_stub.last_instance.ctor_kwargs.get("group") == "ablation_v3"


def test_wandb_group_absent_from_kwargs_when_none(tmp_path, wandb_stub):
    config = TrainConfig(
        model=ModelConfig(name="mock"),
        data=DataConfig(dataset_path=str(tmp_path)),
        loggers=["wandb"],
    )  # wandb_group defaults None
    _build_loggers(config, tmp_path, resume_meta={})
    assert "group" not in wandb_stub.last_instance.ctor_kwargs
