"""Tests for the non-TTY file progress logger and TTY-aware selection.

RichProgressBar renders nothing to a redirected logfile; FileProgressLogger
emits throttled plain-text progress through the logging system instead, so
non-interactive runs (CI / nohup / redirected) get greppable progress.
"""

import logging
from types import SimpleNamespace

import pytest

from pytorch_lightning.callbacks import RichProgressBar

from training_toolkit.callbacks.progress import (
    FileProgressLogger,
    select_progress_callback,
)


class TestThrottle:
    def test_emits_on_first_batch(self):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        assert cb._should_emit(0, now=1000.0) is True

    def test_emits_every_n_batches(self):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        assert cb._should_emit(50, now=1000.0) is True
        assert cb._should_emit(100, now=1000.0) is True

    def test_throttles_between_intervals(self):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        cb._last_emit = 100.0
        assert cb._should_emit(25, now=105.0) is False  # 5s < 30s and 25 % 50 != 0

    def test_emits_after_time_window(self):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        cb._last_emit = 100.0
        assert cb._should_emit(25, now=131.0) is True  # 31s >= 30s

    def test_rejects_non_positive_every_n(self):
        with pytest.raises(ValueError, match="every_n_batches"):
            FileProgressLogger(every_n_batches=0)


class TestFormat:
    def test_includes_epoch_stage_and_one_based_progress(self):
        line = FileProgressLogger._format(1, "val", 149, 400, {"val/loss": 0.5})
        assert line == "epoch 1 | val 150/400 | val/loss=0.5"

    def test_unknown_total_renders_question_mark(self):
        line = FileProgressLogger._format(0, "train", 0, float("inf"), {})
        assert line == "epoch 0 | train 1/?"


class TestStageMetricSelection:
    """Progress line shows ONE representative loss for the current stage, not the ~20
    stale keys from callback_metrics (which still holds the other stage's values)."""

    def test_prefers_stage_total_loss(self):
        cm = {"train/loss_total": 5.0, "train/loss_ce": 1.0, "val/loss_total": 9.0}
        assert FileProgressLogger._select_metrics(cm, "train") == {"train/loss_total": 5.0}

    def test_val_stage_picks_val_loss_not_stale_train(self):
        cm = {"val/loss_total": 9.0, "train/loss_total": 5.0}
        assert FileProgressLogger._select_metrics(cm, "val") == {"val/loss_total": 9.0}

    def test_falls_back_to_any_stage_loss(self):
        cm = {"train/loss_ce": 1.5}
        assert FileProgressLogger._select_metrics(cm, "train") == {"train/loss_ce": 1.5}

    def test_ignores_non_loss_and_other_stage(self):
        cm = {"val/map": 0.1, "train/loss_total": 5.0}
        assert FileProgressLogger._select_metrics(cm, "val") == {}


class TestSelection:
    def test_tty_selects_rich_and_enables_bar(self):
        cb, enable = select_progress_callback(is_tty=True)
        assert isinstance(cb, RichProgressBar)
        assert enable is True

    def test_non_tty_selects_file_logger_and_disables_bar(self):
        cb, enable = select_progress_callback(is_tty=False)
        assert isinstance(cb, FileProgressLogger)
        assert enable is False


def _stub_trainer(rank_zero=True):
    return SimpleNamespace(
        is_global_zero=rank_zero,
        current_epoch=2,
        num_training_batches=400,
        num_val_batches=[400],
        callback_metrics={"train/loss": 0.42, "val/map": 0.1},
    )


class TestHooks:
    def test_logs_progress_on_rank_zero(self, caplog):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        with caplog.at_level(logging.INFO):
            cb.on_train_batch_end(_stub_trainer(), None, None, None, 0)
        assert "epoch 2 | train 1/400" in caplog.text
        assert "train/loss=0.42" in caplog.text
        assert "val/map" not in caplog.text  # only loss-ish metrics

    def test_silent_on_non_zero_rank(self, caplog):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        with caplog.at_level(logging.INFO):
            cb.on_train_batch_end(_stub_trainer(rank_zero=False), None, None, None, 0)
        assert "epoch" not in caplog.text

    def test_validation_hook_uses_val_total(self, caplog):
        cb = FileProgressLogger(every_n_batches=50, every_s=30.0)
        with caplog.at_level(logging.INFO):
            cb.on_validation_batch_end(_stub_trainer(), None, None, None, 0)
        assert "val 1/400" in caplog.text
