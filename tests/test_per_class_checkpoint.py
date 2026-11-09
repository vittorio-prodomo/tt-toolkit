"""Hook-timing tests for PerClassBestCheckpointCallback.

Lightning calls ``on_validation_epoch_end`` on callbacks *before* the
LightningModule's own ``on_validation_epoch_end`` (where the COCO eval runs and
``val/map_<class>`` is logged) and before ``callback_metrics`` is flushed. So a
callback reading per-class metrics in ``on_validation_epoch_end`` sees the
*previous* epoch's values. The correct hook is ``on_validation_end`` — the same
one ``ModelCheckpoint`` uses — which fires after the flush.
"""
from __future__ import annotations

from training_toolkit.callbacks.per_class_checkpoint import (
    PerClassBestCheckpointCallback,
)


class _FakeTrainer:
    """Minimal trainer stub exposing what the callback touches."""

    def __init__(self, callback_metrics: dict[str, float]):
        self.callback_metrics = callback_metrics
        self.sanity_checking = False
        self.is_global_zero = True

    def save_checkpoint(self, path: str) -> None:
        # Write a real file so the callback's os.link() hardlink dedup works.
        with open(path, "wb") as f:
            f.write(b"ckpt")


def test_saves_per_class_best_on_validation_end(tmp_path):
    """``on_validation_end`` (post-flush) must trigger the per-class saves."""
    cb = PerClassBestCheckpointCallback(dirpath=tmp_path)
    trainer = _FakeTrainer({"val/map_crack": 0.5, "val/map_spall": 0.3})

    cb.on_validation_end(trainer, object())

    assert (tmp_path / "best_per_class" / "best_crack.ckpt").exists()
    assert (tmp_path / "best_per_class" / "best_spall.ckpt").exists()


def test_does_not_save_on_epoch_end(tmp_path):
    """``on_validation_epoch_end`` is pre-flush (stale) and must be unused."""
    cb = PerClassBestCheckpointCallback(dirpath=tmp_path)
    trainer = _FakeTrainer({"val/map_crack": 0.5})

    cb.on_validation_epoch_end(trainer, object())

    assert not (tmp_path / "best_per_class").exists()


class _CountingTrainer(_FakeTrainer):
    """Trainer stub that counts ``save_checkpoint`` calls and honors rank.

    Mimics Lightning: ``save_checkpoint`` is a *collective* (every rank calls it,
    barrier inside), but only the global-zero rank actually writes the file.
    """

    def __init__(self, callback_metrics, is_global_zero=True):
        super().__init__(callback_metrics)
        self.is_global_zero = is_global_zero
        self.save_calls = 0

    def save_checkpoint(self, path: str) -> None:
        self.save_calls += 1
        if self.is_global_zero:
            with open(path, "wb") as f:
                f.write(b"ckpt")


def test_save_checkpoint_called_on_nonzero_rank(tmp_path):
    """``trainer.save_checkpoint`` is a DDP collective — a non-global-zero rank
    MUST also call it when a per-class metric improves, or the ranks desync at
    the next collective (e.g. EarlyStopping's reduce). The callback must not
    early-return on rank>0 before issuing the collective."""
    cb = PerClassBestCheckpointCallback(dirpath=tmp_path)
    trainer = _CountingTrainer({"val/map_crack": 0.5}, is_global_zero=False)

    cb.on_validation_end(trainer, object())

    assert trainer.save_calls == 1


def test_no_collective_when_no_improvement(tmp_path):
    """If nothing improves, NO rank calls save_checkpoint — the skip must be
    symmetric so ranks stay aligned (compute candidates from synced metrics)."""
    cb = PerClassBestCheckpointCallback(dirpath=tmp_path)
    cb._bests = {"val/map_crack": 0.9}  # already better than the incoming 0.5
    trainer = _CountingTrainer({"val/map_crack": 0.5}, is_global_zero=False)

    cb.on_validation_end(trainer, object())

    assert trainer.save_calls == 0


def test_nonzero_rank_tracks_bests(tmp_path):
    """Non-zero ranks must update ``_bests`` too, so cross-epoch candidate sets
    stay identical across ranks (else the collective desyncs a later epoch)."""
    cb = PerClassBestCheckpointCallback(dirpath=tmp_path)
    trainer = _CountingTrainer({"val/map_crack": 0.5}, is_global_zero=False)

    cb.on_validation_end(trainer, object())

    assert cb._bests.get("val/map_crack") == 0.5


def test_nonzero_rank_writes_no_files(tmp_path):
    """The rank-0-only filesystem ops (mkdir / hardlinks) must not run on a
    non-zero rank — only the save_checkpoint collective."""
    cb = PerClassBestCheckpointCallback(dirpath=tmp_path)
    trainer = _CountingTrainer(
        {"val/map_crack": 0.5, "val/map_spall": 0.3}, is_global_zero=False
    )

    cb.on_validation_end(trainer, object())

    assert not (tmp_path / "best_per_class").exists()
