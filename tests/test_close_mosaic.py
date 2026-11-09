"""CloseMosaicCallback._close_augmentations must reach tiled WRAPPERS.

Bug: it resolved target = getattr(dataset, "_base", dataset) and zeroed
target._aug_config — but on both tiled paths (offline TiledSegmentationDataset
and streaming) the aug config lives on the WRAPPER and the base's is None
(trainer builds the base without aug_config), so the callback silently no-op'd
while still logging "disabled mosaic...".
"""
import logging
from types import SimpleNamespace

from training_toolkit.callbacks.close_mosaic import CloseMosaicCallback


def _aug(p=0.5):
    return SimpleNamespace(mosaic_p=p, mixup_p=p, cutmix_p=p, copy_paste_p=p)


def _zeroed(aug):
    return (aug.mosaic_p, aug.mixup_p, aug.cutmix_p, aug.copy_paste_p) == (0, 0, 0, 0)


def test_close_reaches_wrapper_config():
    """Tiled shape: wrapper holds _aug_config, base holds None. THE bug."""
    ds = SimpleNamespace(_aug_config=_aug(), _base=SimpleNamespace(_aug_config=None))
    CloseMosaicCallback._close_augmentations(ds)
    assert _zeroed(ds._aug_config)


def test_close_still_reaches_base_config():
    """Non-tiled shape (regression): base holds the config."""
    base = SimpleNamespace(_aug_config=_aug())
    ds = SimpleNamespace(_base=base)          # wrapper without its own _aug_config
    CloseMosaicCallback._close_augmentations(ds)
    assert _zeroed(base._aug_config)


def test_close_handles_shared_object_idempotently():
    """Wrapper and base may alias the SAME AugConfig — zeroing twice is fine."""
    shared = _aug()
    ds = SimpleNamespace(_aug_config=shared, _base=SimpleNamespace(_aug_config=shared))
    CloseMosaicCallback._close_augmentations(ds)
    assert _zeroed(shared)


def test_close_warns_when_nothing_to_close(caplog):
    ds = SimpleNamespace(_base=SimpleNamespace())     # no _aug_config anywhere
    with caplog.at_level(logging.WARNING):
        CloseMosaicCallback._close_augmentations(ds)
    assert any("NOT disabled" in r.message for r in caplog.records)


def _stub_trainer(ds, epoch=5):
    loader = SimpleNamespace(dataset=ds)
    return SimpleNamespace(current_epoch=epoch, train_dataloader=loader)


def test_epoch_start_success_log_gated_on_actual_close(caplog):
    cb = CloseMosaicCallback(close_epoch=3)
    ds_empty = SimpleNamespace(_base=SimpleNamespace())      # nothing to close
    with caplog.at_level(logging.INFO):
        cb.on_train_epoch_start(_stub_trainer(ds_empty), None)
    assert not any("disabled mosaic" in r.message for r in caplog.records
                   if r.levelno == logging.INFO)
    assert any("NOT disabled" in r.message for r in caplog.records)

    caplog.clear()
    cb2 = CloseMosaicCallback(close_epoch=3)
    ds_ok = SimpleNamespace(_aug_config=_aug(), _base=SimpleNamespace(_aug_config=None))
    with caplog.at_level(logging.INFO):
        cb2.on_train_epoch_start(_stub_trainer(ds_ok), None)
    assert any("disabled mosaic" in r.message for r in caplog.records
               if r.levelno == logging.INFO)
