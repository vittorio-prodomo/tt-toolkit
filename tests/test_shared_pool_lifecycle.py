"""Test shared-pool lifecycle: name derivation, startup sweep, owner cleanup."""
from __future__ import annotations
import socket
from multiprocessing import shared_memory
import pytest
from training_toolkit.data.shared_pool_lifecycle import (
    pool_name,
    sweep_stale,
    OwnedControlBlock,
)
from training_toolkit.engine.trainer import _cleanup_shared_pool


def test_pool_name_is_stable_and_node_scoped():
    """Pool name includes run_id and hostname."""
    assert pool_name("run42").startswith("tt_pool_run42_")


def test_sweep_unlinks_stale_segments():
    """Sweep finds and unlinks stale segments + their locks."""
    # Create a stale segment.
    leak = shared_memory.SharedMemory(name="tt_pool_OLD_x", create=True, size=16)
    leak.close()

    # Sweep it.
    swept = sweep_stale("tt_pool_OLD")
    assert "tt_pool_OLD_x" in swept

    # Second sweep finds nothing (it's already gone).
    assert sweep_stale("tt_pool_OLD") == []


def test_sweep_skips_lock_named_segments():
    """A _lock-named shm entry is skipped, not treated as a segment."""
    leak = shared_memory.SharedMemory(name="tt_pool_OLD_y_lock", create=True, size=16)
    leak.close()
    try:
        assert "tt_pool_OLD_y_lock" not in sweep_stale("tt_pool_OLD")  # skipped, no crash
    finally:
        try:
            leak.unlink()
        except FileNotFoundError:
            pass


def test_owned_control_block_unlinks_on_exit_when_creator():
    """Creator: __exit__ unlinks the block."""
    class _CB:
        is_creator = True
        def __init__(self): self.unlinked = False
        def unlink(self): self.unlinked = True
    cb = _CB()
    with OwnedControlBlock(cb) as c:
        assert c is cb
    assert cb.unlinked is True


def test_owned_control_block_skips_unlink_for_non_creator():
    """Non-creator: __exit__ leaves the block alone (some other process owns it)."""
    class _CB:
        is_creator = False
        def __init__(self): self.unlinked = False
        def unlink(self): self.unlinked = True
    cb = _CB()
    with OwnedControlBlock(cb):
        pass
    assert cb.unlinked is False


def test_cleanup_shared_pool_sweeps_stale_pixel_segments():
    """Startup cleanup must remove pixel segments left by a prior crashed run.

    If a run with the same run_name dies hard (kill -9 / OOM) the finally-block
    in trainer.train() never executes, leaving pixel segments on /dev/shm with
    the pattern  {run_id}_ep{epoch}_{host}_img{image_id}.
    SharedImagePool.try_admit calls SharedMemory(create=True) with no
    FileExistsError guard — a collision raises and kills the DataLoader worker.
    _cleanup_shared_pool must sweep those segments so a restarted run is clean.
    """
    run_id = "test_cleanup_pixel_seg_run"
    host = socket.gethostname()
    seg_name = f"{run_id}_ep0_{host}_img99"

    # Simulate a stale pixel segment left by a crashed prior run.
    seg = shared_memory.SharedMemory(name=seg_name, create=True, size=16)
    seg.close()  # close handle; segment persists on /dev/shm until unlinked

    try:
        _cleanup_shared_pool(run_id)
    except Exception:
        # Best-effort cleanup on failure path so we don't leak on CI.
        try:
            shared_memory.SharedMemory(name=seg_name).unlink()
        except Exception:
            pass
        raise

    # The segment must be gone — attaching it should now raise FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=seg_name, create=False)


def test_cleanup_shared_pool_scoped_to_one_epoch():
    """epoch=K unlinks ONLY ep{K}'s segments — both pixel AND control-block — and must
    leave every other epoch intact, including the ep1-vs-ep10 prefix-collision sibling
    (the trailing _{host} delimiter is what anchors the epoch number)."""
    run_id = "test_scoped_epoch_run"
    host = socket.gethostname()
    # Target epoch 1: one pixel segment + one control-block segment — both must go.
    swept = [f"{run_id}_ep1_{host}_img1", f"tt_pool_{run_id}_ep1_{host}"]
    # Must SURVIVE: a different epoch (ep0) and the prefix sibling (ep10 must not match ep1).
    kept = [f"{run_id}_ep0_{host}_img1", f"{run_id}_ep10_{host}_img1"]
    for nm in swept + kept:
        shared_memory.SharedMemory(name=nm, create=True, size=16).close()
    try:
        _cleanup_shared_pool(run_id, epoch=1)
        for nm in swept:
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=nm, create=False)
        for nm in kept:  # ep0 + ep10 untouched — no prefix collision
            shared_memory.SharedMemory(name=nm, create=False).close()
    finally:
        for nm in swept + kept:
            try:
                shared_memory.SharedMemory(name=nm).unlink()
            except FileNotFoundError:
                pass


def test_cleanup_shared_pool_all_epochs_when_epoch_none():
    """epoch=None (default) sweeps every epoch's segments — preserves existing behavior."""
    run_id = "test_all_epochs_run"
    host = socket.gethostname()
    for ep in (0, 1):
        shared_memory.SharedMemory(name=f"{run_id}_ep{ep}_{host}_img1", create=True, size=16).close()
    try:
        _cleanup_shared_pool(run_id)
        for ep in (0, 1):
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=f"{run_id}_ep{ep}_{host}_img1", create=False)
    finally:
        for ep in (0, 1):
            try:
                shared_memory.SharedMemory(name=f"{run_id}_ep{ep}_{host}_img1").unlink()
            except FileNotFoundError:
                pass


def test_epoch_cleanup_callback_sweeps_previous_epoch():
    """The callback, at the start of epoch K>0 on local_rank 0, unlinks ep{K-1};
    it is a no-op at epoch 0 and on non-zero local rank."""
    from training_toolkit.engine.trainer import SharedPoolEpochCleanupCallback
    run_id = "test_cb_epoch_run"
    host = socket.gethostname()
    seg = f"{run_id}_ep0_{host}_img1"

    class _FakeTrainer:
        def __init__(self, current_epoch, local_rank):
            self.current_epoch = current_epoch
            self.local_rank = local_rank

    cb = SharedPoolEpochCleanupCallback(run_id)
    shared_memory.SharedMemory(name=seg, create=True, size=16).close()
    try:
        cb.on_train_epoch_start(_FakeTrainer(0, 0), None)  # epoch 0: no previous epoch -> no-op
        shared_memory.SharedMemory(name=seg, create=False).close()
        cb.on_train_epoch_start(_FakeTrainer(1, 1), None)  # local_rank 1: no-op
        shared_memory.SharedMemory(name=seg, create=False).close()
        cb.on_train_epoch_start(_FakeTrainer(1, 0), None)  # epoch 1, rank 0: sweeps ep0
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=seg, create=False)
    finally:
        try:
            shared_memory.SharedMemory(name=seg).unlink()
        except FileNotFoundError:
            pass
