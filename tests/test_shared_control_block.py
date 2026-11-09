"""Tests for SharedControlBlock — multi-process shm segment with ready barrier."""
import multiprocessing as mp
import subprocess
import sys
import time
from multiprocessing import shared_memory
from training_toolkit.data.shared_control_block import (
    SharedControlBlock,
    disable_shared_memory_tracking,
)
from training_toolkit.data.named_lock import NamedLock

NAME = "tt_test_cb_a"
_RACE_NAME = "tt_test_cb_race"  # used by concurrent-create test


def _cleanup(name):
    """Unlink stale shm segment and lock from a previously-crashed run."""
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    NamedLock.unlink_name("/" + name + "_lock")


def _attach_and_set(name, q):
    cb = SharedControlBlock.create_or_attach(name, capacity=4, buffer_size=8)
    with cb.locked():
        cb.header["cursor"] = 7
    q.put(int(cb.header["cursor"][0]))
    cb.detach()


def test_two_processes_share_one_block():
    _cleanup(NAME)
    ctx = mp.get_context("spawn")
    cb = SharedControlBlock.create_or_attach(NAME, capacity=4, buffer_size=8)
    assert cb.is_creator is True
    q = ctx.Queue()
    p = ctx.Process(target=_attach_and_set, args=(NAME, q))
    p.start()
    assert q.get(timeout=10) == 7
    p.join()
    with cb.locked():
        assert int(cb.header["cursor"][0]) == 7     # attacher's write is visible
    cb.unlink()


def test_attacher_waits_for_ready():
    _cleanup(NAME)
    cb = SharedControlBlock.create_or_attach(NAME, capacity=2, buffer_size=2)
    assert int(cb.header["ready"][0]) == 1          # creator flips ready
    cb.unlink()


# ---------------------------------------------------------------------------
# Fix-9b (redo): disable_shared_memory_tracking — whole-process no-op patch
# ---------------------------------------------------------------------------

def test_disable_shared_memory_tracking_subprocess():
    """Spawned subprocess: after disable_shared_memory_tracking(), creating a
    SharedMemory and exiting WITHOUT unlinking produces NO resource_tracker output
    AND the segment still exists afterwards (tracker did NOT auto-clean it).

    This is the definitive proof that the tracker no longer owns our segments.
    It catches both failure modes from the 2-GPU smoke run:

    (A) Old warning flood — tracker auto-unlinks a segment before another process
        can open it (the process that didn't create it gets "No such file").
    (B) KeyError tracebacks in the daemon — per-segment unregister() calls sent N
        REMOVE messages for the same name; the second REMOVE raised KeyError in the
        daemon process (outside our try/except).

    The new approach avoids both by making register/unregister no-ops for
    "shared_memory" in EVERY process that uses the pool.
    """
    import pathlib

    seg_name = "tt_disable_track_probe"
    # Clean up any stale segment from a previous crash.
    try:
        s = shared_memory.SharedMemory(name=seg_name)
        s.close()
        s.unlink()
    except FileNotFoundError:
        pass

    script = f"""
import sys
from training_toolkit.data.shared_control_block import disable_shared_memory_tracking
from multiprocessing import shared_memory

disable_shared_memory_tracking()

shm = shared_memory.SharedMemory(name={seg_name!r}, create=True, size=64)
shm.close()
# Intentionally do NOT unlink — the tracker must NOT auto-unlink it either.
sys.exit(0)
"""
    src_root = str(pathlib.Path(__file__).parent.parent / "src")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": src_root},
    )

    # 1. Child must exit cleanly (no exception in the patch itself).
    assert result.returncode == 0, f"child failed:\n{result.stderr}"

    # 2. No resource_tracker output on stderr — no warning, no KeyError traceback.
    assert "resource_tracker" not in result.stderr, (
        f"resource_tracker output detected (tracker not silenced):\n{result.stderr}"
    )
    assert "KeyError" not in result.stderr, (
        f"KeyError traceback detected (daemon REMOVE raced):\n{result.stderr}"
    )

    # 3. Segment STILL EXISTS after child exit — tracker did NOT auto-unlink it.
    try:
        s = shared_memory.SharedMemory(name=seg_name)
        s.close()
        s.unlink()   # our explicit unlink — must succeed cleanly
    except FileNotFoundError:
        raise AssertionError(
            "Segment was auto-unlinked by the resource_tracker on child exit "
            "(disable_shared_memory_tracking did not suppress tracking)"
        )


def test_disable_shared_memory_tracking_unit():
    """After disable_shared_memory_tracking():
    - register/unregister for 'shared_memory' are no-ops (don't write to the pipe).
    - register/unregister for other rtypes still delegate to the real tracker.
    - The call is idempotent (safe to call twice).
    """
    from multiprocessing import resource_tracker as rt
    import io

    # Apply the patch (idempotent — safe even if already applied by another test).
    disable_shared_memory_tracking()

    # Patch is in effect — calling register/unregister for shared_memory must be
    # silent: they must NOT attempt to start the daemon or write to its pipe.
    # We verify by checking that no exception is raised even when the pipe fd is
    # not yet open in this process (the real _send would call ensure_running and
    # os.write; the patched version returns immediately).
    try:
        rt.register("fake_seg_name", "shared_memory")
        rt.unregister("fake_seg_name", "shared_memory")
    except Exception as exc:
        raise AssertionError(
            f"register/unregister for 'shared_memory' raised unexpectedly: {exc}"
        )

    # Idempotent: a second call must not raise or double-wrap.
    disable_shared_memory_tracking()
    assert getattr(rt, "_shm_tracking_disabled", False) is True

    # Non-shared_memory rtype still delegates (noop rtype always works as a
    # lightweight smoke — it's registered in _CLEANUP_FUNCS by default).
    # The real register writes to the daemon pipe; we only assert no exception.
    try:
        rt.register("probe_noop", "noop")
        rt.unregister("probe_noop", "noop")
    except Exception as exc:
        raise AssertionError(
            f"Non-shared_memory rtype ('noop') was incorrectly intercepted: {exc}"
        )


# ---------------------------------------------------------------------------
# Fix-9c: concurrent create_or_attach — bootstrap race (lock semaphore lag)
# ---------------------------------------------------------------------------

def _concurrent_create_worker(name, capacity, buffer_size, barrier, q):
    """Worker: all N processes hit create_or_attach at the same instant via barrier.

    Reports (is_creator: bool, ok: bool, exc_repr: str|None) via the queue.
    """
    try:
        barrier.wait(timeout=15)          # synchronize all N processes
        cb = SharedControlBlock.create_or_attach(name, capacity, buffer_size)
        # Do a trivial locked op to confirm the lock is usable.
        with cb.locked():
            _ = int(cb.header["attach_count"][0])
        cb.detach()
        q.put((cb.is_creator, True, None))
    except Exception as exc:             # noqa: BLE001
        q.put((False, False, repr(exc)))


def test_concurrent_create_no_existential_error():
    """N processes all call create_or_attach(SAME_name) at the same instant.

    The bootstrap race: creator creates shm THEN semaphore (non-atomically); a
    loser that attaches the shm before the semaphore exists would see
    posix_ipc.ExistentialError from NamedLock(create=False) and abort.  The
    original fix retried the constructor until the semaphore appeared.

    NOTE (2026-07-11, geometry-validation change): attachers now block on the
    header ready-wait BEFORE constructing views/lock, and the creator makes
    the semaphore strictly before writing ready=1 — so the ready gate itself
    pre-empts this race and the retry loop is defense-in-depth.  This test
    therefore asserts the END-TO-END guarantee (one creator, N successes,
    zero ExistentialError aborts under a concurrent sprint), not which layer
    provides it.  Historical RED proof (pre-retry-fix, pre-ready-reorder):
    the sprint reliably hit the race within the first few iterations.

    Strategy: run the N-process sprint 20 times (each iteration unlinks and
    re-creates the block so the race can fire fresh).
    """
    N = 8
    ITERS = 20
    ctx = mp.get_context("spawn")

    for iteration in range(ITERS):
        _cleanup(_RACE_NAME)           # start clean each iteration

        barrier = ctx.Barrier(N)
        q = ctx.Queue()

        procs = [
            ctx.Process(
                target=_concurrent_create_worker,
                args=(_RACE_NAME, 4, 8, barrier, q),
                daemon=True,
            )
            for _ in range(N)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=20)

        # Collect results — one entry per process.
        results = []
        while not q.empty():
            results.append(q.get_nowait())

        # --- assertions ---
        assert len(results) == N, (
            f"iter {iteration}: only {len(results)}/{N} processes reported "
            f"(some may have hung or crashed)"
        )
        creators = [r for r in results if r[0]]         # is_creator == True
        failures = [r for r in results if not r[1]]     # ok == False
        assert len(creators) == 1, (
            f"iter {iteration}: expected exactly 1 creator, got {len(creators)}: {creators}"
        )
        assert failures == [], (
            f"iter {iteration}: {len(failures)} worker(s) failed:\n"
            + "\n".join(f"  {r[2]}" for r in failures)
        )

    # Final cleanup.
    _cleanup(_RACE_NAME)


def test_attach_rejects_mismatched_geometry():
    """An attacher passing a different (capacity, buffer_size) than the creator
    would build its registry/reservoir numpy views at WRONG OFFSETS (the layout
    is computed from the CALLER's args) and read garbage that can look valid.
    create_or_attach must validate the attacher's geometry against the
    creator's and fail loud."""
    import pytest

    _cleanup("tt_test_cb_geom")
    cb = SharedControlBlock.create_or_attach("tt_test_cb_geom", capacity=4, buffer_size=32)
    try:
        # LARGER geometry: today dies with an incidental numpy buffer error
        # (wrong message), not a deliberate check.
        with pytest.raises(ValueError, match="geometry"):
            SharedControlBlock.create_or_attach("tt_test_cb_geom", capacity=8, buffer_size=128)
        # SMALLER geometry: the truly SILENT case — the views fit inside the
        # larger buffer, so today this attaches "successfully" and reads
        # garbage at wrong offsets.
        with pytest.raises(ValueError, match="geometry"):
            SharedControlBlock.create_or_attach("tt_test_cb_geom", capacity=2, buffer_size=8)
        # Identical geometry still attaches fine.
        cb2 = SharedControlBlock.create_or_attach("tt_test_cb_geom", capacity=4, buffer_size=32)
        assert cb2.is_creator is False
        cb2.detach()
    finally:
        cb.unlink()
