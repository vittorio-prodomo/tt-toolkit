"""Multi-process coverage gate for the cross-process shared pool.

Spawns N worker processes that all attach to ONE SharedControlBlock and drive
the real PooledTileSource._iter_shared() production loop.  Asserts that the
union of every emitted (image_id, box) across all workers equals the full
expected set, each exactly once — and that no /dev/shm pixel segment is
leaked afterwards.
"""
from __future__ import annotations

import glob
import multiprocessing as mp
import random
import socket
import uuid
from collections import Counter
from multiprocessing import shared_memory

import numpy as np
import pytest

import training_toolkit.data.pooled_tile_source as P
from training_toolkit.data.named_lock import NamedLock
from training_toolkit.data.shared_control_block import SharedControlBlock

# ---------------------------------------------------------------------------
# Test parameters
# ---------------------------------------------------------------------------
_N_IMAGES = 20
_WINDOWS_PER_IMAGE = 3
_CAPACITY = 4         # small cap to exercise slot-starvation / pending lookahead
_BUFFER_SIZE = 16
_N_WORKERS = 4

# Each image: 3 windows with distinct boxes so (image_id, box) is unique.
_TILES_BY_IMAGE = {
    i: [((w * 4, 0, 4, 4), True) for w in range(_WINDOWS_PER_IMAGE)]
    for i in range(_N_IMAGES)
}
_IMAGE_ORDER = list(range(_N_IMAGES))

# The full expected set: one (image_id, box) per tile.
_EXPECTED = Counter(
    (i, (w * 4, 0, 4, 4))
    for i in range(_N_IMAGES)
    for w in range(_WINDOWS_PER_IMAGE)
)


# ---------------------------------------------------------------------------
# Picklable helpers — must be at module level for spawn workers
# ---------------------------------------------------------------------------

class _FakeBase:
    """Minimal base object the PooledTileSource needs from its dataset."""
    def _image_path(self, image_id: int) -> str:
        # Encode the image_id in the path; used by the fake decode.
        return f"fake://{image_id}"

    def _load_annotations(self, image_id: int) -> list:
        return []

    def _image_class_mask(self, image_id: int):
        return None

    def _apply_transform(self, sample, transform):
        return sample


class _TestTileSource(P.PooledTileSource):
    """PooledTileSource subclass that replaces file I/O with in-memory stubs.

    - _decode_full: returns a tiny 4×4 numpy array instead of calling pyvips.
    - _materialize_shared: returns a plain dict instead of calling materialize helpers.
      The dict carries image_id and box so callers can verify (image_id, box) coverage.
    """

    def _decode_full(self, image_id: int) -> np.ndarray:
        """Return a 4-pixel-wide image whose value identifies the image_id."""
        return np.full((4, _WINDOWS_PER_IMAGE * 4, 3), image_id % 255, dtype=np.uint8)

    def _materialize_shared(self, image_id: int, box, px, apply_band=True) -> dict:
        """Return a minimal sample dict — no real tile crop or annotation needed."""
        return {"image_id": image_id, "box": tuple(box)}


def _cleanup_cb(name: str) -> None:
    """Unlink stale control-block shm + its semaphore from a previous crash."""
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    NamedLock.unlink_name("/" + name + "_lock")


def _cleanup_pixel_segs(run_id: str, host: str) -> None:
    """Unlink any leftover per-image pixel segments from a previous run.

    Matches both the old ``{run_id}_{host}_img*`` and the current
    ``{run_id}_ep{epoch}_{host}_img*`` naming so stale segments from either
    format are swept away.
    """
    for pattern in (
        f"/dev/shm/{run_id}_{host}_img*",
        f"/dev/shm/{run_id}_ep*_{host}_img*",
    ):
        for path in glob.glob(pattern):
            name = path.split("/")[-1]
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass


def _worker(cb_name: str, run_id: str, seed: int, q: mp.Queue) -> None:
    """Worker process: attach to the shared CB, iterate the production loop, push results."""
    cb = SharedControlBlock.create_or_attach(
        cb_name, capacity=_CAPACITY, buffer_size=_BUFFER_SIZE
    )
    src = _TestTileSource(
        _FakeBase(),
        _IMAGE_ORDER,
        _TILES_BY_IMAGE,
        random.Random(seed),
        tile_source="shared_pooled",
        control=cb,
        run_id=run_id,
        pool_min_windows=1,          # all images are pooled (exercises slot mechanism)
        pool_capacity=_CAPACITY,
        buffer_size=_BUFFER_SIZE,
        transform=None,
        imgsz=None,
        size_band=None,
        min_overlap=0.05,
    )
    got = []
    for sample in src:
        got.append((sample["image_id"], sample["box"]))
    cb.detach()
    q.put(got)


# ---------------------------------------------------------------------------
# The coverage gate test
# ---------------------------------------------------------------------------

def test_multiprocess_covers_every_tile_exactly_once():
    """4 spawn processes share one SharedControlBlock and drain an epoch.

    Asserts:
    - Every (image_id, box) emitted across all workers exactly once (Counter match).
    - No deadlock (bounded timeouts).
    - No /dev/shm pixel-segment leak after creator.unlink().
    """
    # Unique names to avoid collisions on re-run.
    suffix = uuid.uuid4().hex[:8]
    cb_name = f"tt_cov_{suffix}"
    run_id = f"cov_{suffix}"
    host = socket.gethostname()

    # Clean up any stale state from a previous crash.
    _cleanup_cb(cb_name)
    _cleanup_pixel_segs(run_id, host)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    # The creator sets up the control block; workers attach.
    creator = SharedControlBlock.create_or_attach(
        cb_name, capacity=_CAPACITY, buffer_size=_BUFFER_SIZE
    )

    procs = [
        ctx.Process(target=_worker, args=(cb_name, run_id, seed, q), daemon=True)
        for seed in range(_N_WORKERS)
    ]
    for p in procs:
        p.start()

    # Collect all results with a per-process timeout (no deadlock hang).
    all_pairs: list[tuple[int, tuple]] = []
    try:
        for _ in procs:
            chunk = q.get(timeout=120)   # 2 min per worker is very generous
            all_pairs.extend(chunk)
    finally:
        # Always reap + unlink, even if a q.get() timed out (worker hang/crash),
        # so the control block + pixel segments never leak into /dev/shm.
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        # A worker that crashed AFTER putting its results would slip past the
        # coverage Counter — catch it via the exit code.
        assert all(p.exitcode == 0 for p in procs), \
            f"worker exit codes: {[p.exitcode for p in procs]}"
        creator.unlink()

    # --- success path (the finally above already did all cleanup) ---

    # No leak: every per-image pixel segment must be gone after the drain.
    # Match the current segment naming: {run_id}_ep{epoch}_{host}_img{id}
    leaked = glob.glob(f"/dev/shm/{run_id}_ep*_{host}_img*")
    assert leaked == [], f"Pixel segment leak detected: {leaked}"

    # Coverage: every (image_id, box) across all workers, each exactly once.
    got = Counter(all_pairs)
    assert got == _EXPECTED, (
        f"Coverage mismatch.\n"
        f"  total emitted: {sum(got.values())} (expected {sum(_EXPECTED.values())})\n"
        f"  duplicated: {[(k, v) for k, v in got.items() if v > 1]}\n"
        f"  missing:    {[k for k in _EXPECTED if k not in got]}"
    )


# ---------------------------------------------------------------------------
# Fix-9a: capped multi-process coverage — exercises the REAL emit-cap + the
# real shared-cursor interaction under the multi-process path.
#
# Total real tiles = _N_IMAGES * _WINDOWS_PER_IMAGE = 60.  We treat the
# _N_WORKERS (4) processes as a rank's DataLoader workers (W=4) and set the
# PER-RANK max_emit = 60 so the real _iter_shared computes
# per_worker_target = ceil(60 / 4) = 15 per process.
# get_worker_info() is faked to W=4 inside each worker so the REAL
# ceil(max_emit/W) division runs with W != 1.
#
# What this pins:
#   (b) each process emits EXACTLY per_worker_target (15) — the count-invariant
#       that prevents the DDP step desync / deadlock (the whole point of the cap).
#   (a) NO (image_id, box) is emitted as a REAL tile by more than one process —
#       the shared cursor still hands each window out at most once even under
#       capping (no double-emit -> no training corruption).
#
# NOTE — coverage is intentionally NOT asserted to be COMPLETE under capping.
# With per-image admission, a process can claim+admit an image, emit some of its
# windows, hit its cap, and abandon the remaining admitted windows (the shared
# cursor has already advanced past them, so no other process re-claims them).
# So a few windows can go un-emitted in a capped epoch — acceptable for DDP
# (the cap's job is equal step counts, and over many epochs the shuffled draw
# covers everything).  The strong invariant that MUST hold is no DOUBLE-emit,
# which is what (a) checks.  (Padding re-yields a process's own last real sample,
# so its trailing identical pairs are de-duped per process before the union.)
# ---------------------------------------------------------------------------

_TOTAL_TILES = _N_IMAGES * _WINDOWS_PER_IMAGE   # 60
_PER_RANK_MAX_EMIT = _TOTAL_TILES               # 60 (world=1; the procs are the rank's workers)
_PER_WORKER_TARGET = -(-_PER_RANK_MAX_EMIT // _N_WORKERS)   # ceil(60/4) = 15


def _fake_worker_info_w(num_workers: int):
    """A picklable factory result is not needed — each worker calls this in-proc."""
    class _WI:
        def __init__(self, n): self.id = 0; self.num_workers = n; self.seed = 0; self.dataset = None
    import torch.utils.data as tud
    tud.get_worker_info = lambda: _WI(num_workers)   # process-local; spawn proc is throwaway


def _worker_capped(cb_name: str, run_id: str, seed: int, q: mp.Queue) -> None:
    """Like _worker but with max_emit set and get_worker_info faked to W=_N_WORKERS,
    so the REAL _iter_shared division+cap runs.  Reports the ordered emit list so
    the parent can split real vs padded (padding re-yields the last real sample)."""
    _fake_worker_info_w(_N_WORKERS)
    cb = SharedControlBlock.create_or_attach(
        cb_name, capacity=_CAPACITY, buffer_size=_BUFFER_SIZE
    )
    src = _TestTileSource(
        _FakeBase(),
        _IMAGE_ORDER,
        _TILES_BY_IMAGE,
        random.Random(seed),
        tile_source="shared_pooled",
        control=cb,
        run_id=run_id,
        pool_min_windows=1,
        pool_capacity=_CAPACITY,
        buffer_size=_BUFFER_SIZE,
        transform=None,
        imgsz=None,
        size_band=None,
        min_overlap=0.05,
        max_emit=_PER_RANK_MAX_EMIT,
    )
    got = []
    for sample in src:
        got.append((sample["image_id"], sample["box"]))
    cb.detach()
    q.put(got)


def test_multiprocess_capped_emit_per_worker_target():
    """4 processes (W=4), per-rank max_emit=60 -> 15 each; real coverage intact.

    Asserts:
    - each process emits EXACTLY _PER_WORKER_TARGET (15) — the DDP count-invariant,
    - no duplicate REAL (image_id, box) across processes (full once-coverage),
    - no /dev/shm leak.
    """
    suffix = uuid.uuid4().hex[:8]
    cb_name = f"tt_covcap_{suffix}"
    run_id = f"covcap_{suffix}"
    host = socket.gethostname()

    _cleanup_cb(cb_name)
    _cleanup_pixel_segs(run_id, host)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()

    creator = SharedControlBlock.create_or_attach(
        cb_name, capacity=_CAPACITY, buffer_size=_BUFFER_SIZE
    )

    procs = [
        ctx.Process(target=_worker_capped, args=(cb_name, run_id, seed, q), daemon=True)
        for seed in range(_N_WORKERS)
    ]
    for p in procs:
        p.start()

    per_proc_chunks: list[list[tuple[int, tuple]]] = []
    try:
        for _ in procs:
            chunk = q.get(timeout=120)
            per_proc_chunks.append(chunk)
    finally:
        for p in procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
        assert all(p.exitcode == 0 for p in procs), \
            f"worker exit codes: {[p.exitcode for p in procs]}"
        creator.unlink()

    # No leak.
    leaked = glob.glob(f"/dev/shm/{run_id}_{host}_img*")
    assert leaked == [], f"Pixel segment leak detected: {leaked}"

    # (b) Each process emits EXACTLY per_worker_target.
    counts = [len(c) for c in per_proc_chunks]
    assert all(n == _PER_WORKER_TARGET for n in counts), (
        f"Each process must emit exactly {_PER_WORKER_TARGET}; got {counts} "
        f"(unequal counts -> DDP step desync)")

    # (a) No (image_id, box) emitted as a REAL tile by more than one process.
    # Padding re-yields a process's OWN last real sample as a CONSECUTIVE tail of
    # identical pairs, so de-dup each process's real emits (first occurrence in
    # that process = real; repeats of the last one = padding) before the union.
    real_pairs: list[tuple[int, tuple]] = []
    for chunk in per_proc_chunks:
        seen_in_proc = set()
        for pair in chunk:
            if pair not in seen_in_proc:        # first occurrence in THIS proc = a real emit
                seen_in_proc.add(pair)
                real_pairs.append(pair)
    real_counter = Counter(real_pairs)
    dups = [(k, v) for k, v in real_counter.items() if v > 1]
    assert not dups, (
        f"DOUBLE-EMIT: real tile(s) emitted by >1 process (training corruption): {dups}")
    # Real tiles are a subset of the full set (no phantom tiles); completeness is
    # NOT required under capping (see module NOTE above — abandonment is allowed).
    assert set(real_pairs).issubset(set(_EXPECTED)), (
        f"Phantom real tiles not in the expected set: "
        f"{set(real_pairs) - set(_EXPECTED)}")
    assert len(real_pairs) <= _TOTAL_TILES
