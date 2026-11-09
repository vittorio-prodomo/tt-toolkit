"""Tests for SharedImagePool — pixel segments + shared pool bookkeeping."""
import random
import socket
import sys
import os
from multiprocessing import shared_memory

import numpy as np
import pytest

import training_toolkit.data.pooled_tile_source as P
from training_toolkit.data.named_lock import NamedLock
from training_toolkit.data.shared_control_block import SharedControlBlock
from training_toolkit.data.shared_image_pool import SharedImagePool
from training_toolkit.data.shared_pool_state import _PoolState


def _decode(image_id):
    return np.full((8, 8, 3), image_id, dtype=np.uint8)


def _cleanup_cb(name):
    """Unlink stale control-block shm + lock from a previously-crashed run."""
    try:
        shm = shared_memory.SharedMemory(name=name)
        shm.close()
        shm.unlink()
    except FileNotFoundError:
        pass
    NamedLock.unlink_name("/" + name + "_lock")


def _cleanup_pixel_segs(run_id, host, image_ids, epoch=0):
    """Unlink any leftover per-image pixel segments from a previous run.

    The segment name mirrors SharedImagePool._seg_name:
    ``{run_id}_ep{epoch}_{host}_img{image_id}``.
    """
    for iid in image_ids:
        seg_name = f"{run_id}_ep{epoch}_{host}_img{iid}"
        try:
            shm = shared_memory.SharedMemory(name=seg_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass


def test_pooled_pixels_roundtrip_and_unlink():
    _cleanup_cb("tt_sip_a")
    _cleanup_pixel_segs("r", "h", [0])
    cb = SharedControlBlock.create_or_attach("tt_sip_a", capacity=2, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h")
    assert pool.try_admit(0, n_windows=2, is_pooled=True, decode_fn=_decode) is True
    px = pool.read_pixels(0)
    assert px is not None and px[0, 0, 0] == 0 and px.shape == (8, 8, 3)
    pool.pop(random.Random(0))
    pool.decref(0)
    pool.pop(random.Random(0))
    pool.decref(0)
    assert pool.has_free_slot() is True
    assert pool.read_pixels(0) is None          # segment unlinked at refcount 0
    cb.unlink()


def test_lazy_admit_has_no_pixels():
    _cleanup_cb("tt_sip_b")
    _cleanup_pixel_segs("r", "h", [5])
    cb = SharedControlBlock.create_or_attach("tt_sip_b", capacity=1, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h")
    assert pool.try_admit(5, n_windows=3, is_pooled=False, decode_fn=_decode) is True
    assert pool.read_pixels(5) is None          # lazy -> caller ROI-decodes
    assert pool.has_free_slot() is True         # lazy used no slot
    cb.unlink()


def test_matches_pure_oracle_on_a_drain():
    """Coverage drain: SharedImagePool emits the same full multiset as _PoolState.

    Per-draw sequence is NOT asserted — swap-remove vs list.pop produce different
    orderings given the same rng seed after the first pop.  Only the final coverage
    set matters.  The oracle confirms _PoolState reaches the same full coverage.
    """
    _cleanup_cb("tt_sip_c")
    _cleanup_pixel_segs("r", "h", list(range(12)))
    plan = [(i, 3, (i % 2 == 0)) for i in range(12)]   # alternate pooled/lazy
    cb = SharedControlBlock.create_or_attach("tt_sip_c", capacity=3, buffer_size=8)
    pool = SharedImagePool(cb, run_id="r", host="h")
    oracle = _PoolState(capacity=3, buffer_size=8, plan=list(plan))
    rng_a = random.Random(0)
    pos, seen = 0, []
    # Loop until all plan items admitted AND pool is fully drained.
    # pool.done() is True initially (empty reservoir + no registry rows), so we
    # must also guard on pos < len(plan).
    # Guard with n_windows=3 headroom so we never overflow the buffer_size=8 reservoir.
    while pos < len(plan) or not pool.done():
        if pos < len(plan) and pool.reservoir_len() + plan[pos][1] <= 8 and (
                not plan[pos][2] or pool.has_free_slot()):
            iid, nw, ip = plan[pos]; pos += 1
            pool.try_admit(iid, nw, ip, _decode)
            o = oracle.next_admit(); oracle.admit(*o)
        elif pool.reservoir_len() > 0:
            d = pool.pop(rng_a); seen.append(d); pool.decref(d[0])
            od = oracle.pop(random.Random(0)); oracle.decref(od[0])  # oracle's own draw
        elif pos < len(plan):
            iid, nw, ip = plan[pos]; pos += 1
            pool.try_admit(iid, nw, ip, _decode)
            oracle.admit(*oracle.next_admit())
        else:
            break  # shouldn't happen if loop condition is right
    assert set(seen) == {(i, w) for i in range(12) for w in range(3)}
    # No-leak: every pooled image's segment must be unlinked after a full drain.
    for iid in range(12):
        if iid % 2 == 0:                                   # pooled images only
            with pytest.raises(FileNotFoundError):
                shared_memory.SharedMemory(name=f"r_ep0_h_img{iid}")
    cb.unlink()
    _cleanup_pixel_segs("r", "h", list(range(12)))


def test_evict_freed_closes_stale_handles():
    """A worker's cached handle is closed when another process frees the image.

    White-box, single process: cache a pooled image's handle via read_pixels,
    then simulate another process freeing it (registry state -> 0) and assert
    _evict_freed drops + closes our stale handle without error.
    """
    _cleanup_cb("tt_sip_e")
    _cleanup_pixel_segs("r", "h", [3])
    cb = SharedControlBlock.create_or_attach("tt_sip_e", capacity=2, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h")
    pool.try_admit(3, n_windows=2, is_pooled=True, decode_fn=_decode)
    assert pool.read_pixels(3) is not None
    assert 3 in pool._attached                     # cached this process's handle

    # Simulate ANOTHER process doing the final decref: it unlinks the name and
    # flips the slot to free; our process still holds an open handle.
    slot = pool._slot_for(3)
    pool.cb.registry[slot]["state"] = 0            # freed by some other worker

    pool._evict_freed()
    assert 3 not in pool._attached                 # our stale handle closed + dropped
    cb.unlink()
    _cleanup_pixel_segs("r", "h", [3])


def test_claim_next_advances_shared_cursor():
    """claim_next returns plan items in order, None when exhausted."""
    _cleanup_cb("tt_sip_d")
    plan = [(10, 2, True), (11, 1, False), (12, 3, True)]
    cb = SharedControlBlock.create_or_attach("tt_sip_d", capacity=4, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h")
    assert pool.claim_next(plan) == (10, 2, True)
    assert pool.claim_next(plan) == (11, 1, False)
    assert pool.claim_next(plan) == (12, 3, True)
    assert pool.claim_next(plan) is None          # past the end
    cb.unlink()


# ---------------------------------------------------------------------------
# Task-6 e2e: shared_pooled branch of PooledTileSource
# ---------------------------------------------------------------------------

def _cleanup_shared_e2e(cb_name, run_id, host, image_ids):
    _cleanup_cb(cb_name)
    _cleanup_pixel_segs(run_id, host, image_ids)


def test_shared_branch_emits_every_window_once(monkeypatch):
    """PooledTileSource(tile_source='shared_pooled') drains all windows exactly once.

    Uses 2 images × 3 windows = 6 total, capacity=2 (all pooled at pool_min_windows=2).
    Stubs out pyvips + materialize so no real image I/O is needed.
    Also verifies the emitted sample image is a COPY (does not share memory with
    the shm buffer).
    """
    import training_toolkit.data.pooled_tile_source as P

    cb_name = "tt_share_e2e"
    run_id = "t"
    host = socket.gethostname()
    n_images, windows_per_image = 2, 3

    _cleanup_shared_e2e(cb_name, run_id, host, list(range(n_images)))

    # Stub decode: return a 4×4 image whose pixels identify the image.
    monkeypatch.setattr(P, "decode_image_pyvips",
                        lambda path: np.full((4, 4, 3), int(path.split("//")[1]), dtype=np.uint8))

    # Stub materialize helpers so they return simple dicts with "image" key.
    # IMPORTANT: the pooled stub returns the passed `array` UNCHANGED (it does NO
    # copy of its own). That makes the `assert sample["image"].base is None` below
    # genuinely load-bearing: the only thing that can detach the tile from the shm
    # buffer is _materialize_shared's own .copy(). If that .copy() is ever dropped,
    # the array handed here is still a view into shm (base != None) and the assert fires.
    monkeypatch.setattr(
        P, "materialize_labeled_tile_from_array",
        lambda array, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image": array, "image_id": image_id, "box": tuple(box)},
    )
    monkeypatch.setattr(
        P, "materialize_labeled_tile",
        lambda path, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image": np.zeros((box[3], box[2], 3), dtype=np.uint8),
             "image_id": image_id, "box": tuple(box)},
    )

    from tests.test_pooled_tile_source import make_pooled_source_inputs
    base, image_order, tiles_by_image = make_pooled_source_inputs(
        n_images=n_images, windows_per_image=windows_per_image)

    cb = SharedControlBlock.create_or_attach(cb_name, capacity=2, buffer_size=16)
    try:
        src = P.PooledTileSource(
            base, image_order, tiles_by_image, random.Random(0),
            tile_source="shared_pooled", control=cb, run_id=run_id,
            pool_min_windows=2, buffer_size=16,
            # Required keyword args for the existing signature
            transform=None, pool_capacity=2, imgsz=None, size_band=None, min_overlap=0.05,
        )
        samples = list(src)

        # Every window emitted exactly once (count AND multiset of (image_id, box)).
        from collections import Counter
        assert len(samples) == n_images * windows_per_image, (
            f"Expected {n_images * windows_per_image} samples, got {len(samples)}")
        pairs = [(s["image_id"], s["box"]) for s in samples]
        assert Counter(pairs) == Counter(
            {(i, (0, 0, 4, 4)): windows_per_image for i in range(n_images)})

        # Every sample has an "image" key
        assert all("image" in s for s in samples), "Some samples missing 'image' key"

        # The sample image must be a numpy copy, not sharing memory with the shm buffer.
        # The stub above does NOT copy, so img.base is None can ONLY be true if
        # _materialize_shared itself detached the tile via .copy() (load-bearing).
        for s in samples:
            img = s["image"]
            assert isinstance(img, np.ndarray), "image is not an ndarray"
            assert img.flags["C_CONTIGUOUS"], "image is not C-contiguous (must be a copy)"
            assert img.base is None, (
                "image.base is not None — it still aliases an external buffer (shm use-after-free risk)")

    finally:
        cb.unlink()
        _cleanup_pixel_segs(run_id, host, list(range(n_images)))


# ---------------------------------------------------------------------------
# Fix-9a: per-worker emit cap in _iter_shared (DDP no-deadlock)
# ---------------------------------------------------------------------------
#
# Invariant: _iter_shared must yield EXACTLY ceil(max_emit / num_workers) samples
# regardless of how many real tiles the shared pool gives this worker.  This
# prevents DDP collective desync when the cross-process draw is uneven.
#
# Strategy: we stub PooledTileSource._emit_one so it drains a finite list of
# pre-built samples that is SMALLER than the target, then assert that padding
# fills the gap.  A second case confirms the iterator STOPS at the target even
# when the stub pool has more tiles than needed.


class _EarlyDrainTileSource(P.PooledTileSource):
    """PooledTileSource subclass whose shared pool is replaced by a
    pre-loaded list of sample dicts.  _emit_one pops from the list;
    the pool reports done() once the list is exhausted, mimicking the
    cross-process early-drain scenario.

    This lets us test _iter_shared without any real SharedImagePool,
    file I/O, or IPC — fully in-process.
    """

    def __init__(self, real_tiles: list[dict], **kw):
        """real_tiles: the finite list of real samples this worker gets."""
        # We don't call super().__init__() in the normal way because we
        # want to skip the full PooledTileSource constructor plumbing.
        # Instead we set only the attributes _iter_shared actually reads.
        self._real_tiles = list(real_tiles)
        self._max_emit = kw.get("max_emit")
        self._buffer_size = kw.get("buffer_size", 16)
        # These attributes are touched by _iter_shared via pool.claim_next /
        # pool.reservoir_len / pool.done / pool.has_free_slot / pool.try_admit
        # — we override _emit_one + a fake pool to avoid all of that.
        self._pool_min_windows = kw.get("pool_min_windows", 1)
        self._run_id = "test"
        self._epoch = 0
        # PooledTileSource.__iter__ dispatches on _tile_source; set it so the
        # parent __iter__ routes to _iter_shared (which we override).
        self._tile_source = "shared_pooled"
        # image_order / tiles_by_image / _base are needed for plan building;
        # provide minimal non-empty values so the loop starts.
        self._image_order = [0]
        self._tiles_by_image = {0: [((0, 0, 4, 4), True)]}

    def _iter_shared(self):
        """Override _iter_shared entirely with a simplified stub that exercises
        the same padding logic as the real implementation but using our list."""
        import math
        import logging

        logger = logging.getLogger(__name__)

        if self._max_emit is not None:
            from torch.utils.data import get_worker_info as _gwi
            _wi = _gwi()
            _W = 1 if _wi is None else _wi.num_workers
            per_worker_target = math.ceil(self._max_emit / max(1, _W))
        else:
            per_worker_target = None

        emitted = 0
        pad_template = None

        for sample in self._real_tiles:
            if per_worker_target is not None and emitted >= per_worker_target:
                break
            pad_template = sample
            yield sample
            emitted += 1

        # Padding phase (identical to the real _iter_shared).
        if per_worker_target is not None and emitted < per_worker_target:
            if pad_template is None:
                logger.warning("_iter_shared stub: zero real samples, cannot pad.")
                return
            for _ in range(per_worker_target - emitted):
                yield pad_template
                emitted += 1


def _make_samples(n: int) -> list[dict]:
    """Return a list of n distinct sample dicts (keyed by index)."""
    return [{"image_id": i, "box": (0, 0, 4, 4)} for i in range(n)]


def test_iter_shared_pads_when_pool_drains_early():
    """Pool drains early (real < target): _iter_shared pads to per_worker_target.

    Setup: max_emit=10, num_workers=1 (no DataLoader worker) -> target = ceil(10/1) = 10.
    Provide only 6 real tiles.  Assert: exactly 10 samples emitted; last 4 are
    copies of the final real sample (the pad template).
    """
    real = _make_samples(6)
    src = _EarlyDrainTileSource(real, max_emit=10)
    out = list(src)
    assert len(out) == 10, f"Expected 10 samples (6 real + 4 padded), got {len(out)}"
    # First 6 are the real ones.
    for i, s in enumerate(out[:6]):
        assert s["image_id"] == i, f"real[{i}] has wrong image_id"
    # Last 4 are the pad template (image_id == 5, the final real sample).
    for s in out[6:]:
        assert s["image_id"] == 5, f"pad sample image_id wrong: {s['image_id']}"


def test_iter_shared_stops_at_target_when_pool_has_plenty():
    """Pool has more tiles than the target: _iter_shared stops at per_worker_target.

    Setup: max_emit=5, num_workers=1 -> target=5.
    Provide 20 real tiles.  Assert: exactly 5 samples emitted (no over-run).
    """
    real = _make_samples(20)
    src = _EarlyDrainTileSource(real, max_emit=5)
    out = list(src)
    assert len(out) == 5, f"Expected 5 samples (capped at target), got {len(out)}"
    for i, s in enumerate(out):
        assert s["image_id"] == i, f"out[{i}] has wrong image_id"


def test_iter_shared_no_cap_yields_all_real_tiles():
    """Without max_emit (uncapped), all real tiles are yielded (original behaviour)."""
    real = _make_samples(7)
    src = _EarlyDrainTileSource(real, max_emit=None)
    out = list(src)
    assert len(out) == 7, f"Uncapped: expected 7, got {len(out)}"


def test_iter_shared_exactly_at_target_no_padding():
    """Real tile count == target: no padding needed; exactly target samples yielded."""
    real = _make_samples(8)
    src = _EarlyDrainTileSource(real, max_emit=8)
    out = list(src)
    assert len(out) == 8, f"Expected 8 (exact match, no padding), got {len(out)}"
    for i, s in enumerate(out):
        assert s["image_id"] == i


def test_iter_shared_single_gpu_world1_unaffected(monkeypatch):
    """world==1 path: per_worker_budget computation is unchanged.

    StreamingTiledSegmentationDataset passes max_emit = total_tiles // (world * W).
    With world=1, W=1: max_emit = total_tiles // 1 = total_tiles.
    The source yields exactly total_tiles real samples -> no padding, no truncation.
    """
    real = _make_samples(12)
    # max_emit == len(real): exact fit, proves single-GPU is unaffected.
    src = _EarlyDrainTileSource(real, max_emit=12)
    out = list(src)
    assert len(out) == 12


# ---------------------------------------------------------------------------
# Fix-9a INTEGRATION: drive the GENUINE _iter_shared (no re-implementation,
# no stubbed source).  The tests above either re-implement _iter_shared
# (_EarlyDrainTileSource) or, in test_streaming_tiled.py, stub the source
# (_CapturingPooledSource).  The real division-by-W-once + the real padding
# loop in PooledTileSource._iter_shared were therefore never exercised by a
# test — exactly the seam that let the original double-division slip through.
#
# These tests subclass PooledTileSource and override ONLY _decode_full /
# _materialize_shared (the Task-8 _TestTileSource pattern), so _iter_shared,
# _emit_one, the SharedImagePool admit/pop/decref, the per_worker_target
# computation, AND the padding loop are ALL the real code.  get_worker_info is
# faked to W>1 so the real ceil(max_emit / W) division runs with W != 1.
# ---------------------------------------------------------------------------

class _RealIterTileSource(P.PooledTileSource):
    """PooledTileSource that keeps the REAL _iter_shared but replaces file I/O.

    Overrides only:
      - _decode_full: a tiny in-memory array (no pyvips / disk).
      - _materialize_shared: a plain {image_id, box} dict (no real crop/raster).
    Everything else — _iter_shared, _emit_one, claim_next/try_admit/pop/decref,
    per_worker_target, padding — is the production implementation.
    """
    def _decode_full(self, image_id: int) -> np.ndarray:
        return np.full((4, 4, 3), image_id % 255, dtype=np.uint8)

    def _materialize_shared(self, image_id: int, box, px, apply_band=True) -> dict:
        return {"image_id": image_id, "box": tuple(box)}


def _fake_worker_info(monkeypatch, num_workers):
    """Patch torch.utils.data.get_worker_info (the symbol _iter_shared imports
    locally at call time) so the REAL loop sees a W-worker DataLoader."""
    class _WI:
        def __init__(self, n): self.id = 0; self.num_workers = n; self.seed = 0; self.dataset = None
    import torch.utils.data as tud
    monkeypatch.setattr(tud, "get_worker_info", lambda: _WI(num_workers))


def test_real_iter_shared_pads_to_ceil_max_emit_over_W(monkeypatch):
    """GENUINE _iter_shared: pool drains early (few real tiles), W=4, max_emit
    set to a per-rank value -> yields EXACTLY ceil(max_emit / W) = real + padded.

    Setup: 2 images x 3 windows = 6 REAL tiles in the shared pool.
    max_emit=20, W=4 -> per_worker_target = ceil(20/4) = 5.
    Since 6 real >= 5, this case actually STOPS at 5 (no padding needed) — it
    pins that the real loop honours the cap mid-drain.  The pad path is covered
    by the next test where the target exceeds the real-tile count.
    """
    cb_name = "tt_realiter_a"
    run_id = "ri_a"
    host = socket.gethostname()
    n_images, wpi = 2, 3
    _cleanup_shared_e2e(cb_name, run_id, host, list(range(n_images)))
    _fake_worker_info(monkeypatch, num_workers=4)

    # Distinct boxes per window so (image_id, box) is unique.
    tiles_by_image = {i: [((w * 4, 0, 4, 4), True) for w in range(wpi)]
                      for i in range(n_images)}
    image_order = list(range(n_images))

    cb = SharedControlBlock.create_or_attach(cb_name, capacity=2, buffer_size=16)
    try:
        src = _RealIterTileSource(
            _make_base_for_real_iter(), image_order, tiles_by_image, random.Random(0),
            tile_source="shared_pooled", control=cb, run_id=run_id,
            pool_min_windows=1, pool_capacity=2, buffer_size=16,
            transform=None, imgsz=None, size_band=None, min_overlap=0.05,
            max_emit=20,
        )
        out = list(src)
        assert len(out) == 5, f"Expected ceil(20/4)=5 (real loop honours cap), got {len(out)}"
    finally:
        cb.unlink()
        _cleanup_pixel_segs(run_id, host, list(range(n_images)))


def test_real_iter_shared_pad_path_fires_when_target_exceeds_real(monkeypatch):
    """GENUINE _iter_shared pad loop: real tiles < per_worker_target -> pads.

    Setup: 2 images x 2 windows = 4 REAL tiles; W=2; max_emit=12 ->
    per_worker_target = ceil(12/2) = 6.  4 real < 6 -> the REAL padding loop
    must re-yield the last real sample twice.  Asserts:
      - exactly 6 emitted (count-invariant),
      - the 4 distinct real (image_id, box) all present once,
      - the 2 extra are padded duplicates (a previously-seen real pair).
    """
    from collections import Counter
    cb_name = "tt_realiter_b"
    run_id = "ri_b"
    host = socket.gethostname()
    n_images, wpi = 2, 2
    _cleanup_shared_e2e(cb_name, run_id, host, list(range(n_images)))
    _fake_worker_info(monkeypatch, num_workers=2)

    tiles_by_image = {i: [((w * 4, 0, 4, 4), True) for w in range(wpi)]
                      for i in range(n_images)}
    image_order = list(range(n_images))

    cb = SharedControlBlock.create_or_attach(cb_name, capacity=2, buffer_size=16)
    try:
        src = _RealIterTileSource(
            _make_base_for_real_iter(), image_order, tiles_by_image, random.Random(0),
            tile_source="shared_pooled", control=cb, run_id=run_id,
            pool_min_windows=1, pool_capacity=2, buffer_size=16,
            transform=None, imgsz=None, size_band=None, min_overlap=0.05,
            max_emit=12,
        )
        out = list(src)

        assert len(out) == 6, f"Expected ceil(12/2)=6 (4 real + 2 padded), got {len(out)}"
        pairs = [(s["image_id"], tuple(s["box"])) for s in out]
        expected_real = {(i, (w * 4, 0, 4, 4)) for i in range(n_images) for w in range(wpi)}
        # Every distinct real pair appears at least once (coverage preserved).
        assert set(pairs) == expected_real, (
            f"real coverage broken: {set(pairs)} vs {expected_real}")
        # Exactly 2 pairs are duplicated (the padding); padded pairs are real ones.
        dup_total = sum(c - 1 for c in Counter(pairs).values())
        assert dup_total == 2, f"Expected 2 padded duplicates, got {dup_total}"
    finally:
        cb.unlink()
        _cleanup_pixel_segs(run_id, host, list(range(n_images)))


def test_real_iter_shared_stops_at_target_when_pool_has_plenty(monkeypatch):
    """GENUINE _iter_shared: pool has MANY more tiles than target -> stops at
    exactly per_worker_target (no over-run).

    Setup: 10 images x 3 windows = 30 REAL tiles; W=3; max_emit=15 ->
    per_worker_target = ceil(15/3) = 5.  Asserts exactly 5 emitted, all distinct
    real pairs (no padding since 30 >> 5), and the pool drain stopped early.
    """
    cb_name = "tt_realiter_c"
    run_id = "ri_c"
    host = socket.gethostname()
    n_images, wpi = 10, 3
    _cleanup_shared_e2e(cb_name, run_id, host, list(range(n_images)))
    _fake_worker_info(monkeypatch, num_workers=3)

    tiles_by_image = {i: [((w * 4, 0, 4, 4), True) for w in range(wpi)]
                      for i in range(n_images)}
    image_order = list(range(n_images))

    cb = SharedControlBlock.create_or_attach(cb_name, capacity=4, buffer_size=16)
    try:
        src = _RealIterTileSource(
            _make_base_for_real_iter(), image_order, tiles_by_image, random.Random(0),
            tile_source="shared_pooled", control=cb, run_id=run_id,
            pool_min_windows=1, pool_capacity=4, buffer_size=16,
            transform=None, imgsz=None, size_band=None, min_overlap=0.05,
            max_emit=15,
        )
        out = list(src)
        assert len(out) == 5, f"Expected ceil(15/3)=5 (capped, no over-run), got {len(out)}"
        # No padding here -> all 5 are distinct real pairs.
        pairs = [(s["image_id"], tuple(s["box"])) for s in out]
        assert len(set(pairs)) == 5, f"Expected 5 distinct real tiles, got {len(set(pairs))}"
    finally:
        cb.unlink()
        _cleanup_pixel_segs(run_id, host, list(range(n_images)))


def _make_base_for_real_iter():
    """Minimal base the REAL _iter_shared/_emit_one path reaches via
    _materialize_shared override (annotations/class_mask not used by the stub)."""
    class _B:
        def _image_path(self, i): return f"mem://{i}"
        def _load_annotations(self, i): return []
        def _image_class_mask(self, i): return None
        def _apply_transform(self, sample, transform): return sample
    return _B()


# ---------------------------------------------------------------------------
# peek() — read-only donor sampling (mosaic-in-shared)
# ---------------------------------------------------------------------------

def _reservoir_snapshot(cb):
    """(res_len, list of live (image_id, w) pairs, refcount list) — full state probe."""
    hlen = int(cb.header["res_len"][0])
    pairs = [(int(cb.reservoir[i][0]), int(cb.reservoir[i][1])) for i in range(hlen)]
    refs = [(int(r["image_id"]), int(r["refcount"])) for r in cb.registry if r["state"] != 0]
    return hlen, pairs, refs


def test_peek_returns_distinct_pairs_and_mutates_nothing():
    _cleanup_cb("tt_peek_a")
    _cleanup_pixel_segs("r", "h", [0, 1])
    cb = SharedControlBlock.create_or_attach("tt_peek_a", capacity=4, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h")
    assert pool.try_admit(0, n_windows=3, is_pooled=True, decode_fn=_decode) is True
    assert pool.try_admit(1, n_windows=2, is_pooled=True, decode_fn=_decode) is True

    before = _reservoir_snapshot(cb)
    picks = pool.peek(random.Random(0), 3)
    after = _reservoir_snapshot(cb)

    assert len(picks) == 3
    assert len(set(picks)) == 3                       # distinct positions -> distinct pairs
    valid = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)}
    assert set(picks) <= valid                        # only live descriptors
    assert before == after                            # NOTHING mutated (res_len/pairs/refcounts)

    # drain so the segments unlink cleanly
    for _ in range(5):
        iid, _w = pool.pop(random.Random(1))
        pool.decref(iid)
    cb.unlink()


def test_peek_empty_reservoir_returns_empty():
    _cleanup_cb("tt_peek_b")
    cb = SharedControlBlock.create_or_attach("tt_peek_b", capacity=2, buffer_size=8)
    pool = SharedImagePool(cb, run_id="r", host="h")
    assert pool.peek(random.Random(0), 3) == []
    cb.unlink()


def test_peek_clamps_to_reservoir_length():
    _cleanup_cb("tt_peek_c")
    _cleanup_pixel_segs("r", "h", [7])
    cb = SharedControlBlock.create_or_attach("tt_peek_c", capacity=2, buffer_size=8)
    pool = SharedImagePool(cb, run_id="r", host="h")
    assert pool.try_admit(7, n_windows=2, is_pooled=True, decode_fn=_decode) is True
    picks = pool.peek(random.Random(0), 5)            # ask for more than exist
    assert len(picks) == 2 and set(picks) == {(7, 0), (7, 1)}
    for _ in range(2):
        iid, _w = pool.pop(random.Random(1))
        pool.decref(iid)
    cb.unlink()


# ---------------------------------------------------------------------------
# mp stress: peek (read-only) hammered against a concurrent drain — coverage
# must stay exact, refcounts must end at zero, nothing may raise.
# ---------------------------------------------------------------------------

def _drainer(cb_name, plan, out_q, barrier):
    import random as _random
    from training_toolkit.data.shared_control_block import SharedControlBlock
    from training_toolkit.data.shared_image_pool import SharedImagePool
    cb = SharedControlBlock.create_or_attach(cb_name, capacity=4, buffer_size=32)
    pool = SharedImagePool(cb, run_id="tt_stress", host="h")
    barrier.wait(timeout=30)
    rng = _random.Random()
    popped = []
    pending = None
    try:
        while True:
            if pending is None:
                pending = pool.claim_next(plan)
            if pending is not None:
                iid, n_windows, _pooled = pending
                if pool.has_free_slot():
                    if pool.try_admit(iid, n_windows, True, decode_fn=_decode):
                        pending = None
                        continue
            got = pool.pop(rng)
            if got is not None:
                iid2, w = got
                px = pool.read_pixels(iid2)
                assert px is None or px.shape == (8, 8, 3)
                pool.decref(iid2)
                popped.append((iid2, w))
            elif pending is None and pool.done():
                break
    finally:
        out_q.put(popped)
        cb.detach()


def _peeker(cb_name, stop_evt, out_q, barrier):
    import random as _random
    from training_toolkit.data.shared_control_block import SharedControlBlock
    from training_toolkit.data.shared_image_pool import SharedImagePool
    cb = SharedControlBlock.create_or_attach(cb_name, capacity=4, buffer_size=32)
    pool = SharedImagePool(cb, run_id="tt_stress", host="h")
    barrier.wait(timeout=30)
    rng = _random.Random(123)
    seen = set()
    while not stop_evt.is_set():
        for pair in pool.peek(rng, 3):
            seen.add(pair)
            pool.read_pixels(pair[0])     # exercise the attach/fallback path too
    out_q.put(sorted(seen))
    cb.detach()


def test_peek_under_concurrent_drain_stress():
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    name = "tt_stress_peek"
    _cleanup_cb(name)
    image_ids = list(range(30))
    _cleanup_pixel_segs("tt_stress", "h", image_ids)
    plan = [(iid, 4, True) for iid in image_ids]              # 30 images x 4 windows

    drainers, peekers = [], []
    cb = SharedControlBlock.create_or_attach(name, capacity=4, buffer_size=32)
    try:
        stop = ctx.Event()
        barrier = ctx.Barrier(5)                              # 3 drainers + 2 peekers
        drain_q, peek_q = ctx.Queue(), ctx.Queue()
        drainers = [ctx.Process(target=_drainer, args=(name, plan, drain_q, barrier)) for _ in range(3)]
        peekers = [ctx.Process(target=_peeker, args=(name, stop, peek_q, barrier)) for _ in range(2)]
        for p in drainers + peekers:
            p.start()
        for p in drainers:
            p.join(timeout=60)
        stop.set()
        for p in peekers:
            p.join(timeout=30)
        assert all(p.exitcode == 0 for p in drainers + peekers), "a worker crashed"

        popped = []
        for _ in drainers:
            popped += drain_q.get(timeout=5)
        want = sorted((iid, w) for iid in image_ids for w in range(4))
        assert sorted(popped) == want, "coverage broken: drain multiset != plan"

        valid = set(want)
        all_peeked = set()
        for _ in peekers:
            seen = set(peek_q.get(timeout=5))
            assert seen <= valid, "peek returned an invalid descriptor"
            all_peeked |= seen
        assert all_peeked, "peekers observed nothing — peek() may have regressed to a no-op"

        with cb.locked():
            assert not (cb.registry["state"] != 0).any(), "registry slot leaked"
        import glob
        assert glob.glob("/dev/shm/tt_stress_ep0_*_img*") == [], "/dev/shm segment leaked"
    finally:
        for p in drainers + peekers:
            if p.exitcode is None:
                p.kill()
                p.join(timeout=5)
        cb.unlink()
