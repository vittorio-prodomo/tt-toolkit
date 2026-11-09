import random
import numpy as np
import pytest
import training_toolkit.data.pooled_tile_source as P
from training_toolkit.data.pooled_tile_source import ImagePool
from training_toolkit.data.multi_sample import MultiSampleTransforms

def _arr(): return np.zeros((2, 2, 3), np.uint8)

def test_pool_admit_evict_cap():
    pool = ImagePool(capacity=2)
    s0 = pool.admit(_arr(), [], None, image_id=0, pending=2)
    s1 = pool.admit(_arr(), [], None, image_id=1, pending=1)
    assert len(pool) == 2 and not pool.has_free_slot()
    with pytest.raises(RuntimeError):
        pool.admit(_arr(), [], None, image_id=2, pending=1)   # no free slot
    pool.decref(s1)                                           # s1: 1 -> 0, frees
    assert len(pool) == 1 and pool.has_free_slot()
    pool.decref(s0)                                           # s0: 2 -> 1, stays
    assert len(pool) == 1 and pool.get(s0).pending == 1
    pool.decref(s0)                                           # s0: 1 -> 0, frees
    assert len(pool) == 0

def test_pool_rejects_zero_pending():
    with pytest.raises(ValueError):
        ImagePool(2).admit(_arr(), [], None, image_id=0, pending=0)

def test_pool_decref_underflow():
    pool = ImagePool(capacity=1)
    s = pool.admit(_arr(), [], None, image_id=0, pending=1)
    pool.decref(s)              # frees the slot
    with pytest.raises(RuntimeError):
        pool.decref(s)          # underflow -> RuntimeError


class _FakeBase:
    def _image_path(self, i): return f"mem://{i}"
    def _load_annotations(self, i): return []
    def _image_class_mask(self, i): return None
    def _apply_transform(self, sample, transform): return sample


def make_pooled_source_inputs(n_images: int, windows_per_image: int):
    """Return (base, image_order, tiles_by_image) for use in tests.

    Produces a _FakeBase, an ordered list of image ids, and a tiles_by_image
    dict where each image has `windows_per_image` tiles of size (4, 4).
    """
    base = _FakeBase()
    image_order = list(range(n_images))
    tiles_by_image = {
        i: [((0, 0, 4, 4), True)] * windows_per_image for i in range(n_images)
    }
    return base, image_order, tiles_by_image


def _patch_stubs(monkeypatch):
    monkeypatch.setattr(P, "decode_image_pyvips", lambda path: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(
        P, "materialize_labeled_tile_from_array",
        lambda array, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image_id": image_id, "box": tuple(box)},
    )

def _src(tiles, order, **kw):
    defaults = dict(transform=None, pool_capacity=2, buffer_size=3,
                    imgsz=None, size_band=None, min_overlap=0.05)
    defaults.update(kw)
    return P.PooledTileSource(_FakeBase(), order, tiles, random.Random(0), **defaults)

def test_pooled_covers_every_window_once(monkeypatch):
    _patch_stubs(monkeypatch)
    tiles = {0: [((0, 0, 5, 5), True), ((5, 0, 5, 5), True), ((0, 5, 5, 5), True)],
             1: [((0, 0, 5, 5), True)],
             2: [((0, 0, 5, 5), True), ((5, 5, 5, 5), True)]}
    out = list(_src(tiles, [0, 1, 2]))          # pool_capacity=2 < 3 images -> exercises eviction
    got = sorted((s["image_id"], s["box"]) for s in out)
    want = sorted((img, box) for img, ws in tiles.items() for (box, _b) in ws)
    assert got == want                          # every window exactly once: no drop, no dup

def test_pooled_low_window_no_deadlock(monkeypatch):
    _patch_stubs(monkeypatch)
    tiles = {i: [((0, 0, 5, 5), True)] for i in range(5)}    # 1 window/image << buffer_size
    out = list(_src(tiles, list(range(5)), pool_capacity=2, buffer_size=10))
    assert len(out) == 5                         # all emitted, no hang (M binds, reservoir runs shallow)


# ---------------------------------------------------------------------------
# Task-6 tests: mosaic-from-pool (donors borrowed, not consumed)
# ---------------------------------------------------------------------------

class _FakeAug:
    mosaic_p, cutmix_p, mixup_p, copy_paste_p = 1.0, 0.0, 0.0, 0.0

class _FakeBaseAug:
    _aug_config = _FakeAug()
    _mosaic_transform = None
    def _image_path(self, i): return f"mem://{i}"
    def _load_annotations(self, i): return []
    def _image_class_mask(self, i): return None
    def _apply_transform(self, sample, transform): return sample

def _fake_mosaic(primary, extras):
    return {**primary, "mosaic": True, "n_extras": len(extras)}

def test_pooled_mosaic_fires_and_preserves_coverage(monkeypatch):
    monkeypatch.setattr(P, "decode_image_pyvips", lambda path: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(
        P, "materialize_labeled_tile_from_array",
        lambda array, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image_id": image_id, "box": tuple(box)},
    )
    tiles = {i: [((0, 0, 5, 5), True), ((5, 5, 5, 5), True)] for i in range(6)}  # 12 windows
    ms = MultiSampleTransforms(mosaic=_fake_mosaic)
    src = P.PooledTileSource(
        _FakeBaseAug(), list(range(6)), tiles, random.Random(0), transform=None,
        pool_capacity=3, buffer_size=6, imgsz=416, size_band=None, min_overlap=0.05,
        multi_sample=ms, aug_config=_FakeAug(), mosaic_transform=None,
    )
    out = list(src)
    fired = [s for s in out if s.get("mosaic")]
    assert fired                                    # mosaic did fire
    assert all(s["n_extras"] == 3 for s in fired)   # when it fires: exactly 3 donors
    primaries = sorted((s["image_id"], s["box"]) for s in out)
    want = sorted((i, box) for i, ws in tiles.items() for (box, _b) in ws)
    assert primaries == want                                          # donors BORROWED, not consumed


# ---- hybrid window-count routing (pool_min_windows) ----

def test_hybrid_routes_low_window_images_lazy(monkeypatch):
    """High-window image -> pool (decode_image_pyvips); low-window -> lazy ROI (materialize_labeled_tile)."""
    pooled_imgs, lazy_calls = [], []
    monkeypatch.setattr(P, "decode_image_pyvips",
                        lambda path: pooled_imgs.append(path) or np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(P, "materialize_labeled_tile_from_array",
                        lambda array, box, instances, class_mask, realized_scale, band, min_overlap,
                               image_id, assignment_mode="bbox_ioa":
                            {"image_id": image_id, "box": tuple(box), "src": "pool"})
    monkeypatch.setattr(P, "materialize_labeled_tile",
                        lambda path, box, instances, class_mask, realized_scale, band, min_overlap,
                               image_id, assignment_mode="bbox_ioa":
                            (lazy_calls.append(path) or {"image_id": image_id, "box": tuple(box), "src": "lazy"}))
    tiles = {0: [((0, 0, 5, 5), True)] * 4,   # 4 windows -> pooled (>= 3)
             1: [((0, 0, 5, 5), True)]}        # 1 window  -> lazy
    src = P.PooledTileSource(_FakeBase(), [0, 1], tiles, random.Random(0), transform=None,
                             pool_capacity=2, buffer_size=8, imgsz=None, size_band=None,
                             min_overlap=0.05, pool_min_windows=3)
    out = list(src)
    assert sorted((s["image_id"], s["box"]) for s in out) == \
           sorted((i, box) for i, ws in tiles.items() for (box, _b) in ws)   # coverage complete
    assert all(s["src"] == "pool" for s in out if s["image_id"] == 0)
    assert all(s["src"] == "lazy" for s in out if s["image_id"] == 1)
    assert pooled_imgs == ["mem://0"] and lazy_calls == ["mem://1"]           # img1 NEVER decoded into the pool


def test_hybrid_lazy_admission_not_blocked_by_full_pool(monkeypatch):
    _patch_stubs(monkeypatch)
    monkeypatch.setattr(P, "materialize_labeled_tile",
                        lambda path, box, instances, class_mask, realized_scale, band, min_overlap,
                               image_id, assignment_mode="bbox_ioa":
                            {"image_id": image_id, "box": tuple(box)})
    tiles = {0: [((0, 0, 5, 5), True)] * 5, 1: [((0, 0, 5, 5), True)] * 5,   # 2 pooled, cap 1 -> one must wait
             2: [((0, 0, 5, 5), True)], 3: [((0, 0, 5, 5), True)]}            # 2 lazy
    src = P.PooledTileSource(_FakeBase(), [0, 1, 2, 3], tiles, random.Random(0), transform=None,
                             pool_capacity=1, buffer_size=20, imgsz=None, size_band=None,
                             min_overlap=0.05, pool_min_windows=3)
    out = list(src)                       # terminates + covers all 12 windows despite pool_capacity=1
    assert len(out) == 12


def test_hybrid_lazy_admitted_while_pool_full(monkeypatch):
    """A lazy image whose turn comes while the pool is FULL is admitted anyway
    (the lazy branch has no has_free_slot gate) -> reservoir/IID decoupled from M."""
    _patch_stubs(monkeypatch)
    monkeypatch.setattr(P, "materialize_labeled_tile",
                        lambda path, box, instances, class_mask, realized_scale, band, min_overlap,
                               image_id, assignment_mode="bbox_ioa":
                            {"image_id": image_id, "box": tuple(box)})
    tiles = {0: [((0, 0, 5, 5), True)] * 5,   # pooled, admitted first -> pool (cap 1) full
             1: [((0, 0, 5, 5), True)],        # lazy, immediately after img0 -> bypasses slot check
             2: [((0, 0, 5, 5), True)] * 5}    # pooled, must wait for img0 to drain
    src = P.PooledTileSource(_FakeBase(), [0, 1, 2], tiles, random.Random(0), transform=None,
                             pool_capacity=1, buffer_size=20, imgsz=None, size_band=None,
                             min_overlap=0.05, pool_min_windows=3)
    out = list(src)
    assert len(out) == 11                                  # all windows covered, terminates at cap 1
    assert (1, (0, 0, 5, 5)) in [(s["image_id"], s["box"]) for s in out]   # lazy img1 emitted


# ---------------------------------------------------------------------------
# Bug fix: _emit_one (shared path) must apply self._transform
# ---------------------------------------------------------------------------

class _FakeBaseCallTransform:
    """Like _FakeBase but _apply_transform actually calls the transform callable."""
    def _image_path(self, i): return f"mem://{i}"
    def _load_annotations(self, i): return []
    def _image_class_mask(self, i): return None
    def _apply_transform(self, sample, transform):
        return transform(sample)


class _MockSharedPool:
    """Minimal pool mock for exercising _emit_one without real shared memory.

    pop() returns a single valid (image_id, window_idx) on the first call, then
    None on all subsequent calls (signals epoch end to the caller).
    """
    def __init__(self, image_id: int, window_idx: int):
        self._result = (image_id, window_idx)
        self._called = False

    def pop(self, rng):
        if not self._called:
            self._called = True
            return self._result
        return None

    def read_pixels(self, image_id):
        # Return None -> _materialize_shared falls back to path-based decode
        return None

    def decref(self, image_id):
        pass


def test_emit_one_applies_transform(monkeypatch):
    """_emit_one (shared-pool emit path) must apply self._transform to each sample.

    Before the fix _emit_one yields the un-transformed sample; after the fix the
    transform callable is invoked and the marker key "transformed" is present.
    """
    # Stub out the file-based decode so no real I/O is needed.
    monkeypatch.setattr(
        P, "materialize_labeled_tile",
        lambda path, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image_id": image_id, "box": tuple(box)},
    )

    image_id = 7
    window_idx = 0
    tiles = {image_id: [((0, 0, 5, 5), True)]}

    def _stamp(sample):
        return {**sample, "transformed": True}

    src = P.PooledTileSource(
        _FakeBaseCallTransform(),
        [image_id],
        tiles,
        random.Random(0),
        transform=_stamp,
        pool_capacity=1,
        buffer_size=4,
        imgsz=None,
        size_band=None,
        min_overlap=0.05,
    )

    mock_pool = _MockSharedPool(image_id, window_idx)
    emitted = list(src._emit_one(mock_pool))

    assert len(emitted) == 1, "expected exactly one sample from _emit_one"
    assert emitted[0].get("transformed") is True, (
        "_emit_one did not apply self._transform — multi-scale collate will break"
    )


# ---------------------------------------------------------------------------
# Ladder unification: _maybe_multi_sample takes a borrow_fn (one ladder, two
# donor sources — in-process reservoir peek AND shared-pool peek)
# ---------------------------------------------------------------------------

def test_ladder_routes_through_borrow_fn():
    """The ladder asks borrow_fn for donors; 3 donors -> mosaic fires; fewer -> skipped."""
    ms = MultiSampleTransforms(mosaic=_fake_mosaic)
    src = P.PooledTileSource(
        _FakeBaseAug(), [], {}, random.Random(0), transform=None,
        pool_capacity=1, buffer_size=2, imgsz=416, size_band=None, min_overlap=0.05,
        multi_sample=ms, aug_config=_FakeAug(), mosaic_transform=None,
    )
    asked = []

    def borrow3(n):
        asked.append(n)
        return [{"donor": i} for i in range(n)]

    out, used = src._maybe_multi_sample({"image_id": 9}, borrow3)
    assert used is True and out["n_extras"] == 3 and asked == [3]

    def borrow_short(n):
        return [{"donor": 0}]                        # reservoir nearly empty

    out, used = src._maybe_multi_sample({"image_id": 9}, borrow_short)
    assert used is False and "mosaic" not in out     # <3 donors -> mosaic skipped


# ---------------------------------------------------------------------------
# Fix 2: shared path must honor per-window apply_band (whole-frame windows are
# "no band" by design — tile_index.py:103,283 — but _materialize_shared applied
# the size band unconditionally, diverging from buffered/pooled)
# ---------------------------------------------------------------------------

def test_materialize_shared_honors_apply_band(monkeypatch):
    seen_bands = []
    monkeypatch.setattr(
        P, "materialize_labeled_tile",
        lambda path, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            (seen_bands.append(band) or {"image_id": image_id, "box": tuple(box)}),
    )
    SENTINEL_BAND = object()
    tiles = {3: [((0, 0, 8, 8), False)],       # whole-frame-style: apply_band=False
             4: [((0, 0, 4, 4), True)]}         # grid-style: apply_band=True
    src = P.PooledTileSource(
        _FakeBaseCallTransform(), [3, 4], tiles, random.Random(0), transform=None,
        pool_capacity=1, buffer_size=4, imgsz=None, size_band=SENTINEL_BAND,
        min_overlap=0.05,
    )
    # Drive through _emit_one so the passthrough (not just the default) is tested.
    list(src._emit_one(_MockSharedPool(3, 0)))   # window with apply_band=False
    list(src._emit_one(_MockSharedPool(4, 0)))   # window with apply_band=True
    assert seen_bands == [None, SENTINEL_BAND], (
        "shared path must band ONLY apply_band=True windows (parity with "
        "buffered/pooled); whole-frame windows are no-band by design")


# ---------------------------------------------------------------------------
# Mosaic-in-shared: donor borrow via pool.peek + transform switch in _emit_one
# ---------------------------------------------------------------------------

class _MockSharedPoolPeek(_MockSharedPool):
    """_MockSharedPool + a peek() that serves donor descriptors from a fixed list."""
    def __init__(self, image_id, window_idx, peeks):
        super().__init__(image_id, window_idx)
        self._peeks = list(peeks)

    def peek(self, rng, n):
        return self._peeks[:n]


def _stub_materialize(monkeypatch):
    monkeypatch.setattr(
        P, "materialize_labeled_tile",
        lambda path, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image_id": image_id, "box": tuple(box)},
    )


def test_emit_one_mosaic_uses_mosaic_transform(monkeypatch):
    """A mosaic'd sample must go through mosaic_transform, NOT the standard
    transform — mosaic outputs a 2x canvas; resizing it with the standard
    transform crashes collate's torch.stack (multi-scale class of bug, cf.
    the 858f838 shared-transform fix)."""
    _stub_materialize(monkeypatch)
    tiles = {7: [((0, 0, 5, 5), True)],
             1: [((0, 0, 5, 5), True)], 2: [((0, 0, 5, 5), True)], 3: [((0, 0, 5, 5), True)]}
    ms = MultiSampleTransforms(mosaic=_fake_mosaic)
    src = P.PooledTileSource(
        _FakeBaseCallTransform(), [7, 1, 2, 3], tiles, random.Random(0),
        transform=lambda s: {**s, "tx": "standard"},
        pool_capacity=2, buffer_size=8, imgsz=416, size_band=None, min_overlap=0.05,
        multi_sample=ms, aug_config=_FakeAug(),                 # mosaic_p=1.0
        mosaic_transform=lambda s: {**s, "tx": "mosaic"},
    )
    pool = _MockSharedPoolPeek(7, 0, peeks=[(1, 0), (2, 0), (3, 0)])
    out = list(src._emit_one(pool))
    assert len(out) == 1
    assert out[0].get("mosaic") is True and out[0]["n_extras"] == 3
    assert out[0]["tx"] == "mosaic", (
        "_emit_one applied the standard transform to a mosaic'd (2x-canvas) sample")


def test_emit_one_no_mosaic_uses_standard_transform(monkeypatch):
    _stub_materialize(monkeypatch)
    tiles = {7: [((0, 0, 5, 5), True)]}
    src = P.PooledTileSource(
        _FakeBaseCallTransform(), [7], tiles, random.Random(0),
        transform=lambda s: {**s, "tx": "standard"},
        pool_capacity=1, buffer_size=4, imgsz=None, size_band=None, min_overlap=0.05,
    )
    out = list(src._emit_one(_MockSharedPool(7, 0)))
    assert out[0]["tx"] == "standard"


def test_borrow_donors_shared_freed_segment_falls_back_lazy(monkeypatch):
    """A donor whose segment was freed between peek and read resolves through
    the read_pixels->None lazy path (disk ROI decode) and bumps the counter."""
    _stub_materialize(monkeypatch)
    tiles = {5: [((0, 0, 5, 5), False)]}
    src = P.PooledTileSource(
        _FakeBaseCallTransform(), [5], tiles, random.Random(0), transform=None,
        pool_capacity=1, buffer_size=4, imgsz=None, size_band=None, min_overlap=0.05,
    )
    src._donor_lazy_fallbacks = 0
    pool = _MockSharedPoolPeek(5, 0, peeks=[(5, 0)])   # read_pixels always None in the mock
    donors = src._borrow_donors_shared(pool, 1)
    assert donors == [{"image_id": 5, "box": (0, 0, 5, 5)}]
    assert src._donor_lazy_fallbacks == 1


def test_shared_mosaic_end_to_end_preserves_coverage(monkeypatch):
    """Full _iter_shared drain with mosaic_p=1.0 on a real SharedControlBlock:
    mosaic fires with 3 donors AND every window is still emitted exactly once
    as a primary (donors are borrowed, never consumed)."""
    import socket
    from multiprocessing import shared_memory
    from training_toolkit.data.named_lock import NamedLock
    from training_toolkit.data.shared_control_block import SharedControlBlock

    def _cleanup_cb(name):
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close(); shm.unlink()
        except FileNotFoundError:
            pass
        NamedLock.unlink_name("/" + name + "_lock")

    monkeypatch.setattr(P, "decode_image_pyvips", lambda path: np.zeros((6, 6, 3), np.uint8))
    monkeypatch.setattr(
        P, "materialize_labeled_tile_from_array",
        lambda array, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
               assignment_mode="bbox_ioa":
            {"image_id": image_id, "box": tuple(box)},
    )
    _cleanup_cb("tt_moshared_e2e")
    cb = SharedControlBlock.create_or_attach("tt_moshared_e2e", capacity=3, buffer_size=8)
    try:
        tiles = {i: [((0, 0, 5, 5), True), ((5, 5, 5, 5), True)] for i in range(5)}  # 10 windows
        ms = MultiSampleTransforms(mosaic=_fake_mosaic)
        src = P.PooledTileSource(
            _FakeBaseAug(), list(range(5)), tiles, random.Random(0), transform=None,
            pool_capacity=3, buffer_size=8, imgsz=416, size_band=None, min_overlap=0.05,
            multi_sample=ms, aug_config=_FakeAug(), mosaic_transform=None,
            tile_source="shared_pooled", control=cb, run_id="tt_moshared_e2e",
            epoch=0, max_emit=None,
        )
        out = list(src)
        fired = [s for s in out if s.get("mosaic")]
        assert fired and all(s["n_extras"] == 3 for s in fired)
        primaries = sorted((s["image_id"], s["box"]) for s in out)
        want = sorted((i, box) for i, ws in tiles.items() for (box, _b) in ws)
        assert primaries == want            # coverage exactly-once, donors not consumed
    finally:
        cb.unlink()
