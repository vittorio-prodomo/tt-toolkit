"""Byte-aware shared-pool cap (cache_bytes).

The pool's cache_size counts IMAGES; on cohorts whose decoded sizes span
orders of magnitude (one real dataset: 0.2-68.7 MB) an image-count cap can blow
/dev/shm — observed live 2026-07-02 (M=1280 x 2 coexisting epoch pools ==
126 GB == /dev/shm -> torch IPC ENOSPC + DataLoader Bus errors). cache_bytes
adds an opt-in resident-bytes budget enforced at admission from a
declared-dims hint; None (default) = byte-identical to before.
"""
from __future__ import annotations

import random
from multiprocessing import shared_memory

import numpy as np
import pytest

from training_toolkit.data.shared_control_block import SharedControlBlock
from training_toolkit.data.shared_image_pool import SharedImagePool


def _decode(image_id):
    return np.full((8, 8, 3), image_id % 255, dtype=np.uint8)


def _cleanup_cb(name):
    for seg in (name,):
        try:
            s = shared_memory.SharedMemory(name=seg)
            s.close()
            s.unlink()
        except FileNotFoundError:
            pass


def _cleanup_pixel_segs(run_id, host, ids, epoch=0):
    for i in ids:
        try:
            s = shared_memory.SharedMemory(name=f"{run_id}_ep{epoch}_{host}_img{i}")
            s.close()
            s.unlink()
        except FileNotFoundError:
            pass


def _drain_one(pool, image_id, n):
    for _ in range(n):
        pool.pop(random.Random(0))
        pool.decref(image_id)


def test_byte_budget_refuses_admission_when_full():
    _cleanup_cb("tt_bc_a")
    _cleanup_pixel_segs("r", "h", [0, 1])
    cb = SharedControlBlock.create_or_attach("tt_bc_a", capacity=4, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h", cache_bytes=1000)
    assert pool.try_admit(0, n_windows=2, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=600) is True
    # slots are free (capacity 4) but the byte budget is: 600 + 600 > 1000
    assert pool.has_free_slot() is True
    assert pool.try_admit(1, n_windows=2, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=600) is False
    assert int(cb.header["resident_bytes"][0]) == 600
    # freeing image 0 refunds its charge -> image 1 now admits
    _drain_one(pool, 0, 2)
    assert int(cb.header["resident_bytes"][0]) == 0
    assert pool.try_admit(1, n_windows=2, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=600) is True
    _drain_one(pool, 1, 2)
    cb.unlink()


def test_min_progress_oversized_image_admits_into_empty_pool():
    _cleanup_cb("tt_bc_b")
    _cleanup_pixel_segs("r", "h", [7, 8])
    cb = SharedControlBlock.create_or_attach("tt_bc_b", capacity=2, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h", cache_bytes=100)
    # hint 500 > budget 100, but the pool is byte-empty: admit anyway
    # (otherwise this image could NEVER be admitted -> infinite spin)
    assert pool.try_admit(7, n_windows=1, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=500) is True
    # a second oversized image must wait
    assert pool.try_admit(8, n_windows=1, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=500) is False
    _drain_one(pool, 7, 1)
    cb.unlink()


def test_no_budget_default_never_refuses_on_bytes():
    _cleanup_cb("tt_bc_c")
    _cleanup_pixel_segs("r", "h", [2, 3])
    cb = SharedControlBlock.create_or_attach("tt_bc_c", capacity=2, buffer_size=16)
    pool = SharedImagePool(cb, run_id="r", host="h")   # cache_bytes=None
    assert pool.try_admit(2, n_windows=1, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=10**12) is True
    assert pool.try_admit(3, n_windows=1, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=10**12) is True
    # accounting still tracks (telemetry) even without a gate
    assert int(cb.header["resident_bytes"][0]) == 2 * 10**12
    _drain_one(pool, 2, 1)
    _drain_one(pool, 3, 1)
    assert int(cb.header["resident_bytes"][0]) == 0
    cb.unlink()


def test_full_reservoir_rollback_refunds_bytes():
    _cleanup_cb("tt_bc_d")
    _cleanup_pixel_segs("r", "h", [4, 5])
    cb = SharedControlBlock.create_or_attach("tt_bc_d", capacity=2, buffer_size=1)
    pool = SharedImagePool(cb, run_id="r", host="h", cache_bytes=10_000)
    assert pool.try_admit(4, n_windows=1, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=300) is True      # fills the 1-deep reservoir
    # image 5 claims a slot + charges bytes, but pushes 0 descriptors
    # (reservoir full) -> the stranded-slot rollback must refund its charge
    assert pool.try_admit(5, n_windows=1, is_pooled=True, decode_fn=_decode,
                          nbytes_hint=400) is False
    assert int(cb.header["resident_bytes"][0]) == 300
    _drain_one(pool, 4, 1)
    assert int(cb.header["resident_bytes"][0]) == 0
    cb.unlink()


def test_tiling_config_cache_bytes_validation():
    from training_toolkit.configs.tiling_config import TilingConfig

    assert TilingConfig().cache_bytes is None            # default: off
    TilingConfig(cache_bytes=2 * 1024**3)                # valid
    with pytest.raises(ValueError):
        TilingConfig(cache_bytes=0)
    with pytest.raises(ValueError):
        TilingConfig(cache_bytes=-5)
