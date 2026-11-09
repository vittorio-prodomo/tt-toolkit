import random

import numpy as np
import training_toolkit.data.streaming_tiled_dataset as S
import training_toolkit.data.pooled_tile_source as P


class _FakeBase:
    def _image_path(self, i): return f"mem://{i}"
    def _load_annotations(self, i): return []
    def _image_class_mask(self, i): return None
    def _apply_transform(self, sample, transform): return sample


def test_balanced_partition_disjoint_exhaustive_balanced():
    from training_toolkit.data.streaming_tiled_dataset import balanced_partition
    counts = [10, 1, 9, 2, 8, 3, 7, 4]  # total 44
    parts = balanced_partition(counts, num_parts=4, rng=random.Random(0))
    flat = sorted(i for p in parts for i in p)
    assert flat == list(range(len(counts)))          # disjoint + exhaustive
    loads = [sum(counts[i] for i in p) for p in parts]
    assert max(loads) - min(loads) <= 3              # well balanced


def test_balanced_partition_deterministic_across_calls():
    from training_toolkit.data.streaming_tiled_dataset import balanced_partition
    counts = [5, 3, 8, 1, 6, 2]
    a = balanced_partition(counts, 3, random.Random(42))
    b = balanced_partition(counts, 3, random.Random(42))
    assert a == b


def _streaming(root, **kw):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    base = CocoSegmentationDataset(root=root, split="train", layout="coco")
    params = dict(tile_size=32, overlap=0.2, negative_ratio=0.0,
                  buffer_size=4, base_seed=1)
    params.update(kw)
    return StreamingTiledSegmentationDataset(base, **params)


def _make_multi_coco(tmp_path, n=3):
    """A COCO 'train' split with n distinct 64x64 images (so visit order is
    observable via image_id, unlike the single-image mini_coco fixture)."""
    import json
    import numpy as np
    from PIL import Image
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    images, anns = [], []
    for k in range(n):
        Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ).save(img_dir / f"img_{k:03d}.jpg", quality=100)
        images.append({"id": k + 1, "file_name": f"img_{k:03d}.jpg",
                       "width": 64, "height": 64})
        anns.append({"id": k + 1, "image_id": k + 1, "category_id": 0,
                     "iscrowd": 0, "bbox": [10, 10, 20, 20], "area": 400,
                     "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]})
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(
        {"images": images, "annotations": anns,
         "categories": [{"id": 0, "name": "crack"}]}))
    return tmp_path


def test_streaming_len_equals_total_tiles(mini_coco):
    ds = _streaming(mini_coco)
    assert len(ds) == sum(len(v) for v in ds._tiles_by_image.values())
    assert len(ds) >= 1


def test_streaming_len_realized_under_workers_droplast(tmp_path):
    """With drop_last + multiple workers each worker drops its own tail partial
    batch, so the realized batch count is below total_tiles // batch_size.
    __len__ must report that realized count so Lightning's num_training_batches
    <= what the loader actually yields — otherwise val_check_interval=1.0 targets
    a batch the epoch never reaches and per-epoch validation never fires.
    Defaults (batch_size=1, num_workers<=1) must still equal total tiles."""
    root = _make_multi_coco(tmp_path, n=6)
    ds = _streaming(root, tile_size=16, batch_size=4, num_workers=3)
    total = ds._total_tiles
    expected = max(1, total // 4 - 3) * 4
    assert len(ds) == expected
    assert len(ds) <= total


def test_streaming_nw0_covers_all_tiles_once(mini_coco):
    ds = _streaming(mini_coco)
    samples = list(iter(ds))
    assert len(samples) == len(ds)
    assert all(s["image"].shape == (32, 32, 3) for s in samples)


def test_streaming_epoch_deterministic_and_covers(tmp_path):
    root = _make_multi_coco(tmp_path, n=3)
    ds = _streaming(root, tile_size=32, buffer_size=2)
    ds.set_epoch(0); a = [s["image_id"] for s in iter(ds)]
    ds.set_epoch(0); a2 = [s["image_id"] for s in iter(ds)]
    ds.set_epoch(1); b = [s["image_id"] for s in iter(ds)]
    assert a == a2                  # deterministic at a fixed epoch (seeded)
    assert sorted(a) == sorted(b)   # same coverage (multiset) across epochs
    assert len(a) == len(ds)


def test_streaming_dataloader_spawn_covers_all(mini_coco):
    import torch
    from torch.utils.data import DataLoader
    from training_toolkit.data import collate_fn
    ds = _streaming(mini_coco, tile_size=16, buffer_size=4)
    loader = DataLoader(ds, batch_size=2, num_workers=2, collate_fn=collate_fn,
                        multiprocessing_context="spawn", drop_last=False)
    total = sum(batch["images"].shape[0] for batch in loader)
    assert total == len(ds)


def test_streaming_rejects_multisample_aug(mini_coco):
    import pytest
    from training_toolkit.configs import AugConfig
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    base = CocoSegmentationDataset(root=mini_coco, split="train", layout="coco")
    with pytest.raises(ValueError, match="multi-sample"):
        StreamingTiledSegmentationDataset(
            base, tile_size=32, negative_ratio=0.0,
            aug_config=AugConfig(mosaic_p=0.5),
        )


def test_streaming_applies_transform(mini_coco):
    calls = []
    def tf(image, masks, bboxes, labels):
        calls.append(1)
        return {"image": image, "masks": masks, "bboxes": bboxes, "labels": labels}
    ds = _streaming(mini_coco, transform=tf)
    out = list(iter(ds))
    assert len(out) == len(ds)
    assert len(calls) == len(out)


class _MiniBase:
    class_names = {0: "a"}; num_classes = 1
    def __len__(self): return 1
    def _image_path(self, i): return "mem://0"
    def _load_annotations(self, i): return []
    def _image_class_mask(self, i): return None
    def _image_regime(self, i): return "outdoor_tiled"
    def _load_tile_metadata(self, i): return 32, 32, np.zeros((0, 4), np.float32)


def test_pooled_allows_mosaic_buffered_rejects():
    import pytest
    from training_toolkit.configs import AugConfig
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    aug = AugConfig(imgsz=416, mosaic_p=0.5)
    # buffered + mosaic -> rejected
    with pytest.raises(ValueError):
        StreamingTiledSegmentationDataset(_MiniBase(), tile_sizes=[16], aug_config=aug,
                                          imgsz=416, tile_source="buffered")
    # pooled + mosaic -> allowed (builds the multi-sample transforms)
    ds = StreamingTiledSegmentationDataset(_MiniBase(), tile_sizes=[16], aug_config=aug,
                                           imgsz=416, tile_source="pooled", cache_size=4)
    assert ds._multi_sample is not None and ds._multi_sample.mosaic is not None
    # shared_pooled + mosaic -> now allowed too (mosaic-in-shared)
    ds2 = StreamingTiledSegmentationDataset(_MiniBase(), tile_sizes=[16], aug_config=aug,
                                            imgsz=416, tile_source="shared_pooled",
                                            cache_size=4, run_id="tt_guard_shared")
    assert ds2._multi_sample is not None and ds2._multi_sample.mosaic is not None


def test_buffered_and_pooled_emit_same_multiset(monkeypatch):
    mk = (lambda *a, **k: None)  # replaced below; both record (image_id, box)
    def rec(_src_or_array, box, instances, class_mask, realized_scale, band, min_overlap, image_id,
            assignment_mode="bbox_ioa"):
        return {"image_id": image_id, "box": tuple(box)}
    monkeypatch.setattr(S, "materialize_labeled_tile", rec)          # buffered (path-source)
    monkeypatch.setattr(P, "materialize_labeled_tile_from_array", rec)  # pooled (array-source)
    monkeypatch.setattr(P, "decode_image_pyvips", lambda path: np.zeros((4, 4, 3), np.uint8))
    tiles = {0: [((0, 0, 5, 5), True), ((5, 0, 5, 5), True), ((0, 5, 5, 5), True)],
             1: [((0, 0, 5, 5), True), ((5, 5, 5, 5), True)],
             2: [((0, 0, 5, 5), True)]}
    order = [0, 1, 2]
    buf = S.BufferedTileSource(_FakeBase(), order, tiles, random.Random(0),
                               buffer_size=4, imgsz=None, size_band=None, min_overlap=0.05)
    pool = P.PooledTileSource(_FakeBase(), order, tiles, random.Random(0), transform=None,
                              pool_capacity=2, buffer_size=4, imgsz=None, size_band=None, min_overlap=0.05)
    key = lambda s: (s["image_id"], s["box"])
    assert sorted(map(key, buf)) == sorted(map(key, pool))   # same coverage multiset


# ---------------------------------------------------------------------- #
# Fix #2: pooled cache_size IID parity-floor warning
# ---------------------------------------------------------------------- #

def test_pooled_warns_when_cache_size_below_floor(mini_coco, caplog):
    """Pooled with cache_size << IID parity floor emits a WARNING."""
    import logging
    with caplog.at_level(logging.WARNING):
        _streaming(mini_coco, tile_source="pooled", cache_size=1, buffer_size=1000)
    warning_text = "".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    assert "cache_size" in warning_text and "IID parity floor" in warning_text, (
        f"Expected parity-floor WARNING not found.\nCaplog text: {caplog.text}"
    )


def test_buffered_no_parity_warning(mini_coco, caplog):
    """Buffered tile_source never emits the IID parity-floor warning."""
    import logging
    with caplog.at_level(logging.WARNING):
        _streaming(mini_coco, tile_source="buffered")
    warning_text = "".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    assert "IID parity floor" not in warning_text, (
        f"Unexpected parity-floor WARNING on buffered path.\nCaplog text: {caplog.text}"
    )


def test_pooled_no_warning_when_cache_size_large_enough(mini_coco, caplog):
    """Pooled with cache_size >= floor: no parity-floor warning emitted."""
    import logging
    with caplog.at_level(logging.WARNING):
        _streaming(mini_coco, tile_source="pooled", cache_size=10000, buffer_size=1000)
    warning_text = "".join(
        r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    )
    assert "IID parity floor" not in warning_text, (
        f"Unexpected parity-floor WARNING when cache_size is large.\nCaplog text: {caplog.text}"
    )


def test_guardrail_floor_over_pooled_eligible(mini_coco, caplog):
    """The IID-floor warning is computed over POOLED-ELIGIBLE images only:
    pool_min_windows=1 -> image pooled -> warns; huge threshold -> none pooled -> silent."""
    import logging
    with caplog.at_level(logging.WARNING):
        _streaming(mini_coco, tile_source="pooled", cache_size=1, buffer_size=1000, pool_min_windows=1)
    assert "IID parity floor" in "".join(r.getMessage() for r in caplog.records)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        _streaming(mini_coco, tile_source="pooled", cache_size=1, buffer_size=1000, pool_min_windows=10**9)
    assert "IID parity floor" not in "".join(r.getMessage() for r in caplog.records)


def test_resolve_rank_world_explicit_wins(monkeypatch):
    import training_toolkit.data.streaming_tiled_dataset as S
    monkeypatch.setenv("RANK", "3"); monkeypatch.setenv("WORLD_SIZE", "8")
    assert S._resolve_rank_world(1, 2) == (1, 2)   # explicit overrides env


def test_resolve_rank_world_env_fallback(monkeypatch):
    import training_toolkit.data.streaming_tiled_dataset as S
    monkeypatch.setattr(S.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "1"); monkeypatch.setenv("WORLD_SIZE", "2")
    assert S._resolve_rank_world(None, None) == (1, 2)


def test_resolve_rank_world_local_rank_fallback(monkeypatch):
    import training_toolkit.data.streaming_tiled_dataset as S
    monkeypatch.setattr(S.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("LOCAL_RANK", "1"); monkeypatch.setenv("WORLD_SIZE", "2")
    assert S._resolve_rank_world(None, None) == (1, 2)


def test_resolve_rank_world_default_single_gpu(monkeypatch):
    import training_toolkit.data.streaming_tiled_dataset as S
    monkeypatch.setattr(S.torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert S._resolve_rank_world(None, None) == (0, 1)


def test_resolve_rank_world_dist_initialized_wins_over_env(monkeypatch):
    import training_toolkit.data.streaming_tiled_dataset as S
    monkeypatch.setattr(S.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(S.torch.distributed, "get_rank", lambda: 1)
    monkeypatch.setattr(S.torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setenv("RANK", "0"); monkeypatch.setenv("WORLD_SIZE", "1")
    assert S._resolve_rank_world(None, None) == (1, 2)


def test_global_shard_disjoint_exhaustive_over_rank_worker():
    from training_toolkit.data.streaming_tiled_dataset import _global_shard
    counts = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]   # 10 images
    world, workers, seed = 2, 3, 99
    seen = []
    for rank in range(world):
        for w in range(workers):
            seen.extend(_global_shard(counts, rank, world, w, workers,
                                      random.Random(seed)))
    assert sorted(seen) == list(range(len(counts)))   # disjoint + exhaustive


def test_global_shard_rank0_rank1_disjoint_and_covering():
    from training_toolkit.data.streaming_tiled_dataset import _global_shard
    counts = [5] * 12
    r0, r1 = set(), set()
    for w in range(2):
        r0 |= set(_global_shard(counts, 0, 2, w, 2, random.Random(7)))
        r1 |= set(_global_shard(counts, 1, 2, w, 2, random.Random(7)))
    assert r0.isdisjoint(r1)
    assert r0 | r1 == set(range(12))


def test_global_shard_reduces_to_per_worker_when_world1():
    from training_toolkit.data.streaming_tiled_dataset import (
        _global_shard, balanced_partition)
    counts = [3, 7, 2, 8, 1]
    for w in range(3):
        a = _global_shard(counts, 0, 1, w, 3, random.Random(5))
        b = balanced_partition(counts, 3, random.Random(5))[w]
        assert a == b   # world=1 -> global_parts=workers, global_id=worker_id


def test_streaming_ranks_see_disjoint_images(tmp_path):
    """rank 0 and rank 1 over the same dataset (world=2) stream disjoint image
    sets whose union covers exactly the full set -> DDP no-silent-duplication."""
    root = _make_multi_coco(tmp_path, n=8)
    full = _streaming(root, tile_size=32); full.set_epoch(0)
    all_ids = {s["image_id"] for s in iter(full)}

    def ids_for(rank):
        ds = _streaming(root, tile_size=32, rank=rank, world_size=2)
        ds.set_epoch(0)
        return {s["image_id"] for s in iter(ds)}

    r0, r1 = ids_for(0), ids_for(1)
    assert r0.isdisjoint(r1)
    assert r0 | r1 == all_ids


def test_streaming_len_divides_by_world_size(tmp_path):
    root = _make_multi_coco(tmp_path, n=8)
    full = _streaming(root, tile_size=32)                       # world defaults to 1
    sharded = _streaming(root, tile_size=32, rank=0, world_size=2)
    assert sharded._total_tiles == full._total_tiles            # same global index
    assert len(full) == full._total_tiles                       # world=1 byte-identical
    assert len(sharded) <= full._total_tiles // 2 + sharded._batch_size


# ---------------------------------------------------------------------- #
# shared_pooled tile_source — __len__, _uses_global_shard, iter  #
# ---------------------------------------------------------------------- #

def _make_ds_shared(monkeypatch, tile_source="shared_pooled", total_tiles=1000,
                    batch_size=4, world_size=None, num_workers=0, run_id=None):
    """Build a StreamingTiledSegmentationDataset with a fake tile index
    that yields exactly ``total_tiles`` tiles (split across 10 images, 100 each).

    monkeypatch is used to replace build_tile_index so no disk I/O occurs.
    """
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset

    n_images = 10
    tiles_per_img = total_tiles // n_images
    # Build a fake grouped dict: image ids 0..n_images-1, each with tiles_per_img tiles.
    fake_grouped = {
        i: [((0, 0, 32, 32), True)] * tiles_per_img
        for i in range(n_images)
    }
    monkeypatch.setattr(S, "build_tile_index", lambda *a, **k: fake_grouped)

    kw = dict(tile_size=32, overlap=0.0, negative_ratio=0.0,
              batch_size=batch_size, buffer_size=4, num_workers=num_workers,
              tile_source=tile_source)
    if world_size is not None:
        # _resolve_rank_world requires BOTH rank and world_size to honour the
        # explicit override; pass rank=0 alongside world_size.
        kw["rank"] = 0
        kw["world_size"] = world_size
    if run_id is not None:
        kw["run_id"] = run_id
    return StreamingTiledSegmentationDataset(_MiniBase(), **kw)


def _legacy_buffered_len(total_tiles, batch_size, world=1, num_workers=0):
    """The pre-Task-7 buffered/pooled __len__ formula, re-implemented here
    INDEPENDENTLY of the subject code so the test genuinely pins behavior
    rather than tautologically importing whatever __len__ now computes."""
    bs = max(1, batch_size)
    workers = max(1, num_workers)
    per_rank = total_tiles // max(1, world)
    if workers == 1:
        n_batches = per_rank // bs
    else:
        n_batches = per_rank // bs - workers
    return max(1, n_batches) * bs


def test_shared_len_is_fixed_per_rank(monkeypatch):
    """shared_pooled __len__ == bs * (total_tiles // (world * bs)) at workers<=1."""
    ds = _make_ds_shared(monkeypatch, tile_source="shared_pooled",
                         total_tiles=1000, batch_size=4, world_size=2)
    expected = 4 * (1000 // (2 * 4))  # = 4 * 125 = 500
    assert len(ds) == expected


def _shared_realized_len(total_tiles, batch_size, world=1, num_workers=0):
    """Independently re-derived shared_pooled realized __len__ (NOT importing the
    subject formula — pins behavior rather than tautology).

    The emit-cap makes each of the W DataLoader workers yield
    ``ceil(per_rank_samples / W)`` samples (proven empirically by
    test_shared_pool_coverage.test_multiprocess_capped_emit_per_worker_target);
    the DataLoader then batches EACH worker's stream with drop_last, dropping
    every worker's partial tail.  So the rank delivers
    ``W * floor(ceil(S/W) / bs)`` batches.  __len__ must report exactly that
    (× bs) so Lightning's num_training_batches == the batches the loader
    actually yields — else val_check_interval=1.0 targets a batch the epoch
    never reaches and per-epoch validation silently never fires.
    """
    import math
    bs = max(1, batch_size)
    workers = max(1, num_workers)
    S = bs * (total_tiles // max(1, world * bs))      # per-rank sample budget
    per_worker = math.ceil(S / workers)
    realized_batches = workers * (per_worker // bs)
    return max(1, realized_batches) * bs


def test_shared_len_realized_under_workers_droplast(monkeypatch):
    """shared_pooled __len__ must subtract the per-worker drop_last loss, exactly
    like buffered does (test_streaming_len_realized_under_workers_droplast).
    Otherwise num_training_batches > batches the loader yields, the epoch
    StopIterations before val_check_interval=1.0's target, and per-epoch
    validation silently never fires — the build-2 DDP crash.  The bug needs only
    workers>1 (DDP world just scales it); workers<=1 is unchanged."""
    import math
    # (1) single-process, workers>1 — the bug manifests without DDP at all.
    ds = _make_ds_shared(monkeypatch, tile_source="shared_pooled",
                         total_tiles=1000, batch_size=4, world_size=1, num_workers=3)
    assert _shared_realized_len(1000, 4, world=1, num_workers=3) == 996   # pin literal
    assert len(ds) == 996
    assert len(ds) // 4 == 3 * (math.ceil((4 * (1000 // 4)) / 3) // 4)    # 249 realized batches

    # (2) the real build-2 case: DDP world=2, workers=4.
    ds2 = _make_ds_shared(monkeypatch, tile_source="shared_pooled",
                          total_tiles=1000, batch_size=4, world_size=2, num_workers=4)
    assert _shared_realized_len(1000, 4, world=2, num_workers=4) == 496   # pin literal
    assert len(ds2) == 496

    # (3) workers<=1 unchanged (no per-worker drop) — no regression.
    ds1 = _make_ds_shared(monkeypatch, tile_source="shared_pooled",
                          total_tiles=1000, batch_size=4, world_size=2, num_workers=0)
    assert len(ds1) == 4 * (1000 // (2 * 4))   # 500, byte-identical to before


def test_shared_mode_uses_full_image_order_not_global_shard(monkeypatch):
    """_uses_global_shard() is False in shared_pooled mode."""
    ds = _make_ds_shared(monkeypatch, tile_source="shared_pooled")
    assert ds._uses_global_shard() is False


def test_buffered_len_unchanged(monkeypatch):
    """buffered __len__ is byte-identical to the pre-Task-7 formula on BOTH
    branches: num_workers<=1 (per_rank//bs) AND num_workers>1 (per_rank//bs -
    workers). Expected values are computed from the independent _legacy_buffered_len
    so a regression in either branch is caught."""
    total_tiles, batch_size = 1000, 4

    # Branch 1: num_workers=0 -> workers==1 formula -> 1000//4 * 4 = 1000.
    ds0 = _make_ds_shared(monkeypatch, tile_source="buffered",
                          total_tiles=total_tiles, batch_size=batch_size,
                          num_workers=0)
    expected0 = _legacy_buffered_len(total_tiles, batch_size, num_workers=0)
    assert expected0 == 1000                                   # pin the literal
    assert len(ds0) == expected0
    assert ds0._uses_global_shard() is True

    # Branch 2: num_workers=4 -> drop-last branch -> (1000//4 - 4) * 4 = 984.
    ds4 = _make_ds_shared(monkeypatch, tile_source="buffered",
                          total_tiles=total_tiles, batch_size=batch_size,
                          num_workers=4)
    expected4 = _legacy_buffered_len(total_tiles, batch_size, num_workers=4)
    assert expected4 == 984                                    # pin the literal
    assert len(ds4) == expected4
    assert ds4._uses_global_shard() is True


# ---------------------------------------------------------------------------
# Fix 9a composition: dataset -> PooledTileSource max_emit threading (DDP)
#
# The isolated _iter_shared tests pin its arithmetic with a HAND-PICKED
# max_emit, so they cannot see a double-division bug in the dataset->pool
# threading.  These tests exercise the real StreamingTiledSegmentationDataset
# .__iter__ in shared_pooled mode and assert:
#   1. max_emit passed to PooledTileSource == self.__len__()  (the PER-RANK cap,
#      NOT a per-worker value pre-divided by W), and
#   2. one worker yields ceil(__len__/W) samples — NOT ceil(__len__/W^2) — so
#      W workers together deliver W*ceil(__len__/W) >= __len__, the rank's full
#      DataLoader length.  A regression to the per-worker pre-division would
#      yield W* too few -> the DDP deadlock.
# ---------------------------------------------------------------------------

class _CapturingPooledSource:
    """Stand-in for PooledTileSource that (a) records the kwargs the dataset
    passed it (so we can read max_emit) and (b) reproduces _iter_shared's
    per-worker-target + early-drain padding arithmetic over a finite, EARLY-
    DRAINING tile list — so the count it yields is governed by the SAME
    max_emit/W math the real loop uses.

    The class attribute ``last`` captures the most recent instance for assertions.
    ``n_real_tiles`` controls how many real samples drain before padding kicks in.
    """
    last = None
    n_real_tiles = 1          # drains almost immediately -> forces the pad path
    sim_num_workers = 1       # W to simulate (real loop reads get_worker_info())

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.max_emit = kwargs.get("max_emit")
        type(self).last = self

    def __iter__(self):
        import math
        target = (None if self.max_emit is None
                  else math.ceil(self.max_emit / max(1, self.sim_num_workers)))
        emitted = 0
        pad = None
        for i in range(self.n_real_tiles):
            if target is not None and emitted >= target:
                return
            pad = {"image_id": i, "box": (0, 0, 4, 4)}
            yield pad
            emitted += 1
        if target is not None and pad is not None:
            while emitted < target:
                yield pad
                emitted += 1


class _FakeWorkerInfo:
    """Stand-in for torch's WorkerInfo so __iter__'s _worker_id_and_count()
    reports a multi-worker DataLoader (num_workers > 1) WITHOUT a real
    DataLoader.  This is essential: the double-division bug only manifests when
    W > 1, and __iter__ reads W from get_worker_info()."""
    def __init__(self, id, num_workers):
        self.id = id
        self.num_workers = num_workers
        self.seed = 0
        self.dataset = None


def _run_shared_iter_once(monkeypatch, total_tiles, batch_size, world_size,
                          sim_num_workers):
    """Build a shared_pooled dataset and drain ONE worker's __iter__ with the
    SharedControlBlock + PooledTileSource replaced by no-op / capturing stubs,
    and get_worker_info() faked to report ``sim_num_workers`` workers.

    Returns (dataset, list_of_samples).  ``sim_num_workers`` is W as seen by
    BOTH __iter__ (via _worker_id_and_count -> get_worker_info) and the stub
    source's ceil(max_emit / W) division — so a per-worker pre-division in the
    dataset shows up as a W^2 shortfall.
    """
    import training_toolkit.data.shared_control_block as SCB
    import training_toolkit.data.pooled_tile_source as PTS

    # Fake a multi-worker DataLoader so __iter__'s W == sim_num_workers.
    monkeypatch.setattr(S, "get_worker_info",
                        lambda: _FakeWorkerInfo(0, sim_num_workers))

    # No-op control block (no real shared memory).
    class _FakeCB:
        @staticmethod
        def create_or_attach(*a, **k): return _FakeCB()
        def detach(self): pass
        def unlink(self): pass
    monkeypatch.setattr(SCB, "SharedControlBlock", _FakeCB)

    # Capturing source that reproduces the per-worker-target math.
    _CapturingPooledSource.last = None
    _CapturingPooledSource.sim_num_workers = sim_num_workers
    monkeypatch.setattr(PTS, "PooledTileSource", _CapturingPooledSource)

    ds = _make_ds_shared(monkeypatch, tile_source="shared_pooled",
                         total_tiles=total_tiles, batch_size=batch_size,
                         world_size=world_size)
    samples = list(iter(ds))
    return ds, samples


def test_shared_iter_threads_per_rank_len_as_max_emit(monkeypatch):
    """__iter__ passes max_emit == __len__() (per-RANK), not a per-worker value."""
    ds, _ = _run_shared_iter_once(monkeypatch, total_tiles=1000, batch_size=4,
                                  world_size=2, sim_num_workers=4)
    captured = _CapturingPooledSource.last
    assert captured is not None, "PooledTileSource was never constructed"
    # __len__ (shared) = 4 * (1000 // (2*4)) = 4 * 125 = 500.
    assert len(ds) == 500
    assert captured.max_emit == len(ds) == 500, (
        f"max_emit must be the per-rank cap __len__()={len(ds)}, "
        f"got {captured.max_emit} (a per-worker pre-division is the deadlock bug)")


def test_shared_iter_one_worker_yields_ceil_len_over_W(monkeypatch):
    """One worker yields ceil(__len__/W) — NOT ceil(__len__/W^2) — and
    W*per_worker >= __len__ (rank delivers at least its DataLoader length)."""
    import math
    total_tiles, batch_size, world_size, W = 1000, 4, 2, 4
    ds, samples = _run_shared_iter_once(
        monkeypatch, total_tiles=total_tiles, batch_size=batch_size,
        world_size=world_size, sim_num_workers=W)

    L = len(ds)                                   # per-rank cap = 500
    per_worker = len(samples)
    correct = math.ceil(L / W)                    # ceil(500/4) = 125
    buggy = math.ceil(L / (W * W))                # the double-division: ceil(500/16)=32
    assert per_worker == correct, (
        f"worker yielded {per_worker}, expected ceil(__len__/W)={correct} "
        f"(double-division bug would give {buggy})")
    assert per_worker != buggy                    # explicit: the bug is excluded
    # The rank's W workers together deliver >= the DataLoader's per-rank length.
    assert W * per_worker >= L, (
        f"W*per_worker={W * per_worker} < __len__={L}: rank short on batches -> DDP desync")


def test_shared_iter_single_gpu_world1_full_length(monkeypatch):
    """world=1, W=1: max_emit == __len__ == total_tiles; worker yields the full
    per-rank length with no truncation (single-GPU unaffected)."""
    ds, samples = _run_shared_iter_once(
        monkeypatch, total_tiles=1000, batch_size=4, world_size=1,
        sim_num_workers=1)
    # __len__ = 4 * (1000 // (1*4)) = 1000.
    assert len(ds) == 1000
    assert _CapturingPooledSource.last.max_emit == 1000
    assert len(samples) == 1000                   # ceil(1000/1), no padding shortfall
