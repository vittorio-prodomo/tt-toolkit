"""Streaming (IterableDataset) on-the-fly tiling with per-worker locality.

Per worker: take a tile-count-balanced slice of images, stream each image's
tiles sequentially (decode-once locality), and pass them through a shuffle
buffer for within-batch IID. Shards per (rank x worker) for DDP; multi-sample
augmentation requires the pooled tile source.
"""
from __future__ import annotations

import logging
import os
import random
import statistics
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import IterableDataset, get_worker_info

from training_toolkit.configs import AugConfig
from training_toolkit.data.base_dataset import BaseSegmentationDataset
from training_toolkit.data.label_crop import SizeBand, label_crop
from training_toolkit.data.shuffle_buffer import ShuffleBuffer
from training_toolkit.data.tile_index import build_tile_index, load_negative_blocklist
from training_toolkit.data.tile_materialize import (
    decode_crop_pyvips, crop_from_array, materialize_tile,
)

logger = logging.getLogger(__name__)


def _pack_labeled_tile(img, box, instances, class_mask, realized_scale,
                       band, min_overlap, image_id, assignment_mode="bbox_ioa"):
    """Shared tail: size-band label the tile via label_crop + pack the dict.
    Pixel-source-agnostic (label_crop takes no pixels), so the file-ROI and
    array materializers cannot drift."""
    labeled = label_crop(box, instances, realized_scale, class_mask,
                         band=band, min_overlap=min_overlap,
                         assignment_mode=assignment_mode)
    sample = {
        "image": img,
        "masks": labeled["masks"],
        "boxes": labeled["boxes"],
        "labels": labeled["labels"],
        "ignore": labeled["ignore"],
        "image_id": image_id,
    }
    if class_mask is not None:
        sample["class_mask"] = labeled["class_mask"]
    return sample


def materialize_labeled_tile(path, box, instances, class_mask, realized_scale,
                             band, min_overlap, image_id, assignment_mode="bbox_ioa"):
    """Decode one tile (pyvips ROI) + size-band label it (buffered path)."""
    img = decode_crop_pyvips(path, box)
    return _pack_labeled_tile(img, box, instances, class_mask, realized_scale,
                              band, min_overlap, image_id, assignment_mode)


def materialize_labeled_tile_from_array(array, box, instances, class_mask,
                                        realized_scale, band, min_overlap, image_id,
                                        assignment_mode="bbox_ioa"):
    """Crop one tile from a resident decoded array + size-band label it (pooled path)."""
    img = crop_from_array(array, box)
    return _pack_labeled_tile(img, box, instances, class_mask, realized_scale,
                              band, min_overlap, image_id, assignment_mode)


def balanced_partition(counts: list[int], num_parts: int,
                       rng: random.Random) -> list[list[int]]:
    """Partition indices [0, len(counts)) into num_parts groups of ~equal total
    count. Greedy longest-processing-time: shuffle (for deterministic tie-break
    given rng), sort by count desc, assign each to the currently-lightest bin.
    Deterministic given rng; same rng across workers -> identical partition."""
    order = list(range(len(counts)))
    rng.shuffle(order)
    order.sort(key=lambda i: counts[i], reverse=True)
    parts: list[list[int]] = [[] for _ in range(num_parts)]
    loads = [0] * num_parts
    for i in order:
        j = min(range(num_parts), key=lambda k: loads[k])
        parts[j].append(i)
        loads[j] += counts[i]
    return parts


def _global_shard(counts: list[int], rank: int, world_size: int,
                  worker_id: int, num_workers: int,
                  part_rng: random.Random) -> list[int]:
    """Image positions for this (rank, worker) under DDP.

    Partition all images into ``world_size * num_workers`` balanced groups and
    return the one for this process's global id ``rank * num_workers +
    worker_id``. The caller MUST pass a freshly-seeded ``part_rng`` keyed on the
    epoch ONLY (no rank/worker term) so every (rank, worker) computes the
    identical partition -> the per-process slices are disjoint and together
    cover all images.
    """
    global_id = rank * num_workers + worker_id
    global_parts = world_size * num_workers
    return balanced_partition(counts, global_parts, part_rng)[global_id]


def _worker_id_and_count() -> tuple[int, int]:
    info = get_worker_info()
    return (0, 1) if info is None else (info.id, info.num_workers)


def _resolve_rank_world(explicit_rank: int | None,
                        explicit_world: int | None) -> tuple[int, int]:
    """Resolve (rank, world_size) for DDP image sharding.

    Priority: explicit constructor override (both must be set) -> live process
    group (reliable in the rank's main process, e.g. __len__) -> RANK/WORLD_SIZE
    env vars (inherited by spawn workers, e.g. __iter__) -> (0, 1) single-GPU.
    """
    if explicit_rank is not None and explicit_world is not None:
        return explicit_rank, explicit_world
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
    return rank, world


class BufferedTileSource:
    """Option B: per-tile pyvips ROI decode -> ShuffleBuffer of materialized
    tiles. The validated default path; behavior unchanged from the original
    __iter__. Yields PRE-transform samples (the dataset applies the
    single-sample transform)."""

    def __init__(self, base, image_order, tiles_by_image, work_rng, *,
                 buffer_size, imgsz, size_band, min_overlap,
                 assignment_mode="bbox_ioa"):
        self._base = base
        self._image_order = image_order
        self._tiles_by_image = tiles_by_image
        self._rng = work_rng
        self._buffer_size = buffer_size
        self._imgsz = imgsz
        self._size_band = size_band
        self._min_overlap = min_overlap
        self._assignment_mode = assignment_mode

    def __iter__(self):
        def source():
            for img_idx in self._image_order:
                path = self._base._image_path(img_idx)
                instances = self._base._load_annotations(img_idx)
                class_mask = self._base._image_class_mask(img_idx)
                windows = list(self._tiles_by_image[img_idx])
                self._rng.shuffle(windows)
                for box, apply_band in windows:
                    realized_scale = (self._imgsz / max(box[2], box[3])) if self._imgsz else 1.0
                    yield materialize_labeled_tile(
                        path, box, instances, class_mask,
                        realized_scale=realized_scale,
                        band=(self._size_band if apply_band else None),
                        min_overlap=self._min_overlap, image_id=img_idx,
                        assignment_mode=self._assignment_mode,
                    )
        yield from ShuffleBuffer(source(), self._buffer_size, self._rng)


class StreamingTiledSegmentationDataset(IterableDataset):
    def __init__(
        self,
        base_dataset: BaseSegmentationDataset,
        tile_size: int = 416,
        tile_sizes: list[int] | None = None,
        regime_tile_sizes: dict[str, list[int]] | None = None,
        overlap: float = 0.2,
        negative_ratio: float = 0.15,
        min_annotation_overlap: float = 0.05,
        buffer_size: int = 256,
        base_seed: int = 42,
        batch_size: int = 1,
        num_workers: int = 0,
        transform: Callable | None = None,
        aug_config: AugConfig | None = None,
        size_band: SizeBand | None = None,
        imgsz: int | None = None,
        gt_guided_params: dict | None = None,
        giant_ceiling_frac: float = 0.75,
        whole_image_if_giant: bool = True,
        cache_bytes: int | None = None,
        tile_source: str = "buffered",
        cache_size: int = 8,
        mosaic_transform=None,
        pool_min_windows: int = 1,
        rank: int | None = None,
        world_size: int | None = None,
        run_id: str | None = None,
        assignment_mode: str = "bbox_ioa",
        negative_window_blocklist: str | None = None,
    ) -> None:
        super().__init__()
        self._multi_sample = None
        if aug_config is not None and any(
            p and p > 0 for p in (aug_config.mosaic_p, aug_config.cutmix_p,
                                  aug_config.mixup_p, aug_config.copy_paste_p)):
            if tile_source not in ("pooled", "shared_pooled"):
                raise ValueError(
                    "multi-sample augmentation (mosaic/cutmix/mixup/copy-paste) on the "
                    "streaming path requires tiling.tile_source='pooled' or "
                    "'shared_pooled' (Option C). Disable them or switch tile_source.")
            from training_toolkit.data.multi_sample import make_multi_sample_transforms
            self._multi_sample = make_multi_sample_transforms(aug_config)
        self._aug_config = aug_config
        self._mosaic_transform = mosaic_transform
        self._base = base_dataset
        self._transform = transform
        self._tile_source = tile_source
        self._cache_size = cache_size
        self._pool_min_windows = pool_min_windows
        self._min_annotation_overlap = min_annotation_overlap
        self._assignment_mode = assignment_mode
        self._buffer_size = buffer_size
        self._base_seed = base_seed
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._rank = rank
        self._world_size = world_size
        self._run_id = run_id
        self._size_band = size_band
        self._imgsz = imgsz
        self._tile_sizes = tile_sizes or [tile_size]
        self._regime_tile_sizes = regime_tile_sizes
        self._gt_guided_params = gt_guided_params
        self._giant_ceiling_frac = giant_ceiling_frac
        self._whole_image_if_giant = whole_image_if_giant

        self._cache_bytes = cache_bytes
        dims: dict[int, tuple[int, int]] = {}
        blockset = (load_negative_blocklist(negative_window_blocklist)
                    if negative_window_blocklist else None)
        grouped = build_tile_index(
            base_dataset, self._tile_sizes, overlap, negative_ratio,
            min_annotation_overlap, seed=base_seed,
            regime_tile_sizes=self._regime_tile_sizes,
            imgsz=imgsz, gt_guided_params=self._gt_guided_params,
            giant_ceiling_frac=self._giant_ceiling_frac,
            whole_image_if_giant=whole_image_if_giant,
            dims_out=dims,
            assignment_mode=self._assignment_mode,
            negative_blocklist=blockset,
        )
        # Byte hints for the shared pool's cache_bytes budget: decoded RGB
        # uint8 + the 12-byte shape header. Declared dims (w*h is EXIF-rotation
        # invariant), captured for free during the index sweep.
        self._image_nbytes = {i: w * h * 3 + 12 for i, (w, h) in dims.items()}
        self._images: list[int] = sorted(grouped.keys())
        self._tiles_by_image = grouped
        self._counts: list[int] = [len(grouped[i]) for i in self._images]
        self._total_tiles = sum(self._counts)
        # Warn when pooled cache_size is below the IID parity floor.
        # The descriptor reservoir fills to buffer_size only when the M resident
        # images collectively hold >= buffer_size windows, i.e. M >= buffer_size /
        # MEAN(windows_per_image); below that the reservoir runs shallow and
        # distinct-images-per-batch collapses. (Empirically confirmed on a real
        # multi-source cohort: mean=7.7 -> floor ~532, matching the M=512 near-parity / M=1024
        # exact-parity IID sweep. Median is the wrong statistic for a mixed-regime
        # cohort whose median windows/img is 1.)
        # Only POOLED-ELIGIBLE images (>= pool_min_windows) ever sit in the pool and
        # drive its residency demand; lazy images fill the reservoir without a slot.
        if tile_source == "pooled":
            pooled_counts = [c for c in self._counts if c >= pool_min_windows]
            if pooled_counts:
                mean_wpw = statistics.mean(pooled_counts)
                floor = buffer_size / max(1, mean_wpw)
                if cache_size < floor:
                    logger.warning(
                        "pooled tile_source: cache_size=%d is below the IID parity floor ~%d "
                        "(buffer_size=%d / mean_windows_per_image=%.1f over pooled-eligible images); "
                        "distinct-images-per-batch may collapse — raise tiling.cache_size.",
                        cache_size, int(floor), buffer_size, mean_wpw,
                    )
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_()
        logger.info(
            "StreamingTiledSegmentationDataset: %d images, %d tiles, buffer=%d",
            len(self._images), self._total_tiles, buffer_size,
        )

    def _uses_global_shard(self) -> bool:
        """Return True when __iter__ shards images per (rank, worker) via _global_shard.

        shared_pooled bypasses the per-rank shard (all workers see the full
        image list; the shared control block coordinates admission globally);
        buffered and pooled both use _global_shard to partition images.
        """
        return self._tile_source != "shared_pooled"

    def __len__(self) -> int:
        """Number of samples the DataLoader realizes per epoch.

        With ``drop_last=True`` and ``num_workers > 1`` each worker streams a
        disjoint image-slice and drops its own tail partial batch, so the
        realized count is below ``total_tiles``. Lightning derives
        ``num_training_batches`` from this; if it OVERcounts (e.g. by returning
        the full ``total_tiles``), ``val_check_interval=1.0`` targets a batch the
        epoch never reaches and per-epoch validation silently never fires. We
        therefore return a safe realized count: defaults (batch_size=1,
        num_workers<=1) give the full tile count; the multi-worker case drops up
        to one partial batch per worker.

        shared_pooled: the emit-cap makes each of the ``W`` workers yield
        ``ceil(S/W)`` samples, then the DataLoader drops each worker's partial
        tail batch, so the realized count is ``W * floor(ceil(S/W) / bs) * bs``.
        It needs the SAME per-worker drop-last correction as buffered — omitting
        it overcounts, so the epoch StopIterations before val_check_interval=1.0
        and per-epoch validation never fires under workers>1 (the build-2 DDP
        crash).  workers<=1 reduces to the clean floor-division.
        """
        bs = max(1, self._batch_size)
        _, world = _resolve_rank_world(self._rank, self._world_size)
        workers = max(1, self._num_workers)
        if self._tile_source == "shared_pooled":
            S = bs * (self._total_tiles // max(1, world * bs))   # per-rank sample budget
            per_worker = -(-S // workers)                        # ceil(S / workers)
            return max(1, workers * (per_worker // bs)) * bs
        per_rank = self._total_tiles // max(1, world)
        if workers == 1:
            n_batches = per_rank // bs
        else:
            n_batches = per_rank // bs - workers
        return max(1, n_batches) * bs

    @property
    def class_names(self) -> dict[int, str]:
        return self._base.class_names

    @property
    def num_classes(self) -> int:
        return self._base.num_classes

    def set_epoch(self, epoch: int) -> None:
        self._epoch.fill_(int(epoch))

    def __iter__(self) -> Iterator[dict[str, Any]]:
        epoch = int(self._epoch)
        rank, world = _resolve_rank_world(self._rank, self._world_size)
        wid, W = _worker_id_and_count()

        if self._tile_source == "shared_pooled":
            # Shared-pool mode: all workers see the full image list; the
            # SharedImagePool control block coordinates admission across workers
            # and ranks, so _global_shard is bypassed.  Each worker attaches to
            # the node-level control block independently (create_or_attach is
            # idempotent: first caller creates, others spin until ready).
            from training_toolkit.data.shared_control_block import SharedControlBlock
            from training_toolkit.data.shared_pool_lifecycle import pool_name
            from training_toolkit.data.pooled_tile_source import PooledTileSource

            cb = SharedControlBlock.create_or_attach(
                pool_name(self._run_id, epoch), self._cache_size, self._buffer_size,
            )
            try:
                global_id = rank * W + wid
                work_rng = random.Random(self._base_seed + 7919 * epoch + global_id)
                # Build image_order with a SHARED seed (epoch-only, no worker/rank
                # term) so every worker produces the IDENTICAL plan.  The shared
                # cursor then partitions unique positions — concurrent admission of
                # the same image_id is impossible.  `work_rng` (per-worker) is only
                # used for per-tile shuffling within an image, NOT for image ordering.
                plan_rng = random.Random(self._base_seed + epoch)
                image_order = list(self._images)
                plan_rng.shuffle(image_order)
                logger.info(
                    "stream shared_pooled: rank=%d/%d worker=%d/%d -> %d images (full)",
                    rank, world, wid, W, len(image_order),
                )
                # Per-RANK emit cap = __len__() — the drop-last-corrected count
                # W·floor(ceil(S/W)/bs)·bs, so ceil(__len__/W) is an exact multiple
                # of bs: each worker yields that many FULL batches (no partial
                # dropped) and the rank delivers __len__//bs batches == Lightning's
                # num_training_batches, so epoch-end validation fires.  Thread it
                # into PooledTileSource as max_emit; _iter_shared divides by the
                # worker count ONCE (per_worker_target = ceil(max_emit / W)) and pads
                # to it when the pool drains early (the DDP equal-batch-count fix).
                # Do NOT pre-divide by W here (double-divides → W× too few → deadlock).
                per_rank_cap = self.__len__()
                source = PooledTileSource(
                    self._base, image_order, self._tiles_by_image, work_rng,
                    transform=self._transform, pool_capacity=self._cache_size,
                    buffer_size=self._buffer_size, imgsz=self._imgsz,
                    size_band=self._size_band, min_overlap=self._min_annotation_overlap,
                    multi_sample=self._multi_sample,
                    aug_config=self._aug_config,
                    mosaic_transform=self._mosaic_transform,
                    pool_min_windows=self._pool_min_windows,
                    tile_source="shared_pooled", control=cb, run_id=self._run_id,
                    epoch=epoch,
                    max_emit=per_rank_cap,
                    cache_bytes=self._cache_bytes,
                    image_nbytes=self._image_nbytes,
                    assignment_mode=self._assignment_mode,
                )
                yield from source
            finally:
                cb.detach()
            return

        part_rng = random.Random(self._base_seed + epoch)
        positions = _global_shard(self._counts, rank, world, wid, W, part_rng)
        image_order = [self._images[p] for p in positions]
        global_id = rank * W + wid
        work_rng = random.Random(self._base_seed + 7919 * epoch + global_id)
        work_rng.shuffle(image_order)
        logger.info(
            "stream shard: rank=%d/%d worker=%d/%d -> %d images",
            rank, world, wid, W, len(image_order),
        )

        if self._tile_source == "pooled":
            from training_toolkit.data.pooled_tile_source import PooledTileSource
            source = PooledTileSource(
                self._base, image_order, self._tiles_by_image, work_rng,
                transform=self._transform, pool_capacity=self._cache_size,
                buffer_size=self._buffer_size, imgsz=self._imgsz,
                size_band=self._size_band, min_overlap=self._min_annotation_overlap,
                multi_sample=self._multi_sample, aug_config=self._aug_config,
                mosaic_transform=self._mosaic_transform,
                pool_min_windows=self._pool_min_windows,
                assignment_mode=self._assignment_mode,
            )
            yield from source            # pooled owns its transform application
        else:
            source = BufferedTileSource(
                self._base, image_order, self._tiles_by_image, work_rng,
                buffer_size=self._buffer_size, imgsz=self._imgsz,
                size_band=self._size_band, min_overlap=self._min_annotation_overlap,
                assignment_mode=self._assignment_mode,
            )
            for sample in source:
                if self._transform is not None:
                    sample = self._base._apply_transform(sample, self._transform)
                yield sample
