"""Option C: decoded-image pool + live-descriptor reservoir tile source.

Decode each image once into a bounded pool; buffer lightweight tile descriptors
(cap buffer_size); materialize tiles on demand (numpy crop + tile-local raster).
ShuffleBuffer generalized to whole-image admission.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training_toolkit.data.tile_materialize import decode_image_pyvips
from training_toolkit.data.streaming_tiled_dataset import (
    materialize_labeled_tile,
    materialize_labeled_tile_from_array,
)

Box = tuple[int, int, int, int]


@dataclass
class PoolSlot:
    array: np.ndarray
    instances: list
    class_mask: object
    image_id: int
    pending: int


@dataclass
class Desc:
    """A reservoir entry. Either pooled (``slot`` -> a resident ``ImagePool``
    image, materialized by numpy crop) or lazy (``lazy`` carries everything to
    ROI-decode the tile at emit; no pool residency)."""
    box: Box
    apply_band: bool
    slot: "int | None"          # pooled: pool slot id; lazy: None
    lazy: "tuple | None"         # lazy: (path, instances, class_mask, image_id); pooled: None


class ImagePool:
    """Bounded pool of decoded images. admit() occupies a free slot; decref()
    frees it when its last descriptor is materialized. Capacity = M images."""

    def __init__(self, capacity: int) -> None:
        self._capacity = max(1, int(capacity))
        self._slots: dict[int, PoolSlot] = {}
        self._next_id = 0

    def has_free_slot(self) -> bool:
        return len(self._slots) < self._capacity

    def admit(self, array, instances, class_mask, image_id, pending) -> int:
        if not self.has_free_slot():
            raise RuntimeError("ImagePool.admit called with no free slot")
        if pending <= 0:
            raise ValueError("ImagePool.admit requires pending >= 1")
        slot = self._next_id
        self._next_id += 1
        self._slots[slot] = PoolSlot(array, instances, class_mask, image_id, pending)
        return slot

    def get(self, slot: int) -> PoolSlot:
        return self._slots[slot]

    def decref(self, slot: int) -> None:
        if slot not in self._slots:
            raise RuntimeError(f"ImagePool.decref underflow: slot {slot} is not live")
        s = self._slots[slot]
        s.pending -= 1
        if s.pending == 0:
            del self._slots[slot]

    def __len__(self) -> int:
        return len(self._slots)


class PooledTileSource:
    """Decoded-image pool (cap M) + live tile-descriptor reservoir (cap
    buffer_size). Generalizes the tile ShuffleBuffer to whole-image admission:
    decode each image once, draw tiles on demand, rasterize tile-local at emit.
    Yields pre-/post-transform samples in random (reservoir-pop) order;
    coverage-complete per epoch (every window emitted exactly once).

    tile_source choices:
      "buffered"      — the original ShuffleBuffer path (not handled here).
      "pooled"        — in-process decoded-image pool (default).
      "shared_pooled" — cross-process shared pool via SharedImagePool.
    """

    def __init__(self, base, image_order, tiles_by_image, work_rng, *,
                 transform=None, pool_capacity=1, buffer_size=256,
                 imgsz=None, size_band=None, min_overlap=0.05,
                 multi_sample=None, aug_config=None, mosaic_transform=None,
                 pool_min_windows=1,
                 tile_source="pooled", control=None, run_id=None, epoch=0,
                 max_emit: int | None = None,
                 cache_bytes: int | None = None,
                 image_nbytes: dict | None = None,
                 assignment_mode: str = "bbox_ioa"):
        self._base = base
        self._image_order = image_order
        self._tiles_by_image = tiles_by_image
        self._rng = work_rng
        self._transform = transform
        self._pool_capacity = pool_capacity
        self._buffer_size = buffer_size
        self._imgsz = imgsz
        self._size_band = size_band
        self._min_overlap = min_overlap
        self._assignment_mode = assignment_mode
        self._multi_sample = multi_sample
        self._aug_config = aug_config
        self._mosaic_transform = mosaic_transform
        self._pool_min_windows = pool_min_windows
        self._tile_source = tile_source
        self._cache_bytes = cache_bytes
        self._image_nbytes = image_nbytes or {}
        self._control = control
        self._run_id = run_id
        self._epoch = epoch
        self._max_emit = max_emit  # PER-RANK emit cap (shared_pooled DDP safety); /W once in _iter_shared
        self._donor_lazy_fallbacks = 0  # safe default so _borrow_donors_shared works standalone in tests

    def _realized_scale(self, box: Box) -> float:
        return (self._imgsz / max(box[2], box[3])) if self._imgsz else 1.0

    def _materialize(self, desc, pool):
        band = self._size_band if desc.apply_band else None
        rs = self._realized_scale(desc.box)
        if desc.slot is not None:                       # pooled: numpy crop from the resident array
            s = pool.get(desc.slot)
            return materialize_labeled_tile_from_array(
                s.array, desc.box, s.instances, s.class_mask,
                realized_scale=rs, band=band,
                min_overlap=self._min_overlap, image_id=s.image_id,
                assignment_mode=self._assignment_mode,
            )
        path, instances, class_mask, image_id = desc.lazy   # lazy: pyvips ROI decode at emit
        return materialize_labeled_tile(
            path, desc.box, instances, class_mask,
            realized_scale=rs, band=band,
            min_overlap=self._min_overlap, image_id=image_id,
            assignment_mode=self._assignment_mode,
        )

    def _borrow_donors(self, pool, reservoir, n):
        """Peek n distinct reservoir positions WITHOUT consuming (they stay to be
        emitted as their own primaries later — matches offline borrow semantics);
        materialize each from the resident pool."""
        if not reservoir:
            return []
        idxs = self._rng.sample(range(len(reservoir)), min(n, len(reservoir)))
        return [self._materialize(reservoir[j], pool) for j in idxs]

    def _maybe_multi_sample(self, sample, borrow_fn):
        """Apply one mutually-exclusive multi-sample aug (mosaic/cutmix/mixup) +
        optional copy-paste. Donors come from ``borrow_fn(n) -> list[sample]``
        (may return fewer than n) — the in-process caller peeks its local
        reservoir, the shared caller peeks the SHARED reservoir. One ladder,
        two donor sources: keeps the two paths from drifting.
        Returns (sample, used_mosaic)."""
        ms, cfg = self._multi_sample, self._aug_config
        if ms is None or cfg is None:
            return sample, False
        r = self._rng.random()
        used_mosaic = False
        if ms.mosaic is not None and r < cfg.mosaic_p:
            extras = borrow_fn(3)
            if len(extras) == 3:
                sample = ms.mosaic(sample, extras); used_mosaic = True
        elif ms.cutmix is not None and r < cfg.mosaic_p + cfg.cutmix_p:
            d = borrow_fn(1)
            if d: sample = ms.cutmix(sample, d[0])
        elif ms.mixup is not None and r < cfg.mosaic_p + cfg.cutmix_p + cfg.mixup_p:
            d = borrow_fn(1)
            if d: sample = ms.mixup(sample, d[0])
        if ms.copy_paste is not None and self._rng.random() < cfg.copy_paste_p:
            d = borrow_fn(1)
            if d: sample = ms.copy_paste(sample, d[0])
        return sample, used_mosaic

    # ------------------------------------------------------------------
    # Shared-pool helpers (tile_source == "shared_pooled")
    # ------------------------------------------------------------------

    def _decode_full(self, image_id: int) -> np.ndarray:
        """Whole-image decode for the shared pool admit path (pyvips sequential)."""
        return decode_image_pyvips(self._base._image_path(image_id))

    def _materialize_shared(self, image_id: int, box: Box, px, apply_band: bool = True) -> dict:
        """Materialize one tile from the shared pool.

        px is the numpy view returned by SharedImagePool.read_pixels():
          - not None  → pooled: crop the tile from the resident shm array.
                        IMPORTANT: the crop must be an OWNED COPY, not a view into
                        shm, because decref() may free+unlink the segment immediately
                        after this call. The explicit .copy() below is what detaches
                        from the shm mmap — np.ascontiguousarray alone is NOT enough.
          - None      → lazy: ROI-decode via the validated materialize_labeled_tile path.

        Annotations are fetched from self._base, identical to _materialize.
        """
        instances = self._base._load_annotations(image_id)
        class_mask = self._base._image_class_mask(image_id)
        band = self._size_band if apply_band else None   # parity with _materialize (in-process)
        rs = self._realized_scale(box)
        if px is not None:
            # Crop + force an OWNED copy so the sample never aliases the shm buffer.
            # The explicit .copy() is the load-bearing detach: np.ascontiguousarray
            # on an already-contiguous slice returns the SAME underlying buffer
            # (base is not None). Only after .copy() does the array own its memory
            # (img.base is None), so decref() can close+unlink the segment without a
            # use-after-free.
            crop = px[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            img = np.ascontiguousarray(crop, dtype=np.uint8).copy()
            assert img.base is None, (
                "BUG: _materialize_shared produced a view — tile image aliases shm buffer")
            return materialize_labeled_tile_from_array(
                img, box, instances, class_mask,
                realized_scale=rs, band=band,
                min_overlap=self._min_overlap, image_id=image_id,
                assignment_mode=self._assignment_mode,
            )
        path = self._base._image_path(image_id)
        return materialize_labeled_tile(
            path, box, instances, class_mask,
            realized_scale=rs, band=band,
            min_overlap=self._min_overlap, image_id=image_id,
            assignment_mode=self._assignment_mode,
        )

    def _borrow_donors_shared(self, pool, n):
        """Peek n donors from the SHARED reservoir (read-only; donors stay in
        the reservoir and later emit as their own primaries — coverage
        exactly-once preserved). Donors never decref (they hold no descriptor
        unit) and never yield (DDP emit-cap math untouched). A donor whose
        segment was freed mid-borrow resolves through read_pixels' None ->
        lazy disk-ROI fallback; the counter makes that churn visible."""
        out = []
        for image_id, w in pool.peek(self._rng, n):
            box, apply_band = self._tiles_by_image[image_id][w]
            px = pool.read_pixels(image_id)
            if px is None:
                self._donor_lazy_fallbacks += 1
            out.append(self._materialize_shared(image_id, box, px, apply_band))
        return out

    def _emit_one(self, pool):
        """Pop one descriptor from the shared pool, materialize + yield it."""
        result = pool.pop(self._rng)
        if result is None:
            return
        image_id, w = result
        box, apply_band = self._tiles_by_image[image_id][w]
        px = pool.read_pixels(image_id)   # None => lazy; view into shm => pooled
        sample = self._materialize_shared(image_id, box, px, apply_band)
        # decref AFTER materialization so the shm segment is not freed while we
        # still hold a view into it (px is no longer referenced past this line).
        pool.decref(image_id)
        sample, used_mosaic = self._maybe_multi_sample(
            sample, lambda n: self._borrow_donors_shared(pool, n))
        transform = (self._mosaic_transform
                     if used_mosaic and self._mosaic_transform is not None
                     else self._transform)
        if transform is not None:
            sample = self._base._apply_transform(sample, transform)
        yield sample

    def _iter_shared(self):
        """Drain the shared pool: same admit/emit interleave as the in-process
        _iter, with reservoir/pool ops routed to SharedImagePool.

        The 1-image `pending` lookahead mirrors the in-process loop to avoid
        slot-starvation deadlock: a pooled `pending` is never dropped — it waits
        while another worker's emit frees a slot.

        Cross-process slot stealing
        ---------------------------
        `has_free_slot()` and `try_admit()` are SEPARATE lock acquisitions, so a
        slot seen free here can be claimed by another worker before our admit
        runs — `try_admit` then returns False (it pushes NO descriptors).  We
        MUST honor that return: keep `pending` set and retry, never clear it on a
        failed admit (clearing it = the silent-drop coverage bug).  When the
        reservoir is also empty and no slot is free, we spin (a brief sleep) until
        another worker's decref frees a slot — termination is guaranteed because
        every occupied slot belongs to an admitted image whose descriptors are
        in-flight (reservoir or mid-materialize) and will be decref'd.

        DDP emit-cap + padding
        ----------------------
        ``max_emit`` is the PER-RANK emit cap (the dataset passes ``__len__()``,
        the exact per-rank tile count the DataLoader expects).  Each worker must
        yield EXACTLY ``per_worker_target`` samples so the DataLoader always
        delivers the same number of batches on every rank.  The shared cursor
        distributes the epoch's tiles unevenly — one worker may drain its share
        before others finish; if it yields fewer samples its rank's DataLoader
        issues fewer batches and the next NCCL collective on the other rank
        deadlocks.

        Per-worker target: ``ceil(max_emit / num_workers)`` — ``max_emit`` is
        divided by the worker count exactly ONCE here, so the rank's
        ``num_workers`` workers together produce ``W·ceil(max_emit/W) >= max_emit``
        samples.  (The dataset must NOT pre-divide by W; doing so double-divides
        and yields W× too few → the deadlock this guard prevents.)  After the
        shared pool drains, pad by re-yielding the last emitted sample until the
        count reaches the target.  Padding duplicates are negligible for
        convergence and are the standard DDP-uneven trick.

        Degenerate case: if a worker gets zero real samples (whole shard empty —
        not the normal path), it cannot pad because it has no template.  A warning
        is logged and the worker yields nothing; the DataLoader's collate tolerates
        a zero-sample worker (it simply gets no samples from it, reducing the
        effective batch for those steps).  This can cause a batch-count mismatch
        if it happens on only some ranks; avoid by ensuring total_tiles >> world *
        num_workers.
        """
        import logging
        import math
        import socket
        import time
        from training_toolkit.data.shared_image_pool import SharedImagePool

        logger = logging.getLogger(__name__)
        self._donor_lazy_fallbacks = 0
        pool = SharedImagePool(self._control, run_id=self._run_id,
                               host=socket.gethostname(), epoch=self._epoch,
                               cache_bytes=self._cache_bytes)
        # Build the per-image admission plan (identical on every worker — the
        # shared cursor decides which worker admits which image).
        plan = [
            (iid, len(self._tiles_by_image[iid]),
             len(self._tiles_by_image[iid]) >= self._pool_min_windows)
            for iid in self._image_order
        ]

        # Per-worker emit target (DDP safety).  Only active when max_emit is set.
        # num_workers comes from get_worker_info() — None outside a DataLoader
        # (single-process / test context) means W=1.
        if self._max_emit is not None:
            from torch.utils.data import get_worker_info as _gwi
            _wi = _gwi()
            _W = 1 if _wi is None else _wi.num_workers
            per_worker_target = math.ceil(self._max_emit / max(1, _W))
        else:
            per_worker_target = None  # uncapped — original behaviour

        pending = None   # 1-image lookahead: a claimed-but-not-yet-admitted image
        emitted = 0
        pad_template = None  # last real sample; used for padding after pool drains

        while True:
            # --- stop early if we have already hit the per-worker target ---
            if per_worker_target is not None and emitted >= per_worker_target:
                break

            # --- admit phase: fill the reservoir toward buffer_size ---
            if pending is None and pool.reservoir_len() < self._buffer_size:
                pending = pool.claim_next(plan)   # atomic: advances shared cursor

            if pending is not None:
                image_id, n_windows, is_pooled = pending
                if not is_pooled or pool.has_free_slot():
                    admitted = pool.try_admit(
                        image_id, n_windows, is_pooled,
                        decode_fn=self._decode_full,
                        nbytes_hint=self._image_nbytes.get(image_id, 0))
                    if admitted:
                        pending = None        # only clear on a SUCCESSFUL admit
                        continue              # retry the admit phase (may admit more)
                    # try_admit returned False: another worker stole the slot
                    # between our has_free_slot() check and the admit.  Keep
                    # `pending`; fall through to emit (frees a slot) and retry.

            # --- emit phase: pop + materialize one descriptor ---
            if pool.reservoir_len() > 0:
                for sample in self._emit_one(pool):
                    pad_template = sample
                    yield sample
                    emitted += 1
                    # Re-check target after each real emit so we stop promptly.
                    if per_worker_target is not None and emitted >= per_worker_target:
                        return
            elif pending is None and pool.done():
                break   # nothing to admit, nothing to emit — epoch done (real tiles)
            elif pending is not None:
                # A pooled `pending` with an empty reservoir and no free slot:
                # every slot is held by another worker's in-flight image, which
                # will decref and free one.  Spin briefly rather than dropping
                # `pending` (the old `break` here silently dropped the image in
                # the multi-process case).
                time.sleep(0.001)

        if self._donor_lazy_fallbacks:
            logger.debug("_iter_shared: %d donor(s) fell back to lazy disk decode "
                         "(segment freed mid-borrow)", self._donor_lazy_fallbacks)

        # --- padding phase (DDP safety): fill up to per_worker_target ---
        if per_worker_target is not None and emitted < per_worker_target:
            if pad_template is None:
                logger.warning(
                    "_iter_shared: worker got zero real samples but max_emit=%d — "
                    "cannot pad (degenerate empty shard). This may cause a DDP "
                    "batch-count mismatch. Increase total_tiles or reduce world*workers.",
                    self._max_emit,
                )
                return
            n_pad = per_worker_target - emitted
            logger.debug("_iter_shared: padding %d samples to reach per_worker_target=%d",
                         n_pad, per_worker_target)
            for _ in range(n_pad):
                yield pad_template
                emitted += 1

    # ------------------------------------------------------------------
    # Main iterator (in-process pooled path, unchanged)
    # ------------------------------------------------------------------

    def __iter__(self):
        if self._tile_source == "shared_pooled":
            yield from self._iter_shared()
            return

        pool = ImagePool(self._pool_capacity)
        reservoir: list[Desc] = []
        images = iter(self._image_order)
        pending = None            # peeked img awaiting a free pool slot (pooled only)
        exhausted = False

        def admit():
            nonlocal exhausted, pending
            # Fill toward buffer_size. Pooled images (>= pool_min_windows) need a
            # free pool slot, else defer via `pending`. Lazy images never touch the
            # pool, so they keep filling the reservoir even when the pool is full ->
            # reservoir depth (IID) is decoupled from pool residency (RAM).
            while len(reservoir) < self._buffer_size and not exhausted:
                if pending is None:
                    try:
                        pending = next(images)
                    except StopIteration:
                        exhausted = True
                        break
                img_idx = pending
                windows = self._tiles_by_image.get(img_idx, [])
                if not windows:
                    pending = None
                    continue                              # no-window image strands no slot
                if len(windows) >= self._pool_min_windows:
                    if not pool.has_free_slot():
                        break                             # keep `pending`; an emit will free a slot
                    array = decode_image_pyvips(self._base._image_path(img_idx))
                    slot = pool.admit(
                        array, self._base._load_annotations(img_idx),
                        self._base._image_class_mask(img_idx), img_idx, len(windows),
                    )
                    for (box, apply_band) in windows:
                        reservoir.append(Desc(box, apply_band, slot, None))
                else:                                     # lazy: ROI-decode at emit, no residency
                    lazy = (self._base._image_path(img_idx),
                            self._base._load_annotations(img_idx),
                            self._base._image_class_mask(img_idx), img_idx)
                    for (box, apply_band) in windows:
                        reservoir.append(Desc(box, apply_band, None, lazy))
                pending = None

        while True:
            admit()
            if not reservoir:
                break                                     # nothing live + nothing to admit -> epoch done
            desc = reservoir.pop(self._rng.randrange(len(reservoir)))
            sample = self._materialize(desc, pool)
            if desc.slot is not None:                     # only pooled descriptors hold a slot
                pool.decref(desc.slot)
            sample, used_mosaic = self._maybe_multi_sample(
                sample, lambda n: self._borrow_donors(pool, reservoir, n))
            transform = (self._mosaic_transform
                         if used_mosaic and self._mosaic_transform is not None
                         else self._transform)
            if transform is not None:
                sample = self._base._apply_transform(sample, transform)
            yield sample
