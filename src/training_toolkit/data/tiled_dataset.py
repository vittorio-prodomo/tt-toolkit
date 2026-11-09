"""Tiled wrapper for instance segmentation datasets.

Provides deterministic, on-the-fly tiling of high-resolution images into
fixed-size crops, without duplicating data on disk.  Each tile is a virtual
sample in the PyTorch sense: :meth:`__len__` returns the total number of
tiles across all images, and :meth:`__getitem__` lazily loads and crops.

The tile index is pre-computed at init time using only coordinate math on
the label files — no images are loaded until training begins.

Usage::

    base = YoloSegmentationDataset(root, split="train")
    tiled = TiledSegmentationDataset(
        base,
        tile_size=416,
        overlap=0.2,
        negative_ratio=0.15,
    )
    # Use tiled as a drop-in replacement for base in a DataLoader.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Callable

import numpy as np

from training_toolkit.configs import AugConfig
from training_toolkit.data.base_dataset import BaseSegmentationDataset
from training_toolkit.data.tile_index import (
    compute_tile_grid as _compute_tile_grid,
    tile_has_annotation as _tile_has_annotation,
)
from training_toolkit.data.tile_materialize import materialize_tile

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Image cache
# ----------------------------------------------------------------------

class _ImageCache:
    """LRU cache of per-image entries, keyed by image index.

    Holds the data reused across every tile of one image — annotation geometry
    (Instances, no pixel decode) plus the pyvips-decoded (width, height). The
    per-tile pixel cost is the pyvips ROI crop, not a full-image decode, so a
    cache miss is cheap (annotation parse + a header read).
    """

    def __init__(self, max_size: int = 8):
        self._max_size = max_size
        self._cache: dict[int, dict[str, Any]] = {}
        self._order: list[int] = []

    def get(self, image_idx: int, load_fn) -> dict[str, Any]:
        if image_idx in self._cache:
            return self._cache[image_idx]

        image = load_fn()
        if len(self._cache) >= self._max_size:
            evict = self._order.pop(0)
            del self._cache[evict]

        self._cache[image_idx] = image
        self._order.append(image_idx)
        return image


# ----------------------------------------------------------------------
# TiledSegmentationDataset
# ----------------------------------------------------------------------

class TiledSegmentationDataset(BaseSegmentationDataset):
    """Wraps an existing segmentation dataset with deterministic tiling.

    At init time, computes a tile grid for each image and classifies
    tiles as positive (contains annotations) or negative.  Negative tiles
    are subsampled to a configurable ratio.  At getitem time, the image
    is loaded (with LRU caching) and cropped to the tile region.  Masks
    and boxes are clipped to tile-local coordinates.

    Parameters
    ----------
    base_dataset : BaseSegmentationDataset
        The underlying full-image dataset (YOLO or COCO).
    tile_size : int
        Tile side length in pixels.
    overlap : float
        Fractional overlap between adjacent tiles (0–1).
    negative_ratio : float
        Target fraction of the final dataset that should be negative
        tiles (no annotations).  0.0 = no negatives, 0.5 = half negatives.
    min_annotation_overlap : float
        Minimum fraction of an annotation's area that must fall inside
        a tile for the tile to count as positive.
    cache_size : int
        Number of decoded images to keep in the LRU cache per worker.
    transform : callable, optional
        Albumentations transform (applied after cropping).
    aug_config : AugConfig, optional
        Multi-sample augmentation config.
    seed : int, optional
        Seed for negative tile subsampling (for reproducibility).
    """

    def __init__(
        self,
        base_dataset: BaseSegmentationDataset,
        tile_size: int = 416,
        overlap: float = 0.2,
        negative_ratio: float = 0.15,
        min_annotation_overlap: float = 0.05,
        cache_size: int = 8,
        transform: Callable | None = None,
        aug_config: AugConfig | None = None,
        mosaic_transform: Callable | None = None,
        seed: int | None = 42,
        assignment_mode: str = "bbox_ioa",
    ) -> None:
        super().__init__(
            transform=transform, aug_config=aug_config,
            mosaic_transform=mosaic_transform,
        )
        self._base = base_dataset
        self._tile_size = tile_size
        self._overlap = overlap
        self._negative_ratio = negative_ratio
        self._min_annotation_overlap = min_annotation_overlap
        self._assignment_mode = assignment_mode
        self._cache = _ImageCache(max_size=cache_size)
        # Declared (metadata) dims per image, cached on first crop; the grid was
        # built from these, so a decoded image that disagrees is untrustworthy.
        self._declared_dims: dict[int, tuple[int, int]] = {}
        self._dim_mismatch_warned: set[int] = set()

        # Build the tile index
        self._tiles = self._build_tile_index(seed)

    # ------------------------------------------------------------------
    # Abstract interface (delegated to base dataset)
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        return self._base.class_names

    @property
    def num_classes(self) -> int:
        return self._base.num_classes

    def __len__(self) -> int:
        return len(self._tiles)

    # ------------------------------------------------------------------
    # Tile index
    # ------------------------------------------------------------------

    def _build_tile_index(
        self, seed: int | None,
    ) -> list[tuple[int, int, int, int, int, bool]]:
        """Pre-compute all tiles across all images.

        Returns a list of (image_idx, x, y, w, h, is_positive) tuples.
        Only loads label files (coordinate math), not images.
        """
        logger.info(
            "Building tile index: tile_size=%d, overlap=%.2f, "
            "negative_ratio=%.2f, min_annotation_overlap=%.2f",
            self._tile_size, self._overlap,
            self._negative_ratio, self._min_annotation_overlap,
        )

        positives: list[tuple[int, int, int, int, int, bool]] = []
        negatives: list[tuple[int, int, int, int, int, bool]] = []

        n_images = len(self._base)
        try:
            from tqdm import tqdm
            iterator = tqdm(range(n_images), desc="Building tile index", unit="img")
        except ImportError:
            iterator = range(n_images)

        for img_idx in iterator:
            # Lightweight metadata: reads image header + label file only,
            # no pixel decoding.
            img_w, img_h, boxes = self._base._load_tile_metadata(img_idx)

            grid = _compute_tile_grid(img_w, img_h, self._tile_size, self._overlap)

            if self._assignment_mode == "mask_true":
                from training_toolkit.data.tile_index import (
                    mask_true_positive_windows,
                )
                windows = [((tx, ty, tw, th), True) for (tx, ty, tw, th) in grid]
                pos_wi = mask_true_positive_windows(
                    windows, self._base._load_annotations(img_idx))
                for wi, (tx, ty, tw, th) in enumerate(grid):
                    entry = (img_idx, tx, ty, tw, th, wi in pos_wi)
                    (positives if wi in pos_wi else negatives).append(entry)
            else:
                for tx, ty, tw, th in grid:
                    is_pos = _tile_has_annotation(
                        tx, ty, tw, th, boxes, self._min_annotation_overlap)
                    entry = (img_idx, tx, ty, tw, th, is_pos)
                    (positives if is_pos else negatives).append(entry)

        # Subsample negatives to target ratio
        n_pos = len(positives)
        if self._negative_ratio <= 0 or n_pos == 0:
            n_neg_target = 0
        else:
            n_neg_target = int(
                (self._negative_ratio / (1 - self._negative_ratio)) * n_pos
            )

        if n_neg_target > len(negatives):
            logger.warning(
                "Requested negative_ratio=%.2f needs %d negatives but only "
                "%d available — using all",
                self._negative_ratio, n_neg_target, len(negatives),
            )
            n_neg_target = len(negatives)

        rng = random.Random(seed)
        sampled_negatives = rng.sample(negatives, n_neg_target) if n_neg_target > 0 else []

        tiles = positives + sampled_negatives
        # Shuffle so positives and negatives are interleaved
        rng.shuffle(tiles)

        actual_ratio = len(sampled_negatives) / len(tiles) if tiles else 0
        logger.info(
            "Tile index: %d positive + %d negative = %d total "
            "(%.1f%% negative, from %d available negatives)",
            n_pos, len(sampled_negatives), len(tiles),
            actual_ratio * 100, len(negatives),
        )

        return tiles

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def _load_image_entry(self, img_idx: int, path: str) -> dict[str, Any]:
        """Per-image cached payload: annotation geometry (Instances, no pixel
        decode) + the pyvips-decoded (width, height). pyvips ``new_from_file`` is a
        lazy header read here; the pixel decode happens per tile in
        ``materialize_tile``."""
        import pyvips
        img = pyvips.Image.new_from_file(path)
        return {
            "instances": self._base._load_annotations(img_idx),
            "wh": (img.width, img.height),
        }

    def _load_sample(self, idx: int) -> dict[str, Any]:
        img_idx, tx, ty, tw, th, _ = self._tiles[idx]
        path = self._base._image_path(img_idx)

        entry = self._cache.get(
            img_idx, lambda: self._load_image_entry(img_idx, path)
        )

        # Guard: the tile grid and annotations live in the DECLARED (metadata)
        # coordinate space. pyvips decodes raw pixels (no EXIF auto-rotation), so a
        # normal image — or an EXIF-rotated one whose COCO width/height were
        # recorded raw — decodes to dims that MATCH declared and this never fires
        # (val now agrees with the pyvips train decoder). It only catches
        # genuinely-corrupt declared dims, where the annotation-space masks would
        # misalign with the raw pixels: neutralize that image's tiles as benign
        # empty negatives (warn once).
        declared = self._declared_dims.get(img_idx)
        if declared is None:
            mw, mh, _ = self._base._load_tile_metadata(img_idx)
            declared = self._declared_dims[img_idx] = (mw, mh)
        if entry["wh"] != declared:
            if img_idx not in self._dim_mismatch_warned:
                self._dim_mismatch_warned.add(img_idx)
                logger.warning(
                    "Tile image idx=%d decoded %dx%d but declared %dx%d "
                    "(metadata mismatch) — dropping its tiles as empty negatives",
                    img_idx, entry["wh"][0], entry["wh"][1], declared[0], declared[1],
                )
            return {
                "image": np.zeros((max(th, 1), max(tw, 1), 3), dtype=np.uint8),
                "masks": [],
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.array([], dtype=np.int64),
                "image_id": idx,
            }

        return materialize_tile(
            path, (tx, ty, tw, th), entry["instances"],
            min_overlap=self._min_annotation_overlap, image_id=idx,
            assignment_mode=self._assignment_mode,
        )
