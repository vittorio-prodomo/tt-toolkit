"""Tiling configuration for high-resolution image datasets."""

from dataclasses import dataclass


@dataclass
class TilingConfig:
    """Configuration for deterministic on-the-fly tiling.

    When ``enabled=True``, the dataset is wrapped in a
    :class:`TiledSegmentationDataset` that pre-computes a tile grid
    and returns crops instead of full images.
    """

    enabled: bool = False
    tile_size: int = 416
    overlap: float = 0.2
    negative_ratio: float = 0.15
    min_annotation_overlap: float = 0.05
    cache_size: int = 8

    # --- streaming (online) path ---
    online: bool = False
    tile_source: str = "buffered"   # "buffered" (default) | "pooled" (decoded-image pool, Option C) | "shared_pooled" (node-global POSIX shm pool, RAM = 1×M)
    cache_bytes: int | None = None  # shared_pooled: opt-in resident-BYTES budget for the shm pool (image-count cache_size still applies). Size it to <= ~40% of /dev/shm: lag-by-one epoch cleanup lets ~2 epoch pools coexist at boundaries, plus torch IPC headroom. None = off (byte-identical).
    pool_min_windows: int = 1   # pooled hybrid: images with < this many windows skip the pool (lazy ROI). 1 = pool all.
    buffer_size: int = 4096  # per-worker shuffle-buffer depth (IID dial). 4096 ≈ global-shuffle
    # convergence parity (2026-06-09 study); 256 measurably lagged. ~26 GB RAM @ 8 workers.
    base_seed: int = 42     # seed for per-epoch image/tile shuffling
    tile_sizes: list[int] | None = None  # outdoor dual-scale, e.g. [416, 1024]; None -> [tile_size]
    regime_tile_sizes: dict[str, list[int]] | None = None  # per-regime override; None -> tile_sizes

    # Keep in sync with _DEFAULT_GT_GUIDED (data/tile_index.py).
    gt_guided_target_in_model: float = 110.0
    gt_guided_min_crop: int = 416
    gt_guided_translation_jitter: float = 0.3
    gt_guided_scale_jitter: float = 0.15
    gt_guided_dedup_iou: float = 0.6

    giant_ceiling_frac: float = 0.75  # whole-image giant test = giant_ceiling_frac * max(tile_sizes)
    whole_image_if_giant: bool = True  # grid regimes: append a downscaled full-frame window when a defect's raw sqrt-area exceeds giant_ceiling_frac * max(tile_sizes). Off = truncated positives only (e.g. single-scale line-like classes).

    assignment_mode: str = "bbox_ioa"  # "mask_true" = pixel-true window/instance assignment (dataset-v2); default byte-identical

    negative_window_blocklist: str | None = None  # JSON file: [[image_path, x, y, w, h], ...] windows to exclude from train negatives. None = no filtering.

    def __post_init__(self):
        if self.tile_size <= 0:
            raise ValueError(f"tile_size must be > 0, got {self.tile_size}")
        if not 0 <= self.overlap < 1:
            raise ValueError(f"overlap must be in [0, 1), got {self.overlap}")
        if not 0 <= self.negative_ratio < 1:
            raise ValueError(
                f"negative_ratio must be in [0, 1), got {self.negative_ratio}"
            )
        if not 0 < self.min_annotation_overlap <= 1:
            raise ValueError(
                f"min_annotation_overlap must be in (0, 1], "
                f"got {self.min_annotation_overlap}"
            )
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be >= 1, got {self.buffer_size}")
        if self.tile_source not in ("buffered", "pooled", "shared_pooled"):
            raise ValueError(
                f"tile_source must be 'buffered', 'pooled', or 'shared_pooled', got {self.tile_source!r}")
        if self.cache_bytes is not None and self.cache_bytes <= 0:
            raise ValueError(
                f"cache_bytes must be None or > 0, got {self.cache_bytes}"
            )
        if self.pool_min_windows < 1:
            raise ValueError(
                f"pool_min_windows must be >= 1, got {self.pool_min_windows}")
        if self.tile_sizes is not None and len(self.tile_sizes) == 0:
            raise ValueError("tile_sizes must be None or a non-empty list")
        if self.regime_tile_sizes is not None:
            for regime, sizes in self.regime_tile_sizes.items():
                if not sizes:
                    raise ValueError(f"regime_tile_sizes[{regime!r}] must be a non-empty list")
        if not 0 < self.giant_ceiling_frac <= 1:
            raise ValueError(
                f"giant_ceiling_frac must be in (0, 1], got {self.giant_ceiling_frac}"
            )
        if self.assignment_mode not in ("bbox_ioa", "mask_true"):
            raise ValueError(
                f"assignment_mode must be 'bbox_ioa' or 'mask_true', "
                f"got {self.assignment_mode!r}")
