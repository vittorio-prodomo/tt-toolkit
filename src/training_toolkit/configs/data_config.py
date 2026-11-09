"""Data pipeline configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from .tiling_config import TilingConfig

# Auto-detect: cpu_count, capped at 8 to avoid diminishing returns / RAM bloat.
_AUTO_WORKERS = min(os.cpu_count() or 4, 8)


@dataclass
class DataConfig:
    """Configuration for dataset loading."""
    dataset_path: str | Path = "."
    batch_size: int = 8
    workers: int = _AUTO_WORKERS
    prefetch_factor: int = 4
    train_split: str = "train"
    val_split: str = "val"
    layout: str = "coco"
    # Pre-train header-level guard: fail fast if any image carries a rotating EXIF
    # tag or its decoded dims disagree with the declared COCO dims (the cv2/pyvips
    # decoder split that silently drops tiles). Runs on rank 0 before the loaders.
    dim_sanity_check: bool = True
    tiling: TilingConfig = field(default_factory=TilingConfig)

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}")
        if self.workers < 0:
            raise ValueError(f"workers must be >= 0, got {self.workers}")
        if self.prefetch_factor < 1:
            raise ValueError(f"prefetch_factor must be >= 1, got {self.prefetch_factor}")
        if self.layout not in ("coco", "roboflow", "yolo", "fiftyone"):
            raise ValueError(
                f"layout must be 'coco', 'roboflow', 'yolo', or 'fiftyone', "
                f"got {self.layout!r}"
            )
