"""Training engine: Lightning module and trainer wrapper."""

from .lit_module import SegmentationLitModule
from .trainer import train

__all__ = ["SegmentationLitModule", "train"]
