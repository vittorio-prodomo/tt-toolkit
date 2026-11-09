"""Model configuration."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    """Configuration for a model backend.

    Args:
        name: Model identifier (e.g. "mask2former-swin-base", "rf-detr-base").
        backend: Backend name. Auto-detected from name if None.
        pretrained: Pretrained weights identifier (HF hub ID, path, or "none").
        num_classes: Number of object classes. If None, read from dataset.
        extra: Backend-specific pass-through options.
    """
    name: str = "rf-detr-seg-large"
    backend: str | None = None
    pretrained: str = "auto"
    num_classes: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
