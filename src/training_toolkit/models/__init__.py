"""Model adapter registry and factory."""

import importlib
import logging
import re
from typing import Type

from .base import ModelAdapter
from training_toolkit.configs import ModelConfig

logger = logging.getLogger(__name__)

ADAPTER_REGISTRY: dict[str, str] = {
    "rfdetr": "training_toolkit.models.rfdetr.RFDETRAdapter",
    "mask2former": "training_toolkit.models.mask2former.Mask2FormerAdapter",
    "yolo": "training_toolkit.models.yolo.YOLOAdapter",
}

_BACKEND_PATTERNS = [
    (re.compile(r"rf[_-]?detr", re.IGNORECASE), "rfdetr"),
    (re.compile(r"mask2former", re.IGNORECASE), "mask2former"),
    (re.compile(r"yolo", re.IGNORECASE), "yolo"),
]


def detect_backend(model_name: str) -> str:
    for pattern, backend in _BACKEND_PATTERNS:
        if pattern.search(model_name):
            return backend
    raise ValueError(
        f"Cannot auto-detect backend for model {model_name!r}. "
        f"Set config.model.backend explicitly. Known: {list(ADAPTER_REGISTRY)}"
    )


def _import_class(dotted_path: str) -> Type[ModelAdapter]:
    module_path, class_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def get_adapter(config: ModelConfig) -> ModelAdapter:
    backend = config.backend or detect_backend(config.name)
    logger.info("Using model backend: %s", backend)
    if backend not in ADAPTER_REGISTRY:
        raise KeyError(f"Unknown backend {backend!r}. Registered: {list(ADAPTER_REGISTRY)}")
    cls = _import_class(ADAPTER_REGISTRY[backend])
    return cls()


__all__ = ["ModelAdapter", "get_adapter", "detect_backend", "ADAPTER_REGISTRY"]
