"""Tests for ModelAdapter ABC and adapter registry."""

import pytest
import torch
from torch import nn

from training_toolkit.models.base import ModelAdapter
from training_toolkit.models import detect_backend, ADAPTER_REGISTRY


class DummyAdapter(ModelAdapter):
    def create_model(self, config): return nn.Linear(10, 10)
    def create_criterion(self, config): return None
    def format_batch(self, batch): return {"inputs": {}, "targets": batch.get("targets", [])}
    def extract_predictions(self, outputs): return []
    def compute_loss(self, outputs, targets): return {"loss_total": torch.tensor(0.0)}


class TestModelAdapterABC:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ModelAdapter()

    def test_concrete_adapter_instantiates(self):
        adapter = DummyAdapter()
        assert adapter is not None

    def test_optional_hooks_have_defaults(self):
        adapter = DummyAdapter()
        assert adapter.configure_optimizers(nn.Linear(1, 1), None) is None
        loss = torch.tensor(1.0)
        assert adapter.on_before_backward(loss) is loss
        assert adapter.postprocess_for_inference({}) == []


class TestAdapterRegistry:
    def test_registry_has_known_backends(self):
        assert "rfdetr" in ADAPTER_REGISTRY
        assert "mask2former" in ADAPTER_REGISTRY

    def test_detect_backend_rfdetr(self):
        assert detect_backend("rf-detr-base") == "rfdetr"
        assert detect_backend("rfdetr-large") == "rfdetr"
        assert detect_backend("rf-detr-nano") == "rfdetr"
        assert detect_backend("rf-detr-small") == "rfdetr"
        assert detect_backend("rf-detr-medium") == "rfdetr"

    def test_detect_backend_mask2former(self):
        assert detect_backend("mask2former-swin-base") == "mask2former"

    def test_detect_backend_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            detect_backend("totally-unknown-model")
