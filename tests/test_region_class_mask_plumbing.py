"""Focused plumbing tests for RF-DETR region-level class masking."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from training_toolkit.configs import ModelConfig, TrainConfig
from training_toolkit.data.collate import collate_fn
from training_toolkit.engine.lit_module import SegmentationLitModule
from training_toolkit.models.base import ModelAdapter
from training_toolkit.models.rfdetr import RFDETRAdapter


def _sample(**metadata: np.ndarray) -> dict:
    return {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "masks": [],
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        **metadata,
    }


class TestRegionClassMaskConfig:
    def test_defaults_to_false(self):
        assert TrainConfig().region_class_mask is False

    def test_requires_masked_loss(self):
        with pytest.raises(ValueError, match=r"region_class_mask.*masked_loss"):
            TrainConfig(region_class_mask=True)

    def test_accepts_region_mask_with_masked_loss(self):
        config = TrainConfig(masked_loss=True, region_class_mask=True)

        assert config.region_class_mask is True


class TestRegionClassMaskCollate:
    def test_forwards_region_metadata_as_float32_with_shapes_preserved(self):
        region_boxes = np.array([[1, 2, 5, 6], [0, 0, 8, 8]], dtype=np.float64)
        region_class_masks = np.array([[1, 0, 1], [0, 1, 1]], dtype=np.int64)

        target = collate_fn([
            _sample(
                region_boxes=region_boxes,
                region_class_masks=region_class_masks,
            )
        ])["targets"][0]

        assert target["region_boxes"].dtype == torch.float32
        assert target["region_boxes"].shape == (2, 4)
        assert target["region_class_masks"].dtype == torch.float32
        assert target["region_class_masks"].shape == (2, 3)
        torch.testing.assert_close(target["region_boxes"], torch.tensor(region_boxes).float())
        torch.testing.assert_close(
            target["region_class_masks"], torch.tensor(region_class_masks).float()
        )

    def test_omits_absent_region_metadata(self):
        target = collate_fn([_sample()])["targets"][0]

        assert "region_boxes" not in target
        assert "region_class_masks" not in target

    @pytest.mark.parametrize("key", ["region_boxes", "region_class_masks"])
    def test_forwards_each_region_key_independently(self, key):
        values = (
            np.zeros((1, 4), dtype=np.float32)
            if key == "region_boxes"
            else np.ones((1, 3), dtype=np.float32)
        )

        target = collate_fn([_sample(**{key: values})])["targets"][0]

        assert key in target
        other_key = (
            "region_class_masks" if key == "region_boxes" else "region_boxes"
        )
        assert other_key not in target


class TestRFDETRRegionClassMaskPlumbing:
    def test_enable_flips_adapter_flag(self):
        adapter = RFDETRAdapter()

        assert adapter._region_class_mask is False
        adapter.enable_region_class_mask()
        assert adapter._region_class_mask is True

    def test_create_criterion_forwards_enabled_flag(self, monkeypatch):
        from rfdetr.models import lwdetr

        seen = []

        class _Criterion:
            weight_dict = {}

        def _build(args):
            seen.append(args.region_class_mask)
            return _Criterion(), None

        monkeypatch.setattr(lwdetr, "build_criterion_and_postprocessors", _build)
        adapter = RFDETRAdapter()
        adapter._args = SimpleNamespace()
        adapter.enable_region_class_mask()

        adapter.create_criterion(ModelConfig())

        assert seen == [True]

    @pytest.mark.parametrize(
        ("metadata", "expected_keys"),
        [
            ({}, set()),
            ({"region_boxes": torch.zeros(1, 4)}, {"region_boxes"}),
            (
                {"region_class_masks": torch.ones(1, 3)},
                {"region_class_masks"},
            ),
            (
                {
                    "region_boxes": torch.zeros(1, 4),
                    "region_class_masks": torch.ones(1, 3),
                },
                {"region_boxes", "region_class_masks"},
            ),
        ],
    )
    def test_format_batch_forwards_present_region_keys_when_enabled(
        self, metadata, expected_keys
    ):
        adapter = RFDETRAdapter()
        adapter.enable_region_class_mask()
        batch = {
            "images": torch.zeros(1, 3, 8, 8),
            "targets": [
                {
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "masks": torch.zeros(0, 8, 8),
                    **metadata,
                }
            ],
        }

        target = adapter.format_batch(batch)["targets"][0]

        assert expected_keys == ({"region_boxes", "region_class_masks"} & target.keys())

    def test_format_batch_ignores_region_metadata_when_disabled(self):
        batch = {
            "images": torch.zeros(1, 3, 8, 8),
            "targets": [
                {
                    "boxes": torch.zeros(0, 4),
                    "labels": torch.zeros(0, dtype=torch.int64),
                    "masks": torch.zeros(0, 8, 8),
                    "region_boxes": torch.zeros(1, 4),
                    "region_class_masks": torch.ones(1, 3),
                }
            ],
        }

        target = RFDETRAdapter().format_batch(batch)["targets"][0]

        assert "region_boxes" not in target
        assert "region_class_masks" not in target


class _RecordingAdapter(ModelAdapter):
    def __init__(self):
        self.events = []

    def create_model(self, config):
        self.events.append("create_model")
        return nn.Linear(1, 1)

    def enable_masked_loss(self):
        self.events.append("enable_masked_loss")

    def enable_region_class_mask(self):
        self.events.append("enable_region_class_mask")

    def create_criterion(self, config):
        self.events.append("create_criterion")
        return None

    def format_batch(self, batch):
        raise NotImplementedError

    def extract_predictions(self, outputs):
        raise NotImplementedError

    def compute_loss(self, outputs, targets):
        raise NotImplementedError


class _MaskedLossOnlyAdapter(ModelAdapter):
    """Concrete adapter that supports masked loss but not region masking."""

    def create_model(self, config):
        return nn.Linear(1, 1)

    def enable_masked_loss(self):
        pass

    def create_criterion(self, config):
        return None

    def format_batch(self, batch):
        raise NotImplementedError

    def extract_predictions(self, outputs):
        raise NotImplementedError

    def compute_loss(self, outputs, targets):
        raise NotImplementedError


def test_lit_module_enables_region_mask_after_masked_loss_before_criterion():
    adapter = _RecordingAdapter()

    SegmentationLitModule(
        adapter=adapter,
        config=TrainConfig(masked_loss=True, region_class_mask=True),
    )

    assert adapter.events[:4] == [
        "create_model",
        "enable_masked_loss",
        "enable_region_class_mask",
        "create_criterion",
    ]


def test_unsupported_adapter_rejects_region_class_mask_clearly():
    with pytest.raises(
        NotImplementedError,
        match=r"_MaskedLossOnlyAdapter.*region_class_mask",
    ):
        SegmentationLitModule(
            adapter=_MaskedLossOnlyAdapter(),
            config=TrainConfig(masked_loss=True, region_class_mask=True),
        )
