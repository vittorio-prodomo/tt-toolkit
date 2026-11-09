"""Tests for Mask2Former adapter (requires transformers)."""

import pytest
import torch

transformers = pytest.importorskip("transformers")

from training_toolkit.models.mask2former import Mask2FormerAdapter
from training_toolkit.configs import ModelConfig


@pytest.fixture
def adapter():
    return Mask2FormerAdapter()


@pytest.fixture
def model_config():
    return ModelConfig(
        name="mask2former-swin-tiny",
        pretrained="facebook/mask2former-swin-tiny-coco-instance",
        num_classes=2,
    )


@pytest.fixture
def model_config_no_pretrained():
    return ModelConfig(
        name="mask2former",
        pretrained="none",
        num_classes=2,
    )


class TestMask2FormerAdapter:
    def test_create_model_from_config(self, adapter, model_config_no_pretrained):
        """Create model from scratch (no download required)."""
        model = adapter.create_model(model_config_no_pretrained)
        assert model is not None
        assert hasattr(model, "forward")

    @pytest.mark.slow
    def test_create_model_pretrained(self, adapter, model_config):
        """Create model from pretrained weights (requires download)."""
        model = adapter.create_model(model_config)
        assert model is not None
        assert hasattr(model, "forward")

    def test_create_criterion_returns_none(self, adapter, model_config_no_pretrained):
        # Mask2Former computes loss internally
        assert adapter.create_criterion(model_config_no_pretrained) is None

    def test_format_batch(self, adapter):
        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "targets": [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, 64, 64, dtype=torch.uint8),
                },
                {
                    "boxes": torch.tensor(
                        [[5.0, 5.0, 30.0, 30.0], [20.0, 20.0, 60.0, 60.0]]
                    ),
                    "labels": torch.tensor([0, 1]),
                    "masks": torch.ones(2, 64, 64, dtype=torch.uint8),
                },
            ],
        }
        formatted = adapter.format_batch(batch)
        assert "inputs" in formatted
        assert "targets" in formatted
        assert "pixel_values" in formatted["inputs"]
        assert "mask_labels" in formatted["inputs"]
        assert "class_labels" in formatted["inputs"]

        # Check mask_labels are float
        for ml in formatted["inputs"]["mask_labels"]:
            assert ml.dtype == torch.float32

        # Check shapes match
        assert len(formatted["inputs"]["mask_labels"]) == 2
        assert formatted["inputs"]["mask_labels"][0].shape == (1, 64, 64)
        assert formatted["inputs"]["mask_labels"][1].shape == (2, 64, 64)

    def test_masks_to_boxes(self, adapter):
        masks = torch.zeros(2, 10, 10, dtype=torch.bool)
        # First mask: rectangle from (2,3) to (7,8)
        masks[0, 3:9, 2:8] = True
        # Second mask: rectangle from (0,0) to (4,4)
        masks[1, 0:5, 0:5] = True

        boxes = adapter._masks_to_boxes(masks)
        assert boxes.shape == (2, 4)
        # box format: x_min, y_min, x_max, y_max
        assert boxes[0].tolist() == [2.0, 3.0, 7.0, 8.0]
        assert boxes[1].tolist() == [0.0, 0.0, 4.0, 4.0]

    def test_masks_to_boxes_empty(self, adapter):
        masks = torch.zeros(0, 10, 10, dtype=torch.bool)
        boxes = adapter._masks_to_boxes(masks)
        assert boxes.shape == (0, 4)

    def test_extract_predictions(self, adapter):
        """Test prediction extraction with mock outputs."""
        from unittest.mock import MagicMock

        num_queries = 10
        num_classes = 3  # +1 no-object = 4 total in logits
        h, w = 32, 32

        outputs = MagicMock()
        # Shape: (B=1, Q, C+1)
        class_logits = torch.randn(1, num_queries, num_classes + 1)
        # Make first query have high confidence for class 0
        class_logits[0, 0, 0] = 10.0
        class_logits[0, 0, 1:] = -10.0
        outputs.class_queries_logits = class_logits

        # Shape: (B=1, Q, H, W)
        mask_logits = torch.randn(1, num_queries, h, w)
        # Make first query mask very positive (all ones after sigmoid)
        mask_logits[0, 0] = 10.0
        outputs.masks_queries_logits = mask_logits

        preds = adapter.extract_predictions(outputs)
        assert len(preds) == 1
        pred = preds[0]
        assert "boxes" in pred
        assert "scores" in pred
        assert "labels" in pred
        assert "masks" in pred
        # At least the first query should be kept
        assert pred["scores"].shape[0] >= 1

    def test_compute_loss(self, adapter):
        """Test loss extraction from mock outputs."""
        from unittest.mock import MagicMock

        outputs = MagicMock()
        outputs.loss = torch.tensor(1.5)
        outputs.loss_dict = {"loss_ce": torch.tensor(0.5), "loss_mask": torch.tensor(1.0)}

        loss_dict = adapter.compute_loss(outputs, targets=None)
        assert "loss_total" in loss_dict
        assert loss_dict["loss_total"].item() == 1.5
        assert "loss_ce" in loss_dict
        assert "loss_mask" in loss_dict

    def test_resolve_precision_forces_fp16(self, adapter):
        """Mask2Former must force fp16 (16-mixed) — it diverges to NaN under bf16."""
        assert adapter.resolve_precision("bf16-mixed") == "16-mixed"
        assert adapter.resolve_precision("bf16") == "16-mixed"
        assert adapter.resolve_precision("16-mixed") == "16-mixed"
        assert adapter.resolve_precision("32") == "32"


def _mock_m2f_outputs(num_queries: int, num_classes: int, h: int, w: int):
    """A MagicMock outputs object shaped like Mask2Former's forward() output.

    Every query confidently predicts class 0 with an all-positive mask, so
    every query survives the ``keep = max_scores > 0.0`` filter in
    ``postprocess_for_inference`` — gives a deterministic non-empty preds list.
    """
    from unittest.mock import MagicMock

    outputs = MagicMock()
    class_logits = torch.full((1, num_queries, num_classes + 1), -10.0)
    class_logits[0, :, 0] = 10.0  # class 0
    outputs.class_queries_logits = class_logits
    outputs.masks_queries_logits = torch.full((1, num_queries, h, w), 10.0)
    return outputs


class TestValMaskResolution:
    """The unified ``val_mask_downsample`` knob on the Mask2Former adapter.

    M2F emits mask logits at its pixel-decoder stride and the postprocess
    upsamples them to ``image_sizes``. When ``image_sizes`` is None it never
    upsamples, so the masks are already native and there is nothing to do --
    that asymmetry is what ``masks_already_native`` encodes.
    """

    def _adapter(self, extra):
        a = Mask2FormerAdapter()
        a._init_val_mask_resolution(extra)
        return a

    def test_resize_helper_downsamples_nonempty(self):
        masks = torch.ones(3, 64, 64, dtype=torch.bool)
        out = Mask2FormerAdapter._resize_val_masks_hw(masks, (16, 16))
        assert out.shape == (3, 16, 16)
        assert out.dtype == torch.bool
        assert out.all()

    def test_resize_helper_empty_safe(self):
        masks = torch.zeros(0, 64, 64, dtype=torch.bool)
        out = Mask2FormerAdapter._resize_val_masks_hw(masks, (16, 16))
        assert out.shape == (0, 16, 16)
        assert out.dtype == torch.bool

    def test_native_is_a_noop_when_masks_were_never_upsampled(self):
        a = self._adapter({})
        preds = [{"masks": torch.ones(2, 16, 16, dtype=torch.bool)}]
        out = a._apply_val_mask_resolution(
            preds, input_hw=None, native_hw=(16, 16), masks_already_native=True,
        )
        assert out[0]["masks"].shape == (2, 16, 16)

    def test_native_resizes_back_when_the_postprocess_upsampled(self):
        a = self._adapter({})
        preds = [{"masks": torch.ones(2, 64, 64, dtype=torch.bool)}]
        out = a._apply_val_mask_resolution(
            preds, input_hw=(64, 64), native_hw=(16, 16), masks_already_native=False,
        )
        assert out[0]["masks"].shape == (2, 16, 16)

    def test_divisor_is_relative_to_the_input(self):
        a = self._adapter({"val_mask_downsample": 4})
        preds = [{"masks": torch.ones(2, 64, 64, dtype=torch.bool)}]
        out = a._apply_val_mask_resolution(
            preds, input_hw=(64, 64), native_hw=(16, 16), masks_already_native=False,
        )
        assert out[0]["masks"].shape == (2, 16, 16)

    def test_legacy_absolute_val_mask_res_is_still_honoured(self):
        a = self._adapter({"val_mask_res": 8})
        preds = [{
            "boxes": torch.zeros(2, 4),
            "scores": torch.ones(2),
            "masks": torch.ones(2, 64, 64, dtype=torch.bool),
        }]
        out = a._apply_val_mask_resolution(
            preds, input_hw=(64, 64), native_hw=(16, 16), masks_already_native=False,
        )
        assert out[0]["masks"].shape == (2, 8, 8)
        assert out[0]["masks"].dtype == torch.bool
        assert out[0]["scores"].shape == (2,)
        assert out[0]["boxes"].shape == (2, 4)

    def test_postprocess_full_res_request_keeps_input_resolution(self, adapter):
        outputs = _mock_m2f_outputs(num_queries=4, num_classes=2, h=32, w=32)
        adapter._init_val_mask_resolution({"val_mask_downsample": 1})
        preds = adapter.postprocess_for_inference(outputs, image_sizes=(32, 32))
        assert preds[0]["masks"].shape[0] > 0  # sanity: queries were kept
        assert preds[0]["masks"].shape[-2:] == (32, 32)

    def test_postprocess_divisor_downsamples(self, adapter):
        outputs = _mock_m2f_outputs(num_queries=4, num_classes=2, h=32, w=32)
        adapter._init_val_mask_resolution({"val_mask_downsample": 4})
        preds = adapter.postprocess_for_inference(outputs, image_sizes=(32, 32))
        assert preds[0]["masks"].shape[0] > 0
        assert preds[0]["masks"].shape[-2:] == (8, 8)
        assert preds[0]["masks"].dtype == torch.bool

    def test_postprocess_legacy_val_mask_res_downsamples(self, adapter):
        outputs = _mock_m2f_outputs(num_queries=4, num_classes=2, h=32, w=32)
        adapter._init_val_mask_resolution({"val_mask_res": 8})
        preds = adapter.postprocess_for_inference(outputs, image_sizes=(32, 32))
        assert preds[0]["masks"].shape[-2:] == (8, 8)

    def test_create_model_defaults_to_native(self, adapter, model_config_no_pretrained):
        adapter.create_model(model_config_no_pretrained)
        assert adapter._val_mask_spec.is_native

    def test_create_model_reads_the_knob(self, adapter):
        config = ModelConfig(
            name="mask2former", pretrained="none", num_classes=2,
            extra={"val_mask_downsample": 2},
        )
        adapter.create_model(config)
        assert adapter._val_mask_spec.mode == "divisor"
        assert adapter._val_mask_spec.value == 2

    def test_create_model_rejects_conflicting_knobs(self, adapter):
        config = ModelConfig(
            name="mask2former", pretrained="none", num_classes=2,
            extra={"val_mask_downsample": 2, "val_full_res_masks": True},
        )
        with pytest.raises(ValueError, match="Conflicting val-mask-resolution"):
            adapter.create_model(config)



class TestMask2FormerMaskedLoss:
    def test_enable_masked_loss_swaps_criterion(self, adapter, model_config_no_pretrained):
        model = adapter.create_model(model_config_no_pretrained)
        assert type(model.criterion).__name__ == "Mask2FormerLoss"
        adapter.enable_masked_loss()                       # must not raise
        assert type(model.criterion).__name__ == "MaskedMask2FormerLoss"
        assert model.criterion._class_mask is None

    def test_format_batch_stashes_class_mask(self, adapter, model_config_no_pretrained):
        adapter.create_model(model_config_no_pretrained)
        adapter.enable_masked_loss()
        batch = {
            "images": torch.rand(1, 3, 32, 32),
            "targets": [{
                "masks": torch.zeros(0, 32, 32, dtype=torch.uint8),
                "labels": torch.tensor([], dtype=torch.int64),
                "class_mask": torch.tensor([1.0, 0.0]),
            }],
        }
        adapter.format_batch(batch)
        stashed = adapter._model.criterion._class_mask
        assert len(stashed) == 1
        assert torch.equal(stashed[0], torch.tensor([1.0, 0.0]))

    def test_format_batch_stashes_none_when_absent(self, adapter, model_config_no_pretrained):
        adapter.create_model(model_config_no_pretrained)
        adapter.enable_masked_loss()
        batch = {
            "images": torch.rand(1, 3, 32, 32),
            "targets": [{
                "masks": torch.zeros(0, 32, 32, dtype=torch.uint8),
                "labels": torch.tensor([], dtype=torch.int64),
            }],
        }
        adapter.format_batch(batch)
        assert adapter._model.criterion._class_mask == [None]
