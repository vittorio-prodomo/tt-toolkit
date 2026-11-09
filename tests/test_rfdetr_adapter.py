"""Tests for RF-DETR adapter (requires rfdetr)."""

import pytest
import torch

rfdetr_pkg = pytest.importorskip("rfdetr")

from training_toolkit.models.rfdetr import RFDETRAdapter
from training_toolkit.configs import ModelConfig


@pytest.fixture
def adapter():
    return RFDETRAdapter()


@pytest.fixture
def model_config():
    return ModelConfig(
        name="rf-detr-base",
        pretrained="auto",
        num_classes=2,
    )


# Names that select each supported size (nano, small, medium, base, large)
@pytest.mark.parametrize("model_name", [
    "rf-detr-nano",
    "rf-detr-small",
    "rf-detr-medium",
    "rf-detr-base",
    "rf-detr-large",
])
def test_create_model_all_sizes(adapter, model_name):
    """Create model for each supported size from scratch (no pretrained download)."""
    config = ModelConfig(name=model_name, pretrained="none", num_classes=2)
    model = adapter.create_model(config)
    assert model is not None


class TestRFDETRAdapter:
    def test_create_model(self, adapter, model_config):
        model = adapter.create_model(model_config)
        assert model is not None

    def test_create_criterion(self, adapter, model_config):
        # First create model (criterion needs model config)
        adapter.create_model(model_config)
        criterion = adapter.create_criterion(model_config)
        assert criterion is not None

    def test_format_batch_creates_nested_tensor(self, adapter):
        batch = {
            "images": torch.randn(2, 3, 64, 64),
            "targets": [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, 64, 64, dtype=torch.uint8),
                },
            ] * 2,
        }
        formatted = adapter.format_batch(batch)
        assert "inputs" in formatted
        assert "targets" in formatted


def _fake_preds(n: int, res: int) -> list[dict]:
    """A single-image preds list in the postprocess_for_inference output shape."""
    return [{
        "boxes": torch.zeros(n, 4),
        "scores": torch.ones(n),
        "labels": torch.zeros(n, dtype=torch.long),
        "masks": torch.ones(n, res, res, dtype=torch.bool),
    }]


class TestValMaskResolution:
    """The unified ``val_mask_downsample`` knob on the RF-DETR adapter.

    RF-DETR's ``PostProcess`` fuses box rescaling with the mask interpolate, so
    predicted masks always arrive at FULL resolution -- unlike YOLO/M2F this
    adapter cannot simply decline to upsample, and ``native`` has to be reached
    by resizing back down. See ``models/val_mask_resolution.py`` for why the
    knob is a divisor rather than an absolute pixel count.
    """

    def _adapter(self, extra):
        a = RFDETRAdapter()
        a._init_val_mask_resolution(extra)
        return a

    def test_resize_helper_downsamples_nonempty(self):
        masks = torch.ones(3, 504, 504, dtype=torch.bool)
        out = RFDETRAdapter._resize_val_masks_hw(masks, (192, 192))
        assert out.shape == (3, 192, 192)
        assert out.dtype == torch.bool
        assert out.all()  # all-true content preserved under nearest resize

    def test_resize_helper_empty_safe(self):
        masks = torch.zeros(0, 504, 504, dtype=torch.bool)
        out = RFDETRAdapter._resize_val_masks_hw(masks, (192, 192))
        assert out.shape == (0, 192, 192)
        assert out.dtype == torch.bool

    def test_resize_helper_is_identity_at_the_same_size(self):
        masks = torch.ones(2, 64, 64, dtype=torch.bool)
        out = RFDETRAdapter._resize_val_masks_hw(masks, (64, 64))
        assert out is masks  # no pointless interpolate

    def test_default_native_resizes_down_to_the_head_resolution(self):
        """The new default. PostProcess handed us 504; the head only ever
        produced 126, so that is the honest resolution to score at."""
        a = self._adapter({})
        out = a._apply_val_mask_resolution(
            _fake_preds(2, 504), input_hw=(504, 504), native_hw=(126, 126),
        )
        assert out[0]["masks"].shape == (2, 126, 126)

    def test_divisor_one_keeps_full_input_resolution(self):
        """The old default, still reachable explicitly."""
        a = self._adapter({"val_mask_downsample": 1})
        out = a._apply_val_mask_resolution(
            _fake_preds(2, 504), input_hw=(504, 504), native_hw=(126, 126),
        )
        assert out[0]["masks"].shape == (2, 504, 504)

    def test_divisor_tracks_the_input_resolution(self):
        """The property the absolute knob lacked: ONE config line means the
        same fidelity on the 504 arm and the 1008 arm."""
        a = self._adapter({"val_mask_downsample": 2})
        at_504 = a._apply_val_mask_resolution(
            _fake_preds(1, 504), input_hw=(504, 504), native_hw=(126, 126),
        )
        at_1008 = a._apply_val_mask_resolution(
            _fake_preds(1, 1008), input_hw=(1008, 1008), native_hw=(252, 252),
        )
        assert at_504[0]["masks"].shape == (1, 252, 252)
        assert at_1008[0]["masks"].shape == (1, 504, 504)

    def test_legacy_absolute_val_mask_res_is_still_honoured(self):
        """Old configs (written before this option existed) must keep working."""
        a = self._adapter({"val_mask_res": 192})
        out = a._apply_val_mask_resolution(
            _fake_preds(2, 504), input_hw=(504, 504), native_hw=(126, 126),
        )
        assert out[0]["masks"].shape == (2, 192, 192)

    def test_legacy_falsy_val_mask_res_means_full_resolution(self):
        """``val_mask_res: 0`` was the documented opt-out spelling; it meant
        full res and must NOT be swallowed by the new native default."""
        a = self._adapter({"val_mask_res": 0})
        out = a._apply_val_mask_resolution(
            _fake_preds(2, 504), input_hw=(504, 504), native_hw=(126, 126),
        )
        assert out[0]["masks"].shape == (2, 504, 504)

    def test_non_mask_fields_pass_through_untouched(self):
        a = self._adapter({"val_mask_downsample": 2})
        out = a._apply_val_mask_resolution(
            _fake_preds(2, 504), input_hw=(504, 504), native_hw=(126, 126),
        )
        assert out[0]["scores"].shape == (2,)
        assert out[0]["boxes"].shape == (2, 4)

    def test_create_model_defaults_to_native(self, adapter):
        config = ModelConfig(name="rf-detr-base", pretrained="none", num_classes=2)
        adapter.create_model(config)
        assert adapter._val_mask_spec.is_native

    def test_create_model_reads_the_knob(self, adapter):
        config = ModelConfig(
            name="rf-detr-base", pretrained="none", num_classes=2,
            extra={"val_mask_downsample": 2},
        )
        adapter.create_model(config)
        assert adapter._val_mask_spec.mode == "divisor"
        assert adapter._val_mask_spec.value == 2

    def test_create_model_rejects_conflicting_knobs(self, adapter):
        """Fails at construction, not silently mid-run."""
        config = ModelConfig(
            name="rf-detr-base", pretrained="none", num_classes=2,
            extra={"val_mask_downsample": 2, "val_mask_res": 192},
        )
        with pytest.raises(ValueError, match="Conflicting val-mask-resolution"):
            adapter.create_model(config)

    def test_val_mask_keys_never_reach_the_backend(self, adapter):
        """They are toolkit knobs; build_model() would reject them."""
        for key in ("val_mask_downsample", "val_mask_res", "val_full_res_masks"):
            assert key in RFDETRAdapter.TOOLKIT_ONLY_EXTRA_KEYS


class TestPostProcessMaskSizeHandoff:
    """The adapter delegates the mask resolution to PostProcess.

    `mask_size` decouples the mask resolution from `target_sizes` (which still
    has to be the true image frame, since boxes are rescaled with it). Getting
    this handoff wrong is invisible -- the masks come back at some resolution
    either way -- so it is pinned explicitly.
    """

    class _StubPostProcess:
        """Records `mask_size` and honours it, like the real PostProcess."""

        def __init__(self, native_hw):
            self.native_hw = native_hw
            self.mask_size = None
            self.seen = []

        def __call__(self, outputs, target_sizes):
            self.seen.append(self.mask_size)
            if self.mask_size == "native":
                hw = self.native_hw
            elif self.mask_size is not None:
                hw = tuple(self.mask_size)
            else:
                hw = tuple(int(v) for v in target_sizes[0].tolist())
            return [{
                "scores": torch.ones(2),
                "labels": torch.zeros(2, dtype=torch.long),
                "boxes": torch.zeros(2, 4),
                "masks": torch.ones(2, 1, *hw, dtype=torch.bool),
            }]

    def _run(self, extra, native_hw=(126, 126), image_sizes=(504, 504)):
        adapter = RFDETRAdapter()
        adapter._init_val_mask_resolution(extra)
        stub = self._StubPostProcess(native_hw)
        adapter._postprocessor = stub
        outputs = {
            "pred_logits": torch.zeros(1, 4, 2),
            "pred_masks": torch.zeros(1, 4, *native_hw),
        }
        preds = adapter.postprocess_for_inference(outputs, image_sizes=image_sizes)
        return stub, preds

    def test_native_is_requested_from_postprocess_not_resized_afterwards(self):
        """The payoff of the decoupling: no upsample-then-downsample round
        trip, which would resample every mask boundary twice."""
        stub, preds = self._run({})
        assert stub.seen == ["native"]
        assert preds[0]["masks"].shape[-2:] == (126, 126)

    def test_divisor_is_pushed_down_as_an_explicit_size(self):
        stub, preds = self._run({"val_mask_downsample": 2})
        assert stub.seen == [(252, 252)]
        assert preds[0]["masks"].shape[-2:] == (252, 252)

    def test_full_res_leaves_postprocess_on_the_box_frame(self):
        stub, preds = self._run({"val_mask_downsample": 1})
        assert stub.seen == [(504, 504)]
        assert preds[0]["masks"].shape[-2:] == (504, 504)

    def test_legacy_absolute_size_is_pushed_down_too(self):
        stub, preds = self._run({"val_mask_res": 192})
        assert stub.seen == [(192, 192)]
        assert preds[0]["masks"].shape[-2:] == (192, 192)

    def test_an_rfdetr_that_ignores_mask_size_is_still_corrected(self):
        """Older rf-detr builds accept the attribute and do nothing with it.
        The adapter's own resize must catch that rather than silently returning
        full-resolution masks."""
        adapter = RFDETRAdapter()
        adapter._init_val_mask_resolution({})

        class _IgnoresMaskSize(self._StubPostProcess):
            def __call__(self, outputs, target_sizes):
                super().__call__(outputs, target_sizes)
                hw = tuple(int(v) for v in target_sizes[0].tolist())
                return [{
                    "scores": torch.ones(2),
                    "labels": torch.zeros(2, dtype=torch.long),
                    "boxes": torch.zeros(2, 4),
                    "masks": torch.ones(2, 1, *hw, dtype=torch.bool),
                }]

        adapter._postprocessor = _IgnoresMaskSize((126, 126))
        outputs = {
            "pred_logits": torch.zeros(1, 4, 2),
            "pred_masks": torch.zeros(1, 4, 126, 126),
        }
        preds = adapter.postprocess_for_inference(outputs, image_sizes=(504, 504))
        assert preds[0]["masks"].shape[-2:] == (126, 126)
