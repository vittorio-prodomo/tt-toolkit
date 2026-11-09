"""The unified ``val_mask_downsample`` knob on the YOLO adapter.

YOLO is the backend the unification has to be gentlest with: its old
``val_full_res_masks: false`` default ALREADY meant "score at the head's native
resolution" (protos are architecturally input/4), so switching it onto the
shared knob must be a no-op for every existing config. These tests lock that,
plus the two things only YOLO can do: let ``process_mask`` perform the full-res
upsample itself (it crops at full res, which beats cropping at proto res and
scaling up), and bridge the intermediate divisors ``process_mask`` does not
offer.

Runs the real ``process_mask`` — only NMS is stubbed, so the mask path under
test is the production one.
"""
import pytest
import torch

from training_toolkit.models.yolo import YOLOAdapter

pytest.importorskip("ultralytics")  # optional: the `yolo` extra

PROTO = 32          # mask-coefficient channels
PH = PW = 16        # proto spatial size -> input 64x64
IMG = PH * YOLOAdapter._PROTO_DOWNSAMPLE
N_DET = 2


@pytest.fixture
def adapter():
    a = YOLOAdapter()
    a._nc = 2
    a._is_end2end = False
    a._val_conf_thres = 0.001
    a._val_max_det = 100
    a._val_nms_iou = 0.7
    return a


@pytest.fixture
def outputs():
    """Eval-mode YOLO-seg outputs: ((decoded, proto), raw)."""
    decoded = torch.zeros(1, 4 + 2 + PROTO, 10)      # unused: NMS is stubbed
    proto = torch.randn(1, PROTO, PH, PW)
    return ((decoded, proto), {})


@pytest.fixture(autouse=True)
def stub_nms(monkeypatch):
    """Return N_DET detections with in-range boxes; leave process_mask real."""
    det = torch.zeros(N_DET, 6 + PROTO)
    det[:, :4] = torch.tensor([[0.0, 0.0, IMG, IMG], [8.0, 8.0, IMG - 8.0, IMG - 8.0]])
    det[:, 4] = 0.9                                   # score
    det[:, 5] = torch.tensor([0.0, 1.0])              # label
    det[:, 6:] = torch.randn(N_DET, PROTO)

    import ultralytics.utils.nms as nms_mod

    monkeypatch.setattr(nms_mod, "non_max_suppression", lambda *a, **k: [det])


def _mask_hw(adapter, outputs):
    preds = adapter.postprocess_for_inference(outputs)
    assert preds[0]["masks"].shape[0] == N_DET, "sanity: detections survived"
    assert preds[0]["masks"].dtype == torch.bool
    return tuple(preds[0]["masks"].shape[-2:])


def test_default_native_keeps_proto_resolution(adapter, outputs):
    """The pre-existing YOLO default, unchanged by the unification."""
    adapter._init_val_mask_resolution({})
    assert _mask_hw(adapter, outputs) == (PH, PW)


def test_legacy_val_full_res_masks_false_matches_the_new_default(adapter, outputs):
    """Old configs spelling the default explicitly must land in the same place."""
    adapter._init_val_mask_resolution({"val_full_res_masks": False})
    assert _mask_hw(adapter, outputs) == (PH, PW)


def test_divisor_one_gives_full_input_resolution(adapter, outputs):
    adapter._init_val_mask_resolution({"val_mask_downsample": 1})
    assert _mask_hw(adapter, outputs) == (IMG, IMG)


def test_legacy_val_full_res_masks_true_gives_full_resolution(adapter, outputs):
    adapter._init_val_mask_resolution({"val_full_res_masks": True})
    assert _mask_hw(adapter, outputs) == (IMG, IMG)


def test_intermediate_divisor_is_bridged(adapter, outputs):
    """process_mask only offers proto-res or full-res; the adapter fills the gap.

    input 64 / 2 = 32, which is neither the proto size (16) nor full res (64).
    """
    adapter._init_val_mask_resolution({"val_mask_downsample": 2})
    assert _mask_hw(adapter, outputs) == (IMG // 2, IMG // 2)


def test_divisor_matching_the_proto_stride_needs_no_resize(adapter, outputs):
    """`val_mask_downsample: 4` names the proto stride explicitly -- it must
    agree with `native` exactly, not go through a pointless interpolate."""
    adapter._init_val_mask_resolution({"val_mask_downsample": 4})
    assert _mask_hw(adapter, outputs) == (PH, PW)


def test_conflicting_knobs_raise(adapter):
    with pytest.raises(ValueError, match="Conflicting val-mask-resolution"):
        adapter._init_val_mask_resolution(
            {"val_mask_downsample": 2, "val_full_res_masks": True}
        )


def test_empty_detections_keep_the_configured_mask_shape(adapter, outputs, monkeypatch):
    """An image with no detections must still report (0, H, W) at the SAME
    resolution as a populated one, or the metric sees ragged mask shapes."""
    import ultralytics.utils.nms as nms_mod

    monkeypatch.setattr(
        nms_mod, "non_max_suppression", lambda *a, **k: [torch.zeros(0, 6 + PROTO)]
    )
    adapter._init_val_mask_resolution({"val_mask_downsample": 2})
    preds = adapter.postprocess_for_inference(outputs)
    assert preds[0]["masks"].shape == (0, IMG // 2, IMG // 2)
