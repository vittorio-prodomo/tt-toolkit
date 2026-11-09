"""`val_mask_res: 0` must mean FULL RESOLUTION, not "resize to 0x0".

The documented opt-out spelling was ``val_mask_res: 0`` or ``null``. A strict
``is None`` guard once let a 0 fall through to ``F.interpolate(size=(0, 0))``,
which raised as soon as a batch contained any prediction -- and, on an empty
batch, silently returned a malformed (0, 0, 0) tensor instead of (0, H, W).

The knob has since been superseded by the relative ``val_mask_downsample``
(see ``models/val_mask_resolution.py``), and the default flipped from full-res
to native. That makes this regression MORE important to keep locked, not less:
``val_mask_res: 0`` in an old config now has to survive two hazards -- the
original 0x0 crash, and being quietly absorbed into the new native default,
which would silently coarsen a run that explicitly asked for full resolution.
"""
import pytest
import torch

from training_toolkit.models.mask2former import Mask2FormerAdapter
from training_toolkit.models.rfdetr import RFDETRAdapter

ADAPTERS = [RFDETRAdapter, Mask2FormerAdapter]


def _preds(res=64, n=3):
    return [{"masks": torch.zeros(n, res, res, dtype=torch.bool)}]


def _adapter(cls, extra):
    a = cls.__new__(cls)  # bypass __init__ (needs a full config)
    a._init_val_mask_resolution(extra)
    return a


@pytest.mark.parametrize("cls", ADAPTERS)
@pytest.mark.parametrize("off_value", [None, 0])
def test_falsy_val_mask_res_still_means_full_resolution(cls, off_value):
    a = _adapter(cls, {"val_mask_res": off_value})
    out = a._apply_val_mask_resolution(
        _preds(), input_hw=(64, 64), native_hw=(16, 16), masks_already_native=False,
    )
    assert out[0]["masks"].shape == (3, 64, 64), (
        "falsy val_mask_res is the full-resolution opt-out -- it must neither "
        "resize to 0x0 nor be absorbed into the native default"
    )


@pytest.mark.parametrize("cls", ADAPTERS)
def test_truthy_val_mask_res_still_downsamples(cls):
    a = _adapter(cls, {"val_mask_res": 16})
    out = a._apply_val_mask_resolution(
        _preds(), input_hw=(64, 64), native_hw=(16, 16), masks_already_native=False,
    )
    assert out[0]["masks"].shape == (3, 16, 16), "a real value must still downsample"


@pytest.mark.parametrize("cls", ADAPTERS)
def test_empty_batch_keeps_a_well_formed_shape(cls):
    """The other half of the original bug: an empty prediction set must still
    come out as (0, H, W), not (0, 0, 0)."""
    a = _adapter(cls, {"val_mask_downsample": 4})
    out = a._apply_val_mask_resolution(
        _preds(n=0), input_hw=(64, 64), native_hw=(16, 16), masks_already_native=False,
    )
    assert out[0]["masks"].shape == (0, 16, 16)
