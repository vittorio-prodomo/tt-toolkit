"""Toolkit-only val knobs in ModelConfig.extra must never reach a backend ctor.

Mask2Former's create_model splats `extra` into HF from_pretrained(); an
unrecognised kwarg (e.g. the eval harness's val_full_res_masks) raises
TypeError. The strip-set had been hand-copied per adapter and drifted --
val_full_res_masks was missing from every copy, which crashed the M2F full-res
eval. This pins the single source of truth and its membership.
"""
import pytest

from training_toolkit.models.base import ModelAdapter


def test_full_res_masks_is_in_the_shared_strip_set():
    # the exact key whose omission crashed the M2F full-res eval
    assert "val_full_res_masks" in ModelAdapter.TOOLKIT_ONLY_EXTRA_KEYS


@pytest.mark.parametrize("key", [
    "val_mask_res", "val_conf_thres", "val_max_det", "val_nms_iou",
    "val_full_res_masks",
])
def test_every_val_knob_is_stripped(key):
    assert key in ModelAdapter.TOOLKIT_ONLY_EXTRA_KEYS


def test_adapters_reference_the_shared_set_not_a_copy():
    """Guard against a new hand-rolled {"val_..."} literal reappearing."""
    import inspect
    from training_toolkit.models import mask2former, rfdetr
    for mod in (mask2former, rfdetr):
        src = inspect.getsource(mod)
        assert "TOOLKIT_ONLY_EXTRA_KEYS" in src, f"{mod.__name__} must use the shared set"
        # the old literal listed conf/max_det together -- catch its return
        assert '"val_conf_thres", "val_max_det"' not in src, \
            f"{mod.__name__} re-inlined the strip set"
