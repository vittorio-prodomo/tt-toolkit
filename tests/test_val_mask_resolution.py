"""Unit tests for the unified val-mask-resolution knob.

Pure config parsing + arithmetic -- no torch, no model, no GPU.
"""

import pytest

from training_toolkit.models.val_mask_resolution import (
    NATIVE,
    ValMaskResolution,
    resolve_val_mask_resolution,
    target_mask_hw,
)


# ---------------------------------------------------------------- defaults


def test_unset_defaults_to_native():
    spec = resolve_val_mask_resolution({})
    assert spec.is_native
    assert target_mask_hw(spec, (504, 504)) is None


def test_none_extra_defaults_to_native():
    assert resolve_val_mask_resolution(None).is_native


# ---------------------------------------------------------------- new knob


def test_native_string_is_accepted_case_insensitively():
    assert resolve_val_mask_resolution({"val_mask_downsample": "Native"}).is_native


def test_divisor_is_relative_to_the_input_canvas():
    """The whole point of the knob: one config line, consistent fidelity.

    The same `val_mask_downsample: 2` must mean 252 on the 504 arm and 504 on
    the 1008 arm -- unlike `val_mask_res: 192`, which is 2.6x on one and 5.25x
    on the other.
    """
    spec = resolve_val_mask_resolution({"val_mask_downsample": 2})
    assert target_mask_hw(spec, (504, 504)) == (252, 252)
    assert target_mask_hw(spec, (1008, 1008)) == (504, 504)


def test_divisor_one_means_full_input_resolution():
    spec = resolve_val_mask_resolution({"val_mask_downsample": 1})
    assert target_mask_hw(spec, (504, 504)) == (504, 504)


def test_divisor_preserves_aspect_ratio_on_a_non_square_canvas():
    """A divisor cannot introduce the square-canvas assumption the old
    absolute-`(res, res)` helpers carried."""
    spec = resolve_val_mask_resolution({"val_mask_downsample": 4})
    assert target_mask_hw(spec, (1080, 1920)) == (270, 480)


def test_fractional_divisor_is_allowed_and_rounds():
    spec = resolve_val_mask_resolution({"val_mask_downsample": 2.625})
    # the old 192-on-504 operating point, expressed as the ratio it really was
    assert target_mask_hw(spec, (504, 504)) == (192, 192)


def test_divisor_never_rounds_a_tiny_canvas_to_zero():
    spec = resolve_val_mask_resolution({"val_mask_downsample": 64})
    assert target_mask_hw(spec, (16, 16)) == (1, 1)


@pytest.mark.parametrize("bad", [0, 0.5, -1])
def test_divisor_below_one_is_rejected(bad):
    with pytest.raises(ValueError, match="DIVISOR"):
        resolve_val_mask_resolution({"val_mask_downsample": bad})


@pytest.mark.parametrize("bad", ["fast", "", True, None, [2]])
def test_non_numeric_non_native_downsample_is_rejected(bad):
    with pytest.raises(ValueError, match="native"):
        resolve_val_mask_resolution({"val_mask_downsample": bad})


# ------------------------------------------------------- legacy val_mask_res


def test_legacy_val_mask_res_still_works_as_absolute_px():
    spec = resolve_val_mask_resolution({"val_mask_res": 192})
    assert spec == ValMaskResolution("absolute", 192)
    # absolute ignores the canvas -- that is exactly the defect it is
    # deprecated for, and it must keep behaving that way for old configs
    assert target_mask_hw(spec, (504, 504)) == (192, 192)
    assert target_mask_hw(spec, (1008, 1008)) == (192, 192)


@pytest.mark.parametrize("falsy", [0, None])
def test_legacy_val_mask_res_falsy_is_the_full_res_opt_out(falsy):
    """`val_mask_res: 0` was the documented opt-out spelling; it meant full
    resolution, and must not silently become the new native default."""
    spec = resolve_val_mask_resolution({"val_mask_res": falsy})
    assert target_mask_hw(spec, (504, 504)) == (504, 504)


@pytest.mark.parametrize("bad", [-5, 1.5, "192", True])
def test_legacy_val_mask_res_rejects_non_positive_int(bad):
    with pytest.raises(ValueError, match="val_mask_res"):
        resolve_val_mask_resolution({"val_mask_res": bad})


# -------------------------------------------------- legacy val_full_res_masks


def test_legacy_val_full_res_masks_true_means_full_res():
    spec = resolve_val_mask_resolution({"val_full_res_masks": True})
    assert target_mask_hw(spec, (640, 640)) == (640, 640)


def test_legacy_val_full_res_masks_false_means_native():
    """YOLO's old default. It must map onto the new default exactly, so
    flipping YOLO to the unified knob is a no-op for existing configs."""
    assert resolve_val_mask_resolution({"val_full_res_masks": False}).is_native


# ------------------------------------------------------------- conflicts


@pytest.mark.parametrize(
    "extra",
    [
        {"val_mask_downsample": 2, "val_mask_res": 192},
        {"val_mask_downsample": 2, "val_full_res_masks": True},
        {"val_mask_res": 192, "val_full_res_masks": False},
        {"val_mask_downsample": 2, "val_mask_res": 192, "val_full_res_masks": True},
    ],
)
def test_conflicting_knobs_raise_instead_of_picking_a_winner(extra):
    """No defined precedence exists, and silently picking one is how a run
    gets measured at a resolution nobody asked for."""
    with pytest.raises(ValueError, match="Conflicting val-mask-resolution"):
        resolve_val_mask_resolution(extra)


def test_conflict_error_names_every_offending_key():
    with pytest.raises(ValueError) as exc:
        resolve_val_mask_resolution({"val_mask_downsample": 2, "val_mask_res": 192})
    assert "val_mask_downsample" in str(exc.value)
    assert "val_mask_res" in str(exc.value)


# ------------------------------------------------------- unknown canvas


def test_unknown_canvas_falls_back_to_native_for_a_divisor():
    """A divisor needs something to divide. Native is the one policy that
    needs no canvas, so it is the safe fallback."""
    spec = resolve_val_mask_resolution({"val_mask_downsample": 4})
    assert target_mask_hw(spec, None) is None


def test_unknown_canvas_still_honours_an_absolute_request():
    spec = resolve_val_mask_resolution({"val_mask_res": 192})
    assert target_mask_hw(spec, None) == (192, 192)


# ------------------------------------------------------------- describe


@pytest.mark.parametrize(
    "extra,fragment",
    [
        ({}, "native"),
        ({"val_mask_downsample": 1}, "full input resolution"),
        ({"val_mask_downsample": 4}, "input / 4"),
        ({"val_mask_res": 192}, "192x192"),
    ],
)
def test_describe_is_human_readable_for_the_startup_log(extra, fragment):
    assert fragment in resolve_val_mask_resolution(extra).describe()
