from __future__ import annotations

import math
import numpy as np
import pytest

from training_toolkit.data.label_crop import SizeBand, band_status
from training_toolkit.data.tile_materialize import Instance


def test_sizeband_for_imgsz_ceiling_is_three_quarters_of_input():
    band = SizeBand.for_imgsz(504)
    assert band.floor == 24.0
    assert band.ceiling == pytest.approx(378.0)  # 0.75 * 504


@pytest.mark.parametrize(
    "in_model,expected",
    [
        (23.9, "ignore"),   # below floor
        (24.0, "positive"), # exactly floor -> in-band
        (200.0, "positive"),
        (378.0, "positive"),# exactly ceiling -> in-band
        (378.1, "ignore"),  # above ceiling
    ],
)
def test_band_status_boundaries(in_model, expected):
    assert band_status(in_model, SizeBand(floor=24.0, ceiling=378.0)) == expected


def test_band_status_none_is_always_positive():
    # indoor whole-frame regime: no band -> never ignore by size
    assert band_status(1.0, None) == "positive"
    assert band_status(99999.0, None) == "positive"


from training_toolkit.data.label_crop import label_crop


def _square_instance(label, x0, y0, side):
    """An Instance whose polygon is an axis-aligned square (abs px)."""
    ring = np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                    dtype=np.float32)
    return Instance(label=label, polygon=[ring], mask=None, mask_origin=None,
                    bbox=(float(x0), float(y0), float(x0 + side), float(y0 + side)))


# A tile at the image origin, big enough to hold the squares below.
TILE = (0, 0, 100, 100)
# Band [24, 80] in-model -> with realized_scale=1.0 a side-40 square (~40 in-model)
# is positive, a side-18 square (~18) is ignore-small, a side-90 (~90) is ignore-large.
BAND = SizeBand(floor=24.0, ceiling=80.0)
MASK = [1, 1, 0]


def test_label_crop_in_band_is_positive():
    inst = [_square_instance(0, 10, 10, 40)]
    out = label_crop(TILE, inst, realized_scale=1.0, class_mask=MASK, band=BAND)
    assert len(out["labels"]) == 1
    assert out["ignore"].tolist() == [False]
    assert out["negative"] is False
    assert out["class_mask"] == MASK
    assert out["boxes"].shape == (1, 4)
    assert out["masks"][0].shape == (100, 100)


def test_label_crop_below_floor_is_ignored():
    inst = [_square_instance(0, 10, 10, 18)]
    out = label_crop(TILE, inst, realized_scale=1.0, class_mask=MASK, band=BAND)
    assert out["ignore"].tolist() == [True]
    assert out["negative"] is True  # the only instance is ignored


def test_label_crop_above_ceiling_is_ignored():
    inst = [_square_instance(0, 5, 5, 90)]
    out = label_crop(TILE, inst, realized_scale=1.0, class_mask=MASK, band=BAND)
    assert out["ignore"].tolist() == [True]
    assert out["negative"] is True


def test_label_crop_indoor_no_band_supervises_everything():
    # band=None -> even a tiny and a huge instance are positive
    inst = [_square_instance(0, 10, 10, 8), _square_instance(1, 0, 0, 98)]
    out = label_crop(TILE, inst, realized_scale=1.0, class_mask=MASK, band=None)
    assert out["ignore"].tolist() == [False, False]
    assert out["negative"] is False


def test_label_crop_mixed_positive_and_ignore_is_not_negative():
    inst = [_square_instance(0, 5, 5, 40),    # positive
            _square_instance(1, 60, 60, 16)]  # ignore-small
    out = label_crop(TILE, inst, realized_scale=1.0, class_mask=MASK, band=BAND)
    assert out["ignore"].tolist() == [False, True]
    assert out["negative"] is False  # one positive present


def test_label_crop_empty_when_no_instance_overlaps():
    inst = [_square_instance(0, 500, 500, 40)]  # far outside the tile
    out = label_crop(TILE, inst, realized_scale=1.0, class_mask=MASK, band=BAND)
    assert out["labels"].shape == (0,)
    assert out["boxes"].shape == (0, 4)
    assert out["masks"] == []
    assert out["ignore"].tolist() == []
    assert out["negative"] is True


def test_for_imgsz_none_ceiling_is_floor_only():
    band = SizeBand.for_imgsz(504, ceiling_frac=None)
    assert band.floor == 24.0
    assert math.isinf(band.ceiling)


def test_for_imgsz_default_ceiling_unchanged():
    band = SizeBand.for_imgsz(504)
    assert band.ceiling == pytest.approx(378.0)   # 0.75 * 504 — regression pin


def test_label_crop_infinite_ceiling_keeps_giant_positive():
    # side-90 square ≈ 90 in-model at scale 1.0: above the 80 ceiling -> ignore
    # under the bounded band, positive under the floor-only band.
    giant = _square_instance(0, 0, 0, 90)
    bounded = label_crop(TILE, [giant], realized_scale=1.0, class_mask=MASK,
                         band=SizeBand(floor=24.0, ceiling=80.0))
    assert bounded["ignore"].tolist() == [True]
    floor_only = label_crop(TILE, [giant], realized_scale=1.0, class_mask=MASK,
                            band=SizeBand(floor=24.0, ceiling=float("inf")))
    assert floor_only["ignore"].tolist() == [False]
