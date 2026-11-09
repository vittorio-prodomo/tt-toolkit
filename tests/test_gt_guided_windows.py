"""gt_guided per-defect chips."""
from __future__ import annotations

import random

import numpy as np

from training_toolkit.data.tile_index import gt_guided_windows


def _params(**over):
    p = dict(imgsz=504, target_in_model=110.0, min_crop=416,
             translation_jitter=0.3, scale_jitter=0.15, dedup_iou=0.6)
    p.update(over)
    return p


def test_crop_size_scales_with_defect_area():
    # a big defect (sqrt-area ~200) and a small one (~20) on a large frame
    boxes = np.array([[0, 0, 200, 200], [2000, 2000, 2020, 2020]], np.float32)
    wins = gt_guided_windows(4000, 4000, boxes, rng=random.Random(0),
                             **_params(scale_jitter=0.0))
    big = [w for (w, _ab) in wins if w[0] < 1000][0]
    small = [w for (w, _ab) in wins if w[0] >= 1000][0]
    # big: crop ~= 200 * 504/110 ~= 916 ; small clamps to min_crop=416
    assert max(big[2], big[3]) > 700
    assert max(small[2], small[3]) == 416


def test_crop_clamps_to_image_when_defect_huge():
    boxes = np.array([[0, 0, 900, 900]], np.float32)   # sqrt-area 900 -> crop ~4124 > image
    (w, ab), = gt_guided_windows(1000, 800, boxes, rng=random.Random(0), **_params())
    assert ab is True
    assert w[2] <= 1000 and w[3] <= 800     # clamped within image
    assert w[0] >= 0 and w[1] >= 0
    assert w[2] == w[3], "chip must be square"


def test_in_model_size_concentrates_near_target():
    # spread of mid/large defects placed far apart so their chips never overlap
    # (no IoU dedup -> wins stays 1:1 with sizes); each chip sizes its defect to ~target.
    sizes = [120, 160, 200, 260]
    boxes = np.array(
        [[(i + 1) * 1800, (i + 1) * 1800, (i + 1) * 1800 + s, (i + 1) * 1800 + s]
         for i, s in enumerate(sizes)], np.float32)
    wins = gt_guided_windows(10000, 10000, boxes, rng=random.Random(3),
                             **_params(scale_jitter=0.0))
    assert len(wins) == len(sizes)   # far apart -> no dedup -> 1:1 with sizes
    for (s, (w, _ab)) in zip(sizes, wins):
        in_model = s * 504 / max(w[2], w[3])
        assert 88 <= in_model <= 138, f"size {s}: in_model {in_model:.1f} off-band"


def test_iou_dedup_collapses_coincident_defects():
    # two near-identical defects -> their chips overlap heavily -> dedup to one
    boxes = np.array([[1000, 1000, 1100, 1100], [1005, 1005, 1105, 1105]], np.float32)
    wins = gt_guided_windows(4000, 4000, boxes, rng=random.Random(0),
                             **_params(scale_jitter=0.0, translation_jitter=0.0))
    assert len(wins) == 1


def test_windows_contain_defect_centroid():
    boxes = np.array([[500, 500, 560, 560]], np.float32)
    (w, _ab), = gt_guided_windows(4000, 4000, boxes, rng=random.Random(0),
                                  **_params(scale_jitter=0.0))
    cx, cy = 530.0, 530.0
    assert w[0] <= cx <= w[0] + w[2] and w[1] <= cy <= w[1] + w[3]
