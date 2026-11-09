"""Regime-scoped tile_sizes."""
from __future__ import annotations

import json

import cv2
import numpy as np

from training_toolkit.configs.tiling_config import TilingConfig
from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.tile_index import build_tile_index


def _coco_two_regimes(tmp_path):
    """120x120 'tiled_multiscale' + 120x120 'stain_1024', each with one annotation."""
    root = tmp_path / "ds"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    for name in ("t.png", "s.png"):
        cv2.imwrite(str(root / "images" / "train" / name), np.zeros((120, 120, 3), np.uint8))
    ann = {
        "images": [
            {"id": 1, "file_name": "t.png", "width": 120, "height": 120, "regime": "tiled_multiscale"},
            {"id": 2, "file_name": "s.png", "width": 120, "height": 120, "regime": "stain_1024"},
        ],
        "categories": [{"id": 1, "name": "c0"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "bbox": [5, 5, 15, 15],
             "area": 225, "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]]},
            {"id": 2, "image_id": 2, "category_id": 1, "iscrowd": 0, "bbox": [5, 5, 15, 15],
             "area": 225, "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]]},
        ],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(ann))
    return CocoSegmentationDataset(str(root), split="train", layout="coco")


def test_tiling_config_has_regime_tile_sizes_default_none():
    tc = TilingConfig()
    assert tc.regime_tile_sizes is None


def test_regime_tile_sizes_picks_per_regime(tmp_path):
    ds = _coco_two_regimes(tmp_path)
    grouped = build_tile_index(
        ds, tile_sizes=[40], overlap=0.0, negative_ratio=0.0,
        min_annotation_overlap=0.05, seed=1,
        regime_tile_sizes={"tiled_multiscale": [40, 90], "stain_1024": [90]},
    )
    tiled_sides = {max(b[2], b[3]) for (b, _ab) in grouped[0]}      # img id 1
    stain_sides = {max(b[2], b[3]) for (b, _ab) in grouped[1]}      # img id 2
    assert any(s <= 40 for s in tiled_sides) and any(s > 40 for s in tiled_sides)  # both 40 & 90
    assert all(s > 40 for s in stain_sides)                          # only 90-px tiles


def test_regime_tile_sizes_falls_back_to_flat(tmp_path):
    ds = _coco_two_regimes(tmp_path)
    grouped = build_tile_index(
        ds, tile_sizes=[40], overlap=0.0, negative_ratio=0.0,
        min_annotation_overlap=0.05, seed=1,
        regime_tile_sizes={"stain_1024": [90]},   # tiled_multiscale unlisted -> flat [40]
    )
    tiled_sides = {max(b[2], b[3]) for (b, _ab) in grouped[0]}
    assert tiled_sides == {40}
