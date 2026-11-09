"""Whole-image branch for giants on grid regimes."""
from __future__ import annotations

import json

import cv2
import numpy as np

from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.tile_index import build_tile_index, grid_windows


def test_grid_windows_emits_whole_image_for_giant():
    boxes = np.array([[0, 0, 90, 90]], np.float32)   # sqrt-area 90 > 0.75*100 = 75
    wins = grid_windows(120, 120, boxes, tile_sizes=[100], overlap=0.0,
                        whole_image_if_giant=True, giant_ceiling_frac=0.75)
    assert ((0, 0, 120, 120), True) in wins


def test_grid_windows_no_whole_image_when_all_small():
    boxes = np.array([[0, 0, 30, 30]], np.float32)   # sqrt-area 30 < 75
    wins = grid_windows(120, 120, boxes, tile_sizes=[100], overlap=0.0,
                        whole_image_if_giant=True, giant_ceiling_frac=0.75)
    assert ((0, 0, 120, 120), True) not in wins


def test_grid_windows_disabled_by_default():
    boxes = np.array([[0, 0, 90, 90]], np.float32)
    wins = grid_windows(120, 120, boxes, tile_sizes=[100], overlap=0.0)  # default off
    assert ((0, 0, 120, 120), True) not in wins


def _coco(tmp_path, regime, defect_side):
    root = tmp_path / "ds"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "train" / "a.png"), np.zeros((300, 300, 3), np.uint8))
    d = defect_side
    ann = {
        "images": [{"id": 1, "file_name": "a.png", "width": 300, "height": 300, "regime": regime}],
        "categories": [{"id": 1, "name": "c0"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "bbox": [10, 10, d, d], "area": d * d,
             "segmentation": [[10, 10, 10 + d, 10, 10 + d, 10 + d, 10, 10 + d]]},
        ],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(ann))
    return CocoSegmentationDataset(str(root), split="train", layout="coco")


def test_dispatch_whole_image_fires_for_both_grid_regimes(tmp_path):
    for regime in ("tiled_multiscale", "stain_1024"):
        ds = _coco(tmp_path / regime, regime, defect_side=200)   # 200 > 0.75*100 = 75
        grouped = build_tile_index(
            ds, tile_sizes=[100], overlap=0.0, negative_ratio=0.0,
            min_annotation_overlap=0.05, seed=1,
            regime_tile_sizes={"tiled_multiscale": [100], "stain_1024": [100]},
        )
        boxes = [b for (b, _ab) in grouped[0]]
        assert (0, 0, 300, 300) in boxes, f"{regime} missing whole-image window"


def test_dispatch_no_whole_image_for_plain_outdoor(tmp_path):
    ds = _coco(tmp_path, "outdoor_tiled", defect_side=200)
    grouped = build_tile_index(
        ds, tile_sizes=[100], overlap=0.0, negative_ratio=0.0,
        min_annotation_overlap=0.05, seed=1,
    )
    boxes = [b for (b, _ab) in grouped[0]]
    assert (0, 0, 300, 300) not in boxes   # outdoor_tiled is not a giant regime
