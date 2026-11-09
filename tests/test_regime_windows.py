"""Pure window fns + 4-way regime dispatch in build_tile_index."""
from __future__ import annotations

import json

import cv2
import numpy as np

from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.tile_index import (
    build_tile_index,
    grid_windows,
    whole_frame_windows,
)


def _coco(tmp_path, regime):
    """One 80x60 image tagged `regime`, with a single annotation."""
    root = tmp_path / "ds"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    cv2.imwrite(str(root / "images" / "train" / "a.png"), np.zeros((60, 80, 3), np.uint8))
    ann = {
        "images": [{"id": 1, "file_name": "a.png", "width": 80, "height": 60, "regime": regime}],
        "categories": [{"id": 1, "name": "c0"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "bbox": [5, 5, 15, 15], "area": 225,
             "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]]},
        ],
    }
    (root / "annotations" / "instances_train.json").write_text(json.dumps(ann))
    return CocoSegmentationDataset(str(root), split="train", layout="coco")


def test_whole_frame_windows_one_window_no_band():
    boxes = np.array([[5, 5, 20, 20]], np.float32)
    assert whole_frame_windows(80, 60, boxes) == [((0, 0, 80, 60), False)]


def test_whole_frame_windows_empty_when_no_boxes():
    assert whole_frame_windows(80, 60, np.zeros((0, 4), np.float32)) == []


def test_grid_windows_all_banded_dual_scale():
    wins = grid_windows(80, 60, np.zeros((0, 4), np.float32), tile_sizes=[16, 48], overlap=0.0)
    assert all(ab is True for (_b, ab) in wins)
    sides = {max(b[2], b[3]) for (b, _ab) in wins}
    assert any(s <= 16 for s in sides) and any(s > 16 for s in sides)


def test_dispatch_whole_frame_regime(tmp_path):
    ds = _coco(tmp_path, "whole_frame")
    grouped = build_tile_index(ds, tile_sizes=[16], overlap=0.0,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=1)
    assert grouped[0] == [((0, 0, 80, 60), False)]


def test_dispatch_tiled_multiscale_routes_to_grid(tmp_path):
    ds = _coco(tmp_path, "tiled_multiscale")
    grouped = build_tile_index(ds, tile_sizes=[16, 48], overlap=0.0,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=1)
    assert all(ab is True for (_b, ab) in grouped[0])
    sides = {max(b[2], b[3]) for (b, _ab) in grouped[0]}
    assert any(s <= 16 for s in sides) and any(s > 16 for s in sides)


def test_dispatch_legacy_indoor_whole_still_whole_frame(tmp_path):
    ds = _coco(tmp_path, "indoor_whole")
    grouped = build_tile_index(ds, tile_sizes=[16], overlap=0.0,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=1)
    assert grouped[0] == [((0, 0, 80, 60), False)]
