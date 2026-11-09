"""End-to-end streaming smoke across all four tiling regimes."""
from __future__ import annotations

import json

import cv2
import numpy as np

from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.label_crop import SizeBand
from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset


def _multi_regime_dataset(tmp_path):
    """One image per regime, each annotated + carrying a class_mask. Image SIZES are
    chosen so each regime yields a uniquely-shaped sample (107 gt_guided chip / 128
    grid tile / 160x200 whole_frame full frame / 256 giant whole-image), letting the
    smoke verify every regime contributed -- not just a total count."""
    root = tmp_path / "ds"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    # (fname, regime, (W, H), bbox_xyxy)
    names = {
        1: ("tiled.png", "tiled_multiscale", (256, 256), [40, 40, 200, 200]),  # sqrt-area 160 > 0.75*128=96 -> giant
        2: ("gtg.png", "gt_guided", (256, 256), [40, 40, 90, 90]),             # 50px defect -> chip 50*128/60~=107 (unique)
        3: ("whole.png", "whole_frame", (200, 160), [20, 20, 60, 60]),         # band-free full frame -> 160x200 sample (unique)
        4: ("stain.png", "stain_1024", (256, 256), [30, 30, 220, 220]),        # sqrt-area 190 > 96 -> giant
    }
    images, annotations = [], []
    for img_id, (fname, regime, (w, h), (x1, y1, x2, y2)) in names.items():
        cv2.imwrite(str(root / "images" / "train" / fname), np.zeros((h, w, 3), np.uint8))
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h,
                       "regime": regime, "class_mask": [1.0, 1.0, 1.0]})
        annotations.append({
            "id": img_id, "image_id": img_id, "category_id": 1, "iscrowd": 0,
            "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
            "segmentation": [[x1, y1, x2, y1, x2, y2, x1, y2]],
        })
    ann = {"images": images,
           "categories": [{"id": 1, "name": "c0"}, {"id": 2, "name": "c1"}, {"id": 3, "name": "c2"}],
           "annotations": annotations}
    (root / "annotations" / "instances_train.json").write_text(json.dumps(ann))
    return CocoSegmentationDataset(str(root), split="train", layout="coco")


def test_streaming_all_four_regimes_yield_labeled_samples(tmp_path):
    ds = _multi_regime_dataset(tmp_path)
    stream = StreamingTiledSegmentationDataset(
        ds, tile_size=128, tile_sizes=[128],
        regime_tile_sizes={"tiled_multiscale": [128], "stain_1024": [128]},
        gt_guided_params={"target_in_model": 60.0, "min_crop": 64,
                          "translation_jitter": 0.0, "scale_jitter": 0.0, "dedup_iou": 0.6},
        giant_ceiling_frac=0.75, overlap=0.0, negative_ratio=0.0,
        buffer_size=32, imgsz=128, size_band=SizeBand.for_imgsz(128),
    )
    samples = list(stream)

    # every sample is collate-ready
    assert len(samples) >= 4
    for s in samples:
        assert "ignore" in s and "class_mask" in s
        assert s["class_mask"] == [1.0, 1.0, 1.0]

    # each regime contributed a uniquely-shaped sample (shape is (H, W)):
    shapes = {s["image"].shape[:2] for s in samples}
    assert (107, 107) in shapes   # gt_guided per-defect chip (only gt_guided makes 107)
    assert (128, 128) in shapes   # 128 grid tile (tiled_multiscale / stain_1024 grid)
    assert (160, 200) in shapes   # whole_frame band-free full frame (200w x 160h, unique)
    assert (256, 256) in shapes   # whole-image branch full frame for a giant (tiled_multiscale + stain_1024)

    # the band fires somewhere (>=1 out-of-band ignore on the tiled grid)
    assert any(s["ignore"].any() for s in samples if len(s["ignore"]) > 0)
