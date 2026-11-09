# training-toolkit/tests/test_streaming_label_crop.py
from __future__ import annotations

import json

import cv2
import numpy as np

from training_toolkit.data.coco_dataset import CocoSegmentationDataset


def _tiny_coco(tmp_path, with_class_mask: bool):
    """Build a minimal COCO-layout dataset in tmp_path and return the dataset.

    Layout: tmp_path/images/train/a.png + tmp_path/annotations/instances_train.json
    (matches CocoSegmentationDataset's default 'coco' layout).
    """
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "a.png"), img)

    image = {"id": 1, "file_name": "a.png", "width": 40, "height": 40}
    if with_class_mask:
        image["class_mask"] = [1, 0, 1]
    coco = {
        "images": [image],
        "categories": [{"id": 1, "name": "c0"}, {"id": 2, "name": "c1"}, {"id": 3, "name": "c2"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1,
                         "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]],
                         "bbox": [5, 5, 15, 15], "area": 225, "iscrowd": 0}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(coco))
    return CocoSegmentationDataset(root=str(tmp_path), split="train", layout="coco")


def test_image_class_mask_returns_list_when_present(tmp_path):
    ds = _tiny_coco(tmp_path, with_class_mask=True)
    assert ds._image_class_mask(0) == [1, 0, 1]


def test_image_class_mask_none_when_absent(tmp_path):
    ds = _tiny_coco(tmp_path, with_class_mask=False)
    assert ds._image_class_mask(0) is None


from training_toolkit.data.label_crop import SizeBand
from training_toolkit.data.streaming_tiled_dataset import materialize_labeled_tile
from training_toolkit.data.tile_materialize import Instance


def _square_instance(label, x0, y0, side):
    ring = np.array([[x0, y0], [x0 + side, y0], [x0 + side, y0 + side], [x0, y0 + side]],
                    dtype=np.float32)
    return Instance(label=label, polygon=[ring], mask=None, mask_origin=None,
                    bbox=(float(x0), float(y0), float(x0 + side), float(y0 + side)))


def test_materialize_labeled_tile_attaches_ignore_and_class_mask(tmp_path):
    cv2.imwrite(str(tmp_path / "img.png"), np.full((100, 100, 3), 127, np.uint8))
    inst = [_square_instance(0, 10, 10, 40)]   # ~40px -> in-band at scale 1.0
    out = materialize_labeled_tile(
        str(tmp_path / "img.png"), (0, 0, 100, 100), inst, class_mask=[1, 1, 0],
        realized_scale=1.0, band=SizeBand(floor=24.0, ceiling=80.0), min_overlap=0.05,
        image_id=7,
    )
    assert out["image"].shape == (100, 100, 3)
    assert out["image_id"] == 7
    assert out["labels"].tolist() == [0]
    assert out["ignore"].tolist() == [False]
    assert out["class_mask"] == [1, 1, 0]


def test_materialize_labeled_tile_marks_subresolution_ignore(tmp_path):
    cv2.imwrite(str(tmp_path / "img.png"), np.full((100, 100, 3), 127, np.uint8))
    inst = [_square_instance(0, 10, 10, 18)]   # ~18px -> below floor 24 -> ignore
    out = materialize_labeled_tile(
        str(tmp_path / "img.png"), (0, 0, 100, 100), inst, class_mask=None,
        realized_scale=1.0, band=SizeBand(floor=24.0, ceiling=80.0), min_overlap=0.05, image_id=0,
    )
    assert out["ignore"].tolist() == [True]
    assert "class_mask" not in out   # None -> not attached


import torch
from training_toolkit.data.collate import collate_fn


def test_collate_forwards_ignore_and_class_mask():
    sample = {
        "image": np.zeros((8, 8, 3), np.uint8),
        "masks": [np.ones((8, 8), np.uint8)],
        "boxes": np.array([[1, 1, 5, 5]], np.float32),
        "labels": np.array([0], np.int64),
        "ignore": np.array([True], dtype=bool),
        "class_mask": [1, 0, 1],
    }
    out = collate_fn([sample])
    t = out["targets"][0]
    assert torch.equal(t["ignore"], torch.tensor([True]))
    assert t["ignore"].dtype == torch.bool
    assert torch.equal(t["class_mask"], torch.tensor([1.0, 0.0, 1.0]))


def test_collate_no_ignore_key_is_fine():
    sample = {"image": np.zeros((8, 8, 3), np.uint8), "masks": [],
              "boxes": np.zeros((0, 4), np.float32), "labels": np.zeros((0,), np.int64)}
    t = collate_fn([sample])["targets"][0]
    assert "ignore" not in t


from training_toolkit.configs.train_config import TrainConfig
from training_toolkit.models.rfdetr import RFDETRAdapter


def test_train_config_has_size_band_ignore_defaults():
    cfg = TrainConfig.__dataclass_fields__
    assert "size_band_ignore" in cfg
    assert "ignore_ioa_thresh" in cfg


def test_adapter_enable_size_band_ignore_sets_flag():
    a = RFDETRAdapter()
    assert a._size_band_ignore is False
    a.enable_size_band_ignore()
    assert a._size_band_ignore is True


def test_enable_size_band_ignore_sets_ioa_thresh():
    a = RFDETRAdapter()
    a.enable_size_band_ignore(0.3)
    assert a._size_band_ignore is True
    assert a._ignore_ioa_thresh == 0.3


def test_apply_transform_keeps_ignore_in_sync_with_masks(tmp_path):
    # Regression: _apply_transform filters masks/labels/boxes when a transform
    # zeros a mask; the per-instance `ignore` array must shrink in sync, else the
    # criterion's ignore mask index-selects out of bounds (IndexError) at train time.
    ds = _tiny_coco(tmp_path, with_class_mask=False)

    def drop_first(image, masks, bboxes, labels):
        return {"image": image, "masks": [np.zeros_like(masks[0])] + list(masks[1:])}

    sample = {
        "image": np.zeros((40, 40, 3), np.uint8),
        "masks": [np.ones((40, 40), np.uint8), np.ones((40, 40), np.uint8)],
        "boxes": np.array([[1, 1, 10, 10], [20, 20, 30, 30]], np.float32),
        "labels": np.array([0, 1], np.int64),
        "ignore": np.array([True, False], dtype=bool),
    }
    out = ds._apply_transform(sample, transform=drop_first)
    assert len(out["ignore"]) == len(out["labels"]) == len(out["masks"]) == 1
    assert out["ignore"].tolist() == [False]


import cv2
from training_toolkit.data.tile_materialize import Instance, decode_image_pyvips
from training_toolkit.data.streaming_tiled_dataset import (
    materialize_labeled_tile, materialize_labeled_tile_from_array,
)

def test_array_materialize_equals_roi(tmp_path):
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, size=(256, 256, 3), dtype=np.uint8)
    p = str(tmp_path / "i.png"); cv2.imwrite(p, img)
    poly = [np.array([[60, 60], [180, 70], [170, 190], [55, 175]], dtype=np.float32)]
    inst = Instance(label=2, polygon=poly, mask=None, mask_origin=None, bbox=(55, 60, 180, 190))
    box = (40, 40, 160, 160)
    roi = materialize_labeled_tile(p, box, [inst], None, 1.0, None, 0.05, 7)
    full = decode_image_pyvips(p)
    arr = materialize_labeled_tile_from_array(full, box, [inst], None, 1.0, None, 0.05, 7)
    np.testing.assert_array_equal(arr["image"], roi["image"])
    assert len(arr["masks"]) == len(roi["masks"]) == 1
    np.testing.assert_array_equal(arr["masks"][0], roi["masks"][0])
    np.testing.assert_array_equal(arr["boxes"], roi["boxes"])
    np.testing.assert_array_equal(arr["labels"], roi["labels"])
    assert arr["image_id"] == roi["image_id"] == 7
