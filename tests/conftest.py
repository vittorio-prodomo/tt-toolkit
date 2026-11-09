"""Shared test fixtures."""

import json
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def mini_coco(tmp_path):
    """Create a minimal COCO dataset for testing."""
    for split in ("train", "val"):
        img_dir = tmp_path / "images" / split
        img_dir.mkdir(parents=True)
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(img_dir / "img_001.jpg")

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()

    train_ann = {
        "images": [{"id": 1, "file_name": "img_001.jpg", "width": 64, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [10, 10, 20, 20], "area": 400,
             "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]},
            {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "bbox": [40, 40, 15, 15], "area": 225,
             "segmentation": [[40, 40, 55, 40, 55, 55, 40, 55]]},
        ],
        "categories": [{"id": 0, "name": "crack"}, {"id": 1, "name": "spall"}],
    }

    val_ann = {
        "images": [{"id": 2, "file_name": "img_001.jpg", "width": 64, "height": 64}],
        "annotations": [
            {"id": 3, "image_id": 2, "category_id": 0, "iscrowd": 0,
             "bbox": [5, 5, 30, 30], "area": 900,
             "segmentation": [[5, 5, 35, 5, 35, 35, 5, 35]]},
        ],
        "categories": [{"id": 0, "name": "crack"}, {"id": 1, "name": "spall"}],
    }

    with open(ann_dir / "instances_train.json", "w") as f:
        json.dump(train_ann, f)
    with open(ann_dir / "instances_val.json", "w") as f:
        json.dump(val_ann, f)

    return tmp_path


@pytest.fixture
def mini_roboflow(tmp_path):
    """Create a minimal dataset in Roboflow layout: <split>/ with images and _annotations.coco.json."""
    train_ann = {
        "images": [{"id": 1, "file_name": "img_001.jpg", "width": 64, "height": 64}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [10, 10, 20, 20], "area": 400,
             "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]},
            {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 0,
             "bbox": [40, 40, 15, 15], "area": 225,
             "segmentation": [[40, 40, 55, 40, 55, 55, 40, 55]]},
        ],
        "categories": [{"id": 0, "name": "crack"}, {"id": 1, "name": "spall"}],
    }
    val_ann = {
        "images": [{"id": 2, "file_name": "img_001.jpg", "width": 64, "height": 64}],
        "annotations": [
            {"id": 3, "image_id": 2, "category_id": 0, "iscrowd": 0,
             "bbox": [5, 5, 30, 30], "area": 900,
             "segmentation": [[5, 5, 35, 5, 35, 35, 5, 35]]},
        ],
        "categories": [{"id": 0, "name": "crack"}, {"id": 1, "name": "spall"}],
    }
    # Roboflow uses "valid" for the validation folder, not "val"
    for split_dir_name, ann in (("train", train_ann), ("valid", val_ann)):
        split_dir = tmp_path / split_dir_name
        split_dir.mkdir()
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
        img.save(split_dir / "img_001.jpg")
        with open(split_dir / "_annotations.coco.json", "w") as f:
            json.dump(ann, f)
    return tmp_path


@pytest.fixture
def rfdetr_adapter():
    """Adapter with only the postprocess path exercised (no weights loaded)."""
    pytest.importorskip("rfdetr")
    from training_toolkit.models.rfdetr import RFDETRAdapter
    from rfdetr.models.lwdetr import PostProcess
    a = RFDETRAdapter()
    a._postprocessor = PostProcess(num_select=300)
    return a
