"""Offline TiledSegmentationDataset must not emit empty crops (or misaligned masks)
when an image's DECODED dims disagree with its declared COCO dims.

The val tiler now decodes via pyvips ROI (raw pixels, no EXIF auto-rotation). The
dim-mismatch guard compares the pyvips-decoded dims against the declared COCO
width/height and neutralizes an image's tiles when they disagree — i.e. genuinely
corrupt metadata, where annotation-space masks would misalign with the raw pixels.
(The original crash was the EXIF variant of this; that case now
*agrees* under pyvips and is covered in test_tiled_val_pyvips.py.)

Fixture mirrors a clean transpose: declared 64x32, actual file 32x64. The grid for
the 64-wide declared dims places a tile at x=32, off the right edge of the 32-wide
decoded image.
"""
import json

import numpy as np
import pytest
from PIL import Image

from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.tiled_dataset import TiledSegmentationDataset


def _make_coco(tmp_path, declared_w, declared_h, actual_w, actual_h, anns):
    img_dir = tmp_path / "images" / "val"
    img_dir.mkdir(parents=True)
    arr = np.random.randint(0, 255, (actual_h, actual_w, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "a.jpg")

    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": declared_w, "height": declared_h}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [x, y, w, h], "area": w * h,
             "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]}
            for i, (x, y, w, h) in enumerate(anns)
        ],
        "categories": [{"id": 0, "name": "crack"}],
    }
    (ann_dir / "instances_val.json").write_text(json.dumps(coco))
    return tmp_path


def _tiled(root):
    base = CocoSegmentationDataset(root=root, split="val", layout="coco")
    return TiledSegmentationDataset(
        base, tile_size=32, overlap=0.0, negative_ratio=0.0,
        min_annotation_overlap=0.05, seed=0,
    )


# Two annotations so BOTH grid tiles (x=0 in-bounds, x=32 out-of-bounds) are positive
# and therefore kept in the index (negative_ratio=0).
_TRANSPOSED = dict(declared_w=64, declared_h=32, actual_w=32, actual_h=64,
                   anns=[(5, 5, 20, 15), (40, 5, 20, 15)])


def test_transposed_image_yields_no_empty_crops(tmp_path):
    """Every tile of a dim-mismatched image is a non-empty crop (no cv2.resize crash)."""
    ds = _tiled(_make_coco(tmp_path, **_TRANSPOSED))
    assert len(ds) >= 2
    for i in range(len(ds)):
        img = ds._load_sample(i)["image"]
        assert img.shape[0] > 0 and img.shape[1] > 0, \
            f"tile {i} produced an empty crop {img.shape}"


def test_transposed_image_drops_annotations(tmp_path):
    """A dim-mismatched image is untrustworthy (image/masks in different coordinate
    spaces), so every tile is neutralized with no annotations."""
    ds = _tiled(_make_coco(tmp_path, **_TRANSPOSED))
    for i in range(len(ds)):
        s = ds._load_sample(i)
        assert len(s["boxes"]) == 0 and len(s["labels"]) == 0 and len(s["masks"]) == 0, \
            f"tile {i} kept annotations from a dim-mismatched image"


def test_matching_dims_preserves_annotations(tmp_path):
    """Regression: when declared dims match the actual image, annotations and crops
    are unchanged (no neutralization)."""
    ds = _tiled(_make_coco(tmp_path, declared_w=32, declared_h=64,
                           actual_w=32, actual_h=64, anns=[(5, 5, 20, 15)]))
    total = sum(len(ds._load_sample(i)["boxes"]) for i in range(len(ds)))
    assert total > 0
