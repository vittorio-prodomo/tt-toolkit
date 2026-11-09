"""Offline TiledSegmentationDataset now decodes each tile via pyvips ROI +
tile-local rasterization (materialize_tile), not cv2-full-decode-then-numpy-crop.
Verifies pixel parity, that the full-image decoder is never called, mask
alignment, empty tiles, and the EXIF 'double win' (val now agrees with the pyvips
train decoder, so the dim-mismatch guard stops firing on EXIF-rotated images)."""
import json

import cv2
import numpy as np
import pytest
from PIL import Image

from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.tiled_dataset import TiledSegmentationDataset


def _make_coco(tmp_path, w, h, anns, fname="a.png", exif_orientation=None):
    """Write one image (deterministic gradient pixels) + a COCO val JSON declaring
    (w, h). If exif_orientation is set, tag the JPEG so cv2 auto-rotates but pyvips
    does not. Returns (root, stored_rgb_array)."""
    img_dir = tmp_path / "images" / "val"
    img_dir.mkdir(parents=True)
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    img = Image.fromarray(arr)
    if exif_orientation is not None:
        exif = img.getexif()
        exif[0x0112] = exif_orientation  # 0x0112 = EXIF Orientation tag
        img.save(img_dir / fname, exif=exif)
    else:
        img.save(img_dir / fname)
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    coco = {
        "images": [{"id": 1, "file_name": fname, "width": w, "height": h}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [x, y, bw, bh], "area": bw * bh,
             "segmentation": [[x, y, x + bw, y, x + bw, y + bh, x, y + bh]]}
            for i, (x, y, bw, bh) in enumerate(anns)
        ],
        "categories": [{"id": 0, "name": "crack"}],
    }
    (ann_dir / "instances_val.json").write_text(json.dumps(coco))
    return tmp_path, arr


def _tiled(root, tile_size=32, negative_ratio=0.0):
    base = CocoSegmentationDataset(root=root, split="val", layout="coco")
    return TiledSegmentationDataset(
        base, tile_size=tile_size, overlap=0.0, negative_ratio=negative_ratio,
        min_annotation_overlap=0.05, seed=0,
    )


def test_val_no_full_image_decode(tmp_path):
    """The base's full-image _load_sample (cv2 full decode) is NEVER called."""
    root, _ = _make_coco(tmp_path, w=64, h=64, anns=[(5, 5, 20, 20)])
    ds = _tiled(root)

    def _boom(*a, **k):
        pytest.fail("base._load_sample (full cv2 decode) was called by the val path")

    ds._base._load_sample = _boom
    out = ds._load_sample(0)
    assert out["image"].ndim == 3 and out["image"].shape[2] == 3


def test_val_tile_pixels_match_full_then_crop(tmp_path):
    """Every pyvips ROI tile equals cv2-full-decode-then-crop on the same box.
    Two annotations keep positive tiles at (0,0) AND (32,32) so a non-zero-origin
    ROI crop (an x/y swap would slip through a (0,0)-only test) is exercised."""
    root, _ = _make_coco(tmp_path, w=64, h=64, anns=[(5, 5, 20, 20), (37, 37, 20, 20)])
    ds = _tiled(root)
    full_rgb = cv2.cvtColor(
        cv2.imread(str(root / "images" / "val" / "a.png")), cv2.COLOR_BGR2RGB)
    assert len(ds) >= 2
    for i in range(len(ds)):
        _, tx, ty, tw, th, _ = ds._tiles[i]
        ref = full_rgb[ty:ty + th, tx:tx + tw]
        got = ds._load_sample(i)["image"]
        assert got.shape == ref.shape
        assert np.array_equal(got, ref), f"tile {i} pixels differ from cv2 crop"


def test_val_tile_masks_aligned(tmp_path):
    """A polygon inside a tile yields a tile-local mask matching the oracle."""
    from training_toolkit.data.tile_materialize import (
        Instance, rasterize_full_then_crop)
    root, _ = _make_coco(tmp_path, w=64, h=64, anns=[(10, 10, 16, 16)])
    ds = _tiled(root)
    found = False
    for i in range(len(ds)):
        _, tx, ty, tw, th, _ = ds._tiles[i]
        if (tx, ty) != (0, 0):
            continue
        s = ds._load_sample(i)
        assert len(s["masks"]) == 1
        ring = np.array([[10, 10], [26, 10], [26, 26], [10, 26]], dtype=np.float32)
        inst = Instance(0, [ring], None, None, (10, 10, 26, 26))
        ref = rasterize_full_then_crop(inst, (tx, ty, tw, th), img_hw=(64, 64))
        assert np.array_equal(s["masks"][0], ref)
        found = True
    assert found, "expected a tile at (0,0) containing the polygon"


def test_val_uniform_shape_and_empty_negative(tmp_path):
    """All tiles are (32,32,3); at least one is a negative with empty labels."""
    root, _ = _make_coco(tmp_path, w=64, h=64, anns=[(2, 2, 8, 8)])
    ds = _tiled(root, negative_ratio=0.5)
    samples = [ds._load_sample(i) for i in range(len(ds))]
    assert all(s["image"].shape == (32, 32, 3) for s in samples)
    assert all(s["image"].dtype == np.uint8 for s in samples)
    assert any(len(s["labels"]) == 0 for s in samples)


def test_val_exif_double_win_guard_does_not_fire(tmp_path):
    """An EXIF-rotated image whose COCO dims were recorded raw: pyvips agrees with
    declared (guard does NOT fire, annotations kept) — whereas cv2 auto-rotates and
    the OLD path neutralized it. arr is (H=32, W=64) tagged orientation=6."""
    root, _ = _make_coco(
        tmp_path, w=64, h=32, anns=[(5, 5, 20, 20), (40, 5, 18, 20)],
        fname="rot.jpg", exif_orientation=6)
    path = str(root / "images" / "val" / "rot.jpg")
    # Precondition: cv2 honors EXIF → returns the rotated (64,32)-shaped array
    # (un-rotated would be (32,64)). If this fails, the fixture didn't take.
    assert cv2.imread(path).shape[:2] == (64, 32)
    ds = _tiled(root)
    total = 0
    for i in range(len(ds)):
        total += len(ds._load_sample(i)["boxes"])
    assert 0 not in ds._dim_mismatch_warned, "guard fired on a consistent EXIF image"
    assert total > 0, "annotations were dropped on an EXIF image (double win failed)"
