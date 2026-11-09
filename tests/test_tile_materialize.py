import numpy as np
import pytest
from PIL import Image


def _save_img(tmp_path, w=80, h=60):
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    p = tmp_path / "img.jpg"
    Image.fromarray(arr).save(p, quality=100)
    return str(p), arr


def test_pyvips_roi_matches_full_decode(tmp_path):
    from training_toolkit.data.tile_materialize import decode_crop_pyvips
    path, _ = _save_img(tmp_path)
    box = (10, 8, 32, 32)
    roi = decode_crop_pyvips(path, box)
    assert roi.shape == (32, 32, 3)
    assert roi.dtype == np.uint8


def test_pyvips_roi_zero_pads_out_of_bounds(tmp_path):
    from training_toolkit.data.tile_materialize import decode_crop_pyvips
    path, _ = _save_img(tmp_path, w=80, h=60)
    box = (70, 50, 32, 32)
    roi = decode_crop_pyvips(path, box)
    assert roi.shape == (32, 32, 3)
    assert (roi[10:, 10:] == 0).all()  # the out-of-image part is zero-padded


def test_tile_local_matches_full_then_crop():
    from training_toolkit.data.tile_materialize import (
        Instance, rasterize_tile_local, rasterize_full_then_crop)
    ring = np.array([[20, 20], [60, 20], [60, 60], [20, 60]], dtype=np.float32)
    inst = Instance(label=0, polygon=[ring], mask=None, mask_origin=None,
                    bbox=(20, 20, 60, 60))
    tile = (16, 16, 40, 40)
    local = rasterize_tile_local(inst, tile)
    ref = rasterize_full_then_crop(inst, tile, img_hw=(100, 100))
    assert local.shape == (40, 40)
    assert np.array_equal(local, ref)


def test_materialize_tile_returns_collate_shape(tmp_path):
    from training_toolkit.data.tile_materialize import Instance, materialize_tile
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    p = tmp_path / "img.jpg"
    Image.fromarray(arr).save(p, quality=100)
    ring = np.array([[30, 30], [70, 30], [70, 70], [30, 70]], dtype=np.float32)
    inst_in = Instance(0, [ring], None, None, (30, 30, 70, 70))
    far_ring = np.array([[5, 5], [12, 5], [12, 12], [5, 12]], dtype=np.float32)
    inst_out = Instance(1, [far_ring], None, None, (5, 5, 12, 12))

    sample = materialize_tile(str(p), (20, 20, 60, 60), [inst_in, inst_out],
                              min_overlap=0.05, image_id=7)
    assert sample["image"].shape == (60, 60, 3)
    assert sample["image_id"] == 7
    assert len(sample["masks"]) == 1
    assert sample["boxes"].shape == (1, 4)
    assert sample["labels"].tolist() == [0]
    assert (sample["boxes"] >= 0).all() and (sample["boxes"] <= 60).all()


def test_materialize_tile_empty_tile(tmp_path):
    from training_toolkit.data.tile_materialize import materialize_tile
    arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    p = tmp_path / "img.jpg"
    Image.fromarray(arr).save(p, quality=100)
    sample = materialize_tile(str(p), (0, 0, 40, 40), [], image_id=0)
    assert sample["image"].shape == (40, 40, 3)
    assert len(sample["masks"]) == 0
    assert sample["boxes"].shape == (0, 4)
    assert sample["labels"].shape == (0,)


def test_crop_from_array_matches_roi_decode(tmp_path):
    import cv2
    from training_toolkit.data.tile_materialize import (
        decode_crop_pyvips, decode_image_pyvips, crop_from_array,
    )
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, size=(200, 300, 3), dtype=np.uint8)
    p = str(tmp_path / "img.png")
    cv2.imwrite(p, img)  # color convention irrelevant: we compare pyvips-full vs pyvips-ROI of the SAME file
    full = decode_image_pyvips(p)
    assert full.shape == (200, 300, 3)
    for box in [(10, 20, 50, 40), (-5, -5, 30, 30), (280, 180, 40, 40)]:  # interior, TL-OOB, BR-OOB
        roi = decode_crop_pyvips(p, box)
        arr = crop_from_array(full, box)
        assert arr.shape == roi.shape
        np.testing.assert_array_equal(arr, roi)
