"""Pre-train EXIF/declared-dim sanity guard."""
from pathlib import Path

from PIL import Image, ImageDraw

from training_toolkit.configs.data_config import DataConfig
from training_toolkit.data.dim_sanity import (
    assert_declared_dims,
    check_declared_dims,
    records_from_dataset,
)

import pytest


class _FakeCoco:
    def __init__(self, imgs):
        self._imgs = {i["id"]: i for i in imgs}

    def getImgIds(self):
        return list(self._imgs)

    def loadImgs(self, iid):
        return [self._imgs[iid]]


class _FakeCocoDS:
    def __init__(self, imgs, img_dir):
        self.coco = _FakeCoco(imgs)
        self._image_dir = Path(img_dir)
        self.image_ids = [i["id"] for i in imgs]


class _FakeTiled:
    def __init__(self, base):
        self._base = base  # tiled datasets expose the base COCO dataset as ._base


def _oriented(path, w, h, orientation):
    img = Image.new("RGB", (w, h), "white")
    ImageDraw.Draw(img).rectangle([0, 0, w // 3, h // 3], fill=(255, 0, 0))
    exif = Image.Exif()
    exif[0x0112] = orientation
    img.save(path, "JPEG", exif=exif, quality=95)


def test_check_flags_rotating_and_mismatch(tmp_path):
    ok = tmp_path / "ok.jpg"
    Image.new("RGB", (32, 48)).save(ok, "JPEG")
    rot = tmp_path / "rot.jpg"
    _oriented(rot, 64, 48, 8)
    items = [(str(ok), 32, 48), (str(rot), 64, 48), (str(ok), 99, 99)]
    bad = {p for p, _ in check_declared_dims(items, workers=2)}
    assert str(ok) in bad  # 99x99 mismatch
    assert str(rot) in bad  # rotating tag


def test_check_clean_passes(tmp_path):
    ok = tmp_path / "ok.jpg"
    Image.new("RGB", (32, 48)).save(ok, "JPEG")
    assert check_declared_dims([(str(ok), 32, 48)], workers=2) == []


def test_assert_raises_on_violation(tmp_path):
    rot = tmp_path / "rot.jpg"
    _oriented(rot, 64, 48, 8)
    with pytest.raises(ValueError):
        assert_declared_dims([(str(rot), 64, 48)], workers=2)


def test_dataconfig_flag_defaults_on():
    assert DataConfig().dim_sanity_check is True


def test_records_from_dataset_direct_and_wrapped():
    imgs = [
        {"id": 1, "file_name": "a.jpg", "width": 10, "height": 20},
        {"id": 2, "file_name": "b.jpg", "width": 30, "height": 40},
    ]
    ds = _FakeCocoDS(imgs, "/data/imgs")
    recs = records_from_dataset(ds)
    assert ("/data/imgs/a.jpg", 10, 20) in recs and ("/data/imgs/b.jpg", 30, 40) in recs
    # a tiled wrapper is unwrapped via ._base
    assert records_from_dataset(_FakeTiled(ds)) == recs
    # a dataset with no COCO index (e.g. streaming) returns [] rather than raising
    assert records_from_dataset(object()) == []
