"""Tests for FiftyOneSegmentationDataset class auto-discovery.

Covers both label-field types. The Polylines case is a regression test for a bug
where class auto-discovery hardcoded the ``.detections.label`` subpath, which
returns ``{None: N}`` for a Polylines field — yielding a single bogus class and
silently dropping every annotation downstream (0 tiles).
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

fo = pytest.importorskip("fiftyone")

from training_toolkit.data.fiftyone_dataset import FiftyOneSegmentationDataset
from training_toolkit.data.tiled_dataset import TiledSegmentationDataset


def _temp_dataset_with(label_field: str, container):
    ds = fo.Dataset()  # unnamed, non-persistent temp dataset
    sample = fo.Sample(filepath="/tmp/parity_fake_image.jpg")
    sample["metadata"] = fo.ImageMetadata(width=100, height=100)
    sample[label_field] = container
    ds.add_sample(sample)
    return ds


def test_polylines_class_discovery_and_boxes():
    """Regression: classes must be discovered from the Polylines subpath, and
    annotations must survive into _load_tile_metadata."""
    polys = fo.Polylines(
        polylines=[
            fo.Polyline(label="crack",
                        points=[[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]],
                        filled=True, closed=True),
            fo.Polyline(label="spall",
                        points=[[[0.5, 0.5], [0.7, 0.5], [0.7, 0.7]]],
                        filled=True, closed=True),
        ]
    )
    ds = _temp_dataset_with("gt", polys)
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        assert base.num_classes == 2
        assert set(base.class_names.values()) == {"crack", "spall"}
        _, _, boxes = base._load_tile_metadata(0)
        assert boxes.shape == (2, 4)
    finally:
        ds.delete()


def test_detections_class_discovery_and_boxes():
    """Guard: the Detections path keeps working after the fix."""
    dets = fo.Detections(
        detections=[
            fo.Detection(label="crack", bounding_box=[0.1, 0.1, 0.2, 0.2]),
            fo.Detection(label="spall", bounding_box=[0.5, 0.5, 0.2, 0.2]),
        ]
    )
    ds = _temp_dataset_with("gt", dets)
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        assert base.num_classes == 2
        assert set(base.class_names.values()) == {"crack", "spall"}
        _, _, boxes = base._load_tile_metadata(0)
        assert boxes.shape == (2, 4)
    finally:
        ds.delete()


def _dets_with_mask():
    m = np.zeros((20, 20), dtype=bool)
    m[5:15, 5:15] = True  # 10x10 blob inside the 20x20 bbox
    return fo.Detections(detections=[
        fo.Detection(label="crack", bounding_box=[0.1, 0.1, 0.2, 0.2], mask=m),
    ])


def test_image_path_returns_filepath():
    dets = fo.Detections(detections=[
        fo.Detection(label="crack", bounding_box=[0.1, 0.1, 0.2, 0.2])])
    ds = _temp_dataset_with("gt", dets)
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        assert base._image_path(0) == "/tmp/parity_fake_image.jpg"
    finally:
        ds.delete()


def test_load_annotations_polylines():
    polys = fo.Polylines(polylines=[
        fo.Polyline(label="crack",
                    points=[[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3]]],
                    filled=True, closed=True)])
    ds = _temp_dataset_with("gt", polys)
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        anns = base._load_annotations(0)
        assert len(anns) == 1
        inst = anns[0]
        assert inst.polygon is not None and inst.mask is None
        assert inst.label == base._label_to_idx["crack"]
        np.testing.assert_allclose(
            inst.polygon[0], [[10, 10], [30, 10], [30, 30]], atol=1e-3)
        assert inst.bbox == (10.0, 10.0, 30.0, 30.0)
    finally:
        ds.delete()


def test_load_annotations_detections_with_mask():
    ds = _temp_dataset_with("gt", _dets_with_mask())
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        anns = base._load_annotations(0)
        assert len(anns) == 1
        inst = anns[0]
        assert inst.mask is not None and inst.polygon is None
        assert inst.mask_origin == (10, 10)
        assert inst.mask.shape == (20, 20)
        # tight bbox from the 10x10 blob at mask-local [5:15,5:15] -> abs [15,15,24,24]
        assert inst.bbox == (15.0, 15.0, 24.0, 24.0)
    finally:
        ds.delete()


def test_load_annotations_detections_no_mask():
    dets = fo.Detections(detections=[
        fo.Detection(label="crack", bounding_box=[0.1, 0.1, 0.2, 0.2])])
    ds = _temp_dataset_with("gt", dets)
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        inst = base._load_annotations(0)[0]
        assert inst.mask is not None and inst.mask.shape == (20, 20)
        assert int(inst.mask.min()) == 1  # filled rectangle
        assert inst.bbox == (10.0, 10.0, 30.0, 30.0)
    finally:
        ds.delete()


def test_fiftyone_offline_tiling_end_to_end(tmp_path):
    """Regression: TiledSegmentationDataset over a FiftyOne base used to raise
    NotImplementedError in _load_sample; now it materializes a tile via pyvips ROI."""
    img_path = tmp_path / "img.png"
    Image.fromarray(np.full((64, 64, 3), 127, np.uint8)).save(img_path)
    polys = fo.Polylines(polylines=[
        fo.Polyline(label="crack",
                    points=[[[0.1, 0.1], [0.4, 0.1], [0.4, 0.4]]],
                    filled=True, closed=True)])
    ds = fo.Dataset()
    sample = fo.Sample(filepath=str(img_path))
    sample["metadata"] = fo.ImageMetadata(width=64, height=64)
    sample["gt"] = polys
    ds.add_sample(sample)
    try:
        base = FiftyOneSegmentationDataset(ds, label_field="gt")
        tiled = TiledSegmentationDataset(
            base, tile_size=32, overlap=0.0, negative_ratio=0.0, seed=0)
        assert len(tiled) >= 1
        out = tiled._load_sample(0)
        assert out["image"].shape == (32, 32, 3)
        assert out["image"].dtype == np.uint8
        total_masks = sum(len(tiled._load_sample(i)["masks"]) for i in range(len(tiled)))
        assert total_masks >= 1
    finally:
        ds.delete()
