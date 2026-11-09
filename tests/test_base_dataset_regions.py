"""Region-class-mask geometry tests for BaseSegmentationDataset."""

from __future__ import annotations

import numpy as np
import pytest

from training_toolkit.configs import AugConfig
from training_toolkit.data.base_dataset import BaseSegmentationDataset
from training_toolkit.data.multi_sample import MosaicTransform
from training_toolkit.data.transforms import build_transforms


class _DummyDataset(BaseSegmentationDataset):
    def __len__(self):
        return 1

    def _load_sample(self, idx):
        raise NotImplementedError

    @property
    def class_names(self):
        return {0: "a", 1: "b", 2: "c"}

    @property
    def num_classes(self):
        return 3


def _empty_sample(value: int, class_mask: list[float]) -> dict:
    return {
        "image": np.full((12, 12, 3), value, dtype=np.uint8),
        "masks": [],
        "boxes": np.zeros((0, 4), dtype=np.float32),
        "labels": np.zeros((0,), dtype=np.int64),
        "class_mask": np.asarray(class_mask, dtype=np.float32),
    }


def _normalized_half_open_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape
    return [
        float(xs.min()) / w,
        float(ys.min()) / h,
        float(xs.max() + 1) / w,
        float(ys.max() + 1) / h,
    ]


class _RecordingTransform:
    def __init__(self, transform):
        self.transform = transform
        self.result = None

    def __call__(self, **kwargs):
        self.result = self.transform(**kwargs)
        return self.result


def test_mosaic_regions_follow_real_crop_and_horizontal_flip():
    samples = [
        _empty_sample(value, mask)
        for value, mask in zip(
            (32, 96, 160, 224),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        )
    ]
    instance_mask = np.zeros((12, 12), dtype=np.uint8)
    instance_mask[2:10, 2:10] = 1
    samples[3]["masks"] = [instance_mask]
    samples[3]["boxes"] = np.array([[2.0, 2.0, 10.0, 10.0]], dtype=np.float32)
    samples[3]["labels"] = np.array([2], dtype=np.int64)
    mosaic = MosaicTransform(target_size=12, center_range=(0.25, 0.25))
    sample = mosaic(samples[0], samples[1:])
    original_region_class_masks = sample["region_class_masks"].copy()

    cfg = AugConfig(
        imgsz=12,
        mosaic_p=1.0,
        crop_scale_min=0.5,
        crop_scale_max=0.5,
        horizontal_flip=1.0,
    )
    _, mosaic_transform = build_transforms(cfg, is_train=True)
    assert [t.__class__.__name__ for t in mosaic_transform.transforms][:2] == [
        "RandomResizedCrop", "HorizontalFlip"
    ]
    mosaic_transform.set_random_seed(1)
    recording_transform = _RecordingTransform(mosaic_transform)

    result = _DummyDataset()._apply_transform(sample, recording_transform)

    assert recording_transform.result is not None
    transformed_masks = recording_transform.result["masks"]
    transformed_instance_masks = transformed_masks[:1]
    transformed_region_masks = transformed_masks[1:]
    assert len(transformed_masks) == 5
    assert transformed_instance_masks[0].sum() > 0
    assert len(result["masks"]) == 1
    np.testing.assert_array_equal(result["masks"][0], transformed_instance_masks[0])
    surviving = [i for i, mask in enumerate(transformed_region_masks) if mask.sum() > 0]
    assert surviving == [2, 3]
    expected_boxes = np.asarray(
        [_normalized_half_open_box(transformed_region_masks[i]) for i in surviving],
        dtype=np.float32,
    )
    expected_class_masks = original_region_class_masks[surviving]
    np.testing.assert_allclose(result["region_boxes"], expected_boxes)
    np.testing.assert_array_equal(result["region_class_masks"], expected_class_masks)
    assert "region_masks" not in result


def test_non_rectangular_transformed_region_fails_loudly():
    samples = [
        _empty_sample(value, mask)
        for value, mask in zip(
            (32, 96, 160, 224),
            ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
             [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
        )
    ]
    sample = MosaicTransform(
        target_size=12, center_range=(0.5, 0.5)
    )(samples[0], samples[1:])

    def punch_hole(image, masks, bboxes, labels):
        transformed_masks = [mask.copy() for mask in masks]
        transformed_masks[0][1, 1] = 0
        return {"image": image, "masks": transformed_masks}

    with pytest.raises(ValueError, match="non-rectangular"):
        _DummyDataset()._apply_transform(sample, punch_hole)
