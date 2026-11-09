"""Tests for COCO dataset and collate function."""

import numpy as np
import pytest
import torch

from training_toolkit.data import CocoSegmentationDataset, collate_fn


class TestCocoSegmentationDataset:
    """Tests for CocoSegmentationDataset."""

    def test_train_length(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        assert len(ds) == 1

    def test_val_length(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="val")
        assert len(ds) == 1

    def test_getitem_keys(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        sample = ds[0]
        assert "image" in sample
        assert "masks" in sample
        assert "boxes" in sample
        assert "labels" in sample
        assert "image_id" in sample

    def test_image_shape(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        sample = ds[0]
        assert sample["image"].shape == (64, 64, 3)
        assert sample["image"].dtype == np.uint8

    def test_num_masks_train(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        sample = ds[0]
        assert len(sample["masks"]) == 2

    def test_num_masks_val(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="val")
        sample = ds[0]
        assert len(sample["masks"]) == 1

    def test_masks_are_binary(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        sample = ds[0]
        for mask in sample["masks"]:
            unique_vals = set(np.unique(mask))
            assert unique_vals.issubset({0, 1}), f"Mask has non-binary values: {unique_vals}"

    def test_boxes_xyxy(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        sample = ds[0]
        boxes = sample["boxes"]
        assert boxes.shape == (2, 4)
        assert boxes.dtype == np.float32
        # x2 > x1 and y2 > y1
        assert np.all(boxes[:, 2] > boxes[:, 0])
        assert np.all(boxes[:, 3] > boxes[:, 1])

    def test_class_names(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        assert ds.class_names == {0: "crack", 1: "spall"}

    def test_num_classes(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        assert ds.num_classes == 2

    def test_roboflow_layout_loads(self, mini_roboflow):
        """Roboflow layout: <root>/<split>/_annotations.coco.json and images in <root>/<split>/."""
        ds_train = CocoSegmentationDataset(
            mini_roboflow, split="train", layout="roboflow"
        )
        ds_val = CocoSegmentationDataset(
            mini_roboflow, split="val", layout="roboflow"
        )
        assert len(ds_train) == 1
        assert len(ds_val) == 1
        sample = ds_train[0]
        assert "image" in sample and sample["image"].shape == (64, 64, 3)
        assert ds_train.class_names == {0: "crack", 1: "spall"}
        assert ds_train.num_classes == 2

    def test_invalid_layout_raises(self, mini_coco):
        with pytest.raises(ValueError, match="layout must be"):
            CocoSegmentationDataset(mini_coco, split="train", layout="invalid")


class TestCollateFn:
    """Tests for collate_fn."""

    def test_images_stacked(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        batch = [ds[0]]
        result = collate_fn(batch)
        assert isinstance(result["images"], torch.Tensor)
        assert result["images"].shape == (1, 3, 64, 64)
        assert result["images"].dtype == torch.float32
        assert result["images"].min() >= 0.0
        assert result["images"].max() <= 1.0

    def test_targets_are_list_of_dicts(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        batch = [ds[0]]
        result = collate_fn(batch)
        assert isinstance(result["targets"], list)
        assert len(result["targets"]) == 1
        target = result["targets"][0]
        assert isinstance(target, dict)
        assert "boxes" in target
        assert "labels" in target
        assert "masks" in target

    def test_collate_multiple_samples(self, mini_coco):
        ds_train = CocoSegmentationDataset(mini_coco, split="train")
        ds_val = CocoSegmentationDataset(mini_coco, split="val")
        batch = [ds_train[0], ds_val[0]]
        result = collate_fn(batch)
        assert result["images"].shape == (2, 3, 64, 64)
        assert len(result["targets"]) == 2
        # Train image has 2 annotations, val has 1
        assert result["targets"][0]["boxes"].shape[0] == 2
        assert result["targets"][1]["boxes"].shape[0] == 1

    def test_target_tensor_types(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        batch = [ds[0]]
        result = collate_fn(batch)
        target = result["targets"][0]
        assert target["boxes"].dtype == torch.float32
        assert target["labels"].dtype == torch.int64
        assert target["masks"].shape == (2, 64, 64)


class TestDatasetMultiSample:
    """Tests for multi-sample augmentation integration in the dataset."""

    def test_mosaic_produces_2x_canvas(self, mini_coco):
        from training_toolkit.configs import AugConfig

        # Skip the Albumentations post-transform (transform=None) so the
        # 2x mosaic canvas isn't resized back to imgsz — this test verifies
        # the mosaic step itself fires from config and produces 2*imgsz.
        cfg = AugConfig(imgsz=64, mosaic_p=1.0)
        ds = CocoSegmentationDataset(
            mini_coco, split="train", transform=None, aug_config=cfg,
        )
        sample = ds[0]
        # Mosaic at p=1.0 -> 2*imgsz canvas = 128x128 (pre-Albumentations)
        assert sample["image"].shape[:2] == (128, 128)

    def test_no_multi_sample_without_config(self, mini_coco):
        ds = CocoSegmentationDataset(mini_coco, split="train")
        sample = ds[0]
        assert sample["image"].shape == (64, 64, 3)

    def test_masks_labels_consistent_with_mosaic(self, mini_coco):
        from training_toolkit.configs import AugConfig
        from training_toolkit.data.transforms import build_transforms

        cfg = AugConfig(imgsz=64, mosaic_p=1.0, random_resized_crop=True)
        transform = build_transforms(cfg, is_train=True)
        ds = CocoSegmentationDataset(
            mini_coco, split="train", transform=transform, aug_config=cfg,
        )
        sample = ds[0]
        n = len(sample["masks"])
        assert sample["boxes"].shape == (n, 4)
        assert sample["labels"].shape == (n,)


class TestFullPipeline:
    """End-to-end: dataset with multi-sample + albumentations -> collate."""

    def test_mosaic_through_collate(self, mini_coco):
        from training_toolkit.configs import AugConfig
        from training_toolkit.data.transforms import build_transforms

        cfg = AugConfig(imgsz=64, mosaic_p=1.0, random_resized_crop=True)
        transform = build_transforms(cfg, is_train=True)
        ds = CocoSegmentationDataset(
            mini_coco, split="train", transform=transform, aug_config=cfg,
        )
        batch = collate_fn([ds[0]])
        assert batch["images"].shape == (1, 3, 64, 64)
        assert len(batch["targets"]) == 1
        target = batch["targets"][0]
        assert target["masks"].shape[0] == target["boxes"].shape[0]
        assert target["masks"].shape[0] == target["labels"].shape[0]
