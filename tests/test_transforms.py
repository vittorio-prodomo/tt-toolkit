"""Tests for Albumentations pipeline builder."""

from albumentations.core.composition import Compose

from training_toolkit.configs import AugConfig
from training_toolkit.data.transforms import build_transforms


class TestBuildTransforms:
    def test_returns_single_compose_when_no_mosaic(self):
        """build_transforms returns a single Compose when mosaic is off."""
        cfg = AugConfig(imgsz=512, mosaic_p=0.0)
        result = build_transforms(cfg, is_train=True)
        assert isinstance(result, Compose)

    def test_returns_tuple_when_mosaic_needs_separate_pipeline(self):
        """When mosaic_p > 0 and mosaic_crop=True and random_resized_crop=False,
        a (normal, mosaic) tuple is returned so the post-mosaic pipeline can
        include RandomResizedCrop without affecting the no-mosaic path."""
        cfg = AugConfig(imgsz=512, mosaic_p=0.8, mosaic_crop=True, random_resized_crop=False)
        result = build_transforms(cfg, is_train=True)
        assert isinstance(result, tuple) and len(result) == 2
        normal, mosaic_pipeline = result
        assert isinstance(normal, Compose)
        assert isinstance(mosaic_pipeline, Compose)

    def test_no_random_resized_crop_when_disabled(self):
        cfg = AugConfig(imgsz=512, random_resized_crop=False)
        pipeline = build_transforms(cfg, is_train=True)
        transform_names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "RandomResizedCrop" not in transform_names

    def test_random_resized_crop_when_enabled(self):
        cfg = AugConfig(imgsz=512, random_resized_crop=True)
        pipeline = build_transforms(cfg, is_train=True)
        transform_names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "RandomResizedCrop" in transform_names

    def test_val_transforms_no_augmentation(self):
        cfg = AugConfig(imgsz=512, horizontal_flip=0.5)
        pipeline = build_transforms(cfg, is_train=False)
        transform_names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "HorizontalFlip" not in transform_names
        assert "LongestMaxSize" in transform_names

    def test_horizontal_flip_included(self):
        cfg = AugConfig(imgsz=512, horizontal_flip=0.5)
        pipeline = build_transforms(cfg, is_train=True)
        transform_names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "HorizontalFlip" in transform_names

    def test_no_flip_when_zero(self):
        cfg = AugConfig(imgsz=512, horizontal_flip=0.0, vertical_flip=0.0)
        pipeline = build_transforms(cfg, is_train=True)
        transform_names = [t.__class__.__name__ for t in pipeline.transforms]
        assert "HorizontalFlip" not in transform_names
        assert "VerticalFlip" not in transform_names

    def test_scale_zoom_keeps_aspect_ratio(self):
        """`scale` is a ZOOM: one factor for both axes. Albumentations' Affine
        samples x and y independently unless keep_ratio=True, which turns the
        knob into a per-axis stretch (measured: x 0.65 / y 1.16 in one draw)."""
        import numpy as np

        cfg = AugConfig(imgsz=512, scale=0.4, affine_p=1.0)
        pipeline = build_transforms(cfg, is_train=True)
        affine = next(t for t in pipeline.transforms if t.__class__.__name__ == "Affine")
        img = np.zeros((512, 512, 3), np.uint8)
        for _ in range(20):
            s = affine.get_params_dependent_on_data({"shape": img.shape}, {"image": img})["scale"]
            assert s["x"] == s["y"]
            assert 0.6 <= s["x"] <= 1.4
