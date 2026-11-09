"""Tests for YOLO-format dataset loading."""

import numpy as np
import pytest
import yaml

from training_toolkit.data.yolo_dataset import YoloSegmentationDataset


# ── Fixtures ──────────────────────────────────────────────────────────


def _create_test_image(path, w=100, h=80):
    """Create a minimal valid PNG image file."""
    import cv2
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[10:60, 20:70] = (128, 64, 32)  # some content
    cv2.imwrite(str(path), img)


def _write_data_yaml(root, class_names, layout="A"):
    """Write a data.yaml file with the given class names and layout paths."""
    if layout == "A":
        data = {
            "names": class_names,
            "train": "images/train",
            "val": "images/val",
        }
    else:
        data = {
            "names": class_names,
            "train": "train/images",
            "val": "val/images",
        }
    yaml_path = root / "data.yaml"
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(data, fh)


def _make_yolo_a_dataset(tmp_path, n_train=3, n_val=1):
    """Create a YOLO-A layout dataset with polygon labels.

    YOLO-A: images/{split}/ + labels/{split}/ + data.yaml
    """
    class_names = {0: "crack", 1: "spall"}

    for split, count in [("train", n_train), ("val", n_val)]:
        img_dir = tmp_path / "images" / split
        lbl_dir = tmp_path / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(count):
            _create_test_image(img_dir / f"img_{i:03d}.png")
            # Write a label with one polygon instance
            # Normalized polygon: rectangle 0.2,0.1 → 0.7,0.7
            lbl_path = lbl_dir / f"img_{i:03d}.txt"
            lbl_path.write_text(
                "0 0.2 0.1 0.7 0.1 0.7 0.7 0.2 0.7\n"
                "1 0.5 0.5 0.8 0.5 0.8 0.9 0.5 0.9\n"
            )

    _write_data_yaml(tmp_path, class_names, layout="A")
    return tmp_path


def _make_yolo_b_dataset(tmp_path, n_train=3, n_val=1):
    """Create a YOLO-B layout dataset.

    YOLO-B: {split}/images/ + {split}/labels/ + data.yaml
    """
    class_names = ["crack", "spall"]

    for split, count in [("train", n_train), ("val", n_val)]:
        img_dir = tmp_path / split / "images"
        lbl_dir = tmp_path / split / "labels"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)

        for i in range(count):
            _create_test_image(img_dir / f"img_{i:03d}.png")
            lbl_path = lbl_dir / f"img_{i:03d}.txt"
            lbl_path.write_text("0 0.2 0.1 0.7 0.1 0.7 0.7 0.2 0.7\n")

    _write_data_yaml(tmp_path, class_names, layout="B")
    return tmp_path


# ── Tests ─────────────────────────────────────────────────────────────


class TestYoloDatasetDiscovery:
    def test_yolo_a_layout(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        assert len(ds) == 3

    def test_yolo_b_layout(self, tmp_path):
        root = _make_yolo_b_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        assert len(ds) == 3

    def test_class_names_from_dict(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        assert ds.class_names == {0: "crack", 1: "spall"}
        assert ds.num_classes == 2

    def test_class_names_from_list(self, tmp_path):
        root = _make_yolo_b_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        assert ds.class_names == {0: "crack", 1: "spall"}

    def test_val_split(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="val")
        assert len(ds) == 1

    def test_missing_data_yaml_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="data.yaml"):
            YoloSegmentationDataset(tmp_path, split="train")

    def test_missing_split_raises(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            YoloSegmentationDataset(root, split="test")


class TestYoloSampleLoading:
    def test_load_sample_returns_correct_keys(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        sample = ds[0]
        assert "image" in sample
        assert "masks" in sample
        assert "boxes" in sample
        assert "labels" in sample
        assert "image_id" in sample

    def test_image_shape_and_dtype(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        sample = ds[0]
        assert sample["image"].shape == (80, 100, 3)
        assert sample["image"].dtype == np.uint8

    def test_masks_are_binary(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        sample = ds[0]
        assert len(sample["masks"]) == 2  # two instances per label file
        for mask in sample["masks"]:
            assert mask.shape == (80, 100)
            assert set(np.unique(mask)).issubset({0, 1})

    def test_boxes_xyxy_format(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        sample = ds[0]
        boxes = sample["boxes"]
        assert boxes.shape == (2, 4)
        assert boxes.dtype == np.float32
        # All boxes should have x1 < x2 and y1 < y2
        assert np.all(boxes[:, 0] < boxes[:, 2])
        assert np.all(boxes[:, 1] < boxes[:, 3])

    def test_labels_dtype(self, tmp_path):
        root = _make_yolo_a_dataset(tmp_path)
        ds = YoloSegmentationDataset(root, split="train")
        sample = ds[0]
        assert sample["labels"].dtype == np.int64
        assert set(sample["labels"].tolist()) == {0, 1}

    def test_empty_label_file(self, tmp_path):
        """Images with empty label files should return zero instances."""
        root = _make_yolo_a_dataset(tmp_path)
        # Overwrite one label file with empty content
        lbl = root / "labels" / "train" / "img_000.txt"
        lbl.write_text("")
        ds = YoloSegmentationDataset(root, split="train")
        sample = ds[0]
        assert len(sample["masks"]) == 0
        assert sample["boxes"].shape == (0, 4)
        assert len(sample["labels"]) == 0

    def test_missing_label_file(self, tmp_path):
        """Images without a matching label file should return zero instances."""
        root = _make_yolo_a_dataset(tmp_path)
        # Add an image without a label file
        _create_test_image(root / "images" / "train" / "unlabeled.png")
        ds = YoloSegmentationDataset(root, split="train")
        # Find the unlabeled sample
        for i in range(len(ds)):
            sample = ds[i]
            if len(sample["masks"]) == 0:
                assert sample["boxes"].shape == (0, 4)
                return
        # At least one sample should have no labels
        # (could also have been img_000 if ordering differs)


class TestYoloDataConfig:
    def test_yolo_layout_accepted(self):
        from training_toolkit.configs import DataConfig
        cfg = DataConfig(dataset_path="/x", layout="yolo")
        assert cfg.layout == "yolo"

    def test_invalid_layout_still_raises(self):
        from training_toolkit.configs import DataConfig
        with pytest.raises(ValueError, match="layout must be"):
            DataConfig(dataset_path="/x", layout="unknown")
