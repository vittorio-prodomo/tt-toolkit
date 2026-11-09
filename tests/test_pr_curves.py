"""Tests for PR/confidence curve extraction and plotting."""

import numpy as np
import pytest


def _make_fake_pr_data(num_classes: int = 3) -> dict:
    """Build a fake raw compute() result with extended_summary arrays.

    Shape convention: precision/scores are [T, R, K, A, M]
    T=10 IoU thresholds, R=101 recall points, K=num_classes, A=4 areas, M=3 maxDets
    """
    T, R, A, M = 10, 101, 4, 3
    K = num_classes
    precision = np.zeros((T, R, K, A, M), dtype=np.float64)
    scores = np.zeros((T, R, K, A, M), dtype=np.float64)
    for k in range(K):
        for r in range(R):
            precision[:, r, k, :, :] = max(0, 1.0 - r / R - k * 0.1)
            scores[:, r, k, :, :] = max(0, 1.0 - r / R)

    import torch
    return {
        "map": torch.tensor(0.5),
        "map_50": torch.tensor(0.6),
        "precision": torch.tensor(precision),
        "scores": torch.tensor(scores),
        "classes": torch.tensor([0, 1, 2], dtype=torch.int32),
        "map_per_class": torch.tensor([0.7, 0.5, 0.3]),
    }


class TestExtractPrCurves:
    def test_returns_correct_keys(self):
        from training_toolkit.metrics.map import SegmentationMAP

        seg_map = SegmentationMAP(class_names={0: "crack", 1: "spall", 2: "rebar"})
        raw = _make_fake_pr_data(num_classes=3)
        result = seg_map.extract_pr_curves(raw)

        assert "precision" in result
        assert "scores" in result
        assert "rec_thresholds" in result
        assert "class_names" in result

    def test_iou50_slice_shape(self):
        from training_toolkit.metrics.map import SegmentationMAP

        seg_map = SegmentationMAP(class_names={0: "crack", 1: "spall", 2: "rebar"})
        raw = _make_fake_pr_data(num_classes=3)
        result = seg_map.extract_pr_curves(raw)

        assert result["precision"].shape == (101, 3)
        assert result["scores"].shape == (101, 3)
        assert len(result["rec_thresholds"]) == 101

    def test_class_names_ordering(self):
        from training_toolkit.metrics.map import SegmentationMAP

        seg_map = SegmentationMAP(class_names={0: "crack", 1: "spall", 2: "rebar"})
        raw = _make_fake_pr_data(num_classes=3)
        result = seg_map.extract_pr_curves(raw)

        assert result["class_names"] == ["crack", "spall", "rebar"]

    def test_returns_none_without_extended_summary(self):
        from training_toolkit.metrics.map import SegmentationMAP

        seg_map = SegmentationMAP(class_names={0: "crack"})
        raw = {"map": __import__("torch").tensor(0.5)}
        result = seg_map.extract_pr_curves(raw)
        assert result is None


class TestPlotFunctions:
    """Test that the 4 plot functions produce PNG files without errors."""

    @pytest.fixture()
    def curve_data(self):
        """Minimal curve data for plotting."""
        R, K = 101, 3
        precision = np.zeros((R, K))
        scores = np.zeros((R, K))
        for k in range(K):
            for r in range(R):
                precision[r, k] = max(0, 1.0 - r / R - k * 0.1)
                scores[r, k] = max(0, 1.0 - r / R)
        return {
            "precision": precision,
            "scores": scores,
            "rec_thresholds": np.linspace(0, 1, R),
            "class_names": ["crack", "spall", "rebar"],
        }

    def test_save_pr_curve(self, tmp_path, curve_data):
        from training_toolkit.visualization import save_pr_curve

        path = tmp_path / "PR_curve.png"
        save_pr_curve(path=path, **curve_data)
        assert path.exists()
        assert path.stat().st_size > 1000  # non-trivial PNG

    def test_save_p_curve(self, tmp_path, curve_data):
        from training_toolkit.visualization import save_p_curve

        path = tmp_path / "P_curve.png"
        save_p_curve(path=path, **curve_data)
        assert path.exists()

    def test_save_r_curve(self, tmp_path, curve_data):
        from training_toolkit.visualization import save_r_curve

        path = tmp_path / "R_curve.png"
        save_r_curve(path=path, **curve_data)
        assert path.exists()

    def test_save_f1_curve(self, tmp_path, curve_data):
        from training_toolkit.visualization import save_f1_curve

        path = tmp_path / "F1_curve.png"
        save_f1_curve(path=path, **curve_data)
        assert path.exists()

    def test_pr_curve_handles_single_class(self, tmp_path):
        from training_toolkit.visualization import save_pr_curve

        R = 101
        data = {
            "precision": np.linspace(1, 0, R).reshape(R, 1),
            "scores": np.linspace(1, 0, R).reshape(R, 1),
            "rec_thresholds": np.linspace(0, 1, R),
            "class_names": ["crack"],
        }
        path = tmp_path / "PR_curve.png"
        save_pr_curve(path=path, **data)
        assert path.exists()
