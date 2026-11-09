"""Tests for the TideAnalyzer wrapper around tidecv."""

import numpy as np
import pytest
import torch

from training_toolkit.metrics import TideAnalyzer


def _make_preds_targets(num_images: int = 2):
    """Build perfect-match preds + targets: 2 GTs per image, predictions identical to GTs.

    Returns lists matching the convention used by SegmentationMAP.update():
    each entry is a dict with boxes (xyxy float), scores, labels, masks (bool).
    """
    preds, targets = [], []
    for i in range(num_images):
        # Two bounding boxes per image, two classes
        boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 55.0, 55.0]])
        labels = torch.tensor([0, 1])
        masks = torch.zeros(2, 64, 64, dtype=torch.bool)
        masks[0, 10:30, 10:30] = True
        masks[1, 40:55, 40:55] = True

        preds.append({
            "boxes": boxes.clone(),
            "scores": torch.tensor([0.9, 0.9]),
            "labels": labels.clone(),
            "masks": masks.clone(),
        })
        targets.append({
            "boxes": boxes.clone(),
            "labels": labels.clone(),
            "masks": masks.clone(),
        })
    return preds, targets


class TestTideAnalyzer:
    def test_compute_returns_expected_structure(self):
        """Smoke test: update() + compute() should produce a dict with aggregate, per_class, summary_text."""
        analyzer = TideAnalyzer(class_names={0: "crack", 1: "spall"})
        preds, targets = _make_preds_targets(num_images=2)
        analyzer.update(preds, targets, batch_idx=0)

        result = analyzer.compute()

        assert "aggregate" in result
        assert "per_class" in result
        assert "summary_text" in result

        for err_name in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"):
            assert err_name in result["aggregate"]
        for err_name in ("FalsePos", "FalseNeg"):
            assert err_name in result["aggregate"]

        for cls_name in ("crack", "spall"):
            assert cls_name in result["per_class"]
            assert "AP" in result["per_class"][cls_name]
            for err_name in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"):
                assert err_name in result["per_class"][cls_name]

    def test_perfect_predictions_have_zero_errors(self):
        """Perfect preds → ~zero dAP across all categories, AP ~= 100 (tidecv scale).

        Note: tidecv reports AP and dAP on a 0-100 scale (percentage points),
        NOT the 0-1 scale TorchMetrics uses. TideAnalyzer preserves this for
        consistency with tide.summarize() text output.
        """
        analyzer = TideAnalyzer(class_names={0: "crack", 1: "spall"})
        preds, targets = _make_preds_targets(num_images=3)
        analyzer.update(preds, targets, batch_idx=0)

        result = analyzer.compute()

        for err_name in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"):
            assert result["aggregate"][err_name] == pytest.approx(0.0, abs=1e-3), (
                f"Expected zero {err_name} dAP for perfect preds, "
                f"got {result['aggregate'][err_name]}"
            )
        for cls_name in ("crack", "spall"):
            assert result["per_class"][cls_name]["AP"] == pytest.approx(100.0, abs=1e-3)

    def test_multi_batch_accumulation_uses_unique_image_ids(self):
        """update() called with the same batch_idx for different calls must NOT
        collide. The integration calls update(batch_idx=i) once per batch; image
        ids derived from (batch_idx, sample_idx) must be unique across the val
        set so TIDE treats every val image as independent."""
        # Two batches, each with 2 perfect-match images. After both batches,
        # we should have 4 distinct images in the GT buffer (no overlap).
        analyzer = TideAnalyzer(class_names={0: "crack", 1: "spall"})

        preds_a, targets_a = _make_preds_targets(num_images=2)
        preds_b, targets_b = _make_preds_targets(num_images=2)

        analyzer.update(preds_a, targets_a, batch_idx=0)
        analyzer.update(preds_b, targets_b, batch_idx=1)

        gt_image_ids = {entry[0] for entry in analyzer._gt_buf}
        assert len(gt_image_ids) == 4, (
            f"Expected 4 unique image ids across 2 batches of 2, got {gt_image_ids}"
        )
        # Compute should still produce zero errors (all preds perfect)
        result = analyzer.compute()
        for err_name in ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"):
            assert result["aggregate"][err_name] == pytest.approx(0.0, abs=1e-3)

    def test_reset_clears_buffers_between_epochs(self):
        """After reset(), the analyzer should behave as if freshly constructed."""
        analyzer = TideAnalyzer(class_names={0: "crack", 1: "spall"})
        preds, targets = _make_preds_targets(num_images=2)
        analyzer.update(preds, targets, batch_idx=0)

        analyzer.reset()
        assert analyzer._gt_buf == []
        assert analyzer._pred_buf == []

        # Fresh accumulation on a new "epoch" should work identically.
        analyzer.update(preds, targets, batch_idx=0)
        result = analyzer.compute()
        for cls_name in ("crack", "spall"):
            assert result["per_class"][cls_name]["AP"] == pytest.approx(100.0, abs=1e-3)

    def test_format_per_class_table_has_expected_columns_and_rows(self):
        """The custom per-class table (not in tidecv) should have one row per
        class and columns Class | AP | Cls | Loc | Both | Dupe | Bkg | Miss."""
        analyzer = TideAnalyzer(class_names={0: "crack", 1: "spall"})
        preds, targets = _make_preds_targets(num_images=2)
        analyzer.update(preds, targets, batch_idx=0)
        result = analyzer.compute()

        table = analyzer.format_per_class_table(result)

        # Header has all expected columns (in order).
        first_line = table.splitlines()[0]
        for col in ("Class", "AP", "Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"):
            assert col in first_line, f"Column {col!r} missing from header: {first_line!r}"

        # Body has one row per class (substring match — formatting may add padding).
        body = "\n".join(table.splitlines()[1:])
        assert "crack" in body
        assert "spall" in body


class TestDiagnosticsCallbackTideFlush:
    """The DiagnosticsCallback._save_tide() helper writes TIDE artifacts to disk.

    Exercises the callback in isolation (no Lightning Trainer) by stashing a
    real TIDE result on a stub pl_module.
    """

    class _StubLitModule:
        def __init__(self, tide_results=None):
            self._tide_results = tide_results

    def _build_tide_results(self, epoch: int = 5) -> dict:
        analyzer = TideAnalyzer(class_names={0: "crack", 1: "spall"})
        preds, targets = _make_preds_targets(num_images=2)
        analyzer.update(preds, targets, batch_idx=0)
        result = analyzer.compute()
        result["epoch"] = epoch
        return result

    def test_save_tide_writes_summary_and_per_class_text(self, tmp_path):
        from training_toolkit.callbacks.diagnostics import DiagnosticsCallback

        cb = DiagnosticsCallback(run_dir=tmp_path, period=1, class_names={0: "crack", 1: "spall"})
        pl_module = self._StubLitModule(tide_results=self._build_tide_results())

        cb._save_tide(pl_module)

        assert (tmp_path / "tide_summary.txt").exists()
        assert (tmp_path / "tide_per_class.txt").exists()

        per_class_text = (tmp_path / "tide_per_class.txt").read_text()
        assert "crack" in per_class_text
        assert "spall" in per_class_text

    def test_save_tide_is_idempotent_without_new_epoch(self, tmp_path):
        """If stash epoch hasn't advanced since last write, don't rewrite."""
        from training_toolkit.callbacks.diagnostics import DiagnosticsCallback

        cb = DiagnosticsCallback(run_dir=tmp_path, period=1, class_names={0: "crack", 1: "spall"})
        pl_module = self._StubLitModule(tide_results=self._build_tide_results(epoch=5))

        cb._save_tide(pl_module)
        first_mtime = (tmp_path / "tide_summary.txt").stat().st_mtime_ns

        cb._save_tide(pl_module)
        second_mtime = (tmp_path / "tide_summary.txt").stat().st_mtime_ns

        assert first_mtime == second_mtime

    def test_save_tide_handles_missing_stash(self, tmp_path):
        """If no _tide_results is stashed (val_tide=False path), _save_tide is a no-op."""
        from training_toolkit.callbacks.diagnostics import DiagnosticsCallback

        cb = DiagnosticsCallback(run_dir=tmp_path, period=1)
        pl_module = self._StubLitModule(tide_results=None)

        cb._save_tide(pl_module)

        assert not (tmp_path / "tide_summary.txt").exists()


def _make_scored_preds(scores: list[float], size: int = 64):
    """One image's pred dict with N dets at the given scores (distinct boxes)."""
    n = len(scores)
    boxes = torch.tensor(
        [[float(i % 8) * 8, float(i // 8) * 8, float(i % 8) * 8 + 6, float(i // 8) * 8 + 6]
         for i in range(n)]
    )
    masks = torch.zeros(n, size, size, dtype=torch.bool)
    for i in range(n):
        masks[i, i % size, : 4] = True
    return {
        "boxes": boxes,
        "scores": torch.tensor(scores),
        "labels": torch.zeros(n, dtype=torch.long),
        "masks": masks,
    }


def _one_gt_target(size: int = 64):
    return {
        "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]),
        "labels": torch.tensor([0]),
        "masks": torch.zeros(1, size, size, dtype=torch.bool) | (
            torch.zeros(1, size, size, dtype=torch.bool)
        ),
    }


class TestTideInputFilter:
    """Community-standard input filtering: score floor 0.05 + top-100/img.

    Published TIDE numbers implicitly run on COCO results files (<=100
    dets/img, RCNN-lineage 0.05 floor); DETR-family unfiltered decodes
    otherwise drown the FalsePos/Bkg counterfactuals in the structural
    low-score tail.
    """

    def test_default_score_floor_drops_low_scores(self):
        analyzer = TideAnalyzer(class_names={0: "crack"})
        preds = [_make_scored_preds([0.9, 0.5, 0.06, 0.05, 0.01])]
        analyzer.update(preds, [_one_gt_target()], batch_idx=0)
        # strict > 0.05 (detectron2/mmdet convention): 0.05 itself is dropped
        assert len(analyzer._pred_buf) == 3

    def test_default_top_k_caps_per_image_keeping_highest(self):
        analyzer = TideAnalyzer(class_names={0: "crack"})
        scores = torch.linspace(0.9, 0.2, 150).tolist()
        analyzer.update([_make_scored_preds(scores)], [_one_gt_target()], batch_idx=0)
        assert len(analyzer._pred_buf) == 100
        kept_scores = sorted((e[2] for e in analyzer._pred_buf), reverse=True)
        expected = sorted(scores, reverse=True)[:100]
        assert kept_scores == pytest.approx(expected)

    def test_cap_is_per_image_not_global(self):
        analyzer = TideAnalyzer(class_names={0: "crack"})
        scores = [0.9] * 120
        preds = [_make_scored_preds(scores), _make_scored_preds(scores)]
        analyzer.update(preds, [_one_gt_target(), _one_gt_target()], batch_idx=0)
        assert len(analyzer._pred_buf) == 200

    def test_filters_disable(self):
        analyzer = TideAnalyzer(class_names={0: "crack"}, score_thresh=0.0, max_dets=None)
        scores = torch.linspace(0.9, 0.001, 150).tolist()
        analyzer.update([_make_scored_preds(scores)], [_one_gt_target()], batch_idx=0)
        assert len(analyzer._pred_buf) == 150

    def test_compute_returns_none_with_no_predictions(self):
        """Degenerate-model path (fresh-init sanity val): every prediction
        falls below the score floor -> zero preds accumulated -> compute()
        returns None instead of hitting tidecv's ZeroDivisionError
        (sum(aps)/len(aps) over an empty run)."""
        analyzer = TideAnalyzer(class_names={0: "crack"})
        analyzer.update(
            [_make_scored_preds([0.01, 0.02])], [_one_gt_target()], batch_idx=0,
        )
        assert len(analyzer._pred_buf) == 0
        assert analyzer.compute() is None

    def test_gt_untouched_by_filters(self):
        analyzer = TideAnalyzer(class_names={0: "crack"})
        preds = [_make_scored_preds([0.01, 0.02])]  # all below floor
        target = {
            "boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0], [40.0, 40.0, 55.0, 55.0]]),
            "labels": torch.tensor([0, 0]),
            "masks": torch.zeros(2, 64, 64, dtype=torch.bool),
        }
        target["masks"][0, 10:30, 10:30] = True
        target["masks"][1, 40:55, 40:55] = True
        analyzer.update(preds, [target], batch_idx=0)
        assert len(analyzer._pred_buf) == 0
        assert len(analyzer._gt_buf) == 2
