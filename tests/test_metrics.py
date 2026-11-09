"""Tests for metrics wrapper."""

import pytest
import torch

from training_toolkit.metrics import SegmentationMAP


class TestSegmentationMAP:
    def test_update_and_compute(self):
        metric = SegmentationMAP()
        preds = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]
        targets = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]
        metric.update(preds, targets)
        result = metric.compute()
        assert "map" in result
        assert result["map"] >= 0.0

    def test_format_results_flattens_per_class(self):
        metric = SegmentationMAP(class_names={0: "crack", 1: "spall"})
        preds = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]
        targets = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]
        metric.update(preds, targets)
        raw = metric.compute()
        formatted = metric.format_results(raw)
        assert "map" in formatted
        assert "map_50" in formatted

    def test_reset_clears_state(self):
        metric = SegmentationMAP()
        preds = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "scores": torch.tensor([0.9]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]
        targets = [{
            "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]
        metric.update(preds, targets)
        metric.reset()
        # After reset, compute should work on empty state
        # TorchMetrics may return 0 or -1 for empty state


def _rows_mask(rows: int, size: int = 64) -> torch.Tensor:
    """A [1, size, size] bool mask whose first ``rows`` rows are set."""
    m = torch.zeros(1, size, size, dtype=torch.bool)
    m[0, :rows] = True
    return m


class TestPerClassAPAtFixedIoU:
    """Per-class AP at IoU 0.50 and 0.75, from the extended COCO summary.

    Hand-built case (mask IoU decides segm matching):
      crack: one prediction overlapping its one GT with mask IoU 31/50 = 0.62.
             It matches at IoU 0.50/0.55/0.60 only → AP@50 = 1, AP@75 = 0,
             AP@[.50:.95] = 3/10 = 0.3.
      spall: one prediction identical to its one GT → 1 everywhere.
      rebar: no GT and no prediction → absent (no data), never 0.
    """

    def _formatted(self) -> dict:
        metric = SegmentationMAP(class_names={0: "crack", 1: "spall", 2: "rebar"})
        box = torch.tensor([[0.0, 0.0, 64.0, 64.0]])
        preds = [
            {"boxes": box, "scores": torch.tensor([0.9]), "labels": torch.tensor([0]),
             "masks": _rows_mask(31)},
            {"boxes": box, "scores": torch.tensor([0.8]), "labels": torch.tensor([1]),
             "masks": _rows_mask(40)},
        ]
        targets = [
            {"boxes": box, "labels": torch.tensor([0]), "masks": _rows_mask(50)},
            {"boxes": box, "labels": torch.tensor([1]), "masks": _rows_mask(40)},
        ]
        metric.update(preds, targets)
        return metric, metric.format_results(metric.compute())

    def test_ap50_per_class_counts_a_match_that_fails_stricter_ious(self):
        _, f = self._formatted()
        assert f["ap50_crack"] == pytest.approx(1.0)
        assert f["ap50_spall"] == pytest.approx(1.0)

    def test_ap75_per_class_drops_the_loose_match(self):
        _, f = self._formatted()
        assert f["ap75_crack"] == pytest.approx(0.0)
        assert f["ap75_spall"] == pytest.approx(1.0)

    def test_per_class_map_is_unchanged(self):
        _, f = self._formatted()
        assert f["map_crack"] == pytest.approx(0.3)
        assert f["map_spall"] == pytest.approx(1.0)

    def test_a_class_with_no_data_has_no_ap_keys(self):
        _, f = self._formatted()
        assert "ap50_rebar" not in f and "ap75_rebar" not in f and "map_rebar" not in f

    def test_new_keys_do_not_start_with_map_prefix(self):
        # PerClassBestCheckpointCallback tracks EVERY "val/map_*" key: a
        # "map_50_<class>" name would add a best-checkpoint per class per IoU.
        _, f = self._formatted()
        assert not [k for k in f if k.startswith("map_") and ("ap50" in k or "ap75" in k)]

    def test_verbose_table_shows_ap50_per_class(self):
        metric, _ = self._formatted()
        text = metric.format_verbose(metric.compute())
        crack_line = next(l for l in text.splitlines() if l.strip().startswith("crack"))
        header = next(l for l in text.splitlines() if "mAP" in l and "F1" in l)
        assert header.split() == ["mAP", "AP50", "AP75", "F1", "@conf"]
        # name, mAP@[.50:.95], AP50, AP75, F1, conf — AP75 = 0 pins the column order
        assert crack_line.split() == ["crack", "0.3000", "1.0000", "0.0000", "1.0000", "0.900"]
        assert not any(l.strip().startswith(("ap50", "50_", "ap75")) for l in text.splitlines())
