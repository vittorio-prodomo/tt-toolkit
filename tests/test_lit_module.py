"""Tests for SegmentationLitModule with a mock adapter."""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn
import pytest

from training_toolkit.engine.lit_module import SegmentationLitModule
from training_toolkit.models.base import ModelAdapter
from training_toolkit.configs import TrainConfig, ModelConfig, DataConfig


class MockModel(nn.Module):
    """Trivial model that returns fixed outputs."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)

    def forward(self, **kwargs):
        return {
            "pred_logits": torch.randn(1, 10, 2),
            "pred_masks": torch.randn(1, 10, 64, 64),
        }


class MockAdapter(ModelAdapter):
    def create_model(self, config):
        return MockModel()

    def create_criterion(self, config):
        return None

    def format_batch(self, batch):
        return {"inputs": {}, "targets": batch["targets"]}

    def extract_predictions(self, outputs):
        return [
            {
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                "scores": torch.tensor([0.5]),
                "labels": torch.tensor([0]),
                "masks": torch.ones(1, 64, 64, dtype=torch.bool),
            }
        ]

    def compute_loss(self, outputs, targets):
        return {"loss_total": torch.tensor(1.0, requires_grad=True)}


@pytest.fixture
def config():
    return TrainConfig(
        model=ModelConfig(name="mock"),
        data=DataConfig(dataset_path="/tmp"),
    )


@pytest.fixture
def lit_module(config):
    adapter = MockAdapter()
    return SegmentationLitModule(adapter=adapter, config=config)


class TestSegmentationLitModule:
    def test_instantiation(self, lit_module):
        assert lit_module.model is not None

    def test_training_step_returns_loss(self, lit_module):
        batch = {
            "images": torch.randn(1, 3, 64, 64),
            "targets": [
                {
                    "boxes": torch.tensor([[10.0, 10.0, 50.0, 50.0]]),
                    "labels": torch.tensor([0]),
                    "masks": torch.ones(1, 64, 64, dtype=torch.bool),
                }
            ],
        }
        loss = lit_module.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad

    def test_configure_optimizers(self, lit_module):
        result = lit_module.configure_optimizers()
        # Should return optimizer(s) and scheduler(s)
        assert result is not None

    def test_val_tide_is_none_by_default(self, lit_module):
        """When config.val_tide=False (the default), no TIDE analyzer is created."""
        assert lit_module.val_tide is None

    def test_val_tide_is_analyzer_when_enabled(self, config):
        """When config.val_tide=True, a TideAnalyzer is instantiated alongside val_map."""
        from training_toolkit.metrics import TideAnalyzer
        config.val_tide = True
        lit = SegmentationLitModule(
            adapter=MockAdapter(), config=config, class_names={0: "crack"},
        )
        assert isinstance(lit.val_tide, TideAnalyzer)
        assert lit.val_tide.class_names == {0: "crack"}


class TestComputeTide:
    """_compute_tide: guarded TIDE runner — sanity-val skip + empty-run quiet path.

    A fresh-init model produces zero detections at the pre-training sanity
    val, where tidecv raises ZeroDivisionError; the guard skips compute there
    (and stays quiet on empty epochs) while ALWAYS resetting accumulated state
    so sanity batches never leak into epoch 0's real TIDE.
    """

    @pytest.fixture
    def lit_with_mock_tide(self, config):
        lit = SegmentationLitModule(adapter=MockAdapter(), config=config)
        lit.val_tide = Mock()
        return lit

    def test_sanity_check_skips_compute_but_resets(self, lit_with_mock_tide):
        lit = lit_with_mock_tide
        lit._trainer = SimpleNamespace(sanity_checking=True)
        assert lit._compute_tide() is None
        lit.val_tide.compute.assert_not_called()
        lit.val_tide.reset.assert_called_once()

    def test_normal_val_computes_and_resets(self, lit_with_mock_tide):
        lit = lit_with_mock_tide
        lit._trainer = SimpleNamespace(sanity_checking=False)
        sentinel = {"aggregate": {"Cls": 1.0}}
        lit.val_tide.compute.return_value = sentinel
        assert lit._compute_tide() is sentinel
        lit.val_tide.reset.assert_called_once()

    def test_no_trainer_attached_still_computes(self, lit_with_mock_tide):
        """Outside a fit() lifecycle (unit tests, scripts) there is no trainer;
        the guard must not raise Lightning's 'not attached' RuntimeError."""
        lit = lit_with_mock_tide
        sentinel = {"aggregate": {}}
        lit.val_tide.compute.return_value = sentinel
        assert lit._compute_tide() is sentinel

    def test_empty_epoch_logs_info_not_warning(self, lit_with_mock_tide, caplog):
        lit = lit_with_mock_tide
        lit._trainer = SimpleNamespace(sanity_checking=False)
        lit.val_tide.compute.return_value = None
        with caplog.at_level(logging.INFO):
            assert lit._compute_tide() is None
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("TIDE skipped" in r.getMessage() for r in caplog.records)
        lit.val_tide.reset.assert_called_once()

    def test_tide_disabled_returns_none(self, lit_module):
        assert lit_module.val_tide is None
        assert lit_module._compute_tide() is None


class TestSanityValPrint:
    """on_validation_epoch_end during the sanity check: the 2-batch smoke val
    must be labelled as such — its metrics block has repeatedly been mistaken
    for a real validation seconds after a launch/resume — and it must not seed
    best-metric tracking or the PR-curve stash."""

    @pytest.fixture
    def lit_with_mock_map(self, config):
        lit = SegmentationLitModule(adapter=MockAdapter(), config=config)
        lit.val_map = Mock()
        lit.val_map.compute.return_value = {}
        lit.val_map.format_results.return_value = {
            "map": 0.05, "map_50": 0.1, "map_75": 0.02, "f1_best": 0.1,
        }
        lit.val_map.extract_pr_curves.return_value = None
        lit.log = Mock()  # neutralize Lightning's logging plumbing
        return lit

    def test_sanity_val_is_labelled_and_skips_best_tracking(
        self, lit_with_mock_map, caplog
    ):
        lit = lit_with_mock_map
        lit._trainer = SimpleNamespace(sanity_checking=True)
        with caplog.at_level(logging.INFO):
            lit.on_validation_epoch_end()
        msgs = [r.getMessage() for r in caplog.records]
        assert not any(
            "Validation results" in m or m.startswith("Validation —") for m in msgs
        ), "sanity-check val must not print like a real validation"
        assert any("sanity" in m.lower() for m in msgs), (
            "sanity-check val must announce itself as a sanity check"
        )
        assert lit._best_monitored is None
        lit.val_map.reset.assert_called_once()

    def test_real_val_print_unchanged(self, lit_with_mock_map, caplog):
        lit = lit_with_mock_map
        lit._trainer = SimpleNamespace(sanity_checking=False)
        with caplog.at_level(logging.INFO):
            lit.on_validation_epoch_end()
        msgs = [r.getMessage() for r in caplog.records]
        assert any("Validation results" in m for m in msgs)
        assert not any("sanity" in m.lower() for m in msgs)
        assert lit._best_monitored == 0.05
        lit.val_map.reset.assert_called_once()

    def test_no_trainer_attached_prints_normally(self, lit_with_mock_map, caplog):
        """Outside a fit() lifecycle there is no trainer; the gate must not
        raise Lightning's 'not attached' RuntimeError and must not relabel."""
        lit = lit_with_mock_map
        with caplog.at_level(logging.INFO):
            lit.on_validation_epoch_end()
        assert any("Validation results" in r.getMessage() for r in caplog.records)
