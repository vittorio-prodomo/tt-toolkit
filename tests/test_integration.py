"""End-to-end integration test with mock adapter and mini COCO dataset."""

import torch
from torch import nn
import pytest
import pytorch_lightning as L

from training_toolkit.configs import TrainConfig, ModelConfig, DataConfig, AugConfig
from training_toolkit.data import CocoSegmentationDataset, collate_fn
from training_toolkit.data.transforms import build_transforms
from training_toolkit.engine.lit_module import SegmentationLitModule
from training_toolkit.models.base import ModelAdapter


class IntegrationMockModel(nn.Module):
    """Model that produces outputs compatible with the mock adapter."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(3, num_classes)  # dummy param to have gradients

    def forward(self, **kwargs):
        # Return something the adapter can work with
        return {"dummy": True, "param": self.proj.weight}


class IntegrationMockAdapter(ModelAdapter):
    def __init__(self, score: float = 0.9):
        self.score = score  # below 0.05 = degenerate model for the TIDE filter

    def create_model(self, config):
        return IntegrationMockModel(config.num_classes or 2)

    def create_criterion(self, config):
        return None

    def format_batch(self, batch):
        return {"inputs": {}, "targets": batch["targets"]}

    def extract_predictions(self, outputs):
        return [{
            "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
            "scores": torch.tensor([self.score]),
            "labels": torch.tensor([0]),
            "masks": torch.ones(1, 64, 64, dtype=torch.bool),
        }]

    def compute_loss(self, outputs, targets):
        return {"loss_total": outputs["param"].sum() * 0 + 1.0}


def test_full_pipeline(mini_coco):
    """Test full pipeline: dataset -> augmentation -> collate -> LitModule."""
    config = TrainConfig(
        model=ModelConfig(name="mock", num_classes=2),
        data=DataConfig(dataset_path=str(mini_coco), batch_size=1),
        augmentation=AugConfig(imgsz=64),
        epochs=1,
    )

    # Dataset + transforms
    transform = build_transforms(config.augmentation, is_train=True)
    dataset = CocoSegmentationDataset(mini_coco, split="train", transform=transform)
    assert len(dataset) > 0

    # Collate a batch
    sample = dataset[0]
    batch = collate_fn([sample])
    assert batch["images"].shape[0] == 1

    # Lightning module
    adapter = IntegrationMockAdapter()
    lit_module = SegmentationLitModule(adapter=adapter, config=config)

    # Training step
    loss = lit_module.training_step(batch, 0)
    assert loss is not None

    # Validation step (should not raise)
    lit_module.validation_step(batch, 0)


def test_lightning_trainer_runs(mini_coco):
    """Test that Lightning Trainer can run 1 epoch without errors."""
    config = TrainConfig(
        model=ModelConfig(name="mock", num_classes=2),
        data=DataConfig(dataset_path=str(mini_coco), batch_size=1, workers=0),
        augmentation=AugConfig(imgsz=64),
        epochs=1,
        device=[0],
        precision="32",  # CPU doesn't support 16-mixed
    )
    transform = build_transforms(config.augmentation, is_train=True)
    train_ds = CocoSegmentationDataset(mini_coco, split="train", transform=transform)
    val_ds = CocoSegmentationDataset(mini_coco, split="val", transform=build_transforms(config.augmentation, is_train=False))

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    adapter = IntegrationMockAdapter()
    lit_module = SegmentationLitModule(
        adapter=adapter, config=config, class_names={0: "crack", 1: "spall"},
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )

    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


def test_val_tide_degenerate_model_stays_quiet(mini_coco, caplog):
    """val_tide=True + a model whose every score falls below the TIDE input
    filter (the fresh-init sanity-val scenario): no 'TIDE compute failed'
    WARNING traceback anywhere — sanity val is skipped by the guard, the real
    epoch-end val short-circuits to a single INFO line."""
    import logging

    config = TrainConfig(
        model=ModelConfig(name="mock", num_classes=2),
        data=DataConfig(dataset_path=str(mini_coco), batch_size=1, workers=0),
        augmentation=AugConfig(imgsz=64),
        epochs=1,
        device=[0],
        precision="32",
        val_tide=True,
    )
    train_ds = CocoSegmentationDataset(
        mini_coco, split="train",
        transform=build_transforms(config.augmentation, is_train=True),
    )
    val_ds = CocoSegmentationDataset(
        mini_coco, split="val",
        transform=build_transforms(config.augmentation, is_train=False),
    )

    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_ds, batch_size=1, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate_fn)

    adapter = IntegrationMockAdapter(score=0.01)  # everything below the 0.05 floor
    lit_module = SegmentationLitModule(
        adapter=adapter, config=config, class_names={0: "crack", 1: "spall"},
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        num_sanity_val_steps=2,
        enable_progress_bar=False,
        enable_checkpointing=False,
        logger=False,
    )

    with caplog.at_level(logging.INFO):
        trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    assert not [
        r for r in caplog.records if "TIDE compute failed" in r.getMessage()
    ], "degenerate TIDE input must not produce the WARNING traceback"
    assert any("TIDE skipped" in r.getMessage() for r in caplog.records)
