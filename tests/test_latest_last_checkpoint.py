"""Integration test: LatestLastCheckpoint keeps last.ckpt at the LATEST epoch.

Stock PyTorch-Lightning only writes last.ckpt on epochs where a top-k save also
happens, so on a plateaued run last.ckpt freezes at the last top-k epoch. These
tests demonstrate that stock behavior (baseline) and that our subclass fixes it,
using a real Trainer with a scripted val metric that peaks mid-run.
"""
from __future__ import annotations

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from training_toolkit.callbacks.checkpointing import LatestLastCheckpoint

# val/metric rises to a peak at epoch index 2, then falls — so epochs 3 & 4
# never crack the (save_top_k=1) top-k and trigger the stock freeze.
SCORES = [0.1, 0.5, 0.9, 0.4, 0.3]


class _PlateauModel(pl.LightningModule):
    def __init__(self, scores: list[float]):
        super().__init__()
        self.layer = torch.nn.Linear(4, 2)
        self._scores = scores

    def training_step(self, batch, _):
        x, y = batch
        return torch.nn.functional.mse_loss(self.layer(x), y)

    def validation_step(self, batch, _):
        return None

    def on_validation_epoch_end(self):
        self.log("val/metric", float(self._scores[self.current_epoch]))

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def _mk(cls, tmp_path):
    return cls(
        dirpath=str(tmp_path), monitor="val/metric", mode="max",
        save_top_k=1, save_last=True, filename="epoch={epoch:03d}",
        auto_insert_metric_name=False,
    )


def _fit(ckpt, tmp_path):
    ds = TensorDataset(torch.randn(8, 4), torch.randn(8, 2))
    dl = DataLoader(ds, batch_size=4)
    trainer = pl.Trainer(
        max_epochs=len(SCORES), callbacks=[ckpt], logger=False,
        accelerator="cpu", devices=1, num_sanity_val_steps=0,
        enable_progress_bar=False, enable_model_summary=False,
        limit_train_batches=2, limit_val_batches=1,
    )
    trainer.fit(_PlateauModel(SCORES), dl, dl)
    # Self-created, trusted ckpt of a trivial model → weights_only=True is safe/sufficient.
    last = torch.load(tmp_path / "last.ckpt", map_location="cpu", weights_only=True)
    return last["epoch"], ckpt.best_model_path


def test_stock_last_ckpt_freezes_at_last_topk(tmp_path):
    """Baseline: stock ModelCheckpoint's last.ckpt freezes at the peak epoch."""
    ep, best = _fit(_mk(ModelCheckpoint, tmp_path), tmp_path)
    assert "epoch=002" in best            # best kept = the peak
    assert ep == 2                        # last.ckpt stuck at the last top-k epoch, NOT 4


def test_latest_last_ckpt_tracks_latest_epoch(tmp_path):
    """The fix: last.ckpt reflects the latest epoch even after the plateau."""
    ep, best = _fit(_mk(LatestLastCheckpoint, tmp_path), tmp_path)
    assert "epoch=002" in best            # top-k unchanged — best is still the peak
    assert ep == 4                        # last.ckpt now tracks the LATEST epoch
