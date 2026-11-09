"""Per-class best-checkpoint callback with hardlink dedup.

Tracks every ``val/map_<class>`` metric (one per class, emitted by the
lit_module) and writes a separate checkpoint per class on improvement.

Multiple classes that improve on the same epoch share a single inode via
``os.link()`` — so disk footprint is bounded by the number of distinct
improvement epochs, not by the number of classes. Same pattern as the
SAM 3 fork's ``save_best_meters`` hardlink dedup.

Opt-in via ``save_best_per_class: true`` in TrainConfig. Default off.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
from pathlib import Path

from pytorch_lightning import Callback, LightningModule, Trainer

logger = logging.getLogger(__name__)


def _sanitize(name: str) -> str:
    """Convert a class name to a filesystem-safe stem."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
    return s or "unnamed"


class PerClassBestCheckpointCallback(Callback):
    """Save best checkpoints per per-class val metric, with hardlink dedup.

    Parameters
    ----------
    dirpath:
        Run's ``checkpoints/`` directory. A ``best_per_class/`` subfolder
        is created for the per-class files.
    metric_prefix:
        Prefix the lit_module uses to log per-class metrics.
        Default ``"val/map_"``; anything starting with this prefix is tracked.
    verbose:
        Log at INFO level whenever a class's best is updated.
    """

    def __init__(
        self,
        dirpath: str | Path,
        metric_prefix: str = "val/map_",
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath) / "best_per_class"
        self.metric_prefix = metric_prefix
        self.verbose = verbose
        self._bests: dict[str, float] = {}

    def _path_for(self, class_metric: str) -> Path:
        name = class_metric[len(self.metric_prefix):]
        return self.dirpath / f"best_{_sanitize(name)}.ckpt"

    def on_validation_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        # MUST be on_validation_end, NOT on_validation_epoch_end: the latter fires
        # before the lit_module computes val/map_<class> and before callback_metrics
        # is flushed, so it would read the PREVIOUS epoch's per-class values. This
        # hook fires after the flush — the same one ModelCheckpoint uses.
        if trainer.sanity_checking:
            return

        # NOTE: do NOT early-return on non-global-zero ranks here. The
        # save_checkpoint() below is a DDP *collective* (it barriers across all
        # ranks); skipping it on rank>0 desyncs the process group at the next
        # collective (e.g. EarlyStopping's reduce). callback_metrics is
        # DDP-synced, so every rank derives identical candidates.
        candidates: list[tuple[str, float]] = []
        for key, val in trainer.callback_metrics.items():
            if not key.startswith(self.metric_prefix) or key == self.metric_prefix:
                continue
            try:
                v = float(val)
            except (TypeError, ValueError):
                continue
            if math.isnan(v) or math.isinf(v) or v < 0:
                # -1 is the torchmetrics sentinel for "no data this epoch"
                continue
            prior = self._bests.get(key, float("-inf"))
            if v > prior:
                candidates.append((key, v))

        if not candidates:
            return

        # Update tracked bests on EVERY rank so cross-epoch candidate sets stay
        # identical across ranks and the saved callback state_dict is correct.
        for key, val in candidates:
            self._bests[key] = val

        # Save one real checkpoint this epoch; hardlink the rest to it.
        first_key, first_val = candidates[0]
        first_path = self._path_for(first_key)

        # Rank-0-only filesystem prep (no collectives).
        if trainer.is_global_zero:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            if first_path.exists() or first_path.is_symlink():
                first_path.unlink()

        # COLLECTIVE: save_checkpoint() barriers across ranks, so EVERY rank
        # must call it (Lightning writes the file on rank 0 only).
        trainer.save_checkpoint(str(first_path))

        # Past the collective: the remaining work is rank-0-only file I/O.
        if not trainer.is_global_zero:
            return

        if self.verbose:
            logger.info(
                "[per-class-best] %s improved -> %.4f (saved %s)",
                first_key, first_val, first_path.name,
            )

        for key, val in candidates[1:]:
            path = self._path_for(key)
            if path.exists() or path.is_symlink():
                path.unlink()
            try:
                os.link(first_path, path)
            except OSError as e:
                # Cross-device or filesystem without hardlink support.
                shutil.copy2(first_path, path)
                logger.warning(
                    "[per-class-best] hardlink failed (%s); copied instead", e,
                )
            if self.verbose:
                logger.info(
                    "[per-class-best] %s improved -> %.4f (hardlinked %s)",
                    key, val, path.name,
                )

    # ------------------------------------------------------------------
    # State persistence for resume

    def state_dict(self) -> dict:
        return {"bests": dict(self._bests)}

    def load_state_dict(self, state_dict: dict) -> None:
        self._bests = dict(state_dict.get("bests", {}))
