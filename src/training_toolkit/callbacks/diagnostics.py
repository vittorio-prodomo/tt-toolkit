"""Training diagnostics callback: results CSV, training curves, val visualizations."""

import csv
import logging
import sys
from pathlib import Path

import pytorch_lightning as L
import torch

from training_toolkit.visualization import (
    build_val_overlay_payload,
    save_confusion_matrix,
    save_f1_curve,
    save_p_curve,
    save_pr_curve,
    save_r_curve,
    save_training_curves,
    save_val_comparison_grid,
)

logger = logging.getLogger(__name__)

# Cap on interactive val overlays pushed to W&B per logging event (bounds the
# payload; the on-disk comparison grids are written separately and unaffected).
_MAX_OVERLAY_IMAGES = 16


def _curve_kwargs(pr_curves: dict) -> dict:
    """Extract the kwargs needed by the save_*_curve functions."""
    return {
        "precision": pr_curves["precision"],
        "scores": pr_curves["scores"],
        "rec_thresholds": pr_curves["rec_thresholds"],
        "class_names": pr_curves["class_names"],
    }


class DiagnosticsCallback(L.Callback):
    """Generates periodic training artifacts.

    - results.csv: one row per epoch with all logged scalars.
    - results.png: dual-axis loss + mAP plot (every ``period`` epochs).
    - val_batch_N.png: GT-vs-prediction comparison (every ``period`` epochs).
    - (optional) W&B image logging of val comparisons.
    """

    def __init__(
        self,
        run_dir: Path,
        period: int = 5,
        num_val_batches: int = 3,
        class_names: dict[int, str] | None = None,
        log_images: bool = False,
    ) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.period = period
        self.num_val_batches = num_val_batches
        self.class_names = class_names
        self.log_images = log_images

        self._csv_path = self.run_dir / "results.csv"
        self._csv_columns: list[str] | None = None
        self._pr_curves_written_epoch: int = -1
        self._tide_written_epoch: int = -1

    # ------------------------------------------------------------------
    # Val batch capture + CSV + periodic artifacts
    # ------------------------------------------------------------------

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        # Collect all logged metrics for this epoch
        epoch_metrics: dict[str, float] = {"epoch": int(trainer.current_epoch)}
        for k, v in trainer.callback_metrics.items():
            if isinstance(v, torch.Tensor):
                epoch_metrics[k] = v.item()

        if trainer.is_global_zero:      # results.csv is a rank-0 artifact
            self._write_csv_row(epoch_metrics)

        epoch = trainer.current_epoch
        is_periodic = (epoch + 1) % self.period == 0
        is_last = trainer.max_epochs is not None and epoch == trainer.max_epochs - 1

        if is_periodic or is_last:
            # Training curves
            try:
                save_training_curves(
                    self._csv_path, self.run_dir / "results.png",
                )
            except Exception:
                logger.warning("Could not generate training curves", exc_info=True)

            # Val batch GT-vs-prediction visualizations
            self._save_val_visualizations(trainer, pl_module)

            # PR / confidence curve plots (only if metric improved since last write)
            self._save_pr_curves(pl_module)

            # TIDE error-analysis artifacts (only if val_tide was enabled)
            self._save_tide(pl_module)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, metrics: dict[str, float]) -> None:
        """Upsert this epoch's row into results.csv.

        Keyed by epoch, so it is robust to ``--resume`` (which re-instantiates this
        callback on the same run_dir and re-validates epochs): every prior row is
        preserved, columns are unioned, rows stay sorted by epoch, and a blank/None
        new value never overwrites an already-populated cell. This read-modify-write
        replaces the old append + header-rewrite, which on resume duplicated rows and
        blanked earlier epochs' metrics (only recoverable from W&B afterwards).
        """
        try:
            epoch = int(metrics.get("epoch", -1))
        except (TypeError, ValueError):
            epoch = -1

        # Load existing rows (keyed by epoch) + column order from disk.
        existing: dict[int, dict] = {}
        cols: list[str] = ["epoch"]
        if self._csv_path.exists():
            with open(self._csv_path, newline="") as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or ["epoch"])
                for r in reader:
                    try:
                        existing[int(float(r["epoch"]))] = dict(r)
                    except (TypeError, ValueError, KeyError):
                        continue

        # Union columns: keep the existing order, append any new keys (sorted, stable).
        cols = cols + sorted(k for k in metrics if k not in cols)

        # Upsert this epoch; a blank/None new value must not clobber a populated cell.
        row = dict(existing.get(epoch, {}))
        row["epoch"] = epoch
        for k, v in metrics.items():
            if v is None or v == "":
                row.setdefault(k, "")
            else:
                row[k] = v
        existing[epoch] = row

        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for e in sorted(existing):
                writer.writerow({c: existing[e].get(c, "") for c in cols})

        self._csv_columns = cols

    # ------------------------------------------------------------------
    # Val visualizations
    # ------------------------------------------------------------------

    def _save_val_visualizations(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        val_batches = getattr(pl_module, "_captured_val_batches", [])
        val_preds = getattr(pl_module, "_captured_val_preds", [])

        saved_paths: list[Path] = []
        for i, (batch, preds) in enumerate(zip(val_batches, val_preds)):
            if i >= self.num_val_batches:
                break
            path = self.run_dir / f"val_batch_{i}.png"
            save_val_comparison_grid(
                batch,
                preds,
                path,
                class_names=self.class_names,
            )
            saved_paths.append(path)

        if saved_paths:
            sys.stdout.write("\n")  # newline after progress bar
            sys.stdout.flush()
            logger.debug("Saved %d val comparison images to %s", len(saved_paths), self.run_dir)

        if saved_paths and self.log_images:
            self._log_images_to_wandb(
                trainer,
                val_batches[: self.num_val_batches],
                val_preds[: self.num_val_batches],
            )

    # ------------------------------------------------------------------
    # PR / Confidence curve plots
    # ------------------------------------------------------------------

    def _save_pr_curves(self, pl_module: L.LightningModule) -> None:
        """Write the 4 PR/confidence curve plots + confusion matrix if stash is newer."""
        pr_curves = getattr(pl_module, "_pr_curves", None)
        if pr_curves is None:
            return
        stash_epoch = pr_curves.get("epoch", -1)
        if stash_epoch <= self._pr_curves_written_epoch:
            return

        try:
            save_pr_curve(path=self.run_dir / "PR_curve.png", **_curve_kwargs(pr_curves))
            save_p_curve(path=self.run_dir / "P_curve.png", **_curve_kwargs(pr_curves))
            save_r_curve(path=self.run_dir / "R_curve.png", **_curve_kwargs(pr_curves))
            save_f1_curve(path=self.run_dir / "F1_curve.png", **_curve_kwargs(pr_curves))
            self._pr_curves_written_epoch = stash_epoch
            logger.debug("Saved PR/confidence curve plots (epoch %d)", stash_epoch)
        except Exception:
            logger.warning("Could not generate PR curve plots", exc_info=True)

        # Confusion matrix (stashed alongside PR curves)
        cm = getattr(pl_module, "_confusion_matrix", None)
        if cm is not None:
            try:
                save_confusion_matrix(
                    matrix=cm["matrix"],
                    class_names=cm["class_names"],
                    path=self.run_dir / "confusion_matrix.png",
                )
                logger.debug("Saved confusion matrix (epoch %d)", cm.get("epoch", -1))
            except Exception:
                logger.warning("Could not generate confusion matrix", exc_info=True)

        # Persist raw curve data for post-training analysis
        raw: dict = {}
        if pr_curves is not None:
            raw["pr_curves"] = pr_curves
        if cm is not None:
            raw["confusion_matrix"] = cm
        if raw:
            try:
                torch.save(raw, self.run_dir / "curve_data.pt")
                logger.debug("Saved raw curve data to curve_data.pt")
            except Exception:
                logger.warning("Could not save raw curve data", exc_info=True)

    def _save_tide(self, pl_module: L.LightningModule) -> None:
        """Flush TIDE artifacts to the run dir if the stash is newer.

        Writes ``tide_summary.txt`` (tidecv's built-in summarize() output),
        ``tide_per_class.txt`` (our custom-built per-class breakdown), and
        the two PNG plots tidecv generates (main and special errors).
        """
        tide_results = getattr(pl_module, "_tide_results", None)
        if tide_results is None:
            return
        stash_epoch = tide_results.get("epoch", -1)
        if stash_epoch <= self._tide_written_epoch:
            return

        try:
            from training_toolkit.metrics import TideAnalyzer

            (self.run_dir / "tide_summary.txt").write_text(
                tide_results.get("summary_text", ""),
            )
            (self.run_dir / "tide_per_class.txt").write_text(
                TideAnalyzer.format_per_class_table(tide_results),
            )
            tide_obj = tide_results.get("tide")
            if tide_obj is not None:
                try:
                    tide_obj.plot(str(self.run_dir))
                except Exception:
                    logger.warning("Could not generate TIDE PNG plots", exc_info=True)
            # Persist the result dict (sans the live TIDE object) for
            # post-training analysis. Matches the curve_data.pt pattern.
            serializable = {k: v for k, v in tide_results.items() if k != "tide"}
            torch.save(serializable, self.run_dir / "tide_data.pt")
            self._tide_written_epoch = stash_epoch
            logger.debug("Saved TIDE artifacts (epoch %d)", stash_epoch)
        except Exception:
            logger.warning("Could not write TIDE artifacts", exc_info=True)

    def on_fit_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule,
    ) -> None:
        """Final plot generation — covers normal completion and Ctrl+C."""
        # Always generate training curves one last time
        try:
            save_training_curves(self._csv_path, self.run_dir / "results.png")
        except Exception:
            logger.warning("Could not generate final training curves", exc_info=True)

        # PR curves: write if stash is newer than last written
        self._save_pr_curves(pl_module)

        # TIDE error-analysis artifacts: same write-if-newer semantics
        self._save_tide(pl_module)

    # ------------------------------------------------------------------
    # W&B image logging
    # ------------------------------------------------------------------

    def _log_images_to_wandb(
        self,
        trainer: L.Trainer,
        val_batches: list[dict],
        val_preds: list[list[dict]],
    ) -> None:
        """Log interactive GT + prediction mask overlays to W&B, if active.

        Uses W&B's ``wandb.Image(rgb, masks=...)`` overlay API so the dashboard
        renders toggle-able ground-truth / prediction mask layers with per-class
        filtering. Reuses the already-captured val predictions (no extra
        inference); the overlay count is bounded by ``_MAX_OVERLAY_IMAGES``. The
        on-disk comparison grids are written separately and are unaffected.
        """
        try:
            from pytorch_lightning.loggers import WandbLogger
            import wandb
        except ImportError:
            logger.debug("wandb not installed — skipping image logging")
            return

        wandb_logger = None
        for lg in trainer.loggers:
            if isinstance(lg, WandbLogger):
                wandb_logger = lg
                break

        if wandb_logger is None:
            logger.warning(
                "log_images is enabled but no WandbLogger found — "
                "add 'wandb' to your loggers list"
            )
            return

        # Never let a visualization error take down training — degrade to no-op.
        try:
            images = []
            for batch, preds in zip(val_batches, val_preds):
                for img_t, target, pred in zip(batch["images"], batch["targets"], preds):
                    if len(images) >= _MAX_OVERLAY_IMAGES:
                        break
                    rgb, masks = build_val_overlay_payload(
                        img_t, target, pred, class_names=self.class_names,
                    )
                    images.append(wandb.Image(rgb, masks=masks))
                if len(images) >= _MAX_OVERLAY_IMAGES:
                    break

            if images:
                wandb_logger.experiment.log(
                    {"val/comparisons": images},
                    step=trainer.global_step,
                )
        except Exception:
            logger.warning(
                "W&B val overlay logging failed — skipping this epoch's overlays",
                exc_info=True,
            )
