"""Config-driven Lightning Trainer wrapper."""

import logging
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import pytorch_lightning as L
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.logger import Logger as LightningLogger
from pytorch_lightning.profilers import AdvancedProfiler, PyTorchProfiler, SimpleProfiler
from torch.utils.data import DataLoader

from training_toolkit.callbacks import build_early_stopping, build_checkpointing, CloseMosaicCallback, DiagnosticsCallback, DivergenceDetector, PerClassBestCheckpointCallback, select_progress_callback
from training_toolkit.configs import TrainConfig
from training_toolkit.data import collate_fn
from training_toolkit.data.base_dataset import BaseSegmentationDataset
from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.label_crop import SizeBand
from training_toolkit.data.yolo_dataset import YoloSegmentationDataset
from training_toolkit.data.transforms import build_transforms
from training_toolkit.engine.lit_module import SegmentationLitModule
from training_toolkit.logging import setup_logging, tame_third_party_loggers
from training_toolkit.models import get_adapter
from training_toolkit.visualization import (
    save_batch_grid,
    save_labels_summary,
    save_labels_correlogram,
)

logger = logging.getLogger(__name__)


def _streaming_worker_init(worker_id: int) -> None:
    """Single-thread libvips per DataLoader worker to avoid N_workers ×
    libvips_threads CPU oversubscription. pyvips reads VIPS_CONCURRENCY at
    libvips init and has no public concurrency_set() in 3.x."""
    import os
    os.environ["VIPS_CONCURRENCY"] = "1"


def _resolve_batch_limits(config, has_val):
    """Resolve Lightning's (limit_train_batches, limit_val_batches) from config.

    `config.limit_train_batches` (an absolute integer batch cap) takes precedence
    over `config.fraction` (a float ratio) and, like fraction, governs BOTH train
    and val. When neither is set the epoch runs in full (None). `has_val=False`
    disables validation (0). Unlike a float fraction, the int cap reaches the epoch
    boundary on the full pool without shrinking it — the shared-pool repro knob.

    `config.limit_val_batches`, when set, overrides the val side INDEPENDENTLY of the
    train-coupled default (int = batches, float in (0, 1] = fraction, 1.0 = full val) —
    e.g. short train + full val. It is moot when `has_val=False` (val stays disabled).
    """
    limit = config.limit_train_batches
    if limit is not None:
        if config.fraction < 1.0:
            logger.warning(
                "Both --limit-train-batches (%d) and --fraction (%.3f) set; "
                "using --limit-train-batches and ignoring --fraction.",
                limit, config.fraction,
            )
        train_limit = val_limit = limit
    else:
        f = config.fraction
        train_limit = val_limit = f if f < 1.0 else None
    if not has_val:
        return train_limit, 0
    if config.limit_val_batches is not None:
        val_limit = config.limit_val_batches  # independent val override
    return train_limit, val_limit


def _needs_pyvips_spawn(dataset) -> bool:
    """True when a (non-streaming) loader's dataset decodes via pyvips ROI and so
    needs spawn workers — libvips + fork DataLoader workers deadlock. The offline
    TiledSegmentationDataset (val, and the offline-train fallback) now decodes via
    pyvips; the streaming dataset is handled separately via `is_stream`."""
    from training_toolkit.data.tiled_dataset import TiledSegmentationDataset
    return isinstance(dataset, TiledSegmentationDataset)


class SetEpochCallback(L.Callback):
    """Propagate the epoch to a streaming dataset's shared-memory counter so
    persistent spawn workers reshuffle each epoch."""
    def __init__(self, dataset) -> None:
        self._dataset = dataset

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        if hasattr(self._dataset, "set_epoch"):
            self._dataset.set_epoch(trainer.current_epoch)


class SharedPoolEpochCleanupCallback(L.Callback):
    """Lag-by-one shared-pool cleanup: at the start of epoch K (K>0), unlink the
    PREVIOUS epoch's (K-1) /dev/shm segments. Lightning fully completes epoch K-1
    (train + val, DDP-synced) before any rank enters epoch K, so K-1's epoch-scoped
    segments are provably unreferenced — no barrier/refcount needed. Runs on
    local_rank 0 only (segments are host-scoped). The run-end sweep in train()
    remains the backstop for the final epoch and crash leftovers."""

    def __init__(self, run_id: str) -> None:
        self._run_id = run_id

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if epoch > 0 and getattr(trainer, "local_rank", 0) == 0:
            _cleanup_shared_pool(self._run_id, epoch=epoch - 1)


def train(
    config: TrainConfig,
    resume_from: str | None = None,
    profiler: str | None = None,
    extra_callbacks: list | None = None,
) -> float | None:
    """Run a full training pipeline from config.

    This is the main entry point: it sets up logging, data, model, callbacks,
    and starts Lightning training.

    Args:
        config: Training configuration.
        resume_from: Path to a Lightning ``.ckpt`` file, or ``"last"`` to
            auto-find the most recent checkpoint in the run directory.
        profiler: Profiler type — ``"simple"`` (per-hook timing),
            ``"advanced"`` (per-function via cProfile), or ``"pytorch"``
            (GPU kernels, Chrome trace). *None* disables profiling.
        extra_callbacks: Additional Lightning callbacks (e.g. Optuna pruning).

    Returns:
        The best value of ``config.monitor_metric`` after training, or None
        if no validation was performed.
    """
    # Use Tensor Cores for float32 matmuls (free speedup, negligible precision loss)
    torch.set_float32_matmul_precision("high")

    # Output directory (skip auto-increment when resuming)
    run_dir = _resolve_run_dir(config, resuming=bool(resume_from))

    # Logging
    setup_logging(run_dir=run_dir, log_to_file=config.log_to_file)
    logger.info("[bold]Training Toolkit v0.1.0[/bold]", extra={"markup": True})

    # Seed
    if config.seed is not None:
        L.seed_everything(config.seed, workers=True)

    # Model adapter (created early so we can validate imgsz)
    adapter = get_adapter(config.model)

    # Resolve --pretrained run directory to actual weights path
    _resolve_pretrained(config, adapter, logger)

    # Auto-detect imgsz if not set, then validate divisibility
    _resolve_imgsz(config, adapter, logger)

    # Resolve gradient clipping: use adapter default if not set by user
    _resolve_gradient_clip(config, adapter, logger)

    # Resolve precision: adapters may enforce a backend requirement
    # (Mask2Former forces fp16 — it diverges to NaN under bf16-mixed).
    config.precision = adapter.resolve_precision(config.precision)

    # Data
    train_transform_result = build_transforms(config.augmentation, is_train=True)
    val_transform = build_transforms(config.augmentation, is_train=False)

    if isinstance(train_transform_result, tuple):
        train_transform, mosaic_transform = train_transform_result
    else:
        train_transform = train_transform_result
        mosaic_transform = None

    train_dataset = _create_dataset(
        config, config.data.train_split, train_transform, config.augmentation,
        mosaic_transform=mosaic_transform,
    )

    from torch.utils.data import IterableDataset
    is_stream = isinstance(train_dataset, IterableDataset)

    val_dataset = _try_create_dataset(config, config.data.val_split, val_transform)
    has_val = val_dataset is not None
    if not has_val:
        logger.warning(
            "[bold yellow]No validation split found — training without "
            "validation metrics or early stopping. If this is unintended, "
            "check that your dataset contains a '%s' split.[/bold yellow]",
            config.data.val_split, extra={"markup": True},
        )

    # Pre-train dim-sanity: fail fast if any image's declared dims disagree with its
    # decoded dims or it still carries a rotating EXIF tag (a cv2/pyvips decoder split
    # silently drops tiles — seen as a real first-validation crash). Header-level, rank 0
    # only, before loaders/model. Skippable via config.data.dim_sanity_check.
    if config.data.dim_sanity_check and int(os.environ.get("LOCAL_RANK", 0)) == 0:
        from training_toolkit.data.dim_sanity import (
            assert_declared_dims,
            records_from_dataset,
        )

        recs = records_from_dataset(train_dataset)
        if val_dataset is not None:
            recs += records_from_dataset(val_dataset)
        assert_declared_dims(recs)
        logger.info("Dim-sanity check passed for %d images", len(recs))

    # Auto-detect num_classes from dataset if not set
    if config.model.num_classes is None:
        config.model.num_classes = train_dataset.num_classes
        logger.info("Auto-detected %d classes: %s", config.model.num_classes, train_dataset.class_names)

    # Save resolved config (after imgsz correction and num_classes auto-detection)
    config.to_yaml(run_dir / "config.yaml")
    logger.info("Config saved to %s", run_dir / "config.yaml")

    nw = config.data.workers
    train_loader_kwargs = dict(
        batch_size=config.data.batch_size,
        num_workers=nw,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=nw > 0,
        prefetch_factor=config.data.prefetch_factor if nw > 0 else None,
    )
    if is_stream:
        if nw > 0:
            train_loader_kwargs["multiprocessing_context"] = "spawn"
            train_loader_kwargs["worker_init_fn"] = _streaming_worker_init
    else:
        train_loader_kwargs["shuffle"] = True
        # Offline tiled train path also decodes via pyvips ROI -> spawn workers.
        if nw > 0 and _needs_pyvips_spawn(train_dataset):
            train_loader_kwargs["multiprocessing_context"] = "spawn"
            train_loader_kwargs["worker_init_fn"] = _streaming_worker_init
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)

    val_loader = None
    if has_val:
        val_loader_kwargs = dict(
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=nw,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=nw > 0,
            prefetch_factor=config.data.prefetch_factor if nw > 0 else None,
        )
        # The offline TiledSegmentationDataset decodes via pyvips ROI; libvips +
        # fork DataLoader workers deadlock, so use spawn (mirrors the streaming
        # train loader). Val is this dataset whenever tiling is enabled.
        if nw > 0 and _needs_pyvips_spawn(val_dataset):
            val_loader_kwargs["multiprocessing_context"] = "spawn"
            val_loader_kwargs["worker_init_fn"] = _streaming_worker_init
        val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    if has_val:
        logger.info(
            "Dataset: %d train images, %d val images, batch_size=%d",
            len(train_dataset), len(val_dataset), config.data.batch_size,
        )
    else:
        logger.info(
            "Dataset: %d train images (no val split), batch_size=%d",
            len(train_dataset), config.data.batch_size,
        )

    # Pre-training analysis: batch samples + dataset statistics plots
    # Guard with rank check — DDP re-launches the script per GPU, only rank 0 needs this.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if config.analysis and local_rank == 0:
        train_iter = iter(train_loader)
        n_samples = min(3, len(train_loader))
        for i in range(n_samples):
            sample_batch = next(train_iter)
            save_batch_grid(
                sample_batch,
                run_dir / f"train_batch_{i}.png",
                class_names=train_dataset.class_names,
            )
        logger.info("Saved %d augmented batch samples to %s", n_samples, run_dir)

        from training_toolkit.data.tiled_dataset import TiledSegmentationDataset

        stats_dataset = (
            train_dataset._base
            if isinstance(train_dataset, TiledSegmentationDataset)
            else train_dataset
        )
        save_labels_summary(
            stats_dataset, run_dir / "labels.jpg",
            class_names=train_dataset.class_names,
        )
        save_labels_correlogram(stats_dataset, run_dir / "labels_correlogram.jpg")
        logger.info("Saved dataset analysis plots to %s", run_dir)

    lit_module = SegmentationLitModule(
        adapter=adapter,
        config=config,
        class_names=train_dataset.class_names,
    )

    if config.compile:
        logger.info("Compiling model with torch.compile() (first step will be slow)...")
        lit_module.model = torch.compile(lit_module.model)

    # Re-tame after model imports (RF-DETR adds its own handlers on import)
    tame_third_party_loggers()

    # Shared-pool flag — hoisted here so it's available for both the callbacks block
    # below and the lifecycle management around trainer.fit.
    _is_shared_pool = is_stream and config.data.tiling.tile_source == "shared_pooled"

    # Callbacks
    progress_cb, enable_pbar = select_progress_callback(sys.stdout.isatty())
    callbacks = [progress_cb, DivergenceDetector(monitor=config.monitor_metric)]
    if has_val:
        callbacks.append(build_early_stopping(config))
        callbacks.append(build_checkpointing(config))
        if config.save_best_per_class:
            ckpt_dir = Path(config.project_dir) / config.run_name / "checkpoints"
            callbacks.append(PerClassBestCheckpointCallback(dirpath=ckpt_dir))
    else:
        callbacks.append(build_checkpointing(config, train_only=True))

    if config.diagnostics_period > 0:
        callbacks.append(
            DiagnosticsCallback(
                run_dir=run_dir,
                period=config.diagnostics_period,
                num_val_batches=3,
                class_names=train_dataset.class_names,
                log_images=config.log_images,
            )
        )

    if config.augmentation.close_mosaic > 0:
        close_epoch = max(config.epochs - config.augmentation.close_mosaic, 0)
        callbacks.append(CloseMosaicCallback(close_epoch=close_epoch))

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    # Loggers (reconnect to existing runs when resuming)
    logger_meta = _load_logger_metadata(run_dir) if resume_from else {}
    loggers = _build_loggers(config, run_dir, resume_meta=logger_meta)
    _save_logger_metadata(loggers, run_dir)

    # Profiler
    pl_profiler = _build_profiler(profiler, run_dir) if profiler else None

    # Trainer
    train_limit, val_limit = _resolve_batch_limits(config, has_val)
    grad_accum = _compute_grad_accum(config)
    accum_scheduler = _build_accum_scheduler(grad_accum, config.warmup_epochs)
    if accum_scheduler is not None:
        callbacks.append(accum_scheduler)

    if is_stream:
        callbacks.append(SetEpochCallback(train_dataset))
        if _is_shared_pool:
            callbacks.append(SharedPoolEpochCleanupCallback(config.run_name))

    # Use find_unused_parameters for multi-GPU — models like RF-DETR have
    # parameters not used in every forward pass (e.g. auxiliary heads).
    multi_gpu = isinstance(config.device, list) and len(config.device) > 1
    strategy = "ddp_find_unused_parameters_true" if multi_gpu else "auto"

    trainer = L.Trainer(
        max_epochs=config.epochs,
        accelerator="auto",
        devices=config.device,
        strategy=strategy,
        precision=config.precision,
        gradient_clip_val=config.gradient_clip_val,
        accumulate_grad_batches=1 if accum_scheduler else grad_accum,
        callbacks=callbacks,
        logger=loggers,
        profiler=pl_profiler,
        deterministic=False,
        enable_model_summary=False,
        enable_progress_bar=enable_pbar,
        log_every_n_steps=10,
        limit_train_batches=train_limit,
        limit_val_batches=val_limit,
    )

    total_params = sum(p.numel() for p in lit_module.parameters())
    trainable = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)
    logger.info(
        "Model params: %.1f M trainable, %.1f M total",
        trainable / 1e6, total_params / 1e6,
    )

    # Resolve resume checkpoint
    ckpt_path = None
    if resume_from:
        if resume_from.lower() == "last":
            last = Path(config.project_dir) / config.run_name / "checkpoints" / "last.ckpt"
            if last.exists():
                ckpt_path = str(last)
                logger.info("Resuming from last checkpoint: %s", ckpt_path)
            else:
                logger.warning("No last.ckpt found in %s — starting fresh", last.parent)
        else:
            ckpt_path = resume_from
            logger.info("Resuming from checkpoint: %s", ckpt_path)

    logger.info("[bold]Starting training: %d epochs[/bold]", config.epochs, extra={"markup": True})

    # Shared-pool lifecycle: sweep stale segments before fit, unlink cleanly after.
    if _is_shared_pool and local_rank == 0:
        from training_toolkit.data.shared_pool_lifecycle import sweep_stale
        swept = sweep_stale()
        if swept:
            logger.info("shared-pool startup sweep: unlinked stale segments %s", swept)
        # Also clear THIS run's stale pixel segments — sweep_stale() only covers
        # control blocks (tt_pool_* prefix).  If a prior run with the same
        # run_name hard-crashed (kill -9 / OOM), its pixel segments
        # ({run_id}_ep*_{host}_img*) are left behind.  SharedImagePool.try_admit
        # calls SharedMemory(create=True) with no FileExistsError handling, so a
        # stale segment for the same key raises → DataLoader worker death → DDP desync.
        _cleanup_shared_pool(config.run_name)

    try:
        trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    finally:
        if _is_shared_pool:
            _cleanup_shared_pool(config.run_name)

    # Load best checkpoint weights (not last epoch) for export
    best_ckpt_path = getattr(trainer.checkpoint_callback, "best_model_path", "") if has_val else ""
    if best_ckpt_path:
        logger.info("Loading best checkpoint for export: %s", Path(best_ckpt_path).name)
        best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
        lit_module.load_state_dict(best_ckpt["state_dict"])

    # Export model in backend-native format for later fine-tuning
    adapter.export_model(lit_module.model, run_dir)

    logger.info("[bold green]Training complete![/bold green]", extra={"markup": True})

    # Return best monitored metric (used by Optuna sweeps)
    best = trainer.callback_metrics.get(config.monitor_metric)
    return float(best) if best is not None else None


def _resolve_imgsz(
    config: TrainConfig,
    adapter: object,
    log: logging.Logger,
) -> None:
    """Auto-detect imgsz if not set, then round to the adapter's divisor."""
    model_name = config.model.name

    # Auto-detect from adapter if not explicitly set
    if config.augmentation.imgsz is None:
        get_default = getattr(adapter, "get_default_imgsz", None)
        config.augmentation.imgsz = get_default(model_name) if get_default else 512
        log.info("Auto-detected imgsz=%d for %s", config.augmentation.imgsz, model_name)

    # Align to divisor
    get_div = getattr(adapter, "get_imgsz_divisor", None)
    divisor = get_div(model_name) if get_div else getattr(adapter, "IMGSZ_DIVISOR", 1)
    if divisor <= 1:
        return

    imgsz = config.augmentation.imgsz
    if imgsz % divisor == 0:
        return

    aligned = (imgsz // divisor) * divisor
    log.warning(
        "imgsz=%d is not divisible by %d (required by %s backbone). "
        "Rounding down to %d.",
        imgsz,
        divisor,
        type(adapter).__name__,
        aligned,
    )
    config.augmentation.imgsz = aligned


def _resolve_gradient_clip(
    config: TrainConfig,
    adapter: object,
    log: logging.Logger,
) -> None:
    """Use adapter-specific gradient clip default when not set by user."""
    if config.gradient_clip_val is not None:
        log.info("Gradient clipping: max_norm=%.4g (user-specified)", config.gradient_clip_val)
        return

    get_default = getattr(adapter, "get_default_gradient_clip_val", None)
    default = get_default(config.model.name) if get_default else None
    if default is not None:
        config.gradient_clip_val = default
        log.info(
            "Gradient clipping: max_norm=%.4g (default for %s)",
            default, type(adapter).__name__,
        )
    else:
        log.info("Gradient clipping: disabled (no adapter default)")


def _cleanup_shared_pool(run_id: str, epoch: int | None = None) -> None:
    """Best-effort cleanup of epoch-scoped control blocks + leftover pixel segments.

    When ``epoch`` is ``None`` (default), cleans all epochs' segments (used at
    run-start and run-end). When ``epoch`` is an int K, cleans only that epoch's
    segments (used by the lag-by-one callback to sweep epoch K-1 at the start of K).

    Called in the ``finally`` after ``trainer.fit`` on every rank's main process.
    Idempotent: each block unlinks independently; FileNotFoundError is swallowed.
    Control blocks use epoch-scoped names (tt_pool_{run_id}_ep*_{host}),
    so each epoch's block is independent and any stale ones are swept here.
    Pixel segments ({run_id}_ep*_{host}_img*) are also swept as a backstop.
    """
    import glob as _glob
    import os as _os
    import socket as _socket
    from multiprocessing import shared_memory as _shm_mod
    from training_toolkit.data.named_lock import NamedLock

    host = _socket.gethostname()
    ep_token = f"ep{epoch}" if epoch is not None else "ep*"

    # Sweep all epoch-scoped control blocks: tt_pool_{run_id}_ep*_{host}
    for path in _glob.glob(f"/dev/shm/tt_pool_{run_id}_{ep_token}_{host}"):
        name = _os.path.basename(path)
        if name.endswith("_lock"):
            continue
        try:
            shm = _shm_mod.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass  # already unlinked by another rank — idempotent
        except Exception:
            pass  # best-effort
        try:
            NamedLock.unlink_name("/" + name + "_lock")
        except Exception:
            pass

    # Sweep pixel segments: /dev/shm/{run_id}_ep*_{host}_img*
    for path in _glob.glob(f"/dev/shm/{run_id}_{ep_token}_{host}_img*"):
        try:
            seg_name = path.removeprefix("/dev/shm/")
            s = _shm_mod.SharedMemory(name=seg_name)
            s.close()
            s.unlink()
        except Exception:
            pass


def _resolve_run_dir(config: TrainConfig, resuming: bool = False) -> Path:
    """Determine the run directory, handling overwrite and auto-increment."""
    run_dir = Path(config.project_dir) / config.run_name

    if resuming or not run_dir.exists() or not any(run_dir.iterdir()):
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    if config.overwrite:
        shutil.rmtree(run_dir)
        logger.info("Overwrite enabled — cleared %s", run_dir)
    else:
        # Auto-increment: run → run_2 → run_3 → ...
        base = run_dir
        n = 2
        while run_dir.exists() and any(run_dir.iterdir()):
            run_dir = base.parent / f"{base.name}_{n}"
            n += 1
        config.run_name = run_dir.name

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_pretrained(
    config: TrainConfig,
    adapter: object,
    log: logging.Logger,
) -> None:
    """If --pretrained points to a run directory, resolve to the actual weights path."""
    pretrained = config.model.pretrained
    if not pretrained or pretrained in ("auto", "none"):
        return

    p = Path(pretrained)
    if not p.is_dir():
        return  # already a file path or HuggingFace hub ID — leave as-is

    export_name = getattr(adapter, "EXPORT_NAME", "model")
    weights_path = p / export_name
    if weights_path.exists():
        config.model.pretrained = str(weights_path)
        log.info("Resolved pretrained path: %s", weights_path)
    else:
        log.warning(
            "Directory %s does not contain '%s' — passing as-is",
            p, export_name,
        )


def _try_create_dataset(
    config: TrainConfig,
    split: str,
    transform: Any = None,
    aug_config: Any = None,
) -> CocoSegmentationDataset | YoloSegmentationDataset | None:
    """Like _create_dataset, but returns None if the split doesn't exist."""
    try:
        return _create_dataset(config, split, transform, aug_config)
    except (FileNotFoundError, ValueError):
        return None


def _create_dataset(
    config: TrainConfig,
    split: str,
    transform: Any = None,
    aug_config: Any = None,
    mosaic_transform: Any = None,
) -> BaseSegmentationDataset:
    """Instantiate the right dataset class based on config.data.layout.

    When ``config.data.tiling.enabled`` is True and this is a training
    split, the base dataset is wrapped in a :class:`TiledSegmentationDataset`.
    """
    from training_toolkit.data.tiled_dataset import TiledSegmentationDataset

    layout = config.data.layout
    tiling = config.data.tiling

    # ---- Build the base dataset ----

    if layout == "fiftyone":
        base = _create_fiftyone_dataset(config, split)
    elif layout == "yolo":
        base = YoloSegmentationDataset(
            root=config.data.dataset_path,
            split=split,
        )
    else:
        base = CocoSegmentationDataset(
            root=config.data.dataset_path,
            split=split,
            layout=layout,
        )

    # ---- Optionally wrap with tiling ----

    # Tile the train split (online stream or deterministic grid) AND the val
    # split (always the deterministic grid — never streaming — so val metrics
    # are computed at the same object scale as training and stay reproducible).
    if tiling.enabled and split in (config.data.train_split, config.data.val_split):
        if tiling.online and split == config.data.train_split:
            from training_toolkit.data.streaming_tiled_dataset import (
                StreamingTiledSegmentationDataset,
            )
            return StreamingTiledSegmentationDataset(
                base,
                tile_size=tiling.tile_size,
                tile_sizes=tiling.tile_sizes,
                regime_tile_sizes=tiling.regime_tile_sizes,
                overlap=tiling.overlap,
                negative_ratio=tiling.negative_ratio,
                min_annotation_overlap=tiling.min_annotation_overlap,
                buffer_size=tiling.buffer_size,
                base_seed=tiling.base_seed,
                batch_size=config.data.batch_size,
                num_workers=config.data.workers,
                transform=transform,
                aug_config=aug_config,
                size_band=(
                    SizeBand.for_imgsz(
                        aug_config.imgsz,
                        ceiling_frac=getattr(config, "size_band_ceiling_frac", 0.75),
                    )
                    if getattr(config, "size_band_ignore", False)
                    and aug_config is not None and aug_config.imgsz is not None
                    else None
                ),
                imgsz=(aug_config.imgsz if aug_config is not None else None),
                gt_guided_params={
                    "target_in_model": tiling.gt_guided_target_in_model,
                    "min_crop": tiling.gt_guided_min_crop,
                    "translation_jitter": tiling.gt_guided_translation_jitter,
                    "scale_jitter": tiling.gt_guided_scale_jitter,
                    "dedup_iou": tiling.gt_guided_dedup_iou,
                },
                giant_ceiling_frac=tiling.giant_ceiling_frac,
                whole_image_if_giant=tiling.whole_image_if_giant,
                tile_source=tiling.tile_source,
                cache_size=tiling.cache_size,
                cache_bytes=tiling.cache_bytes,
                mosaic_transform=mosaic_transform,
                pool_min_windows=tiling.pool_min_windows,
                run_id=(config.run_name if tiling.tile_source == "shared_pooled" else None),
                assignment_mode=tiling.assignment_mode,
                negative_window_blocklist=tiling.negative_window_blocklist,
            )
        return TiledSegmentationDataset(
            base,
            tile_size=tiling.tile_size,
            overlap=tiling.overlap,
            negative_ratio=tiling.negative_ratio,
            min_annotation_overlap=tiling.min_annotation_overlap,
            cache_size=tiling.cache_size,
            transform=transform,
            aug_config=aug_config,
            mosaic_transform=mosaic_transform,
            seed=config.seed,
            assignment_mode=tiling.assignment_mode,
        )

    # ---- Non-tiled: attach transforms to the base dataset ----

    base._transform = transform
    base._mosaic_transform = mosaic_transform
    if aug_config is not None:
        base._aug_config = aug_config
    return base


def _create_fiftyone_dataset(
    config: TrainConfig, split: str,
) -> "FiftyOneSegmentationDataset":
    """Load a FiftyOne dataset by name, optionally filtered by split tag."""
    import fiftyone as fo
    from training_toolkit.data.fiftyone_dataset import FiftyOneSegmentationDataset

    # dataset_path is used as the FiftyOne dataset name
    dataset_name = str(config.data.dataset_path)
    dataset = fo.load_dataset(dataset_name)

    # If the dataset has split tags, filter to the requested split
    if split and dataset.has_sample_field("tags"):
        tagged = dataset.match_tags(split)
        if len(tagged) > 0:
            dataset = tagged
            logger.info(
                "FiftyOne dataset '%s' filtered by tag '%s': %d samples",
                dataset_name, split, len(dataset),
            )

    # Auto-detect label field
    label_field = "detections"
    for candidate in ("detections", "ground_truth", "segmentations", "polylines"):
        if dataset.has_sample_field(candidate):
            label_field = candidate
            break

    return FiftyOneSegmentationDataset(
        fo_dataset=dataset,
        label_field=label_field,
    )


def _build_profiler(name: str, run_dir: Path):
    """Build a Lightning profiler by name."""
    dirpath = str(run_dir)
    if name == "simple":
        logger.info("SimpleProfiler enabled → %s/profiler.txt", run_dir)
        return SimpleProfiler(dirpath=dirpath, filename="profiler")
    elif name == "advanced":
        logger.info("AdvancedProfiler enabled → %s/profiler.txt", run_dir)
        return AdvancedProfiler(dirpath=dirpath, filename="profiler")
    elif name == "pytorch":
        logger.info("PyTorchProfiler enabled → %s/pytorch_profiler.json", run_dir)
        return PyTorchProfiler(
            dirpath=dirpath,
            filename="pytorch_profiler",
            export_to_chrome=True,
        )
    else:
        raise ValueError(f"Unknown profiler: {name!r} (expected simple/advanced/pytorch)")


def _compute_grad_accum(config: TrainConfig) -> int:
    """Derive gradient accumulation steps from nominal batch size.

    If ``config.nbs`` is set, returns ``max(round(nbs / batch_size), 1)``.
    Otherwise returns 1 (no accumulation).
    """
    if config.nbs is None:
        return 1
    accum = max(round(config.nbs / config.data.batch_size), 1)
    if accum > 1:
        logger.info(
            "Nominal batch size %d with batch_size %d → accumulate_grad_batches=%d "
            "(effective batch size: %d)",
            config.nbs, config.data.batch_size, accum,
            config.data.batch_size * accum,
        )
    return accum


def _build_accum_scheduler(
    target_accum: int, warmup_epochs: int,
) -> "L.callbacks.GradientAccumulationScheduler | None":
    """Build a GradientAccumulationScheduler that ramps accumulation during warmup.

    Returns *None* when no ramping is needed (target_accum <= 1 or
    warmup_epochs <= 0).
    """
    if target_accum <= 1 or warmup_epochs <= 0:
        return None

    # Build epoch → accumulation mapping: linearly step up from 1 to target
    # Example: target=4, warmup=3 → {0: 1, 1: 2, 2: 3, 3: 4}
    schedule = {}
    for epoch in range(warmup_epochs + 1):
        accum = max(1, round(1 + (target_accum - 1) * epoch / warmup_epochs))
        schedule[epoch] = accum

    logger.info(
        "Accumulation warmup schedule: %s",
        {f"epoch {k}": v for k, v in schedule.items()},
    )

    return L.callbacks.GradientAccumulationScheduler(scheduling=schedule)


def _read_dvc_hash(dataset_path: Path) -> str | None:
    """Read the DVC content hash for a tracked dataset directory.

    Looks for a ``<dataset_path>.dvc`` file (created by ``dvc add``) and
    extracts the md5 hash that uniquely identifies the dataset contents.
    Returns *None* if the dataset is not DVC-tracked.
    """
    dvc_file = dataset_path.with_suffix(".dvc")
    if not dvc_file.exists():
        # Also check with the original name + .dvc (handles paths with dots)
        dvc_file = Path(str(dataset_path) + ".dvc")
    if not dvc_file.exists():
        return None
    try:
        import yaml as _yaml

        with open(dvc_file) as f:
            meta = _yaml.safe_load(f)
        return meta["outs"][0]["md5"]
    except Exception:
        logger.warning("Failed to read DVC hash from %s", dvc_file)
        return None


def _read_curation_metadata(dataset_path: Path) -> dict | None:
    """Read curation metadata from a dataset's ``curation.yaml`` sidecar.

    Returns the parsed dict, or *None* if the file doesn't exist.
    """
    curation_file = dataset_path / "curation.yaml"
    if not curation_file.exists():
        return None
    try:
        import yaml as _yaml

        with open(curation_file) as f:
            data = _yaml.safe_load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.warning("Failed to read curation metadata from %s", curation_file)
        return None


def _flatten_dict(d: dict, prefix: str = "") -> dict[str, str]:
    """Flatten a nested dict to dot-notation keys with string values."""
    items: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, key))
        else:
            items[key] = str(v)
    return items


def _build_loggers(
    config: TrainConfig,
    run_dir: Path,
    resume_meta: dict | None = None,
) -> list[LightningLogger]:
    """Build Lightning loggers from ``config.loggers`` list.

    When *resume_meta* is provided, W&B and MLflow loggers reconnect to their
    previous runs instead of creating new ones.
    """
    result: list[LightningLogger] = []
    resume_meta = resume_meta or {}
    dvc_hash = _read_dvc_hash(Path(config.data.dataset_path))
    if dvc_hash:
        logger.info("Dataset DVC hash: %s", dvc_hash)
    else:
        logger.info("No DVC metadata found for %s", config.data.dataset_path)

    curation_meta = _read_curation_metadata(Path(config.data.dataset_path))
    if curation_meta:
        logger.info("Dataset curation metadata loaded (%d sources)", len(curation_meta.get("sources", {})))
    else:
        logger.info("No curation metadata found for %s", config.data.dataset_path)

    for name in config.loggers:
        key = name.lower().replace("-", "").replace("_", "")

        if key == "tensorboard":
            result.append(TensorBoardLogger(
                save_dir=str(run_dir), name="tensorboard", version="",
            ))

        elif key == "wandb":
            try:
                import wandb
                from pytorch_lightning.loggers import WandbLogger

                kwargs = {
                    "project": Path(config.project_dir).name,
                    "name": config.run_name,
                    "save_dir": str(run_dir),
                    "settings": wandb.Settings(silent=True),
                }
                if config.wandb_group:  # bundle related runs (e.g. an ablation) in the project sidebar
                    kwargs["group"] = config.wandb_group
                # Reconnect to previous W&B run on resume
                wandb_id = resume_meta.get("wandb_run_id")
                if wandb_id:
                    kwargs["id"] = wandb_id
                    kwargs["resume"] = "must"
                    logger.info("Resuming W&B run: %s", wandb_id)

                wandb_logger = WandbLogger(**kwargs)
                # Only push config from rank 0 — non-rank-0 DDP processes
                # may not have a fully initialized wandb run.
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    # allow_val_change: a resumed run may legitimately carry an
                    # edited config (e.g. a corrected pool cap) or a rebuilt
                    # dataset (new dvc hash) — W&B hard-errors otherwise.
                    wandb_logger.experiment.config.update(
                        asdict(config), allow_val_change=True)
                    if dvc_hash:
                        wandb_logger.experiment.config.update(
                            {"dataset_dvc_hash": dvc_hash}, allow_val_change=True)
                    if curation_meta:
                        wandb_logger.experiment.config.update(
                            {"dataset_curation": curation_meta}, allow_val_change=True)
                result.append(wandb_logger)
            except (ImportError, ModuleNotFoundError):
                logger.warning("wandb requested but not installed — skipping (pip install wandb)")
                continue

        elif key == "mlflow":
            try:
                from pytorch_lightning.loggers import MLFlowLogger

                mlflow_dir = Path(config.project_dir) / ".mlflow"
                mlflow_dir.mkdir(parents=True, exist_ok=True)
                db_path = mlflow_dir.resolve() / "mlflow.db"
                tracking_uri = f"sqlite:///{db_path}"

                kwargs = {
                    "experiment_name": Path(config.project_dir).name,
                    "tracking_uri": tracking_uri,
                }
                # Reconnect to previous MLflow run on resume
                mlflow_run_id = resume_meta.get("mlflow_run_id")
                if mlflow_run_id:
                    kwargs["run_id"] = mlflow_run_id
                    logger.info("Resuming MLflow run: %s", mlflow_run_id)
                else:
                    kwargs["run_name"] = config.run_name

                mlflow_logger = MLFlowLogger(**kwargs)
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                if local_rank == 0:
                    if dvc_hash:
                        mlflow_logger.experiment.log_param(
                            mlflow_logger.run_id, "dataset_dvc_hash", dvc_hash,
                        )
                    if curation_meta:
                        flat = _flatten_dict(curation_meta, prefix="dataset_curation")
                        for k, v in flat.items():
                            mlflow_logger.experiment.log_param(mlflow_logger.run_id, k, v)
                result.append(mlflow_logger)
            except (ImportError, ModuleNotFoundError):
                logger.warning("mlflow requested but not installed — skipping (pip install mlflow)")
                continue

        else:
            logger.warning("Unknown logger %r — skipping", name)

    if not result:
        result.append(TensorBoardLogger(
            save_dir=str(run_dir), name="tensorboard", version="",
        ))

    logger.info("Loggers: %s", ", ".join(type(l).__name__ for l in result))
    return result


def _save_logger_metadata(loggers: list[LightningLogger], run_dir: Path) -> None:
    """Persist logger run IDs so training can be resumed into the same runs."""
    # Only rank 0 should save metadata — non-rank-0 DDP processes may not
    # have fully initialized logger experiments.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank != 0:
        return

    import json

    meta: dict[str, str] = {}

    for lg in loggers:
        cls_name = type(lg).__name__

        if cls_name == "WandbLogger":
            try:
                run_id = lg.experiment.id
                if run_id and isinstance(run_id, str):
                    meta["wandb_run_id"] = run_id
            except Exception:
                pass

        elif cls_name == "MLFlowLogger":
            try:
                run_id = lg.run_id
                if run_id and isinstance(run_id, str):
                    meta["mlflow_run_id"] = run_id
            except Exception:
                pass

    if meta:
        meta_path = run_dir / "logger_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        logger.debug("Saved logger metadata to %s", meta_path)


def _load_logger_metadata(run_dir: Path) -> dict:
    """Load previously saved logger run IDs for resume."""
    import json

    meta_path = run_dir / "logger_metadata.json"
    if not meta_path.exists():
        return {}

    with open(meta_path) as f:
        meta = json.load(f)

    logger.info("Loaded logger metadata: %s", meta)
    return meta
