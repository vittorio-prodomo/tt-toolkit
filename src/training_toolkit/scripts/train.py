"""CLI entry point: tt-train."""

import argparse


def _int_or_float(s: str):
    """argparse type for a value that may be an int (absolute count) or a float
    (fraction). Preserves the int-vs-float distinction Lightning's limit_val_batches
    needs (int = N batches; float in (0, 1] = fraction; 1.0 = full)."""
    return float(s) if ("." in s or "e" in s.lower()) else int(s)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train an instance segmentation model.\n\n"
            "Usage modes:\n"
            "  1. Config file only:   tt-train --config path/to/config.yaml\n"
            "  2. CLI only:           tt-train --model rf-detr-base --dataset-path /data/my_dataset\n"
            "  3. Hybrid:             tt-train --config config.yaml --lr 5e-4 --epochs 50\n\n"
            "In hybrid mode, CLI arguments override the corresponding values\n"
            "from the config file. When no config file is provided, sensible\n"
            "defaults are used for all unspecified arguments (model defaults\n"
            "to rf-detr-base; backend is inferred from the model name)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", help="Path to YAML config file (optional)")

    # Common overrides
    parser.add_argument("--model", help="Model name (e.g. rf-detr-base, rf-detr-nano, mask2former-swin-base)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Batch size")
    parser.add_argument("--device", nargs="+", type=int, help="GPU indices to use (e.g. --device 0 1 2)")
    parser.add_argument("--nbs", type=int, help="Nominal batch size — accumulate gradient steps to simulate this batch size")
    parser.add_argument("--dataset-path", dest="dataset_path", help="Path to dataset root")
    parser.add_argument("--layout", help="Dataset layout (coco, roboflow, or yolo)")
    parser.add_argument("--imgsz", type=int, help="Image size")
    parser.add_argument("--project-dir", dest="project_dir", help="Output directory for runs (default: runs/)")
    parser.add_argument("--run-name", dest="run_name", help="Run name (subdirectory inside project-dir)")
    parser.add_argument("--seed", help="Random seed (integer or 'none' to disable)")
    parser.add_argument("--scheduler", help="LR scheduler (e.g. cosine, none)")
    parser.add_argument("--patience", type=int, help="Early stopping patience")
    parser.add_argument("--fraction", type=float, help="Fraction of dataset to use (e.g. 0.1 for smoke test)")
    parser.add_argument("--limit-train-batches", type=int, dest="limit_train_batches",
                        help="Cap train AND val batches per epoch to this integer (like "
                             "--fraction, one value governs both). Unlike --fraction (a float "
                             "ratio) it fixes an absolute batch count WITHOUT shrinking the pool "
                             "— reaches the epoch boundary fast on the full pool. Overrides "
                             "--fraction when both are set.")
    parser.add_argument("--limit-val-batches", type=_int_or_float, dest="limit_val_batches",
                        help="Cap VAL batches per epoch INDEPENDENTLY of --limit-train-batches "
                             "(int = absolute batch count; float in (0,1] = fraction; 1.0 = full "
                             "val). Overrides the train-coupled default — e.g. "
                             "--limit-train-batches 100 --limit-val-batches 1.0 = short train, "
                             "full val.")
    parser.add_argument("--multi-scale", action="store_true", dest="multi_scale", help="Enable multi-scale training")
    parser.add_argument("--pretrained", help="Pretrained weights: 'auto' (default), 'none', or path")
    parser.add_argument("--resume", help="Resume from checkpoint: path to .ckpt file, or 'last'")
    parser.add_argument("--workers", type=int, help="Number of DataLoader workers")
    parser.add_argument("--precision", help="Training precision (e.g. bf16-mixed, 16-mixed, 32-true)")
    parser.add_argument("--gradient-clip-val", type=float, dest="gradient_clip_val",
                        help="Max gradient norm for clipping (0 to disable; "
                             "omit to use model-specific default: "
                             "YOLO=10.0, RF-DETR=0.1, Mask2Former=0.01)")
    parser.add_argument("--overwrite", action="store_true", help="Wipe run directory if it already exists")
    parser.add_argument("--profile", nargs="?", const="simple", default=None,
                        choices=["simple", "advanced", "pytorch"],
                        help="Enable profiler: 'simple' (per-hook timing), "
                             "'advanced' (per-function via cProfile), "
                             "'pytorch' (GPU kernels, Chrome trace). "
                             "Passing --profile without a value defaults to 'simple'.")
    parser.add_argument("--no-log-file", action="store_true", help="Disable file logging")
    parser.add_argument("--no-verbose-metrics", action="store_true", dest="no_verbose_metrics", help="Disable full COCO-style mAP breakdown (print summary only)")
    parser.add_argument("--diagnostics-period", type=int, dest="diagnostics_period", help="Epoch interval for results plot and val visualizations (default: 5)")
    parser.add_argument("--merge-overlapping-masks", action="store_true", dest="merge_overlapping_masks", help="Merge same-class instance masks with non-zero overlap before augmentation")
    parser.add_argument("--no-analysis", action="store_true", dest="no_analysis", help="Skip pre-training dataset analysis plots and batch sample grids")
    parser.add_argument("--log-images", action="store_true", dest="log_images", help="Log val images to W&B (requires wandb logger)")
    parser.add_argument("--loggers", nargs="+", help="Experiment loggers (e.g. tensorboard wandb mlflow)")

    return parser


def apply_overrides(config, args):
    """Apply CLI argument overrides to a TrainConfig."""
    if args.model is not None:
        config.model.name = args.model
    if args.lr is not None:
        config.lr = args.lr
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.data.batch_size = args.batch_size
    if args.dataset_path is not None:
        config.data.dataset_path = args.dataset_path
    if args.layout is not None:
        config.data.layout = args.layout
    if args.device is not None:
        config.device = args.device
    if args.nbs is not None:
        config.nbs = args.nbs
    if args.imgsz is not None:
        config.augmentation.imgsz = args.imgsz
    if args.project_dir is not None:
        config.project_dir = args.project_dir
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.seed is not None:
        if args.seed.lower() == "none":
            config.seed = None
        else:
            config.seed = int(args.seed)
    if args.scheduler is not None:
        config.scheduler = args.scheduler
    if args.patience is not None:
        config.patience = args.patience
    if args.fraction is not None:
        config.fraction = args.fraction
    if args.limit_train_batches is not None:
        config.limit_train_batches = args.limit_train_batches
    if args.limit_val_batches is not None:
        config.limit_val_batches = args.limit_val_batches
    if args.multi_scale:
        config.augmentation.multi_scale = True
    if args.pretrained is not None:
        config.model.pretrained = args.pretrained
    if args.workers is not None:
        config.data.workers = args.workers
    if args.precision is not None:
        config.precision = args.precision
    if args.gradient_clip_val is not None:
        config.gradient_clip_val = args.gradient_clip_val
    if args.overwrite:
        config.overwrite = True
    if args.no_log_file:
        config.log_to_file = False
    if args.no_verbose_metrics:
        config.verbose_metrics = False
    if args.diagnostics_period is not None:
        config.diagnostics_period = args.diagnostics_period
    if args.merge_overlapping_masks:
        config.augmentation.merge_overlapping_masks = True
    if args.no_analysis:
        config.analysis = False
    if args.log_images:
        config.log_images = True
    if args.loggers is not None:
        config.loggers = args.loggers
    return config


def main():
    parser = make_parser()
    args = parser.parse_args()

    from training_toolkit.configs import TrainConfig
    from training_toolkit.engine.trainer import train

    if args.config:
        config = TrainConfig.from_yaml(args.config)
    else:
        config = TrainConfig()

    config = apply_overrides(config, args)

    train(config, resume_from=args.resume, profiler=args.profile)


if __name__ == "__main__":
    main()
