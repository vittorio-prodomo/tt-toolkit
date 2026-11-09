"""CLI entry point: tt-sweep."""

import argparse


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an Optuna hyperparameter sweep.\n\n"
            "Usage:\n"
            "  tt-sweep --config config.yaml --sweep-config sweep.yaml\n"
            "  tt-sweep --config config.yaml --sweep-config sweep.yaml --n-trials 50\n\n"
            "The sweep config defines the search space. The training config\n"
            "provides base values for parameters not being tuned."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--config", required=True, help="Path to base training YAML config")
    parser.add_argument("--sweep-config", required=True, dest="sweep_config",
                        help="Path to sweep YAML config (search space, n_trials, etc.)")

    # Sweep overrides
    parser.add_argument("--n-trials", type=int, dest="n_trials", help="Number of trials (overrides sweep config)")
    parser.add_argument("--pruner", help="Pruning strategy: median, hyperband, none")
    parser.add_argument("--metric", help="Metric to optimize (e.g. val/map)")
    parser.add_argument("--direction", help="Optimization direction: maximize or minimize")
    parser.add_argument("--no-storage", action="store_true", dest="no_storage",
                        help="Disable persistent storage (run in-memory only)")
    parser.add_argument("--study-name", dest="study_name", help="Optuna study name (default: auto-generated)")

    # Training overrides (same subset as tt-train for convenience)
    parser.add_argument("--dataset-path", dest="dataset_path", help="Path to dataset root")
    parser.add_argument("--project-dir", dest="project_dir", help="Output directory for runs")
    parser.add_argument("--run-name", dest="run_name", help="Base run name (trials append _trial_N)")
    parser.add_argument("--device", nargs="+", type=int, help="GPU indices to use (e.g. --device 0 1 2)")
    parser.add_argument("--resume", help="Resume from checkpoint")

    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    from training_toolkit.configs import TrainConfig
    from training_toolkit.configs.sweep_config import SweepConfig
    from training_toolkit.engine.sweeper import sweep

    # Load configs
    config = TrainConfig.from_yaml(args.config)
    sweep_cfg = SweepConfig.from_yaml(args.sweep_config)

    # Apply sweep overrides
    if args.n_trials is not None:
        sweep_cfg.n_trials = args.n_trials
    if args.pruner is not None:
        sweep_cfg.pruner = args.pruner
    if args.metric is not None:
        sweep_cfg.metric = args.metric
    if args.direction is not None:
        sweep_cfg.direction = args.direction
    if args.no_storage:
        sweep_cfg.storage = None
    if args.study_name is not None:
        sweep_cfg.study_name = args.study_name

    # Apply training overrides
    if args.dataset_path is not None:
        config.data.dataset_path = args.dataset_path
    if args.project_dir is not None:
        config.project_dir = args.project_dir
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.device is not None:
        config.device = args.device

    sweep(config, sweep_cfg, resume_from=args.resume)


if __name__ == "__main__":
    main()
