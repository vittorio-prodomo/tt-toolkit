"""Benchmark training throughput across batch sizes to find the sweet spot.

Usage::

    tt-bench-batch --config configs/my-run.yaml
    tt-bench-batch --config configs/my-run.yaml --sizes 8 16 32 64 128
    tt-bench-batch --config configs/my-run.yaml --steps 100 --device 1
"""

import argparse
import gc
import time

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark training throughput across batch sizes.",
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[4, 8, 16, 32, 48, 64, 96, 128],
        help="Batch sizes to test (default: 4 8 16 32 48 64 96 128)",
    )
    parser.add_argument("--steps", type=int, default=50, help="Training steps per size (default: 50)")
    parser.add_argument("--device", type=int, default=0, help="GPU index (default: 0)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps to discard (default: 5)")
    args = parser.parse_args()

    from training_toolkit.configs import TrainConfig
    from training_toolkit.data import collate_fn
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.yolo_dataset import YoloSegmentationDataset
    from training_toolkit.data.transforms import build_transforms
    from training_toolkit.models import get_adapter

    config = TrainConfig.from_yaml(args.config)
    device = torch.device(f"cuda:{args.device}")

    # Build adapter + model + criterion
    adapter = get_adapter(config.model)

    # Resolve imgsz
    if config.augmentation.imgsz is None:
        config.augmentation.imgsz = adapter.get_default_imgsz(config.model.name)

    # Auto-detect num_classes from dataset
    layout = config.data.layout
    if layout == "yolo":
        ds = YoloSegmentationDataset(root=config.data.dataset_path, split=config.data.train_split)
    else:
        ds = CocoSegmentationDataset(root=config.data.dataset_path, split=config.data.train_split, layout=layout)

    if config.model.num_classes is None:
        config.model.num_classes = ds.num_classes

    transform = build_transforms(config.augmentation, is_train=False)
    ds._transform = transform

    model = adapter.create_model(config.model).train().to(device)

    # For YOLO: lazy criterion needs model on device first
    has_external_criterion = adapter.create_criterion(config.model) is not None

    gpu_name = torch.cuda.get_device_name(device)
    gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"\nBatch Size Benchmark")
    print(f"  GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    print(f"  Model: {config.model.name}")
    print(f"  Image size: {config.augmentation.imgsz}px")
    print(f"  Steps per size: {args.steps} (+{args.warmup} warmup)")
    print(f"  Dataset: {len(ds)} images")
    print()
    print(f"{'Batch':>7} {'Samples/s':>11} {'Step (ms)':>11} {'VRAM (GB)':>11} {'VRAM %':>8}  Notes")
    print(f"{'-----':>7} {'---------':>11} {'---------':>11} {'---------':>11} {'------':>8}  -----")

    results = []

    for bs in args.sizes:
        # Clean up from previous run
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        loader = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=config.data.workers,
            collate_fn=collate_fn, pin_memory=True, drop_last=True,
        )

        if len(loader) == 0:
            print(f"{bs:>7} {'—':>11} {'—':>11} {'—':>11} {'—':>8}  dataset too small")
            continue

        try:
            data_iter = iter(loader)
            step_times = []

            for step in range(args.steps + args.warmup):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch = next(data_iter)

                # Move to device
                images = batch["images"].to(device, non_blocking=True)
                targets = []
                for t in batch["targets"]:
                    targets.append({
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    })
                gpu_batch = {"images": images, "targets": targets}

                torch.cuda.synchronize(device)
                t0 = time.perf_counter()

                # Forward
                formatted = adapter.format_batch(gpu_batch)
                outputs = model(**formatted["inputs"])

                # Loss
                if has_external_criterion:
                    loss_dict = adapter.compute_loss_with_criterion(
                        adapter.create_criterion(config.model), outputs, formatted["targets"]
                    )
                else:
                    loss_dict = adapter.compute_loss(outputs, formatted["targets"])

                # Backward
                loss_dict["loss_total"].backward()

                # Optimizer step (simulate)
                model.zero_grad()

                torch.cuda.synchronize(device)
                t1 = time.perf_counter()

                if step >= args.warmup:
                    step_times.append(t1 - t0)

            vram_gb = torch.cuda.max_memory_allocated(device) / 1e9
            vram_pct = vram_gb / gpu_mem * 100
            avg_step_ms = sum(step_times) / len(step_times) * 1000
            samples_per_sec = bs / (avg_step_ms / 1000)

            results.append((bs, samples_per_sec, avg_step_ms, vram_gb, vram_pct))
            print(f"{bs:>7} {samples_per_sec:>11.1f} {avg_step_ms:>11.1f} {vram_gb:>11.1f} {vram_pct:>7.0f}%")

        except torch.cuda.OutOfMemoryError:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"{bs:>7} {'OOM':>11} {'—':>11} {'—':>11} {'—':>8}")
            break  # larger sizes will also OOM

        finally:
            del loader
            model.zero_grad(set_to_none=True)

    if results:
        best = max(results, key=lambda r: r[1])
        print()
        print(f"  Best throughput: batch_size={best[0]} → {best[1]:.1f} samples/s "
              f"({best[2]:.0f} ms/step, {best[3]:.1f} GB VRAM = {best[4]:.0f}%)")


if __name__ == "__main__":
    main()
