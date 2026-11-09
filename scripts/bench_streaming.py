"""Throughput gate for the streaming on-the-fly tiling DataLoader (Milestone 1).

Run as a FILE (spawn workers need a real __main__ module — do NOT use python -c
or a heredoc, or spawn workers crash):

    cd training-toolkit && uv run python scripts/bench_streaming.py \\
        --dataset-path /path/to/fullres_coco --layout coco \\
        --tile-size 416 --workers 4 --batch-size 16

Prints samples/sec and whether it clears the GPU-fed budget (~12.95 samp/s, the
demand of a production RF-DETR training config).
"""
from __future__ import annotations

import argparse
import os
import time

# Single-thread libvips (parallelism comes from DataLoader workers). Must be set
# before any pyvips import; see training_toolkit.data.tile_materialize.
os.environ.setdefault("VIPS_CONCURRENCY", "1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", required=True)
    ap.add_argument("--layout", default="coco", choices=["coco", "roboflow", "yolo"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--tile-size", type=int, default=416)
    ap.add_argument("--overlap", type=float, default=0.2)
    ap.add_argument("--negative-ratio", type=float, default=0.15)
    ap.add_argument("--buffer-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-samples", type=int, default=800)
    ap.add_argument("--budget-sps", type=float, default=12.95)
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from training_toolkit.data import collate_fn
    from training_toolkit.data.streaming_tiled_dataset import (
        StreamingTiledSegmentationDataset,
    )

    if args.layout == "yolo":
        from training_toolkit.data.yolo_dataset import YoloSegmentationDataset
        base = YoloSegmentationDataset(root=args.dataset_path, split=args.split)
    else:
        from training_toolkit.data.coco_dataset import CocoSegmentationDataset
        base = CocoSegmentationDataset(root=args.dataset_path, split=args.split,
                                       layout=args.layout)

    ds = StreamingTiledSegmentationDataset(
        base, tile_size=args.tile_size, overlap=args.overlap,
        negative_ratio=args.negative_ratio, buffer_size=args.buffer_size,
    )

    # VIPS_CONCURRENCY=1 is set at module import (and in tile_materialize), which
    # each spawn worker re-runs on import — so no worker_init_fn is needed here.
    kwargs = dict(batch_size=args.batch_size, collate_fn=collate_fn,
                  num_workers=args.workers, drop_last=False)
    if args.workers > 0:
        kwargs["multiprocessing_context"] = "spawn"
    loader = DataLoader(ds, **kwargs)

    print(f"dataset: {len(base)} images, {len(ds)} tiles; "
          f"tile={args.tile_size} workers={args.workers} batch={args.batch_size}")

    it = iter(loader)
    first = next(it)              # warm up one batch (spawn startup) before timing
    warm = first["images"].shape[0]
    seen = warm
    t0 = time.perf_counter()
    for batch in it:
        seen += batch["images"].shape[0]
        if seen >= args.n_samples:
            break
    dt = time.perf_counter() - t0
    timed = seen - warm
    sps = timed / dt if dt > 0 else float("inf")
    print(f"{timed} samples in {dt:.2f}s → {sps:.1f} samples/sec")
    verdict = "PASS" if sps >= args.budget_sps else "FAIL"
    print(f"budget {args.budget_sps} samp/s → {verdict} ({sps / args.budget_sps:.1f}x budget)")


if __name__ == "__main__":
    main()
