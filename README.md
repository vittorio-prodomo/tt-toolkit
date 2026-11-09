# tt-toolkit

A config-driven toolkit for **training, evaluating and deploying instance-segmentation models**: RF-DETR-Seg,
Mask2Former and YOLO-seg behind one PyTorch Lightning loop, with on-the-fly tiling for very large images, exact
multi-GPU mask mAP, hyperparameter sweeps, ONNX/TensorRT export and one-command deployment as CVAT auto-annotation
functions.

It grew out of training defect-segmentation models on 4K drone imagery, where objects can be a few dozen pixels wide and
pre-tiling every dataset at every tile size stopped scaling.

## Highlights

- **One training loop, several architectures.** A single `SegmentationLitModule` plus an adapter per backend
  (RF-DETR-Seg nano→large, Mask2Former Swin tiny→large, Ultralytics YOLO-seg). YAML config and CLI overrides, with
  per-backend defaults for precision and gradient clipping.
- **On-the-fly tiling** (`data.tiling`) instead of pre-tiled copies of the dataset:
  - tiles stream from per-worker image slices through a shuffle buffer;
  - multi-scale tile sizes, ground-truth-guided crops, and a whole-image view for very large objects;
  - pyvips region decoding, so only the pixels a tile needs get decoded;
  - an optional pool of decoded images in POSIX shared memory, shared by every DDP rank and DataLoader worker.
- **Augmentation**: Albumentations pipeline plus mosaic, cutmix, mixup and copy-paste.
- **Evaluation you can trust**:
  - mask mAP computed exactly across GPUs (state gathered, then scored once with faster-coco-eval);
  - per-class AP@50/AP@75 and TIDE error breakdowns;
  - early stopping, plus a best checkpoint per class;
  - validation mask resolution set as a ratio, so numbers stay comparable across input sizes.
- **Partial labels**: per-image class masks (masked loss) and ignore regions, for training on merged datasets that
  were labelled with different class sets. These use the RF-DETR fork pinned in `pyproject.toml`.
- **Tracking**: TensorBoard, Weights & Biases, MLflow. When the dataset is tracked with DVC, its content hash is logged
  with the run.
- **Sweeps**: `tt-sweep` runs Optuna studies with pruning and divergence detection.
- **Export and serving**:
  - `tt-export` exports to ONNX;
  - `tt-infer` runs ONNX Runtime or TensorRT, optionally with SAHI sliced inference;
  - `tt-gen-serverless` and `tt-deploy` turn an export into a CVAT (Nuclio) auto-annotation function.

## Requirements

- Linux (the shared-memory image pool uses POSIX shared memory) with an NVIDIA GPU. PyTorch is pinned to the CUDA 12.8
  wheel index.
- Python 3.10–3.13 and [uv](https://docs.astral.sh/uv/). The backends use patched forks pinned in `pyproject.toml`; uv
  applies those sources for you, plain `pip` does not.
- The libvips system library (`apt install libvips`).

## Install

```bash
git clone https://github.com/vittorio-prodomo/tt-toolkit && cd tt-toolkit
uv sync --extra rfdetr                       # core + RF-DETR-Seg (Mask2Former needs no extra)
uv sync --extra rfdetr --extra sweep --extra export --extra inference --extra wandb   # a typical setup
```

Extras: `rfdetr`, `yolo`, `mask2former`, `sweep`, `wandb`, `mlflow`, `export`, `inference`, `tensorrt`, `sahi`,
`fiftyone`, `all`, `dev`.

## Quickstart

```bash
cp configs/template.yaml my-run.yaml           # set data.dataset_path (COCO, Roboflow or YOLO layout)
uv run tt-train --config my-run.yaml --model rf-detr-seg-medium --epochs 50 --run-name my-run
uv run tt-export --run runs/my-run --output-dir runs/my-run/export   # ONNX; offers the run's checkpoints, best first
uv run tt-infer --export-dir runs/my-run/export --images photo.jpg --sliced --output-dir results/
```

Any config value has a CLI override (`--lr`, `--batch-size`, `--device 0 1`, `--imgsz`, `--resume last`, …);
`tt-train -h` lists them. `configs/template.yaml` and `configs/sweep-template.yaml` document every option.

## Commands

| Command | What it does |
|---|---|
| `tt-train` | Train a model from a YAML config and/or CLI flags (single GPU or DDP) |
| `tt-sweep` | Optuna hyperparameter sweep over a sweep config |
| `tt-export` | Export a trained run to ONNX |
| `tt-infer` | Run an export with ONNX Runtime or TensorRT, optionally sliced (SAHI) |
| `tt-gen-serverless` | Generate a CVAT Nuclio function from an export |
| `tt-deploy` | Deploy that function to a local CVAT instance |
| `tt-bench-batch` | Measure training throughput across batch sizes |

## Data layouts

`data.layout: coco` (default) expects `annotations/instances_<split>.json` plus `images/<split>/`;
`roboflow` expects one folder per split with `_annotations.coco.json`; `yolo` reads YOLO-seg labels. Split names come
from `data.train_split` / `data.val_split`, and the class count from the annotations.

## Development

```bash
uv sync --extra rfdetr --extra yolo --extra dev
uv run pytest -m "not slow"
```

Tests use synthetic data only. Tests for optional backends skip when the backend is not installed, and the
fork-specific RF-DETR criterion tests skip on upstream `rfdetr`. Run the suite serially: the shared-memory and
semaphore tests use fixed names, so parallel workers (`-n`) can collide.

## Licences

The toolkit is MIT-licensed (see [LICENSE](LICENSE)). Optional extras bring their own terms:

- `yolo` installs Ultralytics, which is **AGPL-3.0**. If you use the YOLO backend, its terms apply to that use.
- `tensorrt` installs NVIDIA TensorRT under NVIDIA's licence.
- RF-DETR is Apache-2.0. Its XLarge and 2XLarge models are not supported, because they use a different licence.
  `models/rfdetr.py` reproduces a short excerpt of RF-DETR's `PostProcess.forward`, marked in the source.
