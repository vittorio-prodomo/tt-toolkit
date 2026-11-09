"""CLI entry point: tt-export."""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)

# ImageNet normalization used by both RF-DETR and Mask2Former
_NORM_MEAN = [0.485, 0.456, 0.406]
_NORM_STD = [0.229, 0.224, 0.225]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Export a trained model to ONNX format.\n\n"
            "Usage:\n"
            "  tt-export --run runs/my-run\n"
            "  tt-export --run runs/my-run --no-simplify\n"
            "  tt-export --run runs/my-run --shape 560 --batch-size 4\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run", required=True,
                        help="Path to a training run directory")
    parser.add_argument("--config", default=None,
                        help="Override config file (default: auto-discovered from run dir)")
    parser.add_argument("--output-dir", dest="output_dir", default="export",
                        help="Output directory (default: export/)")
    parser.add_argument("--no-simplify", action="store_true", dest="no_simplify",
                        help="Skip ONNX simplification (simplify is on by default)")
    parser.add_argument("--shape", type=int, default=None,
                        help="Input resolution (single int; must be divisible by "
                             "14 for RF-DETR or 32 for Mask2Former)")
    parser.add_argument("--batch-size", type=int, dest="batch_size", default=None,
                        help="Fixed batch size (default: dynamic batch axis)")
    parser.add_argument("--batch-range", dest="batch_range", default=None,
                        metavar="MIN,OPT,MAX",
                        help="TRT optimization profile batch range, e.g. '1,8,16'. "
                             "Enables dynamic batching with TensorRT. "
                             "Ignored for ORT. Overrides --batch-size.")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17)")
    parser.add_argument("--dataset-path", dest="dataset_path", default=None,
                        help="Path to dataset root (optional, for class names in metadata)")
    return parser


# ------------------------------------------------------------------
# Run directory discovery
# ------------------------------------------------------------------

def _discover_artifacts(run_dir: Path) -> list[dict]:
    """Find all exportable model artifacts in a run directory."""
    artifacts: list[dict] = []

    # Lightning checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for f in sorted(ckpt_dir.glob("*.ckpt")):
            map_val = _parse_map_from_filename(f.name)
            artifacts.append({
                "path": f,
                "type": "Lightning checkpoint",
                "name": f.name,
                "size_mb": f.stat().st_size / 1e6,
                "map": map_val,
            })

    # Toolkit .pth exports (RF-DETR)
    for f in run_dir.glob("*.pth"):
        artifacts.append({
            "path": f,
            "type": "Toolkit export",
            "name": f.name,
            "size_mb": f.stat().st_size / 1e6,
            "map": None,
        })

    # Loose .pt files
    for f in run_dir.glob("*.pt"):
        artifacts.append({
            "path": f,
            "type": "PyTorch export",
            "name": f.name,
            "size_mb": f.stat().st_size / 1e6,
            "map": None,
        })

    # HuggingFace save_pretrained directory (Mask2Former)
    model_dir = run_dir / "model"
    if model_dir.is_dir() and (model_dir / "config.json").exists():
        size = sum(f.stat().st_size for f in model_dir.iterdir() if f.is_file())
        artifacts.append({
            "path": model_dir,
            "type": "HuggingFace export",
            "name": "model/",
            "size_mb": size / 1e6,
            "map": None,
        })

    return _sort_artifacts(artifacts)


def _parse_map_from_filename(name: str) -> float | None:
    """Extract mAP value from checkpoint filename like 'epoch=009-map=0.8671.ckpt'."""
    match = re.search(r"map=([\d.]+)", name)
    if not match:
        return None
    value = match.group(1).rstrip(".")
    return float(value) if value else None


def _sort_artifacts(artifacts: list[dict]) -> list[dict]:
    """Sort: best checkpoints first (by mAP desc), then last.ckpt, then exports."""
    scored = []      # checkpoints with mAP
    last = []        # last.ckpt
    unscored = []    # checkpoints without mAP
    exports = []     # .pth, .pt, model/

    for a in artifacts:
        if a["type"] != "Lightning checkpoint":
            exports.append(a)
        elif a["name"] == "last.ckpt":
            last.append(a)
        elif a["map"] is not None:
            scored.append(a)
        else:
            unscored.append(a)

    scored.sort(key=lambda a: a["map"], reverse=True)
    return scored + unscored + last + exports


def _select_artifact(artifacts: list[dict]) -> dict:
    """Present an interactive selection of discovered artifacts."""
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel

    console = Console()

    table = Table(show_header=True, header_style="bold cyan", padding=(0, 1))
    table.add_column("#", style="bold", width=3, justify="right")
    table.add_column("Artifact", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("mAP", justify="right")
    table.add_column("Size", justify="right", style="dim")

    for i, a in enumerate(artifacts, 1):
        map_str = f"{a['map']:.4f}" if a["map"] is not None else ""
        table.add_row(
            str(i),
            a["name"],
            a["type"],
            map_str,
            f"{a['size_mb']:.0f} MB",
        )

    console.print()
    console.print(Panel(table, title="[bold]Discovered Artifacts[/bold]", border_style="blue"))

    if len(artifacts) == 1:
        console.print(f"  Auto-selecting the only artifact: [green]{artifacts[0]['name']}[/green]")
        return artifacts[0]

    # Non-interactive fallback (piped stdin, CI, etc.)
    if not sys.stdin.isatty():
        console.print(f"  Non-interactive mode — selecting [green]{artifacts[0]['name']}[/green]")
        return artifacts[0]

    while True:
        try:
            raw = console.input("\n  [bold]Select artifact to export[/bold] [dim]\\[1][/dim]: ")
            choice = int(raw.strip()) if raw.strip() else 1
            if 1 <= choice <= len(artifacts):
                return artifacts[choice - 1]
            console.print(f"  [red]Enter a number between 1 and {len(artifacts)}[/red]")
        except ValueError:
            console.print(f"  [red]Enter a number between 1 and {len(artifacts)}[/red]")
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(1)


# ------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------

def _load_weights(checkpoint_path: str) -> dict:
    """Load model state_dict from a Lightning .ckpt, toolkit .pth, or HF directory."""
    p = Path(checkpoint_path)

    # HuggingFace save_pretrained directory (Mask2Former)
    if p.is_dir():
        model_bin = p / "model.safetensors"
        if not model_bin.exists():
            model_bin = p / "pytorch_model.bin"
        if not model_bin.exists():
            raise FileNotFoundError(
                f"No model weights found in {checkpoint_path}. "
                "Expected model.safetensors or pytorch_model.bin."
            )
        return torch.load(str(model_bin), map_location="cpu", weights_only=False)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Lightning checkpoint: state_dict keys are prefixed with "model."
    if "state_dict" in ckpt:
        prefix = "model."
        return {
            k[len(prefix):]: v
            for k, v in ckpt["state_dict"].items()
            if k.startswith(prefix)
        }

    # Toolkit .pth export: raw model state_dict under "model" key
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]

    raise ValueError(
        f"Unrecognised checkpoint format in {checkpoint_path}. "
        "Expected a Lightning .ckpt, toolkit .pth, or HuggingFace directory."
    )


# ------------------------------------------------------------------
# Mask2Former ONNX wrapper
# ------------------------------------------------------------------

class _Mask2FormerONNXWrapper(nn.Module):
    """Thin wrapper that returns flat tensors for ONNX tracing."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(
            pixel_values=pixel_values,
            output_auxiliary_logits=False,
            output_hidden_states=False,
            output_attentions=False,
        )
        return outputs.class_queries_logits, outputs.masks_queries_logits


# ------------------------------------------------------------------
# Backend-specific export
# ------------------------------------------------------------------

def _export_rfdetr(model, shape, batch_size, dynamic_axes, output_dir, opset, simplify):
    """Export RF-DETR model using rfdetr's own export utilities."""
    from rfdetr.deploy.export import make_infer_image, export_onnx

    input_tensors = make_infer_image(None, (shape, shape), batch_size, "cpu")
    input_names = ["input"]
    has_seg_head = getattr(model, "segmentation_head", None) is not None
    output_names = ["dets", "labels", "masks"] if has_seg_head else ["dets", "labels"]

    output_file = export_onnx(
        output_dir=output_dir,
        model=model,
        input_names=input_names,
        input_tensors=input_tensors,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        verbose=False,
    )
    logger.info("Exported ONNX model to %s", output_file)

    simplified = False
    if simplify:
        try:
            from rfdetr.deploy.export import onnx_simplify
            sim_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=True,
            )
            logger.info("Simplified ONNX model to %s", sim_file)
            simplified = True
        except Exception:
            logger.warning(
                "ONNX simplification failed, keeping unsimplified model",
                exc_info=True,
            )

    return simplified


def _export_mask2former(model, shape, batch_size, dynamic_axes, output_dir, opset, simplify):
    """Export Mask2Former model via torch.onnx.export with a thin wrapper."""
    import onnx

    wrapper = _Mask2FormerONNXWrapper(model)
    wrapper.eval()

    dummy = torch.randn(batch_size, 3, shape, shape)
    input_names = ["pixel_values"]
    output_names = ["class_logits", "mask_logits"]

    m2f_dynamic_axes = None
    if dynamic_axes is not None:
        m2f_dynamic_axes = {
            "pixel_values": {0: "batch"},
            "class_logits": {0: "batch"},
            "mask_logits": {0: "batch"},
        }

    output_file = str(Path(output_dir) / "inference_model.onnx")

    torch.onnx.export(
        wrapper,
        (dummy,),
        output_file,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=m2f_dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )
    logger.info("Exported ONNX model to %s", output_file)

    # Validate
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model validation passed")

    simplified = False
    if simplify:
        try:
            import onnxsim
            model_opt, check_ok = onnxsim.simplify(output_file)
            if check_ok:
                sim_file = output_file.replace(".onnx", ".sim.onnx")
                onnx.save(model_opt, sim_file)
                logger.info("Simplified ONNX model to %s", sim_file)
                simplified = True
            else:
                logger.warning("onnxsim simplification check failed, keeping unsimplified model")
        except Exception:
            logger.warning(
                "ONNX simplification failed, keeping unsimplified model",
                exc_info=True,
            )

    return simplified


# ------------------------------------------------------------------
# Main export pipeline
# ------------------------------------------------------------------

def _export(args) -> None:
    """Run the ONNX export pipeline."""
    from training_toolkit.configs import TrainConfig
    from training_toolkit.models import get_adapter, detect_backend

    # --- Load config and build model --------------------------------
    config = TrainConfig.from_yaml(args.config)
    adapter = get_adapter(config.model)
    backend = config.model.backend or detect_backend(config.model.name)

    # If num_classes not set in config, try to read from dataset
    class_names = None
    if config.model.num_classes is None:
        if args.dataset_path is None:
            raise ValueError(
                "num_classes is not set in config and --dataset-path was not "
                "provided. Either set model.num_classes in the YAML or pass "
                "--dataset-path so it can be auto-detected."
            )
        from training_toolkit.data.coco_dataset import CocoSegmentationDataset
        from training_toolkit.data.yolo_dataset import YoloSegmentationDataset

        layout = config.data.layout
        ds_path = Path(args.dataset_path)
        if layout == "yolo":
            ds = YoloSegmentationDataset(root=ds_path, split="train")
        else:
            ds = CocoSegmentationDataset(root=ds_path, split="train", layout=layout)
        config.model.num_classes = ds.num_classes
        class_names = ds.class_names

    model = adapter.create_model(config.model)

    # --- Load checkpoint weights ------------------------------------
    state_dict = _load_weights(args.checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys in checkpoint: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys in checkpoint: %d", len(unexpected))
    logger.info("Loaded weights from %s", args.checkpoint)

    # --- Resolve export parameters ----------------------------------
    divisor = adapter.get_imgsz_divisor(config.model.name)
    shape = args.shape or adapter.get_default_imgsz(config.model.name)
    if shape % divisor != 0:
        raise ValueError(f"Shape must be divisible by {divisor}, got {shape}")

    # Parse batch range if provided
    batch_range = None
    if args.batch_range is not None:
        parts = [int(x.strip()) for x in args.batch_range.split(",")]
        if len(parts) != 3:
            raise ValueError("--batch-range must be MIN,OPT,MAX (e.g. '1,8,16')")
        batch_range = tuple(parts)
        logger.info("TRT batch range: min=%d, opt=%d, max=%d", *batch_range)

    if batch_range is not None:
        batch_size = batch_range[1]  # use opt for tracing
        is_dynamic = True  # need dynamic axes for TRT profiles
    else:
        batch_size = args.batch_size or 1
        is_dynamic = args.batch_size is None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.cpu()
    model.eval()
    simplify = not args.no_simplify

    # --- Dispatch to backend ----------------------------------------
    if backend == "rfdetr":
        dynamic_axes = None
        has_seg_head = getattr(model, "segmentation_head", None) is not None
        if is_dynamic:
            dynamic_axes = {
                "input": {0: "batch"},
                "dets": {0: "batch"},
                "labels": {0: "batch"},
            }
            if has_seg_head:
                dynamic_axes["masks"] = {0: "batch"}
        simplified = _export_rfdetr(
            model, shape, batch_size, dynamic_axes, output_dir, args.opset, simplify,
        )
    elif backend == "mask2former":
        # Pass a non-None marker so _export_mask2former knows to set dynamic axes
        dynamic_axes = {} if is_dynamic else None
        simplified = _export_mask2former(
            model, shape, batch_size, dynamic_axes, output_dir, args.opset, simplify,
        )
    else:
        raise ValueError(f"ONNX export not supported for backend {backend!r}")

    # --- Write metadata.json ----------------------------------------
    metadata = {
        "model": config.model.name,
        "backend": backend,
        "resolution": [shape, shape],
        "batch_size": batch_size if (args.batch_size is not None or batch_range is not None) else "dynamic",
        "normalization": {"mean": _NORM_MEAN, "std": _NORM_STD},
        "opset": args.opset,
        "simplified": simplified,
    }
    if batch_range is not None:
        metadata["batch_range"] = list(batch_range)
    if class_names:
        metadata["class_names"] = class_names

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    logger.info("Wrote metadata to %s", meta_path)


def main():
    parser = make_parser()
    args = parser.parse_args()

    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(show_path=False, show_time=True)],
        format="%(message)s",
    )

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # Auto-discover config
    if args.config is None:
        config_path = run_dir / "config.yaml"
        if not config_path.exists():
            raise SystemExit(
                f"No config.yaml found in {run_dir}. Use --config to specify one."
            )
        args.config = str(config_path)
        logger.info("Using config: %s", args.config)

    # Discover and select artifact
    artifacts = _discover_artifacts(run_dir)
    if not artifacts:
        raise SystemExit(f"No exportable artifacts found in {run_dir}")

    selected = _select_artifact(artifacts)
    args.checkpoint = str(selected["path"])
    logger.info("Selected: %s", selected["name"])

    _export(args)


if __name__ == "__main__":
    main()
