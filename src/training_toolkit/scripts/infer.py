"""CLI entry point: tt-infer."""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run inference with an ONNX model exported by tt-export.\n\n"
            "Usage:\n"
            "  tt-infer --export-dir export/ --images img1.jpg img2.jpg\n"
            "  tt-infer --export-dir export/ --images img.jpg --output-dir results/\n"
            "  tt-infer --export-dir export/ --images img.jpg --json results.json\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export-dir", dest="export_dir", required=True,
                        help="Export directory (with metadata.json + ONNX model)")
    parser.add_argument("--images", nargs="+", required=True,
                        help="One or more image file paths")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Score threshold (default: 0.5)")
    parser.add_argument("--runtime", default="ort", choices=["ort", "trt"],
                        help="Inference runtime: ort (ONNX Runtime) or trt (TensorRT) (default: ort)")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Save annotated images to this directory")
    parser.add_argument("--json", dest="json_path", default=None,
                        help="Save results as JSON to this file")
    # Sliced inference
    parser.add_argument("--sliced", action="store_true", default=False,
                        help="Enable sliced inference via SAHI")
    parser.add_argument("--slice-height", dest="slice_height", type=int, default=None,
                        help="Slice height in pixels (default: model resolution)")
    parser.add_argument("--slice-width", dest="slice_width", type=int, default=None,
                        help="Slice width in pixels (default: model resolution)")
    parser.add_argument("--overlap-ratio", dest="overlap_ratio", type=float, default=0.2,
                        help="Overlap between adjacent slices (default: 0.2)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=1,
                        help="Slices per inference call (default: 1)")
    parser.add_argument("--postprocess-type", dest="postprocess_type", default="GREEDYNMM",
                        choices=["GREEDYNMM", "NMS", "NMM", "LSNMS", "NONE"],
                        help="Merging strategy (default: GREEDYNMM)")
    parser.add_argument("--postprocess-match-metric", dest="postprocess_match_metric",
                        default="IOS", choices=["IOS", "IOU"],
                        help="Match metric for merging (default: IOS)")
    parser.add_argument("--postprocess-match-threshold", dest="postprocess_match_threshold",
                        type=float, default=0.5,
                        help="Merge threshold (default: 0.5)")
    parser.add_argument("--postprocess-class-agnostic", dest="postprocess_class_agnostic",
                        action="store_true", default=False,
                        help="Merge across classes")
    parser.add_argument("--perform-standard-pred", dest="perform_standard_pred",
                        action="store_true", default=False,
                        help="Also run full-image inference")
    parser.add_argument("--auto-slice-resolution", dest="auto_slice_resolution",
                        action="store_true", default=False,
                        help="Auto-determine slice size")
    parser.add_argument("--no-progress-bar", dest="no_progress_bar",
                        action="store_true", default=False,
                        help="Disable tqdm progress bar")
    return parser


def _summarize(
    result: dict[str, np.ndarray],
    class_names: dict[int, str] | None,
) -> str:
    """Build a one-line detection summary."""
    n = len(result["scores"])
    if n == 0:
        return "0 detections"

    counts = Counter(int(l) for l in result["labels"])
    parts = []
    for label, count in sorted(counts.items()):
        name = class_names.get(label, str(label)) if class_names else str(label)
        parts.append(f"{name}: {count}")
    return f"{n} detections ({', '.join(parts)})"


def _draw_result(
    image: np.ndarray,
    result: dict[str, np.ndarray],
    class_names: dict[int, str] | None,
) -> np.ndarray:
    """Draw predictions on an image. Returns BGR image.

    Note: imports ``visualization`` lazily — that module imports torch,
    so this keeps the core CLI torch-free unless --output-dir is used.
    """
    from training_toolkit.visualization import _draw_annotations

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.shape[2] == 3 else image
    masks = result["masks"] if result["masks"] is not None and len(result["masks"]) > 0 else None
    return _draw_annotations(
        bgr,
        result["boxes"],
        result["labels"],
        masks,
        class_names,
        scores=result["scores"],
    )


def _result_to_json(result: dict[str, np.ndarray]) -> dict:
    """Convert a result dict to JSON-serialisable form (no masks)."""
    return {
        "boxes": result["boxes"].tolist(),
        "scores": result["scores"].tolist(),
        "labels": result["labels"].tolist(),
        "num_detections": len(result["scores"]),
    }


def main():
    parser = make_parser()
    args = parser.parse_args()

    from rich.logging import RichHandler
    logging.basicConfig(
        level=logging.INFO,
        handlers=[RichHandler(show_path=False, show_time=True)],
        format="%(message)s",
    )

    from training_toolkit.inference import Predictor

    predictor = Predictor(args.export_dir, runtime=args.runtime, score_threshold=args.threshold)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for img_path in args.images:
        path = Path(img_path)
        if not path.exists():
            logger.warning("Image not found: %s — skipping", img_path)
            continue

        if args.sliced:
            result = predictor.predict_sliced(
                str(path),
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_ratio=args.overlap_ratio,
                batch_size=args.batch_size,
                postprocess_type=args.postprocess_type,
                postprocess_match_metric=args.postprocess_match_metric,
                postprocess_match_threshold=args.postprocess_match_threshold,
                postprocess_class_agnostic=args.postprocess_class_agnostic,
                perform_standard_pred=args.perform_standard_pred,
                auto_slice_resolution=args.auto_slice_resolution,
                progress_bar=not args.no_progress_bar,
            )
        else:
            result = predictor.predict(str(path))
        summary = _summarize(result, predictor.class_names)
        logger.info("%s: %s", path.name, summary)

        if args.output_dir:
            bgr = cv2.imread(str(path))
            if bgr is None:
                logger.warning("Could not read image for annotation: %s", path)
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            annotated = _draw_result(rgb, result, predictor.class_names)
            out_path = Path(args.output_dir) / path.name
            cv2.imwrite(str(out_path), annotated)
            logger.info("  Saved annotated image to %s", out_path)

        if args.json_path:
            all_results[str(path)] = _result_to_json(result)

    if args.json_path:
        Path(args.json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Saved JSON results to %s", args.json_path)


if __name__ == "__main__":
    main()
