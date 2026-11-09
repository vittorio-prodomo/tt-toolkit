"""CLI entry point: tt-gen-serverless."""

import argparse
import logging
import sys
from pathlib import Path

from training_toolkit.serverless.generator import generate, detect_sahi_path

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Nuclio serverless function directory from a tt-export output.\n\n"
            "Usage:\n"
            "  tt-gen-serverless --export-dir runs/my_project/my_run/export/\n"
            "  tt-gen-serverless --export-dir export/ --runtime ort --name my-detector\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--export-dir", dest="export_dir", required=True,
                        help="Export directory (with metadata.json + ONNX model)")
    parser.add_argument("--output-dir", dest="output_dir", default=None,
                        help="Output directory (default: serverless/ sibling to export dir)")
    parser.add_argument("--name", default=None,
                        help="DNS-safe Nuclio function name (default: auto-derived)")
    parser.add_argument("--runtime", default="tensorrt", choices=["ort", "tensorrt"],
                        help="Inference runtime (default: tensorrt)")
    parser.add_argument("--sahi-path", dest="sahi_path", default=None,
                        help="Path to SAHI fork clone (default: auto-detected from pyproject.toml)")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    export_dir = Path(args.export_dir).resolve()
    if not export_dir.is_dir():
        print(f"Error: {export_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else export_dir.parent / "serverless"

    # Resolve SAHI path
    sahi_path = Path(args.sahi_path) if args.sahi_path else detect_sahi_path(Path.cwd())
    if sahi_path is None:
        print(
            "Error: Could not auto-detect SAHI fork path. "
            "Provide --sahi-path explicitly.",
            file=sys.stderr,
        )
        sys.exit(1)

    generate(
        export_dir=export_dir,
        output_dir=output_dir,
        runtime=args.runtime,
        sahi_path=sahi_path,
        name=args.name,
    )


if __name__ == "__main__":
    main()
