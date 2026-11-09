"""CLI entry point: tt-deploy."""

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deploy a Nuclio serverless function to the local CVAT instance.\n\n"
            "Usage:\n"
            "  tt-deploy --serverless-dir runs/my_project/my_run/serverless/\n"
            "  tt-deploy --serverless-dir serverless/ --gpu\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--serverless-dir", dest="serverless_dir", required=True,
                        help="Serverless function directory (from tt-gen-serverless)")
    parser.add_argument("--gpu", action="store_true", default=False,
                        help="Use function-gpu.yaml instead of function.yaml")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Check nuctl first -- most actionable error
    if shutil.which("nuctl") is None:
        sys.exit(
            "Error: nuctl not found on PATH. "
            "Install it from https://nuclio.io/docs/latest/setup/"
        )

    serverless_dir = Path(args.serverless_dir).resolve()
    if not serverless_dir.is_dir():
        print(f"Error: {serverless_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    # Validate expected files
    yaml_file = "function-gpu.yaml" if args.gpu else "function.yaml"
    required = ["main.py", "model_handler.py", yaml_file]
    missing = [f for f in required if not (serverless_dir / f).exists()]
    if missing:
        print(
            f"Error: Missing files in {serverless_dir}: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    cmd = [
        "nuctl", "deploy",
        "--project-name", "cvat",
        "--platform", "local",
        "--path", str(serverless_dir),
        "--file", str(serverless_dir / yaml_file),
        "--env", "CVAT_FUNCTIONS_REDIS_HOST=cvat_redis_ondisk",
        "--env", "CVAT_FUNCTIONS_REDIS_PORT=6666",
        "--platform-config",
        '{"attributes": {"network": "cvat_cvat"}}',
    ]

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    logger.info("Deployment successful.")


if __name__ == "__main__":
    main()
