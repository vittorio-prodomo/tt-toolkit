"""Rich console + optional plain-text file logging for training-toolkit.

Call :func:`setup_logging` once at program startup. All modules then use
standard ``logging.getLogger(__name__)`` — no special imports needed.
"""

import logging
import warnings
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

_CONFIGURED = False

_FILE_FORMAT = "[%(asctime)s] %(levelname)-5s | %(name)s | %(message)s"
_FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    run_dir: Path,
    log_to_file: bool = True,
    level: str = "INFO",
) -> None:
    """Configure rich console logging + optional plain-text file logging.

    Args:
        run_dir: Directory for the log file (``train.log``).
        log_to_file: Write a plain-text log file alongside console output.
        level: Root log level.
    """
    global _CONFIGURED

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    theme = Theme({"log.time": "cyan"})
    console = RichHandler(
        console=Console(theme=theme),
        rich_tracebacks=True,
        show_path=False,
        markup=True,
        show_time=True,
        omit_repeated_times=False,
    )
    console.setLevel(level)
    root.addHandler(console)

    if log_to_file:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(run_dir / "train.log")
        file_handler.setFormatter(logging.Formatter(_FILE_FORMAT, datefmt=_FILE_DATEFMT))
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    tame_third_party_loggers()
    _silence_upstream_warnings()
    _CONFIGURED = True


_THIRD_PARTY_LOGGERS = (
    "lightning_utilities", "pytorch_lightning", "lightning_fabric",
    "transformers", "rf-detr", "rfdetr", "mlflow",
)


def tame_third_party_loggers() -> None:
    """Clear handlers on third-party loggers so they propagate to our root.

    A NullHandler is installed as a sentinel so that libraries which
    guard ``if not logger.handlers:`` (e.g. RF-DETR) won't re-add their
    own StreamHandlers on the next call.
    """
    for name in _THIRD_PARTY_LOGGERS:
        lib_logger = logging.getLogger(name)
        lib_logger.handlers.clear()
        lib_logger.addHandler(logging.NullHandler())
        lib_logger.propagate = True


def _silence_upstream_warnings() -> None:
    """Suppress noisy UserWarnings from upstream libraries we don't control."""
    warnings.filterwarnings("ignore", message="torch.meshgrid:")
    warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
    warnings.filterwarnings("ignore", message=".*more than 100 detections.*")
