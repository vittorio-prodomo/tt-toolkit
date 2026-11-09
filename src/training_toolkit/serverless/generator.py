"""Serverless function generator for Nuclio/CVAT deployment."""

from __future__ import annotations

import json
import logging
import re
import shutil
import warnings
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_VALID_BACKENDS = {"rfdetr", "mask2former"}

_HANDLERS_DIR = Path(__file__).parent / "handlers"

# SAHI extra-params schema (shared by CPU and GPU variants)
_EXTRA_PARAMS_SCHEMA = {
    "use_sliced": {"type": "boolean", "default": False},
    "slice_width": {"type": "number", "default": 1024},
    "slice_height": {"type": "number", "default": 1024},
    "overlap_ratio": {"type": "number", "default": 0.25},
}


# ------------------------------------------------------------------
# Metadata helpers
# ------------------------------------------------------------------

def read_metadata(export_dir: Path) -> dict:
    """Read and validate metadata.json from an export directory.

    Raises ``FileNotFoundError`` if the file is missing and ``ValueError``
    for invalid content (unsupported backend, missing class_names).
    """
    meta_path = export_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {export_dir}")

    meta = json.loads(meta_path.read_text())

    backend = meta.get("backend")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. "
            f"Expected one of {sorted(_VALID_BACKENDS)}."
        )

    class_names = meta.get("class_names")
    if class_names is None:
        raise ValueError(
            "metadata.json is missing 'class_names'. "
            "Re-export with --dataset-path to include class names."
        )

    # Normalize dict-style class_names → sorted list
    if isinstance(class_names, dict):
        meta["class_names"] = [
            class_names[k] for k in sorted(class_names, key=int)
        ]

    return meta


def _dns_sanitize(name: str) -> str:
    """Convert *name* to a DNS-safe identifier (lowercase, hyphens, alphanumeric)."""
    name = name.lower().replace("_", "-")
    name = re.sub(r"[^a-z0-9\-]", "", name)
    # Collapse consecutive hyphens and strip leading/trailing hyphens
    name = re.sub(r"-+", "-", name).strip("-")
    return name


def derive_function_name(export_dir: Path, model_name: str) -> str:
    """Derive a DNS-safe Nuclio function name from the export path.

    When the export directory follows the ``{project}/{run}/export/``
    convention produced by *tt-export*, the function name is
    ``{project}-{model_name}-{run}``.  Otherwise falls back to just
    the sanitised *model_name*.
    """
    if export_dir.name == "export":
        parent = export_dir.parent.name        # run name
        grandparent = export_dir.parent.parent.name  # project name
        raw = f"{grandparent}-{model_name}-{parent}"
        return _dns_sanitize(raw)

    warnings.warn(
        f"Export directory '{export_dir.name}' does not follow the "
        "'{project}/{run}/export/' convention. "
        "Falling back to model name for the function name.",
        stacklevel=2,
    )
    return _dns_sanitize(model_name)


# ------------------------------------------------------------------
# Label / spec helpers
# ------------------------------------------------------------------

def has_mask_support(model_name: str, backend: str) -> bool:
    """Return True if the model/backend combination supports instance masks."""
    if backend == "mask2former":
        return True
    return "seg" in model_name.lower()


def build_label_spec(class_names: list[str], has_masks: bool) -> list[dict]:
    """Build the CVAT annotation-spec label list."""
    label_type = "mask" if has_masks else "rectangle"
    return [
        {"name": name, "type": label_type, "attributes": []}
        for name in class_names
    ]


# ------------------------------------------------------------------
# YAML generation
# ------------------------------------------------------------------

def build_function_yaml(
    function_name: str,
    meta: dict,
    export_dir_abs: Path,
    sahi_path_abs: Path | None,
    runtime: str,
    gpu: bool,
) -> str:
    """Generate the Nuclio ``function.yaml`` content as a YAML string."""
    model_name = meta["model"]
    backend = meta["backend"]
    class_names = meta["class_names"]
    masks = has_mask_support(model_name, backend)

    spec_list = build_label_spec(class_names, masks)
    det_type = "detector"

    extra_params = json.dumps(_EXTRA_PARAMS_SCHEMA)

    # --- base image ---
    if runtime == "trt":
        base_image = "nvcr.io/nvidia/tensorrt:24.08-py3"
    else:
        base_image = "ubuntu:22.04"

    # --- pip directives ---
    pip_packages = [
        "numpy",
        "opencv-python-headless",
        "Pillow",
        "pyyaml",
    ]
    if runtime == "ort":
        pip_packages.append("onnxruntime")
    else:
        pip_packages.append("tensorrt")

    directives = {
        "preCopy": [
            {"kind": "ENV", "value": "DEBIAN_FRONTEND=noninteractive"},
            {
                "kind": "RUN",
                "value": (
                    "apt-get update && apt-get install -y --no-install-recommends "
                    "python3 python3-pip && rm -rf /var/lib/apt/lists/*"
                ),
            },
            {
                "kind": "RUN",
                "value": f"pip3 install --no-cache-dir {' '.join(pip_packages)}",
            },
        ],
    }

    # --- volumes ---
    volumes = [
        {
            "volume": {
                "name": "model",
                "hostPath": {"path": str(export_dir_abs)},
            },
            "volumeMount": {
                "name": "model",
                "mountPath": "/opt/nuclio/model",
            },
        },
    ]
    if sahi_path_abs is not None:
        volumes.append({
            "volume": {
                "name": "sahi",
                "hostPath": {"path": str(sahi_path_abs)},
            },
            "volumeMount": {
                "name": "sahi",
                "mountPath": "/opt/sahi",
            },
        })

    # --- triggers ---
    triggers = {
        "default-http": {
            "kind": "http",
            "workerAvailabilityTimeoutMilliseconds": 10000,
            "attributes": {
                "maxRequestBodySize": 67108864,  # 64 MB
            },
            "maxWorkers": 1,
        },
    }

    # --- assemble spec ---
    func = {
        "metadata": {
            "name": function_name,
            "namespace": "cvat",
            "annotations": {
                "name": function_name,
                "type": det_type,
                "framework": backend,
                "spec": json.dumps(spec_list),
                "extra_params_schema": extra_params,
            },
        },
        "spec": {
            "runtime": "python:3.10",
            "handler": "main:handler",
            "eventTimeout": "300s",
            "build": {
                "image": f"cvat.{function_name}",
                "baseImage": base_image,
                "directives": directives,
            },
            "volumes": volumes,
            "triggers": triggers,
            "platform": {
                "attributes": {
                    "restartPolicy": {
                        "name": "always",
                        "maximumRetryCount": 3,
                    },
                    "mountMode": "volume",
                },
            },
        },
    }

    if gpu:
        func["spec"]["resources"] = {
            "limits": {"nvidia.com/gpu": 1},
        }

    return yaml.dump(func, default_flow_style=False, sort_keys=False)


# ------------------------------------------------------------------
# SAHI path detection
# ------------------------------------------------------------------

def detect_sahi_path(project_root: Path) -> Path | None:
    """Auto-detect the SAHI fork path from ``pyproject.toml`` uv sources.

    Returns the resolved absolute path or ``None`` if not found.
    """
    pyproject = project_root / "pyproject.toml"
    if not pyproject.exists():
        return None

    try:
        import tomllib  # type: ignore[import-not-found]

        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        try:
            sahi_source = data["tool"]["uv"]["sources"]["sahi"]
            rel_path = sahi_source.get("path")
            if rel_path:
                return (project_root / rel_path).resolve()
        except (KeyError, TypeError):
            pass
        return None
    except ModuleNotFoundError:
        pass

    # Regex fallback for Python 3.10 without tomllib
    text = pyproject.read_text()
    match = re.search(r'sahi\s*=\s*\{[\s\S]*?path\s*=\s*"([^"]+)"', text)
    if match:
        return (project_root / match.group(1)).resolve()
    return None


# ------------------------------------------------------------------
# Top-level generation
# ------------------------------------------------------------------

def generate(
    export_dir: Path,
    output_dir: Path,
    runtime: str,
    sahi_path: Path | None,
    name: str | None = None,
) -> Path:
    """Generate a Nuclio serverless function directory.

    Parameters
    ----------
    export_dir:
        Directory containing the ONNX model and ``metadata.json``.
    output_dir:
        Where to write the generated function directory.
    runtime:
        Inference runtime — ``"ort"`` or ``"trt"``.
    sahi_path:
        Absolute path to the SAHI fork, or ``None``.
    name:
        Optional override for the Nuclio function name.

    Returns
    -------
    Path
        The *output_dir* that was created.
    """
    meta = read_metadata(export_dir)
    backend = meta["backend"]
    model_name = meta["model"]

    function_name = name or derive_function_name(export_dir, model_name)

    # Handle existing output directory
    if output_dir.exists() and any(output_dir.iterdir()):
        answer = input(f"Folder exists: {output_dir}\nOverwrite? (y/n) ")
        if answer.strip().lower() != "y":
            raise SystemExit("Aborted.")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy handler files
    main_src = _HANDLERS_DIR / f"{backend}_main.py"
    runtime_suffix = "ort" if runtime == "ort" else "trt"
    handler_src = _HANDLERS_DIR / f"{backend}_model_handler_{runtime_suffix}.py"

    shutil.copy2(main_src, output_dir / "main.py")
    shutil.copy2(handler_src, output_dir / "model_handler.py")

    # Generate function.yaml (CPU) and function-gpu.yaml
    export_abs = export_dir.resolve()
    sahi_abs = sahi_path.resolve() if sahi_path else None

    cpu_yaml = build_function_yaml(
        function_name, meta, export_abs, sahi_abs, runtime, gpu=False,
    )
    gpu_yaml = build_function_yaml(
        function_name, meta, export_abs, sahi_abs, runtime, gpu=True,
    )

    (output_dir / "function.yaml").write_text(cpu_yaml)
    (output_dir / "function-gpu.yaml").write_text(gpu_yaml)

    logger.info("Generated serverless function in %s", output_dir)
    return output_dir
