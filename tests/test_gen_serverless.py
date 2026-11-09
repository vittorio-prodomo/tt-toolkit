"""Tests for serverless function generator."""

import json
import textwrap

import pytest
import yaml

from training_toolkit.serverless.generator import (
    build_function_yaml,
    build_label_spec,
    derive_function_name,
    detect_sahi_path,
    generate,
    has_mask_support,
    read_metadata,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def export_dir(tmp_path):
    """Create a minimal export directory with valid metadata."""
    d = tmp_path / "my-project" / "run-01" / "export"
    d.mkdir(parents=True)
    meta = {
        "model": "rf-detr-base",
        "backend": "rfdetr",
        "resolution": [560, 560],
        "class_names": ["crack", "spall"],
    }
    (d / "metadata.json").write_text(json.dumps(meta))
    (d / "inference_model.onnx").write_bytes(b"\x00")
    return d


@pytest.fixture()
def meta():
    return {
        "model": "rf-detr-base",
        "backend": "rfdetr",
        "class_names": ["crack", "spall"],
        "resolution": [560, 560],
    }


# ------------------------------------------------------------------
# read_metadata
# ------------------------------------------------------------------

class TestReadMetadata:
    def test_happy_path(self, export_dir):
        meta = read_metadata(export_dir)
        assert meta["backend"] == "rfdetr"
        assert meta["class_names"] == ["crack", "spall"]

    def test_dict_class_names_normalised(self, export_dir):
        raw = {
            "model": "rf-detr-base",
            "backend": "rfdetr",
            "resolution": [560, 560],
            "class_names": {"1": "spall", "0": "crack"},
        }
        (export_dir / "metadata.json").write_text(json.dumps(raw))
        meta = read_metadata(export_dir)
        assert meta["class_names"] == ["crack", "spall"]

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="metadata.json"):
            read_metadata(tmp_path)

    def test_missing_class_names(self, export_dir):
        raw = {"model": "rf-detr-base", "backend": "rfdetr"}
        (export_dir / "metadata.json").write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="class_names"):
            read_metadata(export_dir)

    def test_invalid_backend(self, export_dir):
        raw = {
            "model": "yolov8",
            "backend": "yolo",
            "class_names": ["a"],
        }
        (export_dir / "metadata.json").write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="Unsupported backend"):
            read_metadata(export_dir)


# ------------------------------------------------------------------
# derive_function_name
# ------------------------------------------------------------------

class TestDeriveFunctionName:
    def test_standard_layout(self, tmp_path):
        export = tmp_path / "bridge_defects" / "run_42" / "export"
        export.mkdir(parents=True)
        result = derive_function_name(export, "rf-detr-base")
        assert result == "bridge-defects-rf-detr-base-run-42"

    def test_flat_layout_fallback(self, tmp_path):
        flat = tmp_path / "my_model_dir"
        flat.mkdir()
        with pytest.warns(UserWarning, match="convention"):
            result = derive_function_name(flat, "rf-detr-base")
        assert result == "rf-detr-base"


# ------------------------------------------------------------------
# has_mask_support
# ------------------------------------------------------------------

class TestHasMaskSupport:
    def test_seg_model(self):
        assert has_mask_support("rf-detr-seg-medium", "rfdetr") is True

    def test_non_seg_model(self):
        assert has_mask_support("rf-detr-base", "rfdetr") is False

    def test_mask2former(self):
        assert has_mask_support("mask2former-swin-base", "mask2former") is True


# ------------------------------------------------------------------
# build_label_spec
# ------------------------------------------------------------------

class TestBuildLabelSpec:
    def test_with_masks(self):
        spec = build_label_spec(["crack", "spall"], has_masks=True)
        assert len(spec) == 2
        assert spec[0] == {"name": "crack", "type": "mask", "attributes": []}
        assert spec[1] == {"name": "spall", "type": "mask", "attributes": []}

    def test_without_masks(self):
        spec = build_label_spec(["crack"], has_masks=False)
        assert spec == [{"name": "crack", "type": "rectangle", "attributes": []}]


# ------------------------------------------------------------------
# build_function_yaml
# ------------------------------------------------------------------

class TestBuildFunctionYaml:
    def test_cpu_variant(self, meta, tmp_path):
        yaml_str = build_function_yaml(
            "test-func", meta,
            export_dir_abs=tmp_path / "export",
            sahi_path_abs=tmp_path / "sahi",
            runtime="ort", gpu=False,
        )
        doc = yaml.safe_load(yaml_str)
        assert doc["metadata"]["name"] == "test-func"
        assert doc["metadata"]["namespace"] == "cvat"
        assert doc["spec"]["runtime"] == "python:3.10"
        assert doc["spec"]["handler"] == "main:handler"
        assert doc["spec"]["eventTimeout"] == "300s"
        assert doc["spec"]["build"]["baseImage"] == "ubuntu:22.04"
        assert "resources" not in doc["spec"]
        # Volumes
        assert len(doc["spec"]["volumes"]) == 2
        assert doc["spec"]["volumes"][0]["volumeMount"]["mountPath"] == "/opt/nuclio/model"
        assert doc["spec"]["volumes"][1]["volumeMount"]["mountPath"] == "/opt/sahi"
        # Triggers
        trigger = doc["spec"]["triggers"]["default-http"]
        assert trigger["kind"] == "http"
        assert trigger["maxWorkers"] == 1
        # Platform
        assert doc["spec"]["platform"]["attributes"]["restartPolicy"]["name"] == "always"
        assert doc["spec"]["platform"]["attributes"]["mountMode"] == "volume"

    def test_gpu_variant(self, meta, tmp_path):
        yaml_str = build_function_yaml(
            "test-func", meta,
            export_dir_abs=tmp_path / "export",
            sahi_path_abs=None,
            runtime="trt", gpu=True,
        )
        doc = yaml.safe_load(yaml_str)
        assert doc["spec"]["build"]["baseImage"] == "nvcr.io/nvidia/tensorrt:24.08-py3"
        assert doc["spec"]["resources"]["limits"]["nvidia.com/gpu"] == 1
        # No SAHI volume when sahi_path is None
        assert len(doc["spec"]["volumes"]) == 1

    def test_extra_params_schema(self, meta, tmp_path):
        yaml_str = build_function_yaml(
            "test-func", meta,
            export_dir_abs=tmp_path / "export",
            sahi_path_abs=None,
            runtime="ort", gpu=False,
        )
        doc = yaml.safe_load(yaml_str)
        schema = json.loads(doc["metadata"]["annotations"]["extra_params_schema"])
        assert "use_sliced" in schema
        assert schema["slice_width"]["default"] == 1024
        assert schema["overlap_ratio"]["default"] == 0.25


# ------------------------------------------------------------------
# detect_sahi_path
# ------------------------------------------------------------------

class TestDetectSahiPath:
    def test_found_in_pyproject(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(textwrap.dedent("""\
            [tool.uv.sources]
            sahi = { path = "../sahi", editable = true }
        """))
        result = detect_sahi_path(tmp_path)
        assert result is not None
        assert result == (tmp_path / "../sahi").resolve()

    def test_not_found(self, tmp_path):
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[project]\nname = 'foo'\n")
        result = detect_sahi_path(tmp_path)
        assert result is None

    def test_no_pyproject(self, tmp_path):
        result = detect_sahi_path(tmp_path)
        assert result is None


# ------------------------------------------------------------------
# generate (integration-level)
# ------------------------------------------------------------------

class TestGenerate:
    def test_creates_output_dir(self, export_dir, tmp_path):
        out = tmp_path / "nuclio-func"
        result = generate(
            export_dir=export_dir,
            output_dir=out,
            runtime="ort",
            sahi_path=None,
        )
        assert result == out
        assert (out / "main.py").exists()
        assert (out / "model_handler.py").exists()
        assert (out / "function.yaml").exists()
        assert (out / "function-gpu.yaml").exists()

        # Verify GPU yaml has gpu resources
        gpu_doc = yaml.safe_load((out / "function-gpu.yaml").read_text())
        assert gpu_doc["spec"]["resources"]["limits"]["nvidia.com/gpu"] == 1

        # Verify CPU yaml does NOT have gpu resources
        cpu_doc = yaml.safe_load((out / "function.yaml").read_text())
        assert "resources" not in cpu_doc["spec"]

    def test_prompts_on_existing_dir_abort(self, export_dir, tmp_path, monkeypatch):
        out = tmp_path / "nuclio-func"
        out.mkdir()
        (out / "old_file.txt").write_text("old")

        monkeypatch.setattr("builtins.input", lambda _: "n")
        with pytest.raises(SystemExit, match="Aborted"):
            generate(export_dir=export_dir, output_dir=out, runtime="ort", sahi_path=None)

    def test_overwrites_on_confirmation(self, export_dir, tmp_path, monkeypatch):
        out = tmp_path / "nuclio-func"
        out.mkdir()
        (out / "old_file.txt").write_text("old")

        monkeypatch.setattr("builtins.input", lambda _: "y")
        result = generate(export_dir=export_dir, output_dir=out, runtime="ort", sahi_path=None)
        assert result == out
        assert (out / "main.py").exists()
        assert not (out / "old_file.txt").exists()

    def test_mask2former_handler_copy(self, tmp_path):
        """Verify Mask2Former handlers are copied with correct names."""
        export = tmp_path / "project" / "run" / "export"
        export.mkdir(parents=True)
        meta = {
            "model": "mask2former-swin-base",
            "backend": "mask2former",
            "resolution": [512, 512],
            "class_names": ["crack"],
        }
        (export / "metadata.json").write_text(json.dumps(meta))

        out = tmp_path / "nuclio-m2f"
        generate(export_dir=export, output_dir=out, runtime="ort", sahi_path=None)
        assert (out / "main.py").exists()
        assert (out / "model_handler.py").exists()


# ------------------------------------------------------------------
# End-to-end integration tests
# ------------------------------------------------------------------


class TestEndToEnd:
    def test_rfdetr_full_flow(self, export_dir):
        """Full generation flow for RF-DETR produces valid, complete output."""
        output = export_dir.parent / "serverless"
        generate(
            export_dir=export_dir,
            output_dir=output,
            runtime="ort",
            sahi_path=None,
            name="test-rfdetr-func",
        )

        # All expected files exist
        assert (output / "main.py").exists()
        assert (output / "model_handler.py").exists()
        assert (output / "function.yaml").exists()
        assert (output / "function-gpu.yaml").exists()

        # CPU YAML is valid and correct
        cpu_yaml = yaml.safe_load((output / "function.yaml").read_text())
        assert cpu_yaml["metadata"]["name"] == "test-rfdetr-func"
        assert "nvidia.com/gpu" not in str(cpu_yaml)

        # GPU YAML has GPU resource
        gpu_yaml = yaml.safe_load((output / "function-gpu.yaml").read_text())
        assert gpu_yaml["spec"]["resources"]["limits"]["nvidia.com/gpu"] == 1

        # Handler files have expected entry points
        main_content = (output / "main.py").read_text()
        assert "def handler(context, event)" in main_content
        assert "def init_context(context)" in main_content

    def test_mask2former_full_flow(self, tmp_path):
        """Full generation flow for Mask2Former."""
        export = tmp_path / "runs" / "proj" / "run1" / "export"
        export.mkdir(parents=True)
        meta = {
            "model": "mask2former-large",
            "backend": "mask2former",
            "resolution": [512, 512],
            "batch_size": 1,
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "opset": 17,
            "simplified": True,
            "class_names": ["crack", "spall"],
        }
        (export / "metadata.json").write_text(json.dumps(meta))
        (export / "inference_model.onnx").write_bytes(b"fake")

        output = export.parent / "serverless"
        generate(
            export_dir=export,
            output_dir=output,
            runtime="tensorrt",
            sahi_path=None,
        )

        assert (output / "main.py").exists()
        assert (output / "model_handler.py").exists()

        # Auto-derived name includes mask2former
        cpu_yaml = yaml.safe_load((output / "function.yaml").read_text())
        assert "mask2former" in cpu_yaml["metadata"]["name"]

        # All labels should be mask type
        spec = json.loads(cpu_yaml["metadata"]["annotations"]["spec"])
        assert all(s["type"] == "mask" for s in spec)


# ------------------------------------------------------------------
# CLI smoke test
# ------------------------------------------------------------------

class TestCLI:
    def test_gen_serverless_main_with_args(self, export_dir, tmp_path, monkeypatch):
        """Invoke the gen_serverless CLI entry point via sys.argv patching."""
        out = tmp_path / "cli-output"
        sahi_dir = tmp_path / "sahi"
        sahi_dir.mkdir()
        monkeypatch.setattr(
            "sys.argv",
            [
                "tt-gen-serverless",
                "--export-dir", str(export_dir),
                "--output-dir", str(out),
                "--runtime", "ort",
                "--sahi-path", str(sahi_dir),
            ],
        )
        from training_toolkit.scripts.gen_serverless import main
        main()
        assert (out / "main.py").exists()
        assert (out / "model_handler.py").exists()
        assert (out / "function.yaml").exists()
        assert (out / "function-gpu.yaml").exists()
