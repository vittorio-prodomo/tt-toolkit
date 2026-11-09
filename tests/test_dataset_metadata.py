"""Tests for dataset metadata reading functions (_read_dvc_hash, _read_curation_metadata)."""

from pathlib import Path

import yaml


def _make_dvc_file(dataset_dir: Path, md5: str = "abc123") -> None:
    """Create a minimal .dvc file next to a dataset directory."""
    dvc_file = dataset_dir.with_suffix(".dvc")
    dvc_file.write_text(yaml.dump({"outs": [{"md5": md5, "path": dataset_dir.name}]}))


def _make_curation_file(dataset_dir: Path, meta: dict | None = None) -> None:
    """Create a curation.yaml inside a dataset directory."""
    dataset_dir.mkdir(parents=True, exist_ok=True)
    curation = meta or {
        "schema_version": 1,
        "created": "2026-03-30",
        "description": "test dataset",
        "target_classes": ["crack", "spall"],
        "sources": {
            "source_a": {
                "tiling": {"size": 416, "overlap": 0.3},
                "class_mapping": {"crack": "crack"},
            },
            "source_b": {
                "tiling": None,
                "class_mapping": {"spalling": "spall"},
            },
        },
    }
    (dataset_dir / "curation.yaml").write_text(yaml.dump(curation))


# --- _read_dvc_hash tests ---


def test_read_dvc_hash_returns_md5(tmp_path):
    from training_toolkit.engine.trainer import _read_dvc_hash

    ds = tmp_path / "my_dataset"
    ds.mkdir()
    _make_dvc_file(ds, md5="deadbeef")
    assert _read_dvc_hash(ds) == "deadbeef"


def test_read_dvc_hash_returns_none_when_no_dvc_file(tmp_path):
    from training_toolkit.engine.trainer import _read_dvc_hash

    ds = tmp_path / "my_dataset"
    ds.mkdir()
    assert _read_dvc_hash(ds) is None


def test_read_dvc_hash_returns_none_on_malformed_file(tmp_path):
    from training_toolkit.engine.trainer import _read_dvc_hash

    ds = tmp_path / "my_dataset"
    ds.mkdir()
    dvc_file = ds.with_suffix(".dvc")
    dvc_file.write_text("not: valid: dvc: content")
    assert _read_dvc_hash(ds) is None


# --- _read_curation_metadata tests ---


def test_read_curation_metadata_returns_dict(tmp_path):
    from training_toolkit.engine.trainer import _read_curation_metadata

    ds = tmp_path / "my_dataset"
    _make_curation_file(ds)
    result = _read_curation_metadata(ds)
    assert result is not None
    assert result["schema_version"] == 1
    assert result["target_classes"] == ["crack", "spall"]
    assert "source_a" in result["sources"]
    assert "source_b" in result["sources"]


def test_read_curation_metadata_returns_none_when_missing(tmp_path):
    from training_toolkit.engine.trainer import _read_curation_metadata

    ds = tmp_path / "my_dataset"
    ds.mkdir()
    assert _read_curation_metadata(ds) is None


def test_read_curation_metadata_returns_none_when_dir_missing(tmp_path):
    from training_toolkit.engine.trainer import _read_curation_metadata

    ds = tmp_path / "nonexistent"
    assert _read_curation_metadata(ds) is None


def test_read_curation_metadata_returns_none_on_malformed(tmp_path):
    from training_toolkit.engine.trainer import _read_curation_metadata

    ds = tmp_path / "my_dataset"
    ds.mkdir()
    (ds / "curation.yaml").write_text("{{bad yaml")
    assert _read_curation_metadata(ds) is None


def test_read_curation_metadata_preserves_null_tiling(tmp_path):
    from training_toolkit.engine.trainer import _read_curation_metadata

    ds = tmp_path / "my_dataset"
    _make_curation_file(ds)
    result = _read_curation_metadata(ds)
    assert result["sources"]["source_b"]["tiling"] is None


def test_read_curation_metadata_returns_none_on_non_dict(tmp_path):
    from training_toolkit.engine.trainer import _read_curation_metadata

    ds = tmp_path / "my_dataset"
    ds.mkdir()
    (ds / "curation.yaml").write_text("just a string")
    assert _read_curation_metadata(ds) is None


# --- _flatten_dict tests ---


def test_flatten_dict_simple():
    from training_toolkit.engine.trainer import _flatten_dict

    result = _flatten_dict({"a": 1, "b": "hello"})
    assert result == {"a": "1", "b": "hello"}


def test_flatten_dict_nested():
    from training_toolkit.engine.trainer import _flatten_dict

    result = _flatten_dict({"sources": {"ds1": {"size": 416}}})
    assert result == {"sources.ds1.size": "416"}


def test_flatten_dict_with_prefix():
    from training_toolkit.engine.trainer import _flatten_dict

    result = _flatten_dict({"size": 416}, prefix="dataset_curation")
    assert result == {"dataset_curation.size": "416"}


def test_flatten_dict_none_and_list_values():
    from training_toolkit.engine.trainer import _flatten_dict

    result = _flatten_dict({"tiling": None, "classes": ["crack", "spall"]})
    assert result["tiling"] == "None"
    assert result["classes"] == "['crack', 'spall']"
