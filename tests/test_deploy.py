"""Tests for tt-deploy CLI."""

import shutil

import pytest

from training_toolkit.scripts.deploy import main as deploy_main


@pytest.fixture
def serverless_dir(tmp_path):
    """Create a fake serverless function directory."""
    d = tmp_path / "serverless"
    d.mkdir()
    (d / "main.py").write_text("# handler")
    (d / "model_handler.py").write_text("# model handler")
    (d / "function.yaml").write_text("metadata:\n  name: test")
    (d / "function-gpu.yaml").write_text("metadata:\n  name: test")
    return d


def test_deploy_missing_nuctl(serverless_dir, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    monkeypatch.setattr("sys.argv", ["tt-deploy", "--serverless-dir", str(serverless_dir)])
    with pytest.raises(SystemExit) as exc_info:
        deploy_main()
    assert "nuctl" in str(exc_info.value)


def test_deploy_missing_files(tmp_path, monkeypatch):
    d = tmp_path / "empty"
    d.mkdir()
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/local/bin/nuctl")
    monkeypatch.setattr("sys.argv", ["tt-deploy", "--serverless-dir", str(d)])
    with pytest.raises(SystemExit):
        deploy_main()
