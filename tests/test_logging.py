"""Tests for the logging setup module."""

import logging
from pathlib import Path

import pytest

from training_toolkit.logging import setup_logging


@pytest.fixture(autouse=True)
def _clean_handlers():
    """Reset root logger handlers before each test."""
    yield
    logging.getLogger().handlers.clear()


def test_setup_logging_creates_console_handler(tmp_path):
    setup_logging(run_dir=tmp_path, log_to_file=False)
    root = logging.getLogger()
    assert len(root.handlers) >= 1


def test_setup_logging_creates_log_file(tmp_path):
    setup_logging(run_dir=tmp_path, log_to_file=True)
    logger = logging.getLogger("training_toolkit.test")
    logger.info("test message")
    log_file = tmp_path / "train.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "test message" in content


def test_setup_logging_file_has_timestamps(tmp_path):
    setup_logging(run_dir=tmp_path, log_to_file=True)
    logger = logging.getLogger("training_toolkit.test_ts")
    logger.info("timestamped")
    content = (tmp_path / "train.log").read_text()
    assert "[" in content and "]" in content


def test_setup_logging_no_file_when_disabled(tmp_path):
    setup_logging(run_dir=tmp_path, log_to_file=False)
    assert not (tmp_path / "train.log").exists()
