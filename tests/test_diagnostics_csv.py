"""results.csv must survive --resume: re-instantiating the DiagnosticsCallback on
the same run_dir (as resume does) and writing further epochs must PRESERVE every
prior epoch's values, never blank a populated cell, and never duplicate an epoch
row. Regression for a YOLO resume that blanked epochs 0-10 of results.csv
(the pre-resume trajectory was only recoverable from W&B).
"""
import csv

import pytest

from training_toolkit.callbacks.diagnostics import DiagnosticsCallback


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_resume_preserves_prior_epochs(tmp_path):
    cb = DiagnosticsCallback(run_dir=tmp_path)
    for e in range(3):
        cb._write_csv_row({"epoch": e, "train/loss": 1.0 - e, "val/map": 0.10 + 0.05 * e})

    # resume: a FRESH callback on the same run_dir, continue with epochs 3-4
    cb2 = DiagnosticsCallback(run_dir=tmp_path)
    for e in (3, 4):
        cb2._write_csv_row({"epoch": e, "train/loss": 0.5, "val/map": 0.25 + 0.01 * e})

    rows = _read(tmp_path / "results.csv")
    by_epoch = {int(r["epoch"]): r for r in rows}
    assert len(rows) == 5 and sorted(by_epoch) == [0, 1, 2, 3, 4]      # no dups, all epochs
    assert float(by_epoch[0]["val/map"]) == pytest.approx(0.10)                       # early epoch preserved
    assert float(by_epoch[1]["val/map"]) == pytest.approx(0.15)
    assert float(by_epoch[4]["val/map"]) == pytest.approx(0.29)                       # resumed epoch present


def test_resume_reval_neither_blanks_nor_duplicates(tmp_path):
    """The resume sanity-revalidation re-emits an epoch WITHOUT val/map (only
    losses). It must neither blank that epoch's existing val/map nor add a 2nd row."""
    cb = DiagnosticsCallback(run_dir=tmp_path)
    for e in range(3):
        cb._write_csv_row({"epoch": e, "val/map": 0.10 + 0.05 * e})

    cb2 = DiagnosticsCallback(run_dir=tmp_path)
    cb2._write_csv_row({"epoch": 2, "val/loss": 5.0})                  # sanity re-val, no map

    rows = _read(tmp_path / "results.csv")
    ep2 = [r for r in rows if int(r["epoch"]) == 2]
    assert len(ep2) == 1                                               # not duplicated
    assert float(ep2[0]["val/map"]) == pytest.approx(0.20)                           # not blanked


def test_new_column_on_resume_keeps_old_rows(tmp_path):
    """A metric first seen after resume (e.g. val_tide enabled) adds a column; old
    rows must keep their values and get a blank for the new column, not lose data."""
    cb = DiagnosticsCallback(run_dir=tmp_path)
    for e in range(2):
        cb._write_csv_row({"epoch": e, "val/map": 0.10 + 0.05 * e})

    cb2 = DiagnosticsCallback(run_dir=tmp_path)
    cb2._write_csv_row({"epoch": 2, "val/map": 0.22, "val/tide_FalsePos": 12.3})

    rows = _read(tmp_path / "results.csv")
    by_epoch = {int(r["epoch"]): r for r in rows}
    assert float(by_epoch[0]["val/map"]) == pytest.approx(0.10)                      # old row intact
    assert by_epoch[0].get("val/tide_FalsePos", "") == ""             # new col blank for old row
    assert float(by_epoch[2]["val/tide_FalsePos"]) == pytest.approx(12.3)           # new col populated
