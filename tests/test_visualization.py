from training_toolkit.visualization import save_training_curves


def test_save_training_curves_missing_csv_is_noop(tmp_path):
    """A val-off run never writes results.csv; save_training_curves must
    no-op instead of raising FileNotFoundError (regression for the DDP
    post-training plot crash)."""
    missing = tmp_path / "results.csv"  # does not exist
    out = tmp_path / "curves.png"
    save_training_curves(missing, out)  # must return cleanly
    assert not out.exists()  # nothing was plotted
