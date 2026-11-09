"""Tests for B-3b: per-regime + dual-scale window generation."""
from __future__ import annotations

import json

import cv2
import numpy as np

from training_toolkit.data.coco_dataset import CocoSegmentationDataset


def _coco_with_regimes(tmp_path):
    """Two images: one indoor_whole, one outdoor_tiled, each with one annotation."""
    for name in ("indoor.png", "outdoor.png"):
        cv2.imwrite(str(tmp_path / name), np.zeros((60, 80, 3), np.uint8))
    ann = {
        "images": [
            {"id": 1, "file_name": "indoor.png", "width": 80, "height": 60, "regime": "indoor_whole"},
            {"id": 2, "file_name": "outdoor.png", "width": 80, "height": 60, "regime": "outdoor_tiled"},
        ],
        "categories": [{"id": 1, "name": "c0"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]],
             "bbox": [5, 5, 15, 15], "area": 225, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]],
             "bbox": [5, 5, 15, 15], "area": 225, "iscrowd": 0},
        ],
    }
    root = tmp_path / "ds"
    (root / "annotations").mkdir(parents=True)
    (root / "images" / "train").mkdir(parents=True)
    for name in ("indoor.png", "outdoor.png"):
        cv2.imwrite(str(root / "images" / "train" / name), np.zeros((60, 80, 3), np.uint8))
    (root / "annotations" / "instances_train.json").write_text(json.dumps(ann))
    return CocoSegmentationDataset(str(root), split="train", layout="coco")


def test_image_regime_reads_field(tmp_path):
    ds = _coco_with_regimes(tmp_path)
    regimes = {ds._image_regime(i) for i in range(len(ds))}
    assert regimes == {"indoor_whole", "outdoor_tiled"}


def test_image_regime_default_outdoor_when_absent(tmp_path):
    # an image with no 'regime' field -> default outdoor_tiled (tile normally)
    ds = _coco_with_regimes(tmp_path)
    # monkeypatch one image record to drop regime
    img = ds.coco.loadImgs(ds.image_ids[0])[0]
    img.pop("regime", None)
    assert ds._image_regime(0) == "outdoor_tiled"


# ---------------------------------------------------------------------------
# build_tile_index — regime-aware + multi-scale
# ---------------------------------------------------------------------------
from training_toolkit.data.tile_index import build_tile_index


def test_build_tile_index_indoor_one_whole_window_no_band(tmp_path):
    ds = _coco_with_regimes(tmp_path)
    grouped = build_tile_index(ds, tile_sizes=[32], overlap=0.0,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=1)
    # image 0 is indoor_whole -> exactly one window covering the whole frame, apply_band=False
    indoor_windows = grouped[0]
    assert len(indoor_windows) == 1
    (box, apply_band) = indoor_windows[0]
    assert box == (0, 0, 80, 60)
    assert apply_band is False


def test_build_tile_index_outdoor_dual_scale_banded(tmp_path):
    ds = _coco_with_regimes(tmp_path)
    grouped = build_tile_index(ds, tile_sizes=[16, 48], overlap=0.0,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=1)
    outdoor_windows = grouped[1]   # image 1 is outdoor_tiled
    # every outdoor window is apply_band=True; tiles come from BOTH sizes
    assert all(ab is True for (_b, ab) in outdoor_windows)
    sides = {max(b[2], b[3]) for (b, _ab) in outdoor_windows}
    assert any(s <= 16 for s in sides) and any(s > 16 for s in sides)  # both scales present


def test_build_tile_index_negatives_outdoor_only(tmp_path):
    ds = _coco_with_regimes(tmp_path)
    grouped = build_tile_index(ds, tile_sizes=[16], overlap=0.0,
                               negative_ratio=0.5, min_annotation_overlap=0.05, seed=1)
    # indoor image keeps exactly its 1 positive whole-window (no negatives added there)
    assert len(grouped[0]) == 1


# ---------------------------------------------------------------------------
# Streaming driver — tile_sizes + per-window band/realized_scale
# ---------------------------------------------------------------------------
from training_toolkit.data.label_crop import SizeBand
from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset


def test_streaming_indoor_whole_no_band_outdoor_banded(tmp_path):
    ds = _coco_with_regimes(tmp_path)
    stream = StreamingTiledSegmentationDataset(
        ds, tile_sizes=[16, 48], overlap=0.0, negative_ratio=0.0,
        buffer_size=8, imgsz=64, size_band=SizeBand.for_imgsz(64),
    )
    samples = list(stream)
    # indoor frame (80x60) -> one whole-frame sample; outdoor -> several tiles
    assert len(samples) >= 2
    # every sample carries ignore (B-3a); the indoor whole-frame sample is the
    # one whose image is the full 60x80 frame
    shapes = {s["image"].shape[:2] for s in samples}
    assert (60, 80) in shapes            # indoor whole-frame present
    assert all("ignore" in s for s in samples)


# ---------------------------------------------------------------------------
# TilingConfig.tile_sizes + trainer wiring
# ---------------------------------------------------------------------------
from training_toolkit.configs.tiling_config import TilingConfig


def test_tiling_config_has_tile_sizes():
    tc = TilingConfig()
    assert hasattr(tc, "tile_sizes")
    assert tc.tile_sizes is None  # default: fall back to [tile_size]


def test_smoke_indoor_no_band_outdoor_dual_scale_banded(tmp_path):
    """Regime correctness gate (smoke):
    - indoor → one whole-frame sample, band=None (no size-band ignores from the
      band itself; ignores array is empty / all False).
    - outdoor → dual-scale banded tiles, at least one can carry an ignore=True.
    Gate: indoor → one whole-frame, band-free; outdoor → dual-scale banded tiles.
    """
    ds = _coco_with_regimes(tmp_path)
    stream = StreamingTiledSegmentationDataset(
        ds, tile_sizes=[16, 48], overlap=0.0, negative_ratio=0.0,
        buffer_size=16, imgsz=64, size_band=SizeBand.for_imgsz(64),
    )
    samples = list(stream)

    # Separate indoor vs outdoor samples by shape
    indoor_samples = [s for s in samples if s["image"].shape[:2] == (60, 80)]
    outdoor_samples = [s for s in samples if s["image"].shape[:2] != (60, 80)]

    # Indoor: exactly one whole-frame (80x60) sample
    assert len(indoor_samples) == 1, f"Expected 1 indoor whole-frame, got {len(indoor_samples)}"

    # Indoor: apply_band was False → realized_scale = 64/80 = 0.8
    # The annotation bbox=[5,5,15,15] at that scale → effective size ≈ 12px
    # No size-band ignores from the band (band=None → label_crop returns all not-ignored)
    indoor = indoor_samples[0]
    assert "ignore" in indoor
    # ignore should be all False (band=None means no size-band filtering applied)
    assert not indoor["ignore"].any(), "Indoor sample should have no size-band ignores"

    # Outdoor: multiple tiles from both tile sizes (16 and 48)
    assert len(outdoor_samples) >= 2, f"Expected >=2 outdoor tiles, got {len(outdoor_samples)}"
    outdoor_tile_sizes = {max(s["image"].shape[0], s["image"].shape[1]) for s in outdoor_samples}
    assert any(sz <= 16 for sz in outdoor_tile_sizes), "Expected 16-px tiles from outdoor"
    assert any(sz > 16 for sz in outdoor_tile_sizes), "Expected 48-px tiles from outdoor"

    # The band actually fires for outdoor — at least one annotation is out-of-band
    # at one of the scales (regression guard against apply_band always being False).
    assert any(
        s["ignore"].any() for s in outdoor_samples if len(s["ignore"]) > 0
    ), "Expected at least one out-of-band (ignore=True) outdoor annotation"
