"""Functional smoke test: class_mask + ignore flow end-to-end on the online path.

Verifies:
  (a) online+masked_loss does NOT raise (guard is gone)
  (b) collated batch targets carry both class_mask and ignore
  (c) a SetCriterion with masked_loss=True + size_band_ignore=True runs a
      finite backward (CPU-only, no full model needed)

No GPU, no full tt-train invocation.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_coco_with_class_mask(tmp_path, n_images: int = 3):
    """Build a minimal COCO-layout dataset where every image has a class_mask."""
    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)

    images, anns = [], []
    for k in range(n_images):
        img = np.full((64, 64, 3), k * 40, np.uint8)
        cv2.imwrite(str(img_dir / f"img_{k:03d}.png"), img)
        images.append({
            "id": k + 1,
            "file_name": f"img_{k:03d}.png",
            "width": 64,
            "height": 64,
            "class_mask": [1, 0, 1],   # 3-class mask
        })
        # One small annotation (side=15 px) and one big one (side=50 px)
        anns.append({
            "id": 2 * k + 1, "image_id": k + 1, "category_id": 1,
            "segmentation": [[5, 5, 20, 5, 20, 20, 5, 20]],
            "bbox": [5, 5, 15, 15], "area": 225, "iscrowd": 0,
        })
        anns.append({
            "id": 2 * k + 2, "image_id": k + 1, "category_id": 2,
            "segmentation": [[7, 7, 57, 7, 57, 57, 7, 57]],
            "bbox": [7, 7, 50, 50], "area": 2500, "iscrowd": 0,
        })

    coco = {
        "images": images,
        "annotations": anns,
        "categories": [
            {"id": 1, "name": "crack"},
            {"id": 2, "name": "spall"},
            {"id": 3, "name": "rebar"},
        ],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(coco))
    return tmp_path


# ---------------------------------------------------------------------------
# Test A: online path now constructs without error even with masked_loss
# ---------------------------------------------------------------------------

def test_online_masked_loss_no_longer_raises(tmp_path):
    """The guard is gone — online + masked_loss must construct successfully."""
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    from training_toolkit.data.label_crop import SizeBand

    root = _make_coco_with_class_mask(tmp_path)
    base = CocoSegmentationDataset(root=str(root), split="train", layout="coco")

    # This must NOT raise (old code would have raised RuntimeError)
    ds = StreamingTiledSegmentationDataset(
        base,
        tile_size=32,
        overlap=0.0,
        negative_ratio=0.0,
        buffer_size=4,
        size_band=SizeBand.for_imgsz(32),
        imgsz=32,
    )
    assert len(ds) >= 1


# ---------------------------------------------------------------------------
# Test B: collated batch targets carry class_mask AND ignore on the online path
# ---------------------------------------------------------------------------

def test_online_sample_carries_class_mask_and_ignore(tmp_path):
    """Pull one sample from the streaming dataset and collate it — targets must
    have both class_mask and ignore (every tile emits ignore from label_crop,
    class_mask is attached when the COCO image record carries one)."""
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    from training_toolkit.data.label_crop import SizeBand
    from training_toolkit.data.collate import collate_fn

    root = _make_coco_with_class_mask(tmp_path)
    base = CocoSegmentationDataset(root=str(root), split="train", layout="coco")

    ds = StreamingTiledSegmentationDataset(
        base,
        tile_size=32,
        overlap=0.0,
        negative_ratio=0.0,
        buffer_size=4,
        size_band=SizeBand.for_imgsz(32),
        imgsz=32,
    )

    sample = next(iter(ds))

    # Raw sample must carry ignore (always) and class_mask (COCO has it)
    assert "ignore" in sample, "streaming sample must carry 'ignore'"
    assert "class_mask" in sample, "streaming sample must carry 'class_mask' (COCO images have it)"
    assert isinstance(sample["ignore"], np.ndarray)
    assert sample["ignore"].dtype == bool

    # After collate, targets[0] must still carry both
    batch = collate_fn([sample])
    t = batch["targets"][0]
    assert "ignore" in t, "collated target must carry 'ignore'"
    assert "class_mask" in t, "collated target must carry 'class_mask'"
    assert t["ignore"].dtype == torch.bool


# ---------------------------------------------------------------------------
# Test C: criterion with masked_loss+size_band_ignore runs a finite backward
# ---------------------------------------------------------------------------

def test_criterion_masked_and_size_band_backward():
    """Build SetCriterion with both flags, feed synthetic outputs + a target
    carrying class_mask + ignore, assert finite loss_ce and successful backward."""
    from rfdetr.models.lwdetr import SetCriterion
    from rfdetr.models.matcher import HungarianMatcher

    NUM_CLASSES = 3
    NUM_QUERIES = 10

    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0,
                               focal_alpha=0.25)
    criterion = SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        ia_bce_loss=True,
        masked_loss=True,
        size_band_ignore=True,
        ignore_ioa_thresh=0.5,
    )

    g = torch.Generator().manual_seed(0)
    outputs = {
        "pred_logits": torch.randn(1, NUM_QUERIES, NUM_CLASSES, generator=g,
                                   requires_grad=True),
        "pred_boxes": (torch.rand(1, NUM_QUERIES, 4, generator=g) * 0.4 + 0.3),
    }
    targets = [{
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
        "ignore": torch.tensor([False, True]),       # second GT is ignore
        "class_mask": torch.tensor([1.0, 1.0, 0.0]),  # class 2 masked out
    }]

    loss_dict = criterion(outputs, targets)

    loss_ce = loss_dict["loss_ce"]
    assert torch.isfinite(loss_ce), f"loss_ce not finite: {loss_ce}"

    # Backward must complete without error
    total_loss = sum(v for k, v in loss_dict.items()
                     if k in criterion.weight_dict and torch.is_tensor(v))
    total_loss.backward()

    # grad must flow back to pred_logits
    assert outputs["pred_logits"].grad is not None
    assert torch.isfinite(outputs["pred_logits"].grad).all()
