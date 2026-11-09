"""Sanity tests for the masked-loss path in the RF-DETR criterion.

The criterion lives in the rf-detr fork (``rfdetr.models.lwdetr.SetCriterion``);
these tests exercise the patched ``loss_labels`` to verify that:

1. An all-ones ``class_mask`` produces the same loss as no masked-loss at all
   (invariance — flipping the flag on with an empty mask is a no-op).
2. A class with mask=0 contributes zero gradient to ``pred_logits`` along that
   channel (the actual masking semantics).
3. The propagation from a target dict through ``criterion.forward`` works.

Detection-only criteria (no segmentation head) are used to keep the test light.
"""

from __future__ import annotations

import pytest
import torch

rfdetr_pkg = pytest.importorskip("rfdetr")

from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.matcher import HungarianMatcher


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #

NUM_REAL_CLASSES = 3        # e.g. C2 / C5 / C6 for the stain family
NUM_QUERIES = 8             # small to keep matching deterministic-ish
BATCH = 2


def _make_criterion(masked_loss: bool, ia_bce: bool = True) -> SetCriterion:
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, focal_alpha=0.25)
    weight_dict = {"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0}
    return SetCriterion(
        num_classes=NUM_REAL_CLASSES + 1,
        matcher=matcher,
        weight_dict=weight_dict,
        focal_alpha=0.25,
        losses=["labels", "boxes", "cardinality"],
        ia_bce_loss=ia_bce,
        masked_loss=masked_loss,
    )


def _make_outputs(seed: int = 0) -> dict:
    """Synthetic outputs matching what the model produces (no aux/enc/masks)."""
    g = torch.Generator().manual_seed(seed)
    pred_logits = torch.randn(BATCH, NUM_QUERIES, NUM_REAL_CLASSES + 1, generator=g, requires_grad=True)
    # boxes in cxcywh, normalized to [0, 1]
    pred_boxes = torch.rand(BATCH, NUM_QUERIES, 4, generator=g)
    pred_boxes = pred_boxes * 0.5 + 0.1  # keep boxes well inside the unit square
    return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


def _make_targets(class_mask: torch.Tensor | None = None) -> list[dict]:
    """Two-sample targets with one GT instance each (classes 0 and 1)."""
    targets = [
        {
            "labels": torch.tensor([0], dtype=torch.int64),
            "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]]),  # cxcywh, normalized
        },
        {
            "labels": torch.tensor([1], dtype=torch.int64),
            "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
        },
    ]
    if class_mask is not None:
        # Broadcast the same mask to every sample; tests can override per-sample
        if class_mask.ndim == 1:
            for t in targets:
                t["class_mask"] = class_mask.clone()
        elif class_mask.ndim == 2:
            assert class_mask.shape[0] == len(targets)
            for t, cm in zip(targets, class_mask):
                t["class_mask"] = cm.clone()
        else:
            raise ValueError(f"class_mask must be 1D or 2D, got {class_mask.ndim}D")
    return targets


# ---------------------------------------------------------------------- #
# Tests
# ---------------------------------------------------------------------- #

def test_all_ones_mask_matches_unmasked():
    """masked_loss=True with class_mask = all-ones must equal masked_loss=False."""
    outputs_a = _make_outputs(seed=42)
    outputs_b = {k: (v.detach().clone().requires_grad_(v.requires_grad)) for k, v in outputs_a.items()}

    criterion_off = _make_criterion(masked_loss=False)
    criterion_on = _make_criterion(masked_loss=True)

    targets_off = _make_targets(class_mask=None)
    targets_on = _make_targets(class_mask=torch.ones(NUM_REAL_CLASSES))

    losses_off = criterion_off(outputs_a, targets_off)
    losses_on = criterion_on(outputs_b, targets_on)

    for key in ("loss_ce", "loss_bbox", "loss_giou"):
        assert torch.allclose(losses_off[key], losses_on[key], atol=1e-6), (
            f"{key} diverged: off={losses_off[key].item()}, on={losses_on[key].item()}"
        )


def test_zero_mask_zeros_gradient_on_masked_channel():
    """When class c has mask=0, d(loss_ce)/d(pred_logits[..., c]) must be exactly 0."""
    outputs = _make_outputs(seed=1)
    criterion = _make_criterion(masked_loss=True)

    # Mask out class 1 for BOTH samples
    class_mask = torch.tensor([1.0, 0.0, 1.0])
    targets = _make_targets(class_mask=class_mask)
    # Drop the GT of class 1 from sample 2 — masked class can't carry GT in a
    # well-formed dataset (we don't have labels for what we say we don't know).
    targets[1]["labels"] = torch.empty(0, dtype=torch.int64)
    targets[1]["boxes"] = torch.empty(0, 4)

    losses = criterion(outputs, targets)
    losses["loss_ce"].backward()

    grad = outputs["pred_logits"].grad
    # Channel for class 1 (the masked one) must have exactly zero gradient.
    assert torch.all(grad[:, :, 1] == 0), (
        f"Expected zero gradient on masked channel 1, got max |grad| = {grad[:, :, 1].abs().max().item()}"
    )
    # Other real-class channels must NOT be all-zero (the loss should still flow there).
    assert grad[:, :, 0].abs().sum() > 0, "Channel 0 gradient unexpectedly zero"
    assert grad[:, :, 2].abs().sum() > 0, "Channel 2 gradient unexpectedly zero"


def test_per_sample_masks_are_independent():
    """Different class_masks on different samples must take effect per-sample."""
    outputs = _make_outputs(seed=7)
    criterion = _make_criterion(masked_loss=True)

    # Sample 0: only class 0 active.  Sample 1: only class 2 active.
    per_sample_mask = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    targets = _make_targets(class_mask=per_sample_mask)
    # Drop GTs that fall on inactive classes (well-formed dataset assumption).
    targets[0]["labels"] = torch.tensor([0], dtype=torch.int64)
    targets[1]["labels"] = torch.tensor([2], dtype=torch.int64)

    losses = criterion(outputs, targets)
    losses["loss_ce"].backward()

    grad = outputs["pred_logits"].grad
    # Sample 0 should have no gradient on channels 1 and 2.
    assert torch.all(grad[0, :, 1] == 0)
    assert torch.all(grad[0, :, 2] == 0)
    # Sample 1 should have no gradient on channels 0 and 1.
    assert torch.all(grad[1, :, 0] == 0)
    assert torch.all(grad[1, :, 1] == 0)


def test_no_class_mask_in_targets_acts_as_all_ones():
    """When masked_loss=True but no target carries a class_mask, behavior == all-ones."""
    outputs_a = _make_outputs(seed=99)
    outputs_b = {k: (v.detach().clone().requires_grad_(v.requires_grad)) for k, v in outputs_a.items()}

    criterion_on = _make_criterion(masked_loss=True)
    criterion_off = _make_criterion(masked_loss=False)

    # Both criteria see the same targets without any class_mask key.
    losses_on = criterion_on(outputs_a, _make_targets(class_mask=None))
    losses_off = criterion_off(outputs_b, _make_targets(class_mask=None))

    for key in ("loss_ce", "loss_bbox", "loss_giou"):
        assert torch.allclose(losses_off[key], losses_on[key], atol=1e-6), (
            f"{key} diverged when no class_mask was supplied: "
            f"off={losses_off[key].item()}, on={losses_on[key].item()}"
        )


def test_default_branch_focal_loss_masking():
    """Verify masking works in the non-ia_bce_loss path (default sigmoid_focal_loss)."""
    outputs = _make_outputs(seed=3)
    criterion = _make_criterion(masked_loss=True, ia_bce=False)

    class_mask = torch.tensor([1.0, 0.0, 1.0])
    targets = _make_targets(class_mask=class_mask)
    targets[1]["labels"] = torch.empty(0, dtype=torch.int64)
    targets[1]["boxes"] = torch.empty(0, 4)

    losses = criterion(outputs, targets)
    losses["loss_ce"].backward()

    grad = outputs["pred_logits"].grad
    assert torch.all(grad[:, :, 1] == 0), (
        f"Expected zero gradient on masked channel 1 (default branch), got max |grad| = {grad[:, :, 1].abs().max().item()}"
    )


def test_masked_loss_rejects_varifocal_combination():
    """masked_loss=True with use_varifocal_loss must raise NotImplementedError."""
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, focal_alpha=0.25)
    with pytest.raises(NotImplementedError, match="masked_loss"):
        SetCriterion(
            num_classes=NUM_REAL_CLASSES + 1,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0},
            focal_alpha=0.25,
            losses=["labels"],
            use_varifocal_loss=True,
            masked_loss=True,
        )
