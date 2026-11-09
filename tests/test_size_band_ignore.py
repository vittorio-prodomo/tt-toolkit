"""Size-band per-region ignore in rfdetr.models.lwdetr.SetCriterion.

The criterion lives in the rf-detr fork, so these tests exercise it directly,
mirroring test_masked_loss.py.
"""
from __future__ import annotations

import pytest
import torch

# These criterion features exist only in the rf-detr fork pinned in pyproject.toml
# (uv installs it); skip cleanly on a plain install or with upstream rfdetr.
lwdetr = pytest.importorskip("rfdetr.models.lwdetr")
if not hasattr(lwdetr, "split_ignore_targets"):
    pytest.skip("needs the rf-detr fork (masked-loss / ignore-region criterion)", allow_module_level=True)

from rfdetr.models.lwdetr import box_ioa


def test_box_ioa_full_half_disjoint():
    a = torch.tensor([[0.0, 0.0, 1.0, 1.0]])           # one query box (unit area)
    b = torch.tensor([
        [0.0, 0.0, 1.0, 1.0],   # identical -> all of a is inside -> IoA 1.0
        [0.0, 0.0, 0.5, 1.0],   # covers left half of a -> IoA 0.5
        [2.0, 2.0, 3.0, 3.0],   # disjoint -> IoA 0.0
    ])
    ioa = box_ioa(a, b)  # (1, 3)
    assert ioa.shape == (1, 3)
    assert torch.allclose(ioa[0], torch.tensor([1.0, 0.5, 0.0]), atol=1e-6)


def test_box_ioa_is_area_of_a_not_b():
    # a is large, b is a tiny box fully inside a: only a small fraction of a is
    # covered -> IoA(a,b) is small (this is over area of A, not B).
    a = torch.tensor([[0.0, 0.0, 10.0, 10.0]])   # area 100
    b = torch.tensor([[0.0, 0.0, 1.0, 1.0]])     # area 1, inside a
    ioa = box_ioa(a, b)
    assert torch.allclose(ioa[0], torch.tensor([0.01]), atol=1e-6)


from rfdetr.models.lwdetr import split_ignore_targets


def test_split_ignore_targets_partitions_and_stashes():
    boxes = torch.tensor([[0.1, 0.1, 0.2, 0.2],
                          [0.5, 0.5, 0.2, 0.2],   # this one is ignore
                          [0.8, 0.8, 0.1, 0.1]])
    t = {
        "labels": torch.tensor([0, 1, 2]),
        "boxes": boxes,
        "ignore": torch.tensor([False, True, False]),
        "class_mask": torch.ones(3),
    }
    real = split_ignore_targets([t])
    assert len(real) == 1
    rt = real[0]
    assert rt["labels"].tolist() == [0, 2]          # ignore instance removed
    assert rt["boxes"].shape == (2, 4)
    assert "ignore" not in rt
    assert torch.equal(rt["class_mask"], torch.ones(3))   # passthrough
    assert rt["_ignore_boxes"].shape == (1, 4)
    assert torch.allclose(rt["_ignore_boxes"][0], boxes[1])


def test_split_ignore_targets_no_ignore_key_passes_through():
    t = {"labels": torch.tensor([0, 1]), "boxes": torch.zeros(2, 4)}
    real = split_ignore_targets([t])
    assert real[0]["labels"].tolist() == [0, 1]
    assert real[0]["_ignore_boxes"].shape == (0, 4)


from rfdetr.models.lwdetr import unmatched_ignore_overlap


def test_unmatched_ignore_overlap_flags_unmatched_overlappers_only():
    # 1 batch element, 3 queries (cxcywh). q0 sits on the ignore region and is
    # UNMATCHED -> dropped. q1 sits on it but is MATCHED -> kept. q2 is elsewhere.
    pred_boxes = torch.tensor([[
        [0.50, 0.50, 0.20, 0.20],   # q0 over ignore
        [0.50, 0.50, 0.20, 0.20],   # q1 over ignore (but matched)
        [0.10, 0.10, 0.05, 0.05],   # q2 elsewhere
    ]])
    targets = [{"_ignore_boxes": torch.tensor([[0.50, 0.50, 0.30, 0.30]])}]
    # indices: (pred_idx, tgt_idx) per batch — q1 is matched to some real GT
    indices = [(torch.tensor([1]), torch.tensor([0]))]
    drop = unmatched_ignore_overlap(pred_boxes, targets, indices, thresh=0.5)
    assert drop.shape == (1, 3)
    assert drop[0].tolist() == [True, False, False]


def test_unmatched_ignore_overlap_empty_when_no_ignore_boxes():
    pred_boxes = torch.rand(2, 4, 4)
    targets = [{"_ignore_boxes": torch.zeros(0, 4)},
               {"_ignore_boxes": torch.zeros(0, 4)}]
    indices = [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long))] * 2
    drop = unmatched_ignore_overlap(pred_boxes, targets, indices, thresh=0.5)
    assert drop.shape == (2, 4)
    assert not drop.any()


import pytest
from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.matcher import HungarianMatcher

NUM_CLASSES = 3
NUM_QUERIES = 20


def _make_criterion(size_band_ignore: bool) -> SetCriterion:
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, focal_alpha=0.25)
    return SetCriterion(
        num_classes=NUM_CLASSES,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        ia_bce_loss=True,
        size_band_ignore=size_band_ignore,
    )


def _make_outputs(seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "pred_logits": torch.randn(1, NUM_QUERIES, NUM_CLASSES, generator=g),
        "pred_boxes": torch.rand(1, NUM_QUERIES, 4, generator=g) * 0.5 + 0.25,
    }


def _targets(ignore=None):
    t = {
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]]),
    }
    if ignore is not None:
        t["ignore"] = torch.tensor(ignore)
    return [t]


def test_off_vs_on_identical_when_no_ignore_key():
    """size_band_ignore=True but no 'ignore' on targets -> identical to off."""
    off, on = _make_criterion(False), _make_criterion(True)
    lo = off(_make_outputs(1), _targets(ignore=None))
    ln = on(_make_outputs(1), _targets(ignore=None))
    for k in lo:
        assert torch.allclose(lo[k], ln[k], atol=1e-6), f"{k} diverged"


def test_all_real_ignore_false_matches_off():
    off, on = _make_criterion(False), _make_criterion(True)
    lo = off(_make_outputs(2), _targets(ignore=None))
    ln = on(_make_outputs(2), _targets(ignore=[False, False]))
    for k in lo:
        assert torch.allclose(lo[k], ln[k], atol=1e-6), f"{k} diverged"


def test_ignore_box_loss_equals_real_only_baseline():
    """The ignore instance is truly excluded from matching + box loss: the
    box/giou losses must equal a criterion fed ONLY the real instance. (loss_ce
    is expected to differ — the ignore path also drops the no-object term for
    queries overlapping the ignore region.)"""
    on, off = _make_criterion(True), _make_criterion(False)
    outs = _make_outputs(3)
    two = _targets(ignore=[False, True])
    real_only = [{"labels": torch.tensor([0]),
                  "boxes": torch.tensor([[0.3, 0.3, 0.2, 0.2]])}]
    L_on = on(outs, two)
    L_base = off(outs, real_only)
    assert torch.allclose(L_on["loss_bbox"], L_base["loss_bbox"], atol=1e-6)
    assert torch.allclose(L_on["loss_giou"], L_base["loss_giou"], atol=1e-6)


def test_no_object_drop_zeros_unmatched_overlapper_grad():
    """End-to-end: an UNMATCHED query whose box lies inside an ignore region gets
    zero classification gradient (no-object loss dropped), while an unmatched
    query OUTSIDE the ignore region is still supervised as background."""
    crit = _make_criterion(True)
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(1, 4, NUM_CLASSES, generator=g)
    pred_boxes = torch.tensor([[
        [0.1, 0.1, 0.10, 0.10],   # q0 matches the real GT
        [0.9, 0.1, 0.05, 0.05],   # q1 unmatched, OUTSIDE ignore -> supervised
        [0.1, 0.9, 0.05, 0.05],   # q2 unmatched, outside ignore
        [0.5, 0.5, 0.10, 0.10],   # q3 unmatched, INSIDE ignore -> dropped
    ]])
    targets = [{
        "labels": torch.tensor([0, 1]),
        "boxes": torch.tensor([[0.1, 0.1, 0.10, 0.10], [0.5, 0.5, 0.40, 0.40]]),
        "ignore": torch.tensor([False, True]),   # 2nd instance is the ignore region
    }]
    lg = logits.clone().requires_grad_(True)
    crit({"pred_logits": lg, "pred_boxes": pred_boxes}, targets)["loss_ce"].backward()
    assert torch.allclose(lg.grad[0, 3], torch.zeros(NUM_CLASSES), atol=1e-9)  # dropped
    assert lg.grad[0, 1].abs().sum() > 0  # unmatched but outside ignore -> supervised


def test_size_band_ignore_guards_varifocal():
    matcher = HungarianMatcher(cost_class=1.0, cost_bbox=1.0, cost_giou=1.0, focal_alpha=0.25)
    with pytest.raises(NotImplementedError):
        SetCriterion(
            num_classes=NUM_CLASSES, matcher=matcher,
            weight_dict={"loss_ce": 1.0}, focal_alpha=0.25, losses=["labels"],
            use_varifocal_loss=True, size_band_ignore=True,
        )
