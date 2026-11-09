"""Tests for Mask2Former masked-loss (requires transformers)."""
import pytest
import torch

pytest.importorskip("transformers")

from training_toolkit.models.mask2former_masked_loss import apply_class_mask


def test_apply_class_mask_ignores_unmatched_unannotated():
    # B=1, Q=3, logits over num_labels+1=3 (class0, class1, no-object)
    num_labels = 2
    pred_logits = torch.tensor([[
        [5., 0., 0.],   # q0 -> top class0
        [0., 5., 0.],   # q1 -> top class1 (will be masked)
        [5., 0., 0.],   # q2 -> top class0 (annotated)
    ]])
    target_classes = torch.tensor([[0, 2, 2]])   # q0 matched->class0; q1,q2 unmatched->no-object(2)
    class_mask = [torch.tensor([1., 0.])]        # class0 annotated, class1 NOT
    out = apply_class_mask(target_classes.clone(), pred_logits, class_mask, num_labels)
    assert out.tolist() == [[0, -100, 2]]        # only q1 ignored


def test_apply_class_mask_none_is_identity():
    tc = torch.tensor([[0, 2, 2]])
    out = apply_class_mask(tc.clone(), torch.zeros(1, 3, 3), None, 2)
    assert torch.equal(out, tc)


def test_apply_class_mask_matched_never_ignored():
    num_labels = 2
    pred_logits = torch.tensor([[[0., 5., 0.]]])  # q0 top=class1 (masked)
    target_classes = torch.tensor([[0]])          # q0 MATCHED to class0 (not no-object)
    class_mask = [torch.tensor([1., 0.])]
    out = apply_class_mask(target_classes.clone(), pred_logits, class_mask, num_labels)
    assert out.tolist() == [[0]]                  # matched query untouched


def test_apply_class_mask_per_sample_none():
    num_labels = 2
    pred_logits = torch.tensor([
        [[0., 5., 0.]],   # sample0 q0 -> class1
        [[0., 5., 0.]],   # sample1 q0 -> class1
    ])
    target_classes = torch.tensor([[2], [2]])     # both unmatched
    class_mask = [torch.tensor([1., 0.]), None]   # sample1 has no mask
    out = apply_class_mask(target_classes.clone(), pred_logits, class_mask, num_labels)
    assert out.tolist() == [[-100], [2]]          # sample0 ignored; sample1 untouched


def test_apply_class_mask_noobject_prediction_not_ignored():
    # q0 unmatched and its TOP prediction is no-object -> not ignored even if class1 masked
    num_labels = 2
    pred_logits = torch.tensor([[[0., 1., 5.]]])  # top = index 2 = no-object
    target_classes = torch.tensor([[2]])
    class_mask = [torch.tensor([1., 0.])]
    out = apply_class_mask(target_classes.clone(), pred_logits, class_mask, num_labels)
    assert out.tolist() == [[2]]                  # already predicting no-object -> keep


def _make_loss(cls, num_labels=2):
    from transformers import Mask2FormerConfig
    cfg = Mask2FormerConfig(num_labels=num_labels)
    weight_dict = {"loss_cross_entropy": 1.0, "loss_mask": 1.0, "loss_dice": 1.0}
    return cls(cfg, weight_dict)


def test_masked_loss_passthrough_matches_parent():
    from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss
    from training_toolkit.models.mask2former_masked_loss import MaskedMask2FormerLoss
    parent = _make_loss(Mask2FormerLoss)
    masked = _make_loss(MaskedMask2FormerLoss)   # _class_mask unset -> None -> passthrough
    logits = torch.randn(1, 3, 3)
    labels = [torch.tensor([0])]
    indices = [(torch.tensor([0]), torch.tensor([0]))]
    lp = parent.loss_labels(logits, labels, indices)["loss_cross_entropy"]
    lm = masked.loss_labels(logits, labels, indices)["loss_cross_entropy"]
    assert torch.allclose(lp, lm)


def test_masked_loss_differs_when_masking_active():
    from training_toolkit.models.mask2former_masked_loss import MaskedMask2FormerLoss
    masked = _make_loss(MaskedMask2FormerLoss)
    # 3 unmatched queries; q0 predicts class1 STRONGLY (distinct per-query loss),
    # q1/q2 predict class0 — so ignoring q0 changes the weighted-mean CE.
    logits = torch.tensor([[[0., 20., 0.], [5., 0., 0.], [5., 0., 0.]]])
    labels = [torch.tensor([], dtype=torch.int64)]                       # no GT -> all unmatched
    indices = [(torch.tensor([], dtype=torch.int64), torch.tensor([], dtype=torch.int64))]
    before = masked.loss_labels(logits, labels, indices)["loss_cross_entropy"]
    masked._class_mask = [torch.tensor([1., 0.])]                        # mask class1
    after = masked.loss_labels(logits, labels, indices)["loss_cross_entropy"]
    assert not torch.allclose(before, after)                            # q0 now ignored
    assert torch.isfinite(after)
