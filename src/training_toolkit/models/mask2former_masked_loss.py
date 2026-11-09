"""Masked-loss variant of HuggingFace's Mask2FormerLoss.

Couples to transformers internals (Mask2FormerLoss); tested against 4.57.6.
The classification-loss masking is factored into the pure function
``apply_class_mask`` so it can be unit-tested without constructing a model.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss


def apply_class_mask(target_classes: Tensor, pred_logits: Tensor,
                     class_mask: list | None, num_labels: int) -> Tensor:
    """Set ignore_index (-100) on *unmatched* queries (target == num_labels) whose
    overall top prediction is a *real* class that is unannotated in that sample.

    Args:
        target_classes: (B, Q) int64 — per-query CE target (num_labels == no-object).
        pred_logits: (B, Q, num_labels + 1) class logits.
        class_mask: per-sample list; each entry a (num_labels,) 0/1 tensor, or None.
        num_labels: number of real classes (no-object index == num_labels).
    """
    if class_mask is None:
        return target_classes
    out = target_classes.clone()
    pred_cls = pred_logits.argmax(-1)  # (B, Q), in [0, num_labels]
    for b in range(out.shape[0]):
        m = class_mask[b]
        if m is None:
            continue
        assert len(m) == num_labels, "class_mask length != num_labels"
        m = m.to(pred_logits.device)
        unmatched = target_classes[b] == num_labels
        pred = pred_cls[b]
        is_real = pred < num_labels                       # not already predicting no-object
        pred_unannotated = is_real & (m[pred.clamp(max=num_labels - 1)] == 0)
        out[b][unmatched & pred_unannotated] = -100
    return out


class MaskedMask2FormerLoss(Mask2FormerLoss):
    """Mask2FormerLoss that ignores the no-object classification penalty on
    unmatched queries predicting a class the sample's source did not annotate.

    Classification-loss-only; the Hungarian matcher and the mask/dice losses are
    unchanged. Reads a per-batch ``_class_mask`` (per-sample list of (num_labels,)
    0/1 tensors or None) stashed on the instance by the adapter before each forward.
    """

    def loss_labels(self, class_queries_logits, class_labels, indices):
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape
        idx = self._get_predictions_permutation_indices(indices)
        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )
        target_classes = torch.full(
            (batch_size, num_queries), fill_value=self.num_labels,
            dtype=torch.int64, device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o
        target_classes = apply_class_mask(
            target_classes, pred_logits, getattr(self, "_class_mask", None),
            self.num_labels,
        )
        criterion = nn.CrossEntropyLoss(weight=self.empty_weight, ignore_index=-100)
        loss_ce = criterion(pred_logits.transpose(1, 2), target_classes)
        return {"loss_cross_entropy": loss_ce}
