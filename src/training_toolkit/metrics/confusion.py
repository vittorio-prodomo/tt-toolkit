"""Confusion matrix for instance segmentation via IoU-based matching."""

import numpy as np
import torch
from torchvision.ops import box_iou


def update_confusion_matrix(
    matrix: np.ndarray,
    preds: list[dict],
    targets: list[dict],
    iou_threshold: float = 0.5,
    conf_threshold: float = 0.25,
) -> None:
    """Update a confusion matrix in-place for one batch of images.

    Uses bbox IoU to match predictions to ground truths.  Greedy matching
    (predictions sorted by confidence descending) mirrors COCO eval logic.

    Parameters
    ----------
    matrix : ndarray of shape ``(num_classes + 1, num_classes + 1)``
        Rows = true class, columns = predicted class.  The last index
        represents *background* (unmatched GT → last column, unmatched
        prediction → last row).
    preds, targets : list[dict]
        Per-image dicts with ``boxes``, ``labels``, ``scores`` (preds only).
    iou_threshold : float
        Minimum IoU for a valid match.
    conf_threshold : float
        Predictions below this confidence are ignored.
    """
    bg = matrix.shape[0] - 1  # background index

    for pred, target in zip(preds, targets):
        pred_boxes = pred["boxes"]
        pred_labels = pred["labels"]
        pred_scores = pred["scores"]
        gt_boxes = target["boxes"]
        gt_labels = target["labels"]

        # Filter low-confidence predictions
        keep = pred_scores >= conf_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            continue

        # All predictions are FP if no GT
        if len(gt_boxes) == 0:
            for pl in pred_labels:
                matrix[bg, int(pl.item())] += 1
            continue

        # All GTs are FN if no predictions
        if len(pred_boxes) == 0:
            for gl in gt_labels:
                matrix[int(gl.item()), bg] += 1
            continue

        # Compute pairwise bbox IoU: [num_pred, num_gt]
        ious = box_iou(pred_boxes, gt_boxes)

        # Sort predictions by confidence (descending) for greedy matching
        order = torch.argsort(pred_scores, descending=True)
        gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool)

        for idx in order:
            iou_row = ious[idx]
            # Mask already-matched GTs
            iou_row = iou_row.clone()
            iou_row[gt_matched] = 0

            best_gt = iou_row.argmax()
            if iou_row[best_gt] >= iou_threshold:
                gt_matched[best_gt] = True
                gt_cls = int(gt_labels[best_gt].item())
                pred_cls = int(pred_labels[idx].item())
                matrix[gt_cls, pred_cls] += 1
            else:
                # Unmatched prediction → FP
                matrix[bg, int(pred_labels[idx].item())] += 1

        # Unmatched GTs → FN
        for j in range(len(gt_labels)):
            if not gt_matched[j]:
                matrix[int(gt_labels[j].item()), bg] += 1
