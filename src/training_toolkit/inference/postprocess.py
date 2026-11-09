"""Backend-specific postprocessing for ONNX model outputs.

All functions are pure numpy/cv2 — no torch dependency.
"""

import cv2
import numpy as np


def postprocess_rfdetr(
    raw_outputs: dict[str, np.ndarray],
    original_size: tuple[int, int],
    export_size: int,
    score_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Post-process RF-DETR ONNX outputs.

    Handles the raw ONNX export format where:
    - ``dets`` is (B, Q, 4) in normalized cxcywh format
    - ``labels`` is (B, Q, C) class logits (C = num_classes + 1, last is no-object)
    - ``masks`` is (B, Q, H/4, W/4) mask logits

    Args:
        raw_outputs: Dict with keys ``dets``, ``labels``,
            and optionally ``masks``.
        original_size: (H, W) of the original input image.
        export_size: Spatial resolution the model was exported at.
        score_threshold: Minimum detection score.

    Returns:
        Dict with ``boxes``, ``scores``, ``labels``, ``masks``.
    """
    pred_boxes = raw_outputs["dets"][0]     # (Q, 4) cxcywh normalized
    pred_logits = raw_outputs["labels"][0]  # (Q, C) class logits

    h_orig, w_orig = original_size

    # Sigmoid on class logits, drop no-object class (last column)
    probs = _sigmoid(pred_logits)[:, :-1]  # (Q, num_fg_classes)
    scores = probs.max(axis=-1)             # (Q,)
    labels = probs.argmax(axis=-1)          # (Q,)

    # Filter by score
    keep_idx = np.where(scores > score_threshold)[0]
    scores = scores[keep_idx]
    labels = labels[keep_idx]
    kept_boxes = pred_boxes[keep_idx]       # (K, 4) cxcywh

    # Convert cxcywh normalized → xyxy pixel coords
    cx, cy, bw, bh = kept_boxes[:, 0], kept_boxes[:, 1], kept_boxes[:, 2], kept_boxes[:, 3]
    x1 = (cx - bw / 2) * w_orig
    y1 = (cy - bh / 2) * h_orig
    x2 = (cx + bw / 2) * w_orig
    y2 = (cy + bh / 2) * h_orig
    boxes = np.stack([x1, y1, x2, y2], axis=-1)

    # Clamp to image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w_orig)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h_orig)

    # Masks
    masks_out: np.ndarray
    if "masks" in raw_outputs:
        all_masks = raw_outputs["masks"][0]  # (Q, H/4, W/4)
        kept_masks = all_masks[keep_idx]     # (K, H/4, W/4)
        masks_out = _resize_and_threshold_masks(
            kept_masks, original_size, threshold=0.0,
        )
    else:
        masks_out = np.zeros((len(keep_idx), h_orig, w_orig), dtype=bool)

    return {
        "boxes": boxes.astype(np.float32),
        "scores": scores.astype(np.float32),
        "labels": labels.astype(np.int64),
        "masks": masks_out,
    }


def postprocess_mask2former(
    raw_outputs: dict[str, np.ndarray],
    original_size: tuple[int, int],
    export_size: int,
    score_threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Post-process Mask2Former ONNX outputs.

    Args:
        raw_outputs: Dict with keys ``class_logits`` (B, Q, C+1) and
            ``mask_logits`` (B, Q, H/4, W/4).
        original_size: (H, W) of the original input image.
        export_size: Spatial resolution the model was exported at.
        score_threshold: Minimum class score.

    Returns:
        Dict with ``boxes``, ``scores``, ``labels``, ``masks``.
    """
    _ = export_size  # unused — Mask2Former derives boxes from masks, not coordinates
    class_logits = raw_outputs["class_logits"][0]  # (Q, C+1)
    mask_logits = raw_outputs["mask_logits"][0]    # (Q, H/4, W/4)

    # Softmax over classes, drop no-object (last column)
    probs = _softmax(class_logits, axis=-1)[:, :-1]  # (Q, C)
    max_scores = probs.max(axis=-1)                    # (Q,)
    pred_labels = probs.argmax(axis=-1)                # (Q,)

    keep = max_scores > score_threshold
    scores = max_scores[keep]
    labels = pred_labels[keep]
    kept_mask_logits = mask_logits[keep]  # (K, H/4, W/4)

    # Sigmoid + resize + binarize
    masks = _resize_and_threshold_masks(
        kept_mask_logits, original_size, threshold=0.5, apply_sigmoid=True,
    )

    # Compute boxes from binary masks
    boxes = _masks_to_boxes(masks)

    return {
        "boxes": boxes.astype(np.float32),
        "scores": scores.astype(np.float32),
        "labels": labels.astype(np.int64),
        "masks": masks,
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _resize_and_threshold_masks(
    masks: np.ndarray,
    target_size: tuple[int, int],
    threshold: float,
    apply_sigmoid: bool = False,
) -> np.ndarray:
    """Resize mask logits to target_size and binarize.

    Args:
        masks: (K, H, W) float32 logits.
        target_size: (H_target, W_target).
        threshold: Binarization threshold (applied after optional sigmoid).
        apply_sigmoid: Whether to sigmoid before thresholding.

    Returns:
        (K, H_target, W_target) bool array.
    """
    if len(masks) == 0:
        return np.zeros((0, *target_size), dtype=bool)

    h_target, w_target = target_size
    resized = np.stack([
        cv2.resize(m, (w_target, h_target), interpolation=cv2.INTER_LINEAR)
        for m in masks
    ])

    if apply_sigmoid:
        resized = _sigmoid(resized)

    return resized > threshold


def _masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """Compute xyxy bounding boxes from (N, H, W) binary masks."""
    if len(masks) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    n = masks.shape[0]
    boxes = np.zeros((n, 4), dtype=np.float32)

    for i in range(n):
        ys, xs = np.where(masks[i])
        if len(xs) == 0:
            continue
        boxes[i] = [xs.min(), ys.min(), xs.max(), ys.max()]

    return boxes
