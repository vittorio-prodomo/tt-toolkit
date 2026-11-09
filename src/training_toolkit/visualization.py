"""Visualization utilities for training diagnostics."""

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Shorter display names for class labels that are too long for small tiles.
_DISPLAY_NAMES: dict[str, str] = {
    "spalling": "spall",
    "exposed_rebar": "rebar",
}

# Distinct colors for up to 20 classes (BGR for cv2, will convert as needed)
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
]


def save_batch_grid(
    batch: dict,
    path: Path,
    class_names: dict[int, str] | None = None,
    max_images: int = 16,
) -> None:
    """Save a grid of annotated training samples as a PNG image.

    Parameters
    ----------
    batch : dict
        Collated batch with ``images`` (B, 3, H, W) float [0,1] and
        ``targets`` list of dicts with ``boxes``, ``labels``, ``masks``.
    path : Path
        Output file path (e.g. ``run_dir / "train_batch_sample.png"``).
    class_names : dict, optional
        Mapping from label index to class name string.
    max_images : int
        Maximum number of images to include in the grid.
    """
    images = batch["images"][:max_images]
    targets = batch["targets"][:max_images]
    n = len(images)

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    panels = []
    for i in range(n):
        img = _tensor_to_bgr(images[i])
        t = targets[i]
        boxes = t["boxes"].cpu().float().numpy()
        labels = t["labels"].cpu().numpy()
        masks = t["masks"].cpu().float().numpy() if "masks" in t else None

        img = _draw_annotations(img, boxes, labels, masks, class_names)
        panels.append(img)

    # Pad to fill the grid
    h, w = panels[0].shape[:2]
    while len(panels) < rows * cols:
        panels.append(np.zeros((h, w, 3), dtype=np.uint8))

    # Assemble grid
    grid_rows = []
    for r in range(rows):
        row_panels = panels[r * cols : (r + 1) * cols]
        grid_rows.append(np.concatenate(row_panels, axis=1))
    grid = np.concatenate(grid_rows, axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    """Convert (3, H, W) float [0,1] tensor to (H, W, 3) uint8 BGR."""
    img = (tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _draw_annotations(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    masks: np.ndarray | None,
    class_names: dict[int, str] | None,
    scores: np.ndarray | None = None,
    mask_alpha: float = 0.5,
    box_thickness: int = 1,
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw masks, boxes, and labels onto an image.

    Draw order is canonicalised so larger annotations are painted first
    and smaller ones last. For nested masks (a small mask contained in a
    larger one), the smallest mask lands on top and keeps its full
    blending intensity instead of being washed out by a subsequent larger
    overlay. The same order is used for boxes/labels so the smallest
    annotation's label sits on top as well.
    """
    overlay = img.copy()
    img_h, img_w = overlay.shape[:2]
    n = len(labels)

    if n == 0:
        return overlay

    # Determine draw order: largest area first, smallest last.
    # Prefer mask-based areas when masks are present; otherwise fall back
    # to bbox areas.
    if masks is not None and len(masks) > 0:
        resized_masks: list[np.ndarray] = []
        areas = np.empty(n, dtype=np.int64)
        for j, mask in enumerate(masks):
            if mask.shape[:2] != (img_h, img_w):
                mask = cv2.resize(
                    mask.astype(np.uint8), (img_w, img_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            bin_mask = mask.astype(bool)
            resized_masks.append(bin_mask)
            areas[j] = int(bin_mask.sum())
    else:
        resized_masks = []
        if len(boxes) > 0:
            bw = boxes[:, 2] - boxes[:, 0]
            bh = boxes[:, 3] - boxes[:, 1]
            areas = (bw * bh).astype(np.int64)
        else:
            areas = np.zeros(n, dtype=np.int64)

    draw_order = np.argsort(-areas, kind="stable")

    mask_alpha = float(np.clip(mask_alpha, 0.0, 1.0))
    img_alpha = 1.0 - mask_alpha

    # Draw masks with transparency (largest first, smallest last)
    if resized_masks:
        for j in draw_order:
            label = labels[int(j)]
            color = _PALETTE[int(label) % len(_PALETTE)]
            binary = resized_masks[int(j)]
            if not binary.any():
                continue
            overlay[binary] = (
                mask_alpha * np.array(color[::-1], dtype=np.float32)  # BGR
                + img_alpha * overlay[binary].astype(np.float32)
            ).astype(np.uint8)

    # Draw boxes and labels in the same order.
    # Track placed label rectangles so we can nudge later ones to avoid
    # overlap (greedy: labels for larger annotations get first pick).
    placed: list[tuple[int, int, int, int]] = []  # (x1, y1, x2, y2)

    for j in draw_order:
        j = int(j)
        box = boxes[j]
        label = labels[j]
        color_rgb = _PALETTE[int(label) % len(_PALETTE)]
        color_bgr = color_rgb[::-1]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, box_thickness)

        name = class_names.get(int(label), str(int(label))) if class_names else str(int(label))
        name = _DISPLAY_NAMES.get(name, name)
        if scores is not None:
            name = f"{name} {scores[j]:.2f}"

        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        label_w = tw + 4
        label_h = th + 6

        label_x = x1
        label_y_top = y1 - label_h  # default: above the bbox

        # If it would overflow the top, flip inside the bbox.
        if label_y_top < 0:
            label_y_top = y1

        # Clamp to image bounds.
        label_x = max(0, min(label_x, img_w - label_w))
        label_y_top = max(0, min(label_y_top, img_h - label_h))

        # Anti-overlap: if this label collides with an already-placed one,
        # push it downward in label_h steps until it fits or we run out of
        # image space.
        for _ in range(30):
            rect = (label_x, label_y_top, label_x + label_w, label_y_top + label_h)
            if not any(
                r[0] < rect[2] and rect[0] < r[2] and r[1] < rect[3] and rect[1] < r[3]
                for r in placed
            ):
                break
            label_y_top += label_h + 1
        # Re-clamp after shifting.
        label_y_top = max(0, min(label_y_top, img_h - label_h))

        placed.append((label_x, label_y_top, label_x + label_w, label_y_top + label_h))

        label_y_bot = label_y_top + label_h
        text_x = label_x + 2
        text_y = label_y_top + th + 2  # baseline ≈ 4 px above rect bottom

        cv2.rectangle(
            overlay, (label_x, label_y_top),
            (label_x + label_w, label_y_bot), color_bgr, -1,
        )
        # Black text on bright backgrounds, white on dark — decided by
        # perceived luminance of the palette color (ITU-R BT.601).
        lum = 0.299 * color_rgb[0] + 0.587 * color_rgb[1] + 0.114 * color_rgb[2]
        text_color = (0, 0, 0) if lum > 160 else (255, 255, 255)
        cv2.putText(
            overlay, name, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1,
        )

    return overlay


# ------------------------------------------------------------------
# Validation comparison grid (GT vs Prediction)
# ------------------------------------------------------------------


def save_val_comparison_grid(
    batch: dict,
    preds: list[dict],
    path: Path,
    class_names: dict[int, str] | None = None,
    max_images: int = 8,
    score_threshold: float = 0.3,
) -> None:
    """Save a grid comparing ground truth vs predictions side-by-side.

    Each row is one image: [GT | Prediction].
    """
    images = batch["images"][:max_images]
    targets = batch["targets"][:max_images]
    preds = preds[:max_images]
    n = len(images)

    panels = []
    for i in range(n):
        img_gt = _tensor_to_bgr(images[i])
        img_pred = img_gt.copy()

        # GT panel
        t = targets[i]
        gt_boxes = t["boxes"].cpu().float().numpy()
        gt_labels = t["labels"].cpu().numpy()
        gt_masks = (
            t["masks"].cpu().float().numpy()
            if "masks" in t and t["masks"].numel() > 0
            else None
        )
        img_gt = _draw_annotations(img_gt, gt_boxes, gt_labels, gt_masks, class_names)

        # Prediction panel
        p = preds[i]
        p_scores = p["scores"].cpu().float().numpy()
        keep = p_scores >= score_threshold
        pred_boxes = p["boxes"].cpu().float().numpy()[keep]
        pred_labels = p["labels"].cpu().numpy()[keep]
        pred_masks = None
        if "masks" in p and p["masks"].numel() > 0:
            pred_masks = p["masks"].cpu().float().numpy()[keep]
        pred_scores = p_scores[keep]
        img_pred = _draw_annotations(
            img_pred, pred_boxes, pred_labels, pred_masks, class_names,
            scores=pred_scores,
        )

        # Side-by-side with headers
        h, w = img_gt.shape[:2]
        header_h = 30
        gt_panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        gt_panel[header_h:] = img_gt
        cv2.putText(
            gt_panel, "Ground Truth", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        pred_panel = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        pred_panel[header_h:] = img_pred
        cv2.putText(
            pred_panel, "Prediction", (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
        )

        panels.append(np.concatenate([gt_panel, pred_panel], axis=1))

    grid = np.concatenate(panels, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid)


# ------------------------------------------------------------------
# W&B interactive mask-overlay payloads (GT + prediction layers)
# ------------------------------------------------------------------


def _tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0,1] tensor -> (H, W, 3) uint8 RGB (W&B expects RGB)."""
    return (tensor.cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _instance_masks_to_label_map(
    masks: np.ndarray | None,
    labels: np.ndarray,
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """Flatten per-instance binary masks into one HxW int label-map for W&B.

    Each instance's pixels are painted with ``int(label) + 1`` (0 stays
    background). Masks are resized to the image resolution with nearest-
    neighbour (mirrors ``_draw_annotations`` — handles ``val_mask_res``-
    downsampled prediction masks), and larger instances are painted first so
    smaller ones land on top, matching the grid renderer's draw order.
    """
    label_map = np.zeros((img_h, img_w), dtype=np.int32)
    if masks is None or len(masks) == 0:
        return label_map
    bins: list[np.ndarray] = []
    areas: list[int] = []
    for m in masks:
        if m.shape[:2] != (img_h, img_w):
            m = cv2.resize(
                m.astype(np.uint8), (img_w, img_h),
                interpolation=cv2.INTER_NEAREST,
            )
        b = m.astype(bool)
        bins.append(b)
        areas.append(int(b.sum()))
    for j in np.argsort(-np.array(areas), kind="stable"):  # largest first
        label_map[bins[int(j)]] = int(labels[int(j)]) + 1
    return label_map


def build_val_overlay_payload(
    image_tensor: torch.Tensor,
    target: dict,
    pred: dict,
    class_names: dict[int, str] | None = None,
    score_threshold: float = 0.3,
) -> tuple[np.ndarray, dict]:
    """Build interactive GT + prediction MASK-overlay data for one val image.

    Pure (no ``wandb`` import) so it is unit-testable; the diagnostics callback
    wraps the result in ``wandb.Image(rgb, masks=payload)``. Returns
    ``(rgb_uint8_HxWx3, masks_payload)`` where ``masks_payload`` carries the W&B
    ``prediction`` / ``ground_truth`` layers (each ``mask_data`` + shared
    ``class_labels``). Predictions are score-thresholded; class ids are offset
    by +1 so 0 stays background.
    """
    rgb = _tensor_to_rgb(image_tensor)
    img_h, img_w = rgb.shape[:2]

    gt_labels = target["labels"].cpu().numpy()
    gt_masks = (
        target["masks"].cpu().numpy()
        if "masks" in target and target["masks"].numel() > 0
        else None
    )

    scores = pred["scores"].cpu().float().numpy()
    keep = scores >= float(score_threshold)
    pred_labels = pred["labels"].cpu().numpy()[keep]
    pred_masks = (
        pred["masks"].cpu().numpy()[keep]
        if "masks" in pred and pred["masks"].numel() > 0
        else None
    )

    gt_lm = _instance_masks_to_label_map(gt_masks, gt_labels, img_h, img_w)
    pred_lm = _instance_masks_to_label_map(pred_masks, pred_labels, img_h, img_w)

    present = {int(x) for x in gt_labels.tolist()} | {int(x) for x in pred_labels.tolist()}
    class_labels = {0: "background"}
    for lab in sorted(present):
        name = (class_names or {}).get(lab, str(lab))
        class_labels[lab + 1] = _DISPLAY_NAMES.get(name, name)

    masks_payload = {
        "prediction": {"mask_data": pred_lm, "class_labels": class_labels},
        "ground_truth": {"mask_data": gt_lm, "class_labels": class_labels},
    }
    return rgb, masks_payload


# ------------------------------------------------------------------
# Dataset analysis plots
# ------------------------------------------------------------------


_dataset_stats_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _collect_dataset_stats(
    dataset,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect all bboxes and labels from a dataset (no transforms).

    Returns normalized xyxy boxes and labels arrays. Results are cached
    per dataset object so the expensive iteration only happens once.
    """
    cache_key = id(dataset)
    if cache_key in _dataset_stats_cache:
        return _dataset_stats_cache[cache_key]

    from tqdm import tqdm

    all_boxes: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for i in tqdm(range(len(dataset)), desc="Collecting label stats", leave=False):
        sample = dataset._load_sample(i)
        img_h, img_w = sample["image"].shape[:2]
        boxes = sample["boxes"].copy()
        if boxes.size > 0:
            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            all_boxes.append(boxes)
            all_labels.append(sample["labels"])
    if all_boxes:
        result = np.concatenate(all_boxes), np.concatenate(all_labels)
    else:
        result = np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)

    _dataset_stats_cache[cache_key] = result
    return result


def save_labels_summary(
    dataset,
    path: Path,
    class_names: dict[int, str] | None = None,
) -> None:
    """Save an Ultralytics-style labels.jpg with class and bbox distributions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boxes, labels = _collect_dataset_stats(dataset)
    if len(boxes) == 0:
        logger.warning("No annotations found — skipping labels.jpg")
        return

    # Convert xyxy -> cxcywh (all normalized)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), tight_layout=True)

    # (0,0) Class distribution
    ax = axes[0, 0]
    unique, counts = np.unique(labels, return_counts=True)
    if class_names:
        names = [class_names.get(int(u), str(int(u))) for u in unique]
    else:
        names = [str(int(u)) for u in unique]
    ax.barh(range(len(unique)), counts, color="steelblue")
    ax.set_yticks(range(len(unique)))
    ax.set_yticklabels(names)
    ax.set_xlabel("instances")
    ax.set_title("Class Distribution")
    ax.invert_yaxis()

    # (0,1) Box xy centers
    ax = axes[0, 1]
    ax.scatter(cx, cy, s=1, alpha=0.2, edgecolors="none", c="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # y inverted (image coords)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Bounding Box Centers")
    ax.set_aspect("equal")

    # (1,0) Box wh
    ax = axes[1, 0]
    ax.scatter(w, h, s=1, alpha=0.2, edgecolors="none", c="steelblue")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("width")
    ax.set_ylabel("height")
    ax.set_title("Bounding Box Sizes")
    ax.set_aspect("equal")

    # (1,1) Width and height histograms overlaid
    ax = axes[1, 1]
    ax.hist(w, bins=50, alpha=0.5, label="width", color="steelblue")
    ax.hist(h, bins=50, alpha=0.5, label="height", color="coral")
    ax.set_xlabel("normalized size")
    ax.set_ylabel("count")
    ax.set_title("Width / Height Distribution")
    ax.legend()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=200)
    plt.close(fig)


def save_labels_correlogram(
    dataset,
    path: Path,
) -> None:
    """Save a 4x4 correlogram of bbox x, y, w, h (like Ultralytics)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boxes, _ = _collect_dataset_stats(dataset)
    if len(boxes) == 0:
        logger.warning("No annotations found — skipping labels_correlogram.jpg")
        return

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    data = [cx, cy, w, h]
    names = ["x", "y", "width", "height"]

    fig, axes = plt.subplots(4, 4, figsize=(12, 12), tight_layout=True)

    for i in range(4):
        for j in range(4):
            ax = axes[i, j]
            if i == j:
                ax.hist(data[i], bins=50, color="steelblue", edgecolor="none")
            else:
                ax.scatter(data[j], data[i], s=1, alpha=0.1, edgecolors="none", c="steelblue")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

            if j == 0:
                ax.set_ylabel(names[i])
            else:
                ax.set_yticklabels([])
            if i == 3:
                ax.set_xlabel(names[j])
            else:
                ax.set_xticklabels([])

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------
# Training curves
# ------------------------------------------------------------------


def save_training_curves(csv_path: Path, output_path: Path) -> None:
    """Plot training loss and val mAP from the results CSV (dual y-axis)."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not Path(csv_path).exists():
        return  # no results.csv (e.g. validation disabled) -> nothing to plot

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return

    def _safe_float(val: str) -> float | None:
        try:
            return float(val) if val else None
        except ValueError:
            return None

    # Filter to rows that have at least an epoch and a loss value
    filtered = []
    for r in rows:
        epoch = _safe_float(r.get("epoch", ""))
        if epoch is None:
            continue
        loss = _safe_float(r.get("train/loss_total", ""))
        if loss is None:
            continue  # skip sanity-check rows with no train metrics
        filtered.append(r)

    if not filtered:
        return

    epochs = [int(float(r["epoch"])) for r in filtered]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Left axis: loss
    loss = [float(r["train/loss_total"]) for r in filtered]
    ax1.plot(epochs, loss, "b-", label="train/loss_total", linewidth=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="b")
    ax1.tick_params(axis="y", labelcolor="b")

    # Right axis: mAP
    ax2 = ax1.twinx()
    has_map = False
    if "val/map_50" in filtered[0]:
        vals = [_safe_float(r.get("val/map_50", "")) for r in filtered]
        if any(v is not None for v in vals):
            map50 = [v if v is not None else 0.0 for v in vals]
            ax2.plot(epochs, map50, "r-", label="val/mAP50", linewidth=1.5)
            has_map = True
    if "val/map" in filtered[0]:
        vals = [_safe_float(r.get("val/map", "")) for r in filtered]
        if any(v is not None for v in vals):
            mAP = [v if v is not None else 0.0 for v in vals]
            ax2.plot(epochs, mAP, "r--", label="val/mAP", linewidth=1.0, alpha=0.7)
            has_map = True
    if has_map:
        ax2.set_ylabel("mAP", color="r")
        ax2.tick_params(axis="y", labelcolor="r")
        ax2.set_ylim(0, 1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# PR / Confidence curve plots
# ------------------------------------------------------------------

# Palette for per-class curves (maximally distinct, colorblind-friendly)
_CURVE_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def save_pr_curve(
    precision: np.ndarray,
    scores: np.ndarray,
    rec_thresholds: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """Save Precision-Recall curve (per-class + mean).

    Legend shows mAP@0.5 (area under curve) for each class.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    R, K = precision.shape

    # Per-class curves
    for k in range(K):
        p = precision[:, k]
        auc = np.trapezoid(np.maximum(p, 0), rec_thresholds)
        color = _CURVE_COLORS[k % len(_CURVE_COLORS)]
        ax.plot(rec_thresholds, p, color=color, linewidth=1.5,
                label=f"{class_names[k]} {auc:.3f}")

    # Mean curve
    mean_p = np.mean(precision, axis=1)
    mean_auc = np.trapezoid(np.maximum(mean_p, 0), rec_thresholds)
    ax.plot(rec_thresholds, mean_p, color="navy", linewidth=2.5, linestyle="--",
            label=f"all classes {mean_auc:.3f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_p_curve(
    precision: np.ndarray,
    scores: np.ndarray,
    rec_thresholds: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """Save Precision vs Confidence curve (per-class + mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    R, K = precision.shape

    for k in range(K):
        conf = scores[:, k]
        p = precision[:, k]
        order = np.argsort(conf)
        color = _CURVE_COLORS[k % len(_CURVE_COLORS)]
        ax.plot(conf[order], p[order], color=color, linewidth=1.5,
                label=class_names[k])

    # Mean curve
    mean_p = np.mean(precision, axis=1)
    mean_conf = np.mean(scores, axis=1)
    order = np.argsort(mean_conf)
    ax.plot(mean_conf[order], mean_p[order], color="navy", linewidth=2.5,
            linestyle="--", label="all classes")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Confidence Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_r_curve(
    precision: np.ndarray,
    scores: np.ndarray,
    rec_thresholds: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """Save Recall vs Confidence curve (per-class + mean)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    R, K = precision.shape

    for k in range(K):
        conf = scores[:, k]
        order = np.argsort(conf)
        color = _CURVE_COLORS[k % len(_CURVE_COLORS)]
        ax.plot(conf[order], rec_thresholds[order], color=color, linewidth=1.5,
                label=class_names[k])

    # Mean curve
    mean_conf = np.mean(scores, axis=1)
    order = np.argsort(mean_conf)
    ax.plot(mean_conf[order], rec_thresholds[order], color="navy", linewidth=2.5,
            linestyle="--", label="all classes")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_title("Recall-Confidence Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def save_f1_curve(
    precision: np.ndarray,
    scores: np.ndarray,
    rec_thresholds: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    """Save F1 vs Confidence curve (per-class + mean).

    Legend shows best F1 and the confidence threshold that achieved it.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    R, K = precision.shape

    for k in range(K):
        p = precision[:, k]
        r = rec_thresholds
        denom = p + r
        f1 = np.where(denom > 0, 2 * p * r / denom, 0.0)
        conf = scores[:, k]
        order = np.argsort(conf)

        best_idx = np.argmax(f1)
        best_f1 = f1[best_idx]
        best_conf = conf[best_idx]

        color = _CURVE_COLORS[k % len(_CURVE_COLORS)]
        ax.plot(conf[order], f1[order], color=color, linewidth=1.5,
                label=f"{class_names[k]} {best_f1:.2f} @ {best_conf:.3f}")

    # Mean curve
    mean_p = np.mean(precision, axis=1)
    r = rec_thresholds
    denom = mean_p + r
    mean_f1 = np.where(denom > 0, 2 * mean_p * r / denom, 0.0)
    mean_conf = np.mean(scores, axis=1)
    order = np.argsort(mean_conf)

    best_idx = np.argmax(mean_f1)
    best_f1 = mean_f1[best_idx]
    best_conf = mean_conf[best_idx]

    ax.plot(mean_conf[order], mean_f1[order], color="navy", linewidth=2.5,
            linestyle="--",
            label=f"all classes {best_f1:.2f} @ {best_conf:.3f}")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1")
    ax.set_title("F1-Confidence Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------


def save_confusion_matrix(
    matrix: np.ndarray,
    class_names: list[str],
    path: Path,
    normalize: bool = True,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> None:
    """Save an instance segmentation confusion matrix as a heatmap.

    Parameters
    ----------
    matrix : ndarray of shape ``(num_classes + 1, num_classes + 1)``
        Rows = true class, columns = predicted class.  Last index is
        *background* (FP row / FN column).
    class_names : list[str]
        Class names (length ``num_classes``, without background).
    path : Path
        Output PNG path.
    normalize : bool
        If True, normalize each row to sum to 1 (shows confusion rates).
        Raw counts are still shown as annotations.
    conf_threshold, iou_threshold : float
        Displayed in the title for provenance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = class_names + ["background"]
    n = len(labels)

    # Normalized version for color scale
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True).astype(float)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        display = matrix.astype(float) / row_sums
    else:
        display = matrix.astype(float)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(7, n * 0.8)))
    im = ax.imshow(display, cmap="Blues", vmin=0)

    # Tick labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    # Annotate cells with raw counts
    thresh = display.max() / 2
    for i in range(n):
        for j in range(n):
            count = int(matrix[i, j])
            if count == 0:
                continue
            text = str(count)
            if normalize:
                text = f"{count}\n({display[i, j]:.0%})"
            color = "white" if display[i, j] > thresh else "black"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=7, color=color)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion Matrix  "
        f"(IoU\u2265{iou_threshold}, conf\u2265{conf_threshold})"
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label="fraction" if normalize else "count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
