"""Tile-grid computation and a grouped-by-image tile index for streaming."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Keep in sync with TilingConfig.gt_guided_* defaults (configs/tiling_config.py).
_DEFAULT_GT_GUIDED = {
    "target_in_model": 110.0, "min_crop": 416,
    "translation_jitter": 0.3, "scale_jitter": 0.15, "dedup_iou": 0.6,
}


def compute_tile_grid(
    img_w: int,
    img_h: int,
    tile_size: int,
    overlap: float,
) -> list[tuple[int, int, int, int]]:
    """Compute a grid of (x, y, w, h) tile boxes covering the image.

    Tiles are placed on a regular grid with the given overlap ratio.
    Edge tiles are clamped to fit within the image bounds.
    """
    step = max(1, int(tile_size * (1 - overlap)))
    tiles = []
    for y in range(0, img_h, step):
        for x in range(0, img_w, step):
            x1 = min(x, max(0, img_w - tile_size))
            y1 = min(y, max(0, img_h - tile_size))
            w = min(tile_size, img_w - x1)
            h = min(tile_size, img_h - y1)
            tiles.append((x1, y1, w, h))
    return list(dict.fromkeys(tiles))


def tile_has_annotation(
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    boxes: np.ndarray,
    min_overlap: float,
) -> bool:
    """Check whether any annotation overlaps the tile by at least *min_overlap*."""
    if len(boxes) == 0:
        return False
    tx1, ty1 = tile_x, tile_y
    tx2, ty2 = tile_x + tile_w, tile_y + tile_h
    ix1 = np.maximum(boxes[:, 0], tx1)
    iy1 = np.maximum(boxes[:, 1], ty1)
    ix2 = np.minimum(boxes[:, 2], tx2)
    iy2 = np.minimum(boxes[:, 3], ty2)
    inter_w = np.maximum(0, ix2 - ix1)
    inter_h = np.maximum(0, iy2 - iy1)
    inter_area = inter_w * inter_h
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    box_area = np.maximum(box_area, 1e-6)
    overlap_ratio = inter_area / box_area
    return bool(np.any(overlap_ratio >= min_overlap))


def mask_true_positive_windows(windows, instances) -> set[int]:
    """Indices (into *windows*) of windows holding >=1 rasterized pixel of any
    instance. windows: [((x, y, w, h), apply_band), ...]. Instance-major loop:
    a window already marked positive skips further rasterization.

    The coarse bbox pre-filter uses STRICT inequalities (a bbox edge exactly
    touching a tile edge does not short-circuit to "no overlap"): a bbox is
    derived from vertex extrema, and cv2.fillPoly rasterizes boundary pixels
    lying exactly on that extremum (e.g. a degenerate zero-height cap edge at
    bbox.y2 == tile.y) — a non-strict `<=`/`>=` pre-filter would reject that
    window before ever rasterizing it, silently dropping a real pixel-owning
    window. Strict comparisons only reject DEFINITELY-disjoint boxes, so the
    filter can never disagree with the rasterize step it's short-circuiting."""
    from training_toolkit.data.tile_materialize import rasterize_tile_local
    pos: set[int] = set()
    for inst in instances:
        bx1, by1, bx2, by2 = inst.bbox
        for wi, ((tx, ty, tw, th), _band) in enumerate(windows):
            if wi in pos:
                continue
            if bx2 < tx or bx1 > tx + tw or by2 < ty or by1 > ty + th:
                continue
            if rasterize_tile_local(inst, (tx, ty, tw, th)).any():
                pos.add(wi)
    return pos


def whole_frame_windows(
    img_w: int, img_h: int, boxes: np.ndarray,
) -> list[tuple[tuple[int, int, int, int], bool]]:
    """One whole-frame window, no band, positives-only (only if annotated)."""
    if len(boxes) == 0:
        return []
    return [((0, 0, img_w, img_h), False)]


def grid_windows(
    img_w: int, img_h: int, boxes: np.ndarray,
    tile_sizes: list[int], overlap: float,
    whole_image_if_giant: bool = False, giant_ceiling_frac: float = 0.75,
) -> list[tuple[tuple[int, int, int, int], bool]]:
    """Multi-scale tile grid; every window apply_band=True. Cross-scale exact-box
    dedup. When `whole_image_if_giant` and the frame holds a defect whose raw
    sqrt-area exceeds `giant_ceiling_frac * max(tile_sizes)`, append one extra
    downscaled full-frame window (giants land in-band there; small instances ignore)."""
    windows: list[tuple[tuple[int, int, int, int], bool]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for tile_size in tile_sizes:
        for (tx, ty, tw, th) in compute_tile_grid(img_w, img_h, tile_size, overlap):
            if (tx, ty, tw, th) in seen:
                continue
            seen.add((tx, ty, tw, th))
            windows.append(((tx, ty, tw, th), True))
    if whole_image_if_giant and len(boxes) > 0:
        giant_raw = giant_ceiling_frac * max(tile_sizes)
        wh = boxes[:, 2:4] - boxes[:, 0:2]                       # XYXY -> (w, h)
        sqrt_area = np.sqrt(np.clip(wh[:, 0] * wh[:, 1], 0, None))
        if float(sqrt_area.max()) > giant_raw and (0, 0, img_w, img_h) not in seen:
            windows.append(((0, 0, img_w, img_h), True))
    return windows


def _iou_xywh(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2, bx2, by2 = ax1 + aw, ay1 + ah, bx1 + bw, by1 + bh
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _dedup_windows_iou(
    windows: list[tuple[int, int, int, int]], iou_thresh: float,
) -> list[tuple[int, int, int, int]]:
    """Greedy first-wins dedup: drop a window if it overlaps a kept one by >= iou_thresh."""
    kept: list[tuple[int, int, int, int]] = []
    for w in windows:
        if all(_iou_xywh(w, k) < iou_thresh for k in kept):
            kept.append(w)
    return kept


def gt_guided_windows(
    img_w: int, img_h: int, boxes: np.ndarray, *,
    imgsz: int, target_in_model: float, min_crop: int,
    translation_jitter: float, scale_jitter: float, dedup_iou: float,
    rng: random.Random,
) -> list[tuple[tuple[int, int, int, int], bool]]:
    """SNIPER-style per-defect square chips. `boxes` is (N,4) XYXY. Each chip is
    sized so its centered defect lands near `target_in_model` in-model px, jittered
    and clamped to the image, then IoU-deduplicated. Every chip is (box, True)."""
    if imgsz is None:
        raise ValueError("gt_guided regime requires imgsz (got None)")
    raw: list[tuple[int, int, int, int]] = []
    for (x1, y1, x2, y2) in boxes:
        bw, bh = float(x2 - x1), float(y2 - y1)
        a = (max(bw * bh, 1.0)) ** 0.5                      # raw sqrt-area
        crop = a * imgsz / target_in_model
        crop *= 1.0 + rng.uniform(-scale_jitter, scale_jitter)
        crop = max(float(min_crop), crop)          # floor for small defects
        cs = min(int(round(crop)), img_w, img_h)   # cap at BOTH dims -> a square chip always fits
        cx = (x1 + x2) / 2.0 + rng.uniform(-translation_jitter, translation_jitter) * cs
        cy = (y1 + y2) / 2.0 + rng.uniform(-translation_jitter, translation_jitter) * cs
        tx = int(round(cx - cs / 2.0))
        ty = int(round(cy - cs / 2.0))
        tx = max(0, min(tx, max(0, img_w - cs)))
        ty = max(0, min(ty, max(0, img_h - cs)))
        tw = min(cs, img_w - tx)
        th = min(cs, img_h - ty)
        raw.append((tx, ty, tw, th))
    return [((x, y, w, h), True) for (x, y, w, h) in _dedup_windows_iou(raw, dedup_iou)]


def load_negative_blocklist(path) -> set[tuple]:
    """JSON [[image_path, x, y, w, h], ...] -> set of tuples."""
    rows = json.loads(Path(path).read_text())
    return {(str(p), int(x), int(y), int(w), int(h)) for p, x, y, w, h in rows}


def build_tile_index(
    base, tile_sizes, overlap: float, negative_ratio: float,
    min_annotation_overlap: float, seed: int, indoor_regime: str = "indoor_whole",
    regime_tile_sizes: dict[str, list[int]] | None = None,
    imgsz: int | None = None, gt_guided_params: dict | None = None,
    giant_ceiling_frac: float = 0.75,
    whole_image_if_giant: bool = True,
    dims_out: dict | None = None,
    pools_out: dict | None = None,
    assignment_mode: str = "bbox_ioa",
    negative_blocklist: set | None = None,
) -> dict[int, list[tuple[tuple[int, int, int, int], bool]]]:
    """Regime-aware, multi-scale tile index → {img_idx: [(box, apply_band), ...]}.

    Whole-frame regimes (``indoor_whole``, ``whole_frame``, and the ``indoor_regime``
    alias) get ONE whole-frame window ``((0,0,W,H), False)``: forced when the image
    has annotations; routed into the ratio-subsampled negatives pool when it has
    none (a confirmed-empty negative image). Grid regimes (``tiled_multiscale``, ``stain_1024``,
    ``outdoor_tiled``, or any unknown string) get a positive/negative grid at EACH
    size in ``tile_sizes``, every tile ``(box, True)``. Negatives (grid regimes
    only) are globally subsampled to ``negative_ratio``. Skips images whose file is
    missing on disk.

    Parameters
    ----------
    base :
        Dataset with ``_image_path``, ``_load_tile_metadata``, ``_image_regime``.
    tile_sizes : list[int]
        One or more tile sizes for grid-regime images (e.g. ``[416, 1024]``).
    overlap : float
        Fractional tile overlap in [0, 1).
    negative_ratio : float
        Target fraction of the final tile set that are negatives (grid regimes only).
    min_annotation_overlap : float
        Minimum annotation-to-tile IoA to count as a positive.
    seed : int
        RNG seed for negative subsampling.
    indoor_regime : str
        Back-compat alias: the per-image regime string treated as whole-frame
        (default ``"indoor_whole"``). ``"whole_frame"`` is always whole-frame.

    pools_out : dict | None
        When a dict is passed, it is filled with the pre-subsample (PRE-blocklist)
        ``positives`` / ``negatives`` / ``forced`` window lists
        (``[(img_idx, (box, apply_band)), ...]``). Return value unchanged.
        Blocklist filters are applied after ``pools_out`` is populated, so
        out-of-band consumers (e.g. FP mining) see the raw full negative pool.
    assignment_mode : str
        Grid-regime window classification. ``"bbox_ioa"`` (default): a window is
        positive iff an annotation's bbox overlaps it by >= ``min_annotation_overlap``
        (byte-identical to pre-dataset-v2 behavior). ``"mask_true"``: a window is
        positive iff it holds >=1 rasterized pixel of any annotation (pixel-true;
        ``min_annotation_overlap`` is inert) — see :func:`mask_true_positive_windows`.
    negative_blocklist : set | None
        Set of ``(image_path, x, y, w, h)`` tuples (window descriptors) to exclude
        from the negative pool (default None; no filtering). Applied after ``pools_out``
        is populated.

    Notes
    -----
    The ``gt_guided`` regime requires ``imgsz`` (the model input size) to size its
    per-defect SNIPER chips to ``target_in_model`` pixels in-model.
    """
    positives: list[tuple[int, tuple]] = []
    negatives: list[tuple[int, tuple]] = []
    forced: list[tuple[int, tuple]] = []   # whole_frame + gt_guided: always kept, never subsampled
    path_by_idx: dict[int, str] = {}
    skipped_missing = 0
    whole_frame_regimes = {indoor_regime, "whole_frame"}
    grid_giant_regimes = {"tiled_multiscale", "stain_1024"}
    for img_idx in range(len(base)):
        # Skip images whose file is not on disk (e.g. a combined / DVC-partial
        # dataset where some sources aren't materialized) — train on what's
        # present instead of crashing on the first missing file.
        try:
            img_path = base._image_path(img_idx)
        except NotImplementedError:
            img_path = None
        if img_path is not None and not os.path.exists(img_path):
            skipped_missing += 1
            continue
        if img_path is not None:
            path_by_idx[img_idx] = img_path
        img_w, img_h, boxes = base._load_tile_metadata(img_idx)
        if dims_out is not None:
            dims_out[img_idx] = (img_w, img_h)
        regime = base._image_regime(img_idx)
        if regime in whole_frame_regimes:
            if len(boxes) == 0:
                # Confirmed-empty whole-frame image (e.g. a nullified negative from
                # a 1-class crosswalk): one full-frame window into the NEGATIVES
                # pool — ratio-subsampled like grid negatives, never forced.
                negatives.append((img_idx, ((0, 0, img_w, img_h), False)))
            else:
                for window in whole_frame_windows(img_w, img_h, boxes):
                    forced.append((img_idx, window))
            continue
        if regime == "gt_guided":
            grng = random.Random(seed * 1_000_003 + img_idx)   # per-image, order-independent
            for window in gt_guided_windows(
                img_w, img_h, boxes, imgsz=imgsz, rng=grng,
                **(gt_guided_params or _DEFAULT_GT_GUIDED),
            ):
                forced.append((img_idx, window))
            continue
        # grid regimes (tiled_multiscale / stain_1024 / outdoor_tiled / unknown)
        ts = (regime_tile_sizes or {}).get(regime, tile_sizes)
        wig = whole_image_if_giant and regime in grid_giant_regimes
        ws = grid_windows(
            img_w, img_h, boxes, ts, overlap,
            whole_image_if_giant=wig, giant_ceiling_frac=giant_ceiling_frac,
        )
        if assignment_mode == "mask_true":
            insts = base._load_annotations(img_idx)
            pos_wi = mask_true_positive_windows(ws, insts)
            for wi, (box, apply_band) in enumerate(ws):
                bucket = positives if wi in pos_wi else negatives
                bucket.append((img_idx, (box, apply_band)))
        else:
            for (box, apply_band) in ws:
                if tile_has_annotation(box[0], box[1], box[2], box[3], boxes,
                                       min_annotation_overlap):
                    positives.append((img_idx, (box, apply_band)))
                else:
                    negatives.append((img_idx, (box, apply_band)))
    if skipped_missing:
        logger.warning(
            "build_tile_index: skipped %d image(s) whose file is missing on disk",
            skipped_missing,
        )
    if pools_out is not None:
        # Expose the exact pre-subsample pools (construction order, deterministic)
        # for out-of-band consumers (e.g. FP mining over the FULL negative pool).
        pools_out["positives"] = list(positives)
        pools_out["negatives"] = list(negatives)
        pools_out["forced"] = list(forced)
    if negative_blocklist:
        n_before = len(negatives)
        negatives = [
            (i, (box, b)) for (i, (box, b)) in negatives
            if (path_by_idx.get(i), *[int(v) for v in box])
            not in negative_blocklist
        ]
        logger.info("negative blocklist: %d -> %d candidate negatives",
                    n_before, len(negatives))
    n_pos = len(positives)
    n_neg = 0 if (negative_ratio <= 0 or n_pos == 0) else int((negative_ratio / (1 - negative_ratio)) * n_pos)
    n_neg = min(n_neg, len(negatives))
    rng = random.Random(seed)
    kept_neg = rng.sample(negatives, n_neg) if n_neg > 0 else []
    grouped: dict[int, list] = {}
    for img_idx, window in positives + kept_neg + forced:
        grouped.setdefault(img_idx, []).append(window)
    return grouped
