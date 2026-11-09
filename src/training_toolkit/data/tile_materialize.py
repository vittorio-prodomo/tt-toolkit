"""Tile materialization: pyvips ROI decode + tile-local mask rasterization.

Both choices were picked in a profiling study (ROI decode was 60-100x faster
than decoding the full frame). Kept dependency-light (numpy + cv2 + pyvips) so
DataLoader spawn-workers can import it cleanly.
"""
from __future__ import annotations

import os

# libvips must stay single-threaded per process/worker — parallelism comes from
# DataLoader workers, not libvips threads (N_workers × libvips_threads would
# thrash the CPU). pyvips reads VIPS_CONCURRENCY at libvips init and exposes no
# public concurrency_set() in 3.x, so set the env var here, before the lazy
# `import pyvips` in decode_crop_pyvips. Covers every path that imports this
# module first: spawn workers, nw=0, and the bench harness.
os.environ.setdefault("VIPS_CONCURRENCY", "1")

from dataclasses import dataclass

import cv2
import numpy as np

Box = tuple[int, int, int, int]  # x, y, w, h


# --------------------------------------------------------------------------
# Decode (pyvips ROI)
# --------------------------------------------------------------------------

def _clamp_box(box: Box, W: int, H: int) -> tuple[int, int, int, int]:
    """Clamp a requested (x,y,w,h) box to image bounds → in-bounds (cx,cy,cw,ch)."""
    x, y, w, h = box
    cx, cy = max(0, x), max(0, y)
    cx2, cy2 = min(x + w, W), min(y + h, H)
    return cx, cy, max(0, cx2 - cx), max(0, cy2 - cy)


def _place(region: np.ndarray, box: Box, cx: int, cy: int) -> np.ndarray:
    """Place the in-bounds *region* (decoded for clamped origin cx,cy) into a
    zero-padded (h, w, 3) canvas for the full requested *box*."""
    x, y, w, h = box
    out = np.zeros((h, w, 3), dtype=np.uint8)
    if region.size:
        rh, rw = region.shape[:2]
        out[cy - y:cy - y + rh, cx - x:cx - x + rw] = region[:, :, :3]
    return out


def decode_crop_pyvips(path: str, box: Box) -> np.ndarray:
    """ROI / random-access region read — decodes only the (clamped) crop region."""
    import pyvips
    image = pyvips.Image.new_from_file(path, access="random")
    W, H = image.width, image.height
    cx, cy, cw, ch = _clamp_box(box, W, H)
    x, y, w, h = box
    if cw == 0 or ch == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    arr = image.crop(cx, cy, cw, ch).numpy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return _place(np.ascontiguousarray(arr[:, :, :3]), box, cx, cy)


def decode_image_pyvips(path: str) -> np.ndarray:
    """Full-frame decode (sequential access) → (H, W, 3) uint8 RGB.

    No EXIF auto-rotate (matches decode_crop_pyvips), so a pooled image's pixels
    agree with the ROI decoder used by the buffered path."""
    import pyvips
    image = pyvips.Image.new_from_file(path, access="sequential")
    arr = image.numpy()
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return np.ascontiguousarray(arr[:, :, :3])


def crop_from_array(array: np.ndarray, box: Box) -> np.ndarray:
    """Numpy-crop a tile (x,y,w,h) out of a resident full-image HWC array; clamp
    to bounds + zero-pad to (h, w, 3). The array twin of decode_crop_pyvips."""
    H, W = array.shape[:2]
    cx, cy, cw, ch = _clamp_box(box, W, H)
    x, y, w, h = box
    if cw == 0 or ch == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    region = array[cy:cy + ch, cx:cx + cw]
    if region.ndim == 2:
        region = np.stack([region] * 3, axis=-1)
    return _place(np.ascontiguousarray(region[:, :, :3]), box, cx, cy)


# --------------------------------------------------------------------------
# Annotations + tile-local rasterization
# --------------------------------------------------------------------------

@dataclass
class Instance:
    label: int
    polygon: list[np.ndarray] | None  # list of (P, 2) float32 rings (abs px), or None
    mask: np.ndarray | None             # (mh, mw) uint8, for mask-native, or None
    mask_origin: tuple[int, int] | None  # (x0, y0) top-left of mask in image px
    bbox: tuple[float, float, float, float]  # xyxy absolute px
    iscrowd: int = 0  # COCO iscrowd flag (0/1); default keeps every existing
                      # positional Instance(...) call valid. Threaded through to
                      # metric targets so torchmetrics MAP ignores crowd/sliver GT.


def assign_instances_to_tile(instances: list[Instance], tile: Box,
                             min_overlap: float = 0.05,
                             assignment_mode: str = "bbox_ioa") -> list[int]:
    """Indices of instances belonging to the tile.

    "bbox_ioa" (default): bbox-intersection / bbox-area >= min_overlap.
    "mask_true": bbox not STRICTLY disjoint from the tile AND >=1 rasterized
    pixel in it (min_overlap is inert) — every in-window pixel is owned by an
    assignment. The disjointness test is boundary-inclusive, mirroring
    mask_true_positive_windows (tile_index.py): a bbox edge exactly touching
    the tile edge still rasterizes real boundary pixels (cv2 draws the
    extremum cap edge), so gating on intersection AREA > 0 would de-sync
    instance selection from window classification — a window classified
    positive off a boundary pixel would materialize with zero instances.
    """
    tx, ty, tw, th = tile
    tx2, ty2 = tx + tw, ty + th
    keep: list[int] = []
    for i, inst in enumerate(instances):
        bx1, by1, bx2, by2 = inst.bbox
        if assignment_mode == "mask_true":
            disjoint = bx2 < tx or bx1 > tx2 or by2 < ty or by1 > ty2
            if not disjoint and rasterize_tile_local(inst, tile).any():
                keep.append(i)
            continue
        ix1, iy1 = max(bx1, tx), max(by1, ty)
        ix2, iy2 = min(bx2, tx2), min(by2, ty2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area = max((bx2 - bx1) * (by2 - by1), 1e-6)
        if inter / area >= min_overlap:
            keep.append(i)
    return keep


def _poly_fill(canvas: np.ndarray, rings: list[np.ndarray]) -> None:
    cv2.fillPoly(canvas, [r.round().astype(np.int32) for r in rings], color=1)


def rasterize_full_then_crop(inst: Instance, tile: Box,
                             img_hw: tuple[int, int]) -> np.ndarray:
    """Regime 1: rasterize at full image resolution, then crop the tile.
    Used as a test oracle for rasterize_tile_local."""
    tx, ty, tw, th = tile
    H, W = img_hw
    full = np.zeros((H, W), dtype=np.uint8)
    if inst.polygon is not None:
        _poly_fill(full, inst.polygon)
    else:
        x0, y0 = inst.mask_origin
        mh, mw = inst.mask.shape
        y1, x1 = max(0, y0), max(0, x0)
        y2, x2 = min(H, y0 + mh), min(W, x0 + mw)
        if y2 > y1 and x2 > x1:
            full[y1:y2, x1:x2] = (inst.mask[y1 - y0:y2 - y0, x1 - x0:x2 - x0] > 0).astype(np.uint8)
    return np.ascontiguousarray(full[ty:ty + th, tx:tx + tw])


def rasterize_tile_local(inst: Instance, tile: Box) -> np.ndarray:
    """Regime 2: rasterize directly into a tile-sized canvas (no full-res buffer)."""
    tx, ty, tw, th = tile
    canvas = np.zeros((th, tw), dtype=np.uint8)
    if inst.polygon is not None:
        off = np.array([tx, ty], dtype=np.float32)
        _poly_fill(canvas, [r - off for r in inst.polygon])
    else:
        x0, y0 = inst.mask_origin
        mh, mw = inst.mask.shape
        sx1, sy1 = max(x0, tx), max(y0, ty)
        sx2, sy2 = min(x0 + mw, tx + tw), min(y0 + mh, ty + th)
        if sx2 > sx1 and sy2 > sy1:
            sub = inst.mask[sy1 - y0:sy2 - y0, sx1 - x0:sx2 - x0]
            canvas[sy1 - ty:sy2 - ty, sx1 - tx:sx2 - tx] = (sub > 0).astype(np.uint8)
    return canvas


def materialize_tile(image_path: str, tile_box: Box, instances: list[Instance],
                     min_overlap: float = 0.05, image_id: int = 0,
                     assignment_mode: str = "bbox_ioa") -> dict:
    """Build one collate-ready tile sample.

    pyvips ROI crop for pixels + tile-local rasterization for the instances
    selected by :func:`assign_instances_to_tile` (bbox-IoA >= min_overlap by
    default; pixel-true under ``assignment_mode="mask_true"``). Returns the
    dict shape that ``data.collate.collate_fn`` expects (image HWC uint8 RGB,
    masks list of HxW uint8, boxes (N,4) float32 xyxy tile-local, labels (N,)
    int64).
    """
    tile_img = decode_crop_pyvips(image_path, tile_box)  # (th, tw, 3) uint8, zero-padded
    keep = assign_instances_to_tile(instances, tile_box, min_overlap, assignment_mode)

    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    labels: list[int] = []
    iscrowds: list[int] = []
    for i in keep:
        m = rasterize_tile_local(instances[i], tile_box)  # (th, tw) uint8
        if m.sum() == 0:
            continue
        ys, xs = np.where(m > 0)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        masks.append(m)
        labels.append(instances[i].label)
        iscrowds.append(instances[i].iscrowd)

    return {
        "image": tile_img,
        "masks": masks,
        "boxes": (np.array(boxes, dtype=np.float32) if boxes
                  else np.zeros((0, 4), dtype=np.float32)),
        "labels": np.array(labels, dtype=np.int64),
        "iscrowd": np.array(iscrowds, dtype=np.int64),
        "image_id": image_id,
    }
