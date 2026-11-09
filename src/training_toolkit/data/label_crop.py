"""Size-banded crop labeling (SNIP per-region ignore) + masked-loss tags.

Pure labeling for one crop window: given the instances overlapping a crop and
the crop's realized scale, mark each instance POSITIVE (supervise) or IGNORE
(out-of-band: sub-resolution or boundaryless), attach the per-source class_mask,
and tag whether the crop has zero positive supervision (a negative). Reuses
tile_materialize's bbox-overlap assignment + tile-local rasterization. Decode-
free and numpy-only so it stays import-clean for spawn workers and trivially
unit-testable.

"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from training_toolkit.data.tile_materialize import (
    assign_instances_to_tile,
    rasterize_tile_local,
)

DEFAULT_FLOOR_IN_MODEL = 24.0
DEFAULT_CEILING_FRAC = 0.75


@dataclass(frozen=True)
class SizeBand:
    """Valid per-instance size window in IN-MODEL sqrt-area pixels.

    Both bounds are tile-size-independent once expressed in-model: the floor is
    a fixed detectability threshold; the ceiling is a fraction of the model
    input side. The spec's per-tile RAW bands (416->[20,312], 1024->[49,768])
    are this same window scaled by each tile's realized_scale.
    """

    floor: float = DEFAULT_FLOOR_IN_MODEL
    ceiling: float = DEFAULT_CEILING_FRAC * 504

    @classmethod
    def for_imgsz(cls, imgsz: int, floor: float = DEFAULT_FLOOR_IN_MODEL,
                  ceiling_frac: float | None = DEFAULT_CEILING_FRAC) -> "SizeBand":
        if ceiling_frac is None:
            return cls(floor=floor, ceiling=float("inf"))   # floor-only band
        return cls(floor=floor, ceiling=ceiling_frac * imgsz)


def band_status(in_model_sqrt_area: float, band: SizeBand | None) -> str:
    """'positive' if in-band (or band is None), else 'ignore'."""
    if band is None:
        return "positive"
    if in_model_sqrt_area < band.floor or in_model_sqrt_area > band.ceiling:
        return "ignore"
    return "positive"


def label_crop(tile_box, instances, realized_scale, class_mask,
               band=None, min_overlap=0.05, assignment_mode="bbox_ioa"):
    """Label one crop window with size-band ignore flags + masked-loss tags.

    Args:
      tile_box: (x, y, w, h) crop window in source pixels.
      instances: list[Instance] (abs px) for the source image.
      realized_scale: imgsz / effective_crop_side -- converts a tile-local
        sqrt-area to in-model px. Indoor whole-frame: imgsz / max(img_w, img_h).
      class_mask: per-source binary list[num_classes] (trigger b), passed through.
      band: SizeBand for outdoor tiles, or None for the indoor whole-frame
        regime (supervise every instance; no size ignore).
      min_overlap: bbox-overlap fraction to assign an instance to the crop
        (assignment_mode="bbox_ioa" only).
      assignment_mode: "bbox_ioa" (default) or "mask_true" -- see
        :func:`training_toolkit.data.tile_materialize.assign_instances_to_tile`.

    Returns dict:
      masks    : list[(th,tw) uint8] tile-local masks (positives AND ignores)
      boxes    : (N,4) float32 xyxy tile-local
      labels   : (N,) int64
      ignore   : (N,) bool -- True = out-of-band (drop from matching; ignore-region)
      class_mask : the passed-through mask
      negative : bool -- True iff there are zero POSITIVE instances
    """
    keep = assign_instances_to_tile(instances, tile_box, min_overlap, assignment_mode)
    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    labels: list[int] = []
    ignore: list[bool] = []
    for i in keep:
        m = rasterize_tile_local(instances[i], tile_box)
        area = int(m.sum())
        if area == 0:
            continue  # clipped to nothing
        in_model = (area ** 0.5) * realized_scale
        ys, xs = np.where(m > 0)
        boxes.append([float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())])
        masks.append(m)
        labels.append(instances[i].label)
        ignore.append(band_status(in_model, band) == "ignore")
    ignore_arr = np.array(ignore, dtype=bool)
    return {
        "masks": masks,
        "boxes": (np.array(boxes, dtype=np.float32) if boxes
                  else np.zeros((0, 4), dtype=np.float32)),
        "labels": np.array(labels, dtype=np.int64),
        "ignore": ignore_arr,
        "class_mask": class_mask,
        "negative": bool(int((~ignore_arr).sum()) == 0),
    }
