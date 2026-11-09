"""The unified val-mask-resolution knob (``val_mask_downsample``).

Per-epoch validation is dominated by the CPU RLE-encode + device->host copy of
predicted masks inside ``SegmentationMAP.update``; both shrink with mask AREA,
so scoring on smaller masks is the cheapest large win available at val time
(measured ~3x on RF-DETR, 4.8x in the standalone per-source eval). Every
backend already had a way to do this -- but three different ways:

===============  ====================================  ======================
backend          old knob                              old default
===============  ====================================  ======================
RF-DETR          ``val_mask_res: int|None`` (abs px)    ``None`` = full res
Mask2Former      ``val_mask_res: int|None`` (abs px)    ``None`` = full res
YOLO             ``val_full_res_masks: bool``           ``False`` = input/4
===============  ====================================  ======================

Two of them absolute, one relative; two defaulting to slow, one to fast; and
the relative one quantised to a boolean. This module replaces all three with
one relative knob.

WHY RELATIVE, NOT ABSOLUTE PIXELS
---------------------------------
An absolute pixel target silently changes meaning when the input resolution
changes. A concrete example: two arms of one experiment run at 504
and 1008. ``val_mask_res: 192`` is a 2.6x downsample on one arm and a 5.25x
downsample on the other -- the same config line, two different measurement
fidelities, and two ``val/map`` curves that are not on a common footing. That
is exactly the class of silently-incomparable number this codebase keeps
getting bitten by.

It also mis-states the evidence. The ``192`` default was inherited from
an earlier per-source evaluation script's fast mask mode, where it measured a 4.8x
speedup at Spearman rho=1.0 on source/phenotype ranking -- on 504px tiles,
i.e. at a downsample RATIO of ~2.6x. The validated quantity was the ratio; the
pixel count was an artefact of the resolution it was measured at. Pinning the
ratio makes that evidence transfer to any input size.

Finally, absolute-square targets quietly assume a square canvas (the old
helpers resized to ``(res, res)``). True today because the val transform
letterboxes, but an unstated coupling. A divisor preserves aspect ratio by
construction.

WHY THE DEFAULT IS ``"native"``
-------------------------------
A segmentation head emits masks at its own stride -- YOLO-seg protos at
input/4, RF-DETR's seg head at ``imgsz // mask_downsample_ratio``, Mask2Former
at its pixel-decoder ``common_stride``. Then the postprocess UPSAMPLES them to
the full input canvas before the metric sees them. Scoring at native therefore
does not throw information away: it declines to score at a resolution the
model never actually produced. That is a principled default, where ``192`` was
an empirical one.

It is NOT byte-identical to the old full-res default for RF-DETR and
Mask2Former -- interpolating a mask up and scoring it against full-res GT is
not the same computation as scoring both at the head's resolution. It IS the
existing default for YOLO, whose ``val_full_res_masks=False`` already means
exactly this. See ``resolve_val_mask_resolution`` for the compatibility rules.

CAVEAT WORTH KNOWING (thin classes)
-----------------------------------
Coarser val masks hurt THIN structures first: at input/4 a 3px-wide crack in a
504px canvas is sub-pixel, so per-class ``val/map`` for hairline classes reads
lower than a full-res eval would. This is a metric-fidelity effect, not a model
change, and it is why the reported headline number always comes from a separate
full-res eval -- the per-epoch metric exists for best-checkpoint tracking and
early stopping (rho=1.0 keeps ranking/selection safe). If a run's per-class
thin-class trajectory is the object of study, set ``val_mask_downsample: 1``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Config keys this module owns. Used by the adapters' extra-stripping so none
#: of them reach a backend constructor.
VAL_MASK_KEYS = ("val_mask_downsample", "val_mask_res", "val_full_res_masks")

NATIVE = "native"


@dataclass(frozen=True)
class ValMaskResolution:
    """Resolved val-mask-resolution policy.

    ``mode`` is one of:

    * ``"native"``  -- leave masks at the head's own output resolution
      (``value`` is ``None``). The adapter skips its upsample entirely.
    * ``"divisor"`` -- mask side = input side / ``value``. ``value == 1`` means
      full input resolution.
    * ``"absolute"`` -- mask side = ``value`` px, square. Deprecated; reachable
      only via the legacy ``val_mask_res`` key.
    """

    mode: str
    value: float | int | None = None

    @property
    def is_native(self) -> bool:
        return self.mode == NATIVE

    def describe(self) -> str:
        if self.mode == NATIVE:
            return "native (head's own mask stride)"
        if self.mode == "divisor":
            return "full input resolution" if self.value == 1 else f"input / {self.value:g}"
        return f"{self.value}x{self.value} px (absolute, deprecated)"


def resolve_val_mask_resolution(extra: dict | None) -> ValMaskResolution:
    """Parse the val-mask-resolution policy out of ``ModelConfig.extra``.

    Accepts the new ``val_mask_downsample`` and both legacy spellings, but at
    most ONE of them: a config setting two is ambiguous about which wins, and
    silently picking one is how a run ends up measured at a resolution nobody
    intended. Raises ``ValueError`` naming the conflict instead.

    Legacy mappings (each warns once, both remain honoured):

    * ``val_mask_res: N``            -> absolute NxN
    * ``val_mask_res: 0`` / ``None`` -> divisor 1 (full res; the documented
      opt-out spelling)
    * ``val_full_res_masks: true``   -> divisor 1 (full res)
    * ``val_full_res_masks: false``  -> native (what YOLO already did)

    Unset -> ``native``.
    """
    extra = extra or {}
    present = [k for k in VAL_MASK_KEYS if k in extra]

    if len(present) > 1:
        raise ValueError(
            f"Conflicting val-mask-resolution knobs in model.extra: "
            f"{', '.join(present)}. These control the same thing and there is "
            f"no defined precedence between them -- set exactly one. Prefer "
            f"'val_mask_downsample' (relative, resolution-independent); "
            f"'val_mask_res' (absolute px) and 'val_full_res_masks' (bool) are "
            f"deprecated legacy spellings."
        )

    if not present:
        return ValMaskResolution(NATIVE)

    key = present[0]
    raw = extra[key]

    if key == "val_mask_downsample":
        return _parse_downsample(raw)

    if key == "val_full_res_masks":
        _warn_legacy(
            "val_full_res_masks",
            "val_mask_downsample: 1" if raw else f"val_mask_downsample: {NATIVE}",
        )
        return ValMaskResolution("divisor", 1) if raw else ValMaskResolution(NATIVE)

    # val_mask_res -- absolute pixels, or the falsy full-res opt-out.
    if not raw:
        _warn_legacy("val_mask_res: 0/null", "val_mask_downsample: 1")
        return ValMaskResolution("divisor", 1)
    if not isinstance(raw, int) or isinstance(raw, bool) or raw < 1:
        raise ValueError(
            f"val_mask_res must be a positive int (px) or 0/null for full "
            f"resolution, got {raw!r}."
        )
    _warn_legacy(
        f"val_mask_res: {raw}",
        f"val_mask_downsample (e.g. {NATIVE}, or a divisor like 2) -- an "
        f"absolute pixel target silently changes meaning across input "
        f"resolutions",
    )
    return ValMaskResolution("absolute", raw)


def _parse_downsample(raw) -> ValMaskResolution:
    if isinstance(raw, str):
        if raw.strip().lower() == NATIVE:
            return ValMaskResolution(NATIVE)
        raise ValueError(
            f"val_mask_downsample must be {NATIVE!r} or a number >= 1, got {raw!r}."
        )
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(
            f"val_mask_downsample must be {NATIVE!r} or a number >= 1, got {raw!r}."
        )
    if raw < 1:
        raise ValueError(
            f"val_mask_downsample must be >= 1 (it is a DIVISOR of the input "
            f"resolution: 1 = full res, 4 = input/4), got {raw!r}. A value "
            f"below 1 would ask for masks larger than the model's input."
        )
    return ValMaskResolution("divisor", raw)


def target_mask_hw(
    spec: ValMaskResolution, input_hw: tuple[int, int] | None
) -> tuple[int, int] | None:
    """Target mask (height, width), or ``None`` meaning "leave masks alone".

    ``None`` is returned for ``native`` -- and also whenever the input canvas
    size is unknown, because a divisor is meaningless without something to
    divide. Falling back to native there is the safe choice: it is the one
    policy that needs no knowledge of the canvas.
    """
    if spec.mode == "absolute":
        return (int(spec.value), int(spec.value))
    if spec.mode == NATIVE or input_hw is None:
        return None
    h, w = input_hw
    if spec.value == 1:
        return (int(h), int(w))
    return (max(1, round(h / spec.value)), max(1, round(w / spec.value)))


_warned: set[str] = set()


def _warn_legacy(old: str, new: str) -> None:
    if old in _warned:
        return
    _warned.add(old)
    logger.warning(
        "DEPRECATED val-mask knob '%s' -- use %s. The legacy key is still "
        "honoured, but it will not track a change of input resolution.",
        old, new,
    )
