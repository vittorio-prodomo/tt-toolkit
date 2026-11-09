"""Multi-sample augmentation transforms for instance segmentation.

Provides mosaic, cutmix, mixup, and copy-paste transforms that properly
handle per-instance binary masks (unlike Albumentations' built-in mosaic
which only supports single-channel masks).
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class MultiSampleTransforms:
    mosaic: "MosaicTransform | None" = None
    cutmix: "CutMixTransform | None" = None
    mixup: "MixUpTransform | None" = None
    copy_paste: "CopyPasteTransform | None" = None


def make_multi_sample_transforms(aug_config) -> MultiSampleTransforms:
    """Instantiate the enabled multi-sample transforms from aug_config — the
    same rules as BaseSegmentationDataset.__init__, reusable by the streaming
    pooled path."""
    m = MultiSampleTransforms()
    if aug_config is None:
        return m
    if aug_config.mosaic_p > 0:
        assert aug_config.imgsz is not None, "imgsz must be set for mosaic"
        m.mosaic = MosaicTransform(target_size=aug_config.imgsz)
    if aug_config.cutmix_p > 0:
        m.cutmix = CutMixTransform()
    if aug_config.mixup_p > 0:
        m.mixup = MixUpTransform(alpha=aug_config.mixup_alpha)
    if aug_config.copy_paste_p > 0:
        m.copy_paste = CopyPasteTransform(max_instances=aug_config.copy_paste_max_instances)
    return m


def _combine_class_masks(contributing_samples: list[dict]) -> np.ndarray | None:
    """AND-combine ``class_mask`` across samples that contributed pixels.

    Used by multi-sample transforms (mosaic/cutmix/mixup/copy-paste) to
    propagate per-sample annotation-scope information into the combined
    output. The rule: a class is "active" in the combined sample only if
    *every* contributing constituent annotates that class. Any constituent
    that says "I don't know about class X here" makes the combined sample
    also "I don't know about class X" — conservative on purpose, to avoid
    silently reintroducing the false-suppression problem that masked loss
    is designed to prevent.

    Parameters
    ----------
    contributing_samples:
        Samples whose pixels appear in the output. Pass only the ones that
        actually placed pixels (e.g. mosaic quadrants with positive area;
        primary alone when an early-return shortcut bypasses the donor).
        Samples without a ``class_mask`` field are treated as all-ones — no
        information added to the AND.

    Returns
    -------
    np.ndarray | None
        ``None`` when no contributor carried a ``class_mask`` (the combined
        sample also has no mask — the criterion defaults to all-ones for
        it). Otherwise a 1D float32 array of the same length as the input
        masks, holding the element-wise minimum (= logical AND on 0/1
        vectors) across all contributors.
    """
    # class_mask is conventionally a list[num_classes] (base_dataset._image_class_mask,
    # label_crop pass-through); coerce each to a float32 array so list OR ndarray works.
    arrs = [np.asarray(m, dtype=np.float32)
            for m in (s.get("class_mask") for s in contributing_samples)
            if m is not None]
    if not arrs:
        return None
    length = arrs[0].shape[0]
    combined = np.ones(length, dtype=np.float32)
    for a in arrs:
        if a.shape[0] != length:
            raise ValueError(
                f"class_mask length mismatch across multi-sample constituents: "
                f"expected {length}, got {a.shape[0]}"
            )
        combined = np.minimum(combined, a)
    return combined


class MosaicTransform:
    """4-image mosaic augmentation with proper instance mask support.

    Composites four images into a 2x canvas around a random center point.
    Each instance mask is individually resized and placed, and bounding boxes
    are recomputed from the placed mask pixels.

    Parameters
    ----------
    target_size:
        Base size; the output canvas is ``(target_size*2, target_size*2, 3)``.
    center_range:
        Fraction range ``(lo, hi)`` within the canvas for the random center.
    """

    def __init__(
        self,
        target_size: int = 640,
        center_range: tuple[float, float] = (0.25, 0.75),
    ) -> None:
        self.target_size = target_size
        self.center_range = center_range

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #
    def __call__(self, primary: dict, extras: list[dict]) -> dict:
        """Build a 4-image mosaic.

        Parameters
        ----------
        primary:
            Dict with keys ``image`` (H,W,3 uint8), ``masks`` (list of (H,W)
            uint8), ``boxes`` (N,4 float32 xyxy), ``labels`` (N, int64).
        extras:
            List of 3 more dicts with the same structure.

        Returns
        -------
        dict
            Instance fields composited onto a ``2*target_size`` canvas. When
            at least one positive-area contributor carries ``class_mask``,
            the output also contains the conservative AND-combined optional
            ``class_mask``, ``region_boxes`` as an ``(R, 4)`` float32 array
            of canvas-normalized half-open ``xyxy`` contributor bounds, and
            ``region_class_masks`` as an aligned ``(R, L)`` float32 array.
            Maskless contributors receive all-ones region rows. If no
            positive-area contributor carries ``class_mask``, all three
            optional keys are omitted; degenerate quadrants never contribute
            rows.
        """
        assert len(extras) == 3, "MosaicTransform requires exactly 3 extras"
        samples = [primary] + extras

        canvas_h = self.target_size * 2
        canvas_w = self.target_size * 2
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Random center point
        lo, hi = self.center_range
        cy = int(np.random.uniform(lo, hi) * canvas_h)
        cx = int(np.random.uniform(lo, hi) * canvas_w)

        # Quadrant regions: (y_start, y_end, x_start, x_end)
        quadrants = [
            (0, cy, 0, cx),                       # top-left
            (0, cy, cx, canvas_w),                 # top-right
            (cy, canvas_h, 0, cx),                 # bottom-left
            (cy, canvas_h, cx, canvas_w),          # bottom-right
        ]

        out_masks: list[np.ndarray] = []
        out_boxes: list[np.ndarray] = []
        out_labels: list[int] = []
        out_ignore: list[bool] = []
        any_ignore: bool = False
        contributing_samples: list[dict] = []
        contributing_regions: list[tuple[int, int, int, int]] = []

        for sample, (y0, y1, x0, x1) in zip(samples, quadrants):
            qh = y1 - y0
            qw = x1 - x0
            if qh <= 0 or qw <= 0:
                continue
            contributing_samples.append(sample)
            contributing_regions.append((y0, y1, x0, x1))
            any_ignore |= "ignore" in sample

            # Resize image to fit quadrant
            img_resized = cv2.resize(
                sample["image"], (qw, qh), interpolation=cv2.INTER_LINEAR
            )
            canvas[y0:y1, x0:x1] = img_resized

            # Process each instance mask
            masks = sample["masks"]
            labels = sample["labels"]
            for i, mask in enumerate(masks):
                mask_resized = cv2.resize(
                    mask, (qw, qh), interpolation=cv2.INTER_NEAREST
                )
                # Skip if mask becomes empty after resize
                if mask_resized.max() == 0:
                    continue

                # Place on full-canvas mask
                full_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
                full_mask[y0:y1, x0:x1] = mask_resized

                # Compute bbox from mask pixels
                ys, xs = np.where(full_mask > 0)
                if len(ys) == 0:
                    continue

                bbox = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
                )
                out_masks.append(full_mask)
                out_boxes.append(bbox)
                out_labels.append(int(labels[i]))
                out_ignore.append(bool(sample["ignore"][i]) if "ignore" in sample else False)

        # Build output arrays
        if len(out_boxes) > 0:
            boxes = np.stack(out_boxes, axis=0).astype(np.float32)
            labels_arr = np.array(out_labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)

        output = {
            "image": canvas,
            "masks": out_masks,
            "boxes": boxes,
            "labels": labels_arr,
        }
        if any_ignore:
            output["ignore"] = np.array(out_ignore, dtype=bool)
        combined_mask = _combine_class_masks(contributing_samples)
        if combined_mask is not None:
            output["class_mask"] = combined_mask
            num_classes = combined_mask.shape[0]
            output["region_boxes"] = np.asarray(
                [
                    [
                        x0 / canvas_w,
                        y0 / canvas_h,
                        x1 / canvas_w,
                        y1 / canvas_h,
                    ]
                    for y0, y1, x0, x1 in contributing_regions
                ],
                dtype=np.float32,
            )
            output["region_class_masks"] = np.stack(
                [
                    np.asarray(sample["class_mask"], dtype=np.float32)
                    if sample.get("class_mask") is not None
                    else np.ones(num_classes, dtype=np.float32)
                    for sample in contributing_samples
                ],
                axis=0,
            ).astype(np.float32, copy=False)
        return output


class CutMixTransform:
    """CutMix augmentation with instance-aware mask handling.

    Swaps a random rectangular region between two images.  Primary instances
    whose masks are heavily occluded by the cut region are dropped; donor
    instances that overlap sufficiently with the cut region are added.

    Parameters
    ----------
    overlap_thresh:
        Fraction threshold.  For primary instances, keep if
        ``remaining_area / original_area >= overlap_thresh``.  For donor
        instances, include if ``mask_in_cut / total_mask >= overlap_thresh``.
    """

    def __init__(self, overlap_thresh: float = 0.5) -> None:
        self.overlap_thresh = overlap_thresh

    def __call__(self, primary: dict, donor: dict) -> dict:
        """Apply CutMix to *primary* using a region from *donor*.

        Parameters
        ----------
        primary, donor:
            Dicts with keys ``image`` (H,W,3 uint8), ``masks`` (list of
            (H,W) uint8), ``boxes`` (N,4 float32 xyxy), ``labels`` (N, int64).

        Returns
        -------
        dict with the same keys, same spatial size as *primary*.
        """
        h, w = primary["image"].shape[:2]

        # 1. Random lambda from Beta(1, 1) → Uniform(0, 1)
        lam = float(np.random.beta(1.0, 1.0))

        # 2. Cut dimensions
        cut_w = int(w * np.sqrt(1.0 - lam))
        cut_h = int(h * np.sqrt(1.0 - lam))

        # 3. Random centre, clipped to image bounds
        cx = int(np.random.uniform(0, w))
        cy = int(np.random.uniform(0, h))
        x1 = max(cx - cut_w // 2, 0)
        y1 = max(cy - cut_h // 2, 0)
        x2 = min(cx + cut_w // 2, w)
        y2 = min(cy + cut_h // 2, h)

        # 4. If cut region is empty, return primary unchanged
        if x2 <= x1 or y2 <= y1:
            out = {
                "image": primary["image"].copy(),
                "masks": [m.copy() for m in primary["masks"]],
                "boxes": primary["boxes"].copy(),
                "labels": primary["labels"].copy(),
            }
            # Donor never contributed — propagate primary's fields only.
            if "class_mask" in primary:
                out["class_mask"] = primary["class_mask"].copy()
            if "ignore" in primary:
                out["ignore"] = primary["ignore"].copy()
            return out

        # 5. Resize donor image to primary size and paste cut region
        donor_img = cv2.resize(
            donor["image"], (w, h), interpolation=cv2.INTER_LINEAR
        )
        out_image = primary["image"].copy()
        out_image[y1:y2, x1:x2] = donor_img[y1:y2, x1:x2]

        out_masks: list[np.ndarray] = []
        out_boxes: list[np.ndarray] = []
        out_labels: list[int] = []
        out_ignore: list[bool] = []
        any_ignore: bool = "ignore" in primary or "ignore" in donor

        # 6. Primary instances: erase cut region, keep if not heavily occluded
        for i, mask in enumerate(primary["masks"]):
            original_area = int(mask.sum())
            if original_area == 0:
                continue

            new_mask = mask.copy()
            new_mask[y1:y2, x1:x2] = 0
            remaining_area = int(new_mask.sum())

            if remaining_area / original_area >= self.overlap_thresh:
                ys, xs = np.where(new_mask > 0)
                if len(ys) == 0:
                    continue
                bbox = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=np.float32,
                )
                out_masks.append(new_mask)
                out_boxes.append(bbox)
                out_labels.append(int(primary["labels"][i]))
                out_ignore.append(bool(primary["ignore"][i]) if "ignore" in primary else False)

        # 7. Donor instances: resize masks, include if enough is in cut region
        for i, mask in enumerate(donor["masks"]):
            mask_resized = cv2.resize(
                mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            total_area = int(mask_resized.sum())
            if total_area == 0:
                continue

            # Only keep pixels inside the cut region
            cut_mask = np.zeros((h, w), dtype=np.uint8)
            cut_mask[y1:y2, x1:x2] = mask_resized[y1:y2, x1:x2]
            area_in_cut = int(cut_mask.sum())

            if area_in_cut / total_area >= self.overlap_thresh:
                ys, xs = np.where(cut_mask > 0)
                if len(ys) == 0:
                    continue
                bbox = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=np.float32,
                )
                out_masks.append(cut_mask)
                out_boxes.append(bbox)
                out_labels.append(int(donor["labels"][i]))
                out_ignore.append(bool(donor["ignore"][i]) if "ignore" in donor else False)

        # Build output arrays
        if len(out_boxes) > 0:
            boxes = np.stack(out_boxes, axis=0).astype(np.float32)
            labels_arr = np.array(out_labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)

        output = {
            "image": out_image,
            "masks": out_masks,
            "boxes": boxes,
            "labels": labels_arr,
        }
        if any_ignore:
            output["ignore"] = np.array(out_ignore, dtype=bool)
        combined_mask = _combine_class_masks([primary, donor])
        if combined_mask is not None:
            output["class_mask"] = combined_mask
        return output


class MixUpTransform:
    """MixUp augmentation: alpha-blend two images, stack all instances.

    Blends the pixel values of two images using a weight sampled from a Beta
    distribution, while collecting (not blending) all instance masks from both
    images.

    Parameters
    ----------
    alpha:
        Parameter for the ``Beta(alpha, alpha)`` distribution used to sample
        the blending weight ``lam``.
    """

    def __init__(self, alpha: float = 1.5) -> None:
        self.alpha = alpha

    def __call__(self, primary: dict, donor: dict) -> dict:
        """Apply MixUp to *primary* using *donor*.

        Parameters
        ----------
        primary, donor:
            Dicts with keys ``image`` (H,W,3 uint8), ``masks`` (list of
            (H,W) uint8), ``boxes`` (N,4 float32 xyxy), ``labels`` (N, int64).

        Returns
        -------
        dict with the same keys, same spatial size as *primary*.
        """
        h, w = primary["image"].shape[:2]

        # 1. Sample blending weight from Beta distribution
        rng = np.random.default_rng()
        lam = float(rng.beta(self.alpha, self.alpha))

        # 2. Resize donor image to primary's size
        donor_img = cv2.resize(
            donor["image"], (w, h), interpolation=cv2.INTER_LINEAR
        )

        # 3. Blend pixel values
        blended = lam * primary["image"].astype(np.float64) + (1 - lam) * donor_img.astype(np.float64)
        out_image = np.clip(blended, 0, 255).astype(np.uint8)

        out_masks: list[np.ndarray] = []
        out_boxes: list[np.ndarray] = []
        out_labels: list[int] = []
        out_ignore: list[bool] = []
        any_ignore: bool = "ignore" in primary or "ignore" in donor

        # 4. Collect all instances from primary
        for i, mask in enumerate(primary["masks"]):
            if mask.max() == 0:
                continue
            ys, xs = np.where(mask > 0)
            if len(ys) == 0:
                continue
            bbox = np.array(
                [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
            )
            out_masks.append(mask.copy())
            out_boxes.append(bbox)
            out_labels.append(int(primary["labels"][i]))
            out_ignore.append(bool(primary["ignore"][i]) if "ignore" in primary else False)

        # 5. Collect all instances from donor (resize masks)
        for i, mask in enumerate(donor["masks"]):
            mask_resized = cv2.resize(
                mask, (w, h), interpolation=cv2.INTER_NEAREST
            )
            if mask_resized.max() == 0:
                continue
            ys, xs = np.where(mask_resized > 0)
            if len(ys) == 0:
                continue
            bbox = np.array(
                [xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32
            )
            out_masks.append(mask_resized)
            out_boxes.append(bbox)
            out_labels.append(int(donor["labels"][i]))
            out_ignore.append(bool(donor["ignore"][i]) if "ignore" in donor else False)

        # Build output arrays
        if len(out_boxes) > 0:
            boxes = np.stack(out_boxes, axis=0).astype(np.float32)
            labels_arr = np.array(out_labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)

        output = {
            "image": out_image,
            "masks": out_masks,
            "boxes": boxes,
            "labels": labels_arr,
        }
        if any_ignore:
            output["ignore"] = np.array(out_ignore, dtype=bool)
        combined_mask = _combine_class_masks([primary, donor])
        if combined_mask is not None:
            output["class_mask"] = combined_mask
        return output


class CopyPasteTransform:
    """Copy-Paste augmentation: paste donor instances onto the primary image.

    Randomly selects instances from a donor image and pastes them onto the
    primary image.  Primary instances that lose too much area due to occlusion
    are removed; surviving instances have their bounding boxes recomputed.

    Parameters
    ----------
    max_instances:
        Maximum number of donor instances to paste.
    flip_p:
        Probability of horizontally flipping each pasted mask.
    """

    def __init__(self, max_instances: int = 5, flip_p: float = 0.5) -> None:
        self.max_instances = max_instances
        self.flip_p = flip_p

    def __call__(self, primary: dict, donor: dict) -> dict:
        """Paste donor instances onto *primary*.

        Parameters
        ----------
        primary, donor:
            Dicts with keys ``image`` (H,W,3 uint8), ``masks`` (list of
            (H,W) uint8), ``boxes`` (N,4 float32 xyxy), ``labels`` (N, int64).

        Returns
        -------
        dict with the same keys, same spatial size as *primary*.
        """
        h, w = primary["image"].shape[:2]

        # 1. If donor has 0 instances, return primary unchanged
        if len(donor["masks"]) == 0:
            out = {
                "image": primary["image"].copy(),
                "masks": [m.copy() for m in primary["masks"]],
                "boxes": primary["boxes"].copy(),
                "labels": primary["labels"].copy(),
            }
            if "class_mask" in primary:
                out["class_mask"] = primary["class_mask"].copy()
            if "ignore" in primary:
                out["ignore"] = primary["ignore"].copy()
            return out

        # 2. Select random donor instances (without replacement)
        n_donor = len(donor["masks"])
        n_select = min(self.max_instances, n_donor)
        selected_indices = np.random.choice(n_donor, size=n_select, replace=False)

        # 3. Process each selected donor instance
        pasted_masks: list[np.ndarray] = []
        pasted_labels: list[int] = []
        pasted_ignores: list[bool] = []

        for idx in selected_indices:
            mask = donor["masks"][idx]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # With probability flip_p, horizontally flip
            if np.random.random() < self.flip_p:
                mask_resized = np.fliplr(mask_resized).copy()

            # Skip if empty after resize
            if mask_resized.max() == 0:
                continue

            pasted_masks.append(mask_resized)
            pasted_labels.append(int(donor["labels"][idx]))
            pasted_ignores.append(bool(donor["ignore"][idx]) if "ignore" in donor else False)

        # If no masks survived, return primary unchanged
        if len(pasted_masks) == 0:
            out = {
                "image": primary["image"].copy(),
                "masks": [m.copy() for m in primary["masks"]],
                "boxes": primary["boxes"].copy(),
                "labels": primary["labels"].copy(),
            }
            if "class_mask" in primary:
                out["class_mask"] = primary["class_mask"].copy()
            if "ignore" in primary:
                out["ignore"] = primary["ignore"].copy()
            return out

        # 4. Build paste_union and paste donor pixels
        paste_union = np.zeros((h, w), dtype=np.uint8)
        for pm in pasted_masks:
            paste_union = np.maximum(paste_union, pm)

        donor_img = cv2.resize(
            donor["image"], (w, h), interpolation=cv2.INTER_LINEAR
        )
        out_image = primary["image"].copy()
        paste_region = paste_union > 0
        out_image[paste_region] = donor_img[paste_region]

        out_masks: list[np.ndarray] = []
        out_boxes: list[np.ndarray] = []
        out_labels: list[int] = []
        out_ignore: list[bool] = []
        any_ignore: bool = "ignore" in primary or "ignore" in donor

        # 5. Filter primary instances: zero out pixels under paste_union
        for i, mask in enumerate(primary["masks"]):
            original_area = int(mask.sum())
            if original_area == 0:
                continue

            surviving_mask = mask.copy()
            surviving_mask[paste_region] = 0
            surviving_area = int(surviving_mask.sum())

            if surviving_area / original_area >= 0.5:
                ys, xs = np.where(surviving_mask > 0)
                if len(ys) == 0:
                    continue
                bbox = np.array(
                    [xs.min(), ys.min(), xs.max(), ys.max()],
                    dtype=np.float32,
                )
                out_masks.append(surviving_mask)
                out_boxes.append(bbox)
                out_labels.append(int(primary["labels"][i]))
                out_ignore.append(bool(primary["ignore"][i]) if "ignore" in primary else False)

        # 6. Append pasted donor instances with recomputed bboxes
        for pm, lbl, ign in zip(pasted_masks, pasted_labels, pasted_ignores):
            ys, xs = np.where(pm > 0)
            if len(ys) == 0:
                continue
            bbox = np.array(
                [xs.min(), ys.min(), xs.max(), ys.max()],
                dtype=np.float32,
            )
            out_masks.append(pm)
            out_boxes.append(bbox)
            out_labels.append(lbl)
            out_ignore.append(ign)

        # Build output arrays
        if len(out_boxes) > 0:
            boxes = np.stack(out_boxes, axis=0).astype(np.float32)
            labels_arr = np.array(out_labels, dtype=np.int64)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            labels_arr = np.zeros((0,), dtype=np.int64)

        output = {
            "image": out_image,
            "masks": out_masks,
            "boxes": boxes,
            "labels": labels_arr,
        }
        if any_ignore:
            output["ignore"] = np.array(out_ignore, dtype=bool)
        combined_mask = _combine_class_masks([primary, donor])
        if combined_mask is not None:
            output["class_mask"] = combined_mask
        return output
