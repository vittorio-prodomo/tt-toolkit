"""Base class for instance segmentation datasets with multi-sample augmentation."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
from torch.utils.data import Dataset

from training_toolkit.configs import AugConfig
from training_toolkit.data.multi_sample import (
    MosaicTransform, CutMixTransform, MixUpTransform, CopyPasteTransform,
)
from training_toolkit.data.sample_transforms import merge_overlapping_masks

logger = logging.getLogger(__name__)


class BaseSegmentationDataset(Dataset, ABC):
    """Abstract dataset for instance segmentation.

    Subclasses implement :meth:`_load_sample`, :attr:`class_names`,
    :attr:`num_classes`, and :meth:`__len__`.  Multi-sample augmentations
    (mosaic, cutmix, mixup, copy-paste) and single-image transforms are
    handled here.
    """

    def __init__(
        self,
        transform: Callable | None = None,
        aug_config: AugConfig | None = None,
        mosaic_transform: Callable | None = None,
    ) -> None:
        self._transform = transform
        self._mosaic_transform = mosaic_transform
        self._aug_config = aug_config

        # Instantiate multi-sample transforms based on aug_config
        self._mosaic: MosaicTransform | None = None
        self._cutmix: CutMixTransform | None = None
        self._mixup: MixUpTransform | None = None
        self._copy_paste: CopyPasteTransform | None = None

        if aug_config is not None:
            if aug_config.mosaic_p > 0:
                assert aug_config.imgsz is not None, "imgsz must be set for mosaic"
                self._mosaic = MosaicTransform(target_size=aug_config.imgsz)
            if aug_config.cutmix_p > 0:
                self._cutmix = CutMixTransform()
            if aug_config.mixup_p > 0:
                self._mixup = MixUpTransform(alpha=aug_config.mixup_alpha)
            if aug_config.copy_paste_p > 0:
                self._copy_paste = CopyPasteTransform(
                    max_instances=aug_config.copy_paste_max_instances
                )

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _load_sample(self, idx: int) -> dict[str, Any]:
        """Load a single sample without any transforms.

        Returns dict with keys: image, masks, boxes, labels, image_id.
        """

    def _image_path(self, idx: int) -> str:
        """Absolute path to the image file for sample *idx* (no decode)."""
        raise NotImplementedError

    def _load_annotations(self, idx: int) -> list:
        """Return a list of ``tile_materialize.Instance`` for sample *idx*.

        Reads label/annotation geometry only — no pixel decoding. Used by the
        streaming tiled dataset's tile-local rasterizer.
        """
        raise NotImplementedError

    def _image_class_mask(self, idx: int):
        """Per-image class_mask (list of length num_classes) or None.

        Returns None by default (dataset carries no per-image class_mask).
        Subclasses backed by a COCO file with ``class_mask`` records override
        this to read the mask from the COCO image record.
        """
        return None

    def _image_regime(self, idx: int) -> str:
        """Default tiling regime for bases without per-image regimes — tile
        normally (outdoor). Overridden by CocoSegmentationDataset."""
        return "outdoor_tiled"

    def _load_tile_metadata(self, idx: int) -> tuple[int, int, "np.ndarray"]:
        """Return ``(width, height, boxes)`` for the sample at *idx*.

        Used by :class:`TiledSegmentationDataset` to build the tile index
        without decoding pixel data.  The default implementation falls back
        to :meth:`_load_sample`, but subclasses should override this with
        a lightweight implementation that reads only image headers and
        label files.

        Returns
        -------
        width : int
        height : int
        boxes : np.ndarray, shape (N, 4), dtype float32, xyxy pixel coords
        """
        sample = self._load_sample(idx)
        h, w = sample["image"].shape[:2]
        return w, h, sample["boxes"]

    @property
    @abstractmethod
    def class_names(self) -> dict[int, str]:
        """Mapping of category id to category name."""

    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Total number of categories."""

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self._load_sample(idx)

        if self._aug_config is not None and self._aug_config.merge_overlapping_masks:
            sample = merge_overlapping_masks(sample)

        used_mosaic = False
        if self._aug_config is not None:
            sample, used_mosaic = self._apply_multi_sample(sample)

        # Pick the mosaic-specific transform if mosaic fired and one exists
        transform = (
            self._mosaic_transform
            if used_mosaic and self._mosaic_transform is not None
            else self._transform
        )
        if transform is not None:
            sample = self._apply_transform(sample, transform)

        return sample

    # ------------------------------------------------------------------
    # Multi-sample augmentation
    # ------------------------------------------------------------------

    def _apply_multi_sample(self, sample: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Apply one of the mutually exclusive multi-sample transforms.

        Returns the (possibly modified) sample and a boolean indicating
        whether mosaic was applied.
        """
        rng = np.random.default_rng()
        cfg = self._aug_config
        assert cfg is not None
        r = rng.random()
        used_mosaic = False

        if self._mosaic is not None and r < cfg.mosaic_p:
            extras = [self._load_sample(int(rng.integers(len(self)))) for _ in range(3)]
            sample = self._mosaic(sample, extras)
            used_mosaic = True
        elif self._cutmix is not None and r < cfg.mosaic_p + cfg.cutmix_p:
            donor = self._load_sample(int(rng.integers(len(self))))
            sample = self._cutmix(sample, donor)
        elif self._mixup is not None and r < cfg.mosaic_p + cfg.cutmix_p + cfg.mixup_p:
            donor = self._load_sample(int(rng.integers(len(self))))
            sample = self._mixup(sample, donor)

        if self._copy_paste is not None and rng.random() < cfg.copy_paste_p:
            donor = self._load_sample(int(rng.integers(len(self))))
            sample = self._copy_paste(sample, donor)

        return sample, used_mosaic

    # ------------------------------------------------------------------
    # Single-image transform
    # ------------------------------------------------------------------

    def _apply_transform(
        self, sample: dict[str, Any], transform: Callable | None = None,
    ) -> dict[str, Any]:
        """Apply an augmentation pipeline (crop, flip, color, etc.)."""
        if transform is None:
            transform = self._transform
        assert transform is not None

        # Filter out degenerate boxes before passing to Albumentations
        # (which rejects x_max <= x_min or y_max <= y_min)
        boxes = sample["boxes"]
        if len(boxes) > 0:
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not valid.all():
                keep_idx = np.where(valid)[0]
                sample["boxes"] = boxes[keep_idx]
                sample["masks"] = [sample["masks"][i] for i in keep_idx]
                sample["labels"] = sample["labels"][keep_idx]
                if "ignore" in sample:
                    sample["ignore"] = sample["ignore"][keep_idx]
                if "iscrowd" in sample:
                    sample["iscrowd"] = sample["iscrowd"][keep_idx]

        has_region_boxes = "region_boxes" in sample
        has_region_class_masks = "region_class_masks" in sample
        if has_region_boxes != has_region_class_masks:
            raise ValueError(
                "region_boxes and region_class_masks must be present together"
            )

        region_masks: list[np.ndarray] = []
        region_class_masks: np.ndarray | None = None
        if has_region_boxes:
            region_boxes = np.asarray(sample["region_boxes"], dtype=np.float32)
            region_class_masks = np.asarray(
                sample["region_class_masks"], dtype=np.float32
            )
            if (
                region_boxes.ndim != 2
                or region_boxes.shape[1] != 4
                or region_class_masks.ndim != 2
                or region_class_masks.shape[0] != region_boxes.shape[0]
            ):
                raise ValueError(
                    "region_boxes must have shape (R, 4) and "
                    "region_class_masks shape (R, L)"
                )

            image_h, image_w = sample["image"].shape[:2]
            scale = np.array(
                [image_w, image_h, image_w, image_h], dtype=np.float32
            )
            for box in region_boxes:
                x0, y0, x1, y1 = np.rint(box * scale).astype(np.int64)
                if not (
                    0 <= x0 < x1 <= image_w
                    and 0 <= y0 < y1 <= image_h
                ):
                    raise ValueError(
                        f"invalid normalized half-open region box: {box.tolist()}"
                    )
                region_mask = np.zeros((image_h, image_w), dtype=np.uint8)
                region_mask[y0:y1, x0:x1] = 1
                region_masks.append(region_mask)

        original_labels = sample["labels"]
        num_instance_masks = len(sample["masks"])

        transformed = transform(
            image=sample["image"],
            masks=[*sample["masks"], *region_masks],
            bboxes=sample["boxes"].tolist(),
            labels=sample["labels"].tolist(),
        )

        sample["image"] = transformed["image"]
        transformed_masks = transformed["masks"]
        expected_mask_count = num_instance_masks + len(region_masks)
        if len(transformed_masks) != expected_mask_count:
            raise ValueError(
                "transform changed mask count: "
                f"expected {expected_mask_count}, got {len(transformed_masks)}"
            )
        masks = transformed_masks[:num_instance_masks]
        transformed_region_masks = transformed_masks[num_instance_masks:]

        # Albumentations filters bboxes/labels via BboxParams but does NOT
        # filter masks in sync. Use original labels (same count/order as
        # masks) and recompute bboxes from the transformed masks.
        labels = original_labels

        # Keep only instances whose mask survived the spatial transform
        keep = [i for i, m in enumerate(masks) if m.sum() > 0]
        sample["masks"] = [masks[i] for i in keep]
        sample["labels"] = labels[keep] if len(labels) > 0 else np.array([], dtype=np.int64)
        # Keep per-instance ignore flags (size-band SNIP) in sync with the
        # surviving instances — else the criterion's ignore mask desyncs from
        # boxes/labels and index-selects out of bounds.
        if "ignore" in sample:
            sample["ignore"] = sample["ignore"][keep]
        # Keep per-instance iscrowd flags (COCO crowd/sliver GT) in sync with
        # the surviving instances — same rationale as "ignore" above: it is
        # per-instance and aligned with masks/labels (unlike "class_mask",
        # which is per-image and rides through untouched). A stale
        # full-length array here would silently misalign with the surviving
        # masks after a transform drops an instance (e.g. min_visibility).
        if "iscrowd" in sample:
            sample["iscrowd"] = sample["iscrowd"][keep]

        # Recompute bboxes from masks (more accurate than clipped bboxes)
        boxes = []
        for i in keep:
            ys, xs = np.where(masks[i] > 0)
            boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max())])
        boxes = (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )

        # Drop instances whose recomputed box is degenerate. With the
        # inclusive-extremes convention above, a mask that survives the
        # spatial transform as a 1-px-wide/tall sliver (mosaic's quadrant
        # squeeze and RandomResizedCrop produce these routinely) recomputes
        # to x_max == x_min or y_max == y_min. Left unfiltered, a zero-area
        # box reaches the YOLO seg loss's single_mask_loss (divides by box
        # area) -> inf -> NaN in crop_mask's backward, permanently poisoning
        # the optimizer step. Mirrors the pre-transform degenerate-box
        # filter above, applied in the recomputed-box index space (aligned
        # with masks/labels/ignore/iscrowd via `keep` just above).
        if len(boxes) > 0:
            valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not valid.all():
                valid_idx = np.where(valid)[0]
                boxes = boxes[valid_idx]
                sample["masks"] = [sample["masks"][i] for i in valid_idx]
                sample["labels"] = sample["labels"][valid_idx]
                if "ignore" in sample:
                    sample["ignore"] = sample["ignore"][valid_idx]
                if "iscrowd" in sample:
                    sample["iscrowd"] = sample["iscrowd"][valid_idx]

        sample["boxes"] = boxes

        if region_class_masks is not None:
            output_h, output_w = sample["image"].shape[:2]
            surviving_region_boxes: list[list[float]] = []
            surviving_region_indices: list[int] = []
            for i, region_mask in enumerate(transformed_region_masks):
                support = np.asarray(region_mask) > 0
                ys, xs = np.where(support)
                if len(ys) == 0:
                    continue

                x0 = int(xs.min())
                y0 = int(ys.min())
                x1 = int(xs.max()) + 1
                y1 = int(ys.max()) + 1
                if int(support.sum()) != (x1 - x0) * (y1 - y0):
                    raise ValueError(
                        "spatial transform made a surviving class-mask region "
                        f"non-rectangular (region index {i})"
                    )

                surviving_region_boxes.append(
                    [
                        x0 / output_w,
                        y0 / output_h,
                        x1 / output_w,
                        y1 / output_h,
                    ]
                )
                surviving_region_indices.append(i)

            sample["region_boxes"] = (
                np.asarray(surviving_region_boxes, dtype=np.float32)
                if surviving_region_boxes
                else np.zeros((0, 4), dtype=np.float32)
            )
            sample["region_class_masks"] = region_class_masks[
                surviving_region_indices
            ].astype(np.float32, copy=False)
        return sample
