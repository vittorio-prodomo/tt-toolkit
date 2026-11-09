"""FiftyOne-backed dataset for instance segmentation.

Loads images from disk (via filepaths stored in FiftyOne) and annotations
directly from FiftyOne's MongoDB backend.  Supports both ``Detections``
(with dense instance masks) and ``Polylines`` (polygon-based) label fields.

This avoids the need to export FiftyOne datasets to YOLO/COCO format before
training — the dataset can be used directly as-is.

Usage::

    import fiftyone as fo
    ds = fo.load_dataset("my-dataset")
    dataset = FiftyOneSegmentationDataset(ds, label_field="detections")
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable

import cv2
import numpy as np

from training_toolkit.configs import AugConfig
from training_toolkit.data.base_dataset import BaseSegmentationDataset

logger = logging.getLogger(__name__)


class FiftyOneSegmentationDataset(BaseSegmentationDataset):
    """Dataset that reads images from disk and annotations from FiftyOne.

    Parameters
    ----------
    fo_dataset : fiftyone.Dataset or fiftyone.DatasetView
        A loaded FiftyOne dataset or view.
    label_field : str
        Name of the label field containing annotations.  Supports
        ``Detections`` (with ``.mask``) and ``Polylines`` (with ``.points``).
    class_remap : dict[str, int], optional
        Mapping from label name to contiguous class index.  If *None*,
        classes are auto-discovered from the dataset in alphabetical order.
    transform : callable, optional
        An Albumentations transform.
    aug_config : AugConfig, optional
        Augmentation configuration for multi-sample transforms.
    """

    def __init__(
        self,
        fo_dataset,
        label_field: str = "detections",
        class_remap: dict[str, int] | None = None,
        transform: Callable | None = None,
        aug_config: AugConfig | None = None,
    ) -> None:
        super().__init__(transform=transform, aug_config=aug_config)

        self._dataset = fo_dataset
        self._label_field = label_field

        # Materialize sample references for indexed access
        self._sample_ids: list[str] = [s.id for s in fo_dataset]
        self._filepaths: list[str] = fo_dataset.values("filepath")

        # Pre-load metadata (width, height) for tile index building
        self._widths: list[int] = []
        self._heights: list[int] = []
        for w, h in zip(
            fo_dataset.values("metadata.width"),
            fo_dataset.values("metadata.height"),
        ):
            self._widths.append(int(w))
            self._heights.append(int(h))

        # Detect label field type (needed before class discovery so we read the
        # correct sub-path: Detections vs Polylines).
        import fiftyone.core.labels as fol

        schema = fo_dataset.get_field_schema()
        field = schema.get(label_field)
        doc_type = getattr(field, "document_type", None)
        if doc_type is getattr(fol, "Detections", None):
            self._field_type = "detections"
        elif doc_type is getattr(fol, "Polylines", None):
            self._field_type = "polylines"
        else:
            raise ValueError(
                f"Label field '{label_field}' must be Detections or Polylines, "
                f"got {doc_type}"
            )

        # Build class mapping
        if class_remap is not None:
            self._label_to_idx = class_remap
        else:
            # Auto-discover classes alphabetically from the sub-path that matches
            # the field type. Using a hardcoded ".detections.label" here breaks
            # Polylines fields (count_values returns {None: N} -> a single bogus
            # class, silently dropping every annotation).
            label_counts = fo_dataset.count_values(
                f"{label_field}.{self._field_type}.label"
            )
            sorted_names = sorted(
                name for name in label_counts if name is not None
            )
            self._label_to_idx = {name: i for i, name in enumerate(sorted_names)}

        self._class_names = {i: name for name, i in self._label_to_idx.items()}

        logger.info(
            "FiftyOne dataset: %d samples, %d classes, field='%s' (%s)",
            len(self._sample_ids), len(self._class_names),
            label_field, self._field_type,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def __len__(self) -> int:
        return len(self._sample_ids)

    # ------------------------------------------------------------------
    # Tile metadata (lightweight, no pixel decoding)
    # ------------------------------------------------------------------

    def _load_tile_metadata(self, idx: int) -> tuple[int, int, np.ndarray]:
        """Return (width, height, boxes) from FiftyOne metadata.

        Uses pre-loaded dimensions and reads bounding boxes from MongoDB.
        No image decoding.
        """
        warnings.filterwarnings(
            "ignore", message="MongoClient opened before fork", category=UserWarning,
        )

        w = self._widths[idx]
        h = self._heights[idx]

        sample = self._dataset[self._sample_ids[idx]]
        labels_container = sample[self._label_field]

        if labels_container is None:
            return w, h, np.zeros((0, 4), dtype=np.float32)

        boxes: list[list[float]] = []

        if self._field_type == "detections":
            for det in labels_container.detections:
                if det.label not in self._label_to_idx:
                    continue
                bx, by, bw, bh = det.bounding_box  # relative [0, 1]
                boxes.append([
                    bx * w, by * h,
                    (bx + bw) * w, (by + bh) * h,
                ])
        else:  # polylines
            for poly in labels_container.polylines:
                if poly.label not in self._label_to_idx:
                    continue
                all_xs, all_ys = [], []
                for ring in poly.points:
                    for px, py in ring:
                        all_xs.append(px * w)
                        all_ys.append(py * h)
                if all_xs:
                    boxes.append([
                        min(all_xs), min(all_ys),
                        max(all_xs), max(all_ys),
                    ])

        return w, h, (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )

    # ------------------------------------------------------------------
    # Pyvips tiler interface (geometry only, no pixel decoding)
    # ------------------------------------------------------------------

    def _image_path(self, idx: int) -> str:
        """Absolute path to the image file for sample *idx* (no decode)."""
        return self._filepaths[idx]

    def _load_annotations(self, idx: int) -> list:
        """Build ``tile_materialize.Instance`` objects from the FiftyOne labels —
        geometry only, no pixel decoding (dims come from preloaded metadata).
        Mirrors the COCO/YOLO contract so the pyvips ROI tiler windows them
        identically: Detections -> bbox-relative mask + origin; Polylines ->
        absolute-pixel polygon rings."""
        from training_toolkit.data.tile_materialize import Instance

        warnings.filterwarnings(
            "ignore", message="MongoClient opened before fork", category=UserWarning,
        )
        w = self._widths[idx]
        h = self._heights[idx]
        sample = self._dataset[self._sample_ids[idx]]
        labels_container = sample[self._label_field]
        if labels_container is None:
            return []

        out: list[Instance] = []
        if self._field_type == "detections":
            for det in labels_container.detections:
                if det.label not in self._label_to_idx:
                    continue
                bx, by, bw, bh = det.bounding_box
                x1 = max(0, min(int(round(bx * w)), w))
                y1 = max(0, min(int(round(by * h)), h))
                x2 = max(0, min(int(round((bx + bw) * w)), w))
                y2 = max(0, min(int(round((by + bh) * h)), h))
                bbox_w, bbox_h = x2 - x1, y2 - y1
                if bbox_w <= 0 or bbox_h <= 0:
                    continue
                label = self._label_to_idx[det.label]
                if det.mask is not None:
                    m = det.mask.astype(np.uint8)
                    if m.shape != (bbox_h, bbox_w):
                        m = cv2.resize(m, (bbox_w, bbox_h),
                                       interpolation=cv2.INTER_NEAREST)
                    local_ys, local_xs = np.where(m > 0)
                    if len(local_xs) == 0:
                        continue
                    bbox = (float(x1 + int(local_xs.min())),
                            float(y1 + int(local_ys.min())),
                            float(x1 + int(local_xs.max())),
                            float(y1 + int(local_ys.max())))
                    out.append(Instance(label=label, polygon=None, mask=m,
                                        mask_origin=(x1, y1), bbox=bbox))
                else:
                    m = np.ones((bbox_h, bbox_w), dtype=np.uint8)
                    out.append(Instance(label=label, polygon=None, mask=m,
                                        mask_origin=(x1, y1),
                                        bbox=(float(x1), float(y1),
                                              float(x2), float(y2))))
        else:  # polylines
            for poly in labels_container.polylines:
                if poly.label not in self._label_to_idx:
                    continue
                rings: list[np.ndarray] = []
                xs: list[float] = []
                ys: list[float] = []
                for ring in poly.points:
                    pts = np.array([[px * w, py * h] for px, py in ring],
                                   dtype=np.float32)
                    if len(pts) >= 3:
                        rings.append(pts)
                        xs.extend(pts[:, 0].tolist())
                        ys.extend(pts[:, 1].tolist())
                if not rings:
                    continue
                out.append(Instance(
                    label=self._label_to_idx[poly.label],
                    polygon=rings, mask=None, mask_origin=None,
                    bbox=(float(min(xs)), float(min(ys)),
                          float(max(xs)), float(max(ys))),
                ))
        return out

    # ------------------------------------------------------------------
    # Full sample loading
    # ------------------------------------------------------------------

    def _load_sample(self, idx: int) -> dict[str, Any]:
        # Suppress PyMongo fork-safety warning in DataLoader workers.
        # Workers only read from MongoDB; concurrent writes don't occur.
        warnings.filterwarnings(
            "ignore", message="MongoClient opened before fork", category=UserWarning,
        )

        filepath = self._filepaths[idx]
        image = cv2.imread(filepath)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {filepath}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        sample = self._dataset[self._sample_ids[idx]]
        labels_container = sample[self._label_field]

        masks: list[np.ndarray] = []
        boxes_list: list[list[float]] = []
        labels_list: list[int] = []

        if labels_container is not None:
            if self._field_type == "detections":
                self._parse_detections(labels_container, w, h, masks, boxes_list, labels_list)
            else:
                self._parse_polylines(labels_container, w, h, masks, boxes_list, labels_list)

        boxes = (
            np.array(boxes_list, dtype=np.float32)
            if boxes_list
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels = np.array(labels_list, dtype=np.int64)

        return {
            "image": image,
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
        }

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _parse_detections(
        self,
        container,
        img_w: int,
        img_h: int,
        masks: list,
        boxes: list,
        labels: list,
    ) -> None:
        """Parse FiftyOne Detections (with dense .mask) into training format."""
        for det in container.detections:
            if det.label not in self._label_to_idx:
                continue

            bx, by, bw, bh = det.bounding_box
            # Pixel coordinates of the bounding box
            x1 = int(round(bx * img_w))
            y1 = int(round(by * img_h))
            x2 = int(round((bx + bw) * img_w))
            y2 = int(round((by + bh) * img_h))

            # Clamp to image bounds
            x1 = max(0, min(x1, img_w))
            y1 = max(0, min(y1, img_h))
            x2 = max(0, min(x2, img_w))
            y2 = max(0, min(y2, img_h))

            bbox_h, bbox_w = y2 - y1, x2 - x1
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            # Build full-image mask from the bbox-relative mask
            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            if det.mask is not None:
                det_mask = det.mask.astype(np.uint8)
                if det_mask.shape != (bbox_h, bbox_w):
                    det_mask = cv2.resize(
                        det_mask, (bbox_w, bbox_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                full_mask[y1:y2, x1:x2] = det_mask

                # Compute tight bbox from the small bbox-relative mask
                # (avoids scanning the entire full-image array)
                local_ys, local_xs = np.where(det_mask > 0)
                if len(local_xs) == 0:
                    continue
                tight_x1 = x1 + int(local_xs.min())
                tight_y1 = y1 + int(local_ys.min())
                tight_x2 = x1 + int(local_xs.max())
                tight_y2 = y1 + int(local_ys.max())
            else:
                # No mask — fill the bbox as a rectangle
                full_mask[y1:y2, x1:x2] = 1
                tight_x1, tight_y1 = x1, y1
                tight_x2, tight_y2 = x2, y2

            boxes.append([float(tight_x1), float(tight_y1),
                          float(tight_x2), float(tight_y2)])
            masks.append(full_mask)
            labels.append(self._label_to_idx[det.label])

    def _parse_polylines(
        self,
        container,
        img_w: int,
        img_h: int,
        masks: list,
        boxes: list,
        labels: list,
    ) -> None:
        """Parse FiftyOne Polylines into training format (rasterized masks)."""
        for poly in container.polylines:
            if poly.label not in self._label_to_idx:
                continue

            full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
            for ring in poly.points:
                pts = np.array(
                    [[px * img_w, py * img_h] for px, py in ring],
                    dtype=np.int32,
                )
                if len(pts) >= 3:
                    cv2.fillPoly(full_mask, [pts], 1)

            if full_mask.sum() == 0:
                continue

            ys, xs = np.where(full_mask > 0)
            boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max())])
            masks.append(full_mask)
            labels.append(self._label_to_idx[poly.label])
