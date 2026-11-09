"""COCO-format dataset for instance segmentation."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from pycocotools.coco import COCO

from training_toolkit.configs import AugConfig
from training_toolkit.data.base_dataset import BaseSegmentationDataset

logger = logging.getLogger(__name__)


class CocoSegmentationDataset(BaseSegmentationDataset):
    """Dataset that reads COCO-format annotations for instance segmentation.

    Parameters
    ----------
    root : str | Path
        Root directory. Layout depends on ``layout``.
    split : str
        Split name (e.g. ``"train"``, ``"val"``).
    layout : str
        ``"coco"``: ``root/images/<split>/`` and ``root/annotations/instances_<split>.json``.
        ``"roboflow"``: ``root/<split>/`` contains images and ``root/<split>/_annotations.coco.json``.
        For roboflow, split ``"val"`` is mapped to the ``valid`` folder (Roboflow's export name).
    transform : callable, optional
        An Albumentations transform (or any callable that accepts and returns
        a dict with keys ``image``, ``masks``, ``bboxes``, ``labels``, …).
    aug_config : AugConfig, optional
        Augmentation configuration that drives multi-sample transforms
        (mosaic, cutmix, mixup, copy-paste).
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        layout: str = "coco",
        transform: Callable | None = None,
        aug_config: AugConfig | None = None,
    ) -> None:
        super().__init__(transform=transform, aug_config=aug_config)
        self.root = Path(root)
        self.split = split
        self.layout = layout

        # Roboflow exports use "valid" for the validation folder, not "val"
        if layout == "roboflow" and split == "val":
            split_dir = "valid"
        else:
            split_dir = split

        if layout == "coco":
            ann_file = self.root / "annotations" / f"instances_{split}.json"
            self._image_dir = self.root / "images" / split
        elif layout == "roboflow":
            ann_file = self.root / split_dir / "_annotations.coco.json"
            self._image_dir = self.root / split_dir
        else:
            raise ValueError(
                f"layout must be 'coco' or 'roboflow', got {layout!r}"
            )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.coco = COCO(str(ann_file))
        for line in buf.getvalue().splitlines():
            if line.strip():
                logger.debug(line.strip())
        self.image_ids: list[int] = sorted(self.coco.getImgIds())

        # Build category mapping and remap to contiguous 0-indexed labels.
        # COCO category IDs can be arbitrary (e.g. 1-indexed, non-contiguous),
        # but models expect labels in [0, num_classes).
        raw_cats = {
            cat["id"]: cat["name"] for cat in self.coco.loadCats(self.coco.getCatIds())
        }
        sorted_cat_ids = sorted(raw_cats.keys())
        self._cat_id_to_index = {cat_id: idx for idx, cat_id in enumerate(sorted_cat_ids)}
        self._class_names: dict[int, str] = {
            idx: raw_cats[cat_id] for cat_id, idx in self._cat_id_to_index.items()
        }

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        """Return mapping of category id to category name."""
        return dict(self._class_names)

    @property
    def num_classes(self) -> int:
        """Return total number of categories."""
        return len(self._class_names)

    def __len__(self) -> int:
        return len(self.image_ids)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _image_path(self, idx: int) -> str:
        image_id = self.image_ids[idx]
        return str(self._image_dir / self.coco.loadImgs(image_id)[0]["file_name"])

    def _image_class_mask(self, idx: int):
        """Per-image ``class_mask`` (list[float] of length num_real_classes) from
        the COCO record, or None. Used by the streaming path to attach the mask
        to each tile sample (the non-streaming __getitem__ already reads it)."""
        image_id = self.image_ids[idx]
        return self.coco.loadImgs(image_id)[0].get("class_mask")

    def _image_regime(self, idx: int) -> str:
        """Per-image tiling regime ('indoor_whole' | 'outdoor_tiled') from the
        COCO record; defaults to 'outdoor_tiled' (tile normally) when absent."""
        image_id = self.image_ids[idx]
        return self.coco.loadImgs(image_id)[0].get("regime", "outdoor_tiled")

    def _load_annotations(self, idx: int) -> list:
        from training_toolkit.data.tile_materialize import Instance
        image_id = self.image_ids[idx]
        info = self.coco.loadImgs(image_id)[0]
        H, W = int(info["height"]), int(info["width"])
        out: list[Instance] = []
        for ann in self.coco.loadAnns(self.coco.getAnnIds(imgIds=image_id)):
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            label = self._cat_id_to_index[ann["category_id"]]
            bbox = (float(x), float(y), float(x + bw), float(y + bh))
            seg = ann.get("segmentation")
            polygon = None
            mask = None
            mask_origin = None
            if isinstance(seg, list) and seg:
                rings = []
                for poly in seg:
                    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                    if len(pts) >= 3:
                        rings.append(pts)
                polygon = rings or None
            elif isinstance(seg, dict):  # RLE
                from pycocotools import mask as mask_utils
                rle = seg
                if isinstance(rle.get("counts"), list):
                    rle = mask_utils.frPyObjects(rle, H, W)
                m = mask_utils.decode(rle).astype(np.uint8)
                ys, xs = np.where(m > 0)
                if len(xs) == 0:
                    continue
                x0, y0 = int(xs.min()), int(ys.min())
                mask = m[y0:int(ys.max()) + 1, x0:int(xs.max()) + 1]
                mask_origin = (x0, y0)
            if polygon is None and mask is None:
                continue
            out.append(Instance(label=label, polygon=polygon, mask=mask,
                                mask_origin=mask_origin, bbox=bbox,
                                iscrowd=int(ann.get("iscrowd", 0))))
        return out

    def _load_tile_metadata(self, idx: int) -> tuple[int, int, np.ndarray]:
        """Return (width, height, boxes) from the COCO JSON — no pixel decoding."""
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        w, h = img_info["width"], img_info["height"]

        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes: list[list[float]] = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, x + bw, y + bh])

        boxes_arr = (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        if boxes_arr.size > 0:
            boxes_arr[:, [0, 2]] = boxes_arr[:, [0, 2]].clip(0, w)
            boxes_arr[:, [1, 3]] = boxes_arr[:, [1, 3]].clip(0, h)

        return w, h, boxes_arr

    def _load_sample(self, idx: int) -> dict[str, Any]:
        """Load a single sample without any transforms.

        Returns
        -------
        dict with keys:
            image : np.ndarray, shape (H, W, 3), dtype uint8, RGB
            masks : list[np.ndarray], each (H, W) binary uint8
            boxes : np.ndarray, shape (N, 4), dtype float32, xyxy format
            labels : np.ndarray, shape (N,), dtype int64
            iscrowd : np.ndarray, shape (N,), dtype int64 — COCO ``iscrowd``
                flag per kept annotation (0 when the field is absent from the
                COCO record). Threaded through collate/metrics so
                ``MeanAveragePrecision`` ignores crowd/sliver GT instead of
                counting them as false negatives.
            image_id : int
            class_mask : np.ndarray, shape (num_classes,), dtype float32 — only
                present when the per-image COCO record carries a ``class_mask``
                field (a list of 0/1 of length num_real_classes). Consumed by
                masked-loss-aware criteria; absent samples default to all-ones
                downstream.
        """
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]

        # Read image with OpenCV (BGR) and convert to RGB
        img_path = self._image_dir / img_info["file_name"]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]

        # Load annotations for this image
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        masks: list[np.ndarray] = []
        boxes: list[list[float]] = []
        labels: list[int] = []
        iscrowds: list[int] = []

        for ann in anns:
            # Skip annotations with empty or missing segmentation
            seg = ann.get("segmentation")
            if not seg or (isinstance(seg, list) and any(len(p) < 6 for p in seg)):
                logger.debug(
                    "Skipping annotation %d (image %d): empty segmentation",
                    ann.get("id", -1), image_id,
                )
                continue

            # Convert bbox from COCO [x, y, w, h] to xyxy [x1, y1, x2, y2]
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue  # skip degenerate (zero-area) annotations

            # Decode mask (after bbox check to keep masks/labels in sync)
            mask = self.coco.annToMask(ann).astype(np.uint8)
            masks.append(mask)
            boxes.append([x, y, x + bw, y + bh])

            labels.append(self._cat_id_to_index[ann["category_id"]])
            iscrowds.append(int(ann.get("iscrowd", 0)))

        boxes_arr = (
            np.array(boxes, dtype=np.float32)
            if boxes
            else np.zeros((0, 4), dtype=np.float32)
        )
        # Clip to image bounds (COCO annotations can slightly exceed)
        if boxes_arr.size > 0:
            boxes_arr[:, [0, 2]] = boxes_arr[:, [0, 2]].clip(0, w)
            boxes_arr[:, [1, 3]] = boxes_arr[:, [1, 3]].clip(0, h)
        labels_arr = np.array(labels, dtype=np.int64)

        sample: dict[str, Any] = {
            "image": image,
            "masks": masks,
            "boxes": boxes_arr,
            "labels": labels_arr,
            "iscrowd": np.array(iscrowds, dtype=np.int64),
            "image_id": image_id,
        }
        class_mask = img_info.get("class_mask")
        if class_mask is not None:
            sample["class_mask"] = np.asarray(class_mask, dtype=np.float32)
        return sample
