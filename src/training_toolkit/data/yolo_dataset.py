"""YOLO-format dataset for instance segmentation.

Discovery is driven entirely by the ``data.yaml`` file.  Per-split image
paths are read from the YAML; label paths are derived by replacing the
first occurrence of ``images`` with ``labels`` — matching the convention
used by Ultralytics pipelines.  This handles any directory layout
(``images/{split}/``, ``{split}/images/``, etc.) without explicit layout
detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml

from training_toolkit.configs import AugConfig
from training_toolkit.data.base_dataset import BaseSegmentationDataset

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)

_SPLIT_ALIASES = {"valid": "val"}


class YoloSegmentationDataset(BaseSegmentationDataset):
    """Dataset that reads YOLO-format polygon annotations for instance segmentation.

    Each label ``.txt`` file has one line per instance::

        class_id x1 y1 x2 y2 ... xn yn

    where coordinates are normalized to [0, 1].

    Parameters
    ----------
    root : str | Path
        Root directory containing ``data.yaml`` (or ``data.yml``).
    split : str
        Split name (e.g. ``"train"``, ``"val"``).
    transform : callable, optional
        An Albumentations transform.
    aug_config : AugConfig, optional
        Augmentation configuration for multi-sample transforms.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
        aug_config: AugConfig | None = None,
    ) -> None:
        super().__init__(transform=transform, aug_config=aug_config)
        self.root = Path(root)
        self.split = split

        # Parse data.yaml for class names and split paths
        yaml_path = _find_data_yaml(self.root)
        with open(yaml_path) as fh:
            data = yaml.safe_load(fh)

        raw_names = _parse_class_names(data)
        # Remap to contiguous 0-indexed labels (YOLO labels are usually
        # 0-indexed already, but data.yaml keys can be arbitrary).
        sorted_ids = sorted(raw_names.keys())
        self._label_remap = {old_id: new_id for new_id, old_id in enumerate(sorted_ids)}
        self._class_names = {new_id: raw_names[old_id] for old_id, new_id in self._label_remap.items()}

        # Resolve images and labels directories from data.yaml
        canonical = _SPLIT_ALIASES.get(split, split)
        rel_path = data.get(canonical) or data.get(split)
        if not rel_path:
            raise FileNotFoundError(
                f"Split '{split}' not found in {yaml_path}. "
                f"Expected a key like '{canonical}: images/{canonical}' in the YAML."
            )
        rel_path = str(rel_path)

        self._image_dir = _resolve_dir(rel_path, self.root, yaml_path.parent)
        if self._image_dir is None:
            raise FileNotFoundError(
                f"Images directory for split '{split}' not found. "
                f"Tried '{rel_path}' relative to {self.root}"
            )

        # Derive label dir: replace first "images" with "labels" in the path
        label_rel = rel_path.replace("images", "labels", 1)
        self._labels_dir = _resolve_dir(label_rel, self.root, yaml_path.parent)

        # Build sorted list of (image_path, label_path | None)
        self._samples = _discover_samples(self._image_dir, self._labels_dir)
        logger.info(
            "YOLO dataset [%s]: %d images, %d classes from %s",
            split, len(self._samples), len(self._class_names), self.root,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._class_names)

    @property
    def num_classes(self) -> int:
        return len(self._class_names)

    def __len__(self) -> int:
        return len(self._samples)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _image_path(self, idx: int) -> str:
        return str(self._samples[idx][0])

    def _load_annotations(self, idx: int) -> list:
        from training_toolkit.data.tile_materialize import Instance
        img_path, lbl_path = self._samples[idx]
        w, h = _read_image_size(img_path)
        out: list[Instance] = []
        if lbl_path is None or not Path(lbl_path).exists():
            return out
        with open(lbl_path) as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 7:
                    continue
                raw_id = int(parts[0])
                label = (self._label_remap.get(raw_id, raw_id)
                         if self._label_remap else raw_id)
                coords = list(map(float, parts[1:]))
                if len(coords) % 2 != 0:
                    coords = coords[:-1]
                xs = np.array(coords[0::2]) * w
                ys = np.array(coords[1::2]) * h
                if len(xs) < 3:
                    continue
                ring = np.stack([xs, ys], axis=1).astype(np.float32)
                bbox = (float(xs.min()), float(ys.min()),
                        float(xs.max()), float(ys.max()))
                out.append(Instance(label=label, polygon=[ring], mask=None,
                                    mask_origin=None, bbox=bbox))
        return out

    def _load_tile_metadata(self, idx: int) -> tuple[int, int, np.ndarray]:
        """Return (width, height, boxes) without decoding pixel data.

        Reads image dimensions from the file header (via PIL) and parses
        the YOLO label file for bounding boxes only.
        """
        img_path, lbl_path = self._samples[idx]
        w, h = _read_image_size(img_path)
        boxes = _parse_boxes_only(lbl_path, w, h, self._label_remap)
        return w, h, boxes

    def _load_sample(self, idx: int) -> dict[str, Any]:
        img_path, lbl_path = self._samples[idx]

        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = image.shape[:2]
        masks, boxes, labels = _parse_label_file(lbl_path, w, h, self._label_remap)

        return {
            "image": image,
            "masks": masks,
            "boxes": boxes,
            "labels": labels,
            "image_id": idx,
        }


# ======================================================================
# Module-level helpers
# ======================================================================


def _find_data_yaml(root: Path) -> Path:
    """Find data*.yaml / data*.yml in the dataset root."""
    for pattern in ("data*.yaml", "data*.yml"):
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"No data.yaml / data.yml found in {root}")


def _parse_class_names(data: dict) -> dict[int, str]:
    """Extract class names from parsed data.yaml content."""
    names = data.get("names")
    if names is None:
        raise ValueError("data.yaml must contain a 'names' field")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"Unexpected type for 'names' in data.yaml: {type(names)}")


def _resolve_dir(rel_path: str, root: Path, yaml_parent: Path) -> Path | None:
    """Resolve a relative path — try relative to *root*, then *yaml_parent*."""
    for base in (root, yaml_parent):
        candidate = base / rel_path
        if candidate.is_dir():
            return candidate.resolve()
    return None


def _discover_samples(
    images_dir: Path, labels_dir: Path | None,
) -> list[tuple[Path, Path | None]]:
    """Build sorted list of (image_path, label_path_or_None) pairs."""
    image_paths = _find_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    samples: list[tuple[Path, Path | None]] = []
    for img_path in image_paths:
        lbl_path = _match_label(img_path, labels_dir)
        samples.append((img_path, lbl_path))

    n_labelled = sum(1 for _, lp in samples if lp is not None)
    if n_labelled < len(samples):
        logger.warning(
            "%d / %d images have no matching label file",
            len(samples) - n_labelled, len(samples),
        )
    return samples


def _find_images(directory: Path) -> list[Path]:
    """Return sorted image files under *directory*."""
    out: list[Path] = []
    for p in directory.rglob("*"):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            out.append(p)
    return sorted(out)


def _match_label(img_path: Path, labels_dir: Path | None) -> Path | None:
    """Find the .txt label file corresponding to an image."""
    if labels_dir is None:
        return None
    lbl_path = labels_dir / (img_path.stem + ".txt")
    return lbl_path if lbl_path.exists() else None


def _read_image_size(path: Path) -> tuple[int, int]:
    """Read image (width, height) from file header without decoding pixels."""
    from PIL import Image

    with Image.open(path) as img:
        return img.size  # (width, height)


def _parse_boxes_only(
    label_path: Path | None,
    img_w: int,
    img_h: int,
    label_remap: dict[int, int] | None = None,
) -> np.ndarray:
    """Parse a YOLO label file and return only bounding boxes (no masks).

    Computes tight bboxes from polygon coordinates without rasterizing.
    Much faster than :func:`_parse_label_file` since it skips cv2.fillPoly.

    Returns
    -------
    boxes : (N, 4) float32, xyxy pixel coordinates
    """
    if label_path is None or not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)

    boxes: list[list[float]] = []
    with open(label_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                continue

            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]

            xs = np.array(coords[0::2]) * img_w
            ys = np.array(coords[1::2]) * img_h
            boxes.append([
                float(xs.min()), float(ys.min()),
                float(xs.max()), float(ys.max()),
            ])

    return (
        np.array(boxes, dtype=np.float32)
        if boxes
        else np.zeros((0, 4), dtype=np.float32)
    )


def _parse_label_file(
    label_path: Path | None, img_w: int, img_h: int,
    label_remap: dict[int, int] | None = None,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Parse a YOLO segmentation label file into masks, boxes, labels.

    Returns
    -------
    masks : list of (H, W) uint8 binary masks
    boxes : (N, 4) float32, xyxy format
    labels : (N,) int64
    """
    if label_path is None or not label_path.exists():
        return [], np.zeros((0, 4), dtype=np.float32), np.array([], dtype=np.int64)

    masks: list[np.ndarray] = []
    boxes_list: list[list[float]] = []
    labels_list: list[int] = []

    with open(label_path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:
                # Need at least class_id + 3 polygon points (6 coords)
                continue

            raw_id = int(parts[0])
            class_id = label_remap.get(raw_id, raw_id) if label_remap else raw_id
            coords = list(map(float, parts[1:]))

            # Coords are alternating x, y pairs (normalized 0-1)
            if len(coords) % 2 != 0:
                coords = coords[:-1]  # drop trailing odd value

            xs = np.array(coords[0::2]) * img_w
            ys = np.array(coords[1::2]) * img_h
            polygon = np.stack([xs, ys], axis=1).astype(np.int32)

            # Create binary mask from polygon
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.fillPoly(mask, [polygon], 1)

            if mask.sum() == 0:
                continue

            # Compute tight bbox from mask
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            y1, y2 = np.where(rows)[0][[0, -1]]
            x1, x2 = np.where(cols)[0][[0, -1]]

            masks.append(mask)
            boxes_list.append([float(x1), float(y1), float(x2), float(y2)])
            labels_list.append(class_id)

    boxes = (
        np.array(boxes_list, dtype=np.float32)
        if boxes_list
        else np.zeros((0, 4), dtype=np.float32)
    )
    labels = np.array(labels_list, dtype=np.int64)
    return masks, boxes, labels
