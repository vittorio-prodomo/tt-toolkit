"""RF-DETR ONNX Runtime model handler for Nuclio serverless deployment.

Self-contained — no imports from training_toolkit.
Runs inside a Nuclio Docker container.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

logger = logging.getLogger(__name__)


class ModelHandler:
    """Load an RF-DETR ONNX model and run inference with CVAT-format output."""

    def __init__(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)

        # Load metadata
        meta_path = model_dir / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        res = meta["resolution"]
        self._resolution: int = res[0] if isinstance(res, list) else int(res)

        norm = meta["normalization"]
        self._mean = np.array(norm["mean"], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(norm["std"], dtype=np.float32).reshape(3, 1, 1)

        raw_names = meta.get("class_names", [])
        if isinstance(raw_names, dict):
            self._class_names: list[str] = [
                raw_names[k] for k in sorted(raw_names, key=int)
            ]
        else:
            self._class_names = list(raw_names)

        # Detect mask support from model outputs
        self._has_masks: bool = False

        # Find ONNX file
        onnx_path = self._find_onnx(model_dir)

        # Load ONNX Runtime session
        self._session = ort.InferenceSession(
            str(onnx_path),
            providers=ort.get_available_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

        # Check if model has mask output
        if "masks" in self._output_names:
            self._has_masks = True

        logger.info(
            "Loaded RF-DETR ORT model: %s, resolution=%d, classes=%d, masks=%s",
            onnx_path.name, self._resolution, len(self._class_names), self._has_masks,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image) -> tuple[np.ndarray, tuple[int, int]]:
        """Resize, normalize, NCHW. Returns (tensor, original_size)."""
        img = np.array(image)
        h_orig, w_orig = img.shape[:2]

        resized = cv2.resize(
            img, (self._resolution, self._resolution),
            interpolation=cv2.INTER_LINEAR,
        )
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC -> CHW
        tensor = (tensor - self._mean) / self._std
        tensor = tensor[np.newaxis]  # add batch dim
        return tensor, (h_orig, w_orig)

    def predict(self, image, threshold: float) -> list[dict]:
        """Full inference pipeline -> CVAT format."""
        result = self._predict_raw(image, threshold)
        return self._to_cvat(result)

    def predict_sliced(
        self, image, threshold: float,
        slice_width: int, slice_height: int, overlap_ratio: float,
    ) -> list[dict]:
        """Sliced inference via SAHI -> CVAT format."""
        from sahi.predict import get_sliced_prediction

        sahi_model = _SahiRFDETRModel(
            handler=self,
            model_path="",
            confidence_threshold=threshold,
        )

        result = get_sliced_prediction(
            image=np.array(image),
            detection_model=sahi_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NONE",
        )

        detections = []
        for pred in result.object_prediction_list:
            det = {
                "confidence": str(pred.score.value),
                "label": pred.category.name,
            }
            if self._has_masks and pred.mask is not None:
                bool_mask = (
                    pred.mask._bool_mask
                    if hasattr(pred.mask, "_bool_mask")
                    else pred.mask.bool_mask
                )
                bbox = pred.bbox.to_xyxy()
                xtl = int(bbox[0])
                ytl = int(bbox[1])
                xbr = int(bbox[2]) - 1
                ybr = int(bbox[3]) - 1
                det["type"] = "mask"
                det["mask"] = self._encode_mask(bool_mask, [xtl, ytl, xbr, ybr])
            else:
                bbox = pred.bbox.to_xyxy()
                det["type"] = "rectangle"
                det["points"] = [float(x) for x in bbox]
            detections.append(det)
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_raw(self, image, threshold: float) -> dict:
        """Run inference and return raw numpy results.

        Returns dict with keys: boxes, scores, labels, masks (numpy arrays).
        Used by both ``predict`` (CVAT conversion) and the SAHI wrapper.
        """
        tensor, original_size = self.preprocess(image)

        outputs = self._session.run(None, {self._input_name: tensor})
        raw = dict(zip(self._output_names, outputs))

        # Postprocess
        dets = raw["dets"][0]      # (N, 5) — first 4 cols xyxy, 5th score
        labels = raw["labels"][0]  # (N,)

        boxes = dets[:, :4]
        scores = dets[:, 4]

        keep_idx = np.where(scores > threshold)[0]
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

        # Scale boxes from export resolution to original image size
        h_orig, w_orig = original_size
        scale_x = w_orig / self._resolution
        scale_y = h_orig / self._resolution
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # Masks
        if "masks" in raw:
            all_masks = raw["masks"][0]        # (num_queries, H/4, W/4)
            kept_masks = all_masks[keep_idx]   # (K, H/4, W/4)
            if len(kept_masks) > 0:
                masks = np.stack([
                    cv2.resize(m, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR) > 0.0
                    for m in kept_masks
                ])
            else:
                masks = np.zeros((0, h_orig, w_orig), dtype=bool)
        else:
            masks = np.zeros((0, h_orig, w_orig), dtype=bool)

        return {
            "boxes": boxes.astype(np.float32),
            "scores": scores.astype(np.float32),
            "labels": labels.astype(np.int64),
            "masks": masks,
        }

    def _to_cvat(self, result: dict) -> list[dict]:
        """Convert raw result dict to CVAT detection format."""
        detections = []
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        masks = result["masks"]

        for i in range(len(scores)):
            label_idx = int(labels[i])
            class_name = (
                self._class_names[label_idx]
                if label_idx < len(self._class_names)
                else str(label_idx)
            )

            det: dict = {
                "confidence": str(float(scores[i])),
                "label": class_name,
            }

            if self._has_masks and len(masks) > i:
                bool_mask = masks[i]
                box = boxes[i]
                h_img, w_img = bool_mask.shape[:2]
                xtl = max(0, int(np.floor(box[0])))
                ytl = max(0, int(np.floor(box[1])))
                xbr = min(w_img - 1, int(np.ceil(box[2])) - 1)
                ybr = min(h_img - 1, int(np.ceil(box[3])) - 1)
                det["type"] = "mask"
                det["mask"] = self._encode_mask(bool_mask, [xtl, ytl, xbr, ybr])
            else:
                det["type"] = "rectangle"
                det["points"] = [float(x) for x in boxes[i]]

            detections.append(det)

        return detections

    @staticmethod
    def _encode_mask(bitmap: np.ndarray, bbox: list[int]) -> list[int]:
        """Encode binary mask to CVAT RLE format.

        Args:
            bitmap: (H, W) boolean/uint8 array.
            bbox: [xtl, ytl, xbr, ybr] inclusive coordinates.

        Returns:
            Run-length encoded values + [xtl, ytl, xbr, ybr].
        """
        xtl, ytl, xbr, ybr = bbox
        crop = bitmap[ytl:ybr + 1, xtl:xbr + 1].astype(np.uint8).ravel()
        # Run-length encode: alternating bg/fg runs starting from background
        changes = np.diff(crop, prepend=0, append=0)
        run_starts = np.where(changes != 0)[0]
        runs = np.diff(run_starts).tolist()
        if len(crop) > 0 and crop[0] == 1:
            runs = [0] + runs  # prepend 0-length background run
        runs.extend([xtl, ytl, xbr, ybr])
        return runs

    @staticmethod
    def _find_onnx(model_dir: Path) -> Path:
        """Find the best ONNX file in model_dir."""
        for pattern in ("*.sim.onnx", "*.onnx"):
            files = list(model_dir.glob(pattern))
            if files:
                return files[0]
        raise FileNotFoundError(f"No ONNX model found in {model_dir}")


# ------------------------------------------------------------------
# SAHI integration
# ------------------------------------------------------------------

class _SahiRFDETRModel(DetectionModel):
    """SAHI DetectionModel subclass wrapping the ModelHandler.

    Self-contained — does NOT import from training_toolkit.
    """

    def __init__(self, handler: ModelHandler, **kwargs):
        self._handler = handler
        super().__init__(**kwargs)

    def check_dependencies(self) -> None:
        pass

    def set_device(self, device=None) -> None:
        pass

    def load_model(self) -> None:
        self.model = self._handler
        if self.category_mapping is None and self._handler._class_names:
            self.category_mapping = {
                str(i): n for i, n in enumerate(self._handler._class_names)
            }

    @property
    def has_mask(self) -> bool:
        return self._handler._has_masks

    def perform_inference(self, image) -> None:
        from PIL import Image as PILImage

        pil_img = (
            PILImage.fromarray(image) if isinstance(image, np.ndarray) else image
        )
        result = self._handler._predict_raw(pil_img, self.confidence_threshold)
        self._original_predictions = [result]
        self._original_shape = [
            (image.shape[0], image.shape[1])
            if isinstance(image, np.ndarray)
            else (image.size[1], image.size[0])
        ]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ) -> None:
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]
        if full_shape_list is None:
            full_shape_list = [None]

        self._object_prediction_list_per_image = []
        for img_idx, result in enumerate(self._original_predictions):
            predictions = []
            for i in range(len(result["scores"])):
                bbox = result["boxes"][i].tolist()
                category_id = int(result["labels"][i])
                category_name = (
                    self.category_mapping.get(str(category_id), str(category_id))
                    if self.category_mapping
                    else str(category_id)
                )

                segmentation = None
                if self._handler._has_masks and len(result["masks"]) > i:
                    segmentation = get_coco_segmentation_from_bool_mask(
                        result["masks"][i],
                    )
                    if len(segmentation) == 0:
                        continue

                pred = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=float(result["scores"][i]),
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount_list[img_idx],
                    full_shape=full_shape_list[img_idx],
                )
                if (
                    self._handler._has_masks
                    and pred.mask is not None
                    and len(result["masks"]) > i
                ):
                    pred.mask._bool_mask = result["masks"][i]
                predictions.append(pred)
            self._object_prediction_list_per_image.append(predictions)
