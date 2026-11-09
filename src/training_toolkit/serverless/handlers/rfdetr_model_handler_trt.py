"""RF-DETR TensorRT model handler for Nuclio serverless deployment.

Self-contained — no imports from training_toolkit.
Runs inside a Nuclio Docker container.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

logger = logging.getLogger(__name__)


class ModelHandler:
    """Load an RF-DETR model via TensorRT and run inference with CVAT-format output."""

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

        # Detect mask support from model outputs (set after engine load)
        self._has_masks: bool = False

        # TensorRT state
        self._context = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_shapes: dict[str, tuple[int, ...]] = {}
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._device_buffers: dict[str, object] = {}
        self._stream = None
        self._dynamic_batch: bool = False
        self._last_batch_size: int | None = None

        # Find model file and build/load engine
        model_path = self._find_model(model_dir)
        self._load_engine(model_path)

        # Check if model has mask output
        if "masks" in self._output_names:
            self._has_masks = True

        logger.info(
            "Loaded RF-DETR TRT model: %s, resolution=%d, classes=%d, masks=%s",
            model_path.name, self._resolution, len(self._class_names), self._has_masks,
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
    # TensorRT engine loading
    # ------------------------------------------------------------------

    def _load_engine(self, model_path: Path) -> None:
        """Build or load a TensorRT engine from the given model file."""
        import cuda  # from cuda-python

        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)

        suffix = model_path.suffix.lower()
        if suffix in (".engine", ".trt"):
            engine = self._deserialize_engine(runtime, model_path)
        elif suffix == ".onnx":
            engine = self._load_from_onnx(runtime, trt_logger, model_path)
        else:
            raise ValueError(
                f"Unsupported model format: {suffix!r} "
                f"(expected .engine, .trt, or .onnx)"
            )

        self._setup_bindings(engine)

    @staticmethod
    def _deserialize_engine(runtime, engine_path: Path):
        """Deserialize a pre-built engine file."""
        logger.info("Loading TensorRT engine from %s", engine_path.name)
        data = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")
        return engine

    def _load_from_onnx(self, runtime, trt_logger, onnx_path: Path):
        """Load from ONNX, using cached engine if available."""
        cache_path = onnx_path.with_suffix(".engine")

        # Try cached engine first
        if cache_path.exists():
            logger.info(
                "Found cached engine %s — attempting to load", cache_path.name,
            )
            data = cache_path.read_bytes()
            engine = runtime.deserialize_cuda_engine(data)
            if engine is not None:
                return engine
            logger.warning(
                "Cached engine failed to load (GPU mismatch?) — rebuilding",
            )

        # Build from ONNX
        return self._build_engine(runtime, trt_logger, onnx_path, cache_path)

    @staticmethod
    def _build_engine(runtime, trt_logger, onnx_path: Path, cache_path: Path):
        """Build a TensorRT engine from an ONNX graph."""
        logger.info(
            "Building TensorRT engine from %s (this may take 30-60s)...",
            onnx_path.name,
        )

        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
        )
        parser = trt.OnnxParser(network, trt_logger)

        onnx_data = onnx_path.read_bytes()
        if not parser.parse(onnx_data):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parse failed:\n{errors}")

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        # Cache the engine (atomic write)
        tmp_path = cache_path.with_suffix(".engine.tmp")
        try:
            tmp_path.write_bytes(serialized)
            os.replace(tmp_path, cache_path)
            logger.info("Cached engine to %s", cache_path.name)
        except OSError:
            logger.warning(
                "Could not cache engine (permissions?) — using in-memory engine",
            )

        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize freshly built engine")

        logger.info("Engine build complete")
        return engine

    def _setup_bindings(self, engine) -> None:
        """Inspect engine bindings, allocate CUDA buffers, create exec context."""
        import cuda

        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        self._output_shapes = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = tuple(engine.get_tensor_shape(name))

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._input_shapes[name] = shape
                if shape[0] == -1:
                    self._dynamic_batch = True
            else:
                self._output_names.append(name)
                self._output_shapes[name] = shape

        self._context = engine.create_execution_context()

        # Allocate buffers for fixed-batch models
        if not self._dynamic_batch:
            self._allocate_buffers(self._input_shapes[self._input_names[0]][0])

        # Create CUDA stream
        _err, self._stream = cuda.cuStreamCreate(0)

    def _allocate_buffers(self, batch_size: int) -> None:
        """Allocate device memory for inputs and outputs."""
        import cuda

        for name, shape in {**self._input_shapes, **self._output_shapes}.items():
            shape_list = list(shape)
            if shape_list[0] == -1:
                shape_list[0] = batch_size
            nbytes = int(np.prod(shape_list)) * 4  # float32 = 4 bytes
            _err, ptr = cuda.cuMemAlloc(nbytes)
            self._device_buffers[name] = ptr

    def _run_trt(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        """Run TensorRT inference on a (B, C, H, W) float32 tensor."""
        import cuda

        batch_size = input_tensor.shape[0]

        # Handle dynamic batch: update shapes and reallocate if needed
        if self._dynamic_batch and batch_size != self._last_batch_size:
            for name in self._input_names:
                shape = list(self._input_shapes[name])
                shape[0] = batch_size
                self._context.set_input_shape(name, shape)
            self._allocate_buffers(batch_size)
            self._last_batch_size = batch_size
        elif not self._dynamic_batch:
            expected = self._input_shapes[self._input_names[0]][0]
            if batch_size != expected:
                raise RuntimeError(
                    f"Expected batch size {expected}, got {batch_size}. "
                    f"Model has fixed batch size."
                )

        # Copy input to device
        for name in self._input_names:
            host_data = np.ascontiguousarray(input_tensor)
            cuda.cuMemcpyHtoD(
                self._device_buffers[name],
                host_data.ctypes.data,
                host_data.nbytes,
            )
            self._context.set_tensor_address(name, int(self._device_buffers[name]))

        # Set output addresses
        for name in self._output_names:
            self._context.set_tensor_address(
                name, int(self._device_buffers[name]),
            )

        # Execute
        self._context.execute_async_v3(stream_handle=int(self._stream))
        cuda.cuStreamSynchronize(self._stream)

        # Copy outputs to host
        results = {}
        for name in self._output_names:
            shape = list(self._output_shapes[name])
            if self._dynamic_batch:
                shape[0] = batch_size
            host_buf = np.empty(shape, dtype=np.float32)
            cuda.cuMemcpyDtoH(
                host_buf.ctypes.data,
                self._device_buffers[name],
                host_buf.nbytes,
            )
            results[name] = host_buf

        return results

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _predict_raw(self, image, threshold: float) -> dict:
        """Run inference and return raw numpy results.

        Returns dict with keys: boxes, scores, labels, masks (numpy arrays).
        Used by both ``predict`` (CVAT conversion) and the SAHI wrapper.
        """
        tensor, original_size = self.preprocess(image)

        raw = self._run_trt(tensor)

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
    def _find_model(model_dir: Path) -> Path:
        """Find the best model file for TensorRT.

        Search order: *.engine > *.trt > *.sim.onnx > *.onnx
        """
        for pattern in ("*.engine", "*.trt", "*.sim.onnx", "*.onnx"):
            files = list(model_dir.glob(pattern))
            if files:
                return files[0]
        raise FileNotFoundError(f"No model files found in {model_dir}")


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
