"""High-level inference predictor."""

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from training_toolkit.inference.postprocess import postprocess_mask2former, postprocess_rfdetr
from training_toolkit.inference.runtime import InferenceRuntime, OnnxRuntime, TensorRTRuntime

logger = logging.getLogger(__name__)

_RUNTIME_REGISTRY: dict[str, type[InferenceRuntime]] = {
    "ort": OnnxRuntime,
    "trt": TensorRTRuntime,
}

_POSTPROCESSORS = {
    "rfdetr": postprocess_rfdetr,
    "mask2former": postprocess_mask2former,
}


class Predictor:
    """Load an ONNX model and run inference on images.

    Example::

        predictor = Predictor("export/")
        result = predictor.predict("image.jpg")
        # {"boxes": ndarray, "scores": ndarray, "labels": ndarray, "masks": ndarray}
    """

    def __init__(
        self,
        export_dir: str | Path,
        runtime: str = "ort",
        score_threshold: float = 0.5,
        batch_range: tuple[int, int, int] | None = None,
    ) -> None:
        export_dir = Path(export_dir)

        # Load metadata
        meta_path = export_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"No metadata.json found in {export_dir}")
        with open(meta_path) as f:
            self._metadata = json.load(f)

        self._score_threshold = score_threshold
        res = self._metadata["resolution"]
        self._resolution: tuple[int, int] = (res[0], res[1])
        self._backend: str = self._metadata["backend"]
        norm = self._metadata["normalization"]
        self._mean = np.array(norm["mean"], dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(norm["std"], dtype=np.float32).reshape(3, 1, 1)

        self._class_names: dict[int, str] | None = None
        if "class_names" in self._metadata:
            raw = self._metadata["class_names"]
            if isinstance(raw, dict):
                self._class_names = {int(k): v for k, v in raw.items()}
            elif isinstance(raw, list):
                self._class_names = dict(enumerate(raw))

        # Find model file: preference order depends on runtime
        model_path = self._find_model(export_dir, runtime)

        # Postprocessor
        if self._backend not in _POSTPROCESSORS:
            raise ValueError(f"Unknown backend {self._backend!r}")
        self._postprocess = _POSTPROCESSORS[self._backend]

        # Load runtime
        if runtime not in _RUNTIME_REGISTRY:
            raise ValueError(
                f"Unknown runtime {runtime!r}. "
                f"Available: {list(_RUNTIME_REGISTRY)}"
            )
        # Auto-read batch_range from metadata if not explicitly provided
        if batch_range is None and "batch_range" in self._metadata:
            batch_range = tuple(self._metadata["batch_range"])

        runtime_kwargs = {}
        if runtime == "trt" and batch_range is not None:
            runtime_kwargs["batch_range"] = batch_range
        self._runtime = _RUNTIME_REGISTRY[runtime](**runtime_kwargs)
        self._runtime.load(model_path)
        logger.info("Loaded %s model from %s", self._backend, model_path.name)

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def resolution(self) -> tuple[int, int]:
        return self._resolution

    @property
    def class_names(self) -> dict[int, str] | None:
        return self._class_names

    def predict(
        self,
        image: str | Path | np.ndarray,
        score_threshold: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Run inference on a single image.

        Args:
            image: File path (str/Path) or RGB uint8 numpy array (H, W, 3).
            score_threshold: Override the default score threshold.

        Returns:
            Dict with ``boxes``, ``scores``, ``labels``, ``masks`` as numpy arrays.
        """
        img, original_size = self._load_image(image)
        tensor = self._preprocess(img)  # (1, 3, H, W)

        raw = self._runtime.run(tensor)

        threshold = score_threshold if score_threshold is not None else self._score_threshold
        return self._postprocess(
            raw,
            original_size=original_size,
            export_size=self._resolution[0],
            score_threshold=threshold,
        )

    def predict_batch(
        self,
        images: Sequence[str | Path | np.ndarray],
        score_threshold: float | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Run inference on a batch of images.

        Stacks images into a single ``(B, 3, H, W)`` tensor for one runtime
        call when possible.  Falls back to per-image calls if the runtime
        rejects the batch (e.g. fixed-batch ONNX model).

        Args:
            images: List of file paths or RGB uint8 numpy arrays.
            score_threshold: Override the default score threshold.

        Returns:
            List of result dicts, one per image.
        """
        loaded = [self._load_image(img) for img in images]
        tensors = np.concatenate(
            [self._preprocess(img) for img, _ in loaded], axis=0,
        )  # (B, 3, H, W)

        threshold = score_threshold if score_threshold is not None else self._score_threshold

        try:
            raw = self._runtime.run(tensors)
        except Exception:
            # Fixed-batch model — fall back to per-image inference
            return [self.predict(img, score_threshold=score_threshold) for img in images]

        results = []
        for i, (_, original_size) in enumerate(loaded):
            single_raw = {k: v[i:i+1] for k, v in raw.items()}
            results.append(self._postprocess(
                single_raw,
                original_size=original_size,
                export_size=self._resolution[0],
                score_threshold=threshold,
            ))
        return results

    def predict_sliced(
        self,
        image: str | Path | np.ndarray,
        # Slicing
        slice_height: int | None = None,
        slice_width: int | None = None,
        overlap_ratio: float = 0.2,
        # Batching
        batch_size: int = 1,
        # Postprocessing
        postprocess_type: str = "GREEDYNMM",
        postprocess_match_metric: str = "IOS",
        postprocess_match_threshold: float = 0.5,
        postprocess_class_agnostic: bool = False,
        # Full-image
        perform_standard_pred: bool = False,
        auto_slice_resolution: bool = False,
        # Score
        score_threshold: float | None = None,
        # Progress
        progress_bar: bool = True,
        verbose: int = 0,
    ) -> dict[str, np.ndarray]:
        """Run sliced inference on a (potentially large) image via SAHI.

        Slices the image into overlapping tiles, runs inference on each,
        then merges predictions. Requires ``sahi`` to be installed.

        Args:
            image: File path or RGB uint8 numpy array.
            slice_height: Slice height in pixels. Defaults to model resolution.
            slice_width: Slice width in pixels. Defaults to model resolution.
            overlap_ratio: Fractional overlap between slices (applied to both
                height and width).
            batch_size: Number of slices per inference call.
            postprocess_type: Merging strategy — GREEDYNMM, NMS, NMM, LSNMS,
                or NONE (skip merging).
            postprocess_match_metric: IOS or IOU.
            postprocess_match_threshold: Merge IoU/IoS threshold.
            postprocess_class_agnostic: Merge across classes.
            perform_standard_pred: Also run full-image inference.
            auto_slice_resolution: Let SAHI auto-determine slice size.
            score_threshold: Override the default score threshold.
            progress_bar: Show tqdm progress bar.
            verbose: SAHI verbosity (0=silent, 1=slice count, 2=timings).

        Returns:
            Dict with ``boxes``, ``scores``, ``labels``, ``masks`` arrays.
        """
        try:
            from sahi.predict import get_sliced_prediction
        except ImportError:
            raise ImportError(
                "sahi is required for sliced inference. "
                "Install with: pip install training-toolkit[sahi]"
            )

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        # Load image if path
        if not isinstance(image, np.ndarray):
            img, _ = self._load_image(image)
        else:
            img = image

        h_img, w_img = img.shape[:2]

        # Default slice size to model resolution
        if slice_height is None:
            slice_height = self._resolution[0]
        if slice_width is None:
            slice_width = self._resolution[1]

        # Wrap self in SAHI model
        threshold = score_threshold if score_threshold is not None else self._score_threshold
        sahi_model = TrainingToolkitDetectionModel(
            model_path=None,
            confidence_threshold=threshold,
            load_at_init=False,
        )
        sahi_model.set_model(self)

        result = get_sliced_prediction(
            image=img,
            detection_model=sahi_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            perform_standard_pred=perform_standard_pred,
            postprocess_type=postprocess_type,
            postprocess_match_metric=postprocess_match_metric,
            postprocess_match_threshold=postprocess_match_threshold,
            postprocess_class_agnostic=postprocess_class_agnostic,
            auto_slice_resolution=auto_slice_resolution,
            verbose=verbose,
            progress_bar=progress_bar,
            batch_size=batch_size,
        )

        # Convert SAHI PredictionResult → standard dict
        preds = result.object_prediction_list
        if len(preds) == 0:
            return {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=int),
                "masks": np.zeros((0, h_img, w_img), dtype=bool),
            }

        boxes = np.array(
            [p.bbox.to_xyxy() for p in preds], dtype=np.float32,
        )
        scores = np.array(
            [p.score.value for p in preds], dtype=np.float32,
        )
        labels = np.array([p.category.id for p in preds], dtype=int)
        masks = np.stack([p.mask.bool_mask for p in preds])

        return {
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "masks": masks,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(image: str | Path | np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Load an image as RGB uint8 numpy array.  Returns (image, (H, W))."""
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            return image, (h, w)

        path = str(image)
        bgr = cv2.imread(path)
        if bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        return rgb, (h, w)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize, normalize, transpose → (1, 3, H, W) float32."""
        h, w = self._resolution
        resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        tensor = resized.astype(np.float32) / 255.0
        tensor = tensor.transpose(2, 0, 1)  # HWC → CHW
        tensor = (tensor - self._mean) / self._std
        return tensor[np.newaxis]  # add batch dim

    @staticmethod
    def _find_model(export_dir: Path, runtime: str = "ort") -> Path:
        """Find the best model file for the given runtime.

        Search order:
        - ort: *.sim.onnx > *.onnx
        - trt: *.engine > *.trt > *.sim.onnx > *.onnx
        """
        if runtime == "trt":
            for pattern in ("*.engine", "*.trt", "*.sim.onnx", "*.onnx"):
                files = list(export_dir.glob(pattern))
                if files:
                    return files[0]
        else:
            for pattern in ("*.sim.onnx", "*.onnx"):
                files = list(export_dir.glob(pattern))
                if files:
                    return files[0]

        raise FileNotFoundError(
            f"No model files found in {export_dir} for runtime={runtime!r}"
        )
