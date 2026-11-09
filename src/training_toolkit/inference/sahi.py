"""SAHI DetectionModel wrapper for training-toolkit models."""

from __future__ import annotations

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

from training_toolkit.inference.predictor import Predictor


class TrainingToolkitDetectionModel(DetectionModel):
    """SAHI DetectionModel wrapping a training-toolkit Predictor.

    Power-user entry point for sliced inference via SAHI's
    ``get_sliced_prediction()``.

    Example::

        from sahi.predict import get_sliced_prediction
        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        model = TrainingToolkitDetectionModel(
            model_path="export/",
            runtime="trt",
            confidence_threshold=0.3,
        )
        result = get_sliced_prediction(
            image="large_image.jpg",
            detection_model=model,
            slice_height=560,
            slice_width=560,
        )
    """

    def __init__(self, runtime: str = "ort", **kwargs):
        self._runtime = runtime
        super().__init__(**kwargs)

    def check_dependencies(self) -> None:
        pass

    def set_device(self, device=None) -> None:
        pass

    def load_model(self) -> None:
        self.model = Predictor(
            self.model_path,
            runtime=self._runtime,
            score_threshold=self.confidence_threshold,
        )
        if self.category_mapping is None and self.model.class_names is not None:
            self.category_mapping = {
                str(k): v for k, v in self.model.class_names.items()
            }

    def set_model(self, model: Predictor, **kwargs) -> None:
        self.model = model
        if self.category_mapping is None and model.class_names is not None:
            self.category_mapping = {
                str(k): v for k, v in model.class_names.items()
            }

    @property
    def has_mask(self) -> bool:
        return True

    def perform_inference(self, image: np.ndarray | list[np.ndarray]) -> None:
        if isinstance(image, list):
            results = self.model.predict_batch(
                image, score_threshold=self.confidence_threshold,
            )
            self._original_predictions = results
            self._original_shape = [(img.shape[0], img.shape[1]) for img in image]
        else:
            result = self.model.predict(
                image, score_threshold=self.confidence_threshold,
            )
            self._original_predictions = [result]
            self._original_shape = [(image.shape[0], image.shape[1])]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = None,
        full_shape_list: list[list[int]] | None = None,
    ) -> None:
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]
        if full_shape_list is None:
            full_shape_list = [None] * len(self._original_predictions)

        self._object_prediction_list_per_image = []

        for image_ind, result in enumerate(self._original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = full_shape_list[image_ind]

            object_prediction_list = []
            boxes = result["boxes"]
            scores = result["scores"]
            labels = result["labels"]
            masks = result["masks"]

            for i in range(len(scores)):
                bbox = boxes[i].tolist()
                bool_mask = masks[i]

                segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                if len(segmentation) == 0:
                    continue

                category_id = int(labels[i])
                category_name = (
                    self.category_mapping.get(str(category_id), str(category_id))
                    if self.category_mapping
                    else str(category_id)
                )

                pred = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=float(scores[i]),
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )

                # Attach pixel-perfect bool mask cache
                if pred.mask is not None:
                    pred.mask._bool_mask = bool_mask

                object_prediction_list.append(pred)

            self._object_prediction_list_per_image.append(object_prediction_list)
