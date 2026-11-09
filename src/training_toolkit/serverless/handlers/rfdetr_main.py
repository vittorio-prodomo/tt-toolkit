"""Nuclio serverless handler for RF-DETR models."""

import base64
import json
from io import BytesIO
from PIL import Image


def init_context(context):
    """Called once on container startup — load model."""
    from model_handler import ModelHandler

    context.logger.info("Loading RF-DETR model handler...")
    handler = ModelHandler("/opt/nuclio/model")
    context.user_data.handler = handler
    context.logger.info("Model loaded successfully.")


def handler(context, event):
    """Called on every inference request from CVAT."""
    context.logger.info("Run RF-DETR detector")
    data = event.body

    image_bytes = base64.b64decode(data["image"])
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    threshold = float(data.get("threshold", 0.5))
    use_sliced = data.get("use_sliced", False)

    if use_sliced:
        slice_width = int(data.get("slice_width", 1024))
        slice_height = int(data.get("slice_height", 1024))
        overlap_ratio = float(data.get("overlap_ratio", 0.25))
        results = context.user_data.handler.predict_sliced(
            image, threshold, slice_width, slice_height, overlap_ratio,
        )
    else:
        results = context.user_data.handler.predict(image, threshold)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
