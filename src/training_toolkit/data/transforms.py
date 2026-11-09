"""Albumentations pipeline builder from AugConfig."""

from __future__ import annotations

import albumentations as A
from albumentations.core.composition import Compose

from training_toolkit.configs import AugConfig

_BBOX_PARAMS = A.BboxParams(
    format="pascal_voc",
    label_fields=["labels"],
    min_area=1.0,
    min_visibility=0.1,
)


def build_transforms(
    config: AugConfig, is_train: bool,
) -> Compose | tuple[Compose, Compose]:
    """Build Albumentations pipeline(s) from configuration.

    For validation, returns a single Compose.

    For training, returns either:
    - A single Compose (when ``mosaic_crop`` is False or ``random_resized_crop``
      is True — both paths use the same pipeline).
    - A tuple ``(normal_transform, mosaic_transform)`` when ``mosaic_crop`` is
      True and ``random_resized_crop`` is False.  The mosaic transform includes
      ``RandomResizedCrop``; the normal one does not.
    """
    if not is_train:
        return A.Compose(
            _build_val_transforms(config),
            bbox_params=_BBOX_PARAMS,
        )

    normal = A.Compose(
        _build_train_transforms(config),
        bbox_params=_BBOX_PARAMS,
    )

    # When random_resized_crop is already on, both paths are identical.
    # When mosaic_crop is off, no special mosaic pipeline needed.
    needs_mosaic_pipeline = (
        config.mosaic_crop
        and not config.random_resized_crop
        and config.mosaic_p > 0
    )
    if not needs_mosaic_pipeline:
        return normal

    mosaic = A.Compose(
        _build_train_transforms(config, force_random_crop=True),
        bbox_params=_BBOX_PARAMS,
    )
    return normal, mosaic


def _build_train_transforms(config: AugConfig, force_random_crop: bool = False) -> list:
    transforms: list[A.BasicTransform] = []

    if config.random_resized_crop or force_random_crop:
        transforms.append(
            A.RandomResizedCrop(
                size=(config.imgsz, config.imgsz),
                scale=(config.crop_scale_min, config.crop_scale_max),
                ratio=(0.75, 1.333),
            )
        )
    else:
        # Always resize to imgsz even without random crop
        transforms.append(A.LongestMaxSize(max_size=config.imgsz))
        transforms.append(
            A.PadIfNeeded(
                min_height=config.imgsz,
                min_width=config.imgsz,
                border_mode=0,
                fill=0,
                fill_mask=0,
            )
        )

    if config.horizontal_flip > 0:
        transforms.append(A.HorizontalFlip(p=config.horizontal_flip))
    if config.vertical_flip > 0:
        transforms.append(A.VerticalFlip(p=config.vertical_flip))

    # Affine group (rotation, translation, shear — applied together)
    has_affine = (
        config.rotate_degrees > 0 or config.translate > 0
        or config.shear > 0 or config.scale > 0
    )
    if has_affine:
        affine_kwargs: dict = {"p": config.affine_p}
        if config.rotate_degrees > 0:
            affine_kwargs["rotate"] = (-config.rotate_degrees, config.rotate_degrees)
        if config.translate > 0:
            affine_kwargs["translate_percent"] = {
                "x": (-config.translate, config.translate),
                "y": (-config.translate, config.translate),
            }
        if config.shear > 0:
            affine_kwargs["shear"] = (-config.shear, config.shear)
        if config.scale > 0:
            affine_kwargs["scale"] = (1.0 - config.scale, 1.0 + config.scale)
            # one factor for both axes — without it x and y are drawn
            # independently and the "zoom" is a per-axis stretch
            affine_kwargs["keep_ratio"] = True
        transforms.append(A.Affine(**affine_kwargs))
    has_hsv = config.hsv_h > 0 or config.hsv_s > 0 or config.hsv_v > 0
    if has_hsv:
        transforms.append(A.HueSaturationValue(
            hue_shift_limit=int(config.hsv_h * 180),  # fraction of wheel → degrees
            sat_shift_limit=int(config.hsv_s * 255),
            val_shift_limit=int(config.hsv_v * 255),
            p=0.8,
        ))
    elif config.color_jitter > 0:
        s = config.color_jitter
        transforms.append(
            A.ColorJitter(brightness=s, contrast=s, saturation=s, hue=s * 0.3, p=0.8)
        )
    if config.brightness_contrast_p > 0:
        transforms.append(
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=config.brightness_contrast_p,
            )
        )
    if config.clahe_p > 0:
        transforms.append(A.CLAHE(clip_limit=4.0, p=config.clahe_p))
    if config.gaussian_blur_p > 0:
        transforms.append(A.GaussianBlur(blur_limit=(3, 5), p=config.gaussian_blur_p))
    if config.gauss_noise_p > 0:
        transforms.append(A.GaussNoise(std_range=(0.01, 0.05), p=config.gauss_noise_p))
    if config.downscale_p > 0:
        transforms.append(A.Downscale(scale_range=(0.7, 0.9), p=config.downscale_p))
    if config.perspective_p > 0:
        transforms.append(A.Perspective(scale=(0.02, 0.05), p=config.perspective_p))
    if config.elastic_transform_p > 0:
        transforms.append(
            A.ElasticTransform(alpha=1.0, sigma=50, p=config.elastic_transform_p)
        )
    if config.random_shadow_p > 0:
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 3),
                shadow_dimension=5,
                p=config.random_shadow_p,
            )
        )
    if config.random_rain_p > 0:
        transforms.append(
            A.RandomRain(
                brightness_coefficient=0.9, drop_width=1, blur_value=3,
                p=config.random_rain_p,
            )
        )
    if config.sharpen_p > 0:
        transforms.append(
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=config.sharpen_p)
        )
    if config.cutout_p > 0:
        transforms.append(
            A.CoarseDropout(
                num_holes_range=(1, 8), hole_height_range=(16, 32),
                hole_width_range=(16, 32), fill=0, p=config.cutout_p,
            )
        )

    return transforms


def _build_val_transforms(config: AugConfig) -> list:
    return [
        A.LongestMaxSize(max_size=config.imgsz),
        A.PadIfNeeded(
            min_height=config.imgsz,
            min_width=config.imgsz,
            border_mode=0,
            fill=0,
            fill_mask=0,
        ),
    ]
