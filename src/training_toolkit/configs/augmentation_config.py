"""Augmentation pipeline configuration."""

import warnings
from dataclasses import dataclass


@dataclass
class AugConfig:
    """Configuration for the augmentation pipeline."""

    imgsz: int | None = None  # None = auto-detect from model variant

    # Multi-sample (mutually exclusive, sum <= 1.0)
    mosaic_p: float = 0.0
    cutmix_p: float = 0.0
    mixup_p: float = 0.0
    mixup_alpha: float = 1.5

    # Close multi-sample augmentations in the last N epochs
    close_mosaic: int = 0  # 0 = disabled; N > 0 = disable mosaic/mixup/cutmix/copy-paste for last N epochs

    # CopyPaste (independent)
    copy_paste_p: float = 0.0
    copy_paste_max_instances: int = 5

    # Single-image spatial
    random_resized_crop: bool = False
    mosaic_crop: bool = True  # apply RandomResizedCrop only after mosaic (when random_resized_crop is False)
    crop_scale_min: float = 0.5
    crop_scale_max: float = 1.0
    horizontal_flip: float = 0.0
    vertical_flip: float = 0.0

    # Affine transforms (applied together when any is non-zero)
    rotate_degrees: float = 0.0  # max rotation angle in degrees (symmetric: ±degrees)
    translate: float = 0.0       # max translation as fraction of image size
    shear: float = 0.0           # max shear angle in degrees
    scale: float = 0.0           # per-image scale jitter as fraction (e.g. 0.5 → 0.5x to 1.5x)
    affine_p: float = 0.5        # probability of applying the affine group

    # Color / photometric (color_jitter and hsv_* are mutually exclusive)
    color_jitter: float = 0.0        # unified knob: sets all axes at once
    hsv_h: float = 0.0              # hue shift limit (0-1, fraction of hue wheel)
    hsv_s: float = 0.0              # saturation shift limit (0-1)
    hsv_v: float = 0.0              # value/brightness shift limit (0-1)
    brightness_contrast_p: float = 0.0
    clahe_p: float = 0.0

    # Noise / blur / quality
    gaussian_blur_p: float = 0.0
    gauss_noise_p: float = 0.0
    downscale_p: float = 0.0

    # Geometric distortion
    perspective_p: float = 0.0
    elastic_transform_p: float = 0.0

    # Lighting / environment
    random_shadow_p: float = 0.0
    random_rain_p: float = 0.0

    # Sharpening
    sharpen_p: float = 0.0

    # Cutout / erasing
    cutout_p: float = 0.0

    # Multi-scale training (batch-level resize each iteration)
    multi_scale: bool = False
    multi_scale_min: float = 0.7
    multi_scale_max: float = 1.4

    # Annotation cleanup
    merge_overlapping_masks: bool = False

    def __post_init__(self) -> None:
        multi_sample_sum = self.mosaic_p + self.cutmix_p + self.mixup_p
        if multi_sample_sum > 1.0:
            raise ValueError(
                f"mosaic_p + cutmix_p + mixup_p must sum to <= 1.0, "
                f"got {multi_sample_sum}"
            )

        has_hsv = self.hsv_h > 0 or self.hsv_s > 0 or self.hsv_v > 0
        if self.color_jitter > 0 and has_hsv:
            raise ValueError(
                "color_jitter and hsv_h/hsv_s/hsv_v are mutually exclusive — "
                "use one or the other"
            )

        if self.mosaic_p > 0 and not self.random_resized_crop and not self.mosaic_crop:
            warnings.warn(
                "mosaic_p > 0 but both random_resized_crop and mosaic_crop "
                "are disabled; mosaic output will not be resized to imgsz",
                UserWarning,
                stacklevel=2,
            )
