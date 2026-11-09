"""Tests for multi-sample augmentation transforms."""

from __future__ import annotations

import numpy as np
import pytest

from training_toolkit.data.multi_sample import CopyPasteTransform, CutMixTransform, MixUpTransform, MosaicTransform


# ---------------------------------------------------------------------------
# Task-6 tests: make_multi_sample_transforms helper
# ---------------------------------------------------------------------------
from training_toolkit.configs import AugConfig
from training_toolkit.data.multi_sample import make_multi_sample_transforms

def test_make_multi_sample_transforms_respects_probs():
    cfg = AugConfig(imgsz=416, mosaic_p=0.5, cutmix_p=0.0, mixup_p=0.0, copy_paste_p=0.3)
    m = make_multi_sample_transforms(cfg)
    assert m.mosaic is not None and m.cutmix is None and m.mixup is None and m.copy_paste is not None
    assert make_multi_sample_transforms(None).mosaic is None


def _make_sample(h: int, w: int, n_instances: int, seed: int) -> dict:
    """Generate a fake sample with random image and rectangular instance masks.

    Parameters
    ----------
    h, w:
        Image height and width.
    n_instances:
        Number of instance masks to generate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict with keys ``image``, ``masks``, ``boxes``, ``labels``.
    """
    rng = np.random.RandomState(seed)
    image = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)

    masks: list[np.ndarray] = []
    boxes: list[list[float]] = []
    labels: list[int] = []

    for i in range(n_instances):
        mask = np.zeros((h, w), dtype=np.uint8)
        # Random rectangle occupying a meaningful portion
        y0 = rng.randint(0, h // 2)
        x0 = rng.randint(0, w // 2)
        y1 = rng.randint(h // 2, h)
        x1 = rng.randint(w // 2, w)
        mask[y0:y1, x0:x1] = 1
        masks.append(mask)
        boxes.append([float(x0), float(y0), float(x1), float(y1)])
        labels.append(i)

    boxes_arr = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int64)

    return {
        "image": image,
        "masks": masks,
        "boxes": boxes_arr,
        "labels": labels_arr,
    }


class TestMosaicTransform:
    def test_output_is_2x_canvas(self):
        """Output image shape should be (target_size*2, target_size*2, 3)."""
        target_size = 320
        mosaic = MosaicTransform(target_size=target_size)
        primary = _make_sample(400, 500, 2, seed=0)
        extras = [_make_sample(300, 400, 2, seed=i) for i in range(1, 4)]
        result = mosaic(primary, extras)

        expected = (target_size * 2, target_size * 2, 3)
        assert result["image"].shape == expected

    def test_masks_labels_boxes_count_consistent(self):
        """len(masks) == boxes.shape[0] == labels.shape[0]."""
        mosaic = MosaicTransform(target_size=320)
        primary = _make_sample(400, 500, 3, seed=10)
        extras = [_make_sample(300, 400, 2, seed=i) for i in range(11, 14)]
        result = mosaic(primary, extras)

        n_masks = len(result["masks"])
        assert result["boxes"].shape[0] == n_masks
        assert result["labels"].shape[0] == n_masks

    def test_instances_from_all_four_images_present(self):
        """4 images x 2 instances = 8 max; some may be clipped, but > 0."""
        np.random.seed(42)
        mosaic = MosaicTransform(target_size=320)
        primary = _make_sample(400, 500, 2, seed=20)
        extras = [_make_sample(300, 400, 2, seed=i) for i in range(21, 24)]
        result = mosaic(primary, extras)

        n_masks = len(result["masks"])
        assert n_masks > 0, "Expected at least some instances after mosaic"
        # With large-enough masks, we should get close to all 8
        assert n_masks <= 8

    def test_empty_instances_handled(self):
        """0 instances in all 4 images -> empty output."""
        mosaic = MosaicTransform(target_size=320)
        primary = _make_sample(400, 500, 0, seed=30)
        extras = [_make_sample(300, 400, 0, seed=i) for i in range(31, 34)]
        result = mosaic(primary, extras)

        assert len(result["masks"]) == 0
        assert result["boxes"].shape == (0, 4)
        assert result["labels"].shape == (0,)

    def test_output_dtype_correct(self):
        """image uint8, boxes float32, labels int64, masks uint8."""
        mosaic = MosaicTransform(target_size=320)
        primary = _make_sample(400, 500, 2, seed=40)
        extras = [_make_sample(300, 400, 2, seed=i) for i in range(41, 44)]
        result = mosaic(primary, extras)

        assert result["image"].dtype == np.uint8
        assert result["boxes"].dtype == np.float32
        assert result["labels"].dtype == np.int64
        for m in result["masks"]:
            assert m.dtype == np.uint8


class TestCutMixTransform:
    def test_output_same_size_as_primary(self):
        """Result image shape matches primary."""
        np.random.seed(100)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _make_sample(400, 500, 2, seed=0)
        donor = _make_sample(300, 600, 2, seed=1)
        result = cutmix(primary, donor)

        assert result["image"].shape == primary["image"].shape

    def test_masks_labels_boxes_consistent(self):
        """len(masks) == boxes.shape[0] == labels.shape[0]."""
        np.random.seed(101)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _make_sample(400, 500, 3, seed=10)
        donor = _make_sample(300, 600, 3, seed=11)
        result = cutmix(primary, donor)

        n_masks = len(result["masks"])
        assert result["boxes"].shape[0] == n_masks
        assert result["labels"].shape[0] == n_masks

    def test_empty_instances_handled(self):
        """0 instances in both -> empty output."""
        np.random.seed(102)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _make_sample(400, 500, 0, seed=20)
        donor = _make_sample(300, 600, 0, seed=21)
        result = cutmix(primary, donor)

        assert len(result["masks"]) == 0
        assert result["boxes"].shape == (0, 4)
        assert result["labels"].shape == (0,)

    def test_output_dtype_correct(self):
        """image uint8, boxes float32, labels int64."""
        np.random.seed(103)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _make_sample(400, 500, 2, seed=30)
        donor = _make_sample(300, 600, 2, seed=31)
        result = cutmix(primary, donor)

        assert result["image"].dtype == np.uint8
        assert result["boxes"].dtype == np.float32
        assert result["labels"].dtype == np.int64


class TestMixUpTransform:
    def test_output_same_size_as_primary(self):
        """Result image shape matches primary."""
        np.random.seed(200)
        mixup = MixUpTransform(alpha=1.5)
        primary = _make_sample(400, 500, 2, seed=0)
        donor = _make_sample(300, 600, 2, seed=1)
        result = mixup(primary, donor)

        assert result["image"].shape == primary["image"].shape

    def test_all_instances_from_both_images(self):
        """primary has 2, donor has 3 -> result has 5 (all stacked)."""
        np.random.seed(201)
        mixup = MixUpTransform(alpha=1.5)
        primary = _make_sample(400, 500, 2, seed=10)
        donor = _make_sample(300, 600, 3, seed=11)
        result = mixup(primary, donor)

        assert len(result["masks"]) == 5

    def test_masks_labels_boxes_consistent(self):
        """len(masks) == boxes.shape[0] == labels.shape[0]."""
        np.random.seed(202)
        mixup = MixUpTransform(alpha=1.5)
        primary = _make_sample(400, 500, 3, seed=20)
        donor = _make_sample(300, 600, 2, seed=21)
        result = mixup(primary, donor)

        n_masks = len(result["masks"])
        assert result["boxes"].shape[0] == n_masks
        assert result["labels"].shape[0] == n_masks

    def test_blended_image_is_different_from_both(self):
        """Blended image should differ from primary (with different images)."""
        np.random.seed(203)
        mixup = MixUpTransform(alpha=1.5)
        primary = _make_sample(400, 500, 2, seed=30)
        donor = _make_sample(300, 600, 2, seed=31)
        result = mixup(primary, donor)

        # The blended image should not be identical to the primary
        assert not np.array_equal(result["image"], primary["image"])

    def test_output_dtype_correct(self):
        """image uint8, boxes float32, labels int64."""
        np.random.seed(204)
        mixup = MixUpTransform(alpha=1.5)
        primary = _make_sample(400, 500, 2, seed=40)
        donor = _make_sample(300, 600, 2, seed=41)
        result = mixup(primary, donor)

        assert result["image"].dtype == np.uint8
        assert result["boxes"].dtype == np.float32
        assert result["labels"].dtype == np.int64


class TestCopyPasteTransform:
    def test_output_same_size_as_primary(self):
        """Result image shape matches primary."""
        np.random.seed(300)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.5)
        primary = _make_sample(400, 500, 2, seed=0)
        donor = _make_sample(300, 600, 3, seed=1)
        result = cp(primary, donor)

        assert result["image"].shape == primary["image"].shape

    def test_instance_count_increases_or_stays(self):
        """Result has >= primary's instance count."""
        np.random.seed(301)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.0)
        primary = _make_sample(400, 500, 2, seed=10)
        donor = _make_sample(400, 500, 3, seed=11)
        result = cp(primary, donor)

        assert len(result["masks"]) >= len(primary["masks"])

    def test_max_instances_respected(self):
        """max_instances=1, donor has 5 -> at most primary_count + 1."""
        np.random.seed(302)
        cp = CopyPasteTransform(max_instances=1, flip_p=0.0)
        primary = _make_sample(400, 500, 2, seed=20)
        donor = _make_sample(400, 500, 5, seed=21)
        result = cp(primary, donor)

        primary_count = len(primary["masks"])
        assert len(result["masks"]) <= primary_count + 1

    def test_masks_labels_boxes_consistent(self):
        """len(masks) == boxes.shape[0] == labels.shape[0]."""
        np.random.seed(303)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.5)
        primary = _make_sample(400, 500, 3, seed=30)
        donor = _make_sample(300, 600, 3, seed=31)
        result = cp(primary, donor)

        n_masks = len(result["masks"])
        assert result["boxes"].shape[0] == n_masks
        assert result["labels"].shape[0] == n_masks

    def test_empty_donor(self):
        """Donor has 0 instances -> result unchanged (same count as primary)."""
        np.random.seed(304)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.5)
        primary = _make_sample(400, 500, 3, seed=40)
        donor = _make_sample(300, 600, 0, seed=41)
        result = cp(primary, donor)

        assert len(result["masks"]) == len(primary["masks"])


# ---------------------------------------------------------------------- #
# class_mask combination — verifies that masked-loss metadata propagates
# correctly through every multi-sample transform via AND-combination.
# ---------------------------------------------------------------------- #

from training_toolkit.data.multi_sample import _combine_class_masks


def _with_mask(sample: dict, mask: list[float] | None) -> dict:
    """Attach (or remove) a ``class_mask`` field on a sample dict."""
    sample = {**sample}
    if mask is None:
        sample.pop("class_mask", None)
    else:
        sample["class_mask"] = np.asarray(mask, dtype=np.float32)
    return sample


class TestCombineClassMasks:
    def test_returns_none_when_no_masks(self):
        s = _make_sample(64, 64, 1, seed=0)
        assert _combine_class_masks([s, s, s]) is None

    def test_returns_none_when_empty_input(self):
        assert _combine_class_masks([]) is None

    def test_all_ones_returns_all_ones(self):
        s = _with_mask(_make_sample(64, 64, 1, seed=0), [1.0, 1.0, 1.0])
        result = _combine_class_masks([s, s, s])
        assert np.array_equal(result, np.array([1.0, 1.0, 1.0], dtype=np.float32))

    def test_and_of_two_masks(self):
        a = _with_mask(_make_sample(64, 64, 1, seed=0), [1.0, 1.0, 0.0])
        b = _with_mask(_make_sample(64, 64, 1, seed=1), [1.0, 0.0, 1.0])
        result = _combine_class_masks([a, b])
        assert np.array_equal(result, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_missing_mask_treated_as_all_ones(self):
        """A sample without class_mask contributes no constraints to the AND."""
        a = _with_mask(_make_sample(64, 64, 1, seed=0), [1.0, 1.0, 0.0])
        b = _make_sample(64, 64, 1, seed=1)  # no class_mask
        result = _combine_class_masks([a, b])
        assert np.array_equal(result, np.array([1.0, 1.0, 0.0], dtype=np.float32))

    def test_length_mismatch_raises(self):
        a = _with_mask(_make_sample(64, 64, 1, seed=0), [1.0, 1.0, 1.0])
        b = _with_mask(_make_sample(64, 64, 1, seed=1), [1.0, 1.0])
        with pytest.raises(ValueError, match="length mismatch"):
            _combine_class_masks([a, b])

    def test_accepts_list_class_mask(self):
        """Real convention: class_mask is a plain list[num_classes], NOT an ndarray
        (base_dataset._image_class_mask / label_crop pass-through). The ndarray-only
        tests above missed this; mosaic on the pooled path passes raw lists."""
        a = {**_make_sample(64, 64, 1, seed=0), "class_mask": [1, 1, 0]}
        b = {**_make_sample(64, 64, 1, seed=1), "class_mask": [1, 0, 1]}
        result = _combine_class_masks([a, b])
        assert np.array_equal(result, np.array([1.0, 0.0, 0.0], dtype=np.float32))


class TestMosaicClassMask:
    def test_mosaic_ands_class_masks_across_quadrants(self):
        np.random.seed(500)
        mosaic = MosaicTransform(target_size=320)
        primary = _with_mask(_make_sample(400, 500, 1, seed=0), [1.0, 1.0, 0.0])
        e1 = _with_mask(_make_sample(400, 500, 1, seed=1), [1.0, 0.0, 1.0])
        e2 = _with_mask(_make_sample(400, 500, 1, seed=2), [0.0, 1.0, 1.0])
        e3 = _with_mask(_make_sample(400, 500, 1, seed=3), [1.0, 1.0, 1.0])
        result = mosaic(primary, [e1, e2, e3])

        # AND of [1,1,0], [1,0,1], [0,1,1], [1,1,1] = [0,0,0]
        assert "class_mask" in result
        assert np.array_equal(result["class_mask"], np.array([0.0, 0.0, 0.0], dtype=np.float32))

    def test_mosaic_drops_class_mask_when_none_present(self):
        np.random.seed(501)
        mosaic = MosaicTransform(target_size=320)
        primary = _make_sample(400, 500, 1, seed=10)
        extras = [_make_sample(400, 500, 1, seed=i) for i in range(11, 14)]
        result = mosaic(primary, extras)
        assert "class_mask" not in result

    def test_mosaic_handles_partial_masks(self):
        """Only some quadrants have class_mask → AND just over those."""
        np.random.seed(502)
        mosaic = MosaicTransform(target_size=320)
        primary = _with_mask(_make_sample(400, 500, 1, seed=20), [1.0, 0.0, 1.0])
        e1 = _make_sample(400, 500, 1, seed=21)  # no mask
        e2 = _with_mask(_make_sample(400, 500, 1, seed=22), [1.0, 1.0, 0.0])
        e3 = _make_sample(400, 500, 1, seed=23)  # no mask
        result = mosaic(primary, [e1, e2, e3])

        # AND of [1,0,1] and [1,1,0] = [1,0,0]; the two missing ones don't constrain.
        assert "class_mask" in result
        assert np.array_equal(result["class_mask"], np.array([1.0, 0.0, 0.0], dtype=np.float32))


class TestMosaicRegionClassMasks:
    def test_emits_normalized_regions_in_contributor_order(self):
        mosaic = MosaicTransform(target_size=10, center_range=(0.5, 0.5))
        samples = [
            _with_mask(_make_sample(8, 8, 0, seed=i), mask)
            for i, mask in enumerate(
                ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0], [1.0, 1.0, 1.0])
            )
        ]

        result = mosaic(samples[0], samples[1:])

        expected_boxes = np.array(
            [
                [0.0, 0.0, 0.5, 0.5],
                [0.5, 0.0, 1.0, 0.5],
                [0.0, 0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        expected_masks = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        assert result["region_boxes"].dtype == np.float32
        assert result["region_class_masks"].dtype == np.float32
        assert result["region_boxes"].shape == (4, 4)
        assert result["region_class_masks"].shape == (4, 3)
        np.testing.assert_array_equal(result["region_boxes"], expected_boxes)
        np.testing.assert_array_equal(result["region_class_masks"], expected_masks)

    def test_maskless_contributors_use_all_ones(self):
        mosaic = MosaicTransform(target_size=10, center_range=(0.5, 0.5))
        primary = _with_mask(
            _make_sample(8, 8, 0, seed=0), [1.0, 0.0, 1.0]
        )
        extras = [_make_sample(8, 8, 0, seed=i) for i in range(1, 4)]

        result = mosaic(primary, extras)

        expected = np.array(
            [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0],
             [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(result["region_class_masks"], expected)

    def test_omits_region_keys_when_no_contributor_has_class_mask(self):
        mosaic = MosaicTransform(target_size=10, center_range=(0.5, 0.5))
        samples = [_make_sample(8, 8, 0, seed=i) for i in range(4)]

        result = mosaic(samples[0], samples[1:])

        assert "region_boxes" not in result
        assert "region_class_masks" not in result

    def test_skips_degenerate_quadrants_and_their_masks(self):
        mosaic = MosaicTransform(target_size=10, center_range=(0.0, 0.0))
        samples = [
            _with_mask(_make_sample(8, 8, 0, seed=i), mask)
            for i, mask in enumerate(
                ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0], [1.0, 1.0, 0.0])
            )
        ]

        result = mosaic(samples[0], samples[1:])

        np.testing.assert_array_equal(
            result["region_boxes"],
            np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result["region_class_masks"],
            np.array([[1.0, 1.0, 0.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result["class_mask"], np.array([1.0, 1.0, 0.0], dtype=np.float32)
        )


class TestCutMixClassMask:
    def test_cutmix_ands_primary_and_donor(self):
        np.random.seed(600)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _with_mask(_make_sample(400, 500, 2, seed=0), [1.0, 1.0, 0.0])
        donor = _with_mask(_make_sample(400, 500, 2, seed=1), [0.0, 1.0, 1.0])

        # Try a handful of seeds — most yield non-empty cut regions (lam ~ 0.5).
        for trial_seed in range(10):
            np.random.seed(600 + trial_seed)
            result = cutmix(primary, donor)
            if "class_mask" in result:
                assert np.array_equal(
                    result["class_mask"], np.array([0.0, 1.0, 0.0], dtype=np.float32)
                ), f"trial seed {trial_seed}: unexpected mask {result['class_mask']}"
                return
        pytest.fail("CutMix never produced class_mask across 10 trials")

    def test_cutmix_drops_mask_when_neither_has_one(self):
        np.random.seed(601)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _make_sample(400, 500, 2, seed=10)
        donor = _make_sample(400, 500, 2, seed=11)
        result = cutmix(primary, donor)
        assert "class_mask" not in result


class TestMixUpClassMask:
    def test_mixup_ands_primary_and_donor(self):
        np.random.seed(700)
        mixup = MixUpTransform(alpha=1.5)
        primary = _with_mask(_make_sample(400, 500, 2, seed=0), [1.0, 0.0, 1.0])
        donor = _with_mask(_make_sample(400, 500, 2, seed=1), [1.0, 1.0, 0.0])
        result = mixup(primary, donor)

        # AND of [1,0,1] and [1,1,0] = [1,0,0] — both samples blended together,
        # mask combination is invariant to the random blending weight.
        assert "class_mask" in result
        assert np.array_equal(result["class_mask"], np.array([1.0, 0.0, 0.0], dtype=np.float32))


class TestCopyPasteClassMask:
    def test_copy_paste_ands_primary_and_donor(self):
        np.random.seed(800)
        cp = CopyPasteTransform(max_instances=3, flip_p=0.0)
        primary = _with_mask(_make_sample(400, 500, 2, seed=0), [1.0, 0.0, 1.0])
        donor = _with_mask(_make_sample(400, 500, 3, seed=1), [1.0, 1.0, 0.0])
        result = cp(primary, donor)
        assert "class_mask" in result
        assert np.array_equal(result["class_mask"], np.array([1.0, 0.0, 0.0], dtype=np.float32))

    def test_copy_paste_empty_donor_preserves_primary_mask(self):
        """When donor has no instances, the early-return keeps primary's mask."""
        np.random.seed(801)
        cp = CopyPasteTransform(max_instances=3, flip_p=0.0)
        primary = _with_mask(_make_sample(400, 500, 2, seed=10), [1.0, 0.0, 1.0])
        donor = _make_sample(400, 500, 0, seed=11)  # no instances → early return
        result = cp(primary, donor)
        assert "class_mask" in result
        assert np.array_equal(result["class_mask"], np.array([1.0, 0.0, 1.0], dtype=np.float32))

    def test_copy_paste_empty_donor_no_mask_when_primary_lacks_one(self):
        np.random.seed(802)
        cp = CopyPasteTransform(max_instances=3, flip_p=0.0)
        primary = _make_sample(400, 500, 2, seed=20)
        donor = _make_sample(400, 500, 0, seed=21)
        result = cp(primary, donor)
        assert "class_mask" not in result


# ---------------------------------------------------------------------- #
# per-instance ignore propagation — verifies that size-band SNIP markers
# thread through every multi-sample compositor in lockstep with labels,
# and that the offline path (no constituent has "ignore") stays a no-op.
# ---------------------------------------------------------------------- #

def _with_ignore(sample: dict, ignore_vals: list[bool]) -> dict:
    """Attach an ``ignore`` bool array aligned with instance labels."""
    sample = {**sample}
    sample["ignore"] = np.array(ignore_vals, dtype=bool)
    return sample


class TestMosaicIgnore:
    def test_ignore_present_dtype_and_length_aligned(self):
        """Any constituent with ignore -> output has 'ignore', bool, len == len(labels)."""
        np.random.seed(910)
        mosaic = MosaicTransform(target_size=320)
        primary = _with_ignore(_make_sample(400, 500, 2, seed=0), [True, True])
        extras = [_make_sample(400, 500, 2, seed=i) for i in range(1, 4)]
        result = mosaic(primary, extras)
        assert "ignore" in result, "Expected 'ignore' key when a constituent has it"
        assert isinstance(result["ignore"], np.ndarray)
        assert result["ignore"].dtype == bool
        assert len(result["ignore"]) == len(result["labels"])

    def test_all_true_ignore_survives(self):
        """All constituents have ignore=True -> all kept instances are True."""
        np.random.seed(911)
        mosaic = MosaicTransform(target_size=320)
        primary = _with_ignore(_make_sample(400, 500, 2, seed=10), [True, True])
        extras = [_with_ignore(_make_sample(400, 500, 2, seed=i), [True, True])
                  for i in range(11, 14)]
        result = mosaic(primary, extras)
        assert "ignore" in result
        assert all(result["ignore"]), "All kept instances should have ignore=True"

    def test_no_ignore_key_when_no_constituent_has_it(self):
        """Offline no-op: no constituent has 'ignore' -> no 'ignore' key in output."""
        np.random.seed(912)
        mosaic = MosaicTransform(target_size=320)
        primary = _make_sample(400, 500, 2, seed=20)
        extras = [_make_sample(400, 500, 2, seed=i) for i in range(21, 24)]
        result = mosaic(primary, extras)
        assert "ignore" not in result, "offline path must not add 'ignore' key"


class TestCutMixIgnore:
    def test_ignore_propagated_main_path(self):
        """Both primary and donor have ignore -> output has 'ignore' aligned with labels."""
        for trial in range(20):
            np.random.seed(1000 + trial)
            cutmix = CutMixTransform(overlap_thresh=0.5)
            primary = _with_ignore(_make_sample(200, 200, 2, seed=0), [True, False])
            donor = _with_ignore(_make_sample(200, 200, 2, seed=1), [False, True])
            result = cutmix(primary, donor)
            if "ignore" in result:
                assert isinstance(result["ignore"], np.ndarray)
                assert result["ignore"].dtype == bool
                assert len(result["ignore"]) == len(result["labels"])
                return
        pytest.fail("CutMix never produced a non-early-return result across 20 trials")

    def test_early_return_copies_primary_ignore(self):
        """1x1 primary forces zero cut area -> early return preserves primary's ignore exactly."""
        # With h=w=1 and Beta(1,1) ~ Uniform(0,1): cut_w = int(1 * sqrt(1-lam)) = 0
        # for any lam > 0 (probability 1), so x2=x1 -> always early return.
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = {
            "image": np.zeros((1, 1, 3), dtype=np.uint8),
            "masks": [np.ones((1, 1), dtype=np.uint8), np.ones((1, 1), dtype=np.uint8)],
            "boxes": np.array([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
            "labels": np.array([0, 1], dtype=np.int64),
            "ignore": np.array([True, False], dtype=bool),
        }
        donor = _make_sample(50, 50, 2, seed=5)
        result = cutmix(primary, donor)
        # Must copy primary's ignore (early return — donor didn't contribute)
        assert "ignore" in result
        np.testing.assert_array_equal(result["ignore"], [True, False])

    def test_no_ignore_key_when_none_have_it(self):
        """Offline no-op: neither primary nor donor has 'ignore'."""
        np.random.seed(1100)
        cutmix = CutMixTransform(overlap_thresh=0.5)
        primary = _make_sample(200, 200, 2, seed=10)
        donor = _make_sample(200, 200, 2, seed=11)
        result = cutmix(primary, donor)
        assert "ignore" not in result


class TestMixUpIgnore:
    def test_ignore_present_dtype_and_length_aligned(self):
        """Primary has ignore -> output has 'ignore', bool, len == len(labels)."""
        np.random.seed(1200)
        mixup = MixUpTransform(alpha=1.5)
        primary = _with_ignore(_make_sample(300, 300, 2, seed=0), [True, False])
        donor = _make_sample(300, 300, 2, seed=1)
        result = mixup(primary, donor)
        assert "ignore" in result
        assert isinstance(result["ignore"], np.ndarray)
        assert result["ignore"].dtype == bool
        assert len(result["ignore"]) == len(result["labels"])

    def test_all_true_ignore_when_all_constituents_true(self):
        """Both primary and donor all-True ignore -> all kept instances True."""
        np.random.seed(1201)
        mixup = MixUpTransform(alpha=1.5)
        primary = _with_ignore(_make_sample(300, 300, 2, seed=10), [True, True])
        donor = _with_ignore(_make_sample(300, 300, 2, seed=11), [True, True])
        result = mixup(primary, donor)
        assert "ignore" in result
        assert all(result["ignore"])

    def test_no_ignore_key_when_none_have_it(self):
        """Offline no-op: neither primary nor donor has 'ignore'."""
        np.random.seed(1202)
        mixup = MixUpTransform(alpha=1.5)
        primary = _make_sample(300, 300, 2, seed=20)
        donor = _make_sample(300, 300, 2, seed=21)
        result = mixup(primary, donor)
        assert "ignore" not in result


class TestCopyPasteIgnore:
    def test_ignore_present_dtype_and_length_aligned(self):
        """Donor has ignore -> output has 'ignore', bool, len == len(labels)."""
        np.random.seed(1300)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.0)
        primary = _make_sample(300, 300, 2, seed=0)
        donor = _with_ignore(_make_sample(300, 300, 3, seed=1), [True, False, True])
        result = cp(primary, donor)
        assert "ignore" in result
        assert isinstance(result["ignore"], np.ndarray)
        assert result["ignore"].dtype == bool
        assert len(result["ignore"]) == len(result["labels"])

    def test_first_early_return_copies_primary_ignore(self):
        """Empty donor (0 instances) -> first early return copies primary's ignore."""
        np.random.seed(1301)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.0)
        primary = _with_ignore(_make_sample(300, 300, 2, seed=0), [True, False])
        donor = _make_sample(300, 300, 0, seed=1)  # 0 instances -> first early return
        result = cp(primary, donor)
        assert "ignore" in result
        np.testing.assert_array_equal(result["ignore"], [True, False])

    def test_second_early_return_copies_primary_ignore(self):
        """Donor has non-empty mask list but all-zero masks -> second early return preserves primary's ignore."""
        np.random.seed(1302)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.0)
        h, w = 100, 100
        primary = _with_ignore(_make_sample(h, w, 2, seed=0), [True, False])
        # Non-empty mask list so first early-return is NOT triggered,
        # but all-zero mask so mask_resized.max()==0 -> skip -> pasted_masks empty -> second early return
        donor = {
            "image": np.zeros((h, w, 3), dtype=np.uint8),
            "masks": [np.zeros((h, w), dtype=np.uint8)],
            "boxes": np.zeros((1, 4), dtype=np.float32),
            "labels": np.zeros((1,), dtype=np.int64),
            "ignore": np.array([False], dtype=bool),
        }
        result = cp(primary, donor)
        assert "ignore" in result
        np.testing.assert_array_equal(result["ignore"], [True, False])

    def test_no_ignore_key_when_none_have_it(self):
        """Offline no-op: neither primary nor donor has 'ignore'."""
        np.random.seed(1303)
        cp = CopyPasteTransform(max_instances=5, flip_p=0.0)
        primary = _make_sample(300, 300, 2, seed=10)
        donor = _make_sample(300, 300, 3, seed=11)
        result = cp(primary, donor)
        assert "ignore" not in result
