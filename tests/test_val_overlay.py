"""Unit tests for the W&B interactive mask-overlay payload builders."""
import numpy as np
import torch

from training_toolkit.visualization import (
    _DISPLAY_NAMES,
    _instance_masks_to_label_map,
    build_val_overlay_payload,
)


def test_label_map_offsets_by_one_and_reserves_zero_for_background():
    m = np.ones((4, 4), np.uint8)
    lm = _instance_masks_to_label_map(m[None], np.array([0]), 4, 4)
    # class 0 -> value 1 (0 stays background), so a class-0 mask is NOT invisible
    assert lm.dtype == np.int32 and lm.shape == (4, 4)
    assert (lm == 1).all()


def test_label_map_smaller_instance_lands_on_top():
    big = np.ones((4, 4), np.uint8)          # label 0 -> value 1
    small = np.zeros((4, 4), np.uint8)
    small[1:3, 1:3] = 1                        # label 2 -> value 3, smaller => on top
    lm = _instance_masks_to_label_map(np.stack([big, small]), np.array([0, 2]), 4, 4)
    assert lm[0, 0] == 1                       # big only
    assert lm[1, 1] == 3                       # small painted over big
    assert set(np.unique(lm).tolist()) == {1, 3}


def test_label_map_resizes_low_res_masks_to_image_resolution():
    # a 2x2 mask (e.g. val_mask_res-downsampled) resized to a 4x4 image
    m = np.ones((2, 2), np.uint8)
    lm = _instance_masks_to_label_map(m[None], np.array([1]), 4, 4)
    assert lm.shape == (4, 4)
    assert (lm == 2).all()                     # label 1 -> value 2 everywhere


def test_label_map_empty_is_all_background():
    assert (_instance_masks_to_label_map(None, np.array([]), 3, 3) == 0).all()


def test_build_payload_thresholds_predictions_and_builds_layers():
    img = torch.zeros(3, 4, 4)
    target = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        "masks": torch.ones(1, 4, 4),
    }
    pred = {
        "scores": torch.tensor([0.9, 0.1]),    # second is below threshold
        "labels": torch.tensor([2, 1]),
        "boxes": torch.tensor([[0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 1.0, 1.0]]),
        "masks": torch.stack([torch.ones(4, 4), torch.ones(4, 4)]),
    }
    rgb, masks = build_val_overlay_payload(
        img, target, pred,
        class_names={0: "crack", 2: "exposed_rebar"}, score_threshold=0.3,
    )
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    # GT layer: single class-0 instance -> value 1 everywhere
    assert (masks["ground_truth"]["mask_data"] == 1).all()
    # Prediction layer: only the 0.9 pred (class 2 -> value 3) survives the threshold
    assert (masks["prediction"]["mask_data"] == 3).all()
    # class_labels: background + only classes actually present (0 and 2), offset +1
    cl = masks["prediction"]["class_labels"]
    assert cl[0] == "background"
    assert cl[1] == _DISPLAY_NAMES.get("crack", "crack")
    assert cl[3] == _DISPLAY_NAMES.get("exposed_rebar", "exposed_rebar")
    assert 2 not in cl                          # class 1 was thresholded out, absent


def test_build_payload_handles_no_masks_gracefully():
    img = torch.zeros(3, 5, 5)
    target = {"labels": torch.tensor([], dtype=torch.long),
              "boxes": torch.zeros(0, 4), "masks": torch.zeros(0, 5, 5)}
    pred = {"scores": torch.zeros(0), "labels": torch.tensor([], dtype=torch.long),
            "boxes": torch.zeros(0, 4), "masks": torch.zeros(0, 5, 5)}
    rgb, masks = build_val_overlay_payload(img, target, pred)
    assert (masks["ground_truth"]["mask_data"] == 0).all()
    assert (masks["prediction"]["mask_data"] == 0).all()
    assert masks["prediction"]["class_labels"] == {0: "background"}
