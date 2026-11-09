"""Criterion-level tests for per-query region class masking in RF-DETR."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# These criterion features exist only in the rf-detr fork pinned in pyproject.toml
# (uv installs it); skip cleanly on a plain install or with upstream rfdetr.
lwdetr = pytest.importorskip("rfdetr.models.lwdetr")
if not hasattr(lwdetr, "split_ignore_targets"):
    pytest.skip("needs the rf-detr fork (masked-loss / ignore-region criterion)", allow_module_level=True)

from rfdetr.models.lwdetr import SetCriterion
from rfdetr.models.matcher import HungarianMatcher


NUM_REAL_CLASSES = 2
NUM_LOGIT_CLASSES = NUM_REAL_CLASSES + 1


def _make_criterion(
    *,
    region_class_mask: bool,
    masked_loss: bool = True,
    ia_bce: bool = True,
    use_varifocal_loss: bool = False,
    use_position_supervised_loss: bool = False,
) -> SetCriterion:
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        focal_alpha=0.25,
    )
    return SetCriterion(
        num_classes=NUM_LOGIT_CLASSES,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0},
        focal_alpha=0.25,
        losses=["labels"],
        ia_bce_loss=ia_bce,
        masked_loss=masked_loss,
        region_class_mask=region_class_mask,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
    )


def _outputs(
    *,
    logits: torch.Tensor | None = None,
    boxes: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if logits is None:
        logits = torch.tensor(
            [[[0.3, -0.2, 0.1], [-0.4, 0.6, -0.1]]],
            dtype=torch.float32,
        )
    if boxes is None:
        boxes = torch.tensor(
            [[[0.25, 0.5, 0.1, 0.1], [0.75, 0.5, 0.1, 0.1]]],
            dtype=torch.float32,
        )
    return {"pred_logits": logits, "pred_boxes": boxes}


def _empty_target(**metadata: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    return [
        {
            "labels": torch.empty(0, dtype=torch.int64),
            "boxes": torch.empty(0, 4),
            "class_mask": torch.tensor([1.0, 1.0]),
            **metadata,
        }
    ]


def _no_matches():
    empty = torch.empty(0, dtype=torch.int64)
    return [(empty, empty)]


def _label_loss(
    criterion: SetCriterion,
    outputs: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    indices=None,
) -> torch.Tensor:
    if indices is None:
        indices = _no_matches()
    return criterion.loss_labels(
        outputs,
        targets,
        indices,
        num_boxes=1.0,
        log=False,
    )["loss_ce"]


def test_feature_off_is_bitwise_identical_with_region_metadata_present():
    criterion = _make_criterion(region_class_mask=False)
    outputs = _outputs(
        logits=torch.tensor([[[0.3, -0.2, 0.1]]]),
        boxes=torch.tensor([[[0.25, 0.5, 0.1, 0.1]]]),
    )
    plain = [{
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.25, 0.5, 0.1, 0.1]]),
        "class_mask": torch.tensor([1.0, 0.0]),
    }]
    with_regions = [{
        **plain[0],
        "region_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "region_class_masks": torch.tensor([[0.0, 1.0]]),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    plain_loss = _label_loss(criterion, outputs, plain, indices)
    region_loss = _label_loss(criterion, outputs, with_regions, indices)

    assert torch.equal(plain_loss, region_loss)


def test_feature_off_does_not_validate_inert_region_metadata():
    criterion = _make_criterion(region_class_mask=False)
    outputs = _outputs()
    plain = _empty_target()
    malformed = _empty_target(region_boxes=torch.tensor([0.0, 1.0]))

    assert torch.equal(
        _label_loss(criterion, outputs, plain),
        _label_loss(criterion, outputs, malformed),
    )


def test_identical_region_masks_equal_image_fallback_without_matches():
    off = _make_criterion(region_class_mask=False)
    on = _make_criterion(region_class_mask=True)
    outputs = _outputs()
    base = _empty_target()
    base[0]["class_mask"] = torch.tensor([1.0, 0.0])
    regions = _empty_target(
        region_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        region_class_masks=torch.tensor([[1.0, 0.0]]),
    )
    regions[0]["class_mask"] = torch.tensor([1.0, 0.0])

    assert torch.equal(
        _label_loss(off, outputs, base),
        _label_loss(on, outputs, regions),
    )


def test_negative_gradient_is_masked_only_for_its_query_and_class():
    criterion = _make_criterion(region_class_mask=True)
    logits = _outputs()["pred_logits"].clone().requires_grad_(True)
    targets = _empty_target(
        region_boxes=torch.tensor(
            [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
        ),
        region_class_masks=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
    )

    _label_loss(criterion, _outputs(logits=logits), targets).backward()

    assert logits.grad[0, 0, 0] == 0
    assert logits.grad[0, 0, 1].abs() > 0
    assert logits.grad[0, 1, 0].abs() > 0


def test_matched_positive_stays_supervised_after_center_drifts_into_masked_region():
    criterion = _make_criterion(region_class_mask=True)
    logits = torch.tensor([[[0.2, -0.1, 0.0]]], requires_grad=True)
    outputs = _outputs(
        logits=logits,
        boxes=torch.tensor([[[0.75, 0.5, 0.1, 0.1]]]),
    )
    targets = [
        {
            "labels": torch.tensor([0]),
            "boxes": torch.tensor([[0.25, 0.5, 0.1, 0.1]]),
            # The whole-image intersection omits class 0, but the donor region
            # containing the real GT includes it.
            "class_mask": torch.tensor([0.0, 1.0]),
            "region_boxes": torch.tensor(
                [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
            ),
            "region_class_masks": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
        }
    ]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    _label_loss(criterion, outputs, targets, indices).backward()

    assert logits.grad[0, 0, 0].abs() > 0


@pytest.mark.parametrize("missing_key", ["region_boxes", "region_class_masks"])
def test_region_metadata_requires_both_keys(missing_key):
    criterion = _make_criterion(region_class_mask=True)
    metadata = {
        "region_boxes": torch.zeros(1, 4),
        "region_class_masks": torch.ones(1, NUM_REAL_CLASSES),
    }
    metadata.pop(missing_key)

    with pytest.raises(ValueError, match="region_boxes.*region_class_masks"):
        _label_loss(criterion, _outputs(), _empty_target(**metadata))


def test_region_metadata_rejects_row_count_mismatch():
    criterion = _make_criterion(region_class_mask=True)
    targets = _empty_target(
        region_boxes=torch.zeros(2, 4),
        region_class_masks=torch.ones(1, NUM_REAL_CLASSES),
    )

    with pytest.raises(ValueError, match="same number of rows"):
        _label_loss(criterion, _outputs(), targets)


@pytest.mark.parametrize(
    ("region_boxes", "region_class_masks", "message"),
    [
        (torch.zeros(4), torch.ones(1, NUM_REAL_CLASSES), r"region_boxes.*\(R, 4\)"),
        (torch.zeros(1, 4), torch.ones(1, NUM_REAL_CLASSES + 1), r"region_class_masks.*\(R, 2\)"),
    ],
)
def test_region_metadata_rejects_invalid_shapes(
    region_boxes, region_class_masks, message
):
    criterion = _make_criterion(region_class_mask=True)
    targets = _empty_target(
        region_boxes=region_boxes,
        region_class_masks=region_class_masks,
    )

    with pytest.raises(ValueError, match=message):
        _label_loss(criterion, _outputs(), targets)


def test_region_mask_rejects_criterion_and_logit_class_width_mismatch():
    criterion = _make_criterion(region_class_mask=True)
    outputs = _outputs(
        logits=torch.zeros(1, 1, NUM_LOGIT_CLASSES + 1),
        boxes=torch.tensor([[[0.5, 0.5, 0.1, 0.1]]]),
    )
    targets = _empty_target(
        region_boxes=torch.empty(0, 4),
        region_class_masks=torch.empty(0, NUM_REAL_CLASSES + 1),
    )

    with pytest.raises(
        ValueError,
        match=r"src_logits.*4.*criterion num_classes.*3",
    ):
        criterion._build_region_class_mask(
            targets,
            outputs["pred_logits"],
            outputs["pred_boxes"],
        )


def test_empty_paired_region_metadata_uses_image_fallback():
    off = _make_criterion(region_class_mask=False)
    on = _make_criterion(region_class_mask=True)
    outputs = _outputs()
    base = _empty_target()
    base[0]["class_mask"] = torch.tensor([0.0, 1.0])
    empty_regions = _empty_target(
        region_boxes=torch.empty(0, 4),
        region_class_masks=torch.empty(0, NUM_REAL_CLASSES),
    )
    empty_regions[0]["class_mask"] = torch.tensor([0.0, 1.0])

    assert torch.equal(
        _label_loss(off, outputs, base),
        _label_loss(on, outputs, empty_regions),
    )


def test_first_overlapping_region_wins_and_no_object_stays_active():
    criterion = _make_criterion(region_class_mask=True)
    targets = _empty_target(
        region_boxes=torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
        ),
        region_class_masks=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    outputs = _outputs(
        logits=torch.zeros(1, 1, NUM_LOGIT_CLASSES),
        boxes=torch.tensor([[[0.5, 0.5, 0.1, 0.1]]]),
    )

    mask = criterion._build_region_class_mask(
        targets,
        outputs["pred_logits"],
        outputs["pred_boxes"],
    )

    torch.testing.assert_close(mask[0, 0], torch.tensor([0.0, 1.0, 1.0]))


def test_region_boxes_use_half_open_boundaries():
    criterion = _make_criterion(region_class_mask=True)
    targets = _empty_target(
        region_boxes=torch.tensor(
            [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
        ),
        region_class_masks=torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )
    outputs = _outputs(
        logits=torch.zeros(1, 1, NUM_LOGIT_CLASSES),
        boxes=torch.tensor([[[0.5, 0.5, 0.1, 0.1]]]),
    )

    mask = criterion._build_region_class_mask(
        targets,
        outputs["pred_logits"],
        outputs["pred_boxes"],
    )

    torch.testing.assert_close(mask[0, 0], torch.tensor([1.0, 0.0, 1.0]))


def test_forward_composes_region_mask_and_size_ignore_for_main_and_aux_outputs():
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=5.0,
        cost_giou=2.0,
        focal_alpha=0.25,
    )
    criterion = SetCriterion(
        num_classes=NUM_LOGIT_CLASSES,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        focal_alpha=0.25,
        losses=["labels", "boxes"],
        ia_bce_loss=True,
        masked_loss=True,
        region_class_mask=True,
        size_band_ignore=True,
    )
    main_logits = torch.zeros(1, 3, NUM_LOGIT_CLASSES, requires_grad=True)
    aux_logits = torch.zeros(1, 3, NUM_LOGIT_CLASSES, requires_grad=True)
    pred_boxes = torch.tensor(
        [[
            [0.25, 0.5, 0.1, 0.1],
            [0.60, 0.5, 0.1, 0.1],
            [0.85, 0.5, 0.1, 0.1],
        ]]
    )
    outputs = {
        "pred_logits": main_logits,
        "pred_boxes": pred_boxes,
        "aux_outputs": [
            {
                "pred_logits": aux_logits,
                "pred_boxes": pred_boxes.clone(),
            }
        ],
    }
    targets = [
        {
            "labels": torch.tensor([0, 1]),
            "boxes": torch.tensor(
                [[0.25, 0.5, 0.1, 0.1], [0.85, 0.5, 0.2, 0.2]]
            ),
            "ignore": torch.tensor([False, True]),
            "class_mask": torch.tensor([0.0, 1.0]),
            "region_boxes": torch.tensor(
                [
                    [0.0, 0.0, 0.5, 1.0],
                    [0.5, 0.0, 0.75, 1.0],
                    [0.75, 0.0, 1.0, 1.0],
                ]
            ),
            "region_class_masks": torch.tensor(
                [[1.0, 1.0], [0.0, 1.0], [1.0, 1.0]]
            ),
        }
    ]

    losses = criterion(outputs, targets)
    (losses["loss_ce"] + losses["loss_ce_0"]).backward()

    assert all(torch.isfinite(loss).all() for loss in losses.values())
    for logits in (main_logits, aux_logits):
        # The matched class-0 positive remains supervised despite image mask 0.
        assert logits.grad[0, 0, 0].abs() > 0
        # Region masking applies only to class 0 for the foreign-region query.
        assert logits.grad[0, 1, 0] == 0
        assert logits.grad[0, 1, 1].abs() > 0
        # Size-band ignore then removes every negative channel for q2.
        assert torch.equal(logits.grad[0, 2], torch.zeros(NUM_LOGIT_CLASSES))


def test_region_class_mask_requires_masked_loss():
    with pytest.raises(ValueError, match=r"region_class_mask.*masked_loss"):
        _make_criterion(region_class_mask=True, masked_loss=False)


@pytest.mark.parametrize(
    "loss_flag",
    ["use_varifocal_loss", "use_position_supervised_loss"],
)
def test_region_class_mask_rejects_unimplemented_loss_paths(loss_flag):
    with pytest.raises(NotImplementedError, match="region_class_mask"):
        _make_criterion(region_class_mask=True, **{loss_flag: True})


def _build_args(segmentation_head: bool) -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        cls_loss_coef=1.0,
        bbox_loss_coef=1.0,
        giou_loss_coef=1.0,
        segmentation_head=segmentation_head,
        mask_ce_loss_coef=1.0,
        mask_dice_loss_coef=1.0,
        aux_loss=False,
        dec_layers=2,
        two_stage=False,
        num_classes=NUM_REAL_CLASSES,
        focal_alpha=0.25,
        group_detr=1,
        use_varifocal_loss=False,
        use_position_supervised_loss=False,
        ia_bce_loss=True,
        mask_point_sample_ratio=16,
        masked_loss=True,
        region_class_mask=True,
        num_select=10,
    )


@pytest.mark.parametrize("segmentation_head", [False, True])
def test_real_criterion_builder_forwards_region_flag(monkeypatch, segmentation_head):
    matcher = HungarianMatcher(
        cost_class=1.0,
        cost_bbox=1.0,
        cost_giou=1.0,
        focal_alpha=0.25,
    )
    monkeypatch.setattr(lwdetr, "build_matcher", lambda args: matcher)

    criterion, _ = lwdetr.build_criterion_and_postprocessors(
        _build_args(segmentation_head)
    )

    assert criterion.region_class_mask is True


def test_focal_feature_off_is_bitwise_identical_with_region_metadata_present():
    criterion = _make_criterion(region_class_mask=False, ia_bce=False)
    outputs = _outputs(
        logits=torch.tensor([[[0.3, -0.2, 0.1]]]),
        boxes=torch.tensor([[[0.25, 0.5, 0.1, 0.1]]]),
    )
    plain = [{
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.25, 0.5, 0.1, 0.1]]),
        "class_mask": torch.tensor([1.0, 0.0]),
    }]
    with_regions = [{
        **plain[0],
        "region_boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "region_class_masks": torch.tensor([[0.0, 1.0]]),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    plain_loss = _label_loss(criterion, outputs, plain, indices)
    region_loss = _label_loss(criterion, outputs, with_regions, indices)

    assert torch.equal(plain_loss, region_loss)


def test_focal_identical_region_masks_equal_image_fallback_without_matches():
    off = _make_criterion(region_class_mask=False, ia_bce=False)
    on = _make_criterion(region_class_mask=True, ia_bce=False)
    outputs = _outputs()
    base = _empty_target()
    base[0]["class_mask"] = torch.tensor([1.0, 0.0])
    regions = _empty_target(
        region_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        region_class_masks=torch.tensor([[1.0, 0.0]]),
    )
    regions[0]["class_mask"] = torch.tensor([1.0, 0.0])

    assert torch.equal(
        _label_loss(off, outputs, base),
        _label_loss(on, outputs, regions),
    )


def test_focal_negative_gradient_is_masked_only_for_its_query_and_class():
    criterion = _make_criterion(region_class_mask=True, ia_bce=False)
    logits = torch.tensor(
        [[[0.3, -0.2, 0.1], [-0.4, 0.6, -0.1], [0.2, 0.1, -0.3]]],
        requires_grad=True,
    )
    boxes = torch.tensor(
        [[[0.25, 0.5, 0.1, 0.1],
          [0.60, 0.5, 0.1, 0.1],
          [0.90, 0.5, 0.1, 0.1]]]
    )
    targets = _empty_target(
        region_boxes=torch.tensor(
            [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 0.75, 1.0]]
        ),
        region_class_masks=torch.tensor([[0.0, 1.0], [1.0, 1.0]]),
    )
    targets[0]["class_mask"] = torch.tensor([0.0, 1.0])

    _label_loss(criterion, _outputs(logits=logits, boxes=boxes), targets).backward()

    assert logits.grad[0, 0, 0] == 0
    assert logits.grad[0, 0, 1].abs() > 0
    assert logits.grad[0, 0, -1].abs() > 0
    assert logits.grad[0, 1, 0].abs() > 0
    assert logits.grad[0, 2, 0] == 0
    assert logits.grad[0, 2, 1].abs() > 0


def test_focal_matched_positive_stays_supervised_after_center_drifts_into_masked_region():
    criterion = _make_criterion(region_class_mask=True, ia_bce=False)
    logits = torch.tensor([[[0.2, -0.1, 0.0]]], requires_grad=True)
    outputs = _outputs(
        logits=logits,
        boxes=torch.tensor([[[0.75, 0.5, 0.1, 0.1]]]),
    )
    targets = [{
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.25, 0.5, 0.1, 0.1]]),
        "class_mask": torch.tensor([0.0, 1.0]),
        "region_boxes": torch.tensor(
            [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]]
        ),
        "region_class_masks": torch.tensor([[1.0, 1.0], [0.0, 1.0]]),
    }]
    indices = [(torch.tensor([0]), torch.tensor([0]))]

    _label_loss(criterion, outputs, targets, indices).backward()

    assert logits.grad[0, 0, 0].abs() > 0
