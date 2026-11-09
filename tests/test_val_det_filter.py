"""Query-based detectors must bound their val-time prediction count.

RF-DETR takes a bare top-`num_select` (200 for seg-large) with NO score
threshold; Mask2Former keeps every query (`max_scores > 0.0`). YOLO has always
bounded itself via NMS + val_conf_thres + val_max_det. Measured on a real validation set:
200 dets/img vs YOLO's 4.5 -> SegmentationMAP's CPU RLE encode ran 42x slower
and dominated ~90% of eval wall-clock, for predictions COCO discards anyway
(mAP caps at max_detection_thresholds=100).
"""
import pytest
import torch

from training_toolkit.models.base import ModelAdapter
from training_toolkit.models.mask2former import Mask2FormerAdapter
from training_toolkit.models.rfdetr import RFDETRAdapter
from training_toolkit.models.yolo import YOLOAdapter


class _Bare(ModelAdapter):
    """Concrete ModelAdapter so the shared helper can be exercised directly."""
    def create_model(self, config): ...
    def create_criterion(self, config): ...
    def format_batch(self, batch): ...
    def extract_predictions(self, outputs): ...
    def compute_loss(self, outputs, targets): ...


def _pred(scores):
    n = len(scores)
    return [{
        "scores": torch.tensor(scores, dtype=torch.float),
        "labels": torch.arange(n),
        "boxes": torch.arange(n * 4, dtype=torch.float).reshape(n, 4),
        "masks": torch.zeros(n, 4, 4, dtype=torch.bool),
    }]


def test_score_threshold_drops_low_confidence():
    out = _Bare()._apply_val_det_filter(_pred([0.9, 0.5, 0.005]), 0.01, 100)
    assert out[0]["scores"].tolist() == pytest.approx([0.9, 0.5])


def test_max_det_caps_and_keeps_the_highest():
    out = _Bare()._apply_val_det_filter(_pred([0.1, 0.9, 0.5, 0.7]), 0.0, 2)
    assert sorted(out[0]["scores"].tolist(), reverse=True) == pytest.approx([0.9, 0.7])


def test_all_fields_stay_in_lockstep():
    """The classic failure: filtering scores but not masks/labels/boxes."""
    out = _Bare()._apply_val_det_filter(_pred([0.9, 0.001, 0.5]), 0.01, 100)[0]
    n = out["scores"].numel()
    assert n == 2
    for k in ("labels", "boxes", "masks"):
        assert out[k].shape[0] == n, f"{k} desynced from scores"
    # kept rows must be the ORIGINAL rows 0 and 2, not 0 and 1
    assert out["labels"].tolist() == [0, 2]


def test_noop_when_nothing_to_drop_returns_same_object():
    p = _pred([0.9, 0.8])
    assert _Bare()._apply_val_det_filter(p, 0.01, 100)[0] is p[0]


def test_empty_predictions_are_safe():
    p = [{"scores": torch.zeros(0), "masks": torch.zeros(0, 4, 4, dtype=torch.bool)}]
    assert _Bare()._apply_val_det_filter(p, 0.01, 100)[0]["scores"].numel() == 0


def test_both_knobs_falsy_is_a_full_noop():
    p = _pred([0.9, 0.001])
    assert _Bare()._apply_val_det_filter(p, 0.0, 0)[0] is p[0]


def test_detr_style_feed_collapses_to_the_cap():
    """200 dets, 5 real -- the measured RF-DETR shape."""
    scores = [0.9, 0.8, 0.7, 0.6, 0.5] + [0.001] * 195
    out = _Bare()._apply_val_det_filter(_pred(scores), 0.01, 100)
    assert out[0]["scores"].numel() == 5


# (superseded by test_official_defaults_shared_by_all_adapters below,
#  which pins all THREE constants across all THREE adapters)


@pytest.mark.parametrize("cls", [RFDETRAdapter, Mask2FormerAdapter, YOLOAdapter])
def test_all_adapters_share_one_source_of_truth_for_the_bounds(cls):
    """YOLO applies these inside NMS rather than via _apply_val_det_filter, but
    it must not carry its own copy of the numbers -- that is how the two silently
    drift apart when the base default is changed."""
    a = cls.__new__(cls)
    assert a.VAL_CONF_THRES_DEFAULT is ModelAdapter.VAL_CONF_THRES_DEFAULT
    assert a.VAL_MAX_DET_DEFAULT is ModelAdapter.VAL_MAX_DET_DEFAULT


def test_yolo_does_not_hardcode_the_bounds():
    """Regression guard: the literals must come from the base class, so that
    changing ModelAdapter.VAL_* actually changes YOLO's behaviour too."""
    import inspect
    from training_toolkit.models import yolo as ymod
    src = inspect.getsource(ymod.YOLOAdapter.__init__)
    assert "0.01" not in src, "YOLO re-hardcodes the conf default"
    assert "= 100" not in src, "YOLO re-hardcodes the max_det default"


# --- NMS -------------------------------------------------------------------

def _pred_masks(masks, scores, labels):
    import numpy as np
    m = torch.tensor(np.stack(masks), dtype=torch.bool)
    n = len(scores)
    # boxes derived from each mask's extent
    boxes = []
    for mm in m:
        ys, xs = torch.where(mm)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] if xs.numel()
                     else [0, 0, 0, 0])
    return [{"masks": m, "scores": torch.tensor(scores, dtype=torch.float),
             "labels": torch.tensor(labels), "boxes": torch.tensor(boxes, dtype=torch.float)}]


def _blob(h, w, r0, r1, c0, c1):
    import numpy as np
    m = np.zeros((h, w), bool); m[r0:r1, c0:c1] = True; return m


def _diag(h, w, c0, thickness=2):
    """A thin diagonal stroke -- a crack. Its bbox is nearly the whole image."""
    import numpy as np
    m = np.zeros((h, w), bool)
    for i in range(min(h, w)):
        m[i, min(w - 1, max(0, i + c0)):min(w, max(0, i + c0) + thickness)] = True
    return m


def test_nms_suppresses_same_class_duplicates():
    dup = _blob(20, 20, 2, 12, 2, 12)
    p = _pred_masks([dup, dup.copy(), _blob(20, 20, 14, 19, 14, 19)],
                    [0.9, 0.8, 0.7], [0, 0, 0])
    out = _Bare()._apply_val_det_filter(p, 0.0, 0, nms_iou=0.7)
    assert out[0]["scores"].numel() == 2, "the identical duplicate must be cut"
    assert out[0]["scores"].tolist() == pytest.approx([0.9, 0.7])


def test_nms_is_class_aware():
    """DETR top-k runs over query x class, so ONE query legitimately appears
    once per class with an IDENTICAL mask. Those must survive."""
    dup = _blob(20, 20, 2, 12, 2, 12)
    p = _pred_masks([dup, dup.copy()], [0.9, 0.8], [0, 1])   # same mask, DIFFERENT class
    out = _Bare()._apply_val_det_filter(p, 0.0, 0, nms_iou=0.7)
    assert out[0]["scores"].numel() == 2, "cross-class hedges must NOT be suppressed"


def test_mask_iou_not_box_iou_for_thin_diagonals():
    """The reason we NMS on masks, not boxes: two disjoint diagonal cracks have
    nearly identical bounding boxes but almost no mask overlap. Box NMS would
    delete a real crack; mask NMS keeps both."""
    a, b = _diag(40, 40, 0), _diag(40, 40, 12)
    p = _pred_masks([a, b], [0.9, 0.8], [0, 0])
    box = p[0]["boxes"]
    # sanity: their BOXES really do overlap heavily
    inter_w = min(box[0][2], box[1][2]) - max(box[0][0], box[1][0])
    assert inter_w > 0.6 * (box[0][2] - box[0][0]), "fixture must have overlapping boxes"
    out = _Bare()._apply_val_det_filter(p, 0.0, 0, nms_iou=0.7)
    assert out[0]["scores"].numel() == 2, "mask-IoU NMS must keep both cracks"


def test_pipeline_order_is_conf_then_nms_then_cap():
    """The cap must come LAST -- otherwise it fills with duplicates NMS would
    have removed, and a real 2nd object gets squeezed out."""
    dup = _blob(20, 20, 2, 12, 2, 12)
    other = _blob(20, 20, 14, 19, 14, 19)
    p = _pred_masks([dup, dup.copy(), other], [0.9, 0.8, 0.7], [0, 0, 0])
    out = _Bare()._apply_val_det_filter(p, 0.0, max_det=2, nms_iou=0.7)
    # cap-then-NMS would have kept {0.9, 0.8} -> NMS -> just {0.9}, losing `other`
    assert out[0]["scores"].tolist() == pytest.approx([0.9, 0.7])


def test_nms_disabled_by_falsy_iou():
    dup = _blob(20, 20, 2, 12, 2, 12)
    p = _pred_masks([dup, dup.copy()], [0.9, 0.8], [0, 0])
    assert _Bare()._apply_val_det_filter(p, 0.0, 0, nms_iou=0.0)[0]["scores"].numel() == 2


@pytest.mark.parametrize("cls", [RFDETRAdapter, Mask2FormerAdapter, YOLOAdapter])
def test_official_defaults_shared_by_all_adapters(cls):
    a = cls.__new__(cls)
    assert a.VAL_CONF_THRES_DEFAULT == 0.001   # Ultralytics val default
    assert a.VAL_MAX_DET_DEFAULT == 100        # the COCO max_dets cap
    assert a.VAL_NMS_IOU_DEFAULT == 0.7        # Ultralytics NMS default
    assert a.VAL_NMS_IOU_DEFAULT is ModelAdapter.VAL_NMS_IOU_DEFAULT
