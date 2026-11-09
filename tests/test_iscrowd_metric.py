import numpy as np
import torch

from training_toolkit.metrics import SegmentationMAP


def _mask(x0, x1):
    m = np.zeros((64, 64), dtype=np.uint8)
    m[8:24, x0:x1] = 1
    return m


def _pred(masks, scores):
    return {"masks": torch.from_numpy(np.stack(masks)).bool(),
            "scores": torch.tensor(scores),
            "labels": torch.zeros(len(masks), dtype=torch.long),
            "boxes": torch.zeros((len(masks), 4))}


def _target(masks, iscrowd=None):
    t = {"masks": torch.from_numpy(np.stack(masks)).to(torch.uint8),
         "labels": torch.zeros(len(masks), dtype=torch.long),
         "boxes": torch.zeros((len(masks), 4))}
    if iscrowd is not None:
        t["iscrowd"] = torch.tensor(iscrowd, dtype=torch.long)
    return t


def test_iscrowd_gt_is_ignored_not_fn():
    """One perfect match + one UNDETECTED crowd GT: with iscrowd=1 the miss
    must not count as FN -> mAP == 1.0. Without the flag it drags mAP down."""
    m_a, m_b = _mask(8, 24), _mask(40, 56)
    with_flag = SegmentationMAP(class_names={0: "crack"})
    with_flag.update([_pred([m_a], [0.9])], [_target([m_a, m_b], [0, 1])])
    r1 = with_flag.format_results(with_flag.compute())

    without = SegmentationMAP(class_names={0: "crack"})
    without.update([_pred([m_a], [0.9])], [_target([m_a, m_b])])
    r0 = without.format_results(without.compute())
    assert r1["map"] > r0["map"]
    assert abs(r1["map"] - 1.0) < 1e-6


def test_coco_dataset_loads_iscrowd(tmp_path):
    import json, cv2
    (tmp_path / "annotations").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "images" / "val" / "a.png"), img)
    coco = {"images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0,
                 "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]],
                 "bbox": [8, 8, 16, 16], "area": 256},
                {"id": 2, "image_id": 1, "category_id": 1, "iscrowd": 1,
                 "segmentation": [[40, 8, 56, 8, 56, 24, 40, 24]],
                 "bbox": [40, 8, 16, 16], "area": 256}],
            "categories": [{"id": 1, "name": "crack"}]}
    (tmp_path / "annotations" / "instances_val.json").write_text(json.dumps(coco))
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    ds = CocoSegmentationDataset(root=str(tmp_path), split="val")
    s = ds._load_sample(0)
    assert list(s["iscrowd"]) == [0, 1]


def test_iscrowd_survives_instance_drop_in_transform():
    """A per-instance array (unlike the per-image `class_mask`) must be
    reindexed in lockstep with masks/labels through _apply_transform's two
    filter points: the pre-transform degenerate-box filter (`keep_idx`) and
    the post-transform mask-survived-the-spatial-transform filter (`keep`).
    Mirrors how `ignore` (size-band SNIP) is already handled in
    base_dataset.py. A stale full-length iscrowd array would silently
    misalign with the surviving mask once a transform drops an instance
    (e.g. cropped fully out of frame, or filtered by min_visibility)."""
    import albumentations as A

    from training_toolkit.data.base_dataset import BaseSegmentationDataset

    class _Dummy(BaseSegmentationDataset):
        def __len__(self):
            return 1

        def _load_sample(self, idx):
            raise NotImplementedError

        @property
        def class_names(self):
            return {0: "crack"}

        @property
        def num_classes(self):
            return 1

    m_keep = np.zeros((100, 100), dtype=np.uint8)
    m_keep[10:30, 10:30] = 1
    m_drop = np.zeros((100, 100), dtype=np.uint8)
    m_drop[70:90, 70:90] = 1
    sample = {
        "image": np.zeros((100, 100, 3), dtype=np.uint8),
        "masks": [m_keep, m_drop],
        "boxes": np.array([[10, 10, 30, 30], [70, 70, 90, 90]], dtype=np.float32),
        "labels": np.array([0, 0], dtype=np.int64),
        "iscrowd": np.array([0, 1], dtype=np.int64),
    }
    # Crop to the top-left quadrant: m_keep survives intact, m_drop crops to
    # an all-zero mask (fully outside [0,50)x[0,50)) and must be dropped.
    transform = A.Compose(
        [A.Crop(x_min=0, y_min=0, x_max=50, y_max=50)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"],
                                  min_area=1.0, min_visibility=0.1),
    )
    out = _Dummy()._apply_transform(sample, transform)
    assert len(out["masks"]) == 1
    # Must track the SURVIVOR's flag (m_keep, iscrowd=0) -- not a stale
    # positional slice of the original length-2 array.
    assert out["iscrowd"].tolist() == [0]


def test_apply_transform_drops_degenerate_recomputed_box():
    """Regression (NaN-divergence root cause under mosaic + shared_pooled):
    _apply_transform recomputes boxes from the POST-transform mask pixels
    with inclusive extremes (xs.max()/ys.max()) and, unlike the pre-transform
    filter above, applied NO validity check on the result. A mask that
    survives a spatial transform as a 1-px-wide/tall sliver (mosaic's
    quadrant squeeze and RandomResizedCrop produce these routinely)
    recomputes to a zero-area box (`y_max == y_min` or `x_max == x_min`),
    which reaches the YOLO seg loss's `single_mask_loss` (divides by box
    area) -> inf -> NaN in `crop_mask`'s backward -> permanently poisons the
    optimizer step.

    Uses a plain identity callable (not albumentations) so the transform
    itself performs no spatial change -- isolating the POST-recompute path.
    The sliver instance is given a NON-degenerate pre-transform box (a mask
    and its declared box need not agree pre-transform; only the pre-filter
    above inspects the given box, never the mask) but a mask that is
    ALREADY 1px tall. This way the degenerate box is produced purely by the
    mask->bbox recompute under test, not by the earlier pre-transform
    filter (which would have dropped it for the wrong reason had its true
    1px box been supplied upfront).
    """
    from training_toolkit.data.base_dataset import BaseSegmentationDataset

    class _Dummy(BaseSegmentationDataset):
        def __len__(self):
            return 1

        def _load_sample(self, idx):
            raise NotImplementedError

        @property
        def class_names(self):
            return {0: "crack"}

        @property
        def num_classes(self):
            return 1

    def identity(image, masks, bboxes, labels):
        return {"image": image, "masks": masks}

    m_normal = np.zeros((100, 100), dtype=np.uint8)
    m_normal[10:30, 10:30] = 1

    m_sliver = np.zeros((100, 100), dtype=np.uint8)
    m_sliver[50, 20:40] = 1  # a single row -> 1px-tall, already pre-transform

    sample = {
        "image": np.zeros((100, 100, 3), dtype=np.uint8),
        "masks": [m_normal, m_sliver],
        # Sliver's DECLARED box is non-degenerate (height 2, [49,51]) so it
        # clears the pre-transform filter untouched; the recompute below
        # derives the degenerate box straight from the mask, not this box.
        "boxes": np.array([[10, 10, 30, 30], [20, 49, 40, 51]], dtype=np.float32),
        "labels": np.array([0, 0], dtype=np.int64),
    }

    out = _Dummy()._apply_transform(sample, identity)

    assert len(out["masks"]) == len(out["labels"]) == 1
    np.testing.assert_array_equal(out["masks"][0], m_normal)
    assert out["boxes"].shape == (1, 4)
    bx = out["boxes"][0]
    assert bx[2] > bx[0] and bx[3] > bx[1]
    np.testing.assert_allclose(bx, [10.0, 10.0, 29.0, 29.0])


def test_apply_transform_all_degenerate_recomputed_boxes_yields_empty():
    """Edge case of the fix above: when EVERY surviving instance recomputes
    to a degenerate box (all slivers), _apply_transform must return an
    empty (0, 4) float32 box array with empty masks/labels -- not crash, and
    not silently keep zero-area boxes."""
    from training_toolkit.data.base_dataset import BaseSegmentationDataset

    class _Dummy(BaseSegmentationDataset):
        def __len__(self):
            return 1

        def _load_sample(self, idx):
            raise NotImplementedError

        @property
        def class_names(self):
            return {0: "crack"}

        @property
        def num_classes(self):
            return 1

    def identity(image, masks, bboxes, labels):
        return {"image": image, "masks": masks}

    m_sliver_a = np.zeros((100, 100), dtype=np.uint8)
    m_sliver_a[50, 20:40] = 1
    m_sliver_b = np.zeros((100, 100), dtype=np.uint8)
    m_sliver_b[70, 10:30] = 1

    sample = {
        "image": np.zeros((100, 100, 3), dtype=np.uint8),
        "masks": [m_sliver_a, m_sliver_b],
        "boxes": np.array([[20, 49, 40, 51], [10, 69, 30, 71]], dtype=np.float32),
        "labels": np.array([0, 0], dtype=np.int64),
    }

    out = _Dummy()._apply_transform(sample, identity)

    assert out["boxes"].shape == (0, 4)
    assert out["boxes"].dtype == np.float32
    assert out["masks"] == []
    assert len(out["labels"]) == 0


def test_apply_transform_degenerate_recompute_aligns_ignore_iscrowd():
    """Regression: the post-recompute degenerate-box filter (lines 270-280)
    must slice ignore/iscrowd by valid_idx in sync with masks/labels, else
    per-instance arrays desync silently. Instance 0 (normal blob) has
    ignore=False, iscrowd=0; instance 1 (1-px sliver) has ignore=True,
    iscrowd=1. After the identity transform, only the normal instance
    survives; ignore and iscrowd must track its values, not stale positional
    slices of the original length-2 arrays."""
    from training_toolkit.data.base_dataset import BaseSegmentationDataset

    class _Dummy(BaseSegmentationDataset):
        def __len__(self):
            return 1

        def _load_sample(self, idx):
            raise NotImplementedError

        @property
        def class_names(self):
            return {0: "crack"}

        @property
        def num_classes(self):
            return 1

    def identity(image, masks, bboxes, labels):
        return {"image": image, "masks": masks}

    m_normal = np.zeros((100, 100), dtype=np.uint8)
    m_normal[10:30, 10:30] = 1

    m_sliver = np.zeros((100, 100), dtype=np.uint8)
    m_sliver[50, 20:40] = 1  # 1px-tall, recomputes to degenerate box

    sample = {
        "image": np.zeros((100, 100, 3), dtype=np.uint8),
        "masks": [m_normal, m_sliver],
        "boxes": np.array([[10, 10, 30, 30], [20, 49, 40, 51]], dtype=np.float32),
        "labels": np.array([0, 0], dtype=np.int64),
        "ignore": np.array([False, True], dtype=bool),
        "iscrowd": np.array([0, 1], dtype=np.int64),
    }

    out = _Dummy()._apply_transform(sample, identity)

    # Only the normal instance (index 0) survives; sliver (index 1) is dropped
    # by the post-recompute degenerate-box filter.
    assert len(out["masks"]) == 1
    assert out["labels"].tolist() == [0]
    assert out["ignore"].tolist() == [False]
    assert out["iscrowd"].tolist() == [0]
    assert len(out["boxes"]) == 1
