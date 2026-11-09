def test_build_tile_index_groups_by_image(mini_coco):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index
    base = CocoSegmentationDataset(root=mini_coco, split="train", layout="coco")
    grouped = build_tile_index(base, tile_sizes=[32], overlap=0.2,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=0)
    assert isinstance(grouped, dict)
    # each image entry is a list of (box, apply_band) tuples
    assert all(isinstance(v, list) and len(v) > 0 for v in grouped.values())
    assert sum(len(v) for v in grouped.values()) >= 1


def test_build_tile_index_is_deterministic(mini_coco):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index
    base = CocoSegmentationDataset(root=mini_coco, split="train", layout="coco")
    a = build_tile_index(base, [32], 0.2, 0.15, 0.05, seed=7)
    b = build_tile_index(base, [32], 0.2, 0.15, 0.05, seed=7)
    assert a == b


def test_build_tile_index_skips_missing_image_files(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    # image id 1 has a real file; image id 2 references a file that is NOT created.
    Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ).save(img_dir / "present.jpg", quality=100)
    ann = {
        "images": [
            {"id": 1, "file_name": "present.jpg", "width": 64, "height": 64},
            {"id": 2, "file_name": "absent.jpg", "width": 64, "height": 64},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [10, 10, 20, 20], "area": 400,
             "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]},
            {"id": 2, "image_id": 2, "category_id": 0, "iscrowd": 0,
             "bbox": [10, 10, 20, 20], "area": 400,
             "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]]},
        ],
        "categories": [{"id": 0, "name": "crack"}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(ann))

    base = CocoSegmentationDataset(root=tmp_path, split="train", layout="coco")
    grouped = build_tile_index(base, tile_sizes=[32], overlap=0.2,
                               negative_ratio=0.0, min_annotation_overlap=0.05, seed=0)
    present_idx = base.image_ids.index(1)
    absent_idx = base.image_ids.index(2)
    assert present_idx in grouped       # the on-disk image is indexed
    assert absent_idx not in grouped    # the missing-file image is skipped, no crash


def _giant_coco(tmp_path):
    """One 200x200 tiled_multiscale image with a giant ann (sqrt-area 60 >> 0.75*32=24)."""
    import json
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    ).save(img_dir / "big.jpg", quality=100)
    ann = {
        "images": [{"id": 1, "file_name": "big.jpg", "width": 200, "height": 200,
                    "regime": "tiled_multiscale"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
                         "bbox": [70, 70, 60, 60], "area": 3600,
                         "segmentation": [[70, 70, 130, 70, 130, 130, 70, 130]]}],
        "categories": [{"id": 0, "name": "crack"}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(ann))
    return tmp_path


def test_whole_image_if_giant_flag_controls_fullframe_window(tmp_path):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    base = CocoSegmentationDataset(root=_giant_coco(tmp_path), split="train", layout="coco")
    on = build_tile_index(base, tile_sizes=[32], overlap=0.2, negative_ratio=0.0,
                          min_annotation_overlap=0.05, seed=0)
    boxes_on = {box for windows in on.values() for (box, _) in windows}
    assert (0, 0, 200, 200) in boxes_on          # default True: giant full-frame window present

    off = build_tile_index(base, tile_sizes=[32], overlap=0.2, negative_ratio=0.0,
                           min_annotation_overlap=0.05, seed=0,
                           whole_image_if_giant=False)
    boxes_off = {box for windows in off.values() for (box, _) in windows}
    assert (0, 0, 200, 200) not in boxes_off     # knob off: no full-frame window
    assert boxes_off                              # grid windows still there


def _mixed_coco(tmp_path, ratio_target_note="grid positives + one empty whole_frame"):
    """img 1: 96x96 tiled_multiscale with one ann (grid positives).
    img 2: 64x64 whole_frame with NO anns (a nullified negative).
    img 3: 64x64 whole_frame WITH one ann (forced regression control)."""
    import json
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    for name, side in (("grid.jpg", 96), ("empty.jpg", 64), ("full.jpg", 64)):
        Image.fromarray(
            np.random.randint(0, 255, (side, side, 3), dtype=np.uint8)
        ).save(img_dir / name, quality=100)
    ann = {
        "images": [
            {"id": 1, "file_name": "grid.jpg", "width": 96, "height": 96,
             "regime": "tiled_multiscale"},
            {"id": 2, "file_name": "empty.jpg", "width": 64, "height": 64,
             "regime": "whole_frame"},
            {"id": 3, "file_name": "full.jpg", "width": 64, "height": 64,
             "regime": "whole_frame"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [8, 8, 16, 16], "area": 256,
             "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]]},
            {"id": 2, "image_id": 3, "category_id": 0, "iscrowd": 0,
             "bbox": [8, 8, 16, 16], "area": 256,
             "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]]},
        ],
        "categories": [{"id": 0, "name": "crack"}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(ann))
    return tmp_path


def test_empty_whole_frame_image_becomes_a_ratio_negative(tmp_path):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    base = CocoSegmentationDataset(root=_mixed_coco(tmp_path), split="train", layout="coco")
    empty_idx = base.image_ids.index(2)
    full_idx = base.image_ids.index(3)

    # ratio 0.99 -> budget (99x grid positives) exceeds the pool for ANY
    # n_pos >= 1, so ALL negatives are kept deterministically (no lottery).
    grouped = build_tile_index(base, tile_sizes=[32], overlap=0.2, negative_ratio=0.99,
                               min_annotation_overlap=0.05, seed=0)
    assert grouped[empty_idx] == [((0, 0, 64, 64), False)]

    # ratio 0 -> the empty whole_frame image contributes nothing...
    grouped0 = build_tile_index(base, tile_sizes=[32], overlap=0.2, negative_ratio=0.0,
                                min_annotation_overlap=0.05, seed=0)
    assert empty_idx not in grouped0
    # ...while the ANNOTATED whole_frame image is still forced (regression)
    assert grouped0[full_idx] == [((0, 0, 64, 64), False)]


def test_empty_whole_frame_needs_grid_positives_for_budget(tmp_path):
    """Documented edge: the negatives budget keys off GRID positives (n_pos).
    An all-whole_frame dataset has n_pos == 0 -> budget 0 -> empties dropped."""
    import json
    import numpy as np
    from PIL import Image
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    for name in ("empty.jpg", "full.jpg"):
        Image.fromarray(
            np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        ).save(img_dir / name, quality=100)
    ann = {
        "images": [
            {"id": 1, "file_name": "empty.jpg", "width": 64, "height": 64,
             "regime": "whole_frame"},
            {"id": 2, "file_name": "full.jpg", "width": 64, "height": 64,
             "regime": "whole_frame"},
        ],
        "annotations": [
            {"id": 1, "image_id": 2, "category_id": 0, "iscrowd": 0,
             "bbox": [8, 8, 16, 16], "area": 256,
             "segmentation": [[8, 8, 24, 8, 24, 24, 8, 24]]},
        ],
        "categories": [{"id": 0, "name": "crack"}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(ann))

    base = CocoSegmentationDataset(root=tmp_path, split="train", layout="coco")
    grouped = build_tile_index(base, tile_sizes=[32], overlap=0.2, negative_ratio=0.5,
                               min_annotation_overlap=0.05, seed=0)
    assert base.image_ids.index(1) not in grouped   # no grid positives -> no negative budget
    assert base.image_ids.index(2) in grouped


def test_build_tile_index_fills_dims_out(tmp_path):
    """dims_out captures declared (w, h) per indexed image — feeds the shared
    pool's byte-budget hints without a second metadata sweep."""
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    base = CocoSegmentationDataset(root=_giant_coco(tmp_path), split="train", layout="coco")
    dims: dict = {}
    grouped = build_tile_index(base, tile_sizes=[32], overlap=0.2, negative_ratio=0.0,
                               min_annotation_overlap=0.05, seed=0, dims_out=dims)
    assert set(dims) == set(grouped)
    idx = base.image_ids.index(1)
    assert dims[idx] == (200, 200)


def _pools_coco(tmp_path):
    """One 200x200 grid image with one small ann -> many negative grid windows."""
    import json
    import numpy as np
    from PIL import Image

    img_dir = tmp_path / "images" / "train"
    img_dir.mkdir(parents=True)
    Image.fromarray(
        np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
    ).save(img_dir / "grid.jpg", quality=100)
    ann = {
        "images": [{"id": 1, "file_name": "grid.jpg", "width": 200, "height": 200,
                    "regime": "tiled_multiscale"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
                         "bbox": [5, 5, 20, 20], "area": 400,
                         "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]]}],
        "categories": [{"id": 0, "name": "crack"}],
    }
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    (ann_dir / "instances_train.json").write_text(json.dumps(ann))
    return tmp_path


def test_pools_out_exposes_presample_lists(tmp_path):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    base = CocoSegmentationDataset(root=_pools_coco(tmp_path), split="train",
                                   layout="coco")
    pools: dict = {}
    grouped = build_tile_index(base, tile_sizes=[32], overlap=0.2,
                               negative_ratio=0.15, min_annotation_overlap=0.05,
                               seed=0, whole_image_if_giant=False,
                               pools_out=pools)
    assert set(pools) == {"positives", "negatives", "forced"}
    n_pos = len(pools["positives"])
    kept = sum(len(v) for v in grouped.values())
    # pre-sample negatives exceed what the 0.15 ratio kept in the returned index
    assert len(pools["negatives"]) > kept - n_pos - len(pools["forced"])
    # every entry is (img_idx, ((x, y, w, h), apply_band))
    img_idx, (box, band) = pools["negatives"][0]
    assert isinstance(img_idx, int) and len(box) == 4 and isinstance(band, bool)


def test_pools_out_default_is_byte_identical(tmp_path):
    from training_toolkit.data.coco_dataset import CocoSegmentationDataset
    from training_toolkit.data.tile_index import build_tile_index

    base = CocoSegmentationDataset(root=_pools_coco(tmp_path), split="train",
                                   layout="coco")
    a = build_tile_index(base, [32], 0.2, 0.15, 0.05, seed=7,
                         whole_image_if_giant=False)
    b = build_tile_index(base, [32], 0.2, 0.15, 0.05, seed=7,
                         whole_image_if_giant=False, pools_out={})
    assert a == b
