"""The offline TiledSegmentationDataset decodes via pyvips ROI; libvips + fork
DataLoader workers deadlock, so the trainer must run those loaders with spawn.
_needs_pyvips_spawn gates that, and a spawn DataLoader over the dataset must
iterate without hanging."""
import json

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from training_toolkit.data import collate_fn
from training_toolkit.data.coco_dataset import CocoSegmentationDataset
from training_toolkit.data.tiled_dataset import TiledSegmentationDataset
from training_toolkit.engine.trainer import _needs_pyvips_spawn, _streaming_worker_init


def _coco(tmp_path, anns):
    img_dir = tmp_path / "images" / "val"
    img_dir.mkdir(parents=True)
    arr = np.random.RandomState(0).randint(0, 255, (64, 64, 3), dtype=np.uint8)
    Image.fromarray(arr).save(img_dir / "a.png")
    ann_dir = tmp_path / "annotations"
    ann_dir.mkdir()
    coco = {
        "images": [{"id": 1, "file_name": "a.png", "width": 64, "height": 64}],
        "annotations": [
            {"id": i + 1, "image_id": 1, "category_id": 0, "iscrowd": 0,
             "bbox": [x, y, w, h], "area": w * h,
             "segmentation": [[x, y, x + w, y, x + w, y + h, x, y + h]]}
            for i, (x, y, w, h) in enumerate(anns)
        ],
        "categories": [{"id": 0, "name": "crack"}],
    }
    (ann_dir / "instances_val.json").write_text(json.dumps(coco))
    return CocoSegmentationDataset(root=tmp_path, split="val", layout="coco")


def _tiled(base):
    return TiledSegmentationDataset(
        base, tile_size=32, overlap=0.0, negative_ratio=0.0,
        min_annotation_overlap=0.05, seed=0,
    )


def test_needs_pyvips_spawn_predicate(tmp_path):
    base = _coco(tmp_path, anns=[(5, 5, 10, 10)])
    assert _needs_pyvips_spawn(_tiled(base)) is True
    assert _needs_pyvips_spawn(base) is False


def test_val_dataloader_spawn_iterates_without_deadlock(tmp_path):
    # one annotation per grid tile (4 tiles) so all are positive and uniform 32x32
    base = _coco(tmp_path, anns=[(5, 5, 10, 10), (37, 5, 10, 10),
                                 (5, 37, 10, 10), (37, 37, 10, 10)])
    ds = _tiled(base)
    loader = DataLoader(
        ds, batch_size=2, num_workers=2, shuffle=False,
        multiprocessing_context="spawn", worker_init_fn=_streaming_worker_init,
        collate_fn=collate_fn,
    )
    it = iter(loader)
    batch = next(it)
    assert batch["images"].shape[0] == 2
    assert batch["images"].shape[1] == 3  # CHW
    del it, loader
