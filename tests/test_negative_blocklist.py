import json
import numpy as np

from training_toolkit.data.tile_index import (
    build_tile_index, load_negative_blocklist,
)
from training_toolkit.data.tile_materialize import Instance


def _thin_diag_instance():
    ring = np.array([[0, 0], [4, 0], [404, 400], [400, 400]], dtype=np.float32)
    return Instance(label=0, polygon=[ring], mask=None, mask_origin=None,
                    bbox=(0.0, 0.0, 404.0, 400.0))


class _PathBase:
    def __init__(self, instances, w=800, h=800, img_path=None):
        self._insts, self._w, self._h = instances, w, h
        self._img_path = img_path
    def __len__(self):
        return 1
    def _image_path(self, i):
        return self._img_path
    def _load_tile_metadata(self, i):
        boxes = np.array([list(inst.bbox) for inst in self._insts],
                         dtype=np.float32)
        return self._w, self._h, boxes
    def _image_regime(self, i):
        return "tiled_multiscale"
    def _load_annotations(self, i):
        return self._insts


def _index_kw():
    return dict(tile_sizes=[400], overlap=0.0, negative_ratio=0.9,
                min_annotation_overlap=0.05, seed=42,
                whole_image_if_giant=False)


def test_blocklist_filters_negative_windows(tmp_path):
    # Create a temporary image file
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"fake jpeg")
    base = _PathBase([_thin_diag_instance()], img_path=str(img_path))
    plain = build_tile_index(base, **_index_kw())
    n_plain = sum(len(v) for v in plain.values())
    # block one specific negative window: find it from the pools
    pools = {}
    build_tile_index(base, pools_out=pools, **_index_kw())
    img_idx, ((x, y, w, h), _b) = pools["negatives"][0]
    bl = tmp_path / "bl.json"
    bl.write_text(json.dumps([[str(img_path), x, y, w, h]]))
    blocked = build_tile_index(
        base, negative_blocklist=load_negative_blocklist(bl), **_index_kw())
    n_blocked = sum(len(v) for v in blocked.values())
    assert n_blocked == n_plain - 1


def test_no_blocklist_is_identity(tmp_path):
    # Create a temporary image file
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"fake jpeg")
    base = _PathBase([_thin_diag_instance()], img_path=str(img_path))
    assert (build_tile_index(base, **_index_kw())
            == build_tile_index(base, negative_blocklist=None, **_index_kw()))
