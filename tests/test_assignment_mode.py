"""assignment_mode: bbox_ioa (default, byte-identical) vs mask_true."""
import numpy as np
import pytest

from training_toolkit.data.tile_materialize import (
    Instance, assign_instances_to_tile, materialize_tile,
)
from training_toolkit.data.tile_index import (
    build_tile_index, mask_true_positive_windows,
)
from training_toolkit.data.label_crop import label_crop
from training_toolkit.configs.tiling_config import TilingConfig


def _thin_diag_instance():
    """A 4px-wide diagonal stroke from (0,0) to (404,400): pixels satisfy
    x in [y, y+4]. Its bbox (404x400) is mostly empty. Window (350,350,100,100)
    holds REAL stroke pixels but bbox-IoA = (54*50)/161600 = 0.017 < 0.05 ->
    dropped by bbox_ioa, admitted by mask_true."""
    ring = np.array([[0, 0], [4, 0], [404, 400], [400, 400]], dtype=np.float32)
    return Instance(label=0, polygon=[ring], mask=None, mask_origin=None,
                    bbox=(0.0, 0.0, 404.0, 400.0))


def _far_corner_instance():
    """bbox spans the whole image but the actual blob sits at (500..520)^2 —
    a curve-through-corner impostor for window (0,0,100,100)."""
    ring = np.array([[500, 500], [520, 500], [520, 520], [500, 520]],
                    dtype=np.float32)
    # deliberately WRONG-looking wide bbox (as a giant ann's bbox would be)
    return Instance(label=0, polygon=[ring], mask=None, mask_origin=None,
                    bbox=(0.0, 0.0, 520.0, 520.0))


def test_bbox_ioa_default_drops_sliver():
    keep = assign_instances_to_tile([_thin_diag_instance()], (350, 350, 100, 100))
    assert keep == []          # current behavior, unchanged


def test_mask_true_admits_sliver():
    keep = assign_instances_to_tile([_thin_diag_instance()], (350, 350, 100, 100),
                                    assignment_mode="mask_true")
    assert keep == [0]


def test_mask_true_rejects_empty_curve_through_corner():
    keep = assign_instances_to_tile([_far_corner_instance()], (0, 0, 100, 100),
                                    assignment_mode="mask_true")
    assert keep == []


def test_mask_true_positive_windows_marks_only_pixel_bearing():
    windows = [((350, 350, 100, 100), True),   # stroke pixels -> positive
               ((0, 300, 100, 100), True),     # bbox overlaps, zero pixels
               ((500, 500, 100, 100), True)]   # no bbox overlap at all
    pos = mask_true_positive_windows(windows, [_thin_diag_instance()])
    assert 0 in pos
    assert 1 not in pos        # curve-through-corner rejected
    assert 2 not in pos


class _FakeBase:
    """Minimal dataset contract for build_tile_index."""
    def __init__(self, instances, w=800, h=800):
        self._insts, self._w, self._h = instances, w, h
    def __len__(self):
        return 1
    def _image_path(self, i):
        raise NotImplementedError   # build_tile_index treats as "no disk check"
    def _load_tile_metadata(self, i):
        boxes = np.array([[x1, y1, x2, y2] for (x1, y1, x2, y2) in
                          [inst.bbox for inst in self._insts]], dtype=np.float32)
        return self._w, self._h, boxes
    def _image_regime(self, i):
        return "tiled_multiscale"
    def _load_annotations(self, i):
        return self._insts


def test_build_tile_index_mask_true_reclassifies():
    """800x800 image, 400px grid, overlap 0 -> 4 windows. Stroke pixels
    (x in [y, y+4]) land in windows (0,0), (400,0) [x 400-404 at y 396-400]
    and (400,400) — never (0,400) (x >= y >= 400 there). bbox-IoA assigns
    only (0,0): the other three intersect the bbox at < 5% of its area."""
    base = _FakeBase([_thin_diag_instance()])
    kw = dict(tile_sizes=[400], overlap=0.0, negative_ratio=1e-9,
              min_annotation_overlap=0.05, seed=42, whole_image_if_giant=False)
    default = build_tile_index(base, **kw)
    masktrue = build_tile_index(base, assignment_mode="mask_true", **kw)
    n_default = sum(len(v) for v in default.values())
    n_masktrue = sum(len(v) for v in masktrue.values())
    assert n_default == 1      # positives only (negative_ratio ~ 0)
    assert n_masktrue == 3


def test_mask_true_layers_agree_on_boundary_touching_bbox():
    """Cross-layer consistency on a bbox that exactly TOUCHES the window edge
    (bbox.y2 == window.y == 400): cv2 rasterizes the polygon's extremum cap
    edge (real pixels in the window's row 0, x 400-404), so classification AND
    selection must BOTH own it — a window marked positive by
    mask_true_positive_windows must never materialize with zero instances
    (that would train real crack pixels as clean background, the exact V1
    leak this mode exists to fix)."""
    inst = _thin_diag_instance()
    window = (400, 400, 400, 400)
    # classification layer: the window is positive off its boundary pixels
    assert mask_true_positive_windows([(window, True)], [inst]) == {0}
    # selection layer: the same window selects the instance
    keep = assign_instances_to_tile([inst], window, assignment_mode="mask_true")
    assert keep == [0]
    # labeling: the crop owns the pixels
    out = label_crop(window, [inst], realized_scale=1.0, class_mask=None,
                     assignment_mode="mask_true")
    assert len(out["masks"]) == 1


def test_label_crop_mask_true_owns_sliver_pixels():
    out = label_crop((350, 350, 100, 100), [_thin_diag_instance()],
                     realized_scale=504 / 416, class_mask=None,
                     assignment_mode="mask_true")
    assert len(out["masks"]) == 1          # the pixels are OWNED (pos or ignore)


def test_tiling_config_validates_mode():
    TilingConfig(assignment_mode="mask_true")
    with pytest.raises(ValueError):
        TilingConfig(assignment_mode="nope")
