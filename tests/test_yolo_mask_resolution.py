"""Regression test for the YOLO proto->input resolution (the mask_ratio eval bug).

The val postprocess must reconstruct the input size as proto * 4 (YOLO-seg
prototypes are architecturally 1/4 of the input), INDEPENDENT of mask_ratio.
The old code used proto * mask_ratio, which silently produced 4x-too-small mask
crops at mask_ratio != 4 (e.g. the v3 mask_ratio=1) -> segm mAP collapsed ~6x.
"""
from training_toolkit.models.yolo import YOLOAdapter


def test_input_hw_is_proto_times_four_regardless_of_mask_ratio():
    a = YOLOAdapter()
    for mask_ratio in (1, 2, 4, 8):
        a._mask_ratio = mask_ratio
        # 480px input -> proto 120 -> must reconstruct 480 (NOT 120*mask_ratio)
        assert a._input_hw_from_proto(120, 120) == (480, 480), (
            f"mask_ratio={mask_ratio}: expected proto*4, got proto*mask_ratio (the bug)"
        )
        # non-square sanity
        assert a._input_hw_from_proto(104, 160) == (416, 640)


def test_proto_downsample_is_four():
    assert YOLOAdapter._PROTO_DOWNSAMPLE == 4
