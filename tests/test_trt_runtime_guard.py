"""Tests for TensorRTRuntime.run() guarding boolean-returning TRT calls.

TensorRT's Python API signals failure from ``set_input_shape``,
``set_tensor_address`` and ``execute_async_v3`` by returning ``False``
instead of raising. If those return values are ignored, ``run()`` copies
back never-written device buffers and callers receive silent garbage
(e.g. 200 detections/image, all score=sigmoid(0)=0.5, boxes [0,0,0,0]).
It also defeats ``Predictor.predict_batch``'s try/except per-image
fallback, which can only trigger on an exception.

No GPU required: the runtime is built via ``__new__`` and the minimal
instance attributes that ``run()`` dereferences are hand-stubbed.
"""

import numpy as np
import pytest

from training_toolkit.inference.runtime import TensorRTRuntime


class FakeContext:
    """Minimal IExecutionContext stand-in with controllable return values."""

    def __init__(
        self,
        set_input_shape_ok: bool = True,
        set_tensor_address_ok: bool = True,
        execute_ok: bool = True,
    ) -> None:
        self._set_input_shape_ok = set_input_shape_ok
        self._set_tensor_address_ok = set_tensor_address_ok
        self._execute_ok = execute_ok

    def set_input_shape(self, name, shape) -> bool:
        return self._set_input_shape_ok

    def set_tensor_address(self, name, ptr) -> bool:
        return self._set_tensor_address_ok

    def execute_async_v3(self, stream_handle) -> bool:
        return self._execute_ok


class FakeEngine:
    """Engine stub exposing the optimization-profile range query."""

    def get_tensor_profile_shape(self, name, profile_index):
        return [(4, 3, 32, 32), (4, 3, 32, 32), (4, 3, 32, 32)]


class FakeCudaDriver:
    """No-op CUDA driver namespace covering exactly what run() touches."""

    @staticmethod
    def cuMemAlloc(nbytes):
        return (0, 1234)

    @staticmethod
    def cuMemcpyHtoD(dst, src, nbytes):
        return (0,)

    @staticmethod
    def cuMemcpyDtoH(dst, src, nbytes):
        return (0,)

    @staticmethod
    def cuStreamSynchronize(stream):
        return (0,)


def _make_runtime(context: FakeContext, dynamic_batch: bool) -> TensorRTRuntime:
    """Build a TensorRTRuntime without __init__ (no cuda import, no GPU)."""
    rt = TensorRTRuntime.__new__(TensorRTRuntime)
    batch_dim = -1 if dynamic_batch else 1
    rt._context = context
    rt._input_names = ["images"]
    rt._output_names = ["dets"]
    rt._input_shapes = {"images": (batch_dim, 3, 32, 32)}
    rt._output_shapes = {"dets": (batch_dim, 300, 6)}
    rt._dynamic_batch = dynamic_batch
    rt._last_batch_size = None
    rt._stream = 7
    rt._device_buffers = {"images": 1111, "dets": 2222}
    rt._cuda_driver = FakeCudaDriver()
    return rt


def test_run_raises_when_set_input_shape_fails():
    """A batch shape outside the engine profile must raise, not return garbage."""
    ctx = FakeContext(set_input_shape_ok=False)
    # Profile-range diagnostics, queried best-effort via context.engine
    ctx.active_optimization_profile = 0
    ctx.engine = FakeEngine()
    rt = _make_runtime(ctx, dynamic_batch=True)

    inp = np.zeros((2, 3, 32, 32), dtype=np.float32)
    with pytest.raises(RuntimeError, match="set_input_shape") as excinfo:
        rt.run(inp)

    # Message carries the attempted shape and the valid profile range
    assert "(2, 3, 32, 32)" in str(excinfo.value)
    assert "(4, 3, 32, 32)" in str(excinfo.value)


def test_run_raises_when_execute_async_v3_fails():
    """A rejected enqueue must raise (with the batch size), not return stale buffers."""
    ctx = FakeContext(execute_ok=False)
    rt = _make_runtime(ctx, dynamic_batch=False)

    inp = np.zeros((1, 3, 32, 32), dtype=np.float32)
    with pytest.raises(RuntimeError, match="execute_async_v3") as excinfo:
        rt.run(inp)

    assert "batch size 1" in str(excinfo.value)


def test_run_raises_when_set_tensor_address_fails():
    """A failed binding-address set must raise before execution."""
    ctx = FakeContext(set_tensor_address_ok=False)
    rt = _make_runtime(ctx, dynamic_batch=False)

    inp = np.zeros((1, 3, 32, 32), dtype=np.float32)
    with pytest.raises(RuntimeError, match="set_tensor_address"):
        rt.run(inp)
