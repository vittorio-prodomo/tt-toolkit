"""Pluggable inference runtime abstraction."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import logging

logger = logging.getLogger(__name__)


class InferenceRuntime(ABC):
    """Abstract base for inference runtimes (ORT, TensorRT, etc.)."""

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load a model from disk (ONNX or serialized engine)."""

    @abstractmethod
    def run(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        """Run inference on a (B, C, H, W) float32 input tensor.

        Returns a dict mapping output names to numpy arrays.
        """


class OnnxRuntime(InferenceRuntime):
    """ONNX Runtime inference backend."""

    def __init__(self) -> None:
        self._session = None
        self._input_name: str = ""
        self._output_names: list[str] = []

    def load(self, model_path: Path) -> None:
        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(model_path),
            providers=ort.get_available_providers(),
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]

    def run(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        if self._session is None:
            raise RuntimeError("Call load() before run()")
        outputs = self._session.run(None, {self._input_name: input_tensor})
        return dict(zip(self._output_names, outputs))


class TensorRTRuntime(InferenceRuntime):
    """TensorRT inference backend with FP16 support and engine caching."""

    def __init__(
        self,
        fp16: bool = True,
        batch_range: tuple[int, int, int] | None = None,
    ) -> None:
        """
        Args:
            fp16: Enable FP16 precision.
            batch_range: (min, opt, max) batch sizes for TRT optimization profile.
                Enables dynamic batching. If None, uses static shapes from the ONNX.
        """
        self._fp16 = fp16
        self._batch_range = batch_range
        self._context = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._output_shapes: dict[str, tuple[int, ...]] = {}
        self._input_shapes: dict[str, tuple[int, ...]] = {}
        self._stream = None
        self._device_buffers: dict[str, object] = {}
        self._host_outputs: dict[str, np.ndarray] = {}
        self._last_batch_size: int | None = None
        self._dynamic_batch: bool = False
        self._cuda_driver = self._import_cuda_driver()

    @staticmethod
    def _import_cuda_driver():
        """Import the CUDA driver API (new and old cuda-python layouts).

        Returns None when cuda-python is not installed, so that constructing the
        runtime never fails; ``load()`` then reports the missing extra clearly.
        """
        try:
            from cuda.bindings import driver
            return driver
        except ImportError:
            try:
                import cuda
                return cuda
            except ImportError:
                return None

    def load(self, model_path: Path) -> None:
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError(
                "TensorRT is not installed. "
                "Install the `tensorrt` extra (e.g. uv sync --extra tensorrt)."
            )
        if self._cuda_driver is None:
            raise RuntimeError(
                "cuda-python (cuda-bindings) is not installed. "
                "Install the `tensorrt` extra (e.g. uv sync --extra tensorrt)."
            )

        self._trt = trt
        trt_logger = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(trt_logger)

        suffix = model_path.suffix.lower()
        if suffix in (".engine", ".trt"):
            engine = self._load_engine(runtime, model_path)
        elif suffix == ".onnx":
            engine = self._load_onnx(runtime, trt_logger, model_path)
        else:
            raise ValueError(
                f"Unsupported model format: {suffix!r} "
                f"(expected .engine, .trt, or .onnx)"
            )

        self._setup_bindings(engine)

    def run(self, input_tensor: np.ndarray) -> dict[str, np.ndarray]:
        if self._context is None:
            raise RuntimeError("Call load() before run()")

        batch_size = input_tensor.shape[0]

        # Handle dynamic batch: update shapes and reallocate if needed
        if self._dynamic_batch and batch_size != self._last_batch_size:
            for name in self._input_names:
                shape = list(self._input_shapes[name])
                shape[0] = batch_size
                if not self._context.set_input_shape(name, shape):
                    profile_hint = ""
                    try:  # best-effort diagnostics only
                        profile_idx = max(
                            0, int(self._context.active_optimization_profile),
                        )
                        min_s, opt_s, max_s = (
                            self._context.engine.get_tensor_profile_shape(
                                name, profile_idx,
                            )
                        )
                        profile_hint = (
                            f" (engine profile {profile_idx}: min={tuple(min_s)}, "
                            f"opt={tuple(opt_s)}, max={tuple(max_s)})"
                        )
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"TensorRT set_input_shape failed for tensor {name!r} "
                        f"with shape {tuple(shape)}{profile_hint} — the shape "
                        f"does not satisfy the engine's optimization profile."
                    )
            self._allocate_buffers(batch_size)
            self._last_batch_size = batch_size
        elif not self._dynamic_batch:
            expected = self._input_shapes[self._input_names[0]][0]
            if batch_size != expected:
                raise RuntimeError(
                    f"Expected batch size {expected}, got {batch_size}. "
                    f"Model has fixed batch size."
                )

        cuda_drv = self._cuda_driver

        # Copy input to device
        for name in self._input_names:
            host_data = np.ascontiguousarray(input_tensor)
            cuda_drv.cuMemcpyHtoD(
                self._device_buffers[name],
                host_data.ctypes.data,
                host_data.nbytes,
            )
            if not self._context.set_tensor_address(
                name, int(self._device_buffers[name]),
            ):
                raise RuntimeError(
                    f"TensorRT set_tensor_address failed for input tensor {name!r}"
                )

        # Set output addresses
        for name in self._output_names:
            if not self._context.set_tensor_address(
                name, int(self._device_buffers[name]),
            ):
                raise RuntimeError(
                    f"TensorRT set_tensor_address failed for output tensor {name!r}"
                )

        # Execute
        if not self._context.execute_async_v3(stream_handle=int(self._stream)):
            raise RuntimeError(
                f"TensorRT execute_async_v3 failed (enqueue rejected) for "
                f"batch size {batch_size} — aborting instead of returning "
                f"never-written device buffers."
            )
        cuda_drv.cuStreamSynchronize(self._stream)

        # Copy outputs to host
        results = {}
        for name in self._output_names:
            shape = list(self._output_shapes[name])
            if self._dynamic_batch:
                shape[0] = batch_size
            host_buf = np.empty(shape, dtype=np.float32)
            cuda_drv.cuMemcpyDtoH(
                host_buf.ctypes.data,
                self._device_buffers[name],
                host_buf.nbytes,
            )
            results[name] = host_buf

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_engine(runtime, engine_path: Path):
        """Deserialize a pre-built engine file."""
        logger.info("Loading TensorRT engine from %s", engine_path.name)
        data = engine_path.read_bytes()
        engine = runtime.deserialize_cuda_engine(data)
        if engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {engine_path}"
            )
        return engine

    def _load_onnx(self, runtime, trt_logger, onnx_path: Path):
        """Load from ONNX, using cached engine if available."""
        # Include batch range in cache key so different profiles get separate engines
        if self._batch_range is not None:
            min_b, opt_b, max_b = self._batch_range
            suffix = f".b{min_b}-{opt_b}-{max_b}.engine"
        else:
            suffix = ".engine"
        cache_path = onnx_path.with_suffix(suffix)

        # Try cached engine first
        if cache_path.exists():
            logger.info(
                "Found cached engine %s — attempting to load",
                cache_path.name,
            )
            data = cache_path.read_bytes()
            engine = runtime.deserialize_cuda_engine(data)
            if engine is not None:
                return engine
            logger.warning(
                "Cached engine failed to load (GPU mismatch?) — rebuilding",
            )

        # Build from ONNX
        return self._build_engine(runtime, trt_logger, onnx_path, cache_path)

    def _build_engine(self, runtime, trt_logger, onnx_path: Path, cache_path: Path):
        """Build a TensorRT engine from an ONNX graph."""
        trt = self._trt
        logger.info("Building TensorRT engine from %s (this may take 30-60s)...", onnx_path.name)

        builder = trt.Builder(trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH),
        )
        parser = trt.OnnxParser(network, trt_logger)

        onnx_data = onnx_path.read_bytes()
        if not parser.parse(onnx_data):
            errors = "\n".join(
                str(parser.get_error(i)) for i in range(parser.num_errors)
            )
            raise RuntimeError(f"ONNX parse failed:\n{errors}")

        config = builder.create_builder_config()
        if self._fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        # Optimization profile for dynamic batching
        if self._batch_range is not None:
            min_b, opt_b, max_b = self._batch_range
            profile = builder.create_optimization_profile()
            for i in range(network.num_inputs):
                inp = network.get_input(i)
                shape = inp.shape  # e.g. (-1, 3, 504, 504) or (1, 3, 504, 504)
                base = list(shape)
                base[0] = min_b
                min_shape = tuple(base)
                base[0] = opt_b
                opt_shape = tuple(base)
                base[0] = max_b
                max_shape = tuple(base)
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
                logger.info(
                    "Optimization profile for '%s': min=%s, opt=%s, max=%s",
                    inp.name, min_shape, opt_shape, max_shape,
                )
            config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT engine build failed")

        # Cache the engine (atomic write)
        tmp_path = cache_path.with_suffix(".engine.tmp")
        try:
            tmp_path.write_bytes(serialized)
            import os
            os.replace(tmp_path, cache_path)
            logger.info("Cached engine to %s", cache_path.name)
        except OSError:
            logger.warning("Could not cache engine (permissions?) — using in-memory engine")

        engine = runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError("Failed to deserialize freshly built engine")

        logger.info("Engine build complete")
        return engine

    def _setup_bindings(self, engine) -> None:
        """Inspect engine bindings, allocate CUDA buffers, create exec context."""
        trt = self._trt

        self._input_names = []
        self._output_names = []
        self._input_shapes = {}
        self._output_shapes = {}

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = tuple(engine.get_tensor_shape(name))

            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
                self._input_shapes[name] = shape
                if shape[0] == -1:
                    self._dynamic_batch = True
            else:
                self._output_names.append(name)
                self._output_shapes[name] = shape

        self._context = engine.create_execution_context()

        # Allocate buffers for fixed-batch models
        if not self._dynamic_batch:
            self._allocate_buffers(self._input_shapes[self._input_names[0]][0])

        # Create CUDA stream
        cuda_drv = self._cuda_driver
        err, self._stream = cuda_drv.cuStreamCreate(0)

    def _allocate_buffers(self, batch_size: int) -> None:
        """Allocate device memory for inputs and outputs."""
        cuda_drv = self._cuda_driver

        for name, shape in {**self._input_shapes, **self._output_shapes}.items():
            shape_list = list(shape)
            if shape_list[0] == -1:
                shape_list[0] = batch_size
            nbytes = int(np.prod(shape_list)) * 4  # float32 = 4 bytes
            err, ptr = cuda_drv.cuMemAlloc(nbytes)
            self._device_buffers[name] = ptr
