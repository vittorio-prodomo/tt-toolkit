"""Tests for the inference runtime."""

import importlib.util
import json
import numpy as np

import pytest

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestOnnxRuntime:
    """Test OnnxRuntime.load() and .run() with a minimal ONNX model."""

    @pytest.fixture
    def trivial_onnx(self, tmp_path):
        """Build a minimal ONNX model: identity op, input (1,3,32,32) -> output (1,3,32,32)."""
        onnx = pytest.importorskip("onnx")
        from onnx import helper, TensorProto

        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 32, 32])
        node = helper.make_node("Identity", ["input"], ["output"])
        graph = helper.make_graph([node], "test", [X], [Y])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        path = tmp_path / "model.onnx"
        onnx.save(model, str(path))
        return path

    def test_load_and_run(self, trivial_onnx):
        pytest.importorskip("onnxruntime")
        from training_toolkit.inference.runtime import OnnxRuntime

        rt = OnnxRuntime()
        rt.load(trivial_onnx)
        inp = np.random.randn(1, 3, 32, 32).astype(np.float32)
        result = rt.run(inp)

        assert "output" in result
        np.testing.assert_array_almost_equal(result["output"], inp)


class TestPostprocessRfdetr:
    """Test RF-DETR ONNX output → standardized result dict."""

    def test_filters_by_threshold(self):
        from training_toolkit.inference.postprocess import postprocess_rfdetr

        # New ONNX schema (see postprocess_rfdetr docstring):
        #   dets:   (B, Q, 4) cxcywh normalized
        #   labels: (B, Q, C) class logits, last column is no-object
        #   masks:  (B, Q, H/4, W/4)
        #
        # 4 queries, 2 classes (+ 1 no-object); 2 queries above threshold 0.5.
        # sigmoid(3.0) ≈ 0.953, sigmoid(-3.0) ≈ 0.047.
        raw = {
            "dets": np.array([[
                [0.25, 0.25, 0.5, 0.5],
                [0.5,  0.5,  0.4, 0.4],
                [0.7,  0.7,  0.3, 0.3],
                [0.3,  0.3,  0.2, 0.2],
            ]], dtype=np.float32),                                # (1, 4, 4)
            "labels": np.array([[
                [ 3.0, -3.0, -3.0],   # class 0 strong
                [-3.0,  3.0, -3.0],   # class 1 strong
                [-3.0, -3.0,  3.0],   # no-object strong → drops below threshold
                [-3.0, -3.0,  3.0],   # no-object strong → drops below threshold
            ]], dtype=np.float32),                                # (1, 4, 3)
            "masks": np.random.randn(1, 4, 16, 16).astype(np.float32),
        }
        result = postprocess_rfdetr(
            raw, original_size=(64, 64), export_size=64, score_threshold=0.5,
        )
        assert result["boxes"].shape == (2, 4)
        assert result["scores"].shape == (2,)
        assert result["labels"].shape == (2,)
        assert result["masks"].shape == (2, 64, 64)
        assert result["masks"].dtype == bool

    def test_scales_boxes_to_original(self):
        from training_toolkit.inference.postprocess import postprocess_rfdetr

        # cxcywh (0.25, 0.25, 0.5, 0.5) normalized → xyxy in 128px image = [0, 0, 64, 64].
        # Note: export_size is unused by postprocess_rfdetr (boxes are normalized,
        # original_size alone drives scaling).
        raw = {
            "dets": np.array([[[0.25, 0.25, 0.5, 0.5]]], dtype=np.float32),
            "labels": np.array([[[3.0, -3.0]]], dtype=np.float32),  # 1 fg + 1 no-object
            "masks": np.random.randn(1, 1, 16, 16).astype(np.float32),
        }
        result = postprocess_rfdetr(
            raw, original_size=(128, 128), export_size=64, score_threshold=0.5,
        )
        np.testing.assert_array_almost_equal(result["boxes"][0], [0, 0, 64, 64])

    def test_empty_after_threshold(self):
        from training_toolkit.inference.postprocess import postprocess_rfdetr

        raw = {
            "dets": np.array([[[0.25, 0.25, 0.5, 0.5]]], dtype=np.float32),
            # Both fg classes well below threshold (sigmoid(-3) ≈ 0.047).
            "labels": np.array([[[-3.0, -3.0]]], dtype=np.float32),
            "masks": np.random.randn(1, 1, 8, 8).astype(np.float32),
        }
        result = postprocess_rfdetr(
            raw, original_size=(64, 64), export_size=64, score_threshold=0.5,
        )
        assert result["boxes"].shape == (0, 4)
        assert result["scores"].shape == (0,)
        assert result["masks"].shape == (0, 64, 64)


class TestPostprocessMask2former:
    """Test Mask2Former ONNX output → standardized result dict."""

    def _make_raw(self, num_queries=3, num_classes=2, h=8, w=8):
        """Build synthetic Mask2Former ONNX outputs."""
        # class_logits: (1, Q, C+1) — last column is no-object
        logits = np.full((1, num_queries, num_classes + 1), -5.0, dtype=np.float32)
        # Make query 0 strongly predict class 0
        logits[0, 0, 0] = 5.0
        # Make query 1 strongly predict class 1 (only when present)
        if num_queries > 1:
            logits[0, 1, 1] = 5.0
        # Query 2 predicts no-object (only when present)
        if num_queries > 2:
            logits[0, 2, num_classes] = 5.0

        # mask_logits: (1, Q, H/4, W/4) — big positive = mask present
        masks = np.full((1, num_queries, h, w), -10.0, dtype=np.float32)
        masks[0, 0, 2:6, 2:6] = 10.0  # query 0: center square
        if num_queries > 1:
            masks[0, 1, 0:4, 0:4] = 10.0  # query 1: top-left square
        return {"class_logits": logits, "mask_logits": masks}

    def test_filters_no_object_queries(self):
        from training_toolkit.inference.postprocess import postprocess_mask2former

        raw = self._make_raw()
        result = postprocess_mask2former(
            raw, original_size=(32, 32), export_size=32, score_threshold=0.5,
        )
        # Query 2 (no-object) should be filtered out
        assert result["labels"].shape[0] == 2
        assert 0 in result["labels"]
        assert 1 in result["labels"]

    def test_masks_are_binary_and_resized(self):
        from training_toolkit.inference.postprocess import postprocess_mask2former

        raw = self._make_raw(num_queries=1, num_classes=2, h=8, w=8)
        raw["class_logits"][0, 0, 0] = 10.0
        result = postprocess_mask2former(
            raw, original_size=(64, 64), export_size=32, score_threshold=0.1,
        )
        assert result["masks"].dtype == bool
        assert result["masks"].shape == (1, 64, 64)

    def test_boxes_computed_from_masks(self):
        from training_toolkit.inference.postprocess import postprocess_mask2former

        raw = self._make_raw(num_queries=1, num_classes=2, h=16, w=16)
        raw["class_logits"][0, 0, 0] = 10.0
        # Mask: positive square from (4,4) to (12,12) in 16x16 space
        raw["mask_logits"] = np.full((1, 1, 16, 16), -10.0, dtype=np.float32)
        raw["mask_logits"][0, 0, 4:12, 4:12] = 10.0

        result = postprocess_mask2former(
            raw, original_size=(16, 16), export_size=16, score_threshold=0.1,
        )
        # Box should cover the mask region
        box = result["boxes"][0]
        assert box[0] >= 3 and box[0] <= 5    # x1 ≈ 4
        assert box[2] >= 10 and box[2] <= 12   # x2 ≈ 11

    def test_empty_when_all_no_object(self):
        from training_toolkit.inference.postprocess import postprocess_mask2former

        raw = self._make_raw()
        # Set all queries to predict no-object
        raw["class_logits"][:] = -5.0
        raw["class_logits"][0, :, -1] = 5.0
        result = postprocess_mask2former(
            raw, original_size=(32, 32), export_size=32, score_threshold=0.5,
        )
        assert result["boxes"].shape == (0, 4)
        assert result["masks"].shape == (0, 32, 32)


class TestPredictor:
    """Test Predictor end-to-end with a fake export directory."""

    @pytest.fixture
    def fake_rfdetr_export(self, tmp_path):
        """Create a fake RF-DETR export dir with metadata.json and a stub ONNX model.

        The stub ONNX model takes (1, 3, 64, 64) and emits the raw RF-DETR
        export schema (see ``postprocess_rfdetr``):
        - dets:   (1, 2, 4) — normalized cxcywh boxes
        - labels: (1, 2, 3) — class logits (2 fg classes + 1 no-object)
        - masks:  (1, 2, 16, 16) — mask logits

        Query 0 strongly predicts class 0 (kept above threshold 0.5); query 1
        strongly predicts no-object (dropped).
        """
        onnx = pytest.importorskip("onnx")
        from onnx import helper, TensorProto, numpy_helper

        # Constant tensors for deterministic outputs.
        # sigmoid(3.0) ≈ 0.953 (kept), sigmoid(-3.0) ≈ 0.047 (dropped).
        dets_data = np.array([[[0.5, 0.5, 0.4, 0.4],
                               [0.7, 0.7, 0.3, 0.3]]], dtype=np.float32)
        labels_data = np.array([[[3.0, -3.0, -3.0],
                                 [-3.0, -3.0, 3.0]]], dtype=np.float32)
        masks_data = np.random.randn(1, 2, 16, 16).astype(np.float32)

        dets_const = numpy_helper.from_array(dets_data, name="dets_const")
        labels_const = numpy_helper.from_array(labels_data, name="labels_const")
        masks_const = numpy_helper.from_array(masks_data, name="masks_const")

        # Input (required but unused — model outputs constants)
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 64, 64])

        # Outputs
        dets_out = helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, 2, 4])
        labels_out = helper.make_tensor_value_info("labels", TensorProto.FLOAT, [1, 2, 3])
        masks_out = helper.make_tensor_value_info("masks", TensorProto.FLOAT, [1, 2, 16, 16])

        # Identity nodes from constants to outputs
        n1 = helper.make_node("Constant", [], ["dets"], value=dets_const)
        n2 = helper.make_node("Constant", [], ["labels"], value=labels_const)
        n3 = helper.make_node("Constant", [], ["masks"], value=masks_const)

        graph = helper.make_graph([n1, n2, n3], "rfdetr_stub", [X],
                                  [dets_out, labels_out, masks_out])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save(model, str(tmp_path / "inference_model.onnx"))

        metadata = {
            "model": "rf-detr-seg-large",
            "backend": "rfdetr",
            "resolution": [64, 64],
            "batch_size": "dynamic",
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "class_names": {"0": "crack", "1": "spall"},
        }
        (tmp_path / "metadata.json").write_text(json.dumps(metadata))
        return tmp_path

    def test_predict_returns_standard_dict(self, fake_rfdetr_export):
        pytest.importorskip("onnxruntime")
        from training_toolkit.inference import Predictor

        predictor = Predictor(str(fake_rfdetr_export))
        # Create a test image
        img = np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8)
        result = predictor.predict(img, score_threshold=0.5)

        assert "boxes" in result
        assert "scores" in result
        assert "labels" in result
        assert "masks" in result
        assert result["boxes"].ndim == 2 and result["boxes"].shape[1] == 4
        assert result["scores"].ndim == 1
        # One detection above 0.5 threshold (query 0: class 0; query 1 is no-object)
        assert len(result["scores"]) == 1

    def test_predict_from_file_path(self, fake_rfdetr_export, tmp_path):
        pytest.importorskip("onnxruntime")
        from training_toolkit.inference import Predictor
        from PIL import Image

        predictor = Predictor(str(fake_rfdetr_export))
        img = Image.fromarray(np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8))
        img_path = tmp_path / "test_img.jpg"
        img.save(img_path)

        result = predictor.predict(str(img_path))
        assert "boxes" in result

    def test_metadata_properties(self, fake_rfdetr_export):
        pytest.importorskip("onnxruntime")
        from training_toolkit.inference import Predictor

        predictor = Predictor(str(fake_rfdetr_export))
        assert predictor.backend == "rfdetr"
        assert predictor.resolution == (64, 64)
        assert predictor.class_names is not None

    def test_predict_batch(self, fake_rfdetr_export):
        pytest.importorskip("onnxruntime")
        from training_toolkit.inference import Predictor

        predictor = Predictor(str(fake_rfdetr_export))
        imgs = [
            np.random.randint(0, 255, (100, 80, 3), dtype=np.uint8),
            np.random.randint(0, 255, (50, 120, 3), dtype=np.uint8),
        ]
        results = predictor.predict_batch(imgs)
        assert len(results) == 2
        for r in results:
            assert "boxes" in r and "scores" in r and "labels" in r and "masks" in r


class TestFindModel:
    """Test Predictor._find_model() with different runtime keys."""

    def _make_export_dir(self, tmp_path, files):
        """Create files in tmp_path and return the path."""
        for name in files:
            (tmp_path / name).touch()
        return tmp_path

    def test_ort_prefers_sim_onnx(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        d = self._make_export_dir(tmp_path, ["model.onnx", "model.sim.onnx"])
        result = Predictor._find_model(d, runtime="ort")
        assert result.name == "model.sim.onnx"

    def test_ort_falls_back_to_plain_onnx(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        d = self._make_export_dir(tmp_path, ["model.onnx"])
        result = Predictor._find_model(d, runtime="ort")
        assert result.name == "model.onnx"

    def test_trt_prefers_engine(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        d = self._make_export_dir(
            tmp_path, ["model.onnx", "model.sim.onnx", "model.engine"],
        )
        result = Predictor._find_model(d, runtime="trt")
        assert result.name == "model.engine"

    def test_trt_falls_back_to_sim_onnx(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        d = self._make_export_dir(tmp_path, ["model.onnx", "model.sim.onnx"])
        result = Predictor._find_model(d, runtime="trt")
        assert result.name == "model.sim.onnx"

    def test_trt_falls_back_to_plain_onnx(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        d = self._make_export_dir(tmp_path, ["model.onnx"])
        result = Predictor._find_model(d, runtime="trt")
        assert result.name == "model.onnx"

    def test_raises_when_no_model_found(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        with pytest.raises(FileNotFoundError):
            Predictor._find_model(tmp_path, runtime="ort")

    def test_trt_also_finds_trt_extension(self, tmp_path):
        from training_toolkit.inference.predictor import Predictor

        d = self._make_export_dir(tmp_path, ["model.trt"])
        result = Predictor._find_model(d, runtime="trt")
        assert result.name == "model.trt"


class TestTensorRTRuntimeMissingExtra:
    """Clear errors when the optional TensorRT stack is absent."""

    def test_import_error_gives_clear_message(self):
        """Verify clear error when tensorrt is not installed."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        rt = TensorRTRuntime()
        with patch.dict("sys.modules", {"tensorrt": None}):
            with pytest.raises(RuntimeError, match="TensorRT is not installed"):
                rt.load(Path("/fake/model.onnx"))

    def test_missing_cuda_bindings_gives_clear_message(self):
        """Constructing never fails; load() names the missing extra."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        rt = TensorRTRuntime()
        rt._cuda_driver = None  # as if cuda-python were not installed
        with patch.dict("sys.modules", {"tensorrt": MagicMock()}):
            with pytest.raises(RuntimeError, match="cuda-python"):
                rt.load(Path("/fake/model.onnx"))


_HAS_CUDA_BINDINGS = importlib.util.find_spec("cuda") is not None


@pytest.mark.skipif(not _HAS_CUDA_BINDINGS, reason="needs cuda-python (the `tensorrt` extra)")
class TestTensorRTRuntime:
    """Test TensorRTRuntime with mocked tensorrt (no GPU required).

    The CUDA driver bindings (cuda-python) are used for real, so these tests need
    the `tensorrt` extra installed; they skip otherwise.
    """

    def test_load_engine_file(self, tmp_path):
        """Verify .engine files are deserialized directly."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"fake-engine-data")

        mock_trt = MagicMock()
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_trt.Runtime.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        # Engine has one input and one output binding
        mock_engine.num_io_tensors = 2
        mock_engine.get_tensor_name.side_effect = ["input", "output"]
        mock_engine.get_tensor_mode.side_effect = [
            mock_trt.TensorIOMode.INPUT,
            mock_trt.TensorIOMode.OUTPUT,
        ]
        mock_engine.get_tensor_shape.side_effect = [[1, 3, 64, 64], [1, 10, 5]]
        mock_engine.get_tensor_dtype.return_value = mock_trt.float32

        # Logger
        mock_trt.Logger.return_value = MagicMock()

        mock_cuda = MagicMock()
        mock_cuda.cuMemAlloc.return_value = (0, MagicMock())
        mock_cuda.cuStreamCreate.return_value = (0, MagicMock())

        rt = TensorRTRuntime()
        with patch.dict("sys.modules", {"tensorrt": mock_trt, "cuda": mock_cuda}):
            rt.load(engine_path)

        mock_runtime.deserialize_cuda_engine.assert_called_once()

    def test_load_onnx_builds_and_caches_engine(self, tmp_path):
        """Verify .onnx triggers engine build and saves cache."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx-data")
        engine_path = tmp_path / "model.engine"

        mock_trt = MagicMock()
        mock_logger = MagicMock()
        mock_trt.Logger.return_value = mock_logger

        # Builder + network + parser
        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser

        # Config
        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        # Serialized engine
        mock_builder.build_serialized_network.return_value = b"serialized-engine"

        # Runtime + engine from deserialization
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_trt.Runtime.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        mock_engine.num_io_tensors = 2
        mock_engine.get_tensor_name.side_effect = ["input", "output"]
        mock_engine.get_tensor_mode.side_effect = [
            mock_trt.TensorIOMode.INPUT,
            mock_trt.TensorIOMode.OUTPUT,
        ]
        mock_engine.get_tensor_shape.side_effect = [[1, 3, 64, 64], [1, 10, 5]]
        mock_engine.get_tensor_dtype.return_value = mock_trt.float32

        # NetworkDefinitionCreationFlag for EXPLICIT_BATCH
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 1

        mock_cuda = MagicMock()
        mock_cuda.cuMemAlloc.return_value = (0, MagicMock())
        mock_cuda.cuStreamCreate.return_value = (0, MagicMock())

        rt = TensorRTRuntime()
        with patch.dict("sys.modules", {"tensorrt": mock_trt, "cuda": mock_cuda}):
            rt.load(onnx_path)

        # Engine file should have been cached
        assert engine_path.exists()
        assert engine_path.read_bytes() == b"serialized-engine"

    def test_load_onnx_uses_cached_engine(self, tmp_path):
        """Verify cached .engine is loaded instead of rebuilding."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx")
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"cached-engine")

        mock_trt = MagicMock()
        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_trt.Logger.return_value = MagicMock()
        mock_trt.Runtime.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine

        mock_engine.num_io_tensors = 2
        mock_engine.get_tensor_name.side_effect = ["input", "output"]
        mock_engine.get_tensor_mode.side_effect = [
            mock_trt.TensorIOMode.INPUT,
            mock_trt.TensorIOMode.OUTPUT,
        ]
        mock_engine.get_tensor_shape.side_effect = [[1, 3, 64, 64], [1, 10, 5]]
        mock_engine.get_tensor_dtype.return_value = mock_trt.float32

        mock_cuda = MagicMock()
        mock_cuda.cuMemAlloc.return_value = (0, MagicMock())
        mock_cuda.cuStreamCreate.return_value = (0, MagicMock())

        rt = TensorRTRuntime()
        with patch.dict("sys.modules", {"tensorrt": mock_trt, "cuda": mock_cuda}):
            rt.load(onnx_path)

        # Builder should NOT have been called — cache was used
        mock_trt.Builder.assert_not_called()

    def test_load_onnx_rebuilds_on_stale_cache(self, tmp_path):
        """Verify stale cache (deserialization failure) triggers rebuild."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx")
        engine_path = tmp_path / "model.engine"
        engine_path.write_bytes(b"stale-engine")

        mock_trt = MagicMock()
        mock_logger = MagicMock()
        mock_trt.Logger.return_value = mock_logger

        mock_runtime = MagicMock()
        mock_trt.Runtime.return_value = mock_runtime

        # First deserialize fails (stale cache), second succeeds (fresh build)
        mock_engine_good = MagicMock()
        mock_engine_good.num_io_tensors = 2
        mock_engine_good.get_tensor_name.side_effect = ["input", "output"]
        mock_engine_good.get_tensor_mode.side_effect = [
            mock_trt.TensorIOMode.INPUT,
            mock_trt.TensorIOMode.OUTPUT,
        ]
        mock_engine_good.get_tensor_shape.side_effect = [
            [1, 3, 64, 64], [1, 10, 5],
        ]
        mock_engine_good.get_tensor_dtype.return_value = mock_trt.float32

        mock_runtime.deserialize_cuda_engine.side_effect = [
            None,            # stale cache fails
            mock_engine_good,  # fresh build succeeds
        ]

        # Builder for rebuild
        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_serialized_network.return_value = b"new-engine"
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 1

        mock_cuda = MagicMock()
        mock_cuda.cuMemAlloc.return_value = (0, MagicMock())
        mock_cuda.cuStreamCreate.return_value = (0, MagicMock())

        rt = TensorRTRuntime()
        with patch.dict("sys.modules", {"tensorrt": mock_trt, "cuda": mock_cuda}):
            rt.load(onnx_path)

        # Builder should have been called for rebuild
        mock_trt.Builder.assert_called_once()
        # New engine should be cached
        assert engine_path.read_bytes() == b"new-engine"

    def test_fp16_flag_sets_builder_config(self, tmp_path):
        """Verify FP16 flag is passed to builder config."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        onnx_path = tmp_path / "model.onnx"
        onnx_path.write_bytes(b"fake-onnx")

        mock_trt = MagicMock()
        mock_trt.Logger.return_value = MagicMock()

        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser

        mock_config = MagicMock()
        mock_builder.create_builder_config.return_value = mock_config

        mock_builder.build_serialized_network.return_value = b"engine"

        mock_runtime = MagicMock()
        mock_engine = MagicMock()
        mock_trt.Runtime.return_value = mock_runtime
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine
        mock_engine.num_io_tensors = 2
        mock_engine.get_tensor_name.side_effect = ["input", "output"]
        mock_engine.get_tensor_mode.side_effect = [
            mock_trt.TensorIOMode.INPUT,
            mock_trt.TensorIOMode.OUTPUT,
        ]
        mock_engine.get_tensor_shape.side_effect = [[1, 3, 64, 64], [1, 10, 5]]
        mock_engine.get_tensor_dtype.return_value = mock_trt.float32
        mock_trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH = 1
        mock_trt.BuilderFlag.FP16 = 8  # arbitrary flag value

        mock_cuda = MagicMock()
        mock_cuda.cuMemAlloc.return_value = (0, MagicMock())
        mock_cuda.cuStreamCreate.return_value = (0, MagicMock())

        # fp16=True (default)
        rt = TensorRTRuntime(fp16=True)
        with patch.dict("sys.modules", {"tensorrt": mock_trt, "cuda": mock_cuda}):
            rt.load(onnx_path)

        mock_config.set_flag.assert_any_call(8)

    def test_run_raises_before_load(self):
        """Verify run() raises if load() was not called."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        rt = TensorRTRuntime()
        with pytest.raises(RuntimeError, match="Call load\\(\\) before run\\(\\)"):
            rt.run(np.zeros((1, 3, 64, 64), dtype=np.float32))

    def test_run_fixed_batch_mismatch_raises(self):
        """Verify RuntimeError when batch size doesn't match fixed-batch engine."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        rt = TensorRTRuntime()
        # Simulate a loaded fixed-batch engine
        rt._context = MagicMock()
        rt._dynamic_batch = False
        rt._input_names = ["input"]
        rt._output_names = ["output"]
        rt._input_shapes = {"input": (1, 3, 64, 64)}
        rt._output_shapes = {"output": (1, 10, 5)}

        # Send batch size 4 to a batch-1 engine
        with pytest.raises(RuntimeError, match="Expected batch size 1, got 4"):
            rt.run(np.zeros((4, 3, 64, 64), dtype=np.float32))

    def test_run_dynamic_batch_updates_shape(self):
        """Verify dynamic-batch engine calls set_input_shape on batch change."""
        from training_toolkit.inference.runtime import TensorRTRuntime

        rt = TensorRTRuntime()
        rt._context = MagicMock()
        rt._dynamic_batch = True
        rt._last_batch_size = None
        rt._input_names = ["input"]
        rt._output_names = ["output"]
        rt._input_shapes = {"input": (-1, 3, 64, 64)}
        rt._output_shapes = {"output": (-1, 10, 5)}
        rt._device_buffers = {"input": MagicMock(), "output": MagicMock()}
        rt._stream = MagicMock()

        mock_cuda = MagicMock()
        mock_cuda.cuMemAlloc.return_value = (0, MagicMock())
        mock_cuda.cuMemcpyHtoD.return_value = None
        mock_cuda.cuMemcpyDtoH.return_value = None
        mock_cuda.cuStreamSynchronize.return_value = None

        with patch.dict("sys.modules", {"cuda": mock_cuda}):
            rt._allocate_buffers = MagicMock()
            rt.run(np.zeros((2, 3, 64, 64), dtype=np.float32))

        rt._context.set_input_shape.assert_called_with("input", [2, 3, 64, 64])


class TestCLI:
    """Test tt-infer CLI argument parsing."""

    def test_runtime_argument_parsed(self):
        from training_toolkit.scripts.infer import make_parser

        args = make_parser().parse_args(
            ["--export-dir", "x", "--images", "y", "--runtime", "trt"],
        )
        assert args.runtime == "trt"

    def test_runtime_defaults_to_ort(self):
        from training_toolkit.scripts.infer import make_parser

        args = make_parser().parse_args(
            ["--export-dir", "x", "--images", "y"],
        )
        assert args.runtime == "ort"


class TestTrainingToolkitDetectionModel:
    """Test the SAHI DetectionModel wrapper around Predictor."""

    def test_load_model_creates_predictor(self):
        """load_model() should create a Predictor and set category_mapping."""
        sahi_base = pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        with patch("training_toolkit.inference.sahi.Predictor") as MockPredictor:
            mock_predictor = MagicMock()
            mock_predictor.class_names = {0: "crack", 1: "spall"}
            MockPredictor.return_value = mock_predictor

            model = TrainingToolkitDetectionModel(
                model_path="/fake/export",
                confidence_threshold=0.3,
                load_at_init=True,
            )

            MockPredictor.assert_called_once_with(
                "/fake/export",
                runtime="ort",
                score_threshold=0.3,
            )
            assert model.model is mock_predictor
            assert model.category_mapping == {"0": "crack", "1": "spall"}

    def test_load_model_with_trt_runtime(self):
        """runtime kwarg should be forwarded to Predictor."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        with patch("training_toolkit.inference.sahi.Predictor") as MockPredictor:
            mock_predictor = MagicMock()
            mock_predictor.class_names = {0: "crack"}
            MockPredictor.return_value = mock_predictor

            TrainingToolkitDetectionModel(
                model_path="/fake/export",
                runtime="trt",
                confidence_threshold=0.5,
                load_at_init=True,
            )

            MockPredictor.assert_called_once_with(
                "/fake/export",
                runtime="trt",
                score_threshold=0.5,
            )

    def test_set_model_accepts_predictor(self):
        """set_model() should accept an existing Predictor instance."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        mock_predictor = MagicMock()
        mock_predictor.class_names = {0: "crack", 1: "spall"}

        model = TrainingToolkitDetectionModel(
            model_path=None,
            load_at_init=False,
        )
        model.set_model(mock_predictor)

        assert model.model is mock_predictor
        assert model.category_mapping == {"0": "crack", "1": "spall"}

    def test_set_model_preserves_existing_category_mapping(self):
        """set_model() should not override explicitly set category_mapping."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        mock_predictor = MagicMock()
        mock_predictor.class_names = {0: "crack", 1: "spall"}

        custom_mapping = {"0": "fissure", "1": "spalling"}
        model = TrainingToolkitDetectionModel(
            model_path=None,
            category_mapping=custom_mapping,
            load_at_init=False,
        )
        model.set_model(mock_predictor)

        assert model.category_mapping == custom_mapping

    def test_perform_inference_single_image(self):
        """perform_inference with ndarray should call predict()."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        mock_predictor = MagicMock()
        mock_predictor.class_names = {0: "crack"}
        mock_result = {
            "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
            "scores": np.array([0.9], dtype=np.float32),
            "labels": np.array([0]),
            "masks": np.ones((1, 100, 100), dtype=bool),
        }
        mock_predictor.predict.return_value = mock_result

        model = TrainingToolkitDetectionModel(
            model_path=None,
            load_at_init=False,
        )
        model.set_model(mock_predictor)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        model.perform_inference(image)

        mock_predictor.predict.assert_called_once()
        assert len(model._original_predictions) == 1
        assert model._original_predictions[0] is mock_result

    def test_perform_inference_batch(self):
        """perform_inference with list of ndarrays should call predict_batch()."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        mock_predictor = MagicMock()
        mock_predictor.class_names = {0: "crack"}
        mock_results = [
            {
                "boxes": np.array([[10, 10, 50, 50]], dtype=np.float32),
                "scores": np.array([0.9], dtype=np.float32),
                "labels": np.array([0]),
                "masks": np.ones((1, 100, 100), dtype=bool),
            },
            {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.zeros((0,), dtype=np.float32),
                "labels": np.zeros((0,), dtype=int),
                "masks": np.zeros((0, 100, 100), dtype=bool),
            },
        ]
        mock_predictor.predict_batch.return_value = mock_results

        model = TrainingToolkitDetectionModel(
            model_path=None,
            load_at_init=False,
        )
        model.set_model(mock_predictor)

        images = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]
        model.perform_inference(images)

        mock_predictor.predict_batch.assert_called_once()
        assert len(model._original_predictions) == 2
        assert model._original_predictions is mock_results

    def test_create_object_prediction_list(self):
        """_create_object_prediction_list_from_original_predictions should produce ObjectPredictions with masks."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel
        from sahi.prediction import ObjectPrediction

        mock_predictor = MagicMock()
        mock_predictor.class_names = {0: "crack", 1: "spall"}

        model = TrainingToolkitDetectionModel(
            model_path=None,
            category_mapping={"0": "crack", "1": "spall"},
            load_at_init=False,
        )
        model.set_model(mock_predictor)

        # Simulate stored predictions from perform_inference
        mask = np.zeros((50, 50), dtype=bool)
        mask[10:30, 10:30] = True
        model._original_predictions = [{
            "boxes": np.array([[5, 5, 40, 40]], dtype=np.float32),
            "scores": np.array([0.85], dtype=np.float32),
            "labels": np.array([0]),
            "masks": np.array([mask]),
        }]
        model._original_shape = [(50, 50)]

        model._create_object_prediction_list_from_original_predictions(
            shift_amount_list=[[0, 0]],
            full_shape_list=[[50, 50]],
        )

        pred_list = model._object_prediction_list_per_image
        assert len(pred_list) == 1
        assert len(pred_list[0]) == 1

        pred = pred_list[0][0]
        assert isinstance(pred, ObjectPrediction)
        assert pred.score.value == pytest.approx(0.85)
        assert pred.category.id == 0
        assert pred.mask is not None
        # Verify pixel-perfect mask is preserved
        np.testing.assert_array_equal(pred.mask.bool_mask, mask)

    def test_has_mask_returns_true(self):
        """has_mask should always return True."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        model = TrainingToolkitDetectionModel(
            model_path=None,
            load_at_init=False,
        )
        assert model.has_mask is True

    def test_set_device_is_noop(self):
        """set_device should not raise regardless of input."""
        pytest.importorskip("sahi.models.base")

        from training_toolkit.inference.sahi import TrainingToolkitDetectionModel

        model = TrainingToolkitDetectionModel(
            model_path=None,
            load_at_init=False,
        )
        # Should not raise
        model.set_device("cuda")
        model.set_device(None)
        model.set_device("cpu")


class TestCLISliced:
    """Test tt-infer CLI sliced inference arguments."""

    def test_sliced_flag_parsed(self):
        from training_toolkit.scripts.infer import make_parser

        args = make_parser().parse_args(
            ["--export-dir", "x", "--images", "y", "--sliced"],
        )
        assert args.sliced is True

    def test_sliced_defaults(self):
        from training_toolkit.scripts.infer import make_parser

        args = make_parser().parse_args(
            ["--export-dir", "x", "--images", "y", "--sliced"],
        )
        assert args.slice_height is None
        assert args.slice_width is None
        assert args.overlap_ratio == 0.2
        assert args.batch_size == 1
        assert args.postprocess_type == "GREEDYNMM"
        assert args.postprocess_match_metric == "IOS"
        assert args.postprocess_match_threshold == 0.5
        assert args.postprocess_class_agnostic is False
        assert args.perform_standard_pred is False
        assert args.auto_slice_resolution is False
        assert args.no_progress_bar is False

    def test_sliced_custom_args(self):
        from training_toolkit.scripts.infer import make_parser

        args = make_parser().parse_args([
            "--export-dir", "x", "--images", "y", "--sliced",
            "--slice-height", "256",
            "--slice-width", "256",
            "--overlap-ratio", "0.3",
            "--batch-size", "8",
            "--postprocess-type", "NONE",
            "--postprocess-match-metric", "IOU",
            "--postprocess-match-threshold", "0.6",
            "--postprocess-class-agnostic",
            "--perform-standard-pred",
            "--auto-slice-resolution",
            "--no-progress-bar",
        ])
        assert args.slice_height == 256
        assert args.slice_width == 256
        assert args.overlap_ratio == 0.3
        assert args.batch_size == 8
        assert args.postprocess_type == "NONE"
        assert args.postprocess_match_metric == "IOU"
        assert args.postprocess_match_threshold == 0.6
        assert args.postprocess_class_agnostic is True
        assert args.perform_standard_pred is True
        assert args.auto_slice_resolution is True
        assert args.no_progress_bar is True

    def test_sliced_not_set_by_default(self):
        from training_toolkit.scripts.infer import make_parser

        args = make_parser().parse_args(
            ["--export-dir", "x", "--images", "y"],
        )
        assert args.sliced is False


class TestPredictSliced:
    """Test Predictor.predict_sliced() with mocked SAHI."""

    def test_predict_sliced_returns_standard_dict(self):
        """predict_sliced should return dict with boxes, scores, labels, masks."""
        pytest.importorskip("sahi")

        from training_toolkit.inference.predictor import Predictor
        from sahi.prediction import ObjectPrediction, PredictionResult
        from sahi.utils.cv import get_coco_segmentation_from_bool_mask

        # Create mock ObjectPredictions with mask
        mask1 = np.zeros((100, 100), dtype=bool)
        mask1[10:30, 10:30] = True
        seg1 = get_coco_segmentation_from_bool_mask(mask1)

        pred = ObjectPrediction(
            bbox=[10, 10, 30, 30],
            category_id=0,
            score=0.9,
            segmentation=seg1,
            category_name="crack",
            shift_amount=[0, 0],
            full_shape=[100, 100],
        )
        # Attach pixel-perfect bool mask cache
        pred.mask._bool_mask = mask1

        mock_result = MagicMock(spec=PredictionResult)
        mock_result.object_prediction_list = [pred]

        with patch.object(Predictor, "__init__", lambda self, *a, **kw: None):
            predictor = Predictor.__new__(Predictor)
            predictor._resolution = (64, 64)
            predictor._score_threshold = 0.5
            predictor._class_names = None

            with patch(
                "sahi.predict.get_sliced_prediction",
                return_value=mock_result,
            ):
                result = predictor.predict_sliced(
                    np.zeros((100, 100, 3), dtype=np.uint8),
                )

        assert set(result.keys()) == {"boxes", "scores", "labels", "masks"}
        assert result["boxes"].shape == (1, 4)
        assert result["boxes"].dtype == np.float32
        assert result["scores"].shape == (1,)
        assert result["scores"].dtype == np.float32
        assert result["labels"].shape == (1,)
        assert result["masks"].shape == (1, 100, 100)
        assert result["masks"].dtype == bool

    def test_predict_sliced_empty_result(self):
        """predict_sliced with no detections should return correctly shaped empty arrays."""
        pytest.importorskip("sahi")

        from training_toolkit.inference.predictor import Predictor
        from sahi.prediction import PredictionResult

        mock_result = MagicMock(spec=PredictionResult)
        mock_result.object_prediction_list = []

        with patch.object(Predictor, "__init__", lambda self, *a, **kw: None):
            predictor = Predictor.__new__(Predictor)
            predictor._resolution = (64, 64)
            predictor._score_threshold = 0.5
            predictor._class_names = None

            with patch(
                "sahi.predict.get_sliced_prediction",
                return_value=mock_result,
            ):
                image = np.zeros((100, 100, 3), dtype=np.uint8)
                result = predictor.predict_sliced(image)

        assert result["boxes"].shape == (0, 4)
        assert result["scores"].shape == (0,)
        assert result["labels"].shape == (0,)
        assert result["masks"].shape == (0, 100, 100)

    def test_predict_sliced_default_slice_size(self):
        """When slice_height/width are None, should default to model resolution."""
        pytest.importorskip("sahi")

        from training_toolkit.inference.predictor import Predictor
        from sahi.prediction import PredictionResult

        mock_result = MagicMock(spec=PredictionResult)
        mock_result.object_prediction_list = []

        with patch.object(Predictor, "__init__", lambda self, *a, **kw: None):
            predictor = Predictor.__new__(Predictor)
            predictor._resolution = (504, 504)
            predictor._score_threshold = 0.5
            predictor._class_names = None

            with patch(
                "sahi.predict.get_sliced_prediction",
                return_value=mock_result,
            ) as mock_get:
                predictor.predict_sliced(
                    np.zeros((1000, 1000, 3), dtype=np.uint8),
                )

            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["slice_height"] == 504
            assert call_kwargs["slice_width"] == 504

    def test_predict_sliced_passes_sahi_params(self):
        """All SAHI parameters should be forwarded correctly."""
        pytest.importorskip("sahi")

        from training_toolkit.inference.predictor import Predictor
        from sahi.prediction import PredictionResult

        mock_result = MagicMock(spec=PredictionResult)
        mock_result.object_prediction_list = []

        with patch.object(Predictor, "__init__", lambda self, *a, **kw: None):
            predictor = Predictor.__new__(Predictor)
            predictor._resolution = (64, 64)
            predictor._score_threshold = 0.5
            predictor._class_names = None

            with patch(
                "sahi.predict.get_sliced_prediction",
                return_value=mock_result,
            ) as mock_get:
                predictor.predict_sliced(
                    np.zeros((200, 200, 3), dtype=np.uint8),
                    slice_height=128,
                    slice_width=128,
                    overlap_ratio=0.3,
                    batch_size=4,
                    postprocess_type="NMS",
                    postprocess_match_metric="IOS",
                    postprocess_match_threshold=0.6,
                    postprocess_class_agnostic=True,
                    perform_standard_pred=True,
                    auto_slice_resolution=True,
                    progress_bar=False,
                    verbose=2,
                )

            call_kwargs = mock_get.call_args[1]
            assert call_kwargs["slice_height"] == 128
            assert call_kwargs["slice_width"] == 128
            assert call_kwargs["overlap_height_ratio"] == 0.3
            assert call_kwargs["overlap_width_ratio"] == 0.3
            assert call_kwargs["postprocess_type"] == "NMS"
            assert call_kwargs["postprocess_match_metric"] == "IOS"
            assert call_kwargs["postprocess_match_threshold"] == 0.6
            assert call_kwargs["postprocess_class_agnostic"] is True
            assert call_kwargs["perform_standard_pred"] is True
            assert call_kwargs["auto_slice_resolution"] is True
            assert call_kwargs["progress_bar"] is False
            assert call_kwargs["verbose"] == 2

    def test_predict_sliced_import_error(self):
        """Should raise clear ImportError when sahi is not installed."""
        from training_toolkit.inference.predictor import Predictor

        with patch.object(Predictor, "__init__", lambda self, *a, **kw: None):
            predictor = Predictor.__new__(Predictor)
            predictor._resolution = (64, 64)
            predictor._score_threshold = 0.5

            with patch.dict("sys.modules", {"sahi": None, "sahi.predict": None}):
                with pytest.raises(ImportError, match="sahi"):
                    predictor.predict_sliced(
                        np.zeros((100, 100, 3), dtype=np.uint8),
                    )
