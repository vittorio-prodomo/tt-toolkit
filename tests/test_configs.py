"""Tests for configuration dataclasses."""

from pathlib import Path

import pytest

from training_toolkit.configs import TrainConfig, ModelConfig, DataConfig, AugConfig


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig(name="mask2former-swin-base")
        assert cfg.name == "mask2former-swin-base"
        assert cfg.backend is None
        assert cfg.pretrained == "auto"

    def test_backend_override(self):
        cfg = ModelConfig(name="custom", backend="mask2former")
        assert cfg.backend == "mask2former"


class TestDataConfig:
    def test_path_coerced_to_pathlib(self):
        cfg = DataConfig(dataset_path="/some/path")
        assert isinstance(cfg.dataset_path, Path)

    def test_default_layout_is_coco(self):
        cfg = DataConfig(dataset_path="/x")
        assert cfg.layout == "coco"

    def test_roboflow_layout(self):
        cfg = DataConfig(dataset_path="/x", layout="roboflow")
        assert cfg.layout == "roboflow"

    def test_invalid_layout_raises(self):
        with pytest.raises(ValueError, match="layout must be"):
            DataConfig(dataset_path="/x", layout="unknown")

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError, match="batch_size"):
            DataConfig(dataset_path="/x", batch_size=0)


class TestAugConfig:
    def test_defaults(self):
        cfg = AugConfig()
        assert cfg.imgsz is None  # auto-detected from model variant at train time
        assert cfg.mosaic_p == 0.0
        assert cfg.cutmix_p == 0.0
        assert cfg.mixup_p == 0.0
        assert cfg.mixup_alpha == 1.5
        assert cfg.copy_paste_p == 0.0
        assert cfg.copy_paste_max_instances == 5
        assert cfg.random_resized_crop is False
        assert cfg.crop_scale_min == 0.5
        assert cfg.crop_scale_max == 1.0
        assert cfg.horizontal_flip == 0.0
        assert cfg.vertical_flip == 0.0
        assert cfg.rotate_degrees == 0.0
        assert cfg.translate == 0.0
        assert cfg.shear == 0.0
        assert cfg.color_jitter == 0.0

    def test_multi_sample_probabilities_must_sum_to_at_most_one(self):
        with pytest.raises(ValueError, match="sum"):
            AugConfig(mosaic_p=0.5, cutmix_p=0.3, mixup_p=0.3)

    def test_multi_sample_probabilities_at_boundary(self):
        cfg = AugConfig(mosaic_p=0.5, cutmix_p=0.3, mixup_p=0.2)
        assert cfg.mosaic_p + cfg.cutmix_p + cfg.mixup_p == pytest.approx(1.0)

    def test_mosaic_without_random_resized_crop_warns(self):
        # Warning now fires only when *both* crop pipelines are disabled —
        # mosaic_crop=True (the default) provides a RandomResizedCrop after
        # mosaic so no warning is needed in that case.
        with pytest.warns(UserWarning, match="random_resized_crop"):
            AugConfig(mosaic_p=0.5, random_resized_crop=False, mosaic_crop=False)

    def test_multi_scale_defaults(self):
        cfg = AugConfig()
        assert cfg.multi_scale is False
        assert cfg.multi_scale_min == 0.7
        assert cfg.multi_scale_max == 1.4


class TestMultiScaleSizes:
    def test_rfdetr_seg_medium_divisor_24(self):
        from training_toolkit.models.rfdetr import RFDETRAdapter
        adapter = RFDETRAdapter()
        cfg = AugConfig(imgsz=432, multi_scale=True)
        sizes = adapter.compute_multi_scale_sizes(cfg, "rf-detr-seg-medium")
        assert all(s % 24 == 0 for s in sizes)
        assert 432 in sizes
        assert min(sizes) >= 24 * 2

    def test_rfdetr_seg_nano_divisor_12(self):
        from training_toolkit.models.rfdetr import RFDETRAdapter
        adapter = RFDETRAdapter()
        cfg = AugConfig(imgsz=312, multi_scale=True)
        sizes = adapter.compute_multi_scale_sizes(cfg, "rf-detr-seg-nano")
        assert all(s % 12 == 0 for s in sizes)
        assert 312 in sizes

    def test_mask2former_divisor_32(self):
        from training_toolkit.models.mask2former import Mask2FormerAdapter
        adapter = Mask2FormerAdapter()
        cfg = AugConfig(imgsz=512, multi_scale=True)
        sizes = adapter.compute_multi_scale_sizes(cfg, "mask2former-swin-base")
        assert all(s % 32 == 0 for s in sizes)
        assert 512 in sizes

    def test_disabled_returns_empty(self):
        from training_toolkit.models.base import ModelAdapter
        cfg = AugConfig(imgsz=640, multi_scale=False)
        # compute_multi_scale_sizes still returns sizes; the LitModule
        # checks multi_scale flag to decide whether to use them.
        # Just verify the method works on the base class.
        class DummyAdapter(ModelAdapter):
            IMGSZ_DIVISOR = 32
            def create_model(self, config): ...
            def create_criterion(self, config): ...
            def format_batch(self, batch): ...
            def extract_predictions(self, outputs): ...
            def compute_loss(self, outputs, targets): ...
        adapter = DummyAdapter()
        sizes = adapter.compute_multi_scale_sizes(cfg)
        assert all(s % 32 == 0 for s in sizes)


class TestTrainConfig:
    def test_from_yaml_roundtrip(self, tmp_path):
        original = TrainConfig(
            model=ModelConfig(name="rf-detr-base"),
            data=DataConfig(dataset_path="/data/test"),
            augmentation=AugConfig(imgsz=800),
            epochs=100,
            lr=5e-5,
        )
        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)
        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.model.name == "rf-detr-base"
        assert loaded.data.dataset_path == Path("/data/test")
        assert loaded.augmentation.imgsz == 800
        assert loaded.epochs == 100
        assert loaded.lr == 5e-5

    def test_invalid_epochs(self):
        with pytest.raises(ValueError, match="epochs"):
            TrainConfig(
                model=ModelConfig(name="x"),
                data=DataConfig(dataset_path="/x"),
                epochs=0,
            )

    def test_defaults(self):
        cfg = TrainConfig(
            model=ModelConfig(name="x"),
            data=DataConfig(dataset_path="/x"),
        )
        assert cfg.optimizer == "AdamW"
        assert cfg.scheduler == "cosine"
        assert cfg.patience == 50
        assert cfg.precision == "bf16-mixed"
        assert cfg.seed == 42
        assert cfg.val_tide is False  # opt-in; tidecv error analysis adds ~1s per val epoch


class TestLimitTrainBatches:
    """The --limit-train-batches knob: an absolute integer batch cap applied to
    BOTH train and val (like --fraction's single value), distinct from --fraction
    (a float ratio). It does not shrink the pool and takes precedence over fraction.
    """

    def _cfg(self, fraction=1.0, limit=None, limit_val=None):
        kwargs = dict(
            model=ModelConfig(name="x"),
            data=DataConfig(dataset_path="/x"),
            fraction=fraction,
            limit_train_batches=limit,
        )
        # Only pass when set so existing tests don't depend on the new field existing.
        if limit_val is not None:
            kwargs["limit_val_batches"] = limit_val
        return TrainConfig(**kwargs)

    # --- config field ---
    def test_default_is_none(self):
        cfg = TrainConfig(model=ModelConfig(name="x"), data=DataConfig(dataset_path="/x"))
        assert cfg.limit_train_batches is None

    def test_accepts_positive_int(self):
        assert self._cfg(limit=50).limit_train_batches == 50

    def test_rejects_zero(self):
        with pytest.raises(ValueError, match="limit_train_batches"):
            self._cfg(limit=0)

    def test_rejects_negative(self):
        with pytest.raises(ValueError, match="limit_train_batches"):
            self._cfg(limit=-5)

    def test_yaml_coerces_string_to_int(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "model:\n  name: rf-detr-base\n"
            "data:\n  dataset_path: /data/test\n"
            'limit_train_batches: "50"\n'
        )
        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.limit_train_batches == 50
        assert isinstance(loaded.limit_train_batches, int)

    def test_yaml_val_coerces_string_to_int(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "model:\n  name: rf-detr-base\n"
            "data:\n  dataset_path: /data/test\n"
            'limit_val_batches: "50"\n'
        )
        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.limit_val_batches == 50
        assert isinstance(loaded.limit_val_batches, int)

    def test_yaml_val_coerces_string_to_float(self, tmp_path):
        yaml_path = tmp_path / "config.yaml"
        yaml_path.write_text(
            "model:\n  name: rf-detr-base\n"
            "data:\n  dataset_path: /data/test\n"
            'limit_val_batches: "1.0"\n'
        )
        loaded = TrainConfig.from_yaml(yaml_path)
        assert loaded.limit_val_batches == 1.0
        assert isinstance(loaded.limit_val_batches, float)

    # --- CLI override ---
    def test_cli_override_maps_flag(self):
        from training_toolkit.scripts.train import make_parser, apply_overrides
        args = make_parser().parse_args(["--limit-train-batches", "50"])
        cfg = apply_overrides(
            TrainConfig(model=ModelConfig(name="x"), data=DataConfig(dataset_path="/x")), args
        )
        assert cfg.limit_train_batches == 50

    def test_cli_default_leaves_none(self):
        from training_toolkit.scripts.train import make_parser, apply_overrides
        args = make_parser().parse_args([])
        cfg = apply_overrides(
            TrainConfig(model=ModelConfig(name="x"), data=DataConfig(dataset_path="/x")), args
        )
        assert cfg.limit_train_batches is None

    # --- trainer limit resolution (precedence) ---
    def test_resolve_fraction_only(self):
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(fraction=0.5), has_val=True) == (0.5, 0.5)

    def test_resolve_no_limits(self):
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(), has_val=True) == (None, None)

    def test_resolve_int_limit(self):
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(limit=50), has_val=True) == (50, 50)

    def test_resolve_int_limit_overrides_fraction(self, caplog):
        import logging
        from training_toolkit.engine.trainer import _resolve_batch_limits
        with caplog.at_level(logging.WARNING):
            result = _resolve_batch_limits(self._cfg(fraction=0.5, limit=50), has_val=True)
        assert result == (50, 50)
        assert "limit-train-batches" in caplog.text.lower()

    def test_resolve_val_disabled_when_no_val(self):
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(fraction=0.5), has_val=False) == (0.5, 0)

    def test_resolve_int_limit_val_disabled(self):
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(limit=50), has_val=False) == (50, 0)

    # --- decoupled val limit (--limit-val-batches override) ---
    def test_resolve_val_override_alone(self):
        # val capped independently; train runs full
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(limit_val=50), has_val=True) == (None, 50)

    def test_resolve_val_override_decouples_from_train_limit(self):
        # --limit-train-batches 100 --limit-val-batches 1.0 -> short train, FULL val
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(limit=100, limit_val=1.0), has_val=True) == (100, 1.0)

    def test_resolve_val_override_float_fraction(self):
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(limit_val=0.5), has_val=True) == (None, 0.5)

    def test_resolve_val_override_disabled_when_no_val(self):
        # val override is moot when there is no val split (stays 0/disabled)
        from training_toolkit.engine.trainer import _resolve_batch_limits
        assert _resolve_batch_limits(self._cfg(limit_val=50), has_val=False) == (None, 0)

    # --- limit_val_batches config field ---
    def test_limit_val_default_none(self):
        cfg = TrainConfig(model=ModelConfig(name="x"), data=DataConfig(dataset_path="/x"))
        assert cfg.limit_val_batches is None

    def test_limit_val_accepts_int_and_float(self):
        # int stays int (batch count), float stays float (fraction) — must NOT coerce
        assert self._cfg(limit_val=50).limit_val_batches == 50
        assert isinstance(self._cfg(limit_val=50).limit_val_batches, int)
        assert self._cfg(limit_val=1.0).limit_val_batches == 1.0
        assert isinstance(self._cfg(limit_val=1.0).limit_val_batches, float)

    def test_limit_val_rejects_zero_int(self):
        with pytest.raises(ValueError, match="limit_val_batches"):
            self._cfg(limit_val=0)

    def test_limit_val_rejects_fraction_over_one(self):
        with pytest.raises(ValueError, match="limit_val_batches"):
            self._cfg(limit_val=1.5)

    # --- CLI ---
    def test_cli_limit_val_int(self):
        from training_toolkit.scripts.train import make_parser, apply_overrides
        args = make_parser().parse_args(["--limit-val-batches", "50"])
        cfg = apply_overrides(
            TrainConfig(model=ModelConfig(name="x"), data=DataConfig(dataset_path="/x")), args
        )
        assert cfg.limit_val_batches == 50 and isinstance(cfg.limit_val_batches, int)

    def test_cli_limit_val_full(self):
        from training_toolkit.scripts.train import make_parser, apply_overrides
        args = make_parser().parse_args(["--limit-val-batches", "1.0"])
        cfg = apply_overrides(
            TrainConfig(model=ModelConfig(name="x"), data=DataConfig(dataset_path="/x")), args
        )
        assert cfg.limit_val_batches == 1.0 and isinstance(cfg.limit_val_batches, float)


def test_tiling_config_online_defaults():
    from training_toolkit.configs import TilingConfig
    cfg = TilingConfig()
    assert cfg.online is False
    assert cfg.buffer_size == 4096  # 2026-06-09 parity study: 4096 ≈ offline convergence (256 lagged)
    assert cfg.base_seed == 42


def test_tiling_config_rejects_bad_buffer_size():
    import pytest
    from training_toolkit.configs import TilingConfig
    with pytest.raises(ValueError):
        TilingConfig(buffer_size=0)


def test_tiling_config_pool_min_windows_default_and_validation():
    import pytest
    from training_toolkit.configs import TilingConfig
    assert TilingConfig().pool_min_windows == 1  # default: pool every image (Option C unchanged)
    with pytest.raises(ValueError):
        TilingConfig(pool_min_windows=0)
