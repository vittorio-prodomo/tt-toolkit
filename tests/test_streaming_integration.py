"""Integration tests for streaming tiling path wired into the trainer."""

import pytest


def test_single_gpu_streaming_guard_removed():
    """The single-GPU streaming guard was lifted once per-rank DDP sharding
    shipped — the function must no longer exist on the trainer module."""
    import training_toolkit.engine.trainer as T
    assert not hasattr(T, "_assert_streaming_single_gpu")


def test_create_dataset_online_returns_streaming_with_multi_gpu_config(mini_coco):
    """Building the streaming dataset via _create_dataset works with a
    multi-GPU device list in the config — construction does not depend on
    device count."""
    from training_toolkit.engine.trainer import _create_dataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    from training_toolkit.configs import TrainConfig, ModelConfig, DataConfig, TilingConfig
    cfg = TrainConfig(
        model=ModelConfig(name="rf-detr-seg-large"),
        data=DataConfig(
            dataset_path=str(mini_coco), layout="coco", train_split="train",
            workers=0,
            tiling=TilingConfig(enabled=True, online=True, tile_size=32,
                                negative_ratio=0.0, buffer_size=4),
        ),
        device=[0, 1],
    )
    ds = _create_dataset(cfg, cfg.data.train_split, transform=None, aug_config=None)
    assert isinstance(ds, StreamingTiledSegmentationDataset)


def test_create_dataset_returns_streaming_when_online(mini_coco):
    from training_toolkit.engine.trainer import _create_dataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    from training_toolkit.configs import TrainConfig, ModelConfig, DataConfig, TilingConfig

    cfg = TrainConfig(
        model=ModelConfig(name="rf-detr-seg-large"),
        data=DataConfig(
            dataset_path=str(mini_coco),
            layout="coco",
            train_split="train",
            workers=0,
            tiling=TilingConfig(
                enabled=True,
                online=True,
                tile_size=32,
                negative_ratio=0.0,
                buffer_size=4,
            ),
        ),
    )
    ds = _create_dataset(cfg, cfg.data.train_split, transform=None, aug_config=None)
    assert isinstance(ds, StreamingTiledSegmentationDataset)


def test_create_dataset_online_masked_loss_now_succeeds(mini_coco):
    """After lifting the guard, online=True + masked_loss=True should construct
    the streaming dataset without error (class_mask is now propagated by the
    streaming path, so masked_loss is no longer a silent no-op)."""
    from training_toolkit.engine.trainer import _create_dataset
    from training_toolkit.data.streaming_tiled_dataset import StreamingTiledSegmentationDataset
    from training_toolkit.configs import TrainConfig, ModelConfig, DataConfig, TilingConfig

    cfg = TrainConfig(
        model=ModelConfig(name="rf-detr-seg-large"),
        data=DataConfig(
            dataset_path=str(mini_coco),
            layout="coco",
            train_split="train",
            workers=0,
            tiling=TilingConfig(
                enabled=True,
                online=True,
                tile_size=32,
                negative_ratio=0.0,
                buffer_size=4,
            ),
        ),
        masked_loss=True,
    )
    ds = _create_dataset(cfg, cfg.data.train_split, transform=None, aug_config=None)
    assert isinstance(ds, StreamingTiledSegmentationDataset)
