"""Top-level training configuration."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml

from .model_config import ModelConfig
from .data_config import DataConfig
from .augmentation_config import AugConfig


@dataclass
class TrainConfig:
    """Top-level config composing model, data, and augmentation settings."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugConfig = field(default_factory=AugConfig)

    epochs: int = 200
    lr: float = 1e-4
    weight_decay: float = 0.01
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    warmup_epochs: int = 3

    nbs: Optional[int] = None  # nominal batch size — accumulate grad steps to simulate this batch size (None = no accumulation)

    patience: int = 50
    monitor_metric: str = "val/map"

    save_top_k: int = 3
    save_period: int = -1
    save_best_per_class: bool = False  # if true, write a separate best-checkpoint per per-class val metric (hardlinked for dedup)

    device: list[int] = field(default_factory=lambda: [0])
    precision: str = "bf16-mixed"
    gradient_clip_val: Optional[float] = None  # max gradient norm (None = no clipping)
    compile: bool = False  # torch.compile() the model for ~5-15% speedup (slow first step)
    masked_loss: bool = False  # if true, the criterion honors per-sample `class_mask` fields on targets — classes with mask=0 contribute zero gradient. Currently only the RF-DETR adapter supports this.
    region_class_mask: bool = False  # if true, masked loss also honors region-level class masks. RF-DETR adapter only.
    size_band_ignore: bool = False  # if true, the (online-tiling) criterion drops out-of-band GTs from matching and zeros the no-object loss for unmatched ignore-region overlaps. RF-DETR adapter only.
    size_band_ceiling_frac: float | None = 0.75  # size-band ceiling as a fraction of imgsz; None/null -> floor-only banding (no ceiling). Read only when size_band_ignore is true.
    ignore_ioa_thresh: float = 0.5  # IoA above which an unmatched query counts as overlapping an ignore region

    fraction: float = 1.0  # fraction of dataset to use (useful for smoke tests)
    limit_train_batches: Optional[int] = None  # absolute cap on train+val batches per epoch (int); None = no cap. Unlike `fraction` (a float ratio) it does NOT shrink the pool, and it takes precedence over `fraction`. Mirrors Lightning's limit_train_batches.
    limit_val_batches: Optional[int | float] = None  # INDEPENDENT cap on VAL batches per epoch, overriding the train-coupled default. int = absolute batch count; float in (0, 1] = fraction (1.0 = full val). Enables e.g. short train + full val. Mirrors Lightning's limit_val_batches.

    project_dir: str = "runs"
    run_name: str = "run"
    wandb_group: str | None = None  # optional W&B `group` (bundles related runs in the project sidebar); None = ungrouped
    overwrite: bool = False  # wipe run directory if it already exists
    log_to_file: bool = True
    loggers: list[str] = field(default_factory=lambda: ["mlflow", "wandb"])
    log_images: bool = False  # log val images to W&B (requires wandb logger)
    verbose_metrics: bool = True
    val_tide: bool = False  # if true, run tidecv error analysis at every val epoch — adds aggregate dAP metrics (val/tide_dap/{cls,loc,...}) and dumps tide_summary.txt + per-class table to run dir
    analysis: bool = True  # generate pre-training dataset analysis plots and batch samples
    diagnostics_period: int = 5  # epoch interval for results.png and val visualizations
    seed: Optional[int] = 42

    def __post_init__(self):
        if self.region_class_mask and not self.masked_loss:
            raise ValueError("region_class_mask requires masked_loss=True")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be > 0, got {self.epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.limit_train_batches is not None and self.limit_train_batches < 1:
            raise ValueError(
                "limit_train_batches must be >= 1 (an absolute batch count), "
                f"got {self.limit_train_batches}"
            )
        v = self.limit_val_batches
        if v is not None:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"limit_val_batches must be an int or float, got {v!r}")
            if isinstance(v, int):
                if v < 1:
                    raise ValueError(
                        "limit_val_batches as an int must be >= 1 (an absolute batch "
                        f"count), got {v}"
                    )
            elif not (0.0 < v <= 1.0):
                raise ValueError(
                    "limit_val_batches as a float must be in (0.0, 1.0] (a fraction; "
                    f"1.0 = full val), got {v}"
                )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        with open(path) as fh:
            data = yaml.safe_load(fh)
        data["model"] = ModelConfig(**data.get("model", {}))
        raw_data = data.get("data", {})
        # Handle nested tiling config
        if "tiling" in raw_data and isinstance(raw_data["tiling"], dict):
            from .tiling_config import TilingConfig
            raw_data["tiling"] = TilingConfig(**raw_data["tiling"])
        data["data"] = DataConfig(**raw_data)
        data["augmentation"] = AugConfig(**data.get("augmentation", {}))

        # Coerce numeric fields that may be loaded as strings from YAML (e.g. lr: "1e-4")
        int_keys = (
            "epochs", "warmup_epochs", "patience", "save_top_k", "save_period",
            "diagnostics_period", "limit_train_batches",
        )
        # device: accept int (e.g. 2 → [2]) or list of ints
        if "device" in data:
            raw = data["device"]
            if isinstance(raw, int):
                data["device"] = [raw]
            elif isinstance(raw, list):
                data["device"] = [int(d) for d in raw]
        float_keys = ("lr", "weight_decay", "fraction")
        for key in int_keys:
            if key in data and data[key] is not None:
                data[key] = int(data[key])
        for key in float_keys:
            if key in data and data[key] is not None:
                data[key] = float(data[key])

        # limit_val_batches keeps Lightning's int(=batches)-vs-float(=fraction)
        # distinction, so coerce a quoted YAML string without collapsing the two.
        if isinstance(data.get("limit_val_batches"), str):
            v = data["limit_val_batches"]
            data["limit_val_batches"] = float(v) if ("." in v or "e" in v.lower()) else int(v)

        # seed supports None (null in YAML or the string "none")
        if "seed" in data:
            raw = data["seed"]
            if raw is None or (isinstance(raw, str) and raw.lower() == "none"):
                data["seed"] = None
            else:
                data["seed"] = int(raw)

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = asdict(self)
        d["data"]["dataset_path"] = str(d["data"]["dataset_path"])
        with open(path, "w") as fh:
            yaml.dump(d, fh, default_flow_style=False, sort_keys=False)
