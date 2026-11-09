"""Custom collate function for instance segmentation batches."""

from __future__ import annotations

import numpy as np
import torch


def collate_fn(batch: list[dict]) -> dict:
    """Collate variable-size instance segmentation samples.

    Each sample in *batch* is expected to have:
    - ``"image"``: (H, W, 3) uint8 numpy array (RGB)
    - ``"masks"``: list of (H, W) uint8 numpy arrays
    - ``"boxes"``: (N, 4) float32 numpy array (xyxy)
    - ``"labels"``: (N,) int64 numpy array

    Returns
    -------
    dict with:
        images : torch.Tensor, shape (B, 3, H, W), dtype float32, values in [0, 1]
        targets : list[dict] each containing
            - ``"boxes"``  : torch.Tensor (N, 4) float32
            - ``"labels"`` : torch.Tensor (N,) int64
            - ``"masks"``  : torch.Tensor (M, H, W) uint8
    """
    images: list[torch.Tensor] = []
    targets: list[dict[str, torch.Tensor]] = []

    for sample in batch:
        # HWC uint8 -> CHW float32 [0, 1]
        img = torch.from_numpy(sample["image"].transpose(2, 0, 1).copy()).float() / 255.0
        images.append(img)

        # Masks: list of (H, W) numpy -> stacked tensor (M, H, W)
        if sample["masks"]:
            masks_tensor = torch.from_numpy(np.stack(sample["masks"]))
        else:
            h, w = sample["image"].shape[:2]
            masks_tensor = torch.zeros((0, h, w), dtype=torch.uint8)

        target: dict[str, torch.Tensor] = {
            "boxes": torch.from_numpy(np.asarray(sample["boxes"], dtype=np.float32)),
            "labels": torch.from_numpy(np.asarray(sample["labels"], dtype=np.int64)),
            "masks": masks_tensor,
        }
        # Propagate optional per-sample class_mask (for masked-loss training).
        # Absent samples default to no mask downstream → criterion treats all
        # classes as active for that sample.
        if "class_mask" in sample:
            target["class_mask"] = torch.from_numpy(
                np.asarray(sample["class_mask"], dtype=np.float32)
            )
        # Forward region metadata independently; the enabled criterion owns
        # paired-key validation so malformed metadata stays inert otherwise.
        if "region_boxes" in sample:
            target["region_boxes"] = torch.from_numpy(
                np.asarray(sample["region_boxes"], dtype=np.float32)
            )
        if "region_class_masks" in sample:
            target["region_class_masks"] = torch.from_numpy(
                np.asarray(sample["region_class_masks"], dtype=np.float32)
            )
        # Propagate optional per-instance ignore flags (size-band SNIP ignore).
        # Absent samples default to no ignores downstream.
        if "ignore" in sample:
            target["ignore"] = torch.from_numpy(
                np.asarray(sample["ignore"], dtype=bool)
            )
        # Propagate optional per-instance COCO iscrowd flags (crowd/sliver GT).
        # Absent samples default to no crowds downstream (torchmetrics MAP
        # treats a missing "iscrowd" key as all-zeros — see _format_targets_for_metrics).
        if "iscrowd" in sample:
            target["iscrowd"] = torch.from_numpy(
                np.asarray(sample["iscrowd"], dtype=np.int64))
        targets.append(target)

    return {
        "images": torch.stack(images),
        "targets": targets,
    }
