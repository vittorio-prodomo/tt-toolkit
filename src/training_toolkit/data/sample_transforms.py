"""Sample-level transforms applied after loading, before augmentation."""

import numpy as np


def merge_overlapping_masks(sample: dict) -> dict:
    """Merge same-class instance masks that share any overlapping pixels.

    Uses union-find to handle chain reactions: if A overlaps B and B overlaps C,
    all three merge into one instance even if A and C don't touch.

    Boxes are recomputed from the merged masks.
    """
    masks: list[np.ndarray] = sample["masks"]
    labels: np.ndarray = sample["labels"]

    if len(masks) <= 1:
        return sample

    n = len(masks)

    # --- Union-Find ---
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Group indices by class label
    label_to_indices: dict[int, list[int]] = {}
    for i, lbl in enumerate(labels):
        label_to_indices.setdefault(int(lbl), []).append(i)

    # Within each class, union masks that overlap
    for indices in label_to_indices.values():
        if len(indices) <= 1:
            continue
        for i_pos in range(len(indices)):
            for j_pos in range(i_pos + 1, len(indices)):
                mi, mj = indices[i_pos], indices[j_pos]
                if find(mi) == find(mj):
                    continue  # already in the same component
                if np.any(masks[mi] & masks[mj]):
                    union(mi, mj)

    # Collect connected components
    components: dict[int, list[int]] = {}
    for i in range(n):
        components.setdefault(find(i), []).append(i)

    # Build merged output
    new_masks: list[np.ndarray] = []
    new_labels: list[int] = []
    new_boxes: list[list[float]] = []

    for members in components.values():
        merged = masks[members[0]].copy()
        for idx in members[1:]:
            merged = merged | masks[idx]

        ys, xs = np.where(merged)
        if len(xs) == 0:
            continue

        new_masks.append(merged)
        new_labels.append(int(labels[members[0]]))
        new_boxes.append([float(xs.min()), float(ys.min()),
                          float(xs.max()), float(ys.max())])

    sample["masks"] = new_masks
    sample["labels"] = np.array(new_labels, dtype=labels.dtype)
    sample["boxes"] = (
        np.array(new_boxes, dtype=np.float32)
        if new_boxes
        else np.zeros((0, 4), dtype=np.float32)
    )
    return sample
