"""Header-level pre-train guard: declared COCO dims must equal decoded dims and no
rotating EXIF tag may remain.

An image with a rotating EXIF orientation tag decodes to different dims under cv2
(auto-rotates) vs pyvips (does not); if its declared COCO dims then disagree with the
decoder in use, the tile grid emits out-of-bounds crops that get silently dropped
(a real crash at the first validation of a training run). This check fails fast at pre-train instead.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # header-only reads on large source images
_ROTATING = {5, 6, 7, 8}


def _check_one(item):
    path, dw, dh = item
    try:
        with Image.open(path) as im:
            rw, rh = im.size
            o = im.getexif().get(0x0112)
    except Exception as e:  # noqa: BLE001
        return (path, f"unreadable: {e}")
    if o in _ROTATING:
        return (path, f"rotating EXIF orientation {o} (raw {rw}x{rh}, declared {dw}x{dh})")
    if (rw, rh) != (dw, dh):
        return (path, f"decoded {rw}x{rh} != declared {dw}x{dh}")
    return None


def check_declared_dims(items, workers=8):
    """items: iterable of (path, declared_w, declared_h). Returns [(path, reason)] for
    each image with a rotating EXIF tag or decoded-vs-declared dim mismatch."""
    items = list(items)
    if not items:
        return []
    workers = max(1, min(workers, len(items)))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(_check_one, items, chunksize=64)
    return [r for r in results if r is not None]


def assert_declared_dims(items, workers=8):
    """Raise ValueError (first <=20 offenders) if any image fails the dim-sanity check."""
    violations = check_declared_dims(items, workers)
    if violations:
        head = "\n".join(f"  {p}: {r}" for p, r in violations[:20])
        raise ValueError(
            f"Dataset dim-sanity check failed for {len(violations)} image(s):\n{head}\n"
            "(EXIF orientation / declared-vs-decoded mismatch — normalize the source, "
            "e.g. bake the EXIF rotation into the pixels and drop the tag)"
        )


def records_from_dataset(dataset):
    """(path, w, h) records from a CocoSegmentationDataset or a tiled wrapper around one.
    Returns [] if the dataset does not expose a COCO index (e.g. streaming)."""
    base = getattr(dataset, "_base", dataset)
    coco = getattr(base, "coco", None)
    img_dir = getattr(base, "_image_dir", None)
    if coco is None:
        return []
    out = []
    ids = getattr(base, "image_ids", None) or coco.getImgIds()
    for iid in ids:
        info = coco.loadImgs(iid)[0]
        p = str(img_dir / info["file_name"]) if img_dir is not None else info["file_name"]
        out.append((p, int(info["width"]), int(info["height"])))
    return out
