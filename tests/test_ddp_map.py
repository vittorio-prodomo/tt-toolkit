"""Exact DDP segmentation mAP — Pattern A (gather-to-rank-0 + faster-coco-eval).

The property under test: COCO mAP decomposes — per-image matching is independent —
so concatenating each rank's state and evaluating the union yields the SAME numbers
as evaluating all data at once. We pin that against stock torchmetrics
MeanAveragePrecision (the ground truth), verified on CPU (the DDP-free
``_eval_merged_states`` core + a real 2-rank gloo run of ``compute()``). Pattern C
(per-shard evaluate + evalImgs merge) is the escape-hatch path (``TT_DDP_EXACT_MAP=0``)
and the in-process oracle — tested via ``_compute_shards`` + a real 2-rank gloo
escape-hatch run with uneven counts.
"""

import pytest
import torch
from torchmetrics.detection import MeanAveragePrecision

from training_toolkit.metrics.ddp_map import DDPSegmentationMAP

_KW = dict(iou_type="segm", class_metrics=True, backend="faster_coco_eval", extended_summary=True)
_SCALARS = ("map", "map_50", "map_75", "map_small", "map_medium", "map_large",
            "mar_1", "mar_10", "mar_100")


def _rect(h, w, y0, x0, y1, x1):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def _synth():
    """3 images, classes {1,2}, class coverage varies per image (incl. a class
    absent from some images) — exercises the per-class / empty-shard paths."""
    H = W = 32
    preds = [
        {"masks": torch.stack([_rect(H, W, 2, 2, 12, 12), _rect(H, W, 20, 20, 30, 29), _rect(H, W, 0, 0, 4, 4)]),
         "scores": torch.tensor([0.9, 0.8, 0.3]), "labels": torch.tensor([1, 2, 1])},
        {"masks": torch.stack([_rect(H, W, 5, 5, 15, 15)]),
         "scores": torch.tensor([0.95]), "labels": torch.tensor([1])},
        {"masks": torch.stack([_rect(H, W, 8, 8, 20, 21), _rect(H, W, 25, 25, 30, 30)]),
         "scores": torch.tensor([0.7, 0.2]), "labels": torch.tensor([2, 1])},
    ]
    targets = [
        {"masks": torch.stack([_rect(H, W, 2, 2, 12, 12), _rect(H, W, 20, 20, 30, 30)]), "labels": torch.tensor([1, 2])},
        {"masks": torch.stack([_rect(H, W, 5, 5, 15, 15)]), "labels": torch.tensor([1])},
        {"masks": torch.stack([_rect(H, W, 8, 8, 20, 20)]), "labels": torch.tensor([2])},
    ]
    return preds, targets


def _ref_compute(preds, targets):
    m = MeanAveragePrecision(**_KW)
    m.update(preds, targets)
    return m.compute()


def _assert_map_close(got, ref, tol=1e-5):
    for k in _SCALARS:
        g, r = got[k].item(), ref[k].item()
        assert g == pytest.approx(r, abs=tol), f"{k}: {g} vs {r}"
    assert torch.equal(got["classes"], ref["classes"]), f"classes: {got['classes']} vs {ref['classes']}"
    assert torch.allclose(got["map_per_class"].float(), ref["map_per_class"].float(), atol=tol), \
        f"map_per_class: {got['map_per_class']} vs {ref['map_per_class']}"


def test_single_shard_matches_stock():
    """The sharded pipeline on ONE shard (build coco → evaluate → merge → accumulate)
    must equal stock torchmetrics exactly. Drives the seam directly so it tests the
    sharded code rather than the single-GPU fall-through."""
    preds, targets = _synth()
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    _assert_map_close(DDPSegmentationMAP._compute_shards([m]), _ref_compute(preds, targets))


def test_compute_falls_through_to_stock_without_ddp():
    """Single-process compute() keeps stock behavior (minimal blast radius)."""
    preds, targets = _synth()
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    _assert_map_close(m.compute(), _ref_compute(preds, targets))


def test_two_shard_split_matches_full():
    """The core property: evaluate two disjoint shards locally, merge their per-image
    results, accumulate — must equal evaluating all data at once."""
    preds, targets = _synth()
    a = DDPSegmentationMAP(**_KW)
    a.update(preds[:2], targets[:2])
    b = DDPSegmentationMAP(**_KW)
    b.update(preds[2:], targets[2:])
    merged = DDPSegmentationMAP._compute_shards([a, b])
    _assert_map_close(merged, _ref_compute(preds, targets))


def test_extended_summary_arrays_match_stock():
    """precision/scores arrays (used downstream for PR curves + best-F1) must match."""
    preds, targets = _synth()
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    got = DDPSegmentationMAP._compute_shards([m])
    ref = _ref_compute(preds, targets)
    assert torch.allclose(got["precision"].float(), ref["precision"].float(), atol=1e-5)
    assert torch.allclose(got["scores"].float(), ref["scores"].float(), atol=1e-5)


def test_split_with_class_absent_from_one_shard():
    """Class 1 only on shard A, class 2 only on shard B — exercises the None-placeholder
    alignment when a rank evaluates a globally-present class it has no local data for."""
    H = W = 32
    predsA = [{"masks": torch.stack([_rect(H, W, 2, 2, 12, 12)]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}]
    tgtA = [{"masks": torch.stack([_rect(H, W, 2, 2, 12, 12)]), "labels": torch.tensor([1])}]
    predsB = [{"masks": torch.stack([_rect(H, W, 20, 20, 30, 29)]), "scores": torch.tensor([0.8]), "labels": torch.tensor([2])}]
    tgtB = [{"masks": torch.stack([_rect(H, W, 20, 20, 30, 30)]), "labels": torch.tensor([2])}]
    a = DDPSegmentationMAP(**_KW)
    a.update(predsA, tgtA)
    b = DDPSegmentationMAP(**_KW)
    b.update(predsB, tgtB)
    got = DDPSegmentationMAP._compute_shards([a, b])
    _assert_map_close(got, _ref_compute(predsA + predsB, tgtA + tgtB))


def test_payload_carries_no_masks():
    """The gathered payload must be small — per-image match results, not RLE masks."""
    import pickle
    preds, targets = _synth()
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    payload = m._local_eval_payload(m._local_class_ids())
    blob = pickle.dumps(payload)
    # Stock would pickle the full mask state; here the payload is just match flags/scores.
    assert len(blob) < 50_000, f"payload unexpectedly large: {len(blob)} bytes"


# --- real 2-process DDP (gloo/CPU): exercises Pattern A end-to-end —
#     _wrap_compute -> _sync_dist no-op -> compute() -> _compute_rank0
#     (all_gather_object -> rank-0 faster-coco-eval -> broadcast_object_list).
#     BOTH ranks must return the identical broadcast result; a desync would hang. ---
def _ddp_worker(rank, world_size, port, shards, out_q):
    import os
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        m = DDPSegmentationMAP(**_KW)
        preds, targets = shards[rank]
        m.update(preds, targets)
        res = m.compute()  # wrapped: sync -> _sync_dist (no-op) -> _compute_rank0
        out_q.put({
            "rank": rank,
            "map": res["map"].item(), "map_50": res["map_50"].item(),
            "map_75": res["map_75"].item(), "mar_100": res["mar_100"].item(),
            "map_per_class": res["map_per_class"].tolist(),
            "classes": res["classes"].tolist(),
            # checksums prove the broadcast carries the extended arrays intact too
            "precision_sum": res["precision"].sum().item(),
            "scores_sum": res["scores"].sum().item(),
        })
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_compute_matches_stock():
    """Pattern A over a real 2-rank process group == stock over the union, AND the
    broadcast delivers the identical result to both ranks."""
    import socket
    import torch.multiprocessing as mp

    preds, targets = _synth()
    ref = _ref_compute(preds, targets)
    shards = [(preds[:2], targets[:2]), (preds[2:], targets[2:])]

    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_ddp_worker, args=(r, 2, port, shards, out_q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in range(2):
            r = out_q.get(timeout=120)  # a desync would time out here
            results[r["rank"]] = r
    finally:
        for p in procs:
            p.join(timeout=30)

    assert set(results) == {0, 1}
    # broadcast identity: both ranks return the same values (scalars, per-class, arrays)
    assert results[0]["map"] == pytest.approx(results[1]["map"], abs=1e-9)
    assert results[0]["map_per_class"] == pytest.approx(results[1]["map_per_class"], abs=1e-9)
    assert results[0]["classes"] == results[1]["classes"]
    assert results[0]["precision_sum"] == pytest.approx(results[1]["precision_sum"], abs=1e-9)
    assert results[0]["scores_sum"] == pytest.approx(results[1]["scores_sum"], abs=1e-9)
    # exactness vs stock over the union (scalars, per-class, and extended arrays)
    got = results[0]
    assert got["map"] == pytest.approx(ref["map"].item(), abs=1e-5)
    assert got["map_50"] == pytest.approx(ref["map_50"].item(), abs=1e-5)
    assert got["map_75"] == pytest.approx(ref["map_75"].item(), abs=1e-5)
    assert got["mar_100"] == pytest.approx(ref["mar_100"].item(), abs=1e-5)
    assert got["classes"] == ref["classes"].tolist()
    assert got["map_per_class"] == pytest.approx(ref["map_per_class"].tolist(), abs=1e-5)
    assert got["precision_sum"] == pytest.approx(ref["precision"].sum().item(), abs=1e-4)
    assert got["scores_sum"] == pytest.approx(ref["scores"].sum().item(), abs=1e-4)


def test_local_state_has_nine_lists_and_rle_masks():
    """_local_state snapshots the 9 compute() state lists; masks are compact RLE
    tuples (torchmetrics stores segm state as RLE), NOT dense tensors."""
    preds, targets = _synth()
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    state = m._local_state()
    assert set(state) == set(DDPSegmentationMAP._STATE_ATTRS)
    assert len(DDPSegmentationMAP._STATE_ATTRS) == 9
    # one entry per image
    assert len(state["detection_mask"]) == len(preds)
    # each per-image mask entry is an RLE tuple, never a dense tensor
    for entry in state["detection_mask"]:
        assert isinstance(entry, tuple)
        assert not torch.is_tensor(entry)


def test_local_state_snapshot_is_compact():
    """The gathered snapshot is small — RLE, not raw masks."""
    import pickle
    preds, targets = _synth()
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    assert len(pickle.dumps(m._local_state())) < 50_000


def test_merge_states_concatenates_per_key():
    preds, targets = _synth()
    a = DDPSegmentationMAP(**_KW); a.update(preds[:2], targets[:2])
    b = DDPSegmentationMAP(**_KW); b.update(preds[2:], targets[2:])
    merged = DDPSegmentationMAP._merge_states([a._local_state(), b._local_state()])
    # segm-only iou_type: box attrs are never populated (torchmetrics fills them
    # only for "bbox"); the other 7 carry one entry per image (2 + 1 = 3).
    empty = {"groundtruth_box", "detection_box"}
    for name in DDPSegmentationMAP._STATE_ATTRS:
        expected = 0 if name in empty else 3
        assert len(merged[name]) == expected, f"{name}: {len(merged[name])} != {expected}"
    # order is rank-major (shard a's images first)
    assert len(merged["detection_labels"][0]) == len(preds[0]["labels"])


def _state_of(preds, targets):
    m = DDPSegmentationMAP(**_KW)
    m.update(preds, targets)
    return m._local_state()


def test_eval_merged_two_states_matches_full():
    """The Pattern A core: eval over concatenated shard snapshots == stock over all."""
    preds, targets = _synth()
    sA, sB = _state_of(preds[:2], targets[:2]), _state_of(preds[2:], targets[2:])
    got = DDPSegmentationMAP(**_KW)._eval_merged_states([sA, sB])
    _assert_map_close(got, _ref_compute(preds, targets))


def test_eval_merged_extended_arrays_match_stock():
    """precision/scores arrays (PR curves + best-F1 inputs) must match stock."""
    preds, targets = _synth()
    sA, sB = _state_of(preds[:2], targets[:2]), _state_of(preds[2:], targets[2:])
    got = DDPSegmentationMAP(**_KW)._eval_merged_states([sA, sB])
    ref = _ref_compute(preds, targets)
    assert torch.allclose(got["precision"].float(), ref["precision"].float(), atol=1e-5)
    assert torch.allclose(got["scores"].float(), ref["scores"].float(), atol=1e-5)


def test_eval_merged_class_absent_from_one_shard():
    """Class 1 only on shard A, class 2 only on shard B."""
    H = W = 32
    predsA = [{"masks": torch.stack([_rect(H, W, 2, 2, 12, 12)]), "scores": torch.tensor([0.9]), "labels": torch.tensor([1])}]
    tgtA = [{"masks": torch.stack([_rect(H, W, 2, 2, 12, 12)]), "labels": torch.tensor([1])}]
    predsB = [{"masks": torch.stack([_rect(H, W, 20, 20, 30, 29)]), "scores": torch.tensor([0.8]), "labels": torch.tensor([2])}]
    tgtB = [{"masks": torch.stack([_rect(H, W, 20, 20, 30, 30)]), "labels": torch.tensor([2])}]
    got = DDPSegmentationMAP(**_KW)._eval_merged_states([_state_of(predsA, tgtA), _state_of(predsB, tgtB)])
    _assert_map_close(got, _ref_compute(predsA + predsB, tgtA + tgtB))


def test_eval_merged_with_empty_shard():
    """A rank that saw no val batches contributes an empty snapshot (must be inert)."""
    preds, targets = _synth()
    got = DDPSegmentationMAP(**_KW)._eval_merged_states([_state_of(preds, targets), _state_of([], [])])
    _assert_map_close(got, _ref_compute(preds, targets))


# --- escape hatch (TT_DDP_EXACT_MAP=0): Pattern C shard-merge over a real 2-rank
#     process group with UNEVEN per-rank counts (2 vs 1) — the exact case stock
#     torchmetrics crashes on (its _sync_dist can't gather uneven segm state).
#     Regression guard for the smoke-found IndexError / collective-size bug. ---
def _escape_hatch_worker(rank, world_size, port, shards, out_q):
    import os
    import torch.distributed as dist
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["TT_DDP_EXACT_MAP"] = "0"  # escape hatch -> Pattern C shard-merge
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        m = DDPSegmentationMAP(**_KW)
        preds, targets = shards[rank]
        m.update(preds, targets)
        res = m.compute()  # -> _compute_shards_ddp (Pattern C); every rank gets the result
        out_q.put({"rank": rank, "map": res["map"].item(),
                   "map_per_class": res["map_per_class"].tolist(),
                   "classes": res["classes"].tolist()})
    except Exception as e:  # stock B would crash/desync on uneven segm state
        out_q.put({"rank": rank, "error": f"{type(e).__name__}: {e}"})
    finally:
        dist.destroy_process_group()


def test_escape_hatch_pattern_c_uneven_gloo():
    """TT_DDP_EXACT_MAP=0 routes to Pattern C (shard-merge). Over a real 2-rank group
    with UNEVEN per-rank image counts (2 vs 1) — the exact case stock torchmetrics
    crashes on — it must return the exact global mAP on every rank."""
    import socket
    import torch.multiprocessing as mp

    preds, targets = _synth()
    ref = _ref_compute(preds, targets)
    shards = [(preds[:2], targets[:2]), (preds[2:], targets[2:])]  # 2 vs 1 = uneven

    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = [ctx.Process(target=_escape_hatch_worker, args=(r, 2, port, shards, out_q)) for r in range(2)]
    for p in procs:
        p.start()
    results = {}
    try:
        for _ in range(2):
            r = out_q.get(timeout=120)  # a desync would time out here
            results[r["rank"]] = r
    finally:
        for p in procs:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()

    errors = [r["error"] for r in results.values() if "error" in r]
    assert not errors, f"escape-hatch Pattern C crashed on uneven shards: {errors}"
    assert set(results) == {0, 1}
    for rank in (0, 1):  # Pattern C computes on every rank -> both hold the exact global result
        assert results[rank]["map"] == pytest.approx(ref["map"].item(), abs=1e-5)
        assert results[rank]["classes"] == ref["classes"].tolist()
        assert results[rank]["map_per_class"] == pytest.approx(ref["map_per_class"].tolist(), abs=1e-5)
