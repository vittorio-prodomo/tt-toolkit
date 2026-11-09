"""Exact multi-GPU segmentation mAP without gathering dense masks (Pattern A).

For ``iou_type="segm"`` under multi-rank DDP, stock torchmetrics
:class:`~torchmetrics.detection.MeanAveragePrecision` synchronises by
``all_gather_object``-ing *every* predicted and ground-truth mask (hundreds of MB)
to *every* rank each epoch end, then recomputes on every rank — a pathologically
slow, GPU-idle collective that reads as a hang.

``DDPSegmentationMAP`` instead uses **gather-to-rank-0 + faster-coco-eval**:
torchmetrics already stores segm state as compact COCO-RLE, so each rank
``all_gather_object``s its (small) state snapshot, rank 0 concatenates the
snapshots and runs the stock reference eval once (faster-coco-eval, C++), and the
result dict is ``broadcast_object_list``-ed back to all ranks. This is the
single-process reference computation over the union, so it is exact; keys/shapes
match ``super().compute()`` by construction. It kills the per-rank pycocotools
accumulate (~70 s at full-val scale) and re-fixes the original mask-gather bug.

Scoped tightly: the custom paths trigger ONLY under ``world_size > 1`` **and**
segm. Single-GPU and non-segm fall through to stock torchmetrics unchanged.

Escape hatch: ``TT_DDP_EXACT_MAP=0`` selects **Pattern C** (per-shard pycocotools
match + evalImgs merge) instead of Pattern A — a genuinely independent exact path
(shard-then-merge vs gather-then-eval), a fallback if Pattern A is ever suspect.
Stock torchmetrics' own multi-GPU segm sync can't gather uneven per-rank counts
(it crashes / desyncs), so it is never used under DDP. Pattern C's core is also the
in-process exactness oracle for the test suite (``_compute_shards``).
"""

import copy
import os

import numpy as np
import torch
import torch.distributed as dist
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.detection.helpers import CocoBackend, _get_classes, _calculate_map_with_coco


class DDPSegmentationMAP(MeanAveragePrecision):
    """MeanAveragePrecision that computes exact global segm mAP under DDP by
    gathering each rank's compact RLE state to rank 0 and running faster-coco-eval
    once there (Pattern A), then broadcasting the result to all ranks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pycocotools backend: pure-Python, mergeable evalImgs (see module docstring).
        self._pyc_backend = CocoBackend("pycocotools")

    # Pattern A state = the 9 lists MeanAveragePrecision.compute() consumes, in order.
    # torchmetrics stores segm masks in these as COCO-RLE tuples (see
    # helpers._get_safe_item_values), so a snapshot is compact — no dense masks.
    _STATE_ATTRS = (
        "groundtruth_labels", "groundtruth_box", "groundtruth_mask",
        "groundtruth_crowds", "groundtruth_area",
        "detection_labels", "detection_box", "detection_mask", "detection_scores",
    )

    def _local_state(self) -> dict[str, list]:
        """Picklable CPU snapshot of this rank's metric state (RLE masks, no CUDA)."""
        return {
            name: [v.cpu() if torch.is_tensor(v) else v for v in getattr(self, name)]
            for name in self._STATE_ATTRS
        }

    @staticmethod
    def _merge_states(states: list[dict]) -> dict[str, list]:
        """Concatenate per-rank state lists (rank-major). image_ids are (re)assigned
        positionally by _get_coco_format downstream, so plain concatenation yields
        globally-unique ids — no manual offsetting needed."""
        return {
            name: [item for s in states for item in s[name]]
            for name in DDPSegmentationMAP._STATE_ATTRS
        }

    def _eval_merged_states(self, states: list) -> dict:
        """Concatenate per-rank state snapshots and run the stock reference
        computation over the union (faster-coco-eval, C++). DDP-free: this is
        Pattern A's rank-0 core and the in-process oracle the tests drive. Output
        keys/shapes match super().compute() by construction (same function)."""
        m = self._merge_states(states)
        return _calculate_map_with_coco(
            self._coco_backend,
            m["groundtruth_labels"], m["groundtruth_box"], m["groundtruth_mask"],
            m["groundtruth_crowds"], m["groundtruth_area"],
            m["detection_labels"], m["detection_box"], m["detection_mask"],
            m["detection_scores"],
            self.iou_type, self.average, self.iou_thresholds, self.rec_thresholds,
            self.max_detection_thresholds, self.class_metrics, self.extended_summary,
        )

    # ------------------------------------------------------------------ gating
    def _is_ddp_segm(self) -> bool:
        """True when we need a custom path: multi-rank DDP + segm. Both Pattern A
        (default) and Pattern C (escape hatch) apply here; single-GPU and non-segm
        fall through to stock torchmetrics."""
        return (
            any("segm" in t for t in self.iou_type)
            and dist.is_available()
            and dist.is_initialized()
            and dist.get_world_size() > 1
        )

    def _sync_dist(self, dist_sync_fn=None, process_group=None) -> None:
        # Skip stock's mask all_gather; keep state local. The custom compute() path
        # (Pattern A or C) gathers what it needs itself. (Stock's own segm sync can't
        # gather uneven per-rank counts anyway — that's why the custom paths exist.)
        if self._is_ddp_segm():
            return
        super()._sync_dist(dist_sync_fn=dist_sync_fn, process_group=process_group)

    def compute(self) -> dict:
        if not self._is_ddp_segm():
            return super().compute()  # single-GPU / non-segm: stock, byte-identical
        if os.environ.get("TT_DDP_EXACT_MAP", "1") == "0":
            return self._compute_shards_ddp()  # escape hatch: Pattern C (shard-merge)
        return self._compute_rank0()           # default: Pattern A (gather-to-rank-0)

    def _compute_rank0(self) -> dict:
        """Pattern A: all-gather each rank's compact RLE state, evaluate ONCE on
        rank 0 with faster-coco-eval (C++), broadcast the result to all ranks.

        Transport is all_gather_object (NOT gather_object — the latter uses
        `gather`, unsupported on NCCL); only rank 0 evaluates, then broadcasts, so
        the eval runs once regardless of world size. Every rank returns the same
        dict → sync_dist logging + EarlyStopping's collective stay balanced.

        Note: a rank-0 eval exception surfaces as a process-group-timeout hang on
        the other ranks (blocked in the broadcast), not a clean crash — check
        rank 0's traceback first if the val boundary hangs.
        """
        world_size = dist.get_world_size()
        gathered = [None] * world_size
        dist.all_gather_object(gathered, self._local_state())
        result = self._eval_merged_states(gathered) if dist.get_rank() == 0 else None
        holder = [result]
        dist.broadcast_object_list(holder, src=0)
        return holder[0]

    def _compute_shards_ddp(self) -> dict:
        """Escape hatch (TT_DDP_EXACT_MAP=0): Pattern C — each rank matches its shard
        locally (pycocotools), all_gather_objects the tiny per-image evalImgs, and
        merges + accumulates the global tally on every rank. A genuinely independent
        exact path from Pattern A (shard-then-merge vs gather-then-eval), and
        uneven-safe by construction (_reduce_payloads merges by each rank's own image
        count). Symmetric collectives → every rank returns the identical result."""
        world_size = dist.get_world_size()
        # Phase 1: agree on the global class set (tiny) so every rank evaluates the
        # same catId blocks (keeps the positional evalImgs merge aligned).
        local_classes = self._local_class_ids()
        gathered_classes = [None] * world_size
        dist.all_gather_object(gathered_classes, local_classes)
        cat_ids = sorted({c for cs in gathered_classes for c in cs})
        # Phase 2: match locally, gather the small per-image results, accumulate.
        payloads = [None] * world_size
        dist.all_gather_object(payloads, self._local_eval_payload(cat_ids))
        return self._reduce_payloads(payloads, cat_ids)

    # ---- Pattern C (shard-match + evalImgs merge): the escape-hatch runtime path
    # ---- (TT_DDP_EXACT_MAP=0, via _compute_shards_ddp above) AND the in-process
    # ---- exactness oracle for tests (_compute_shards). An independent exact path
    # ---- from Pattern A. ----

    @staticmethod
    def _compute_shards(shards):
        """In-process equivalent of the DDP flow (tests / single-proc): agree on the
        global class set, evaluate each shard locally, merge, accumulate."""
        cat_ids = sorted({c for s in shards for c in s._local_class_ids()})
        payloads = [s._local_eval_payload(cat_ids) for s in shards]
        return shards[0]._reduce_payloads(payloads, cat_ids)

    # ------------------------------------------------------------------- seams
    def _local_class_ids(self) -> list:
        return _get_classes(detection_labels=self.detection_labels,
                            groundtruth_labels=self.groundtruth_labels)

    def _bare_cocoeval(self):
        """A pycocotools COCOeval with no datasets — accumulate() is self-contained
        given evalImgs + params (npig comes from gtIgnore, not cocoGt)."""
        return self._pyc_backend.cocoeval(iouType=self.iou_type[0])

    def _local_coco_eval(self, cat_ids):
        coco_preds, coco_target = self._pyc_backend._get_coco_datasets(
            self.groundtruth_labels, self.groundtruth_box, self.groundtruth_mask,
            self.groundtruth_crowds, self.groundtruth_area,
            self.detection_labels, self.detection_box, self.detection_mask,
            self.detection_scores, self.iou_type, average="macro",
        )
        coco_eval = self._pyc_backend.cocoeval(coco_target, coco_preds, iouType=self.iou_type[0])
        coco_eval.params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
        coco_eval.params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
        coco_eval.params.maxDets = self.max_detection_thresholds
        coco_eval.params.catIds = list(cat_ids)
        return coco_eval

    def _local_eval_payload(self, cat_ids) -> dict:
        """Match this rank's shard locally; return the tiny per-image results.

        evalImgs is a flat list ordered [catId, areaRng, imgId] of pure-Python dicts
        (dtMatches/gtMatches/dtScores/ignore flags) — no masks.
        """
        coco_eval = self._local_coco_eval(cat_ids)
        coco_eval.evaluate()
        return {
            "evalImgs": coco_eval.evalImgs,
            "n_img": len(coco_eval.params.imgIds),
            "A": len(coco_eval.params.areaRng),
        }

    def _reduce_payloads(self, payloads, cat_ids) -> dict:
        """Merge per-rank evalImgs into the global flat list and accumulate."""
        A = payloads[0]["A"]
        K = len(cat_ids)
        n_imgs = [p["n_img"] for p in payloads]
        i_total = sum(n_imgs)

        # Rebuild the global [catId, areaRng, imgId] flat list. Within a (k, a) block
        # concatenate ranks' per-image entries (images are disjoint; accumulate indexes
        # positionally, so the image_id values themselves are irrelevant).
        merged = []
        for k in range(K):
            for a in range(A):
                for payload, i_r in zip(payloads, n_imgs):
                    base = k * A * i_r + a * i_r
                    merged.extend(payload["evalImgs"][base: base + i_r])

        stats, eval_dict = self._accumulate(merged, cat_ids, i_total, A)
        result = self._pyc_backend._coco_stats_to_tensor_dict(
            stats, prefix="", max_detection_thresholds=self.max_detection_thresholds)

        if self.extended_summary:
            result["precision"] = torch.tensor(eval_dict["precision"])
            result["recall"] = torch.tensor(eval_dict["recall"])
            result["scores"] = torch.tensor(eval_dict["scores"])

        mar_key = f"mar_{self.max_detection_thresholds[-1]}_per_class"
        if self.class_metrics:
            map_pc, mar_pc = [], []
            for k in range(K):
                block = merged[k * A * i_total: (k + 1) * A * i_total]
                class_stats, _ = self._accumulate(block, [cat_ids[k]], i_total, A)
                map_pc.append(class_stats[0])
                mar_pc.append(class_stats[8])
            result["map_per_class"] = torch.tensor(map_pc, dtype=torch.float32)
            result[mar_key] = torch.tensor(mar_pc, dtype=torch.float32)
        else:
            result["map_per_class"] = torch.tensor([-1], dtype=torch.float32)
            result[mar_key] = torch.tensor([-1], dtype=torch.float32)

        result["classes"] = torch.tensor(cat_ids, dtype=torch.int32)
        return result

    def _accumulate(self, eval_imgs, cat_ids, i_total, area_ranges):
        """pycocotools accumulate + summarize on a prebuilt flat evalImgs list."""
        ce = self._bare_cocoeval()
        params = ce.params
        params.iouThrs = np.array(self.iou_thresholds, dtype=np.float64)
        params.recThrs = np.array(self.rec_thresholds, dtype=np.float64)
        params.maxDets = self.max_detection_thresholds
        params.catIds = list(cat_ids)
        params.imgIds = list(range(i_total))
        params.useCats = 1
        assert len(params.areaRng) == area_ranges, "areaRng mismatch between evaluate and accumulate"
        ce.params = params
        ce._paramsEval = copy.deepcopy(params)
        ce.evalImgs = eval_imgs
        ce.accumulate()
        ce.summarize()
        return ce.stats, ce.eval
