"""Pool/reservoir/cursor over a SharedControlBlock, plus per-image pixel segments.

Mirrors _PoolState's admit/pop/decref/coverage logic, running it under the
control block's NamedLock against shared memory so multiple processes share the
same pool state.

Correctness invariant
---------------------
refcount == descriptors admitted but not yet materialised (popped+decrefed).
A pooled image's pixel segment stays alive until its last reader calls decref,
whereupon refcount hits 0, the registry slot is freed, and the segment is
closed+unlinked.  Lazy images (is_pooled=False) use no slot and have no segment.

Lock discipline
---------------
NamedLock is NOT reentrant — every locked section must be flat (no helper that
tries to acquire the same lock).  _slot_for(), _push_descriptors() are private
helpers meant to be called only while the caller already holds the lock.
"""
from __future__ import annotations

import random
from multiprocessing import shared_memory

import numpy as np


class SharedImagePool:
    def __init__(self, control, run_id: str, host: str, epoch: int = 0,
                 cache_bytes: int | None = None) -> None:
        self.cb = control
        self._run_id = run_id
        self._host = host
        self._epoch = epoch
        # Opt-in resident-bytes budget (None = image-count cap only). The gate
        # is byte-EMPTY-safe: an oversized image admits into an empty pool
        # (min-progress), otherwise it could never be admitted at all.
        self._cache_bytes = cache_bytes
        # Local cache of attached pixel segments for this process.
        # We own the segments we created (pooled); we only borrow them for
        # reading (another process created them). Either way, the handle must be
        # closed when we're done.
        self._attached: dict[int, shared_memory.SharedMemory] = {}

    # ------------------------------------------------------------------
    # Helpers (lock-free — callers must hold cb.locked() before calling)
    # ------------------------------------------------------------------

    def _seg_name(self, image_id: int) -> str:
        return f"{self._run_id}_ep{self._epoch}_{self._host}_img{image_id}"

    def _slot_for(self, image_id: int):
        """Return the registry row index for image_id if it is resident, else None.

        Caller must hold the lock.
        """
        reg = self.cb.registry
        # state != 0 means the slot is in-use (decoding=1 or ready=2)
        hits = np.where((reg["state"] != 0) & (reg["image_id"] == image_id))[0]
        return int(hits[0]) if len(hits) else None

    def _push_descriptors(self, image_id: int, n_windows: int) -> int:
        """Append n_windows (image_id, window_idx) pairs to the shared reservoir.

        Caller must hold the lock.  Clamps to the reservoir capacity so a race
        where multiple workers simultaneously push into a nearly-full reservoir
        never causes an out-of-bounds write.  Returns the number of descriptors
        actually pushed (may be < n_windows if the reservoir was nearly full).
        """
        res = self.cb.reservoir
        cap = self.cb.buffer_size
        hlen = int(self.cb.header["res_len"][0])
        pushed = 0
        for w in range(n_windows):
            if hlen >= cap:
                break
            res[hlen] = (image_id, w)
            hlen += 1
            pushed += 1
        self.cb.header["res_len"] = hlen
        return pushed

    # ------------------------------------------------------------------
    # Public API (match _PoolState's interface where applicable)
    # ------------------------------------------------------------------

    def has_free_slot(self) -> bool:
        with self.cb.locked():
            return bool((self.cb.registry["state"] == 0).any())

    def reservoir_len(self) -> int:
        with self.cb.locked():
            return int(self.cb.header["res_len"][0])

    def try_admit(
        self,
        image_id: int,
        n_windows: int,
        is_pooled: bool,
        decode_fn,
        nbytes_hint: int = 0,
    ) -> bool:
        """Admit an image into the pool.

        Pooled: claim a registry slot, decode pixels OUTSIDE the lock, write
        the pixel segment, then push descriptors.  Returns False immediately if
        no free slot is available (caller should retry later).

        Lazy: just push descriptors; no slot, no segment.  Always returns True.
        """
        if is_pooled:
            # --- claim a slot under the lock ---
            with self.cb.locked():
                free = np.where(self.cb.registry["state"] == 0)[0]
                if len(free) == 0:
                    return False
                if self._cache_bytes is not None and nbytes_hint > 0:
                    res_bytes = int(self.cb.header["resident_bytes"][0])
                    # min-progress: only refuse when something IS resident —
                    # a byte-empty pool always admits (else an image larger
                    # than the budget would spin forever).
                    if res_bytes > 0 and res_bytes + nbytes_hint > self._cache_bytes:
                        return False
                self.cb.header["resident_bytes"] = (
                    int(self.cb.header["resident_bytes"][0]) + nbytes_hint)
                slot = int(free[0])
                # dtype order: (image_id, refcount, state, is_pooled, n_windows)
                # state=1 (decoding), refcount=n_windows (one per descriptor)
                self.cb.registry[slot]["image_id"] = image_id
                self.cb.registry[slot]["refcount"] = n_windows
                self.cb.registry[slot]["state"] = 1       # decoding
                self.cb.registry[slot]["is_pooled"] = 1
                self.cb.registry[slot]["n_windows"] = n_windows
                self.cb.registry[slot]["nbytes"] = nbytes_hint

            # --- decode OUTSIDE the lock (potentially slow) ---
            array = np.ascontiguousarray(decode_fn(image_id), dtype=np.uint8)

            # --- write pixel segment ---
            # Layout: [i4×3 shape header (12 bytes)] [uint8 pixel data]
            seg = shared_memory.SharedMemory(
                name=self._seg_name(image_id),
                create=True,
                size=12 + array.nbytes,
            )
            # Write shape header — create and immediately delete the temporary
            # numpy view so it doesn't keep a reference to seg.buf alive after
            # this function returns (avoids a Py3.12 BufferError in decref when
            # seg.close() is called while an exported view still exists).
            hdr_view = np.frombuffer(seg.buf, dtype="i4", count=3)
            hdr_view[:] = array.shape
            del hdr_view
            # Write pixel data
            px_view = np.ndarray(array.shape, dtype=np.uint8, buffer=seg.buf, offset=12)
            px_view[:] = array
            del px_view
            self._attached[image_id] = seg

            # --- mark ready + push descriptors under the lock ---
            with self.cb.locked():
                pushed = self._push_descriptors(image_id, n_windows)
                # Adjust refcount to the number of descriptors actually pushed.
                # If the reservoir was full we push fewer; we must not leave a
                # dangling refcount or the slot will never be freed.
                if pushed < n_windows:
                    self.cb.registry[slot]["refcount"] = pushed
                if pushed == 0:
                    # Reservoir completely full: free the slot immediately so
                    # we don't strand it (no descriptors → it will never decref).
                    self.cb.header["resident_bytes"] = max(
                        0, int(self.cb.header["resident_bytes"][0])
                        - int(self.cb.registry[slot]["nbytes"]))
                    self.cb.registry[slot]["nbytes"] = 0
                    self.cb.registry[slot]["state"] = 0
                    # Close and unlink the pixel segment we just created.
                    seg = self._attached.pop(image_id, None)
                    if seg is not None:
                        seg.close()
                        try:
                            seg.unlink()
                        except FileNotFoundError:
                            pass
                    return False
                self.cb.registry[slot]["state"] = 2       # ready
            return True

        # Lazy path — no slot, no segment
        with self.cb.locked():
            pushed = self._push_descriptors(image_id, n_windows)
        # If we pushed 0 descriptors (reservoir full), return False so the
        # caller keeps the lazy pending image and retries after the reservoir drains.
        return pushed > 0

    def _evict_freed(self) -> None:
        """Close this process's cached handles for images no longer resident.

        On Linux an unlinked shm segment's pages aren't reclaimed until ALL
        handles close.  Only the process that did the final decref unlinks the
        NAME + closes its own handle; every other worker that read a tile of
        that image keeps a stale open handle.  Without this sweep those handles
        accumulate toward "every pooled image touched this epoch", defeating
        the 1×M RAM goal.  We only CLOSE here (never unlink) — the name was
        already unlinked exactly once by the freeing process.
        """
        if not self._attached:
            return
        with self.cb.locked():
            resident = {int(r["image_id"]) for r in self.cb.registry if r["state"] != 0}
        for image_id in list(self._attached):
            if image_id not in resident:
                self._attached.pop(image_id).close()

    def pop(self, rng: random.Random):
        """Swap-remove a random descriptor from the reservoir.  Returns (image_id, window)."""
        # Bound this process's open handles to ~the resident set (M) — close
        # any stale handles for images another worker has since freed.
        self._evict_freed()
        with self.cb.locked():
            hlen = int(self.cb.header["res_len"][0])
            if hlen == 0:
                return None
            idx = rng.randrange(hlen)
            image_id = int(self.cb.reservoir[idx][0])
            w = int(self.cb.reservoir[idx][1])
            # Swap-remove: fill the gap with the last element
            self.cb.reservoir[idx] = self.cb.reservoir[hlen - 1]
            self.cb.header["res_len"] = hlen - 1
        return (image_id, w)

    def peek(self, rng: random.Random, n: int) -> list[tuple[int, int]]:
        """Read up to n distinct reservoir descriptors WITHOUT consuming them.

        Read-only under the control-block lock: no refcount change, no reservoir
        mutation, no new wait state. Donor semantics match the in-process
        _borrow_donors (pooled_tile_source.py): peeked windows remain in the
        reservoir and are later emitted as their own primaries, so coverage
        (every window exactly once per epoch) is preserved by construction.

        The caller materializes each pick via read_pixels(); a segment freed
        between this peek and that read resolves through read_pixels' existing
        None -> lazy-disk-ROI fallback (see the spec's race table).
        """
        with self.cb.locked():
            hlen = int(self.cb.header["res_len"][0])
            if hlen == 0:
                return []
            k = min(n, hlen)
            idxs = rng.sample(range(hlen), k)
            return [(int(self.cb.reservoir[i][0]), int(self.cb.reservoir[i][1]))
                    for i in idxs]

    def read_pixels(self, image_id: int):
        """Return a numpy view of the pixel segment, or None (lazy / already freed).

        The returned array is a view into shared memory.  The caller must not
        hold this view across a decref that could free the segment (which would
        trigger a BufferError in Python 3.12+).  Callers should copy if they
        need the data beyond the point where they call decref.
        """
        # Check whether the image is still resident (under the lock)
        with self.cb.locked():
            slot = self._slot_for(image_id)
        if slot is None:
            return None                                   # lazy or already freed

        # Try local cache first; fall back to attaching the named segment
        if image_id not in self._attached:
            try:
                seg = shared_memory.SharedMemory(name=self._seg_name(image_id))
            except FileNotFoundError:
                return None
            self._attached[image_id] = seg

        seg = self._attached[image_id]
        shape = tuple(int(x) for x in np.frombuffer(seg.buf, dtype="i4", count=3))
        return np.ndarray(shape, dtype=np.uint8, buffer=seg.buf, offset=12)

    def decref(self, image_id: int) -> None:
        """Decrement the refcount; free+unlink the segment when it hits 0."""
        freed = False
        with self.cb.locked():
            slot = self._slot_for(image_id)
            if slot is None:
                return                                    # lazy image or already freed
            self.cb.registry[slot]["refcount"] -= 1
            if int(self.cb.registry[slot]["refcount"]) <= 0:
                self.cb.header["resident_bytes"] = max(
                    0, int(self.cb.header["resident_bytes"][0])
                    - int(self.cb.registry[slot]["nbytes"]))
                self.cb.registry[slot]["nbytes"] = 0
                self.cb.registry[slot]["state"] = 0      # free the slot
                freed = True

        if freed:
            seg = self._attached.pop(image_id, None)
            if seg is not None:
                # Release the numpy view before closing/unlinking to avoid a
                # Python 3.12 BufferError (the view holds a reference to the
                # mmap; closing with an active view raises on 3.12).
                seg.close()
                try:
                    seg.unlink()
                except FileNotFoundError:
                    pass

    def done(self) -> bool:
        """True when the reservoir is empty AND no images are resident.

        Reflects pool-INTERNAL state only — it is not plan/cursor-aware, so a
        freshly-constructed (un-admitted) pool reports done.  Callers driving an
        admission plan must also guard on their own cursor (see claim_next).
        """
        with self.cb.locked():
            return (
                int(self.cb.header["res_len"][0]) == 0
                and not (self.cb.registry["state"] != 0).any()
            )

    def claim_next(self, plan: list):
        """Atomically advance the shared cursor and return plan[cursor].

        Returns None if the cursor is already at or past the end of plan.
        This is the node-global admission point — all workers contend on this
        single atomic increment under the NamedLock.
        """
        with self.cb.locked():
            c = int(self.cb.header["cursor"][0])
            if c >= len(plan):
                return None
            self.cb.header["cursor"] = c + 1
            return plan[c]
