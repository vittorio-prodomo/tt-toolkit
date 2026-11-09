"""Pure single-process reference logic for the cross-process shared pool.

Encodes admit/pop/decref/coverage. Unit-tested here; SharedImagePool runs the
same steps against shared memory under a lock and is cross-checked against this.
"""
from __future__ import annotations
import random


class _PoolState:
    def __init__(self, capacity: int, buffer_size: int,
                 plan: list[tuple[int, int, bool]]) -> None:
        self.capacity = capacity
        self.buffer_size = buffer_size
        self._plan = plan
        self._cursor = 0
        self._reservoir: list[tuple[int, int]] = []   # (image_id, window_index)
        self._refcount: dict[int, int] = {}           # pooled image_id -> remaining
        self._resident: set[int] = set()              # pooled image_ids holding a slot

    def has_free_slot(self) -> bool:
        return len(self._resident) < self.capacity

    def reservoir_len(self) -> int:
        return len(self._reservoir)

    def cursor_exhausted(self) -> bool:
        return self._cursor >= len(self._plan)

    def next_admit(self) -> tuple[int, int, bool] | None:
        if self.cursor_exhausted():
            return None
        item = self._plan[self._cursor]
        self._cursor += 1
        return item

    def admit(self, image_id: int, n_windows: int, is_pooled: bool) -> None:
        if is_pooled:
            self._resident.add(image_id)
            self._refcount[image_id] = n_windows
        for w in range(n_windows):
            self._reservoir.append((image_id, w))

    def pop(self, rng: random.Random) -> tuple[int, int] | None:
        if not self._reservoir:
            return None
        idx = rng.randrange(len(self._reservoir))
        return self._reservoir.pop(idx)

    def decref(self, image_id: int) -> None:
        if image_id not in self._refcount:
            return                                   # lazy image: nothing to free
        self._refcount[image_id] -= 1
        if self._refcount[image_id] <= 0:
            del self._refcount[image_id]
            self._resident.discard(image_id)

    def n_resident(self) -> int:
        return len(self._resident)

    def done(self) -> bool:
        return self.cursor_exhausted() and not self._reservoir and not self._resident
