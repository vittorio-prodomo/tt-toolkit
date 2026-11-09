"""A tf.data-style shuffle buffer: reorders an iterator with bounded memory."""
from __future__ import annotations

import random
from typing import Any, Iterable, Iterator


class ShuffleBuffer:
    """Yield items from *source* in shuffled order using a buffer of *size*.

    Fills the buffer to ``size`` items, then for each new source item emits one
    random buffered item (and appends the new one). Drains the buffer (shuffled)
    at the end. ``size=1`` is the identity (passthrough). Larger sizes mix items
    spanning more of the source — the IID dial for the streaming sampler.
    """

    def __init__(self, source: Iterable[Any], size: int, rng: random.Random) -> None:
        self._source = source
        self._size = max(1, int(size))
        self._rng = rng

    def __iter__(self) -> Iterator[Any]:
        buf: list[Any] = []
        for item in self._source:
            buf.append(item)
            if len(buf) >= self._size:
                yield buf.pop(self._rng.randrange(len(buf)))
        self._rng.shuffle(buf)
        yield from buf
