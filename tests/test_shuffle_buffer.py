import random


def test_buffer_is_exhaustive_and_preserves_multiset():
    from training_toolkit.data.shuffle_buffer import ShuffleBuffer
    src = list(range(100))
    out = list(ShuffleBuffer(iter(src), size=16, rng=random.Random(0)))
    assert sorted(out) == src  # every item exactly once


def test_buffer_size_one_is_identity():
    from training_toolkit.data.shuffle_buffer import ShuffleBuffer
    src = list(range(20))
    out = list(ShuffleBuffer(iter(src), size=1, rng=random.Random(0)))
    assert out == src


def test_buffer_reorders_with_large_size_deterministically():
    from training_toolkit.data.shuffle_buffer import ShuffleBuffer
    src = list(range(50))
    out_a = list(ShuffleBuffer(iter(src), size=32, rng=random.Random(123)))
    out_b = list(ShuffleBuffer(iter(src), size=32, rng=random.Random(123)))
    assert out_a != src          # actually shuffled
    assert out_a == out_b        # deterministic under the same seed


def test_buffer_handles_empty_source():
    from training_toolkit.data.shuffle_buffer import ShuffleBuffer
    assert list(ShuffleBuffer(iter([]), size=8, rng=random.Random(0))) == []
