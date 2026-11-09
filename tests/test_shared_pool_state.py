import random
from training_toolkit.data.shared_pool_state import _PoolState

def _plan(n_imgs, windows=3, pooled=True):
    return [(i, windows, pooled) for i in range(n_imgs)]

def test_admit_pushes_one_descriptor_per_window():
    st = _PoolState(capacity=4, buffer_size=100, plan=_plan(1, windows=3))
    img = st.next_admit(); st.admit(*img)
    assert st.reservoir_len() == 3
    assert st.n_resident() == 1

def test_pooled_admit_consumes_a_slot_lazy_does_not():
    st = _PoolState(capacity=1, buffer_size=100,
                    plan=[(0, 2, True), (1, 2, False)])
    st.admit(*st.next_admit())            # pooled -> uses the only slot
    assert st.has_free_slot() is False
    st.admit(*st.next_admit())            # lazy -> no slot needed
    assert st.n_resident() == 1           # only the pooled image is resident

def test_drain_covers_every_tile_exactly_once():
    st = _PoolState(capacity=4, buffer_size=8, plan=_plan(20, windows=3))
    rng = random.Random(0)
    seen = []
    pending = None
    while not st.done():
        if pending is None and st.reservoir_len() < st.buffer_size and not st.cursor_exhausted():
            pending = st.next_admit()
        if pending is not None and (not pending[2] or st.has_free_slot()):
            st.admit(*pending); pending = None
        elif st.reservoir_len() > 0:
            img_id, w = st.pop(rng); seen.append((img_id, w)); st.decref(img_id)
        elif pending is None:
            break
    expected = {(i, w) for i in range(20) for w in range(3)}
    assert set(seen) == expected
    assert len(seen) == len(expected)      # each exactly once

def test_slot_frees_only_after_last_window_decref():
    st = _PoolState(capacity=1, buffer_size=100, plan=[(0, 2, True), (1, 2, True)])
    st.admit(*st.next_admit())
    st.pop(random.Random(0)); st.decref(0)    # 1 of 2 windows materialized
    assert st.has_free_slot() is False        # still occupied
    st.pop(random.Random(0)); st.decref(0)    # last window
    assert st.has_free_slot() is True         # now free

def test_next_admit_returns_none_when_exhausted():
    assert _PoolState(1, 1, []).next_admit() is None

def test_pop_returns_none_on_empty_reservoir():
    assert _PoolState(1, 1, []).pop(random.Random(0)) is None
