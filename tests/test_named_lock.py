import multiprocessing as mp, time, pytest
from training_toolkit.data.named_lock import NamedLock

NAME = "/tt_test_lock_a"

def test_acquire_release_roundtrip():
    NamedLock.unlink_name(NAME)
    lock = NamedLock(NAME, create=True, timeout_s=1.0)
    with lock:
        pass
    lock.unlink()

def _hold(name, secs):
    l = NamedLock(name, create=False, timeout_s=5.0)
    with l:
        time.sleep(secs)

def test_timeout_raises_when_held():
    NamedLock.unlink_name(NAME)
    owner = NamedLock(NAME, create=True, timeout_s=5.0)
    p = mp.get_context("spawn").Process(target=_hold, args=(NAME, 2.0))
    with owner:                       # hold it ourselves first to force contention
        p.start(); time.sleep(0.2)
        contender = NamedLock(NAME, create=False, timeout_s=0.3)
        with pytest.raises(TimeoutError):
            with contender:
                pass
    p.join()
    owner.unlink()

def test_lock_released_on_body_exception():
    NamedLock.unlink_name(NAME)
    lock = NamedLock(NAME, create=True, timeout_s=1.0)
    with pytest.raises(RuntimeError):
        with lock:
            raise RuntimeError("oops")
    with lock:            # must be re-acquirable immediately
        pass
    lock.unlink()
