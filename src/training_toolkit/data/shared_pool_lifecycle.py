"""Name derivation, startup sweep of stale segments, owner cleanup."""
from __future__ import annotations
import glob
import os
import signal
import socket
from multiprocessing import shared_memory

from training_toolkit.data.named_lock import NamedLock


def pool_name(run_id: str, epoch: int = 0) -> str:
    """Derive a unique shared-pool name for this node and epoch.

    Including the epoch makes each epoch's control block independent — no
    cross-epoch FileExistsError race when some workers start epoch N+1 before
    all epoch-N workers have finished detaching.
    """
    return f"tt_pool_{run_id}_ep{epoch}_{socket.gethostname()}"


def sweep_stale(prefix: str = "tt_pool_") -> list[str]:
    """Unlink leftover /dev/shm/<prefix>* segments + their _lock semaphores.

    Returns the list of segment names swept (excluding _lock suffixes).
    """
    swept = []
    for path in glob.glob(f"/dev/shm/{prefix}*"):
        name = os.path.basename(path)
        if name.endswith("_lock"):
            continue
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            continue  # raced away between glob and open — nothing to sweep
        # Segment is gone; unlink its lock independently (best-effort) so a
        # lock-unlink failure can't abort the sweep or drop the swept record.
        try:
            NamedLock.unlink_name("/" + name + "_lock")
        except Exception:
            pass
        swept.append(name)
    return swept


def install_signal_cleanup(cb) -> None:
    """Install SIGTERM/SIGINT handlers to unlink the control block if creator."""

    def _handler(signum, frame):
        if getattr(cb, "is_creator", False):
            cb.unlink()
        # Deliberately replaces the default SIGINT->KeyboardInterrupt path with a
        # clean SystemExit(128+signum) (conventional signal exit code) after cleanup.
        raise SystemExit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


class OwnedControlBlock:
    """Context manager that unlinks the control block in __exit__ if creator.

    Used by rank-0 main to ensure cleanup happens even on exception.
    """

    def __init__(self, cb):
        self._cb = cb

    def __enter__(self):
        return self._cb

    def __exit__(self, *exc):
        try:
            if self._cb.is_creator:
                self._cb.unlink()
        except Exception:
            pass  # best-effort cleanup; never mask the original exception
