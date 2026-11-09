"""Named POSIX semaphore with timeout-acquire-or-raise (fail-loud)."""
from __future__ import annotations
import posix_ipc


class NamedLock:
    def __init__(self, name: str, create: bool, timeout_s: float = 10.0) -> None:
        self._timeout_s = timeout_s
        flags = posix_ipc.O_CREAT if create else 0
        # initial_value=1 => a mutex. Existing semaphore: flags=0 just opens it.
        self._sem = posix_ipc.Semaphore(name, flags=flags, initial_value=1)

    def __enter__(self) -> "NamedLock":
        try:
            self._sem.acquire(timeout=self._timeout_s)
        except posix_ipc.BusyError as exc:
            raise TimeoutError(
                f"NamedLock {self._sem.name}: timed out after {self._timeout_s}s "
                f"(holder presumed dead — failing loud)"
            ) from exc
        return self

    def __exit__(self, *exc) -> None:
        # __exit__ runs only if __enter__ succeeded; a timed-out acquire raises in
        # __enter__ and __exit__ is never called -> no spurious release.
        self._sem.release()

    def unlink(self) -> None:
        try:
            self._sem.unlink()
        except posix_ipc.ExistentialError:
            pass

    @classmethod
    def unlink_name(cls, name: str) -> None:
        try:
            posix_ipc.unlink_semaphore(name)
        except posix_ipc.ExistentialError:
            pass
