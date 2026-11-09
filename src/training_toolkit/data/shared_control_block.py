"""Node-level shm control block: registry + reservoir + header, behind a NamedLock.

Layout (one SharedMemory buffer): [header | registry | reservoir], each a numpy
view. The first process to create() initializes + flips ready=1; attachers spin
until ready so nobody reads a half-built block.
"""
from __future__ import annotations
import time
import numpy as np
from multiprocessing import shared_memory, resource_tracker
from training_toolkit.data.named_lock import NamedLock


def disable_shared_memory_tracking() -> None:
    """Make the per-process resource_tracker a no-op for shared_memory segments.

    Background
    ----------
    Python's resource_tracker daemon auto-registers every SharedMemory a process
    creates OR attaches to, intending to unlink any that are still registered when
    all processes exit.  We manage our POSIX shm lifecycle manually and
    cross-process (explicit .unlink() in decref/eviction/cleanup), so the
    resource_tracker must not touch our segments.  The two failure modes we hit:

    1. Warning flood (original): tracker auto-unlinks a segment after the creating
       process exits, then other processes try to open a name that no longer exists
       → "No such file or directory" warnings from the daemon.

    2. KeyError tracebacks (attempt 3b899d8): calling unregister() at every create/
       attach site meant the daemon received N REMOVE messages for the same name
       (one per process that holds/held the segment).  The daemon's internal set
       removes the name on the FIRST REMOVE and raises KeyError on every subsequent
       one — in the DAEMON process, so our try/except can't catch it.

    Correct fix: make register/unregister no-ops for "shared_memory" in EVERY
    process that uses the pool.  This is safe because:
    - Our .unlink() calls are explicit and unconditional.
    - Spawned DataLoader workers import this module (via SharedControlBlock), so
      they patch themselves before creating or attaching any segment.
    - The patch is idempotent and affects only "shared_memory" (our only use of
      SharedMemory); other rtypes (e.g. "semaphore") still route normally.

    CPython 3.12 internals (verified)
    ----------------------------------
    ``multiprocessing/resource_tracker.py`` defines a module-level
    ``ResourceTracker`` instance (``_resource_tracker``) and then binds:

        register   = _resource_tracker.register    # (self, name, rtype) bound method
        unregister = _resource_tracker.unregister  # (self, name, rtype) bound method

    ``shared_memory.SharedMemory.__init__`` / ``.unlink()`` call these via
    module-attribute lookup (``resource_tracker.register(...)``), NOT via
    ``_resource_tracker`` directly.  Replacing the module-level attributes
    intercepts all future calls from SharedMemory without touching the instance.

    ``_CLEANUP_FUNCS`` is a module-level dict populated at import time; its
    ``"shared_memory"`` entry holds ``_posixshmem.shm_unlink``.  Removing it
    prevents the daemon's end-of-life sweep from calling shm_unlink on any names
    that somehow slipped through before this patch ran (defence in depth).
    """
    if getattr(resource_tracker, "_shm_tracking_disabled", False):
        return
    _orig_register = resource_tracker.register
    _orig_unregister = resource_tracker.unregister

    def register(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_register(name, rtype)

    def unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return _orig_unregister(name, rtype)

    resource_tracker.register = register
    resource_tracker.unregister = unregister
    resource_tracker._CLEANUP_FUNCS.pop("shared_memory", None)
    resource_tracker._shm_tracking_disabled = True


_HEADER = np.dtype([("res_len", "i8"), ("cursor", "i8"),
                    ("attach_count", "i8"), ("ready", "i1"),
                    ("resident_bytes", "i8"),    # sum of resident images' byte hints
                    ("capacity", "i8"),          # creator's geometry — attachers MUST
                    ("buffer_size", "i8")])      # match it (views are offset-computed
                                                 # from the CALLER's args)
_REGENTRY = np.dtype([("image_id", "i8"), ("refcount", "i4"),
                      ("state", "i1"), ("is_pooled", "i1"), ("n_windows", "i4"),
                      ("nbytes", "i8")])          # byte hint charged at admission


class SharedControlBlock:
    def __init__(self, name, capacity, buffer_size, shm, is_creator):
        self.name = name
        self.capacity = capacity
        self.buffer_size = buffer_size
        self._shm = shm
        self.is_creator = is_creator
        self._lock = NamedLock("/" + name + "_lock", create=is_creator)
        h = _HEADER.itemsize
        r = capacity * _REGENTRY.itemsize
        self.header = np.frombuffer(shm.buf, dtype=_HEADER, count=1, offset=0)
        self.registry = np.frombuffer(shm.buf, dtype=_REGENTRY, count=capacity, offset=h)
        self.reservoir = np.frombuffer(shm.buf, dtype="i4",
                                       count=buffer_size * 2, offset=h + r).reshape(buffer_size, 2)

    @classmethod
    def _nbytes(cls, capacity, buffer_size):
        return _HEADER.itemsize + capacity * _REGENTRY.itemsize + buffer_size * 2 * 4

    @classmethod
    def create_or_attach(cls, name, capacity, buffer_size):
        # Disable resource_tracker auto-management for shared_memory in this process.
        # Must run before any SharedMemory create/attach so the tracker never registers
        # our segments (idempotent; safe to call in every process including workers).
        disable_shared_memory_tracking()
        size = cls._nbytes(capacity, buffer_size)
        try:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            cb = cls(name, capacity, buffer_size, shm, is_creator=True)
            cb.registry["state"] = 0
            cb.header["res_len"] = 0
            cb.header["cursor"] = 0
            cb.header["attach_count"] = 1
            cb.header["capacity"] = capacity         # geometry record: attachers
            cb.header["buffer_size"] = buffer_size   # validate against these
            cb.header["ready"] = 1                   # initialized — release attachers
            return cb
        except FileExistsError:
            import posix_ipc as _posix_ipc
            shm = shared_memory.SharedMemory(name=name)         # attach
            # Validate geometry BEFORE building the full views: the registry/
            # reservoir offsets are computed from the CALLER's (capacity,
            # buffer_size), so a mismatched attacher would read/write at wrong
            # offsets — silently, when its layout fits inside the creator's
            # buffer. The header sits at offset 0 with a geometry-independent
            # dtype, so it is readable first. Wait for ready here (the creator
            # writes the geometry fields before flipping ready=1).
            hdr = np.frombuffer(shm.buf, dtype=_HEADER, count=1, offset=0)
            deadline = time.time() + 30
            while int(hdr["ready"][0]) != 1:
                if time.time() > deadline:
                    del hdr                          # release view before close (Py3.12)
                    shm.close()
                    raise TimeoutError(f"control block {name} never became ready")
                time.sleep(0.005)
            creator_cap = int(hdr["capacity"][0])
            creator_buf = int(hdr["buffer_size"][0])
            del hdr                                  # release view before any close
            if (creator_cap, creator_buf) != (capacity, buffer_size):
                shm.close()
                raise ValueError(
                    f"control block {name}: attach geometry (capacity={capacity}, "
                    f"buffer_size={buffer_size}) does not match the creator's "
                    f"(capacity={creator_cap}, buffer_size={creator_buf}) — the "
                    "registry/reservoir views would be misaligned. Every "
                    "create_or_attach of the same pool must pass identical values.")
            # Semaphore-existence retry — now DEFENSE-IN-DEPTH: the ready-wait
            # above already subsumes the bootstrap race (the creator makes the
            # semaphore inside cls() strictly BEFORE writing ready=1, so an
            # attacher past the ready gate is guaranteed the semaphore exists).
            # Kept because it is cheap and guards any future reordering; 30 s
            # bounds a crashed creator either way.
            lock_deadline = time.time() + 30.0
            while True:
                try:
                    cb = cls(name, capacity, buffer_size, shm, is_creator=False)
                    break
                except _posix_ipc.ExistentialError:
                    # Partial cls() — _lock never set, numpy views never built;
                    # shm itself is unaffected (same object reused on retry).
                    if time.time() > lock_deadline:
                        raise TimeoutError(
                            f"control block {name}: semaphore never appeared "
                            "(creator may have crashed)"
                        )
                    time.sleep(0.002)
            with cb.locked():
                cb.header["attach_count"] += 1
            return cb

    def locked(self):
        return self._lock

    def _release_views(self):
        """Release numpy buffer references so shm.close() can release the mmap."""
        # np.frombuffer views hold a reference to shm.buf; deleting them lets the
        # mmap be closed.  Python's GC doesn't guarantee immediate collection, so we
        # also call del explicitly here.
        del self.header, self.registry, self.reservoir
        self.header = self.registry = self.reservoir = None  # type: ignore[assignment]

    def detach(self):
        import gc
        with self.locked():
            self.header["attach_count"] -= 1
        self._release_views()
        # Explicitly collect GC so numpy buffer views rooted in exception tracebacks
        # are freed before shm.close() — otherwise Python 3.12 raises BufferError.
        gc.collect()
        self._shm.close()

    def unlink(self):
        self._release_views()
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
        self._lock.unlink()
