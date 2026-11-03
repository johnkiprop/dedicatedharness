from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .system import HarnessError

_locks: dict[str, threading.RLock] = {}
_guard = threading.Lock()
_depth = threading.local()


@contextmanager
def activity_lock(home: Path) -> Iterator[None]:
    """Serialize model changes and supervised runs, including across processes."""
    home = home.resolve()
    home.mkdir(parents=True, exist_ok=True)
    key = str(home)
    with _guard:
        lock = _locks.setdefault(key, threading.RLock())
    if not lock.acquire(blocking=False):
        raise HarnessError("Another model operation or supervised run is active.")
    depths = getattr(_depth, "homes", {})
    _depth.homes = depths
    descriptor = None
    try:
        if not depths.get(key):
            descriptor = (home / "activity.lock").open("a+b")
            descriptor.seek(0, os.SEEK_END)
            if descriptor.tell() == 0:
                descriptor.write(b"\0")
                descriptor.flush()
            descriptor.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(descriptor.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise HarnessError(
                    "Another model operation or supervised run is active; "
                    "wait for it to stop before switching models."
                ) from error
        depths[key] = depths.get(key, 0) + 1
        try:
            yield
        finally:
            depths[key] -= 1
    finally:
        if descriptor is not None:
            descriptor.close()
        lock.release()
