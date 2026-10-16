"""Single-consumer enforcement for Telegram ``getUpdates``.

Two processes long-polling the same bot token fight over updates: each
``getUpdates`` call consumes updates the other never sees, and Telegram
answers the loser with 409 Conflict. The failure used to be silent —
messages just vanished for one of the processes.

This module makes it loud: the poll loop holds an exclusive lockfile
(``O_CREAT | O_EXCL``). A second process that tries to start refuses to
start with a clear error instead of polling into a 409. Stale locks
(dead PID) are reclaimed, so a crashed process does not brick the bot
forever.

Invariant: at most one live holder per lock path; ``acquire()`` returns
False (never blocks, never raises for contention) when another live
process holds it.
"""

from __future__ import annotations

import errno
import json
import os
import time
from pathlib import Path
from types import TracebackType

__all__ = ["SingleConsumerLock", "ConsumerAlreadyRunning"]


class ConsumerAlreadyRunning(RuntimeError):
    """A second getUpdates consumer tried to start while one is live."""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except OSError:
        return False
    return True


class SingleConsumerLock:
    """Exclusive lockfile for one polling consumer.

    Usage::

        lock = SingleConsumerLock(Path("data/telegram-bot.lock"))
        if not lock.acquire():
            raise ConsumerAlreadyRunning(...)
        try:
            ... poll loop ...
        finally:
            lock.release()

    Or as a context manager (``__enter__`` raises
    :class:`ConsumerAlreadyRunning` on contention).
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._fd: int | None = None
        self._owner_pid: int | None = None

    # -- introspection ---------------------------------------------------
    def holder(self) -> dict[str, object] | None:
        """Lock holder info (``{"pid", "started_at"}``) or None."""
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return None
        try:
            data = json.loads(raw)
        except ValueError:
            return None
        if isinstance(data, dict) and isinstance(data.get("pid"), int):
            return data
        return None

    def is_held(self) -> bool:
        """True iff a live process holds the lock."""
        info = self.holder()
        if info is None:
            return False
        pid = int(info.get("pid") or 0)
        return pid > 0 and _pid_alive(pid)

    # -- acquire / release -----------------------------------------------
    def acquire(self) -> bool:
        """Take the lock. False when a live process already holds it.

        A stale lock (holder PID dead, or an unreadable lockfile from a
        crashed process) is reclaimed: the file is removed and the
        acquire retried once.
        """
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _attempt in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self.is_held():
                    return False
                # Stale: holder dead or lockfile unreadable — reclaim.
                try:
                    self.path.unlink()
                except OSError:
                    return False
                continue
            except OSError:
                return False
            try:
                payload = json.dumps(
                    {"pid": os.getpid(), "started_at": time.time()}
                ).encode("utf-8")
                os.write(fd, payload)
            except OSError:
                try:
                    os.close(fd)
                except OSError:
                    pass
                return False
            self._fd = fd
            self._owner_pid = os.getpid()
            return True
        return False

    def release(self) -> None:
        """Release the lock. Only the holder may release it."""
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except OSError:
            pass
        self._fd = None
        # Remove only our own lockfile: if another process reclaimed it
        # after we died (it can't — we're alive), this guard still holds.
        try:
            info = self.holder()
            if info is not None and int(info.get("pid") or 0) == (self._owner_pid or -1):
                self.path.unlink()
        except OSError:
            pass
        self._owner_pid = None

    # -- context manager ---------------------------------------------------
    def __enter__(self) -> "SingleConsumerLock":
        if not self.acquire():
            info = self.holder() or {}
            raise ConsumerAlreadyRunning(
                "another getUpdates consumer is already running "
                f"(pid={info.get('pid')}); refusing to start — a second "
                "poll loop would silently steal updates / hit 409 Conflict"
            )
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None,
                 tb: TracebackType | None) -> bool:
        self.release()
        return False
