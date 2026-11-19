"""The file layer (Craft-AI port of ai-native ``shared.rs`` +
``history_file.rs``).

The launcher writes the connections file; every app reads it. The
document format and its validation live in the seam's
:class:`~nomorals.craftlibs.ai.store.StoreFile`; this module adds the
file handling around it:

- **Reads** never block on a writer. Writes replace the file
  atomically, so a reader sees the old file or the new one.
- **Writes** go through :meth:`SharedStore.update`: lock, read,
  change, validate, write. A change that fails validation writes
  nothing.
- **A newer schema is never overwritten.** An older build that meets
  one refuses to write.
- **Unknown top-level fields survive** a rewrite by an older build of
  the same schema version, so a newer launcher's additions are not
  silently dropped. (Nested unknown fields are not preserved; only
  the launcher should rewrite the file in practice.)
- A missing file reads as an empty store, so an app works with no
  launcher installed.

The lock is a lockfile created with ``O_EXCL`` beside the target,
with a stale-expiry so a crashed writer does not wedge the file
forever. It is best-effort across processes; in-process callers also
hold a thread lock.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path

from .store import MAX_STORE_BYTES, StoreError, StoreFile

__all__ = [
    "HistoryFile",
    "HistoryRead",
    "SharedError",
    "SharedStore",
    "default_store_path",
]

KNOWN_FIELDS = ("schema_version", "connections", "defaults", "apps",
                "custom_tools", "model_folders")

DEFAULT_MAX_BYTES = 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
_LOCK_STALE_SECS = 30.0

_thread_locks: dict[str, threading.Lock] = {}
_thread_locks_guard = threading.Lock()


def _thread_lock(key: str) -> threading.Lock:
    with _thread_locks_guard:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _thread_locks[key] = lock
        return lock


class SharedError(Exception):
    """Why a shared store operation failed."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message

    @classmethod
    def no_location(cls) -> SharedError:
        return cls("no_location",
                   "no place to keep the connections file on this system")

    @classmethod
    def io(cls, message: str) -> SharedError:
        return cls("io", f"connections file: {message}")

    @classmethod
    def store(cls, error: StoreError) -> SharedError:
        err = cls("store", str(error))
        err.store_error = error
        return err


class _LockFile:
    """A lockfile beside the target, created with O_EXCL. Stale
    locks (older than :data:`_LOCK_STALE_SECS`) are broken, so a
    crashed writer does not wedge the file forever."""

    def __init__(self, target: Path) -> None:
        self._path = target.with_suffix(target.suffix + ".lock")
        self._fd = None

    def __enter__(self):
        deadline = time.monotonic() + 10.0
        while True:
            try:
                self._fd = os.open(self._path, os.O_CREAT | os.O_EXCL
                                   | os.O_WRONLY)
                os.write(self._fd, str(os.getpid()).encode())
                return self
            except FileExistsError:
                try:
                    age = time.time() - self._path.stat().st_mtime
                except OSError:
                    age = 0.0
                if age > _LOCK_STALE_SECS:
                    with contextlib.suppress(OSError):
                        self._path.unlink()
                if time.monotonic() > deadline:
                    raise SharedError.io(
                        f"timed out waiting for {self._path}") from None
                time.sleep(0.05)

    def __exit__(self, *exc):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        with contextlib.suppress(OSError):
            self._path.unlink()
        return False


class SharedStore:
    """The connections file at a path."""

    def __init__(self, path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def _locked(self, fn):
        with _thread_lock(str(self._path)), _LockFile(self._path):
            return fn()

    def _read_raw(self):
        """Parse the file, also returning its top-level fields this
        build does not know. ``None`` when the file does not exist."""
        try:
            raw = self._path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SharedError.io(str(exc)) from exc
        if len(raw) > MAX_STORE_BYTES:
            raise SharedError.store(StoreError(
                "too_large",
                f"store file is {len(raw)} bytes (max {MAX_STORE_BYTES})"))
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise SharedError.store(StoreError(
                "syntax", "the file is not valid text")) from None
        try:
            head = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SharedError.store(StoreError(
                "syntax", f"not JSON: {exc}")) from None
        if not isinstance(head, dict):
            raise SharedError.store(StoreError(
                "syntax", "a store file is a JSON object"))
        try:
            store = StoreFile.parse(text)
        except StoreError as exc:
            raise SharedError.store(exc) from None
        extras = {k: v for k, v in head.items() if k not in KNOWN_FIELDS}
        return store, extras

    def load(self) -> StoreFile:
        """Read the store. A missing file is an empty store."""
        found = self._read_raw()
        return StoreFile() if found is None else found[0]

    def load_existing(self):
        """Read the store, returning ``None`` when the file does not
        exist (no launcher has written one yet), for callers that
        want to tell "empty" from "absent"."""
        found = self._read_raw()
        return None if found is None else found[0]

    def update(self, fn):
        """Read, change and write the store under the lock, returning
        what ``fn`` returns. Nothing is written if reading fails, if
        the file is a newer schema, or if the changed store is
        invalid or too large."""
        def _run():
            found = self._read_raw()
            if found is None:
                store, extras = StoreFile(), {}
            else:
                store, extras = found
            out = fn(store)
            self._write(store, extras)
            return out

        return self._locked(_run)

    def replace(self, store: StoreFile) -> None:
        """Replace the file with ``store``. A corrupt existing file
        is kept beside it as ``<name>.corrupt``; a newer-schema file
        is refused, as in :meth:`update`."""
        def _run():
            try:
                found = self._read_raw()
            except SharedError as exc:
                if (getattr(exc, "store_error", None) is not None
                        and exc.store_error.kind == "too_new"):
                    raise
                try:
                    corrupt = self._path.with_suffix(
                        self._path.suffix + ".corrupt")
                    corrupt.write_bytes(self._path.read_bytes())
                except OSError:
                    pass
                extras = {}
            else:
                extras = {} if found is None else found[1]
            self._write(store, extras)

        self._locked(_run)

    def _write(self, store: StoreFile, extras: dict) -> None:
        store.validate()
        data = store.to_dict()
        for key, value in extras.items():
            data.setdefault(key, value)
        text = json.dumps(data, indent=2, ensure_ascii=False)
        if len(text.encode("utf-8")) > MAX_STORE_BYTES:
            raise SharedError.store(StoreError(
                "too_large",
                f"store file is {len(text.encode('utf-8'))} bytes "
                f"(max {MAX_STORE_BYTES})"))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=self._path.name + ".",
            suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(text.encode("utf-8"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, self._path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp_name)
            raise


class HistoryRead:
    """What :meth:`HistoryFile.read` found, oldest first."""

    def __init__(self, records: list, skipped: int) -> None:
        self.records = records
        self.skipped = skipped


class HistoryFile:
    """One JSON line per run record, oldest first."""

    def __init__(self, path, max_bytes: int = DEFAULT_MAX_BYTES) -> None:
        self._path = Path(path)
        self._max_bytes = max(max_bytes, 1024)

    @property
    def path(self) -> Path:
        return self._path

    def with_max_bytes(self, max_bytes: int) -> HistoryFile:
        return HistoryFile(self._path, max(max_bytes, 1024))

    def _rotated_path(self) -> Path:
        stem = self._path.stem or "jobs"
        suffix = self._path.suffix
        return self._path.with_name(f"{stem}.1{suffix}")

    def append(self, record: dict) -> None:
        """Append one record. Safe to call from several processes at
        once. When the file would grow past the cap, the old file is
        rotated aside first."""
        line = json.dumps(record, ensure_ascii=False).encode("utf-8")
        if len(line) > MAX_LINE_BYTES:
            raise ValueError("history record is too large")
        line += b"\n"

        def _run():
            size = (self._path.stat().st_size
                    if self._path.exists() else 0)
            if size > 0 and size + len(line) > self._max_bytes:
                rotated = self._rotated_path()
                with contextlib.suppress(OSError):
                    rotated.unlink()
                os.replace(self._path, rotated)
                size = 0
            self._path.parent.mkdir(parents=True, exist_ok=True)
            needs_newline = False
            if size > 0:
                # A mid-line crash leaves the file without its
                # trailing newline; the next record starts on a fresh
                # line. Read the last byte separately: an
                # append-mode handle cannot read.
                with open(self._path, "rb") as check:
                    check.seek(size - 1)
                    needs_newline = check.read(1) != b"\n"
            with open(self._path, "ab") as handle:
                if needs_newline:
                    handle.write(b"\n")
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())

        self._locked(_run)

    def _locked(self, fn):
        with _thread_lock(str(self._path)), _LockFile(self._path):
            return fn()

    def read(self) -> HistoryRead:
        """Read the rotated file then the live one. A missing file
        is an empty history."""
        records: list = []
        skipped = 0
        for path in (self._rotated_path(), self._path):
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            for raw_line in raw.split(b"\n"):
                if not raw_line.strip():
                    continue
                try:
                    records.append(json.loads(raw_line.decode("utf-8")))
                except (ValueError, UnicodeDecodeError):
                    skipped += 1
        return HistoryRead(records, skipped)

    def clear(self) -> None:
        """Delete the history (both files)."""
        def _run():
            for path in (self._rotated_path(), self._path):
                with contextlib.suppress(FileNotFoundError):
                    path.unlink()

        self._locked(_run)


def default_store_path():
    """The standard per-OS location for the connections file, or
    ``None`` when the OS reports no config directory."""
    home = os.environ.get("CRAFT_AI_DIR")
    if home:
        return Path(home) / "ai.json"
    config = os.environ.get("XDG_CONFIG_HOME")
    if config:
        return Path(config) / "craft-ai" / "ai.json"
    home_dir = os.environ.get("HOME") or os.path.expanduser("~")
    if home_dir:
        return Path(home_dir) / ".config" / "craft-ai" / "ai.json"
    return None
