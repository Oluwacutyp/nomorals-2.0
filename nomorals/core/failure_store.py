"""Durable sink for the subsystem failure ledgers (pass #141).

One shared organ: every ``failures.py`` ring ledger binds a :class:`Store`
for its subsystem namespace, and the ledger writes through to it.
Stdlib only; nothing in this module imports ``nomorals``, so it is safe
to import from any layer (L1, imported downward).

Design notes:
- Store format is one JSON document per subsystem namespace:
  ``{"version": 1, "total": <ever-recorded>, "entries": [...]}``.
  ``total`` preserves ``failure_count()`` semantics across restarts
  (the ring alone cannot, once the cap drops old entries).
- Home resolution mirrors the app's data-home convention: the
  ``NM_HOME`` environment override first (the config layer honors it),
  else ``~/.nomorals`` — the ``Settings.home`` default. A custom home
  configured only via the ``.env`` file is NOT honored here: replicating
  ``.env`` merging would force the config layer into 26 leaf modules.
  The path is resolved fresh on every operation (never cached), so
  tests can redirect stores with ``NM_HOME`` after import.
- Writes are atomic (temporary file in the same directory +
  ``os.replace``), pruned to the 200-entry ring cap on every write,
  and best-effort: a failed write bumps :func:`write_failures` and
  returns ``False`` — it never raises and never records into a ledger
  (no recursion). The latest failure cause is retained via
  :func:`last_error` for diagnostics, not as a ledger entry. Protections are the default umask, the same as the
  app's other plain state files; failure records hold site + exception
  text, the same exposure class as logs, so no encryption is added
  that the rest of the app does not have.
- Load strategy: lazy on first access (first record or first read),
  one-shot per process, under the caller's lock. No import-time I/O,
  no startup cost, no import failure mode; write-through after load
  keeps disk in sync. A fresh process reloads exactly once.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

__all__ = ["Store", "MAX_ENTRIES", "home_dir", "store_path", "write_failures",
           "last_error"]

MAX_ENTRIES = 200
_VERSION = 1
_DIRNAME = "failures"

_write_failures = 0
_write_failures_lock = threading.Lock()
_last_error: str | None = None


def _note_error(err: BaseException) -> None:
    """Retain the latest store failure cause for diagnostics.

    This is the deliberate trace for best-effort degradation: it is a
    module-level diagnostic string, never a ledger entry, so recording
    a store failure can never recurse into the ledgers themselves.
    """
    global _last_error
    try:
        text = str(err)
    except Exception as _inner:
        text = f"<unprintable {type(_inner).__name__}>"
    _last_error = f"{type(err).__name__}: {text}"


def last_error() -> str | None:
    """Latest store failure cause, or ``None`` if none has occurred."""
    return _last_error
_tmp_counter = 0


def _next_tmp_suffix() -> str:
    global _tmp_counter
    with _write_failures_lock:
        _tmp_counter += 1
        return f"{os.getpid()}-{threading.get_ident()}-{_tmp_counter}"


def write_failures() -> int:
    """How many store writes have failed since process start (diagnostic only)."""
    with _write_failures_lock:
        return _write_failures


def home_dir() -> Path:
    """Data home: ``NM_HOME`` override, else ``~/.nomorals``."""
    raw = os.environ.get("NM_HOME") or "~/.nomorals"
    return Path(raw).expanduser()


def _sanitize(namespace: str) -> str:
    cleaned = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in namespace)
    return cleaned.strip(".") or "default"


def store_path(namespace: str) -> Path:
    """File holding ``namespace``'s persisted ledger."""
    return home_dir() / _DIRNAME / (_sanitize(namespace) + ".json")


def _coerce_entry(row: object) -> dict | None:
    if not isinstance(row, dict):
        return None
    try:
        ts = float(row["ts"])
        site = str(row["site"])
        exc_type = str(row["exc_type"])
        message = str(row["message"])
    except (KeyError, TypeError, ValueError):
        return None
    return {"ts": ts, "site": site, "exc_type": exc_type, "message": message}


class Store:
    """Durable sink for one subsystem ledger. Construction does no I/O."""

    def __init__(self, namespace: str, max_entries: int = MAX_ENTRIES) -> None:
        self.namespace = namespace
        self.max_entries = max(1, int(max_entries))

    def path(self) -> Path:
        """Store file for this namespace (resolved fresh every call)."""
        return store_path(self.namespace)

    def load(self) -> tuple[list[dict], int]:
        """``(entries, total)`` from disk; ``([], 0)`` on missing/corrupt/any error.

        Never raises. Malformed entries are skipped; ``total`` never
        undercounts the returned ring.
        """
        try:
            raw = self.path().read_text(encoding="utf-8")
            doc = json.loads(raw)
        except Exception as _err:
            _note_error(_err)
            return [], 0
        if not isinstance(doc, dict):
            return [], 0
        entries = []
        for row in doc.get("entries", []):
            entry = _coerce_entry(row)
            if entry is not None:
                entries.append(entry)
        entries = entries[-self.max_entries:]
        try:
            total = int(doc.get("total", len(entries)))
        except (TypeError, ValueError):
            total = len(entries)
        return entries, max(total, len(entries))

    def save(self, entries: list[dict], total: int) -> bool:
        """Atomically persist ``entries`` (pruned to the cap) + ``total``.

        Never raises: on any error the write is abandoned, the failure
        counter bumps, and ``False`` returns. The temporary file lives
        in the same directory so ``os.replace`` is atomic on every
        platform; a concurrent reader never sees a torn document.
        """
        kept = []
        for row in entries[-self.max_entries:]:
            entry = _coerce_entry(row)
            if entry is not None:
                kept.append(entry)
        try:
            total_int = int(total)
        except (TypeError, ValueError):
            total_int = len(kept)
        doc = {"version": _VERSION, "total": total_int, "entries": kept}
        tmp = None
        try:
            # Path resolution lives INSIDE the try: home_dir() calls
            # expanduser(), which can raise (no home, mocked home,
            # broken env). save() never raises — a resolution
            # failure is a write failure like any other.
            path = self.path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp-{_next_tmp_suffix()}")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(doc, fh)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError as _err:
                    _note_error(_err)
            os.replace(tmp, path)
            return True
        except Exception as _err:
            _note_error(_err)
            global _write_failures
            with _write_failures_lock:
                _write_failures += 1
            if tmp is not None:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception as _unlink_err:
                    _note_error(_unlink_err)
            return False

    def clear(self) -> None:
        """Best-effort delete of the store file. Never raises."""
        try:
            self.path().unlink(missing_ok=True)
        except Exception as _err:
            _note_error(_err)
