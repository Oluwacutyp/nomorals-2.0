"""Fresh start: a clean game database, never an import of the old one.

The owner approved a fresh start for the game: the new database
starts at **schema v86** with zero gameplay rows, and the old
database is never auto-imported. There is deliberately *no*
``import_old`` parameter, no fallback path, and no "helpful"
migration on this module — if the owner ever changes their mind,
:func:`backup_old` has kept the old file as a plain file copy and a
human can decide what to do with it then.

Guarantees:

* :func:`fresh_start` creates a new database at ``db_path`` and
  nothing else. It never opens, reads, or copies any other file.
  If ``db_path`` already holds a non-empty database, it refuses
  rather than overwriting — call :func:`backup_old` first and
  remove/rename the file explicitly if a rebuild is really wanted.
* :func:`backup_old` copies the old file to a timestamped backup
  file and **never deletes** the source. The source still exists
  afterwards, byte for byte.
* :func:`verify_clean` checks the result: schema version is 86,
  the fresh-start marker says ``imported_from: none``, and no
  import marker exists. Gameplay rows earned *after* the fresh
  start are not "old data" and do not fail verification.
"""

from __future__ import annotations

import hashlib
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..storage.db import Database
from .early_quests import ensure_early_quest_tables
from .onboarding import ensure_onboarding_tables

__all__ = [
    "SCHEMA_VERSION", "SCHEMA_V86", "FRESH_START_MARKER",
    "fresh_start", "backup_old", "verify_clean",
    "fresh_start_marker",
]

_log = get_logger(__name__)

#: The schema version a fresh game database starts at.
SCHEMA_VERSION = 86
SCHEMA_V86 = 86

#: Marker row written into ``fresh_start_meta`` at creation.
FRESH_START_MARKER = {"imported_from": "none", "schema_version": "86"}


def _meta_table(db: Database) -> None:
    try:
        db.execute(
            "CREATE TABLE IF NOT EXISTS fresh_start_meta ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL DEFAULT '')")
    except Exception:  # noqa: BLE001
        _log.debug("fresh_start_meta ensure failed", exc_info=True)


def _set_meta(db: Database, key: str, value: str) -> None:
    _meta_table(db)
    try:
        db.execute(
            "INSERT INTO fresh_start_meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value))
    except Exception:  # noqa: BLE001
        _log.debug("fresh_start_meta write failed", exc_info=True)


def _get_meta(db: Any, key: str) -> str | None:
    try:
        row = db.query_one(
            "SELECT value FROM fresh_start_meta WHERE key = ?", (key,))
    except Exception:  # noqa: BLE001
        _log.debug("fresh_start_meta read failed", exc_info=True)
        return None
    if row is None:
        return None
    return str(row["value"])


def _user_version(db: Any) -> int | None:
    try:
        row = db.query_one("PRAGMA user_version")
    except Exception:  # noqa: BLE001
        _log.debug("user_version read failed", exc_info=True)
        return None
    if row is None:
        return None
    try:
        return int(next(iter(row.values())))
    except (StopIteration, TypeError, ValueError):
        return None


def fresh_start_marker(db: Any) -> dict[str, Any]:
    """The fresh-start marker rows, or {} when there is no marker."""
    out: dict[str, Any] = {}
    for key in ("imported_from", "schema_version", "created_at",
                "created_by"):
        value = _get_meta(db, key)
        if value is not None:
            out[key] = value
    return out


def fresh_start(db_path: str | Path,
                *, created_by: str = "fresh_start") -> dict[str, Any]:
    """Create a brand-new game database at ``db_path`` at schema v86.

    Never reads any other file: there is no old-DB argument and no
    import step. Refuses when ``db_path`` already holds a non-empty
    database, so a fresh start can never silently clobber data.
    Returns ``{"ok": True, "db_path", "schema_version": 86,
    "created_at", "sha256"}`` on success.
    """
    path = Path(db_path)
    if path.exists() and path.stat().st_size > 0:
        return {
            "ok": False,
            "reason": "already_exists",
            "db_path": str(path),
            "detail": (
                "A non-empty database already exists at this path. "
                "Back it up with backup_old() and remove/rename it "
                "explicitly if a rebuild is really wanted — "
                "fresh_start never overwrites."),
        }
    if path.parent != Path("") and str(path.parent):
        path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(path))
    created_at = datetime.fromtimestamp(time.time()).isoformat()
    try:
        ensure_onboarding_tables(db)
        ensure_early_quest_tables(db)
        _meta_table(db)
        _set_meta(db, "imported_from", "none")
        _set_meta(db, "schema_version", str(SCHEMA_VERSION))
        _set_meta(db, "created_at", created_at)
        _set_meta(db, "created_by", created_by)
        try:
            db.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        except Exception:  # noqa: BLE001
            _log.debug("user_version set failed", exc_info=True)
    finally:
        db.close()
    digest = ""
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        _log.debug("fresh db hash failed", exc_info=True)
    return {
        "ok": True,
        "db_path": str(path),
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "sha256": digest,
    }


def backup_old(path: str | Path,
               backup_dir: str | Path | None = None) -> dict[str, Any]:
    """Copy an old database file to a timestamped backup. Never deletes.

    The source file is left exactly where it was, untouched. The
    backup name is ``<stem>.backup-YYYYMMDD-HHMMSS.db`` (with a
    numeric suffix if that name already exists, so a backup never
    overwrites another backup). Returns ``{"ok": True, "source",
    "backup_path", "bytes"}``.
    """
    src = Path(path)
    if not src.exists():
        return {"ok": False, "reason": "not_found", "source": str(src)}
    if not src.is_file():
        return {"ok": False, "reason": "not_a_file", "source": str(src)}
    dest_dir = Path(backup_dir) if backup_dir is not None else src.parent
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")
    stem = src.stem
    suffix = src.suffix or ".db"
    candidate = dest_dir / f"{stem}.backup-{stamp}{suffix}"
    counter = 1
    while candidate.exists():
        counter += 1
        candidate = dest_dir / f"{stem}.backup-{stamp}-{counter}{suffix}"
    try:
        shutil.copy2(src, candidate)
    except OSError as exc:
        _log.debug("backup_old copy failed: %s", exc, exc_info=True)
        return {"ok": False, "reason": "copy_failed", "source": str(src),
                "detail": str(exc)}
    # The source must still be there, untouched — verify, don't assume.
    if not src.exists():
        return {"ok": False, "reason": "source_missing_after_backup",
                "source": str(src), "backup_path": str(candidate)}
    return {
        "ok": True,
        "source": str(src),
        "backup_path": str(candidate),
        "bytes": candidate.stat().st_size,
        "source_still_exists": True,
    }


def verify_clean(db_or_path: Any) -> dict[str, Any]:
    """Verify a database is a genuine fresh start: schema v86, no import.

    Accepts an open ``Database`` or a path. Checks:

    * ``PRAGMA user_version`` == 86,
    * ``fresh_start_meta.imported_from`` == ``"none"``,
    * no import marker / legacy-import table exists.

    Gameplay rows earned after the fresh start do NOT fail this
    check — only an actual import does. Returns ``{"ok": bool,
    "schema_version", "imported_from", "reasons": [...]}``.
    """
    reasons: list[str] = []
    db = None
    opened_here = False
    try:
        if isinstance(db_or_path, (str, Path)):
            path = Path(db_or_path)
            if not path.exists():
                return {"ok": False, "schema_version": None,
                        "imported_from": None,
                        "reasons": ["database file does not exist"]}
            db = Database(str(path))
            opened_here = True
        else:
            db = db_or_path
        if db is None:
            return {"ok": False, "schema_version": None,
                    "imported_from": None, "reasons": ["no database given"]}
        version = _user_version(db)
        if version != SCHEMA_VERSION:
            reasons.append(
                f"schema version is {version}, expected {SCHEMA_VERSION}")
        marker = fresh_start_marker(db)
        imported_from = marker.get("imported_from")
        if imported_from is None:
            reasons.append("no fresh-start marker (fresh_start_meta missing)")
        elif imported_from != "none":
            reasons.append(
                f"fresh-start marker says imported_from={imported_from!r}, "
                "expected 'none' — old data was imported")
        # Any table that claims to hold imported/legacy data fails.
        try:
            tables = {str(t).lower() for t in db.tables()}
        except Exception:  # noqa: BLE001
            tables = set()
        bad = sorted(t for t in tables
                     if t.startswith("imported_") or t.startswith("legacy_import"))
        if bad:
            reasons.append(f"import marker tables present: {bad}")
        return {
            "ok": not reasons,
            "schema_version": version,
            "imported_from": imported_from,
            "marker": marker,
            "reasons": reasons,
        }
    finally:
        if opened_here and db is not None and hasattr(db, "close"):
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass
