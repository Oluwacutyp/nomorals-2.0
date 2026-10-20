"""Verify: the monthly restore drill, unattended.

A backup never restored is a hypothesis. The drill restores the latest
generation to a temp dir and checks, in order:

1. manifest present + schema known,
2. every blob present, ciphertext SHA-256 matches the manifest,
3. decrypt with the backup key (wrong key / tampered = fail),
4. gunzip, open each DB, ``PRAGMA integrity_check`` == ``ok``,
5. schema version + table count match the manifest,
6. spot-check row counts (up to 5 tables per DB, ``SELECT COUNT(*)``),
7. memory backup verified via :func:`nomorals.memory.backup.verify_backup`
   when the generation contains one.

The drill NEVER raises on a failed step: each step records PASS/FAIL
and the report says PASS or names the exact step that failed. The
highest-value test is a from-zero restore on a clean machine — the
drill is the automated rehearsal for it.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import sqlite3
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from .errors import ChecksumMismatchError
from .manifest import load_manifest, resolve_blob_path, verify_manifest_files
from .protect import decrypt_bytes, join_parts

__all__ = ["DrillStep", "DrillReport", "verify_database_file", "drill", "run_monthly_drill"]

_log = get_logger(__name__)

SPOT_CHECK_TABLES = 5


@dataclass
class DrillStep:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class DrillReport:
    backup_id: str
    ok: bool
    steps: list[DrillStep] = field(default_factory=list)
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id, "ok": self.ok,
            "steps": [s.to_dict() for s in self.steps],
            "started_at": self.started_at, "finished_at": self.finished_at,
        }

    def summary(self) -> str:
        failed = [s.name for s in self.steps if not s.ok]
        if self.ok:
            return f"restore drill PASS for {self.backup_id} ({len(self.steps)} steps)"
        return (
            f"restore drill FAIL for {self.backup_id}: "
            f"failed step(s): {', '.join(failed) or 'unknown'}"
        )


def verify_database_file(path: Path) -> dict[str, Any]:
    """Open a DB file: integrity_check + table count + schema version."""
    conn = sqlite3.connect(str(path))
    try:
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        tables = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
        ).fetchone()[0]
        try:
            schema_version = int(conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()[0] or 0)
        except sqlite3.Error:
            schema_version = int(
                conn.execute("PRAGMA user_version").fetchone()[0] or 0
            )
        counts: dict[str, int] = {}
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT ?",
            (SPOT_CHECK_TABLES,),
        ).fetchall()]
        for name in names:
            try:
                counts[name] = int(conn.execute(
                    f'SELECT COUNT(*) FROM "{name}"'
                ).fetchone()[0])
            except sqlite3.Error:
                counts[name] = -1
        return {"integrity": integrity, "tables": int(tables),
                "schema_version": schema_version, "row_counts": counts}
    finally:
        conn.close()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def drill(
    manifest_path: str | Path,
    blobs_dir: str | Path,
    key: bytes,
    *,
    work_dir: str | Path | None = None,
) -> DrillReport:
    """Run the full restore drill for one generation. Never raises."""
    started = time.time()
    steps: list[DrillStep] = []
    backup_id = "unknown"

    def _step(name: str, ok: bool, detail: str = "") -> bool:
        steps.append(DrillStep(name, ok, detail))
        return ok

    try:
        manifest = load_manifest(manifest_path)
        backup_id = str(manifest.get("backup_id", "unknown"))
        _step("manifest_present", True,
              f"schema v{manifest.get('schema_version')}")
    except Exception as exc:  # noqa: BLE001 - step failure, not a raise
        _step("manifest_present", False, str(exc)[:300])
        return DrillReport(backup_id, False, steps, started, time.time())

    blobs = Path(blobs_dir)
    problems = verify_manifest_files(blobs, manifest)
    if not _step("blobs_present_and_hash", not problems,
                 "; ".join(problems) or "all blobs present"):
        return DrillReport(backup_id, False, steps, started, time.time())

    tmp = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="nm-restore-drill-")
    )
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        restored_dir = tmp / "restored"
        restored_dir.mkdir(exist_ok=True)
        all_ok = True
        for entry in manifest.get("files", []):
            rel = str(entry.get("rel", "?"))
            parts = entry.get("parts") or [entry.get("encrypted_path", "")]
            part_paths = [resolve_blob_path(blobs, candidate)
                          for candidate in parts]
            joined = tmp / (Path(rel.replace("/", "__")).name + ".joined.enc")
            try:
                join_parts(part_paths, joined)
                if _sha256_file(joined) != entry.get("sha256"):
                    raise ChecksumMismatchError(
                        f"ciphertext hash mismatch for {rel}")
                plain = gzip.decompress(decrypt_bytes(joined.read_bytes(), key))
                restored = restored_dir / Path(rel).name
                restored.write_bytes(plain)
            except Exception as exc:  # noqa: BLE001 - per-file step failure
                all_ok = False
                _step(f"decrypt:{rel}", False, str(exc)[:300])
                continue
            _step(f"decrypt:{rel}", True, f"{len(plain)} bytes")
            if entry.get("kind") == "database":
                try:
                    info = verify_database_file(restored)
                except Exception as exc:  # noqa: BLE001
                    all_ok = False
                    _step(f"integrity:{rel}", False, str(exc)[:300])
                    continue
                ok = (
                    info["integrity"] == "ok"
                    and info["tables"] == int(entry.get("tables", -1))
                    and info["schema_version"] == int(
                        entry.get("schema_version", -1))
                )
                if not ok:
                    all_ok = False
                _step(f"integrity:{rel}", ok,
                      f"integrity={info['integrity']} tables="
                      f"{info['tables']}/{entry.get('tables')} schema="
                      f"{info['schema_version']}/"
                      f"{entry.get('schema_version')} rows="
                      f"{info['row_counts']}")
        # Memory leg, when the generation contains a real memory/backup.py
        # backup (kind "memory-backup") — delegates to memory/backup.py.
        # Plain memory/ files are ordinary snapshot files, not memory
        # backups, and must NOT trigger this leg.
        if any(f.get("kind") == "memory-backup"
               for f in manifest.get("files", [])):
            try:
                from ..memory.backup import verify_backup as _verify_memory

                mem = _verify_memory(restored_dir)
                _step("memory_backup_verify", bool(mem.get("ok")),
                      str(mem.get("bad") or mem.get("error") or "ok"))
                if not mem.get("ok"):
                    all_ok = False
            except Exception as exc:  # noqa: BLE001
                _step("memory_backup_verify", False, str(exc)[:300])
                all_ok = False
        overall = all_ok and all(s.ok for s in steps)
        return DrillReport(backup_id, overall, steps, started, time.time())
    finally:
        if work_dir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def run_monthly_drill(
    manifest_path: str | Path,
    blobs_dir: str | Path,
    key: bytes,
    *,
    report: Any | None = None,
) -> DrillReport:
    """Unattended monthly drill: run, report, never raise."""
    try:
        result = drill(manifest_path, blobs_dir, key)
    except Exception as exc:  # noqa: BLE001 - the drill itself must not die
        result = DrillReport(
            "unknown", False,
            [DrillStep("drill_harness", False, str(exc)[:300])],
            time.time(), time.time(),
        )
    if report is not None:
        try:
            from .report import report_drill

            report_drill(report, result)
        except Exception:  # noqa: BLE001 - reporting must not mask the drill
            _log.warning("restore drill report failed", exc_info=True)
    _log.info("%s", result.summary())
    return result


# Re-exported for callers that want the manifest JSON alongside the report.
def manifest_summary(manifest_path: str | Path) -> dict[str, Any]:
    return load_manifest(manifest_path)
