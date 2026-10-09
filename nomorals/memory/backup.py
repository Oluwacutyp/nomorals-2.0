"""Memory backup, restore, and import — the whole store, portably.

``export_model()`` (in ``persona.py``) dumps the *user model* as JSON.  This
module handles the *store itself*:

- :func:`backup_to` — atomic snapshot of every memory file (the memories
  DB, the repetition/knowledge-state/two-tier DBs, typed entities, vector
  sidecars) plus a manifest with SHA-256 hashes and record counts.
- :func:`restore_from` — verify-then-restore: hashes are checked before a
  single byte is written, and the live files get a pre-restore safety copy.
- :func:`import_records` — merge an export bundle's ``records`` list into a
  live store: exact-content dedupe, private flags restored, additive only.

Everything here is additive and never raises — a backup that crashes the
chat is worse than no backup.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable

from ..core.logging_setup import get_logger
from .base import MemoryKind

_log = get_logger(__name__)

__all__ = [
    "backup_to",
    "import_records",
    "restore_from",
    "verify_backup",
]

_MANIFEST = "manifest.json"


# ── file inventory ───────────────────────────────────────────────────────

def _settings_home(manager: Any) -> Path | None:
    try:
        settings = getattr(getattr(manager, "context", None), "settings", None)
        home = getattr(settings, "home_path", None)
        return Path(home) if home else None
    except Exception:  # noqa: BLE001
        return None


def _memory_files(manager: Any) -> dict[str, Path]:
    """Every file that makes up the memory system.  Missing files are
    simply absent from the map — a store without a two-tier DB is fine."""
    files: dict[str, Path] = {}
    home = _settings_home(manager)

    def _resolve(db_path_fn: Any, key: str) -> None:
        try:
            p = db_path_fn(home and _FakeSettings(home))
            if p and Path(p).is_file():
                files[key] = Path(p)
        except Exception:  # noqa: BLE001
            pass

    # main memories DB
    try:
        db = getattr(manager, "db", None)
        db_path = getattr(db, "path", None)
        if db_path and Path(db_path).is_file():
            files["memories.db"] = Path(db_path)
            # usearch sidecars live next to the DB
            for sidecar in Path(db_path).parent.glob("*.usearch"):
                files[f"sidecar/{sidecar.name}"] = sidecar
    except Exception:  # noqa: BLE001
        pass

    from .repetition import repetition_db_path
    from .delivery import knowledge_state_db_path
    from .tiers import two_tier_db_path
    from .types import entity_home
    _resolve(repetition_db_path, "repetition.db")
    _resolve(knowledge_state_db_path, "knowledge_state.db")
    _resolve(two_tier_db_path, "two_tier.db")
    try:
        entities = entity_home(_FakeSettings(home) if home else None)
        if entities.is_dir():
            files["entities/"] = entities
    except Exception:  # noqa: BLE001
        pass
    return files


class _FakeSettings:
    """Adapter so the ``*_db_path(settings)`` helpers accept a home path."""

    def __init__(self, home_path: Any) -> None:
        self.home_path = str(home_path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ── backup ───────────────────────────────────────────────────────────────

def backup_to(manager: Any, dest_dir: str | Path, *,
              label: str = "") -> dict[str, Any]:
    """Snapshot the whole memory system into ``dest_dir``.  Never raises.

    Copies go to a temp sibling first, then the directory is renamed into
    place — a backup is never half-written.  Returns the manifest dict.
    """
    dest = Path(dest_dir)
    report: dict[str, Any] = {"ok": False, "dest": str(dest),
                              "files": [], "label": label}
    try:
        files = _memory_files(manager)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        # ``dest_dir`` is the backup *collection*; each run gets its own
        # timestamped subdirectory so backups never overwrite each other.
        target = dest / f"memory-backup-{stamp}"
        tmp = dest / f".tmp-{stamp}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        for key, src in files.items():
            rel = Path(key)
            out = tmp / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, out, dirs_exist_ok=True)
                size = sum(p.stat().st_size for p in out.rglob("*")
                           if p.is_file())
                entries.append({"key": key, "kind": "dir", "size": size})
            else:
                shutil.copy2(src, out)
                entries.append({"key": key, "kind": "file",
                                "size": out.stat().st_size,
                                "sha256": _sha256(out)})
            report["files"].append(key)

        try:
            record_count = int(manager.repo.count())
        except Exception:  # noqa: BLE001
            record_count = -1
        manifest = {
            "created_at": time.time(),
            "label": label,
            "record_count": record_count,
            "vector_backend": getattr(getattr(manager, "semantic", None),
                                      "name", ""),
            "embedder": getattr(getattr(manager, "embedder", None),
                                "stats_snapshot", lambda: {})(),
            "files": entries,
        }
        (tmp / _MANIFEST).write_text(json.dumps(manifest, indent=2),
                                     encoding="utf-8")
        # atomic publish
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        tmp.rename(target)
        report.update({"ok": True, "backup_dir": str(target),
                       "manifest": manifest})
    except Exception as exc:  # noqa: BLE001
        _log.warning("memory backup failed: %s", exc)
        report["error"] = str(exc)[:300]
    return report


def verify_backup(backup_dir: str | Path) -> dict[str, Any]:
    """Check a backup's manifest against its files.  Never raises."""
    root = Path(backup_dir)
    out: dict[str, Any] = {"ok": False, "dir": str(root), "checked": 0,
                           "bad": []}
    try:
        manifest_path = root / _MANIFEST
        if not manifest_path.is_file():
            out["error"] = "no manifest.json — not a memory backup"
            return out
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in manifest.get("files", []):
            if entry.get("kind") != "file":
                continue
            path = root / entry["key"]
            out["checked"] += 1
            if not path.is_file():
                out["bad"].append({"key": entry["key"],
                                   "problem": "missing"})
            elif _sha256(path) != entry.get("sha256"):
                out["bad"].append({"key": entry["key"],
                                   "problem": "hash mismatch"})
        out["ok"] = not out["bad"]
        out["manifest"] = {"created_at": manifest.get("created_at"),
                           "record_count": manifest.get("record_count"),
                           "label": manifest.get("label")}
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)[:300]
    return out


def restore_from(backup_dir: str | Path, manager: Any, *,
                 dry_run: bool = False) -> dict[str, Any]:
    """Verify-then-restore a backup into the live store.  Never raises.

    Live files get a ``.pre-restore-<stamp>`` safety copy before anything
    is overwritten.  ``dry_run=True`` verifies and reports without writing.
    """
    root = Path(backup_dir)
    report: dict[str, Any] = {"ok": False, "dry_run": dry_run,
                              "restored": []}
    try:
        check = verify_backup(root)
        report["verified"] = check["ok"]
        if not check["ok"]:
            report["error"] = (f"backup failed verification: "
                               f"{check.get('bad') or check.get('error')}")
            return report
        if dry_run:
            report["ok"] = True
            report["would_restore"] = check["checked"]
            return report
        live = _memory_files(manager)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        manifest = json.loads((root / _MANIFEST).read_text(encoding="utf-8"))
        for entry in manifest.get("files", []):
            key = entry["key"]
            src = root / key
            if entry.get("kind") == "dir":
                continue  # entities merge below
            dst = live.get(key)
            if dst is None:
                # file belongs to a store this manager doesn't have —
                # note it, don't invent a home for it
                report.setdefault("skipped", []).append(key)
                continue
            if dst.is_file():
                safety = dst.with_name(f"{dst.name}.pre-restore-{stamp}")
                shutil.copy2(dst, safety)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            report["restored"].append(key)
        # entities: merge (never clobber — existing files win)
        ent_src = root / "entities"
        if ent_src.is_dir():
            try:
                from .types import entity_home
                live_home = _settings_home(manager)
                ent_dst = entity_home(_FakeSettings(live_home)
                                      if live_home else None)
                ent_dst.mkdir(parents=True, exist_ok=True)
                merged = 0
                for src_file in ent_src.rglob("*.json"):
                    rel = src_file.relative_to(ent_src)
                    dst_file = ent_dst / rel
                    if not dst_file.exists():
                        dst_file.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                        merged += 1
                report["entities_merged"] = merged
            except Exception as exc:  # noqa: BLE001
                report["entities_error"] = str(exc)[:200]
        report["ok"] = True
    except Exception as exc:  # noqa: BLE001
        _log.warning("memory restore failed: %s", exc)
        report["error"] = str(exc)[:300]
    return report


# ── import (merge an export bundle) ─────────────────────────────────────

def _normalize_content(text: str) -> str:
    return " ".join((text or "").lower().split())


def import_records(manager: Any,
                   records: Iterable[dict[str, Any]],
                   *,
                   dry_run: bool = False,
                   source: str = "import") -> dict[str, Any]:
    """Merge exported records into the live store.  Never raises.

    Additive: records whose normalized content already exists are skipped
    (not duplicated); private flags are restored; nothing is deleted or
    overwritten.  ``dry_run=True`` reports without writing.
    """
    report: dict[str, Any] = {"ok": True, "dry_run": dry_run,
                              "imported": 0, "skipped_duplicate": 0,
                              "failed": 0, "ids": []}
    try:
        items = list(records or [])
        # one query for the existing content set — not one LIKE per record
        try:
            existing = {_normalize_content(r["content"]) for r in
                        manager.db.query("SELECT content FROM memories")}
        except Exception:  # noqa: BLE001
            existing = set()
        for item in items:
            try:
                content = (item.get("content") or "").strip()
                if not content:
                    report["failed"] += 1
                    continue
                if _normalize_content(content) in existing:
                    report["skipped_duplicate"] += 1
                    continue
                kind = str(item.get("kind") or MemoryKind.EPISODE)
                if kind not in ("episode", "fact", "preference", "skill",
                                "lesson", "decision", "relationship"):
                    kind = MemoryKind.EPISODE
                metadata = dict(item.get("metadata") or {})
                if item.get("private"):
                    metadata["private"] = True
                if dry_run:
                    report["imported"] += 1
                    continue
                record_id = manager.remember(
                    content,
                    kind=kind,
                    importance=float(item.get("importance", 0.5) or 0.5),
                    source=item.get("source") or source,
                    agent=item.get("agent") or "",
                    metadata=metadata,
                    tags=item.get("tags") or "",
                    origin=item.get("origin") or "",
                    trust=item.get("trust") or "",
                )
                if record_id:
                    report["imported"] += 1
                    report["ids"].append(record_id)
                    existing.add(_normalize_content(content))
                    if metadata.get("private"):
                        manager.mark_private(record_id)
                else:
                    report["failed"] += 1
            except Exception as exc:  # noqa: BLE001
                _log.debug("import record failed: %s", exc)
                report["failed"] += 1
    except Exception as exc:  # noqa: BLE001
        report["ok"] = False
        report["error"] = str(exc)[:300]
    return report
