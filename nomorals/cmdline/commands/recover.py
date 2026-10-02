"""``nm recover`` — interactive-safe recovery from snapshots.

Lists snapshots newest-first, verifies each, restores the last good one,
re-runs migrations, and reports health.  Interactive by default (asks
before touching the live state); ``--yes`` makes it non-interactive.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ..emit import _emit
from .snapshot import _manager
from ...os.snapshots import SnapshotRefused, system_state


def _health(settings: Any) -> dict[str, Any]:
    from ...storage.db import Database

    db = Database(str(settings.db_path))
    try:
        summary = db.migrate()
        integrity = db.integrity_check()
        tables = len(db.tables())
    finally:
        db.close()
    try:
        from ... import version as _v  # noqa: F401
        import_ok = True
    except Exception:  # noqa: BLE001
        import_ok = False
    return {
        "schema_version": summary.version,
        "tables": tables,
        "integrity": str(integrity),
        "import_ok": import_ok,
    }


def _ask(prompt: str) -> bool:
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):  # noqa: E106 - deliberate: aborted prompt = "no"
        print()
        return False
    return answer in ("y", "yes")


def _cmd_recover(args: argparse.Namespace, settings: Any) -> int:
    """Runs on settings only (no open DB) like ``nm snapshot``."""
    yes = bool(getattr(args, "yes", False))
    mgr = _manager(settings, getattr(args, "config", None))

    want = getattr(args, "snapshot_id", "") or ""
    if want:
        snap = mgr.get(want)
        candidates = [snap]
    else:
        candidates = list(reversed(mgr.list()))
    if not candidates:
        print("no snapshots to recover from")
        return 1

    # Pick the newest snapshot that verifies clean.
    chosen = None
    checked = 0
    for snap in candidates:
        checked += 1
        problems = mgr.verify(snap.id)
        if not problems:
            chosen = snap
            break
        print(f"snapshot {snap.id} failed verification, skipping: "
              f"{'; '.join(problems)}")
    if chosen is None:
        print(f"no verified snapshot found ({checked} checked)")
        return 1
    print(f"last good snapshot: {chosen.id}"
          + (f" [{chosen.label}]" if chosen.label else ""))

    state = system_state(settings.home_path, settings.db_path)
    if state["running"] or state["dirty"]:
        why = " and ".join(k for k, v in state.items() if v)
        print(f"live system is {why}")
        if not yes and not _ask("restore over it anyway (--force)?"):
            print("aborted")
            return 2
        force = True
    else:
        force = False
        if not yes and not _ask(f"restore snapshot {chosen.id}?"):
            print("aborted")
            return 2

    try:
        mgr.restore(chosen.id, force=force)
    except SnapshotRefused as exc:
        print(f"restore refused: {exc}")
        return 2
    except Exception as exc:  # noqa: BLE001 - recovery must report, not crash
        print(f"restore failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"restored snapshot {chosen.id}")

    # Re-run migrations in case the snapshot predates the current schema.
    try:
        from ...storage.db import Database
        db = Database(str(settings.db_path))
        try:
            summary = db.migrate()
        finally:
            db.close()
        print(f"migrations: schema v{summary.version} "
              f"({len(summary.applied)} applied)")
    except Exception as exc:  # noqa: BLE001
        print(f"migrations failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    health = _health(settings)
    ok = (str(health["integrity"]).lower() == "ok" and health["import_ok"])
    _emit(
        args,
        {"restored": chosen.id, "health": health},
        f"health: schema v{health['schema_version']}, "
        f"{health['tables']} tables, integrity {health['integrity']}, "
        f"import {'ok' if health['import_ok'] else 'FAILED'}",
    )
    return 0 if ok else 1
