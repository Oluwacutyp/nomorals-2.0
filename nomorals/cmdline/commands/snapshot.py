"""``nm snapshot`` — point-in-time snapshots of Devon's live state."""

from __future__ import annotations

import argparse
import contextlib
import json
from typing import Any

from ...os.snapshots import SnapshotManager, SnapshotRefused
from ..emit import _emit


def _manager(settings: Any, config_path: str | None) -> SnapshotManager:
    settings_dict: dict[str, Any] = {}
    with contextlib.suppress(Exception):  # noqa: BLE001 - best effort
        settings_dict = settings.to_dict()
    return SnapshotManager(
        home=settings.home_path,
        db_path=settings.db_path,
        blob_dir=getattr(settings, "blob_dir", None),
        config_path=config_path,
        settings_dict=settings_dict,
    )


def _cmd_snapshot(args: argparse.Namespace, settings: Any) -> int:
    """Runs on settings only (no open DB) so restore can detect a live system."""
    mgr = _manager(settings, getattr(args, "config", None))
    action = getattr(args, "snapshot_action", "list") or "list"

    if action == "create":
        snap = mgr.create(label=getattr(args, "label", "") or "")
        _emit(args, snap.to_dict(), f"snapshot {snap.id} ({snap.size_bytes} bytes)")
        return 0

    if action == "verify":
        snap_id = getattr(args, "snapshot_id", "")
        snap = mgr.get(snap_id) if snap_id else mgr.latest()
        if snap is None:
            print("no snapshots")
            return 1
        problems = mgr.verify(snap.id)
        _emit(args, {"id": snap.id, "problems": problems},
              "snapshot clean" if not problems else "\n".join(problems))
        return 1 if problems else 0

    if action == "restore":
        snap_id = getattr(args, "snapshot_id", "")
        snap = mgr.get(snap_id) if snap_id else mgr.latest()
        if snap is None:
            print("no snapshots")
            return 1
        try:
            mgr.restore(snap.id, force=bool(getattr(args, "force", False)))
        except SnapshotRefused as exc:
            print(f"restore refused: {exc}")
            return 2
        _emit(args, {"restored": snap.id}, f"restored snapshot {snap.id}")
        return 0

    if action == "delete":
        snap_id = getattr(args, "snapshot_id", "")
        if not snap_id:
            print("snapshot delete needs an id")
            return 2
        mgr.delete(snap_id)
        _emit(args, {"deleted": snap_id}, f"deleted snapshot {snap_id}")
        return 0

    # list (default)
    snaps = mgr.list()
    payload = [s.to_dict() for s in snaps]
    if getattr(args, "json", False):
        print(json.dumps(payload, indent=2, default=str))
        return 0
    if not snaps:
        print("no snapshots")
        return 0
    for snap in snaps:
        import datetime as _dt
        ts = _dt.datetime.fromtimestamp(
            snap.created_at, tz=_dt.UTC).strftime("%Y-%m-%d %H:%M:%SZ")
        label = f"  [{snap.label}]" if snap.label else ""
        print(f"{snap.id}  {ts}  {snap.size_bytes:>10} bytes{label}")
    return 0
