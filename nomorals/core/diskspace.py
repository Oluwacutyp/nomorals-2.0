"""Free-disk measurement shared by the update preflight, the
doctor, and the housekeeping disk watch.

Lives in core (stdlib only) so every layer above — the updater,
the partner runtime, the CLI doctor — measures with the same
stick without an upward import.

The 2026-11-08 EC2 incident: the owner's /update failed at its
snapshot step — "database or disk is full" — because pre-update
snapshots had accumulated with no retention bound and nothing
anywhere measured the disk. The update discovered the full disk
by failing into it, mid-transaction.

These helpers are the ONE measuring stick every surface shares,
so the preflight's "needs", the doctor's floor, and the
housekeeping warning can never drift apart:

* one update needs room for the pre-update snapshot it is about
  to take — the database file, its WAL sidecar, and the blob
  store the snapshot copies wholesale;
* plus an EQUAL amount of headroom for the pull/checkout churn
  of the update itself.  ``need = 2 x snapshot`` everywhere.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

__all__ = [
    "disk_report",
    "free_bytes",
    "human_bytes",
    "snapshot_bytes_estimate",
    "snapshot_store_stats",
    "update_need_bytes",
]


def human_bytes(n: int) -> str:
    """Same rendering as the snapshot list (os/snapshots)."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(max(0, int(n)))
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024
    return f"{value:.1f}TB"  # pragma: no cover - unreachable


def _existing_anchor(path: Path) -> Path:
    """The nearest existing ancestor of ``path`` (disk_usage needs
    a path that exists; a fresh install's home may not yet)."""
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return p


def free_bytes(path: str | os.PathLike[str]) -> int:
    """Free bytes on the volume holding ``path``."""
    return int(shutil.disk_usage(_existing_anchor(Path(path))).free)


def _file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _dir_size(root: Path) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).stat().st_size
            except OSError:
                continue
    return total


def snapshot_bytes_estimate(db_path: str | os.PathLike[str],
                            blob_dir: str | os.PathLike[str] | None = None,
                            ) -> int:
    """What one snapshot of the current state will occupy: the
    database file + its WAL/SHM sidecars + the blob store (a
    snapshot copies all three)."""
    db = Path(db_path)
    total = (_file_size(db)
             + _file_size(Path(str(db) + "-wal"))
             + _file_size(Path(str(db) + "-shm")))
    if blob_dir:
        blob = Path(blob_dir)
        if blob.is_dir():
            total += _dir_size(blob)
    return total


def update_need_bytes(db_path: str | os.PathLike[str],
                      blob_dir: str | os.PathLike[str] | None = None,
                      ) -> int:
    """What one update needs free: its pre-update snapshot plus
    an equal headroom for the pull/checkout churn."""
    return 2 * snapshot_bytes_estimate(db_path, blob_dir)


def snapshot_store_stats(home: str | os.PathLike[str],
                         ) -> tuple[int, int]:
    """(count, total bytes) of the snapshot store under ``home``."""
    root = Path(home) / "snapshots"
    count = 0
    total = 0
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                count += 1
                total += _dir_size(child)
    return count, total


def disk_report(home: str | os.PathLike[str],
                db_path: str | os.PathLike[str],
                blob_dir: str | os.PathLike[str] | None = None,
                ) -> dict[str, Any]:
    """The shared disk verdict: free space on the install volume
    (measured at ``home`` — the snapshot the update takes lands
    there), what one snapshot costs, what one update needs, the
    snapshot store's footprint, and whether the volume is low
    (free < one update's need)."""
    snap = snapshot_bytes_estimate(db_path, blob_dir)
    need = 2 * snap
    free = free_bytes(home)
    count, store_bytes = snapshot_store_stats(home)
    return {
        "free_bytes": free,
        "snapshot_bytes": snap,
        "need_bytes": need,
        "store_count": count,
        "store_bytes": store_bytes,
        "low": free < need,
    }
