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
import time
from pathlib import Path
from typing import Any

__all__ = [
    "disk_report",
    "evict_cache_dir",
    "free_bytes",
    "home_areas",
    "human_bytes",
    "largest_area_note",
    "media_cache_stats",
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


def home_areas(home: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Size of every top-level area under ``home``, largest first:
    ``[{"name": "workspace", "bytes": N}, ...]``.  The whole-tree
    accounting the 2026-11-08 incident proved necessary: the pile
    that filled the disk was NOT the snapshot store this module
    was built around — it was a WhatsApp download cache three
    levels down inside ``workspace/``, invisible to every number
    the doctor and the preflight used to print.  An area list
    makes the biggest writer a measured fact, not a discovery
    made by failing into a full disk."""
    root = Path(home)
    areas: list[dict[str, Any]] = []
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir():
                areas.append({"name": child.name,
                              "bytes": _dir_size(child)})
    areas.sort(key=lambda a: -int(a["bytes"]))
    return areas


def media_cache_stats(home: str | os.PathLike[str],
                      ) -> tuple[list[str], int]:
    """(dirs, total bytes) of the WhatsApp bridges' inbound-media
    download caches under ``home``.  The layout is the pairing
    supervisor's: sessions live at ``<workspace>/data/
    whatsapp_sessions/wa-<number>`` and each account's download
    cache is the sibling ``wa-<number>-media`` directory.  The
    workspace's own name is configurable, so the sessions root is
    found one level down from any top-level area (and directly
    under home when the workspace IS home)."""
    root = Path(home)
    sessions_roots: list[Path] = []
    direct = root / "data" / "whatsapp_sessions"
    if direct.is_dir():
        sessions_roots.append(direct)
    if root.is_dir():
        for child in root.iterdir():
            cand = child / "data" / "whatsapp_sessions"
            if child.is_dir() and cand.is_dir() \
                    and cand not in sessions_roots:
                sessions_roots.append(cand)
    dirs: list[str] = []
    total = 0
    for sessions in sessions_roots:
        for child in sessions.iterdir():
            if child.is_dir() and child.name.endswith("-media"):
                dirs.append(str(child))
                total += _dir_size(child)
    return sorted(dirs), total


def evict_cache_dir(root: str | os.PathLike[str], *,
                    cap_bytes: int,
                    min_age_seconds: float = 0.0,
                    now: float | None = None,
                    ) -> dict[str, Any]:
    """Bound one cache directory to ``cap_bytes``, oldest first.

    Files younger than ``min_age_seconds`` are evicted only when
    the older files alone cannot bring the total under the cap —
    the cap is HARD (a cache that outruns its grace window in an
    hour still stops at the cap), the grace only decides the
    order within one sweep.  A file that cannot be removed is
    counted in ``errors`` and left in place — never silently
    swallowed, and the sweep carries on with the next victim.
    Returns the run's ledger: files/bytes evicted, what remains,
    and the error count."""
    root_path = Path(root)
    result: dict[str, Any] = {"files_evicted": 0, "bytes_freed": 0,
                              "bytes_remaining": 0, "errors": 0}
    if not root_path.is_dir():
        return result
    stamp = time.time() if now is None else float(now)
    files: list[tuple[float, int, Path]] = []
    total = 0
    for dirpath, _dirnames, filenames in os.walk(root_path):
        for name in filenames:
            p = Path(dirpath) / name
            try:
                st = p.stat()
            except OSError:
                result["errors"] += 1
                continue
            files.append((float(st.st_mtime), int(st.st_size), p))
            total += int(st.st_size)
    result["bytes_remaining"] = total
    if total <= int(cap_bytes):
        return result
    files.sort(key=lambda f: f[0])
    grace = float(min_age_seconds or 0.0)
    ordered = [f for f in files if stamp - f[0] >= grace] \
        + [f for f in files if stamp - f[0] < grace]
    for _mtime, size, p in ordered:
        if total <= int(cap_bytes):
            break
        try:
            p.unlink()
        except OSError:
            result["errors"] += 1
            continue
        total -= size
        result["files_evicted"] += 1
        result["bytes_freed"] += size
    result["bytes_remaining"] = total
    return result


def disk_report(home: str | os.PathLike[str],
                db_path: str | os.PathLike[str],
                blob_dir: str | os.PathLike[str] | None = None,
                ) -> dict[str, Any]:
    """The shared disk verdict: free space on the install volume
    (measured at ``home`` — the snapshot the update takes lands
    there), what one snapshot costs, what one update needs, the
    snapshot store's footprint, whether the volume is low
    (free < one update's need) — and the whole home tree's area
    accounting (``areas``, ``largest_area``, the WhatsApp media
    caches' footprint), so the biggest writer is named by the
    same report that judges the free space."""
    snap = snapshot_bytes_estimate(db_path, blob_dir)
    need = 2 * snap
    free = free_bytes(home)
    count, store_bytes = snapshot_store_stats(home)
    areas = home_areas(home)
    cache_dirs, cache_bytes = media_cache_stats(home)
    return {
        "free_bytes": free,
        "snapshot_bytes": snap,
        "need_bytes": need,
        "store_count": count,
        "store_bytes": store_bytes,
        "low": free < need,
        "areas": areas,
        "largest_area": areas[0] if areas else None,
        "media_cache_bytes": cache_bytes,
        "media_cache_dirs": cache_dirs,
    }


def largest_area_note(rep: dict[str, Any]) -> str:
    """The clause every disk surface appends when the home tree's
    biggest area outweighs the snapshot the update is about to
    take: the pile an update will trip over is named, with its
    size, instead of the report discussing only the snapshot
    store while a cache three levels down eats the volume.
    Empty when no area exceeds the snapshot estimate."""
    largest = rep.get("largest_area") or None
    if not largest:
        return ""
    if int(largest.get("bytes") or 0) <= int(
            rep.get("snapshot_bytes") or 0):
        return ""
    note = (f"biggest area: {largest.get('name')} "
            f"{human_bytes(int(largest.get('bytes') or 0))}")
    cache = int(rep.get("media_cache_bytes") or 0)
    if cache > 0:
        note += f" (whatsapp media cache {human_bytes(cache)})"
    return note
