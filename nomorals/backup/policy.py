"""Retention: keep last 7 + 30 daily + 12 monthly — locally AND on GitHub.

One policy, two enforcement points: local staging dirs and GitHub
Releases. The rule is a union — a generation survives if ANY rule keeps
it — so the newest 7 are always safe even if the daily/monthly buckets
would have dropped them.
"""

from __future__ import annotations

import time
from typing import Any

from ..core.logging_setup import get_logger
from .ship import delete_release, list_releases

__all__ = [
    "KEEP_LAST",
    "KEEP_DAILY",
    "KEEP_MONTHLY",
    "select_keepers",
    "prune_local",
    "prune_github_releases",
]

KEEP_LAST = 7
KEEP_DAILY = 30
KEEP_MONTHLY = 12

_log = get_logger(__name__)


def select_keepers(
    entries: list[dict[str, Any]],
    *,
    now: float | None = None,
    keep_last: int = KEEP_LAST,
    keep_daily: int = KEEP_DAILY,
    keep_monthly: int = KEEP_MONTHLY,
) -> set[str]:
    """Backup ids to keep. Entries need ``backup_id`` + ``created_at``.

    - Last ``keep_last`` generations, always.
    - Newest generation per day for the last ``keep_daily`` days.
    - Newest generation per month for the last ``keep_monthly`` months.
    """
    now = time.time() if now is None else now
    if not entries:
        return set()
    ordered = sorted(entries, key=lambda e: float(e.get("created_at", 0)),
                     reverse=True)
    keepers: set[str] = set()

    def _bid(entry: dict[str, Any]) -> str:
        return str(entry.get("backup_id") or entry.get("tag") or entry.get("id"))

    for entry in ordered[:max(0, keep_last)]:
        keepers.add(_bid(entry))
    # Daily: one (newest) per UTC day, within the window.
    day_cutoff = now - keep_daily * 86400
    seen_days: set[str] = set()
    for entry in ordered:
        created = float(entry.get("created_at", 0))
        if created < day_cutoff:
            break
        day = time.strftime("%Y-%m-%d", time.gmtime(created))
        if day in seen_days:
            continue
        seen_days.add(day)
        keepers.add(_bid(entry))
    # Monthly: one (newest) per UTC month, within the window.
    month_cutoff = now - keep_monthly * 30.44 * 86400
    seen_months: set[str] = set()
    for entry in ordered:
        created = float(entry.get("created_at", 0))
        if created < month_cutoff:
            break
        month = time.strftime("%Y-%m", time.gmtime(created))
        if month in seen_months:
            continue
        seen_months.add(month)
        keepers.add(_bid(entry))
    return keepers


def prune_local(
    entries: list[dict[str, Any]],
    *,
    now: float | None = None,
    delete: Any | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Prune local generations not selected by :func:`select_keepers`.

    ``delete`` is a callable taking one entry (defaults to removing the
    entry's ``path``/``staging_dir`` on disk). Returns keep/prune lists.
    """
    from pathlib import Path
    import shutil

    keepers = select_keepers(entries, now=now)
    pruned: list[str] = []
    kept: list[str] = []
    for entry in entries:
        bid = str(entry.get("backup_id") or entry.get("id"))
        if bid in keepers:
            kept.append(bid)
            continue
        pruned.append(bid)
        if dry_run:
            continue
        if delete is not None:
            delete(entry)
        else:
            for key in ("staging_dir", "path"):
                target = entry.get(key)
                if target:
                    path = Path(str(target))
                    if path.is_dir():
                        shutil.rmtree(path, ignore_errors=True)
                    elif path.is_file():
                        path.unlink(missing_ok=True)
    return {"kept": kept, "pruned": pruned, "dry_run": dry_run}


def prune_github_releases(
    connector: Any,
    full_name: str,
    *,
    now: float | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Enforce retention on GitHub Releases — same policy as local.

    Release tags are backup ids (``backup-<stamp>``); the release
    ``created_at`` is the generation time. An empty release list prunes
    nothing; a release whose tag does not parse as a backup id is left
    alone (never delete what you cannot identify).
    """
    import datetime

    releases = list_releases(connector, full_name)
    entries: list[dict[str, Any]] = []
    for release in releases:
        tag = str(release.get("tag_name", ""))
        if not tag.startswith("backup-"):
            continue
        raw = str(release.get("created_at", ""))
        try:
            created = datetime.datetime.fromisoformat(
                raw.replace("Z", "+00:00")
            ).timestamp()
        except ValueError:
            created = 0.0
        entries.append({"backup_id": tag, "tag": tag, "created_at": created,
                        "release_id": release.get("id")})
    keepers = select_keepers(entries, now=now)
    deleted: list[str] = []
    kept: list[str] = []
    for entry in entries:
        if entry["backup_id"] in keepers:
            kept.append(entry["backup_id"])
            continue
        deleted.append(entry["backup_id"])
        if not dry_run:
            delete_release(connector, full_name, int(entry["release_id"]))
            _log.info("retention: deleted GitHub release %s",
                      entry["backup_id"])
    return {"kept": kept, "deleted": deleted, "dry_run": dry_run}
