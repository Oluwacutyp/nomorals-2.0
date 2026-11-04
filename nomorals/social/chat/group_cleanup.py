"""Member cleanup analysis — deleted accounts + inactive members.

The group build's member ledger (DESIGN P1) has always KNOWN who
is quiet: the community Roster classifies members ``silent`` and
the surface ledger keeps ``last_seen`` / ``joined_ts`` /
``message_count`` per member.  What was missing — named in the
DESIGN's own gap list since 2026-10-25 — was anything ACTING on
it: no scan for deleted accounts, no removal path for the dead
and the long-gone.  This organ is the analysis half of that
class; the runtime owns the commands and the adapter calls.

Inputs are normalised member rows (built by the caller from
whatever enumeration the platform offered) plus the group's
ledger.  The honesty rules are the point:

* **Deleted** is Telegram's own marker — the MTProto ``deleted``
  flag on the user object, or, on the Bot API path where a member
  can only be probed one at a time, Telegram's rendering of a
  deleted account (name "Deleted Account", no username, no last
  name).  Never a guess from silence: a quiet member is not a
  deleted one.
* **Inactive** means no activity past the group's window, where
  activity evidence is, in order of truth: Devon's own ledger
  (messages she observed, the join she recorded) and Telegram's
  last-seen for the account (the ``was_online`` timestamp the
  userbot surface can read).  The Bot API alone exposes NEITHER a
  member list NOR last-seen — on that path the ledger is the
  whole universe and the scan says so.
* A last-seen CLASS (recently / last week / last month) can only
  ever RESCUE a member (prove recent activity) or leave them
  unknown — it never convicts.  Where no evidence exists at all,
  the member lands in ``unknown`` and the report names them as
  unjudged instead of guessing.
* The creator, administrators, Devon's own accounts and bots
  (for the inactivity sweep — a utility bot sits quiet by
  design) are never findings.  A deleted account is a finding
  even if it was a bot: a dead account is dead.
"""

from __future__ import annotations

import time
from typing import Any

__all__ = [
    "CLEANUP_DEFAULTS",
    "DEFAULT_INACTIVE_DAYS",
    "analyze_cleanup",
    "is_deleted_row",
    "member_label",
]

#: Standing inactivity window when a group has not set its own.
DEFAULT_INACTIVE_DAYS = 90

#: The ``cleanup`` block of the group settings store — the same
#: defaults-merged settings machinery as every other block.
CLEANUP_DEFAULTS: dict[str, Any] = {
    "enabled": True,        # may removals act at all, commanded
                            # or automatic?  (the scan is
                            # read-only and always runs)
    "auto_deleted": True,   # the upkeep cycle removes deleted
                            # accounts on its own (owner order
                            # 2026-11-04: automatic, default on;
                            # still under the defense autonomy
                            # switch, like every automatic admin)
    "inactive_days": DEFAULT_INACTIVE_DAYS,
                            # inactivity is NEVER auto-removed
                            # on a default: real people are
                            # removed by command only
}

_DAY_S = 86400.0


def member_label(row: dict[str, Any]) -> str:
    """How a member is named in member-facing text: display name,
    else @username, else a plain placeholder.  Never the numeric
    id (internal ids never print)."""
    name = str(row.get("name") or "").strip()
    if name:
        return name
    username = str(row.get("username") or "").strip()
    if username:
        return f"@{username}"
    return "a member"


def is_deleted_row(row: dict[str, Any]) -> bool:
    """Telegram's own deleted-account markers, either surface's
    shape: the explicit MTProto flag, or the Bot API rendering
    (a deleted account's user object reads first_name "Deleted
    Account" with no username and no last name beside it)."""
    if row.get("deleted"):
        return True
    name = str(row.get("name") or "").strip().lower()
    return name == "deleted account" \
        and not str(row.get("username") or "").strip() \
        and not str(row.get("last_name") or "").strip()


def _ts(value: Any) -> float | None:
    try:
        out = float(value or 0)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _finding(row: dict[str, Any], *, basis: str,
             last_activity: float | None,
             now: float) -> dict[str, Any]:
    return {
        "user_id": str(row.get("user_id") or ""),
        "label": member_label(row),
        "username": str(row.get("username") or ""),
        "bot": bool(row.get("bot")),
        "basis": basis,
        "last_activity": last_activity,
        "days_inactive": (
            int((now - last_activity) // _DAY_S)
            if last_activity else None),
    }


def analyze_cleanup(
    rows: list[dict[str, Any]],
    ledger: dict[str, dict[str, Any]],
    *,
    now: float | None = None,
    inactive_days: int = DEFAULT_INACTIVE_DAYS,
    protected_ids: frozenset[str] | set[str] = frozenset(),
    self_ids: frozenset[str] | set[str] = frozenset(),
) -> dict[str, Any]:
    """Classify one group's current members for cleanup.

    ``rows`` — normalised member rows for members believed to be
    IN the group now: ``{user_id, name, username, bot, deleted,
    status_kind, was_online}`` (fields beyond user_id optional;
    ``status_kind`` is the MTProto last-seen class — online /
    recently / last_week / last_month / unknown — and
    ``was_online`` its exact timestamp when Telegram gives one).

    ``ledger`` — the group's member ledger (``surface.members``):
    Devon's own activity record per user id.

    Returns ``{deleted, inactive, unknown, protected, coverage}``
    — findings carry the evidence basis so the report can say
    WHY, and coverage counts what was skipped and why.
    """
    now = now if now is not None else time.time()
    window_s = max(1, int(inactive_days)) * _DAY_S
    deleted: list[dict[str, Any]] = []
    inactive: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    protected: list[dict[str, Any]] = []
    coverage = {"scanned": 0, "bots_skipped": 0,
                "self_skipped": 0, "left_skipped": 0}
    seen: set[str] = set()
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        user_id = str(row.get("user_id") or row.get("id") or "")
        if not user_id or user_id in seen:
            continue
        seen.add(user_id)
        row = {**row, "user_id": user_id}
        if user_id in self_ids:
            coverage["self_skipped"] += 1
            continue
        led = ledger.get(user_id) or {}
        if led.get("left_ts"):
            # Devon's ledger says they already left — nothing to
            # clean (only reachable when the ledger itself was
            # the enumeration source).
            coverage["left_skipped"] += 1
            continue
        coverage["scanned"] += 1
        is_bot = bool(row.get("bot") or led.get("is_bot"))
        row["bot"] = is_bot
        finding: dict[str, Any] | None = None
        bucket: list[dict[str, Any]] | None = None
        if is_deleted_row(row):
            finding = _finding(row, basis="telegram",
                               last_activity=None, now=now)
            bucket = deleted
        elif is_bot:
            # Bots are exempt from the inactivity sweep: a
            # utility bot is quiet by design.  (A deleted bot
            # was still caught above — dead is dead.)
            coverage["bots_skipped"] += 1
            continue
        else:
            # Activity evidence: Devon's ledger first (it is
            # group-specific truth), Telegram's account
            # last-seen beside it; the LATER signal is the
            # member's real last activity.
            signals: list[tuple[float, str]] = []
            for key in ("last_seen", "joined_ts", "first_seen"):
                ts = _ts(led.get(key))
                if ts:
                    signals.append((ts, "ledger"))
            was_online = _ts(row.get("was_online"))
            if was_online:
                signals.append((was_online, "telegram"))
            if signals:
                last_activity, basis = max(signals)
                if now - last_activity > window_s:
                    finding = _finding(
                        row, basis=basis,
                        last_activity=last_activity, now=now)
                    bucket = inactive
            else:
                # No timestamp anywhere: the last-seen CLASS
                # can rescue (prove recency) or abstain — it
                # never convicts.
                kind = str(row.get("status_kind") or "").lower()
                rescued = (
                    kind in ("online", "recently")
                    or (kind == "last_week" and inactive_days >= 7)
                    or (kind == "last_month"
                        and inactive_days >= 30))
                if not rescued:
                    unknown.append({
                        "user_id": user_id,
                        "label": member_label(row),
                        "username": str(row.get("username")
                                        or ""),
                        "bot": is_bot,
                        "reason": "no activity on record, and "
                                  "no last-seen Telegram will "
                                  "show",
                    })
        if finding is not None and bucket is not None:
            if user_id in protected_ids:
                protected.append(finding)
            else:
                bucket.append(finding)
    deleted.sort(key=lambda f: f["label"].lower())
    inactive.sort(key=lambda f: (-(f["days_inactive"] or 0),
                                 f["label"].lower()))
    unknown.sort(key=lambda f: f["label"].lower())
    protected.sort(key=lambda f: f["label"].lower())
    return {"deleted": deleted, "inactive": inactive,
            "unknown": unknown, "protected": protected,
            "coverage": coverage}
