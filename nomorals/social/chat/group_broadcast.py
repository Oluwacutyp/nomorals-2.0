"""Owner broadcast + group re-add recovery (DESIGN §20b, §9.8).

§20b: the owner speaks through Devon to all groups/channels at
once with pacing and dry-run — the mass-tools class (§14), under
owner authority.  This is the fourth mass tool and the one that
speaks as DEVON (the bot), not as him: the other three act through
his paired accounts, this one posts into the rooms her bot serves.

§9.8 (recovery half): a kicked-and-re-added group resumes from its
backup — the backup organ already builds/restores it; this module
supplies the missing DETECTION + OFFER: on re-add, notice, find
the latest backup, and tell the owner the exact restore command.
The restore itself never auto-applies — the owner decides, and
the existing preview + ``yes`` flow is the only way it lands.

The organ takes the gateway as data (the roster-diff discipline):
it never imports the gateway, so tests ride a fake and the layering
test stays green.
"""
from __future__ import annotations

import contextlib
import time
from typing import Any

from .base import ChatKind, ChatRef

#: Pacing floor between two rooms in one run (seconds).  Telegram
#: groups allow 20 messages/minute — the gateway's own docstring
#: blesses ≥3.0s there, and the same conservative gap rides
#: WhatsApp rooms.  The owner can raise it; it cannot go under.
PACE_FLOOR_S = 3.0

#: Default pace between rooms when the owner never set one.
DEFAULT_PACE_S = 3.0

#: The warn-first text for a tool that speaks as the BOT (the §14
#: warning is written for tools that act through his paired
#: accounts — its "never the bot" line is false here, so this tool
#: carries its own).  Shown when the tool is armed, in its status,
#: and in every refusal.  Plain words, once, as state.
BROADCAST_WARN_TEXT = (
    "⚠️ Before this runs live, the risk in plain words: one message "
    "to all your groups and channels at once is exactly what "
    "platform anti-abuse systems look for. Telegram or WhatsApp "
    "may restrict the bot, or members may read a bulk post as "
    "spam — pacing lowers the temperature, it does not make it "
    "safe. It posts as Devon herself, into the claimed rooms "
    "only, and every run is journaled with the rooms it reached.")

#: Broadcast scopes, as the command's ``|`` segment spells them.
SCOPES = ("all", "groups", "channels")


class BroadcastRefused(RuntimeError):
    """The broadcast will not run as asked — empty text, a bad
    scope, pacing under the floor, or no rooms to reach.  The
    refusal names the reason; nothing was sent."""


def _title_of(surface: Any, key: str, *, channel: bool) -> str:
    if channel:
        try:
            title = surface.channel_title(key) or ""
        except Exception:  # noqa: BLE001 - the key is enough
            title = ""
        return title or key
    entry = (surface.store.data.get(key) or {})
    return str(entry.get("title") or key)


def broadcast_targets(surface: Any) -> dict[str, list[dict[str, Any]]]:
    """Claimed groups + channels, split, with titles.

    The §33 gate decides: a foreign room Devon merely sits in
    publishes nothing.  The enumeration is a consistent cut — it
    walks a snapshot of the store's keys taken as the pass begins,
    so a room registered mid-pass publishes whole on the next
    pass instead of half-joining this one (the same discipline
    as ``claimed_group_keys`` in site.py).
    """
    groups: list[dict[str, Any]] = []
    channels: list[dict[str, Any]] = []
    for key in sorted(list(surface.store.data.keys())):
        if not surface.system_output_allowed(key):
            continue
        platform = str(key).split(":", 1)[0]
        channel = bool(surface.is_channel(key))
        row = {"key": str(key), "platform": platform,
               "kind": "channel" if channel else "group",
               "title": _title_of(surface, key, channel=channel)}
        (channels if channel else groups).append(row)
    return {"groups": groups, "channels": channels}


def _pick_targets(targets: dict[str, list[dict[str, Any]]],
                  scope: str) -> list[dict[str, Any]]:
    if scope == "groups":
        return list(targets["groups"])
    if scope == "channels":
        return list(targets["channels"])
    return list(targets["groups"]) + list(targets["channels"])


def _check(text: str, scope: str, pace_seconds: float) -> str:
    body = str(text or "").strip()
    if not body:
        raise BroadcastRefused(
            "nothing to broadcast — the message is empty.")
    if scope not in SCOPES:
        raise BroadcastRefused(
            f"unknown scope {scope!r} — all, groups, or channels.")
    pace = float(pace_seconds)
    if pace < PACE_FLOOR_S:
        raise BroadcastRefused(
            f"pacing floor is {PACE_FLOOR_S:.0f}s between rooms — "
            f"{pace:g}s is below it. The floor is the safety "
            "control, not a suggestion; raise it freely, it "
            "cannot go under.")
    return body


def preview_broadcast(surface: Any, text: str,
                      scope: str = "all") -> dict[str, Any]:
    """The dry-run: what a live run WOULD reach.  Touches nothing
    — no gateway call, no journal row.  The command's default."""
    body = _check(text, scope, PACE_FLOOR_S)
    targets = _pick_targets(broadcast_targets(surface), scope)
    return {"tool": "obroadcast", "live": False, "text": body,
            "scope": scope, "targets": targets,
            "sent": 0, "failed": 0, "aborted": False, "results": []}


def run_owner_broadcast(*, surface: Any, gateway: Any, text: str,
                        scope: str = "all",
                        pace_seconds: float = DEFAULT_PACE_S
                        ) -> dict[str, Any]:
    """Fan the message out across the claimed rooms, per platform.

    One ``gateway.broadcast`` call per platform (its pacing +
    stop-on-failures ride along); one dead room never kills the
    rest — every room is its own result row.  Returns the run
    record: sent/failed/aborted + the per-room results, ready to
    journal and to render as the run report.
    """
    body = _check(text, scope, pace_seconds)
    targets = _pick_targets(broadcast_targets(surface), scope)
    if not targets:
        raise BroadcastRefused(
            "no claimed groups or channels to broadcast to — "
            "claim a room first (the /groups board shows what "
            "Devon serves).")
    by_platform: dict[str, list[dict[str, Any]]] = {}
    for row in targets:
        by_platform.setdefault(row["platform"], []).append(row)
    results: list[dict[str, Any]] = []
    sent = failed = 0
    aborted = False
    for platform in sorted(by_platform):
        rows = by_platform[platform]
        chats = [ChatRef.parse(
            row["key"],
            kind=(ChatKind.CHANNEL if row["kind"] == "channel"
                  else ChatKind.GROUP)) for row in rows]
        outcome = gateway.broadcast(
            platform, chats, body, pace_seconds=float(pace_seconds))
        for item in outcome.get("results") or []:
            results.append({"chat": str(item.get("chat") or ""),
                            "ok": bool(item.get("ok")),
                            "error": str(item.get("error") or "")})
        sent += int(outcome.get("sent") or 0)
        failed += int(outcome.get("failed") or 0)
        aborted = bool(aborted or outcome.get("aborted"))
    return {"tool": "obroadcast", "live": True, "text": body,
            "scope": scope, "targets": targets,
            "sent": sent, "failed": failed, "aborted": aborted,
            "results": results}


def _has_prior_state(surface: Any, group_key: str, bot_id: str) -> bool:
    """Did Devon serve this room before the kick?  A member other
    than the bot, a backup on disk, or audit rows — any one of
    them proves it.  The claim probe's own ``bot_claim`` stamp
    does NOT count: it fires on the very re-add being judged
    (first add flips None → admin), so counting it would cry
    "served before" for every room the bot just entered.  A pure
    read: asking never materialises the room into the store."""
    entry = surface.store.data.get(group_key) or {}
    members = entry.get("members") or {}
    if any(str(uid) != str(bot_id) for uid in members):
        return True
    if any((row or {}).get("kind") != "bot_claim"
           for row in entry.get("audit") or []):
        return True
    try:
        from .group_backup import list_backups

        infos, _corrupt = list_backups(surface, group_key)
        if infos:
            return True
    except Exception:  # noqa: BLE001 - no backup is a no
        pass
    return False


def readd_recovery_offer(surface: Any, group_key: str, *,
                         bot_id: str) -> dict[str, Any] | None:
    """The recovery offer for a bot that was just re-added.

    None when the room has no prior state (a first add — nothing
    to resume).  Otherwise the offer: the latest backup when one
    exists, else the honest "no backup" variant — both carrying
    the exact restore command.  The re-add is written to the
    room's audit log either way; the restore itself is never
    applied here.
    """
    if not _has_prior_state(surface, group_key, bot_id):
        return None
    title = _title_of(surface, group_key, channel=False)
    try:
        from .group_backup import list_backups

        infos, _corrupt = list_backups(surface, group_key)
    except Exception:  # noqa: BLE001 - no backup is a no
        infos = []
    surface._audit(group_key, "bot_readded", user_id=str(bot_id))
    with contextlib.suppress(Exception):
        surface.store.save()
    if infos:
        newest = infos[0]
        backup_id = str(newest.get("backup_id") or "")
        when = time.strftime(
            "%Y-%m-%d %H:%M",
            time.localtime(float(newest.get("created_ts") or 0)))
        by = str(newest.get("created_by") or "—")
        restore = f"/gbackup restore {group_key} {backup_id}"
        text = (
            f"I'm back in {title} — I was kicked and re-added. "
            f"Latest backup {backup_id} (made {when}, by {by}). "
            f"To resume the group from it: {restore} — it shows "
            "a preview first; reply yes to apply it.")
        return {"has_backup": True, "backup_id": backup_id,
                "created_ts": float(newest.get("created_ts") or 0),
                "created_by": by, "restore_command": restore,
                "text": text}
    text = (
        f"I'm back in {title} — I was kicked and re-added. "
        "No backup exists for this group yet, so there's nothing "
        "to restore from. Once the group is settled, take one "
        f"with /gbackup export {group_key} — or restore from an "
        f"older backup with /gbackup restore {group_key} <id>.")
    return {"has_backup": False, "backup_id": "",
            "created_ts": 0.0, "created_by": "",
            "restore_command": "", "text": text}
