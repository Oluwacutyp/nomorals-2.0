"""Scheduled roster diffing (DESIGN.md §9.1 / §3): the
change-detection backbone.

Observation alone misses changes made while she is offline — a
member joins, leaves, or renames and Devon only notices on their
next observed message.  The diff polls the live member list on a
schedule per group and diffs it against the last snapshot: joins,
leaves, username/display-name changes and role changes land in the
per-group audit log and the roster stays current without anyone
asking.  Routine and silent — nothing is ever posted; the diff is
held internally and surfaced on demand (console / /gsettings /
/groups dashboard).

The live list comes from the userbot (MTProto
``iter_participants`` via the adapter's ``admin_members_full``) —
the Bot API has no member list, so on that surface the diff falls
back to the cleanup scan's per-member probe, and when no route can
enumerate the group it is skipped with the reason recorded, never
faked.

First-run rule: a live member with no ledger record on the first
run is "not yet known", not "joined" — reporting the whole roster
as joins would be false noise.  Leaves on the first run ARE
asserted (absence from a complete live list is a leave whenever it
is observed).  Leaves are never asserted from a truncated list:
absence proves nothing there.

The snapshot lives in the group entry's ``extras`` (``roster_diff``
key), so the backup organ carries it with the rest of the group
state.  The last-run summary rides the settings block beside the
cadence, the way the backup organ carries ``last_run_ts``.
"""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

from .base import ChatRef
from .failures import record_failure
from .group_cleanup import is_deleted_row

#: The scheduler job name and its firing spec.  The tick fires
#: every hour; each group's own cadence decides whether it is due.
ROSTER_DIFF_JOB = "group roster diff tick"
ROSTER_DIFF_SPEC = "every 1h"
ROSTER_DIFF_TOOL = "__group_roster_diff_tick__"

#: The per-group schedule block (see ROSTER_DIFF_DEFAULTS).
#: Cadence defaults to daily: the diff is the design's
#: change-detection backbone, and a backbone that is off
#: everywhere by default is not one.  One bounded MTProto fetch
#: per group per day; an admin turns it off per group.
ROSTER_DIFF_DEFAULTS: dict[str, Any] = {
    "cadence": "daily",   # off | daily | weekly
    "hour": 3,            # wall-clock hour the run becomes due
    "last_run_ts": 0.0,
    "last_diff": {},      # the last run's counts, for the panels
}

CADENCES = ("off", "daily", "weekly")

#: A per-member probe is only worth it for a small ledger: the
#: Bot API has no member list, so each known member costs one
#: getChatMember.  Above this the group is skipped with the
#: reason recorded instead of hammering the API.
MAX_PROBE_LEDGER = 200

#: The full-list cap the adapter promises (admin_members_full).
FULL_LIST_CAP = 20000


class GroupRosterDiffRefused(Exception):
    """The schedule was not set: the ask itself was invalid."""


class GroupRosterDiffError(Exception):
    """A diff run failed for a reason the caller should see."""


def roster_diff_settings(surface: Any, group_key: str
                         ) -> dict[str, Any]:
    """The merged roster-diff block for a group (defaults filled
    in)."""
    merged = surface.store.settings(group_key)
    block = dict(merged.get("roster_diff") or {})
    out = dict(ROSTER_DIFF_DEFAULTS)
    out.update({k: v for k, v in block.items() if k in out})
    try:
        out["hour"] = int(out["hour"])
    except (TypeError, ValueError):
        out["hour"] = ROSTER_DIFF_DEFAULTS["hour"]
    out["hour"] = max(0, min(23, out["hour"]))
    if str(out["cadence"]) not in CADENCES:
        out["cadence"] = "off"
    try:
        out["last_run_ts"] = float(out["last_run_ts"] or 0.0)
    except (TypeError, ValueError):
        out["last_run_ts"] = 0.0
    if not isinstance(out.get("last_diff"), dict):
        out["last_diff"] = {}
    return out


def _write_roster_diff_settings(surface: Any, group_key: str,
                                cfg: dict[str, Any]) -> None:
    entry = surface.store.group(group_key)
    stored = entry.setdefault("settings", {})
    stored["roster_diff"] = {
        "cadence": str(cfg.get("cadence") or "off"),
        "hour": int(cfg.get("hour") or 0),
        "last_run_ts": float(cfg.get("last_run_ts") or 0.0),
        "last_diff": dict(cfg.get("last_diff") or {}),
    }
    surface.store.save()


def roster_diff_due(cfg: dict[str, Any], now: float) -> bool:
    """Is a scheduled run due? One run per interval, never before
    the configured wall-clock hour."""
    cadence = str(cfg.get("cadence") or "off")
    if cadence not in ("daily", "weekly"):
        return False
    interval = 86400.0 if cadence == "daily" else 604800.0
    last = float(cfg.get("last_run_ts") or 0.0)
    if now - last < interval and last:
        return False
    return time.localtime(now).tm_hour >= int(cfg.get("hour") or 0)


def next_run_desc(cfg: dict[str, Any], now: float) -> str:
    cadence = str(cfg.get("cadence") or "off")
    if cadence not in ("daily", "weekly"):
        return "off — the roster is checked only when you ask"
    interval = 86400.0 if cadence == "daily" else 604800.0
    last = float(cfg.get("last_run_ts") or 0.0)
    hour = int(cfg.get("hour") or 0)
    lt = time.localtime(now)
    candidate = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday,
                             hour, 0, 0, lt.tm_wday, lt.tm_yday,
                             lt.tm_isdst))
    if candidate <= now:
        candidate += interval
    if last and candidate - last < interval:
        candidate = last + interval
    return ("next diff "
            + time.strftime("%a %d %b %H:%M",
                            time.localtime(candidate)))


def set_roster_diff_schedule(surface: Any, group_key: str, *,
                             cadence: str, hour: int | None = None,
                             by: str = "") -> dict[str, Any]:
    """Set one group's roster-diff cadence. Returns the merged
    block plus the next-run line for the reply."""
    cadence = str(cadence or "").lower()
    if cadence not in CADENCES:
        raise GroupRosterDiffRefused(
            f"cadence is off, daily or weekly — not {cadence!r}.")
    cfg = roster_diff_settings(surface, group_key)
    cfg["cadence"] = cadence
    if hour is not None:
        if not 0 <= int(hour) <= 23:
            raise GroupRosterDiffRefused(
                f"hour is 0–23 — not {hour}.")
        cfg["hour"] = int(hour)
    _write_roster_diff_settings(surface, group_key, cfg)
    surface._audit(group_key, "roster_diff_schedule",
                   by=str(by or ""), cadence=cadence,
                   hour=cfg["hour"])
    with contextlib.suppress(OSError):
        surface.store.save()
    return {"settings": cfg,
            "next": next_run_desc(cfg, time.time())}


def describe_roster_diff(cfg: dict[str, Any]) -> str:
    """The one-line roster-diff state the panels share: cadence
    + last run + last counts."""
    cadence = str(cfg.get("cadence") or "off")
    if cadence == "off":
        return "off"
    line = f"{cadence} at {int(cfg.get('hour') or 0):02d}:00"
    last = float(cfg.get("last_run_ts") or 0.0)
    if last:
        line += (" · last "
                 + time.strftime("%d %b %H:%M",
                                 time.localtime(last)))
    diff = cfg.get("last_diff") or {}
    if diff:
        parts = []
        if diff.get("joined"):
            parts.append(f"+{diff['joined']} joined")
        if diff.get("left"):
            parts.append(f"−{diff['left']} left")
        if diff.get("changed"):
            parts.append(f"~{diff['changed']} changed")
        if diff.get("deleted"):
            parts.append(f"{diff['deleted']} deleted")
        if diff.get("roles"):
            parts.append(f"{diff['roles']} role changes")
        if parts:
            line += " · " + ", ".join(parts)
        elif not diff.get("seeded"):
            line += " · no changes"
        if diff.get("truncated"):
            line += " · list truncated"
    return line


def _normalise_row(row: dict[str, Any]) -> dict[str, Any]:
    """One live member row in the diff's shape (the cleanup
    scan's row shape, so the two agree on what a member is)."""
    return {
        "user_id": str(row.get("id") or row.get("user_id") or ""),
        "name": str(row.get("name") or ""),
        "username": str(row.get("username") or ""),
        "bot": bool(row.get("bot") or row.get("is_bot")),
        "deleted": bool(is_deleted_row(row)),
        "status_kind": str(row.get("status_kind") or ""),
        "was_online": row.get("was_online"),
        "last_name": str(row.get("last_name") or ""),
    }


def _chat_ref(platform: str, chat_id: str) -> ChatRef:
    return ChatRef(platform=platform, chat_id=chat_id,
                   kind="group", title="")


def fetch_live_roster(adapters: list, group_key: str, *,
                      surface: Any = None) -> dict[str, Any]:
    """Fetch the live member list for one group from the best
    enumeration the adapters actually offer:

    1. any adapter's FULL member list (MTProto) — every member,
       with the honest ``truncated`` flag;
    2. any adapter's capped member list;
    3. the Bot API probe path: no member list exists there, so
       each known ledger member is probed once via getChatMember
       (small ledgers only — above MAX_PROBE_LEDGER the group is
       skipped with the reason recorded).

    ``adapters`` is a list of (platform, adapter) pairs, the
    cleanup scan's candidate shape.  Returns {"ok": True, "rows",
    "source", "truncated", "admin_ids"} or {"ok": False,
    "error"} — absence is never dressed as an empty roster.
    """
    platform = str(group_key).split(":", 1)[0]
    chat_id = str(group_key).split(":", 1)[1] \
        if ":" in group_key else ""
    if not chat_id:
        return {"ok": False, "error": "no chat id in the key"}
    admin_ids: set | None = None
    for _plat, adapter in adapters:
        fn = getattr(adapter, "admin_administrators", None)
        if fn is None:
            continue
        try:
            result = fn(_chat_ref(_plat, chat_id))
        except Exception:  # noqa: BLE001 - a path, not a need
            result = None
        if isinstance(result, dict) and result.get("ok"):
            admin_ids = {str(row.get("id") or "")
                         for row in result.get("administrators")
                         or []
                         if isinstance(row, dict)
                         and row.get("id")}
            break
    rows: list | None = None
    source = ""
    truncated = False
    for _plat, adapter in adapters:
        fn = getattr(adapter, "admin_members_full", None)
        if fn is None:
            continue
        try:
            result = fn(_chat_ref(_plat, chat_id))
        except Exception:  # noqa: BLE001 - degrade a path
            result = None
        if isinstance(result, dict) and result.get("ok"):
            rows = [_normalise_row(r)
                    for r in result.get("members") or []
                    if isinstance(r, dict)]
            source = "the userbot's full member list"
            truncated = bool(result.get("truncated"))
            break
    if rows is None:
        for _plat, adapter in adapters:
            fn = getattr(adapter, "admin_members", None)
            if fn is None:
                continue
            try:
                result = fn(_chat_ref(_plat, chat_id), limit=500)
            except Exception:  # noqa: BLE001 - degrade a path
                result = None
            if isinstance(result, dict) and result.get("ok"):
                rows = [_normalise_row(r)
                        for r in result.get("members") or []
                        if isinstance(r, dict)]
                source = "the member list (first 500)"
                truncated = len(rows) >= 500
                break
    if rows is None:
        # WhatsApp: neither bridge speaks the Telegram-shaped
        # admin_members verbs, but both answer group_participants
        # (Go gobridge/groups.go, Node whatsapp-groups.mjs) — the
        # member roster with admin roles.  Without this branch
        # every WhatsApp group failed the diff with "no
        # member-list route", so churn was traffic-only.
        for _plat, adapter in adapters:
            fn = getattr(adapter, "group_participants", None)
            if fn is None:
                continue
            try:
                parts = fn(_chat_ref(_plat, chat_id))
            except Exception:  # noqa: BLE001 - degrade a path
                parts = None
            if not parts:
                continue
            rows = []
            if admin_ids is None:
                admin_ids = set()
            for entry in parts:
                if not isinstance(entry, dict):
                    continue
                jid = str(entry.get("jid")
                          or entry.get("id") or "")
                if not jid:
                    continue
                rows.append(_normalise_row({
                    "id": jid,
                    "name": entry.get("name") or "",
                    "username": "",
                    "bot": False,
                    "deleted": False,
                    "status_kind": "",
                    "was_online": None,
                    "last_name": "",
                }))
                if str(entry.get("role") or "").lower() in (
                        "admin", "superadmin"):
                    admin_ids.add(jid)
            if rows:
                source = "the bridge's group roster"
                truncated = False
                break
            rows = None
    if rows is None and surface is not None \
            and platform.startswith("telegram"):
        probe = None
        probe_ref = None
        for _plat, adapter in adapters:
            fn = getattr(adapter, "admin_member_profile", None)
            if fn is not None:
                probe, probe_ref = fn, _chat_ref(_plat, chat_id)
                break
        ledger = surface.members(group_key)
        if probe is None:
            return {"ok": False,
                    "error": "no member-list route on this "
                             "platform — the Bot API has no "
                             "member list and no adapter here "
                             "offers one"}
        if len(ledger) > MAX_PROBE_LEDGER:
            return {"ok": False,
                    "error": f"no member-list route and the "
                             f"ledger ({len(ledger)} members) is "
                             f"too large to probe one by one"}
        rows = []
        for uid, led in ledger.items():
            if led.get("left_ts"):
                continue
            row = {
                "user_id": uid,
                "name": str(led.get("name") or ""),
                "username": str(led.get("username") or ""),
                "bot": bool(led.get("is_bot")),
                "deleted": False,
                "status_kind": "",
                "was_online": None,
                "last_name": str(led.get("last_name") or ""),
            }
            try:
                prof = probe(probe_ref, uid)
            except Exception:  # noqa: BLE001 - unprobed
                prof = None
            if isinstance(prof, dict) and prof.get("ok"):
                if str(prof.get("status") or "").lower() \
                        in ("left", "kicked"):
                    continue  # already out — no ghost
                user = prof.get("user") or {}
                row["name"] = " ".join(filter(None, [
                    str(user.get("first_name") or ""),
                    str(user.get("last_name") or ""),
                ])).strip() or row["name"]
                row["last_name"] = str(
                    user.get("last_name") or "")
                row["username"] = str(
                    user.get("username") or "") or row["username"]
                row["bot"] = bool(user.get("is_bot"))
                row["deleted"] = is_deleted_row(row)
            rows.append(row)
        source = ("the member ledger, probed one by one — the "
                  "Bot API has no member list")
        truncated = False
    if rows is None:
        return {"ok": False,
                "error": "no member-list route on this platform"}
    return {"ok": True, "rows": rows, "source": source,
            "truncated": truncated, "admin_ids": admin_ids}


def diff_group_roster(surface: Any, group_key: str,
                      rows: list[dict[str, Any]], *,
                      now: float | None = None,
                      admin_ids: set | None = None,
                      truncated: bool = False,
                      source: str = "") -> dict[str, Any]:
    """Diff live rows against the last snapshot and apply what
    changed to the ledger.  Joins ride ``note_join`` (silent —
    no welcome render), identity changes ride
    ``_observe_identity`` (change ring + durable history +
    audit), leaves ride the silent ``note_leave`` (no goodbye).
    Returns the run summary; the snapshot + last-run summary are
    stored on the group entry."""
    now = now if now is not None else time.time()
    entry = surface.store.group(group_key)
    ledger = entry["members"]
    state = surface._extras(group_key).setdefault(
        "roster_diff", {})
    snapshot = dict(state.get("snapshot") or {})
    snap_admins = set(state.get("admin_ids") or [])
    seeded = not snapshot
    live_ids = {r["user_id"] for r in rows if r.get("user_id")}
    summary: dict[str, Any] = {
        "ok": True, "group": group_key, "seeded": seeded,
        "joined": [], "left": [], "changed": [], "deleted": [],
        "roles": [], "total_live": len(live_ids),
        "truncated": bool(truncated), "source": source,
    }
    for row in rows:
        uid = row.get("user_id") or ""
        if not uid:
            continue
        known = uid in ledger
        if not known and seeded:
            # First run: seed silently.  _observe_identity
            # creates the member record (first seen by Devon:
            # now) without a join audit — the member is "not
            # yet known", not "joined".
            surface._observe_identity(
                group_key, uid, username=row.get("username", ""),
                name=row.get("name", ""), ts=now)
            continue
        if not known:
            member = surface.note_join(
                group_key, uid, username=row.get("username", ""),
                name=row.get("name", ""),
                is_bot=bool(row.get("bot")))
            summary["joined"].append(uid)
            if row.get("deleted") and member is not None:
                member["deleted"] = True
                surface._audit(group_key, "deleted_account",
                               user_id=uid, deleted=True)
                summary["deleted"].append(uid)
            continue
        events = surface._observe_identity(
            group_key, uid, username=row.get("username", ""),
            name=row.get("name", ""), ts=now)
        for event in events:
            summary["changed"].append({
                "user_id": uid, "kind": event.get("kind", ""),
                "old": event.get("old", ""),
                "new": event.get("new", "")})
        member = ledger.get(uid) or {}
        if member.get("left_ts"):
            # Back after a recorded leave: the join clears it.
            surface.note_join(
                group_key, uid, username=row.get("username", ""),
                name=row.get("name", ""),
                is_bot=bool(row.get("bot")))
            summary["joined"].append(uid)
        was_deleted = bool(member.get("deleted"))
        if row.get("deleted") and not was_deleted:
            member["deleted"] = True
            surface._audit(group_key, "deleted_account",
                           user_id=uid, deleted=True)
            summary["deleted"].append(uid)
        elif not row.get("deleted") and was_deleted:
            member["deleted"] = False
            surface._audit(group_key, "deleted_account",
                           user_id=uid, deleted=False)
    if not truncated:
        # Absence from a complete live list IS a leave.  From a
        # truncated list it proves nothing — never asserted.
        for uid in sorted(ledger):
            if uid in live_ids:
                continue
            member = ledger.get(uid) or {}
            if member.get("left_ts"):
                continue
            surface.note_leave(group_key, uid, ts=now)
            summary["left"].append(uid)
    if admin_ids is not None and not seeded:
        gained = sorted(set(admin_ids) - snap_admins)
        lost = sorted(snap_admins - set(admin_ids))
        for uid in gained:
            surface._audit(group_key, "role_changed",
                           user_id=uid, admin=True)
            summary["roles"].append({"user_id": uid,
                                     "admin": True})
        for uid in lost:
            surface._audit(group_key, "role_changed",
                           user_id=uid, admin=False)
            summary["roles"].append({"user_id": uid,
                                     "admin": False})
    state["snapshot"] = {
        row["user_id"]: {
            "username": row.get("username", ""),
            "name": row.get("name", ""),
            "bot": bool(row.get("bot")),
            "deleted": bool(row.get("deleted")),
        }
        for row in rows if row.get("user_id")}
    if admin_ids is not None:
        state["admin_ids"] = sorted(admin_ids)
    state["last_run_ts"] = now
    cfg = roster_diff_settings(surface, group_key)
    cfg["last_run_ts"] = now
    cfg["last_diff"] = {
        "joined": len(summary["joined"]),
        "left": len(summary["left"]),
        "changed": len(summary["changed"]),
        "deleted": len(summary["deleted"]),
        "roles": len(summary["roles"]),
        "seeded": seeded,
        "truncated": bool(truncated),
        "total_live": len(live_ids),
    }
    _write_roster_diff_settings(surface, group_key, cfg)
    surface._audit(group_key, "roster_diff",
                   joined=len(summary["joined"]),
                   left=len(summary["left"]),
                   changed=len(summary["changed"]),
                   deleted=len(summary["deleted"]),
                   roles=len(summary["roles"]),
                   seeded=seeded, truncated=bool(truncated),
                   total_live=len(live_ids), source=source)
    with contextlib.suppress(OSError):
        surface.store.save()
    return summary


def run_roster_diff_tick_once(*, context: Any = None,
                              surface: Any = None,
                              workspace_dir: str | Path | None = None,
                              now: float | None = None,
                              fetch: Any = None,
                              adapters: list | None = None
                              ) -> dict[str, Any]:
    """One scheduled roster-diff pass: every CLAIMED group whose
    cadence is due gets its live roster diffed against the last
    snapshot.  Routine and silent — nothing is posted anywhere;
    each run journals to its group's audit log.  One bad group
    never kills the pass.

    ``fetch`` (tests / the manual verb) overrides the live
    fetch: a callable ``(adapters, group_key, surface)`` returning
    the ``fetch_live_roster`` shape.  ``adapters`` is the live
    gateway's adapter list when the caller resolved it; when both
    are absent the tick resolves the gateway from the context.
    """
    now = now if now is not None else time.time()
    if surface is None:
        ws = workspace_dir
        if ws is None and context is not None:
            ws = getattr(getattr(context, "settings", None),
                         "workspace_dir", ".") or "."
        from .group_surface import default_surface

        surface = default_surface(ws or ".")
    if adapters is None and fetch is None and context is not None:
        try:
            # Layered: resolve_gateway() (nomorals/agents/notifier, L5)
            # is a plain attribute lookup — social/chat (L4) must not
            # import upward for it (the patch's `...agents.notifier`
            # import adds a new layering violation). The runtime stores
            # the gateway on context.gateway or in
            # context.extras["gateway"]; read it here instead.
            gateway = getattr(context, "gateway", None)
            if gateway is None:
                extras = getattr(context, "extras", None)
                if isinstance(extras, dict):
                    gateway = extras.get("gateway")
            adapters = list((getattr(gateway, "adapters", None)
                             or {}).items()) if gateway else []
        except Exception as exc:  # noqa: BLE001 - no gateway, no live fetch
            record_failure("group_roster_diff.gateway", exc)
            adapters = []
    summary: dict[str, Any] = {"ok": True, "diffed": [],
                               "failed": [], "skipped": 0}
    for key in sorted(surface.store.data.keys()):
        try:
            if surface.is_channel(key):
                continue
        except Exception:  # noqa: BLE001 - an unreadable key is skipped, not fatal
            continue
        try:
            if not surface.claim_state(key)["claimed"]:
                summary["skipped"] += 1
                continue
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            record_failure("group_roster_diff.claim", exc)
            summary["failed"].append({"group": key,
                                      "error": str(exc)})
            continue
        try:
            cfg = roster_diff_settings(surface, key)
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            record_failure("group_roster_diff.tick_settings", exc)
            summary["failed"].append({"group": key,
                                      "error": str(exc)})
            continue
        if not roster_diff_due(cfg, now):
            summary["skipped"] += 1
            continue
        try:
            if fetch is not None:
                fetched = fetch(adapters or [], key, surface)
            else:
                fetched = fetch_live_roster(adapters or [], key,
                                            surface=surface)
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            record_failure("group_roster_diff.tick_fetch", exc)
            summary["failed"].append({"group": key,
                                      "error": str(exc)})
            continue
        if not isinstance(fetched, dict) or not fetched.get("ok"):
            summary["failed"].append({
                "group": key,
                "error": str((fetched or {}).get("error")
                             or "no member-list route")})
            continue
        try:
            result = diff_group_roster(
                surface, key, fetched.get("rows") or [],
                now=now, admin_ids=fetched.get("admin_ids"),
                truncated=bool(fetched.get("truncated")),
                source=str(fetched.get("source") or ""))
            summary["diffed"].append(result)
        except Exception as exc:  # noqa: BLE001 - one bad group never kills the pass
            record_failure("group_roster_diff.tick", exc)
            summary["failed"].append({"group": key,
                                      "error": str(exc)})
    return summary


def ensure_group_roster_diff_job(scheduler: Any, *,
                                 spec: str = ROSTER_DIFF_SPEC
                                 ) -> Any:
    """Register the durable group-roster-diff tick job.
    Idempotent — mirrors ``ensure_group_backup_job``: an existing
    job at the requested cadence is returned as-is, a stale one
    is replaced.  Best-effort: never raises."""
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name == ROSTER_DIFF_JOB:
                job_spec = (job.get("spec") if isinstance(job, dict)
                            else getattr(job, "spec", None))
                if job_spec == spec:
                    return job
                with contextlib.suppress(Exception):
                    job_id = (job.get("id") if isinstance(job, dict)
                              else getattr(job, "id", None))
                    scheduler.remove(job_id)
                break
        return scheduler.add(ROSTER_DIFF_JOB, spec, "tool",
                             {"tool": ROSTER_DIFF_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
