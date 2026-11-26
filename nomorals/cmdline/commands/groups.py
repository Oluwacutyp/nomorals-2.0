"""``nm groups`` — the groups board from the console.

The console twin of chat /groups (``agents/partner/runtime_group.py:
_control_groups`` + ``_groups_board`` + ``_groups_group_lines``) over
the same ``GroupSurface`` (``social.chat.group_surface.
default_surface`` from the console workspace dir): every group/channel
Devon serves — status, member count, economy health, pending actions.
The per-group lines are the board's own lines, rendered as plain
text instead of a Card: the deep-link buttons and the console-session
map have no console equivalent, and the board's Status block is
console-session state, not group data, so it is not shown. The
console is the owner's seat, so the board shows every group — no
admin gate (chat's gate proves each sender's adminship through the
platform; the console needs no proof).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_groups"]


def _workspace_dir(context: Any) -> Path:
    settings = getattr(context, "settings", None)
    ws = getattr(settings, "workspace_dir", None) if settings else None
    if not ws:
        raise ValueError("no workspace_dir on context")
    return Path(ws)


def _surface(context: Any) -> Any:
    """The one surface construction site on the console — the twin of
    ``runtime_group._group_surface`` (minus the owner-registry merge,
    a runtime concern)."""
    from ...social.chat.group_surface import default_surface

    return default_surface(_workspace_dir(context))


def _duration_text(seconds: int) -> str:
    """Mirror of ``runtime_group._duration_text`` (same thresholds,
    same text) — the console must not import the runtime for one
    formatting helper."""
    if seconds >= 86400:
        return f"{seconds // 86400}d"
    if seconds >= 3600:
        return f"{seconds // 3600}h"
    if seconds >= 60:
        return f"{seconds // 60}m"
    return f"{seconds}s"


def _waiting(surface: Any, key: str) -> str:
    """Pending actions for one group — the mirror of
    ``runtime_group._groups_waiting`` (§8's list, nonzero counts
    only; "" means nothing waits)."""
    parts: list = []
    joins = len(surface.acceptance_queue(key))
    if joins:
        parts.append(f"join {joins}")
    entry = surface.store.group(key)
    restricted = sum(
        1 for m in (entry.get("members") or {}).values()
        if isinstance(m, dict)
        and isinstance(m.get("acceptance"), dict)
        and m["acceptance"].get("state") == "restricted")
    if restricted:
        parts.append(f"restricted {restricted}")
    flags = len(surface.flag_proposals(key))
    if flags:
        parts.append(f"flags {flags}")
    reports = len(surface.open_reports(key))
    if reports:
        parts.append(f"reports {reports}")
    proposals = len(surface.open_proposals(key))
    if proposals:
        parts.append(f"proposals {proposals}")
    appeals = len([a for a in surface.bans.appeals(status="open")
                   if a.group_id == key])
    if appeals:
        parts.append(f"appeals {appeals}")
    heist = surface.heist_waiting(key)
    if heist:
        parts.append(heist)
    return " · ".join(parts)


def _group_lines(surface: Any, key: str, token: str) -> list[str]:
    """One group's board rows — the mirror of
    ``runtime_group._groups_group_lines`` (same calls, same text)."""
    from ...social.chat import group_community as _gc
    from ...social.chat import group_defense as _gd
    from ...social.chat import group_health as _gh
    from ...social.chat import group_roster_diff as _rd

    members = surface.store.group(key).get("members") or {}
    claim = surface.claim_state(key)
    claim_txt = (f"system output on ({claim['source']})"
                 if claim["claimed"] else "system output off")
    lines = [f"{token} — {key} ({len(members)} in the ledger) · "
             f"{claim_txt}"]
    frozen = surface.economy_frozen(key)
    if frozen is not None:
        lines.append(f"economy frozen: {frozen}")
    elif not surface.economy_on(key):
        lines.append("economy off")
    else:
        health = surface.economy_health(key)
        econ = (f"treasury {health['treasury']} · "
                f"holders {health['holders']} · "
                f"active 7d {health['active_7d']}")
        if health["policy"].get("throttled"):
            econ += " · faucets halved"
        lines.append(econ)
    delay = surface.slowmode_delay(key)
    lines.append("slow mode "
                 f"{'off' if not delay else _duration_text(delay)}")
    lines.append(
        "roster diff "
        f"{_rd.describe_roster_diff(_rd.roster_diff_settings(surface, key))}")
    if key.partition(":")[0].strip().lower() in ("whatsapp", "wa"):
        # WhatsApp group health (census §6 #6): the board's
        # one-liner per WA group.  Best-effort — a board
        # never dies on one group's health.
        try:
            _wa_health = _gh.build_group_health(surface, key)
        except Exception:  # noqa: BLE001 - board survives
            _wa_health = None
        if _wa_health is not None:
            lines.append(f"health {_gh.one_line(_wa_health)}")
    lines.append(
        "media locks "
        f"{_gd.describe_media_locks((surface.store.settings(key).get('defense') or {}).get('media_locks') or {})}")
    if _gc.community_of(surface, key) is not None:
        lines.append(_gc.describe_community(surface, key))
    waiting = _waiting(surface, key)
    lines.append(f"waiting: {waiting}" if waiting
                 else "nothing waiting")
    return lines


def _channel_lines(surface: Any, key: str, token: str) -> list[str]:
    """One channel's board row — the mirror of the board's channel
    section in ``runtime_group._groups_board``."""
    claim = surface.claim_state(key)
    title = surface.channel_title(key) or key
    return [f"{token} — {title} · "
            f"{'claimed' if claim['claimed'] else 'unclaimed'}"
            f" · {len(surface.channel_queue(key))} queued"
            f" · {surface.channel_analytics(key)['posts']} posted"]


def _board(context: Any) -> dict[str, Any]:
    """The board as data: groups (key + the board's own lines) and
    channels (key + title + claim + queue + posts)."""
    surface = _surface(context)
    keys = [k for k in sorted(surface.store.data.keys())
            if not surface.is_channel(k)]
    channel_keys = [k for k in sorted(surface.store.data.keys())
                    if surface.is_channel(k)]
    groups = [{"key": key,
               "lines": _group_lines(surface, key, f"g{i}")}
              for i, key in enumerate(keys, 1)]
    channels = []
    for i, key in enumerate(channel_keys, 1):
        claim = surface.claim_state(key)
        channels.append({
            "key": key,
            "title": surface.channel_title(key) or key,
            "claimed": bool(claim["claimed"]),
            "queued": len(surface.channel_queue(key)),
            "posted": surface.channel_analytics(key)["posts"],
            "lines": _channel_lines(surface, key, f"c{i}"),
        })
    return {"groups": groups, "channels": channels}


def _render(board: dict[str, Any]) -> str:
    lines = ["your groups:"]
    for group in board["groups"]:
        lines.extend(group["lines"])
        lines.append("")
    if board["channels"]:
        lines.append("your channels:")
        for channel in board["channels"]:
            lines.extend(channel["lines"])
    return "\n".join(lines).rstrip() + "\n"


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_groups(args: argparse.Namespace, context: Any) -> int:
    """``nm groups`` — the /groups board from the console."""
    action = getattr(args, "groups_action", None)
    as_json = bool(getattr(args, "json", False))
    if action is not None:
        return _usage_error(
            "usage: nm groups — the board takes no arguments")
    try:
        board = _board(context)
    except Exception as exc:  # noqa: BLE001 - surface path is the whole command
        return _usage_error(f"groups unavailable: {exc}")
    if not board["groups"] and not board["channels"]:
        if as_json:
            print(json.dumps(board, indent=2, default=str))
            return 0
        print("no groups in the store yet — the board fills in "
              "once a group is claimed.")
        return 0
    if as_json:
        print(json.dumps(board, indent=2, default=str))
        return 0
    print(_render(board), end="")
    return 0
