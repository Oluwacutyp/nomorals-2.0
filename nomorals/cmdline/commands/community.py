"""``nm community`` — the community board from the console.

The console twin of chat /community's read verbs
(``agents/partner/runtime_group.py: _control_community`` +
``_community_list`` + ``_community_show`` + ``_community_usage``)
over the same communities index (``social.chat.group_community``)
and the same durable arms store (``social.mass_actions``): list
every community, show one community's groups, see the arm state.
The console is the owner's seat, so it shows every community — no
admin gate (chat's gate proves each sender's adminship through the
platform; the console needs no proof). Post/mirror/arm/disarm/link/
unlink/sync/member change state or send, so they stay chat-only and
are refused in words from the console, the way ``nm museum send``
is refused. ``nm community mirror`` is deliberately not built
(R2-26: the mirror engine's loop-safety contract is the live
inbound path, not a console verb).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_community"]

# Verbs that change state or send — chat-only, refused in words.
_REFUSALS = {
    "post": ("posting needs a live chat — run /community post "
             "from a chat, not the console"),
    "mirror": ("mirroring needs a live chat — /community mirror "
               "stays chat-only; nm community mirror is "
               "deliberately not built"),
    "arm": ("arming needs a live chat — run /community arm "
            "from a chat, not the console"),
    "disarm": ("disarming needs a live chat — run /community "
               "disarm from a chat, not the console"),
    "link": ("linking needs a live chat — run /community link "
             "from a chat, not the console"),
    "unlink": ("unlinking needs a live chat — run /community "
               "unlink from a chat, not the console"),
    "sync": ("syncing needs a live chat — run /community sync "
             "from a chat, not the console"),
    "member": ("roster changes need a live chat — run "
               "/community member from a chat, not the console"),
}


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


def _usage() -> str:
    return ("nm community — list every community carrying a group\n"
            "nm community show <id|name> — the groups in one community\n"
            "nm community status — the post/mirror arm state\n"
            "post · mirror · arm · disarm · link · unlink · sync · "
            "member stay chat-only — run /community from a chat")


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _list(surface: Any) -> str:
    """Bare /community — the mirror of
    ``runtime_group._community_list`` minus the admin gate."""
    from ...social.chat import group_community as _gc

    index = _gc.communities_index(surface)
    rows: list[str] = []
    for cid in sorted(index):
        row = index[cid]
        name = row["name"] or cid
        platforms = ", ".join(row["platforms"])
        rows.append(f"{name} ({cid}) — {platforms} · "
                    f"{len(row['groups'])} group(s)")
    if not rows:
        return ("no community carries a group yet — link one with "
                "/community link <g1> <id>, or sync a group whose "
                "platform already put it in a community "
                "(/community sync <g1>).")
    return "communities:\n" + "\n".join(rows)


def _resolve_id(surface: Any, query: str):
    """Resolve a community id or unique name to (cid, row, None),
    or (None, None, error) — the mirror of
    ``runtime_group._community_resolve_id`` (same words)."""
    from ...social.chat import group_community as _gc

    index = _gc.communities_index(surface)
    query = str(query or "").strip()
    if query in index:
        return query, index[query], None
    matches = [(cid, row) for cid, row in index.items()
               if str(row.get("name") or "").lower()
               == query.lower()]
    if len(matches) == 1:
        return matches[0][0], matches[0][1], None
    if len(matches) > 1:
        return None, None, (
            f"more than one community is named “{query}” — "
            "use its id instead.")
    return None, None, (
        f"no community “{query}” carries a group I serve — "
        "/community lists the ones that do.")


def _show(surface: Any, query: str) -> str:
    """One community's member groups, with claim state — the mirror
    of ``runtime_group._community_show`` minus the admin gate."""
    from ...social.chat import group_community as _gc

    cid, row, error = _resolve_id(surface, query)
    if error:
        return error
    groups = row["groups"]
    if not groups:
        return (f"community {row['name'] or cid} carries no "
                "group in the store.")
    lines = [f"community {row['name'] or cid} ({cid}):"]
    for key in groups:
        claim = surface.claim_state(key)
        tag = _gc.community_of(surface, key) or {}
        lines.append(
            f"· {key} — "
            f"{'claimed' if claim['claimed'] else 'unclaimed'}"
            f" · tag: {tag.get('source') or '?'}")
    return "\n".join(lines)


def _status(context: Any) -> tuple[str, dict[str, Any]]:
    """The post/mirror arm state — the mirror of the ``status`` leg
    of ``runtime_group._control_community``. The rate line names the
    per-minute limit only: the limiter is per-process, and a console
    run is a fresh process, so a remaining count would always read
    full and mean nothing."""
    from ...social.chat.whatsapp_communities import CommunityRateLimiter
    from ...social.mass_actions import MassArms, default_mass_root

    arms = MassArms(default_mass_root(_workspace_dir(context))
                    / "arms.json")
    limiter = CommunityRateLimiter()
    post_state = "armed" if arms.armed("community-post") else "disarmed"
    mirror_state = ("armed" if arms.armed("community-mirror")
                    else "disarmed")
    text = (f"/community post: {post_state} · live runs: "
            f"{arms.live_runs('community-post')} · rate: "
            f"{limiter.limit} post/min\n"
            f"/community mirror: {mirror_state} · live runs: "
            f"{arms.live_runs('community-mirror')}")
    data = {
        "post": {"armed": arms.armed("community-post"),
                 "live_runs": arms.live_runs("community-post"),
                 "rate_per_min": limiter.limit},
        "mirror": {"armed": arms.armed("community-mirror"),
                   "live_runs": arms.live_runs("community-mirror")},
    }
    return text, data


def _cmd_community(args: argparse.Namespace, context: Any) -> int:
    """``nm community [list|show <id|name>|status]``."""
    action = getattr(args, "community_action", None)
    rest = list(getattr(args, "community_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    verb = str(action).lower() if action is not None else "list"
    if verb in ("list", "ls"):
        try:
            surface = _surface(context)
        except Exception as exc:  # noqa: BLE001 - surface path is the whole command
            return _usage_error(f"community unavailable: {exc}")
        if as_json:
            from ...social.chat import group_community as _gc

            print(json.dumps({"communities": _gc.communities_index(surface)},
                             indent=2, default=str))
            return 0
        print(_list(surface))
        return 0
    if verb in ("help", "?"):
        print(_usage())
        return 0
    if verb == "show":
        if not rest:
            return _usage_error(
                "usage: nm community show <id|name> — nm community "
                "lists the communities and their ids")
        try:
            surface = _surface(context)
        except Exception as exc:  # noqa: BLE001 - surface path is the whole command
            return _usage_error(f"community unavailable: {exc}")
        if as_json:
            from ...social.chat import group_community as _gc

            cid, row, error = _resolve_id(surface, " ".join(rest))
            if error:
                print(error)
                return 0
            groups = []
            for key in row["groups"]:
                claim = surface.claim_state(key)
                tag = _gc.community_of(surface, key) or {}
                groups.append({"key": key,
                               "claimed": bool(claim["claimed"]),
                               "tag": tag.get("source") or None})
            print(json.dumps({"community": cid, "name": row["name"],
                              "groups": groups},
                             indent=2, default=str))
            return 0
        print(_show(surface, " ".join(rest)))
        return 0
    if verb == "status":
        try:
            text, data = _status(context)
        except Exception as exc:  # noqa: BLE001 - arms path is the whole command
            return _usage_error(f"community unavailable: {exc}")
        if as_json:
            print(json.dumps(data, indent=2, default=str))
            return 0
        print(text)
        return 0
    if verb in _REFUSALS:
        print(_REFUSALS[verb])
        return 0
    return _usage_error(_usage())
