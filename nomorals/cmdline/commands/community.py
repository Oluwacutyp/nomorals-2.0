"""``nm community`` — the community board from the console.

The console twin of chat /community's read verbs
(``agents/partner/runtime_group.py: _control_community`` +
``_community_list`` + ``_community_show`` + ``_community_usage``)
over the same communities index (``social.chat.group_community``)
and the same durable arms store (``social.mass_actions``): list
every community, show one community's groups, see the arm state,
list one group's roster. The console is the owner's seat, so it
shows every community — no admin gate (chat's gate proves each
sender's adminship through the platform; the console needs no
proof). Post/mirror/arm/disarm/link/unlink/sync/member add|link|
unlink change state or send, so they stay chat-only and are refused
in words from the console, the way ``nm museum send`` is refused.
``nm community mirror`` is deliberately not built (R2-26: the
mirror engine's loop-safety contract is the live inbound path, not
a console verb). The roster read (``member list``) is the mirror
of chat ``/community member list`` over the same per-group
``BridgeStore`` (``community/bridges.py``) — the store is
home-based (``~/.devon/community/bridges``), the same store the
runtime reads, so the console uses the default too; a
workspace-based dir would read a different, empty store. The
group handle resolves the way chat's ``_community_resolve`` does
minus the console session-map leg (no session on the console) and
minus the admin gate: the cold gN/cN ordinal, then the exact key,
then one unique substring match.
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
}

# Roster writes — chat-only, refused in words (``member list`` is
# the read half and lives in _member below).
_MEMBER_REFUSAL = ("roster changes need a live chat — run "
                   "/community member from a chat, not the console")


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


def _bridge_store() -> Any:
    """The per-group roster store — the same home-based default the
    runtime reads (``runtime_group._community_bridge_store`` builds
    ``BridgeStore(None)`` too). Tests monkeypatch this name."""
    from ...community.bridges import BridgeStore

    return BridgeStore()


def _usage() -> str:
    return ("nm community — list every community carrying a group\n"
            "nm community show <id|name> — the groups in one community\n"
            "nm community status — the post/mirror arm state\n"
            "nm community member list <g1|key> — the roster: members "
            "and their linked platform identities\n"
            "post · mirror · arm · disarm · link · unlink · sync · "
            "member add/link/unlink stay chat-only — run /community "
            "from a chat")


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


def _resolve_group(surface: Any, token: str):
    """Resolve a group handle to a store key: the cold gN/cN
    ordinal, then the exact key, then one unique substring match —
    the mirror of ``runtime_group._community_resolve`` minus the
    console session-map leg (no session on the console) and minus
    the admin gate (the console is the owner's seat). Returns
    (key, None) or (None, error), in chat's words."""
    import re as _re

    token = str(token or "").strip()
    keys = sorted(surface.store.data.keys())
    group_keys = [k for k in keys if not surface.is_channel(k)]
    channel_keys = [k for k in keys if surface.is_channel(k)]
    key = None
    match = _re.fullmatch(r"g(\d+)", token.lower())
    match_c = _re.fullmatch(r"c(\d+)", token.lower())
    if match:
        index = int(match.group(1)) - 1
        if 0 <= index < len(group_keys):
            key = group_keys[index]
    elif match_c:
        index = int(match_c.group(1)) - 1
        if 0 <= index < len(channel_keys):
            key = channel_keys[index]
    if key is None and token in keys:
        key = token
    elif key is None:
        matches = [k for k in keys if token.lower() in k.lower()]
        if len(matches) == 1:
            key = matches[0]
        elif len(matches) > 1:
            return None, (
                f"more than one group matches “{token}” — "
                "use the handle from the community board "
                "(g1, g2…).")
    if key is None:
        return None, ("that group handle expired — pick the "
                      "group again from the community board.")
    return key, None


def _member_list(surface: Any, token: str) -> str:
    """One group's roster — the mirror of the ``list`` branch of
    ``runtime_group._community_member``, same words, minus the
    admin gate."""
    key, error = _resolve_group(surface, token)
    if error:
        return error
    roster = _bridge_store().load(key)
    if not roster.members:
        return (f"no members in {key} yet — add one with "
                f"/community member add {key} <name>.")
    lines = [f"members in {key}:"]
    for member in roster.members:
        links = ", ".join(
            f"{p}:{u}"
            for p, u in sorted(member.platforms.items()))
        lines.append(
            f"  · {member.display_name} ({member.id})"
            + (f" — {links}" if links
               else " — no linked identities"))
    return "\n".join(lines)


def _member(context: Any, rest: list[str],
            as_json: bool) -> int:
    """``nm community member list <g1|key>`` — the roster read.
    add/link/unlink change the roster, so they stay chat-only and
    are refused in words."""
    sub = str(rest[0]).lower() if rest else ""
    if sub in ("add", "link", "unlink"):
        print(_MEMBER_REFUSAL)
        return 0
    if sub != "list":
        return _usage_error(
            "usage: nm community member list <g1|key> — nm community "
            "lists the communities and their groups")
    if len(rest) < 2:
        return _usage_error(
            "usage: nm community member list <g1|key> — nm community "
            "lists the communities and their groups")
    token = " ".join(rest[1:])
    try:
        surface = _surface(context)
    except Exception as exc:  # noqa: BLE001 - surface path is the whole command
        return _usage_error(f"community unavailable: {exc}")
    if as_json:
        key, error = _resolve_group(surface, token)
        if error:
            print(error)
            return 0
        roster = _bridge_store().load(key)
        print(json.dumps(
            {"group": key,
             "members": [{"id": m.id,
                          "display_name": m.display_name,
                          "platforms": dict(m.platforms)}
                         for m in roster.members]},
            indent=2, default=str))
        return 0
    print(_member_list(surface, token))
    return 0


def _cmd_community(args: argparse.Namespace, context: Any) -> int:
    """``nm community [list|show <id|name>|status|member list <g1|key>]``."""
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
    if verb == "member":
        return _member(context, rest, as_json)
    if verb in _REFUSALS:
        print(_REFUSALS[verb])
        return 0
    return _usage_error(_usage())
