"""``nm digest`` — group digests from the console.

The console twin of chat /digest (``agents/partner/runtime_group.py:
_control_digest`` + ``_digest_visible`` + ``_digest_pick`` +
``_digest_build`` + ``_digest_reply``) over the same
``whatsapp_digest.build_group_digest_text`` and the same
ConversationLog: the board over every WhatsApp group, or one group
by name. The console is the owner's seat, so it shows every
WhatsApp group — no admin gate (chat's gate proves each sender's
adminship through the platform; the console needs no proof).
``policy`` sets state, so it stays chat-only and is refused in words
from the console, the way ``nm museum send`` is refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

__all__ = ["_cmd_digest"]


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


def _log(context: Any) -> Any:
    """The message log chat's digests read — the same ConversationLog
    over the console context's database."""
    from ...storage.conversations import ConversationLog

    db = getattr(context, "db", None)
    if db is None:
        raise ValueError("no message log on this context")
    return ConversationLog(db)


def _visible(surface: Any) -> list[str]:
    """WhatsApp group keys the board covers — the mirror of
    ``runtime_group._digest_visible`` minus the admin gate."""
    visible = []
    for group_key in sorted(surface.store.data.keys()):
        if str(group_key).split(":", 1)[0] != "whatsapp":
            continue
        if surface.is_channel(group_key):
            continue
        visible.append(group_key)
    return visible


def _pick(surface: Any, selector: str) -> str:
    """One WhatsApp group by name/key substring, or a refusal — the
    mirror of ``runtime_group._digest_pick`` (same words)."""
    from ...social.chat import whatsapp_digest as _wd

    needle = selector.lower()
    matches = []
    for group_key in _visible(surface):
        title = _wd._group_title(surface, group_key)
        if needle in title.lower() or needle in group_key.lower():
            matches.append(group_key)
    if not matches:
        return f"no WhatsApp group matching {selector!r}."
    if len(matches) > 1:
        names = ", ".join(
            _wd._group_title(surface, group_key)
            for group_key in matches)
        return (f"more than one group matches {selector!r}: "
                f"{names} — name one.")
    return matches[0]


def _build(surface: Any, log: Any, group_key: str) -> dict[str, Any] | None:
    """This group's digest dict, or None when the group is unknown —
    the mirror of ``runtime_group._digest_build``."""
    from ...social.chat import whatsapp_digest as _wd

    if group_key not in (surface.store.data or {}):
        return None
    return _wd.build_group_digest_text(surface, log, group_key)


def _reply(surface: Any, log: Any, group_key: str) -> str:
    """This group's digest as a reply, with honest no-digest lines —
    the mirror of ``runtime_group._digest_reply`` (same words)."""
    digest = _build(surface, log, group_key)
    if digest is None:
        return "I have no record of this group yet."
    if digest.get("included"):
        return str(digest.get("text") or "")
    if digest.get("policy") == "never":
        return ("digests are off for this group (policy never) "
                "— /digest policy only-when-mentioned to turn "
                "them back on.")
    return (f"nothing needs you in {digest.get('title') or group_key} "
            "in the last 24h.")


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def _cmd_digest(args: argparse.Namespace, context: Any) -> int:
    """``nm digest [name]`` — the /digest board from the console."""
    action = getattr(args, "digest_action", None)
    rest = list(getattr(args, "digest_rest", None) or [])
    as_json = bool(getattr(args, "json", False))
    words = ([str(action)] if action is not None else []) + [
        str(w) for w in rest]
    # Policy sets state — chat-only, refused in words.
    if any(w.lower() == "policy" for w in words):
        print("digest policy needs a live chat — run "
              "/digest <group> policy <value> from a chat, "
              "not the console")
        return 0
    try:
        surface = _surface(context)
        log = _log(context)
    except Exception as exc:  # noqa: BLE001 - surface/log path is the whole command
        return _usage_error(f"digest unavailable: {exc}")
    if not words:
        included = []
        for group_key in _visible(surface):
            digest = _build(surface, log, group_key)
            if digest is not None and digest.get("included"):
                included.append({"key": group_key, "digest": digest})
        if as_json:
            print(json.dumps({"groups": included}, indent=2, default=str))
            return 0
        if not included:
            print("no group needs you right now.")
            return 0
        print("\n\n".join(str(row["digest"].get("text") or "")
                          for row in included))
        return 0
    selector = " ".join(words).strip()
    group_key = _pick(surface, selector)
    if not (isinstance(group_key, str)
            and group_key.startswith("whatsapp:")):
        print(group_key)  # the pick's refusal, verbatim
        return 0
    if as_json:
        digest = _build(surface, log, group_key)
        print(json.dumps({"key": group_key, "digest": digest},
                         indent=2, default=str))
        return 0
    print(_reply(surface, log, group_key))
    return 0
