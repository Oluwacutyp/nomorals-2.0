"""Group-health tool: ``group_health`` — the brain's answer to
"how is <group> doing?", "is <group> growing?" and "what is
<group> talking about?"

One tool, action-dispatched, over
:mod:`nomorals.social.chat.group_health`; the organ owns the
numbers — churn rate, growth rate, trend over two windows, topic
clusters from the recorded messages — this module is only the
surface.

``action=report`` answers for one group (``group=<name or key>``,
``days=<window, default 7>``): the full /ghealth report — growth
and topics included. ``action=overview`` boards every WhatsApp
group. ``action=topics`` answers just the topics block for one
group. An unknown group is answered honestly as nothing on
record, never invented. A group with no recorded messages gets
the honest "no message record" line, never a faked cluster.

The phrasing parser :func:`parse_group_health_question` is shared
with the chat NL hook, so the question and the tool agree on what
counts as the question. The parser takes the groups Devon
actually serves (titles/keys) and only matches a name it can see
there — a question about anything else ("the weather", "the
market", "you") is never stolen.
"""

from __future__ import annotations

import re
import time
from typing import Any

from ..core.policy import Capability

__all__ = [
    "register",
    "parse_group_health_question",
]

_GENERIC_NAMES = ("the group", "this group", "the room", "this room")

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # "how is Lagos Life doing?" / "how's the group doing?"
    (re.compile(r"^how(?:'s| is) (.+?) doing[?.!]*$", re.IGNORECASE),
     "report"),
    # "is Lagos Life growing?" / "is the group shrinking?"
    (re.compile(r"^is (.+?) (?:growing|shrinking|churning)[?.!]*$",
                re.IGNORECASE),
     "growth"),
    # "what is Lagos Life talking about?" / "what's the group talking about?"
    (re.compile(r"^what(?:'s| is) (.+?) talking about[?.!]*$",
                re.IGNORECASE),
     "topics"),
)


def parse_group_health_question(
    text: str, known: Any = (),
) -> tuple[str, str] | None:
    """Extract (intent, group name) from a group-health question.

    ``known`` is the groups Devon serves (titles and/or keys). A
    name only matches when it names one of them (substring, either
    direction) — anything else is None, so the hook never steals
    an unrelated question. A generic "the group" / "this group"
    matches with an empty name, meaning "this group" in a group
    chat and the board in DM. None when the text is not that
    question at all."""
    raw = (text or "").strip()
    if not raw:
        return None
    for pattern, intent in _PATTERNS:
        match = pattern.match(raw)
        if match is None:
            continue
        name = match.group(1).strip().rstrip("?.!").strip()
        if not name:
            return None
        if name.lower() in _GENERIC_NAMES:
            return (intent, "")
        needle = name.lower()
        for entry in known or ():
            known_text = str(entry or "").lower()
            if not known_text:
                continue
            if needle in known_text or known_text in needle:
                return (intent, name)
        return None
    return None


def _find_groups(surface: Any, needle: str) -> list[str]:
    """Group keys whose title or key contains ``needle``."""
    needle = (needle or "").strip().lower()
    if not needle:
        return []
    matches = []
    for key in sorted(getattr(surface.store, "data", None) or {}):
        entry = surface.store.group(key)
        title = str(entry.get("title") or entry.get("name") or "")
        if needle in title.lower() or needle in str(key).lower():
            matches.append(str(key))
    return matches


def _surface(context: Any) -> Any:
    from ..social.chat.group_surface import default_surface

    settings = getattr(context, "settings", None)
    ws_dir = getattr(settings, "workspace_dir", ".") or "."
    return default_surface(ws_dir)


def _messages(context: Any, group_key: str, *,
              now: float, window_days: int) -> list[dict[str, Any]] | None:
    """The group's recorded messages for the window, or None when
    the brain log is unavailable."""
    from ..social.chat import group_health as _gh
    from ..storage.conversations import ConversationLog

    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        return _gh.read_group_messages(
            ConversationLog(db), group_key,
            since=now - window_days * 86400.0)
    except Exception:  # noqa: BLE001 - no log, no topics
        return None


def register(registry: Any) -> None:
    """Attach the ``group_health`` tool."""
    context = registry.context

    @registry.register(
        "group_health",
        description=(
            "WhatsApp group health: churn rate, growth rate and trend "
            "over two windows, and what the group actually talks about "
            "(topic clusters from the recorded messages). "
            "action=report answers one group (group=<name or key>, "
            "days=<window, default 7>) with the full report. "
            "action=overview boards every WhatsApp group. "
            "action=topics answers just the topics block for one group. "
            "An unknown group is answered honestly as nothing on "
            "record, never invented; a group with no recorded messages "
            "says so, never a faked cluster."
        ),
        capability=Capability.MEM_READ,
    )
    def group_health_tool(
        action: str = "report",
        group: str = "",
        days: str = "",
    ) -> dict[str, Any]:
        from ..social.chat import group_health as _gh

        try:
            window_days = int(days) if str(days or "").strip() else 7
        except (TypeError, ValueError):
            window_days = 7
        if window_days <= 0:
            window_days = 7
        action = str(action or "report").strip().lower()
        if action not in ("report", "overview", "topics"):
            return {"ok": False,
                    "error": f"unknown action {action!r}"}
        surface = _surface(context)
        now = time.time()
        if action == "overview":
            overview = _gh.build_overview(surface, now=now,
                                          window_days=window_days)
            return {"ok": True,
                    "text": _gh.render_overview(overview),
                    "overview": overview}
        matches = _find_groups(surface, group)
        if not matches:
            return {"ok": True,
                    "text": (f"no record of a WhatsApp group matching "
                             f"{group!r}.")}
        if len(matches) > 1:
            names = ", ".join(
                str(surface.store.group(key).get("title") or key)
                for key in matches)
            return {"ok": True,
                    "text": (f"more than one group matches {group!r}: "
                             f"{names} — name one.")}
        key = matches[0]
        messages = _messages(context, key, now=now,
                             window_days=window_days)
        health = _gh.build_group_health(
            surface, key, now=now, window_days=window_days,
            messages=messages)
        if health is None:
            return {"ok": True,
                    "text": f"no record of a WhatsApp group matching "
                            f"{group!r}."}
        if action == "topics":
            title = health.get("title") or key
            return {"ok": True,
                    "text": (f"Talking about — {title}\n"
                             + _gh.render_topics(health.get("topics"))),
                    "topics": health.get("topics")}
        return {"ok": True,
                "text": _gh.render_group_health(health),
                "health": health}
