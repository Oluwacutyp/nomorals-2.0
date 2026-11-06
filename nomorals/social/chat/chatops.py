"""Chat & privacy command set — pure helpers (FEI gap wave F).

The transport-free half of Devon's chat & privacy commands:
the kind set itself, the disappearing-timer vocabulary, the
tagall chunking law, the poll-results and group-status
formatters, and the mass-tag test.  Everything here is a pure
function of its inputs so the command layer
(``agents/partner/runtime_chatops.py``) and its tests exercise
the shapes without a bridge, a client, or a network.

The platform truth these helpers encode:

* disappearing timers speak WhatsApp's preset vocabulary
  (off / 24h / 7d / 90d) plus raw seconds — Telegram's TTL and
  the Go bridge accept any duration, so one parser serves
  every surface and each adapter refuses what it cannot do;
* a poll's results are whatever the message itself carries
  (Telegram's poll media, re-read by the personal adapter) —
  the formatter prints only counted votes, never estimates;
* a group's "status" is composed ONLY from data a surface
  enumerated (member/admin lists, counts, group info) and
  every line is labeled with where it came from, because no
  platform offers a group-stats endpoint to any of Devon's
  surfaces (the Bot API least of all);
* tagall is capped and chunked: a giant group gets the cap
  NAMED in the reply, never a silent partial tag and never
  an unbounded mention storm.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CHATOPS_COMMANDS",
    "TAG_CHUNK_SIZE",
    "TAG_CAP",
    "MASS_TAG_MIN_MENTIONS",
    "parse_disappear_duration",
    "chunk_mentions",
    "is_mass_tag",
    "format_poll_results",
    "compose_group_status",
    "kind_for_chat_key",
]

#: The wave's kinds — one set, spelled once here; the runtime
#: dispatch, the registry tables and the tests all read it.
CHATOPS_COMMANDS = frozenset({
    "archive", "unarchive", "mutechat", "unmutechat",
    "readchat", "unreadchat", "star", "unstar",
    "unsend", "edit", "editmsg", "antiedit", "poststatus",
    "block", "unblock", "anticall", "disappear",
    "statusreact", "banchecker", "gcstatus", "statusinfo",
    "pollres", "tagall", "hidetag", "antitag",
})

#: Tagall pacing law: one message carries at most this many
#: mentions, and one command tags at most this many members —
#: the reply names the cap when a group is bigger.
TAG_CHUNK_SIZE = 50
TAG_CAP = 100

#: A message mentioning at least this many people counts as a
#: mass tag for the antitag sensor.
MASS_TAG_MIN_MENTIONS = 5

#: WhatsApp's disappearing-message presets, in seconds (the
#: only durations WhatsApp itself offers; 0 = off).
_DISAPPEAR_PRESETS = {
    "off": 0,
    "0": 0,
    "24h": 24 * 3600,
    "1d": 24 * 3600,
    "7d": 7 * 24 * 3600,
    "90d": 90 * 24 * 3600,
}


def parse_disappear_duration(raw: str) -> int | None:
    """Seconds for one /disappear argument, or None when the
    argument is not a duration this command speaks.

    Accepts the preset vocabulary (off, 24h, 7d, 90d) and a
    bare number of seconds ("3600").  Anything else is None —
    the caller answers with the vocabulary, never a guess.
    """
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if text in _DISAPPEAR_PRESETS:
        return _DISAPPEAR_PRESETS[text]
    if text.isdigit():
        return int(text)
    return None


def chunk_mentions(members: list[Any], *,
                   size: int = TAG_CHUNK_SIZE,
                   cap: int = TAG_CAP
                   ) -> tuple[list[list[Any]], int, bool]:
    """Split a member list into sendable mention chunks.

    Returns (chunks, tagged_count, capped): at most ``cap``
    members are tagged, in chunks of at most ``size``; when
    the group is bigger than the cap, ``capped`` is True and
    the caller names the cap in its reply.
    """
    members = list(members or [])
    capped = len(members) > cap
    chosen = members[:cap]
    chunks = [chosen[i:i + size]
              for i in range(0, len(chosen), size)]
    return chunks, len(chosen), capped


def is_mass_tag(mention_count: int, *,
                threshold: int = MASS_TAG_MIN_MENTIONS) -> bool:
    """True when a message's mention payload is a mass tag."""
    try:
        return int(mention_count) >= int(threshold)
    except (TypeError, ValueError):
        return False


def format_poll_results(verdict: dict[str, Any]) -> str:
    """Render one poll-results verdict (the personal Telegram
    adapter's channel_poll_results shape) as plain lines.

    Only counted votes print: an option whose voter count the
    platform did not expose shows "not exposed", never a
    fabricated zero; the total prints only when present.
    """
    options = list(verdict.get("options") or [])
    lines: list[str] = []
    question = str(verdict.get("question") or "").strip()
    if question:
        lines.append(f"Poll: {question}")
    total = verdict.get("total_voters")
    for row in options:
        text = str(row.get("text") or "option")
        voters = row.get("voters")
        if voters is None:
            lines.append(f"• {text} — votes not exposed")
            continue
        voters = int(voters)
        share = ""
        if total:
            share = f" ({round(100.0 * voters / float(total))}%)"
        mark = " ✓" if row.get("correct") else ""
        lines.append(f"• {text} — {voters} vote"
                     f"{'' if voters == 1 else 's'}{share}{mark}")
    if total is not None:
        lines.append(f"Total: {int(total)} voter"
                     f"{'' if int(total) == 1 else 's'}")
    if not options:
        lines.append("The poll carries no options to read.")
    return "\n".join(lines)


def compose_group_status(*, title: str, platform: str,
                         kind: str,
                         member_count: int | None,
                         admins: list[str],
                         extra: list[tuple[str, str]],
                         source: str) -> str:
    """Compose /gcstatus output from ENUMERATED data only.

    ``member_count`` is None when no surface counted the
    group — the line then says "not counted on this surface"
    rather than inventing a number.  ``source`` names the
    surface the data came from and always prints: these
    numbers are a composition from member/admin data, not a
    platform stats endpoint (none exists on any surface).
    """
    lines = [f"{title or 'This group'} — {platform} {kind}"]
    if member_count is None:
        lines.append("Members: not counted on this surface")
    else:
        lines.append(f"Members: {int(member_count)}")
    if admins:
        shown = ", ".join(admins[:10])
        more = (f" (+{len(admins) - 10} more)"
                if len(admins) > 10 else "")
        lines.append(f"Admins ({len(admins)}): {shown}{more}")
    else:
        lines.append("Admins: not enumerated on this surface")
    for label, value in extra:
        if value:
            lines.append(f"{label}: {value}")
    lines.append(f"(composed from {source} — no platform "
                 "offers a group-stats endpoint)")
    return "\n".join(lines)


def kind_for_chat_key(chat_key: str) -> str:
    """ChatRef kind for one resolved chat key's id part.

    WhatsApp: ``@g.us`` is a group, ``@newsletter`` a channel,
    anything else a DM.  Telegram: ``-100…`` ids are
    supergroups/channels (reported "group" — the chat ops
    treat them alike), other negatives are basic groups,
    positives are DMs.
    """
    key = str(chat_key or "")
    platform, _, chat_id = key.partition(":")
    if platform == "whatsapp":
        if chat_id.endswith("@g.us"):
            return "group"
        if chat_id.endswith("@newsletter"):
            return "channel"
        return "dm"
    if platform.startswith("telegram"):
        if chat_id.startswith("-"):
            return "group"
        return "dm"
    return "dm"
