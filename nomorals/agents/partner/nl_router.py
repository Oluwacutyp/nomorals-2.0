"""Natural-language intent router: capability triggers without slash commands.

The owner shouldn't need to memorize /commands. When a plain-text message
unambiguously asks for a capability (research this, continue my story, mine
my chats, ask the wisdom corpus), route it to the real handler instead of
letting it die as small talk.

High precision only: patterns here must be unambiguous. When in doubt,
return None and let the normal conversation flow handle it.
"""

from __future__ import annotations

import re
from typing import Any

# (pattern, handler-name, topic-group-index)
_PATTERNS: list[tuple[re.Pattern[str], str, int | None]] = [
    # research — "research X", "look deeply into X", "investigate X"
    (re.compile(r"^(?:please\s+)?research\s+(.+)$", re.I), "research", 1),
    (re.compile(r"^(?:please\s+)?look\s+(?:deeply\s+)?into\s+(.+)$", re.I),
     "research", 1),
    (re.compile(r"^(?:please\s+)?investigate\s+(.+)$", re.I), "research", 1),
    (re.compile(r"^do\s+(?:some\s+)?research\s+(?:on|into|about)\s+(.+)$", re.I),
     "research", 1),
    # mining — "mine my conversations", "mine training data"
    (re.compile(r"^mine\s+(?:my\s+)?(?:conversations?|chats?|training\s+data)\s*$",
                re.I), "mine", None),
    # novel — "continue my story", "read chapter 5 of X"
    (re.compile(r"^continue\s+(?:my\s+)?(?:story|novel)\s*(.*)$", re.I),
     "novel_continue", 1),
]

# Wisdom: esoteric topics that clearly belong to the corpus, not general chat.
_WISDOM_KEYWORDS = frozenset({
    "chakra", "chakras", "kundalini", "astral", "akashic", "apocrypha",
    "gnostic", "hermetic", "hermes trismegistus", "enoch", "pistis sophia",
    "gospel of thomas", "dead sea scrolls", "nag hammadi", "kabbalah",
    "sufi", "vedas", "upanishads", "bhagavad gita", "tao te ching",
    "i ching", "book of the dead", "emanation", "demiurge", "aeon",
    "reincarnation", "past life", "third eye", "void state",
    "timeline shift", "quantum jump",
})

_WISDOM_QUESTION = re.compile(
    r"\b(when was|how was|who created|what created|why (?:was|did)|"
    r"what is the (?:true |real |hidden |secret )?(?:meaning|origin|nature|purpose) of)\b",
    re.I)


def match_nl_intent(text: str) -> tuple[str, str] | None:
    """Return (handler, topic) for a plain-text capability request, or None."""
    t = (text or "").strip()
    if not t or t.startswith("/"):
        return None
    # never hijack long conversational messages — intents are short asks
    if len(t) > 220:
        return None
    for pattern, handler, group in _PATTERNS:
        m = pattern.match(t)
        if m:
            topic = m.group(group).strip() if group else ""
            return handler, topic
    # wisdom: esoteric keyword + question shape
    low = t.lower()
    if any(k in low for k in _WISDOM_KEYWORDS):
        return "wisdom", t
    if _WISDOM_QUESTION.search(t) and any(
            w in low for w in ("earth", "world", "universe", "creation",
                               "god", "soul", "spirit", "humanity",
                               "bible", "scripture")):
        return "wisdom", t
    return None
