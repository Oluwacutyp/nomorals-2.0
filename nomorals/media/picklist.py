"""Old-school WhatsApp-bot track picking for /play.

When a query is vague (``/play lifestyle``), auto-downloading the top
guess is worse than asking: the wrong song wastes the user's data and
patience.  So vague queries get a numbered pick-list — tappable buttons
on Telegram, "reply with the number" on WhatsApp — and the chosen track
is the one that downloads.

When a query is specific (``/play lifestyle ya man``), the top result
is downloaded directly, no list.

The :class:`PickCache` is short-lived (5-minute TTL) and in-memory:
stale picks simply vanish instead of downloading the wrong track later.
Never raises.
"""

from __future__ import annotations

import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "PICK_TTL",
    "PICK_LIMIT",
    "PickCache",
    "PickEntry",
    "is_specific",
    "parse_title_artist",
    "search_query_for",
    "format_picklist",
    "format_duration",
]

#: How long a pick-list stays valid.
PICK_TTL = 300.0
#: Max candidates shown.
PICK_LIMIT = 8

#: Words that carry no identifying weight for the specificity heuristic.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for",
    "with", "by", "ft", "feat", "featuring", "vs", "x",
})

#: Marker embedded in the pick-list message so the Telegram adapter can
#: derive tappable buttons from the reply text (see
#: ``tgbot_buttons.buttons_for_text``).
PICK_MARKER_RE = re.compile(r"pick:([0-9a-f]{8})")


@dataclass
class PickEntry:
    """One stored pick-list: candidates + expiry + owning chat."""

    candidates: list[Any] = field(default_factory=list)
    chat_key: str = ""
    expires_at: float = 0.0

    def expired(self) -> bool:
        return time.time() >= self.expires_at


class PickCache:
    """Short-lived in-memory pick-list store.  All classmethods, never raises."""

    _by_token: dict[str, PickEntry] = {}
    _by_chat: dict[str, str] = {}  # chat_key -> latest token

    @classmethod
    def store(cls, candidates: list[Any], chat_key: str,
              ttl: float = PICK_TTL) -> str:
        """Store candidates, return the short token.  Never raises."""
        try:
            token = secrets.token_hex(4)
            entry = PickEntry(candidates=list(candidates or []),
                              chat_key=chat_key or "",
                              expires_at=time.time() + ttl)
            cls._by_token[token] = entry
            if chat_key:
                cls._by_chat[chat_key] = token
            cls.prune()
            return token
        except Exception:  # noqa: BLE001 — cache must never break chat
            _log.warning("PickCache.store failed", exc_info=True)
            return ""

    @classmethod
    def get(cls, token: str) -> PickEntry | None:
        """Fetch by token; expired/missing → None.  Never raises."""
        try:
            entry = cls._by_token.get((token or "").strip().lower())
            if entry is None:
                return None
            if entry.expired():
                cls._by_token.pop(token, None)
                return None
            return entry
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def get_for_chat(cls, chat_key: str) -> tuple[str, PickEntry] | None:
        """Latest live pick-list for a chat.  Never raises."""
        try:
            token = cls._by_chat.get(chat_key or "")
            if not token:
                return None
            entry = cls.get(token)
            if entry is None:
                cls._by_chat.pop(chat_key, None)
                return None
            return token, entry
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def consume(cls, token: str) -> PickEntry | None:
        """Fetch once and invalidate (a pick is single-use)."""
        entry = cls.get(token)
        if entry is not None:
            cls._by_token.pop((token or "").strip().lower(), None)
            if entry.chat_key:
                cls._by_chat.pop(entry.chat_key, None)
        return entry

    @classmethod
    def prune(cls) -> None:
        """Drop expired entries.  Never raises."""
        try:
            now = time.time()
            dead = [t for t, e in cls._by_token.items() if e.expires_at <= now]
            for t in dead:
                entry = cls._by_token.pop(t, None)
                if entry and entry.chat_key:
                    if cls._by_chat.get(entry.chat_key) == t:
                        cls._by_chat.pop(entry.chat_key, None)
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def clear(cls) -> None:
        """Test helper: empty the cache."""
        cls._by_token.clear()
        cls._by_chat.clear()


def is_specific(query: str) -> bool:
    """Is this query specific enough to auto-download the top result?

    Specific: "artist - title" separator, "title by artist", quoted
    phrase, or 2+ significant words (``lifestyle ya man``).  Vague: a
    single word (``lifestyle``) — show the pick-list instead.  When in
    doubt the list wins: downloading the wrong song is worse than one
    extra tap.  Never raises.
    """
    try:
        q = (query or "").strip()
        if not q:
            return False
        if " - " in q:
            return True
        if re.search(r"\bby\b", q, re.IGNORECASE):
            return True
        if len(q) >= 2 and q[0] in "\"'“”" and q[-1] in "\"'“”":
            return True
        words = [w for w in re.findall(r"[a-z0-9]+", q.lower())
                 if len(w) > 1 and w not in _STOPWORDS]
        return len(words) >= 2
    except Exception:  # noqa: BLE001
        return False


def parse_title_artist(query: str) -> tuple[str, str]:
    """Split a query into (title, artist).  Never raises.

    - ``"lifestyle (YA MAN) by ayo maff"`` → (``"lifestyle (YA MAN)"``,
      ``"ayo maff"``)
    - ``"burna boy - last last"`` → (``"last last"``, ``"burna boy"``)
      (standard "Artist - Title" convention)
    - no separator → (``query``, ``""``); the caller still searches the
      full string, the specificity flag is what matters.
    """
    try:
        q = (query or "").strip()
        if not q:
            return "", ""
        # "title by artist" — word-boundary, case-insensitive
        parts = re.split(r"\bby\b", q, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            title, artist = parts[0].strip(), parts[1].strip()
            if title and artist:
                return title, artist
        # "artist - title"
        if " - " in q:
            left, _, right = q.partition(" - ")
            if left.strip() and right.strip():
                return right.strip(), left.strip()
        return q, ""
    except Exception:  # noqa: BLE001
        q = (query or "").strip()
        return q, ""


def search_query_for(title: str, artist: str) -> str:
    """Search string with artist-first ordering (better SoundCloud/YouTube
    results).  Never raises."""
    try:
        title = (title or "").strip()
        artist = (artist or "").strip()
        if artist and title:
            return f"{artist} {title}"
        return title or artist
    except Exception:  # noqa: BLE001
        return (title or artist or "").strip()


def format_duration(seconds: float) -> str:
    """Seconds → "3:24".  Never raises."""
    try:
        total = max(0, int(seconds or 0))
        return f"{total // 60}:{total % 60:02d}"
    except Exception:  # noqa: BLE001
        return ""


def format_picklist(candidates: list[Any], query: str, token: str) -> str:
    """Compact numbered pick-list.  Never raises."""
    try:
        lines = [f"🎵 “{query}” — pick one (reply with the number):"]
        for i, c in enumerate(candidates[:PICK_LIMIT], 1):
            title = (getattr(c, "title", "") or "untitled").strip()
            artist = (getattr(c, "artist", "") or "").strip()
            dur = format_duration(getattr(c, "duration", 0))
            who = f" — {artist}" if artist else ""
            when = f" ({dur})" if dur and dur != "0:00" else ""
            lines.append(f"{i}. {title}{who}{when}")
        lines.append("tap a number below 👇")
        if token:
            lines.append(f"pick:{token}")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"🎵 “{query}” — pick one by number."
