"""Old-school WhatsApp-bot track picking for /play.

When a query is vague (``/play lifestyle``), auto-downloading the top
guess is worse than asking: the wrong song wastes the user's data and
patience.  So vague queries get a numbered pick-list — tappable buttons
on Telegram and WhatsApp (a ListMessage row list on WhatsApp) — and
the chosen track is the one that downloads.  Replying with the number
still works as the fallback rung.

When a query is specific (``/play lifestyle ya man``), the top result
is downloaded directly, no list.

The :class:`PickCache` is short-lived (30-minute TTL) and in-memory:
stale picks simply vanish instead of downloading the wrong track later.
A short-lived tombstone remembers a slot that HAD a pick for 30 minutes
past expiry/consumption, so a late bare-number reply gets an honest
"that pick-list expired" instead of silence — a bare number in a chat
that never had a pick still flows on untouched.
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
    "PICK_TOMBSTONE_TTL",
    "PICK_LIMIT",
    "MATCH_RECALL_FLOOR",
    "MATCH_SCORE_FLOOR",
    "PickCache",
    "PickEntry",
    "is_specific",
    "match_confidence",
    "is_confident_match",
    "parse_title_artist",
    "search_query_for",
    "format_picklist",
    "format_duration",
]

#: How long a pick-list stays valid. 30 minutes: a human reads a list
#: slower than 5 minutes, and the list is single-use and consumed on
#: success, so a longer TTL cannot download the wrong track — it is
#: the same list the user was shown.
PICK_TTL = 1800.0
#: How long the "this slot HAD a pick" memory lasts past expiry or
#: consumption. A late bare-number reply inside this window gets the
#: honest expired answer; past it, the number is a stray number again.
PICK_TOMBSTONE_TTL = 1800.0
#: Max candidates shown.
PICK_LIMIT = 8

#: Words that carry no identifying weight for the specificity heuristic.
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "and", "or", "in", "on", "to", "for",
    "with", "by", "ft", "feat", "featuring", "vs", "x",
})

#: Marker embedded in the pick-list message so the adapters can
#: derive tappable buttons from the reply text (see
#: ``tgbot_buttons.buttons_for_text``).  It also marks a final text as
#: a LIVE pick-list offer (see ``runtime_media._body``): an ack-edit
#: carrying it derives the keyboard instead of clearing it.
PICK_MARKER_RE = re.compile(r"pick:([0-9a-f]{8})")


@dataclass
class PickEntry:
    """One stored pick-list: candidates + expiry + owning chat."""

    candidates: list[Any] = field(default_factory=list)
    chat_key: str = ""
    expires_at: float = 0.0
    #: The user the list was offered to ("" = chat-scoped, the
    #: legacy binding). With an owner set, the bare-number lookup
    #: for a DIFFERENT user in the same chat does not resolve —
    #: one person's card never serves another person's number
    #: (Living Chat P1 operation targeting, 2026-11-08).
    owner_key: str = ""

    def expired(self) -> bool:
        return time.time() >= self.expires_at


def _slot(chat_key: str, owner_key: str = "") -> str:
    """The (chat, owner) slot key for the latest-token index."""
    return f"{chat_key or ''}\x1f{owner_key or ''}"


class PickCache:
    """Short-lived in-memory pick-list store.  All classmethods, never raises."""

    _by_token: dict[str, PickEntry] = {}
    _by_chat: dict[str, str] = {}  # (chat, owner) slot -> latest token
    #: (chat, owner) slot -> time the slot's pick expired or was
    #: consumed. Lets a late bare-number reply tell "your list ran
    #: out" from "there never was a list". Pruned with the cache.
    _expired: dict[str, float] = {}

    @classmethod
    def _note_expired(cls, chat_key: str, owner_key: str = "") -> None:
        """Remember that this slot HAD a pick, now gone. Never raises."""
        try:
            if not chat_key:
                return
            cls._expired[_slot(chat_key, owner_key)] = time.time()
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def store(cls, candidates: list[Any], chat_key: str,
              ttl: float = PICK_TTL, *, owner_key: str = "") -> str:
        """Store candidates, return the short token.  Never raises."""
        try:
            token = secrets.token_hex(4)
            entry = PickEntry(candidates=list(candidates or []),
                              chat_key=chat_key or "",
                              expires_at=time.time() + ttl,
                              owner_key=owner_key or "")
            cls._by_token[token] = entry
            if chat_key:
                slot = _slot(chat_key, owner_key)
                cls._by_chat[slot] = token
                # A fresh list supersedes the expired memory: the
                # chat has a LIVE pick again, not a gone one.
                cls._expired.pop(slot, None)
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
                cls._note_expired(entry.chat_key, entry.owner_key)
                cls._by_token.pop(token, None)
                return None
            return entry
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def get_for_chat(cls, chat_key: str,
                     owner_key: str = "") -> tuple[str, PickEntry] | None:
        """Latest live pick-list for a chat.  Never raises.

        With ``owner_key`` given, a list owned by a DIFFERENT user
        does not resolve (the bare-number collision fix); a list
        with no owner (the legacy chat binding) resolves for any
        requester.
        """
        try:
            slot = _slot(chat_key, owner_key)
            token = cls._by_chat.get(slot)
            if not token:
                # Fall back across the chat's slots: the requester's
                # own slot first, then the chat-shared slot, then —
                # for the legacy no-owner lookup — the chat's most
                # recent list under any owner slot.
                prefix = f"{chat_key or ''}\x1f"
                best: tuple[str, PickEntry] | None = None
                for s, t in cls._by_chat.items():
                    if not s.startswith(prefix):
                        continue
                    e = cls.get(t)
                    if e is not None and (best is None
                                          or e.expires_at > best[1].expires_at):
                        best = (t, e)
                if best is None:
                    return None
                token, entry = best
                if owner_key and entry.owner_key \
                        and entry.owner_key != owner_key:
                    return None
                return token, entry
            entry = cls.get(token)
            if entry is None:
                cls._by_chat.pop(slot, None)
                return None
            if owner_key and entry.owner_key \
                    and entry.owner_key != owner_key:
                return None
            return token, entry
        except Exception:  # noqa: BLE001
            return None

    @classmethod
    def had_expired_pick(cls, chat_key: str,
                         owner_key: str = "") -> bool:
        """True when this (chat, owner) slot HAD a pick-list that is
        now gone (expired or consumed) inside the tombstone window —
        the difference between "your list ran out" and "there never
        was a list". A bare number with a live pick never reaches
        here (the live lookup runs first); a bare number with no
        history at all returns False so the turn flows on untouched.
        Owner scoping mirrors :meth:`get_for_chat`: another user's
        expired list never answers this user's number. Never raises."""
        try:
            now = time.time()
            slot = _slot(chat_key, owner_key)
            noted = cls._expired.get(slot)
            if noted is not None:
                if noted > now - PICK_TOMBSTONE_TTL:
                    return True
                cls._expired.pop(slot, None)
            # Fall back across the chat's slots, mirroring
            # get_for_chat: the most recent expired memory under
            # any owner slot, with the same owner guard.
            prefix = f"{chat_key or ''}\x1f"
            best_slot: str | None = None
            best_noted = 0.0
            for s, n in list(cls._expired.items()):
                if not s.startswith(prefix):
                    continue
                if n <= now - PICK_TOMBSTONE_TTL:
                    cls._expired.pop(s, None)
                    continue
                if n > best_noted:
                    best_noted, best_slot = n, s
            if best_slot is None:
                return False
            _chat, _, slot_owner = best_slot.partition("\x1f")
            return not (owner_key and slot_owner
                        and slot_owner != owner_key)
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def is_superseded(cls, token: str) -> bool:
        """True when a NEWER list has replaced ``token``'s list in
        its (chat, owner) slot — a tap on the old card must answer
        "replaced", not download from a list the chat has moved
        past.  Unknown/expired tokens are not superseded (they
        take the expired path).  Never raises."""
        try:
            entry = cls._by_token.get((token or "").strip().lower())
            if entry is None:
                return False
            latest = cls._by_chat.get(
                _slot(entry.chat_key, entry.owner_key))
            return bool(latest) and latest != (token or "").strip().lower()
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def consume(cls, token: str) -> PickEntry | None:
        """Fetch once and invalidate (a pick is single-use)."""
        entry = cls.get(token)
        if entry is not None:
            cls._by_token.pop((token or "").strip().lower(), None)
            if entry.chat_key:
                slot = _slot(entry.chat_key, entry.owner_key)
                if cls._by_chat.get(slot) == (token or "").strip().lower():
                    cls._by_chat.pop(slot, None)
            # Consumed, not vanished: a second number still gets
            # the honest expired answer inside the window.
            cls._note_expired(entry.chat_key, entry.owner_key)
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
                    slot = _slot(entry.chat_key, entry.owner_key)
                    if cls._by_chat.get(slot) == t:
                        cls._by_chat.pop(slot, None)
                    cls._note_expired(entry.chat_key, entry.owner_key)
            stale = [s for s, n in cls._expired.items()
                     if n <= now - PICK_TOMBSTONE_TTL]
            for s in stale:
                cls._expired.pop(s, None)
        except Exception:  # noqa: BLE001
            pass

    @classmethod
    def clear(cls) -> None:
        """Test helper: empty the cache."""
        cls._by_token.clear()
        cls._by_chat.clear()
        cls._expired.clear()


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


#: Auto-delivery confidence floors (owner delivery contract,
#: 2026-10-20).  A candidate is only auto-sent when it covers at
#: least this fraction of the requested title's significant words
#: AND its title F1 clears the score floor AND (below) the head-word
#: and artist gates pass.  Anything less goes to the pick-list —
#: never a silent substitution, never a hard refusal either.
MATCH_RECALL_FLOOR = 0.6
MATCH_SCORE_FLOOR = 0.5


def _match_tokens(text: str) -> list[str]:
    """Significant tokens of a title/artist string, in order.

    Same tokenisation the specificity heuristic uses (lowercase
    alnum words, >1 char, non-stopword) so the whole module shares
    one notion of what a word is.  Never raises.
    """
    try:
        return [w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
                if len(w) > 1 and w not in _STOPWORDS]
    except Exception:  # noqa: BLE001
        return []


def match_confidence(cand_title: str, cand_artist: str,
                     query_title: str, query_artist: str) -> float:
    """How well a candidate matches a parsed request, 0.0–1.0.

    Title F1 over significant tokens: recall = how much of the
    requested title the candidate covers (its artist counts toward
    coverage — uploaders put the artist in the title and the title
    in the channel name), precision = how much of the candidate's
    title is actually the requested title.  This is the same
    word-overlap notion the resolver's workspace scan thresholds
    local files with, applied to search candidates.  Never raises.
    """
    try:
        qt = set(_match_tokens(query_title))
        if not qt:
            return 0.0
        ct = set(_match_tokens(cand_title))
        hay = ct | set(_match_tokens(cand_artist))
        recall = len(qt & hay) / len(qt)
        precision = (len(qt & ct) / len(ct)) if ct else 0.0
        if recall <= 0.0 or precision <= 0.0:
            return 0.0
        return round(2.0 * recall * precision / (recall + precision), 3)
    except Exception:  # noqa: BLE001
        return 0.0


def is_confident_match(cand_title: str, cand_artist: str,
                       query_title: str, query_artist: str) -> bool:
    """May this candidate be auto-delivered for the request?

    All of these must hold:

    * the request's HEAD word is present — a namesake that only
      shares the parenthetical ("Ya Man Ain't Me" for "Lifestyle
      (YA MAN)") is a different song, however many words overlap;
    * title recall clears :data:`MATCH_RECALL_FLOOR`;
    * when the request named an artist, the candidate's
      title/artist covers at least half of it;
    * the F1 clears :data:`MATCH_SCORE_FLOOR`.

    Borderline fails CLOSED to the pick-list, never to a coin flip.
    Never raises.
    """
    try:
        qt = _match_tokens(query_title)
        if not qt:
            return False
        ct = set(_match_tokens(cand_title))
        hay = ct | set(_match_tokens(cand_artist))
        if qt[0] not in hay:
            return False
        qset = set(qt)
        if len(qset & hay) / len(qset) < MATCH_RECALL_FLOOR:
            return False
        qa = set(_match_tokens(query_artist))
        if qa and len(qa & hay) / len(qa) < 0.5:
            return False
        return match_confidence(
            cand_title, cand_artist, query_title,
            query_artist) >= MATCH_SCORE_FLOOR
    except Exception:  # noqa: BLE001
        return False


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


#: Pick-list source labels, by candidate kind.  The list must name the
#: source — the user is choosing WHERE the track comes from, and a
#: bare title hides whether they are about to get a full track or a
#: 30-second preview clip.
_KIND_LABELS = {
    "file": "Local",
    "netnaija": "NetNaija",
    "notjustok": "NotJustOk",
    "tooxclusive": "TooXclusive",
    "soundcloud": "SoundCloud",
    "bandcamp": "Bandcamp",
    "youtube": "YouTube",
    "youtube-music": "YouTube Music",
    "archiveorg": "Archive",
    "boomplay": "Boomplay",
    "musicbrainz": "MusicBrainz",
    "openverse": "Openverse",
    "wikimedia": "Wikimedia",
    "itunes": "iTunes",
    "deezer": "Deezer",
    "spotify": "Spotify",
    "audiomack": "Audiomack",
    "url": "Link",
}


def format_picklist(candidates: list[Any], query: str, token: str) -> str:
    """Compact numbered pick-list.  Never raises.

    Every entry names its source, and 30s-preview entries say so —
    picking a preview blind is a wasted download and a confused user.
    """
    try:
        lines = [f"🎵 “{query}” — pick one (reply with the number):"]
        for i, c in enumerate(candidates[:PICK_LIMIT], 1):
            title = (getattr(c, "title", "") or "untitled").strip()
            artist = (getattr(c, "artist", "") or "").strip()
            dur = format_duration(getattr(c, "duration", 0))
            who = f" — {artist}" if artist else ""
            when = f" ({dur})" if dur and dur != "0:00" else ""
            kind = str(getattr(c, "kind", "") or "").lower()
            label = _KIND_LABELS.get(kind, "") or kind
            source_name = str(getattr(c, "source_name", "") or "")
            if kind in ("itunes", "deezer") \
                    or "preview" in source_name.lower():
                tag = f" [{label} · 30s preview]" if label \
                    else " [30s preview]"
            elif getattr(c, "downloadable", True) is False:
                tag = f" [{label} · metadata only]" if label \
                    else " [metadata only]"
            elif label:
                tag = f" [{label}]"
            else:
                tag = ""
            lines.append(f"{i}. {title}{who}{when}{tag}")
        lines.append("tap a button below 👇")
        if token:
            lines.append(f"pick:{token}")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return f"🎵 “{query}” — pick one by number."
