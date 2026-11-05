"""Chat targeting by human reference (register #30).

Owner-facing commands that act on a chat (``/chatmode`` first)
used to demand the full raw chat key — ``whatsapp:<jid>`` — as
their target.  Nobody should have to paste a JID.  A target may
now be given the way a human names a chat:

1. nothing / ``this`` / ``here`` — the chat the command was sent in;
2. a full chat key (``platform:chat_id``) — the original form,
   still accepted verbatim;
3. a phone number (``+234 801 234 5678`` or bare digits) — that
   person's WhatsApp DM.  From a Telegram-family chat a bare
   numeric id is that chat's platform id instead (a number means
   different things on different networks; the reply always names
   the key it resolved to, so a surprise is visible, never silent);
4. a chat or group NAME — resolved against every name source
   Devon has: the title index (every chat title she has seen,
   recorded from live traffic), the current chat's own title, and
   any adapter that can list its chats/groups (WhatsApp's bridge
   lists both).  An exact title match wins; otherwise a unique
   substring match resolves.  Several matches are REPORTED as
   candidates — never guessed between.

Resolution is honest on exhaustion: when nothing matches, the
result names every form that was tried, so the caller can answer
in one plain line instead of a generic usage scold.

The title index is a hint store, not an authority: a chat renamed
since its last seen title resolves by its new name as soon as a
message from it is seen again (the runtime records titles on
intake, only when they change).
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "ChatTargetResolution",
    "note_chat_title",
    "chat_title_index",
    "resolve_chat_target",
]

#: Platform tokens a full chat key may start with. A "name" that
#: merely contains a colon (a clock time, a ratio) must not be
#: mistaken for a key — only these prefixes take the key path.
_KEY_PLATFORMS = frozenset({
    "telegram", "telegram-bot", "telegram-personal", "whatsapp",
    "discord", "local", "web", "webhook", "sms",
})

_PHONE_RE = re.compile(r"^\+?[\d][\d\s\-()]{5,}$")
_TELEGRAM_ID_RE = re.compile(r"^-?\d{5,}$")


@dataclass
class ChatTargetResolution:
    """The outcome of resolving one human target reference."""

    #: The resolved chat key, or "" when unresolved.
    key: str = ""
    #: The matched chat's title, when the name path found it.
    title: str = ""
    #: (key, title) pairs when the name matched several chats.
    candidates: list[tuple[str, str]] = field(default_factory=list)
    #: Every form/path tried, in order — the exhaustion report.
    tried: list[str] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return bool(self.key)

    @property
    def ambiguous(self) -> bool:
        return not self.key and len(self.candidates) > 1

    def describe(self, raw: str) -> str:
        """One plain line for the unresolved/ambiguous cases."""
        if self.ambiguous:
            lines = [f"{raw!r} matches more than one chat — "
                     "which one did you mean?"]
            for key, title in self.candidates[:8]:
                lines.append(f"  {title} — {key}" if title else f"  {key}")
            lines.append("use the full key to pick one.")
            return "\n".join(lines)
        tried = "; ".join(self.tried) if self.tried else "nothing"
        return (f"no chat matches {raw!r} — I tried: {tried}. "
                "A full key (platform:chat_id), a phone number, or "
                "the chat's name all work.")


# ── the title index ────────────────────────────────────────────────────────

def _ensure_index(db: Any) -> None:
    db.execute(
        """CREATE TABLE IF NOT EXISTS chat_title_index (
            chat_key TEXT PRIMARY KEY,
            title TEXT NOT NULL DEFAULT '',
            updated_at REAL NOT NULL DEFAULT 0
        )"""
    )


def note_chat_title(db: Any, chat: Any) -> None:
    """Record one chat's title from live traffic.

    Called on message intake with the message's ChatRef. Writes
    only when the title is non-empty AND differs from the stored
    one (one indexed read per titled message, a write only on
    change). Best-effort: never raises — a title hint must not be
    able to break intake.
    """
    if db is None or chat is None:
        return
    title = str(getattr(chat, "title", "") or "").strip()
    key = str(getattr(chat, "key", "") or "").strip()
    if not title or not key:
        return
    try:
        _ensure_index(db)
        row = db.query_one(
            "SELECT title FROM chat_title_index WHERE chat_key = ?",
            (key,))
        if row is not None and str(row.get("title") or "") == title:
            return
        db.execute(
            "INSERT OR REPLACE INTO chat_title_index "
            "(chat_key, title, updated_at) VALUES (?, ?, ?)",
            (key, title, time.time()))
    except Exception:  # noqa: BLE001 - a hint store never breaks intake
        pass


def chat_title_index(db: Any) -> list[tuple[str, str]]:
    """Every (chat_key, title) pair the index holds. Never raises."""
    if db is None:
        return []
    try:
        _ensure_index(db)
        rows = db.query(
            "SELECT chat_key, title FROM chat_title_index") or []
        return [(str(r["chat_key"]), str(r["title"]))
                for r in rows if str(r["title"] or "").strip()]
    except Exception:  # noqa: BLE001
        return []


# ── resolution ─────────────────────────────────────────────────────────────

def _adapter_chat_rows(adapter: Any, platform: str) -> list[tuple[str, str]]:
    """(chat_key, title) rows from an adapter that can list chats.

    Duck-typed on purpose: any adapter exposing ``chats()`` (recent
    chats with titles) or ``groups()`` (groups with subjects)
    contributes its names — WhatsApp's bridge backs both today.
    Failures contribute nothing (the caller reports which sources
    were actually consulted via the tried list only when the
    adapter answered).
    """
    rows: list[tuple[str, str]] = []
    chats = getattr(adapter, "chats", None)
    if callable(chats):
        try:
            for entry in chats() or []:
                cid = str(entry.get("id") or "").strip()
                title = str(entry.get("title") or "").strip()
                if cid:
                    rows.append((f"{platform}:{cid}", title))
        except Exception:  # noqa: BLE001 - a listing is a hint source
            pass
    groups = getattr(adapter, "groups", None)
    if callable(groups):
        try:
            for entry in groups() or []:
                cid = str(entry.get("id") or "").strip()
                title = str(entry.get("subject")
                            or entry.get("title") or "").strip()
                if cid:
                    rows.append((f"{platform}:{cid}", title))
        except Exception:  # noqa: BLE001
            pass
    return rows


def resolve_chat_target(
    raw: str,
    *,
    current_key: str,
    current_title: str = "",
    current_platform: str = "",
    db: Any = None,
    adapters: Iterable[Any] = (),
) -> ChatTargetResolution:
    """Resolve one human chat reference to a chat key.

    See the module docstring for the accepted forms and the
    honesty rules. ``adapters`` are duck-typed chat adapters (for
    name sources); ``db`` backs the title index.
    """
    res = ChatTargetResolution()
    text = (raw or "").strip()
    if not text or text.lower() in ("this", "here", "self"):
        res.key = current_key
        res.title = current_title
        return res

    # 2. Full key form — verbatim, exactly as before.
    platform, sep, chat_id = text.partition(":")
    if sep and chat_id.strip() \
            and platform.strip().lower() in _KEY_PLATFORMS:
        res.tried.append("full chat key")
        res.key = f"{platform.strip().lower()}:{chat_id.strip()}"
        return res

    # 3. Numbers: a phone number names a WhatsApp DM; from a
    # Telegram-family chat a bare numeric id names a Telegram chat.
    compact = re.sub(r"[\s\-()]", "", text)
    if _PHONE_RE.match(text) and compact.lstrip("+").isdigit() \
            and 7 <= len(compact.lstrip("+")) <= 15:
        if (current_platform or "").lower().startswith("telegram") \
                and _TELEGRAM_ID_RE.match(compact) \
                and not compact.startswith("+"):
            res.tried.append("numeric id → Telegram chat")
            res.key = f"telegram:{compact}"
            return res
        res.tried.append("phone number → WhatsApp DM")
        res.key = f"whatsapp:{compact.lstrip('+')}@s.whatsapp.net"
        return res
    if _TELEGRAM_ID_RE.match(compact) \
            and (current_platform or "").lower().startswith("telegram"):
        res.tried.append("numeric id → Telegram chat")
        res.key = f"telegram:{compact}"
        return res

    # 4. Name — every source, exact first, then unique substring.
    res.tried.append("chat name")
    sources: list[tuple[str, str]] = []
    if current_title:
        sources.append((current_key, current_title))
    sources.extend(chat_title_index(db))
    for adapter in adapters or ():
        plat = str(getattr(adapter, "name", "") or "").strip()
        if plat:
            sources.extend(_adapter_chat_rows(adapter, plat))
    seen: dict[str, str] = {}
    for key, title in sources:
        if title and key not in seen:
            seen[key] = title
    want = text.lower()
    exact = [(k, t) for k, t in seen.items() if t.lower() == want]
    if len(exact) == 1:
        res.key, res.title = exact[0]
        return res
    if len(exact) > 1:
        res.candidates = exact
        return res
    partial = [(k, t) for k, t in seen.items() if want in t.lower()]
    if len(partial) == 1:
        res.key, res.title = partial[0]
        return res
    if len(partial) > 1:
        res.candidates = partial
        return res
    return res
