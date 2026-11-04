"""The media expression organ (DESIGN §21 + §22): how Devon
EXPRESSES herself — emoji-led buttons everywhere, GIF/sticker
moments, her own library, reactions, and the full emoji space.

The owner's signal (2026-11-03 12:11, watching live moderation
in his own group): "Good, but the inline button emojis and gifs
still missing."  The mechanics all worked; the expression layer
was the visible gap.  This organ is that layer, in one place:

- **Buttons** (§27's map, completed): one per-action emoji
  vocabulary (:data:`BUTTON_EMOJI`) that every surface's buttons
  are labelled through — notices, console, games, economy — with
  per-group overrides riding the presentation settings, in the
  same spirit as the theme system.  No surface hand-rolls its
  own button emoji after this.
- **Moments**: welcome / quest completion / jackpot / season
  podium / duel wins / estate milestones / a ban landing get a
  GIF or sticker from her library; a warning stays dry, by
  design.  Selection is by mood/tag with no immediate repeats
  per room and a per-room cooldown, so she reads as a member of
  the room, not a printer attached to it.
- **The library**: stolen + made + provider-fetched media,
  indexed by tag/mood with provenance, persisted beside the
  group store with the S3 atomic + quarantine pattern.  On
  Telegram the first upload's file_id is cached per item and
  re-sent by id afterwards — no re-uploads.
- **Providers**: GIF search APIs (Tenor / Giphy, key-configured)
  are SEED sources behind a health breaker mirroring
  ``agency/sourcehealth`` semantics (stand down after 3 straight
  failures, disable after 6) — the open-sources law: the list
  grows via settings, dead sources shed, the local library is
  the always-available floor.  No key and an empty library means
  the moment ships text-only: expression NEVER blocks and never
  fakes the underlying action.
- **Reactions** (§22): a per-platform capability truth table +
  the moment vocabulary for placing them; the adapters carry
  the actual verbs.
- **The full emoji space**: a table built from the Unicode
  standard itself (every named codepoint in the emoji blocks,
  skin-tone variants, flags, keycaps, composed ZWJ sequences) —
  coverage defined by Unicode, never by a phone's font — with
  name/keyword search behind /emoji.
- **Her own custom emoji set**: a named registry (unicode
  fallback + per-platform refs + sticker art), managed through
  /customemoji.

Everything here is transport-free: the runtime injects the
send callable, exactly like the group surface.  A moment that
cannot send is a journaled non-event at the call site, never a
claimed send.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from ...core.logging_setup import get_logger

_log = get_logger("nomorals.social.expression")

__all__ = [
    "BUTTON_EMOJI", "button_label", "buttons_from_spec",
    "EXPRESSION_DEFAULTS",
    "EmojiEntry", "emoji_table", "search_emoji", "emoji_info",
    "describe_sequence",
    "CustomEmojiRegistry",
    "ExpressionLibrary",
    "ProviderHealth", "TenorProvider", "GiphyProvider",
    "default_providers", "download_to_library",
    "MOMENT_TAGS", "DRY_MOMENTS", "ExpressionService",
    "REACTION_SETS", "reaction_allowed", "pick_reaction",
    "classify_notice_moment",
]


# ── the button vocabulary (§27 completed) ────────────────────────

#: One emoji per button ACTION — the single map every surface
#: inherits.  Handlers name the action; the map owns the glyph,
#: so a member learns the map once (the §27 law for titles,
#: extended to buttons).
BUTTON_EMOJI: dict[str, str] = {
    "undo": "↩️", "warn": "⚠️", "mute": "🔇", "unmute": "🔊",
    "ban": "🔨", "unban": "🕊️", "kick": "🥾", "restrict": "🔒",
    "approve": "✅", "decline": "🚫", "check": "✅",
    "settings": "⚙️", "back": "⬅️", "home": "🏠",
    "rules": "📜", "games": "🎮", "game": "🎮",
    "balance": "🪙", "sparks": "🪙", "quests": "🗺️",
    "economy": "🪙", "estate": "🏘️", "defense": "🛡️",
    "acceptance": "🚪", "presentation": "🎨",
    "welcome": "👋", "goodbye": "🚪", "cleanup": "🧹",
    "theme": "🎨", "stamp": "🔖", "leaderboard": "🏆",
    "queue": "📋", "next": "⏭️", "rematch": "🔁",
    "recall": "🧠", "memories": "📋", "jobs": "📋",
    "rerun": "🔁", "status": "📊", "search": "🔍",
    "music": "🎵", "autonomy": "🤖", "anticrash": "💥",
    "antichannel": "📣", "antitag": "🏷️",
    "claim": "🎁", "drop": "🎁", "steal": "🕵️",
    "gif": "🎞️", "sticker": "🌟", "convert": "🔄",
    "react": "😀", "customemoji": "💠", "library": "🗂️",
    "treasury": "🏦", "glog": "📋", "expression": "🎭",
    "dismiss": "🔕", "flag": "🚩", "rep": "🗂️",
    "spin": "🎰", "flip": "🪙", "quit": "🚪", "pvp": "⚔️",
    "events": "🎉", "save": "💾", "research": "🔬",
}


def _already_led(text: str) -> bool:
    """True when a label already opens with a symbol/emoji —
    labelling is idempotent, so migrating a hand-rolled site
    never double-stamps it."""
    if not text:
        return False
    ch = text[0]
    if unicodedata.category(ch) in ("So", "Sk"):
        return True
    code = ord(ch)
    return 0x2190 <= code <= 0x2BFF or code >= 0x1F000


def button_label(action: str, text: str, *,
                 settings: dict[str, Any] | None = None) -> str:
    """The emoji-led label for one button: the vocabulary's
    glyph for ``action`` in front of ``text`` — unless the
    group's presentation settings turn emoji buttons off or
    override this action's glyph, or the text already leads
    with a symbol of its own."""
    text = str(text or "")
    if not text:
        return text
    block = (settings or {}).get("presentation") or {}
    if block.get("emoji_buttons") is False:
        return text
    overrides = block.get("button_emoji") or {}
    emoji = str(overrides.get(str(action or "")) or "") \
        or BUTTON_EMOJI.get(str(action or "").strip().lower(), "")
    if not emoji or _already_led(text):
        return text
    return f"{emoji} {text}"


def buttons_from_spec(spec: Any, *,
                      settings: dict[str, Any] | None = None
                      ) -> list[list[tuple[str, str]]]:
    """Author a keyboard from a spec of rows; each entry is
    either an ``(action, text, callback)`` triple (labelled
    through the vocabulary) or a ready ``(label, callback)``
    pair (passed through untouched)."""
    rows: list[list[tuple[str, str]]] = []
    for row in spec or ():
        out: list[tuple[str, str]] = []
        for entry in row or ():
            if len(entry) == 3:
                action, text, callback = entry
                out.append((button_label(str(action), str(text),
                                         settings=settings),
                            str(callback)))
            else:
                label, callback = entry
                out.append((str(label), str(callback)))
        if out:
            rows.append(out)
    return rows


# ── per-group expression settings ────────────────────────────────

#: The group store's ``expression`` block: whether she expresses
#: at all, how much GIF (off | low | normal — low keeps welcome
#: + celebrations, normal adds ban theatre), and the two moment
#: switches.  Warnings are dry at every setting (tone law, §31).
EXPRESSION_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "gifs": "normal",
    "welcome_media": True,
    "ban_theatre": True,
}


# ── the full emoji space (§22) ───────────────────────────────────

_EMOJI_RANGES = (
    (0x1F000, 0x1FAFF),   # emoticons, transport, supplemental…
    (0x2300, 0x23FF),     # misc technical (⌚ ⏰ ⏳…)
    (0x2600, 0x27BF),     # misc symbols + dingbats
    (0x2B00, 0x2BFF),     # stars, arrows-as-symbols
    (0x1F1E6, 0x1F1FF),   # regional indicators (flags compose)
)
_EMOJI_SINGLES = (0x00A9, 0x00AE, 0x203C, 0x2049, 0x2122,
                  0x2139, 0x3030, 0x303D, 0x3297, 0x3299)

#: Extra search words for the emoji people actually ask for —
#: the Unicode name alone ("HEAVY BLACK HEART") misses "love".
_KEYWORD_OVERRIDES: dict[str, tuple[str, ...]] = {
    "😀": ("happy", "smile", "grin"), "😂": ("laugh", "funny", "lol", "tears", "joy"),
    "🤣": ("rofl", "laugh", "funny", "lol"), "😍": ("love", "crush", "hearts"),
    "😘": ("kiss", "love"), "😎": ("cool", "sunglasses", "shades"),
    "🤔": ("thinking", "hmm", "ponder"), "🤨": ("skeptical", "doubt", "sus"),
    "🫠": ("melting", "embarrassed", "sinking"), "🥴": ("woozy", "drunk", "tipsy"),
    "😴": ("sleep", "tired", "zzz"), "🤯": ("mindblown", "explode", "shocked"),
    "🥳": ("party", "celebrate", "birthday"), "😭": ("cry", "sob", "sad", "tears"),
    "😡": ("angry", "mad", "rage"), "🥺": ("pleading", "puppy", "begging"),
    "🙏": ("pray", "please", "thanks", "namaste"), "👍": ("yes", "ok", "good", "like", "approve"),
    "👎": ("no", "bad", "dislike", "down"), "👏": ("clap", "applause", "bravo"),
    "🔥": ("fire", "hot", "lit", "trending"), "💯": ("hundred", "perfect", "score"),
    "❤️": ("love", "heart", "like"), "💔": ("heartbreak", "broken", "sad"),
    "💰": ("money", "rich", "bag", "cash"), "🪙": ("coin", "money", "sparks"),
    "📈": ("chart", "stocks", "up", "gains", "bullish"), "📉": ("chart", "down", "loss", "bearish"),
    "🚀": ("rocket", "moon", "launch", "blast"), "💎": ("diamond", "gem", "precious"),
    "🏆": ("trophy", "win", "champion", "gold"), "🎉": ("party", "celebrate", "confetti"),
    "🎁": ("gift", "present", "drop", "surprise"), "👀": ("eyes", "looking", "watching", "seen"),
    "🚨": ("alarm", "alert", "emergency", "siren"), "⚠️": ("warning", "caution", "alert"),
    "🔨": ("hammer", "ban", "judge"), "🛡️": ("shield", "defense", "guard", "protect"),
    "👋": ("wave", "hello", "hi", "bye", "welcome"), "🤝": ("handshake", "deal", "agree"),
    "🧠": ("brain", "smart", "think", "mind"), "💡": ("idea", "bulb", "bright"),
    "⏰": ("alarm", "clock", "time", "wake"), "🌙": ("moon", "night", "sleep"),
    "☀️": ("sun", "day", "morning", "weather"), "🌧️": ("rain", "weather", "sad"),
    "🍕": ("pizza", "food", "hungry"), "🍔": ("burger", "food", "hungry"),
    "⚽": ("football", "soccer", "sport"), "🏀": ("basketball", "sport"),
    "🎮": ("game", "gaming", "play", "controller"), "🎲": ("dice", "gamble", "random"),
    "🎵": ("music", "song", "note"), "🎧": ("music", "headphones", "listen"),
    "📚": ("books", "read", "study", "library"), "✈️": ("plane", "travel", "flight"),
    "🏠": ("home", "house"), "🏘️": ("houses", "estate", "property", "neighbourhood"),
    "🔑": ("key", "keys", "access", "estate"), "🗺️": ("map", "quest", "adventure", "world"),
    "↩️": ("undo", "back", "return", "reverse"), "✅": ("check", "yes", "done", "approve"),
    "🚫": ("no", "blocked", "decline", "forbidden"), "🥾": ("boot", "kick"),
    "🔇": ("mute", "silent", "quiet"), "🔊": ("loud", "sound", "unmute", "volume"),
    "🕊️": ("dove", "peace", "unban", "free"), "🔒": ("lock", "restricted", "private"),
}

_TONE_NAMES = ("light", "medium-light", "medium", "medium-dark", "dark")
_TONE_WORDS = ("🏻", "🏼", "🏽", "🏾", "🏿")
#: Bases that take a skin-tone modifier (people + hands + body).
_TONE_BASES = (
    "👍👎👏🙏👋🤲🖐️✋🖖👌🤌🤏✌️🤞🫰🤟🤘🤙👈👉👆🖕👇☝️✍️💅🤳💪🦾🦿🦵🦶👂🦻👃"
    "👶🧒👦👧🧑👱👨👩🧓👴👵🙍🙎🙅🙆💁🙋🧏🙇🤦🤷🧑‍🤝‍🧑👮🕵️💂🥷👷🫅🤴👸👳👲🧕🤵👰"
    "🤰🤱👼🎅🤶🧑‍🎄🦸🦹🧙🧚🧛🧜🧝🧞🧟🧖🧗🤺🏇🏂🏄🚣🚵🏊🧘🏋️🤸🤼🤾🏌️🧎🧍🧑‍🦯"
    "🧑‍🦱🧑‍🦰🧑‍🦳🧑‍🦲👱‍♀️👱‍♂️"
)

#: Well-known ZWJ sequences, named as people name them.  Any
#: other sequence is still READABLE here — describe_sequence()
#: composes a name from its parts, so nothing a member sends
#: is invisible to her.
_ZWJ_NAMED: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("👨‍👩‍👧", "family: man, woman and girl", ("family",)),
    ("👨‍👩‍👦", "family: man, woman and boy", ("family",)),
    ("👨‍👩‍👧‍👦", "family: man, woman, girl and boy", ("family",)),
    ("👩‍👩‍👧‍👦", "family: two women, girl and boy", ("family",)),
    ("👨‍👨‍👧‍👦", "family: two men, girl and boy", ("family",)),
    ("👨‍👧‍👦", "family: man, girl and boy", ("family",)),
    ("👩‍👧‍👦", "family: woman, girl and boy", ("family",)),
    ("👨‍💻", "man technologist", ("coder", "developer", "programmer", "computer")),
    ("👩‍💻", "woman technologist", ("coder", "developer", "programmer", "computer")),
    ("👨‍🎤", "man singer", ("singer", "music", "rockstar")),
    ("👩‍🎤", "woman singer", ("singer", "music", "rockstar")),
    ("👨‍🚀", "man astronaut", ("astronaut", "space")),
    ("👩‍🚀", "woman astronaut", ("astronaut", "space")),
    ("👨‍🍳", "man cook", ("chef", "cook", "kitchen")),
    ("👩‍🍳", "woman cook", ("chef", "cook", "kitchen")),
    ("👨‍⚕️", "man health worker", ("doctor", "nurse", "medical")),
    ("👩‍⚕️", "woman health worker", ("doctor", "nurse", "medical")),
    ("👨‍🎓", "man student", ("student", "graduate")),
    ("👩‍🎓", "woman student", ("student", "graduate")),
    ("👨‍🏫", "man teacher", ("teacher", "school")),
    ("👩‍🏫", "woman teacher", ("teacher", "school")),
    ("👨‍🔧", "man mechanic", ("mechanic", "fix", "repair")),
    ("👩‍🔧", "woman mechanic", ("mechanic", "fix", "repair")),
    ("👨‍⚖️", "man judge", ("judge", "law", "court")),
    ("👩‍⚖️", "woman judge", ("judge", "law", "court")),
    ("👨‍🌾", "man farmer", ("farmer", "farm")),
    ("👩‍🌾", "woman farmer", ("farmer", "farm")),
    ("👨‍🎨", "man artist", ("artist", "paint")),
    ("👩‍🎨", "woman artist", ("artist", "paint")),
    ("🧑‍🤝‍🧑", "people holding hands", ("couple", "hands", "together")),
    ("👩‍❤️‍👨", "couple with heart: woman and man", ("couple", "love")),
    ("👨‍❤️‍👨", "couple with heart: two men", ("couple", "love")),
    ("👩‍❤️‍👩", "couple with heart: two women", ("couple", "love")),
    ("👩‍❤️‍💋‍👨", "kiss: woman and man", ("kiss", "couple", "love")),
    ("🏳️‍🌈", "rainbow flag", ("pride", "rainbow", "flag")),
    ("🏴‍☠️", "pirate flag", ("pirate", "flag", "skull")),
    ("🧑‍🧑‍🧒", "family: adult, adult and child", ("family",)),
    ("🧑‍🧒‍🧒", "family: adult and two children", ("family",)),
    ("🫱🏻‍🫲🏼", "handshake: light and medium-light", ("handshake", "deal")),
    ("🐈‍⬛", "black cat", ("cat", "black", "pet")),
    ("🐻‍❄️", "polar bear", ("bear", "polar", "arctic")),
    ("🦁‍🦒", "lion and giraffe (face combo)", ("combo",)),
)

_FLAG_CODES = (
    "ng gh ke za eg ma dz tn ly sd et so tz ug rw bi mz zm zw bw na "
    "ao cm ci sn ml bf ne td gm gw lr sl gn mr mu mg mg sc cv st "
    "us gb ca mx br ar cl co pe ve ec bo py uy py cr pa do cu jm ht "
    "fr de it es pt nl be ch at pl cz sk hu gr se no dk fi ie is "
    "ru ua by ro bg hr rs ba al mk me xk lv lt ee cy mt "
    "tr sa ae qa kw om ye iq sy lb jo il ps eg kw bh "
    "in pk bd lk np bt mm th vn kh la my sg id ph jp kr cn hk tw "
    "au nz fj pg sb vu nc pf ws to ki tv "
    "ir af kz uz tm kg tj az ge am md"
).split()


class EmojiEntry:
    """One row of the emoji table: the grapheme itself, its
    Unicode name, and the words that find it."""

    __slots__ = ("char", "name", "keywords")

    def __init__(self, char: str, name: str,
                 keywords: tuple[str, ...] = ()) -> None:
        self.char = char
        self.name = name
        self.keywords = keywords

    def __repr__(self) -> str:  # pragma: no cover - debugging
        return f"EmojiEntry({self.char!r}, {self.name!r})"

    def to_dict(self) -> dict[str, Any]:
        return {"char": self.char, "name": self.name,
                "keywords": list(self.keywords)}


def _overrides(ch: str) -> tuple[str, ...]:
    """Curated keywords for one codepoint — the overrides are
    keyed by the presentation form (❤️ with its selector), the
    table by the bare codepoint; bridge the two."""
    return _KEYWORD_OVERRIDES.get(ch) \
        or _KEYWORD_OVERRIDES.get(ch + "️", ())


_TABLE: list[EmojiEntry] | None = None


def _name_tokens(name: str) -> tuple[str, ...]:
    words = re.split(r"[^a-z0-9]+", name.lower())
    return tuple(w for w in words if len(w) >= 2)


def emoji_table() -> list[EmojiEntry]:
    """The full table, built once from the Unicode standard:
    every named codepoint in the emoji blocks (the newest
    blocks included — coverage follows Unicode, not a phone),
    skin-tone variants of the people/hand bases, ISO flags,
    keycaps, and the named ZWJ sequences."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE
    entries: list[EmojiEntry] = []
    seen: set[str] = set()

    def add(char: str, name: str,
            extra: tuple[str, ...] = ()) -> None:
        if not char or char in seen:
            return
        seen.add(char)
        kws = set(_name_tokens(name)) | set(extra)
        entries.append(EmojiEntry(
            char, name, tuple(sorted(kws))))

    for lo, hi in _EMOJI_RANGES:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            name = unicodedata.name(ch, "")
            if not name:
                continue
            if 0x1F1E6 <= cp <= 0x1F1FF:
                continue  # lone regional indicators are not emoji
            add(ch, name.lower(), _overrides(ch))
    for cp in _EMOJI_SINGLES:
        ch = chr(cp)
        name = unicodedata.name(ch, "")
        if name:
            add(ch, name.lower(), _overrides(ch))
    # skin-tone variants: base + modifier, named as Unicode does
    for base in _TONE_BASES:
        base = base.replace("️", "")
        if len(base) != 1:
            continue
        bname = unicodedata.name(base, "")
        if not bname:
            continue
        for tone, word in zip(_TONE_WORDS, _TONE_NAMES):
            add(base + tone,
                f"{bname.lower()}: {word} skin tone",
                (word.split("-")[0], "skin", "tone"))
    # flags: regional-indicator pairs over the ISO list
    for code in _FLAG_CODES:
        pair = "".join(
            chr(0x1F1E6 + ord(c) - ord("a")) for c in code)
        add(pair, f"flag: {code.upper()}",
            ("flag", code, "country"))
    # keycaps
    for digit in "0123456789#*":
        add(digit + "️⃣", f"keycap: {digit}",
            ("keycap", "digit" if digit.isdigit() else "symbol"))
    # named ZWJ sequences
    for seq, name, extra in _ZWJ_NAMED:
        add(seq, name, extra)
    _TABLE = entries
    return entries


def _grapheme_clusters(text: str) -> list[str]:
    """Split into user-perceived emoji-ish clusters: a base
    plus its variation selectors, tone modifiers and ZWJ
    chains stays one cluster."""
    clusters: list[str] = []
    cur = ""
    join_next = False
    for ch in str(text or ""):
        code = ord(ch)
        attach = join_next or ch in ("️", "⃣") \
            or 0x1F3FB <= code <= 0x1F3FF or code == 0x200D
        if cur and not attach:
            clusters.append(cur)
            cur = ch
        else:
            cur += ch
        join_next = code == 0x200D
    if cur:
        clusters.append(cur)
    return clusters


def _find_entry(char: str) -> EmojiEntry | None:
    """Table lookup tolerant of the U+FE0F presentation
    selector: ❤️ (as typed) and ❤ (as tabled) are one emoji."""
    for entry in emoji_table():
        if entry.char == char:
            return entry
    bare = char.replace("️", "")
    for entry in emoji_table():
        if entry.char.replace("️", "") == bare:
            return entry
    return None


def describe_sequence(char: str) -> str:
    """Name ANY emoji grapheme, listed or not: a ZWJ chain is
    described from its parts, so an unlisted sequence a member
    invents is still legible to her."""
    char = str(char or "")
    if not char:
        return ""
    entry = _find_entry(char)
    if entry is not None:
        return entry.name
    parts = [p for p in char.replace("️", "").split("‍") if p]
    if len(parts) > 1:
        names = []
        for part in parts:
            base = part.rstrip("".join(_TONE_WORDS))
            names.append(unicodedata.name(base, "symbol").lower()
                         if len(base) == 1 else "symbol")
        return " + ".join(names) + " (sequence)"
    if len(char) == 1:
        return unicodedata.name(char, "unknown symbol").lower()
    return "emoji sequence"


def emoji_info(char: str) -> dict[str, Any] | None:
    """What one grapheme IS: table name, codepoints, keywords."""
    char = str(char or "").strip()
    if not char:
        return None
    clusters = _grapheme_clusters(char)
    target = clusters[0] if clusters else char
    name = describe_sequence(target)
    if not name:
        return None
    found = _find_entry(target)
    return {
        "char": target, "name": name,
        "codepoints": " ".join(
            f"U+{ord(c):04X}" for c in target),
        "keywords": list(found.keywords) if found else [],
    }


def search_emoji(query: str, *, limit: int = 8) -> list[EmojiEntry]:
    """Find emoji by meaning: query words score against names
    and keywords (exact token > keyword > prefix > substring),
    so 'melting' finds 🫠 and 'family' finds the ZWJ chains."""
    words = [w for w in re.split(r"[^a-z0-9]+",
                                 str(query or "").lower()) if w]
    if not words:
        return []
    scored: list[tuple[int, EmojiEntry]] = []
    for entry in emoji_table():
        score = 0
        name_words = set(_name_tokens(entry.name))
        for word in words:
            if word in name_words:
                score += 4
            elif word in entry.keywords:
                score += 3
            elif any(nw.startswith(word) for nw in name_words):
                score += 2
            elif word in entry.name:
                score += 1
        if score:
            scored.append((score, entry))
    # Ties break toward the SHORTER name: "thumbs up: medium
    # skin tone" is a tighter match for 'thumbs up medium'
    # than the medium-light variant that merely contains it.
    scored.sort(key=lambda pair: (-pair[0], len(pair[1].name)))
    return [entry for _score, entry in scored[:max(1, limit)]]


# ── her own custom emoji set (§22) ───────────────────────────────

class CustomEmojiRegistry:
    """Devon's signature set: named entries with a unicode
    fallback (always renderable), per-platform custom refs
    where the platform carries them, and optional sticker art.

    Platform truth, in code: a Telegram BOT can re-send a
    custom-emoji STICKER by file_id and can name a
    ``custom_emoji_id`` in message entities — but this gateway
    exposes no entity channel, so her set renders through the
    sticker path (sendSticker by cached ref) with the unicode
    as the everywhere fallback.  Discord custom emoji need an
    uploaded guild emoji id (``<:name:id>``) — stored as a ref
    when the owner supplies one.  WhatsApp has no custom emoji
    concept; the set ships there as stickers.  Nothing here
    invents a mechanic a platform doesn't have."""

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, dict[str, Any]] = dict(data or {})

    def names(self) -> list[str]:
        return sorted(self._data)

    def get(self, name: str) -> dict[str, Any] | None:
        entry = self._data.get(str(name or "").strip().lower())
        return dict(entry) if entry else None

    def add(self, name: str, *, unicode: str = "",
            platform: str = "", ref: str = "",
            image_path: str = "") -> dict[str, Any]:
        key = str(name or "").strip().lower()
        if not key:
            raise ValueError("a custom emoji needs a name")
        entry = dict(self._data.get(key) or {})
        if unicode:
            entry["unicode"] = str(unicode)
        if platform and ref:
            refs = dict(entry.get("refs") or {})
            refs[str(platform)] = str(ref)
            entry["refs"] = refs
        if image_path:
            entry["image_path"] = str(image_path)
        entry.setdefault("unicode", "")
        self._data[key] = entry
        return dict(entry)

    def remove(self, name: str) -> bool:
        return self._data.pop(
            str(name or "").strip().lower(), None) is not None

    def render(self, name: str, platform: str) -> dict[str, Any]:
        """How this entry goes out on ``platform``: a custom
        ref when the platform carries one, else the sticker
        art, else the unicode fallback — in that order, and
        always SOMETHING honest."""
        entry = self.get(name)
        if entry is None:
            return {"kind": "missing", "value": ""}
        refs = entry.get("refs") or {}
        platform = str(platform or "")
        if platform in refs:
            kind = "sticker_ref" if platform.startswith("telegram") \
                else "custom_ref"
            return {"kind": kind, "value": str(refs[platform]),
                    "name": str(name)}
        if entry.get("image_path"):
            return {"kind": "sticker_file",
                    "value": str(entry["image_path"]),
                    "name": str(name)}
        return {"kind": "unicode",
                "value": str(entry.get("unicode") or ""),
                "name": str(name)}

    def to_dict(self) -> dict[str, Any]:
        return dict(self._data)


# ── the library ──────────────────────────────────────────────────

def _item_id() -> str:
    return "mx_" + hashlib.sha256(
        f"{time.time()}-{random.random()}".encode()
    ).hexdigest()[:12]


class ExpressionLibrary:
    """Her persistent GIF/sticker library: items indexed by
    tag/mood with provenance, per-room recent-use memory (the
    no-repeat rule), and the custom-emoji registry — one JSON
    document beside the group store, saved atomically
    (tmp + replace).  A corrupt file is quarantined and the
    library starts empty: collected media is re-earnable,
    unlike a ledger — but the quarantine keeps the evidence."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.recovered = False
        self.data: dict[str, Any] = {"items": [], "custom_emoji": {},
                                     "meta": {}}
        self.load()

    # -- persistence -----------------------------------------
    def load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) \
                    or not isinstance(raw.get("items"), list):
                raise ValueError("library root is not an object")
            self.data = raw
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_suffix(
                f".corrupt-{int(time.time())}")
            try:
                self.path.replace(quarantine)
            except OSError:
                pass
            _log.warning(
                "expression library at %s unreadable (%s); "
                "quarantined to %s — starting empty",
                self.path, exc, quarantine)
            self.recovered = True
            self.data = {"items": [], "custom_emoji": {}, "meta": {}}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1),
                       encoding="utf-8")
        tmp.replace(self.path)

    # -- items -------------------------------------------------
    @property
    def items(self) -> list[dict[str, Any]]:
        return list(self.data.get("items") or [])

    def __len__(self) -> int:
        return len(self.data.get("items") or [])

    def add_item(self, *, kind: str, path: str = "",
                 file_ids: dict[str, str] | None = None,
                 tags: Any = (), moods: Any = (),
                 source: str = "", origin: str = ""
                 ) -> dict[str, Any]:
        """Add one item (or return the existing twin: same path,
        or a shared platform file_id — her library never holds
        the same media twice)."""
        file_ids = {str(k): str(v) for k, v in
                    (file_ids or {}).items() if v}
        for item in self.data["items"]:
            if path and item.get("path") == path:
                return item
            held = item.get("file_ids") or {}
            if any(held.get(k) == v for k, v in file_ids.items()):
                return item
        item = {
            "id": _item_id(), "kind": str(kind or "gif"),
            "path": str(path or ""), "file_ids": file_ids,
            "tags": [str(t).lower() for t in tags or () if str(t).strip()],
            "moods": [str(m).lower() for m in moods or () if str(m).strip()],
            "source": str(source or ""), "origin": str(origin or ""),
            "uses": 0, "created": time.time(),
        }
        self.data["items"].append(item)
        self.save()
        return item

    def get(self, item_id: str) -> dict[str, Any] | None:
        for item in self.data["items"]:
            if item.get("id") == item_id:
                return item
        return None

    def note_file_id(self, item_id: str, platform: str,
                     file_id: str) -> None:
        """Cache a platform file_id after a successful send —
        the next send of this item on that platform goes by id,
        no re-upload (the Telegram rule; harmless elsewhere)."""
        item = self.get(item_id)
        if item is None or not file_id:
            return
        ids = dict(item.get("file_ids") or {})
        if ids.get(platform) != file_id:
            ids[platform] = str(file_id)
            item["file_ids"] = ids
            self.save()

    def search(self, query: str) -> list[dict[str, Any]]:
        words = [w for w in re.split(r"[^a-z0-9]+",
                                     str(query or "").lower()) if w]
        if not words:
            return self.items
        out = []
        for item in self.data["items"]:
            hay = set(item.get("tags") or []) | set(item.get("moods") or [])
            if any(w in hay or any(w in h for h in hay) for w in words):
                out.append(item)
        return out

    def pick(self, tags: Any = (), moods: Any = (), *,
             room: str = "", kinds: Any = None,
             rng: random.Random | None = None
             ) -> dict[str, Any] | None:
        """Choose media for a moment: tag/mood overlap scores,
        the room's last two picks are excluded (no immediate
        repeats), and the choice among the best is random —
        she has a rotation, not a loop of one GIF."""
        rng = rng or random
        wanted_tags = {str(t).lower() for t in tags or ()}
        wanted_moods = {str(m).lower() for m in moods or ()}
        kind_set = {str(k) for k in kinds} if kinds else None
        recent = self._recent(room)
        scored: list[tuple[int, dict[str, Any]]] = []
        for item in self.data["items"]:
            if kind_set and item.get("kind") not in kind_set:
                continue
            if item.get("id") in recent:
                continue
            score = 2 * len(wanted_tags & set(item.get("tags") or [])) \
                + len(wanted_moods & set(item.get("moods") or []))
            if score or not (wanted_tags or wanted_moods):
                scored.append((score, item))
        if not scored:
            # every candidate was in the room's recent window:
            # the no-repeat rule yields to having a voice at all
            for item in self.data["items"]:
                if kind_set and item.get("kind") not in kind_set:
                    continue
                scored.append((0, item))
        if not scored:
            return None
        best = max(score for score, _ in scored)
        top = [item for score, item in scored if score == best]
        return rng.choice(top)  # type: ignore[arg-type]

    # -- per-room memory ---------------------------------------
    def _recent(self, room: str) -> list[str]:
        meta = self.data.setdefault("meta", {})
        return list((meta.get("room_recent") or {}).get(room) or [])

    def note_use(self, item_id: str, *, room: str = "") -> None:
        item = self.get(item_id)
        if item is None:
            return
        item["uses"] = int(item.get("uses") or 0) + 1
        if room:
            meta = self.data.setdefault("meta", {})
            recent = meta.setdefault("room_recent", {})
            ring = [item_id] + [i for i in recent.get(room) or []
                                if i != item_id]
            recent[room] = ring[:2]
            sent = meta.setdefault("room_last_sent", {})
            sent[room] = time.time()
        self.save()

    def last_sent(self, room: str) -> float:
        meta = self.data.get("meta") or {}
        return float((meta.get("room_last_sent") or {}).get(room) or 0.0)

    # -- custom emoji ------------------------------------------
    @property
    def custom_emoji(self) -> CustomEmojiRegistry:
        return CustomEmojiRegistry(self.data.get("custom_emoji") or {})

    def save_custom_emoji(self, registry: CustomEmojiRegistry) -> None:
        self.data["custom_emoji"] = registry.to_dict()
        self.save()


# ── GIF providers (seed sources, health-checked) ─────────────────

_STAND_DOWN_AFTER = 3     # agency/sourcehealth semantics
_DISABLE_AFTER = 6
_STAND_DOWN_S = 300.0


class ProviderHealth:
    """The sourcehealth breaker, in miniature: consecutive
    failures stand a source down for a spell, a longer streak
    disables it, any success restores it.  A source that
    doesn't work is worse than no source."""

    def __init__(self) -> None:
        self._state: dict[str, dict[str, Any]] = {}

    def available(self, name: str, *,
                  now: float | None = None) -> bool:
        st = self._state.get(name) or {}
        if st.get("disabled"):
            return False
        until = float(st.get("stood_down_until") or 0.0)
        return (now if now is not None else time.time()) >= until

    def note(self, name: str, ok: bool, *,
             now: float | None = None) -> None:
        now = now if now is not None else time.time()
        st = self._state.setdefault(
            name, {"fails": 0, "stood_down_until": 0.0,
                   "disabled": False})
        if ok:
            st.update(fails=0, stood_down_until=0.0, disabled=False)
            return
        st["fails"] = int(st.get("fails") or 0) + 1
        if st["fails"] >= _DISABLE_AFTER:
            st["disabled"] = True
        elif st["fails"] >= _STAND_DOWN_AFTER:
            st["stood_down_until"] = now + _STAND_DOWN_S

    def status(self, name: str) -> dict[str, Any]:
        return dict(self._state.get(name) or {"fails": 0})


def _http_json(url: str, *, timeout: float = 15.0) -> dict[str, Any]:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-expression/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _http_bytes(url: str, *, timeout: float = 30.0,
                cap: int = 20 * 1024 * 1024) -> bytes:
    req = urllib.request.Request(
        url, headers={"User-Agent": "nomorals-expression/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(cap + 1)
    if len(data) > cap:
        raise ValueError("media exceeds the 20MB library cap")
    return data


class TenorProvider:
    """Tenor GIF search (a configured API key is required —
    Tenor has no keyless tier; without one this source simply
    never enters the chain)."""

    name = "tenor"

    def __init__(self, api_key: str, *,
                 fetcher: Callable[[str], dict[str, Any]] | None = None
                 ) -> None:
        self.api_key = str(api_key or "")
        self._fetch = fetcher or _http_json

    def search(self, query: str, *, limit: int = 4
               ) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        url = ("https://tenor.googleapis.com/v2/search?"
               + urllib.parse.urlencode({
                   "q": query, "key": self.api_key,
                   "limit": max(1, limit), "media_filter": "mp4,gif",
                   "contentfilter": "medium"}))
        data = self._fetch(url)
        out: list[dict[str, Any]] = []
        for row in data.get("results") or []:
            formats = row.get("media_formats") or {}
            pick = formats.get("tinymp4") or formats.get("mp4") \
                or formats.get("gif") or {}
            if pick.get("url"):
                out.append({"url": str(pick["url"]),
                            "title": str(row.get("title") or query),
                            "kind": "gif"})
        return out


class GiphyProvider:
    """Giphy GIF search (configured API key, like Tenor)."""

    name = "giphy"

    def __init__(self, api_key: str, *,
                 fetcher: Callable[[str], dict[str, Any]] | None = None
                 ) -> None:
        self.api_key = str(api_key or "")
        self._fetch = fetcher or _http_json

    def search(self, query: str, *, limit: int = 4
               ) -> list[dict[str, Any]]:
        if not self.api_key:
            return []
        url = ("https://api.giphy.com/v1/gifs/search?"
               + urllib.parse.urlencode({
                   "api_key": self.api_key, "q": query,
                   "limit": max(1, limit), "rating": "pg-13"}))
        data = self._fetch(url)
        out: list[dict[str, Any]] = []
        for row in data.get("data") or []:
            images = row.get("images") or {}
            pick = images.get("fixed_height") or images.get("original") or {}
            mp4 = pick.get("mp4") or ""
            url_out = mp4 or pick.get("url") or ""
            if url_out:
                out.append({"url": str(url_out),
                            "title": str(row.get("title") or query),
                            "kind": "gif"})
        return out


def default_providers(*, tenor_key: str = "", giphy_key: str = "",
                      extra: Any = ()) -> list[Any]:
    """The seed chain: keyed providers that are configured,
    plus any caller-supplied sources (settings grow the list —
    the open-sources law; nothing here is a closed list)."""
    out: list[Any] = []
    if tenor_key:
        out.append(TenorProvider(tenor_key))
    if giphy_key:
        out.append(GiphyProvider(giphy_key))
    out.extend(extra or ())
    return out


def download_to_library(url: str, dest_dir: Path | str, *,
                        fetcher: Callable[[str], bytes] | None = None
                        ) -> Path:
    """Fetch one provider result into the library's media dir,
    content-named.  Raises on failure — the caller records the
    provider's health from the outcome."""
    data = (fetcher or _http_bytes)(url)
    ext = ".mp4" if ".mp4" in url.split("?")[0] else ".gif"
    digest = hashlib.sha256(data).hexdigest()[:16]
    dest = Path(dest_dir) / f"gif_{digest}{ext}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        dest.write_bytes(data)
    return dest


# ── moments ──────────────────────────────────────────────────────

#: Moment → (library tags, moods).  The moment vocabulary is
#: the hook layer's whole vocabulary: hosts name a moment,
#: never a file.
MOMENT_TAGS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "welcome": (("welcome", "wave", "hello", "hi"),
                ("welcoming", "happy")),
    "quest_complete": (("celebration", "party", "win", "quest"),
                       ("celebrate",)),
    "jackpot": (("jackpot", "money", "celebration", "rich"),
                ("celebrate", "rich")),
    "season_podium": (("trophy", "champion", "celebration", "win"),
                      ("celebrate",)),
    "duel_win": (("win", "victory", "celebration", "duel"),
                 ("celebrate",)),
    "estate_milestone": (("house", "keys", "celebration", "estate"),
                         ("celebrate",)),
    "ban": (("ban", "hammer", "goodbye", "justice"),
            ("strict",)),
    "celebration": (("celebration", "party", "win"),
                    ("celebrate",)),
}

#: Moments that NEVER carry media, at any setting: a warning
#: is a record, not a show (§31 tone discipline).
DRY_MOMENTS = frozenset({"warn", "warning"})

_CELEBRATION_MOMENTS = frozenset({
    "quest_complete", "jackpot", "season_podium", "duel_win",
    "estate_milestone", "celebration"})

DEFAULT_COOLDOWN_S = 25.0


def classify_notice_moment(text: str) -> str:
    """Which moment (if any) a surface notice line belongs to:
    the surface's own icon vocabulary names it — 🎉 + 'quest
    complete' is a quest, 🏆 + 'champions' is a podium.  Any
    other notice is just a notice (no media)."""
    text = str(text or "")
    low = text.lower()
    if "quest complete" in low:
        return "quest_complete"
    if "champions" in low and "season" in low:
        return "season_podium"
    if "jackpot" in low and ("🎉" in text or "won" in low
                             or "takes" in low):
        return "jackpot"
    return ""


class ExpressionService:
    """Fires moments: settings + cooldown + library/providers
    → one media send through the caller's callable.  Every
    outcome is a plain dict — ``sent`` says what ACTUALLY
    happened and ``reason`` says why when it didn't."""

    def __init__(self, library: ExpressionLibrary, *,
                 providers: Any = (),
                 media_dir: Path | str = "",
                 health: ProviderHealth | None = None,
                 rng: random.Random | None = None,
                 clock: Callable[[], float] = time.time,
                 cooldown_s: float = DEFAULT_COOLDOWN_S) -> None:
        self.library = library
        self.providers = list(providers or ())
        self.media_dir = Path(media_dir) if media_dir else None
        self.health = health or ProviderHealth()
        self.rng = rng or random.Random()
        self.clock = clock
        self.cooldown_s = float(cooldown_s)

    # -- gating, in the group's own terms ----------------------
    def _allowed(self, moment: str,
                 settings: dict[str, Any] | None) -> str:
        """'' when the moment may fire under the group's
        expression settings, else the reason it may not."""
        block = settings or {}
        if moment in DRY_MOMENTS:
            return "dry by design"
        if block.get("enabled", True) is False:
            return "expression off in this group"
        level = str(block.get("gifs") or "normal").lower()
        if level == "off":
            return "gifs off in this group"
        if moment == "welcome" \
                and block.get("welcome_media", True) is False:
            return "welcome media off in this group"
        if moment == "ban":
            if block.get("ban_theatre", True) is False:
                return "ban theatre off in this group"
            if level == "low":
                return "gifs set to low — celebrations only"
        return ""

    # -- media choice -------------------------------------------
    def _send_kind(self, item: dict[str, Any],
                   platform: str) -> str:
        kind = str(item.get("kind") or "gif")
        if platform.startswith("telegram") and kind == "gif":
            return "animation"
        return kind

    def _fetch_from_providers(self, tags: tuple[str, ...]
                              ) -> dict[str, Any] | None:
        """Walk the healthy seed chain for the moment's first
        good tag; a fetched GIF JOINS the library (the library
        grows from use, not just from steals)."""
        if not self.providers or self.media_dir is None:
            return None
        query = " ".join(tags[:2]) or "celebration"
        for provider in self.providers:
            name = str(getattr(provider, "name", "?"))
            if not self.health.available(name, now=self.clock()):
                continue
            try:
                results = provider.search(query, limit=4)
            except Exception as exc:  # noqa: BLE001 - a dead source sheds
                _log.debug("expression provider %s failed: %s",
                           name, exc)
                self.health.note(name, False, now=self.clock())
                continue
            if not results:
                continue
            self.health.note(name, True, now=self.clock())
            hit = self.rng.choice(list(results))
            try:
                path = download_to_library(
                    str(hit["url"]), self.media_dir)
            except Exception as exc:  # noqa: BLE001
                _log.debug("expression download failed: %s", exc)
                self.health.note(name, False, now=self.clock())
                continue
            return self.library.add_item(
                kind=str(hit.get("kind") or "gif"), path=str(path),
                tags=tags, source=f"provider:{name}")
        return None

    def fire(self, moment: str, *, room: str = "",
             platform: str = "",
             settings: dict[str, Any] | None = None,
             send: Callable[[Any], Any]) -> dict[str, Any]:
        """Fire one moment.  ``send`` takes a MediaRef and
        returns a SendResult-shaped object.  Never raises."""
        moment = str(moment or "")
        try:
            return self._fire(moment, room=room, platform=platform,
                              settings=settings, send=send)
        except Exception as exc:  # noqa: BLE001 - expression never breaks its host
            _log.debug("expression fire(%s) failed: %s", moment, exc)
            return {"sent": False, "moment": moment,
                    "reason": f"expression error: {exc}"[:120]}

    def _fire(self, moment: str, *, room: str, platform: str,
              settings: dict[str, Any] | None,
              send: Callable[[Any], Any]) -> dict[str, Any]:
        reason = self._allowed(moment, settings)
        if reason:
            return {"sent": False, "moment": moment, "reason": reason}
        tags, moods = MOMENT_TAGS.get(
            moment, MOMENT_TAGS["celebration"])
        if room and self.clock() - self.library.last_sent(room) \
                < self.cooldown_s:
            return {"sent": False, "moment": moment,
                    "reason": "cooldown — she just expressed here"}
        item = self.library.pick(tags, moods, room=room,
                                 rng=self.rng)
        if item is None:
            item = self._fetch_from_providers(tags)
        if item is None:
            return {"sent": False, "moment": moment,
                    "reason": "no media available — text only"}
        from .base import MediaRef

        file_id = str((item.get("file_ids") or {}).get(platform) or "")
        ref = MediaRef(path=str(item.get("path") or ""),
                       kind=self._send_kind(item, platform),
                       file_id=file_id,
                       name=f"expression {moment}")
        result = send(ref)
        if not getattr(result, "ok", False):
            return {"sent": False, "moment": moment,
                    "reason": f"send failed: "
                              f"{getattr(result, 'error', '')}"[:160],
                    "item_id": item.get("id")}
        self.library.note_use(str(item.get("id")), room=room)
        captured = str(getattr(result, "media_file_id", "") or "")
        if captured:
            self.library.note_file_id(
                str(item.get("id")), platform, captured)
        return {"sent": True, "moment": moment,
                "item_id": item.get("id"),
                "via": "file_id" if file_id else "upload"}

    def gif_for(self, query: str, *, room: str = "",
                platform: str = "",
                send: Callable[[Any], Any]) -> dict[str, Any]:
        """/gif: an explicit ask — library first (the words are
        tags), then the provider chain on the raw query."""
        words = tuple(w for w in re.split(
            r"[^a-z0-9]+", str(query or "").lower()) if w)
        item = None
        hits = self.library.search(str(query or ""))
        if hits:
            item = self.rng.choice(hits)
        if item is None and words:
            item = self._fetch_from_providers(words)
        if item is None:
            return {"sent": False,
                    "reason": "nothing in my library for that yet"}
        from .base import MediaRef

        file_id = str((item.get("file_ids") or {}).get(platform) or "")
        ref = MediaRef(path=str(item.get("path") or ""),
                       kind=self._send_kind(item, platform),
                       file_id=file_id, name="expression gif")
        result = send(ref)
        if not getattr(result, "ok", False):
            return {"sent": False,
                    "reason": f"send failed: "
                              f"{getattr(result, 'error', '')}"[:160]}
        self.library.note_use(str(item.get("id")), room=room)
        captured = str(getattr(result, "media_file_id", "") or "")
        if captured:
            self.library.note_file_id(
                str(item.get("id")), platform, captured)
        return {"sent": True, "item_id": item.get("id")}


# ── reactions (§22) ──────────────────────────────────────────────

#: The Bot API's standard reaction set — a Telegram BOT may
#: only react with these (custom-emoji reactions need a
#: premium/owned set the bot doesn't hold).  WhatsApp and
#: Discord accept any emoji.  This table is the capability
#: TRUTH the surfaces report from, not a wish.
REACTION_SETS: dict[str, frozenset[str] | None] = {
    "telegram-bot": frozenset(
        "👍 👎 ❤️ 🔥 🥰 👏 😁 🤔 🤯 😱 🤬 😢 🎉 🤩 🤮 💩 🙏 👌 🕊️ "
        "🤡 🌚 🌝 😐 🦄 😘 💯 🤣 ⚡ 🍌 🏆 💔 🤨 🍾 🍩 🥱 🖕 😴 "
        "💋 😈 👻 🎃 👀".split()),
    "telegram": None,        # MTProto userbot: any emoji
    "telegram-personal": None,
    "whatsapp": None,        # any emoji — but see the adapter:
                             # the current bridge carries no
                             # reaction verb at all (reported)
    "discord": None,         # any unicode emoji + guild customs
}


def reaction_allowed(platform: str, emoji: str) -> bool:
    """May a bot on ``platform`` place this reaction?  (The
    adapter may still lack the verb — that's reported by the
    adapter's own result, separately.)"""
    allowed = REACTION_SETS.get(str(platform or ""))
    if allowed is None:
        return True
    clusters = _grapheme_clusters(str(emoji or ""))
    return bool(clusters) and clusters[0] in allowed


#: The moment vocabulary for reactions: context word → her
#: reaction.  The brain may choose freely through the same
#: function shape later; the rules cover the standing moments.
_REACTION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("win", "won", "jackpot", "champion", "podium", "complete"), "🏆"),
    (("welcome", "join", "hello", "new member"), "👋"),
    (("funny", "lol", "haha", "joke", "😂"), "😂"),
    (("thanks", "thank", "love", "sweet"), "❤️"),
    (("question", "?", "who", "what", "help"), "👀"),
    (("sad", "sorry", "lost", "rip"), "😢"),
    (("wow", "amazing", "incredible", "🤯"), "🤯"),
    (("fire", "hot", "streak", "🔥"), "🔥"),
)


def pick_reaction(context: str, *, platform: str = "") -> str:
    """Her reaction for a moment of text/context — the first
    matching rule, adjusted to what the platform allows (a
    disallowed pick degrades to 👍, which every set holds)."""
    low = str(context or "").lower()
    for words, emoji in _REACTION_RULES:
        if any(w in low for w in words):
            return emoji if reaction_allowed(platform, emoji) else "👍"
    return "👍" if reaction_allowed(platform, "👍") else ""
