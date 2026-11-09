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
from .failures import record_failure

_log = get_logger("nomorals.social.expression")

__all__ = [
    "BUTTON_EMOJI", "button_label", "buttons_from_spec",
    "EXPRESSION_COMMANDS",
    "EXPRESSION_DEFAULTS",
    "EmojiEntry", "emoji_table", "search_emoji", "emoji_info",
    "confident_emoji_match", "describe_sequence",
    "emoji_prompt_for", "EmojiCreateUnavailable",
    "create_emoji_sticker",
    "CustomEmojiRegistry",
    "ExpressionLibrary",
    "CaptureError", "capture_library_sticker",
    "AUTO_COLLECT_ROOM_CAP", "auto_collected_from",
    "ProviderHealth", "TenorProvider", "GiphyProvider",
    "default_providers", "download_to_library",
    "WaifuPicsProvider", "NekosBestProvider",
    "REACTION_GIFS", "REACTION_GIF_COMMANDS",
    "MEDIA_DELIVERED_KINDS", "reaction_caption",
    "MOMENT_TAGS", "DRY_MOMENTS", "ExpressionService",
    "REACTION_SETS", "reaction_allowed", "pick_reaction",
    "classify_notice_moment",
]


#: The expression organ's own command family — the control
#: kinds whose handlers live in the expression runtime mixin
#: (/steal /sticker /convert /gif /emoji /customemoji
#: /emojiart /react). The per-emotion reaction commands are the
#: separate live registry REACTION_GIF_COMMANDS below, and the
#: moment vocabulary is MOMENT_TAGS. The surface manifest's
#: expression surface declares exactly this union as its
#: members; the Living Chat P3 drift test pins the two equal,
#: so a new expression command that is not registered in the
#: manifest fails by name.
EXPRESSION_COMMANDS: tuple[str, ...] = (
    "steal", "sticker", "convert", "gif", "emoji",
    "customemoji", "emojiart", "react")


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
    "gang": "🏴", "heist": "💰", "war": "⚔️", "bounty": "🎯",
    "gear": "🧰", "police": "🚔", "conflict": "⚔️",
    "rhythm": "🌙", "poll": "🗳️", "milestones": "📖",
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
    # §21 sticker stealer, second capture mode: stickers
    # landing in the room join her library passively.  The
    # design names it "an auto-collect mode a group admin
    # enables" — so it is OFF until an admin turns it on
    # (/gsettings expression collect on|off), and the §33
    # claimed-room gate applies on top, always.
    "auto_collect": False,
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


def _word_scores(entry: EmojiEntry,
                 words: list[str]) -> list[int]:
    """Per-word scores of one entry against a query: exact name
    token 4 > keyword 3 > name prefix 2 > name substring 1."""
    name_words = set(_name_tokens(entry.name))
    scores: list[int] = []
    for word in words:
        if word in name_words:
            scores.append(4)
        elif word in entry.keywords:
            scores.append(3)
        elif any(nw.startswith(word) for nw in name_words):
            scores.append(2)
        elif word in entry.name:
            scores.append(1)
        else:
            scores.append(0)
    return scores


def _query_words(query: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9]+",
                                str(query or "").lower()) if w]


def search_emoji(query: str, *, limit: int = 8) -> list[EmojiEntry]:
    """Find emoji by meaning: query words score against names
    and keywords (exact token > keyword > prefix > substring),
    so 'melting' finds 🫠 and 'family' finds the ZWJ chains."""
    words = _query_words(query)
    if not words:
        return []
    scored: list[tuple[int, EmojiEntry]] = []
    for entry in emoji_table():
        score = sum(_word_scores(entry, words))
        if score:
            scored.append((score, entry))
    # Ties break toward the SHORTER name: "thumbs up: medium
    # skin tone" is a tighter match for 'thumbs up medium'
    # than the medium-light variant that merely contains it.
    scored.sort(key=lambda pair: (-pair[0], len(pair[1].name)))
    return [entry for _score, entry in scored[:max(1, limit)]]


def confident_emoji_match(query: str) -> EmojiEntry | None:
    """The one entry that covers EVERY query word with a real
    hit (name token or keyword, score ≥ 3 per word) — the
    lookup is sure it means this.  'melting face' → 🫠;
    'thumbs up medium' → the tone entry.  'family rocket'
    covers nothing: the table holds the parts, never the
    whole — that is a creation request, not a lookup."""
    words = _query_words(query)
    if not words:
        return None
    best: tuple[int, EmojiEntry] | None = None
    for entry in emoji_table():
        scores = _word_scores(entry, words)
        if scores and all(s >= 3 for s in scores):
            total = sum(scores)
            if best is None or total > best[0] or (
                    total == best[0]
                    and len(entry.name) < len(best[1].name)):
                best = (total, entry)
    return best[1] if best is not None else None


# ── creating what the table doesn't have ─────────────────────

#: The emoji-art direction every creation prompt carries:
#: one glyph-like subject a phone keyboard would ship —
#: bold silhouette, thick outline, flat color, nothing else.
_EMOJI_STYLE = (
    "a single emoji icon of {subject}: one bold glyph-like "
    "subject, thick dark outline, flat vibrant colors, simple "
    "shapes, centered, plain transparent background, no text, "
    "no letters, no watermark, no shadow, digital emoji art "
    "style like a phone keyboard emoji")


def emoji_prompt_for(description: str) -> str:
    """The generation prompt for one created emoji: the ask
    wrapped in the house emoji-art direction."""
    subject = " ".join(str(description or "").split()).strip()
    return _EMOJI_STYLE.format(subject=subject)


class EmojiCreateUnavailable(Exception):
    """Emoji creation cannot run on this install — the message
    says, in plain words, exactly what is missing.  Never a
    fabricated emoji, never a near-match passed off as made."""


def create_emoji_sticker(description: str, out_dir: Any, *,
                         backend: Any = None) -> Path:
    """Create a NEW STICKER (the maker — named for what it
    makes; it was mislabeled "emoji creation" until the
    owner's 2026-11-07 correction: emoji are not stickers).

    The ladder: the image pipeline's best configured rung
    first; when no backend is configured — or the rung fails —
    HER OWN studio draws the design locally (deterministic,
    zero-config: the rung that works on every install).  The
    art is post-processed through the house sticker seam
    (512×512 transparent canvas, WebP).

    Raises :class:`EmojiCreateUnavailable` only when every
    rung fails — the caller answers with that plain reason,
    journaled."""
    subject = " ".join(str(description or "").split()).strip()
    if not subject:
        raise EmojiCreateUnavailable(
            "a sticker creation needs a description")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw: Path | None = None
    backend_error = ""
    if backend is None:
        from ...media_edit.generate import (
            GenerativeEditError,
            get_backend,
        )

        try:
            backend = get_backend()
        except GenerativeEditError as exc:
            backend_error = str(exc)
            backend = None
    if backend is not None:
        try:
            images = backend.generate(emoji_prompt_for(subject))
        except Exception as exc:  # noqa: BLE001 - fall to her own hand
            backend_error = f"the image rung failed: {exc}"
            images = []
        if images:
            image = images[0]
            raw = out / f"sticker_{time.time_ns()}.png"
            try:
                if hasattr(image, "save"):
                    image.save(str(raw))
                else:
                    from PIL import Image

                    Image.fromarray(image).save(str(raw))
            except Exception as exc:  # noqa: BLE001 - fall to her hand
                backend_error = f"the generated art could not " \
                                f"be saved: {exc}"
                raw = None
    if raw is None:
        # Her own hand: the studio's local renderer.  This is
        # the rung that makes creation real on installs with
        # no image backend configured at all.
        try:
            from . import emoji_studio

            spec = emoji_studio.parse_design(subject)
            raw = emoji_studio.sticker_master_png(
                spec, out, name=f"sticker_{time.time_ns()}")
        except Exception as exc:  # noqa: BLE001 - every rung failed
            raise EmojiCreateUnavailable(
                f"{backend_error + '; ' if backend_error else ''}"
                f"her local renderer failed too: {exc}") from exc
    from .media_convert import ConversionError, image_to_sticker

    try:
        return image_to_sticker(raw, out_dir=out)
    except ConversionError as exc:
        raise EmojiCreateUnavailable(str(exc)) from exc


# ── her own custom emoji set (§22) ───────────────────────────────

class CustomEmojiRegistry:
    """Devon's signature set: named entries with a unicode
    fallback (always renderable), per-platform custom refs
    where the platform carries them, and optional sticker art.

    Platform truth, in code: emoji SHE creates are registered
    in her own Telegram custom-emoji packs (Bot API, packs the
    bot owns) and the pack hands back a real ``custom_emoji_id``
    — stored here as the telegram ref; the bot adapter's
    ``send_custom_emoji`` carries it inline through message
    entities, so on Telegram her art renders AS an emoji in
    text.  The unicode face is the everywhere fallback.
    Discord custom emoji need an uploaded guild emoji id
    (``<:name:id>``) — stored as a ref when the owner supplies
    one.  WhatsApp has no custom emoji concept at all (a real
    platform wall); the set ships there as stickers.  Nothing
    here invents a mechanic a platform doesn't have."""

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
            # A telegram ref IS a custom_emoji_id (her packs);
            # other platforms' refs are their own custom ids.
            return {"kind": "custom_ref",
                    "value": str(refs[platform]),
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
            except OSError as _exc:
                record_failure("expression.load", _exc)
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
                 source: str = "", origin: str = "",
                 sha: str = "", note: str = "",
                 by_reference: bool = False
                 ) -> dict[str, Any]:
        """Add one item (or return the existing twin: same path,
        a shared platform file_id, or the same content hash —
        her library never holds the same media twice).  ``sha``
        / ``note`` / ``by_reference`` are additive: captures
        that hold bytes record the hash (the dedupe key for
        media whose path changes between downloads), and
        captures that could only hold a platform handle say
        so, with the reason in ``note``."""
        file_ids = {str(k): str(v) for k, v in
                    (file_ids or {}).items() if v}
        for item in self.data["items"]:
            if path and item.get("path") == path:
                return item
            if sha and item.get("sha") == sha:
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
            "sha": str(sha or ""), "note": str(note or ""),
            "by_reference": bool(by_reference),
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


# ── auto-collect (§21: the stealer's passive mode) ─────────────

#: Per-room bound on PASSIVE collection: one room may grow
#: the library by at most this many stickers on its own.
#: /steal is a human's deliberate claim and is not bounded
#: by this; the bound exists so an ordinary busy (or sticker-
#: happy) room cannot grow the shared library without limit
#: just by existing.  At the bound, further captures from
#: that room are plain no-ops.
AUTO_COLLECT_ROOM_CAP = 300


class CaptureError(RuntimeError):
    """A passive capture that could not honestly land."""


def auto_collected_from(library: ExpressionLibrary,
                        origin: str) -> int:
    """How many items one room has passively contributed."""
    return sum(1 for item in library.items
               if item.get("source") == "auto-collect"
               and item.get("origin") == origin)


def capture_library_sticker(library: ExpressionLibrary, *,
                            path: str = "",
                            file_ids: dict[str, str] | None = None,
                            media_dir: Path | str = "",
                            origin: str = "", name: str = ""
                            ) -> dict[str, Any]:
    """Land one passively-seen sticker in her library, under
    the same capture discipline as the packs (§21 media2):
    format is read from magic bytes, never the filename —
    WebP stays WebP, WebM stays WebM (copied into the
    library's own media dir, content-named, so the same
    sticker always lands at the same path), another raster
    shape is normalized to a real WebP through the
    conversion organ, a .tgs (Lottie) is held BY REFERENCE
    with the reason named, and media whose bytes never
    reached us is held by reference too.  Dedupe is by
    content hash first (the library's path/file_id twin
    rules ride along inside add_item): a sticker already
    held is a no-op returning the held item.  Provenance is
    the library's own shape — source "auto-collect", origin
    the room key.  Raises CaptureError when there is
    honestly nothing to hold, and when the room has reached
    AUTO_COLLECT_ROOM_CAP passive contributions."""
    from .sticker_packs import _sniff

    file_ids = {str(k): str(v) for k, v in
                (file_ids or {}).items() if v}

    def _twin() -> dict[str, Any] | None:
        """The held item for these platform handles, if any
        (the add_item twin rule, readable before the cap:
        a duplicate is a no-op, never a cap casualty)."""
        for item in library.items:
            held = item.get("file_ids") or {}
            if any(held.get(k) == v for k, v in file_ids.items()):
                return item
        return None

    def _cap_gate() -> None:
        if auto_collected_from(library, origin) >= AUTO_COLLECT_ROOM_CAP:
            raise CaptureError(
                "this room has already auto-collected "
                f"{AUTO_COLLECT_ROOM_CAP} stickers — the "
                "passive bound; /steal still claims "
                "deliberately")

    src = Path(path) if path else None
    if src is not None and not src.exists():
        src = None
    if src is None:
        if file_ids:
            twin = _twin()
            if twin is not None:
                return twin
            _cap_gate()
            return library.add_item(
                kind="sticker", path="", file_ids=file_ids,
                tags=["sticker"], source="auto-collect",
                origin=origin, by_reference=True,
                note=("held by platform reference — this "
                      "surface couldn't hand me the bytes, so "
                      "re-sends go by the platform's own handle"))
        raise CaptureError(
            "nothing to collect — no bytes and no platform "
            "handle reached me for that sticker")

    raw = src.read_bytes()
    if not raw:
        raise CaptureError("that sticker download came back "
                           "empty — nothing to collect")
    sha = hashlib.sha256(raw).hexdigest()
    for item in library.items:
        if item.get("sha") == sha:
            return item
    fmt = _sniff(raw[:16]) or src.suffix.lower().lstrip(".")
    if fmt == "tgs":
        if not file_ids:
            raise CaptureError(
                "a .tgs (Lottie) sticker with no platform "
                "handle — nothing honest to hold")
        twin = _twin()
        if twin is not None:
            return twin
        _cap_gate()
        return library.add_item(
            kind="sticker", path="", file_ids=file_ids,
            tags=["sticker"], source="auto-collect",
            origin=origin, sha=sha, by_reference=True,
            note=("a .tgs (Lottie) sticker — held by platform "
                  "reference; I don't render Lottie, so the "
                  "handle is the honest copy"))
    _cap_gate()

    dest_dir = Path(media_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    note = ""
    if fmt in ("webp", "webm"):
        dest = dest_dir / f"stk_{sha[:16]}.{fmt}"
        if not dest.exists():
            import shutil

            shutil.copyfile(src, dest)
    else:
        # Another raster shape (png/jpg photo-sticker): the
        # conversion organ normalizes it to a real WebP.
        from . import media_convert

        try:
            made = media_convert.image_to_sticker(
                src, out_dir=dest_dir)
        except media_convert.ConversionError as exc:
            raise CaptureError(
                "that media isn't a sticker I can keep: "
                f"{exc}") from exc
        dest = dest_dir / f"stk_{sha[:16]}.webp"
        made = Path(made)
        if made != dest:
            import shutil

            shutil.move(str(made), str(dest))
        fmt = "webp"
        note = "normalized to WebP from the original image"
    return library.add_item(
        kind="sticker", path=str(dest), file_ids=file_ids,
        tags=["sticker"], source="auto-collect", origin=origin,
        sha=sha, note=note)


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


# ── reaction GIF providers (keyless category seeds) ──────────────

class WaifuPicsProvider:
    """waifu.pics reaction GIFs — a keyless category API: one
    GET per category answers ``{"url": …}``.  Its SFW category
    list is the per-emotion command set's primary seed (the
    FEI XMD reaction menu is this shape).  Category APIs are
    not search APIs — ``search`` honestly returns nothing, so
    the moment chain walks past this source untouched."""

    name = "waifu.pics"

    def __init__(self, *,
                 fetcher: Callable[[str], dict[str, Any]] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def fetch(self, category: str) -> dict[str, Any]:
        data = self._fetch(
            "https://api.waifu.pics/sfw/" + str(category))
        url = str((data or {}).get("url") or "")
        return {"url": url, "title": str(category), "kind": "gif"}

    def search(self, query: str, *, limit: int = 4
               ) -> list[dict[str, Any]]:
        return []


class NekosBestProvider:
    """nekos.best reaction GIFs — the second seed: one GET per
    category answers ``{"results": [{"url": …,
    "anime_name": …}]}``.  Its action categories cover the
    emotions waifu.pics doesn't carry (baka, pout, stare…)."""

    name = "nekos.best"

    def __init__(self, *,
                 fetcher: Callable[[str], dict[str, Any]] | None = None
                 ) -> None:
        self._fetch = fetcher or _http_json

    def fetch(self, category: str) -> dict[str, Any]:
        data = self._fetch(
            "https://nekos.best/api/v2/" + str(category))
        rows = (data or {}).get("results") or []
        first = rows[0] if rows else {}
        return {"url": str(first.get("url") or ""),
                "title": str(first.get("anime_name") or category),
                "kind": "gif"}

    def search(self, query: str, *, limit: int = 4
               ) -> list[dict[str, Any]]:
        return []


def default_reaction_providers() -> list[Any]:
    """The keyless reaction chain, in seed order.  Settings
    may grow it later; the two seeds are never a closed list
    in spirit — the health breaker sheds a dead one and the
    library floor stays."""
    return [WaifuPicsProvider(), NekosBestProvider()]


# ── the per-emotion reaction commands (FEI gap wave C) ───────────

_WP = "waifu.pics"
_NB = "nekos.best"

#: command → its seed chain ((provider, category), … in the
#: order tried), the caption's verb phrases, and its glyph.
#: ``obj`` is the targeted phrase ("Ada " + obj + " 💞");
#: a transitive command's solo form borrows ``obj`` with
#: Devon as actor, an intransitive one uses ``plain``.
#: Every category named here is a REAL category of its
#: source (the canonical waifu.pics SFW / nekos.best v2
#: lists) — a command whose emotion neither source carries
#: is not shipped at all (see the wave report: think, love),
#: and a name Devon already owns for another job is never
#: hijacked (kick = member moderation, kill = the conflict
#: system; the FEI reactions of those names stay unshipped).
REACTION_GIFS: dict[str, dict[str, Any]] = {
    "awoo": {"sources": ((_WP, "awoo"),),
             "obj": "awoos at {obj}", "plain": "awoos",
             "transitive": False, "emoji": "🐺"},
    "baka": {"sources": ((_NB, "baka"),),
             "obj": "calls {obj} a baka", "plain": "",
             "transitive": True, "emoji": "💢"},
    "bite": {"sources": ((_WP, "bite"), (_NB, "bite")),
             "obj": "bites {obj}", "plain": "",
             "transitive": True, "emoji": "😬"},
    "blush": {"sources": ((_WP, "blush"), (_NB, "blush")),
              "obj": "blushes at {obj}", "plain": "blushes",
              "transitive": False, "emoji": "😊"},
    "bonk": {"sources": ((_WP, "bonk"),),
             "obj": "bonks {obj}", "plain": "",
             "transitive": True, "emoji": "🔨"},
    "bored": {"sources": ((_NB, "bored"),),
              "obj": "yawns at {obj}", "plain": "is bored",
              "transitive": False, "emoji": "🥱"},
    "bully": {"sources": ((_WP, "bully"),),
              "obj": "bullies {obj}", "plain": "",
              "transitive": True, "emoji": "😈"},
    "cringe": {"sources": ((_WP, "cringe"),),
               "obj": "cringes at {obj}", "plain": "cringes",
               "transitive": False, "emoji": "😖"},
    "cry": {"sources": ((_WP, "cry"), (_NB, "cry")),
            "obj": "cries beside {obj}", "plain": "cries",
            "transitive": False, "emoji": "😭"},
    "cuddle": {"sources": ((_WP, "cuddle"), (_NB, "cuddle")),
               "obj": "cuddles {obj}", "plain": "",
               "transitive": True, "emoji": "🥰"},
    "dance": {"sources": ((_WP, "dance"), (_NB, "dance")),
              "obj": "dances with {obj}", "plain": "dances",
              "transitive": False, "emoji": "💃"},
    "facepalm": {"sources": ((_NB, "facepalm"),),
                 "obj": "facepalms at {obj}", "plain": "facepalms",
                 "transitive": False, "emoji": "🤦"},
    "feed": {"sources": ((_NB, "feed"),),
             "obj": "feeds {obj}", "plain": "",
             "transitive": True, "emoji": "🍜"},
    "glomp": {"sources": ((_WP, "glomp"),),
              "obj": "glomps {obj}", "plain": "",
              "transitive": True, "emoji": "🤗"},
    "handhold": {"sources": ((_WP, "handhold"), (_NB, "handhold")),
                 "obj": "holds hands with {obj}",
                 "plain": "holds out a hand",
                 "transitive": False, "emoji": "🫶"},
    "handshake": {"sources": ((_NB, "handshake"),),
                  "obj": "shakes hands with {obj}",
                  "plain": "offers a handshake",
                  "transitive": False, "emoji": "🤝"},
    "happy": {"sources": ((_WP, "happy"), (_NB, "happy")),
              "obj": "smiles at {obj}", "plain": "is happy",
              "transitive": False, "emoji": "☺️"},
    "highfive": {"sources": ((_WP, "highfive"), (_NB, "highfive")),
                 "obj": "high-fives {obj}", "plain": "",
                 "transitive": True, "emoji": "✋"},
    "hug": {"sources": ((_WP, "hug"), (_NB, "hug")),
            "obj": "hugs {obj}", "plain": "",
            "transitive": True, "emoji": "💞"},
    "kiss": {"sources": ((_WP, "kiss"), (_NB, "kiss")),
             "obj": "kisses {obj}", "plain": "",
             "transitive": True, "emoji": "💋"},
    "laugh": {"sources": ((_NB, "laugh"),),
              "obj": "laughs at {obj}", "plain": "laughs",
              "transitive": False, "emoji": "😂"},
    "lick": {"sources": ((_WP, "lick"),),
             "obj": "licks {obj}", "plain": "",
             "transitive": True, "emoji": "😛"},
    "nom": {"sources": ((_WP, "nom"), (_NB, "nom")),
            "obj": "noms {obj}", "plain": "",
            "transitive": True, "emoji": "😋"},
    "pat": {"sources": ((_WP, "pat"), (_NB, "pat")),
            "obj": "pats {obj}", "plain": "",
            "transitive": True, "emoji": "🫳"},
    "poke": {"sources": ((_WP, "poke"), (_NB, "poke")),
             "obj": "pokes {obj}", "plain": "",
             "transitive": True, "emoji": "👉"},
    "pout": {"sources": ((_NB, "pout"),),
             "obj": "pouts at {obj}", "plain": "pouts",
             "transitive": False, "emoji": "😤"},
    "shrug": {"sources": ((_NB, "shrug"),),
              "obj": "shrugs at {obj}", "plain": "shrugs",
              "transitive": False, "emoji": "🤷"},
    "slap": {"sources": ((_WP, "slap"), (_NB, "slap")),
             "obj": "slaps {obj}", "plain": "",
             "transitive": True, "emoji": "🖐️"},
    "sleep": {"sources": ((_NB, "sleep"),),
              "obj": "sleeps beside {obj}", "plain": "sleeps",
              "transitive": False, "emoji": "😴"},
    "smile": {"sources": ((_WP, "smile"), (_NB, "smile")),
              "obj": "smiles at {obj}", "plain": "smiles",
              "transitive": False, "emoji": "😄"},
    "smug": {"sources": ((_WP, "smug"), (_NB, "smug")),
             "obj": "looks smug at {obj}", "plain": "looks smug",
             "transitive": False, "emoji": "😏"},
    "stare": {"sources": ((_NB, "stare"),),
              "obj": "stares at {obj}", "plain": "stares",
              "transitive": False, "emoji": "👀"},
    "thumbsup": {"sources": ((_NB, "thumbsup"),),
                 "obj": "gives {obj} a thumbs up", "plain": "",
                 "transitive": True, "emoji": "👍"},
    "tickle": {"sources": ((_NB, "tickle"),),
               "obj": "tickles {obj}", "plain": "",
               "transitive": True, "emoji": "🤣"},
    "wave": {"sources": ((_WP, "wave"), (_NB, "wave")),
             "obj": "waves at {obj}", "plain": "waves",
             "transitive": False, "emoji": "👋"},
    "wink": {"sources": ((_WP, "wink"), (_NB, "wink")),
             "obj": "winks at {obj}", "plain": "winks",
             "transitive": False, "emoji": "😉"},
    "yeet": {"sources": ((_WP, "yeet"), (_NB, "yeet")),
             "obj": "yeets {obj}", "plain": "",
             "transitive": True, "emoji": "🙌"},
}

#: Every per-emotion reaction command name.
REACTION_GIF_COMMANDS: frozenset[str] = frozenset(REACTION_GIFS)

#: Commands whose handlers answer by SENDING media through
#: the expression path (the moments' send, a steal, a made
#: sticker…): for these, an empty text result from the
#: handler means "the media WAS the reply" — the only shape
#: their success takes.  The control path's never-empty law
#: (register #32) reads this set: a media kind's silence is a
#: delivery, never the /sudo defect.
MEDIA_DELIVERED_KINDS: frozenset[str] = frozenset(
    {"steal", "sticker", "convert", "gif", "react",
     # /emoji answers in text when the table has the emoji;
     # when it CREATES one, the made sticker is the reply and
     # the handler's "" is a delivery, same contract.
     "emoji",
     # the wave-D image one-shots: /dog /cat /fox /meme /nasa
     # /pokemon answer by attaching the picture, caption riding
     # it — the same delivery contract as the kinds above.
     "dog", "cat", "fox", "meme", "nasa", "pokemon",
     # the wave-E makers: the quote cards, /removebg, /qr and
     # /couplepp (+ /pp) answer by attaching their render or
     # cutout — the same delivery contract again.
     "qcard", "canvasquote", "iqc", "wndwqute", "meme2",
     "fakemsg", "fakettc", "nokiamsg", "fakeff", "fakeffduo",
     "ffgirl", "removebg", "qr", "couplepp", "pp",
     # the download family (presentation law, 2026-11-05): /play,
     # /download, /tiktok, /instagram, /spotify and /aio answer by
     # sending their own ONE ack message and editing it as the job
     # runs — the handler's empty text means the ack already
     # posted, the same delivery contract as the kinds above.
     "play", "download", "tiktok", "ttdl", "instagram",
     "spotify", "aio"})


def reaction_caption(command: str, *, actor: str, target: str = "",
                     owner: bool = False) -> str:
    """The caption riding a reaction GIF.  Targeted: "Ada
    hugs Bola 💞".  Solo, Devon is the actor: "Devon hugs
    you 💞" when the owner summons it on himself, "Devon
    hugs Ada 💞" for a member's transitive solo, "Devon
    dances 💃" for a one-person reaction."""
    entry = REACTION_GIFS.get(str(command or "").strip().lower())
    if entry is None:
        return ""
    emoji = str(entry["emoji"])
    actor = str(actor or "").strip() or "Someone"
    target = str(target or "").strip()
    if target:
        return f"{actor} {entry['obj'].format(obj=target)} {emoji}"
    if entry["transitive"]:
        obj = "you" if owner else actor
        return f"Devon {entry['obj'].format(obj=obj)} {emoji}"
    return f"Devon {entry['plain']} {emoji}"


def _sniff_animation(data: bytes) -> tuple[str, str] | None:
    """(extension, library kind) for REAL animation bytes —
    the magic decides, never the URL's claim: GIF, MP4 (an
    animation on the chat platforms), WebM, animated-capable
    WebP.  A still image (PNG/JPEG) is not an animation and
    never rides a reaction command."""
    head = bytes(data or b"")[:16]
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return (".gif", "gif")
    if head[4:8] == b"ftyp":
        return (".mp4", "gif")
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return (".webm", "video")
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return (".webp", "sticker")
    return None


def _store_media_bytes(data: bytes, ext: str,
                       dest_dir: Path | str) -> Path:
    """Hold fetched bytes in the media dir, content-named —
    the twin of download_to_library for bytes already in
    hand (the reaction path sniffs before it stores)."""
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
                 cooldown_s: float = DEFAULT_COOLDOWN_S,
                 reaction_providers: Any = None,
                 bytes_fetcher: Callable[[str], bytes] | None = None
                 ) -> None:
        self.library = library
        self.providers = list(providers or ())
        self.media_dir = Path(media_dir) if media_dir else None
        self.health = health or ProviderHealth()
        self.rng = rng or random.Random()
        self.clock = clock
        self.cooldown_s = float(cooldown_s)
        # The reaction commands' own seed chain + the byte
        # fetcher behind it (injectable, like every provider
        # fetcher here — tests never touch the network).
        self.reaction_providers = list(
            default_reaction_providers() if reaction_providers is None
            else reaction_providers)
        self.bytes_fetcher = bytes_fetcher or _http_bytes

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

    # ── her own hand: self-created media (2026-11-07) ───────
    def _self_create(self, moment: str, tags: tuple[str, ...],
                     moods: tuple[str, ...]
                     ) -> dict[str, Any] | None:
        """Render the moment's media HERSELF: celebration-class
        moments get a studio animation, the rest a studio
        sticker — deterministic, zero-config, journaled with
        provenance ``self:render``.  The piece joins the
        library under the moment's tags, so later fires pick
        it from the shelf instead of re-rendering."""
        if self.media_dir is None:
            return None
        try:
            from . import emoji_studio
            from .media_convert import image_to_sticker

            mood = (moods[0] if moods else "") or moment
            spec = emoji_studio.mood_design(mood)
            if moment in ("celebration", "quest_complete",
                          "jackpot", "season_podium", "win",
                          "welcome", "party"):
                path = emoji_studio.mood_animation(
                    mood, self.media_dir, name=f"moment_{moment}")
                kind = "animation"
            else:
                master = emoji_studio.sticker_master_png(
                    spec, self.media_dir, name=f"moment_{moment}")
                path = image_to_sticker(master,
                                        out_dir=self.media_dir)
                kind = "sticker"
            return self.library.add_item(
                kind=kind, path=str(path), tags=tags,
                moods=moods, source="self:render")
        except Exception as exc:  # noqa: BLE001 - degrade to text
            _log.debug("expression self-create failed: %s", exc)
            return None

    def _self_create_reaction(self, command: str,
                              entry: dict[str, Any] | None
                              ) -> dict[str, Any] | None:
        """The reaction floor when every outside source is
        down: she renders the command's OWN face — an animated
        studio GIF of the entry's emoji/design — tagged with
        the command, so /hug and friends still land as media
        instead of an apology line."""
        if self.media_dir is None:
            return None
        try:
            from . import emoji_studio

            mood = str((entry or {}).get("emoji") or command)
            path = emoji_studio.mood_animation(
                command, self.media_dir, name=f"reaction_{command}")
            return self.library.add_item(
                kind="animation", path=str(path),
                tags=(command,), moods=(mood,),
                source="self:render")
        except Exception as exc:  # noqa: BLE001 - degrade to text
            _log.debug("reaction self-create failed: %s", exc)
            return None

    def menu_banner_item(self) -> dict[str, Any] | None:
        """The menu's picture head: her library's menu piece
        when one exists, else her studio banner, rendered once
        and shelved (source ``self:render``)."""
        item = self.library.pick(tags=("menu",), room="",
                                 rng=self.rng)
        if item is not None:
            return item
        if self.media_dir is None:
            return None
        try:
            from . import emoji_studio

            path = emoji_studio.menu_banner(self.media_dir)
            return self.library.add_item(
                kind="image", path=str(path),
                tags=("menu", "banner"), source="self:render")
        except Exception as exc:  # noqa: BLE001 - menu stays text
            _log.debug("menu banner render failed: %s", exc)
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
            # Her own hand (owner, 2026-11-07: "She can do
            # this stuffs herself"): when no library piece and
            # no provider answers, she CREATES the moment's
            # media — a studio render with provenance — instead
            # of going silently text-only.  This is the rung
            # that keeps greetings/celebrations/theatre alive
            # on installs with no GIF keys at all.
            item = self._self_create(moment, tags, moods)
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

    # -- per-emotion reaction commands (wave C) -----------------
    def _reaction_provider(self, name: str) -> Any:
        for provider in self.reaction_providers:
            if str(getattr(provider, "name", "")) == name:
                return provider
        return None

    def _fetch_reaction_item(self, command: str,
                             entry: dict[str, Any]
                             ) -> dict[str, Any] | None:
        """Walk the command's seed chain in order: a source
        is health-checked, its answer's bytes are fetched and
        SNIFFED as a real animation before they can ride —
        a still, a dead source or an empty answer falls
        through to the next seed, and every outcome is noted
        in the source's health.  A good fetch joins the
        library tagged with the command (the floor grows)."""
        for provider_name, category in entry["sources"]:
            provider = self._reaction_provider(str(provider_name))
            if provider is None:
                continue
            if not self.health.available(str(provider_name),
                                         now=self.clock()):
                continue
            try:
                hit = provider.fetch(str(category))
            except Exception as exc:  # noqa: BLE001 - a dead source sheds
                _log.debug("reaction provider %s failed: %s",
                           provider_name, exc)
                self.health.note(str(provider_name), False,
                                 now=self.clock())
                continue
            url = str((hit or {}).get("url") or "")
            if not url:
                self.health.note(str(provider_name), False,
                                 now=self.clock())
                continue
            try:
                data = self.bytes_fetcher(url)
            except Exception as exc:  # noqa: BLE001
                _log.debug("reaction download failed: %s", exc)
                self.health.note(str(provider_name), False,
                                 now=self.clock())
                continue
            sniffed = _sniff_animation(data)
            if sniffed is None:
                # The source answered with something that is
                # not an animation — a wrong asset; it is noted
                # against the source and NEVER sent.
                _log.debug(
                    "reaction provider %s served a non-animation "
                    "for %s", provider_name, command)
                self.health.note(str(provider_name), False,
                                 now=self.clock())
                continue
            self.health.note(str(provider_name), True,
                             now=self.clock())
            if self.media_dir is None:
                continue
            ext, kind = sniffed
            path = _store_media_bytes(data, ext, self.media_dir)
            return self.library.add_item(
                kind=kind, path=str(path), tags=[command],
                source=f"provider:{provider_name}")
        return None

    def reaction_gif(self, command: str, *, room: str = "",
                     platform: str = "",
                     send: Callable[[Any], Any]) -> dict[str, Any]:
        """One per-emotion reaction command (/hug, /pat, …):
        the command's seed chain first (a fresh fetch), then
        her library floor — items earlier fetches tagged with
        this command — then one honest line.  Never a bare
        URL, never a still, never an unrelated GIF.  Like
        every expression path: never raises."""
        command = str(command or "").strip().lower()
        entry = REACTION_GIFS.get(command)
        if entry is None:
            return {"sent": False,
                    "reason": f"no reaction called /{command}"}
        try:
            item = self._fetch_reaction_item(command, entry)
            if item is None:
                # Beyond-anime rung (owner, 2026-11-07): the
                # general GIF providers (Tenor/Giphy) answer
                # the command word too — the reaction set is
                # not locked to anime sources.
                item = self._fetch_from_providers((command,))
            if item is None:
                item = self.library.pick(tags=(command,),
                                         room=room, rng=self.rng)
            if item is None:
                # Her own hand: she renders the reaction
                # herself rather than apologizing in text.
                item = self._self_create_reaction(command, entry)
            if item is None:
                return {"sent": False, "reason": (
                    f"no /{command} GIF reached me — every "
                    "source is down and my library has none yet")}
            from .base import MediaRef

            file_id = str((item.get("file_ids") or {}).get(platform)
                          or "")
            ref = MediaRef(path=str(item.get("path") or ""),
                           kind=self._send_kind(item, platform),
                           file_id=file_id,
                           name=f"reaction {command}")
            result = send(ref)
            if not getattr(result, "ok", False):
                return {"sent": False,
                        "reason": f"send failed: "
                                  f"{getattr(result, 'error', '')}"[:160],
                        "item_id": item.get("id")}
            self.library.note_use(str(item.get("id")), room=room)
            captured = str(getattr(result, "media_file_id", "")
                           or "")
            if captured:
                self.library.note_file_id(
                    str(item.get("id")), platform, captured)
            return {"sent": True, "item_id": item.get("id")}
        except Exception as exc:  # noqa: BLE001 - expression never breaks its host
            _log.debug("reaction gif %s failed: %s", command, exc)
            return {"sent": False,
                    "reason": f"expression error: {exc}"[:120]}


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
