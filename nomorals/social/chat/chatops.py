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
    "PRIVACY_CATEGORIES",
    "PRIVACY_VALUE_WORDS",
    "privacy_pair_error",
    "format_privacy_settings",
    "TG_PRIVACY_CATEGORIES",
    "tg_privacy_pair_error",
    "format_tg_privacy_settings",
    "parse_makepoll_args",
    "parse_sendloc_args",
    "parse_sendcontact_args",
    "compose_group_status",
    "kind_for_chat_key",
]

#: The wave's kinds — one set, spelled once here; the runtime
#: dispatch, the registry tables and the tests all read it.
CHATOPS_COMMANDS = frozenset({
    "archive", "unarchive", "mutechat", "unmutechat",
    "readchat", "unreadchat", "star", "unstar",
    "forward",
    "unsend", "edit", "editmsg", "antiedit", "poststatus",
    "sendonce", "sendnote",
    "chanstory", "channeltexts",
    "block", "unblock", "anticall", "disappear",
    "statusreact", "banchecker", "gcstatus", "statusinfo",
    "pollres", "tagall", "hidetag", "antitag",
    "viewonce", "vv",
    "joinlink", "newgroup", "newchannel", "leave",
    "myprofile", "privacy", "makepoll", "vcf", "addcontact",
    "sendloc", "sendcontact",
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


# -- own identity + polls (parity G3, census bundles #5/#6) --

#: The WhatsApp privacy vocabulary, shared by the adapter,
#: both bridges and the /privacy command so the category and
#: value a command accepts are exactly the ones the protocol
#: documents (whatsmeow types/user.go).  Keys are the
#: owner-facing category names; the lists are the values that
#: category takes.  A pair outside this table refuses in words
#: BEFORE any traffic — the server silently ignoring a bad
#: value would be a fake success.
PRIVACY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "groupadd": ("all", "contacts", "contact_blacklist", "none"),
    "lastseen": ("all", "contacts", "contact_blacklist", "none"),
    "status": ("all", "contacts", "contact_blacklist", "none"),
    "profile": ("all", "contacts", "contact_blacklist", "none"),
    "readreceipts": ("all", "none"),
    "online": ("all", "match_last_seen"),
    "calladd": ("all", "known"),
    "messages": ("all", "contacts"),
    "defense": ("on_standard", "off"),
    "stickers": ("contacts", "contact_allowlist", "none"),
}

#: Plain-words labels for the privacy values, so a /privacy
#: read answers in language instead of protocol tokens.
PRIVACY_VALUE_WORDS: dict[str, str] = {
    "all": "everyone",
    "contacts": "my contacts",
    "contact_blacklist": "my contacts except…",
    "contact_allowlist": "only some contacts",
    "none": "nobody",
    "match_last_seen": "same as last seen",
    "known": "people I know",
    "on_standard": "on (standard)",
    "off": "off",
}


def privacy_pair_error(category: str, value: str) -> str:
    """'' when (category, value) is a settable privacy pair,
    else the refusal in words — naming the category's real
    values so the owner's next try is informed, not a guess."""
    category = str(category or "").strip().lower()
    value = str(value or "").strip().lower()
    if category not in PRIVACY_CATEGORIES:
        return ("privacy categories are: " +
                ", ".join(sorted(PRIVACY_CATEGORIES)))
    values = PRIVACY_CATEGORIES[category]
    if value not in values:
        return f"{category} can be: " + ", ".join(values)
    return ""


def format_privacy_settings(settings: Any,
                            *, header: str = "") -> str:
    """The /privacy read: one line per category the bridge
    could actually read, in plain words.  A category missing
    from the map was NOT read — it is listed as not readable
    rather than dressed up with a value nobody fetched."""
    settings = settings if isinstance(settings, dict) else {}
    lines = [header or "🔒 Your WhatsApp privacy:"]
    for category in sorted(PRIVACY_CATEGORIES):
        value = settings.get(category)
        if value in (None, ""):
            lines.append(f"• {category}: couldn't read")
        else:
            word = PRIVACY_VALUE_WORDS.get(str(value),
                                           str(value))
            lines.append(f"• {category}: {word} ({value})")
    lines.append("Change one: /privacy lastseen contacts")
    return "\n".join(lines)


# -- own identity on Telegram (parity G5, census bundle #8) --

#: The Telegram privacy vocabulary for /privacy on the
#: personal (userbot) surface: the same command grammar as
#: WhatsApp's table above, over Telegram's own privacy keys.
#: Category names overlap WhatsApp's wherever the MEANING is
#: the same (lastseen = status timestamp, profile = who sees
#: the profile photo, groupadd = who can add the account to
#: chats); the rest are Telegram's own keys in plain names.
#: Values are the shared plain-words set (all / contacts /
#: none — everyone / my contacts / nobody): those are the
#: three rule shapes a bare word can honestly name on
#: Telegram (InputPrivacyValueAllowAll / AllowContacts /
#: DisallowAll).  Finer Telegram rules (specific people,
#: close friends, premium, chat-member lists) READ BACK as
#: named labels — see the Telegram adapter's rule mapping —
#: never squeezed into a word they are not.
TG_PRIVACY_CATEGORIES: dict[str, tuple[str, ...]] = {
    "lastseen": ("all", "contacts", "none"),
    "profile": ("all", "contacts", "none"),
    "about": ("all", "contacts", "none"),
    "phone": ("all", "contacts", "none"),
    "groupadd": ("all", "contacts", "none"),
    "calls": ("all", "contacts", "none"),
    "p2p": ("all", "contacts", "none"),
    "forwards": ("all", "contacts", "none"),
    "voice": ("all", "contacts", "none"),
    "birthday": ("all", "contacts", "none"),
}


def tg_privacy_pair_error(category: str, value: str) -> str:
    """'' when (category, value) is a settable Telegram
    privacy pair, else the refusal in words — the Telegram
    twin of privacy_pair_error, over TG_PRIVACY_CATEGORIES."""
    category = str(category or "").strip().lower()
    value = str(value or "").strip().lower()
    if category not in TG_PRIVACY_CATEGORIES:
        return ("Telegram privacy categories are: " +
                ", ".join(sorted(TG_PRIVACY_CATEGORIES)))
    values = TG_PRIVACY_CATEGORIES[category]
    if value not in values:
        return f"{category} can be: " + ", ".join(values)
    return ""


def format_tg_privacy_settings(settings: Any,
                               *, header: str = "") -> str:
    """The /privacy read on Telegram: one line per category
    the surface could actually read.  A value in the shared
    vocabulary prints as plain words with the token beside
    it; a value that is a named label (a Telegram rule shape
    with no shared word — close friends, specific people…)
    prints as the label itself, so it is NAMED, never dressed
    as a vocabulary value it is not.  A category missing from
    the map was NOT read — listed as not readable, the
    WhatsApp read's same honesty."""
    settings = settings if isinstance(settings, dict) else {}
    lines = [header or "🔒 Your Telegram privacy:"]
    for category in sorted(TG_PRIVACY_CATEGORIES):
        value = settings.get(category)
        if value in (None, ""):
            lines.append(f"• {category}: couldn't read")
        elif str(value) in PRIVACY_VALUE_WORDS:
            word = PRIVACY_VALUE_WORDS[str(value)]
            lines.append(f"• {category}: {word} ({value})")
        else:
            lines.append(f"• {category}: {value}")
    lines.append("Change one: /privacy lastseen contacts")
    return "\n".join(lines)


def parse_makepoll_args(tail: str) -> dict[str, Any]:
    """Parse /makepoll's tail: ``[multi |] Question | Opt | Opt
    [| …]`` — pipe-separated.  A leading ``multi`` segment makes
    the poll multi-choice.  Shape problems (a missing question,
    fewer than two options) come back as ``error`` in words;
    the platform's own count/length caps are the adapter's
    refusal, so one grammar lives in each layer."""
    parts = [p.strip() for p in str(tail or "").split("|")]
    parts = [p for p in parts if p != ""]
    multiple = False
    if parts and parts[0].lower() == "multi":
        multiple = True
        parts = parts[1:]
    if not parts:
        return {"ok": False, "error": (
            "usage: /makepoll Question | Option | Option — "
            "start with 'multi |' to allow several choices")}
    question, options = parts[0], parts[1:]
    if len(options) < 2:
        return {"ok": False, "error": (
            "a poll needs a question and at least two options: "
            "/makepoll Question | Option | Option")}
    return {"ok": True, "question": question,
            "options": options, "multiple": multiple}


def parse_sendloc_args(tail: str) -> dict[str, Any]:
    """Parse /sendloc's tail: ``<lat>, <lng> [| Place name |
    Address]`` — the coordinates comma- or space-separated,
    then an optional place name and address as pipe segments
    (a named place rides the platform's venue form; bare
    coordinates drop a plain pin).  Shape problems come back
    as ``error`` in words; the coordinate range rule itself
    is base.location_shape_error's, shared with the adapters
    and mirrored in the bridges."""
    from .base import location_shape_error

    parts = [p.strip() for p in str(tail or "").split("|")]
    head = parts[0] if parts else ""
    words = head.replace(",", " ").split()
    if len(words) != 2:
        return {"ok": False, "error": (
            "usage: /sendloc latitude, longitude — e.g. "
            "/sendloc 6.5244, 3.3792 — add '| Place name | "
            "Address' to name the pin")}
    try:
        latitude = float(words[0])
        longitude = float(words[1])
    except ValueError:
        return {"ok": False, "error": (
            "usage: /sendloc latitude, longitude — e.g. "
            "/sendloc 6.5244, 3.3792 — add '| Place name | "
            "Address' to name the pin")}
    problem = location_shape_error(latitude, longitude)
    if problem:
        return {"ok": False, "error": problem}
    name = parts[1] if len(parts) > 1 else ""
    address = parts[2] if len(parts) > 2 else ""
    return {"ok": True, "latitude": latitude,
            "longitude": longitude, "name": name,
            "address": address}


def parse_sendcontact_args(tail: str) -> dict[str, Any]:
    """Parse /sendcontact's tail: ``Name | +number`` — the
    /addcontact grammar, for the card form: the name before
    the pipe, the number after it.  Shape problems come back
    as ``error`` in words (base.contact_shape_error is the
    shared rule)."""
    from .base import contact_shape_error

    name_part, sep, phone_part = str(tail or "").partition("|")
    name = name_part.strip()
    phone = phone_part.strip()
    if not sep or not name or not phone:
        return {"ok": False, "error": (
            "usage: /sendcontact Name | +number — e.g. "
            "/sendcontact Ada Lovelace | +2348000000000")}
    problem = contact_shape_error(phone, name)
    if problem:
        return {"ok": False, "error": problem}
    return {"ok": True, "name": name, "phone": phone}


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
