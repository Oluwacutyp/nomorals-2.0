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

import time
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
    "parse_schedule_args",
    "format_schedule_instant",
    "parse_invoice_args",
    "invoice_payload_for",
    "format_search_results",
    "parse_statusaud_args",
    "format_status_audience",
    "format_status_viewers",
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
    "sendloc", "sendcontact", "find", "findall",
    "statusaud", "statusviewers",
    "sendlater", "deletechat", "clearhistory",
    "botprofile", "invoice",
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


# -- scheduled sends + chat deletion (parity bundle #14) --

_SCHEDULE_SCALES = {"s": 1, "m": 60, "h": 3600, "d": 86400}


def parse_schedule_args(tail: str, now: float) -> dict[str, Any]:
    """Parse /sendlater's tail: ``<when> <words…>`` — the
    pure half of the scheduled-send grammar.  (The kind is
    sendlater, not schedule: /schedule is the cron surface,
    the makepoll-vs-/poll precedent.)

    Two when-forms, one vocabulary: a RELATIVE offset
    (``+45s``, ``+30m``, ``+2h``, ``+1d`` — the codebase's
    duration token with a leading plus) or an ABSOLUTE UTC
    instant (``2026-11-10 14:30`` — every surface here
    speaks UTC, and the reply restates the parsed instant
    so nothing lands at a silent guess).  Returns
    ``{ok, at, text}`` with ``at`` an epoch float, or
    ``{ok: False, error}`` — a misshapen when and a past
    instant both refuse in words here, before any traffic.

    The words may come back EMPTY: /sendlater can ride
    attached or replied-to media, whose caption is
    optional — whether an empty text is a usage error is
    the handler's call, not the parser's."""
    import calendar
    import re

    usage = ("usage: /sendlater +2h the words to send — "
             "or an absolute UTC time: /sendlater "
             "2026-11-10 14:30 the words to send")
    text = str(tail or "").strip()
    if not text:
        return {"ok": False, "error": usage}
    head, _, rest = text.partition(" ")
    rest = rest.strip()
    match = re.fullmatch(r"\+(\d+)([smhd])", head.lower())
    if match:
        at = float(now) + int(match.group(1)) * \
            _SCHEDULE_SCALES[match.group(2)]
    else:
        # an absolute instant is TWO tokens: the date,
        # then the clock — everything after is the message
        clock, _, remainder = rest.partition(" ")
        stamp = f"{head} {clock}".strip()
        parsed = None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                parsed = time.strptime(stamp, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return {"ok": False, "error": (
                f"“{head}” isn't a when I speak — {usage}")}
        at = float(calendar.timegm(parsed))
        rest = remainder.strip()
    if at <= float(now):
        return {"ok": False, "error": (
            "that time is already past — a scheduled "
            "message needs a future when")}
    return {"ok": True, "at": at, "text": rest}


def format_schedule_instant(at: float) -> str:
    """One scheduled instant the way the surfaces speak
    instants: ``2026-11-10 14:30 UTC`` (gmtime — the
    codebase's one clock face for owner-visible times)."""
    return time.strftime("%Y-%m-%d %H:%M UTC",
                         time.gmtime(float(at)))
def parse_invoice_args(tail: str) -> dict[str, Any]:
    """Parse /invoice's tail: ``Title | Description |
    Stars`` — what is sold, what the buyer gets, and the
    price as a whole number of Telegram Stars (parity
    bundle #18).  Shape problems come back as ``error`` in
    words; the field bounds themselves are
    base.invoice_shape_error's, shared with the adapter so
    the command and the verb refuse identically."""
    from .base import invoice_shape_error

    usage = ("usage: /invoice Title | Description | Stars "
             "— e.g. /invoice Signal pack | One month of "
             "premium signals | 50 (add 'link' first for a "
             "shareable invoice link instead)")
    parts = [p.strip() for p in str(tail or "").split("|")]
    if len(parts) != 3 or not all(parts):
        return {"ok": False, "error": usage}
    title, description, amount_raw = parts
    try:
        amount = int(amount_raw)
    except ValueError:
        return {"ok": False, "error": (
            f"“{amount_raw}” isn't a whole number of Stars — "
            f"the price is a count of ⭐, e.g. 50")}
    problem = invoice_shape_error(title, description, amount)
    if problem:
        return {"ok": False, "error": problem}
    return {"ok": True, "title": title,
            "description": description, "amount": amount}


def invoice_payload_for(title: str) -> str:
    """The invoice payload slug for a title — the opaque
    string Telegram hands back in the pre-checkout query
    and the successful_payment receipt, tying the payment
    to what was sold.  Derived from the title only (never
    from buyer data): lowercased, non-alphanumerics folded
    to dashes, capped well under the Bot API's 128-byte
    payload ceiling."""
    slug = "".join(
        ch if ch.isalnum() else "-"
        for ch in str(title or "").lower())
    slug = "-".join(p for p in slug.split("-") if p)[:48]
    return f"devon:{slug}" if slug else "devon-invoice"


# -- message search (parity bundle #11, census "search depth") --

def _search_snippet(text: str, query: str,
                    width: int = 140) -> str:
    """One hit's text, flattened to a line and windowed
    around the match when it is long — a search list is a
    finding surface: the window shows WHY the message
    matched, and the conversation it names is where the
    full text lives."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= width:
        return flat
    idx = flat.lower().find(str(query or "").lower())
    if idx < 0:
        return flat[:width].rstrip() + "…"
    start = max(0, idx - 40)
    end = min(len(flat), start + width)
    start = max(0, end - width)
    out = flat[start:end].strip()
    if start > 0:
        out = "…" + out
    if end < len(flat):
        out += "…"
    return out


def format_search_results(verdict: Any, *, scope: str = "",
                            show_chat: bool = False) -> str:
    """Render one search verdict (the adapters' shared
    search shape) as plain lines for the owner.

    Only what the surface actually found prints: the
    header states the match count — the TOTAL when the
    page was capped, so "20 shown of 57" never reads as
    "20 exist" — and the closing line names the coverage,
    what was actually searched, because each surface's
    reach differs (a bridge's session memory is not an
    archive).  A hit prints its date, its sender ("you"
    for the owner's own messages) and a snippet around
    the match; with ``show_chat`` it also names its
    conversation.  Message ids and chat ids never print —
    this is a reading surface, not a debug dump."""
    verdict = verdict if isinstance(verdict, dict) else {}
    if not verdict.get("ok"):
        return ("the search didn't run — "
                + str(verdict.get("error")
                      or "the surface refused"))
    messages = list(verdict.get("messages") or [])
    total = verdict.get("total")
    if not isinstance(total, int) or total < len(messages):
        total = len(messages)
    query = str(verdict.get("query") or "")
    if not messages:
        head = f"no messages match “{query}”"
        if scope:
            head += f" in {scope}"
        lines = [head + "."]
    else:
        head = (f"{total} message"
                f"{'' if total == 1 else 's'} match "
                f"“{query}”")
        if scope:
            head += f" in {scope}"
        if total > len(messages):
            head += (f" — showing the {len(messages)} "
                     f"most recent")
        lines = [head + ":"]
        for msg in messages:
            lines.append(_search_hit_line(msg, query,
                                          show_chat=show_chat))
    coverage = str(verdict.get("coverage") or "")
    if coverage:
        lines.append(f"(searched {coverage})")
    return "\n".join(lines)


def _search_hit_line(msg: Any, query: str, *,
                     show_chat: bool = False) -> str:
    when = ""
    ts = getattr(msg, "ts", None)
    if ts:
        try:
            when = time.strftime(
                "%Y-%m-%d %H:%M",
                time.localtime(float(ts)))
        except (TypeError, ValueError, OverflowError,
                OSError):
            when = ""
    who = ("you" if not getattr(msg, "incoming", True)
           else str(getattr(msg, "sender", "") or "")
           or "someone")
    parts = []
    if show_chat:
        chat = getattr(msg, "chat", None)
        title = str(getattr(chat, "title", "") or "")
        if title:
            parts.append(title)
    if when:
        parts.append(when)
    parts.append(who)
    snippet = _search_snippet(
        str(getattr(msg, "text", "") or ""), query)
    return "• " + " · ".join(parts) + f" — {snippet}"


# -- status depth (parity bundle #13, census "WhatsApp
#    status depth") --

def parse_statusaud_args(tail: str) -> dict[str, Any]:
    """Parse /statusaud's tail: empty shows the current
    audience; ``contacts`` sets everyone; ``except`` /
    ``only`` set the list mode with the people after the
    mode word, comma-separated — names or phone numbers,
    resolved to person JIDs by the caller (a name is a
    directory lookup, not a parse).  Shape problems come
    back as ``error`` in words."""
    text = str(tail or "").strip()
    if not text:
        return {"ok": True, "action": "show"}
    head, _, rest = text.partition(" ")
    mode = head.strip().lower()
    if mode not in ("contacts", "except", "only"):
        return {"ok": False, "error": (
            "usage: /statusaud shows the audience · "
            "/statusaud contacts · /statusaud except "
            "Name, +234… · /statusaud only Name, +234…")}
    people = [p.strip() for p in rest.split(",")
              if p.strip()]
    if mode == "contacts" and people:
        return {"ok": False, "error": (
            "the contacts audience takes no list — "
            "it is everyone")}
    if mode != "contacts" and not people:
        return {"ok": False, "error": (
            "an " + mode + " audience needs at least "
            "one person — name them after the word, "
            "comma-separated")}
    return {"ok": True, "action": "set", "mode": mode,
            "people": people}


def _audience_person_label(jid: str,
                           names: dict[str, str]) -> str:
    """One audience/viewer JID as the owner should read
    it: the contact's name when known, else their phone
    number.  A raw LID never prints — an unresolvable LID
    is worded, not dumped."""
    name = str(names.get(jid) or "").strip()
    if name:
        return name
    if jid.endswith("@s.whatsapp.net"):
        return "+" + jid.split("@", 1)[0]
    return "a contact whose name isn't known"


def format_status_audience(verdict: Any, *,
                           names: dict[str, str] | None = None
                           ) -> str:
    """Render one status-audience verdict (the adapters'
    shared shape) as plain lines: the mode in words, then
    the people by name or number.  When the surface could
    not hand back the list itself (``list_known`` False),
    the reply says exactly that instead of printing an
    empty list as if nobody were on it."""
    verdict = verdict if isinstance(verdict, dict) else {}
    if not verdict.get("ok"):
        return ("the status audience didn't answer — "
                + str(verdict.get("error")
                      or "the surface refused"))
    names = names or {}
    mode = str(verdict.get("mode") or "")
    entries = [str(j) for j in (verdict.get("list") or [])]
    labels = [_audience_person_label(j, names)
              for j in entries]
    written = "set: " if verdict.get("written") else ""
    if mode == "contacts":
        return (written + "your WhatsApp status audience "
                "is all your contacts.")
    if not verdict.get("list_known", True):
        return (written + "your WhatsApp status audience "
                "is " + ("everyone except a list"
                         if mode == "except"
                         else "a chosen list only")
                + " — the list itself isn't readable on "
                "this surface, so I won't guess the names.")
    if mode == "except":
        head = ("your WhatsApp status goes to all your "
                "contacts except:")
    elif mode == "only":
        head = "your WhatsApp status is shared only with:"
    else:
        return ("the status audience came back in a form "
                "I don't know: " + (mode or "empty"))
    lines = [written + head]
    lines += ["• " + label for label in labels]
    return "\n".join(lines)


def format_status_viewers(verdict: Any, *,
                          names: dict[str, str] | None = None
                          ) -> str:
    """Render one status-viewers verdict as plain lines:
    the count, each viewer by name or number with the
    time they were first seen, and the coverage line —
    the count is views the bridge has SEEN, so zero seen
    reads "no views seen yet", never "nobody viewed".
    No message id prints."""
    verdict = verdict if isinstance(verdict, dict) else {}
    if not verdict.get("ok"):
        return ("the status viewers didn't answer — "
                + str(verdict.get("error")
                      or "the surface refused"))
    names = names or {}
    viewers = [v for v in (verdict.get("viewers") or [])
               if isinstance(v, dict)]
    coverage = str(verdict.get("coverage") or "")
    if not viewers:
        lines = ["no views seen on your status yet."]
    else:
        head = (f"{len(viewers)} "
                f"{'person has' if len(viewers) == 1 else 'people have'}"
                " viewed your status:")
        lines = [head]
        for row in viewers:
            label = _audience_person_label(
                str(row.get("jid") or ""), names)
            when = ""
            ts = row.get("ts")
            if ts:
                try:
                    when = time.strftime(
                        "%Y-%m-%d %H:%M",
                        time.localtime(float(ts) / 1000.0))
                except (TypeError, ValueError,
                        OverflowError, OSError):
                    when = ""
            lines.append("• " + label
                         + (f" · {when}" if when else ""))
    if coverage:
        lines.append(f"(counting {coverage})")
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
