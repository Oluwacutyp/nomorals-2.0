"""vCard (VCF) building and group-roster merging for /vcf.

The owner's ask (2026-11-08): a VCF creator — each number
with its name, like the people on the group — for Telegram
and WhatsApp.  This module is the pure organ: it knows the
vCard 3.0 wire shape and how to merge people-rows arriving
from several roster sources into one person with one best
name and every known number.  It does NO platform I/O: the
runtime handler assembles the rows (bridge roster, member
ledger, admin lists, the personal surface's member fetch),
hands them here, and delivers the file this module renders.

Name preference, stated once and pinned: a SAVED CONTACT
name outranks a push/display name, which outranks an
observed ledger name, which outranks a bare @username.
Ties go to the longer (more complete) string.  A person
whose number no source can see is never carded — the
handler counts them instead (no number, no card, no
invention).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

__all__ = [
    "RANK_CONTACT",
    "RANK_DISPLAY",
    "RANK_LEDGER",
    "RANK_USERNAME",
    "normalize_number",
    "whatsapp_number_from_jid",
    "merge_people",
    "build_vcard",
    "build_vcf",
    "parse_manual_entries",
    "slugify_filename",
    "compose_export_line",
]

#: Name-source ranks for :func:`merge_people` — lower wins.
RANK_CONTACT = 0    # saved address-book contact name
RANK_DISPLAY = 1    # push name / profile display name
RANK_LEDGER = 2     # name observed in Devon's member ledger
RANK_USERNAME = 3   # @username — an identifier, not a name

_MIN_DIGITS = 7     # shorter than this is not a phone number
_MAX_DIGITS = 15    # E.164's own ceiling


def normalize_number(raw: Any) -> tuple[str, str]:
    """(display, digits) for one raw number, or (\"\", \"\").

    The display keeps a leading ``+`` when the input carried
    one, or when the digits plainly read as an international
    number (>= 10 digits, no trunk 0).  A local-format number
    (leading 0) and short numbers are kept exactly as dialed
    — no country code is invented for them.  ``digits`` is
    the bare digit string used as the dedup key.
    """
    text = str(raw or "").strip()
    if not text:
        return "", ""
    digits = re.sub(r"\D", "", text)
    if not (_MIN_DIGITS <= len(digits) <= _MAX_DIGITS):
        return "", ""
    if text.startswith("+"):
        return f"+{digits}", digits
    if digits.startswith("0"):
        # A local-format number (trunk 0): kept exactly as
        # dialed — inventing a "+" would corrupt it.
        return digits, digits
    if len(digits) >= 10:
        return f"+{digits}", digits
    return digits, digits


def whatsapp_number_from_jid(jid: Any) -> tuple[str, str]:
    """(display, digits) for a WhatsApp JID, or (\"\", \"\").

    The phone-number form (``…@s.whatsapp.net``, or a bare
    number) resolves; an ``@lid`` JID is an anonymous id whose
    number this surface cannot see — it resolves to nothing,
    honestly, and the caller counts the person as numberless
    rather than carding an id as if it were a number.  A
    device suffix (``…:12@s.whatsapp.net``) is stripped.
    """
    text = str(jid or "").strip()
    if not text:
        return "", ""
    user, _, domain = text.partition("@")
    user = user.split(":", 1)[0]
    if domain == "lid":
        return "", ""
    if domain and domain != "s.whatsapp.net":
        return "", ""
    return normalize_number(user)


def _better_name(current: tuple[str, int],
                 candidate: tuple[str, int]) -> tuple[str, int]:
    """The winning (name, rank): lower rank wins; on a tie
    the longer name wins; empty always loses."""
    cur_name, cur_rank = current
    new_name, new_rank = candidate
    if not new_name:
        return current
    if not cur_name or new_rank < cur_rank:
        return candidate
    if new_rank == cur_rank and len(new_name) > len(cur_name):
        return candidate
    return current


def merge_people(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Merge roster rows from every source into people.

    Row shape (all fields optional but id/number): ::

        {"platform_id": str, "name": str, "name_rank": int,
         "username": str, "numbers": [raw, ...],
         "bot": bool, "deleted": bool, "source": str}

    Identity: rows sharing a platform id are one person;
    rows sharing a number are one person (a ledger row keyed
    by number meets the roster row carrying that number).
    Bots and deleted accounts are not people — dropped and
    counted, never carded.

    Returns ``{"people": [...], "dropped_bots": int,
    "dropped_deleted": int}`` where each person is
    ``{"name", "numbers": [display, ...], "sources": [...]}``
    sorted by name (number as the tiebreak).  A person may
    carry NO numbers — the caller counts those as
    number-hidden and leaves them uncarded.
    """
    people: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    by_num: dict[str, dict[str, Any]] = {}
    dropped_bots = 0
    dropped_deleted = 0

    def _new_person() -> dict[str, Any]:
        person = {"name": "", "name_rank": 99, "username": "",
                  "numbers": [], "digits": set(), "sources": []}
        people.append(person)
        return person

    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if row.get("deleted"):
            dropped_deleted += 1
            continue
        if row.get("bot"):
            dropped_bots += 1
            continue
        pid = str(row.get("platform_id") or "").strip()
        numbers: list[tuple[str, str]] = []
        for raw in row.get("numbers") or []:
            disp, digs = normalize_number(raw)
            if digs and (disp, digs) not in numbers:
                numbers.append((disp, digs))
        person = by_id.get(pid) if pid else None
        if person is None:
            for _disp, digs in numbers:
                if digs in by_num:
                    person = by_num[digs]
                    break
        if person is None:
            person = _new_person()
        if pid and pid not in by_id:
            by_id[pid] = person
        for disp, digs in numbers:
            by_num.setdefault(digs, person)
            if digs not in person["digits"]:
                person["digits"].add(digs)
                person["numbers"].append(disp)
        rank = row.get("name_rank")
        rank = int(rank) if isinstance(rank, int) else RANK_DISPLAY
        person["name"], person["name_rank"] = _better_name(
            (str(person["name"]), int(person["name_rank"])),
            (str(row.get("name") or "").strip(), rank))
        if not person["username"]:
            person["username"] = str(
                row.get("username") or "").strip()
        source = str(row.get("source") or "").strip()
        if source and source not in person["sources"]:
            person["sources"].append(source)

    out = []
    for person in people:
        name = person["name"]
        if not name and person["username"]:
            name = person["username"]
        out.append({"name": name,
                    "numbers": list(person["numbers"]),
                    "sources": list(person["sources"])})
    out.sort(key=lambda p: ((p["name"] or "~").lower(),
                            p["numbers"][0] if p["numbers"] else ""))
    return {"people": out, "dropped_bots": dropped_bots,
            "dropped_deleted": dropped_deleted}


def _escape(value: str) -> str:
    """vCard text escaping: backslash first, then the
    structural characters and newlines."""
    return (str(value).replace("\\", "\\\\")
            .replace("\r\n", "\\n").replace("\n", "\\n")
            .replace("\r", "\\n")
            .replace(",", "\\,").replace(";", "\\;"))


def _fold(line: str) -> str:
    """Fold one logical line at 75 octets (RFC 2425): the
    continuation lines start with a single space, and a
    multi-byte UTF-8 character is never split."""
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    chunks: list[bytes] = []
    budget = 75
    while raw:
        cut = min(budget, len(raw))
        while cut < len(raw) and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        chunks.append(raw[:cut])
        raw = raw[cut:]
        budget = 74  # the leading space takes one octet
    return "\r\n ".join(c.decode("utf-8") for c in chunks)


def build_vcard(name: str, numbers: Iterable[str]) -> str:
    """One vCard 3.0 card (CRLF endings), or \"\" when there
    is no number to card.  A nameless person's FN falls back
    to their first number — the card stays importable and
    honest instead of carrying a placeholder name."""
    nums = [n for n in (str(x).strip() for x in numbers or []) if n]
    if not nums:
        return ""
    display = str(name or "").strip() or nums[0]
    parts = display.split()
    if len(parts) > 1:
        family, given = parts[-1], " ".join(parts[:-1])
    else:
        family, given = "", display
    lines = [
        "BEGIN:VCARD",
        "VERSION:3.0",
        f"N:{_escape(family)};{_escape(given)};;;",
        f"FN:{_escape(display)}",
    ]
    for num in nums:
        lines.append(f"TEL;TYPE=CELL:{num}")
    lines.append("END:VCARD")
    return "".join(_fold(line) + "\r\n" for line in lines)


def build_vcf(people: Iterable[dict[str, Any]]) -> str:
    """The whole .vcf file body: one card per person who
    carries at least one number, in the order given."""
    cards = []
    for person in people or []:
        card = build_vcard(str(person.get("name") or ""),
                           person.get("numbers") or [])
        if card:
            cards.append(card)
    return "".join(cards)


def parse_manual_entries(tail: str) -> dict[str, Any]:
    """The manual rung's grammar: one person per line,
    ``Name | number [| number …]``.  Returns
    ``{"ok": True, "entries": [{"name", "numbers"}]}`` or
    ``{"ok": False, "error": <one plain line>}`` — a bad
    line refuses the whole batch in words rather than
    silently dropping the person it couldn't read."""
    entries: list[dict[str, Any]] = []
    for line in str(tail or "").splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            return {"ok": False, "error": (
                f"couldn't read “{line}” — the manual shape "
                f"is Name | number, one person per line.")}
        parts = [p.strip() for p in line.split("|")]
        name = parts[0]
        if not name:
            return {"ok": False, "error": (
                f"“{line}” starts with no name — the shape "
                f"is Name | number.")}
        numbers = []
        for raw in parts[1:]:
            if not raw:
                continue
            disp, _digs = normalize_number(raw)
            if not disp:
                return {"ok": False, "error": (
                    f"“{raw}” in “{line}” doesn't read as a "
                    f"phone number — fix that line and send "
                    f"the batch again; nothing was made.")}
            if disp not in numbers:
                numbers.append(disp)
        if not numbers:
            return {"ok": False, "error": (
                f"“{line}” carries no number — a card needs "
                f"one.")}
        entries.append({"name": name, "numbers": numbers})
    if not entries:
        return {"ok": False, "error": (
            "nothing to make — the shape is "
            "/vcf Name | +number, one person per line.")}
    return {"ok": True, "entries": entries}


def slugify_filename(title: str) -> str:
    """A friendly file stem for a group's contacts file."""
    slug = re.sub(r"[^a-z0-9]+", "-",
                  str(title or "").strip().lower()).strip("-")
    return (slug[:40].rstrip("-") or "group")


def compose_export_line(*, title: str, people: int, cards: int,
                        hidden: int,
                        sources: Iterable[str]) -> str:
    """The coverage line that rides the file as its caption:
    exactly what went in, what was left out and why, and
    which sources the roster came from — counted, never
    rounded."""
    line = (f"📇 {title} — {people} "
            f"{'person' if people == 1 else 'people'} · "
            f"{cards} contact card"
            f"{'' if cards == 1 else 's'} written")
    if hidden:
        line += (f" · {hidden} without a visible number "
                 f"(left out — no number, no card)")
    labels = [str(s) for s in sources or [] if str(s)]
    if labels:
        line += f" · from: {', '.join(labels)}"
    return line
