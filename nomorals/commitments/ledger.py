"""Behavioural fabric / commitment ledger — the obligation layer.

``nomorals/social/relationships.py`` knows *who matters* (closeness,
cadence, warmth). It does not know what you *owe* anyone. This package
is that missing layer: every promise the owner makes, tracked until it
is kept or broken, with the receipts.

Design rules (the whole module stands on these):

* **Never hallucinate.** Commitments are extracted ONLY from explicit
  promise language in supplied message dicts (``"I will …"``,
  ``"I promise …"``, and equivalents in the shipped language packs).
  Hedges (``might``, ``could``, ``I'll try``), questions, third-person
  statements, and past tense produce NOTHING — an empty list, not a
  guess. If the text does not commit, there is no commitment.
* **Append-only audit.** Every change is a ledger entry. Status changes,
  corrections, and reversals append a new entry; nothing is ever edited
  or deleted. Current state is a replay of the entries, and
  :meth:`CommitmentLedger.verify` replays them to prove the snapshot
  matches the log.
* **Reliability is earned, not assumed.** Kept/broken counts come from
  the ledger alone. No resolved commitments → rate is ``None``, never
  0% (0% would claim "you break everything" on zero evidence).
* **Universal.** Language packs cover English, Nigerian Pidgin, French,
  Spanish, Portuguese, Hausa, Yoruba, and Igbo promise verbs and
  weekday/month names. Matching is diacritic-insensitive, so
  ``ránṣẹ́`` and ``ranse`` match the same pattern. Nigeria is a default
  that is never a boundary: packs are data, callers register more via
  :func:`register_language_pack`. Deadlines are epoch seconds; timezone
  handling belongs to the caller (see ``nomorals.pulses.timezone``).

Example — the sentence this module exists to make true::

    >>> ledger = CommitmentLedger()
    >>> added = ledger.extract_and_record([{
    ...     "text": "I will send you the deck by Tuesday",
    ...     "sender": "owner", "recipient": "Chidi",
    ...     "timestamp": 1792000000.0, "id": "m1",
    ... }])
    >>> len(added)
    1
    >>> nudge_text(added[0], now=1792000000.0 + 3 * 86400)
    "⏰ You told Chidi you'd send the deck by Tuesday."

Only the standard library is imported: these packages are new, so the
layering test treats them as layer 0 — importing ``nomorals.core``
here would read as an upward import. The ledger therefore persists to
plain JSON, atomically (tmp file + ``os.replace``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "Commitment",
    "LedgerEntry",
    "CommitmentLedger",
    "extract_commitments",
    "parse_deadline",
    "register_language_pack",
    "LANGUAGE_PACKS",
    "PROMISEE_FIELDS",
    "MESSAGE_TEXT_FIELDS",
    "MESSAGE_TIME_FIELDS",
    "MESSAGE_ID_FIELDS",
]


# ── message-dict field aliases ───────────────────────────────────────
# Callers hand us message dicts from any chat surface; the field names
# vary (Telegram vs WhatsApp vs email vs in-app). Accept them all.

MESSAGE_TEXT_FIELDS = ("text", "body", "content", "message", "caption")
MESSAGE_TIME_FIELDS = ("timestamp", "ts", "created_at", "sent_at", "date")
MESSAGE_ID_FIELDS = ("id", "message_id", "msg_id", "uid", "key")
PROMISEE_FIELDS = ("recipient", "to", "to_name", "counterparty", "peer",
                   "chat_name", "channel")


def _fold(text: str) -> str:
    """Lowercase + strip diacritics, for robust multilingual matching."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return stripped.lower()


# ── language packs ───────────────────────────────────────────────────
# Each pack: explicit promise markers (regexes over folded text),
# deliverable verbs (the thing being promised), weekday names
# (0 = Monday), month names. Packs are DATA — adding a language never
# touches the extraction logic.

_EN_VERBS = (
    "send|deliver|finish|complete|share|call|pay|bring|write|prepare|"
    "submit|fix|book|arrange|attend|meet|join|come|get|do|make|provide|"
    "return|forward|upload|review|approve|sign|ship|order|pick|drop|"
    "transfer|cover|handle|sort|check|look|read|watch|listen|help|support|"
    "teach|show|explain|introduce|recommend|confirm|update|follow|remind|"
    "visit|drive|cook|buy|sell|build|create|design|test|deploy|publish|"
    "record|edit|translate|drive|fly|travel"
)
_EN_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    "mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3,
    "thurs": 3, "fri": 4, "sat": 5, "sun": 6,
}
_EN_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
    "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
    "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


@dataclass
class LanguagePack:
    """Promise language for one language code."""

    code: str
    # Regexes (over folded text) that mark an explicit promise.
    promise_markers: tuple[str, ...]
    # Deliverable verbs — the action being promised.
    verbs: str
    weekdays: dict[str, int]
    months: dict[str, int]


LANGUAGE_PACKS: dict[str, LanguagePack] = {}


def register_language_pack(pack: LanguagePack) -> None:
    """Add (or replace) a language pack. Packs are data, not code."""
    LANGUAGE_PACKS[pack.code] = pack


def _pack(code: str, markers: tuple[str, ...], verbs: str,
          weekdays: dict[str, int], months: dict[str, int]) -> LanguagePack:
    return LanguagePack(code=code, promise_markers=markers, verbs=verbs,
                        weekdays=weekdays, months=months)


register_language_pack(_pack(
    "en",
    (r"\b((i|we)\s+(will|shall)|i'll|we'll)\b",
     r"\b(i|we)\s+(promise[sd]?\s+(?:to|that)?|commit(?:ted|s)?\s+to|"
     r"guarantee[sd]?|swear|vow|owe\s+you|am\s+going\s+to|"
     r"are\s+going\s+to)\b"),
    _EN_VERBS, _EN_WEEKDAYS, _EN_MONTHS))
register_language_pack(_pack(
    "pcm",  # Nigerian Pidgin
    (r"\bi\s+go\b", r"\bi\s+promise\b", r"\bwe\s+go\b",
     r"\bi\s+go\s+make\s+sure\b"),
    _EN_VERBS, _EN_WEEKDAYS, _EN_MONTHS))
register_language_pack(_pack(
    "fr",
    (r"\bje\s+(promets|vais|m'engage)\b",
     r"\bnous\s+(promettons|allons)\b",
     r"\bje\s+vous\s+(enverrai|appellerai|paierai)\b"),
    "envoyer|livrer|finir|terminer|partager|appeler|payer|apporter|"
    "ecrire|preparer|soumettre|reparer|reserver|organiser|venir|rejoindre|"
    "rencontrer|faire|donner|fournir|retourner|transferer|aider|soutenir|"
    "enseigner|montrer|expliquer|presenter|recommander|confirmer|"
    "mettre|suivre|rappeler|visiter|conduire|cuisiner|acheter|vendre|"
    "construire|creer|concevoir|tester|deployer|publier|enregistrer|"
    "editer|traduire",
    {"lundi": 0, "mardi": 1, "mercredi": 2, "jeudi": 3, "vendredi": 4,
     "samedi": 5, "dimanche": 6,
     "lun": 0, "mar": 1, "mer": 2, "jeu": 3, "ven": 4, "sam": 5, "dim": 6},
    {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5,
     "juin": 6, "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10,
     "novembre": 11, "decembre": 12,
     "janv": 1, "fevr": 2, "avr": 4, "juil": 7, "sept": 9, "oct": 10,
     "nov": 11, "dec": 12}))
register_language_pack(_pack(
    "es",
    (r"\bprometo\b", r"\bvoy\s+a\b", r"\bnosotros\s+prometemos\b",
     r"\bte\s+prometo\b"),
    "enviar|entregar|terminar|compartir|llamar|pagar|traer|escribir|"
    "preparar|enviar|arreglar|reservar|organizar|venir|unirse|reunirse|"
    "hacer|dar|proporcionar|devolver|transferir|ayudar|apoyar|ensenar|"
    "mostrar|explicar|presentar|recomendar|confirmar|actualizar|seguir|"
    "recordar|visitar|conducir|cocinar|comprar|vender|construir|crear|"
    "disenar|probar|desplegar|publicar|grabar|editar|traducir",
    {"lunes": 0, "martes": 1, "miercoles": 2, "jueves": 3, "viernes": 4,
     "sabado": 5, "domingo": 6,
     "lun": 0, "mar": 1, "mie": 2, "jue": 3, "vie": 4, "sab": 5, "dom": 6},
    {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5,
     "junio": 6, "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10,
     "noviembre": 11, "diciembre": 12,
     "ene": 1, "feb": 2, "mar": 3, "abr": 4, "jun": 6, "jul": 7,
     "ago": 8, "sept": 9, "oct": 10, "nov": 11, "dic": 12}))
register_language_pack(_pack(
    "pt",
    (r"\bprometo\b", r"\bvou\b", r"\bnos\s+prometemos\b"),
    "enviar|entregar|terminar|compartilhar|ligar|pagar|trazer|escrever|"
    "preparar|consertar|reservar|organizar|vir|juntar|reunir|fazer|dar|"
    "fornecer|devolver|transferir|ajudar|apoiar|ensinar|mostrar|explicar|"
    "apresentar|recomendar|confirmar|atualizar|seguir|lembrar|visitar|"
    "dirigir|cozinhar|comprar|vender|construir|criar|projetar|testar|"
    "publicar|gravar|editar|traduzir",
    {"segunda": 0, "terca": 1, "quarta": 2, "quinta": 3, "sexta": 4,
     "sabado": 5, "domingo": 6,
     "seg": 0, "ter": 1, "qua": 2, "qui": 3, "sex": 4, "sab": 5, "dom": 6},
    {"janeiro": 1, "fevereiro": 2, "marco": 3, "abril": 4, "maio": 5,
     "junho": 6, "julho": 7, "agosto": 8, "setembro": 9, "outubro": 10,
     "novembro": 11, "dezembro": 12,
     "jan": 1, "fev": 2, "mar": 3, "abr": 4, "jun": 6, "jul": 7,
     "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12}))
register_language_pack(_pack(
    "ha",  # Hausa — zan = "I will", na yi alkawari = "I promise"
    (r"\bzan\b", r"\bna\s+yi\s+alkawari\b", r"\bmu\s+yi\s+alkawari\b"),
    r"aika|bayar|kira|rubuta|shirya|gyara|kawo|bada|dawo|taimaka|"
    r"koyar|nuna|bayyana|gabatar|ba\s+shawara|tabbatar|sabunta|bi|"
    r"tunatar|ziyara|tuƙi|dafa|saya|sayar|gina|ƙirƙira|gwada|buga|"
    r"rikodi|fassara|tafiya|zo|halarta|haɗu",
    {"litinin": 0, "talata": 1, "laraba": 2, "alhamis": 3, "juma'a": 4,
     "juma": 4, "asabar": 5, "lahadi": 6},
    {"janairu": 1, "fabrairu": 2, "maris": 3, "afrilu": 4, "mayu": 5,
     "yun": 6, "yuli": 7, "agusta": 8, "satumba": 9, "oktoba": 10,
     "nuwamba": 11, "disamba": 12}))
register_language_pack(_pack(
    "yo",  # Yoruba — mo se ileri = "I promise", maa = future "will"
    (r"\bmo\s+se\s+ileri\b", r"\bmaa\b", r"\ba\s+se\s+ileri\b"),
    r"ranse|san|pe|ko|pese|se\s+atunse|mu|fi|pada|tayọ|ran|lọwọ|"
    r"kọ|fi\s+han|gbekalẹ|ṣe\s+aba|jẹrisi|ṣe\s+imudojuiwọn|tẹle|"
    r"ran\s+letı|bẹwo|wakọ|se|ra|ta|kọ|ṣe|danwo|tẹjade|gba\s+silẹ|"
    r"tumọ|rin\s+ajo|wa|wa\s+nibe|pade",
    {r"ojo\s+aje": 0, r"ojo\s+isegun": 1, "ojoru": 2, "ojobo": 3,
     r"ojo\s+eti": 4, r"ojo\s+abameta": 5, r"ojo\s+aiku": 6},
    {r"osu\s+kinni": 1, r"osu\s+keji": 2, r"osu\s+keta": 3,
     r"osu\s+kerin": 4, r"osu\s+karun": 5, r"osu\s+kefa": 6,
     r"osu\s+keje": 7, r"osu\s+kejo": 8, r"osu\s+kesan": 9,
     r"osu\s+kewa": 10, r"osu\s+kokanla": 11, r"osu\s+kejila": 12}))
register_language_pack(_pack(
    "ig",  # Igbo — m ga- = "I will", ekwere m = "I promise"
    (r"\bm\s+ga\b", r"\bekwere\s+m\b", r"\banyi\s+ga\b"),
    r"ziga|kwụọ|kpọọ|dee|kwadebe|mezie|weta|nye|laghachi|nyere|"
    r"kuzi|gosi|kọwaa|webata|nye\s+ndụmọdụ|gosiputa|melite|so|"
    r"chetara|leta|kwọ|sie|zụta|re|wu|mepụta|nwale|bipụta|dekọọ|"
    r"tụgharịa|gaa|bia|bịa|zute",
    {"monde": 0, "tusde": 1, "wensde": 2, "tosde": 3, "fraide": 4,
     "satode": 5, "sonde": 6},
    {"januarị": 1, "febrụwarị": 2, "maachị": 3, "eprel": 4, "mee": 5,
     "juun": 6, "julaị": 7, "ọgọst": 8, "septemba": 9, "ọktoba": 10,
     "novemba": 11, "disemba": 12}))


# ── hedges: language that is NOT a commitment ─────────────────────────
# If any of these appear, the sentence is a maybe, a question, a wish,
# or a conditional — never a promise. Checked on folded text.

_HEDGES = (
    "maybe", "might", "could", "possibly", "probably", "hopefully",
    "i'll try", "i will try", "ill try", "i will attempt", "i'll attempt",
    "i should", "i should probably", "i wish", "i'd like to",
    "if i have time", "if i can", "if i get time", "if i get a chance",
    "i'll see", "i will see", "i'll check", "i will check", "i might", "i could",
    "i'll think about", "i will think about", "i'll consider",
    "try to", "attempt to", "aim to", "plan to", "intend to",
    "god willing", "insha allah", "in sha allah", "by god's grace",
    "if god wills", "hopefully", "fingers crossed",
    "je vais essayer", "j'essaierai", "peut-etre", "peut etre",
    "voy a intentar", "quizas", "tal vez", "vou tentar", "talvez",
)
_QUESTION_STARTS = (
    "will you", "would you", "can you", "could you", "do you",
    "did you", "are you", "is it", "should i", "shall i",
)


# Relative-day words (folded) across the packed languages → day offset.
_RELATIVE_DAYS = {
    "tomorrow": 1, "demain": 1, "manana": 1, "amanha": 1, "gobe": 1,
    "ola": 1, "echi": 1,
    "today": 0, "tonight": 0, "aujourd'hui": 0, "hoy": 0, "hoje": 0,
    "yau": 0, "loni": 0, "taa": 0,
}


def _is_hedged(folded: str) -> bool:
    return any(h in folded for h in _HEDGES)


def _is_question(text: str, folded: str) -> bool:
    stripped = (text or "").strip()
    if not stripped.endswith("?"):
        return False
    return (any(folded.startswith(q) for q in _QUESTION_STARTS)
            or folded.startswith(("will ", "would ", "can ", "could ",
                                  "do ", "did ", "are ", "should ")))


# ── deadline parsing ─────────────────────────────────────────────────
# Parses deadline expressions from folded text. Returns epoch seconds
# (UTC-naive, same clock as ref_ts) or None. Supports: tomorrow/today,
# "next/this <weekday>", bare weekday, "in N days/weeks", "on 20 Oct",
# "20 Oct 2026", "Oct 20", ISO dates, "by 5pm / 17:00 / noon / midnight".

_TIME_OF_DAY = {
    "midnight": (0, 0), "noon": (12, 0), "morning": (9, 0),
    "afternoon": (15, 0), "evening": (18, 0), "night": (21, 0),
    "tonight": (21, 0),
}


def _epoch(year: int, month: int, day: int, hour: int = 17,
           minute: int = 0) -> float:
    import datetime as _dt
    return _dt.datetime(year, month, day, hour, minute,
                        tzinfo=_dt.timezone.utc).timestamp()


def _all_weekdays() -> dict[str, int]:
    merged: dict[str, int] = {}
    for pack in LANGUAGE_PACKS.values():
        merged.update(pack.weekdays)
    return merged


def _all_months() -> dict[str, int]:
    merged: dict[str, int] = {}
    for pack in LANGUAGE_PACKS.values():
        merged.update(pack.months)
    return merged


def _parse_time_of_day(expr: str) -> tuple[int, int] | None:
    """Parse a bare time-of-day expression (``5pm``, ``17:00``, ``noon``).

    Returns ``(hour, minute)`` or ``None``. Used when a deadline
    clause contains nothing but a time — the date is the reference
    day (rolled forward a day when that time has already passed).
    """
    text = (expr or "").strip().strip(".,;")
    if text in _TIME_OF_DAY:
        return _TIME_OF_DAY[text]
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    suffix = m.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def parse_deadline(text: str, ref_ts: float | None = None) -> float | None:
    """Parse a deadline expression out of ``text`` (folded internally).

    ``ref_ts`` is the clock the expression is relative to (the message
    timestamp). Returns epoch seconds or ``None`` when no deadline is
    stated. Never guesses: an unrecognised expression is ``None``.

    Discipline: a bare weekday / absolute date counts as a deadline
    only when it is introduced by ``by`` / ``on`` / ``before`` or sits
    at the end of the clause (``"I'll send it Tuesday"``). A weekday
    buried mid-clause (``"the Tuesday report"``) is a noun, not a
    deadline, and yields ``None``.
    """
    import datetime as _dt
    if not text:
        return None
    folded = _fold(text)
    ref = _dt.datetime.fromtimestamp(
        ref_ts if ref_ts is not None else time.time(),
        tz=_dt.timezone.utc)
    hour, minute = 17, 0  # default: end of the working day
    # Time-of-day overrides (checked first, applies to any date found).
    # The digits must be a clock time, not a day-of-month: "by 20 Oct"
    # is a date (at the 17:00 default), not 20:00 — refuse the override
    # when a month name follows the number.
    _month_alt = "|".join(
        re.escape(n) for n in sorted(_all_months(), key=len, reverse=True))
    tod = re.search(
        r"\bby\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b"
        rf"(?!\s*(?:{_month_alt})\b)", folded)
    if tod:
        h = int(tod.group(1))
        minute = int(tod.group(2) or 0)
        suffix = tod.group(3)
        if suffix == "pm" and h < 12:
            h += 12
        elif suffix == "am" and h == 12:
            h = 0
        hour = h
    else:
        for name, (h, m) in _TIME_OF_DAY.items():
            if re.search(rf"\bby\s+{name}\b", folded):
                hour, minute = h, m
                break

    # An explicit by/on/before clause wins: parse its remainder alone,
    # so "the Tuesday report by Friday" resolves to Friday, not Tuesday.
    clause = re.search(r"\b(?:by|on|before)\s+(.+)$", folded)
    if clause:
        inner = _parse_date_expr(clause.group(1), ref, hour, minute,
                                 bare_allowed=True)
        if inner is not None:
            return inner
        # A clause that is ONLY a time of day ("by 5pm", "before
        # noon") is a deadline on the reference day at that time —
        # rolled to the next day when that time has already passed
        # ("by midnight" said at 09:00 means the coming midnight,
        # never 00:00 today, which is already gone).
        pure_time = _parse_time_of_day(clause.group(1))
        if pure_time is not None:
            ts = _epoch(ref.year, ref.month, ref.day,
                        pure_time[0], pure_time[1])
            if ts <= ref.timestamp():
                nxt = ref + _dt.timedelta(days=1)
                ts = _epoch(nxt.year, nxt.month, nxt.day,
                            pure_time[0], pure_time[1])
            return ts
        # A by/on/before with no parseable date is still not a licence
        # to fall through and grab a weekday from the noun phrase.
        return None
    return _parse_date_expr(folded, ref, hour, minute, bare_allowed=False)


def _parse_date_expr(expr: str, ref: Any, hour: int, minute: int,
                     *, bare_allowed: bool) -> float | None:
    """Parse one date expression. ``bare_allowed`` gates bare weekday /
    absolute dates (only valid inside an explicit deadline clause or
    at the end of the text — the caller decides)."""
    import datetime as _dt
    folded = _fold(expr)
    # ISO date: 2026-10-20 — unambiguous anywhere.
    iso = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", folded)
    if iso:
        try:
            return _epoch(int(iso.group(1)), int(iso.group(2)),
                          int(iso.group(3)), hour, minute)
        except ValueError:
            return None
    # "in N days/weeks" — unambiguous anywhere.
    rel = re.search(r"\bin\s+(\d+)\s+(day|days|week|weeks)\b", folded)
    if rel:
        n = int(rel.group(1))
        delta = _dt.timedelta(days=n if "day" in rel.group(2) else n * 7)
        target = ref + delta
        return _epoch(target.year, target.month, target.day, hour, minute)
    # "tomorrow" / "today" / "tonight" in the packed languages —
    # unambiguous anywhere in the text.
    for word, days in _RELATIVE_DAYS.items():
        if re.search(rf"\b{re.escape(word)}\b", folded):
            target = ref + _dt.timedelta(days=days)
            return _epoch(target.year, target.month, target.day, hour, minute)
    # Bare weekday / absolute date: only inside an explicit clause or
    # at the very end of the text ("I'll send it Tuesday"), in any
    # packed language.
    _wd_names = sorted(_all_weekdays(), key=len, reverse=True)
    _wd_re = "|".join(re.escape(n) for n in _wd_names)
    at_end = bool(re.search(
        rf"({_wd_re}|\d{{1,2}}\s+\w+|\w+\s+\d{{1,2}})\s*$", folded))
    if not (bare_allowed or at_end):
        return None
    weekdays = _all_weekdays()
    if weekdays:
        names = sorted(weekdays, key=len, reverse=True)
        wd_re = r"\b(next|this)?\s*(" + "|".join(re.escape(n) for n in names) + r")\b"
        m = re.search(wd_re, folded)
        if m:
            qualifier, name = m.group(1), m.group(2)
            target_wd = weekdays[name]
            ahead = (target_wd - ref.weekday()) % 7
            if qualifier == "next":
                # "next Tuesday" = the Tuesday of next week: this
                # week's Tuesday + 7, or 7 days out if today is Tuesday.
                ahead = ahead + 7 if ahead else 7
            target = ref + _dt.timedelta(days=ahead)
            return _epoch(target.year, target.month, target.day, hour, minute)
    months = _all_months()
    if months:
        month_names = sorted(months, key=len, reverse=True)
        month_re = "|".join(re.escape(n) for n in month_names)
        m1 = re.search(rf"\b(\d{{1,2}})\s+({month_re})(?:\s+(\d{{4}}))?\b", folded)
        m2 = None
        if not m1:
            m2 = re.search(rf"\b({month_re})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b", folded)
        if m1:
            day, month, year = int(m1.group(1)), months[m1.group(2)], \
                int(m1.group(3)) if m1.group(3) else ref.year
            try:
                ts = _epoch(year, month, day, hour, minute)
            except ValueError:
                return None
            if not m1.group(3) and ts < ref.timestamp():
                ts = _epoch(year + 1, month, day, hour, minute)
            return ts
        if m2:
            month, day = months[m2.group(1)], int(m2.group(2))
            year = int(m2.group(3)) if m2.group(3) else ref.year
            try:
                ts = _epoch(year, month, day, hour, minute)
            except ValueError:
                return None
            if not m2.group(3) and ts < ref.timestamp():
                ts = _epoch(year + 1, month, day, hour, minute)
            return ts
    return None


# ── extraction ───────────────────────────────────────────────────────

def _message_get(msg: dict[str, Any], fields: tuple[str, ...]) -> Any:
    for f in fields:
        if f in msg and msg[f] not in (None, ""):
            return msg[f]
    return None


def _promisee_from_text(original: str) -> str | None:
    """``to <Name>`` / ``for <Name>`` with a capitalised name in the text."""
    m = re.search(r"\b(?:to|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
                  original or "")
    if m:
        name = m.group(1)
        # "to Tuesday" / "to send" are not people — guard the common ones.
        if name.lower() not in {"tuesday", "monday", "wednesday",
                                "thursday", "friday", "saturday", "sunday",
                                "tomorrow", "today"}:
            return name
    return None


def extract_commitments(message: dict[str, Any]) -> list[Commitment]:
    """Extract explicit commitments from ONE supplied message dict.

    Pure function: no state, no I/O. Returns ``[]`` unless the text
    contains an explicit promise + a deliverable verb. Never invents a
    promisee, a deliverable, or a deadline that is not in the text.
    """
    text = _message_get(message, MESSAGE_TEXT_FIELDS)
    if not isinstance(text, str) or not text.strip():
        return []
    folded = _fold(text)
    if _is_hedged(folded) or _is_question(text, folded):
        return []
    ts_raw = _message_get(message, MESSAGE_TIME_FIELDS)
    try:
        ref_ts = float(ts_raw) if ts_raw is not None else time.time()
    except (TypeError, ValueError):
        ref_ts = time.time()
    sender = _message_get(message, ("sender", "from", "author", "promisor"))
    promisee = (
        _message_get(message, PROMISEE_FIELDS)
        or _promisee_from_text(text)
    )
    if isinstance(promisee, str):
        promisee = promisee.strip() or None
    source_id = _message_get(message, MESSAGE_ID_FIELDS)
    source_id = str(source_id) if source_id is not None else None

    found: list[Commitment] = []
    for pack in LANGUAGE_PACKS.values():
        for marker_re in pack.promise_markers:
            for marker in re.finditer(marker_re, folded):
                # The deliverable verb must follow the marker closely
                # (within the next ~8 words) — "I will think about it"
                # has no deliverable verb, so it yields nothing.
                tail = folded[marker.end():marker.end() + 90]
                is_owe = "owe" in marker.group(0)
                if is_owe:
                    # "I owe you $50 by Friday" — the deliverable is the
                    # remainder itself; there is no action verb.
                    raw_what = tail.strip()
                    raw_what = re.split(r"\s+but\s+|;|,?\s+and then\s+",
                                        raw_what)[0]
                    deadline = parse_deadline(folded[marker.end():], ref_ts)
                    if deadline is not None:
                        for pat in (r"\s*\bby\s+[^,;]+$",
                                    r"\s*\bon\s+[^,;]+$",
                                    r"\s*\bbefore\s+[^,;]+$",
                                    r"\s*\b(tomorrow|today|tonight|demain|manana|"
                                    r"amanha|gobe|ola|echi|aujourd'hui|hoy|hoje|"
                                    r"yau|loni|taa)\s*$"):
                            raw_what = re.sub(pat, "", raw_what)
                    raw_what = raw_what.strip(" .,;:!?'\"")
                    # "I owe you $50" — the marker consumed "you", so
                    # the tail is just the thing owed: phrase it as an
                    # action so nudges read naturally.
                    if raw_what and not raw_what.startswith("give "):
                        raw_what = "give " + raw_what
                    if raw_what and len(raw_what) >= 2:
                        digest_src = (f"{source_id or ''}|{pack.code}|"
                                      f"{raw_what}|{deadline}")
                        cid = "cm-" + hashlib.sha1(
                            digest_src.encode("utf-8")).hexdigest()[:12]
                        found.append(Commitment(
                            id=cid, promisor=str(sender or "owner"),
                            promisee=promisee, what=raw_what,
                            deadline=deadline, source_text=text.strip(),
                            source_message_id=source_id,
                            language=pack.code, created_at=ref_ts))
                    break
                verb = re.search(
                    rf"^\s*(?:to\s+|that\s+|de\s+|t['’]|you\s+|the\s+|a\s+|an\s+|"
                    rf"your\s+|my\s+)*({pack.verbs})\b", tail)
                if not verb:
                    continue
                # "what": from the verb to the end of the clause, minus
                # the deadline clause and trailing punctuation.
                what_start = marker.end() + verb.start(1)
                # Rebuild from the FOLDED text for a clean, uniform form,
                # then title-case nothing — keep it verbatim-ish.
                raw_what = folded[what_start:].strip()
                # Cut at clause boundaries: " and then ", ";", " but ".
                raw_what = re.split(r"\s+but\s+|;|,?\s+and then\s+", raw_what)[0]
                # Strip the deadline clause from "what".
                deadline = parse_deadline(folded[marker.end():], ref_ts)
                if deadline is not None:
                    # Remove the deadline expression text from what.
                    for pat in (r"\s*\bby\s+[^,;]+$",
                                r"\s*\bon\s+[^,;]+$",
                                r"\s*\bbefore\s+[^,;]+$",
                                r"\s*\bin\s+\d+\s+(?:day|days|week|weeks)\s*$",
                                r"\s*\bnext\s+\w+\s*$",
                                r"\s*\bthis\s+\w+\s*$",
                                r"\s*\b(tomorrow|today|tonight|demain|manana|"
                                r"amanha|gobe|ola|echi|aujourd'hui|hoy|hoje|"
                                r"yau|loni|taa)\s*$"):
                        raw_what = re.sub(pat, "", raw_what)
                raw_what = raw_what.strip(" .,;:!?'\"")
                if not raw_what or len(raw_what) < 2:
                    continue
                digest_src = f"{source_id or ''}|{pack.code}|{raw_what}|{deadline}"
                cid = "cm-" + hashlib.sha1(
                    digest_src.encode("utf-8")).hexdigest()[:12]
                found.append(Commitment(
                    id=cid,
                    promisor=str(sender or "owner"),
                    promisee=promisee,
                    what=raw_what,
                    deadline=deadline,
                    source_text=text.strip(),
                    source_message_id=source_id,
                    language=pack.code,
                    created_at=ref_ts,
                ))
                break  # one commitment per marker position
            if found and pack.code == "en":
                break
        if found:
            break
    # Dedupe: two marker spellings can describe the same promise.
    seen: set[str] = set()
    unique: list[Commitment] = []
    for commitment in found:
        if commitment.id not in seen:
            seen.add(commitment.id)
            unique.append(commitment)
    return unique


# ── records ──────────────────────────────────────────────────────────

@dataclass
class Commitment:
    """One promise: who owes whom what, by when."""

    id: str
    promisor: str = "owner"
    promisee: str | None = None
    what: str = ""
    deadline: float | None = None
    source_text: str = ""
    source_message_id: str | None = None
    language: str = "en"
    status: str = "open"  # open | kept | broken | cancelled | superseded
    created_at: float = field(default_factory=time.time)
    closed_at: float | None = None
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "promisor": self.promisor,
            "promisee": self.promisee, "what": self.what,
            "deadline": self.deadline, "source_text": self.source_text,
            "source_message_id": self.source_message_id,
            "language": self.language, "status": self.status,
            "created_at": self.created_at, "closed_at": self.closed_at,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> Commitment:
        return cls(
            id=str(row.get("id", "")),
            promisor=str(row.get("promisor", "owner")),
            promisee=row.get("promisee"),
            what=str(row.get("what", "")),
            deadline=row.get("deadline"),
            source_text=str(row.get("source_text", "")),
            source_message_id=row.get("source_message_id"),
            language=str(row.get("language", "en")),
            status=str(row.get("status", "open")),
            created_at=float(row.get("created_at") or 0.0),
            closed_at=row.get("closed_at"),
            evidence=str(row.get("evidence", "")),
        )

    def is_overdue(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return (self.status == "open" and self.deadline is not None
                and self.deadline < now)


@dataclass
class LedgerEntry:
    """One append-only event in the audit log."""

    seq: int
    ts: float
    kind: str  # created | kept | broken | cancelled | superseded |
    # reminded | note | correction
    commitment_id: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"seq": self.seq, "ts": self.ts, "kind": self.kind,
                "commitment_id": self.commitment_id, "detail": self.detail,
                "payload": self.payload}

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> LedgerEntry:
        return cls(seq=int(row.get("seq", 0)),
                   ts=float(row.get("ts") or 0.0),
                   kind=str(row.get("kind", "")),
                   commitment_id=str(row.get("commitment_id", "")),
                   detail=str(row.get("detail", "")),
                   payload=dict(row.get("payload") or {}))


_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"]
_MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul",
                "Aug", "Sep", "Oct", "Nov", "Dec"]


def format_deadline(deadline: float | None,
                    now: float | None = None) -> str:
    """Human deadline: ``Tuesday`` within a week, else ``Tue, 20 Oct``."""
    import datetime as _dt
    if deadline is None:
        return "no deadline set"
    now_dt = _dt.datetime.fromtimestamp(
        now if now is not None else time.time(), tz=_dt.timezone.utc)
    dt = _dt.datetime.fromtimestamp(deadline, tz=_dt.timezone.utc)
    if 0 <= (dt.date() - now_dt.date()).days <= 7:
        return _WEEKDAY_NAMES[dt.weekday()]
    return f"{_WEEKDAY_NAMES[dt.weekday()]}, {dt.day} {_MONTH_NAMES[dt.month]}"


def _render_what(what: str) -> str:
    """Drop the ``you`` object from the deliverable for nudge phrasing:
    ``send you the deck`` → ``send the deck``; ``call you`` → ``call``."""
    w = (what or "").strip()
    w = re.sub(r"\s+you\s+(the|a|an|my|your)\s+", r" \1 ", w)
    w = re.sub(r"\s+you\s*$", "", w)
    w = re.sub(r"^(you|to you|for you)\s+", "", w)
    return w or what


class CommitmentLedger:
    """The obligation layer: append-only, replay-derived, auditable.

    State is ALWAYS derived by replaying :attr:`entries` — the snapshot
    in :attr:`commitments` is a cache, and :meth:`verify` proves the
    cache matches the log. Persistence (when ``path`` is given) stores
    the entries only; loading replays them.
    """

    def __init__(self, path: str | None = None) -> None:
        self.path = path
        self.entries: list[LedgerEntry] = []
        self.commitments: dict[str, Commitment] = {}
        if path:
            self._load()

    # ── persistence ──────────────────────────────────────────────
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return
        for row in raw.get("entries") or []:
            try:
                self.entries.append(LedgerEntry.from_dict(row))
            except (TypeError, ValueError):
                continue
        self._replay()

    def _save(self) -> None:
        if not self.path:
            return
        try:
            parent = os.path.dirname(self.path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"entries": [e.to_dict() for e in self.entries]},
                          fh, ensure_ascii=False)
            os.replace(tmp, self.path)
        except OSError:
            pass  # fail-open: the in-memory ledger is still the truth

    # ── append-only core ─────────────────────────────────────────
    def _append(self, kind: str, commitment_id: str, detail: str = "",
                payload: dict[str, Any] | None = None,
                ts: float | None = None) -> LedgerEntry:
        entry = LedgerEntry(
            seq=len(self.entries) + 1,
            ts=float(ts if ts is not None else time.time()),
            kind=kind, commitment_id=commitment_id, detail=detail,
            payload=dict(payload or {}))
        self.entries.append(entry)
        self._replay()
        self._save()
        return entry

    def _replay(self) -> None:
        """Rebuild current state from the entry log. The log is truth."""
        state: dict[str, Commitment] = {}
        for entry in sorted(self.entries, key=lambda e: e.seq):
            cid = entry.commitment_id
            if entry.kind == "created":
                state[cid] = Commitment.from_dict(
                    {"id": cid, **entry.payload})
            elif cid in state:
                current = state[cid]
                if entry.kind in ("kept", "broken", "cancelled",
                                  "superseded"):
                    current.status = entry.kind
                    current.closed_at = entry.ts
                    current.evidence = entry.detail
                elif entry.kind == "correction":
                    # A correction never rewrites history: it annotates
                    # the live record with the corrected fields while
                    # the original entries stay intact.
                    for key, value in (entry.payload.get("fields") or {}).items():
                        if hasattr(current, key):
                            setattr(current, key, value)
        self.commitments = state

    def verify(self) -> dict[str, Any]:
        """Prove the snapshot matches a fresh replay of the log."""
        saved = dict(self.commitments)
        self._replay()
        seqs = [e.seq for e in self.entries]
        contiguous = seqs == list(range(1, len(seqs) + 1))
        matches = (
            {k: v.to_dict() for k, v in saved.items()}
            == {k: v.to_dict() for k, v in self.commitments.items()})
        return {"entries": len(self.entries), "commitments": len(saved),
                "seq_contiguous": contiguous, "snapshot_matches_replay": matches,
                "ok": contiguous and matches}

    # ── intake ───────────────────────────────────────────────────
    def extract_and_record(
            self, messages: list[dict[str, Any]]) -> list[Commitment]:
        """Extract explicit commitments from messages and append them.

        Idempotent: re-feeding the same message (same deterministic id)
        does not duplicate the commitment.
        """
        added: list[Commitment] = []
        for msg in messages or []:
            for commitment in extract_commitments(msg):
                if commitment.id in self.commitments:
                    continue
                self._append("created", commitment.id,
                             f"promise extracted: {commitment.what}",
                             payload=commitment.to_dict(),
                             ts=commitment.created_at)
                added.append(self.commitments[commitment.id])
        return added

    def add(self, *, promisor: str = "owner", promisee: str | None = None,
            what: str, deadline: float | None = None,
            source_text: str = "", source_message_id: str | None = None,
            language: str = "en",
            ts: float | None = None) -> Commitment:
        """Record an explicitly stated commitment (manual intake)."""
        now = float(ts if ts is not None else time.time())
        cid = "cm-" + uuid.uuid4().hex[:12]
        commitment = Commitment(id=cid, promisor=promisor, promisee=promisee,
                                what=what, deadline=deadline,
                                source_text=source_text,
                                source_message_id=source_message_id,
                                language=language, created_at=now)
        self._append("created", cid, f"promise recorded: {what}",
                     payload=commitment.to_dict(), ts=now)
        return self.commitments[cid]

    # ── lifecycle ────────────────────────────────────────────────
    def _close(self, commitment_id: str, kind: str, detail: str,
               ts: float | None) -> Commitment:
        current = self.commitments.get(commitment_id)
        if current is None:
            raise KeyError(f"unknown commitment {commitment_id!r}")
        if current.status != "open":
            raise ValueError(
                f"commitment {commitment_id!r} already {current.status}")
        self._append(kind, commitment_id, detail, ts=ts)
        return self.commitments[commitment_id]

    def mark_kept(self, commitment_id: str, evidence: str = "",
                  ts: float | None = None) -> Commitment:
        """The promise was kept. Evidence is a note, not a rewrite."""
        return self._close(commitment_id, "kept", evidence, ts)

    def mark_broken(self, commitment_id: str, evidence: str = "",
                    ts: float | None = None) -> Commitment:
        """The promise was broken. Recorded, not erased."""
        return self._close(commitment_id, "broken", evidence, ts)

    def cancel(self, commitment_id: str, reason: str = "",
               ts: float | None = None) -> Commitment:
        """Both sides agreed it is off — cancelled, never deleted."""
        return self._close(commitment_id, "cancelled", reason, ts)

    def supersede(self, old_id: str, *, what: str,
                  deadline: float | None = None,
                  promisee: str | None = None,
                  reason: str = "",
                  ts: float | None = None) -> Commitment:
        """Replace a promise with a new one (renegotiated deadline etc.).

        The old commitment is marked superseded (kept in the log); the
        new one links back via ``supersedes`` in its payload.
        """
        old = self.commitments.get(old_id)
        if old is None:
            raise KeyError(f"unknown commitment {old_id!r}")
        if old.status != "open":
            raise ValueError(f"commitment {old_id!r} already {old.status}")
        new = self.add(promisor=old.promisor,
                       promisee=promisee if promisee is not None else old.promisee,
                       what=what, deadline=deadline,
                       source_text=old.source_text,
                       source_message_id=old.source_message_id,
                       language=old.language, ts=ts)
        self._append("superseded", old_id,
                     reason or f"superseded by {new.id}",
                     payload={"superseded_by": new.id}, ts=ts)
        return new

    def correct(self, commitment_id: str, *, fields: dict[str, Any],
                reason: str = "") -> Commitment:
        """Correct a live record (e.g. wrong deadline was extracted).

        Appends a ``correction`` entry — the original ``created`` entry
        is never edited, so the audit trail shows both versions.
        """
        if commitment_id not in self.commitments:
            raise KeyError(f"unknown commitment {commitment_id!r}")
        allowed = {"promisee", "what", "deadline", "evidence"}
        clean = {k: v for k, v in fields.items() if k in allowed}
        if not clean:
            raise ValueError("nothing correctable in fields")
        self._append("correction", commitment_id,
                     reason or "record corrected", payload={"fields": clean})
        return self.commitments[commitment_id]

    def note(self, commitment_id: str, text: str) -> LedgerEntry:
        if commitment_id not in self.commitments:
            raise KeyError(f"unknown commitment {commitment_id!r}")
        return self._append("note", commitment_id, text)

    # ── queries ──────────────────────────────────────────────────
    def open_commitments(self, promisee: str | None = None) -> list[Commitment]:
        rows = [c for c in self.commitments.values() if c.status == "open"]
        if promisee is not None:
            rows = [c for c in rows if c.promisee == promisee]
        return sorted(rows, key=lambda c: (c.deadline is None,
                                           c.deadline or 0.0))

    def overdue(self, now: float | None = None) -> list[Commitment]:
        now = time.time() if now is None else now
        return [c for c in self.open_commitments() if c.is_overdue(now)]

    def due_soon(self, within_seconds: float = 86400,
                 now: float | None = None) -> list[Commitment]:
        """Open commitments due within the window (overdue included)."""
        now = time.time() if now is None else now
        return [c for c in self.open_commitments()
                if c.deadline is not None and c.deadline <= now + within_seconds]

    def nudges(self, within_seconds: float = 86400,
               now: float | None = None) -> list[tuple[Commitment, str]]:
        """(commitment, nudge text) for everything due soon / overdue.

        Nudging appends a ``reminded`` entry per commitment, so the log
        shows every time the owner was warned — no silent nagging.
        """
        now = time.time() if now is None else now
        out: list[tuple[Commitment, str]] = []
        for commitment in self.due_soon(within_seconds, now):
            out.append((commitment, nudge_text(commitment, now)))
            self._append("reminded", commitment.id,
                         f"nudged: due {format_deadline(commitment.deadline, now)}",
                         ts=now)
        return out

    # ── reliability ──────────────────────────────────────────────
    def reliability(self, promisee: str | None = None) -> dict[str, Any]:
        """Kept/broken counts and rate. No resolved data → rate ``None``."""
        rows = [c for c in self.commitments.values()
                if (promisee is None or c.promisee == promisee)]
        kept = sum(1 for c in rows if c.status == "kept")
        broken = sum(1 for c in rows if c.status == "broken")
        resolved = kept + broken
        return {
            "promisee": promisee,
            "kept": kept, "broken": broken, "resolved": resolved,
            "open": sum(1 for c in rows if c.status == "open"),
            "overdue": sum(1 for c in rows if c.is_overdue()),
            "rate": (kept / resolved) if resolved else None,
        }

    def reliability_by_person(self) -> dict[str, dict[str, Any]]:
        people = sorted({c.promisee for c in self.commitments.values()
                         if c.promisee})
        return {person: self.reliability(person) for person in people}

    def reliability_sentence(self, promisee: str | None = None) -> str | None:
        """Plain words: ``You've kept 4 of 5 promises to Chidi (80%).``"""
        stats = self.reliability(promisee)
        if stats["resolved"] == 0:
            return None
        who = f" to {promisee}" if promisee else ""
        return (f"You've kept {stats['kept']} of {stats['resolved']} "
                f"promises{who} ({stats['rate']:.0%}).")

    def audit_log(self, commitment_id: str | None = None) -> list[LedgerEntry]:
        if commitment_id is None:
            return list(self.entries)
        return [e for e in self.entries if e.commitment_id == commitment_id]


def nudge_text(commitment: Commitment, now: float | None = None) -> str:
    """The nudge: ``You told Chidi you'd send the deck Tuesday.``"""
    now = time.time() if now is None else now
    who = f" {commitment.promisee}" if commitment.promisee else ""
    what = _render_what(commitment.what)
    if commitment.deadline is None:
        return f"⏰ You told{who} you'd {what} (no deadline set)."
    due = format_deadline(commitment.deadline, now)
    if commitment.is_overdue(now):
        days = max(1, int((now - commitment.deadline) // 86400))
        late = f"{days} day{'s' if days != 1 else ''} late"
        return (f"⚠️ Overdue: you told{who} you'd {what} by {due} "
                f"— {late}.")
    return f"⏰ You told{who} you'd {what} by {due}."


# Callback type for callers that want extraction wired into a pipeline:
# ``on_commitment(commitment)`` fires for every newly recorded promise.
OnCommitment = Callable[[Commitment], None]
