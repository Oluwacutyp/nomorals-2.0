"""Voice money commands — Yoruba (Ekiti/Ilawe Ekiti flagship), Pidgin,
Hausa, Igbo, English. Code-switching is the norm, not the edge case.

Inbound: voice-note transcript → :func:`parse_voice_money` → ``MoneyIntent``
→ :func:`handle_voice_money` (log expense / check balance / stage transfer).
Outbound replies are short templates in the detected language.

Money NEVER moves on voice alone. A transfer intent is staged in the
finance dir and requires :func:`nomorals.core.policy.approve_with_biometric`
— the owner's fingerprint — before anything proceeds. The staged record
(with its biometric token) is the handoff point for the connector
human-checkpoint flow
(:func:`nomorals.connectors._confirm.confirm_or_checkpoint`); no transfer
function exists in the connectors yet, so nothing executes past staging.
Parsing is offline and rule-based (no LLM required).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..core.policy import Policy, approve_with_biometric
from ..finance.budgets import finance_paths
from ..finance.ledger import (
    CATEGORY_KEYWORDS,
    Ledger,
    categorize,
    format_naira,
    parse_amount,
)

_log = get_logger("nomorals.voice.money")

__all__ = [
    "DIALECT_NOTES",
    "MoneyIntent",
    "StagedTransfer",
    "TransferStaging",
    "detect_language",
    "parse_voice_money",
    "transcribe_money_voice",
    "handle_voice_money",
    "voice_money_precheck",
]

#: Honest documentation of what the dialect patterns below actually cover.
#: Ekiti is an Eastern Yoruba dialect; the user (Ilawe Ekiti) speaks it as
#: the flagship. The single most-cited Ekiti marker is the first-person
#: subject pronoun "mi" where Standard Yoruba uses "mo"
#: (Ekiti "mi lọ" vs Standard "mo lọ" = "I go/went"). Note the possessive
#: "mi" ("owo mi" = "my money") exists in Standard Yoruba too, so Ekiti
#: detection keys on SUBJECT-position "mi" + verb, never bare "mi".
#: Financial vocabulary (owo/money, fún/give, ránṣẹ/send, ná/spend,
#: elo/how-much, jọwọ/please) is largely shared between Ekiti and
#: Standard Yoruba — the patterns reuse it for both and let the mi/mo
#: distinction pick the dialect label. LIMITS: the author is not a native
#: Ekiti speaker; these patterns cover the verifiable core (the mi/mo
#: distinction + shared Yoruba money lexicon) and deliberately do NOT
#: invent Ekiti-specific vocabulary. Full dialect coverage needs
#: native-speaker validation — ambiguous Yoruba leans Ekiti (flagship
#: bias, documented here). Code-switched speech is labelled "mixed".
DIALECT_NOTES = __doc__

# ── language / dialect marker banks ──────────────────────────────────────────
# Each entry: (phrase, weight). Matching is case-insensitive substring on the
# lowercased transcript. Weights keep distinctive markers (abeg, jare, aika)
# above shared ones (send, money).

# Ekiti dialect: subject-position "mi" + verb. Keep tight — bare "mi" is the
# Standard possessive too and must NOT count.
_EKITI_MARKERS: tuple[tuple[str, float], ...] = (
    ("mi fẹ́", 2.0), ("mi fe ", 1.5), ("mi ná", 2.0), ("mi na ", 1.5),
    ("mi lọ", 2.0), ("mi lo ", 1.5), ("mi ti ", 1.5), ("mi ó ", 1.5),
    ("mi o ", 1.0), ("mi ní", 1.5), ("mi ni ", 1.0),
)

# Standard Yoruba subject pronoun + shared money lexicon.
_YORUBA_STD_MARKERS: tuple[tuple[str, float], ...] = (
    ("mo fẹ́", 2.0), ("mo fe ", 1.5), ("mo ná", 2.0), ("mo na ", 1.5),
    ("mo lọ", 2.0), ("mo lo ", 1.5), ("mo ti ", 1.5),
)

_YORUBA_LEXICON: tuple[tuple[str, float], ...] = (
    ("jọwọ", 2.0), ("jowo", 1.5), ("ẹ ṣe", 1.5), ("e se ", 1.0),
    ("fún", 1.5), ("fun ", 1.0), ("ránṣẹ", 2.0), ("ranse", 1.5),
    ("rán ", 1.0), ("ran ", 0.5), ("owo", 1.5), ("owó", 1.5), ("elo", 1.5),
    ("ná ", 1.0), ("na ", 0.5), ("sí ", 0.5), ("si ", 0.25),
    ("ṣeun", 1.0), ("seun", 1.0),
)

_PIDGIN_MARKERS: tuple[tuple[str, float], ...] = (
    ("abeg", 2.5), ("jare", 2.5), ("how far", 2.0), ("i don ", 2.0),
    ("don spend", 2.0), ("don buy", 2.0), ("wahala", 2.0),
    ("sharp", 1.5), ("no be ", 1.5), ("make i", 1.5), ("dey ", 1.0),
)

_HAUSA_MARKERS: tuple[tuple[str, float], ...] = (
    ("aika", 2.5), ("na so", 2.0), ("nawa", 2.0), ("na biya", 2.0),
    ("dan allah", 2.0), ("na gode", 2.0), ("kawo", 1.5), ("zuwa", 1.5),
    ("zua ", 1.0), ("kudi", 1.5),
)

_IGBO_MARKERS: tuple[tuple[str, float], ...] = (
    ("ego", 2.0), ("biko", 2.5), ("daalụ", 2.0), ("daalu", 1.5),
    ("zipu", 2.0), ("kedu", 1.5), ("nye ", 1.0),
)

_EN_MARKERS: tuple[tuple[str, float], ...] = (
    ("please", 1.5), ("spent", 1.5), ("balance", 2.0), ("transfer", 2.0),
    ("how much", 1.5), ("send", 1.0), ("expense", 1.5), ("money", 1.0),
    ("bought", 1.5), ("paid", 1.5),
)

# ── intent verb patterns ────────────────────────────────────────────────────

_BALANCE_RES = (
    re.compile(r"\bbalance\b", re.IGNORECASE),
    re.compile(r"\bhow much\b.{0,20}\b(spen[td]|ná|naya)\b", re.IGNORECASE),
    re.compile(r"\belo\b.{0,20}\b(ná|ti ná|na)\b", re.IGNORECASE),
)

_TRANSFER_RES = (
    re.compile(r"\bsend\b", re.IGNORECASE),
    re.compile(r"ránṣẹ|ranse|\bran\b", re.IGNORECASE),
    re.compile(r"\bfún\b|\bfun\b", re.IGNORECASE),
    re.compile(r"\baika\b", re.IGNORECASE),
    re.compile(r"\bzipu\b", re.IGNORECASE),
    re.compile(r"\btransfer\b", re.IGNORECASE),
    re.compile(r"\bgive\b", re.IGNORECASE),
)

#: Particles that trail a name but are never part of it.
_RECIPIENT_TRAILING_STOP = frozenset(
    {"jare", "na", "ni", "o", "oo", "ooh", "sha"})


def _transfer_verb_present(raw: str) -> bool:
    """True when a transfer verb fires. ``fun/fún <category>`` ("spent for
    data") states a purpose, not a transfer — those occurrences are
    blanked before matching so they can't count as "give"."""
    def _blank(m: re.Match) -> str:
        if m.group(1).lower() in _CATEGORY_WORDS:
            return " " * len(m.group(0))
        return m.group(0)

    text = re.sub(r"\bf[uú]n\b\s+(\w+)", _blank, raw or "",
                  flags=re.IGNORECASE)
    return any(rx.search(text) for rx in _TRANSFER_RES)

_EXPENSE_RES = (
    re.compile(r"\bspen[td]\b", re.IGNORECASE),
    re.compile(r"\bdon\s+(spend|buy|pay)\b", re.IGNORECASE),
    re.compile(r"\bná\b|\bna\b", re.IGNORECASE),
    re.compile(r"\bna\s+biya\b", re.IGNORECASE),
    re.compile(r"\bbought\b|\bpaid\b", re.IGNORECASE),
)

# Recipient introducers: to / fún / sí / give / for / zuwa / nye + name.
_RECIPIENT_RE = re.compile(
    r"(?:\bto\b|\bf[uú]n\b|\bs[ií]\b|\bgive\b|\bfor\b|\bzuwa\b|\bzua\b|\bnye\b)"
    r"\s+([A-ZÀ-Þa-zà-þ][\wÀ-ÿ'’]*(?:\s+[A-ZÀ-Þa-zà-þ][\wÀ-ÿ'’]*){0,2})"
)
_SEND_ME_RE = re.compile(r"\bsend\s+me\b", re.IGNORECASE)

# Amount-looking token inside free text; fed to parse_amount for validation.
_AMOUNT_TOKEN_RE = re.compile(
    r"(?:₦|ngn)?\s*\d[\d,]*(?:\.\d+)?\s*[kKmM]?\s*(?:naira|ngn)?",
    re.IGNORECASE,
)


@dataclass
class MoneyIntent:
    """Parsed money intent from a voice transcript. Offline, rule-based."""

    kind: str  # log_expense | check_balance | transfer | unknown
    amount_kobo: int | None
    recipient: str | None
    category: str | None
    note: str
    language: str  # ekiti | yoruba | pidgin | hausa | igbo | en | mixed
    confidence: float
    raw_transcript: str


def _score(text: str, bank: tuple[tuple[str, float], ...]) -> float:
    return sum(w for phrase, w in bank if phrase in text)


def detect_language(transcript: str) -> tuple[str, float]:
    """(language, confidence). Never raises; empty text → ("en", 0.0).

    Cross-language mixing → "mixed" (code-switching is the norm). Within
    Yoruba, Ekiti dialect markers win ties (documented flagship bias);
    Standard markers alone → "yoruba".
    """
    text = f" {(transcript or '').lower()} "
    ekiti_hits = _score(text, _EKITI_MARKERS)
    scores = {
        "pidgin": _score(text, _PIDGIN_MARKERS),
        "hausa": _score(text, _HAUSA_MARKERS),
        "igbo": _score(text, _IGBO_MARKERS),
        "en": _score(text, _EN_MARKERS),
        "yoruba": _score(text, _YORUBA_STD_MARKERS) + _score(text, _YORUBA_LEXICON),
        "ekiti": ekiti_hits,
    }
    total = sum(scores.values())
    if total <= 0:
        return "en", 0.0
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_lang, top = ranked[0]
    # Second-strongest *language* (Ekiti/Yoruba are one language family —
    # dialect mixing is not code-switching).
    second = max((s for lang, s in ranked[1:]
                  if not {lang, top_lang} <= {"ekiti", "yoruba"}), default=0.0)
    confidence = top / total
    if top_lang == "ekiti" or (top_lang == "yoruba" and ekiti_hits > 0):
        # Documented flagship bias: Ekiti markers win within Yoruba.
        return "ekiti", confidence
    # Code-switching: any real second-language presence means "mixed".
    if second >= 1.0:
        return "mixed", confidence
    return top_lang, confidence


def _extract_amount(text: str) -> int | None:
    """First amount-like token that parse_amount accepts. None when absent."""
    for m in _AMOUNT_TOKEN_RE.finditer(text or ""):
        kobo = parse_amount(m.group(0))
        if kobo is not None and kobo > 0:
            return kobo
    return None


#: Every category keyword, lowercased — a "recipient" that is actually a
#: spending category ("fún data", "for fuel") is not a person.
_CATEGORY_WORDS: frozenset[str] = frozenset(
    w.lower() for kws in CATEGORY_KEYWORDS.values() for w in kws
)


def _extract_recipient(text: str) -> str | None:
    """Recipient after to/fún/sí/give/for/zuwa/nye, or 'me' for 'send me'.

    Returns None when the captured word is a spending category rather
    than a person ("fún data" → data is a category, not a recipient).
    """
    if not text:
        return None
    if _SEND_ME_RE.search(text):
        return "me"
    m = _RECIPIENT_RE.search(text)
    if m:
        parts = m.group(1).strip().strip(",.").split()
        while parts and parts[-1].lower().strip(",.") in _RECIPIENT_TRAILING_STOP:
            parts.pop()
        if not parts:
            return None
        if parts[0].lower() in _CATEGORY_WORDS:
            return None
        return " ".join(parts)
    return None


def parse_voice_money(transcript: str) -> MoneyIntent:
    """Parse a voice transcript into a MoneyIntent. Never raises; never
    guesses a money action — unparseable input → kind="unknown"."""
    raw = transcript or ""
    language, confidence = detect_language(raw)
    amount_kobo = _extract_amount(raw)
    note = raw.strip()

    def _intent(kind: str, **kw: Any) -> MoneyIntent:
        return MoneyIntent(
            kind=kind,
            amount_kobo=kw.get("amount_kobo", amount_kobo),
            recipient=kw.get("recipient"),
            category=kw.get("category") or (categorize(note) if note else None),
            note=note,
            language=language,
            confidence=confidence,
            raw_transcript=raw,
        )

    # Balance questions first: "how much have I spent" contains "spent".
    if any(rx.search(raw) for rx in _BALANCE_RES):
        return _intent("check_balance", amount_kobo=None, recipient=None)
    # Transfer with a named person-recipient beats bare expense markers:
    # Pidgin "na" (emphasis particle) must not reroute "send 5k to Mama na"
    # into an expense, and "fún data" is a category, not a recipient.
    recipient = _extract_recipient(raw)
    has_transfer_verb = _transfer_verb_present(raw)
    if has_transfer_verb and (recipient is not None or amount_kobo is not None):
        return _intent("transfer", recipient=recipient)
    if any(rx.search(raw) for rx in _EXPENSE_RES):
        return _intent("log_expense", recipient=None)
    # A bare transfer verb with no amount and no recipient is not an
    # actionable money intent — unknown, never a guess.
    return _intent("unknown", amount_kobo=None, recipient=None,
                   category=None, note=note)


def transcribe_money_voice(audio_path: str, stt: Any) -> MoneyIntent:
    """Transcribe with language AUTO-DETECT, then parse.

    ``stt`` is duck-typed: a ``UniversalSTT`` (or anything with
    ``transcribe(path, language=...)``) or a plain callable. Auto-detect
    (Whisper ``language=None``) handles code-switched speech better than
    forcing "en" — at the cost that monolingual English may transcribe
    marginally worse than a forced-en pass. Honest tradeoff, documented
    here; no STT quality claims beyond what the backend reports.
    """
    text = ""
    try:
        transcribe = getattr(stt, "transcribe", None)
        if callable(transcribe):
            try:
                result = transcribe(audio_path, language=None)
            except TypeError:
                # Backend requires an explicit language string.
                result = transcribe(audio_path, language="auto")
        elif callable(stt):
            result = stt(audio_path, language=None)
        else:
            raise TypeError("stt has no transcribe() and is not callable")
        if isinstance(result, dict):
            text = str(result.get("text", "") or "")
        else:
            text = str(result or "")
    except Exception as exc:  # noqa: BLE001 - hearing is a bonus, never fatal
        _log.warning("money STT failed on %s: %s", audio_path, exc)
    return parse_voice_money(text.strip())


# ── short per-language reply templates ───────────────────────────────────────
# Template-based, deliberately simple. Ekiti replies use the "mi" subject
# form (the documented dialect marker); anything uncertain falls back to
# Pidgin-leaning phrasing for "mixed".

_REPLIES: dict[str, dict[str, str]] = {
    "en": {
        "logged": "Logged {amt} for {cat}.",
        "balance": "You've spent {amt} this month.",
        "need_amount": "How much? Try 'I spent 5k on data'.",
        "need_details": "How much, and to whom? Try 'send 5k to Mama'.",
        "transfer_denied": "Not confirmed — I won't move the money.",
        "transfer_staged": "Confirmed — {amt} to {who} is staged for transfer.",
        "unknown": "I didn't catch the money part — try 'I spent 5k on data'.",
        "error": "Something went wrong with that money request.",
    },
    "pidgin": {
        "logged": "Don log {amt} for {cat}.",
        "balance": "You don spend {amt} this month.",
        "need_amount": "How much be that? Try 'I don spend 5k for data'.",
        "need_details": "How much, and give who? Try 'abeg send 5k give Mama'.",
        "transfer_denied": "No confirm — I no go move the money.",
        "transfer_staged": "Confirmed — {amt} for {who} don stage for transfer.",
        "unknown": "I no catch the money part — try 'I don spend 5k for data'.",
        "error": "Something spoil for that money request.",
    },
    "yoruba": {
        "logged": "Mo ti kọ {amt} sílẹ̀ fún {cat}.",
        "balance": "O ti ná {amt} ní oṣù yìí.",
        "need_amount": "Elo? Gbìyànjú 'mo ná 5k fún data'.",
        "need_details": "Elo, ta ni? Gbìyànjú 'rán 5k sí Mama'.",
        "transfer_denied": "A kò fọwọ́ sí — mi ò ní gbé owó náà.",
        "transfer_staged": "A fọwọ́ sí — {amt} sí {who} ti ṣetán.",
        "unknown": "N kò gbọ́ apá owó náà — gbìyànjú 'mo ná 5k fún data'.",
        "error": "Nǹkan kan ṣìṣe níbẹ̀.",
    },
    "ekiti": {
        "logged": "Mi ti kọ {amt} sílẹ̀ fún {cat}.",
        "balance": "O ti ná {amt} ní oṣù yìí.",
        "need_amount": "Elo? Gbìyànjú 'mi ná 5k fún data'.",
        "need_details": "Elo, ta ni? Gbìyànjú 'mi fẹ́ rán 5k sí Mama'.",
        "transfer_denied": "A kò fọwọ́ sí — mi ò ní gbé owó náà.",
        "transfer_staged": "A fọwọ́ sí — {amt} sí {who} ti ṣetán.",
        "unknown": "Mi ò gbọ́ apá owó náà — gbìyànjú 'mi ná 5k fún data'.",
        "error": "Nǹkan kan ṣìṣe níbẹ̀.",
    },
    "hausa": {
        "logged": "An rubuta {amt} don {cat}.",
        "balance": "Ka kashe {amt} a wannan wata.",
        "need_amount": "Nawa ne? Ka ce 'na biya 5k'.",
        "need_details": "Nawa, kuma ga wa? Ka ce 'aika 5k zuwa Mama'.",
        "transfer_denied": "Ba a tabbatar ba — ba zan motsa kuɗin ba.",
        "transfer_staged": "An tabbatar — {amt} ga {who} a shirye.",
        "unknown": "Ban ji ɓangaren kuɗin ba — ka ce 'na biya 5k'.",
        "error": "Akwai matsala.",
    },
    "igbo": {
        # Minimal Igbo: only the safest templates; anything richer falls
        # back to English rather than invented fluency.
        "logged": "Edela {amt} maka {cat}.",
        "balance": "I mefuru {amt} n'ọnwa a.",
        "need_amount": "Ego ole? Kwuo 'I spent 5k on data'.",
        "need_details": "Ego ole, nye onye? Kwuo 'send 5k to Mama'.",
        "transfer_denied": "No confirm — I no go move the money.",
        "transfer_staged": "Confirmed — {amt} nye {who} adịla njikere.",
        "unknown": "I didn't catch the money part — try 'I spent 5k on data'.",
        "error": "Something went wrong with that money request.",
    },
    "mixed": {
        "logged": "Don log {amt} for {cat}.",
        "balance": "You don spend {amt} this month.",
        "need_amount": "How much be that? Try 'I don spend 5k for data'.",
        "need_details": "How much, and give who? Try 'abeg send 5k give Mama jare'.",
        "transfer_denied": "No confirm — I no go move the money.",
        "transfer_staged": "Confirmed — {amt} for {who} don stage for transfer.",
        "unknown": "I no catch the money part — try 'I don spend 5k for data'.",
        "error": "Something went wrong with that money request.",
    },
}


def _reply(language: str, key: str, **kw: Any) -> str:
    bank = _REPLIES.get(language, _REPLIES["en"])
    template = bank.get(key, _REPLIES["en"][key])
    try:
        return template.format(**kw)
    except Exception:  # noqa: BLE001 - a bad template must not kill the reply
        return template


# ── staged transfers (the biometric gate + connector handoff point) ─────────

@dataclass
class StagedTransfer:
    """A voice-requested transfer. Money moves ONLY via the connector
    human-checkpoint flow consuming this record — never from voice alone."""

    id: str
    ts: float
    amount_kobo: int
    recipient: str
    note: str
    status: str = "awaiting_biometric"
    biometric_token: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "ts": self.ts, "amount_kobo": self.amount_kobo,
            "recipient": self.recipient, "note": self.note,
            "status": self.status, "biometric_token": self.biometric_token,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StagedTransfer":
        return cls(
            id=str(data.get("id", "")), ts=float(data.get("ts", 0)),
            amount_kobo=int(data.get("amount_kobo", 0)),
            recipient=str(data.get("recipient", "")),
            note=str(data.get("note", "")),
            status=str(data.get("status", "awaiting_biometric")),
            biometric_token=data.get("biometric_token"),
        )


class TransferStaging:
    """JSON-lines store for staged voice transfers. Thread-safety via the
    GIL + atomic appends is enough for this low-volume path.

    HANDOFF POINT: a record with status "biometric_confirmed" carries a
    capability-bound token from approve_with_biometric(). The connector
    layer (nomorals/connectors/_confirm.py::confirm_or_checkpoint) is the
    only sanctioned consumer: it should treat the token as the owner's
    explicit approval of the exact payload, then execute through its own
    human-checkpoint resume path. Nothing else may act on these records.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            _, budgets_path = finance_paths(None)
            path = Path(budgets_path).parent / "staged_transfers.jsonl"
        self.path = Path(path)

    def _ensure(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def stage(self, amount_kobo: int, recipient: str, note: str = "") -> StagedTransfer:
        import hashlib
        import os as _os

        tid = "vtr_" + hashlib.sha256(
            f"{time.time()}:{_os.getpid()}:{recipient}:{amount_kobo}".encode()
        ).hexdigest()[:12]
        rec = StagedTransfer(
            id=tid, ts=time.time(), amount_kobo=int(amount_kobo),
            recipient=recipient, note=note,
        )
        self._ensure()
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec.to_dict()) + "\n")
        _log.info("staged voice transfer %s: %s → %s (awaiting_biometric)",
                  tid, format_naira(amount_kobo), recipient)
        return rec

    def _rewrite(self, mutate: Any) -> None:
        self._ensure()
        recs: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except ValueError:
                _log.debug("skipping malformed staged-transfer line")
        mutate(recs)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            "\n".join(json.dumps(r) for r in recs) + ("\n" if recs else ""),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def mark(self, transfer_id: str, status: str,
             biometric_token: str | None = None) -> bool:
        """Update a staged record's status. Returns False when not found."""
        found = []

        def _mut(recs: list[dict[str, Any]]) -> None:
            for r in recs:
                if r.get("id") == transfer_id:
                    r["status"] = status
                    if biometric_token is not None:
                        r["biometric_token"] = biometric_token
                    found.append(True)

        self._rewrite(_mut)
        return bool(found)

    def pending(self) -> list[StagedTransfer]:
        """Records still awaiting action (awaiting_biometric or
        biometric_confirmed)."""
        out: list[StagedTransfer] = []
        if not self.path.exists():
            return out
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = StagedTransfer.from_dict(json.loads(line))
            except (ValueError, TypeError):
                continue
            if rec.status in ("awaiting_biometric", "biometric_confirmed"):
                out.append(rec)
        return out


# ── routing ──────────────────────────────────────────────────────────────────

def _month_start_ts(now: float | None = None) -> float:
    import datetime as _dt

    now = time.time() if now is None else now
    dt = _dt.datetime.fromtimestamp(now)
    return _dt.datetime(dt.year, dt.month, 1).timestamp()


def _ledger_for(context: Any) -> Ledger:
    settings = getattr(context, "settings", None)
    ledger_path, _ = finance_paths(settings)
    return Ledger(ledger_path)


def _staging_for(context: Any) -> TransferStaging:
    settings = getattr(context, "settings", None)
    _, budgets_path = finance_paths(settings)
    return TransferStaging(Path(budgets_path).parent / "staged_transfers.jsonl")


def handle_voice_money(transcript: str, context: Any) -> str:
    """Route a voice money transcript. Never raises — worst case returns
    an error reply in the detected language."""
    try:
        intent = parse_voice_money(transcript)
        lang = intent.language
        if intent.kind == "log_expense":
            if intent.amount_kobo is None:
                return _reply(lang, "need_amount")
            txn = _ledger_for(context).log(
                intent.amount_kobo, note=intent.note or transcript,
                kind="spend", source="voice",
            )
            return _reply(lang, "logged",
                          amt=format_naira(txn.amount_kobo), cat=txn.category)
        if intent.kind == "check_balance":
            total = _ledger_for(context).total_spent(since=_month_start_ts())
            return _reply(lang, "balance", amt=format_naira(total))
        if intent.kind == "transfer":
            if intent.amount_kobo is None or not intent.recipient:
                return _reply(lang, "need_details")
            staged = _staging_for(context).stage(
                intent.amount_kobo, intent.recipient, note=intent.note)
            token = approve_with_biometric(
                Policy(), "finance.transfer",
                title=f"Send {format_naira(intent.amount_kobo)} "
                      f"to {intent.recipient}",
            )
            if token:
                _staging_for(context).mark(
                    staged.id, "biometric_confirmed", biometric_token=token)
                who = ("you" if intent.recipient.lower() == "me"
                       else intent.recipient)
                _log.info("voice transfer %s biometric-confirmed", staged.id)
                return _reply(lang, "transfer_staged",
                              amt=format_naira(intent.amount_kobo), who=who)
            _staging_for(context).mark(staged.id, "denied")
            _log.info("voice transfer %s denied at biometric gate", staged.id)
            return _reply(lang, "transfer_denied")
        return _reply(lang, "unknown")
    except Exception:  # noqa: BLE001 - money path must never crash the chat
        _log.exception("handle_voice_money failed")
        try:
            lang = parse_voice_money(transcript).language
        except Exception:  # noqa: BLE001
            lang = "en"
        return _reply(lang, "error")


def voice_money_precheck(transcript: str, context: Any) -> str | None:
    """Bridge hook: confident money intent → reply string (caller sends it
    and skips the normal path); anything else → None (fall through).

    The confidence gate keeps ordinary chat ("send me that file") on the
    normal path — only clear money commands short-circuit.
    """
    try:
        intent = parse_voice_money(transcript)
    except Exception:  # noqa: BLE001
        return None
    if intent.kind == "unknown" or intent.confidence < 0.5:
        return None
    if intent.kind in ("log_expense", "transfer") and intent.amount_kobo is None:
        # "I spent money on data" — no amount to act on; let the brain ask.
        return None
    return handle_voice_money(transcript, context)
