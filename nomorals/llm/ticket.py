"""TranslationTicket: the typed facade over the translation organ.

Doc 19 §4 (adopted 2026-11-10, pass T1): a caller who needs to say
WHAT KIND of translation it is asking for — and to prove afterwards
what it got — wraps the request in a ticket.  The ticket carries the
exactness class (gist/fidelity/legal/code), glossary + preserve
intent, audience, a local-only requirement and the requester.  The
organ (``llm.translate``) does the work exactly as it does for every
other caller and mints its durable receipt; the ticket result ties
the three together: the outcome, the receipt id, the quality gate's
verdict, and whether the result may be LABELLED exact.

The label rule is doc 19's honesty rule: a fidelity-class result is
labelled exact only when the bidirectional quality gate passed it.
A failed gate never swaps or blocks the translation (the gate is a
mark — see ``Translator._gate_verdict``); it refuses the label.

Glossary application and structure preservation are T2's build; the
ticket carries those fields from day one so T2's consumers do not
change shape, and the ticket result says plainly (in ``notes``) that
they were carried, not applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .languages import detect_language, normalize_code
from .translate import (
    EXACTNESS_CLASSES,
    FIDELITY_CLASSES,
    TranslationOutcome,
    Translator,
)

__all__ = [
    "TranslationTicket",
    "TicketOutcome",
    "run_ticket",
]


@dataclass
class TranslationTicket:
    """One translation request, with its intent stated."""

    text: str
    target_lang: str = ""
    source_lang: str = ""
    exactness: str = "gist"
    glossary_id: str = ""
    preserve: tuple[str, ...] = ()
    audience: str = ""
    local_only: bool = False
    requester: str = ""


@dataclass
class TicketOutcome:
    """A ticket's result: the organ's outcome + its proof."""

    outcome: TranslationOutcome
    receipt_id: str = ""
    gate: dict[str, Any] = field(default_factory=dict)
    exact_label: bool = False
    exactness: str = "gist"
    requester: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.outcome.text

    @property
    def translated(self) -> bool:
        return self.outcome.translated

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.outcome.text,
            "source_lang": self.outcome.source_lang,
            "target_lang": self.outcome.target_lang,
            "rung": self.outcome.rung,
            "translated": self.outcome.translated,
            "note": self.outcome.note,
            "receipt_id": self.receipt_id,
            "gate": self.gate,
            "exact_label": self.exact_label,
            "exactness": self.exactness,
            "requester": self.requester,
            "notes": list(self.notes),
        }


def _run_local_only(translator: Translator,
                    ticket: TranslationTicket,
                    ) -> tuple[TranslationOutcome, dict[str, Any]]:
    """A local-only ticket: the local rung or the marked identity —
    no other rung is consulted, for the translation OR for the
    quality gate's round trip."""
    raw = ticket.text or ""
    dst = normalize_code(ticket.target_lang) \
        or translator.owner_language()
    if not raw.strip():
        outcome = TranslationOutcome(text=raw, target_lang=dst,
                                     rung="identity",
                                     note="empty text")
        gate = Translator._gate_not_run("empty text")
        translator._mint_receipt(
            outcome, chars=0, cache="miss",
            exactness=ticket.exactness, gate=gate,
            requester=ticket.requester)
        return outcome, gate
    src = normalize_code(ticket.source_lang)
    if not src:
        detected, _conf = detect_language(raw)
        src = "" if detected == "und" else detected
    if src and src == dst:
        outcome = TranslationOutcome(
            text=raw, source_lang=src, target_lang=dst,
            rung="same-language",
            note="already in the target language")
        gate = Translator._gate_not_run(
            "already in the target language")
        translator._mint_receipt(
            outcome, chars=len(raw), cache="miss",
            exactness=ticket.exactness, gate=gate,
            requester=ticket.requester)
        return outcome, gate
    ok, reason = translator.local.available()
    out = translator.local.translate(raw, src, dst) if ok else None
    if out:
        outcome = TranslationOutcome(
            text=out, source_lang=src, target_lang=dst,
            rung="local", translated=True)
    else:
        why = reason or "the local rung declined this pair"
        outcome = TranslationOutcome(
            text=raw, source_lang=src, target_lang=dst,
            rung="identity",
            note="local-only ticket: " + why + " — so this is the "
                 "ORIGINAL text, not a translation")

    def _reverse(text: str, s: str, d: str) -> str:
        return translator.local.translate(text, s, d) or ""

    gate = translator._gate_verdict(
        raw, outcome, exactness=ticket.exactness, reverse_fn=_reverse)
    translator._mint_receipt(
        outcome, chars=len(raw), cache="miss",
        exactness=ticket.exactness, gate=gate,
        requester=ticket.requester)
    return outcome, gate


def run_ticket(translator: Translator,
               ticket: TranslationTicket) -> TicketOutcome:
    """Execute one ticket against the organ.

    Raises ValueError on an unknown exactness class — a misspelled
    class must not silently degrade to gist.
    """
    if ticket.exactness not in EXACTNESS_CLASSES:
        raise ValueError(
            "unknown exactness class " + repr(ticket.exactness)
            + " — one of " + ", ".join(EXACTNESS_CLASSES))
    notes: list[str] = []
    if ticket.glossary_id:
        notes.append(
            "glossary " + repr(ticket.glossary_id) + " carried on "
            "the ticket — the glossary store lands in T2; no terms "
            "were applied in this pass")
    if ticket.preserve:
        notes.append(
            "preserve rules carried on the ticket — structure "
            "preservation lands in T2's section-wise translation")
    if ticket.local_only:
        outcome, gate = _run_local_only(translator, ticket)
        receipt = translator.last_receipt or {}
    else:
        outcome = translator.translate(
            ticket.text, target=ticket.target_lang,
            source=ticket.source_lang, exactness=ticket.exactness,
            requester=ticket.requester)
        receipt = translator.last_receipt or {}
        gate = receipt.get("gate") or Translator._gate_not_run(
            "no receipt was minted for this call")
    exact_label = bool(
        ticket.exactness in FIDELITY_CLASSES and outcome.translated
        and gate.get("status") == "passed")
    if (ticket.exactness in FIDELITY_CLASSES and outcome.translated
            and not exact_label):
        notes.append(
            "not labelled exact: the quality gate did not pass this "
            "translation (" + str(gate.get("status", "not_run"))
            + ")")
    return TicketOutcome(
        outcome=outcome,
        receipt_id=str(receipt.get("receipt_id", "")),
        gate=gate, exact_label=exact_label,
        exactness=ticket.exactness, requester=ticket.requester,
        notes=notes)
