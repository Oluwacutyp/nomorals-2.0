"""TranslationTicket: the typed facade over the translation organ.

Doc 19 §4 (adopted 2026-11-10): a caller who needs to say WHAT KIND
of translation it is asking for — and to prove afterwards what it
got — wraps the request in a ticket.  The ticket carries the
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

Pass T1 carried glossary/preserve intent without applying it and
said so.  Pass T2 applies both, and the notes now say what actually
happened:

- **Glossary** (:mod:`nomorals.llm.glossary`): the named glossary's
  terms for the pair are protected by placeholders before the rung
  runs and restored — verified — after.  A term whose placeholder
  does not survive is re-translated unprotected and NAMED in the
  notes; it is never silently dropped.  The glossary's fingerprint
  rides the cache key, so editing a glossary invalidates its cached
  translations.
- **Structure** (:mod:`nomorals.llm.sections`): a document-shaped
  input is parsed into sections, each translated through this same
  ticket path (one receipt per section), and reassembled with its
  structure intact — code fences byte-identical, tables cell by
  cell, link destinations untouched.  The result carries the
  section map: per-section rung, gate verdict and receipt.  Plain
  prose keeps the single-shot path unchanged.  Preserve rules:
  ``citations``/``ids`` widen placeholder protection, ``tables``
  passes tables through whole, ``verbatim`` suppresses sectioning;
  a rule with no mechanism is still carried — and the notes say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .glossary import GlossaryStore, GlossaryTerm, protect_terms, restore_terms
from .languages import detect_language, normalize_code
from .sections import Document, parse_document, protected_unit_document
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

#: Preserve rules with a mechanism behind them in this pass.
_KNOWN_PRESERVE = ("citations", "ids", "tables", "verbatim")


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
    """A ticket's result: the organ's outcome + its proof.

    ``sections`` is the section map for a document ticket (empty on
    the plain path); ``receipt_ids`` lists every receipt the job
    minted — one on the plain path, one per section for a document,
    whose composite outcome therefore carries ``receipt_id == ""``
    with the ids here and in the section map.
    """

    outcome: TranslationOutcome
    receipt_id: str = ""
    gate: dict[str, Any] = field(default_factory=dict)
    exact_label: bool = False
    exactness: str = "gist"
    requester: str = ""
    notes: list[str] = field(default_factory=list)
    sections: list[dict[str, Any]] = field(default_factory=list)
    receipt_ids: list[str] = field(default_factory=list)

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
            "receipt_ids": list(self.receipt_ids),
            "gate": self.gate,
            "exact_label": self.exact_label,
            "exactness": self.exactness,
            "requester": self.requester,
            "notes": list(self.notes),
            "sections": list(self.sections),
        }


# ----------------------------------------------------------------------
# one unit of work (a whole plain text, or one document section)
# ----------------------------------------------------------------------

def _call_organ(translator: Translator, ticket: TranslationTicket,
                text: str, cache_key_extra: str
                ) -> tuple[TranslationOutcome, dict[str, Any], str]:
    """One organ call for one unit of text: (outcome, gate,
    receipt_id).  Local-only tickets ride the local rung alone."""
    if ticket.local_only:
        outcome, gate = _run_local_only(translator, ticket, text)
        receipt = translator.last_receipt or {}
        return outcome, gate, str(receipt.get("receipt_id", ""))
    outcome = translator.translate(
        text, target=ticket.target_lang, source=ticket.source_lang,
        exactness=ticket.exactness, requester=ticket.requester,
        cache_key_extra=cache_key_extra)
    receipt = translator.last_receipt or {}
    gate = receipt.get("gate") or Translator._gate_not_run(
        "no receipt was minted for this call")
    return outcome, gate, str(receipt.get("receipt_id", ""))


def _translate_unit(translator: Translator, ticket: TranslationTicket,
                    text: str, *, terms: list[GlossaryTerm],
                    cache_key_extra: str) -> dict[str, Any]:
    """Translate one unit with glossary enforcement wrapped around
    the organ call.  The returned outcome's text is always in the
    input's own space (placeholders never leak into it)."""
    if not terms:
        outcome, gate, receipt_id = _call_organ(
            translator, ticket, text, cache_key_extra)
        return {"outcome": outcome, "gate": gate,
                "receipt_id": receipt_id, "applied": [],
                "unenforced": []}
    protected, occurrences = protect_terms(text, terms)
    if not occurrences:
        outcome, gate, receipt_id = _call_organ(
            translator, ticket, text, cache_key_extra)
        return {"outcome": outcome, "gate": gate,
                "receipt_id": receipt_id, "applied": [],
                "unenforced": []}
    outcome, gate, receipt_id = _call_organ(
        translator, ticket, protected, cache_key_extra)
    if not outcome.translated:
        # Identity / same-language: nothing was translated, so
        # nothing was enforced either — hand back the original
        # text, never the placeholder-bearing form.
        return {"outcome": replace(outcome, text=text), "gate": gate,
                "receipt_id": receipt_id, "applied": [],
                "unenforced": []}
    restored, lost = restore_terms(outcome.text, occurrences)
    if restored is not None:
        applied = list({term.source_term: term
                        for _, term in occurrences}.values())
        return {"outcome": replace(outcome, text=restored),
                "gate": gate, "receipt_id": receipt_id,
                "applied": applied, "unenforced": []}
    # The rung mangled placeholder(s): re-translate this unit
    # unprotected — the fallback call's outcome AND gate govern —
    # and disclose the unenforced terms to the caller.
    outcome, gate, receipt_id = _call_organ(
        translator, ticket, text, cache_key_extra)
    return {"outcome": outcome, "gate": gate,
            "receipt_id": receipt_id, "applied": [],
            "unenforced": lost}


def _run_local_only(translator: Translator,
                    ticket: TranslationTicket, raw: str
                    ) -> tuple[TranslationOutcome, dict[str, Any]]:
    """A local-only unit: the local rung or the marked identity —
    no other rung is consulted, for the translation OR for the
    quality gate's round trip."""
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


# ----------------------------------------------------------------------
# documents
# ----------------------------------------------------------------------

def _run_document(translator: Translator, ticket: TranslationTicket,
                  document: Document, *, terms: list[GlossaryTerm],
                  cache_key_extra: str, src: str, dst: str,
                  notes: list[str]) -> TicketOutcome:
    """Translate a parsed document section by section and
    reassemble it with its structure intact."""
    rendered: dict[int, str] = {}
    sections: list[dict[str, Any]] = []
    receipt_ids: list[str] = []
    applied: dict[str, GlossaryTerm] = {}
    unenforced: dict[str, GlossaryTerm] = {}
    prose_total = 0
    prose_translated = 0
    gate_statuses: list[str] = []
    label_blockers: list[str] = []
    first_outcome: TranslationOutcome | None = None
    table_entry: dict[str, Any] | None = None
    table_group = ""

    def flush_table() -> None:
        nonlocal table_entry, table_group
        if table_entry is not None:
            sections.append(table_entry)
            table_entry = None
            table_group = ""

    for idx, unit in enumerate(document.units):
        if table_entry is not None and unit.group != table_group:
            flush_table()
        if unit.prose is None:
            if unit.kind == "code":
                sections.append({
                    "index": idx, "kind": "code", "translated": False,
                    "rung": "preserved", "gate_status": "",
                    "receipt_id": "", "receipt_ids": [],
                    "note": "code fence carried byte-identical — "
                            "never translated"})
            elif unit.kind == "table-cell":
                if table_entry is None:
                    table_group = unit.group
                    table_entry = {
                        "index": idx, "kind": "table",
                        "translated": False, "rung": "preserved",
                        "gate_status": "", "receipt_id": "",
                        "receipt_ids": [], "cells": [],
                        "note": "table passed through whole "
                                "(preserve: tables)"}
                table_entry["cells"].append(
                    {"index": idx, "translated": False,
                     "rung": "preserved", "receipt_id": ""})
            continue
        result = _translate_unit(
            translator, ticket, unit.prose, terms=terms,
            cache_key_extra=cache_key_extra)
        outcome = result["outcome"]
        gate = result["gate"]
        receipt_id = result["receipt_id"]
        if receipt_id:
            receipt_ids.append(receipt_id)
        for term in result["applied"]:
            applied[term.source_term] = term
        for term in result["unenforced"]:
            unenforced[term.source_term] = term
        if first_outcome is None:
            first_outcome = outcome
        framed = unit.render(outcome.text)
        section_translated = bool(outcome.translated
                                  and framed is not None)
        if framed is None:
            label_blockers.append(
                f"section {idx} ({unit.kind}) kept its original text — the "
                "rung did not return its structure placeholders "
                "intact")
            rendered[idx] = unit.raw
        else:
            rendered[idx] = framed
        gate_status = str(gate.get("status", ""))
        if unit.group:
            if table_entry is None:
                table_group = unit.group
                table_entry = {
                    "index": idx, "kind": "table", "translated": True,
                    "rung": outcome.rung, "gate_status": gate_status,
                    "receipt_id": receipt_id,
                    "receipt_ids": [], "cells": [], "note": ""}
            table_entry["receipt_ids"].append(receipt_id)
            table_entry["cells"].append({
                "index": idx, "translated": section_translated,
                "rung": outcome.rung, "receipt_id": receipt_id})
            if not section_translated:
                table_entry["translated"] = False
            if table_entry["rung"] != outcome.rung:
                table_entry["rung"] = "mixed"
            if gate_status != "passed":
                table_entry["gate_status"] = gate_status
        else:
            sections.append({
                "index": idx, "kind": unit.kind,
                "translated": section_translated,
                "rung": outcome.rung, "gate_status": gate_status,
                "receipt_id": receipt_id, "receipt_ids": [receipt_id],
                "note": "" if framed is not None else
                        "structure placeholders did not survive "
                        "the rung — original text kept"})
        prose_total += 1
        if section_translated:
            prose_translated += 1
            gate_statuses.append(gate_status)
        else:
            label_blockers.append(
                f"section {idx} ({unit.kind}) was not translated")
    flush_table()
    for entry in sections:
        if entry["kind"] == "table" and entry.get("cells"):
            entry["note"] = entry.get("note") or (
                "{} cell(s), each translated as its own unit"
                .format(len(entry["cells"])))
    text = document.reassemble(rendered)
    translated = prose_translated > 0
    if gate_statuses and all(s == "passed" for s in gate_statuses) \
            and prose_translated == prose_total:
        gate = {"status": "passed",
                "basis": "every translated section's own gate "
                         "verdict — see the section map"}
    elif gate_statuses:
        gate = {"status": "failed",
                "basis": "at least one section's gate did not pass "
                         "or a section went untranslated — see "
                         "the section map"}
    else:
        gate = Translator._gate_not_run(
            "no section was translated, so no gate verdict exists")
    exact_label = bool(
        ticket.exactness in FIDELITY_CLASSES and translated
        and not label_blockers and not unenforced
        and gate.get("status") == "passed")
    _glossary_notes(notes, ticket, applied, unenforced)
    if (ticket.exactness in FIDELITY_CLASSES and translated
            and not exact_label):
        reason = label_blockers[0] if label_blockers else (
            "the quality gate did not pass every section ("
            + str(gate.get("status", "not_run")) + ")")
        notes.append("not labelled exact: " + reason)
    notes.append(
        "document receipt: one receipt per section — the composite "
        "carries receipt_id '' by design; every id is in "
        "receipt_ids and the section map")
    outcome = TranslationOutcome(
        text=text,
        source_lang=(first_outcome.source_lang if first_outcome
                     else src),
        target_lang=dst, rung="document", translated=translated,
        note=f"document: {prose_total} prose section(s), {prose_translated} translated, "
             "structure preserved by construction")
    return TicketOutcome(
        outcome=outcome, receipt_id="", gate=gate,
        exact_label=exact_label, exactness=ticket.exactness,
        requester=ticket.requester, notes=notes, sections=sections,
        receipt_ids=receipt_ids)


def _glossary_notes(notes: list[str], ticket: TranslationTicket,
                    applied: dict[str, GlossaryTerm],
                    unenforced: dict[str, GlossaryTerm]) -> None:
    if applied:
        names = ", ".join(sorted(applied))
        notes.append(
            "glossary " + repr(ticket.glossary_id) + ": enforced "
            + str(len(applied)) + " term(s) — " + names)
        if ticket.exactness in FIDELITY_CLASSES:
            notes.append(
                "the quality gate marked the placeholder-protected "
                "candidate; glossary terms were restored after "
                "the gate ran")
    if unenforced:
        names = ", ".join(repr(n) for n in sorted(unenforced))
        notes.append(
            "glossary " + repr(ticket.glossary_id) + ": term(s) NOT "
            "enforced — the rung did not return their placeholders "
            "intact, so those units were re-translated "
            "unprotected: " + names)


# ----------------------------------------------------------------------
# the ticket entry point
# ----------------------------------------------------------------------

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
    raw = ticket.text or ""
    dst = normalize_code(ticket.target_lang) \
        or translator.owner_language()
    src = normalize_code(ticket.source_lang)
    if not src and raw.strip():
        detected, _conf = detect_language(raw)
        src = "" if detected == "und" else detected

    # The glossary resolves for the pair up front: its fingerprint
    # joins the cache key, so a glossary edit invalidates cached
    # translations instead of serving them stale.
    terms: list[GlossaryTerm] = []
    cache_key_extra = ""
    if ticket.glossary_id:
        store = GlossaryStore(getattr(translator, "db", None),
                              getattr(translator, "home", ""))
        terms = store.terms_for(ticket.glossary_id, src, dst)
        cache_key_extra = "glossary:" + store.fingerprint(
            ticket.glossary_id, src, dst)
        if terms:
            notes.append(
                "glossary " + repr(ticket.glossary_id) + ": "
                + str(len(terms)) + " term(s) in force for "
                + (src or "any") + "→" + dst
                + " — protected before translation, restored and "
                  "verified after")
        else:
            notes.append(
                "glossary " + repr(ticket.glossary_id) + ": no terms "
                "stored for " + (src or "any") + "→" + dst
                + " — nothing to apply")
    if ticket.preserve:
        cache_key_extra += "|preserve:" + ",".join(
            sorted(set(ticket.preserve)))
        unknown = [rule for rule in ticket.preserve
                   if rule not in _KNOWN_PRESERVE]
        for rule in unknown:
            notes.append(
                "preserve rule " + repr(rule) + " carried on the "
                "ticket — no mechanism applies it")

    document = None
    if raw.strip() and "verbatim" not in ticket.preserve:
        document = parse_document(raw, preserve=ticket.preserve)
        if document is None and ("citations" in ticket.preserve
                                 or "ids" in ticket.preserve):
            # Plain prose, but the preserve rules name token-level
            # protection: run the whole text as one protected unit.
            document = protected_unit_document(
                raw, preserve=ticket.preserve)
    if document is not None:
        if "tables" in ticket.preserve:
            for unit in document.units:
                if unit.kind == "table-cell":
                    unit.prose = None  # whole-table passthrough
        return _run_document(
            translator, ticket, document, terms=terms,
            cache_key_extra=cache_key_extra, src=src, dst=dst,
            notes=notes)

    result = _translate_unit(
        translator, ticket, raw, terms=terms,
        cache_key_extra=cache_key_extra)
    outcome = result["outcome"]
    gate = result["gate"]
    receipt_id = result["receipt_id"]
    _glossary_notes(notes, ticket,
                    {t.source_term: t for t in result["applied"]},
                    {t.source_term: t for t in result["unenforced"]})
    exact_label = bool(
        ticket.exactness in FIDELITY_CLASSES and outcome.translated
        and not result["unenforced"]
        and gate.get("status") == "passed")
    if (ticket.exactness in FIDELITY_CLASSES and outcome.translated
            and not exact_label):
        if result["unenforced"]:
            notes.append(
                "not labelled exact: glossary term(s) went "
                "unenforced — see the glossary note above")
        else:
            notes.append(
                "not labelled exact: the quality gate did not pass "
                "this translation ("
                + str(gate.get("status", "not_run")) + ")")
    return TicketOutcome(
        outcome=outcome, receipt_id=receipt_id, gate=gate,
        exact_label=exact_label, exactness=ticket.exactness,
        requester=ticket.requester, notes=notes,
        receipt_ids=[receipt_id] if receipt_id else [])
