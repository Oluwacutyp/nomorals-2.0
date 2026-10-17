"""Chat control for /legal — one front door over nomorals.legal.

Never raises. Owner-only (same gate as /contract, /contracts, /regwatch).

Routing — bare questions keep the EXISTING /legal aid behaviour
(``control_legal`` in :mod:`nomorals.legal.aid`); the new subcommands
route to the matching existing control:

    /legal contract ...    → control_contract (Nigerian playbooks)
    /legal portfolio ...   → control_contracts (obligation tracker)
    /legal contracts ...   → control_contracts (alias)
    /legal regulatory ...  → control_regwatch (CBN/SEC/NDPA/FIRS/…)
    /legal regwatch ...    → control_regwatch (alias)
    /legal aid ...         → control_legal (plain-language aid)
    /legal research ...    → control_research (grounded RAG)
    /legal citations       → precedent citations (not advice)
    /legal <question>      → control_legal (unchanged aid behaviour)

Every output this router generates itself carries the "not legal
advice" disclaimer. Delegated outputs already carry it (each legal
module attaches :data:`DISCLAIMER` to every public output).

The DoNotPay FTC $193K precedent and the NBA 2024 AI guidelines are
surfaced here as CITATIONS for why Devon gives information, never
advice — never as a claim that Devon is, or performs like, a lawyer.
"""

from __future__ import annotations

from typing import Any

from ..core.logging_setup import get_logger
from .contracts import DISCLAIMER
from ..core.tool_render import describe_exc

_log = get_logger(__name__)

__all__ = ["control_legal_full", "LEGAL_USAGE"]


def LEGAL_USAGE() -> str:
    return (
        "⚖️ /legal — legal information, never legal advice\n"
        "  /legal contract review [tenancy|employment|freelance] <paste contract text>\n"
        "  /legal contract types | meta | rights | rewrite — contract risk review (Nigerian playbooks)\n"
        "  /legal portfolio — obligation tracker (renewals, notice deadlines, SLA)\n"
        "  /legal regulatory regulators | add <REG> [topics] | list | check | calendar\n"
        "  /legal aid [en|pcm|yo|ha|ig] <question> — plain-language aid (English/Pidgin/Yoruba/Hausa/Igbo)\n"
        "  /legal research <question> — grounded research with citations\n"
        "  /legal citations — why information-only (precedent citations)\n"
        "  /legal [en|pcm|yo|ha|ig] <question> — ask directly (aid)\n"
        + DISCLAIMER
    )


CITATIONS_TEXT = (
    "📚 Why Devon gives legal information, never legal advice — citations:\n"
    "  • DoNotPay — FTC settlement, $193,000 (2024): the FTC held AI "
    "legal-service claims to substantiation; an AI product that claimed "
    "to perform like a lawyer was penalised. Devon therefore never "
    "claims to be or perform like a lawyer.\n"
    "  • Nigerian Bar Association (NBA) 2024 AI guidelines: technology "
    "supports professional judgment, never replaces it. Devon points "
    "consequential decisions at real, qualified Nigerian lawyers.\n"
    "  • Every /legal output carries the disclaimer below and cites the "
    "corpus provisions it used, so you can take it to a lawyer.\n\n"
    + DISCLAIMER
)


def control_legal_full(tail: str, context: Any = None, chat: Any = None,
                       sender_id: str = "", sender: str = "") -> str:
    """``/legal`` router — subcommands + unchanged aid fallback. Never raises."""
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() == "help":
            return LEGAL_USAGE()
        verb, _, sub = rest.partition(" ")
        verb = verb.lower()
        sub = sub.strip()
        if verb == "citations":
            return CITATIONS_TEXT
        if verb == "contract":
            from .contracts import control_contract
            return control_contract(sub, context=context, chat=chat,
                                    sender_id=sender_id, sender=sender)
        if verb in ("portfolio", "contracts"):
            from .portfolio import control_contracts
            return control_contracts(sub, context=context, chat=chat,
                                     sender_id=sender_id, sender=sender)
        if verb in ("regulatory", "regwatch"):
            from .regulatory import control_regwatch
            return control_regwatch(sub, context=context, chat=chat,
                                    sender_id=sender_id, sender=sender)
        if verb in ("aid", "ask"):
            from .aid import control_legal
            return control_legal(sub, context=context, chat=chat,
                                 sender_id=sender_id, sender=sender)
        if verb == "research":
            from .research import control_research
            return control_research(sub, context=context, chat=chat,
                                    sender_id=sender_id, sender=sender)
        # Unchanged aid behaviour: bare question (optionally language-first).
        from .aid import control_legal
        return control_legal(rest, context=context, chat=chat,
                             sender_id=sender_id, sender=sender)
    except Exception as exc:  # noqa: BLE001 — never raise from chat
        return f"Legal hit an error ({describe_exc(exc)}). {DISCLAIMER}"
