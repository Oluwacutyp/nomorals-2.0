"""Legal product — contract risk review, portfolio obligations,
regulatory monitoring, and legal aid as one product surface.

Wraps the real engines in :mod:`nomorals.legal` (contracts, portfolio,
regulatory, aid). This module adds NO new legal reasoning — it is the
product surface: Nigerian playbook descriptors, the obligation tracker
(computed from *supplied* dates — never invented), regulatory checks
against *supplied* updates (no fetching, no invented URLs), and legal
aid in en/pidgin/yo/ha/ig.

Invariant: EVERY public output carries the "not legal advice" disclaimer
AND the citations block (DoNotPay FTC $193K precedent + NBA 2024 AI
guidelines). :func:`with_disclaimer` enforces it; every product function
routes through it.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

from .aid import (
    LANGUAGES as _AID_LANGUAGES,
)
from .aid import (
    answer_legal_question as _answer_legal_question,
)
from .aid import (
    format_answer as _format_answer,
)
from .aid import (
    letter_kinds as _letter_kinds,
)
from .aid import (
    refer_lawyer as _refer_lawyer,
)
from .aid import (
    render_letter as _render_letter,
)
from .contracts import DISCLAIMER, format_review, review_contract
from .portfolio import ContractPortfolio
from .regulatory import (
    REGULATORS,
    alert_text,
    controls_for,
    impact_score,
    item_from_dict,
    upcoming_deadlines,
)

__all__ = [
    "DISCLAIMER",
    "CITATIONS",
    "NIGERIAN_PLAYBOOKS",
    "REGULATORY_DESCRIPTORS",
    "with_disclaimer",
    "review_contract_product",
    "obligation_status",
    "format_obligations",
    "portfolio_product_summary",
    "check_regulatory_updates",
    "regulatory_calendar_product",
    "legal_aid",
    "legal_referral",
    "render_letter_product",
    "legal_flow",
    "LEGAL_FLOW_USAGE",
]

#: Citations carried on every product output. Wording matches the
#: precedent as stated in nomorals.legal.contracts / aid.
CITATIONS: tuple[str, ...] = (
    "DoNotPay FTC $193K precedent — AI legal services must not perform "
    "like a lawyer; technology supports professional judgment, never "
    "replaces it.",
    "NBA 2024 AI guidelines — technology supports professional judgment, "
    "never replaces it.",
)

_CITATIONS_BLOCK = (
    "Citations:\n" + "\n".join(f"• {c}" for c in CITATIONS)
)


def with_disclaimer(text: str) -> str:
    """Append disclaimer + citations if not already present. Idempotent."""
    out = text or ""
    if DISCLAIMER not in out:
        out = out.rstrip() + "\n\n" + DISCLAIMER
    if "DoNotPay FTC $193K" not in out:
        out = out.rstrip() + "\n\n" + _CITATIONS_BLOCK
    return out


# ── Nigerian playbook descriptors ───────────────────────────────────────────
# Descriptors of the playbooks that live in legal/contracts.py — rule
# areas + governing instruments, NOT a re-implementation of the rules.

NIGERIAN_PLAYBOOKS: dict[str, dict[str, Any]] = {
    "tenancy": {
        "name": "Lagos tenancy agreement",
        "instruments": ["Lagos Tenancy Law 2011"],
        "rule_areas": ["rent-increase notice", "service-charge amount/cap",
                       "notice-to-quit periods", "agency fee (10% norm)",
                       "auto-renewal", "repairs allocation",
                       "late-rent penalty", "governing law",
                       "statutory-rights waiver", "liability / indemnity"],
        "implemented_in": "nomorals.legal.contracts review_contract (T-rules)",
    },
    "employment": {
        "name": "Nigerian employment agreement",
        "instruments": ["Labour Act, Cap L1 LFN 2004",
                        "Pension Reform Act 2014 (employer 10% / employee 8%)"],
        "rule_areas": ["probation length (3–6 month norm)",
                       "termination notice", "non-compete reasonableness",
                       "pension / Contributory Pension Scheme",
                       "salary deductions", "confidentiality",
                       "working hours / overtime"],
        "implemented_in": "nomorals.legal.contracts review_contract (E-rules)",
    },
    "freelance": {
        "name": "Freelance service agreement",
        "instruments": ["Copyright Act 2022 (ownership; assignment in writing)"],
        "rule_areas": ["payment deadline (7–30 day norm)", "IP ownership",
                       "mutual termination", "late-payment provision",
                       "bounded scope / revisions", "confidentiality / NDA",
                       "dispute resolution", "liability cap",
                       "cancellation / kill fee"],
        "implemented_in": "nomorals.legal.contracts review_contract (F-rules)",
    },
}


# ── contract risk review (wraps legal/contracts.py) ────────────────────────

def review_contract_product(doc_text: str, contract_type: str = "auto",
                            style: str = "full") -> dict[str, Any]:
    """Review a contract; returns grade/findings + formatted text.

    Every output carries disclaimer + citations. Delegates scoring to
    :func:`review_contract`.
    """
    review = review_contract(doc_text, contract_type)
    text = with_disclaimer(format_review(review, style=style))
    return {
        "contract_type": review.contract_type,
        "score": review.score,
        "grade": review.grade,
        "findings": [{"rule_id": f.rule_id, "title": f.title,
                       "severity": f.severity, "explanation": f.explanation,
                       "playbook_ref": f.playbook_ref,
                       "suggestion": f.suggestion}
                      for f in review.findings],
        "needs_attention": review.needs_attention,
        "clauses_seen": review.clauses_seen,
        "text": text,
    }


# ── portfolio obligation tracker ────────────────────────────────────────────
# Pure date math over SUPPLIED obligations. Dates are never invented:
# an obligation without a supplied due date reports "no date supplied".

def _parse_supplied_date(value: Any) -> float | None:
    """Parse a supplied due date (ISO string or unix timestamp)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # Naive ISO dates are read as UTC — deterministic, never
            # host-timezone-dependent.
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def obligation_status(obligation: dict[str, Any],
                      *, now: float | None = None) -> dict[str, Any]:
    """Status of ONE supplied obligation, computed from its supplied date.

    ``obligation``: {"description": str, "due": ISO date | timestamp | None,
    "kind": str}. Returns days-until (negative = overdue), a human line
    like "tenancy renews in 60 days", and ``date_supplied`` False when no
    date was given — in which case NOTHING is computed.
    """
    now = now if now is not None else time.time()
    desc = str(obligation.get("description", "") or "").strip()
    due_ts = _parse_supplied_date(obligation.get("due"))
    if due_ts is None:
        return {"description": desc,
                "kind": str(obligation.get("kind", "deadline")),
                "date_supplied": False,
                "days_until": None, "status": "no date supplied — cannot compute",
                "line": f"{desc or 'obligation'} — no due date supplied, "
                        "so no countdown can be computed."}
    days = int((due_ts - now) // 86400)
    if days < 0:
        status, line = "overdue", f"{desc} — OVERDUE by {-days} day(s)"
    elif days == 0:
        status, line = "due today", f"{desc} — due today"
    else:
        status, line = "upcoming", f"{desc} — in {days} day(s)"
    return {"description": desc, "kind": str(obligation.get("kind", "deadline")),
            "date_supplied": True, "due_ts": due_ts, "days_until": days,
            "status": status, "line": line}


def format_obligations(obligations: list[dict[str, Any]],
                       *, now: float | None = None) -> str:
    """Render supplied obligations with computed countdowns + disclaimer."""
    now = now if now is not None else time.time()
    if not obligations:
        return with_disclaimer("No obligations supplied.")
    lines = ["📋 Obligation tracker (dates as supplied):"]
    for ob in obligations:
        lines.append(f"• {obligation_status(ob, now=now)['line']}")
    lines.append("")
    lines.append("Countdowns are computed from the dates you supplied; "
                 "nothing here invents a date.")
    return with_disclaimer("\n".join(lines))


def portfolio_product_summary(portfolio: ContractPortfolio | None = None) -> str:
    """Portfolio summary via the real ContractPortfolio + disclaimer."""
    try:
        text = (portfolio or ContractPortfolio()).summary()
    except Exception as exc:  # noqa: BLE001
        text = f"portfolio unavailable ({exc})."
    return with_disclaimer(text)


# ── regulatory monitoring (wraps legal/regulatory.py) ───────────────────────

REGULATORY_DESCRIPTORS: dict[str, dict[str, Any]] = {
    code: {"code": reg.code, "name": reg.name, "site": reg.site,
           "blurb": reg.blurb, "typical_topics": list(reg.typical_topics)}
    for code, reg in REGULATORS.items()
}


def check_regulatory_updates(supplied_updates: list[dict[str, Any]],
                             *, profile: dict[str, Any] | None = None,
                             style: str = "full") -> list[str]:
    """Check SUPPLIED regulatory updates against watches/descriptors.

    ``supplied_updates`` are dicts for ``item_from_dict`` (regulator,
    title, summary, source_url, …). No fetching, no invented URLs —
    an update without a supplied URL falls back to the regulator's
    official homepage, exactly as the underlying module does.
    Every alert carries disclaimer + citations.
    """
    alerts: list[str] = []
    for raw in supplied_updates or []:
        item = item_from_dict(raw)
        if item is None:
            continue
        controls = controls_for(item)
        alert = alert_text(item, profile=profile, style=style)
        if controls and style == "full":
            alert += "\n\nRelated controls: " + "; ".join(
                c.obligation[:120] for c in controls[:2])
        alerts.append(with_disclaimer(alert))
    return alerts


def regulatory_calendar_product(months: int = 6,
                                *, now: float | None = None) -> str:
    """Upcoming recurring regulatory deadlines (wraps ``upcoming_deadlines``)."""
    entries = upcoming_deadlines(months, now=now)
    if not entries:
        return with_disclaimer(f"No recurring deadlines in the next {months} months.")
    lines = [f"📅 Regulatory calendar (next {months} months):"]
    for e in entries:
        d = datetime.fromtimestamp(e["date_ts"], tz=timezone.utc).date().isoformat()
        lines.append(f"• {d} — [{e['regulator']}] {e['title']}")
    return with_disclaimer("\n".join(lines))


# ── legal aid (wraps legal/aid.py) ──────────────────────────────────────────

_LANGUAGE_ALIASES = {"pidgin": "pcm", "naija": "pcm", "nigerian pidgin": "pcm",
                     "english": "en", "yoruba": "yo", "hausa": "ha",
                     "igbo": "ig", "ibo": "ig", "en": "en", "pcm": "pcm",
                     "yo": "yo", "ha": "ha", "ig": "ig"}


def legal_aid(question: str, language: str = "en",
              *, style: str = "full") -> dict[str, Any]:
    """Plain-language legal information in en/pidgin/yo/ha/ig.

    Wraps :func:`answer_legal_question`. If the aid engine ever reports
    itself unavailable, the answer says so honestly — never faked.
    Every output carries disclaimer + citations.
    """
    lang = _LANGUAGE_ALIASES.get((language or "en").strip().lower(), "en")
    if lang not in _AID_LANGUAGES:
        lang = "en"
    try:
        answer = _answer_legal_question(question, lang)
        text = with_disclaimer(_format_answer(answer, style=style))
        return {"question": question, "language": lang,
                "escalate": answer.escalate,
                "citations": list(answer.citations),
                "text": text, "available": True}
    except Exception as exc:  # noqa: BLE001
        return {"question": question, "language": lang, "escalate": True,
                "citations": [], "available": False,
                "text": with_disclaimer(
                    f"legal aid is unavailable right now ({exc}). "
                    "See the referral list for a real lawyer.")}


def legal_referral(topic: str = "") -> str:
    """Real Nigerian legal-aid institutions (wraps ``refer_lawyer``)."""
    return with_disclaimer(_refer_lawyer(topic))


def render_letter_product(kind: str, **fields: str) -> str:
    """Fill-in-the-blank letter template (wraps ``render_letter``)."""
    kinds = _letter_kinds()
    if (kind or "").lower() not in kinds:
        return with_disclaimer(
            f"Unknown letter kind '{kind}'. Available: {', '.join(kinds)}.")
    return with_disclaimer(_render_letter(kind, **fields))


# ── flow ────────────────────────────────────────────────────────────────────

LEGAL_FLOW_USAGE = (
    "nm legal — legal product\n"
    "  nm legal review <contract text> — contract risk review\n"
    "  nm legal obligations — obligation tracker (supplied dates only)\n"
    "  nm legal regcheck — regulatory monitoring (supplied updates only)\n"
    "  nm legal aid [en|pidgin|yo|ha|ig] <question> — plain-language legal info\n"
    "  nm legal refer [topic] — real Nigerian legal-aid institutions\n"
    "Every output: 'not legal advice' disclaimer + DoNotPay/NBA citations."
)


def legal_flow(command: str) -> str:
    """The ``nm legal`` flow. Never raises; every output carries disclaimer."""
    try:
        rest = (command or "").strip()
        low = rest.lower()
        if not low or low in ("help", "usage"):
            return with_disclaimer(LEGAL_FLOW_USAGE)
        if low.startswith("review "):
            return review_contract_product(rest[len("review "):].strip())["text"]
        if low == "obligations":
            return with_disclaimer(
                "Obligation tracker needs supplied obligations — pass them "
                "to format_obligations([...]) with real due dates. "
                "No dates are invented.")
        if low == "regcheck":
            return with_disclaimer(
                "Regulatory monitoring needs supplied updates — pass them "
                "to check_regulatory_updates([...]). No fetching, no "
                "invented URLs.")
        if low.startswith("aid "):
            body = rest[len("aid "):].strip()
            parts = body.split(None, 1)
            if len(parts) == 2 and parts[0].lower() in _LANGUAGE_ALIASES:
                return legal_aid(parts[1], parts[0])["text"]
            return legal_aid(body)["text"]
        if low.startswith("refer"):
            return legal_referral(rest[len("refer"):].strip())
        return with_disclaimer("unknown legal subcommand.\n" + LEGAL_FLOW_USAGE)
    except Exception as exc:  # noqa: BLE001 — flow never raises
        return with_disclaimer(f"legal flow hit an error ({exc}).")
