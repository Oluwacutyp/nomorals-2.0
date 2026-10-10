"""Contract risk review — letter grades + Nigerian playbooks.

Positioning (factual, not moralizing): this module provides legal
*information*, never legal *advice*. It extracts clauses, explains them
in plain language, and scores them against playbooks. It never tells the
owner what to do about a specific situation, never claims to be or
perform like a lawyer, and always points at real lawyers for
consequential decisions (DoNotPay FTC $193K precedent; NBA 2024 AI
guidelines: technology supports professional judgment, never replaces
it). Every public output carries :data:`DISCLAIMER`.

Scope is deliberate: Nigerian contract types (Lagos tenancy, Nigerian
employment, freelance service agreements). We do not out-corpus
LawPavilion — we out-UX them: conversational, WhatsApp-native,
consumer-facing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

__all__ = [
    "DISCLAIMER",
    "CONTRACT_TYPES",
    "REVIEW_STYLES",
    "CLAUSE_CATEGORIES",
    "Finding",
    "Review",
    "ReviewDiff",
    "ContractMeta",
    "detect_contract_type",
    "review_contract",
    "format_review",
    "format_diff",
    "extract_clauses",
    "extract_metadata",
    "extract_rights",
    "categorize_clauses",
    "compare_reviews",
    "plain_english",
    "information_only_check",
    "control_contract",
]

#: Carried on every review output. Legal information, not legal advice.
DISCLAIMER = (
    "ℹ️ Legal information, not legal advice: this review explains clauses "
    "in plain language. It does not tell you what to do, and Devon is not "
    "a lawyer. For any consequential decision (signing, disputing, "
    "terminating, or relying on a clause), speak to a qualified Nigerian "
    "lawyer."
)

#: Phrases that cross the information→advice line. Explanations are
#: written to avoid these; :func:`information_only_check` enforces it.
_ADVICE_PHRASES = (
    "you should",
    "you must",
    "you need to",
    "don't sign",
    "do not sign",
    "never sign",
    "sign it",
    "reject the",
    "i recommend",
    "i advise",
    "my advice",
    "you ought to",
)

CONTRACT_TYPES = ("tenancy", "employment", "freelance")

# ── severities / scoring ──────────────────────────────────────────────

CRITICAL = "critical"
HIGH = "high"
MEDIUM = "medium"
LOW = "low"

_DEDUCTION = {CRITICAL: 25, HIGH: 15, MEDIUM: 8, LOW: 3}
#: Missing-clause ("gap") findings deduct less: absence of a good clause is
#: weaker evidence than presence of a bad one.
_GAP_DEDUCTION = 2
_SEV_ICON = {CRITICAL: "🔴", HIGH: "🟠", MEDIUM: "🟡", LOW: "⚪"}


def _grade(score: int) -> str:
    bands = (
        (97, "A+"), (93, "A"), (90, "A-"),
        (87, "B+"), (83, "B"), (80, "B-"),
        (77, "C+"), (73, "C"), (70, "C-"),
        (67, "D+"), (63, "D"), (60, "D-"),
    )
    for cutoff, letter in bands:
        if score >= cutoff:
            return letter
    return "F"


# ── models ────────────────────────────────────────────────────────────


@dataclass
class Finding:
    """One flagged clause. Information only — never advice."""

    rule_id: str
    title: str
    severity: str
    clause: str  # verbatim snippet from the document (may be "")
    explanation: str  # plain-language information about the clause
    playbook_ref: str = ""
    anomaly: bool = False  # deviation from playbook norms, not a direct rule
    suggestion: str = ""  # information-only: how similar agreements handle it
    citation_ok: bool = True  # snippet verified verbatim against the document
    gap: bool = False  # True when the finding is a *missing* clause (absence),
    # not a bad present clause — scored lighter (2 pts), because absence of
    # a good clause is weaker evidence than presence of a bad one.

    def headline(self) -> str:
        icon = _SEV_ICON.get(self.severity, "⚪")
        tag = " (unusual clause)" if self.anomaly else ""
        return f"{icon} {self.title}{tag}"


@dataclass
class Review:
    contract_type: str
    score: int
    grade: str
    findings: list[Finding] = field(default_factory=list)
    clauses_seen: int = 0
    disclaimer: str = DISCLAIMER
    rights: list[str] = field(default_factory=list)  # rights the text grants (Do Not Sign pattern)

    @property
    def needs_attention(self) -> int:
        return sum(1 for f in self.findings
                   if f.severity in (CRITICAL, HIGH, MEDIUM))


# ── clause extraction ─────────────────────────────────────────────────


_HEADING_RE = re.compile(
    r"^\s*(?:(?:clause|article|section|schedule|part)\s+[\w.\-]+"
    r"|(?:\d{1,2}(?:\.\d{1,2})*)\.?"
    r"|(?:[IVXLCDM]{1,6})\.?)\s*[:.)–—-]?\s*(?P<title>.{0,80})$",
    re.IGNORECASE,
)
_ALLCAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9\s/&,'-]{4,60}$")
_DURATION_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?", re.IGNORECASE)
_MONEY_RE = re.compile(r"(?:₦|NGN|N)\s?([\d,]+(?:\.\d{1,2})?)", re.IGNORECASE)


@dataclass
class Clause:
    heading: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.heading} {self.body}".strip()


def extract_clauses(doc_text: str) -> list[Clause]:
    """Split a contract into clauses on numbered/all-caps headings.

    Never raises on weird input; worst case returns one big clause.
    """
    try:
        lines = (doc_text or "").splitlines()
    except Exception:
        return [Clause(heading="", body="")]
    clauses: list[Clause] = []
    cur_heading = ""
    cur_body: list[str] = []

    def flush() -> None:
        body = " ".join(l.strip() for l in cur_body if l.strip())
        if cur_heading or body:
            clauses.append(Clause(heading=cur_heading.strip(), body=body))

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        m = _HEADING_RE.match(line)
        if m and len(stripped) < 100:
            flush()
            cur_heading, cur_body = stripped, []
        elif (_ALLCAPS_RE.match(stripped) and len(stripped) < 60
                and len(stripped.split()) <= 8):
            flush()
            cur_heading, cur_body = stripped, []
        else:
            cur_body.append(stripped)
    flush()
    if not clauses:
        body = " ".join(l.strip() for l in lines if l.strip())
        clauses.append(Clause(heading="", body=body))
    return clauses


def _durations_months(text: str) -> list[float]:
    out = []
    for num, unit in _DURATION_RE.findall(text or ""):
        try:
            v = float(num)
        except ValueError:
            continue
        factor = {"day": 1 / 30, "week": 1 / 4.345, "month": 1, "year": 12}[unit.lower()]
        out.append(round(v * factor, 2))
    return out


def _has(text: str, *patterns: str) -> bool:
    """Regex search with whitespace normalized (PDF/OCR line breaks)."""
    t = re.sub(r"\s+", " ", text or "")
    return any(re.search(p, t, re.IGNORECASE) for p in patterns)


def _money(text: str) -> list[float]:
    vals = []
    for raw in _MONEY_RE.findall(text or ""):
        try:
            vals.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    return vals


# ── rules ─────────────────────────────────────────────────────────────


@dataclass
class Rule:
    """One playbook rule. ``check`` returns a Finding or None."""

    rule_id: str
    title: str
    severity: str
    types: tuple[str, ...]
    check: Callable[[str, list[Clause]], Optional[Finding]]
    playbook_ref: str = ""


def _mk(rule_id: str, title: str, severity: str, clause: str,
        explanation: str, playbook_ref: str = "",
        anomaly: bool = False, gap: bool = False) -> Finding:
    return Finding(rule_id=rule_id, title=title, severity=severity,
                   clause=clause.strip()[:400], explanation=explanation,
                   playbook_ref=playbook_ref, anomaly=anomaly, gap=gap)


# — Lagos tenancy playbook ———————————————————————————————————————————

def _t_rent_increase(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"rent.{0,40}(increas|review|raise|adjust)",
               r"(increas|rais|review|adjust).{0,40}rent", r"rent review"):
        return None
    if _has(text, r"rent.{0,60}(month|year).{0,30}notice.{0,20}(\d+)",
            r"notice of.{0,20}(\d+).{0,20}(month|year).{0,20}rent"):
        return None  # a notice period is stated
    snippet = next((c.text for c in clauses
                    if _has(c.text, r"rent.{0,40}(increas|review|raise)")), "")[:300]
    return _mk(
        "T-01", "Rent increase with no stated notice period", HIGH, snippet,
        "This agreement lets the rent go up but does not say how much "
        "notice you get first. In Lagos, rent reviews are normal, but a "
        "clear notice period (many agreements use 3+ months) gives time to "
        "plan. The Lagos Tenancy Law 2011 sets rules around rent review "
        "and notices; a qualified lawyer can confirm what applies here.",
        playbook_ref="Lagos Tenancy Law 2011 — rent review / notice provisions",
    )


def _t_service_charge(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"service charge|estate dues|maintenance (fee|levy)"):
        return None
    if _has(text, r"service charge.{0,40}(₦|NGN|N\s?[\d,]+)",
            r"(₦|NGN).{0,20}service charge"):
        return None  # amount or cap is stated
    if _has(text, r"service charge.{0,60}(cap|fixed|maximum|not exceed)"):
        return None
    snippet = next((c.text for c in clauses if _has(c.text, r"service charge")), "")[:300]
    return _mk(
        "T-02", "Service charge with no amount or cap", MEDIUM, snippet,
        "A service charge is mentioned but no amount, formula, or cap is "
        "given. Service charges in Lagos estates vary widely and can be a "
        "large part of the real cost of renting. It is common for "
        "agreements to state the amount or how it is calculated.",
        playbook_ref="Lagos tenancy market practice — service charges stated or capped",
    )


def _t_notice_to_quit(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"notice to quit|quit notice|vacat"):
        return None
    years = _durations_months(text)
    months = [d for d in years if 0 < d <= 120]
    if not months:
        return None
    # find notice periods stated near quit language
    shortest = min(months)
    snippet = next((c.text for c in clauses if _has(c.text, r"notice to quit")), "")[:300]
    if shortest < 3 and _has(text, r"year|annual|12 month"):
        return _mk(
            "T-03", "Short notice-to-quit for a yearly tenancy", HIGH, snippet,
            "The shortest notice period here is under 3 months, but this "
            "looks like a yearly tenancy. Under the Lagos Tenancy Law 2011, "
            "the minimum notice to quit for a yearly tenancy is 3 months "
            "(monthly tenancies: 1 month; weekly: 7 days). A clause giving "
            "less than the statutory minimum may not be enforceable — a "
            "lawyer can confirm.",
            playbook_ref="Lagos Tenancy Law 2011 — statutory notice-to-quit periods",
        )
    return None


def _t_agency_fee(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"agency fee|commission|agent.{0,20}fee"):
        return None
    amounts = _money(text)
    if amounts and all(a <= 0.10 * max(amounts) or a < 100000 for a in amounts):
        return None
    pcts = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\s*%", text) if _has(text, r"agency|commission")]
    snippet = next((c.text for c in clauses if _has(c.text, r"agency fee|commission")), "")[:300]
    if pcts and max(pcts) > 10:
        return _mk(
            "T-04", "Agency fee above the usual 10%", MEDIUM, snippet,
            "The agency fee here is above 10% of the rent. In Lagos, 10% "
            "of annual rent is the widely used market rate for agency "
            "fees. Anything above that is unusual and worth querying.",
            playbook_ref="Lagos market practice — 10% agency fee norm",
        )
    if not amounts and not pcts:
        return _mk(
            "T-04", "Agency fee mentioned without an amount", LOW, snippet,
            "An agency fee is mentioned but the amount is not stated, so "
            "the total cost of this tenancy is unclear. Lagos market "
            "practice is typically 10% of annual rent.",
            playbook_ref="Lagos market practice — 10% agency fee norm",
        )
    return None


def _t_auto_renew(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"automatically renew|auto.?renew|deemed to be renewed"):
        return None
    snippet = next((c.text for c in clauses if _has(c.text, r"renew")), "")[:300]
    return _mk(
        "T-05", "Automatic renewal clause", MEDIUM, snippet,
        "The tenancy renews automatically unless someone acts. Auto-renewal "
        "is common, but it can lock in the same rent (or a new rent) "
        "without a fresh negotiation. Check what notice is needed to stop "
        "the renewal.",
        playbook_ref="Tenancy playbook — renewal mechanics should be explicit",
    )


def _t_missing_repairs(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"rent|tenancy|landlord|tenant"):
        return None
    if _has(text, r"repair|maintenance|structural|dilapidation"):
        return None
    return _mk(
        "T-06", "No repairs / maintenance clause found", LOW, "",
        "Nothing in this agreement says who handles repairs and "
        "maintenance. Most tenancy agreements assign this (often: "
        "landlord for structural, tenant for minor). Its absence is not "
        "fatal, but it is a common source of disputes.",
        playbook_ref="Tenancy playbook — repairs allocation",
        gap=True
    )


def _t_statutory_waiver(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"waive.{0,30}(right|rights)|rights.{0,20}waived|contracting out"):
        return None
    snippet = next((c.text for c in clauses if _has(c.text, r"waive")), "")[:300]
    return _mk(
        "T-99", "Clause asks a party to waive statutory rights", CRITICAL, snippet,
        "This clause asks someone to give up rights given by law. Under "
        "the Lagos Tenancy Law 2011 and Nigerian law generally, parties "
        "usually cannot contract out of statutory protections — such "
        "clauses are often unenforceable. This is one to run past a lawyer.",
        playbook_ref="Lagos Tenancy Law 2011 — statutory protections",
        anomaly=True,
    )


def _t_unlimited_liability(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"unlimited liability|liab.{0,25}(without|no).{0,10}limit"
                      r"|without.{0,10}limit.{0,25}liab|liable for all|indemnif"):
        return None
    snippet = next((c.text for c in clauses if _has(c.text, r"unlimited liability|indemnif")), "")[:300]
    return _mk(
        "T-98", "Unlimited liability / broad indemnity", HIGH, snippet,
        "Someone here takes on unlimited liability or a very broad "
        "indemnity. Most consumer and tenancy agreements cap liability "
        "at something reasonable (e.g. the annual rent). An uncapped "
        "indemnity is unusual and worth a lawyer's look.",
        playbook_ref="Tenancy playbook — liability caps",
        anomaly=True,
    )


# — Nigerian employment playbook —————————————————————————————————————

def _e_probation(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"probation"):
        return None
    durs = _durations_months(" ".join(c.text for c in clauses if _has(c.text, r"probation")))
    if not durs:
        return None
    longest = max(durs)
    snippet = next((c.text for c in clauses if _has(c.text, r"probation")), "")[:300]
    if longest > 6:
        return _mk(
            "E-01", f"Probation of {longest:g} months is unusually long", HIGH, snippet,
            "Probation here runs longer than 6 months. Nigerian employers "
            "commonly use 3–6 months; longer probation keeps employment "
            "terms uncertain for an extended period. There is no single "
            "statutory maximum, but anything beyond 6 months stands out "
            "against market practice.",
            playbook_ref="Nigerian employment market practice — 3–6 month probation norm",
        )
    return None


def _e_termination_notice(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"terminat"):
        return None
    if not _has(text, r"notice"):
        return _mk(
            "E-02", "Termination with no notice period stated", HIGH, "",
            "Employment here can be terminated but no notice period is "
            "stated. The Nigerian Labour Act sets minimum notice periods "
            "that scale with length of service (roughly: 1 week under 2 "
            "years, 2 weeks for 2–5 years, 1 month for 5+ years). A clear "
            "notice clause protects both sides.",
            playbook_ref="Labour Act, Cap L1 LFN 2004 — minimum notice periods",
        )
    return None


def _e_non_compete(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"non.?compete|restraint of trade|not compete"):
        return None
    durs = _durations_months(" ".join(c.text for c in clauses if _has(c.text, r"non.?compete|restraint")))
    snippet = next((c.text for c in clauses if _has(c.text, r"non.?compete|restraint")), "")[:300]
    longest = max(durs) if durs else 0
    if longest > 12:
        return _mk(
            "E-03", f"Non-compete of {longest:g} months is unusually long", HIGH, snippet,
            "The non-compete runs longer than 12 months. Nigerian courts "
            "test restraints for reasonableness (duration, geography, "
            "scope) — long or open-ended restraints are frequently treated "
            "as unreasonable. A lawyer can assess whether this one would "
            "hold up.",
            playbook_ref="Nigerian common law — reasonableness test for restraints of trade",
        )
    if _has(text, r"non.?compete.{0,80}(worldwide|globally|anywhere|without (any )?limit|unlimited)"):
        return _mk(
            "E-03", "Non-compete with unlimited geography", HIGH, snippet,
            "The non-compete has no geographic limit. Nigerian courts weigh "
            "geography when testing reasonableness — a worldwide restraint "
            "on an ordinary employee is unusual.",
            playbook_ref="Nigerian common law — reasonableness test for restraints of trade",
            anomaly=True,
        )
    return None


def _e_pension(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"employ|salary|wage|remuneration"):
        return None
    if _has(text, r"pension|PenCom|RSA|retirement savings|contributory"):
        return None
    return _mk(
        "E-04", "No pension / Contributory Pension Scheme mention", MEDIUM, "",
        "This employment agreement says nothing about pension. Under the "
        "Pension Reform Act 2014, employers with 15+ employees must run "
        "the Contributory Pension Scheme (employer 10%, employee 8% of "
        "monthly emoluments) via a licensed PFA. Its absence does not "
        "remove the legal duty, but it is worth confirming.",
        playbook_ref="Pension Reform Act 2014 — 10% employer / 8% employee",
    )


def _e_salary_deduction(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"deduct"):
        return None
    if _has(text, r"deduct.{0,60}(statutory|tax|PAYE|pension|consent)"):
        return None
    snippet = next((c.text for c in clauses if _has(c.text, r"deduct")), "")[:300]
    return _mk(
        "E-05", "Broad salary-deduction clause", MEDIUM, snippet,
        "The employer can deduct from salary without clear limits or the "
        "employee's consent spelled out. The Labour Act restricts "
        "deductions (generally to statutory ones and agreed items); broad "
        "deduction clauses are a common dispute point.",
        playbook_ref="Labour Act — permitted deductions from wages",
    )


# — Freelance / service playbook —————————————————————————————————————

def _f_payment_terms(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"payment|invoice|fee|compensation"):
        return None
    if not _has(text, r"\d+\s*days?|net\s?\d+|within.{0,20}days?"):
        return _mk(
            "F-01", "No payment deadline stated", HIGH, "",
            "Money is discussed but no payment deadline is set (e.g. "
            "'within 14 days of invoice'). Without a deadline, late "
            "payment has no contractual trigger. Freelance agreements "
            "commonly use 7–30 day terms.",
            playbook_ref="Freelance playbook — explicit payment terms",
        )
    durs = _durations_months(" ".join(c.text for c in clauses if _has(c.text, r"payment|invoice")))
    if durs and max(durs) > 2.1:  # > ~60 days
        snippet = next((c.text for c in clauses if _has(c.text, r"payment|invoice")), "")[:300]
        return _mk(
            "F-01", "Payment terms longer than 60 days", MEDIUM, snippet,
            "Payment takes more than 60 days after invoicing. That is a "
            "long cash-flow gap for a freelancer; 7–30 days is the common "
            "range in service agreements.",
            playbook_ref="Freelance playbook — 7–30 day norm",
        )
    return None


def _f_ip_ownership(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"intellectual property|IP\b|copyright|ownership|work product|deliverable"):
        return None
    if _has(text, r"(assign|transfer|vest).{0,40}(client|company)|ownership.{0,30}(client|company)"):
        return None  # clear assignment to client
    if _has(text, r"(retain|remain).{0,40}(freelancer|contractor|consultant)"):
        return None  # clear retention by freelancer
    snippet = next((c.text for c in clauses if _has(c.text, r"intellectual property|ownership|deliverable")), "")[:300]
    return _mk(
        "F-02", "IP ownership is ambiguous", HIGH, snippet,
        "Intellectual property is mentioned but it is not clear who owns "
        "the finished work. Under Nigerian copyright law, the creator "
        "generally owns the work unless it is assigned in writing — "
        "silence here is the classic freelance dispute. Clear assignment "
        "language (to the client, or retained by the freelancer with a "
        "licence) is the norm.",
        playbook_ref="Copyright Act 2022 — ownership defaults; assignment in writing",
    )


def _f_termination(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"terminat"):
        return None
    if not _has(text, r"either party|both parties|mutual"):
        if _has(text, r"client may terminat|company may terminat"):
            snippet = next((c.text for c in clauses if _has(c.text, r"terminat")), "")[:300]
            return _mk(
                "F-03", "One-sided termination right", MEDIUM, snippet,
                "Only the client can terminate early. Balanced agreements "
                "give both sides a termination right (often with 14–30 "
                "days' notice). One-sided termination leaves the "
                "freelancer exposed if the project stalls.",
                playbook_ref="Freelance playbook — mutual termination rights",
            )
    return None


def _f_late_fees(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"payment|invoice"):
        return None
    if _has(text, r"late (fee|payment|charge)|interest.{0,20}late|penalt"):
        return None
    return _mk(
        "F-04", "No late-payment provision", LOW, "",
        "There is no late fee or interest on overdue invoices. Late-payment "
        "terms (e.g. a small monthly percentage) are a common nudge that "
        "keeps invoices paid on time.",
        playbook_ref="Freelance playbook — late-payment terms",
        gap=True
    )


def _f_scope_creep(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"unlimited revision|any (and all )?changes|additional work.{0,20}no (extra|additional) (charge|fee|cost)|scope.{0,20}unlimited"):
        return None
    snippet = next((c.text for c in clauses if _has(c.text, r"revision|scope")), "")[:300]
    return _mk(
        "F-05", "Open-ended scope / unlimited revisions", MEDIUM, snippet,
        "The scope of work has no boundary (unlimited revisions or "
        "changes at no extra cost). This is the classic scope-creep "
        "setup. Service agreements commonly cap revisions and price "
        "extra work separately.",
        playbook_ref="Freelance playbook — bounded scope",
    )


# — new rules: CUAD-taxonomy coverage (governing law, confidentiality,
# indemnification, assignment, dispute resolution, force majeure) ————

def _t_late_rent_penalty(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if not _has(text, r"interest.{0,40}late.{0,20}rent|penalt.{0,40}late.{0,20}rent"
                      r"|late.{0,20}rent.{0,40}(interest|penalt|surcharge)"):
        return None
    snippet = next((c.text for c in clauses
                    if _has(c.text, r"late.{0,20}rent|interest.{0,20}penalt")), "")[:300]
    return _mk(
        "T-07", "Late-rent penalty / interest clause", MEDIUM, snippet,
        "Late rent attracts a penalty or interest here. Lagos tenancy "
        "agreements often include a modest late fee, but the rate and how "
        "it compounds should be stated clearly — an open-ended penalty can "
        "grow fast.",
        playbook_ref="Tenancy playbook — late-payment terms should be explicit",
        anomaly=False,
    )


def _t_no_governing_law(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"governed by|governing law|jurisdiction|laws of.{0,20}(lagos|nigeria|state)"):
        return None
    if not _has(text, r"tenancy|rent|landlord|tenant|lease"):
        return None
    return _mk(
        "T-08", "No governing-law / jurisdiction clause", MEDIUM, "",
        "This agreement never says which law governs it or which court "
        "settles disputes. Tenancy disputes in Lagos normally sit under "
        "the Lagos Tenancy Law 2011, but spelling it out avoids "
        "arguments later. Most professionally drafted agreements include "
        "one line on this.",
        playbook_ref="Tenancy playbook — governing law stated",
        gap=True
    )


def _e_confidentiality(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"confidential"):
        return None
    if not _has(text, r"employ"):
        return None
    return _mk(
        "E-06", "No confidentiality clause found", MEDIUM, "",
        "This employment agreement has no confidentiality clause. Many "
        "Nigerian employers — especially in tech, finance, and roles "
        "handling customer data — include one covering trade secrets and "
        "client information, usually surviving termination. Its absence is "
        "not a legal defect, but it is a gap against market practice.",
        playbook_ref="Employment playbook — confidentiality / IP protection",
        gap=True
    )


def _e_working_hours(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"working hours|work hours|overtime|resumption|8am|9am"):
        return None
    if not _has(text, r"employ|salary"):
        return None
    return _mk(
        "E-07", "No working-hours / overtime terms", LOW, "",
        "Nothing here sets working hours, overtime, or rest days. The "
        "Labour Act limits normal working hours and provides for rest "
        "periods; agreements commonly spell these out so expectations "
        "match on both sides.",
        playbook_ref="Labour Act — hours of work and rest periods",
        gap=True
    )


def _f_confidentiality(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"confidential|non.?disclosure|NDA"):
        return None
    return _mk(
        "F-06", "No confidentiality / NDA terms", MEDIUM, "",
        "This service agreement has no confidentiality or non-disclosure "
        "terms. Freelance work often exposes the freelancer to the "
        "client's business information (and vice versa); agreements "
        "commonly include a mutual confidentiality clause.",
        playbook_ref="Freelance playbook — mutual confidentiality",
        gap=True
    )


def _f_dispute_resolution(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"arbitration|dispute.{0,30}resol|mediation"):
        return None
    has_govlaw = _has(text, r"jurisdiction|court of|governed by|governing law")
    detail = ("It names a governing law but no forum or process for "
              "resolving disputes. " if has_govlaw else
              "Nothing here says how disputes get settled or which law governs. ")
    return _mk(
        "F-07", "No dispute-resolution mechanism", MEDIUM, "",
        detail + "Service agreements commonly pick arbitration (e.g. Lagos "
        "Court of Arbitration) or mediation, plus the governing law. "
        "Without an agreed forum, a dispute over payment or scope has "
        "nowhere agreed to go.",
        playbook_ref="Freelance playbook — dispute resolution stated",
        gap=True
    )


def _f_liability_cap_missing(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"liab.{0,30}cap|cap.{0,30}liab|liability.{0,30}limit|limit.{0,30}liability"
                  r"|unlimited liability|liab.{0,25}(without|no).{0,10}limit"):
        return None  # a cap — or unlimited liability — is stated (T-98 covers the latter)
    if not _has(text, r"liab|indemnif|warrant"):
        return None
    return _mk(
        "F-08", "Liability mentioned but never capped or excluded", MEDIUM, "",
        "Liability or indemnity is discussed but no cap is stated. Service "
        "agreements commonly cap the freelancer's liability at the fees "
        "paid (or a multiple). An uncapped exposure is worth a lawyer's "
        "look before signing.",
        playbook_ref="Freelance playbook — liability caps",
    )


def _f_kill_fee(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"kill fee|cancellation fee|termination fee|early termination.{0,30}(fee|payment)"):
        return None
    if not _has(text, r"terminat"):
        return None
    return _mk(
        "F-09", "No cancellation / kill fee", LOW, "",
        "The agreement can be terminated but says nothing about payment "
        "for work already started. Many freelance agreements include a "
        "kill fee (e.g. a percentage of the project fee) so cancelled "
        "work is not done for free.",
        playbook_ref="Freelance playbook — cancellation compensation",
        gap=True
    )


# — universal rules (all contract types) ——————————————————————————————

def _u_auto_renew(text: str, clauses: list[Clause]) -> Optional[Finding]:
    """Universal auto-renewal detection (T-05 is tenancy-scoped)."""
    return _t_auto_renew(text, clauses)


def _u_foreign_governing_law(text: str, clauses: list[Clause]) -> Optional[Finding]:
    m = re.search(r"(?:governed by|governing law).{0,60}(england|wales|new york|delaware|"
                  r"singapore|dubai|united kingdom|laws of england)",
                  text or "", re.IGNORECASE)
    if not m:
        return None
    snippet = next((c.text for c in clauses
                    if _has(c.text, r"governed by|governing law")), "")[:300]
    return _mk(
        "U-01", "Foreign governing law in a Nigerian agreement", HIGH, snippet,
        "This Nigerian-facing agreement is governed by foreign law "
        "(e.g. England & Wales). That means disputes may be decided under "
        "a different legal system — and potentially in a foreign forum — "
        "which is costly for a Nigerian party. Cross-border governing law "
        "is normal for international deals, but unusual for a purely "
        "domestic one.",
        playbook_ref="Cross-border playbook — governing law should match the deal",
        anomaly=True,
    )


def _u_no_force_majeure(text: str, clauses: list[Clause]) -> Optional[Finding]:
    if _has(text, r"force majeure|act of god|unforeseen circumstances|beyond.{0,20}control"):
        return None
    if not _has(text, r"terminat|obligation|perform"):
        return None
    return _mk(
        "U-02", "No force-majeure clause", LOW, "",
        "Nothing covers what happens if performance becomes impossible "
        "(floods, strikes, government action). Most commercial agreements "
        "include a force-majeure clause suspending obligations during such "
        "events. Nigerian common law does not imply one automatically.",
        playbook_ref="General playbook — force majeure",
        gap=True
    )


def _u_assignment(text: str, clauses: list[Clause]) -> Optional[Finding]:
    # Only agreement-assignment counts — IP/copyright assignment is a
    # different concept and must not trip this rule.
    cands = [c.text for c in clauses
             if _has(c.text, r"\bassign")
             and not _has(c.text, r"intellectual property|copyright|work product|\bIP\b")]
    if not cands:
        return None
    if _has(" ".join(cands), r"assign.{0,60}(consent|written|not|prior)"):
        return None
    snippet = cands[0][:300]
    return _mk(
        "U-03", "Assignment allowed without clear consent terms", MEDIUM, snippet,
        "Assignment of rights/obligations is mentioned but the consent "
        "terms are unclear. Agreements commonly require the other party's "
        "prior written consent before assignment, so you know who you are "
        "dealing with throughout the term.",
        playbook_ref="General playbook — assignment with consent",
    )


#: Information-only "how similar agreements handle it" lines, keyed by
#: rule id. Applied to findings after the rule checks run. Phrasing is
#: deliberately neutral — never "you should".#: rule id. Applied to findings after the rule checks run. Phrasing is
#: deliberately neutral — never "you should".
_PLAYBOOK_SUGGESTIONS: dict[str, str] = {
    "T-01": ("Many Lagos tenancy agreements state a notice period of 3+ "
             "months before a rent increase takes effect."),
    "T-05": ("Auto-renewal is common, but check what notice is needed to "
             "stop the renewal — many agreements require written notice "
             "30–90 days before expiry."),
    "U-04": ("Auto-renewal is common, but check what notice is needed to "
             "stop the renewal — many agreements require written notice "
             "30–90 days before expiry."),
    "T-02": ("Many Lagos tenancy agreements state the service-charge amount "
             "or the formula used to compute it."),
    "T-03": ("The statutory minimum notice to quit for a yearly tenancy in "
             "Lagos is 3 months (1 month for monthly, 7 days for weekly)."),
    "T-04": ("10% of annual rent is the widely used Lagos market rate for "
             "agency fees."),
    "T-07": ("Many Lagos tenancy agreements state a late-rent penalty as a "
             "fixed amount or a modest monthly percentage, in writing."),
    "E-01": ("Nigerian employers commonly use 3–6 months of probation."),
    "E-02": ("Labour Act minimums scale with service: roughly 1 week under "
             "2 years, 2 weeks for 2–5 years, 1 month for 5+ years."),
    "E-03": ("Non-competes that hold up in Nigerian courts are usually "
             "bounded in time (often ≤12 months) and geography."),
    "F-01": ("Freelance agreements commonly use payment terms of 7–30 days "
             "from invoice."),
    "F-02": ("Common practice is a written IP assignment to the client on "
             "full payment, or freelancer ownership with a client licence."),
    "F-06": ("Mutual confidentiality clauses commonly cover both parties' "
             "business information during and after the engagement."),
    "F-07": ("Service agreements commonly pick arbitration (e.g. Lagos "
             "Court of Arbitration) or a named court, plus governing law."),
    "F-08": ("Liability is commonly capped at the fees paid under the "
             "agreement, or a stated multiple."),
    "U-03": ("Common practice is assignment only with the other party's "
             "prior written consent."),
}


# — playbook registry ————————————————————————————————————————————————

_PLAYBOOKS: dict[str, list[Rule]] = {
    "tenancy": [
        Rule("T-01", "Rent increase with no stated notice period", HIGH, ("tenancy",), _t_rent_increase),
        Rule("T-02", "Service charge with no amount or cap", MEDIUM, ("tenancy",), _t_service_charge),
        Rule("T-03", "Short notice-to-quit for a yearly tenancy", HIGH, ("tenancy",), _t_notice_to_quit),
        Rule("T-04", "Agency fee above the usual 10%", MEDIUM, ("tenancy",), _t_agency_fee),
        Rule("T-05", "Automatic renewal clause", MEDIUM, ("tenancy",), _t_auto_renew),
        Rule("T-06", "No repairs / maintenance clause found", LOW, ("tenancy",), _t_missing_repairs),
        Rule("T-07", "Late-rent penalty / interest clause", MEDIUM, ("tenancy",), _t_late_rent_penalty),
        Rule("T-08", "No governing-law / jurisdiction clause", MEDIUM, ("tenancy",), _t_no_governing_law),
        Rule("U-01", "Foreign governing law in a Nigerian agreement", HIGH, ("tenancy", "employment", "freelance"), _u_foreign_governing_law),
        Rule("U-02", "No force-majeure clause", LOW, ("tenancy", "employment", "freelance"), _u_no_force_majeure),
        Rule("U-03", "Assignment allowed without clear consent terms", MEDIUM, ("tenancy", "employment", "freelance"), _u_assignment),
        Rule("T-98", "Unlimited liability / broad indemnity", HIGH, ("tenancy", "employment", "freelance"), _t_unlimited_liability),
        Rule("T-99", "Clause asks a party to waive statutory rights", CRITICAL, ("tenancy", "employment", "freelance"), _t_statutory_waiver),
    ],
    "employment": [
        Rule("E-01", "Probation unusually long", HIGH, ("employment",), _e_probation),
        Rule("E-02", "Termination notice issue", HIGH, ("employment",), _e_termination_notice),
        Rule("E-03", "Non-compete unusually broad", HIGH, ("employment",), _e_non_compete),
        Rule("E-04", "No pension / Contributory Pension Scheme mention", MEDIUM, ("employment",), _e_pension),
        Rule("E-05", "Broad salary-deduction clause", MEDIUM, ("employment",), _e_salary_deduction),
        Rule("E-06", "No confidentiality clause found", MEDIUM, ("employment",), _e_confidentiality),
        Rule("E-07", "No working-hours / overtime terms", LOW, ("employment",), _e_working_hours),
        Rule("U-04", "Automatic renewal clause", MEDIUM, ("tenancy", "employment", "freelance"), _u_auto_renew),
        Rule("U-01", "Foreign governing law in a Nigerian agreement", HIGH, ("tenancy", "employment", "freelance"), _u_foreign_governing_law),
        Rule("U-02", "No force-majeure clause", LOW, ("tenancy", "employment", "freelance"), _u_no_force_majeure),
        Rule("U-03", "Assignment allowed without clear consent terms", MEDIUM, ("tenancy", "employment", "freelance"), _u_assignment),
        Rule("T-98", "Unlimited liability / broad indemnity", HIGH, ("tenancy", "employment", "freelance"), _t_unlimited_liability),
        Rule("T-99", "Clause asks a party to waive statutory rights", CRITICAL, ("tenancy", "employment", "freelance"), _t_statutory_waiver),
    ],
    "freelance": [
        Rule("F-01", "Payment terms issue", HIGH, ("freelance",), _f_payment_terms),
        Rule("F-02", "IP ownership ambiguous", HIGH, ("freelance",), _f_ip_ownership),
        Rule("F-03", "One-sided termination right", MEDIUM, ("freelance",), _f_termination),
        Rule("F-04", "No late-payment provision", LOW, ("freelance",), _f_late_fees),
        Rule("F-05", "Open-ended scope / unlimited revisions", MEDIUM, ("freelance",), _f_scope_creep),
        Rule("F-06", "No confidentiality / NDA terms", MEDIUM, ("freelance",), _f_confidentiality),
        Rule("F-07", "No dispute-resolution mechanism", MEDIUM, ("freelance",), _f_dispute_resolution),
        Rule("F-08", "Liability mentioned but never capped or excluded", MEDIUM, ("freelance",), _f_liability_cap_missing),
        Rule("F-09", "No cancellation / kill fee", LOW, ("freelance",), _f_kill_fee),
        Rule("U-04", "Automatic renewal clause", MEDIUM, ("tenancy", "employment", "freelance"), _u_auto_renew),
        Rule("U-01", "Foreign governing law in a Nigerian agreement", HIGH, ("tenancy", "employment", "freelance"), _u_foreign_governing_law),
        Rule("U-02", "No force-majeure clause", LOW, ("tenancy", "employment", "freelance"), _u_no_force_majeure),
        Rule("U-03", "Assignment allowed without clear consent terms", MEDIUM, ("tenancy", "employment", "freelance"), _u_assignment),
        Rule("T-98", "Unlimited liability / broad indemnity", HIGH, ("tenancy", "employment", "freelance"), _t_unlimited_liability),
        Rule("T-99", "Clause asks a party to waive statutory rights", CRITICAL, ("tenancy", "employment", "freelance"), _t_statutory_waiver),
    ],
}


def _playbook_names() -> dict[str, str]:
    return {
        "tenancy": "Lagos tenancy agreement",
        "employment": "Nigerian employment agreement",
        "freelance": "Freelance service agreement",
    }


# ── contract-type detection ─────────────────────────────────────────


_TYPE_SIGNALS: dict[str, tuple[str, ...]] = {
    "tenancy": (r"\btenant\b", r"\blandlord\b", r"\brent\b", r"\btenancy\b",
                r"\bpremises\b", r"\bservice charge\b"),
    "employment": (r"\bemploye[er]\b", r"\bsalary\b", r"\bprobation\b",
                   r"\bwages?\b", r"\bHR\b", r"\bjob (title|role)\b"),
    "freelance": (r"\bfreelanc", r"\bdeliverable", r"\binvoice\b",
                  r"\bcontractor\b", r"\bclient\b.{0,20}\bproject\b",
                  r"\bscope of work\b"),
}


def detect_contract_type(doc_text: str) -> str:
    """Best-effort contract-type detection from keyword signals."""
    text = doc_text or ""
    scores = {t: sum(1 for p in sigs if re.search(p, text, re.IGNORECASE))
              for t, sigs in _TYPE_SIGNALS.items()}
    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "tenancy"


# ── review ──────────────────────────────────────────────────────────


def review_contract(doc_text: str, contract_type: str = "auto") -> Review:
    """Extract clauses, score against the playbook, return a letter grade.

    Never raises on bad input — worst case is a low-information review.
    """
    try:
        text = doc_text or ""
        ctype = (contract_type or "auto").strip().lower()
        if ctype == "auto" or ctype not in _PLAYBOOKS:
            ctype = detect_contract_type(text)
        clauses = extract_clauses(text)
        findings: list[Finding] = []
        for rule in _PLAYBOOKS[ctype]:
            try:
                hit = rule.check(text, clauses)
            except Exception:  # noqa: BLE001 — one bad rule never kills a review
                continue
            if hit is not None:
                hit.rule_id = rule.rule_id
                hit.title = rule.title
                hit.severity = rule.severity
                if not hit.suggestion:
                    hit.suggestion = _PLAYBOOK_SUGGESTIONS.get(rule.rule_id, "")
                hit.citation_ok = _verbatim_in(hit.clause, text)
                findings.append(hit)
        score = max(0, 100 - sum(
            _GAP_DEDUCTION if f.gap else _DEDUCTION.get(f.severity, 0)
            for f in findings))
        order = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3}
        findings.sort(key=lambda f: (order.get(f.severity, 4), f.rule_id))
        rights = extract_rights(text)
        return Review(contract_type=ctype, score=score, grade=_grade(score),
                      findings=findings, clauses_seen=len(clauses),
                      rights=rights)
    except Exception:  # noqa: BLE001 — fail-closed review, never raise
        return Review(contract_type="tenancy", score=0, grade="F",
                      findings=[], clauses_seen=0)


# ── formatting ──────────────────────────────────────────────────────

#: Output styles for contract reviews.
#: - "full": the complete review (default)
#: - "compact": grade + one line per finding
#: - "triage": executive issues table (legalquants pattern — severity
#:   sorted, deviation + note columns)
REVIEW_STYLES = ("full", "compact", "triage")


def format_review(review: Review, style: str = "full") -> str:
    """WhatsApp-native, plain-language review. Disclaimer always attached.

    ``style`` is "full" (default), "compact", or "triage" (executive
    issues table sorted by severity).
    """
    style = (style or "full").lower()
    if style not in REVIEW_STYLES:
        style = "full"
    names = _playbook_names()
    if style == "triage":
        return _format_triage(review)
    lines = [
        f"📄 Contract review — {names.get(review.contract_type, review.contract_type)}",
        f"Grade: *{review.grade}* ({review.score}/100)",
    ]
    if review.needs_attention:
        lines.append(f"{review.needs_attention} clause{'s' if review.needs_attention != 1 else ''} need attention.")
    else:
        lines.append("No clauses flagged against the playbook.")
    if style == "compact":
        for f in review.findings:
            lines.append(f.headline())
    else:
        for f in review.findings:
            lines.append("")
            lines.append(f.headline())
            if f.clause:
                lines.append(f"_{f.clause[:220]}_")
            lines.append(f.explanation)
            if f.suggestion:
                lines.append(f"💡 {f.suggestion}")
    if review.rights:
        lines.append("")
        lines.append("*Rights the agreement states it gives you:*")
        for r in review.rights[:8]:
            lines.append(f"  ✓ {r}")
        if len(review.rights) > 8:
            lines.append(f"  …and {len(review.rights) - 8} more.")
    lines.append("")
    lines.append(review.disclaimer)
    return "\n".join(lines)


def _format_triage(review: Review) -> str:
    """Executive issues table — severity-sorted triage (legalquants pattern)."""
    names = _playbook_names()
    lines = [
        f"📄 Triage — {names.get(review.contract_type, review.contract_type)}",
        f"Grade: *{review.grade}* ({review.score}/100) · "
        f"{review.needs_attention} need attention · "
        f"{review.clauses_seen} clauses scanned",
        "",
        "| # | Risk | Clause topic | Verbatim ref | Note |",
        "|---|------|--------------|--------------|------|",
    ]
    for i, f in enumerate(review.findings, 1):
        icon = _SEV_ICON.get(f.severity, "⚪")
        ref = f.clause[:48].replace("|", "/").replace("\n", " ") if f.clause else "—"
        note = (f.suggestion or f.explanation).split(".")[0][:80].replace("|", "/")
        lines.append(f"| {i} | {icon} {f.severity} | {f.title} | _{ref}_ | {note} |")
    if not review.findings:
        lines.append("| — | — | _No issues flagged_ | — | — |")
    lines.append("")
    lines.append(review.disclaimer)
    return "\n".join(lines)


def information_only_check(text: str) -> list[str]:
    """Return advice-line phrases found in ``text`` (empty = clean)."""
    t = (text or "").lower()
    return [p for p in _ADVICE_PHRASES if p in t]


# ── verbatim citation check ─────────────────────────────────────────


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _verbatim_in(snippet: str, doc_text: str) -> bool:
    """Is the finding's clause snippet verbatim in the source document?

    Deterministic validation (contract-risk-assessment VAL step): a rule
    may only cite text that actually appears in the document.
    """
    if not snippet:
        return True  # nothing cited → nothing to verify
    return _norm_ws(snippet)[:200] in _norm_ws(doc_text)


# ── contract metadata (CUAD "Basic Info" categories) ───────────────────


@dataclass
class ContractMeta:
    """Structured facts about a contract (CUAD Basic Info pattern).

    Purely extracted strings — no interpretation, no advice.
    """
    parties: list[str] = field(default_factory=list)
    effective_date: str = ""
    expiry_date: str = ""
    governing_law: str = ""
    notice_period: str = ""
    auto_renew: bool = False
    term_length: str = ""
    amounts: list[str] = field(default_factory=list)


_BETWEEN_RE = re.compile(
    r"\bbetween\s+([A-Z][^,;\n]{2,80}?)\s+and\s+([A-Z][^,;\n]{2,80}?)(?:,|\n|$)",
    re.IGNORECASE,
)
_PARTY_ROLE_RE = re.compile(
    r'([A-Z][A-Za-z0-9 .,&\'()-]{2,70}?)\s*\(\s*["\']?(?:the\s+)?'
    r"(landlord|tenant|employer|employee|client|contractor|freelancer|"
    r"consultant|company|lessor|lessee)[\"']?\s*\)",
    re.IGNORECASE,
)
_GOVLAW_RE = re.compile(
    r"governed by.{0,80}?laws? of ([A-Z][A-Za-z ()-]{2,60})"
    r"|governing law.{0,20}?([A-Z][A-Za-z ()-]{2,60})",
    re.IGNORECASE,
)
_EFFDATE_RE = re.compile(
    r"(?:effective|commencement)\s+date.{0,20}?(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}"
    r"|\d{4}-\d{1,2}-\d{1,2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)
_TERM_RE = re.compile(
    r"(?:term|duration|tenancy|period).{0,40}?(\d+(?:\.\d+)?\s*(?:day|week|month|year)s?)",
    re.IGNORECASE,
)
_NOTICEPER_RE = re.compile(
    r"notice.{0,40}?(\d+(?:\.\d+)?\s*(?:day|week|month|year)s?)",
    re.IGNORECASE,
)


def extract_metadata(doc_text: str) -> ContractMeta:
    """Extract structured metadata: parties, dates, governing law, terms.

    CUAD "Basic Info" pattern — deterministic regex extraction over the
    whole document. Never raises; missing values stay empty.
    """
    meta = ContractMeta()
    try:
        text = doc_text or ""
        seen: set[str] = set()
        m = _BETWEEN_RE.search(text)
        if m:
            for g in (m.group(1), m.group(2)):
                g = re.sub(r"\s+", " ", g).strip(" ,.;")
                g = re.sub(r"^and\s+", "", g, flags=re.IGNORECASE)
                if g and g.lower() not in seen and len(g) < 90:
                    seen.add(g.lower())
                    meta.parties.append(g)
        for pm in _PARTY_ROLE_RE.finditer(text):
            g = re.sub(r"\s+", " ", pm.group(1)).strip(" ,.;")
            if (g and g.lower() not in seen and len(g) < 90
                    and "between" not in g.lower()):
                seen.add(g.lower())
                meta.parties.append(f"{g} ({pm.group(2).title()})")
        gm = _GOVLAW_RE.search(text)
        if gm:
            meta.governing_law = re.sub(
                r"\s+", " ", next(g for g in gm.groups() if g)).strip(" ,.;")[:80]
        em = _EFFDATE_RE.search(text)
        if em:
            meta.effective_date = em.group(1).strip()
        tm = _TERM_RE.search(text)
        if tm:
            meta.term_length = tm.group(1).strip()
        nm = _NOTICEPER_RE.search(text)
        if nm:
            meta.notice_period = nm.group(1).strip()
        meta.auto_renew = bool(_has(
            text, r"automatically renew|auto.?renew|deemed to be renewed"))
        meta.amounts = [m2.group(0).strip()[:40]
                        for m2 in _MONEY_RE.finditer(text)][:12]
        # crude expiry: "expir… on <date>" near an expiry keyword
        xm = re.search(
            r"expir.{0,60}?(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}"
            r"|\d{4}-\d{1,2}-\d{1,2})",
            text, re.IGNORECASE)
        if xm:
            meta.expiry_date = xm.group(1).strip()
    except Exception:  # noqa: BLE001 — metadata is best-effort
        pass
    return meta


# ── clause categorization (CUAD-style taxonomy) ────────────────────────

#: CUAD-style clause categories → keyword patterns. Contracts the model
#: has never seen still get a typed map of what they contain.
CLAUSE_CATEGORIES: dict[str, tuple[str, ...]] = {
    "parties": (r"\bparties\b", r"between.{0,20}and", r"hereinafter"),
    "term": (r"\bterm\b.{0,20}(year|month)", r"duration", r"commencement"),
    "payment": (r"\bpayment\b", r"\binvoice\b", r"\bfee\b", r"₦|NGN", r"\brent\b"),
    "termination": (r"\bterminat", r"\bcancel", r"notice to quit"),
    "renewal": (r"\brenew", r"\bexpir", r"extend.{0,20}term"),
    "confidentiality": (r"\bconfidential", r"non.?disclosure", r"\bNDA\b"),
    "ip_ownership": (r"intellectual property", r"\bcopyright\b", r"\bIP\b",
                     r"ownership", r"work product", r"deliverable"),
    "liability": (r"\bliab", r"\bindemnif", r"\bwarrant"),
    "non_compete": (r"non.?compete", r"restraint of trade", r"not compete",
                    r"non.?solicit"),
    "dispute_resolution": (r"\barbitration\b", r"\bmediation\b",
                           r"dispute.{0,20}resol", r"\bjurisdiction\b",
                           r"governing law", r"governed by"),
    "governing_law": (r"governing law", r"governed by", r"laws of"),
    "force_majeure": (r"force majeure", r"act of god", r"beyond.{0,15}control"),
    "assignment": (r"\bassign", r"transfer.{0,20}rights"),
    "repairs": (r"\brepair", r"\bmaintenance\b", r"dilapidation"),
    "notice": (r"\bnotice\b", r"\bnotify\b", r"in writing"),
    "pension": (r"\bpension\b", r"PenCom", r"\bRSA\b", r"retirement savings"),
    "probation": (r"\bprobation\b",),
    "scope": (r"scope of work", r"\bdeliverable", r"\brevision"),
}


def categorize_clauses(clauses: list[Clause]) -> dict[str, list[str]]:
    """Map clauses to CUAD-style categories. Returns category → headings.

    A clause can land in several categories. Never raises.
    """
    out: dict[str, list[str]] = {}
    try:
        for clause in clauses or []:
            text = clause.text
            label = clause.heading or text[:60]
            for cat, patterns in CLAUSE_CATEGORIES.items():
                if _has(text, *patterns):
                    out.setdefault(cat, [])
                    if label not in out[cat]:
                        out[cat].append(label)
    except Exception:  # noqa: BLE001
        pass
    return out


# ── plain-English clause rewriting ─────────────────────────────────────


#: Legalese → plain-language replacements (deterministic, offline).
_LEGALESE: tuple[tuple[str, str], ...] = (
    (r"\bhereinafter referred to as\b", "called"),
    (r"\bhereinafter\b", ""),
    (r"\bhereby\b", ""),
    (r"\bherein\b", "in this agreement"),
    (r"\bhereof\b", "of this agreement"),
    (r"\bhereto\b", "to this agreement"),
    (r"\bpursuant to\b", "under"),
    (r"\bin witness whereof\b", "signed"),
    (r"\bnotwithstanding anything to the contrary\b", "despite anything else in this agreement"),
    (r"\bforce majeure\b", "events nobody can control (floods, strikes, etc.)"),
    (r"\bindemnify and hold harmless\b", "protect against loss and cover the costs of"),
    (r"\bindemnify\b", "cover the losses of"),
    (r"\btime is of the essence\b", "deadlines are strict"),
    (r"\bjointly and severally\b", "together and individually"),
    (r"\bnull and void\b", "invalid"),
    (r"\bcease and desist\b", "stop"),
    (r"\bprior written consent\b", "written permission in advance"),
    (r"\bwithout prejudice\b", "without giving up any rights"),
    (r"\bipso facto\b", "automatically"),
    (r"\binter alia\b", "among other things"),
    (r"\bmutatis mutandis\b", "with the necessary changes"),
)


def plain_english(clause_text: str) -> str:
    """Rewrite legalese in plain language (deterministic, offline).

    This is a wording aid, not legal advice: it restates the same clause
    more readably. Never raises.
    """
    try:
        out = clause_text or ""
        for pat, repl in _LEGALESE:
            out = re.sub(pat, repl, out, flags=re.IGNORECASE)
        out = re.sub(r"\s{2,}", " ", out).strip()
        # split monster sentences on "; " for readability
        out = re.sub(r";\s+(?=[A-Z(])", ".\n", out)
        return out
    except Exception:  # noqa: BLE001
        return clause_text or ""


# ── rights extraction (Do Not Sign pattern) ────────────────────────────

_RIGHTS_RES = (
    r"shall be entitled to",
    r"is entitled to",
    r"are entitled to",
    r"shall have the right to",
    r"has the right to",
    r"have the right to",
    r"may terminate",
    r"may cancel",
    r"quiet enjoyment",
    r"entitled to (a )?refund",
    r"right to (a )?refund",
    r"may withhold",
    r"may deduct",
    r"shall not be liable",
    r"right to renew",
    r"option to renew",
    r"first right of refusal",
)


def extract_rights(doc_text: str, limit: int = 12) -> list[str]:
    """Extract rights the contract text *grants* (the Do Not Sign pattern).

    Reviews flag risks; this surfaces what the agreement gives the reader:
    refund rights, termination rights, renewal options, quiet enjoyment.
    Phrased as reported facts ("The agreement states…"), never advice.
    """
    rights: list[str] = []
    try:
        text = re.sub(r"\s+", " ", doc_text or "")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        for sent in sentences:
            if len(sent) < 25 or len(sent) > 400:
                continue
            if any(re.search(p, sent, re.IGNORECASE) for p in _RIGHTS_RES):
                clean = sent.strip()
                if clean not in rights:
                    rights.append(clean)
                if len(rights) >= limit:
                    break
    except Exception:  # noqa: BLE001
        pass
    return rights


# ── review comparison (amendment tracking) ──────────────────────────────


@dataclass
class ReviewDiff:
    """What changed between two reviews of (probably) the same contract."""
    old_grade: str
    new_grade: str
    old_score: int
    new_score: int
    new_findings: list[Finding] = field(default_factory=list)
    resolved_findings: list[Finding] = field(default_factory=list)
    carried_findings: list[Finding] = field(default_factory=list)

    def summary(self) -> str:
        delta = self.new_score - self.old_score
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
        return (f"{self.old_grade} ({self.old_score}) → {self.new_grade} "
                f"({self.new_score}) {arrow}{abs(delta)} · "
                f"{len(self.new_findings)} new, "
                f"{len(self.resolved_findings)} resolved, "
                f"{len(self.carried_findings)} carried over")


def compare_reviews(old: Review, new: Review) -> ReviewDiff:
    """Diff two reviews by rule id — for tracking amendments/negotiations.

    Shows which flagged issues the new version fixed, which appeared, and
    which carried over. Never raises.
    """
    try:
        old_map = {f.rule_id: f for f in old.findings}
        new_map = {f.rule_id: f for f in new.findings}
        diff = ReviewDiff(
            old_grade=old.grade, new_grade=new.grade,
            old_score=old.score, new_score=new.score,
            new_findings=[f for rid, f in new_map.items() if rid not in old_map],
            resolved_findings=[f for rid, f in old_map.items() if rid not in new_map],
            carried_findings=[f for rid, f in new_map.items() if rid in old_map],
        )
        return diff
    except Exception:  # noqa: BLE001
        return ReviewDiff(old_grade="?", new_grade="?", old_score=0, new_score=0)


def format_diff(diff: ReviewDiff) -> str:
    """Render a review comparison for chat. Never raises."""
    try:
        lines = ["🔁 Contract comparison", diff.summary()]
        if diff.resolved_findings:
            lines.append("")
            lines.append("✅ Fixed in the new version:")
            for f in diff.resolved_findings:
                lines.append(f"  • [{f.rule_id}] {f.title}")
        if diff.new_findings:
            lines.append("")
            lines.append("⚠️ New in this version:")
            for f in diff.new_findings:
                lines.append(f"  • {f.headline()} [{f.rule_id}]")
        if diff.carried_findings:
            lines.append("")
            lines.append("📌 Still present:")
            for f in diff.carried_findings:
                lines.append(f"  • {f.headline()} [{f.rule_id}]")
        lines.append("")
        lines.append(DISCLAIMER)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001
        return DISCLAIMER


# ── chat control ────────────────────────────────────────────────────


def _usage() -> str:
    return ("/contract review [tenancy|employment|freelance] [style] <paste the contract text> — "
            "letter grade + flagged clauses in plain language. style: full|compact|triage.\n"
            "/contract meta <paste the contract text> — extracted facts: parties, dates, governing law, amounts.\n"
            "/contract rights <paste the contract text> — rights the agreement grants you.\n"
            "/contract rewrite <paste one clause> — plain-English rewrite of the clause.\n"
            "/contract types — supported contract types.")


def control_contract(tail: str, context: Any = None, chat: Any = None,
                     sender_id: str = "", sender: str = "") -> str:
    """/contract — contract risk review. Owner-only; never raises."""
    try:
        rest = (tail or "").strip()
        if not rest or rest.lower() == "help":
            return _usage()
        low = rest.lower()
        if low == "types":
            names = _playbook_names()
            return ("Supported contract types:\n" +
                    "\n".join(f"• {k} — {v}" for k, v in names.items()) +
                    "\n\n" + DISCLAIMER)
        if low.startswith("meta"):
            text = rest[4:].strip()
            if len(text) < 40:
                return ("Paste the contract text after the command.\n" + _usage())
            meta = extract_metadata(text)
            lines = ["📋 Contract facts (extracted, not interpreted):"]
            lines.append(f"Parties: {', '.join(meta.parties) or '—'}")
            lines.append(f"Effective date: {meta.effective_date or '—'}")
            lines.append(f"Expiry date: {meta.expiry_date or '—'}")
            lines.append(f"Term: {meta.term_length or '—'}")
            lines.append(f"Governing law: {meta.governing_law or '—'}")
            lines.append(f"Notice period: {meta.notice_period or '—'}")
            lines.append(f"Auto-renew: {'yes' if meta.auto_renew else 'not stated'}")
            lines.append(f"Amounts: {', '.join(meta.amounts) or '—'}")
            return "\n".join(lines) + "\n\n" + DISCLAIMER
        if low.startswith("rights"):
            text = rest[6:].strip()
            if len(text) < 40:
                return ("Paste the contract text after the command.\n" + _usage())
            rights = extract_rights(text)
            if not rights:
                return ("No explicit rights-granting language found in this "
                        "text.\n\n" + DISCLAIMER)
            lines = ["✓ Rights this agreement states it grants:"]
            lines += [f"  • {r}" for r in rights]
            return "\n".join(lines) + "\n\n" + DISCLAIMER
        if low.startswith("rewrite"):
            text = rest[7:].strip()
            if len(text) < 20:
                return ("Paste one clause after the command.\n" + _usage())
            out = plain_english(text[:2000])
            return ("📝 Plain-English rewrite (same clause, simpler words — "
                    "not legal advice):\n\n" + out + "\n\n" + DISCLAIMER)
        if low.startswith("review"):
            rest = rest[6:].strip()
        ctype = "auto"
        style = "full"
        for t in CONTRACT_TYPES:
            if rest.lower().startswith(t + " ") or rest.lower().startswith(t + "\n"):
                ctype = t
                rest = rest[len(t):].strip()
                break
        for s in REVIEW_STYLES:
            if rest.lower().startswith(s + " ") or rest.lower().startswith(s + "\n"):
                style = s
                rest = rest[len(s):].strip()
                break
        if len(rest) < 80:
            return ("Paste the contract text after the command — "
                    "I need the actual clauses to review.\n" + _usage())
        review = review_contract(rest, ctype)
        return format_review(review, style=style)
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Contract review hit an error ({e}). {DISCLAIMER}"
