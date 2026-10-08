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
    "Finding",
    "Review",
    "detect_contract_type",
    "review_contract",
    "format_review",
    "extract_clauses",
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
        anomaly: bool = False) -> Finding:
    return Finding(rule_id=rule_id, title=title, severity=severity,
                   clause=clause.strip()[:400], explanation=explanation,
                   playbook_ref=playbook_ref, anomaly=anomaly)


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


# — playbook registry ————————————————————————————————————————————————

_PLAYBOOKS: dict[str, list[Rule]] = {
    "tenancy": [
        Rule("T-01", "Rent increase with no stated notice period", HIGH, ("tenancy",), _t_rent_increase),
        Rule("T-02", "Service charge with no amount or cap", MEDIUM, ("tenancy",), _t_service_charge),
        Rule("T-03", "Short notice-to-quit for a yearly tenancy", HIGH, ("tenancy",), _t_notice_to_quit),
        Rule("T-04", "Agency fee above the usual 10%", MEDIUM, ("tenancy",), _t_agency_fee),
        Rule("T-05", "Automatic renewal clause", MEDIUM, ("tenancy",), _t_auto_renew),
        Rule("T-06", "No repairs / maintenance clause found", LOW, ("tenancy",), _t_missing_repairs),
        Rule("T-98", "Unlimited liability / broad indemnity", HIGH, ("tenancy", "employment", "freelance"), _t_unlimited_liability),
        Rule("T-99", "Clause asks a party to waive statutory rights", CRITICAL, ("tenancy", "employment", "freelance"), _t_statutory_waiver),
    ],
    "employment": [
        Rule("E-01", "Probation unusually long", HIGH, ("employment",), _e_probation),
        Rule("E-02", "Termination notice issue", HIGH, ("employment",), _e_termination_notice),
        Rule("E-03", "Non-compete unusually broad", HIGH, ("employment",), _e_non_compete),
        Rule("E-04", "No pension / Contributory Pension Scheme mention", MEDIUM, ("employment",), _e_pension),
        Rule("E-05", "Broad salary-deduction clause", MEDIUM, ("employment",), _e_salary_deduction),
        Rule("T-98", "Unlimited liability / broad indemnity", HIGH, ("tenancy", "employment", "freelance"), _t_unlimited_liability),
        Rule("T-99", "Clause asks a party to waive statutory rights", CRITICAL, ("tenancy", "employment", "freelance"), _t_statutory_waiver),
    ],
    "freelance": [
        Rule("F-01", "Payment terms issue", HIGH, ("freelance",), _f_payment_terms),
        Rule("F-02", "IP ownership ambiguous", HIGH, ("freelance",), _f_ip_ownership),
        Rule("F-03", "One-sided termination right", MEDIUM, ("freelance",), _f_termination),
        Rule("F-04", "No late-payment provision", LOW, ("freelance",), _f_late_fees),
        Rule("F-05", "Open-ended scope / unlimited revisions", MEDIUM, ("freelance",), _f_scope_creep),
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
                findings.append(hit)
        score = max(0, 100 - sum(_DEDUCTION.get(f.severity, 0) for f in findings))
        order = {CRITICAL: 0, HIGH: 1, MEDIUM: 2, LOW: 3}
        findings.sort(key=lambda f: (order.get(f.severity, 4), f.rule_id))
        return Review(contract_type=ctype, score=score, grade=_grade(score),
                      findings=findings, clauses_seen=len(clauses))
    except Exception:  # noqa: BLE001 — fail-closed review, never raise
        return Review(contract_type="tenancy", score=0, grade="F",
                      findings=[], clauses_seen=0)


# ── formatting ──────────────────────────────────────────────────────


def format_review(review: Review) -> str:
    """WhatsApp-native, plain-language review. Disclaimer always attached."""
    names = _playbook_names()
    lines = [
        f"📄 Contract review — {names.get(review.contract_type, review.contract_type)}",
        f"Grade: *{review.grade}* ({review.score}/100)",
    ]
    if review.needs_attention:
        lines.append(f"{review.needs_attention} clause{'s' if review.needs_attention != 1 else ''} need attention.")
    else:
        lines.append("No clauses flagged against the playbook.")
    for f in review.findings:
        lines.append("")
        lines.append(f.headline())
        if f.clause:
            lines.append(f"_{f.clause[:220]}_")
        lines.append(f.explanation)
    lines.append("")
    lines.append(review.disclaimer)
    return "\n".join(lines)


def information_only_check(text: str) -> list[str]:
    """Return advice-line phrases found in ``text`` (empty = clean)."""
    t = (text or "").lower()
    return [p for p in _ADVICE_PHRASES if p in t]


# ── chat control ────────────────────────────────────────────────────


def _usage() -> str:
    return ("/contract review [tenancy|employment|freelance] <paste the contract text> — "
            "letter grade + flagged clauses in plain language.\n"
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
        if low.startswith("review"):
            rest = rest[6:].strip()
        ctype = "auto"
        for t in CONTRACT_TYPES:
            if rest.lower().startswith(t + " ") or rest.lower().startswith(t + "\n"):
                ctype = t
                rest = rest[len(t):].strip()
                break
        if len(rest) < 80:
            return ("Paste the contract text after the command — "
                    "I need the actual clauses to review.\n" + _usage())
        review = review_contract(rest, ctype)
        return format_review(review)
    except Exception as e:  # noqa: BLE001 — never raise from chat
        return f"Contract review hit an error ({e}). {DISCLAIMER}"
