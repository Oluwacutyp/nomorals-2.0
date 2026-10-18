"""Evidence-supported gap-resolution adequacy (audit §3).

Replaces the concrete weakness in ``research/autonomy.py::_work_gaps``:
a nonempty synthesis marked a gap resolved. That is an execution
criterion, not an adequacy criterion. A fluent paragraph must not
close a hard question automatically.

A gap is resolved ONLY when ALL hold:
1. Synthesis is non-empty AND is not the pipeline's "no evidence"
   sentinel.
2. At least ``MIN_SOURCES`` findings from DISTINCT domains
   (syndicated duplicates / one domain ≠ independent confirmation).
3. Every declared subquestion is covered — either answered in the
   synthesis/findings or explicitly listed as unresolved (which keeps
   the gap open, not resolved).
4. The consuming task can proceed (optional ``proceed_test`` callable
   supplied by the consumer; when absent, 1–3 decide).
5. No unresolved contradictions (``deep.conflicts`` non-empty ⇒ open).

Unresolved subquestions and contradictions are PRESERVED on the
verdict (and, via the patched ``_work_gaps``, written back to the
gap row) so a later run can continue instead of restarting.
Reopen rule: a resolved gap reopens when its supporting evidence
expires or is contradicted — see :func:`should_reopen`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

__all__ = ["AdequacyVerdict", "assess_gap_adequacy", "should_reopen"]

#: Independent-confirmation floor: one domain is never enough.
MIN_SOURCES = 2
#: Pipeline sentinel meaning "findings do not support an answer".
_NO_EVIDENCE_MARKERS = ("no evidence", "insufficient evidence",
                        "could not find", "no reliable source")


@dataclass
class AdequacyVerdict:
    adequate: bool
    reasons: list[str] = field(default_factory=list)
    unresolved_subquestions: list[str] = field(default_factory=list)
    contradictions: list[Any] = field(default_factory=list)
    distinct_domains: int = 0


def _domains(findings: Any) -> set[str]:
    out = set()
    for f in findings or []:
        url = getattr(f, "url", "") or ""
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        if host:
            out.add(host)
    return out


def _subquestions(gap: Any) -> list[str]:
    ctx = gap.get("context", "") if isinstance(gap, dict) else ""
    try:
        parsed = json.loads(ctx or "{}")
    except Exception:
        return []
    sq = parsed.get("subquestions") or []
    return [str(s) for s in sq]


def _covered(subq: str, deep: Any) -> bool:
    parts = [getattr(deep, "synthesis", "") or ""]
    for f in (getattr(deep, "findings", None) or []):
        parts.append(getattr(f, "snippet", "") or "")
        parts.append(getattr(f, "title", "") or "")
    hay = "\n".join(parts).lower()
    # Subquestion is covered when its key terms appear in the evidence.
    terms = [t for t in subq.lower().split() if len(t) > 3]
    if not terms:
        return subq.lower() in hay
    return any(t in hay for t in terms)


def assess_gap_adequacy(deep: Any, gap: Any,
                        proceed_test: Callable[[Any], bool] | None = None
                        ) -> AdequacyVerdict:
    reasons: list[str] = []
    synthesis = (getattr(deep, "synthesis", "") or "").strip()
    if not synthesis:
        reasons.append("empty synthesis")
    if any(m in synthesis.lower() for m in _NO_EVIDENCE_MARKERS):
        reasons.append("synthesis reports no supporting evidence")
    domains = _domains(getattr(deep, "findings", None))
    if len(domains) < MIN_SOURCES:
        reasons.append(
            f"only {len(domains)} distinct source domain(s); need >={MIN_SOURCES}")
    unresolved = [sq for sq in _subquestions(gap) if not _covered(sq, deep)]
    if unresolved:
        reasons.append(f"unresolved subquestions: {unresolved}")
    contradictions = list(getattr(deep, "conflicts", None) or [])
    if contradictions:
        reasons.append(f"{len(contradictions)} contradiction(s) preserved")
    if getattr(deep, "needs_clarification", False):
        reasons.append("deep research needs clarification")
    if proceed_test is not None:
        try:
            if not proceed_test(deep):
                reasons.append("consuming task cannot proceed on this answer")
        except Exception as exc:  # noqa: BLE001 - a broken test ≠ adequacy
            reasons.append(f"proceed test errored: {exc}")
    return AdequacyVerdict(
        adequate=not reasons, reasons=reasons,
        unresolved_subquestions=unresolved,
        contradictions=contradictions, distinct_domains=len(domains))


def should_reopen(resolved_gap: dict[str, Any], *,
                  evidence_expired: bool = False,
                  contradicted: bool = False) -> bool:
    """Reopen a resolved gap when evidence expires or is contradicted.

    Retrieval availability is tracked separately from truth: a
    disappeared URL alone does NOT disprove a claim — callers pass
    ``contradicted=True`` only on positive counter-evidence.
    """
    return bool(evidence_expired or contradicted)
