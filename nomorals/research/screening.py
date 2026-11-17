"""Title/abstract screening for evidence reviews (Elicit core).

The systematic-review state machine around the extraction table:
a paper set × inclusion criteria → one decision per paper —
include | exclude | borderline — each naming the criterion that
decided it and the reason in words.

Rules, in order (first match decides; the decision is reproducible
from the criteria alone — re-running the same criteria over the same
paper set returns the same decisions):

1. No abstract and abstracts required → exclude ("abstract").
2. Known year outside [min_year, max_year] → exclude ("year").
3. Unknown year while a year bound is set → borderline ("year"):
   the paper cannot be judged on recency, so it is flagged for the
   owner instead of silently dropped or silently kept.
4. An excluded keyword in title or abstract → exclude ("keywords").
5. Required keywords given and none present → exclude ("keywords").
6. Otherwise → include ("all").

Nothing here ever raises: a paper with missing fields is decided
by the rules above, not by an exception.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .extraction import Paper

_WORD = re.compile(r"[a-z0-9]+")


@dataclass
class InclusionCriteria:
    """What a paper must satisfy to enter the extraction table."""

    min_year: int | None = None
    max_year: int | None = None
    require_abstract: bool = True
    keywords_include: list[str] = field(default_factory=list)
    keywords_exclude: list[str] = field(default_factory=list)
    require_doi: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"min_year": self.min_year, "max_year": self.max_year,
                "require_abstract": self.require_abstract,
                "keywords_include": list(self.keywords_include),
                "keywords_exclude": list(self.keywords_exclude),
                "require_doi": self.require_doi}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InclusionCriteria:
        return cls(min_year=data.get("min_year"),
                   max_year=data.get("max_year"),
                   require_abstract=data.get("require_abstract", True),
                   keywords_include=list(
                       data.get("keywords_include") or []),
                   keywords_exclude=list(
                       data.get("keywords_exclude") or []),
                   require_doi=data.get("require_doi", False))


@dataclass(frozen=True)
class ScreenDecision:
    """One screening decision: the verdict, the criterion, the reason."""

    paper_id: str
    decision: str  # include | exclude | borderline
    criterion: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"paper_id": self.paper_id, "decision": self.decision,
                "criterion": self.criterion, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScreenDecision:
        return cls(paper_id=data.get("paper_id", ""),
                   decision=data.get("decision", ""),
                   criterion=data.get("criterion", ""),
                   reason=data.get("reason", ""))


def _words(text: str) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _screen_one(paper: Paper, criteria: InclusionCriteria) -> ScreenDecision:
    pid = paper.id
    if criteria.require_abstract and not (paper.abstract or "").strip():
        return ScreenDecision(
            paper_id=pid, decision="exclude", criterion="abstract",
            reason="no abstract to screen — nothing to extract from")
    if criteria.require_doi and not paper.doi:
        return ScreenDecision(
            paper_id=pid, decision="exclude", criterion="doi",
            reason="no DOI — cannot identify the paper")
    if paper.year is not None:
        if criteria.min_year is not None and paper.year < criteria.min_year:
            return ScreenDecision(
                paper_id=pid, decision="exclude", criterion="year",
                reason=f"published {paper.year}, before min year "
                       f"{criteria.min_year}")
        if criteria.max_year is not None and paper.year > criteria.max_year:
            return ScreenDecision(
                paper_id=pid, decision="exclude", criterion="year",
                reason=f"published {paper.year}, after max year "
                       f"{criteria.max_year}")
    elif criteria.min_year is not None or criteria.max_year is not None:
        return ScreenDecision(
            paper_id=pid, decision="borderline", criterion="year",
            reason="year unknown — cannot judge recency; needs a human look")
    haystack = f"{paper.title} {paper.abstract}".lower()
    for keyword in criteria.keywords_exclude:
        if keyword.strip().lower() in haystack:
            return ScreenDecision(
                paper_id=pid, decision="exclude", criterion="keywords",
                reason=f"excluded keyword “{keyword.strip()}” present")
    if criteria.keywords_include:
        missing = [k for k in criteria.keywords_include
                   if k.strip().lower() not in haystack]
        if len(missing) == len(criteria.keywords_include):
            return ScreenDecision(
                paper_id=pid, decision="exclude", criterion="keywords",
                reason="none of the required keywords "
                       f"({', '.join(criteria.keywords_include)}) present")
    return ScreenDecision(
        paper_id=pid, decision="include", criterion="all",
        reason="meets all inclusion criteria")


def screen_papers(papers: list[Paper],
                  criteria: InclusionCriteria) -> list[ScreenDecision]:
    """Screen a paper set. Pure function — same input, same decisions."""
    return [_screen_one(paper, criteria) for paper in papers]


class ScreenSession:
    """The screening state of one review: decisions over its paper set."""

    def __init__(self, decisions: list[ScreenDecision],
                 papers: list[Paper]) -> None:
        self.decisions = list(decisions)
        self._papers = {p.id: p for p in papers}

    def included(self) -> list[Paper]:
        return [self._papers[d.paper_id] for d in self.decisions
                if d.decision == "include" and d.paper_id in self._papers]

    def excluded(self) -> list[tuple[Paper, ScreenDecision]]:
        return [(self._papers[d.paper_id], d) for d in self.decisions
                if d.decision == "exclude" and d.paper_id in self._papers]

    def borderline(self) -> list[tuple[Paper, ScreenDecision]]:
        return [(self._papers[d.paper_id], d) for d in self.decisions
                if d.decision == "borderline"
                and d.paper_id in self._papers]

    def counts(self) -> dict[str, int]:
        return {"include": sum(1 for d in self.decisions
                               if d.decision == "include"),
                "exclude": sum(1 for d in self.decisions
                               if d.decision == "exclude"),
                "borderline": sum(1 for d in self.decisions
                                  if d.decision == "borderline")}
