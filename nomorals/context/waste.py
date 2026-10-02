"""Waste detection for assembled context.

:class:`WasteDetector` analyzes a built context and reports reclaimable
tokens: duplicated tool outputs, stale history beyond the useful window, and
oversized low-signal sections.  Estimates are conservative — a reported
reclaimable token is one the engine could actually remove without touching
load-bearing content.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ..core.text import approx_token_count
from .budget import ContextBudget
from .engine import BuiltContext

__all__ = ["WasteFinding", "WasteReport", "WasteDetector"]

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text.strip().lower())


@dataclass
class WasteFinding:
    kind: str  # duplicate_output | stale_history | oversized_section
    section: str
    detail: str
    wasted_tokens: int
    reclaimable_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "section": self.section,
            "detail": self.detail,
            "wasted_tokens": self.wasted_tokens,
            "reclaimable_tokens": self.reclaimable_tokens,
        }


@dataclass
class WasteReport:
    findings: list[WasteFinding] = field(default_factory=list)

    @property
    def total_wasted(self) -> int:
        return sum(f.wasted_tokens for f in self.findings)

    @property
    def total_reclaimable(self) -> int:
        return sum(f.reclaimable_tokens for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "findings": [f.to_dict() for f in self.findings],
            "total_wasted": self.total_wasted,
            "total_reclaimable": self.total_reclaimable,
        }

    def summary(self) -> str:
        if not self.findings:
            return "No context waste detected."
        lines = [
            f"Context waste: ~{self.total_wasted} tokens wasted, "
            f"~{self.total_reclaimable} reclaimable."
        ]
        for finding in self.findings:
            lines.append(
                f"- [{finding.kind}] {finding.section}: {finding.detail} "
                f"(~{finding.reclaimable_tokens} tokens reclaimable)"
            )
        return "\n".join(lines)


class WasteDetector:
    """Find duplicated, stale, and oversized content in a built context."""

    def __init__(
        self,
        *,
        stale_history_window: int = 10,
        min_stale_tokens: int = 100,
        low_signal_priority_below: float = 30.0,
        oversize_factor: float = 2.0,
        min_duplicate_tokens: int = 40,
    ) -> None:
        self.stale_history_window = stale_history_window
        self.min_stale_tokens = min_stale_tokens
        self.low_signal_priority_below = low_signal_priority_below
        self.oversize_factor = oversize_factor
        self.min_duplicate_tokens = min_duplicate_tokens

    def analyze(
        self, built: BuiltContext, budget: ContextBudget | None = None
    ) -> WasteReport:
        report = WasteReport()
        report.findings.extend(self._find_duplicates(built))
        report.findings.extend(self._find_stale_history(built))
        report.findings.extend(
            self._find_oversized(built, budget or ContextBudget.default()))
        # Reclaimable can never exceed wasted, by construction each finding
        # obeys it; sort biggest-first for readability.
        report.findings.sort(key=lambda f: f.reclaimable_tokens, reverse=True)
        return report

    # ── duplicated tool outputs ───────────────────────────────────────────
    def _find_duplicates(self, built: BuiltContext) -> list[WasteFinding]:
        findings: list[WasteFinding] = []
        for section in built.sections:
            lines = [ln for ln in section.content.splitlines() if _normalize(ln)]
            counts = Counter(_normalize(ln) for ln in lines)
            for norm, count in counts.items():
                if count < 2:
                    continue
                tokens = approx_token_count(norm)
                if tokens < self.min_duplicate_tokens:
                    continue
                extra = count - 1
                wasted = extra * tokens
                sample = next(ln for ln in lines if _normalize(ln) == norm)
                findings.append(WasteFinding(
                    kind="duplicate_output",
                    section=section.name,
                    detail=(f"{extra} duplicate cop{'y' if extra == 1 else 'ies'} of "
                            f"{sample[:80]!r}"),
                    wasted_tokens=wasted,
                    reclaimable_tokens=wasted,  # one copy stays, rest can go
                ))
        return findings

    # ── stale history ─────────────────────────────────────────────────────
    def _find_stale_history(self, built: BuiltContext) -> list[WasteFinding]:
        section = built.section("history")
        if section is None:
            return []
        entries = [ln for ln in section.content.splitlines()
                   if ln.strip().startswith("- [")]
        if len(entries) <= self.stale_history_window:
            return []
        stale = entries[: len(entries) - self.stale_history_window]
        stale_tokens = approx_token_count("\n".join(stale))
        if stale_tokens < self.min_stale_tokens:
            return []
        return [WasteFinding(
            kind="stale_history",
            section="history",
            detail=(f"{len(stale)} of {len(entries)} history entries are older "
                    f"than the useful window ({self.stale_history_window})"),
            wasted_tokens=stale_tokens,
            reclaimable_tokens=stale_tokens,
        )]

    # ── oversized low-signal sections ─────────────────────────────────────
    def _find_oversized(
        self, built: BuiltContext, budget: ContextBudget
    ) -> list[WasteFinding]:
        findings: list[WasteFinding] = []
        for section in built.sections:
            if section.load_bearing:
                continue  # load-bearing is never waste by definition
            if section.priority >= self.low_signal_priority_below:
                continue
            cap = budget.allocation_for(section.name)
            limit = int(cap * self.oversize_factor)
            tokens = section.tokens
            if tokens > limit:
                findings.append(WasteFinding(
                    kind="oversized_section",
                    section=section.name,
                    detail=(f"{tokens} tokens at priority {section.priority:g} "
                            f"vs ~{cap} token allocation"),
                    wasted_tokens=tokens - cap,
                    reclaimable_tokens=tokens - cap,
                ))
        return findings
