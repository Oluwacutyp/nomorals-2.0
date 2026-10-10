"""Token budgets for context assembly.

A :class:`ContextBudget` splits a total token budget across the named
sections and :meth:`ContextBudget.fit` trims a section list down to the
budget: lowest-priority non-load-bearing sections are dropped first, and
load-bearing sections are never dropped, ever.

Named profiles (:meth:`ContextBudget.profile`) encode task-specific
allocations — a coding turn, a chat turn, and a research turn do not want
the same context mix.
"""

from __future__ import annotations

from typing import Any

from .sections import Section, priority_for

__all__ = ["ContextBudget", "DEFAULT_ALLOCATIONS", "BUDGET_PROFILES"]


#: Default per-section shares of the total budget.  History is the swing
#: section: it is large but the first thing the budget sacrifices.
DEFAULT_ALLOCATIONS: dict[str, float] = {
    "system": 0.10,
    "mission": 0.225,
    "artifacts": 0.175,
    "tools": 0.15,
    "project": 0.075,
    "user_profile": 0.05,
    "history": 0.175,
}


#: Task-specific allocation profiles.  Each maps section name -> share of
#: the total budget.  ``default`` matches :data:`DEFAULT_ALLOCATIONS`.
BUDGET_PROFILES: dict[str, dict[str, float]] = {
    "default": dict(DEFAULT_ALLOCATIONS),
    "chat": {
        "system": 0.08,
        "mission": 0.12,
        "artifacts": 0.10,
        "tools": 0.12,
        "project": 0.05,
        "user_profile": 0.10,
        "memory": 0.12,
        "history": 0.31,
    },
    "coding": {
        "system": 0.06,
        "mission": 0.20,
        "artifacts": 0.22,
        "tools": 0.18,
        "project": 0.08,
        "user_profile": 0.02,
        "memory": 0.06,
        "history": 0.18,
    },
    "research": {
        "system": 0.06,
        "mission": 0.15,
        "artifacts": 0.25,
        "tools": 0.12,
        "project": 0.08,
        "user_profile": 0.02,
        "memory": 0.12,
        "history": 0.20,
    },
    "minimal": {
        "system": 0.20,
        "mission": 0.30,
        "artifacts": 0.10,
        "tools": 0.20,
        "project": 0.05,
        "user_profile": 0.05,
        "memory": 0.05,
        "history": 0.05,
    },
}


def _sparkbar(ratio: float, width: int = 18) -> str:
    filled = max(0, min(width, int(round(ratio * width))))
    return "█" * filled + "░" * (width - filled)


class ContextBudget:
    """Total budget plus per-section allocations, in tokens."""

    def __init__(self, total: int = 8000, **allocations: float) -> None:
        if total <= 0:
            raise ValueError("budget total must be positive")
        self.total = int(total)
        self.allocations: dict[str, float] = dict(DEFAULT_ALLOCATIONS)
        for name, share in allocations.items():
            share = float(share)
            if share < 0:
                raise ValueError(f"allocation for {name!r} must be non-negative")
            self.allocations[name] = share
        #: tokens of load-bearing content that did not fit, from the last
        #: :meth:`fit` call.  Nonzero means the budget was exceeded by
        #: content the budget is not allowed to drop.
        self.last_overrun: int = 0

    @classmethod
    def default(cls, total: int = 8000) -> "ContextBudget":
        return cls(total)

    @classmethod
    def profile(cls, name: str, total: int = 8000) -> "ContextBudget":
        """Build a budget from a named task profile.

        Profiles: ``default``, ``chat``, ``coding``, ``research``,
        ``minimal``.  Raises ``ValueError`` for unknown names.
        """
        key = name.strip().lower()
        if key not in BUDGET_PROFILES:
            known = ", ".join(sorted(BUDGET_PROFILES))
            raise ValueError(f"unknown budget profile {name!r} (known: {known})")
        return cls(total, **BUDGET_PROFILES[key])

    def allocation_for(self, name: str) -> int:
        """Token allocation for a section name.

        Named allocations come from the configured shares; unknown names
        split the unclaimed remainder evenly-ish by falling back to a flat
        5% floor.
        """
        share = self.allocations.get(name, 0.05)
        return max(16, int(self.total * share))

    def fit(self, sections: list[Section]) -> list[Section]:
        """Trim ``sections`` to the budget, returning the survivors.

        Drops lowest-priority non-load-bearing sections first.  Load-bearing
        sections are never dropped; if they alone exceed the budget the
        overrun is recorded in :attr:`last_overrun` and every section is
        kept (the engine surfaces the overrun instead of deleting truth).
        """
        self.last_overrun = 0
        for section in sections:
            section.dropped = False

        load_bearing = [s for s in sections if s.load_bearing]
        flexible = [s for s in sections if not s.load_bearing]
        load_tokens = sum(s.tokens for s in load_bearing)
        if load_tokens > self.total:
            self.last_overrun = load_tokens - self.total

        used = load_tokens + sum(s.tokens for s in flexible)
        # Drop cheapest-to-lose first: ascending priority, ties by size.
        for section in sorted(flexible, key=lambda s: (s.priority, s.tokens)):
            if used <= self.total:
                break
            if section.tokens == 0:
                continue  # free sections never cost a drop
            used -= section.tokens
            section.dropped = True

        return [s for s in sections if not s.dropped]

    def priority_of(self, section: Section) -> float:
        if section.priority:
            return section.priority
        return priority_for(section.name)

    def utilization_report(self, sections: list[Section]) -> list[dict[str, Any]]:
        """Per-section budget utilization: used vs allocated.

        Each row carries ``name``, ``used``, ``allocated``, ``ratio``
        (used/allocated), ``priority``, ``load_bearing`` and ``status``
        (``under``/``over``/``unused``).
        """
        rows: list[dict[str, Any]] = []
        for section in sections:
            allocated = self.allocation_for(section.name)
            used = 0 if section.dropped else section.tokens
            ratio = used / allocated if allocated else 0.0
            if section.dropped or used == 0:
                status = "unused"
            elif ratio > 1.0:
                status = "over"
            else:
                status = "under"
            rows.append({
                "name": section.name,
                "used": used,
                "allocated": allocated,
                "ratio": round(ratio, 3),
                "priority": section.priority,
                "load_bearing": section.load_bearing,
                "dropped": section.dropped,
                "status": status,
            })
        rows.sort(key=lambda r: r["ratio"], reverse=True)
        return rows

    def render_table(
        self,
        sections: list[Section] | None = None,
        *,
        style: str = "plain",
    ) -> str:
        """Render the allocation table, optionally with per-section usage.

        ``style`` is ``"plain"`` (ASCII) or ``"fancy"`` (unicode bars,
        no color codes — safe on any terminal).
        """
        fancy = style == "fancy"
        header = (
            f"Context budget: {self.total} tokens"
            + (f" (overrun +{self.last_overrun})" if self.last_overrun else "")
        )
        lines = [header, "-" * 64]
        rows = self.utilization_report(sections) if sections is not None else None
        if rows is None:
            for name in self.allocations:
                lines.append(
                    f"  {name:<14} {self.allocation_for(name):>6} tokens "
                    f"({self.allocations[name]:.0%})"
                )
            return "\n".join(lines)
        for row in rows:
            if fancy:
                gauge = _sparkbar(min(1.0, row["ratio"]))
                status = {"under": "ok", "over": "OVER", "unused": "—"}[row["status"]]
                mark = "◆" if row["load_bearing"] else " "
                drop = " ✕dropped" if row["dropped"] else ""
                lines.append(
                    f"{mark} {row['name']:<14} {gauge} "
                    f"{row['used']:>6}/{row['allocated']:<6} {status}{drop}"
                )
            else:
                mark = "*" if row["load_bearing"] else " "
                drop = " [dropped]" if row["dropped"] else ""
                lines.append(
                    f"{mark} {row['name']:<14} {row['used']:>6}/{row['allocated']:<6} "
                    f"tokens ({row['ratio']:.0%} of allocation){drop}"
                )
        lines.append("-" * 64)
        total_used = sum(r["used"] for r in rows)
        lines.append(
            f"  {'TOTAL':<14} {total_used:>6}/{self.total:<6} tokens "
            f"({total_used / self.total:.0%} of budget)"
        )
        return "\n".join(lines)
