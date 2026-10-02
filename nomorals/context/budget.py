"""Token budgets for context assembly.

A :class:`ContextBudget` splits a total token budget across the named
sections and :meth:`ContextBudget.fit` trims a section list down to the
budget: lowest-priority non-load-bearing sections are dropped first, and
load-bearing sections are never dropped, ever.
"""

from __future__ import annotations

from .sections import Section, priority_for

__all__ = ["ContextBudget", "DEFAULT_ALLOCATIONS"]


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
