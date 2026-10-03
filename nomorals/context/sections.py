"""Sections: the unit of assembly for token-budgeted context.

A :class:`Section` is one named slab of a prompt (system instructions, mission
state, tool manifests, history, ...).  Sections carry the metadata the budget
and compressor need to make load-aware decisions: a priority, a load-bearing
flag, and ``keep`` — verbatim content that must survive compression.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.text import approx_token_count

__all__ = ["Section", "SECTION_PRIORITIES", "priority_for"]


#: Canonical assembly order is this dict's insertion order.  Higher number =
#: more important; the budget drops low-priority sections first.
SECTION_PRIORITIES: dict[str, float] = {
    "system": 100.0,
    "mission": 90.0,
    "artifacts": 70.0,
    "tools": 60.0,
    "project": 40.0,
    "user_profile": 30.0,
    "memory": 25.0,
    "history": 20.0,
}


def priority_for(name: str, default: float = 50.0) -> float:
    """Priority for a section name, falling back to ``default``."""
    return SECTION_PRIORITIES.get(name, default)


@dataclass
class Section:
    """One named slab of assembled context.

    ``load_bearing`` sections (acceptance criteria, active mission state,
    safety-of-state) are never dropped by the budget and their ``keep``
    content is never silently removed by the compressor: if it cannot fit,
    the section is marked ``truncated`` explicitly rather than vanishing.
    """

    name: str
    content: str = ""
    priority: float = 50.0
    load_bearing: bool = False
    keep: tuple[str, ...] = ()
    truncated: bool = False
    dropped: bool = False

    #: free-form provenance for this section (artifact ids, tool names, ...).
    meta: dict = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return approx_token_count(self.content)

    def keep_text(self) -> str:
        return "\n".join(k for k in self.keep if k)

    def render(self) -> str:
        """The section as it appears in the final prompt."""
        title = self.name.replace("_", " ").title()
        return f"## {title}\n{self.content}" if self.content else f"## {title}\n(empty)"
