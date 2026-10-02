"""Token-budgeted context assembly for model calls.

The engine builds the canonical sections (system / user profile / project /
mission / artifacts / tools / history), fits them to a
:class:`ContextBudget`, compresses each survivor with explicit truncation
markers, and can snapshot/restore the result or report reclaimable waste.

Layer L5: imports core (L1) only.  Everything else is duck-typed.
"""

from __future__ import annotations

from .budget import DEFAULT_ALLOCATIONS, ContextBudget
from .compress import (
    compress_section,
    extractive_summary,
    summarize_then_truncate,
)
from .engine import DEFAULT_PREAMBLE, BuiltContext, ContextEngine
from .sections import SECTION_PRIORITIES, Section, priority_for
from .snapshots import SnapshotStore, snapshot_from_dict, snapshot_to_dict
from .waste import WasteDetector, WasteFinding, WasteReport

__all__ = [
    "ContextBudget",
    "DEFAULT_ALLOCATIONS",
    "DEFAULT_PREAMBLE",
    "BuiltContext",
    "ContextEngine",
    "SECTION_PRIORITIES",
    "Section",
    "SnapshotStore",
    "WasteDetector",
    "WasteFinding",
    "WasteReport",
    "compress_section",
    "extractive_summary",
    "priority_for",
    "snapshot_from_dict",
    "snapshot_to_dict",
    "summarize_then_truncate",
]
