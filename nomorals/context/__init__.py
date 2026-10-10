"""Token-budgeted context assembly for model calls.

The engine builds the canonical sections (system / user profile / project /
mission / artifacts / tools / history), fits them to a
:class:`ContextBudget`, compresses each survivor with explicit truncation
markers, and can snapshot/restore the result or report reclaimable waste.

Assembly ordering modes (``ordering``): ``"priority"`` (canonical order),
``"cache"`` (stable prefix first, volatile tail last — keeps provider
prompt-caches hot), ``"rot"`` (load-bearing first with a key-fact echo at
the tail — fights lost-in-the-middle attention decay).

Layer L5: imports core (L1) only.  Everything else is duck-typed.
"""

from __future__ import annotations

from .budget import BUDGET_PROFILES, DEFAULT_ALLOCATIONS, ContextBudget
from .cache import (
    CachePlan,
    StabilityWarning,
    audit_prefix_stability,
    cache_plan,
    stable_fingerprint,
)
from .compress import (
    compress_section,
    extractive_summary,
    salient_extract,
    summarize_then_truncate,
)
from .engine import (
    DEFAULT_PREAMBLE,
    ORDERING_MODES,
    BuiltContext,
    ContextEngine,
)
from .sections import (
    SECTION_PRIORITIES,
    Section,
    priority_for,
    set_token_counter,
    token_count,
    try_tiktoken_counter,
)
from .snapshots import (
    SnapshotStore,
    format_snapshot_diff,
    snapshot_diff,
    snapshot_from_dict,
    snapshot_to_dict,
)
from .waste import (
    WasteDetector,
    WasteFinding,
    WasteReport,
    classify_recoverability,
    simhash,
)

__all__ = [
    "BUDGET_PROFILES",
    "ORDERING_MODES",
    "CachePlan",
    "ContextBudget",
    "DEFAULT_ALLOCATIONS",
    "DEFAULT_PREAMBLE",
    "BuiltContext",
    "ContextEngine",
    "SECTION_PRIORITIES",
    "Section",
    "SnapshotStore",
    "StabilityWarning",
    "WasteDetector",
    "WasteFinding",
    "WasteReport",
    "audit_prefix_stability",
    "cache_plan",
    "classify_recoverability",
    "compress_section",
    "extractive_summary",
    "format_snapshot_diff",
    "priority_for",
    "salient_extract",
    "set_token_counter",
    "simhash",
    "snapshot_diff",
    "snapshot_from_dict",
    "snapshot_to_dict",
    "stable_fingerprint",
    "summarize_then_truncate",
    "token_count",
    "try_tiktoken_counter",
]
