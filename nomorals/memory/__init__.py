"""L3 memory: episodic, semantic, working, and consolidation."""

from __future__ import annotations

from .backup import backup_to, import_records, restore_from, verify_backup
from .base import (
    TRUSTED,
    UNTRUSTED,
    MemoryRecord,
    format_recall,
    format_record,
    score_memory,
)
from .cadence import (
    consolidate_now,
    ensure_consolidation_job,
    maybe_run as maybe_consolidate,
    status as consolidation_status,
)
from .contradictions import (
    Contradiction,
    detect_against,
    detect_for,
    resolve as resolve_contradiction,
)
from .deep_recall import (
    DeepRecallResult,
    decisions_about,
    recall_deep,
    recall_window,
    related,
    timeline,
)
from .embeddings import (
    Embedder,
    contextualize_chunk,
    matryoshka_truncate,
    select_for_profile,
)
from .extract import MemoryDecision, decide, forget_targets
from .hybrid import Hit, mmr_select, rrf_fuse
from .manager import CoreBlocks, MemoryManager
from .persona import Community
from .repetition import (
    FSRS,
    FSCard,
    RepetitionScheduler,
    next_interval_days,
    retrievability,
)
from .scopes import (
    normalize_scope,
    record_matches_scope,
    scope_of,
    scope_tag,
    scopes_summary,
)
from .vector_backends import (
    BACKEND_NAMES,
    LegacyStoreBackend,
    SqliteVecBackend,
    USearchBackend,
    VectorBackend,
    available_backends,
    select_vector_backend,
)

__all__ = [
    "BACKEND_NAMES",
    "Community",
    "Contradiction",
    "CoreBlocks",
    "DeepRecallResult",
    "Embedder",
    "FSCard",
    "FSRS",
    "Hit",
    "LegacyStoreBackend",
    "MemoryDecision",
    "MemoryManager",
    "MemoryRecord",
    "RepetitionScheduler",
    "SqliteVecBackend",
    "TRUSTED",
    "UNTRUSTED",
    "USearchBackend",
    "VectorBackend",
    "available_backends",
    "backup_to",
    "consolidate_now",
    "consolidation_status",
    "contextualize_chunk",
    "decide",
    "decisions_about",
    "detect_against",
    "detect_for",
    "ensure_consolidation_job",
    "forget_targets",
    "format_recall",
    "format_record",
    "import_records",
    "matryoshka_truncate",
    "maybe_consolidate",
    "mmr_select",
    "next_interval_days",
    "normalize_scope",
    "recall_deep",
    "recall_window",
    "record_matches_scope",
    "related",
    "resolve_contradiction",
    "restore_from",
    "retrievability",
    "rrf_fuse",
    "scope_of",
    "scope_tag",
    "scopes_summary",
    "score_memory",
    "select_for_profile",
    "select_vector_backend",
    "timeline",
    "verify_backup",
]
