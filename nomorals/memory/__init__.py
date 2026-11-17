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
    CONSOLIDATION_JOB_NAME,
    LEGACY_CONSOLIDATION_JOB_NAME,
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
from .ephemeral_notes import (
    BurnNoteStore,
    ensure_burn_note_job,
    list_notes,
    make_link,
    parse_expiry,
    parse_link,
    purge_note,
    read_note,
    run_burn_note_tick_once,
    seal_note,
    sweep_expired,
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
    "BurnNoteStore",
    "CONSOLIDATION_JOB_NAME",
    "Community",
    "Contradiction",
    "CoreBlocks",
    "DeepRecallResult",
    "Embedder",
    "FSCard",
    "FSRS",
    "Hit",
    "LEGACY_CONSOLIDATION_JOB_NAME",
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
    "ensure_burn_note_job",
    "ensure_consolidation_job",
    "forget_targets",
    "format_recall",
    "format_record",
    "import_records",
    "list_notes",
    "make_link",
    "matryoshka_truncate",
    "maybe_consolidate",
    "mmr_select",
    "next_interval_days",
    "normalize_scope",
    "parse_expiry",
    "parse_link",
    "purge_note",
    "read_note",
    "recall_deep",
    "recall_window",
    "record_matches_scope",
    "related",
    "resolve_contradiction",
    "restore_from",
    "retrievability",
    "rrf_fuse",
    "run_burn_note_tick_once",
    "scope_of",
    "scope_tag",
    "scopes_summary",
    "score_memory",
    "seal_note",
    "select_for_profile",
    "select_vector_backend",
    "sweep_expired",
    "timeline",
    "verify_backup",
]
