"""L3 memory: episodic, semantic, working, and consolidation."""

from __future__ import annotations

from .backup import backup_to, import_records, restore_from, verify_backup
from .base import TRUSTED, UNTRUSTED, MemoryRecord, score_memory
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
from .embeddings import Embedder, select_for_profile
from .manager import MemoryManager
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
    "Contradiction",
    "DeepRecallResult",
    "Embedder",
    "LegacyStoreBackend",
    "MemoryManager",
    "MemoryRecord",
    "SqliteVecBackend",
    "TRUSTED",
    "UNTRUSTED",
    "USearchBackend",
    "VectorBackend",
    "available_backends",
    "backup_to",
    "consolidate_now",
    "consolidation_status",
    "decisions_about",
    "detect_against",
    "detect_for",
    "ensure_consolidation_job",
    "import_records",
    "maybe_consolidate",
    "normalize_scope",
    "recall_deep",
    "recall_window",
    "record_matches_scope",
    "related",
    "resolve_contradiction",
    "restore_from",
    "scope_of",
    "scope_tag",
    "scopes_summary",
    "score_memory",
    "select_for_profile",
    "select_vector_backend",
    "timeline",
    "verify_backup",
]
