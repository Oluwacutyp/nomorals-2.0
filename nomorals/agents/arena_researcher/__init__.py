"""Arena researcher: Devon researching herself.

A cadenced organ beside the research loop (owner phase 2026-10-21
item 2): it surveys the live system — organs, wiring, dead/decorative
paths, weak spots — mines external state of the art per organ through
the existing research pipeline, and files RANKED proposals with
evidence into the upgrade queue for the owner's word. Never
auto-approved, never auto-applied.
"""
from __future__ import annotations

from .loop import (
                   ARENA_RESEARCHER_JOB,
                   SOURCE,
                   ArenaResearcher,
                   arena_dedupe_days,
                   arena_hours,
                   arena_max_proposals,
                   ensure_arena_researcher_job,
                   register,
                   status,
)
from .rank import finding_to_ticket, rank_findings
from .store import ArenaResearcherStore, default_arena_researcher_path
from .survey import default_survey_cache_path, survey_repo, verify_anchor

__all__ = [
    "ARENA_RESEARCHER_JOB",
    "SOURCE",
    "ArenaResearcher",
    "ArenaResearcherStore",
    "arena_dedupe_days",
    "arena_hours",
    "arena_max_proposals",
    "default_arena_researcher_path",
    "default_survey_cache_path",
    "ensure_arena_researcher_job",
    "finding_to_ticket",
    "rank_findings",
    "register",
    "status",
    "survey_repo",
    "verify_anchor",
]
