"""BugHunter — authorized bug-bounty hunting inside Devon's security organ.

Port of elementalsouls/Claude-BugHunter v2.1.0 (MIT code / CC BY 4.0
content, commit 0e70a02): scope rules, class weights / skill map,
URL-pattern triggers, the 7-Question Gate, the coverage-closure
ledger, and the H1/Bugcrowd/Intigriti/Immunefi report templates —
over Devon's HttpClient, store, brain (injected ``suggest``), audit
book and evidence chain. The ``claude -p`` subprocess is gone; the
LLM is only ever the judge in hunt/validate, and with no model
those phases record items as blocked, never as clean.

Vendored skill packs + pattern library live in
``nomorals/skills/library/bughunter/`` (83 skills, 36 pattern files,
15 commands, upstream's skill index); every file is SHA-256-pinned
in ``bughunter-attribution.toml``.
"""

from __future__ import annotations

from .agent import extract_json, run_agent
from .attribution import verify_against_clone, verify_attribution
from .classify import URL_PATTERN_TO_SKILLS, classify_url
from .closure import coverage_complete, coverage_gaps, evaluate
from .engine import Engine, run_engagement, worklist_from_audit
from .hackerone import (
    get_my_program,
    get_program,
    get_program_scope,
    list_my_programs,
    list_my_reports,
    search_disclosed_reports,
)
from .memory import (
    DEAD_CLASS_THRESHOLD,
    record_finding,
    record_run,
    skip_decision,
)
from .parity import run_parity
from .recon_map import (
    auth_enforcement_sweep,
    build_manifest,
    classify_all,
    gather_params,
    openapi_recon,
    platform_recon,
    prune_inert_params,
    spa_recon,
)
from .report import render_engagement_report, render_report
from .scope import Scope
from .skillmap import (
    CLASS_PROBE,
    CLASS_SKILL,
    CLASS_WEIGHT,
    TECH_SKILL,
    probe_for,
    rank_priority,
    skill_map_selftest,
    skills_for,
    tech_skills,
)
from .state import Engagement, engagements_root
from .surface import register
from .triage import TRIAGE_QUESTIONS, gate_finding, gate_text

__all__ = [
    "Scope",
    "Engagement",
    "engagements_root",
    "Engine",
    "run_engagement",
    "worklist_from_audit",
    "classify_url",
    "URL_PATTERN_TO_SKILLS",
    "skills_for",
    "tech_skills",
    "probe_for",
    "rank_priority",
    "skill_map_selftest",
    "CLASS_WEIGHT",
    "CLASS_SKILL",
    "TECH_SKILL",
    "CLASS_PROBE",
    "gate_text",
    "gate_finding",
    "TRIAGE_QUESTIONS",
    "coverage_complete",
    "coverage_gaps",
    "evaluate",
    "render_report",
    "render_engagement_report",
    "run_agent",
    "extract_json",
    "build_manifest",
    "classify_all",
    "spa_recon",
    "platform_recon",
    "openapi_recon",
    "auth_enforcement_sweep",
    "gather_params",
    "prune_inert_params",
    "search_disclosed_reports",
    "get_program",
    "get_program_scope",
    "get_my_program",
    "list_my_programs",
    "list_my_reports",
    "record_finding",
    "record_run",
    "skip_decision",
    "DEAD_CLASS_THRESHOLD",
    "run_parity",
    "verify_attribution",
    "verify_against_clone",
    "register",
]
