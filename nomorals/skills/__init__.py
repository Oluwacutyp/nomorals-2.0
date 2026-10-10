"""Executable skill packages (L4).

A skill here is a named, versioned chain of tool calls with declared
input/output schemas and explicit data wiring between steps — something
you *run*, via the same capability-gated :class:`ToolRegistry.call`
dispatch every agent uses.

This package is the execution substrate.  The knowledge-skill library
in ``nomorals/agents/skills.py`` (L5) is the complementary half: proven
strategies and playbooks as texts.  One remembers *how*; this one
*does*.
"""

from __future__ import annotations

from .bench import (BENCH_RUNS_DDL, BENCH_STEPS_DDL, SkillBench,
                    format_score, sparkline)
from .manifest import (RESERVED_NAMES, NAME_RE, SEMVER_RE, TYPE_NAMES,
                       ManifestError, SkillManifest, WiringError,
                       diff_manifests, parse_wiring_root,
                       resolve_expression, sanitize_description,
                       validate_schema)
from .registry import SKILL_PACKAGES_DDL, InstalledSkill, SkillRegistry
from .repair import (REPAIR_TICKETS_DDL, RepairTicket, RepairTicketStore,
                     build_ticket, format_ticket, suggest_fix)
from .runner import (DRY_RUN_SKIPPED, SkillResult, SkillRunner, StepResult,
                     format_result)

__all__ = [
    "SkillManifest",
    "ManifestError",
    "WiringError",
    "validate_schema",
    "resolve_expression",
    "parse_wiring_root",
    "sanitize_description",
    "diff_manifests",
    "SEMVER_RE",
    "NAME_RE",
    "TYPE_NAMES",
    "RESERVED_NAMES",
    "SkillRegistry",
    "InstalledSkill",
    "SKILL_PACKAGES_DDL",
    "SkillRunner",
    "SkillResult",
    "StepResult",
    "format_result",
    "DRY_RUN_SKIPPED",
    "SkillBench",
    "BENCH_RUNS_DDL",
    "BENCH_STEPS_DDL",
    "format_score",
    "sparkline",
    "RepairTicket",
    "RepairTicketStore",
    "build_ticket",
    "suggest_fix",
    "format_ticket",
    "REPAIR_TICKETS_DDL",
]
