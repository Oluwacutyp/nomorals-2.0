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

from .bench import BENCH_RUNS_DDL, SkillBench
from .manifest import (ManifestError, SkillManifest, WiringError,
                       resolve_expression, validate_schema)
from .registry import SKILL_PACKAGES_DDL, InstalledSkill, SkillRegistry
from .repair import (REPAIR_TICKETS_DDL, RepairTicket, RepairTicketStore,
                     build_ticket, suggest_fix)
from .runner import SkillResult, SkillRunner, StepResult

__all__ = [
    "SkillManifest",
    "ManifestError",
    "WiringError",
    "validate_schema",
    "resolve_expression",
    "SkillRegistry",
    "InstalledSkill",
    "SKILL_PACKAGES_DDL",
    "SkillRunner",
    "SkillResult",
    "StepResult",
    "SkillBench",
    "BENCH_RUNS_DDL",
    "RepairTicket",
    "RepairTicketStore",
    "build_ticket",
    "suggest_fix",
    "REPAIR_TICKETS_DDL",
]
