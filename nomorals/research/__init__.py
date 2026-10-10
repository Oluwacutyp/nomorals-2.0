"""Autonomous background research: scheduled web research, worth-interrupting
judgment, and delivery to the owner's DMs.

The pipeline is: ResearchJob -> run_job() -> assess_worth() -> deliver().
The scheduler in scheduler.py ticks the jobs. The core rule is conservative:
better to miss something than to spam the owner.
"""

from .costs import COST_TABLE
from .pipeline import (
    Assessment,
    DeepReport,
    ResearchBudget,
    ResearchContext,
    ResearchFinding,
    ResearchJob,
    assess_worth,
    clarify,
    decompose,
    deliver,
    ensure_schema,
    execute_job,
    research_deep,
    run_job,
    synthesize,
)
from .scheduler import ResearchScheduler, default_jobs

__all__ = [
    "Assessment",
    "COST_TABLE",
    "DeepReport",
    "ResearchBudget",
    "ResearchContext",
    "ResearchFinding",
    "ResearchJob",
    "ResearchScheduler",
    "assess_worth",
    "clarify",
    "decompose",
    "default_jobs",
    "deliver",
    "ensure_schema",
    "execute_job",
    "research_deep",
    "run_job",
    "synthesize",
]
