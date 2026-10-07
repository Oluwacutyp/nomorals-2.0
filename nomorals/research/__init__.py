"""Autonomous background research: scheduled web research, worth-interrupting
judgment, and delivery to the owner's DMs.

The pipeline is: ResearchJob -> run_job() -> assess_worth() -> deliver().
The scheduler in scheduler.py ticks the jobs. The core rule is conservative:
better to miss something than to spam the owner.
"""

from .pipeline import (
    Assessment,
    ResearchContext,
    ResearchFinding,
    ResearchJob,
    assess_worth,
    deliver,
    ensure_schema,
    execute_job,
    run_job,
)
from .scheduler import ResearchScheduler, default_jobs

__all__ = [
    "Assessment",
    "ResearchContext",
    "ResearchFinding",
    "ResearchJob",
    "ResearchScheduler",
    "assess_worth",
    "default_jobs",
    "deliver",
    "ensure_schema",
    "execute_job",
    "run_job",
]
