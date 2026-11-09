"""Careers — the job-pipeline organ (WS2 P4).

The careers goal class's home: a personal application pipeline
(store), its board (surface), and — via
:mod:`nomorals.goals.providers` — the measured-progress provider
that reads the pipeline for careers-class goals.  The recruitment
*business* machinery (``matching/``) is a separate organ and stays
separate; this package is the owner's own career: applications,
interviews, offers.
"""

from __future__ import annotations

from .store import (
    ACTIVE_STAGES,
    FUNNEL_STAGES,
    STAGE_PROGRESS,
    STAGES,
    TERMINAL_STAGES,
    Application,
    CareerError,
    CareerStore,
    careers_path,
)

__all__ = [
    "ACTIVE_STAGES",
    "FUNNEL_STAGES",
    "STAGES",
    "STAGE_PROGRESS",
    "TERMINAL_STAGES",
    "Application",
    "CareerError",
    "CareerStore",
    "careers_path",
]
