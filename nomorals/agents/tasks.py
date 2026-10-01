"""Deprecated shim: task primitives moved to :mod:`nomorals.core.tasks`.

Import from ``nomorals.core.tasks`` instead.  This module stays as a
re-export so older branches and plugins keep working.
"""
from ..core.tasks import (  # noqa: F401
    Task,
    TaskGraph,
    TaskKind,
    TaskState,
    cycle_in,
)

__all__ = ["Task", "TaskGraph", "TaskKind", "TaskState", "cycle_in"]
