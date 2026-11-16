"""Workflow errors.

One error type for the whole workflow organ: a bad script, a bad
state transition, a bad schema, a missing workflow — all surface as
WorkflowError with the reason in the message, never as a bare
KeyError or a crash in a tick.
"""

from __future__ import annotations

__all__ = ["WorkflowError"]


class WorkflowError(Exception):
    """The workflow organ refused something, with the reason named."""
