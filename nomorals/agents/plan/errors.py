"""Errors for the /plan organ (L5)."""

from __future__ import annotations

__all__ = ["PlanError"]


class PlanError(Exception):
    """A plan could not be surveyed, built, saved, or run."""
