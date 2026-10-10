"""Trajectory learning: what worked, what failed, and how recently.

This package learns from Devon's own execution outcomes so callers
(e.g. the LLM broker) can route a task to the model / skill / tool that
has actually been succeeding at it *lately*.

Design notes:

- Everything is SQLite-backed with the same pattern as
  :mod:`nomorals.os.session` (``Database`` + ``_DDL`` + ``RLock``).
- Layer rule: L3 — only :mod:`nomorals.core` (layer 1) and
  :mod:`nomorals.storage` (layer 2) are imported.
- No priors are fabricated: an empty store reports a 0.5 success
  rate ("unknown") rather than 0.0 ("known-bad"), and all cluster
  listings come back empty.
"""

from .failure_kb import FailureKB
from .trajectories import (
    TrajectoryStore,
    cluster_key_for,
    exception_class_of,
    normalize_error,
    wilson_interval,
    wilson_lower,
)

__all__ = [
    "TrajectoryStore",
    "FailureKB",
    "normalize_error",
    "cluster_key_for",
    "exception_class_of",
    "wilson_lower",
    "wilson_interval",
]
