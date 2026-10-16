"""Instant yield: a real request preempts a run, and the fight resumes.

A game run must never hold the owner hostage: when a real request
arrives (a command, a DM, anything the owner actually needs), the run
**yields instantly** — the fight pauses, the request runs first, and
the fight resumes from the exact snapshot it paused at. No turn is
lost, no state is recomputed, no "the game ate my progress".

:class:`PreemptibleRun` owns that contract:

* ``pause(reason)`` snapshots the mutable fight state (deep-copied —
  later mutation of the live state cannot corrupt the snapshot).
* ``resume()`` restores the snapshot verbatim and returns to ``running``.
* :func:`preempt` is the one-call form: pause → run the real request →
  resume, in a ``finally`` so the fight resumes even if the request
  raises.

Invariant: ``snapshot == state`` after a pause→resume round-trip —
the fight state is bit-identical before and after preemption. A real
request always preempts: :func:`preempt` pauses *before* invoking the
request callable, never after.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, TypeVar

__all__ = ["PreemptibleRun", "preempt"]

T = TypeVar("T")


class PreemptibleRun:
    """A game run that yields instantly to real requests."""

    def __init__(self, state: dict[str, Any] | None = None) -> None:
        self.state: dict[str, Any] = dict(state or {})
        self.status: str = "running"  # running | paused | finished
        self._snapshot: dict[str, Any] | None = None
        self.pause_reason: str = ""
        self.preemptions: int = 0

    def pause(self, reason: str = "real request") -> dict[str, Any]:
        """Snapshot the fight and pause. Returns the snapshot."""
        if self.status == "finished":
            raise ValueError("cannot pause a finished run")
        self._snapshot = copy.deepcopy(self.state)
        self.status = "paused"
        self.pause_reason = reason or "real request"
        self.preemptions += 1
        return copy.deepcopy(self._snapshot)

    def resume(self) -> dict[str, Any]:
        """Restore the snapshot verbatim. Raises if never paused."""
        if self.status != "paused" or self._snapshot is None:
            raise ValueError("cannot resume a run that is not paused")
        self.state = copy.deepcopy(self._snapshot)
        self._snapshot = None
        self.status = "running"
        self.pause_reason = ""
        return dict(self.state)

    def finish(self) -> None:
        self.status = "finished"
        self._snapshot = None


def preempt(run: PreemptibleRun, request: Callable[[], T], *,
            reason: str = "real request") -> T:
    """Pause the run, run the real request, resume the fight.

    The request runs while the fight is paused; the fight resumes in a
    ``finally`` — even when the request raises, the run is not left
    paused forever.
    """
    run.pause(reason)
    try:
        return request()
    finally:
        run.resume()
