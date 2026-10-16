"""Case file — append-only evidence bundles.

A case file is the immune system's memory of one incident: the
triggering event, a timeline (replayed by an INJECTED ``timeline_fn``,
e.g. ``os/replay.py``'s ``replay_mission`` — never imported here),
and an evidence list that is append-only: evidence is never deleted,
and once sealed, no further evidence may be added.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = ["CaseFile", "build_case"]


@dataclass
class CaseFile:
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    event: Any = None
    timeline: List[Any] = field(default_factory=list)
    evidence: List[Any] = field(default_factory=list)
    created_ts: float = field(default_factory=time.time)
    sealed: bool = False

    def add_evidence(self, item: Any) -> "CaseFile":
        """Append evidence. Raises if the case file is sealed — a sealed
        file accepts no further adds, and nothing is ever deleted."""
        if self.sealed:
            raise ValueError("case file is sealed — no further evidence may be added")
        self.evidence.append(item)
        return self

    def seal(self) -> "CaseFile":
        """Seal the case file: no further adds after this point."""
        self.sealed = True
        return self


def build_case(
    event: Any,
    timeline_fn: Callable[[Any], List[Any]] | None = None,
) -> CaseFile:
    """Build a case file for ``event``.

    ``timeline_fn`` is injected (e.g. ``replay_mission`` bound to a
    mission id by the caller) and is called with the event; its return
    value becomes the timeline. No timeline function → empty timeline,
    not a fabricated one.
    """
    timeline: List[Any] = []
    if timeline_fn is not None:
        result = timeline_fn(event)
        timeline = list(result) if result is not None else []
    return CaseFile(event=event, timeline=timeline)
