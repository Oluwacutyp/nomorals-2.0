"""Arrival / departure triggers: evaluate SUPPLIED updates, never track.

The only entry point is ``evaluate_update``. The caller supplies a
location (e.g. the owner's phone reporting where it is); this module
compares it against the caller's fences and emits ``entered`` /
``exited`` events on inside↔outside transitions. It never polls, never
opens GPS, never runs a background loop — there is nothing here that
can observe a position unasked.

State: a ``FenceState`` per (person, fence) remembers the last known
inside/outside. The first update for a fence establishes the baseline
and emits NO event (arriving somewhere is not "entering" until we
already knew you were outside). Callers persist states however they
like; ``FenceState`` round-trips through ``to_dict`` / ``from_dict``.

Consent: ``consent`` must be True or ``ConsentError`` is raised and
nothing is evaluated or stored.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .geofence import Geofence
from .primitives import LatLon
from .privacy import require_consent

__all__ = ["FenceState", "evaluate_update"]


@dataclass
class FenceState:
    """Last-known inside/outside for one (person, fence) pair."""

    person_id: str
    fence_id: str
    inside: bool | None = None  # None = no baseline yet
    last_loc: dict[str, float] | None = None
    last_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"person_id": self.person_id, "fence_id": self.fence_id,
                "inside": self.inside, "last_loc": self.last_loc,
                "last_ts": self.last_ts}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FenceState":
        return cls(person_id=str(data["person_id"]),
                   fence_id=str(data["fence_id"]),
                   inside=data.get("inside"),
                   last_loc=data.get("last_loc"),
                   last_ts=float(data.get("last_ts", 0.0)))


def evaluate_update(*, person_id: str, loc: LatLon,
                    fences: list[Geofence],
                    states: dict[str, FenceState],
                    consent: bool,
                    ts: float | None = None) -> list[dict[str, Any]]:
    """Evaluate one supplied location update against ``fences``.

    Returns enter/exit events. ``states`` is mutated in place (caller
    owns persistence). First sighting per fence sets the baseline and
    emits no event. ``consent=False`` raises ConsentError before any
    evaluation.
    """
    require_consent(consent, scope="geofence")
    now = ts if ts is not None else time.time()
    events: list[dict[str, Any]] = []
    for fence in fences:
        state = states.get(fence.id)
        if state is None:
            state = FenceState(person_id=person_id, fence_id=fence.id)
            states[fence.id] = state
        now_inside = fence.contains(loc)
        prev = state.inside
        state.inside = now_inside
        state.last_loc = loc.to_dict()
        state.last_ts = now
        if prev is None or prev == now_inside:
            continue  # baseline established, or no transition
        events.append({
            "person_id": person_id, "fence_id": fence.id,
            "fence_name": fence.name,
            "event": "entered" if now_inside else "exited",
            "loc": loc.to_dict(), "ts": now,
        })
    return events
