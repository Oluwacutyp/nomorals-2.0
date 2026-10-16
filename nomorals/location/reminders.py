"""Location-aware reminders: fire when inside (or leaving) a fence.

A reminder is a plain record: ``fire when a supplied update produces
an <entered|exited> event for my fence``. This module never tracks
anything — it only matches reminders against events produced by
``triggers.evaluate_update``. A reminder fires once (``fired_ts`` is
set); re-arm with ``rearm()`` for recurring reminders.

Consent: ``check_reminders`` requires ``consent=True`` — reminders
are location use, so they refuse without opt-in consent too.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .privacy import require_consent

__all__ = ["LocationReminder", "check_reminders"]


@dataclass
class LocationReminder:
    """Fire ``message`` when ``fire_on`` happens at ``fence_id``."""

    id: str
    person_id: str
    fence_id: str
    message: str
    fire_on: str = "enter"  # "enter" | "exit"
    active: bool = True
    fired_ts: float | None = None
    created_ts: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.fire_on not in ("enter", "exit"):
            raise ValueError(f"fire_on must be 'enter'|'exit', got {self.fire_on!r}")

    @property
    def fired(self) -> bool:
        return self.fired_ts is not None

    def rearm(self) -> None:
        """Make a fired reminder fireable again."""
        self.fired_ts = None
        self.active = True

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "person_id": self.person_id,
                "fence_id": self.fence_id, "message": self.message,
                "fire_on": self.fire_on, "active": self.active,
                "fired_ts": self.fired_ts, "created_ts": self.created_ts}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocationReminder":
        return cls(id=str(data["id"]), person_id=str(data["person_id"]),
                   fence_id=str(data["fence_id"]),
                   message=str(data["message"]),
                   fire_on=str(data.get("fire_on", "enter")),
                   active=bool(data.get("active", True)),
                   fired_ts=data.get("fired_ts"),
                   created_ts=float(data.get("created_ts", 0.0)))


def check_reminders(reminders: list[LocationReminder],
                    events: list[dict[str, Any]],
                    *, consent: bool,
                    ts: float | None = None) -> list[LocationReminder]:
    """Match reminders against fence events. Returns the ones that fired.

    A reminder fires when an event for its fence matches its
    ``fire_on`` direction, the reminder is active, and it has not
    already fired. Fired reminders get ``fired_ts`` set in place.
    """
    require_consent(consent, scope="reminders")
    now = ts if ts is not None else time.time()
    fired: list[LocationReminder] = []
    by_fence: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        by_fence.setdefault(str(ev.get("fence_id")), []).append(ev)
    for rem in reminders:
        if not rem.active or rem.fired:
            continue
        for ev in by_fence.get(rem.fence_id, []):
            want = "entered" if rem.fire_on == "enter" else "exited"
            if ev.get("event") == want and ev.get("person_id") == rem.person_id:
                rem.fired_ts = now
                fired.append(rem)
                break
    return fired
