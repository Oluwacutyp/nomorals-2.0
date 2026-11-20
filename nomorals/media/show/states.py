"""Show states — the named states a live show moves through.

The Bevy pattern, in Python: a ``States`` enum plus the table of
legal transitions. A transition is requested (``Show.request``)
and applied at the next schedule boundary (``Show.step``) — never
mid-tick, so a state change can never cut a phase in half. An
illegal request is refused naming both states and the legal
exits from the current one.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ShowState",
    "TRANSITIONS",
    "legal",
]


class ShowState(str, Enum):  # noqa: UP042 - StrEnum needs 3.11+; CI tests 3.10
    """The states of a live show."""

    IDLE = "idle"
    ARMING = "arming"
    LIVE = "live"
    PAUSED = "paused"
    STOPPING = "stopping"
    DONE = "done"
    FAILED = "failed"


#: The legal transitions. Everything else is refused.
TRANSITIONS: dict[ShowState, set[ShowState]] = {
    ShowState.IDLE: {ShowState.ARMING},
    ShowState.ARMING: {ShowState.LIVE, ShowState.STOPPING,
                       ShowState.FAILED},
    ShowState.LIVE: {ShowState.PAUSED, ShowState.STOPPING,
                     ShowState.FAILED},
    ShowState.PAUSED: {ShowState.LIVE, ShowState.STOPPING,
                       ShowState.FAILED},
    ShowState.STOPPING: {ShowState.DONE, ShowState.FAILED},
    ShowState.DONE: {ShowState.ARMING},
    ShowState.FAILED: {ShowState.ARMING},
}


def legal(from_state: ShowState, to_state: ShowState) -> bool:
    """True when the transition is in the table."""
    return to_state in TRANSITIONS.get(from_state, set())
