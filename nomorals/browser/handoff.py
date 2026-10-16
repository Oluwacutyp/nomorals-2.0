"""Interactive handoff: the owner takes the mouse, then the agent resumes.

When the agent hits something it cannot — or should not — do alone (a
captcha only the owner may solve, a login, a judgment call), the run
pauses and hands control to the owner through the live view. The owner
acts in the real browser, then hands back, and the agent resumes from
exactly where it stopped.

State machine (the only legal transitions):

    IDLE --start_live()--> LIVE --pause()--> PAUSED --hand_mouse()--> HANDED
    HANDED --resume()--> RESUMED --finish()--> DONE
    PAUSED --resume()--> RESUMED   (owner had nothing to do)
    LIVE --pause()--> PAUSED,  PAUSED --show_live()--> PAUSED (read-only),
    LIVE --show_live()--> LIVE (read-only)

Every illegal transition raises :class:`HandoffError` naming the current
state and the attempted move — a handoff that silently skipped a state
would be a lie about who was in control. State persists to JSON so a
handoff survives a restart: a session loaded as HANDED still waits for
the owner, never auto-resumes.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..core.logging_setup import get_logger

__all__ = [
    "HandoffError",
    "HandoffSession",
    "STATE_IDLE",
    "STATE_LIVE",
    "STATE_PAUSED",
    "STATE_HANDED",
    "STATE_RESUMED",
    "STATE_DONE",
]

_log = get_logger(__name__)

STATE_IDLE = "idle"
STATE_LIVE = "live"
STATE_PAUSED = "paused"
STATE_HANDED = "handed"
STATE_RESUMED = "resumed"
STATE_DONE = "done"

#: Legal transitions: {from_state: {method: to_state}}.
_TRANSITIONS: dict[str, dict[str, str]] = {
    STATE_IDLE: {"start_live": STATE_LIVE},
    STATE_LIVE: {"pause": STATE_PAUSED, "show_live": STATE_LIVE},
    STATE_PAUSED: {"hand_mouse": STATE_HANDED, "resume": STATE_RESUMED,
                   "show_live": STATE_PAUSED},
    STATE_HANDED: {"resume": STATE_RESUMED, "show_live": STATE_HANDED},
    STATE_RESUMED: {"pause": STATE_PAUSED, "finish": STATE_DONE,
                    "show_live": STATE_RESUMED},
    STATE_DONE: {},
}


class HandoffError(ValueError):
    """An illegal handoff transition, or a handoff used in the wrong state."""


@dataclass
class _HandoffEvent:
    action: str
    from_state: str
    to_state: str
    ts: float = 0.0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HandoffSession:
    """One interactive handoff, as a strict state machine.

    ``view`` is an optional callable invoked by :meth:`show_live` with
    the current state — e.g. a ``LiveView.update``-style hook. It is
    never required: handoff works with no live view attached.
    """

    def __init__(self, session_id: str = "", *,
                 task_label: str = "",
                 view: Callable[[str, str], Any] | None = None,
                 state_path: str | Path | None = None) -> None:
        self.session_id = (session_id or "").strip() or f"handoff-{int(time.time() * 1000)}"
        self.task_label = (task_label or "").strip() or "browsing"
        self.view = view
        self.state_path = Path(state_path).expanduser() if state_path else None
        self.state = STATE_IDLE
        self.events: list[_HandoffEvent] = []
        self.owner_notes: list[str] = []
        self.started_at: float = 0.0
        self.handed_at: float = 0.0
        self.resumed_at: float = 0.0

    # -- machine ------------------------------------------------------------------
    def _move(self, action: str, *, note: str = "") -> str:
        allowed = _TRANSITIONS.get(self.state, {})
        if action not in allowed:
            raise HandoffError(
                f"handoff {self.session_id!r}: cannot {action}() from "
                f"state {self.state!r} (legal from here: "
                f"{sorted(allowed) or ['none — session is done']})")
        previous = self.state
        self.state = allowed[action]
        self.events.append(_HandoffEvent(
            action=action, from_state=previous, to_state=self.state,
            ts=time.time(), note=note or ""))
        _log.info("handoff %s: %s -> %s (%s)", self.session_id, previous, self.state, action)
        self._persist()
        return self.state

    def start_live(self, *, note: str = "") -> str:
        """Agent starts driving with the live view attached."""
        if self.started_at == 0.0:
            self.started_at = time.time()
        return self._move("start_live", note=note)

    def pause(self, *, note: str = "") -> str:
        """Agent stops driving and waits (for the owner, or for resume)."""
        return self._move("pause", note=note)

    def show_live(self, *, note: str = "") -> dict[str, Any]:
        """Show the current live frame. Read-only: the state does not
        change (a view is not a transfer of control)."""
        self._move("show_live", note=note)
        if self.view is not None:
            try:
                self.view(self.state, self.task_label)
            except Exception as exc:  # noqa: BLE001 - a view failure is not a handoff failure
                _log.debug("handoff %s: view hook failed: %r", self.session_id, exc)
        return self.status()

    def hand_mouse(self, *, note: str = "") -> str:
        """Transfer control to the owner. Only from PAUSED — the agent
        must have stopped driving before the owner takes the mouse."""
        self.handed_at = time.time()
        return self._move("hand_mouse", note=note)

    def owner_note(self, note: str) -> None:
        """Record what the owner did while in control. Allowed only
        while HANDED — notes from any other state would misattribute
        who acted."""
        if self.state != STATE_HANDED:
            raise HandoffError(
                f"handoff {self.session_id!r}: owner_note() only while "
                f"HANDED, currently {self.state!r}")
        text = (note or "").strip()
        if text:
            self.owner_notes.append(text)
        self._persist()

    def resume(self, *, note: str = "") -> str:
        """Owner hands back; the agent resumes. From HANDED (owner acted)
        or PAUSED (owner had nothing to do)."""
        self.resumed_at = time.time()
        return self._move("resume", note=note)

    def finish(self, *, note: str = "") -> str:
        """The resumed run completed."""
        return self._move("finish", note=note)

    # -- status / persistence -------------------------------------------------------
    def status(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "task_label": self.task_label,
            "state": self.state,
            "owner_notes": list(self.owner_notes),
            "events": [e.to_dict() for e in self.events],
            "started_at": self.started_at,
            "handed_at": self.handed_at,
            "resumed_at": self.resumed_at,
        }

    def _persist(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.status(), indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(self.state_path)

    def save(self, path: str | Path | None = None) -> Path:
        dest = Path(path).expanduser() if path else self.state_path
        if dest is None:
            raise HandoffError("no state path — pass one to save() or the constructor")
        self.state_path = dest
        self._persist()
        return dest

    @classmethod
    def load(cls, path: str | Path, *,
             view: Callable[[str, str], Any] | None = None) -> "HandoffSession":
        """Restore a handoff. A session that was HANDED stays HANDED —
        it never auto-resumes across a restart."""
        p = Path(path).expanduser()
        if not p.is_file():
            raise HandoffError(f"no handoff state at {p}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise HandoffError(f"corrupt handoff state {p}: {exc}") from exc
        if not isinstance(raw, dict) or "state" not in raw:
            raise HandoffError(f"corrupt handoff state {p}: missing state")
        session = cls.__new__(cls)
        session.session_id = str(raw.get("session_id") or "")
        session.task_label = str(raw.get("task_label") or "")
        session.view = view
        session.state_path = p
        state = str(raw.get("state") or "")
        if state not in _TRANSITIONS:
            raise HandoffError(f"corrupt handoff state {p}: unknown state {state!r}")
        session.state = state
        session.events = [
            _HandoffEvent(action=str(e.get("action") or ""),
                          from_state=str(e.get("from_state") or ""),
                          to_state=str(e.get("to_state") or ""),
                          ts=float(e.get("ts") or 0.0),
                          note=str(e.get("note") or ""))
            for e in (raw.get("events") or []) if isinstance(e, dict)
        ]
        session.owner_notes = [str(n) for n in (raw.get("owner_notes") or [])]
        session.started_at = float(raw.get("started_at") or 0.0)
        session.handed_at = float(raw.get("handed_at") or 0.0)
        session.resumed_at = float(raw.get("resumed_at") or 0.0)
        return session
