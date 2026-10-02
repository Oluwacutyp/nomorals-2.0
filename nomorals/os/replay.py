"""Timeline-based mission replay and redrive (Wave J, L6).

:func:`replay_mission` reconstructs a mission's history from the H2 event
timeline — state transitions, task activity, verdicts, and the artifacts
it produced — as a structured report with both a human narrative and a
JSON-serializable dict. It is strictly read-only: it never writes to the
timeline, the mission store, or the artifact store.

:func:`redrive_mission` re-drives a *terminal* (completed/failed/cancelled)
mission's plan from its recorded history: it creates a new mission with the
same goal and a copy of the recorded plan, transitions it to PLANNED, and
records a redrive artifact whose provenance links ``derived_from`` the
original mission's artifacts in the artifact graph. It refuses without
``confirm=True`` and refuses to redrive a mission that is still running.

Tool calls: the H2 timeline persists the ``mission.*``, ``artifact.*``,
``session.*`` and ``task.*`` bus families. There is no ``tool.call`` bus
topic yet, so per-tool invocations appear under task activity only when a
sibling emits ``task.*`` events for them — replay surfaces whatever the
timeline actually recorded rather than inventing tool calls.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..core.errors import ValidationError
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from .mission_state import PLANNED, current_state, transition

__all__ = [
    "MissionReplay",
    "RedriveRefused",
    "RedriveReport",
    "redrive_mission",
    "replay_mission",
]

_log = get_logger(__name__)


class RedriveRefused(ValidationError):
    """Raised when a redrive is not allowed.

    Reasons: ``confirm`` not given, mission unknown, or the mission is
    still live (running/paused). A dead runner's zombie "running" mission
    is reconciled first, so only genuinely live missions are refused.
    """


# ── replay ───────────────────────────────────────────────────────────────────


def _fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC).strftime(
            "%Y-%m-%d %H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return "?"


def _event_summary(topic: str, data: dict[str, Any]) -> str:
    """One-line human summary of a task/other event's payload."""
    for key in ("summary", "detail", "note", "message", "step", "task",
                "name", "status", "tool"):
        value = data.get(key)
        if value:
            return f"{key}={value}"
    items = [f"{k}={v}" for k, v in list(data.items())[:3]]
    return " ".join(items) if items else topic


@dataclass
class MissionReplay:
    """A mission's history, rebuilt from the persisted event timeline."""

    mission_id: str
    event_count: int = 0
    covered_from: float | None = None
    covered_to: float | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    tasks: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    verifications: list[dict[str, Any]] = field(default_factory=list)
    redrives: list[dict[str, Any]] = field(default_factory=list)
    other: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "event_count": self.event_count,
            "covered_from": self.covered_from,
            "covered_to": self.covered_to,
            "transitions": self.transitions,
            "tasks": self.tasks,
            "artifacts": self.artifacts,
            "verifications": self.verifications,
            "redrives": self.redrives,
            "other": self.other,
        }

    def narrative(self) -> str:
        """Human-readable retelling of the mission, oldest event first."""
        lines = [f"mission {self.mission_id} — replay of "
                 f"{self.event_count} timeline events"]
        if self.covered_from is not None:
            lines[0] += (f" ({_fmt_ts(self.covered_from)} → "
                         f"{_fmt_ts(self.covered_to or self.covered_from)})")

        def section(title: str, rows: list[dict[str, Any]],
                    fmt: Any) -> None:
            lines.append(f"{title} ({len(rows)}):")
            if not rows:
                lines.append("  (none recorded)")
            for row in rows:
                lines.append(f"  {fmt(row)}")

        section("state transitions", self.transitions,
                lambda t: f"[{_fmt_ts(t['ts'])}] {t['from']} -> {t['to']}"
                + (f" — {t['note']}" if t.get("note") else ""))
        section("task activity", self.tasks,
                lambda t: f"[{_fmt_ts(t['ts'])}] {t['topic']} — {t['summary']}")
        section("artifacts produced", self.artifacts,
                lambda a: f"[{_fmt_ts(a['ts'])}] artifact://{a['artifact_id']} "
                f"({a['type']})" + (f" by {a['creator']}" if a.get("creator")
                                    else ""))
        section("verdicts", self.verifications,
                lambda v: f"[{_fmt_ts(v['ts'])}] verdict={v['verdict']}"
                + (f" — {v['details']}" if v.get("details") else ""))
        if self.redrives:
            section("redrives", self.redrives,
                    lambda r: f"[{_fmt_ts(r['ts'])}] redriven as "
                    f"{r.get('new_mission_id', '?')}")
        if self.other:
            section("other events", self.other,
                    lambda o: f"[{_fmt_ts(o['ts'])}] {o['topic']} — "
                    f"{o['summary']}")
        return "\n".join(lines)


def replay_mission(timeline: Any, mission_id: str,
                   *, limit: int = 2000) -> MissionReplay:
    """Rebuild ``mission_id``'s history from the timeline. Read-only.

    ``timeline`` is a :class:`~nomorals.os.timeline.Timeline` (duck-typed:
    anything with ``query(mission_id=..., limit=...)``). Events are
    classified by topic; unknown topics land in ``other`` rather than being
    dropped.
    """
    rows = timeline.query(mission_id=mission_id, limit=limit)
    rows = sorted(rows, key=lambda r: (r.get("ts") or 0.0,
                                       str(r.get("event_id") or "")))
    report = MissionReplay(mission_id=mission_id, event_count=len(rows))
    if rows:
        report.covered_from = rows[0].get("ts")
        report.covered_to = rows[-1].get("ts")
    for row in rows:
        topic = str(row.get("topic") or "")
        data = row.get("data") or {}
        ts = row.get("ts")
        if topic == "mission.transition":
            report.transitions.append({
                "ts": ts,
                "from": str(data.get("from_state") or "?"),
                "to": str(data.get("to_state") or "?"),
                "note": str(data.get("note") or ""),
            })
        elif topic == "mission.verify":
            report.verifications.append({
                "ts": ts,
                "verdict": str(data.get("verdict") or "?"),
                "details": str(data.get("details") or ""),
            })
        elif topic == "mission.redrive":
            report.redrives.append({
                "ts": ts,
                "new_mission_id": str(data.get("mission_id") or ""),
                "note": str(data.get("note") or ""),
            })
        elif topic == "artifact.created":
            report.artifacts.append({
                "ts": ts,
                "artifact_id": str(data.get("artifact_id") or ""),
                "uri": str(data.get("uri") or ""),
                "type": str(data.get("type") or ""),
                "creator": str(data.get("creator") or ""),
            })
        elif topic.startswith("task."):
            report.tasks.append({
                "ts": ts,
                "topic": topic,
                "summary": _event_summary(topic, data),
            })
        else:
            report.other.append({
                "ts": ts,
                "topic": topic,
                "summary": _event_summary(topic, data),
            })
    return report


# ── redrive ──────────────────────────────────────────────────────────────────


@dataclass
class RedriveReport:
    """What :func:`redrive_mission` created."""

    new_mission_id: str
    original_mission_id: str
    plan_steps: int
    artifact_id: str = ""
    derived_from_artifact_ids: list[str] = field(default_factory=list)
    os_state: str = PLANNED

    def to_dict(self) -> dict[str, Any]:
        return {
            "new_mission_id": self.new_mission_id,
            "original_mission_id": self.original_mission_id,
            "plan_steps": self.plan_steps,
            "artifact_id": self.artifact_id,
            "derived_from_artifact_ids": self.derived_from_artifact_ids,
            "os_state": self.os_state,
        }


def redrive_mission(
    mission_store: Any,
    mission_id: str,
    *,
    confirm: bool = False,
    artifact_store: Any = None,
    note: str = "",
) -> RedriveReport:
    """Re-drive a terminal mission's recorded plan as a brand-new mission.

    The new mission carries the original's goal and a deep copy of its
    recorded plan (``state["plan"]``), fresh budgets copied from the
    original, and ``metadata["derived_from_mission"]`` pointing at the
    original. A redrive-record artifact is written with
    ``provenance.derived_from`` naming every artifact the original mission
    produced — that is the artifact-graph link back to the original.

    The new mission is left in PLANNED; it is *not* auto-executed — the
    operator starts it with ``nm missions --resume <new-id>`` (or the
    runner picks it up via ``resume_all``).

    Raises :class:`RedriveRefused` without ``confirm=True``, for an unknown
    mission, or when the mission is still live.
    """
    if not confirm:
        raise RedriveRefused(
            "redrive creates a new mission — pass confirm=True "
            "(CLI: --confirm) to proceed")
    # A dead runner can leave status="running" with nobody driving it;
    # reconcile flips those to failed so they become redrivable.
    mission_store.reconcile(mission_id)
    mission = mission_store.get(mission_id)  # raises NotFound when unknown
    if not mission.terminal:
        state = current_state(mission)
        raise RedriveRefused(
            f"mission {mission_id} is {mission.status} (os: {state}): "
            "only completed/failed/cancelled missions can be redriven")

    plan = copy.deepcopy(mission.state.get("plan") or [])
    plan_steps = [p for p in plan if isinstance(p, dict)
                  and "__plan_error__" not in p]
    new_mission = mission_store.create_new(
        mission.goal,
        name=f"{mission.name} (redrive)",
        budget_wall=mission.budget_wall,
        budget_tokens=mission.budget_tokens,
        state={
            "plan": plan,
            "completed_steps": [],
            "redrive": {
                "of": mission.id,
                "at": time.time(),
                "note": note,
                "original_status": mission.status,
                "original_iterations": mission.iterations,
                "original_success": mission.success,
            },
        },
        metadata={
            "derived_from_mission": mission.id,
            "redrive_of": mission.id,
        },
    )
    transition(mission_store, new_mission.id, PLANNED,
               note=note or f"redrive of {mission.id}")

    artifact_id = ""
    parent_artifact_ids: list[str] = []
    if artifact_store is not None:
        parent_artifact_ids = [a.id for a in
                               artifact_store.for_mission(mission.id)]
        record = {
            "kind": "redrive_record",
            "redrive_of": mission.id,
            "new_mission_id": new_mission.id,
            "goal": mission.goal,
            "plan_steps": len(plan_steps),
            "derived_from_mission": mission.id,
            "created_at": time.time(),
        }
        artifact = artifact_store.derive(
            json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8"),
            from_ids=parent_artifact_ids,
            type="json",
            creator="mission.redrive",
            mission_id=new_mission.id,
            metadata={"derived_from_mission": mission.id,
                      "kind": "redrive_record"},
        )
        artifact_id = artifact.id

    _emit_redrive(new_mission.id, mission.id, artifact_id, note)
    _log.info("mission %s redriven as %s (%d plan steps)",
              mission.id, new_mission.id, len(plan_steps))
    return RedriveReport(
        new_mission_id=new_mission.id,
        original_mission_id=mission.id,
        plan_steps=len(plan_steps),
        artifact_id=artifact_id,
        derived_from_artifact_ids=parent_artifact_ids,
        os_state=PLANNED,
    )


def _emit_redrive(new_mission_id: str, original_mission_id: str,
                  artifact_id: str, note: str) -> None:
    """Publish ``mission.redrive`` so the timeline records it. Best-effort."""
    try:
        global_bus.publish(Event(
            topic="mission.redrive",
            data={"mission_id": new_mission_id,
                  "redrive_of": original_mission_id,
                  "artifact_id": artifact_id,
                  "note": note},
            source="nomorals.os.replay",
        ))
    except Exception:  # noqa: BLE001 - events never break a redrive
        _log.debug("mission.redrive event failed", exc_info=True)
