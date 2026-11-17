"""The idle trigger: speak first at a natural breakpoint, never twice.

danpeg/clicky's Tutor Mode (``UserActivityIdleDetector.swift``) is
the reference design: idle = >= 3.0 s since the last user input
**AND** the user has performed at least one action since the last
observation. That second condition is the key decision — after each
observation the flag is cleared, so the trigger can never fire twice
on the same pause, and never fires while the user is simply AFK.
The wiring adds an overlap guard (no observation already in
flight) and an arm/disarm switch.

Devon's input is her own DM traffic: the runtime stamps every
owner-DM arrival through :meth:`note_activity`. The tick
(:func:`run_visual_idle_tick_once`, driven by the durable
``visual idle tick`` job) fires the trigger and, when there is
concrete unfinished work — a queued or running workflow, or an
open plan — publishes exactly one Notifier note naming it. When
nothing is open the tick stays silent: an idle owner with nothing
waiting gets no ping at all.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

from ..core.failure_store import home_dir as _devon_home_dir

__all__ = [
    "DEFAULT_IDLE_SECONDS",
    "IDLE_JOB",
    "IDLE_SPEC",
    "IDLE_TOOL",
    "IdleTrigger",
    "ensure_visual_idle_job",
    "get_idle_trigger",
    "run_visual_idle_tick_once",
]

#: danpeg's threshold: 3.0 s of no input.
DEFAULT_IDLE_SECONDS = 3.0

IDLE_JOB = "visual idle tick"
IDLE_SPEC = "every 5m"
IDLE_TOOL = "__visual_idle_tick__"

#: States with unfinished work (workflow store + plan store).
_OPEN_WORKFLOW_STATES = ("queued", "running")
_OPEN_PLAN_STATES = ("running",)


class IdleTrigger:
    """A breakpoint detector with the acted-since-last-check gate."""

    def __init__(self, *, threshold_seconds: float = DEFAULT_IDLE_SECONDS,
                 home: Path | str | None = None) -> None:
        self.threshold_seconds = float(threshold_seconds)
        self._home = home
        self._last_activity: float | None = None
        self._acted = False
        self._in_flight = False
        self._armed = True
        self._load_flag()

    # ── durable arm flag ─────────────────────────────────────────

    def _flag_path(self) -> Path:
        base = (Path(self._home) if self._home is not None
                else Path(_devon_home_dir()))
        return base / "visuals" / "idle.json"

    def _load_flag(self) -> None:
        try:
            raw = json.loads(self._flag_path().read_text(
                encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(raw, dict) and isinstance(raw.get("armed"), bool):
            self._armed = raw["armed"]

    def _save_flag(self) -> None:
        try:
            path = self._flag_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps({"armed": self._armed}),
                           encoding="utf-8")
            os.replace(tmp, path)
        except OSError:
            pass

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self) -> None:
        self._armed = True
        self._save_flag()

    def disarm(self) -> None:
        self._armed = False
        self._save_flag()

    # ── activity ─────────────────────────────────────────────────

    def note_activity(self, now: float | None = None) -> None:
        """The owner did something: stamp it, set the acted flag."""
        self._last_activity = time.time() if now is None else now
        self._acted = True

    def idle_seconds(self, now: float | None = None) -> float | None:
        """Seconds since the last activity (``None``: never acted)."""
        if self._last_activity is None:
            return None
        moment = time.time() if now is None else now
        return moment - self._last_activity

    def check(self, now: float | None = None) -> bool:
        """True when a breakpoint is reached: armed, idle past the
        threshold, acted since the last observation, and no
        observation already in flight."""
        if not self._armed or not self._acted or self._in_flight:
            return False
        idle = self.idle_seconds(now)
        if idle is None:
            return False
        return idle >= self.threshold_seconds

    def begin_observation(self, now: float | None = None) -> bool:
        """Claim the breakpoint (overlap guard). False when there is
        nothing to observe or one is already running."""
        if not self.check(now):
            return False
        self._in_flight = True
        return True

    def observation_complete(self) -> None:
        """The observation is done: clear the acted flag (so this
        pause can never fire again) and the in-flight guard."""
        self._acted = False
        self._in_flight = False


_TRIGGERS: dict[str, IdleTrigger] = {}


def get_idle_trigger(home: Path | str | None = None) -> IdleTrigger:
    """The process-wide trigger for a home (one trigger per home)."""
    key = str(Path(home) if home is not None
              else Path(_devon_home_dir()))
    trigger = _TRIGGERS.get(key)
    if trigger is None:
        trigger = IdleTrigger(home=home)
        _TRIGGERS[key] = trigger
    return trigger


# ── the tick ─────────────────────────────────────────────────────


def _open_work(home: Path | str | None) -> dict[str, Any] | None:
    """The newest unfinished item, if any: a queued/running
    workflow first, else an open plan. ``None`` when nothing is
    waiting — the tick then stays silent."""
    with contextlib.suppress(Exception):
        from ..agents.workflow.store import WorkflowStore, default_workflows_path
        for item in WorkflowStore(
                default_workflows_path(home)).list_workflows():
            if item.get("state") in _OPEN_WORKFLOW_STATES:
                return {"kind": "workflow",
                        "name": str(item.get("name") or item.get("id")),
                        "task": str(item.get("task") or "")}
    with contextlib.suppress(Exception):
        from ..agents.plan.store import PlanStore, default_plans_path

        for item in PlanStore(
                default_plans_path(home)).list_plans():
            if item.get("state") in _OPEN_PLAN_STATES:
                return {"kind": "plan",
                        "name": str(item.get("name") or item.get("id")),
                        "task": str(item.get("task") or "")}
    return None


def run_visual_idle_tick_once(
        home: Path | str | None = None, *,
        notifier: Any = None,
        trigger: IdleTrigger | None = None,
        now: float | None = None) -> dict[str, Any]:
    """One idle tick: fire the trigger, notify once about open work.

    Returns ``{"ok", "fired", "notified", "item"}``. A fired tick
    with nothing open notifies nobody. The trigger's acted flag is
    cleared on every fired tick, so a pause can never produce two
    notes. Never raises.
    """
    active = trigger if trigger is not None else get_idle_trigger(home)
    if not active.begin_observation(now):
        return {"ok": True, "fired": False, "notified": False,
                "item": None}
    try:
        item = _open_work(home)
        if item is None or notifier is None:
            return {"ok": True, "fired": True, "notified": False,
                    "item": item}
        title = (f"Still working on {item['kind']} "
                 f"'{item['name']}'")
        body = item["task"] or (
            f"You went quiet — {item['kind']} '{item['name']}' "
            "is still open.")
        try:
            notifier.publish("visual_idle", title, body)
        except Exception as exc:  # noqa: BLE001 - a failed note is data
            return {"ok": False, "fired": True, "notified": False,
                    "item": item, "error": str(exc)}
        return {"ok": True, "fired": True, "notified": True,
                "item": item}
    finally:
        active.observation_complete()


def ensure_visual_idle_job(scheduler: Any, *,
                           spec: str = IDLE_SPEC) -> Any:
    """Register the durable visual idle tick on the live scheduler.

    Mirror of the distill / continual-update / swarm-board ensures:
    a routine tool job, one per scheduler, replaced when its spec
    goes stale. Never raises — scheduling is best-effort.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name != IDLE_JOB:
                continue
            job_spec = (job.get("spec") if isinstance(job, dict)
                        else getattr(job, "spec", None))
            if job_spec == spec:
                return job
            with contextlib.suppress(Exception):
                job_id = (job.get("id") if isinstance(job, dict)
                          else getattr(job, "id", None))
                scheduler.remove(job_id)
            break
        return scheduler.add(IDLE_JOB, spec, "tool",
                             {"tool": IDLE_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
