"""Render queue = the mission queue. No new scheduler.

A render is a :class:`Mission` with tags ``["render", "media"]`` and
its plan in ``state["render"]``: the canonical (sources, spec, pack)
triple, the completed stages, and the outputs. Enqueue with
:func:`enqueue_render`; advance with :func:`advance_render`, which
runs exactly one stage (``proxy`` then ``full``) per call and
checkpoints after it.

Crash-resumable: completed stages are recorded in mission state and
checkpointed (``render:<stage>``), so a crash between stages costs at
most the in-flight stage — re-calling :func:`advance_render` resumes
at the next uncompleted stage and never re-runs a finished one.
Budgeted: the mission's own ``budget_wall`` / ``budget_tokens`` apply
— a budget-exhausted render fails without running another stage, and
each successful stage charges its wall time to the mission so a
restart cannot launder the budget.

There is no timer, thread, or poller here: advancement is
caller-driven (a runner tick, a CLI call, a webhook). A queue that
advances itself would be a second scheduler.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from ..core.errors import ValidationError
from ..media.edit_engine.pure_render import (
    canonical_spec_dict,
    render_cache_key,
)
from .mission import MissionStatus

__all__ = [
    "RENDER_STAGES",
    "enqueue_render",
    "advance_render",
    "render_mission_status",
]

#: stages in order. Proxy first (fast preview), full-res second.
RENDER_STAGES: tuple[str, ...] = ("proxy", "full")


def _render_state(mission: Any) -> dict[str, Any]:
    state = getattr(mission, "state", None) or {}
    render = state.get("render")
    if not isinstance(render, dict) or "cache_key" not in render:
        raise ValidationError(
            f"mission {getattr(mission, 'id', '?')} is not a render "
            "mission (no state['render'])")
    return render


def enqueue_render(store: Any, *, sources: Any, spec: Any,
                   pack: dict[str, Any] | None = None,
                   budget_wall: float = 0.0, budget_tokens: int = 0,
                   name: str = "") -> Any:
    """Enqueue a render as a mission. Returns the created mission."""
    if isinstance(sources, (str, bytes)) or hasattr(sources, "__fspath__"):
        sources = [sources]
    sources = [str(s) for s in sources]
    if not sources:
        raise ValidationError("enqueue_render needs at least one source")
    canonical = canonical_spec_dict(spec)
    key = render_cache_key(sources, canonical, pack or {})
    return store.create_new(
        f"render {key[:12]}",
        name=name or f"render {key[:12]}",
        budget_wall=budget_wall,
        budget_tokens=budget_tokens,
        state={"render": {
            "cache_key": key,
            "sources": sources,
            "spec": canonical,
            "pack": pack or {},
            "stages": {},
            "stage": "queued",
        }},
        metadata={"render_cache_key": key},
        tags=["render", "media"],
    )


def advance_render(store: Any, mission_id: str, *,
                   render_fn: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    """Run the next uncompleted stage of a render mission.

    ``render_fn(stage=..., state=...)`` performs one stage and returns
    ``{"output": <path>}``. Raises from ``render_fn`` propagate — the
    mission keeps its completed stages and checkpoints, so the caller
    can retry (a crash mid-stage is the same story: re-advance resumes
    after the last checkpointed stage).
    """
    if not callable(render_fn):
        raise ValidationError("advance_render needs a render_fn callable")
    mission = store.get(mission_id)
    if mission.terminal:
        return {"mission_id": mission.id, "status": mission.status,
                "done": True, "stage": None, "output": None}
    render = _render_state(mission)
    if mission.budget_exhausted:
        store.set_status(mission_id, MissionStatus.FAILED,
                         "render budget exhausted before the next stage")
        return {"mission_id": mission.id, "status": MissionStatus.FAILED,
                "done": True, "stage": None, "output": None,
                "note": "budget exhausted"}
    stages: dict[str, Any] = dict(render.get("stages") or {})
    pending = [s for s in RENDER_STAGES if s not in stages]
    if not pending:
        if mission.status != MissionStatus.DONE:
            store.set_status(mission_id, MissionStatus.DONE,
                             "all render stages complete")
        return {"mission_id": mission.id, "status": MissionStatus.DONE,
                "done": True, "stage": None,
                "output": stages.get("full", {}).get("output")}
    stage = pending[0]
    if mission.status != MissionStatus.RUNNING:
        store.set_status(mission_id, MissionStatus.RUNNING,
                         f"render stage {stage}")
        mission = store.get(mission_id)
        render = _render_state(mission)
        stages = dict(render.get("stages") or {})
    started = time.monotonic()
    result = render_fn(stage=stage, state=dict(render))
    output = result.get("output") if isinstance(result, dict) else None
    if not output:
        raise ValidationError(
            f"render_fn for stage {stage!r} returned no output")
    mission.charge(wall=max(0.0, time.monotonic() - started))
    stages[stage] = {"output": str(output),
                     "completed_at": time.time()}
    render = {**render, "stages": stages, "stage": stage}
    mission.state = {**(mission.state or {}), "render": render}
    store.save(mission)
    store.checkpoint(mission, label=f"render:{stage}")
    done = all(s in stages for s in RENDER_STAGES)
    if done:
        store.set_status(mission_id, MissionStatus.DONE,
                         "all render stages complete")
        status = MissionStatus.DONE
    else:
        status = MissionStatus.RUNNING
    return {"mission_id": mission.id, "status": status, "done": done,
            "stage": stage, "output": str(output)}


def render_mission_status(store: Any, mission_id: str) -> dict[str, Any]:
    """Small status summary for a render mission."""
    mission = store.get(mission_id)
    render = _render_state(mission)
    stages = render.get("stages") or {}
    return {"mission_id": mission.id, "status": mission.status,
            "cache_key": render.get("cache_key"),
            "stage": render.get("stage"),
            "completed_stages": sorted(stages),
            "pending_stages": [s for s in RENDER_STAGES if s not in stages],
            "outputs": {s: v.get("output") for s, v in stages.items()},
            "budget_exhausted": bool(mission.budget_exhausted),
            "done": all(s in stages for s in RENDER_STAGES)}
