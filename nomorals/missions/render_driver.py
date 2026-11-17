"""Driver for the mission render queue (audit gap R2-33).

``missions/render_queue.py`` was complete and test-pinned but had zero
production callers: no tick ever advanced a render mission, nothing
could enqueue one, and a failed stage was recorded nowhere. This
module is the wiring that makes it run — it is not a second renderer:

* :func:`run_render_tick_once` — one tick: every non-terminal
  render-tagged mission advances exactly one stage (``proxy`` then
  ``full``) through :func:`advance_render`, whose stage function is
  the real engine — :func:`pure_render.render_pure` (cache-aware
  ffmpeg render), proxy stage via ``proxy_spec``. The segment-level
  media ``RenderQueue`` is deliberately NOT used: it renders segments,
  not proxy/full stages, and routing the mission queue through it
  would make two queues own the same work. One queue owns the work;
  ``render_pure`` is the engine, not a queue.
* A stage failure is recorded on the mission (``last_error``,
  ``failed_stage``, ``consecutive_failures`` in ``state["render"]``)
  and retried next tick; three consecutive failures fail the mission
  with an explicit note. A success resets the counter. A
  budget-exhausted mission never renders (``advance_render`` fails it
  without running a stage).
* The owner is told once when a render finishes (or fails) via
  ``Notifier.publish("render", …)`` — marker
  ``state["render"]["notified"]`` so later ticks never re-publish.
* ``ensure_render_tick_job`` + the ``__render_tick__`` routine tool —
  a durable every-5m scheduler job, the swarm-board / workflow /
  drift / continual pattern.

Layering: L5 (``missions``), same package as ``render_queue.py``.
It never imports the notifier — the tick takes an injected one (the
digest lane's lesson).
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..core.failures import record_failure
from ..core.logging_setup import get_logger
from ..media.edit_engine.pure_render import proxy_spec, render_pure
from .mission import Mission, MissionStatus, MissionStore
from .render_queue import RENDER_STAGES, advance_render

_log = get_logger(__name__)

__all__ = [
    "MAX_STAGE_FAILURES",
    "RENDER_JOB",
    "RENDER_SPEC",
    "RENDER_TOOL",
    "default_render_cache_dir",
    "ensure_render_tick_job",
    "run_render_tick_once",
    "stage_render_fn",
]

RENDER_JOB = "render tick"
RENDER_SPEC = "every 5m"
RENDER_TOOL = "__render_tick__"

#: A stage that fails this many times in a row fails the mission.
#: One crash is a retry, not a verdict; three in a row is a verdict.
MAX_STAGE_FAILURES = 3


def default_render_cache_dir(home: Path | str | None = None) -> Path:
    """Where stage renders are cached under a runtime's home."""
    base = Path(home) if home is not None else Path("~/.nomorals").expanduser()
    return base / "data" / "renders" / "cache"


def _home_of(context: Any) -> Path | None:
    home = getattr(getattr(context, "settings", None), "home", None)
    if home is None:
        home = getattr(context, "home", None)
    return Path(home) if home is not None else None


def stage_render_fn(stage: str, state: dict[str, Any], *,
                    cache_dir: Path | str,
                    renderer: Any = None) -> dict[str, Any]:
    """Render one mission-queue stage through the real engine.

    ``proxy`` renders the 480p proxy spec, ``full`` the full spec —
    both through :func:`render_pure`, so a re-run with an unchanged
    triple is a cache hit and zero ffmpeg. Returns ``{"output": ...}``,
    the shape :func:`advance_render` requires.
    """
    if stage not in RENDER_STAGES:
        raise ValueError(f"unknown render stage {stage!r}")
    spec = state["spec"]
    eff_spec = proxy_spec(spec) if stage == "proxy" else spec
    result = render_pure(state["sources"], eff_spec, state.get("pack") or {},
                         cache_dir=cache_dir, renderer=renderer)
    return {"output": result["output"]}


def _record_stage_failure(store: MissionStore, mission: Mission,
                          stage: str, exc: Exception) -> int:
    """Record a failed stage on the mission. Returns the new count."""
    render = dict((mission.state or {}).get("render") or {})
    count = int(render.get("consecutive_failures") or 0) + 1
    render["last_error"] = str(exc)[:500]
    render["failed_stage"] = stage
    render["consecutive_failures"] = count
    mission.state = {**(mission.state or {}), "render": render}
    store.save(mission)
    return count


def _notify_once(store: MissionStore, mission: Mission, *,
                 notifier: Any, title: str, body: str) -> bool:
    """Publish once for a terminal render; later ticks stay silent."""
    render = dict((mission.state or {}).get("render") or {})
    if render.get("notified"):
        return False
    if notifier is not None:
        with contextlib.suppress(Exception):
            notifier.publish("render", title, body)
    render["notified"] = True
    mission.state = {**(mission.state or {}), "render": render}
    store.save(mission)
    return True


def _next_stage(store: MissionStore, mission: Mission) -> str | None:
    """The next uncompleted stage, or None when all are done."""
    render = (mission.state or {}).get("render") or {}
    stages = render.get("stages") or {}
    for stage in RENDER_STAGES:
        if stage not in stages:
            return stage
    return None


def run_render_tick_once(context: Any, *,
                         store: MissionStore | None = None,
                         render_fn: Callable[..., dict[str, Any]] | None = None,
                         notifier: Any = None,
                         cache_dir: Path | str | None = None
                         ) -> dict[str, Any]:
    """Run one render tick. Never raises.

    Every non-terminal render-tagged mission advances exactly one
    stage. A stage failure is recorded on the mission and retried
    next tick; three in a row fails it. A finished render notifies
    the owner once with the output path.
    """
    try:
        if store is None:
            store = MissionStore(getattr(context, "db", None))
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("render.tick.setup", exc)
        return {"ok": False, "status": "failed",
                "error": f"render tick failed: {exc}"}
    try:
        return _run_once(context, store=store, render_fn=render_fn,
                         notifier=notifier, cache_dir=cache_dir)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("render.tick", exc)
        return {"ok": False, "status": "failed",
                "error": f"render tick failed: {exc}"}


def _run_once(context: Any, *, store: MissionStore,
              render_fn: Callable[..., dict[str, Any]] | None,
              notifier: Any, cache_dir: Path | str | None
              ) -> dict[str, Any]:
    if cache_dir is None:
        cache_dir = default_render_cache_dir(_home_of(context))
    if render_fn is None:
        def render_fn(*, stage: str, state: dict[str, Any],
                      _cache_dir: Path | str = cache_dir
                      ) -> dict[str, Any]:
            return stage_render_fn(stage, state, cache_dir=_cache_dir)
    try:
        due = store.list(tag="render", active_only=True, limit=50)
    except Exception as exc:  # noqa: BLE001 - a bad store is not a crash
        record_failure("render.tick.list", exc)
        return {"ok": False, "status": "failed",
                "error": f"render tick failed: {exc}"}
    advanced: list[str] = []
    done: list[str] = []
    failed: list[str] = []
    for mission in due:
        stage = _next_stage(store, mission)
        if stage is None:
            continue
        try:
            result = advance_render(store, mission.id, render_fn=render_fn)
        except Exception as exc:  # noqa: BLE001 - one bad mission
            # never kills the tick; the failure is recorded on the
            # mission itself, never dropped.
            count = _record_stage_failure(store, mission, stage, exc)
            _log.warning("render %s stage %s failed (%d/%d): %s",
                         mission.id, stage, count, MAX_STAGE_FAILURES, exc)
            if count >= MAX_STAGE_FAILURES:
                store.set_status(
                    mission.id, MissionStatus.FAILED,
                    f"render stage {stage} failed {count} times: {exc}")
                # Re-get AFTER set_status: it does its own get + save,
                # so a pre-status copy would clobber the FAILED flip
                # (and its note) when the notify marker is saved.
                fresh = store.get(mission.id)
                _notify_once(
                    store, fresh, notifier=notifier,
                    title="🎬 render failed",
                    body=(f"render {mission.id} failed — stage {stage} "
                          f"failed {count} times: {exc}"))
                failed.append(mission.id)
            continue
        # A stage completed: the failure counter resets.
        fresh = store.get(mission.id)
        render = dict((fresh.state or {}).get("render") or {})
        if render.get("consecutive_failures"):
            render["consecutive_failures"] = 0
            render.pop("last_error", None)
            render.pop("failed_stage", None)
            fresh.state = {**(fresh.state or {}), "render": render}
            store.save(fresh)
        status = str(result.get("status") or "")
        if status == MissionStatus.FAILED:
            failed.append(mission.id)
            _notify_once(
                store, fresh, notifier=notifier,
                title="🎬 render failed",
                body=(f"render {mission.id} failed — "
                      f"{result.get('note') or 'budget exhausted'}"))
        elif result.get("done"):
            done.append(mission.id)
            _notify_once(
                store, fresh, notifier=notifier,
                title="🎬 render done",
                body=(f"render {mission.id} is ready\n"
                      f"output: {result.get('output') or ''}"))
        else:
            advanced.append(mission.id)
    return {"ok": True, "advanced": advanced, "done": done,
            "failed": failed, "considered": len(due)}


def ensure_render_tick_job(scheduler: Any, *,
                           spec: str = RENDER_SPEC) -> Any:
    """Register the durable render tick on the live scheduler.

    Mirror of the swarm-board / digest / watch-loop / drift / workflow
    / continual ensures: a routine tool job, one per scheduler,
    replaced when its spec goes stale. Never raises — scheduling is
    best-effort; a missing or broken scheduler yields None.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name != RENDER_JOB:
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
        return scheduler.add(RENDER_JOB, spec, "tool",
                             {"tool": RENDER_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None
