"""The /plan service: survey → plan+spawn → save → show.

One call does the whole verb: survey the real code first (hard
failure when the survey is unavailable), plan and spawn through the
existing delegation manager, save the Implementation Plan durably,
and render the chat display.  The plan RUNS — there is no approval
gate (owner order 2026-11-16).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..delegation_manager import DelegationManager
from .errors import PlanError
from .render import plan_display
from .store import PlanStore, default_plans_dir, default_plans_path
from .survey import survey_task, survey_text

__all__ = ["plan_and_run"]


def _run_record(run: Any) -> dict[str, Any]:
    """One run as a plain JSON-safe dict.

    The manager returns ``DelegationRun`` objects; the record (and
    its JSON index) holds only plain data.  Dicts pass through with
    the same three keys.
    """
    if isinstance(run, dict):
        run_id = run.get("id", "")
        role = run.get("role", "")
        status = run.get("status", "")
    else:
        run_id = getattr(run, "id", "")
        role = getattr(run, "role", "")
        status = getattr(run, "status", "")
    return {
        "id": str(run_id or ""),
        "role": str(role or ""),
        "status": str(status or "unknown"),
    }


def _unassigned_record(entry: Any) -> dict[str, Any]:
    if isinstance(entry, dict):
        return {
            "step": str(entry.get("step", "") or ""),
            "role": str(entry.get("role", "") or ""),
            "reason": str(entry.get("reason", "") or "no roster worker"),
        }
    return {
        "step": str(getattr(entry, "step", "") or ""),
        "role": str(getattr(entry, "role", "") or ""),
        "reason": str(getattr(entry, "reason", "") or "no roster worker"),
    }
def _home_from_context(context: Any) -> Path | None:
    settings = getattr(context, "settings", None) if context is not None else None
    if settings is not None:
        for attr in ("home_path", "home"):
            value = getattr(settings, attr, None)
            if value:
                return Path(value)
    return None


def _default_store(context: Any) -> PlanStore:
    home = _home_from_context(context)
    if home is not None:
        return PlanStore(path=default_plans_path(home),
                         plans_dir=default_plans_dir(home))
    return PlanStore(path=None, plans_dir=None)


def _default_manager(context: Any) -> Any:
    if context is None:
        raise PlanError("/plan needs a runtime context to spawn workers")
    try:
        return DelegationManager.from_context(context)
    except Exception as exc:  # noqa: BLE001 — surface as a plan error
        raise PlanError(f"delegation manager unavailable: {exc}") from exc


def plan_and_run(context: Any, task: str, *, chat_key: str = "",
                 store: PlanStore | None = None, manager: Any = None,
                 graph: Any = None) -> dict[str, Any]:
    """Survey, plan, spawn, save, and render one task.

    ``store`` / ``manager`` / ``graph`` are injected seams (tests,
    and runtimes that already hold them); each defaults to the live
    object from ``context`` when omitted.
    """
    task = " ".join(str(task or "").split())
    if not task:
        raise PlanError("/plan needs a task to plan")
    survey = survey_task(task, graph=graph)
    if survey.get("unavailable"):
        raise PlanError(
            "cannot plan without a code survey: "
            f"{survey['unavailable']}")
    live_manager = manager if manager is not None else _default_manager(context)
    hint = survey_text(survey)
    try:
        result = live_manager.plan_and_spawn(
            task, chat_key=chat_key, context_hint=hint)
    except Exception as exc:  # noqa: BLE001 — surface as a plan error
        raise PlanError(f"planning failed: {exc}") from exc
    plan = result.get("plan") or {}
    if not isinstance(plan, dict):
        to_dict = getattr(plan, "to_dict", None)
        plan = to_dict() if callable(to_dict) else {}
    runs = [_run_record(run) for run in (result.get("runs") or [])]
    unassigned = [_unassigned_record(entry)
                  for entry in (result.get("unassigned") or [])]
    plan_error = result.get("plan_error") or ""
    live_store = store if store is not None else _default_store(context)
    record = live_store.save_plan(
        task=task, survey=survey, plan=plan, runs=runs,
        unassigned=unassigned, plan_error=plan_error, chat_key=chat_key)
    # The artifact is written once, now that the id exists.
    artifact_path = live_store.write_artifact(record)
    if artifact_path is not None:
        record["artifact_path"] = str(artifact_path)
    return {
        "ok": True,
        "plan_id": record["id"],
        "record": record,
        "display": plan_display(record),
        "survey": survey,
        "plan_error": plan_error,
        "runs": runs,
        "unassigned": unassigned,
    }
