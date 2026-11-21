"""The workflow runner: a durable scheduler tick + lifecycle verbs.

The runner is the ``__workflow_tick__`` routine tool job on the
live scheduler (every 5m, the delegation-manager tick's cadence).
The tick itself never executes a script: it starts queued runs on
background threads, polls live threads, delivers finished
summaries into the origin chat, and marks runs whose thread died
with the process ``interrupted`` (record + journal survive; the
owner re-runs the saved script with ``/workflow resume``).

Lifecycle verbs (``rerun`` / ``cancel`` / ``resume``) are thin
functions over the store so chat control stays thin; each needs
an actor (``by``), never an anonymous action.
"""

from __future__ import annotations

import contextlib
import threading
from pathlib import Path
from typing import Any

from .errors import WorkflowError
from .failures import record_failure
from .script import ScriptRuntime
from .store import WorkflowStore, default_workflows_dir, default_workflows_path

__all__ = [
    "WORKFLOW_JOB",
    "WORKFLOW_SPEC",
    "WORKFLOW_TOOL",
    "cancel_workflow",
    "ensure_workflow_job",
    "rerun_workflow",
    "resume_workflow",
    "run_workflow_tick_once",
]

WORKFLOW_JOB = "workflow tick"
WORKFLOW_SPEC = "every 5m"
WORKFLOW_TOOL = "__workflow_tick__"

_threads: dict[str, threading.Thread] = {}
_threads_lock = threading.Lock()


def ensure_workflow_job(scheduler: Any, *,
                        spec: str = WORKFLOW_SPEC) -> Any:
    """Register the durable workflow tick on the live scheduler.

    Mirror of the digest / watch-loop / drift ensures: a routine
    tool job, one per scheduler, replaced when its spec goes stale.
    Never raises — scheduling is best-effort; a missing or broken
    scheduler yields None.
    """
    if scheduler is None:
        return None
    try:
        for job in (scheduler.list_jobs() or []):
            name = (job.get("name") if isinstance(job, dict)
                    else getattr(job, "name", None))
            if name != WORKFLOW_JOB:
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
        return scheduler.add(WORKFLOW_JOB, spec, "tool",
                             {"tool": WORKFLOW_TOOL, "args": {}},
                             routine=True)
    except Exception:  # noqa: BLE001 - scheduling is best-effort
        return None


def _repo_root() -> Path:
    """The git checkout this code runs from (the code file is inside it)."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() and (parent / "nomorals").is_dir():
            return parent
    return Path.cwd()


def _store_from_context(context: Any) -> WorkflowStore:
    home = None
    settings = getattr(context, "settings", None)
    if settings is not None:
        home = getattr(settings, "home", None)
    if home is None:
        home = getattr(context, "home", None)
    return WorkflowStore(path=default_workflows_path(home),
                         scripts_dir=default_workflows_dir(home))


def _manager_from_context(context: Any) -> Any:
    """The process-long delegation manager (one per context)."""
    from ..delegation_manager import _tick_manager

    return _tick_manager(context)


def _conversation_log_from_context(context: Any) -> Any:
    db = getattr(context, "db", None)
    if db is None:
        return None
    try:
        from ...storage.conversations import ConversationLog

        return ConversationLog(db)
    except Exception:  # noqa: BLE001 - trajectory is advisory
        return None


def _execute_run(store: WorkflowStore, manager: Any,
                 record: dict[str, Any], context: Any) -> None:
    """One script, on its thread. Never raises out of the thread."""
    try:
        runtime = ScriptRuntime(
            run=record, store=store, manager=manager,
            conversation_log=_conversation_log_from_context(context),
            repo=_repo_root(), args=record.get("args") or {},
            context=context)
        runtime.run(record.get("script") or "")
    except Exception as exc:  # noqa: BLE001 - the thread never dies loud
        record_failure(f"workflow:{record['id']}:thread", exc)
        with contextlib.suppress(WorkflowError):
            store.mark_failed(record["id"],
                              error=f"{type(exc).__name__}: {exc}")


def _start_run(store: WorkflowStore, manager: Any,
               record: dict[str, Any], context: Any) -> bool:
    """Mark a queued run running and start its thread. False on failure."""
    try:
        store.mark_running(record["id"])
    except WorkflowError as exc:
        record_failure(f"workflow:{record['id']}:start", exc)
        return False
    thread = threading.Thread(
        target=_execute_run,
        args=(store, manager, record, context),
        name=f"workflow-{record['id']}", daemon=True)
    try:
        thread.start()
    except Exception as exc:  # noqa: BLE001 - a thread that never starts is a failed run
        record_failure(f"workflow:{record['id']}:start", exc)
        with contextlib.suppress(WorkflowError):
            store.mark_failed(record["id"],
                              error=f"could not start workflow thread: {exc}")
        return False
    with _threads_lock:
        _threads[record["id"]] = thread
    return True


def run_workflow_tick_once(context: Any, *,
                           notifier: Any = None,
                           manager: Any = None,
                           store: WorkflowStore | None = None
                           ) -> dict[str, Any]:
    """Run one workflow tick. Never raises.

    Starts every ``queued`` run, reaps finished threads, delivers
    finished summaries into the origin chat (once each, via the
    injected notifier), and marks runs whose thread is gone
    ``interrupted``. The notifier is injected — the runner never
    constructs one itself, so the tick stays testable without a
    runtime.
    """
    try:
        store = store if store is not None else _store_from_context(context)
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("workflow.tick.setup", exc)
        return {"ok": False, "error": f"workflow tick failed: {exc}"}
    started = 0
    delivered = 0
    interrupted = 0
    try:
        queued = [record for record in store.list_workflows()
                  if record.get("state") == "queued"]
        if queued and manager is None:
            # The manager is resolved only when a run actually
            # needs one — a tick with nothing to start must not
            # fail for want of a database.
            try:
                manager = _manager_from_context(context)
            except Exception as exc:  # noqa: BLE001 - the tick never dies
                record_failure("workflow.tick.setup", exc)
                return {"ok": False,
                        "error": f"workflow tick failed: {exc}"}
        for record in queued:
            if _start_run(store, manager, record, context):
                started += 1
        with _threads_lock:
            live = dict(_threads)
        for run_id, thread in live.items():
            if thread.is_alive():
                continue
            with _threads_lock:
                _threads.pop(run_id, None)
            thread.join(timeout=1.0)
        for record in store.list_workflows():
            state = record.get("state")
            if state == "running" and record["id"] not in _threads:
                # No live thread and still marked running: the
                # process died mid-run (restart) — the record and
                # journal survive; the owner re-runs the script.
                with contextlib.suppress(WorkflowError):
                    store.mark_interrupted(record["id"])
                    interrupted += 1
            elif state in ("done", "failed") \
                    and not record.get("summary_delivered") \
                    and notifier is not None:
                text = str(record.get("summary") or "")
                if not text and state == "failed":
                    text = (f"workflow {record.get('name', '')} "
                            f"failed: {record.get('error', '')}")
                if text:
                    try:
                        notifier.publish(
                            "workflow",
                            title=(f"Workflow "
                                   f"{record.get('name', '')} {state}"),
                            body=text)
                    except Exception as exc:  # noqa: BLE001 - delivery is best-effort
                        record_failure(
                            f"workflow:{record['id']}:deliver", exc)
                    else:
                        with contextlib.suppress(WorkflowError):
                            store.mark_summary_delivered(record["id"])
                            delivered += 1
        return {"ok": True, "started": started, "delivered": delivered,
                "interrupted": interrupted}
    except Exception as exc:  # noqa: BLE001 - the tick never dies
        record_failure("workflow.tick", exc)
        return {"ok": False, "error": f"workflow tick failed: {exc}"}


# ── lifecycle verbs (chat control stays thin) ───────────────────────

def _resolve_or_raise(store: WorkflowStore, ref: str) -> dict[str, Any]:
    record = store.resolve(ref)
    if record is None:
        raise WorkflowError(
            f"no workflow {ref!r} — /workflow lists them")
    return record


def rerun_workflow(store: WorkflowStore, ref: str, *,
                   by: str) -> dict[str, Any]:
    """Re-run a saved script: state → ``queued``. No approval step."""
    actor = (by or "").strip()
    if not actor:
        raise WorkflowError(
            "a workflow action needs an actor — who is asking?")
    record = _resolve_or_raise(store, ref)
    return store.requeue(record["id"], by=actor)


def cancel_workflow(store: WorkflowStore, ref: str, *,
                    by: str) -> dict[str, Any]:
    """Cancel a queued or running workflow. The script stays saved."""
    actor = (by or "").strip()
    if not actor:
        raise WorkflowError(
            "a workflow action needs an actor — who is asking?")
    record = _resolve_or_raise(store, ref)
    return store.cancel(record["id"], by=actor)


def resume_workflow(store: WorkflowStore, ref: str, *,
                    by: str) -> dict[str, Any]:
    """Re-run an ``interrupted`` / ``failed`` workflow's saved script."""
    actor = (by or "").strip()
    if not actor:
        raise WorkflowError(
            "a workflow action needs an actor — who is asking?")
    record = _resolve_or_raise(store, ref)
    return store.resume(record["id"], by=actor)
