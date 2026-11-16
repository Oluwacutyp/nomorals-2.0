"""The workflow script format + runtime primitives.

A workflow script is synchronous Python (a deliberate Devon
deviation from Antigravity's ``async`` form — the scheduler tick
is synchronous, and an event loop inside it would be a second
runtime). Scripts coordinate roster agents; they never touch the
filesystem, the network, or Devon's internals directly. That is
enforced two ways:

1. Static validation (``validate_script``) before every run:
   no imports, no ``open``/``exec``/``eval``/``__import__``/
   ``compile``, no dunder access, no ``while`` loops (a script
   runs on a background thread with no kill switch — finite work
   uses ``for`` / ``parallel`` / ``pipeline`` over a list).
2. Restricted execution globals: the script runs under ``exec``
   with only the primitives + a small safe-builtin set. Anything
   else (``os``, ``open``, ``eval``) is simply not defined and
   fails at runtime with a NameError, recorded as a failed run.

Primitives: ``phase(name)``, ``log(*args)``, ``agent(...)``
(one-shot, or a multi-turn handle when called with no prompt),
``parallel(items, fn)``, ``pipeline(items, fn)``,
``trajectory(limit, kind)``, ``args``, ``CONVERSATION_ID``,
``STEP_INDEX``, ``WORKSPACES``, ``print`` (collected into the
phase log, as in Antigravity).
"""

from __future__ import annotations

import ast
import concurrent.futures
import contextlib
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import WorkflowError
from .failures import record_failure
from .schema import validate_result
from .store import WorkflowStore

__all__ = ["ScriptRuntime", "WorkflowHandle", "validate_script"]

#: Names a script may call that are NOT the primitives. Anything
#: else is a NameError at runtime. Exception types are included so
#: a script can fail fast with a reason (``raise ValueError(...)``).
_SAFE_BUILTINS: dict[str, Any] = {
    "len": len, "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    "sorted": sorted, "min": min, "max": max, "range": range,
    "enumerate": enumerate, "zip": zip, "abs": abs, "round": round,
    "sum": sum, "any": any, "all": all, "isinstance": isinstance,
    "Exception": Exception, "RuntimeError": RuntimeError,
    "ValueError": ValueError, "TypeError": TypeError,
}

#: Calls that are never allowed in a script, however they are spelled.
_FORBIDDEN_CALLS = {
    "open", "exec", "eval", "__import__", "compile", "input",
    "globals", "locals", "vars", "getattr", "setattr", "delattr",
    "breakpoint", "exit", "quit",
}

_DEFAULT_SCHEMA: dict[str, Any] = {"type": "string"}


def validate_script(text: str) -> list[str]:
    """Static-check a workflow script. Empty list = valid.

    Never raises: a syntax error is reported as an error string,
    so the drafter can be told what to fix.
    """
    if not (text or "").strip():
        return ["script is empty"]
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        line = f"line {exc.lineno}" if exc.lineno else "line ?"
        return [f"syntax error on {line}: {exc.msg}"]
    errors: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(
                f"line {line}: import is not allowed in workflow "
                f"scripts — agents and tools come through the "
                f"primitives")
        elif isinstance(node, ast.While):
            errors.append(
                f"line {line}: while loops are not allowed — finite "
                f"work uses for / parallel / pipeline over a list")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) \
                    and func.id in _FORBIDDEN_CALLS:
                errors.append(
                    f"line {line}: {func.id}() is not allowed in "
                    f"workflow scripts")
        elif isinstance(node, ast.Attribute) \
                and node.attr.startswith("__"):
            errors.append(
                f"line {line}: dunder attribute {node.attr!r} is "
                f"not allowed in workflow scripts")
        elif isinstance(node, ast.Name) \
                and node.id.startswith("__"):
            errors.append(
                f"line {line}: dunder name {node.id!r} is not "
                f"allowed in workflow scripts")
    return errors


class _ScriptCancelled(Exception):
    """Internal: the run was cancelled between phases."""


class WorkflowHandle:
    """A multi-turn agent: one roster agent across several turns.

    ``handle = agent(role="reviewer")`` then ``handle(prompt)``
    runs the next turn of the SAME roster agent, carrying the
    earlier turns as context inside the next goal text — the
    Reviewer pattern (lists bugs, later verifies fixes). The
    turns are journaled on the run record through the runtime.
    """

    def __init__(self, runtime: ScriptRuntime, *, role: str,
                 workspace: str = "inherit", name: str = "") -> None:
        self._runtime = runtime
        self.role = role
        self.workspace = workspace
        self.name = name
        self.turns: list[dict[str, Any]] = []

    def __call__(self, prompt: str, schema: dict[str, Any] | None = None,
                 *, input: dict[str, Any] | None = None) -> Any:
        out = self._runtime._call_agent(
            prompt, schema, role=self.role, input=input,
            workspace=self.workspace, name=self.name,
            prior_turns=list(self.turns))
        self.turns.append({"prompt": prompt, "result": out})
        return out

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"WorkflowHandle(role={self.role!r}, "
                f"turns={len(self.turns)})")


class ScriptRuntime:
    """The primitives a workflow script runs with, bound to one run.

    One runtime per run. ``run(text)`` validates, executes, and
    finalizes the run record (``done``/``failed``) with a rendered
    summary. The runtime never raises out of ``run`` — a script
    error is a failed run with the reason recorded, plus a
    failure-ledger row.
    """

    def __init__(self, *, run: dict[str, Any], store: WorkflowStore,
                 manager: Any, conversation_log: Any = None,
                 repo: Path | str | None = None,
                 args: dict[str, Any] | None = None,
                 agent_timeout_s: float = 120.0) -> None:
        self.record = run
        self.store = store
        self.manager = manager
        self._conversation_log = conversation_log
        self._conversation_id = str(run.get("chat_key") or "")
        self._repo = Path(repo) if repo is not None else Path.cwd()
        self._args = dict(args if args is not None
                          else run.get("args") or {})
        self._agent_timeout_s = float(agent_timeout_s)
        self._phase_open: str | None = None
        self._cancel_requested = False
        self._worktree_seq = 0
        self._lock = threading.Lock()

    # ── cancellation ───────────────────────────────────────────────

    def request_cancel(self) -> None:
        """Flag the run cancelled; the script stops at the next phase."""
        self._cancel_requested = True

    def _check_cancelled(self) -> None:
        if self._cancel_requested:
            raise _ScriptCancelled("workflow run was cancelled")

    # ── primitives ─────────────────────────────────────────────────

    def phase(self, name: str) -> None:
        """Start a named phase; the previous open phase is closed."""
        self._check_cancelled()
        name = " ".join(str(name or "").split()).strip()
        if not name:
            raise WorkflowError("phase() needs a name")
        if self._phase_open is not None:
            with contextlib.suppress(WorkflowError):
                self.store.finish_phase(self.record["id"], self._phase_open)
        self.store.start_phase(self.record["id"], name)
        self._phase_open = name

    def log(self, *parts: Any) -> None:
        """One line in the current phase (opens a ``main`` phase if needed)."""
        text = " ".join(str(p) for p in parts)
        if self._phase_open is None:
            self.phase("main")
        self.store.append_log(self.record["id"], text)

    def _print(self, *parts: Any, **_kw: Any) -> None:
        """``print`` inside a script is collected into the phase log."""
        self.log(*parts)

    def agent(self, prompt: str | None = None,
              schema: dict[str, Any] | None = None, *,
              role: str = "planner",
              input: dict[str, Any] | None = None,
              workspace: str = "inherit", name: str = "") -> Any:
        """Run one roster agent, or return a multi-turn handle.

        ``agent(prompt, schema, role=...)`` spawns the roster role
        with ``goal=prompt``, waits bounded, collects, and returns
        the schema-validated result. A failure (unknown role,
        missing required input, timeout, worker error, schema
        failure) is returned as ``{"ok": False, "error": ...}`` —
        the script decides what to do with it; ``parallel``
        isolates it per item. ``agent(role=...)`` with no prompt
        returns a :class:`WorkflowHandle` for multi-turn work.
        """
        if prompt is None:
            return WorkflowHandle(self, role=role, workspace=workspace,
                                  name=name)
        return self._call_agent(prompt, schema, role=role, input=input,
                                workspace=workspace, name=name)

    def _call_agent(self, prompt: str,
                    schema: dict[str, Any] | None, *,
                    role: str, input: dict[str, Any] | None,
                    workspace: str, name: str,
                    prior_turns: list[dict[str, Any]] | None = None) -> Any:
        schema = dict(schema) if schema is not None else dict(
            _DEFAULT_SCHEMA)
        if self._phase_open is None:
            self.phase("main")
        goal = " ".join(str(prompt or "").split()).strip()
        if prior_turns:
            context_lines = ["Earlier turns in this conversation:"]
            for i, turn in enumerate(prior_turns, 1):
                result = turn.get("result")
                result_text = str(result)
                if len(result_text) > 500:
                    result_text = result_text[:500] + "…"
                context_lines.append(
                    f"{i}. {turn.get('prompt')!r} → {result_text}")
            goal = "\n".join(context_lines) + "\n\nNext: " + goal
        worktree_path = ""
        if workspace not in ("inherit", "branch", "share"):
            return self._agent_failed(
                role, f"unknown workspace {workspace!r} — want "
                f"'inherit', 'branch' or 'share'")
        if workspace in ("branch", "share"):
            worktree_path = self._prepare_worktree(workspace, name or role)
            if isinstance(worktree_path, dict):
                return worktree_path  # an error dict, already recorded
            goal += (f"\n\nWork in this isolated worktree: "
                     f"{worktree_path}")
        merged = dict(input or {})
        try:
            spawned = self.manager.spawn(
                role=role, goal=goal, input=merged,
                mission_id=str(self.record["id"]),
                chat_key=str(self.record.get("chat_key") or ""),
                name=name or f"workflow-{role}")
            run_id = getattr(spawned, "id", None) or str(spawned)
            finished = self.manager.wait(run_id,
                                         timeout=self._agent_timeout_s)
        except Exception as exc:  # noqa: BLE001 - failures are records
            return self._agent_failed(role, f"{type(exc).__name__}: {exc}")
        finished_run = self.manager.get_run(run_id) \
            if hasattr(self.manager, "get_run") else finished
        if finished_run is None:
            return self._agent_failed(role, f"no delegation run {run_id!r}")
        status = getattr(finished_run, "status", "")
        if status != "completed":
            error = getattr(finished_run, "error", "") \
                or f"worker ended {status or 'unknown'}"
            return self._agent_failed(role, error)
        outcome = finished_run.result_dict() \
            if hasattr(finished_run, "result_dict") else {}
        result = outcome.get("result")
        try:
            validated = validate_result(result, schema,
                                        where=f"agent:{role}")
        except WorkflowError as exc:
            return self._agent_failed(role, str(exc))
        self.store.record_agent_result(self.record["id"], role, validated)
        return validated

    def _agent_failed(self, role: str, error: str) -> dict[str, Any]:
        """One agent failure: ledger row + journal entry + error dict."""
        record_failure(f"workflow:{self.record['id']}:{role}",
                       WorkflowError(error))
        with contextlib.suppress(WorkflowError):
            self.store.record_agent_result(
                self.record["id"], role, {"ok": False, "error": error})
        return {"ok": False, "error": error}

    def _prepare_worktree(self, workspace: str,
                          handle: str) -> str | dict[str, Any]:
        """Create an isolated worktree (or copy) for one agent call."""
        from .worktree import create_worktree

        with self._lock:
            self._worktree_seq += 1
            seq = self._worktree_seq
        dest = (self._repo / ".workflow-worktrees" / self.record["id"]
                / f"{handle}-{seq}")
        try:
            base = subprocess.run(
                ["git", "-C", str(self._repo), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False)
            if base.returncode != 0:
                raise WorkflowError(
                    f"cannot start a {workspace} workspace: "
                    f"{self._repo} is not a git checkout")
            base_revision = base.stdout.strip()
            create_worktree(self._repo, dest, base_revision)
        except WorkflowError as exc:
            return self._agent_failed("worktree", str(exc))
        self.store.record_worktree(self.record["id"], handle, str(dest))
        return str(dest)

    def parallel(self, items: Any,
                 fn: Callable[[Any], Any]) -> list[Any]:
        """Run ``fn(item)`` for every item at the same time.

        Results in item order. One item failing does NOT stop the
        others — the script receives ``{"ok": False, "error": ...}``
        for that item, and a failure-ledger row is written.
        """
        items = list(items)
        if not items:
            return []
        results: list[Any] = [None] * len(items)

        def _one(index: int, item: Any) -> Any:
            try:
                return fn(item)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                record_failure(
                    f"workflow:{self.record['id']}:parallel:{item!r}", exc)
                return {"ok": False,
                        "error": f"{type(exc).__name__}: {exc}"}

        workers = min(len(items), 8)
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers) as pool:
            futures = {pool.submit(_one, i, item): i
                       for i, item in enumerate(items)}
            for future in concurrent.futures.as_completed(futures):
                results[futures[future]] = future.result()
        return results

    def pipeline(self, items: Any,
                 fn: Callable[[Any], Any]) -> list[Any]:
        """Run ``fn(item)`` one item at a time, results in item order.

        A failing item yields an error dict for that item (same
        isolation as ``parallel``) and the pipeline moves on.
        """
        results: list[Any] = []
        for item in items:
            try:
                results.append(fn(item))
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                record_failure(
                    f"workflow:{self.record['id']}:pipeline:{item!r}", exc)
                results.append({"ok": False,
                                "error": f"{type(exc).__name__}: {exc}"})
        return results

    def trajectory(self, limit: int = 10,
                   kind: str | None = None) -> list[dict[str, Any]]:
        """Earlier steps of the current conversation (oldest first).

        Reads the durable conversation ledger through the context
        db; without one, or without a conversation id, returns [].
        ``kind`` filters by message role when given.
        """
        if self._conversation_log is None or not self._conversation_id:
            return []
        try:
            rows = self._conversation_log.history(
                self._conversation_id, limit=max(1, int(limit)))
        except Exception:  # noqa: BLE001 - trajectory is advisory
            return []
        rows = list(rows or [])[::-1]
        if kind is not None:
            rows = [r for r in rows if r.get("role") == kind]
        return rows

    # ── execution ──────────────────────────────────────────────────

    def _globals(self) -> dict[str, Any]:
        workspaces = [str(self._repo)]
        try:
            record = self.store.get_workflow(self.record["id"])
            if record is not None:
                workspaces.extend(
                    str(p) for p in (record.get("worktrees") or {}).values())
        except WorkflowError:
            pass
        env: dict[str, Any] = dict(_SAFE_BUILTINS)
        env.update({
            "__builtins__": dict(_SAFE_BUILTINS),
            "phase": self.phase,
            "log": self.log,
            "print": self._print,
            "agent": self.agent,
            "parallel": self.parallel,
            "pipeline": self.pipeline,
            "trajectory": self.trajectory,
            "args": dict(self._args),
            "CONVERSATION_ID": self._conversation_id,
            "STEP_INDEX": str(self.record["id"]),
            "WORKSPACES": workspaces,
        })
        return env

    def run(self, text: str) -> dict[str, Any]:
        """Validate, execute, and finalize one script.

        Returns a summary dict (``state`` ``done``/``failed``).
        Never raises: a script error is a failed run with the
        reason recorded, plus a failure-ledger row.
        """
        # The store only fails/completes a RUNNING record; the
        # runner usually marks it first, but run() must hold on
        # its own when called directly (a second mark is refused
        # and suppressed — the record is already running).
        with contextlib.suppress(WorkflowError):
            self.store.mark_running(self.record["id"])
        errors = validate_script(text)
        if errors:
            reason = "workflow script is invalid: " + "; ".join(errors)
            record_failure(f"workflow:{self.record['id']}:validate",
                           WorkflowError(reason))
            with contextlib.suppress(WorkflowError):
                self.store.mark_failed(self.record["id"], error=reason)
            return {"state": "failed", "error": reason}
        try:
            exec(text, self._globals())  # noqa: S102 - scripts are sandboxed by validate_script + restricted globals
        except _ScriptCancelled as exc:
            reason = str(exc)
            with contextlib.suppress(WorkflowError):
                self.store.mark_failed(self.record["id"], error=reason)
            return {"state": "failed", "error": reason}
        except Exception as exc:  # noqa: BLE001 - script errors are records
            reason = f"{type(exc).__name__}: {exc}"
            record_failure(f"workflow:{self.record['id']}:script", exc)
            with contextlib.suppress(WorkflowError):
                self.store.mark_failed(self.record["id"], error=reason)
            return {"state": "failed", "error": reason}
        if self._phase_open is not None:
            with contextlib.suppress(WorkflowError):
                self.store.finish_phase(self.record["id"], self._phase_open)
            self._phase_open = None
        summary = self._render_summary()
        with contextlib.suppress(WorkflowError):
            self.store.mark_done(self.record["id"], summary=summary)
        return {"state": "done", "summary": summary}

    def _render_summary(self) -> str:
        """One readable summary from the phase journal."""
        try:
            record = self.store.get_workflow(self.record["id"])
        except WorkflowError:
            return f"workflow {self.record.get('name', '')} finished."
        lines = [f"workflow {record.get('name', '')} — done."]
        for phase in record.get("phases") or []:
            lines.append(f"— {phase.get('name', '')}:")
            for line in phase.get("log") or []:
                lines.append(f"  {line}")
            for call in phase.get("agents") or []:
                result = call.get("result")
                result_text = str(result)
                if len(result_text) > 300:
                    result_text = result_text[:300] + "…"
                lines.append(f"  [{call.get('role', '')}] {result_text}")
        if record.get("error"):
            lines.append(f"error: {record['error']}")
        return "\n".join(lines)

