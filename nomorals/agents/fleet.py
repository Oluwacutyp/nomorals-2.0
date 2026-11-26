"""Fleet: parallel coding-agent runs in isolated git worktrees.

The composed surface the 305-agent survey found missing (family F1):
Agent 007, agent-manager, AgentBox, Agent Teams, AGX, Dorothy,
EliaAgent, LoopTroop, Maestro, Mysti, Opus Manager, Orkas, Tale,
Okto-Nexus, TeamHero, Agents Squads, Hive, Kodo, Okto-Pulse, ORCH,
Stoneforge, WritBase — all of them are one capability: run several
coding agents at once, each in its own isolated checkout, watch them
on a board, review the diff, approve the merge. Devon had every part
and no surface: worktree isolation (`agents/workflow/worktree.py`,
over `agents/bench/staging.py`), async managed runs
(`agents/delegation_manager.py`), the coding agent itself
(`agents/coding.py`, `CodingAgent` with an explicit `root=`), and
git (`tools/git.py`). The fleet composes them; it re-implements none
of them.

Lifecycle of one task: ``queued`` (worktree created) → ``running``
(worker executing in the worktree) → ``review`` (worker finished
with a real diff) or ``failed`` (worker error, worker ok-but-empty,
worktree creation failure) → ``merged`` (owner approved; the
worktree diff applied to the repo as a patch). Approval is the only
way changes reach the repo: the fleet never writes to the repo
working tree except through ``git apply`` of a reviewed diff, and a
patch that does not apply cleanly leaves the task in ``review``
with the conflict named — never a forced merge, never a silent
skip.

Workers are injectable: ``dispatch(..., worker=fn)`` takes any
``fn(root: Path, brief: str) -> dict``. The default worker is the
real ``CodingAgent`` (needs a configured model, exactly like
``nm code``). Tests drive the whole loop offline with a fake worker
that writes files; nothing in the organ itself needs a model.

Durability: one JSON record per fleet and per task under the fleet
root (``<root>/<fleet-id>/fleet.json``,
``<root>/<fleet-id>/tasks/<task-id>.json``) — the same sidecar
shape as the bench staging manager, so a restart (or a second
process over the same root) sees the same board. A background
thread is only a handle: a task found ``running`` with no live
thread (restart) is reported ``interrupted`` by :meth:`recover`
and can be re-dispatched.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_id
from .workflow.worktree import create_worktree, remove_worktree

__all__ = [
    "FleetError",
    "FleetStore",
    "FleetRecord",
    "FleetTaskRecord",
    "FleetBoard",
    "default_worker",
    "render_board",
    "STATES",
    "STATE_QUEUED",
    "STATE_RUNNING",
    "STATE_REVIEW",
    "STATE_MERGED",
    "STATE_FAILED",
]

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_REVIEW = "review"
STATE_MERGED = "merged"
STATE_FAILED = "failed"
STATES = (STATE_QUEUED, STATE_RUNNING, STATE_REVIEW, STATE_MERGED,
          STATE_FAILED)

#: A task whose worker finished but produced no diff is a failure,
#: not a success — an empty run must never reach review.
_NO_CHANGES = "worker finished but changed nothing in the worktree"


class FleetError(Exception):
    """A fleet operation refused or failed (named, never silent)."""


# -- git helpers (subprocess, mirroring tools/git.py's runner) --------

def _git(args: list[str], cwd: str | Path, *,
         input_text: str | None = None,
         timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        input=input_text, capture_output=True, text=True,
        timeout=timeout, check=False)


def _git_ok(args: list[str], cwd: str | Path, *,
            input_text: str | None = None) -> str:
    proc = _git(args, cwd, input_text=input_text)
    if proc.returncode != 0:
        raise FleetError(
            f"git {' '.join(args)} in {cwd} failed: "
            f"{(proc.stderr or proc.stdout or '').strip()}")
    return proc.stdout


def _resolve_revision(repo: str | Path, revision: str) -> str:
    return _git_ok(["rev-parse", "--verify", revision], repo).strip()


def _atomic_write(path: Path, text: str) -> None:
    """Write a record atomically (temp file + rename) so a concurrent
    reader never sees a torn JSON file — the fleet board is read
    from these records while worker threads write them."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


# -- records ------------------------------------------------------------

@dataclass
class FleetRecord:
    fleet_id: str
    name: str
    repo: str
    base_revision: str
    created_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"fleet_id": self.fleet_id, "name": self.name,
                "repo": self.repo, "base_revision": self.base_revision,
                "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FleetRecord:
        return cls(fleet_id=str(data["fleet_id"]),
                   name=str(data.get("name") or data["fleet_id"]),
                   repo=str(data["repo"]),
                   base_revision=str(data["base_revision"]),
                   created_at=float(data.get("created_at") or 0.0))


@dataclass
class FleetTaskRecord:
    task_id: str
    fleet_id: str
    key: str
    brief: str
    state: str = STATE_QUEUED
    worktree: str = ""
    mechanism: str = ""
    base_revision: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "fleet_id": self.fleet_id,
                "key": self.key, "brief": self.brief,
                "state": self.state, "worktree": self.worktree,
                "mechanism": self.mechanism,
                "base_revision": self.base_revision,
                "result": self.result, "error": self.error,
                "created_at": self.created_at,
                "updated_at": self.updated_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FleetTaskRecord:
        return cls(
            task_id=str(data["task_id"]),
            fleet_id=str(data["fleet_id"]),
            key=str(data.get("key") or data["task_id"]),
            brief=str(data.get("brief") or ""),
            state=str(data.get("state") or STATE_QUEUED),
            worktree=str(data.get("worktree") or ""),
            mechanism=str(data.get("mechanism") or ""),
            base_revision=str(data.get("base_revision") or ""),
            result=dict(data.get("result") or {}),
            error=str(data.get("error") or ""),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0))


@dataclass
class FleetBoard:
    """One fleet's Kanban board: tasks grouped by state."""

    fleet: dict[str, Any]
    columns: dict[str, list[dict[str, Any]]]

    def to_dict(self) -> dict[str, Any]:
        return {"fleet": self.fleet, "columns": self.columns}


def render_board(board: FleetBoard) -> str:
    """Kanban text for a board: one column per state, one row per task."""
    fleet = board.fleet
    lines = [f"fleet {fleet.get('name')} ({fleet.get('fleet_id')})",
             f"repo {fleet.get('repo')} @ {fleet.get('base_revision', '')[:12]}"]
    for state in STATES:
        tasks = board.columns.get(state, [])
        lines.append(f"\n[{state}] ({len(tasks)})")
        for task in tasks:
            lines.append(f"  {task['task_id'][:8]}  {task['key']} — "
                         f"{(task.get('brief') or '')[:72]}")
            if task.get("error"):
                lines.append(f"      ! {(task['error'])[:120]}")
    return "\n".join(lines)


# -- default worker: the real coding agent ------------------------------

def default_worker(context: Any, root: Path, brief: str,
                   *, max_iterations: int = 5) -> dict[str, Any]:
    """Run Devon's own coding agent against ``root``.

    This is the same agent ``nm code`` runs (`agents/coding.py`,
    `CodingAgent` with an explicit `root=`), so a fleet task is a
    real coding run, not a scripted edit. Needs a configured model
    — a missing model fails through the agent's own error, reported
    on the task, never as fake success.
    """
    from .coding import CodingAgent

    agent = CodingAgent(context, root=str(root))
    result = agent.run(brief, max_iterations=max_iterations)
    return {"ok": bool(result.ok), "files": list(result.files),
            "output": str(result.output or ""),
            "error": str(result.error or ""),
            "verification": str(result.verification or ""),
            "iterations": int(result.iterations)}


# -- the store ------------------------------------------------------------

class FleetStore:
    """Owns fleets and their tasks under one root directory.

    ``root`` is the fleet home (``<home>/fleet`` in production).
    ``context`` is the agent context the default worker needs; it
    may be None when every dispatch passes an explicit ``worker``.
    """

    def __init__(self, root: str | Path, *,
                 context: Any = None) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.context = context
        self._lock = threading.RLock()
        self._live: dict[str, threading.Thread] = {}

    # -- paths ----------------------------------------------------------

    def _fleet_dir(self, fleet_id: str) -> Path:
        return self.root / str(fleet_id)

    def _fleet_file(self, fleet_id: str) -> Path:
        return self._fleet_dir(fleet_id) / "fleet.json"

    def _task_file(self, fleet_id: str, task_id: str) -> Path:
        return self._fleet_dir(fleet_id) / "tasks" / f"{task_id}.json"

    def _worktree_dir(self, fleet_id: str, task_id: str) -> Path:
        return self._fleet_dir(fleet_id) / task_id / "wt"

    # -- records ----------------------------------------------------------

    def _save_fleet(self, fleet: FleetRecord) -> None:
        path = self._fleet_file(fleet.fleet_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(fleet.to_dict(), indent=1))

    def _load_fleet(self, fleet_id: str) -> FleetRecord | None:
        try:
            data = json.loads(
                self._fleet_file(fleet_id).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return FleetRecord.from_dict(data)

    def _save_task(self, task: FleetTaskRecord) -> None:
        task.updated_at = time.time()
        path = self._task_file(task.fleet_id, task.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(task.to_dict(), indent=1))

    def _load_task(self, fleet_id: str,
                   task_id: str) -> FleetTaskRecord | None:
        try:
            data = json.loads(
                self._task_file(fleet_id, task_id).read_text(
                    encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        return FleetTaskRecord.from_dict(data)

    # -- fleet lookup -------------------------------------------------------

    def resolve_fleet(self, ref: str) -> FleetRecord:
        """A fleet by id, id prefix, or name. Ambiguous refs refuse."""
        ref = (ref or "").strip()
        if not ref:
            raise FleetError("a fleet id or name is required")
        exact = self._load_fleet(ref)
        if exact is not None:
            return exact
        found: list[FleetRecord] = []
        for path in sorted(self.root.glob("*/fleet.json")):
            fleet = self._load_fleet(path.parent.name)
            if fleet is None:
                continue
            if fleet.fleet_id.startswith(ref) or fleet.name == ref:
                found.append(fleet)
        if len(found) == 1:
            return found[0]
        if not found:
            raise FleetError(f"no fleet {ref!r}")
        raise FleetError(
            f"fleet {ref!r} is ambiguous: {len(found)} fleets match — "
            "use the full id")

    def list_fleets(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.root.glob("*/fleet.json")):
            fleet = self._load_fleet(path.parent.name)
            if fleet is not None:
                record = fleet.to_dict()
                record["tasks"] = len(self.list_tasks(fleet.fleet_id))
                out.append(record)
        return out

    def create_fleet(self, name: str, repo: str | Path, *,
                     base: str | None = None) -> dict[str, Any]:
        """Create a fleet over a git repo. The base revision is the
        fleet's shared starting point; every task worktrees from it."""
        repo_path = Path(repo).expanduser().resolve()
        if not repo_path.is_dir():
            raise FleetError(f"not a directory: {repo}")
        _git_ok(["rev-parse", "--git-dir"], repo_path)
        base_revision = _resolve_revision(
            repo_path, base or "HEAD")
        fleet = FleetRecord(
            fleet_id=new_id("fleet"), name=(name or "").strip() or "fleet",
            repo=str(repo_path), base_revision=base_revision,
            created_at=time.time())
        with self._lock:
            self._save_fleet(fleet)
        return fleet.to_dict()

    # -- tasks -----------------------------------------------------------

    def list_tasks(self, fleet_id: str) -> list[dict[str, Any]]:
        out = []
        task_dir = self._fleet_dir(fleet_id) / "tasks"
        if not task_dir.is_dir():
            return out
        for path in sorted(task_dir.glob("*.json")):
            task = self._load_task(fleet_id, path.stem)
            if task is not None:
                out.append(task.to_dict())
        out.sort(key=lambda t: t["created_at"])
        return out

    def resolve_task(self, ref: str) -> FleetTaskRecord:
        """A task by id or id prefix, across fleets. Ambiguous refs refuse."""
        ref = (ref or "").strip()
        if not ref:
            raise FleetError("a task id is required")
        found: list[FleetTaskRecord] = []
        for fleet_file in sorted(self.root.glob("*/fleet.json")):
            fleet_id = fleet_file.parent.name
            task_dir = fleet_file.parent / "tasks"
            if not task_dir.is_dir():
                continue
            for path in sorted(task_dir.glob("*.json")):
                if path.stem == ref or path.stem.startswith(ref):
                    task = self._load_task(fleet_id, path.stem)
                    if task is not None:
                        found.append(task)
        if len(found) == 1:
            return found[0]
        if not found:
            raise FleetError(f"no task {ref!r}")
        raise FleetError(
            f"task {ref!r} is ambiguous: {len(found)} tasks match — "
            "use the full id")

    def dispatch(self, fleet_ref: str, brief: str, *,
                 key: str | None = None,
                 worker: Callable[[Path, str], dict[str, Any]] | None = None,
                 run_sync: bool = False) -> dict[str, Any]:
        """Dispatch a coding brief as a fleet task.

        Creates the isolated worktree first; the worker runs inside
        it and nowhere else. ``run_sync`` runs the worker in the
        calling thread (CLI `--wait` / tests); otherwise the worker
        runs on a daemon thread and dispatch returns at once.
        """
        brief = (brief or "").strip()
        if not brief:
            raise FleetError("a task brief is required")
        fleet = self.resolve_fleet(fleet_ref)
        if worker is None:
            if self.context is None:
                raise FleetError(
                    "no worker and no agent context: pass a worker "
                    "or build the store with a context")
            def _default(root: Path, text: str) -> dict[str, Any]:
                return default_worker(self.context, root, text)

            worker = _default
        task = FleetTaskRecord(
            task_id=new_id("task"), fleet_id=fleet.fleet_id,
            key=(key or "").strip() or brief[:48],
            brief=brief, state=STATE_QUEUED,
            base_revision=fleet.base_revision,
            created_at=time.time())
        worktree = self._worktree_dir(fleet.fleet_id, task.task_id)
        with self._lock:
            self._save_task(task)
        try:
            task.mechanism = create_worktree(
                fleet.repo, worktree, fleet.base_revision)
        except Exception as exc:  # noqa: BLE001 - worktree failure is a task failure
            task.state = STATE_FAILED
            task.error = f"worktree creation failed: {exc}"
            with self._lock:
                self._save_task(task)
            return task.to_dict()
        task.worktree = str(worktree)
        with self._lock:
            self._save_task(task)
        if run_sync:
            self._run_task(task.task_id, worker)
        else:
            thread = threading.Thread(
                target=self._run_task,
                args=(task.task_id, worker),
                name=f"fleet-{task.task_id[:8]}", daemon=True)
            with self._lock:
                self._live[task.task_id] = thread
            thread.start()
        return self._load_task(fleet.fleet_id,
                               task.task_id).to_dict()  # type: ignore[union-attr]

    def _run_task(self, task_id: str,
                  worker: Callable[[Path, str], dict[str, Any]]) -> None:
        fleet_id = self._fleet_of_task(task_id)
        if fleet_id is None:
            return
        try:
            with self._lock:
                task = self._load_task(fleet_id, task_id)
                if task is None or task.state != STATE_QUEUED:
                    return
                task.state = STATE_RUNNING
                self._save_task(task)
            try:
                result = worker(Path(task.worktree), task.brief)
            except Exception as exc:  # noqa: BLE001 - worker failure is a task failure
                with self._lock:
                    task = self._load_task(fleet_id, task_id)
                    if task is None:
                        return
                    task.state = STATE_FAILED
                    task.error = f"worker failed: {exc}"
                    self._save_task(task)
                return
            if not isinstance(result, dict):
                result = {"ok": False, "output": str(result),
                          "error": "worker returned a non-dict result"}
            with self._lock:
                task = self._load_task(fleet_id, task_id)
                if task is None:
                    return
                task.result = result
                if not result.get("ok"):
                    task.state = STATE_FAILED
                    task.error = str(result.get("error") or
                                     "worker reported failure")
                else:
                    try:
                        diff = self._diff(task)
                    except FleetError as exc:
                        task.state = STATE_FAILED
                        task.error = ("could not read the worktree "
                                      f"diff: {exc}")
                    else:
                        if not diff.strip():
                            task.state = STATE_FAILED
                            task.error = _NO_CHANGES
                        else:
                            task.state = STATE_REVIEW
                self._save_task(task)
        finally:
            with self._lock:
                self._live.pop(task_id, None)

    def _fleet_of_task(self, task_id: str) -> str | None:
        for fleet_file in sorted(self.root.glob("*/fleet.json")):
            if (fleet_file.parent / "tasks" /
                    f"{task_id}.json").is_file():
                return fleet_file.parent.name
        return None

    def wait(self, task_ref: str, *, timeout: float = 300.0) -> dict[str, Any]:
        """Block until a task leaves queued/running, or time out.

        A timeout returns the current record with ``timed_out`` set —
        it never claims a state the task has not reached.
        """
        deadline = time.monotonic() + max(float(timeout), 0.0)
        task = self.resolve_task(task_ref)
        while task.state in (STATE_QUEUED, STATE_RUNNING):
            if time.monotonic() >= deadline:
                record = task.to_dict()
                record["timed_out"] = True
                return record
            time.sleep(0.2)
            task = self.resolve_task(task_ref)
        record = task.to_dict()
        record["timed_out"] = False
        return record

    def recover(self) -> list[dict[str, Any]]:
        """Mark orphaned running tasks (restart: no live thread)
        ``failed``/``interrupted`` so the board never shows a
        running task that is not running. Returns what changed."""
        changed = []
        for fleet_file in sorted(self.root.glob("*/fleet.json")):
            fleet_id = fleet_file.parent.name
            for task_dict in self.list_tasks(fleet_id):
                if task_dict["state"] != STATE_RUNNING:
                    continue
                if task_dict["task_id"] in self._live:
                    continue
                task = self._load_task(fleet_id, task_dict["task_id"])
                if task is None:
                    continue
                task.state = STATE_FAILED
                task.error = ("interrupted: the fleet restarted while "
                              "this task was running; re-dispatch the brief")
                with self._lock:
                    self._save_task(task)
                changed.append(task.to_dict())
        return changed

    # -- board / review / approve ----------------------------------------

    def board(self, fleet_ref: str) -> FleetBoard:
        fleet = self.resolve_fleet(fleet_ref)
        columns: dict[str, list[dict[str, Any]]] = {
            state: [] for state in STATES}
        for task in self.list_tasks(fleet.fleet_id):
            columns.setdefault(task["state"], []).append(task)
        return FleetBoard(fleet=fleet.to_dict(), columns=columns)

    def _diff(self, task: FleetTaskRecord) -> str:
        """The worktree diff vs the fleet base, staged so new files
        count. Stages everything in the fleet-owned worktree
        (`git add -A`); a staging failure raises FleetError."""
        worktree = Path(task.worktree)
        if not worktree.is_dir():
            raise FleetError(f"worktree is gone: {task.worktree}")
        proc = _git(["add", "-A"], worktree)
        if proc.returncode != 0:
            raise FleetError(
                f"could not stage the worktree diff: "
                f"{(proc.stderr or proc.stdout or '').strip()}")
        return _git_ok(["diff", "--cached", task.base_revision],
                       worktree)

    def _diff_stat(self, task: FleetTaskRecord) -> dict[str, Any]:
        worktree = Path(task.worktree)
        out = _git_ok(["diff", "--cached", "--stat",
                       task.base_revision], worktree)
        files: list[str] = []
        proc = _git(["diff", "--cached", "--name-only",
                     task.base_revision], worktree)
        if proc.returncode == 0:
            files = [line for line in proc.stdout.splitlines()
                     if line.strip()]
        return {"stat": out, "files": files}

    def review(self, task_ref: str) -> dict[str, Any]:
        """The task's diff for review. Any state except merged
        shows its current diff; a merged task's worktree is gone,
        so review reports that honestly."""
        task = self.resolve_task(task_ref)
        if task.state == STATE_MERGED:
            return {"task": task.to_dict(), "diff": "",
                    "files": [], "stat": "",
                    "note": "merged — the worktree was removed on merge"}
        if not task.worktree:
            raise FleetError(f"task {task.task_id} has no worktree")
        diff = self._diff(task)
        stat = self._diff_stat(task)
        return {"task": task.to_dict(), "diff": diff,
                "files": stat["files"], "stat": stat["stat"]}

    def approve(self, task_ref: str) -> dict[str, Any]:
        """Merge a reviewed task into the repo.

        Only ``review`` tasks merge. The worktree diff is applied
        to the repo as a patch (`git apply --check` first); a patch
        that does not apply cleanly refuses with the conflict named
        and the task stays in ``review``. On success the worktree is
        removed and the task is ``merged`` with the applied files
        recorded.
        """
        task = self.resolve_task(task_ref)
        if task.state != STATE_REVIEW:
            raise FleetError(
                f"task {task.task_id} is {task.state}, not review — "
                "only a reviewed task can be approved")
        fleet = self.resolve_fleet(task.fleet_id)
        diff = self._diff(task)
        if not diff.strip():
            raise FleetError(
                f"task {task.task_id} has no diff to merge")
        proc = _git(["apply", "--check", "-"], fleet.repo,
                    input_text=diff)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            task.error = ("patch does not apply cleanly on the repo "
                          f"(the repo moved since {fleet.base_revision[:12]}): "
                          f"{detail[:400]}")
            with self._lock:
                self._save_task(task)
            raise FleetError(task.error)
        _git_ok(["apply", "-"], fleet.repo, input_text=diff)
        stat = self._diff_stat(task)
        remove_worktree(fleet.repo, task.worktree)
        task.state = STATE_MERGED
        task.error = ""
        task.result = {**task.result, "merged_files": stat["files"]}
        task.worktree = ""
        with self._lock:
            self._save_task(task)
        return task.to_dict()

    def cleanup(self, fleet_ref: str) -> dict[str, Any]:
        """Remove worktrees of finished tasks (merged/failed).
        Running/review tasks keep their worktrees. Returns counts."""
        fleet = self.resolve_fleet(fleet_ref)
        removed = 0
        kept = 0
        for task_dict in self.list_tasks(fleet.fleet_id):
            task = self._load_task(fleet.fleet_id,
                                   task_dict["task_id"])
            if task is None or not task.worktree:
                continue
            if task.state in (STATE_MERGED, STATE_FAILED):
                if remove_worktree(fleet.repo, task.worktree):
                    removed += 1
                task.worktree = ""
                with self._lock:
                    self._save_task(task)
            else:
                kept += 1
        return {"fleet_id": fleet.fleet_id, "removed": removed,
                "kept": kept}
