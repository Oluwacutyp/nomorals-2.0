"""Durable workflow store.

One atomic JSON index (``<home>/data/workflows.json``) plus one
readable ``.py`` file per saved script under ``<home>/workflows/``.
Shape follows ``location/fences.py`` (FenceStore): tmp+replace
writes; a corrupt store is quarantined and degrades to empty with a
named reason; bad rows are skipped; reads never raise.

States (owner correction 2026-11-16, "Not necessary I review" —
there is NO review-before-run gate, so there are no
``awaiting_approval`` / ``approved`` / ``denied`` states)::

    queued → running → done | failed | cancelled
                          ↘ interrupted → (resume/requeue) → queued

A saved script is ``queued`` the moment it is written; the runner
tick picks it up. The index is the source of truth; the ``.py``
file is the readable copy.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import WorkflowError

__all__ = [
    "WorkflowStore",
    "default_workflows_dir",
    "default_workflows_path",
    "new_workflow_id",
]

#: The states a workflow record can be in. No approval states exist
#: by owner order (2026-11-16): a saved script runs, it is not gated.
STATES = ("queued", "running", "done", "failed", "cancelled",
          "interrupted")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def new_workflow_id() -> str:
    """A fresh workflow id: ``wf-`` + 12 hex chars."""
    return f"wf-{uuid.uuid4().hex[:12]}"


def default_workflows_path(home: Path | str | None = None) -> Path:
    """The durable workflows index under a home (``<home>/data/workflows.json``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "data" / "workflows.json"


def default_workflows_dir(home: Path | str | None = None) -> Path:
    """The directory holding the readable saved scripts (``<home>/workflows``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "workflows"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _blank_record(workflow_id: str, name: str, task: str,
                  script: str, args: dict[str, Any],
                  chat_key: str) -> dict[str, Any]:
    now = time.time()
    return {
        "id": workflow_id,
        "name": name,
        "task": task,
        "script": script,
        "script_path": "",
        "state": "queued",
        "args": dict(args),
        "chat_key": chat_key,
        "phases": [],
        "runs": [],
        "worktrees": {},
        "summary": "",
        "error": "",
        "created_at": now,
        "updated_at": now,
        "finished_at": 0.0,
        "summary_delivered": False,
    }


class WorkflowStore:
    """Saved workflow scripts + run records, durable JSON, one file.

    ``path=None`` means in-memory only (tests, and runtimes with no
    home); the record shape is identical either way.
    """

    def __init__(self, path: Path | str | None = None,
                 scripts_dir: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if scripts_dir is not None:
            self.scripts_dir: Path | None = Path(scripts_dir)
        elif self.path is not None:
            self.scripts_dir = self.path.parent.parent / "workflows"
        else:
            self.scripts_dir = None
        self._lock = threading.RLock()
        self._workflows: dict[str, dict[str, Any]] = {}
        self._degraded = ""
        if self.path is not None:
            self._load()

    # ── load / save ────────────────────────────────────────────────

    @property
    def degraded_reason(self) -> str:
        """Why this store degraded, or "" when it is healthy."""
        return self._degraded

    def _load(self) -> None:
        assert self.path is not None
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            self._quarantine(exc)
            return
        if not isinstance(raw, dict):
            self._quarantine(
                ValueError(
                    f"workflows store must hold an object, found "
                    f"{type(raw).__name__}"))
            return
        for row in raw.get("workflows", []) or []:
            if not isinstance(row, dict):
                continue
            if not isinstance(row.get("id"), str) \
                    or not isinstance(row.get("name"), str):
                continue
            record = _blank_record(
                row["id"], row["name"], str(row.get("task", "")),
                str(row.get("script", "")),
                row.get("args") if isinstance(row.get("args"), dict)
                else {}, str(row.get("chat_key", "")))
            for key in ("script_path", "state", "summary", "error",
                        "created_at", "updated_at", "finished_at",
                        "summary_delivered"):
                if key in row:
                    record[key] = row[key]
            if record["state"] not in STATES:
                record["state"] = "queued"
            for key in ("phases", "runs"):
                if isinstance(row.get(key), list):
                    record[key] = row[key]
            if isinstance(row.get("worktrees"), dict):
                record["worktrees"] = row["worktrees"]
            self._workflows[record["id"]] = record

    def _quarantine(self, exc: Exception) -> None:
        assert self.path is not None
        bad = self.path.with_name(
            f"{self.path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(self.path, bad)
        except OSError:
            bad = self.path
        self._degraded = (
            f"workflows store is corrupt — quarantined to {bad}: {exc}")

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.path,
            json.dumps({"workflows": list(self._workflows.values())},
                       ensure_ascii=False, indent=1))

    # ── lookups ────────────────────────────────────────────────────

    def get_workflow(self, ref: str) -> dict[str, Any] | None:
        """A copy of the record with this exact id or name, else None."""
        with self._lock:
            for record in self._workflows.values():
                if record["id"] == ref or record["name"] == ref:
                    return copy.deepcopy(record)
        return None

    def resolve(self, ref: str) -> dict[str, Any] | None:
        """Resolve by exact id, unique id prefix, or exact name.

        An ambiguous prefix resolves to nothing — the caller names
        the workflow it means instead of guessing.
        """
        ref = (ref or "").strip()
        if not ref:
            return None
        with self._lock:
            exact = self.get_workflow(ref)
            if exact is not None:
                return exact
            matches = [r for r in self._workflows.values()
                       if r["id"].startswith(ref)]
            if len(matches) == 1:
                return copy.deepcopy(matches[0])
        return None

    def list_workflows(self) -> list[dict[str, Any]]:
        """All records, oldest first."""
        with self._lock:
            records = sorted(self._workflows.values(),
                             key=lambda r: r.get("created_at", 0.0))
        return copy.deepcopy(records)

    # ── creation ───────────────────────────────────────────────────

    def save_workflow(self, *, name: str, task: str, script: str,
                      args: dict[str, Any] | None = None,
                      chat_key: str = "") -> dict[str, Any]:
        """Save a script as a runnable workflow (state ``queued``).

        The script is written to ``<scripts_dir>/<name>.py`` when a
        scripts dir exists, and always kept in the index (the index
        is the source of truth). A duplicate name is refused with
        the name in the message.
        """
        name = (name or "").strip()
        if not _NAME_RE.match(name):
            raise WorkflowError(
                f"bad workflow name {name!r} — use letters, digits, "
                f"'-' and '_', starting with a letter or digit")
        if not (script or "").strip():
            raise WorkflowError("cannot save an empty workflow script")
        with self._lock:
            if any(r["name"] == name for r in self._workflows.values()):
                raise WorkflowError(
                    f"a workflow named {name!r} already exists — "
                    f"pick another name or /workflow run {name}")
            record = _blank_record(new_workflow_id(), name,
                                   " ".join(str(task or "").split()),
                                   script, dict(args or {}),
                                   str(chat_key or ""))
            if self.scripts_dir is not None:
                self.scripts_dir.mkdir(parents=True, exist_ok=True)
                script_path = self.scripts_dir / f"{name}.py"
                script_path.write_text(script, encoding="utf-8")
                record["script_path"] = str(script_path)
            self._workflows[record["id"]] = record
            self._save()
            return copy.deepcopy(record)

    # ── state transitions ──────────────────────────────────────────

    def _get(self, workflow_id: str) -> dict[str, Any]:
        record = self._workflows.get(workflow_id)
        if record is None:
            raise WorkflowError(f"no workflow {workflow_id!r}")
        return record

    def _require_actor(self, by: str) -> str:
        actor = (by or "").strip()
        if not actor:
            raise WorkflowError(
                "a workflow action needs an actor — who is asking?")
        return actor

    def _transition(self, workflow_id: str, from_states: tuple[str, ...],
                    to_state: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            record = self._get(workflow_id)
            if record["state"] not in from_states:
                raise WorkflowError(
                    f"workflow {record['name']!r} is "
                    f"{record['state']}, not "
                    f"{' or '.join(from_states)} — cannot move to "
                    f"{to_state}")
            record["state"] = to_state
            record.update(fields)
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)

    def mark_running(self, workflow_id: str) -> dict[str, Any]:
        """``queued`` → ``running`` (the runner starting the thread)."""
        return self._transition(workflow_id, ("queued",), "running")

    def mark_done(self, workflow_id: str,
                  summary: str = "") -> dict[str, Any]:
        """``running`` → ``done``."""
        return self._transition(workflow_id, ("running",), "done",
                                summary=summary,
                                finished_at=time.time())

    def mark_failed(self, workflow_id: str,
                    error: str = "") -> dict[str, Any]:
        """``running`` → ``failed``."""
        return self._transition(workflow_id, ("running",), "failed",
                                error=error,
                                finished_at=time.time())

    def mark_interrupted(self, workflow_id: str) -> dict[str, Any]:
        """``running`` → ``interrupted`` (its thread died with the process)."""
        return self._transition(workflow_id, ("running",), "interrupted")

    def cancel(self, workflow_id: str, by: str) -> dict[str, Any]:
        """``queued``/``running`` → ``cancelled``. The script stays saved."""
        self._require_actor(by)
        return self._transition(workflow_id, ("queued", "running"),
                                "cancelled")

    def requeue(self, workflow_id: str, by: str) -> dict[str, Any]:
        """A finished/cancelled run → ``queued`` again (re-run).

        Re-queueing a ``queued`` run is a no-op; re-queueing a
        ``running`` run is refused — it is already running.
        """
        self._require_actor(by)
        with self._lock:
            record = self._get(workflow_id)
            if record["state"] == "queued":
                return copy.deepcopy(record)
            if record["state"] == "running":
                raise WorkflowError(
                    f"workflow {record['name']!r} is already running")
        return self._transition(
            workflow_id, ("done", "failed", "interrupted", "cancelled"),
            "queued", error="", summary="")

    def resume(self, workflow_id: str, by: str) -> dict[str, Any]:
        """An ``interrupted``/``failed`` run → ``queued`` (pick it up)."""
        self._require_actor(by)
        return self._transition(workflow_id, ("interrupted", "failed"),
                                "queued", error="")

    def mark_summary_delivered(self, workflow_id: str) -> dict[str, Any]:
        """Record that the run's summary reached the owner (once)."""
        with self._lock:
            record = self._get(workflow_id)
            record["summary_delivered"] = True
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)

    # ── phase journal ──────────────────────────────────────────────

    def _open_phase(self, record: dict[str, Any]) -> dict[str, Any]:
        for phase in reversed(record["phases"]):
            if phase.get("finished_at") is None:
                return phase
        raise WorkflowError(
            f"workflow {record['name']!r} has no open phase — "
            f"call phase() first")

    def start_phase(self, workflow_id: str,
                    name: str) -> dict[str, Any]:
        """Open a phase entry in the journal (durable, at once)."""
        name = " ".join(str(name or "").split()).strip()
        if not name:
            raise WorkflowError("a phase needs a name")
        with self._lock:
            record = self._get(workflow_id)
            for phase in record["phases"]:
                if phase.get("name") == name \
                        and phase.get("finished_at") is None:
                    return copy.deepcopy(record)
            record["phases"].append({
                "name": name, "started_at": time.time(),
                "finished_at": None, "log": [], "agents": []})
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)

    def append_log(self, workflow_id: str, text: str) -> dict[str, Any]:
        """Append one log line to the open phase."""
        with self._lock:
            record = self._get(workflow_id)
            phase = self._open_phase(record)
            phase["log"].append(str(text))
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)

    def record_agent_result(self, workflow_id: str, role: str,
                            result: Any) -> dict[str, Any]:
        """Append one agent call's outcome to the open phase."""
        with self._lock:
            record = self._get(workflow_id)
            phase = self._open_phase(record)
            phase["agents"].append({
                "role": role, "result": result, "at": time.time()})
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)

    def finish_phase(self, workflow_id: str,
                     name: str) -> dict[str, Any]:
        """Close the open phase with this name."""
        with self._lock:
            record = self._get(workflow_id)
            for phase in reversed(record["phases"]):
                if phase.get("name") == name \
                        and phase.get("finished_at") is None:
                    phase["finished_at"] = time.time()
                    record["updated_at"] = time.time()
                    self._save()
                    return copy.deepcopy(record)
        raise WorkflowError(
            f"workflow {record['name']!r} has no open phase {name!r}")

    def record_worktree(self, workflow_id: str, handle: str,
                        path: str) -> dict[str, Any]:
        """Note where a handle's isolated worktree lives."""
        with self._lock:
            record = self._get(workflow_id)
            record["worktrees"][str(handle)] = str(path)
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)
