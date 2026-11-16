"""Durable plan store.

One atomic JSON index (``<home>/data/plans.json``) plus one readable
``.md`` Implementation Plan artifact per saved plan under
``<home>/plans/``.  Shape follows ``agents/workflow/store.py``
(FenceStore): tmp+replace writes; a corrupt store is quarantined and
degrades to empty with a named reason; bad rows are skipped; reads
never raise.

States: a plan is saved ``running`` (its runs are spawned at once by
:func:`plan_and_run`) and its terminal state is DERIVED from its
delegation runs — see :func:`derive_plan_state`.  There are no
approval states (owner order 2026-11-16): a saved plan runs, it is
not gated.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import PlanError
from .render import render_plan_md

__all__ = [
    "PlanStore",
    "derive_plan_state",
    "default_plans_dir",
    "default_plans_path",
    "new_plan_id",
]

#: Stored states. Terminal states are derived (see derive_plan_state);
#: the stored value is the last known state, starting at "running".
STATES = ("running", "done", "failed")

_LIVE_RUN_STATES = ("running", "steering", "finalizing")
_FAILED_RUN_STATES = ("failed", "cancelled")


def new_plan_id() -> str:
    """A fresh plan id: ``pl-`` + 12 hex chars."""
    return f"pl-{uuid.uuid4().hex[:12]}"


def default_plans_path(home: Path | str | None = None) -> Path:
    """The durable plans index under a home (``<home>/data/plans.json``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "data" / "plans.json"


def default_plans_dir(home: Path | str | None = None) -> Path:
    """The directory holding the readable plan artifacts (``<home>/plans``)."""
    base = Path(home) if home is not None else Path.home()
    return base / "plans"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically (tmp file + ``os.replace``)."""
    tmp = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _blank_record(plan_id: str, task: str, survey: dict[str, Any],
                  plan: dict[str, Any], runs: list[dict[str, Any]],
                  unassigned: list[dict[str, Any]], plan_error: str,
                  chat_key: str) -> dict[str, Any]:
    now = time.time()
    return {
        "id": plan_id,
        "task": " ".join(str(task or "").split()),
        "survey": survey,
        "plan": plan,
        "runs": runs,
        "unassigned": unassigned,
        "plan_error": plan_error,
        "chat_key": str(chat_key or ""),
        "state": "running",
        "created_at": now,
        "updated_at": now,
    }


def derive_plan_state(record: dict[str, Any], get_run: Any) -> str:
    """The plan's state, derived from its delegation runs.

    ``get_run`` maps a run id to a run with a ``status`` (or None
    when the run is gone).  Any live run → ``running``; any failed,
    cancelled, or missing run → ``failed``; every run terminal and
    successful → ``done``.  A plan with no runs at all failed to
    start (empty plan, or nothing the roster could run).
    """
    runs = record.get("runs") or []
    if not runs:
        return "failed"
    failed = False
    for run in runs:
        run_id = run.get("id") if isinstance(run, dict) else None
        live = get_run(run_id) if run_id and get_run is not None else None
        status = getattr(live, "status", None) if live is not None else None
        if live is None or status in _FAILED_RUN_STATES:
            failed = True
        elif status in _LIVE_RUN_STATES:
            return "running"
        elif status in ("completed", "collected"):
            continue
        else:
            # An unknown status is not success — the run is not done.
            return "running"
    return "failed" if failed else "done"


class PlanStore:
    """Saved Implementation Plans + run records, durable JSON, one file.

    ``path=None`` means in-memory only (tests, and runtimes with no
    home); the record shape is identical either way.
    """

    def __init__(self, path: Path | str | None = None,
                 plans_dir: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        if plans_dir is not None:
            self.plans_dir: Path | None = Path(plans_dir)
        elif self.path is not None:
            self.plans_dir = self.path.parent.parent / "plans"
        else:
            self.plans_dir = None
        self._lock = threading.RLock()
        self._plans: dict[str, dict[str, Any]] = {}
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
                    f"plans store must hold an object, found "
                    f"{type(raw).__name__}"))
            return
        for row in raw.get("plans", []) or []:
            if not isinstance(row, dict):
                continue
            if not isinstance(row.get("id"), str):
                continue
            record = _blank_record(
                row["id"], str(row.get("task", "")),
                row.get("survey") if isinstance(row.get("survey"), dict) else {},
                row.get("plan") if isinstance(row.get("plan"), dict) else {},
                row.get("runs") if isinstance(row.get("runs"), list) else [],
                row.get("unassigned") if isinstance(row.get("unassigned"), list) else [],
                str(row.get("plan_error", "")),
                str(row.get("chat_key", "")))
            for key in ("state", "created_at", "updated_at"):
                if key in row:
                    record[key] = row[key]
            if record["state"] not in STATES:
                record["state"] = "running"
            self._plans[record["id"]] = record

    def _quarantine(self, exc: Exception) -> None:
        assert self.path is not None
        bad = self.path.with_name(
            f"{self.path.name}.corrupt-{int(time.time())}")
        try:
            os.replace(self.path, bad)
        except OSError:
            bad = self.path
        self._degraded = (
            f"plans store is corrupt — quarantined to {bad}: {exc}")

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            self.path,
            json.dumps({"plans": list(self._plans.values())},
                       ensure_ascii=False, indent=1))

    # ── lookups ────────────────────────────────────────────────────

    def get_plan(self, ref: str) -> dict[str, Any] | None:
        """A copy of the record with this exact id, else None."""
        with self._lock:
            record = self._plans.get(ref)
            if record is None:
                return None
            return copy.deepcopy(record)

    def resolve(self, ref: str) -> dict[str, Any] | None:
        """Resolve by exact id or unique id prefix.

        An ambiguous prefix resolves to nothing — the caller names
        the plan it means instead of guessing.
        """
        ref = (ref or "").strip()
        if not ref:
            return None
        with self._lock:
            exact = self.get_plan(ref)
            if exact is not None:
                return exact
            matches = [r for r in self._plans.values()
                       if r["id"].startswith(ref)]
            if len(matches) == 1:
                return copy.deepcopy(matches[0])
        return None

    def list_plans(self) -> list[dict[str, Any]]:
        """All records, oldest first."""
        with self._lock:
            records = sorted(self._plans.values(),
                             key=lambda r: r.get("created_at", 0.0))
        return copy.deepcopy(records)

    # ── creation ───────────────────────────────────────────────────

    def save_plan(self, *, task: str, survey: dict[str, Any],
                  plan: dict[str, Any], runs: list[dict[str, Any]],
                  unassigned: list[dict[str, Any]],
                  plan_error: str = "", chat_key: str = "") -> dict[str, Any]:
        """Save a plan + its runs (state ``running``).

        The readable ``.md`` artifact is written separately by
        :meth:`write_artifact`, once the id exists.
        """
        task = " ".join(str(task or "").split())
        if not task:
            raise PlanError("cannot save a plan with no task")
        with self._lock:
            record = _blank_record(
                new_plan_id(), task, survey, plan, runs, unassigned,
                plan_error, chat_key)
            self._plans[record["id"]] = record
            self._save()
            return copy.deepcopy(record)

    def write_artifact(self, record: dict[str, Any]) -> Path | None:
        """Write (or rewrite) the ``.md`` artifact for a saved record.

        Records the artifact path on the stored record.  Returns the
        path written, or None when this store has no plans dir.
        """
        if self.plans_dir is None or not record.get("id"):
            return None
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = self.plans_dir / f"{record['id']}.md"
        artifact_path.write_text(render_plan_md(record), encoding="utf-8")
        with self._lock:
            held = self._plans.get(record["id"])
            if held is not None:
                held["artifact_path"] = str(artifact_path)
                held["updated_at"] = time.time()
            self._save()
        return artifact_path

    def refresh_state(self, plan_id: str, get_run: Any) -> dict[str, Any]:
        """Recompute a plan's stored state from its runs and persist it."""
        with self._lock:
            record = self._plans.get(plan_id)
            if record is None:
                raise PlanError(f"no plan {plan_id!r}")
            record["state"] = derive_plan_state(record, get_run)
            record["updated_at"] = time.time()
            self._save()
            return copy.deepcopy(record)
