"""Autonomous Project Mode (wave 50) — complex multi-step projects with
reduced human intervention.

A *project* is a long, multi-step undertaking that the system plans on its
own, executes step by step, self-corrects when a step fails, and reports on
progress. It is deliberately self-contained (its steps live in the kv_store)
so it can run without a linked goal, but it can also be backed by a goal for
cross-session tracking.

Design notes
------------
* **Self-planning.** :meth:`ProjectManager.plan` asks the model to break the
  objective into concrete, ordered steps; a heuristic fallback (split on the
  objective's clauses) keeps it working with no model.
* **Self-correction.** A failing step is retried up to ``max_attempts``; on
  each retry the step is *reworded* by the model (a genuinely different
  approach) rather than replayed verbatim.
* **Progress reporting.** :meth:`ProjectManager.report` renders a human-
  readable status (done/total, current step, blockers) for the owner.
* **Injectable executor.** The thing that actually *does* a step is a
  callable; the default delegates to the agent orchestrator/tools, and tests
  inject a deterministic one. No egress in tests.

Modular and callable by the main AI and sub-agents.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..llm.base import Message
from ..storage.kv import KVStore
from ..llm.brain import brain_for

__all__ = ["Project", "ProjectStep", "ProjectManager", "register"]

_log = get_logger(__name__)

STEP_STATUSES = ("pending", "in_progress", "done", "failed", "skipped")
PROJECT_STATUSES = ("planning", "running", "paused", "done", "failed")
_KV_PREFIX = "project.steps."
DEFAULT_MAX_ATTEMPTS = 3


class BudgetSuspended(RuntimeError):
    """The daily model budget cannot cover the NEXT unit of work (wave 66).

    Not a step failure: nothing was broken, there is just not enough model
    budget left today to reserve what this build will spend.  The project
    pauses with an honest report and resumes after the ledger rolls over —
    the owner (or the loop) can simply continue it later.
    """


@dataclass
class ProjectStep:
    id: str
    description: str
    status: str = "pending"
    result: str = ""
    attempts: int = 0
    revised: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "description": self.description,
            "status": self.status, "result": self.result,
            "attempts": self.attempts, "revised": self.revised,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProjectStep":
        return cls(
            id=str(d.get("id", "") or new_short_id()),
            description=str(d.get("description", "")),
            status=str(d.get("status", "pending")),
            result=str(d.get("result", "")),
            attempts=int(d.get("attempts", 0) or 0),
            revised=bool(d.get("revised", False)),
        )


@dataclass
class Project:
    id: str
    title: str
    objective: str = ""
    status: str = "planning"
    steps: list[ProjectStep] = field(default_factory=list)
    budget_wall: float = 0.0
    budget_tokens: int = 0
    progress: float = 0.0
    report: str = ""
    goal_id: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    task_kind: str = ""      # build|investigate|research|chat (wave 65)
    artifact: str = ""       # build only: the concrete thing to produce
    verify_cmd: str = ""     # build only: the REAL sandbox command that
                             # must exit 0 for a step to count as done

    # ── derived ──────────────────────────────────────────────────────────
    @property
    def total(self) -> int:
        return len(self.steps)

    @property
    def done_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "done")

    @property
    def failed_count(self) -> int:
        return sum(1 for s in self.steps if s.status == "failed")

    @property
    def next_step(self) -> ProjectStep | None:
        for s in self.steps:
            if s.status == "pending":
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "objective": self.objective,
            "status": self.status, "steps": [s.to_dict() for s in self.steps],
            "progress": round(self.progress, 3), "report": self.report,
            "goal_id": self.goal_id, "created_at": self.created_at,
            "updated_at": self.updated_at, "finished_at": self.finished_at,
            "task_kind": self.task_kind, "artifact": self.artifact,
            "verify_cmd": self.verify_cmd,
        }


class ProjectManager:
    """Create, plan, autonomously run, and report on multi-step projects."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = context.db
        self.max_attempts = 3

    # ── rooms (Prompt 05) ────────────────────────────────────────────────
    def _room_manager(self) -> Any | None:
        """Lazily build the RoomManager (None when rooms unavailable)."""
        if not hasattr(self, "_rooms_cache"):
            self._rooms_cache: Any = None
        if self._rooms_cache is False:
            return None
        if self._rooms_cache is None:
            try:
                from ..workspace.rooms import RoomManager
                from pathlib import Path
                root = Path(self.context.settings.workspace_dir)
                self._rooms_cache = RoomManager(root=root, db=self.db)
            except Exception:  # noqa: BLE001 — rooms are optional
                self._rooms_cache = False
                return None
        return self._rooms_cache

    def _linked_room(self, project_id: str) -> Any | None:
        mgr = self._room_manager()
        if mgr is None:
            return None
        try:
            return mgr.get_by_linked("project", project_id)
        except Exception:  # noqa: BLE001
            return None

    def _rooms_auto_create(self) -> bool:
        try:
            return bool(getattr(self.context.settings,
                                "rooms_auto_create", True))
        except Exception:  # noqa: BLE001
            return True

    def _maybe_auto_room(self, project_id: str, title: str,
                         steps: list[str] | None = None) -> None:
        if not self._rooms_auto_create():
            return
        mgr = self._room_manager()
        if mgr is None:
            return
        try:
            if mgr.get_by_linked("project", project_id) is None:
                mgr.create(title, kind="project", linked_id=project_id,
                           plan=steps)
        except Exception as exc:  # noqa: BLE001
            _log.warning("auto-room create failed for project %s: %s",
                         project_id, exc)

    # ── persistence ──────────────────────────────────────────────────────
    def _upsert_row(self, p: Project) -> None:
        self.db.execute(
            """
            INSERT INTO projects (id, title, objective, status, goal_id,
                                  budget_wall, budget_tokens, progress,
                                  report, created_at, updated_at, finished_at,
                                  task_kind, artifact, verify_cmd)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title, objective=excluded.objective,
                status=excluded.status, goal_id=excluded.goal_id,
                budget_wall=excluded.budget_wall, budget_tokens=excluded.budget_tokens,
                progress=excluded.progress, report=excluded.report,
                updated_at=excluded.updated_at, finished_at=excluded.finished_at,
                task_kind=excluded.task_kind, artifact=excluded.artifact,
                verify_cmd=excluded.verify_cmd
            """,
            (p.id, p.title, p.objective, p.status, p.goal_id,
             p.budget_wall, p.budget_tokens, p.progress, p.report,
             p.created_at, p.updated_at, p.finished_at or 0.0,
             p.task_kind, p.artifact, p.verify_cmd),
        )
        KVStore(self.db).set_raw(
            _KV_PREFIX + p.id,
            json.dumps([s.to_dict() for s in p.steps], default=str),
            "json",
        )

    def _load(self, project_id: str) -> Project | None:
        row = self.db.query_one(
            "SELECT * FROM projects WHERE id = ?", (project_id,)
        )
        if not row:
            return None
        steps: list[ProjectStep] = []
        kv = KVStore(self.db).get(_KV_PREFIX + project_id)
        if kv:
            try:
                steps = [ProjectStep.from_dict(d) for d in kv]
            except (ValueError, TypeError):
                steps = []
        return Project(
            id=row["id"], title=row["title"], objective=row["objective"] or "",
            status=row["status"], steps=steps,
            budget_wall=row["budget_wall"] or 0.0,
            budget_tokens=int(row["budget_tokens"] or 0),
            progress=row["progress"] or 0.0, report=row["report"] or "",
            goal_id=row["goal_id"] or "", created_at=row["created_at"],
            updated_at=row["updated_at"], finished_at=row["finished_at"],
            task_kind=(row.get("task_kind") if isinstance(row, dict)
                       else row["task_kind"]) or "",
            artifact=(row.get("artifact") if isinstance(row, dict)
                      else row["artifact"]) or "",
            verify_cmd=(row.get("verify_cmd") if isinstance(row, dict)
                        else row["verify_cmd"]) or "",
        )

    # ── create ───────────────────────────────────────────────────────────
    def create(self, title: str, objective: str = "",
               steps: list[str] | None = None,
               goal_id: str = "",
               budget_wall: float = 0.0, budget_tokens: int = 0) -> Project:
        now = time.time()
        p = Project(
            id=new_short_id(), title=title or objective or "project",
            objective=objective or title, status="planning",
            budget_wall=budget_wall, budget_tokens=budget_tokens,
            goal_id=goal_id, created_at=now, updated_at=now,
        )
        # The system KNOWS its task type from the pure text (wave 65):
        # deterministic keywords first, model tiebreak only when silent.
        from .task_type import acceptance_command, artifact_filename

        tt = self._classify_tt(p.objective or p.title)
        p.task_kind, p.artifact = tt.kind, tt.artifact
        # and a build carries its REAL acceptance command (wave 66):
        # the exact sandbox command that must exit 0 for work to count.
        if tt.kind == "build":
            p.verify_cmd = acceptance_command(tt.verify,
                                              artifact_filename(tt.artifact))
        if steps:
            p.steps = [ProjectStep(id=new_short_id(), description=s) for s in steps]
            p.status = "running"
        self._upsert_row(p)
        # Prompt 05: every project gets a room (configurable, default ON)
        self._maybe_auto_room(p.id, p.title,
                              [s.description for s in p.steps] or None)
        return p

    def _classify_tt(self, text: str):
        """Full task-type classification of a project objective. Never fatal."""
        try:
            from .task_type import TaskType, classify_task

            return classify_task(self.context, text)
        except Exception as exc:  # noqa: BLE001
            _log.debug("task classification failed: %s", exc)
            from .task_type import TaskType

            return TaskType()

    # ── planning ─────────────────────────────────────────────────────────
    def plan(self, project_id: str, *, context_note: str = "") -> Project:
        p = self._load(project_id)
        if p is None:
            raise KeyError(f"unknown project: {project_id!r}")
        if p.steps:
            return p  # already planned
        objective = p.objective or p.title
        step_texts = self._plan_steps(
            objective, context_note=context_note,
            task_kind=p.task_kind, artifact=p.artifact,
            verify_cmd=p.verify_cmd)
        p.steps = [ProjectStep(id=new_short_id(), description=s) for s in step_texts]
        if p.steps:
            p.status = "running"
        p.report = f"Planned {len(p.steps)} steps."
        p.updated_at = time.time()
        self._upsert_row(p)
        return p

    def _plan_steps(self, objective: str, *, context_note: str = "",
                    task_kind: str = "", artifact: str = "",
                    verify_cmd: str = "") -> list[str]:
        # 1) model
        user = f"Objective: {objective}"
        # closed loop (wave 65): proven skills for this kind of work steer
        # the plan — what the system already learned actually works.
        skills_block = self._skill_context(f"{objective} {artifact}"[:240])
        if skills_block:
            user = f"{skills_block}\n\n{user}"
        # build-aware planning (wave 65): a BUILD objective must plan REAL,
        # runnable work — no "describe the implementation" theater.
        if task_kind == "build":
            target = artifact or "the described deliverable"
            user += (
                f"\n\nThis is a BUILD task. The artifact is: {target}. "
                "Every step must produce or extend REAL runnable files. "
                "Include an explicit final step that RUNS the artifact and "
                "checks its actual output. Never plan a step that only "
                "describes, discusses, or narrates the work.")
            if verify_cmd:
                user += (f"\nSteps are verified for real: the sandbox runs "
                         f"`{verify_cmd}` and the step only counts as done "
                         f"when it exits 0. Plan the artifact so that "
                         f"command genuinely passes.")
        if context_note:
            user += (f"\n\nContext from past attempts (plan around these — "
                     f"the previous plan failed or was superseded):"
                     f"\n{context_note}")
        try:
            resp = brain_for(self.context).chat(
                [Message.system(
                     "You break a project objective into 3-7 concrete, ordered, "
                     "self-contained action steps. Reply with a JSON array of "
                     "strings. No commentary."),
                 Message.user(user)],
            task_kind="plan")
            if resp.ok:
                arr = _extract_json_array(resp.text or "")
                if arr:
                    return [str(x).strip() for x in arr if str(x).strip()]
        except Exception as exc:  # noqa: BLE001
            _log.debug("project planning via model failed: %s", exc)
        # 2) heuristic: split on sentence/conjunction boundaries
        parts = [c.strip(" .!?,;") for c in _split_clauses(objective)]
        parts = [c for c in parts if c]
        if not parts:
            parts = ["Investigate the objective and produce a concrete deliverable"]
        if context_note:
            parts = parts[:6] + ["Work around the known failures from the "
                                 "previous attempts (see learned lessons)"]
        return parts[:7]

    # ── autonomous replanning (wave 63) ───────────────────────────────────
    def replan(self, project_id: str, *, context_note: str = "") -> dict[str, Any]:
        """Re-plan a project that is stuck or failing.

        Done steps are KEPT (no work is thrown away); failed and pending
        steps are discarded and the remaining work is planned fresh — with
        the failure context (what was tried, what the failure analyzer
        learned) folded into the planning prompt so the new plan is a
        genuinely different approach, not a repeat.
        """
        p = self._load(project_id)
        if p is None:
            return {"ok": False, "error": f"unknown project {project_id!r}"}
        if p.status == "done":
            return {"ok": False, "error": "project is done, nothing to replan"}
        if not (p.objective or p.title):
            return {"ok": False, "error": "project has no objective to replan"}
        if not context_note:
            context_note = self._failure_context(p)
        done_steps = [s for s in p.steps if s.status == "done"]
        step_texts = self._plan_steps(p.objective or p.title,
                                      context_note=context_note,
                                      task_kind=p.task_kind,
                                      artifact=p.artifact,
                                      verify_cmd=p.verify_cmd)
        p.steps = ([ProjectStep(id=new_short_id(), description=s.description,
                                status="done", result=s.result,
                                attempts=s.attempts, revised=s.revised)
                    for s in done_steps]
                   + [ProjectStep(id=new_short_id(), description=s)
                      for s in step_texts])
        p.status = "running"
        p.finished_at = None
        p.report = (f"Replanned: kept {len(done_steps)} done step(s), "
                    f"new plan of {len(step_texts)} step(s) built with failure "
                    f"context.")
        p.updated_at = time.time()
        self._upsert_row(p)
        return {"ok": True, "project_id": project_id, "status": p.status,
                "kept_steps": len(done_steps), "new_steps": len(step_texts),
                "steps": [s.description for s in p.steps]}

    def _failure_context(self, p: Project) -> str:
        """What the failure analyzer has learned about this project's
        failures, plus what was already tried. Empty when there is none."""
        bits: list[str] = []
        tried = [s for s in p.steps if s.status in {"failed", "done"}]
        if tried:
            bits.append("Already attempted:\n" + "\n".join(
                f"  - [{s.status}] {s.description[:120]}"
                + (f" → {s.result[:120]}" if s.result else "")
                for s in tried[:6]))
        try:
            from .failure import enrich_with_lessons

            lessons = enrich_with_lessons(
                self.context,
                (p.objective or p.title) + " " + p.report[:200], limit=4)
            if lessons:
                bits.append(lessons)
        except Exception:  # noqa: BLE001 — context is a bonus, never fatal
            pass
        return "\n\n".join(bits)[:3000]

    # ── execution ────────────────────────────────────────────────────────
    def advance(self, project_id: str,
                executor: Callable[[str], str] | None = None,
                max_attempts: int | None = None) -> Project:
        p = self._load(project_id)
        if p is None:
            raise KeyError(f"unknown project: {project_id}")
        attempts = max_attempts or self.max_attempts
        if p.status not in {"running", "planning"}:
            # wave 67: a project paused ON MODEL BUDGET resumes itself
            # the moment the budget can cover its next step — advance()
            # is the single resume path (heartbeat, goal drive, manual).
            if (p.status == "paused"
                    and str(p.report or "").startswith(
                        "Suspended on model budget")):
                from .cognition import ModelBudget

                cost = 3 if p.task_kind == "build" else 1
                if ModelBudget(self.context).affordable(cost):
                    p.status = "running"
                    p.report = ("Resumed after budget rollover — "
                                "continuing where it suspended.")
                    self._upsert_row(p)
                else:
                    return p  # still waiting on the rollover
            else:
                return p
        if p.status == "planning":
            p = self.plan(project_id)
        step = p.next_step
        if step is None:
            p = self._finalise(p)
            return p
        p.status = "running"
        step.status = "in_progress"
        step.attempts += 1
        self._current_project_id = project_id
        self._current_step_id = step.id  # wave 67: acceptance regression
        exec_fn = executor or self._default_executor
        # Prompt 05: a room-linked project runs its step inside the
        # RoomContext — activity logged, state checkpointed afterwards.
        # Without a room, exec_fn is untouched (original behavior).
        room = self._linked_room(project_id)
        room_ctx = None
        if room is not None:
            mgr = self._room_manager()
            if mgr is not None:
                room_ctx = mgr.enter(room.slug)
                room_ctx.set_step(step.description[:200])
                _inner, _ctx, _sid = exec_fn, room_ctx, step.id

                def exec_fn(desc: str, _i=_inner, _c=_ctx,
                            _s=_sid) -> str:  # noqa: F811
                    _c.log(_s, "start", {"description": desc[:500]})
                    try:
                        out = _i(desc)
                    except Exception as exc:
                        _c.log(_s, "error",
                               {"error": type(exc).__name__})
                        raise
                    _c.log(_s, "end", {"status": "ok"})
                    return out
        try:
            result = exec_fn(step.description)
            step.status = "done"
            step.result = str(result or "ok")[:4000]
        except BudgetSuspended as exc:  # noqa: BLE001 — budget, not a failure
            # The budget can't cover this step's reservation: pause (don't
            # fail, don't burn retries) — the step stays pending for a
            # later advance, and the linked goal pauses with it.
            step.status = "pending"
            step.result = f"suspended: {exc}"
            p.status = "paused"
            p.report = f"Suspended on model budget: {exc}"
            p.updated_at = time.time()
            self._upsert_row(p)
            self._sync_goal(p)
            return p
        except Exception as exc:  # noqa: BLE001 — self-correction
            if step.attempts >= attempts:
                step.status = "failed"
                step.result = f"failed after {attempts} attempts: {exc}"
                p.status = "failed"
                p.report = f"Step '{step.description[:60]}' failed after {attempts} attempts: {exc}"
                self._record_failure(step.description, str(exc))
            else:
                # wave 85: mid-task re-evaluation — the reasoning agent
                # decides whether the approach is weak or wrong and names
                # the pivot; the retry follows the pivot instead of just
                # rewording, and a "stop" verdict abandons the step early
                # instead of burning the remaining attempts.
                pivot = self._course_pivot(
                    step.description, str(exc), step.attempts)
                verdict = str(pivot.get("pivot") or "")
                if pivot.get("should_pivot") and "stop" in verdict.lower():
                    step.status = "failed"
                    step.result = (f"abandoned after {step.attempts} "
                                   f"attempt(s): {exc} (reasoning: stop — "
                                   f"{verdict[:120]})")
                    p.status = "failed"
                    p.report = (f"Step '{step.description[:60]}' abandoned — "
                                f"{str(pivot.get('rationale') or '')[:200]}")
                    self._record_failure(step.description, str(exc))
                else:
                    # revise the step (a genuinely different approach)
                    step.status = "pending"
                    step.revised = True
                    step.result = f"attempt {step.attempts} failed: {exc}"
                    step.description = self._revise_step(
                        step.description, str(exc), hint=verdict)
            p.updated_at = time.time()
            self._upsert_row(p)
            if p.status == "failed":
                # a project that gives up pauses its linked goal (wave 51
                # two-way sync) — the owner can adapt and resume it later.
                self._sync_goal(p)
            return p
        finally:
            # Prompt 05: checkpoint + release the room after the step,
            # whichever way the step ended (done / paused / failed)
            if room_ctx is not None:
                try:
                    room_ctx.checkpoint(
                        f"project step {step.id}: {step.status}")
                finally:
                    room_ctx.close()
        p.updated_at = time.time()
        p = self._recompute(p)
        self._upsert_row(p)
        return p

    def run(self, project_id: str, *, max_steps: int | None = None,
            executor: Callable[[str], str] | None = None,
            max_attempts: int | None = None) -> dict[str, Any]:
        """Autonomously execute until done/failed or *max_steps* steps ran."""
        ran = 0
        last = self._load(project_id)
        if last is not None and last.status == "planning" and not last.steps:
            # a fresh project (nm project create) is still unplanned — plan
            # it so `run` works in one command
            last = self.plan(project_id)
        while last is not None and last.status == "running" and last.next_step is not None:
            if max_steps is not None and ran >= max_steps:
                last.status = "paused"
                last.report = f"Paused after {ran} steps (max_steps reached)."
                self._upsert_row(last)
                break
            last = self.advance(project_id, executor=executor, max_attempts=max_attempts)
            ran += 1
            if last.status in {"done", "failed"}:
                break
        return {
            "id": project_id, "status": last.status if last else "missing",
            "steps_ran": ran, "progress": last.progress if last else 0.0,
            "report": last.report if last else "",
        }

    def _finalise(self, p: Project) -> Project:
        p.progress = (p.done_count / p.total) if p.total else 0.0
        if p.failed_count and not p.done_count:
            p.status = "failed"
        else:
            p.status = "done"
        p.finished_at = time.time()
        p.report = self.report_text(p)
        self._upsert_row(p)
        self._sync_goal(p)
        return p

    def _sync_goal(self, p: Project) -> None:
        """Mirror this project's outcome onto its linked goal (wave 51)."""
        if not p.goal_id:
            return
        try:
            row = self.db.query_one("SELECT id FROM agent_goals WHERE id=?",
                                    (p.goal_id,))
            if not row:
                return
            now = time.time()
            if p.status == "done":
                self.db.execute(
                    "UPDATE agent_goals SET status='done', progress=1.0, "
                    "finished_at=?, updated_at=? WHERE id=?",
                    (now, now, p.goal_id))
                self.db.execute(
                    "UPDATE agent_goal_steps SET status='done' "
                    "WHERE goal_id=? AND status IN ('pending','in_progress',"
                    "'blocked')", (p.goal_id,))
                try:
                    from .goals import GoalSystem

                    GoalSystem(self.context)._record_completion_knowledge(
                        p.goal_id)
                except Exception:  # noqa: BLE001
                    pass
            elif p.status == "failed":
                self.db.execute(
                    "UPDATE agent_goals SET status='paused', progress=?, "
                    "updated_at=? WHERE id=?",
                    (p.progress, now, p.goal_id))
            elif p.status == "paused":
                # wave 67: a suspended (budget) or otherwise paused
                # project pauses its goal with it — the goal must not
                # look active while its project is waiting.
                self.db.execute(
                    "UPDATE agent_goals SET status='paused', progress=?, "
                    "updated_at=? WHERE id=?",
                    (p.progress, now, p.goal_id))
        except Exception as exc:  # noqa: BLE001 — goal sync is best-effort
            _log.debug("goal sync failed: %s", exc)

    def _recompute(self, p: Project) -> Project:
        p.progress = (p.done_count / p.total) if p.total else 0.0
        if (p.total and p.done_count == p.total
                and p.status not in {"done", "failed"}):
            p = self._finalise(p)
        return p

    def _course_pivot(self, description: str, error: str,
                      attempts: int) -> dict[str, Any]:
        """One bounded course-correction consult on a failed step.

        ``{}`` when reasoning is off or the budget is exhausted — the
        caller then falls back to the plain reworded retry, so a dead
        model never blocks a project.
        """
        try:
            from .reasoning import ReasoningAgent, reasoning_enabled

            if not reasoning_enabled(self.context):
                return {}
            agent = ReasoningAgent(self.context)
            return agent.course_correct(
                f"Project step failed on attempt {attempts}. "
                f"Step: {description[:200]}. Error: {error[:300]}.",
                scope="project", attempts=attempts,
            )
        except Exception:  # noqa: BLE001 — correction is a bonus, never fatal
            return {}

    def _revise_step(self, description: str, error: str,
                     hint: str = "") -> str:
        # Self-healing (wave 62): fold in what the failure analyzer has
        # ALREADY LEARNED about similar failures, so the retry is informed
        # by past experience, not just the last error message.
        known = ""
        try:
            from .failure import enrich_with_lessons

            known = enrich_with_lessons(
                self.context, f"{description} {error}", limit=3)
        except Exception:  # noqa: BLE001 — lessons are a bonus, never fatal
            known = ""
        user = f"Step: {description}\nError: {error}"
        # wave 85: the reasoning agent's named pivot, when it has one
        if hint:
            user += f"\n\nThe mid-task re-evaluation says to pivot: {hint[:200]}"
        if known:
            user += f"\n\n{known}\nRewrite the step so it avoids these known failure patterns."
        try:
            resp = brain_for(self.context).chat(
                [Message.system("A project step failed. Rewrite it as a different, "
                                "more robust approach in ONE short sentence."),
                 Message.user(user)],
            task_kind="plan")
            if resp.ok and (resp.text or "").strip():
                return resp.text.strip()[:300]
        except Exception:  # noqa: BLE001
            pass
        return description

    def _record_failure(self, description: str, error: str) -> None:
        """Feed a final step failure into the failure-analyzer pipeline.

        Closes the learning loop: the failure is journaled (coding_log),
        analyzed, and persisted as a durable lesson + prevention skill that
        future steps (and ``heal``) can use. Best-effort — learning must
        never block execution.
        """
        try:
            import uuid

            from .failure import FailureAnalyzer, FailureCase

            db = self.db
            db.execute(
                "INSERT INTO coding_log (id, task, filename, attempt, "
                "exit_code, stdout, stderr, code, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (f"cl{uuid.uuid4().hex}",
                 f"project step: {description[:180]}", "", 1, 1,
                 "", str(error)[:4000], "", time.time()))
            analyzer = FailureAnalyzer(self.context)
            analyzer.learn_from_failure(FailureCase(
                source="project", summary=description[:200],
                error=str(error)[:1500], ts=time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("failure learning failed: %s", exc)

    def _default_executor(self, description: str) -> str:
        p = None
        try:
            p = self._load(self._current_project_id) if getattr(
                self, "_current_project_id", "") else None
        except Exception:  # noqa: BLE001
            p = None

        # BUILD tasks execute for REAL (wave 65): draft -> sandbox run ->
        # fix, with a real file and a real exit code.  This path takes
        # priority over the orchestrator on purpose — a build must produce
        # a working artifact, not a narration of one.
        if p is not None and p.task_kind == "build":
            return self._execute_build_step(p, description)

        # predictive budget reservation (wave 66, extended wave 67):
        # NON-build steps reserve their model call(s) too — every kind
        # of project work meters alike, and an unaffordable step
        # suspends cleanly (advance() pauses it) instead of dying
        # mid-execution with the budget half-eaten.
        from .cognition import ModelBudget

        budget = ModelBudget(self.context)
        res = budget.reserve(1, task=f"step: {description[:60]}")
        if not res.get("ok"):
            raise BudgetSuspended(res.get("reason") or "model budget")
        snap = ModelBudget.usage_snapshot(self.context)
        try:
            # Delegate to the orchestrator if present, else a single model call.
            orch = getattr(self.context, "orchestrator", None)
            if orch is not None and hasattr(orch, "run"):
                out = orch.run(description)
                return str(out)
            # cross-goal intelligence (wave 63): if this project serves a goal
            # whose dependencies already finished, give the executor what they
            # actually produced
            upstream = ""
            if p is not None and p.goal_id:
                try:
                    from .goals import GoalSystem

                    upstream = GoalSystem(self.context).upstream_knowledge(
                        p.goal_id)
                except Exception:  # noqa: BLE001 — bonus context, never fatal
                    upstream = ""
            # closed loop (wave 65): recall proven skills for this step; record
            # the outcome so the skill's use stats track reality.
            recalled = self._recall_skills(description, p)
            user = description
            if recalled:
                user = f"{self._skill_context_block(recalled)}\n\n{user}"
            if upstream:
                user = f"{upstream}\n\nYour step now: {user}"
            resp = brain_for(self.context).chat(
                [Message.system("You are executing one step of a project. Do it and "
                                "report the concrete result in 1-3 sentences."),
                 Message.user(user)],
            task_kind="plan")
            if not resp.ok:
                self._record_skill_use(recalled, success=False, task=description)
                raise RuntimeError(resp.error or "execution failed")
            self._record_skill_use(recalled, success=True, task=description)
            return resp.text or ""
        finally:
            # settle the step's REAL spend in the daily ledger and close
            # the reservation — even when the step raised (retries
            # reserve again, which is the point).
            try:
                budget.meter_session(snap)
            finally:
                budget.commit(res.get("reservation_id", ""))

    # ── real build execution (wave 65) ─────────────────────────────────────
    def _execute_build_step(self, p: "Project", description: str) -> str:
        """Execute one step of a BUILD project with the real coding agent.

        Draft -> sandbox run -> fix, with the previous step's file content
        seeded in (multi-step builds EXTEND the artifact).  Raises on a
        red exit code so the project's self-correction (revise -> retry ->
        heal) kicks in exactly like any other failure.
        """
        from .coding import CodingAgent
        from .task_type import artifact_filename
        from ..tools.filesystem import safe_path

        filename = artifact_filename(p.artifact)
        path = safe_path(self.context, filename)
        seed = ""
        try:
            if path.exists():
                seed = path.read_text(encoding="utf-8",
                                      errors="replace")[:20000]
        except Exception:  # noqa: BLE001
            seed = ""

        task = f"{p.objective or p.title}\n\nCurrent step: {description}"
        if p.goal_id:
            try:
                from .goals import GoalSystem

                upstream = GoalSystem(self.context).upstream_knowledge(
                    p.goal_id)
                if upstream:
                    task = f"{upstream}\n\n{task}"
            except Exception:  # noqa: BLE001 — bonus context, never fatal
                pass
        recalled = self._recall_skills(f"{description} {p.artifact}", p)
        if recalled:
            task = f"{self._skill_context_block(recalled)}\n\n{task}"

        # predictive budget reservation (wave 66): a build session spends
        # up to max_iterations draft calls (+ review calls); if the daily
        # cap can't cover it, suspend NOW instead of dying mid-build.
        from .cognition import ModelBudget

        budget = ModelBudget(self.context)
        res = budget.reserve(3, task=f"build step: {description[:60]}")
        if not res.get("ok"):
            raise BudgetSuspended(res.get("reason") or "model budget")
        snap = ModelBudget.usage_snapshot(self.context)
        result = CodingAgent(self.context).run(
            task, filename=filename, max_iterations=3, timeout=60.0,
            seed_code=seed, accept=p.verify_cmd)
        # settle the session's REAL spend in the daily ledger (out-of-tick
        # work counts the same as tick work) and close the reservation.
        try:
            budget.meter_session(snap)
        finally:
            budget.commit(res.get("reservation_id", ""))
        if not result.ok:
            self._record_skill_use(recalled, success=False, task=description)
            raise RuntimeError(
                f"build of {filename} failed after "
                f"{result.iterations} iteration(s): "
                f"{(result.error or 'unknown error')[:300]}")
        self._record_skill_use(recalled, success=True, task=description)
        # acceptance regression (wave 67): freeze THIS step's acceptance
        # run, then re-run every PRIOR step's frozen acceptance command
        # against the current workspace — a step that "fixes" the current
        # step by breaking an earlier one must fail here, not pass.
        step_id = getattr(self, "_current_step_id", "")
        self._record_acceptance_run(p, step_id, p.verify_cmd,
                                    result.output or "")
        reg = self._replay_prior_acceptances(p, step_id)
        if not reg["ok"]:
            raise RuntimeError("regression detected — " + reg["detail"][:300])
        out = (result.output or "").strip().replace("\n", " ")
        return (f"built {filename} for real (iteration {result.iterations}, "
                f"exit 0): {out[:400]}")

    # ── acceptance regression suite (wave 67) ──────────────────────────────
    def _record_acceptance_run(self, p: "Project", step_id: str,
                               cmd: str, output: str) -> None:
        """Freeze one acceptance run (command + output hash) for the
        project's regression suite.  Best-effort — a logging hiccup
        must never fail a step that genuinely passed."""
        try:
            if not (cmd and step_id):
                return
            import hashlib

            digest = hashlib.sha256(str(output or "").encode(
                "utf-8", "replace")).hexdigest()[:16]
            self.db.execute(
                "INSERT INTO acceptance_runs "
                "(id, project_id, step_id, cmd, output_hash, ok, ts) "
                "VALUES (?,?,?,?,?,1,?)",
                (new_short_id("acc"), p.id, step_id, cmd, digest,
                 time.time()))
        except Exception as exc:  # noqa: BLE001
            _log.debug("acceptance record failed: %s", exc)

    def acceptance_history(self, project_id: str,
                           limit: int = 50) -> list[dict[str, Any]]:
        """The frozen acceptance runs for a project, oldest first."""
        try:
            return list(self.db.query(
                "SELECT id, project_id, step_id, cmd, output_hash, ok, ts "
                "FROM acceptance_runs WHERE project_id=? "
                "ORDER BY ts ASC LIMIT ?", (project_id, limit)))
        except Exception:  # noqa: BLE001
            return []

    def _prior_acceptance_cmds(self, p: "Project",
                               exclude_step_id: str = "") -> list[str]:
        """The latest PASSING acceptance command of every other step of
        this project — the regression set for the current step."""
        try:
            rows = self.db.query(
                "SELECT step_id, cmd FROM acceptance_runs "
                "WHERE project_id=? AND ok=1 AND step_id != ? "
                "ORDER BY ts ASC", (p.id, exclude_step_id))
        except Exception:  # noqa: BLE001
            return []
        latest: dict[str, str] = {}
        for r in rows:  # ascending ts: last write per step wins
            latest[str(r.get("step_id") or "")] = str(r.get("cmd") or "")
        return [c for c in latest.values() if c]

    def _replay_prior_acceptances(self, p: "Project",
                                  exclude_step_id: str = "") -> dict[str, Any]:
        """Re-run the frozen acceptance commands of every PRIOR step
        against the CURRENT workspace.  All green -> the new step can
        stand; any red -> the project's self-correction takes over."""
        cmds = self._prior_acceptance_cmds(p, exclude_step_id)
        if not cmds:
            return {"ok": True, "results": []}
        from .coding import CodingAgent
        from ..tools.filesystem import safe_path

        runner = CodingAgent(self.context)
        results: list[dict[str, Any]] = []
        for cmd in cmds:
            try:
                r = runner._run(cmd, safe_path(self.context, "."), 60.0)
                ok = int(r.get("exit_code", -1)) == 0 \
                    and not r.get("timed_out")
                detail = " ".join(
                    str(r.get("stderr") or r.get("stdout") or "").split())
                results.append({"cmd": cmd, "ok": ok,
                                "detail": detail[:200]})
                if not ok:
                    return {"ok": False, "results": results,
                            "detail": (f"'{cmd[:80]}' regressed (exit "
                                       f"{r.get('exit_code')}): "
                                       f"{detail[:150]}")}
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "results": results,
                        "detail": f"'{cmd[:80]}' could not be replayed: {exc}"}
        return {"ok": True, "results": results}

    def regression_check(self, project_id: str) -> dict[str, Any]:
        """Public regression suite: re-run EVERY frozen acceptance
        command of the project against the current workspace and report
        step by step.  `nm project regress <id>` is this."""
        p = self._load(project_id)
        if p is None:
            raise KeyError(f"unknown project: {project_id!r}")
        try:
            rows = self.db.query(
                "SELECT step_id, cmd FROM acceptance_runs "
                "WHERE project_id=? AND ok=1 ORDER BY ts ASC", (p.id,))
        except Exception:  # noqa: BLE001
            return {"ok": True, "results": [], "project_id": p.id,
                    "note": "no acceptance runs recorded"}
        latest: dict[str, str] = {}
        for r in rows:
            latest[str(r.get("step_id") or "")] = str(r.get("cmd") or "")
        if not latest:
            return {"ok": True, "results": [], "project_id": p.id,
                    "note": "no acceptance runs recorded"}
        from .coding import CodingAgent
        from ..tools.filesystem import safe_path

        runner = CodingAgent(self.context)
        results: list[dict[str, Any]] = []
        all_ok = True
        for step_id, cmd in latest.items():
            try:
                r = runner._run(cmd, safe_path(self.context, "."), 60.0)
                ok = int(r.get("exit_code", -1)) == 0 \
                    and not r.get("timed_out")
                detail = " ".join(
                    str(r.get("stderr") or r.get("stdout") or "").split())
                results.append({"step_id": step_id, "cmd": cmd, "ok": ok,
                                "detail": detail[:200]})
                all_ok = all_ok and ok
            except Exception as exc:  # noqa: BLE001
                results.append({"step_id": step_id, "cmd": cmd, "ok": False,
                                "detail": f"could not be replayed: {exc}"})
                all_ok = False
        return {"ok": all_ok, "results": results, "project_id": p.id}

    # ── closed-loop skills (wave 65) ───────────────────────────────────────
    def _recall_skills(self, query: str,
                       p: "Project | None" = None) -> list:
        """Relevant proven skills for a step/query; [] when none apply."""
        try:
            from .skills import SkillLibrary

            lib = SkillLibrary(self.db)
            return lib.recall((query or "")[:200], limit=2)
        except Exception:  # noqa: BLE001 — skills must never block execution
            return []

    def _skill_context(self, query: str) -> str:
        """Formatted prior-art block for planning prompts (may be empty)."""
        try:
            from .skills import SkillLibrary

            return SkillLibrary(self.db).context_block((query or "")[:200],
                                                       limit=3)
        except Exception:  # noqa: BLE001
            return ""

    @staticmethod
    def _skill_context_block(recalled: list) -> str:
        if not recalled:
            return ""
        # One line per skill: the body is squashed so the block stays
        # parseable by coding.core_request (which strips it to judge the
        # owner's ACTUAL request for the reasoning gates).
        import re as _re

        lines = ["Relevant proven skills (apply their lessons):"]
        for s, _score in recalled[:3]:
            body = _re.sub(r"\s+", " ", s.body or "").strip()[:160]
            lines.append(f"- {s.name} [{s.kind}]: {body}")
        return "\n".join(lines)

    def _record_skill_use(self, recalled: list, *, success: bool,
                          task: str = "") -> None:
        """Feed step outcomes back into the skills that helped (wave 65)."""
        if not recalled:
            return
        try:
            from .skills import SkillLibrary

            lib = SkillLibrary(self.db)
            for s, _score in recalled:
                lib.record_use(s.id, success=success, task=(task or "")[:200])
        except Exception:  # noqa: BLE001 — reinforcement is best-effort
            pass

    # ── reporting ───────────────────────────────────────────────────────
    def report_text(self, p: Project) -> str:
        kind_note = ""
        if p.task_kind:
            kind_note = f" [{p.task_kind}"
            if p.artifact:
                kind_note += f" → {p.artifact}"
            kind_note += "]"
        if not p.steps:
            return (f"Project '{p.title}' ({p.status}): not planned yet."
                    f"{kind_note}")
        lines = [f"Project '{p.title}' — {p.status} — {p.done_count}/"
                 f"{p.total} steps done{kind_note}"]
        for s in p.steps:
            mark = {"done": "[x]", "failed": "[!]", "in_progress": "[>]",
                    "pending": "[ ]", "skipped": "[-]"}.get(s.status, "[ ]")
            extra = f"  ({s.result[:80]})" if s.result else ""
            lines.append(f"  {mark} {s.description}{extra}")
        nxt = p.next_step
        if nxt and p.status == "running":
            lines.append(f"Next: {nxt.description}")
        return "\n".join(lines)

    def report(self, project_id: str) -> dict[str, Any]:
        p = self._load(project_id)
        if p is None:
            return {"ok": False, "error": f"unknown project: {project_id}"}
        p.report = self.report_text(p)
        return {
            "id": p.id, "title": p.title, "status": p.status,
            "progress": round(p.progress, 3), "report": p.report,
            "total": p.total, "done": p.done_count, "failed": p.failed_count,
            "task_kind": p.task_kind, "artifact": p.artifact,
        }

    def status(self, project_id: str) -> dict[str, Any]:
        p = self._load(project_id)
        if p is None:
            return {"ok": False, "error": f"unknown project: {project_id}"}
        d = p.to_dict()
        d["ok"] = True
        return d

    def list_projects(self) -> list[dict[str, Any]]:
        rows = self.db.query(
            "SELECT id, title, status, progress, goal_id, updated_at, "
            "task_kind, artifact FROM projects ORDER BY updated_at DESC LIMIT 50"
        )
        return [
            {"id": r["id"], "title": r["title"], "status": r["status"],
             "progress": round(r["progress"] or 0.0, 3),
             "goal_id": r.get("goal_id", "") or "",
             "task_kind": r.get("task_kind", "") or "",
             "artifact": r.get("artifact", "") or ""}
            for r in rows
        ]

    def pause(self, project_id: str) -> Project:
        p = self._load(project_id)
        if p is None:
            raise KeyError(f"unknown project: {project_id}")
        p.status = "paused"
        p.updated_at = time.time()
        self._upsert_row(p)
        return p

    def resume(self, project_id: str) -> Project:
        p = self._load(project_id)
        if p is None:
            raise KeyError(f"unknown project: {project_id}")
        if p.status in {"done", "failed"}:
            return p
        p.status = "running"
        p.updated_at = time.time()
        self._upsert_row(p)
        return p

    # ── self-healing (wave 62) ─────────────────────────────────────────────
    def heal(self, project_id: str) -> dict[str, Any]:
        """Revive a FAILED project using what the system has learned.

        Escalating self-heal (waves 62-63):

        * **retry** — every exhausted step is reworded with the failure
          analyzer's prevention lessons, attempt counters reset, resume.
        * **replan** — when the linked goal has already been healed
          ``autonomy.replan_after_heals`` times (default 2), a reword is
          unlikely to help: the project is RE-PLANNED with failure context
          (what was tried + what was learned), keeping done steps.

        A linked goal paused by the failure is re-activated either way.
        A completed project is left alone; a project with nothing to heal
        is reported as such.
        """
        p = self._load(project_id)
        if p is None:
            return {"ok": False, "error": f"unknown project {project_id!r}"}
        if p.status in {"done", "planning"}:
            return {"ok": False, "error": f"project is {p.status}, nothing to heal"}
        failed = [s for s in p.steps if s.status == "failed"]
        if not failed:
            # not failed as such (e.g. paused mid-run): just let it continue
            p.status = "running"
            p.updated_at = time.time()
            self._upsert_row(p)
            self._heal_goal(p)
            return {"ok": True, "project_id": project_id, "status": p.status,
                    "mode": "resume", "healed_steps": 0, "resuming": True}
        # escalation: enough failed retries already? replan instead
        if self._should_replan(p):
            result = self.replan(project_id)
            if result.get("ok"):
                self._heal_goal(p)
                result["mode"] = "replan"
                result["resuming"] = True
                return result
            _log.warning("heal: replan fell back to retry (%s)",
                         result.get("error"))
        healed = 0
        for step in failed:
            revised = self._revise_step(step.description, step.result or "failed")
            step.description = revised
            step.status = "pending"
            step.attempts = 0
            step.result = ""
            step.revised = True
            healed += 1
        p.status = "running"
        p.finished_at = None
        p.report = (f"Self-healed: {healed} failed step(s) reworded with "
                    f"learned failure lessons; resuming.")
        p.updated_at = time.time()
        self._upsert_row(p)
        self._heal_goal(p)
        return {"ok": True, "project_id": project_id, "status": p.status,
                "mode": "retry", "healed_steps": healed, "resuming": True}

    def _should_replan(self, p: Project) -> bool:
        """True when the linked goal has burned enough retries that a fresh
        plan (with failure context) is the better next move."""
        threshold = int(getattr(getattr(self.context.settings, "autonomy", None),
                                "replan_after_heals", 2) or 0)
        if threshold <= 0 or not p.goal_id:
            return False
        try:
            row = self.db.query_one(
                "SELECT heals FROM agent_goals WHERE id=?", (p.goal_id,))
            return bool(row) and int(row.get("heals", 0) or 0) >= threshold
        except Exception:  # noqa: BLE001
            return False


    def _heal_goal(self, p: Project) -> None:
        """Re-activate the linked goal when its project comes back to life."""
        if not p.goal_id or p.status != "running":
            return
        try:
            row = self.db.query_one("SELECT id, status FROM agent_goals WHERE id=?",
                                    (p.goal_id,))
            if row and row.get("status") == "paused":
                self.db.execute(
                    "UPDATE agent_goals SET status='active', updated_at=? WHERE id=?",
                    (time.time(), p.goal_id))
        except Exception as exc:  # noqa: BLE001 — goal sync is best-effort
            _log.debug("goal re-activation failed: %s", exc)


# ── helpers ────────────────────────────────────────────────────────────────────

def _split_clauses(text: str) -> list[str]:
    import re
    parts = re.split(r"(?<=[.!?])\s+|\s+(?:and|then|next|after that|finally)\s+|\n",
                     text.strip())
    return parts


def _extract_json_array(text: str) -> list[Any]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except (ValueError, TypeError):
        return []
    return data if isinstance(data, list) else []


# ── tool registration ──────────────────────────────────────────────────────────

def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "project",
        description=("Autonomous multi-step project mode: self-plans, self-corrects "
                     "and reports. action=create (start), plan (break into steps), "
                     "advance (do next step), run (autonomously execute), report "
                     "(progress), list, status, pause, resume, heal (revive a "
                     "failed project with learned failure lessons, escalating "
                     "to replan), replan (fresh plan with failure context)."),
        capability="model.call",
        parameters={
            "action": "str — create|plan|advance|run|report|list|status|pause|resume|heal|replan",
            "project_id": "str (for most actions)",
            "title": "str (create)",
            "objective": "str (create)",
            "steps_json": "json str [..] (create with explicit steps)",
            "max_steps": "str (run)",
        },
    )
    def project(
        action: str = "list", *, project_id: str = "", title: str = "",
        objective: str = "", steps_json: str = "", max_steps: str = "",
    ) -> dict[str, Any]:
        mgr = ProjectManager(context)
        action = (action or "list").strip().lower()
        if action == "list":
            return {"ok": True, "projects": mgr.list_projects()}
        if action == "create":
            steps = None
            if steps_json:
                try:
                    steps = json.loads(steps_json)
                except (ValueError, TypeError):
                    return {"ok": False, "error": "steps_json must be a JSON array"}
            p = mgr.create(title or objective or "project", objective=objective,
                           steps=steps if isinstance(steps, list) else None)
            return {"ok": True, "id": p.id, "status": p.status,
                    "steps": len(p.steps)}
        if not project_id:
            return {"ok": False, "error": "project_id required"}
        try:
            if action == "plan":
                p = mgr.plan(project_id)
                return {"ok": True, "id": p.id, "status": p.status,
                        "steps": [s.description for s in p.steps]}
            if action == "advance":
                p = mgr.advance(project_id)
                return {"ok": True, "id": p.id, "status": p.status,
                        "progress": round(p.progress, 3)}
            if action == "run":
                ms = int(max_steps) if str(max_steps).isdigit() else None
                return mgr.run(project_id, max_steps=ms)
            if action == "report":
                return mgr.report(project_id)
            if action == "status":
                return mgr.status(project_id)
            if action == "pause":
                p = mgr.pause(project_id)
                return {"ok": True, "id": p.id, "status": p.status}
            if action == "resume":
                p = mgr.resume(project_id)
                return {"ok": True, "id": p.id, "status": p.status}
            if action == "heal":
                return mgr.heal(project_id)
            if action == "replan":
                return mgr.replan(project_id)
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": False, "error": f"unknown action: {action}"}
