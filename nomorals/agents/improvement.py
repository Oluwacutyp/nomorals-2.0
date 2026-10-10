"""Closed-loop self-improvement — the system improves itself, verified.

One cycle:

  1. **benchmark** the live system (bounded)
  2. **detect the weakest** measurable dimension
  3. if it is already at/above ``target``, skip (no churn)
  4. **map the weakness to a concrete lever** — the exact prompt/code that
     governs that dimension (reasoning→reasoning prompts, planning→
     orchestrator planner, tool_use→devon planner, self_correction→coding
     prompt)
  5. **propose** a surgical edit via the EvolutionAgent, seeded with the
     specific failing benchmark tasks as evidence
  6. **apply** it (autonomous mode) or hold it for owner approval
     (approval mode).  Applying runs the full gate: tests AND the benchmark
     regression check — a change that makes the system dumber is reverted.
  7. **re-benchmark** the same dimension to verify REAL improvement
  8. **record the full history** (before/after, delta, status) so the loop
     is continuous, auditable, and self-correcting: a dimension whose edit
     regressed is marked, and the loop won't blindly retry the same edit.

The loop is continuous (``tick`` / scheduler hook) and self-correcting
(reverts on regression, remembers what already failed).  It is gated by
``settings.improvement.mode`` — off by default; ``approval`` proposes and
waits for the owner; ``autonomous`` applies (always through the gate).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..storage.kv import KVStore
from .benchmark import measurable, run_benchmark

_log = get_logger(__name__)

__all__ = ["ImprovementLoop", "CycleRecord", "LEVERS", "register"]


# ── the levers: which concrete code improves which dimension ────────────────
# Each lever names the file that governs the dimension and the focused
# instruction handed to the evolver. The evolver makes a *surgical* edit;
# the gate verifies it.
LEVERS: dict[str, dict[str, str]] = {
    "reasoning": {
        "file": "nomorals/agents/reasoning.py",
        "instruction": (
            "Improve the reasoning engine's step-parsing and strategy "
            "prompts so it extracts cleaner, higher-confidence step chains "
            "and better-decomposes multi-part problems. Strengthen the "
            "REASONING:/ANSWER:/CONFIDENCE: parsing to tolerate more "
            "formats. Do not change the public API or the trace schema."
        ),
    },
    "planning": {
        "file": "nomorals/agents/orchestrator.py",
        "instruction": (
            "Improve the orchestrator planner so it emits more specific, "
            "correctly-ordered steps with valid depends_on. Strengthen the "
            "planner prompt to require a verify step that depends on the "
            "execution step, and make _repair catch more invalid plans. Do "
            "not change the Plan/PlanStep dataclasses."
        ),
    },
    "tool_use": {
        "file": "nomorals/agents/devon.py",
        "instruction": (
            "Improve Devon's planner prompt and TOOL_CATALOG descriptions so "
            "he picks valid tools that exist, supplies every required "
            "argument, and orders calls so dependencies run first. Tighten "
            "the catalog descriptions to name required args explicitly. Do "
            "not remove any tools."
        ),
    },
    "self_correction": {
        "file": "nomorals/agents/coding.py",
        "instruction": (
            "Improve the coding agent's draft prompt so it writes code that "
            "runs on the first attempt: guard empty/None inputs, use "
            "defined names, and handle the edge cases the task implies. "
            "Strengthen the failure->fix prompt to address the exact error "
            "shown. Do not change the sandbox or the run loop."
        ),
    },
}


@dataclass
class CycleRecord:
    id: str
    dimension: str
    before_score: Optional[float]
    after_score: Optional[float]
    delta: Optional[float]
    proposal_id: str
    action: str
    rationale: str
    edit_summary: str
    status: str
    created_at: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "dimension": self.dimension,
            "before_score": self.before_score,
            "after_score": self.after_score, "delta": self.delta,
            "proposal_id": self.proposal_id, "action": self.action,
            "rationale": self.rationale, "edit_summary": self.edit_summary,
            "status": self.status, "created_at": self.created_at,
            "details": self.details,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CycleRecord":
        try:
            details = json.loads(row.get("details_json") or "{}")
        except Exception:  # noqa: BLE001
            details = {}
        return cls(
            id=row["id"], dimension=row.get("dimension", ""),
            before_score=row.get("before_score"),
            after_score=row.get("after_score"), delta=row.get("delta"),
            proposal_id=row.get("proposal_id", ""), action=row.get("action", ""),
            rationale=row.get("rationale", ""),
            edit_summary=row.get("edit_summary", ""),
            status=row.get("status", "pending"),
            created_at=float(row.get("created_at", 0)), details=details)


class ImprovementLoop:
    def __init__(self, context: Any) -> None:
        self.context = context
        self.db = context.db

    # ── config ─────────────────────────────────────────────────────────────
    @property
    def settings(self):
        return self.context.settings.improvement

    # ── core cycle ──────────────────────────────────────────────────────────
    def run_cycle(self, mode: str = "", *, dimension: str = "") -> CycleRecord:
        """Run one closed-loop cycle. ``mode`` defaults to the configured
        mode; ``dimension`` optionally forces which dimension to target."""
        mode = (mode or self.settings.mode or "off").strip().lower()
        cycle_id = new_short_id("improve")
        if mode == "off":
            return self._record(cycle_id, dimension, None, None, None, "",
                                "off", "improvement is off", "",
                                "skipped", {"reason": "mode off"})

        if not measurable(self.context):
            return self._record(cycle_id, dimension, None, None, None, "",
                                "unmeasurable",
                                "active provider cannot be measured "
                                "(mock/offline)", "", "skipped",
                                {"reason": "unmeasurable"})

        # 1. benchmark (bounded) — or measure just the forced dimension
        dims = [dimension] if dimension and dimension in LEVERS else None
        report = run_benchmark(self.context, dimensions=dims, limit=2)
        measured = {k: v for k, v in report.scores.items()
                    if v.score is not None}
        if not measured:
            return self._record(cycle_id, dimension, None, None, None, "",
                                "unmeasurable", "no measurable dimensions",
                                "", "skipped", {"reason": "no measurable dim"})

        # 2. weakest dimension
        weakest = min(measured, key=lambda k: measured[k])
        before = measured[weakest]
        if dimension and dimension in LEVERS:
            weakest = dimension
            before = measured.get(weakest, before)

        # 3. at/above target -> no churn
        if before >= self.settings.target:
            return self._record(
                cycle_id, weakest, before, before, 0.0, "", "meets_target",
                f"{weakest} already at {before:.2f} >= target "
                f"{self.settings.target:.2f}", "", "skipped",
                {"reason": "meets_target"})

        lever = LEVERS[weakest]
        rationale = self._weakness_evidence(report, weakest)

        # self-correcting: don't retry an edit that just regressed
        if self._recently_regressed(weakest):
            return self._record(
                cycle_id, weakest, before, None, None, "", "cooldown",
                f"{weakest}: a recent edit regressed; cooling down", "",
                "skipped", {"reason": "recent_regression"})

        # 5. propose via the evolver
        instruction = (
            f"{lever['instruction']}\n\n"
            f"EVIDENCE (benchmark tasks currently failing for "
            f"{weakest}):\n{rationale}\n\n"
            "Make the smallest edit that addresses this evidence."
        )
        proposal_id = ""
        edit_summary = ""
        try:
            from .evolution import EvolutionAgent
            evolver = EvolutionAgent(self.context)
            proposal = evolver.plan(instruction, focus=lever["file"])
            proposal_id = proposal.id
            edit_summary = "; ".join(
                e.get("path", "") for e in proposal.edits)[:400]
        except Exception as exc:  # noqa: BLE001 — a bad proposal is a cycle
            return self._record(
                cycle_id, weakest, before, None, None, proposal_id,
                "proposal_failed", f"could not plan an edit: {exc}",
                "", "skipped", {"error": str(exc)[:200]})

        # 6. approval mode: hold for the owner
        if mode == "approval":
            return self._record(
                cycle_id, weakest, before, None, None, proposal_id,
                "proposed",
                f"improve {weakest} ({before:.2f}) — awaiting approval",
                edit_summary, "pending",
                {"proposal": proposal_id, "lever_file": lever["file"]})

        # 7. autonomous: apply (runs the full gate), then re-benchmark
        after = before
        status = "neutral"
        action = "applied"
        details: dict[str, Any] = {"proposal": proposal_id,
                                   "lever_file": lever["file"]}
        try:
            out = evolver.apply(proposal_id)
            action = out.get("status", "applied")
            if not out.get("applied"):
                # the gate reverted it (tests or benchmark regression)
                return self._record(
                    cycle_id, weakest, before, None, None, proposal_id,
                    "reverted", out.get("reason", "gate reverted the edit"),
                    edit_summary, "regressed", details)
            # 7b. re-benchmark the same dimension to verify REAL improvement
            rerun = run_benchmark(self.context, dimensions=[weakest], limit=2)
            after_dim = rerun.scores.get(weakest)
            if after_dim is not None and after_dim.score is not None:
                after = after_dim.score
            else:
                after = before
            delta = after - before
            if delta > self.settings.tolerance:
                status = "improved"
            elif delta < -self.settings.tolerance:
                status = "regressed"
            else:
                status = "neutral"
            details["after_provider"] = rerun.provider
        except Exception as exc:  # noqa: BLE001
            return self._record(
                cycle_id, weakest, before, None, None, proposal_id,
                "apply_failed", f"apply raised: {exc}", edit_summary,
                "regressed", {**details, "error": str(exc)[:200]})
        return self._record(cycle_id, weakest, before, after,
                            (after - before) if after is not None else None,
                            proposal_id, action,
                            f"{weakest}: {before:.2f} -> "
                            f"{after if after is not None else '?'}",
                            edit_summary, status, details)

    # ── helpers ─────────────────────────────────────────────────────────────
    def _weakness_evidence(self, report: Any, dimension: str) -> str:
        dim = report.scores.get(dimension)
        if dim is None:
            return "(no details)"
        lines = []
        for d in dim.details[:6]:
            mark = "PASS" if d.get("pass") else "FAIL"
            lines.append(f"  [{mark}] {d.get('task', '')}: {d.get('detail', '')}")
        return "\n".join(lines) or "(no task details)"

    def _recently_regressed(self, dimension: str, *, window_h: float = 24.0
                            ) -> bool:
        cutoff = time.time() - window_h * 3600
        row = self.db.query_one(
            "SELECT 1 FROM improvement_runs WHERE dimension=? AND status="
            "'regressed' AND created_at > ? LIMIT 1", (dimension, cutoff))
        return row is not None

    def _record(self, cycle_id: str, dimension: str,
                before: Optional[float], after: Optional[float],
                delta: Optional[float], proposal_id: str, action: str,
                rationale: str, edit_summary: str, status: str,
                details: dict[str, Any]) -> CycleRecord:
        dimension = dimension or ""  # NOT NULL column; empty = auto-selected
        rec = CycleRecord(
            id=cycle_id, dimension=dimension, before_score=before,
            after_score=after, delta=delta, proposal_id=proposal_id,
            action=action, rationale=rationale[:500],
            edit_summary=edit_summary[:400], status=status,
            created_at=time.time(), details=details)
        self.db.execute(
            "INSERT INTO improvement_runs (id, dimension, before_score, "
            "after_score, delta, proposal_id, action, rationale, "
            "edit_summary, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (cycle_id, dimension, before, after, delta, proposal_id, action,
             rec.rationale, rec.edit_summary, status, rec.created_at))
        # stash full details in kv so history can show them
        try:
            KVStore(self.db).set_raw(
                f"improvement.details.{cycle_id}",
                json.dumps(details, default=str), "json")
        except Exception:  # noqa: BLE001
            pass
        _log.info("improvement cycle [%s] dim=%s %s -> %s", action, dimension,
                  before, status)
        return rec

    # ── approval flow ───────────────────────────────────────────────────────
    def pending_approvals(self) -> list[CycleRecord]:
        rows = self.db.query(
            "SELECT * FROM improvement_runs WHERE status='pending' ORDER BY "
            "created_at DESC LIMIT 50")
        return [CycleRecord.from_row(r) for r in rows]

    def approve(self, cycle_id: str) -> dict[str, Any]:
        """Apply a held (approval-mode) cycle now, then re-benchmark."""
        row = self.db.query_one(
            "SELECT * FROM improvement_runs WHERE id=?", (cycle_id,))
        if not row or row["status"] != "pending":
            return {"ok": False, "error": "no pending cycle with that id"}
        proposal_id = row["proposal_id"]
        dimension = row["dimension"]
        before = row["before_score"]
        try:
            from .evolution import EvolutionAgent
            evolver = EvolutionAgent(self.context)
            out = evolver.apply(proposal_id)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:200]}
        if not out.get("applied"):
            self.db.execute(
                "UPDATE improvement_runs SET status='regressed', action="
                "'reverted', rationale=? WHERE id=?",
                (out.get("reason", "reverted")[:500], cycle_id))
            return {"ok": False, "error": out.get("reason", "reverted")}
        rerun = run_benchmark(self.context, dimensions=[dimension], limit=2)
        dim = rerun.scores.get(dimension)
        after = dim.score if (dim is not None and dim.score is not None) else before
        if after is not None and before is not None:
            delta = after - before
        else:
            delta = None
        if delta is not None and delta > self.settings.tolerance:
            status = "improved"
        elif delta is not None and delta < -self.settings.tolerance:
            status = "regressed"
        else:
            status = "neutral"
        self.db.execute(
            "UPDATE improvement_runs SET after_score=?, delta=?, status=? "
            "WHERE id=?", (after, delta, status, cycle_id))
        return {"ok": True, "status": status, "before": before,
                "after": after, "delta": delta}

    def deny(self, cycle_id: str) -> bool:
        row = self.db.query_one(
            "SELECT * FROM improvement_runs WHERE id=? AND status='pending'",
            (cycle_id,))
        if not row:
            return False
        self.db.execute("UPDATE improvement_runs SET status='skipped', "
                        "action='denied' WHERE id=?", (cycle_id,))
        return True

    # ── history / status ────────────────────────────────────────────────────
    def history(self, *, limit: int = 20) -> list[CycleRecord]:
        rows = self.db.query(
            "SELECT * FROM improvement_runs ORDER BY created_at DESC LIMIT ?",
            (limit,))
        out = []
        for r in rows:
            rec = CycleRecord.from_row(r)
            try:
                rec.details = KVStore(self.db).get(
                    f"improvement.details.{rec.id}", default={})
            except Exception:  # noqa: BLE001
                pass
            out.append(rec)
        return out

    def status(self) -> dict[str, Any]:
        report = run_benchmark(self.context, limit=1)
        measured = {k: round(v.score, 3) for k, v in report.scores.items()
                    if v.score is not None}
        weakest = min(measured, key=lambda k: measured[k]) if measured else None
        recent = self.history(limit=10)
        return {
            "mode": self.settings.mode,
            "target": self.settings.target,
            "measurable": report.measurable,
            "scores": measured,
            "weakest": weakest,
            "weakest_score": measured.get(weakest) if weakest else None,
            "pending_approvals": len(self.pending_approvals()),
            "recent": [r.to_dict() for r in recent],
        }

    # ── continuous ──────────────────────────────────────────────────────────
    def tick(self) -> list[CycleRecord]:
        """Run up to ``max_cycles`` cycles (for the scheduler / autonomous
        mode).  Stops early once every measurable dimension meets target."""
        out: list[CycleRecord] = []
        if self.settings.mode == "off":
            return out
        for _ in range(max(1, self.settings.max_cycles)):
            cycle = self.run_cycle()
            out.append(cycle)
            if cycle.status == "skipped" and cycle.action in (
                    "meets_target", "unmeasurable", "off"):
                break
            if cycle.status == "improved":
                continue
            if cycle.status in ("regressed", "skipped"):
                # give the loop a chance to try a different dimension next
                continue
        return out


# ── registry ────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "improve",
        description=(
            "Closed-loop self-improvement: benchmark -> weakest dimension -> "
            "propose a surgical edit -> apply (gated) -> re-benchmark -> "
            "verify -> history. action=cycle | status | history | "
            "pending | approve <id> | deny <id>. mode: off|approval|autonomous."
        ),
        capability="model.call",
        parameters={
            "action": "str — cycle|status|history|pending|approve|deny",
            "mode": "str — off|approval|autonomous (for cycle)",
            "dimension": "str — force a dimension (for cycle)",
            "cycle_id": "str — for approve/deny",
            "limit": "int — for history",
        },
    )
    def improve(
        action: str = "status", *, mode: str = "", dimension: str = "",
        cycle_id: str = "", limit: str = "10",
    ) -> dict[str, Any]:
        loop = ImprovementLoop(context)
        action = (action or "status").strip().lower()
        if action == "cycle":
            rec = loop.run_cycle(mode=mode, dimension=dimension)
            return {"ok": True, "cycle": rec.to_dict()}
        if action == "history":
            try:
                n = int(limit or 10)
            except ValueError:
                n = 10
            return {"cycles": [r.to_dict() for r in loop.history(limit=n)]}
        if action == "pending":
            return {"pending": [r.to_dict() for r in loop.pending_approvals()]}
        if action == "approve":
            return loop.approve(cycle_id)
        if action == "deny":
            return {"ok": loop.deny(cycle_id)}
        return loop.status()
