"""Upgrade-proposal suggestion pipeline (owner approve/deny -> implement -> tests).

Research/digest findings arrive as plain-data *tickets*; the owner
explicitly approves or denies each one.  An approved proposal is handed
to :class:`EvolutionAgent` — the single tested apply path (plan ->
write edits -> import smoke -> full test suite, with git rollback on
failure).  This module never edits the tree itself.

States: ``proposed -> approved|denied`` ; approved proposals become
``implemented`` or ``failed`` once the evolution agent reports back.
"""
from __future__ import annotations

import json
import time
from typing import Any

from ..core.errors import ToolError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .evolution import EvolutionAgent
from .notifier import notify

_log = get_logger(__name__)

__all__ = ["UpgradeQueue", "UpgradePipeline", "register"]

_JSON_COLUMNS = ("patch_plan", "files", "tests", "claim_ids", "applied_result")

# Ticket confidence below this never reaches the queue — weak findings
# would just spam the owner.  This is explicit (a warning is logged).
_MIN_TICKET_CONFIDENCE = 0.4


def _decode(row: dict[str, Any]) -> dict[str, Any]:
    """Decode the JSON columns of an upgrade_proposals row into a dict."""
    data = dict(row)
    for col in _JSON_COLUMNS:
        raw = data.get(col)
        if isinstance(raw, str):
            try:
                data[col] = json.loads(raw) if raw else ({} if col in ("patch_plan", "applied_result") else [])
            except (json.JSONDecodeError, ValueError):
                data[col] = {} if col in ("patch_plan", "applied_result") else []
        elif raw is None:
            data[col] = {} if col in ("patch_plan", "applied_result") else []
    # ``source`` has no dedicated column: it is stashed inside patch_plan
    # at write time and lifted back out here.
    plan = data.get("patch_plan")
    if isinstance(plan, dict):
        data["source"] = plan.pop("source", "")
    else:
        data["source"] = ""
    return data


class UpgradeQueue:
    """Owner-gated store of proposed framework upgrades."""

    def __init__(self, context: Any):
        self.context = context
        self.db = context.db

    # ------------------------------------------------------------------ api

    def propose(
        self,
        *,
        title: str,
        rationale: str,
        patch_plan: dict,
        files: list[str],
        tests: list[str],
        claim_ids: list[str] | None = None,
        source: str = "",
    ) -> str:
        """File a new upgrade proposal. Returns the proposal id.

        Fail-fast validation: short titles/rationales raise ValueError so
        the queue never accumulates junk.
        """
        title = (title or "").strip()
        rationale = (rationale or "").strip()
        if len(title) < 8:
            raise ValueError("title must be at least 8 characters")
        if len(rationale) < 20:
            raise ValueError("rationale must be at least 20 characters")
        if not isinstance(patch_plan, dict):
            raise ValueError("patch_plan must be a dict")
        files = [f for f in (files or []) if f]
        tests = [t for t in (tests or []) if t]

        plan = dict(patch_plan)
        if source:
            plan["source"] = source
        pid = new_id("upg")
        self.db.execute(
            "INSERT INTO upgrade_proposals "
            "(id, title, rationale, patch_plan, files, tests, claim_ids, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed', ?)",
            (pid, title, rationale, json.dumps(plan),
             json.dumps(files), json.dumps(tests),
             json.dumps(claim_ids or []), time.time()),
        )
        body = rationale[:400]
        if source:
            body = f"[source: {source}]\n{body}"
        notify(self.context, "upgrade_proposal",
               f"new upgrade proposal: {title}", body)
        return pid

    def list(self, status: str = "proposed", limit: int = 50) -> list[dict]:
        rows = self.db.query(
            "SELECT * FROM upgrade_proposals WHERE status = ? "
            "ORDER BY created_at DESC LIMIT ?",
            (status, limit),
        )
        return [_decode(r) for r in (rows or [])]

    def get(self, proposal_id: str) -> dict | None:
        row = self.db.query_one(
            "SELECT * FROM upgrade_proposals WHERE id = ?", (proposal_id,))
        return _decode(row) if row else None

    def approve(self, proposal_id: str, by: str = "owner") -> dict:
        """Move a ``proposed`` proposal to ``approved``."""
        row = self._require(proposal_id, "proposed")
        self.db.execute(
            "UPDATE upgrade_proposals SET status = 'approved', "
            "decided_at = ?, decided_by = ? WHERE id = ?",
            (time.time(), by, proposal_id),
        )
        notify(self.context, "upgrade_proposal",
               f"upgrade approved: {row['title']}",
               f"proposal_id={proposal_id} decided_by={by} — "
               "dispatching to the evolution agent for implementation.")
        return self.get(proposal_id) or {}

    def deny(self, proposal_id: str, reason: str, by: str = "owner") -> dict:
        """Move a ``proposed`` proposal to ``denied``. A reason is required."""
        reason = (reason or "").strip()
        if not reason:
            raise ValueError("a denial reason is required")
        row = self._require(proposal_id, "proposed")
        self.db.execute(
            "UPDATE upgrade_proposals SET status = 'denied', reason = ?, "
            "decided_at = ?, decided_by = ? WHERE id = ?",
            (reason, time.time(), by, proposal_id),
        )
        notify(self.context, "upgrade_proposal",
               f"upgrade denied: {row['title']}",
               f"proposal_id={proposal_id} decided_by={by}\nReason: {reason}")
        return self.get(proposal_id) or {}

    def record_implemented(self, proposal_id: str, result: dict) -> dict:
        """Record the outcome of an approved proposal's implementation."""
        result = dict(result or {})
        ok = result.get("ok") is not False
        status = "implemented" if ok else "failed"
        evo_id = str(result.get("proposal_id") or result.get("evolution_proposal_id") or "")
        self.db.execute(
            "UPDATE upgrade_proposals SET status = ?, applied_result = ?, "
            "evolution_proposal_id = ? WHERE id = ?",
            (status, json.dumps(result), evo_id, proposal_id),
        )
        notify(self.context, "upgrade_proposal",
               f"upgrade {status}: {proposal_id}",
               f"evolution_proposal_id={evo_id}\n{json.dumps(result)[:400]}")
        return self.get(proposal_id) or {}

    # --------------------------------------------------------------- helpers

    def _require(self, proposal_id: str, expected_status: str) -> dict:
        row = self.get(proposal_id)
        if row is None:
            raise ToolError(f"unknown upgrade proposal: {proposal_id}")
        if row.get("status") != expected_status:
            raise ToolError(
                f"proposal {proposal_id} is '{row.get('status')}', "
                f"expected '{expected_status}'")
        return row


class UpgradePipeline:
    """Ticket -> queue -> owner decision -> EvolutionAgent implementation."""

    def __init__(self, context: Any,
                 evolution: EvolutionAgent | None = None):
        self.context = context
        self._evolution = evolution

    @property
    def evolution(self) -> EvolutionAgent:
        if self._evolution is None:
            self._evolution = EvolutionAgent(self.context)
        return self._evolution

    # ------------------------------------------------------------------ api

    def propose_from_ticket(self, ticket: dict, *, source: str = "research") -> str:
        """Turn a digest-stream ticket into a queued proposal.

        Returns the proposal id, or ``""`` when the ticket's confidence is
        too low to bother the owner with (a warning is logged).
        """
        ticket = ticket or {}
        confidence = float(ticket.get("confidence") or 0.0)
        if confidence < _MIN_TICKET_CONFIDENCE:
            _log.warning(
                "upgrade ticket rejected: confidence %.2f < %.2f (%s)",
                confidence, _MIN_TICKET_CONFIDENCE,
                ticket.get("title") or "<untitled>")
            return ""
        queue = UpgradeQueue(self.context)
        raw_plan = ticket.get("patch_plan") or {}
        # digest tickets carry patch_plan as an ordered step list; the
        # queue stores a dict — normalize, never crash on the shape.
        patch_plan = (raw_plan if isinstance(raw_plan, dict)
                      else {"steps": list(raw_plan)})
        return queue.propose(
            title=str(ticket.get("title") or ""),
            rationale=str(ticket.get("rationale") or ""),
            patch_plan=patch_plan,
            files=list(ticket.get("suggested_files") or []),
            tests=list(ticket.get("test_plan") or []),
            claim_ids=list(ticket.get("claim_ids") or []),
            source=source,
        )

    def approve_and_implement(
        self, proposal_id: str, by: str = "owner", *, verify: bool = True,
    ) -> dict:
        """Approve a proposal and implement it via the EvolutionAgent.

        When the proposal references an existing evolution proposal
        (``patch_plan.evolution_proposal_id`` — filed by ``evolve_plan``)
        it is applied directly instead of re-planning.  When it
        references a staged skill edit (``patch_plan.skill_edit_id`` —
        filed by the skill loop in approval mode) the staged edit is
        committed.  Otherwise a fresh evolution plan is generated from
        the ticket, as before.

        On any exception the proposal is recorded as ``failed`` with the
        error text, then the exception is re-raised so the owner sees the
        failure instead of a quiet log line.
        """
        queue = UpgradeQueue(self.context)
        approved = queue.approve(proposal_id, by=by)
        plan = approved.get("patch_plan") or {}
        evo_id = plan.get("evolution_proposal_id") if isinstance(plan, dict) else ""
        skill_edit_id = plan.get("skill_edit_id") if isinstance(plan, dict) else ""
        try:
            if evo_id:
                result = self.evolution.apply(str(evo_id), verify=verify,
                                             commit=True)
                result = dict(result or {})
                result.setdefault("proposal_id", str(evo_id))
                return queue.record_implemented(proposal_id, result)
            if skill_edit_id:
                from .skill_evolution import SkillEvolutionLoop

                result = SkillEvolutionLoop(
                    self.context).apply_staged_edit(str(skill_edit_id))
                return queue.record_implemented(proposal_id, result)
            instruction = _build_instruction(approved)
            evo_proposal = self.evolution.plan(instruction)
            evo_new_id = getattr(evo_proposal, "id", None) or str(
                (evo_proposal or {}).get("id") or "")
            result = self.evolution.apply(evo_new_id, verify=verify)
            result = dict(result or {})
            result.setdefault("proposal_id", evo_new_id)
            return queue.record_implemented(proposal_id, result)
        except Exception as exc:  # noqa: BLE001 — recorded, then re-raised
            queue.record_implemented(
                proposal_id,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            raise

    def deny_with_reason(self, proposal_id: str, reason: str,
                         by: str = "owner") -> dict:
        """Deny a proposal; when it references an evolution proposal or a
        staged skill edit, mark that side rejected/denied too so nothing
        lingers in a plannable/stageable limbo."""
        queue = UpgradeQueue(self.context)
        denied = queue.deny(proposal_id, reason, by=by)
        plan = denied.get("patch_plan") or {}
        if isinstance(plan, dict):
            evo_id = plan.get("evolution_proposal_id")
            if evo_id:
                try:
                    self.evolution.reject(str(evo_id))
                except Exception:  # noqa: BLE001 - the deny itself stands
                    _log.debug("could not reject evolution proposal %s",
                               evo_id, exc_info=True)
            skill_edit_id = plan.get("skill_edit_id")
            if skill_edit_id:
                try:
                    from .skill_evolution import SkillEvolutionLoop

                    SkillEvolutionLoop(
                        self.context).deny_staged_edit(str(skill_edit_id))
                except Exception:  # noqa: BLE001 - the deny itself stands
                    _log.debug("could not deny staged skill edit %s",
                               skill_edit_id, exc_info=True)
        return denied


def _build_instruction(proposal: dict) -> str:
    """Compose the evolution instruction from an approved proposal."""
    plan = proposal.get("patch_plan") or {}
    lines = [
        f"Title: {proposal.get('title')}",
        f"Rationale: {proposal.get('rationale')}",
    ]
    files = proposal.get("files") or []
    if files:
        lines.append("Target files:")
        lines.extend(f"  - {f}" for f in files)
    tests = proposal.get("tests") or []
    if tests:
        lines.append("Acceptance tests:")
        lines.extend(f"  - {t}" for t in tests)
    if isinstance(plan, dict) and plan:
        lines.append("Patch plan:")
        lines.append(json.dumps(plan, indent=2))
    return "\n".join(lines)


def register(registry: Any) -> None:
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "upgrade_queue",
        description=(
            "Upgrade-proposal pipeline: file, list, approve or deny framework "
            "upgrade suggestions. Approving dispatches the proposal to the "
            "evolution agent for implementation and testing. "
            "Actions: propose | list | approve | deny | get."
        ),
        capability=Capability.SYS_CONFIG,
        parameters={
            "action": "str — propose | list | approve | deny | get",
            "id": "str (optional) — proposal id for approve/deny/get",
            "title": "str (optional) — required for propose, >=8 chars",
            "rationale": "str (optional) — required for propose, >=20 chars",
            "patch_plan": "dict (optional) — structured edit plan",
            "files": "list (optional) — target file paths",
            "tests": "list (optional) — acceptance test descriptions",
            "reason": "str (optional) — required for deny",
            "by": "str (optional) — who decided, default 'owner'",
            "status": "str (optional) — filter for list, default 'proposed'",
            "source": "str (optional) — proposal origin, default 'research'",
        },
    )
    def upgrade_queue(
        action: str = "list",
        id: str = "",
        title: str = "",
        rationale: str = "",
        patch_plan: Any = None,
        files: Any = None,
        tests: Any = None,
        reason: str = "",
        by: str = "owner",
        status: str = "proposed",
        source: str = "research",
        **_: Any,
    ) -> dict[str, Any]:
        pipeline = UpgradePipeline(context)
        queue = UpgradeQueue(context)
        action = (action or "list").lower()

        if action == "propose":
            pid = queue.propose(
                title=title, rationale=rationale,
                patch_plan=dict(patch_plan or {}),
                files=list(files or []), tests=list(tests or []),
                source=source,
            )
            return {"ok": True, "id": pid}
        if action == "list":
            return {"ok": True, "proposals": queue.list(status=status)}
        if action == "get":
            proposal = queue.get(id)
            if proposal is None:
                raise ToolError(f"unknown upgrade proposal: {id}")
            return {"ok": True, "proposal": proposal}
        if action == "approve":
            return {"ok": True, "proposal": pipeline.approve_and_implement(id, by=by)}
        if action == "deny":
            return {"ok": True,
                    "proposal": pipeline.deny_with_reason(id, reason, by=by)}
        raise ToolError(f"unknown action: {action}")
