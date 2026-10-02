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

import hashlib
import json
import time
from typing import Any

from ..core.errors import ToolError
from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .evolution import EvolutionAgent
from .notifier import notify
from .research_digest import gate_ticket
from .research_swarm import _NEGATION, _content_words, _same_claim

_log = get_logger(__name__)

__all__ = ["UpgradeQueue", "UpgradePipeline", "register"]

_JSON_COLUMNS = ("patch_plan", "files", "tests", "claim_ids", "applied_result")

# Bookkeeping stashed inside the patch_plan JSON payload (same trick as
# ``source``): dedup merge records and conflict notes travel with the
# proposal row and are lifted back out by _decode.
_BOOKKEEPING_KEYS = ("merged_duplicates", "conflict_notes")

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
    # at write time and lifted back out here.  The same goes for the
    # dedup/conflict bookkeeping (``_BOOKKEEPING_KEYS``).
    plan = data.get("patch_plan")
    if isinstance(plan, dict):
        data["source"] = plan.pop("source", "")
        for key in _BOOKKEEPING_KEYS:
            val = plan.pop(key, [])
            data[key] = list(val) if isinstance(val, list) else []
    else:
        data["source"] = ""
        for key in _BOOKKEEPING_KEYS:
            data[key] = []
    return data


def _norm_file(path: str) -> str:
    """Normalize a suggested file path for overlap comparison."""
    p = (path or "").strip()
    while p.startswith("./"):
        p = p[2:]
    return p


def _files_overlap(a: list[str], b: list[str]) -> list[str]:
    """Normalized suggested-file overlap between two proposals."""
    sa = {_norm_file(f) for f in (a or []) if _norm_file(f)}
    sb = {_norm_file(f) for f in (b or []) if _norm_file(f)}
    return sorted(sa & sb)


def _proposal_text(p: dict[str, Any]) -> str:
    return f"{p.get('title') or ''} {p.get('rationale') or ''}".strip()


def _proposal_fingerprint(title: str, rationale: str,
                          files: list[str], tests: list[str]) -> str:
    """Stable fingerprint of a proposal's substance (for idempotency:
    refiling the exact same proposal is a no-op)."""
    h = hashlib.sha256()
    h.update(title.strip().lower().encode("utf-8"))
    h.update(b"\x00")
    h.update(rationale.strip().lower().encode("utf-8"))
    h.update(b"\x00")
    h.update("\x00".join(sorted(_norm_file(f) for f in (files or []))
                          ).encode("utf-8"))
    h.update(b"\x00")
    h.update("\x00".join(sorted((t or "").strip().lower()
                                for t in (tests or []))).encode("utf-8"))
    return h.hexdigest()[:32]


def _is_near_duplicate(new: dict[str, Any],
                       existing: dict[str, Any]) -> bool:
    """Near-duplicate test for propose-time dedup — the same philosophy
    as ``research_swarm.dedupe_findings``: the same claim in different
    words (``_same_claim``, which never fires across an
    affirmation/negation boundary), plus overlapping suggested files."""
    if not _same_claim(_proposal_text(new), _proposal_text(existing)):
        return False
    return bool(_files_overlap(new.get("files") or [],
                               existing.get("files") or []))


def _is_conflict(new: dict[str, Any], existing: dict[str, Any]) -> bool:
    """Genuine-conflict test — the same guard logic as
    ``research_swarm._detect_conflicts``: the SAME subject (two shared
    content words, >= 4 letters) over overlapping suggested files, with
    exactly one side negating. Affirmations never conflict with
    affirmations; those are duplicates or unrelated."""
    tn, te = _proposal_text(new), _proposal_text(existing)
    shared = {w for w in (_content_words(tn) & _content_words(te))
              if len(w) >= 4}
    if len(shared) < 2:
        return False
    if not _files_overlap(new.get("files") or [],
                          existing.get("files") or []):
        return False
    return bool(_NEGATION.search(tn)) != bool(_NEGATION.search(te))


def _conflict_point(title_a: str, title_b: str,
                    shared_files: list[str]) -> str:
    files_txt = ", ".join(shared_files[:4])
    if len(shared_files) > 4:
        files_txt += f" (+{len(shared_files) - 4} more)"
    return (f"contradictory proposals touching the same files "
            f"({files_txt}): {title_a[:80]!r} vs {title_b[:80]!r}")


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

        Dedup/conflict handling (automatic, propose-time):
        * near-duplicate of an open proposal (same claim in different
          words + overlapping suggested files) — merged into the
          original: the new filing is attached as a merged duplicate with
          full provenance, extra files/tests are unioned in, and the
          ORIGINAL id is returned. Nothing is silently dropped.
        * genuine conflict with an open proposal (same subject + same
          files, exactly one side negating) — never merged, never
          last-write-wins: an explicit ``conflict_notes`` entry is
          attached to BOTH items and both stay open for the owner.
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
        claim_ids = [c for c in (claim_ids or []) if c]

        new = {"title": title, "rationale": rationale, "files": files,
               "tests": tests, "claim_ids": claim_ids, "source": source}
        pid = new_id("upg")
        fingerprint = _proposal_fingerprint(title, rationale, files, tests)
        now = time.time()

        open_props = sorted(
            self.list(status="proposed", limit=500),
            key=lambda p: p.get("created_at") or 0)
        target = None
        for existing in open_props:
            if _is_near_duplicate(new, existing):
                if _proposal_fingerprint(
                        existing.get("title") or "",
                        existing.get("rationale") or "",
                        existing.get("files") or [],
                        existing.get("tests") or []) == fingerprint:
                    return existing["id"]  # exact re-file: no-op
                if any(d.get("fingerprint") == fingerprint
                       for d in existing.get("merged_duplicates") or []):
                    return existing["id"]  # already merged: no double-merge
                target = existing
                break
        conflicts = [e for e in open_props if _is_conflict(new, e)]

        if target is not None:
            self._merge_duplicate(target, pid=pid, new=new,
                                  fingerprint=fingerprint, now=now,
                                  conflicts=conflicts)
            return target["id"]

        plan = dict(patch_plan)
        if source:
            plan["source"] = source
        note_list = [
            self._make_conflict_note(e, other_id=e["id"],
                                     other_title=e.get("title") or "",
                                     now=now,
                                     shared_files=_files_overlap(
                                         files, e.get("files") or []))
            for e in conflicts
        ]
        if note_list:
            plan["conflict_notes"] = note_list
        self.db.execute(
            "INSERT INTO upgrade_proposals "
            "(id, title, rationale, patch_plan, files, tests, claim_ids, "
            "status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'proposed', ?)",
            (pid, title, rationale, json.dumps(plan),
             json.dumps(files), json.dumps(tests),
             json.dumps(claim_ids), now),
        )
        for e in conflicts:
            self._attach_conflict_note(e["id"], other_id=pid,
                                       other_title=title, now=now)
        body = rationale[:400]
        if source:
            body = f"[source: {source}]\n{body}"
        if conflicts:
            names = ", ".join(
                f"{c['id']}: {str(c.get('title') or '')[:60]}"
                for c in conflicts[:4])
            body = f"{body}\n\nconflicts with: {names}"
        notify(self.context, "upgrade_proposal",
               f"new upgrade proposal: {title}", body)
        return pid

    # --------------------------------------------- dedup / conflict internals

    def _merge_duplicate(self, target: dict[str, Any], *, pid: str,
                         new: dict[str, Any], fingerprint: str,
                         now: float, conflicts: list[dict]) -> None:
        """Fold a near-duplicate filing into the original open proposal:
        provenance record appended, files/tests/claim_ids unioned, and
        conflict notes attached both ways (the survivor is still open,
        so its conflicts stay visible)."""
        merged = list(target.get("merged_duplicates") or [])
        merged.append({
            "id": pid,
            "title": new["title"],
            "rationale": new["rationale"],
            "source": new["source"],
            "merged_at": now,
            "files": list(new["files"]),
            "tests": list(new["tests"]),
            "claim_ids": list(new["claim_ids"]),
            "fingerprint": fingerprint,
        })
        files = list(target.get("files") or [])
        for f in new["files"]:
            if _norm_file(f) and all(
                    _norm_file(e) != _norm_file(f) for e in files):
                files.append(f)
        tests = list(target.get("tests") or [])
        for t in new["tests"]:
            if t and all((e or "").strip() != t.strip() for e in tests):
                tests.append(t)
        claim_ids = list(target.get("claim_ids") or [])
        for c in new["claim_ids"]:
            if c and c not in claim_ids:
                claim_ids.append(c)
        plan = dict(target.get("patch_plan") or {})
        plan["source"] = target.get("source") or ""
        plan["merged_duplicates"] = merged
        if target.get("conflict_notes"):
            plan["conflict_notes"] = list(target["conflict_notes"])
        self.db.execute(
            "UPDATE upgrade_proposals SET files = ?, tests = ?, "
            "claim_ids = ?, patch_plan = ? WHERE id = ?",
            (json.dumps(files), json.dumps(tests), json.dumps(claim_ids),
             json.dumps(plan), target["id"]),
        )
        for e in conflicts:
            self._attach_conflict_note(target["id"], other_id=e["id"],
                                       other_title=e.get("title") or "",
                                       now=now)
            self._attach_conflict_note(e["id"], other_id=target["id"],
                                       other_title=target.get("title") or "",
                                       now=now)
        _log.info("upgrade proposal %s merged into %s", pid, target["id"])
        notify(self.context, "upgrade_proposal",
               f"upgrade proposal merged: {new['title']}",
               f"filed as {pid}; near-duplicate of open proposal "
               f"{target['id']} — merged in with provenance instead of a "
               "new queue entry.")

    def _make_conflict_note(self, existing: dict[str, Any], *,
                            other_id: str, other_title: str,
                            now: float,
                            shared_files: list[str] | None = None) -> dict[str, Any]:
        return {
            "proposal_id": other_id,
            "title": other_title,
            "point": _conflict_point(other_title,
                                     existing.get("title") or "",
                                     shared_files or []),
            "noted_at": now,
        }

    def _attach_conflict_note(self, proposal_id: str, *,
                              other_id: str, other_title: str,
                              now: float) -> None:
        """Append a conflict note to one side of a conflicting pair."""
        row = self.get(proposal_id)
        if row is None:
            return
        if any(n.get("proposal_id") == other_id
               for n in row.get("conflict_notes") or []):
            return  # already noted: idempotent
        other = self.get(other_id)
        shared = _files_overlap(row.get("files") or [],
                                (other or {}).get("files") or [])
        notes = list(row.get("conflict_notes") or [])
        notes.append({
            "proposal_id": other_id,
            "title": other_title,
            "point": _conflict_point(other_title, row.get("title") or "",
                                     shared),
            "noted_at": now,
        })
        plan = dict(row.get("patch_plan") or {})
        plan["source"] = row.get("source") or ""
        plan["conflict_notes"] = notes
        if row.get("merged_duplicates"):
            plan["merged_duplicates"] = list(row["merged_duplicates"])
        self.db.execute(
            "UPDATE upgrade_proposals SET patch_plan = ? WHERE id = ?",
            (json.dumps(plan), proposal_id),
        )

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

        Returns the proposal id, or ``""`` when the ticket is too weak or
        too vague to bother the owner with (a warning is logged in both
        cases):
        * confidence below ``_MIN_TICKET_CONFIDENCE`` — weak finding.
        * ``gate_ticket`` finds blocking problems — vague ticket (no real
          files, no concrete test names, no acceptance criteria).
        """
        ticket = ticket or {}
        confidence = float(ticket.get("confidence") or 0.0)
        if confidence < _MIN_TICKET_CONFIDENCE:
            _log.warning(
                "upgrade ticket rejected: confidence %.2f < %.2f (%s)",
                confidence, _MIN_TICKET_CONFIDENCE,
                ticket.get("title") or "<untitled>")
            return ""
        problems = gate_ticket(ticket)
        if problems:
            _log.warning(
                "upgrade ticket rejected as vague (%s): %s",
                "; ".join(problems), ticket.get("title") or "<untitled>")
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
