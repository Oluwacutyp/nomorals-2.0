"""Adversarial debate: coder vs critic, plus disagreement arbitration.

A :class:`Debate` runs a unit of work through coder→critic rounds until the
critic approves, the score crosses the acceptance threshold, the critic
escalates a stalemate, or ``max_rounds`` is hit (then ``unresolved`` — never
a silent accept).  Every round is recorded on the Blackboard so the
reflector and Prompt 01's lesson memory can learn from it.

The coder and critic are injected callables, so debates are fully
testable without a model; in production they are thin wrappers around the
router with the ``coder`` / ``critic`` role system prompts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable

from ..core.ids import new_id
from ..core.logging_setup import get_logger
from .blackboard import Blackboard

__all__ = [
    "Debate", "DebateResult", "Issue", "Critique", "WorkArtifact",
    "DEFAULT_RUBRIC", "arbitrate", "ArbitrationResult",
]

_log = get_logger(__name__)

#: What the critic grades against.
DEFAULT_RUBRIC = (
    "correctness",       # does it do what was asked, on all inputs?
    "edge_cases",        # empty / huge / malformed / boundary inputs
    "security",          # injection, traversal, secret leakage, unsafe exec
    "test_coverage",     # is the claim verified, not just asserted?
)


@dataclass
class Issue:
    id: str
    severity: str  # critical | major | minor
    location: str
    detail: str
    round: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "severity": self.severity,
                "location": self.location, "detail": self.detail,
                "round": self.round}


@dataclass
class Critique:
    verdict: str  # approve | request_changes | reject
    issues: list[Issue] = field(default_factory=list)
    score: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "score": self.score,
                "notes": self.notes,
                "issues": [i.to_dict() for i in self.issues]}


@dataclass
class WorkArtifact:
    content: Any
    summary: str = ""
    #: Issue ids the revision claims to fix (coder must quote them).
    addresses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"summary": self.summary, "addresses": self.addresses,
                "content": self.content if isinstance(
                    self.content, (str, int, float, bool, type(None)))
                else str(self.content)[:4000]}


@dataclass
class DebateResult:
    debate_id: str
    verdict: str  # approved | unresolved | needs_arbitration | rejected
    rounds: int
    score: float
    final_artifact: WorkArtifact | None
    transcript: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    @property
    def approved(self) -> bool:
        return self.verdict == "approved"

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id, "verdict": self.verdict,
            "rounds": self.rounds, "score": self.score, "reason": self.reason,
            "final_artifact": (self.final_artifact.to_dict()
                               if self.final_artifact else None),
            "transcript": self.transcript,
        }


class Debate:
    """Orchestrates coder-vs-critic rounds over one unit of work."""

    def __init__(
        self,
        *,
        coder_fn: Callable[[str, list[Issue] | None, WorkArtifact | None],
                           WorkArtifact],
        critic_fn: Callable[[WorkArtifact, tuple[str, ...]], Critique],
        blackboard: Blackboard | None = None,
        rubric: tuple[str, ...] = DEFAULT_RUBRIC,
        max_rounds: int = 3,
        accept_score: float = 85.0,
        context: Any = None,
    ) -> None:
        self.coder_fn = coder_fn
        self.critic_fn = critic_fn
        self.blackboard = blackboard if blackboard is not None else Blackboard()
        self.rubric = rubric
        self.max_rounds = max_rounds
        self.accept_score = accept_score
        self.context = context
        self.debate_id = f"debate-{new_id()[-8:]}"

    # ── main loop ────────────────────────────────────────────────────────────
    def run(self, brief: str,
            initial_artifact: WorkArtifact | None = None) -> DebateResult:
        transcript: list[dict[str, Any]] = []
        artifact = initial_artifact
        feedback: list[Issue] | None = None
        # issue id -> consecutive rounds it survived unaddressed
        survivors: dict[str, int] = {}

        for round_no in range(1, self.max_rounds + 1):
            artifact = self.coder_fn(brief, feedback, artifact)
            self._post(f"round.{round_no}.coder", artifact.to_dict(),
                       author="coder", round_no=round_no)
            critique = self.critic_fn(artifact, self.rubric)
            self._post(f"round.{round_no}.critic", critique.to_dict(),
                       author="critic", round_no=round_no)
            transcript.append({
                "round": round_no,
                "coder": artifact.to_dict(),
                "critic": critique.to_dict(),
            })
            if self.context is not None:
                try:
                    self.context.emit(
                        "swarm.debate_round", debate_id=self.debate_id,
                        round=round_no, verdict=critique.verdict,
                        score=critique.score)
                except Exception:  # noqa: BLE001 - telemetry never breaks debate
                    pass

            if critique.verdict == "approve" or critique.score >= self.accept_score:
                return self._finish("approved", round_no, critique.score,
                                    artifact, transcript,
                                    "critic approved the work")
            if critique.verdict == "reject":
                return self._finish("rejected", round_no, critique.score,
                                    artifact, transcript,
                                    "critic rejected the work outright")

            # Anti-stalemate: an issue the critic repeats across 2 rounds
            # without the coder addressing it escalates instead of looping.
            addressed = set(artifact.addresses or [])
            escalate: list[str] = []
            current_ids = {i.id for i in critique.issues}
            for issue in critique.issues:
                if issue.id in addressed:
                    survivors.pop(issue.id, None)
                else:
                    survivors[issue.id] = survivors.get(issue.id, 0) + 1
                    if survivors[issue.id] >= 2:
                        escalate.append(issue.id)
            for stale in [k for k in survivors if k not in current_ids]:
                survivors.pop(stale, None)
            if escalate:
                return self._finish(
                    "needs_arbitration", round_no, critique.score, artifact,
                    transcript,
                    f"stalemate: critic repeated issue(s) {escalate} across "
                    f"2 rounds without resolution")

            feedback = critique.issues

        return self._finish("unresolved", self.max_rounds,
                            transcript[-1]["critic"]["score"] if transcript else 0.0,
                            artifact, transcript,
                            f"max_rounds ({self.max_rounds}) hit without approval")

    # ── helpers ──────────────────────────────────────────────────────────────
    def _post(self, key: str, value: Any, *, author: str, round_no: int) -> None:
        self.blackboard.post(
            f"{self.debate_id}.{key}", value, author=author,
            topic=self.debate_id,
            metadata={"round": round_no, "debate_id": self.debate_id})

    def _finish(self, verdict: str, rounds: int, score: float,
                artifact: WorkArtifact | None,
                transcript: list[dict[str, Any]], reason: str) -> DebateResult:
        result = DebateResult(
            debate_id=self.debate_id, verdict=verdict, rounds=rounds,
            score=score, final_artifact=artifact, transcript=transcript,
            reason=reason)
        self._post("result", result.to_dict(), author="debate", round_no=rounds)
        _log.info("debate %s finished: %s after %d round(s) (%s)",
                  self.debate_id, verdict, rounds, reason)
        return result


@dataclass
class ArbitrationResult:
    decision: Any
    rationale: str
    level: int  # 1 = judge agent, 2 = escalated to owner
    judge_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"decision": self.decision, "rationale": self.rationale,
                "level": self.level, "judge_notes": self.judge_notes}


def arbitrate(
    positions: list[dict[str, Any]],
    rubric: tuple[str, ...] = DEFAULT_RUBRIC,
    *,
    judge_fn: Callable[[list[dict[str, Any]], tuple[str, ...]], dict[str, Any]] | None = None,
    blackboard: Blackboard | None = None,
    context: Any = None,
    autonomous: bool = True,
    topic: str = "arbitration",
) -> ArbitrationResult:
    """Settle a disagreement between parallel agents.

    Level 1: a ``judge`` (critic role, fresh context — it sees only the
    artifacts plus the rubric) picks.  Level 2: when there is no judge or
    the judge abstains, escalate to the owner with a concise brief in
    approval-style modes; in autonomous mode, log and proceed with the
    first position.  The outcome and rationale are recorded on the
    Blackboard and returned as a lesson candidate.
    """
    board = blackboard if blackboard is not None else Blackboard()
    arb_id = f"arb-{new_id()[-8:]}"

    board.post(f"{arb_id}.positions", positions, author="arbitrator",
               topic=topic, metadata={"n": len(positions)})

    decision: Any = None
    rationale = ""
    level = 1
    judge_notes = ""
    if judge_fn is not None:
        try:
            ruling = judge_fn(positions, rubric)
        except Exception as exc:  # noqa: BLE001 - a bad judge escalates
            ruling = None
            judge_notes = f"judge failed: {exc}"
        if isinstance(ruling, dict) and ruling.get("decision") is not None:
            decision = ruling["decision"]
            rationale = str(ruling.get("rationale", ""))
            judge_notes = str(ruling.get("notes", ""))
        else:
            judge_notes = judge_notes or "judge abstained"

    if decision is None:
        level = 2
        brief = {
            "positions": [
                {"label": p.get("label"), "evidence": str(p.get("evidence", ""))[:500]}
                for p in positions
            ],
            "recommended": positions[0].get("label") if positions else None,
            "judge_notes": judge_notes,
        }
        if autonomous:
            decision = positions[0].get("decision", positions[0]) if positions else None
            rationale = ("no decisive judge; autonomous mode proceeds with "
                         "the first position")
            _log.warning("arbitration %s: %s", arb_id, rationale)
        else:
            decision = {"escalate_to_owner": brief}
            rationale = "escalated to owner with positions and recommendation"

    result = ArbitrationResult(decision=decision, rationale=rationale,
                               level=level, judge_notes=judge_notes)
    board.post(f"{arb_id}.result", result.to_dict(), author="arbitrator",
               topic=topic)
    if context is not None:
        try:
            context.emit("swarm.arbitration", arbitration_id=arb_id,
                         level=level, topic=topic)
        except Exception:  # noqa: BLE001 - telemetry never breaks arbitration
            pass
    return result
