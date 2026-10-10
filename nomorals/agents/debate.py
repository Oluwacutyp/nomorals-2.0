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
    "PanelDebate", "PanelResult", "Position",
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

    def settle(
        self,
        brief: str,
        initial_artifact: WorkArtifact | None = None,
        *,
        judge_fn: Callable[
            [list[dict[str, Any]], tuple[str, ...]], dict[str, Any]] | None = None,
        autonomous: bool = True,
    ) -> tuple[DebateResult, ArbitrationResult | None]:
        """Run the debate, then settle it no matter how it ends.

        ``approved`` comes back untouched. Anything else
        (``unresolved``, ``needs_arbitration``, ``rejected``) goes to
        :func:`arbitrate` between the final artifact and the critic's
        position, so a hard decision never dies quietly — it gets a
        named decision with a rationale. Returns ``(debate_result,
        arbitration_or_None)``.
        """
        result = self.run(brief, initial_artifact)
        if result.verdict == "approved":
            return result, None
        positions: list[dict[str, Any]] = []
        if result.final_artifact is not None:
            positions.append({
                "label": "coder",
                "decision": result.final_artifact.content,
                "evidence": result.final_artifact.summary,
                "score": result.score,
            })
        if result.transcript:
            last_critic = result.transcript[-1].get("critic", {})
            issues = [i.get("detail", "") for i in last_critic.get("issues", [])]
            positions.append({
                "label": "critic",
                "decision": None,
                "evidence": "; ".join(issues) or last_critic.get("notes", ""),
                "score": last_critic.get("score", 0.0),
            })
        arbitration = arbitrate(
            positions, self.rubric, judge_fn=judge_fn,
            blackboard=self.blackboard, context=self.context,
            autonomous=autonomous, topic=self.debate_id)
        self._post("settlement", arbitration.to_dict(), author="debate",
                   round_no=result.rounds)
        return result, arbitration

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


@dataclass
class Position:
    """One contender in a panel debate: a labelled artifact plus the
    structured claim → evidence → risk framing the research says keeps
    debates honest (and terminating)."""

    label: str
    artifact: WorkArtifact
    claim: str = ""
    evidence: str = ""
    risk: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "claim": self.claim,
                "evidence": self.evidence, "risk": self.risk,
                "artifact": self.artifact.to_dict()}


@dataclass
class PanelResult:
    """Outcome of a :class:`PanelDebate`."""

    panel_id: str
    winner: Position | None
    scores: dict[str, float]  # label -> mean critic score
    verdict: str  # decided | tie_arbitrated | no_contenders
    rounds: int
    transcript: list[dict[str, Any]] = field(default_factory=list)
    arbitration: ArbitrationResult | None = None
    reason: str = ""

    @property
    def decided(self) -> bool:
        return self.verdict in {"decided", "tie_arbitrated"} and self.winner is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "panel_id": self.panel_id, "verdict": self.verdict,
            "rounds": self.rounds, "reason": self.reason,
            "scores": self.scores,
            "winner": self.winner.to_dict() if self.winner else None,
            "arbitration": (self.arbitration.to_dict()
                            if self.arbitration else None),
            "transcript": self.transcript,
        }


class PanelDebate:
    """N-position debate: every position is scored by the critic(s), the
    winner is picked by confidence-weighted vote, and a tie (or a
    near-tie inside ``tie_margin``) escalates to :func:`arbitrate`
    instead of pretending the vote was decisive.

    Positions are injected (``position_fns``: label → callable returning a
    :class:`Position`), so panels are fully testable without a model; in
    production each position fn is a role-prompted model call.
    """

    def __init__(
        self,
        *,
        position_fns: dict[str, Callable[[], Position]],
        critic_fn: Callable[[WorkArtifact, tuple[str, ...]], Critique],
        blackboard: Blackboard | None = None,
        rubric: tuple[str, ...] = DEFAULT_RUBRIC,
        max_rounds: int = 2,
        tie_margin: float = 5.0,
        judge_fn: Callable[
            [list[dict[str, Any]], tuple[str, ...]], dict[str, Any]] | None = None,
        context: Any = None,
    ) -> None:
        if not position_fns:
            raise ValueError("PanelDebate needs at least one position")
        self.position_fns = dict(position_fns)
        self.critic_fn = critic_fn
        self.blackboard = blackboard if blackboard is not None else Blackboard()
        self.rubric = rubric
        self.max_rounds = max(1, int(max_rounds))
        self.tie_margin = max(0.0, float(tie_margin))
        self.judge_fn = judge_fn
        self.context = context
        self.panel_id = f"panel-{new_id()[-8:]}"

    # ── main loop ────────────────────────────────────────────────────
    def run(self, brief: str) -> PanelResult:
        transcript: list[dict[str, Any]] = []
        positions: list[Position] = []
        for label, make in self.position_fns.items():
            try:
                pos = make()
            except Exception as exc:  # noqa: BLE001 - a dead position is a record
                _log.warning("panel position %r failed to produce: %s", label, exc)
                continue
            if not isinstance(pos, Position):
                _log.warning("panel position %r returned %s, not a Position",
                             label, type(pos).__name__)
                continue
            pos.label = label
            positions.append(pos)
            self._post(f"position.{label}", pos.to_dict(), author=label)

        if not positions:
            return self._finish(None, {}, "no_contenders", 0, transcript,
                                "every position failed to produce")

        totals: dict[str, float] = {p.label: 0.0 for p in positions}
        rounds = 0
        for round_no in range(1, self.max_rounds + 1):
            rounds = round_no
            round_scores: dict[str, float] = {}
            for pos in positions:
                try:
                    critique = self.critic_fn(pos.artifact, self.rubric)
                    score = float(critique.score or 0.0)
                except Exception as exc:  # noqa: BLE001 - a dead critic abstains
                    _log.warning("panel critic failed on %r: %s", pos.label, exc)
                    score = 0.0
                    critique = Critique(verdict="request_changes", score=0.0,
                                        notes=f"critic failed: {exc}")
                totals[pos.label] += score
                round_scores[pos.label] = round(score, 2)
                self._post(f"round.{round_no}.critic.{pos.label}",
                           {"score": score, **critique.to_dict()},
                           author="critic")
            transcript.append({"round": round_no, "scores": round_scores})
            if self.context is not None:
                try:
                    self.context.emit("swarm.panel_round", panel_id=self.panel_id,
                                      round=round_no, scores=round_scores)
                except Exception:  # noqa: BLE001 - telemetry never breaks debate
                    pass

        mean = {label: totals[label] / rounds for label in totals}
        ranked = sorted(mean.items(), key=lambda kv: (-kv[1], kv[0]))
        best_label, best_score = ranked[0]
        # A tie — or a near-tie inside tie_margin — is not a decision.
        # The judge must decide (research: the mitigation that actually
        # stops debate non-termination).
        contenders = [label for label, score in ranked
                      if best_score - score <= self.tie_margin]
        winner = next(p for p in positions if p.label == best_label)
        arbitration: ArbitrationResult | None = None
        if len(contenders) > 1:
            arbitration = arbitrate(
                [{"label": label,
                  "decision": next(p for p in positions if p.label == label).artifact.content,
                  "evidence": next(p for p in positions if p.label == label).evidence,
                  "score": round(mean[label], 2)}
                 for label in contenders],
                self.rubric,
                judge_fn=self.judge_fn,
                blackboard=self.blackboard,
                context=self.context,
                topic=self.panel_id,
            )
            decided_label = None
            if isinstance(arbitration.decision, dict):
                decided_label = arbitration.decision.get("label")
            if decided_label in mean:
                winner = next(p for p in positions if p.label == decided_label)
            return self._finish(
                winner, mean, "tie_arbitrated", rounds, transcript,
                f"scores within {self.tie_margin} points "
                f"({', '.join(f'{l}={mean[l]:.1f}' for l in contenders)}); "
                f"judge decided: {winner.label}",
                arbitration=arbitration)
        return self._finish(winner, mean, "decided", rounds, transcript,
                            f"{winner.label} won outright "
                            f"({best_score:.1f} mean critic score)")

    # ── helpers ──────────────────────────────────────────────────────
    def _post(self, key: str, value: Any, *, author: str) -> None:
        self.blackboard.post(
            f"{self.panel_id}.{key}", value, author=author,
            topic=self.panel_id,
            metadata={"panel_id": self.panel_id})

    def _finish(self, winner: Position | None, scores: dict[str, float],
                verdict: str, rounds: int, transcript: list[dict[str, Any]],
                reason: str,
                arbitration: ArbitrationResult | None = None) -> PanelResult:
        result = PanelResult(
            panel_id=self.panel_id, winner=winner,
            scores={k: round(v, 2) for k, v in scores.items()},
            verdict=verdict, rounds=rounds, transcript=transcript,
            arbitration=arbitration, reason=reason)
        self._post("result", result.to_dict(), author="panel")
        _log.info("panel debate %s finished: %s (%s)",
                  self.panel_id, verdict, reason)
        return result


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


# ── symmetric debate (Du et al. 2023) ─────────────────────────────────
# "Improving Factuality and Reasoning in Language Models through
# Multiagent Debate" (arXiv:2305.14325): every agent answers
# INDEPENDENTLY first, then each round every agent sees ALL others'
# answers and revises its own. Final answer comes from converged
# consensus — 3 agents × 2 rounds beat single-model baselines by ~15pp
# on GSM8K/MMLU and cut biography hallucinations. This is the
# reveal-and-revise protocol: agreement must be EARNED, not assumed.


@dataclass
class SymmetricDebateResult:
    """Outcome of a :class:`SymmetricDebate`."""
    debate_id: str
    verdict: str  # consensus | judge_decided | no_consensus
    rounds: int
    final_answer: str
    answers: dict[str, str]  # label -> final answer
    converged: bool
    transcript: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "debate_id": self.debate_id, "verdict": self.verdict,
            "rounds": self.rounds, "final_answer": self.final_answer,
            "answers": dict(self.answers), "converged": self.converged,
            "reason": self.reason, "transcript": self.transcript,
        }


class SymmetricDebate:
    """Peer debate, not coder-vs-critic.

    ``debater_fns`` maps label → ``fn(brief, others) -> str`` where
    ``others`` is ``{label: answer}`` (empty on the independent round).
    Debaters are injected callables, so the protocol is fully testable
    without a model; in production they are router calls with peer
    personas.

    Protocol (Du et al. 2023):
      1. Round 0 — independent answers, no one sees anyone.
      2. Rounds 1..N — each debater sees every other answer and revises.
      3. Stop early when all answers converge (exact match) or the judge
         says they've converged.
      4. No convergence by ``max_rounds`` → ``judge_fn`` picks; without a
         judge → ``no_consensus`` (honest, never a fake consensus).
    """

    def __init__(
        self,
        *,
        debater_fns: dict[str, Callable[[str, dict[str, str]], str]],
        judge_fn: Callable[[dict[str, str]], dict[str, Any]] | None = None,
        max_rounds: int = 2,
        blackboard: Blackboard | None = None,
        context: Any = None,
        parallel: bool = True,
    ) -> None:
        if len(debater_fns) < 2:
            raise ValueError("symmetric debate needs at least 2 debaters")
        self.debater_fns = dict(debater_fns)
        self.judge_fn = judge_fn
        self.max_rounds = max(1, int(max_rounds))
        self.blackboard = (blackboard if blackboard is not None
                           else Blackboard())
        self.context = context
        self.parallel = bool(parallel)
        self.debate_id = f"sdebate-{new_id()[-8:]}"

    # ── main loop ────────────────────────────────────────────────
    def run(self, brief: str) -> SymmetricDebateResult:
        transcript: list[dict[str, Any]] = []
        answers: dict[str, str] = {}

        def _post(key: str, value: Any, *, author: str, round_no: int) -> None:
            self.blackboard.post(
                f"{self.debate_id}.{key}", value, author=author,
                topic=self.debate_id,
                metadata={"debate_id": self.debate_id, "round": round_no})

        # Round 0: independent answers. No one sees anyone.
        answers = self._collect(brief, {}, round_no=0)
        transcript.append({"round": 0, "answers": dict(answers),
                           "mode": "independent"})
        for label, ans in answers.items():
            _post(f"round.0.{label}", {"answer": ans}, author=label, round_no=0)

        if self._converged(answers):
            return self._finish("consensus", 0, answers, transcript,
                                "independent answers already agreed")

        for round_no in range(1, self.max_rounds + 1):
            round_answers: dict[str, str] = {}
            new_answers = self._collect(brief, answers, round_no=round_no)
            for label, ans in new_answers.items():
                round_answers[label] = ans
                _post(f"round.{round_no}.{label}", {"answer": ans},
                      author=label, round_no=round_no)
            answers = round_answers
            transcript.append({"round": round_no, "answers": dict(answers),
                               "mode": "reveal_and_revise"})
            if self.context is not None:
                try:
                    self.context.emit("swarm.symmetric_round",
                                      debate_id=self.debate_id,
                                      round=round_no, n_debaters=len(answers))
                except Exception:  # noqa: BLE001 - telemetry never breaks debate
                    pass
            if self._converged(answers):
                return self._finish("consensus", round_no, answers, transcript,
                                    f"debaters converged after round {round_no}")

        # No convergence: the judge decides, or honesty.
        return self._judge_or_honest(brief, answers, transcript)

    def _collect(self, brief: str, others: dict[str, str],
                 *, round_no: int) -> dict[str, str]:
        """Run every debater; parallel when enabled (debates are slow)."""
        labels = list(self.debater_fns)
        if not self.parallel or len(labels) < 2:
            return {label: self._safe_call(label, brief, others)
                    for label in labels}
        import concurrent.futures as _cf
        out: dict[str, str] = {}
        with _cf.ThreadPoolExecutor(
                max_workers=min(len(labels), 8),
                thread_name_prefix="symmetric-debate") as pool:
            futs = {pool.submit(self._safe_call, label, brief, others): label
                    for label in labels}
            for fut in _cf.as_completed(futs):
                label = futs[fut]
                try:
                    out[label] = fut.result()
                except Exception:  # noqa: BLE001
                    out[label] = "(no answer)"
        # Preserve debater order for deterministic transcripts.
        return {label: out.get(label, "(no answer)") for label in labels}

    def _safe_call(self, label: str, brief: str,
                   others: dict[str, str]) -> str:
        try:
            visible = {k: v for k, v in others.items() if k != label}
            ans = self.debater_fns[label](brief, visible)
            text = str(ans or "").strip()
            return text or "(no answer)"
        except Exception as exc:  # noqa: BLE001 - a dead debater is a record
            _log.warning("symmetric debater %r failed: %s", label, exc)
            return f"(error: {exc})"

    @staticmethod
    def _converged(answers: dict[str, str]) -> bool:
        """All answers agree (exact match after whitespace normalization)."""
        texts = {str(a or "").strip() for a in answers.values()}
        return len(texts) == 1 and next(iter(texts)) not in {"", "(no answer)"}

    def _judge_or_honest(self, brief: str, answers: dict[str, str],
                         transcript: list[dict[str, Any]]) -> SymmetricDebateResult:
        if self.judge_fn is not None:
            try:
                ruling = self.judge_fn(dict(answers))
            except Exception as exc:  # noqa: BLE001 - a bad judge abstains
                ruling = None
                _log.warning("symmetric judge failed: %s", exc)
            if isinstance(ruling, dict) and ruling.get("answer"):
                final = str(ruling["answer"])
                reason = str(ruling.get("rationale", "judge decided"))
                self.blackboard.post(
                    f"{self.debate_id}.judge",
                    {"final": final, "rationale": reason},
                    author="judge", topic=self.debate_id)
                return self._finish("judge_decided", self.max_rounds, answers,
                                    transcript, reason, final_answer=final)
        return self._finish("no_consensus", self.max_rounds, answers,
                            transcript,
                            f"{self.max_rounds} rounds without convergence; "
                            "no judge — refusing to fake consensus")

    def _finish(self, verdict: str, rounds: int, answers: dict[str, str],
                transcript: list[dict[str, Any]], reason: str,
                final_answer: str = "") -> SymmetricDebateResult:
        if not final_answer:
            # Consensus verdicts share one answer by construction.
            final_answer = next(iter(answers.values()), "")
        result = SymmetricDebateResult(
            debate_id=self.debate_id, verdict=verdict, rounds=rounds,
            final_answer=final_answer, answers=dict(answers),
            converged=(verdict == "consensus"), transcript=transcript,
            reason=reason)
        self.blackboard.post(f"{self.debate_id}.result", result.to_dict(),
                             author="symmetric_debate", topic=self.debate_id)
        _log.info("symmetric debate %s finished: %s (%s)",
                  self.debate_id, verdict, reason)
        return result


# ── rendering ─────────────────────────────────────────────────────────
# Debate transcripts are decision records: they should read like one.

def render_debate(result: Any) -> str:
    """Render any debate result as human-readable markdown.

    Accepts DebateResult, PanelResult, or SymmetricDebateResult.
    Never raises.
    """
    from .render import ICONS, banner, bullets, kv, section, truncate

    try:
        verdict = getattr(result, "verdict", "?")
        rounds = getattr(result, "rounds", 0)
        reason = getattr(result, "reason", "") or ""
        did = (getattr(result, "debate_id", None)
               or getattr(result, "panel_id", "?"))
        good = verdict in {"approved", "decided", "tie_arbitrated",
                           "consensus", "judge_decided"}
        head = {
            "verdict": verdict,
            "rounds": rounds,
            "id": did,
            "why": reason,
        }
        lines = [banner(f"Debate {did}", ICONS["debate"]),
                 kv(head.items())]
        transcript = getattr(result, "transcript", None) or []
        for entry in transcript:
            rnd = entry.get("round", "?")
            lines.append("")
            lines.append(f"**Round {rnd}**")
            if "answers" in entry:  # symmetric: peer answers
                for label, ans in entry["answers"].items():
                    lines.append(f"_{label}_: {truncate(ans, 300)}")
            if "coder" in entry:  # coder-vs-critic
                coder = entry["coder"] or {}
                critic = entry["critic"] or {}
                lines.append(f"🧑‍💻 coder: "
                             f"{truncate(str(coder.get('content', '')), 200)}")
                lines.append(f"🔍 critic [{critic.get('verdict', '?')}, "
                             f"score {critic.get('score', '?')}]: "
                             f"{truncate(str(critic.get('notes', '')), 200)}")
            if "scores" in entry and "answers" not in entry:  # panel
                scores = ", ".join(
                    f"{k}={v}" for k, v in entry["scores"].items())
                lines.append(f"scores: {scores}")
        final = (getattr(result, "final_answer", None)
                 or (getattr(result, "final_artifact", None)
                     and getattr(result.final_artifact, "content", None))
                 or (getattr(result, "winner", None)
                     and getattr(result.winner, "label", None)))
        lines.append("")
        lines.append(section("Final",
                             f"{'✅' if good else '⚠️'} {truncate(str(final), 600)}",
                             ICONS["star"]))
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — rendering never breaks callers
        return f"debate result ({type(result).__name__}): {verdict}"


__all__ = __all__ + ["SymmetricDebate", "SymmetricDebateResult", "render_debate"]
