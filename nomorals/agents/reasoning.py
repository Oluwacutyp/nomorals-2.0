"""Reasoning engine — explicit, auditable, multi-strategy thought.

The difference between an LLM call and a reasoning system is that a
reasoning system shows its work.  Every call here produces a TRACE —
an ordered list of typed steps (plan, subgoal, action, observation,
critique, verdict) with confidence — plus a final answer.  The trace
is:

  * **auditable**  — stored with the result, rendered by ``nm reason``,
                     ``/think`` in chat, and the ``reason`` tool
  * **composable** — strategies call each other (decompose → cot;
                     auto dispatches on problem shape)
  * **testable**   — fully hermetic: a scripted router proves each
                     strategy's control flow without any real model

Strategies
----------
  cot         linear step-by-step chain, parsed into typed steps
  decompose   subgoals → solve each (recursive, depth-limited) → synthesize
  hypothesize hypotheses → evidence scoring → best hypothesis re-derived
  critique    draft → critic → revise loop until the critic passes it
  tree        bounded plan-branch-evaluate search over action plans
  auto        classify the problem shape, dispatch to the right strategy

Budgets: ``max_llm_calls`` and ``max_seconds`` — every strategy honours
them and reports ``stopped`` = complete | budget.  Tool access is
optional: pass a ``tools`` callable for action steps (devon, the
orchestrator); the partner brain runs pure reasoning by default.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..core.errors import ToolError
from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..llm.base import Message, SamplingParams

_log = get_logger(__name__)

__all__ = ["ReasoningAgent", "ReasoningEngine", "ReasoningResult",
           "ReasoningStep", "looks_complex", "reasoning_eval", "register",
           "trace_text"]

_COMPLEX_RE = re.compile(
    r"\b(why|how (do|should|can|would)|explain|prove|design|architect|"
    r"calculate|compare|should i|plan(ning)?|step(s)? (by )?through|"
    r"trade-?off|pros? (and|vs)|consequence|impact (of|on)|root cause|"
    r"most likely|which (is|are) (best|right|safe))\b", re.I)


def looks_complex(text: str) -> bool:
    """Conservative heuristic: does this message warrant a reasoning pass?

    Long enough, multi-part, or carrying a genuinely analytical marker.
    A one-line "hey" never triggers; a real question usually does.
    """
    t = (text or "").strip()
    if not t:
        return False
    if t.count("?") >= 2:
        return True
    if len(t) >= 100:
        return True
    if len(t) >= 50 and _COMPLEX_RE.search(t):
        return True
    return False

_MAX_BRANCHES = 3
_MAX_HYPOTHESES = 4
_MAX_CRITIQUE_ROUNDS = 2


# ── trace model ──────────────────────────────────────────────────────────────


@dataclass
class ReasoningStep:
    index: int
    kind: str          # plan | subgoal | action | observation | critique | verdict | note
    text: str
    confidence: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"index": self.index, "kind": self.kind, "text": self.text,
                "confidence": self.confidence, "meta": self.meta}


@dataclass
class ReasoningResult:
    answer: str
    strategy: str
    trace: list[ReasoningStep]
    confidence: float
    subgoals: list[str] = field(default_factory=list)
    hypotheses: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: int = 0
    seconds: float = 0.0
    stopped: str = "complete"

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer, "strategy": self.strategy,
            "confidence": round(self.confidence, 3),
            "stopped": self.stopped, "llm_calls": self.llm_calls,
            "seconds": round(self.seconds, 2),
            "subgoals": self.subgoals, "hypotheses": self.hypotheses,
            "trace": [s.as_dict() for s in self.trace],
        }


def trace_text(result: ReasoningResult) -> str:
    """Render a result for a human: the work, then the answer."""
    lines = [f"reasoning [{result.strategy}] "
             f"({result.llm_calls} calls, {result.seconds:.1f}s, "
             f"conf {result.confidence:.2f}, {result.stopped})"]
    icons = {"plan": "◆", "subgoal": "◇", "action": "→",
             "observation": "•", "critique": "✎", "verdict": "■",
             "note": "·"}
    for s in result.trace:
        text = s.text if len(s.text) <= 240 else s.text[:237] + "…"
        lines.append(f"  {icons.get(s.kind, '·')} [{s.kind}] {text}")
    lines.append(f"■ ANSWER: {result.answer}")
    return "\n".join(lines)


# ── plumbing ─────────────────────────────────────────────────────────────────


class _Budget:
    def __init__(self, max_calls: int, max_seconds: float) -> None:
        self.max_calls = max_calls
        self.max_seconds = max_seconds
        self.calls = 0
        self.started = time.monotonic()

    def spend(self) -> None:
        self.calls += 1

    def exhausted(self) -> bool:
        return (self.calls >= self.max_calls
                or time.monotonic() - self.started > self.max_seconds)


def _extract_json(text: str) -> Optional[Any]:
    """Pull the first balanced JSON object/array out of a model reply.

    Handles raw JSON, ```json fences, and JSON embedded in prose.
    Returns None when nothing parseable is there.
    """
    if not text:
        return None
    candidates: list[str] = []
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        candidates.append(fence.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start = 0
        while True:
            start = text.find(opener, start)
            if start == -1:
                break
            end = _balanced_end(text, start, opener, closer)
            if end is None:
                break
            candidates.append(text[start:end + 1])
            # keep scanning past this candidate so a malformed one does
            # not hide a valid JSON blob later in the same reply
            start = end + 1
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def _balanced_end(text: str, start: int, opener: str, closer: str) -> int | None:
    """Index of the closer matching text[start], or None if unbalanced.

    String-aware: braces inside quoted values do not count.
    """
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return i
    return None


def _confidence(raw: Any, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return default


# ── the engine ───────────────────────────────────────────────────────────────


class ReasoningEngine:
    """Multi-strategy reasoning with an explicit, budgeted trace."""

    def __init__(
        self,
        context: Any,
        *,
        tools: Optional[Callable[..., str]] = None,
        max_llm_calls: int = 12,
        max_seconds: float = 120.0,
        system_note: str = "",
    ) -> None:
        self.context = context
        self.tools = tools
        self.system_note = system_note
        self.budget = _Budget(max_llm_calls, max_seconds)

    # ── llm ──────────────────────────────────────────────────────────────────

    def _llm(self, prompt: str, *, temperature: float = 0.2) -> str:
        if self.budget.exhausted():
            raise _BudgetExhausted()
        self.budget.spend()
        system = ("You are a rigorous reasoning engine. Work step by step, "
                  "never skip a step you claim, and when a format is "
                  "requested, follow it EXACTLY. No preamble, no apologies.")
        if self.system_note:
            system += f"\nContext: {self.system_note}"
        response = self.context.router.chat(
            [Message.system(system), Message.user(prompt)],
            SamplingParams(temperature=temperature, max_tokens=1400),
        )
        if not response.ok:
            raise ToolError(f"reasoning: {getattr(response, 'error', 'model error')}")
        text = (response.text or "").strip()
        if not text:
            raise ToolError("reasoning: empty model response")
        return text

    # ── entry point ──────────────────────────────────────────────────────────

    def reason(self, goal: str, *, strategy: str = "auto", depth: int = 1,
               extra_context: str = "") -> ReasoningResult:
        goal = (goal or "").strip()
        if not goal:
            raise ToolError("reasoning needs a goal or question")
        # KG-aware reasoning: fold in what the system already knows about
        # this goal. Always available; "" when the graph is empty/off.
        kg_block = _knowledge_block(self.context, goal)
        if kg_block and kg_block not in extra_context:
            extra_context = (f"{extra_context}\n\n{kg_block}"
                             if extra_context else kg_block)
        started = time.monotonic()
        trace: list[ReasoningStep] = []
        if kg_block:
            self._step(trace, "note",
                       "reasoning over stored knowledge "
                       f"({kg_block.count(chr(10)) + 1} lines of context)",
                       0.5)
        strategy = (strategy or "auto").lower()
        depth = max(0, min(int(depth), 3))
        if strategy == "auto":
            strategy = self._classify(goal)
        self._step(trace, "plan",
                   f"goal: {goal[:300]}  [strategy={strategy}, "
                   f"depth={depth}]", 0.6)
        answer = ""
        confidence = 0.5
        subgoals: list[str] = []
        hypotheses: list[dict[str, Any]] = []
        stopped = "complete"
        try:
            if strategy == "decompose":
                answer, confidence, subgoals = self._decompose(
                    goal, trace, depth, extra_context)
            elif strategy == "hypothesize":
                answer, confidence, hypotheses = self._hypothesize(
                    goal, trace, extra_context)
            elif strategy == "critique":
                answer, confidence = self._critique(goal, trace, extra_context)
            elif strategy == "tree":
                answer, confidence = self._tree(goal, trace, depth,
                                                 extra_context)
            else:
                strategy = "cot"
                answer, confidence = self._cot(goal, trace, extra_context)
        except _BudgetExhausted:
            stopped = "budget"
            self._step(trace, "note",
                       "budget exhausted — best partial answer "
                       "below", 0.3)
            answer = answer or ("(incomplete: reasoning budget exhausted "
                                "before a final answer)")
        result = ReasoningResult(
            answer=answer, strategy=strategy, trace=trace,
            confidence=confidence, subgoals=subgoals, hypotheses=hypotheses,
            llm_calls=self.budget.calls,
            seconds=time.monotonic() - started, stopped=stopped)
        self._step(trace, "verdict",
                   f"confidence {confidence:.2f} — {stopped}", confidence)
        return result

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _step(trace: list[ReasoningStep], kind: str, text: str,
              confidence: float = 0.5, **meta: Any) -> ReasoningStep:
        s = ReasoningStep(index=len(trace), kind=kind, text=text,
                          confidence=confidence, meta=meta)
        trace.append(s)
        return s

    def _classify(self, goal: str) -> str:
        """Heuristic problem-shape classifier (cheap, deterministic first)."""
        low = goal.lower()
        if re.search(r"\b(why|most likely|possible causes?|hypothes|explain "
                     r"the (reason|cause)|what went wrong)\b", low):
            return "hypothesize"
        if re.search(r"\b(is (this|that) correct|review|check (for|my)|"
                     r"find (the )?(flaw|error|bug|contradiction)|audit)\b",
                     low):
            return "critique"
        if re.search(r"\b(which option|compare (the )?(options|approaches)|"
                     r"which of|should i (pick|choose))\b", low):
            return "tree"
        if re.search(r"\b(design|build|plan|architect|implement|roadmap|"
                     r"step by step how to)\b", low) or \
                (len(goal) > 260 and goal.count(".") >= 2) or \
                re.search(r"\b(then|after that|finally|next),\b", low) \
                and low.count(" and ") >= 1:
            return "decompose"
        return "cot"

    def _parse_cot_reply(self, raw: str,
                         trace: list[ReasoningStep]) -> tuple[str, float]:
        """Parse the REASONING:/ANSWER:/CONFIDENCE: protocol; degrade
        gracefully to the raw text when the model ignored the format."""
        answer = ""
        confidence = 0.5
        reasoning_block = ""
        if "ANSWER:" in raw.upper():
            head, _, tail = raw.partition("ANSWER:")
            reasoning_block = head
            answer = tail.strip()
            if "CONFIDENCE:" in answer.upper():
                answer, _, conf = answer.partition("CONFIDENCE:")
                confidence = _confidence(conf.strip()[:12])
                answer = answer.strip()
        else:
            answer = raw.strip()
            reasoning_block = ""
        for line in reasoning_block.splitlines():
            m = re.match(r"\s*(?:\d+[\.\)]|[-*])\s+(.*)", line)
            if m:
                self._step(trace, "note", m.group(1).strip()[:400], 0.6)
        return answer[:4000], confidence

    # ── strategies ───────────────────────────────────────────────────────────

    def _cot(self, goal: str, trace: list[ReasoningStep],
             extra_context: str) -> tuple[str, float]:
        ctx = f"\nRelevant context:\n{extra_context}" if extra_context else ""
        prompt = (
            f"Reason about this, step by step.{ctx}\n\n"
            f"QUESTION:\n{goal}\n\n"
            "Respond in EXACTLY this format (no extra text):\n"
            "REASONING:\n1. <first step>\n2. <next step>\n"
            "...\nANSWER: <final answer>\nCONFIDENCE: <0.0-1.0>")
        raw = self._llm(prompt)
        answer, confidence = self._parse_cot_reply(raw, trace)
        if not answer:
            answer = "(no answer produced)"
        self._step(trace, "verdict", answer[:300], confidence)
        return answer, confidence

    def _decompose(self, goal: str, trace: list[ReasoningStep],
                   depth: int, extra_context: str
                   ) -> tuple[str, float, list[str]]:
        ctx = f"\nRelevant context:\n{extra_context}" if extra_context else ""
        prompt = (
            f"Decompose this into the subgoals that must each be solved "
            f"first.{ctx}\n\n"
            f"PROBLEM:\n{goal}\n\n"
            'Respond with JSON ONLY: {"subgoals": ["<subgoal 1>", '
            '"<subgoal 2>", ...]}  (2-5 subgoals, each self-contained)')
        data = _extract_json(self._llm(prompt))
        subgoals: list[str] = []
        if isinstance(data, dict):
            subgoals = [str(s).strip() for s in data.get("subgoals", [])
                        if str(s).strip()][:6]
        if not subgoals:
            # degradation: model ignored the format — single chain
            self._step(trace, "note",
                       "decomposition unparsable — falling back to a "
                       "single chain", 0.4)
            answer, confidence = self._cot(goal, trace, extra_context)
            return answer, confidence, []
        self._step(trace, "plan",
                   "subgoals: " + " / ".join(s[:60] for s in subgoals), 0.6)
        answers: list[tuple[str, str, float]] = []
        for i, sub in enumerate(subgoals, 1):
            self._step(trace, "subgoal", f"({i}/{len(subgoals)}) {sub[:300]}",
                       0.6)
            sub_trace: list[ReasoningStep] = []
            if depth > 1:
                a, c, _subs = self._decompose(
                    sub, sub_trace, depth - 1, extra_context)
            else:
                a, c = self._cot(sub, sub_trace, extra_context)
            # keep the sub-work in the main trace — auditable means the
            # full work, including what happened inside each subgoal
            for s in sub_trace:
                self._step(trace, s.kind, s.text, s.confidence, **s.meta)
            answers.append((sub, a, c))
            self._step(trace, "observation",
                       f"subgoal {i} → {a[:240]}", c)
        digest = "\n".join(f"- {sub}: {a[:400]}" for sub, a, _c in answers)
        prompt = (
            f"These subgoals were solved for the original problem:\n\n"
            f"{digest}\n\n"
            f"Original problem:\n{goal}\n\n"
            "Synthesize the FINAL answer. Respond in EXACTLY this format:\n"
            "ANSWER: <final answer>\nCONFIDENCE: <0.0-1.0>")
        raw = self._llm(prompt)
        if "ANSWER:" in raw.upper():
            answer = raw.partition("ANSWER:")[2].strip()
            if "CONFIDENCE:" in answer.upper():
                answer, _, conf = answer.partition("CONFIDENCE:")
                confidence = _confidence(conf.strip()[:12])
                answer = answer.strip()
        else:
            answer, confidence = raw.strip(), 0.5
        self._step(trace, "verdict", answer[:300], confidence)
        return answer[:4000], confidence, subgoals

    def _hypothesize(self, goal: str, trace: list[ReasoningStep],
                     extra_context: str) -> tuple[str, float, list[dict]]:
        ctx = f"\nRelevant context:\n{extra_context}" if extra_context else ""
        prompt = (
            f"Generate the most plausible competing hypotheses for this."
            f"{ctx}\n\n"
            f"QUESTION:\n{goal}\n\n"
            'Respond with JSON ONLY: {"hypotheses": [{"claim": "...", '
            '"why": "..."}]}  (2-4 hypotheses, genuinely different)')
        data = _extract_json(self._llm(prompt))
        hyps: list[dict[str, str]] = []
        if isinstance(data, dict):
            hyps = [{"claim": str(h.get("claim", "")).strip(),
                     "why": str(h.get("why", "")).strip()}
                    for h in data.get("hypotheses", [])
                    if str(h.get("claim", "")).strip()][:_MAX_HYPOTHESES]
        if not hyps:
            self._step(trace, "note",
                       "hypotheses unparsable — falling back to a chain",
                       0.4)
            a, c = self._cot(goal, trace, extra_context)
            return a, c, []
        for i, h in enumerate(hyps, 1):
            self._step(trace, "subgoal",
                       f"H{i}: {h['claim'][:240]}", 0.5)
        scored: list[dict[str, Any]] = []
        for i, h in enumerate(hyps, 1):
            prompt = (
                f"Evaluate ONE hypothesis against the question. Be honest — "
                f"refuting evidence counts double.\n\n"
                f"QUESTION:\n{goal}\n\n"
                f"HYPOTHESIS {i}: {h['claim']}\nBasis: {h['why']}\n\n"
                'Respond with JSON ONLY: {"supports": ["..."], "refutes": '
                '["..."], "score": <0.0-1.0>}')
            data = _extract_json(self._llm(prompt))
            if isinstance(data, dict):
                score = _confidence(data.get("score"), 0.3)
                supports = [str(s)[:160] for s in
                            data.get("supports", [])][:4]
                refutes = [str(s)[:160] for s in data.get("refutes", [])][:4]
            else:
                score, supports, refutes = 0.3, [], []
            entry = {**h, "score": round(score, 3), "supports": supports,
                     "refutes": refutes}
            scored.append(entry)
            self._step(trace, "observation",
                       f"H{i} score {score:.2f} — supports: "
                       f"{'; '.join(supports[:2]) or '—'} | refutes: "
                       f"{'; '.join(refutes[:2]) or '—'}", score)
        best = max(scored, key=lambda h: h["score"])
        prompt = (
            f"Evidence scored these hypotheses; the best is quoted.\n\n"
            f"QUESTION:\n{goal}\n\n"
            "SCORING:\n" +
            "\n".join(f"- ({h['score']:.2f}) {h['claim']} "
                      f"[refuted by: {'; '.join(h['refutes'][:2]) or 'none'}]"
                      for h in scored) +
            f"\n\nGiven this evidence, what is the answer? Respond in "
            "EXACTLY this format:\nANSWER: <final answer>\n"
            "CONFIDENCE: <0.0-1.0>")
        raw = self._llm(prompt)
        answer, confidence = self._parse_cot_reply(raw, trace)
        confidence = min(confidence, 0.95)  # hypothesis space was real
        self._step(trace, "verdict",
                   f"best hypothesis: {best['claim'][:200]} — "
                   f"{answer[:200]}", confidence)
        return answer, confidence, scored

    def _critique(self, goal: str, trace: list[ReasoningStep],
                  extra_context: str) -> tuple[str, float]:
        draft, confidence = self._cot(goal, trace, extra_context)
        for _round in range(_MAX_CRITIQUE_ROUNDS):
            prompt = (
                "You are a harsh reviewer. Find every error, weak step, "
                "unjustified claim, or missing case in the reasoning below. "
                "If it is genuinely sound, say so.\n\n"
                f"QUESTION:\n{goal}\n\n"
                f"REASONING + ANSWER:\n{draft}\n\n"
                'Respond with JSON ONLY: {"passed": <true|false>, '
                '"flaws": ["<specific flaw>", ...]}')
            data = _extract_json(self._llm(prompt))
            if isinstance(data, dict) and data.get("passed") is True \
                    and not data.get("flaws"):
                self._step(trace, "critique", "reviewer: passed — no flaws",
                           max(confidence, 0.7))
                confidence = max(confidence, 0.7)
                return draft, confidence
            flaws = [str(f)[:200] for f in
                     (data.get("flaws") if isinstance(data, dict)
                      else [])][:5]
            if not flaws:
                # unparsable or vacuous critique: keep the draft
                self._step(trace, "critique",
                           "reviewer verdict unparsable — keeping draft",
                           0.5)
                return draft, confidence
            self._step(trace, "critique",
                       "flaws: " + " | ".join(flaws[:4]), 0.5)
            prompt = (
                f"Revise the answer. Address EVERY flaw listed.\n\n"
                f"QUESTION:\n{goal}\n\n"
                f"CURRENT ANSWER:\n{draft}\n\n"
                "FLAWS TO FIX:\n" + "\n".join(f"- {f}" for f in flaws) +
                "\n\nRespond in EXACTLY this format:\n"
                "ANSWER: <revised answer>\nCONFIDENCE: <0.0-1.0>")
            raw = self._llm(prompt)
            revised, new_conf = self._parse_cot_reply(raw, trace)
            if revised:
                draft, confidence = revised, new_conf
        self._step(trace, "note",
                   "critique rounds exhausted — returning last revision",
                   confidence)
        return draft, confidence

    def _tree(self, goal: str, trace: list[ReasoningStep], depth: int,
              extra_context: str) -> tuple[str, float]:
        ctx = f"\nRelevant context:\n{extra_context}" if extra_context else ""
        prompt = (
            f"Plan the first move to solve this, and sketch the main "
            f"branches.{ctx}\n\n"
            f"PROBLEM:\n{goal}\n\n"
            'Respond with JSON ONLY: {"plan": "<first concrete step>", '
            '"branches": [{"action": "<branch>", "expected": "<what it '
            'yields>"}]}  (2-3 branches)')
        data = _extract_json(self._llm(prompt))
        if not (isinstance(data, dict) and data.get("branches")):
            self._step(trace, "note",
                       "plan unparsable — falling back to a chain", 0.4)
            return self._cot(goal, trace, extra_context)
        plan = str(data.get("plan", "")).strip()[:300]
        branches = [b for b in data.get("branches", [])
                    if str(b.get("action", "")).strip()
                    ][:_MAX_BRANCHES]
        self._step(trace, "plan", f"plan: {plan} — "
                                  f"{len(branches)} branches", 0.6)
        scored: list[tuple[dict, float, str]] = []
        for i, branch in enumerate(branches, 1):
            action = str(branch["action"]).strip()[:300]
            self._step(trace, "action", f"branch {i}: {action}", 0.5)
            observation = ""
            if self.tools is not None:
                prompt = (
                    "Express this branch as ONE concrete tool call that "
                    "advances the problem. Use a real tool name only.\n\n"
                    f"PROBLEM:\n{goal}\n\nBRANCH: {action}\n\n"
                    'Respond with JSON ONLY: {"tool": "<tool name or '
                    'empty>", "args": {"<param>": "<string value>", ...}}')
                data = _extract_json(self._llm(prompt))
                tool_name = str((data or {}).get("tool", "")).strip() \
                    if isinstance(data, dict) else ""
                args = {str(k): str(v)[:400] for k, v in
                        ((data or {}).get("args") or {}).items()
                        if isinstance(data, dict)} \
                    if isinstance(data, dict) else {}
                if not tool_name:
                    viable, value = False, 0.0
                    observation = "branch has no matching tool — " \
                                  "left unevaluated"
                else:
                    try:
                        observation = str(self.tools(
                            tool=tool_name, **args))[:400]
                        viable, value = True, 0.5  # it ran; that's evidence
                    except Exception as exc:  # noqa: BLE001 — branch dies
                        viable, value = False, 0.1
                        observation = f"error: {type(exc).__name__}: {exc}"
                self._step(trace, "observation", observation[:240], 0.5)
            else:
                prompt = (
                    "Judge this branch: does it meaningfully advance the "
                    "problem? Be strict.\n\n"
                    f"PROBLEM:\n{goal}\n\n"
                    f"BRANCH: {action}\nEXPECTED: "
                    f"{str(branch.get('expected', ''))[:200]}\n\n"
                    'Respond with JSON ONLY: {"viable": <true|false>, '
                    '"value": <0.0-1.0>, "reason": "<one line>"}')
                data = _extract_json(self._llm(prompt))
                if isinstance(data, dict):
                    value = _confidence(data.get("value"), 0.3)
                    viable = bool(data.get("viable", value > 0.4))
                    reason = str(data.get("reason", ""))[:160]
                    observation = f"viable={viable} value={value:.2f} " \
                                  f"({reason})"
                else:
                    value, viable, observation = 0.3, True, "unparsed"
            scored.append((branch, 0.0 if not viable else value, observation))
        usable = [s for s in scored if s[1] > 0]
        if not usable:
            self._step(trace, "note", "no viable branch — falling back "
                                      "to a chain", 0.3)
            return self._cot(goal, trace, extra_context)
        branch, value, obs = max(usable, key=lambda s: s[1])
        self._step(trace, "observation",
                   f"chosen branch: {str(branch.get('action', ''))[:200]} "
                   f"(value {value:.2f})", value)
        if depth > 1:
            sub_goal = (f"{goal}\n\nYou already established: "
                        f"{obs[:300]}\nContinue from there.")
            return self._tree(sub_goal, trace, depth - 1, extra_context)
        prompt = (
            f"Complete the answer following the chosen branch.\n\n"
            f"PROBLEM:\n{goal}\n\n"
            f"ESTABLISHED: {obs[:400]}\n\n"
            "Respond in EXACTLY this format:\n"
            "ANSWER: <final answer>\nCONFIDENCE: <0.0-1.0>")
        raw = self._llm(prompt)
        answer, confidence = self._parse_cot_reply(raw, trace)
        self._step(trace, "verdict", answer[:300], confidence)
        return answer, confidence


class _BudgetExhausted(Exception):
    pass


# ── built-in eval (the seed of the agent benchmark) ─────────────────────────


_EVAL_TASKS: list[dict[str, Any]] = [
    {"goal": "What is 17*24 + 8*6? Show each operation before the total.",
     "strategy": "cot",
     "expect": lambda a: "432" in a and "48" in a},
    {"goal": "All bloops are razzies. No razzies are lazzies. Can any "
             "bloops be lazzies? Answer yes or no and justify.",
     "strategy": "cot",
     "expect": lambda a: re.search(r"\bno\b", a.lower())
     and "razz" in a.lower()},
    {"goal": "Plan how to back up a 50GB production database when the "
             "target server has only 1GB of free disk, then verify the "
             "backup is usable.",
     "strategy": "decompose",
     "expect": lambda a: len(a) > 80 and
     re.search(r"(compress|chunk|split|transfer|remote)", a.lower())},
    {"goal": "Review this claim for flaws: 'I always lie, so this very "
             "statement must be false.' What is actually wrong with it?",
     "strategy": "critique",
     "expect": lambda a: re.search(r"paradox|contradict|liar|self",
                                   a.lower())},
    {"goal": "Which of these comes last in a healthy release: code "
             "review, merge, staging test, deploy to production?",
     "strategy": "cot",
     "expect": lambda a: re.search(r"deploy|production", a.lower())},
    {"goal": "Why did the egress check report direct when the proxy pool "
             "was alive? Most likely causes, ranked.",
     "strategy": "hypothesize",
     "expect": lambda a: len(a) > 40},
]


def reasoning_eval(context: Any, *, limit: int = 0) -> dict[str, Any]:
    """Run the built-in reasoning eval against the active model.

    This is the seed of the agent benchmark: a fixed task set with
    predicates, scored 0-1, stable across runs — the promotion gate
    can consume it as the reasoning dimension.
    """
    tasks = _EVAL_TASKS[:limit] if limit else _EVAL_TASKS
    results = []
    for task in tasks:
        started = time.monotonic()
        # fresh engine per task: the budget is per-reasoning, not per-run
        engine = ReasoningEngine(context, max_llm_calls=10, max_seconds=60)
        try:
            res = engine.reason(task["goal"], strategy=task["strategy"])
            ok = bool(task["expect"](res.answer))
            detail = res.answer[:160]
        except Exception as exc:  # noqa: BLE001 — one bad task, on
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        results.append({
            "goal": task["goal"][:100], "strategy": task["strategy"],
            "pass": ok, "seconds": round(time.monotonic() - started, 1),
            "detail": detail,
        })
    passed = sum(1 for r in results if r["pass"])
    return {
        "score": round(passed / len(results), 3) if results else 0.0,
        "passed": passed, "total": len(results), "tasks": results,
    }


# ── the universal pre-flight hook ───────────────────────────────────────────
#
# Every agentic system that *thinks before it acts* (the orchestrator's
# plans, Devon's tool sequences, the coding bot's drafts, the partner's
# answers) gets the same two primitives:
#
#   reasoning_enabled(context, complex_ok=…)  →  should we review now?
#   review_text(context, text, focus=…)       →  flaws, or [] (passed)
#   revise_text(context, text, flaws, focus)  →  the revised artifact
#
# Mode: NM_REASONING_MODE = off | auto | always.  ``auto`` reviews
# non-trivial work only; ``always`` reviews everything.  Power mode
# forces ``always`` and relaxes the budget — the capability is
# automatically available to power mode.


def reasoning_enabled(context: Any, *, complex_ok: bool = False) -> bool:
    """Should this system run its reasoning pre-flight right now?"""
    mode = "auto"
    try:
        mode = str(getattr(context.settings, "reasoning_mode", "auto")
                   or "auto").strip().lower()
    except Exception:  # noqa: BLE001 — bad config must never kill agents
        mode = "auto"
    if mode == "off":
        return False
    if mode == "always":
        return True
    # power mode: reasoning is automatically fully on
    try:
        from .power import power_mode_for
        if power_mode_for(context).active:
            return True
    except Exception:  # noqa: BLE001
        pass
    return bool(complex_ok)


def _hook_budget(context: Any) -> tuple[int, float]:
    """(max_calls, max_seconds) — relaxed in power mode."""
    try:
        from .power import power_mode_for
        if power_mode_for(context).active:
            return 8, 120.0
    except Exception:  # noqa: BLE001
        pass
    return 3, 45.0


def _knowledge_block(context: Any, query: str) -> str:
    """Knowledge-graph context for the query — "" when none/off. Never raises."""
    try:
        from .kg import knowledge_context
        return knowledge_context(context, query)
    except Exception:  # noqa: BLE001
        return ""


def review_text(context: Any, text: str, *, focus: str = "",
                max_calls: int | None = None) -> list[str]:
    """One harsh-reviewer pass over an artifact (a plan, a tool sequence,
    a code draft).  Returns the list of flaws — empty when it passed.

    Never raises: a failed review means "no flaws found", which is the
    conservative outcome (the original artifact stays as-is).
    """
    calls, seconds = _hook_budget(context)
    if max_calls is not None:
        calls = max_calls
    engine = ReasoningEngine(context, max_llm_calls=calls, max_seconds=seconds)
    focus_line = f"Focus: {focus}\n" if focus else ""
    known = _knowledge_block(context, f"{focus} {text[:400]}")
    known_block = f"{known}\n\n" if known else ""
    prompt = (
        "You are a harsh reviewer. Find every real error, weak step, "
        "missing case, or unjustified assumption in the artifact below. "
        "Ignore style; flag only things that would make it fail or be "
        "wrong. If it is genuinely sound, say so.\n\n"
        f"{known_block}{focus_line}"
        f"ARTIFACT:\n{text[:6000]}\n\n"
        'Respond with JSON ONLY: {"passed": <true|false>, '
        '"flaws": ["<specific flaw>", ...]}')
    try:
        raw = engine._llm(prompt)
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return []
        if data.get("passed") is True and not data.get("flaws"):
            return []
        return [str(f).strip()[:200] for f in (data.get("flaws") or [])
                if str(f).strip()][:5]
    except Exception:  # noqa: BLE001 — review must never break the agent
        _log.debug("reasoning review failed", exc_info=True)
        return []


def revise_text(context: Any, text: str, flaws: list[str], *,
                focus: str = "", max_calls: int | None = None) -> str:
    """One revision pass that addresses the listed flaws.

    Returns the revised artifact, or the original text unchanged when
    the revision is unusable.  Never raises.
    """
    if not flaws:
        return text
    calls, seconds = _hook_budget(context)
    if max_calls is not None:
        calls = max_calls
    engine = ReasoningEngine(context, max_llm_calls=calls, max_seconds=seconds)
    focus_line = f"Keep this constraint: {focus}\n" if focus else ""
    known = _knowledge_block(context, f"{focus} {text[:400]}")
    known_block = f"{known}\n\n" if known else ""
    prompt = (
        "Revise the artifact. Address EVERY flaw listed — fix each one, "
        "do not just mention it.\n\n"
        f"{known_block}{focus_line}"
        f"ARTIFACT:\n{text[:6000]}\n\n"
        "FLAWS TO FIX:\n" + "\n".join(f"- {f}" for f in flaws) +
        "\n\nRespond with ONLY the revised artifact, in exactly the same "
        "format as it was given — nothing before or after it.")
    try:
        revised = engine._llm(prompt).strip()
        if revised and len(revised) >= 20:
            return revised
        return text
    except Exception:  # noqa: BLE001
        _log.debug("reasoning revision failed", exc_info=True)
        return text


# ── registry ──────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "reason",
        description=(
            "Explicit multi-strategy reasoning with a full auditable trace: "
            "cot (step chain), decompose (subgoals), hypothesize (evidence-"
            "scored hypotheses), critique (draft/review/revise), tree "
            "(plan-branch-evaluate), auto (classify + dispatch). Optional "
            "tool access for action steps. Returns answer + trace + "
            "confidence."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "goal": "str — the question, problem, or thing to review",
            "strategy": "str (optional, auto) — cot|decompose|hypothesize|critique|tree|auto",
            "depth": "int (optional, 1, max 3) — recursion for decompose/tree",
            "use_tools": "bool (optional, false) — let action steps call tools",
            "context": "str (optional) — extra facts to reason with",
            "show_trace": "bool (optional, true) — include the full trace",
        },
    )
    def reason(
        goal: str,
        *,
        strategy: str = "auto",
        depth: str = "1",
        use_tools: str = "",
        context: str = "",
        show_trace: str = "true",
    ) -> dict[str, Any]:
        def tool_caller(**kwargs: str) -> str:
            outcome = context.tools.call(
                str(kwargs.get("tool") or ""),
                actor="reasoning",
                **{k: v for k, v in kwargs.items() if k != "tool"},
            )
            return json.dumps(outcome.value, default=str) if outcome.ok \
                else f"error: {getattr(outcome.error, 'message', outcome.error)}"

        try:
            d = max(0, min(int(depth or 1), 3))
        except ValueError:
            d = 1
        engine = ReasoningEngine(
            context,
            tools=tool_caller if str(use_tools).lower() in {"1", "true", "yes"}
            else None,
            max_llm_calls=16, max_seconds=180.0)
        result = engine.reason(goal, strategy=strategy, depth=d,
                               extra_context=context)
        out = result.as_dict()
        if str(show_trace).lower() not in {"1", "true", "yes"}:
            out.pop("trace", None)
        out["trace_text"] = trace_text(result)
        return out

    @registry.register(
        "reasoning_eval",
        description=(
            "Run the built-in reasoning eval (fixed task set: arithmetic, "
            "logic, planning, self-review, sequencing, causal ranking) and "
            "score the active model 0-1. Seed of the agent benchmark — "
            "the evolution promotion gate can consume it."
        ),
        capability=Capability.MODEL_CALL,
        parameters={"limit": "int (optional) — only first N tasks"},
    )
    def reasoning_eval_tool(limit: str = "") -> dict[str, Any]:
        try:
            n = int(limit or 0)
        except ValueError:
            n = 0
        return reasoning_eval(context, limit=n)

# ── the permanent reasoning agent (wave 68) ─────────────────────────────────

class ReasoningAgent:
    """The always-on reasoning layer — one object every major decision
    goes through.

    * ``think(text, strategy)`` — a full multi-step trace with the work
      shown (what ``/think`` does, callable from any agent).
    * ``advise(decision, focus)`` — ONE bounded reasoning call: a short,
      sharp piece of advice for a decision that is happening right now
      (a tool plan to run, code stuck on the same error, a mission step
      that failed).  Returns ``""`` when reasoning is off or the budget
      is exhausted — callers always continue.  This is what makes the
      system "think mid-task" instead of only when asked.
    * ``plan_tools(task, catalog)`` — reasoning-assisted tool planning:
      a chain-of-thought step that must end in the strict JSON step
      list.  The fallback the devon planner uses when the raw model
      plan didn't parse, so "planned by heuristic" becomes a last
      resort, not the norm.
    * ``course_correct(outcome, ...)`` — wave 78: the mid-task
      re-evaluation loop.  After a step/attempt outcome, deterministic
      signals (repeated same-family failures, weak skills in play,
      known traps) decide ``should_pivot`` and the model names the
      concrete pivot (different tool, different decomposition, stop).
      The orchestrator and mission runner call this on every failure so
      a weak approach gets corrected while the task is still live.
    * ``self_challenge(conclusion, ...)`` — wave 78: adversarial
      falsification of the system's own previous conclusions.  One
      bounded red-team pass per conclusion (plus deterministic
      overconfidence checks); contested conclusions are journaled so
      later reasoning knows what the system is NOT sure about.
    * ``traces(limit)`` — the persisted, inspectable record of full
      reasoning traces (the system's thinking on audit).
    """

    def __init__(self, context: Any, *, max_calls: int = 0,
                 max_seconds: float = 0.0) -> None:
        self.context = context
        calls, seconds = _hook_budget(context)
        self.engine = ReasoningEngine(
            context,
            max_llm_calls=max_calls or calls,
            max_seconds=max_seconds or seconds,
        )

    # ── full trace ─────────────────────────────────────────────────────────
    def think(self, text: str, *, strategy: str = "auto",
              depth: int = 1, persist: bool = True) -> ReasoningResult:
        """A complete reasoning trace for ``text`` (the /think path).

        The result is journaled (bounded, inspectable via
        ``ReasoningAgent.traces``) so the system's thinking can be
        audited after the fact.
        """
        result = self.engine.reason(text, strategy=strategy, depth=depth)
        if persist:
            self.record_think_trace(text, result)
        return result

    # ── mid-task pre-action thinking ───────────────────────────────────────
    _MIDTASK_KEY = "reasoning.mid_task"
    #: action words that mark an action as hard to undo
    _IRREVERSIBLE_RE = re.compile(
        r"apply|install|commit|push|delete|drop|overwrite|publish|revert",
        re.I)

    def mid_task_check(self, action: str, *, details: str = "",
                       hard_risks: list[str] | None = None,
                       log: bool = True) -> dict[str, Any]:
        """One bounded thinking pass *before* a high-cost action runs —
        the always-on mid-task reasoning the system is missing when a
        worker is about to commit, install, apply, or spend real budget.

        Deterministic risk checklist (no model, never fails):
        * irreversible wording in the action
        * caller-declared hard risks (dirty tree, untested code, …)
        * a recent failure of the same action family in the ledger
        * the action name matching a known systemic skill trap
        Plus ONE model advisory when reasoning is enabled and budget
        remains — ``""`` otherwise.  The trace is appended to a bounded
        kv journal so the system can show *what it thought* before each
        major act.

        Returns ``{"action", "risks": [...], "advice": str, "proceed":
        bool}``.  ``proceed`` is False only on hard risks — advisory
        reasoning never blocks; the safety contracts do.
        """
        action = (action or "").strip()
        risks: list[str] = []
        if self._IRREVERSIBLE_RE.search(action):
            risks.append("action is hard to undo — verify the after-state "
                         "plan before starting")
        for r in (hard_risks or []):
            if r:
                risks.append(r)
        # recent failure of the same family?
        try:
            db = getattr(self.context, "db", None)
            if db is not None:
                head = re.sub(r"[^a-z0-9]+", " ",
                              action.lower())[:40].strip()
                if head:
                    rows = db.query(
                        "SELECT COUNT(*) AS n FROM failures WHERE ts > ? "
                        "AND summary LIKE ?",
                        (time.time() - 3600.0, f"%{head[:24]}%"))
                    n = rows[0]["n"] if rows else 0
                    if n:
                        risks.append(f"same action family failed {n}x in "
                                     "the last hour")
        except Exception:  # noqa: BLE001 - pre-migration-27 db
            pass
        # known systemic trap?  (plus prevention skills mined from
        # repeated pre-action aborts — the memory loop closing back)
        try:
            from .skills import SkillLibrary
            lib = SkillLibrary(getattr(self.context, "db"))
            traps = lib.match_errors(action + " " + details, limit=2)
            for t in traps:
                risks.append(f"known trap: {t.name}")
            prevs = lib.match_preventions(action + " " + details, limit=2)
            for p in prevs:
                hits = self._prevention_hits(p)
                if hits >= self.PREVENTION_HARD_THRESHOLD:
                    # wave 83: hardened memory — this family has aborted
                    # too many times, the warning is now a hard stop
                    risks.append(f"hard risk: repeated abort ({hits}x): "
                                 f"{p.name} — {p.description[:80]}")
                else:
                    risks.append(f"repeated abort: {p.name} — "
                                 f"{p.description[:100]}")
        except Exception:  # noqa: BLE001
            pass
        # one bounded model advisory — a bonus, never a gate
        advice = ""
        try:
            if reasoning_enabled(self.context, complex_ok=True):
                advice = self.advise(
                    f"About to run: {action[:400]}"
                    + (f" Context: {details[:300]}" if details else ""),
                    focus="ONE risk to double-check before proceeding "
                          "(or 'proceed' if none)")
        except Exception:  # noqa: BLE001
            advice = ""
        proceed = not any(r.startswith(("hard risk:")) for r in risks)
        out = {"action": action[:200], "risks": risks[:8],
               "advice": advice[:600], "proceed": proceed}
        if log:
            try:
                db = getattr(self.context, "db", None)
                if db is not None:
                    row = db.query_one(
                        "SELECT value FROM kv_store WHERE key=?",
                        (self._MIDTASK_KEY,))
                    journal: list[dict[str, Any]] = []
                    if row:
                        try:
                            journal = json.loads(row["value"])
                        except ValueError:
                            journal = []
                    journal.append({"ts": time.time(), **out})
                    journal = journal[-100:]
                    db.execute(
                        "INSERT INTO kv_store (key, value, updated_at) "
                        "VALUES (?,?,?) "
                        "ON CONFLICT(key) DO UPDATE SET "
                        "value=excluded.value, updated_at=excluded.updated_at",
                        (self._MIDTASK_KEY, json.dumps(journal),
                         time.time()))
            except Exception:  # noqa: BLE001 - journal is a bonus
                pass
        # wave 82: an abort is evidence — after it is journaled, mine the
        # journal into prevention skills so the repeated family becomes
        # durable memory (best-effort)
        if not proceed:
            try:
                self.mine_preventions()
            except Exception:  # noqa: BLE001 - memory never blocks the check
                pass
        return out

    def mid_task_journal(self, limit: int = 20) -> list[dict[str, Any]]:
        """The last pre-action thinking passes (oldest first)."""
        try:
            db = getattr(self.context, "db", None)
            if db is None:
                return []
            row = db.query_one("SELECT value FROM kv_store WHERE key=?",
                               (self._MIDTASK_KEY,))
            if not row:
                return []
            return json.loads(row["value"])[-max(1, limit):]
        except Exception:  # noqa: BLE001
            return []

    # ── mid-task advice ────────────────────────────────────────────────────
    def advise(self, decision: str, *, focus: str = "",
               max_tokens: int = 400) -> str:
        """One bounded reasoning pass over a decision in progress.

        The reply is a short, direct recommendation (a few sentences,
        or an exact JSON block when ``focus`` asks for a format).
        ``""`` when reasoning is off, the model is down, or the budget
        is exhausted — advice is a bonus, never a gate.
        """
        decision = (decision or "").strip()
        if not decision:
            return ""
        try:
            if not reasoning_enabled(self.context, complex_ok=True):
                return ""
            prompt = (f"Deciding right now: {decision[:1200]}\n")
            if focus:
                prompt += (f"Give ONE direct recommendation that fixes "
                           f"this: {focus[:300]}\n")
            else:
                prompt += ("Give ONE direct, concrete recommendation "
                           "(2-4 sentences). No preamble.\n")
            text = self.engine._llm(prompt, temperature=0.1)
            return (text or "").strip()[:max_tokens * 4]
        except _BudgetExhausted:
            return ""
        except Exception as exc:  # noqa: BLE001 — advice must never break a flow
            _log.debug("reasoning advise failed: %s", exc)
            return ""

    # ── reasoning-assisted tool planning ───────────────────────────────────
    def plan_tools(self, task: str, catalog: list[tuple[str, str]],
                   limit: int = 8) -> list[dict[str, Any]]:
        """Plan a tool sequence WITH a reasoning trace behind it.

        Chain-of-thought first (pick the tools that actually answer the
        task, in an order that can work), then a strict JSON contract.
        Returns [] when reasoning is off/unavailable or the model
        couldn't produce valid known tools — the caller falls back to
        its own heuristic, which is now a LAST resort.
        """
        if not task or not catalog:
            return []
        try:
            if not reasoning_enabled(self.context, complex_ok=True):
                return []
        except Exception:  # noqa: BLE001
            return []
        names = {name for name, _ in catalog}
        rendered = "\n".join(f"- {name}: {desc}" for name, desc in catalog)
        prompt = (
            "Plan the tool calls that answer this task. Think step by "
            "step first: which tools actually fit, in an order that can "
            "really work. Then give the plan.\n"
            f"Available tools:\n{rendered}\n\n"
            f"Task: {task[:1000]}\n\n"
            "End your reply with ONLY JSON of the form "
            '{"steps":[{"tool":"<name>","args":{...},"why":"<short>"}]}. '
            "Use only tools from the list above."
        )
        try:
            raw = self.engine._llm(prompt, temperature=0.1)
        except _BudgetExhausted:
            return []
        except Exception as exc:  # noqa: BLE001
            _log.debug("reasoning plan_tools failed: %s", exc)
            return []
        data = _extract_json(raw)
        if not isinstance(data, dict):
            return []
        raw_steps = data.get("steps")
        if not isinstance(raw_steps, list):
            return []
        cleaned: list[dict[str, Any]] = []
        for item in raw_steps:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            if tool not in names:
                continue
            args = item.get("args")
            if not isinstance(args, dict):
                args = {}
            cleaned.append({"tool": tool, "args": args,
                            "why": str(item.get("why") or "reasoned")})
        return cleaned[: max(1, limit)]

    # ── wave 78: outcome evaluation + course correction ────────────────────
    _TRACE_KEY = "reasoning.traces"
    _CHALLENGE_KEY = "reasoning.challenges"
    _OVERCONFIDENT_RE = re.compile(
        r"\b(definitely|certainly|guaranteed|guarantee|always|never|"
        r"100%|no doubt|without question)\b", re.I)

    def _journal(self, key: str, entry: dict[str, Any],
                 bound: int) -> None:
        """Append ``entry`` to the bounded kv journal ``key`` (best-effort)."""
        try:
            db = getattr(self.context, "db", None)
            if db is None:
                return
            row = db.query_one("SELECT value FROM kv_store WHERE key=?",
                               (key,))
            journal: list[dict[str, Any]] = []
            if row:
                try:
                    journal = json.loads(row["value"])
                except ValueError:
                    journal = []
            journal.append(entry)
            journal = journal[-bound:]
            db.execute(
                "INSERT INTO kv_store (key, value, updated_at) "
                "VALUES (?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "value=excluded.value, updated_at=excluded.updated_at",
                (key, json.dumps(journal), time.time()))
        except Exception:  # noqa: BLE001 - journals are a bonus
            pass

    @classmethod
    def _read_journal(cls, context: Any, key: str,
                     limit: int = 50) -> list[dict[str, Any]]:
        try:
            db = getattr(context, "db", None)
            if db is None:
                return []
            row = db.query_one("SELECT value FROM kv_store WHERE key=?",
                               (key,))
            if not row:
                return []
            data = json.loads(row["value"])
            return data[-max(1, limit):] if isinstance(data, list) else []
        except Exception:  # noqa: BLE001
            return []

    def record_think_trace(self, text: str, result: ReasoningResult) -> None:
        """Persist one full reasoning trace (bounded, inspectable)."""
        self._journal(self._TRACE_KEY, {
            "ts": time.time(),
            "text": (text or "")[:300],
            "strategy": getattr(result, "strategy", ""),
            "answer": (getattr(result, "answer", "") or "")[:500],
            "confidence": getattr(result, "confidence", None),
            "steps": len(getattr(result, "trace", None) or []),
        }, bound=50)

    def traces(self, limit: int = 20) -> list[dict[str, Any]]:
        """The recent full reasoning traces (newest last)."""
        return self._read_journal(self.context, self._TRACE_KEY,
                                  limit=limit)

    def record_outcome(self, scope: str, task: str, ok: bool,
                       detail: str = "") -> None:
        """Feed a completed step/attempt outcome back into the failure
        ledger so course correction can see the pattern.  Best-effort —
        bookkeeping never blocks the task."""
        if ok:
            return
        try:
            from .failure import FailureAnalyzer
            FailureAnalyzer(self.context).record(scope, task, detail or "failed")
        except Exception:  # noqa: BLE001
            pass

    def course_correct(self, outcome: str, *,
                       scope: str = "agent",
                       attempts: int = 1) -> dict[str, Any]:
        """Mid-task re-evaluation: is the current approach weak or wrong,
        and if so, what exactly to do differently.

        Deterministic signals (no model, always run):
        * the same outcome/failure family repeated within the last hour
          in the failure ledger (``attempts`` >= 2 or a ledger hit)
        * a skill matching the work that is underperforming (used >= 5x,
          success < 34%) — the system's own experience says this
          approach is weak
        * a known systemic trap matching the outcome text
        Then ONE bounded model pass names the pivot (different tool,
        different decomposition, different source, or stop) — ``""``
        when reasoning is off or the budget is exhausted, in which case
        ``should_pivot`` is driven by the signals alone.

        Returns ``{"outcome", "signals": [...], "should_pivot": bool,
        "pivot": str, "rationale": str}``.  Never raises.
        """
        outcome = (outcome or "").strip()
        if not outcome:
            return {"outcome": "", "signals": [], "should_pivot": False,
                    "pivot": "", "rationale": "no outcome given"}
        signals: list[str] = []
        # 1) repeated same-family failures
        family_hits = 0
        try:
            db = getattr(self.context, "db", None)
            if db is not None:
                head = re.sub(r"[^a-z0-9]+", " ", outcome.lower())
                head = " ".join(head.split())[:40].strip()
                if head:
                    rows = db.query(
                        "SELECT COUNT(*) AS n FROM failures WHERE ts > ? "
                        "AND summary LIKE ?",
                        (time.time() - 3600.0, f"%{head[:24]}%"))
                    family_hits = rows[0]["n"] if rows else 0
        except Exception:  # noqa: BLE001
            family_hits = 0
        if family_hits or attempts >= 2:
            why = (f"failed {family_hits}x in the last hour"
                   if family_hits else f"attempt {attempts} of the "
                                       "same approach failed again")
            signals.append(f"approach is not working: {why}")
        # 2) weak skill in play
        try:
            from .skills import SkillLibrary
            lib = SkillLibrary(getattr(self.context, "db"))
            for skill in lib.recall(outcome, limit=3):
                uses = int(getattr(skill, "uses", 0) or 0)
                wins = int(getattr(skill, "success_count", 0) or 0)
                if uses >= 5 and wins / uses < 0.34:
                    signals.append(
                        f"skill '{getattr(skill, 'name', '?')}' has a "
                        f"{wins}/{uses} record — this approach is "
                        f"historically weak")
                    break
        except Exception:  # noqa: BLE001
            pass
        # 3) known trap
        try:
            from .skills import SkillLibrary
            for t in SkillLibrary(getattr(self.context, "db")).match_errors(
                    outcome, limit=1):
                signals.append(f"known trap: {t.name}")
        except Exception:  # noqa: BLE001
            pass
        should_pivot = bool(signals)
        # 4) one bounded model pivot
        pivot = ""
        rationale = ""
        try:
            if reasoning_enabled(self.context, complex_ok=True):
                prompt = (
                    f"The current approach just failed again: "
                    f"{outcome[:900]}\n")
                if signals:
                    prompt += (
                        "Signals from the system's own ledger:\n- "
                        + "\n- ".join(signals[:4]) + "\n")
                prompt += (
                    "Is the current approach itself the problem? If yes, "
                    "give ONE concrete pivot: a different tool, a "
                    "different decomposition, a different source, or "
                    "'stop: <reason>'. One short line, no preamble. "
                    "If the approach is fine and this was a transient "
                    "failure, reply exactly: proceed")
                raw = (self.engine._llm(prompt, temperature=0.1) or "").strip()
                if raw:
                    if raw.lower().startswith("proceed"):
                        pivot = "proceed (transient failure)"
                        rationale = "model judged the approach sound"
                    else:
                        pivot = raw[:300]
                        rationale = "model red-team of the approach"
                        should_pivot = True
        except _BudgetExhausted:
            pass
        except Exception as exc:  # noqa: BLE001
            _log.debug("reasoning course_correct failed: %s", exc)
        # deterministic pivot: when the signals say the approach is weak
        # but no model is available to name the correction, the system
        # still changes course (heuristic pivot), not just complains.
        if should_pivot and not pivot:
            pivot = ("change approach: " + signals[0] +
                     " — try a different tool, decomposition, or source")
            rationale = rationale or "deterministic pivot from ledger " \
                                    "signals (no model available)"
        out = {
            "outcome": outcome[:200],
            "signals": signals[:6],
            "should_pivot": should_pivot,
            "pivot": pivot,
            "rationale": rationale or
                         ("; ".join(signals[:2]) if signals else "no signals"),
            "attempts": attempts,
        }
        self._journal(self._CHALLENGE_KEY, {
            "ts": time.time(), "kind": "course_correct",
            "should_pivot": should_pivot, "pivot": pivot[:120],
            "outcome": outcome[:160],
        }, bound=100)
        return out

    # ── wave 78: self-challenge of previous conclusions ─────────────────────
    def self_challenge(self, conclusion: str, *, evidence: str = "",
                       context: str = "") -> dict[str, Any]:
        """Adversarially attack the system's own previous conclusion.

        One bounded red-team pass: the model is asked to FALSIFY
        ``conclusion`` — what would prove it wrong, what is missing,
        where is the weakest link.  Plus deterministic overconfidence
        checks (absolute claims with no cited evidence).  The result is
        journaled so later reasoning knows which conclusions are
        contested.  ``contested`` is True when the attack lands or the
        conclusion is overconfident without evidence — the caller
        should re-verify before acting on a contested conclusion.

        Returns ``{"conclusion", "attacks": [...], "contested": bool,
        "verdict": str}``.  Never raises.
        """
        conclusion = (conclusion or "").strip()
        if not conclusion:
            return {"conclusion": "", "attacks": [], "contested": False,
                    "verdict": "no conclusion given"}
        attacks: list[str] = []
        # deterministic overconfidence check
        if self._OVERCONFIDENT_RE.search(conclusion) and not evidence.strip():
            attacks.append("absolute wording with no cited evidence — "
                           "treat as unverified until checked")
        attack = ""
        try:
            if reasoning_enabled(self.context, complex_ok=True):
                prompt = (
                    "You are challenging your own previous conclusion. "
                    "Try hard to FALSIFY it: what observation would prove "
                    "it wrong, what is the weakest assumption, what "
                    "evidence is missing?\n"
                    f"Conclusion: {conclusion[:900]}\n")
                if evidence:
                    prompt += f"Evidence cited: {evidence[:600]}\n"
                if context:
                    prompt += f"Surrounding context: {context[:400]}\n"
                prompt += (
                    "Give 1-3 sharp challenges, one per line, each "
                    "starting with '- '. If the conclusion survives your "
                    "attack, end with 'holds' on the last line.")
                raw = (self.engine._llm(prompt, temperature=0.3) or "").strip()
                for line in raw.splitlines():
                    line = line.strip()
                    if line.lower().startswith("-"):
                        attacks.append(line[1:].strip()[:200])
                    elif line.lower() == "holds":
                        attack = "holds under challenge"
                if raw and not attacks and "holds" not in attack:
                    attacks.append(raw[:300])
        except _BudgetExhausted:
            pass
        except Exception as exc:  # noqa: BLE001
            _log.debug("reasoning self_challenge failed: %s", exc)
        contested = bool(attacks) and "holds under challenge" not in attack
        verdict = (attack or
                   ("contested: " + attacks[0][:160] if contested else
                    "no attack landed"))
        out = {
            "conclusion": conclusion[:200],
            "attacks": attacks[:5],
            "contested": contested,
            "verdict": verdict[:300],
        }
        self._journal(self._CHALLENGE_KEY, {
            "ts": time.time(), "kind": "self_challenge",
            "contested": contested, "conclusion": conclusion[:160],
            "attacks": attacks[:3],
        }, bound=100)
        return out

    def challenges(self, limit: int = 20) -> list[dict[str, Any]]:
        """Recent self-challenges + course-correction decisions."""
        return self._read_journal(self.context, self._CHALLENGE_KEY,
                                  limit=limit)

    #: wave 83: a prevention family aborted this many times graduates
    #: from advisory warning to HARD abort — the memory hardens
    PREVENTION_HARD_THRESHOLD = 5

    @staticmethod
    def _prevention_hits(skill: Any) -> int:
        try:
            body = json.loads(skill.body)
            return int(body.get("hits") or 0)
        except Exception:  # noqa: BLE001
            return 0

    # ── wave 82: the audit → lessons loop ──────────────────────────────────
    def mine_preventions(self, *, window_seconds: float = 7 * 86400.0,
                         min_hits: int = 2) -> list[dict[str, Any]]:
        """Close the loop between "what we thought" and "what we prevent".

        The mid-task journal is read for pre-action ABORTS (proceed=False)
        inside the window; an action family that was aborted at least
        ``min_hits`` times becomes a durable ``prevention`` skill — and
        :meth:`~nomorals.agents.skills.SkillLibrary.match_preventions`
        feeds it back into the next pre-action check on a similar action.
        Idempotent: re-mining upserts the same skill with fresher samples.
        Returns the skills written/refreshed.
        """
        from .skills import _STOP

        now = time.time()
        fams: dict[str, list[dict[str, Any]]] = {}
        for entry in self.mid_task_journal(limit=100):
            if entry.get("proceed", True):
                continue
            ts = float(entry.get("ts") or 0)
            if not ts or now - ts > window_seconds:
                continue
            # family key: first four meaningful tokens — rewordings of
            # the same action ("drop the users table" vs
            # "drop the users table now") land in one family
            words = re.sub(r"[^a-z0-9]+", " ",
                           str(entry.get("action", "")).lower()).split()
            words = [w for w in words if w and not _STOP.fullmatch(w)]
            head = " ".join(words[:4])
            if not head:
                continue
            fams.setdefault(head, []).append(entry)
        from .skills import SkillLibrary
        lib = SkillLibrary(getattr(self.context, "db"))
        written: list[dict[str, Any]] = []
        for head, group in sorted(fams.items()):
            if len(group) < min_hits:
                continue
            risks: list[str] = []
            for e in group:
                for r in (e.get("risks") or [])[:4]:
                    if r not in risks:
                        risks.append(r)
            name = f"prevent:{head[:48]}"
            tier = ""
            if len(group) >= self.PREVENTION_HARD_THRESHOLD:
                tier = " — HARD TIER: hard-aborts similar actions"
            skill = lib.save(
                name, kind="prevention", source="reasoning.mine",
                description=(f"Pre-action abort repeated {len(group)}x: "
                             f"{head}" + tier),
                body=json.dumps({
                    "hits": len(group),
                    "mined_at": now,
                    "risks": risks[:8],
                    "samples": [str(e.get("action", ""))[:160]
                                for e in group[:5]],
                }, ensure_ascii=False))
            written.append(skill.to_dict())
        return written

