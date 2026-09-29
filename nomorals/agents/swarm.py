"""Multi-agent swarm: parallel sub-investigators with a synthesis step.

A swarm splits one goal into N subtasks, runs each on its own Devon agent in
a thread (real parallelism — the devon loop is I/O- and model-bound), then
fuses the digests into one answer.

Decomposition is model-driven when a real model is answering (the same
decompose contract the search engine uses); otherwise it falls back to
heuristic splits ("A and B", "A, then B") and, failing that, to
perspective workers — evidence / risks / plan — so a swarm is still useful
offline.

Coordination:
* each worker gets its own DevonAgent (own journal, own run id — no shared
  mutable state to lock);
* a wall-clock budget caps the whole swarm and each worker;
* a crashed worker is reported as a failed leg, never fatal to the rest;
* the synthesis is a single model call over the digests, with a plain
  concatenation fallback.

``/swarm <goal> [workers]`` from chat, and Devon's own ``swarm`` tool for
recursive delegation.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from .devon import DevonAgent, DevonResult

_log = get_logger(__name__)

__all__ = ["SwarmAgent", "SwarmResult"]

_PERSPECTIVES = (
    "evidence and facts",
    "risks, unknowns, and failure modes",
    "concrete step-by-step plan",
)


@dataclass
class SwarmResult:
    goal: str
    subtasks: list[str]
    legs: list[dict[str, Any]] = field(default_factory=list)
    synthesis: str = ""
    seconds: float = 0.0
    run_id: str = field(default_factory=lambda: new_short_id("swarm"))

    @property
    def ok(self) -> bool:
        return any(leg.get("ok") for leg in self.legs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "subtasks": self.subtasks,
            "legs": self.legs,
            "synthesis": self.synthesis,
            "seconds": round(self.seconds, 2),
            "run_id": self.run_id,
            "ok": self.ok,
        }


class SwarmAgent:
    """Plan → parallel devon legs → fuse."""

    def __init__(
        self,
        context: Any,
        *,
        brain: Any = None,
        gateway: Any = None,
        wall_seconds: float = 420.0,
    ) -> None:
        self.context = context
        self.brain = brain
        self.gateway = gateway
        self.wall_seconds = float(wall_seconds)

    # ── decomposition ────────────────────────────────────────────────────────
    def _model_available(self) -> bool:
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return False
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001
            return False
        active = str(snap.get("active") or "")
        return bool(active) and active not in {"mock", "offline", "test", ""}

    def decompose(self, goal: str, workers: int) -> tuple[list[str], str]:
        """→ (subtasks, planner name)."""
        if self._model_available():
            try:
                from ..llm.base import Message, SamplingParams

                response = self.context.router.chat(
                    [
                        Message.system(
                            f"Split this goal into at most {workers} parallel subtasks that "
                            "independent investigators can each complete without talking to "
                            "each other. Complementary, non-overlapping, each self-contained. "
                            "Reply with ONLY a JSON array of strings."
                        ),
                        Message.user(goal),
                    ],
                    SamplingParams(temperature=0.2, max_tokens=300),
                )
                text = (response.text or "").strip()
                start, end = text.find("["), text.rfind("]")
                if start != -1 and end > start:
                    subs = [
                        str(x).strip()
                        for x in json.loads(text[start : end + 1])
                        if str(x).strip()
                    ]
                    if subs:
                        return subs[:workers], "llm"
            except Exception as exc:  # noqa: BLE001 - fall through to heuristics
                _log.debug("swarm decomposition failed: %s", exc)

        return self._heuristic_decompose(goal, workers), "heuristic"

    def _heuristic_decompose(self, goal: str, workers: int) -> list[str]:
        """Explicit conjunctions first; else numbered items; else perspectives."""
        parts = [p.strip() for p in re.split(r"\s+(?:and|then|,)\s+(?=[a-z])", goal, flags=re.IGNORECASE) if p.strip()]
        if len(parts) >= 2:
            return parts[:workers]
        numbered = re.split(r"(?:^|\n)\s*\d+[.)]\s*", goal)
        parts = [p.strip() for p in numbered if p.strip()]
        # a preamble ("do these:") survives as part 0 — drop it
        if len(parts) >= 2 and not re.search(r"\d+[.)]", goal.split("\n")[0].strip()[:3]):
            parts = parts[1:]
        if len(parts) >= 2:
            return parts[:workers]
        return [f"investigate the {pers} for: {goal}" for pers in _PERSPECTIVES[:workers]]

    # ── workers ──────────────────────────────────────────────────────────────
    def _run_leg(self, subtask: str, deadline: float) -> dict[str, Any]:
        started = time.time()
        remaining = max(20.0, deadline - started)
        try:
            # step_timeout doubles as the per-leg wall budget (× max_steps
            # inside devon), so scale it to the swarm's remaining time.
            agent = DevonAgent(self.context, brain=self.brain, gateway=self.gateway,
                               max_steps=8, step_timeout=remaining / 8.0)
            result: DevonResult = agent.run(subtask, chat_key="")
            return {
                "subtask": subtask,
                "ok": True,
                "digest": result.digest,
                "planned_by": result.planned_by,
                "steps": len(result.steps),
                "seconds": round(result.seconds, 1),
            }
        except Exception as exc:  # noqa: BLE001 - a dead leg is a result
            _log.exception("swarm leg failed: %r", subtask)
            return {
                "subtask": subtask,
                "ok": False,
                "digest": f"leg failed: {exc}",
                "planned_by": "error",
                "steps": 0,
                "seconds": round(time.time() - started, 1),
            }

    # ── synthesis ────────────────────────────────────────────────────────────
    def _synthesize(self, goal: str, legs: list[dict[str, Any]]) -> tuple[str, bool]:
        if self._model_available():
            try:
                from ..llm.base import Message, SamplingParams

                leg_text = "\n\n".join(
                    f"--- worker {i + 1} ({leg['subtask']}) ---\n{leg['digest']}"
                    for i, leg in enumerate(legs)
                    if leg.get("ok")
                )
                response = self.context.router.chat(
                    [
                        Message.system(
                            "You are fusing the results of a parallel multi-agent swarm "
                            "into one coherent answer. Merge overlaps, keep conflicts "
                            "visible, and lead with the direct answer."
                        ),
                        Message.user(f"Goal: {goal}\n\nWorker results:\n{leg_text[:12000]}"),
                    ],
                    SamplingParams(temperature=0.3, max_tokens=900),
                )
                text = (response.text or "").strip()
                if text:
                    return text, True
            except Exception as exc:  # noqa: BLE001
                _log.debug("swarm synthesis failed; concatenating: %s", exc)
        merged = "\n\n".join(f"• {leg['subtask']}\n  {leg['digest'][:800]}"
                             for leg in legs if leg.get("ok"))
        return (merged or "all swarm legs failed"), False

    # ── the run ──────────────────────────────────────────────────────────────
    def run(self, goal: str, workers: int = 3) -> SwarmResult:
        started = time.time()
        goal = (goal or "").strip()
        if not goal:
            raise ValueError("swarm needs a goal")
        workers = max(1, min(int(workers), 5))
        deadline = started + self.wall_seconds

        subtasks, planner = self.decompose(goal, workers)
        result = SwarmResult(goal=goal, subtasks=subtasks)

        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="swarm")
        futures = [pool.submit(self._run_leg, sub, deadline) for sub in subtasks]
        for i, future in enumerate(futures):
            try:
                leg = future.result(timeout=max(10.0, deadline - time.time()))
            except Exception as exc:  # noqa: BLE001 - a late leg is a result
                leg = {
                    "subtask": subtasks[i], "ok": False,
                    "digest": f"leg timed out: {exc}", "planned_by": "timeout",
                    "steps": 0, "seconds": 0,
                }
            result.legs.append(leg)
        # Don't wait on late legs past the swarm's own deadline: they are
        # bounded by devon's internal step budgets and finish on their own.
        pool.shutdown(wait=False, cancel_futures=True)

        result.synthesis, model_ok = self._synthesize(goal, result.legs)
        result.seconds = time.time() - started
        _log.info(
            "swarm %s done: %d/%d legs ok, %.0fs (%s planner%s)",
            result.run_id, sum(1 for l in result.legs if l["ok"]), len(result.legs),
            result.seconds, planner, ", model-synthesized" if model_ok else "",
        )
        return result
