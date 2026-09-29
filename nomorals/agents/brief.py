"""Mission briefing — the prompt/mission structuring sub-agent (wave 68).

Its ONLY job: take a normal request and rewrite it into a maximum-quality
mission specification — a crisp objective, explicit constraints, testable
success criteria, and a step order that can actually be executed.

Why a dedicated agent: a mission given to the runner as raw owner prose
("scrape stuff and send me the files") is planned by whatever model is
active on the day, with all the ambiguity intact.  Structured BEFORE it
runs, it is planned against a spec — and the spec is persisted with the
mission, so anyone (chat, CLI, a future session) can see what "done" was
supposed to mean.

Design rules:
  * ONE bounded model call per refine (temperature 0.2, strict JSON);
  * a deterministic heuristic fallback keeps it working model-less;
  * short requests skip it entirely (no model call to rephrase one line);
  * it NEVER refuses — structuring is a capability, not a gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger
from ..llm.base import LLMResponse, Message, SamplingParams

_log = get_logger(__name__)

__all__ = ["MissionBrief", "BriefAgent", "should_brief"]

#: Requests longer than this (or with several clauses) are worth a brief.
_MIN_BRIEF_CHARS = 80


def should_brief(request: str) -> bool:
    """Is this request worth structuring?  One-line asks are not —
    structuring them would spend a model call to say the same thing."""
    text = (request or "").strip()
    if not text:
        return False
    if len(text) >= _MIN_BRIEF_CHARS:
        return True
    # several explicit clauses in a short sentence still counts
    clauses = re.split(r"(?:, |; | and then | and | then )", text, maxsplit=4)
    return len([c for c in clauses if len(c.strip()) > 12]) >= 3


@dataclass
class MissionBrief:
    """A structured mission specification."""

    objective: str
    constraints: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    persona: str = ""
    tool_hints: list[str] = field(default_factory=list)
    by: str = "heuristic"  # "model" | "heuristic"

    def to_dict(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "constraints": self.constraints,
            "success_criteria": self.success_criteria,
            "steps": self.steps,
            "persona": self.persona,
            "tool_hints": self.tool_hints,
            "by": self.by,
        }

    def as_goal(self) -> str:
        """The structured spec rendered back into ONE goal string for the
        mission runner — richer than the raw ask, still one field."""
        parts = [f"MISSION: {self.objective}"]
        if self.steps:
            parts.append("Steps: " + " → ".join(self.steps[:8]))
        if self.success_criteria:
            parts.append("Done means: " + " | ".join(self.success_criteria[:4]))
        if self.constraints:
            parts.append("Constraints: " + " | ".join(self.constraints[:4]))
        if self.persona:
            parts.append(f"Persona: {self.persona[:160]}")
        return "\n".join(parts)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MissionBrief":
        return cls(
            objective=str(d.get("objective") or ""),
            constraints=[str(c) for c in (d.get("constraints") or [])],
            success_criteria=[str(c) for c in
                              (d.get("success_criteria") or [])],
            steps=[str(s) for s in (d.get("steps") or [])],
            persona=str(d.get("persona") or ""),
            tool_hints=[str(t) for t in (d.get("tool_hints") or [])],
            by=str(d.get("by") or "heuristic"),
        )


class BriefAgent:
    """``refine(request)`` -> MissionBrief.  Model-first, heuristic-backed.

    The model sees a strict JSON contract; anything it misses is backfilled
    by the heuristic, so the output is always a complete spec.
    """

    def __init__(self, context: Any) -> None:
        self.context = context

    def refine(self, request: str, *, kind: str = "mission") -> MissionBrief:
        """Structure one request into a mission spec.

        ``kind`` tunes the prompt ("mission" for background work, "code"
        for build tasks) — the contract is the same.
        """
        request = (request or "").strip()
        brief = self._heuristic_brief(request, kind)
        if not should_brief(request):
            brief.objective = request
            brief.by = "skip"
            return brief
        model_brief = self._model_brief(request, kind)
        if model_brief is not None:
            merged = self._merge(model_brief, brief)
            merged.by = "model"
            return merged
        return brief

    # ── model path ─────────────────────────────────────────────────────────
    def _model_brief(self, request: str, kind: str) -> MissionBrief | None:
        router = getattr(self.context, "router", None)
        if router is None:
            return None
        noun = "mission" if kind == "mission" else "build task"
        system = (
            f"You structure raw owner requests into maximum-quality "
            f"{noun} specifications. Rewrite for EXECUTION: a single crisp "
            "objective (one sentence, no filler), the hard constraints the "
            "work must respect, 2-4 CHECKABLE success criteria (each one "
            "verifiable by a test, a file, or an output — no vibes), 2-6 "
            "concrete steps in executable order, and the tools that fit. "
            "Preserve the owner's intent and any persona exactly; do not "
            "add scope they didn't ask for. Reply with ONLY JSON of the "
            'form {"objective": "...", "constraints": ["..."], '
            '"success_criteria": ["..."], "steps": ["..."], '
            '"persona": "", "tool_hints": ["..."]}.'
        )
        user = f"Raw request ({noun}):\n{request[:2000]}"
        try:
            response: LLMResponse = router.chat(
                [Message.system(system), Message.user(user)],
                SamplingParams(temperature=0.2, max_tokens=700),
            )
        except Exception as exc:  # noqa: BLE001 - structuring is best-effort
            _log.debug("brief model call failed: %s", exc)
            return None
        if not getattr(response, "ok", False) or not getattr(response, "text", ""):
            return None
        data = self._extract_json(response.text)
        if not isinstance(data, dict) or not str(data.get("objective") or "").strip():
            return None
        return MissionBrief(
            objective=str(data.get("objective")).strip()[:400],
            constraints=[str(c).strip() for c in
                         (data.get("constraints") or [])
                         if str(c).strip()][:6],
            success_criteria=[str(c).strip() for c in
                              (data.get("success_criteria") or [])
                              if str(c).strip()][:6],
            steps=[str(s).strip() for s in
                   (data.get("steps") or []) if str(s).strip()][:8],
            persona=str(data.get("persona") or "").strip()[:200],
            tool_hints=[str(t).strip() for t in
                        (data.get("tool_hints") or [])
                        if str(t).strip()][:8],
        )

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        text = (text or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
        return None

    # ── heuristic path ─────────────────────────────────────────────────────
    @staticmethod
    def _heuristic_brief(request: str, kind: str) -> MissionBrief:
        """A complete (if plain) spec, derived deterministically — the
        floor the model path backfills onto."""
        sentences = [s.strip(" .!?") for s in
                     re.split(r"(?<=[.!?])\s+|; |, and |, then ",
                              request) if s.strip(" .!?")]
        objective = sentences[0][:300] if sentences else request[:300]
        rest = sentences[1:]
        steps: list[str] = []
        constraints: list[str] = []
        criteria: list[str] = []
        persona = ""
        for clause in rest:
            low = clause.lower()
            m = re.search(r'persona\s+"([^"]+)"|persona\s*([^.,]+)', low)
            if m and len(clause) < 400:
                persona = m.group(1) or m.group(2)
                continue
            if any(w in low for w in ("must ", "stay true", "no more than",
                                      "at least", "only ", "never")):
                constraints.append(clause[:200])
                continue
            if any(w in low for w in ("then ", "once ", "after ",
                                      "send", "deliver", "verify", "check")):
                if "send" in low or "deliver" in low:
                    criteria.append(clause[:200])
                steps.append(clause[:200])
                continue
            steps.append(clause[:200])
        if not steps:
            steps = ["plan the work", "execute the work", "verify the result"]
        if not criteria:
            criteria = ["the objective is met as stated"]
        if kind == "code":
            criteria = criteria[:4] + ["it runs without errors"]
            criteria = list(dict.fromkeys(criteria))[:6]
        return MissionBrief(
            objective=objective,
            constraints=list(dict.fromkeys(constraints))[:6],
            success_criteria=list(dict.fromkeys(criteria))[:6],
            steps=list(dict.fromkeys(steps))[:8],
            persona=persona[:200],
        )

    @staticmethod
    def _merge(model_brief: MissionBrief, floor: MissionBrief) -> MissionBrief:
        """Model output on top of the heuristic floor: any field the model
        left empty is backfilled so the spec is always complete."""
        return MissionBrief(
            objective=model_brief.objective or floor.objective,
            constraints=model_brief.constraints or floor.constraints,
            success_criteria=model_brief.success_criteria
            or floor.success_criteria,
            steps=model_brief.steps or floor.steps,
            persona=model_brief.persona or floor.persona,
            tool_hints=model_brief.tool_hints,
        )
