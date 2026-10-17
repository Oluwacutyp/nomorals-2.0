"""Stage 2 — FORMULATE. The stage everyone skips and where most value hides.

A wish is not a problem. Formulation extracts: goal (function, not form),
current pain, constraints, context, success criteria, and what is
underspecified. Underspecified problems return clarification questions
instead of a guessed answer — that is human-like, not a failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Frames: re-state the same problem in different frames. Formulating in
# several frames is what lets transfer + reframe find the achievable version.
FRAMES = ("function_not_form", "first_principles", "inversion", "trivial_move",
          "analogy", "scale", "user_ache")

_VAGUE = re.compile(r"\b(better|faster|cheaper|good|nice|great|impossible|amazing)\b", re.I)
_QUANTITY = re.compile(r"\d+\s*(?:%|x|times|kg|g|m|km|cm|mm|s|min|h|hr|w|kw|mw|mw|°c|k|n|pa|bar|\$|usd|naira|₦|litres?|liters?|hours?|days?|years?|minutes?)", re.I)
_COST = re.compile(r"\b(cost|budget|price|afford|cheap|expensive|\$|₦|usd|naira)\b", re.I)
_ENV = re.compile(r"\b(lagos|nigeria|humid|heat|grid|power|outage|outdoor|indoor|kitchen|car|house|roof|farm|clinic|school)\b", re.I)
_TIMEFRAME = re.compile(r"\b(by|within|in)\s+\d+\s*(days?|weeks?|months?|years?)\b", re.I)


@dataclass
class Formulation:
    raw: str
    goal_function: str           # what it must DO, not what it must BE
    pain: str                    # the ache underneath the wish
    constraints: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    success_criteria: list[str] = field(default_factory=list)
    frames: dict[str, str] = field(default_factory=dict)
    clarifications: list[str] = field(default_factory=list)
    underspecified: bool = False

    def to_dict(self) -> dict:
        return {"raw": self.raw, "goal_function": self.goal_function, "pain": self.pain,
                "constraints": self.constraints, "context": self.context,
                "success_criteria": self.success_criteria, "frames": self.frames,
                "clarifications": self.clarifications, "underspecified": self.underspecified}


def _strip_wish(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"^(i want|i need|i wish|can you|please|help me|make|build|create|design)\s+", "", t, flags=re.I)
    return t.strip(" .")


def formulate(problem: str) -> Formulation:
    """Turn a wish into a well-posed problem statement + frames."""
    raw = (problem or "").strip()
    if not raw:
        return Formulation(raw="", goal_function="", pain="",
                            clarifications=["What problem should I solve? Describe the goal in your own words."],
                            underspecified=True)
    goal = _strip_wish(raw)
    constraints: list[str] = []
    context: list[str] = []
    clarifications: list[str] = []
    if _COST.search(raw):
        constraints.append("cost-sensitive (budget mentioned or implied)")
    if _ENV.search(raw):
        context.append("environment/context mentioned in the wish")
    if not _QUANTITY.search(raw):
        clarifications.append("What does success look like in numbers — how much, how fast, how cheap, by when?")
    if not _TIMEFRAME.search(raw):
        clarifications.append("Is there a deadline or time horizon?")
    if _VAGUE.search(raw) and not _QUANTITY.search(raw):
        clarifications.append("Which part is the real pain — cost, time, effort, reliability, or something else?")
    if len(raw.split()) < 5:
        clarifications.append("One more detail: who is it for, and where will it be used?")
    underspecified = len(raw.split()) < 5 or len(clarifications) >= 3

    # Frames — each is a real re-statement, not a slogan.
    frames = {
        "function_not_form": f"What must it DO: {goal} — the form (what it looks like / is made of) is negotiable.",
        "first_principles": f"Strip to physics/economics: what energy, mass, information, or money must move, and what is the cheapest way to move it for: {goal}?",
        "inversion": f"Solve the opposite: how would you guarantee {goal} FAILS? Avoid those causes.",
        "trivial_move": f"What would make this trivial — unlimited budget, no regulation, a different material, someone else already solving it? Then ask which of those is actually removable.",
        "analogy": f"Where is something structurally like '{goal}' already solved — in another zone, course, or industry?",
        "scale": f"Does '{goal}' work at bench scale, room scale, house scale, city scale — and where does square-cube / cost / maintenance break it?",
        "user_ache": f"The owner never wants the impossible thing itself — they want the ache underneath: what pain does '{goal}' remove, and is there a cheaper way to remove that pain?",
    }
    pain = f"Underneath '{raw}': the owner wants the outcome without the cost/effort/failure that currently blocks it."
    success = ["Achieves the goal function within stated constraints",
               "Cheaper / more reliable than the current way",
               "Falsifiable: a test exists that would prove it wrong"]
    return Formulation(raw=raw, goal_function=goal, pain=pain, constraints=constraints,
                       context=context, success_criteria=success, frames=frames,
                       clarifications=clarifications, underspecified=underspecified)
