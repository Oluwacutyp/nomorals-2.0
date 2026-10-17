"""Thinks like a human — common sense, intuitive physics, causal chains,
mental simulation, goal-directed planning with backtracking, and learning
from failed attempts. Not a slogan: each is real logic used by the core.
"""

from __future__ import annotations

import re

# Intuitive physics: the rules a human uses without thinking.
INTUITIVE_PHYSICS = {
    "gravity": "Unsupported objects fall. Support everything, or it falls.",
    "liquids": "Liquids spill, flow downhill, find leaks, and evaporate. Seal, slope, and contain.",
    "heat": "Heat flows hot -> cold. Hot things cool, cold things warm, insulation only slows it.",
    "friction": "Friction resists motion and wears surfaces. Moving parts need clearance, lubrication, or replacement.",
    "leverage": "Leverage multiplies force. Long handles, gears, and pulleys trade distance for force.",
    "pressure": "Pressure acts everywhere on a surface. Seals fail at the weakest point, not the average.",
    "breakage": "Brittle things snap; ductile things bend. Design for the failure mode, not just the load.",
    "wear": "Everything that moves or rubs wears. Parts that wear must be replaceable.",
    "electricity": "Electricity needs a complete circuit. A break anywhere stops it; a short bypasses the load.",
    "humans": "Humans forget, misread, get tired, and improvise. Design for the tired human, not the ideal one.",
}

_COMMON_SENSE = [
    (re.compile(r"\b(water|liquid|fluid|tank|pipe|leak)\b", re.I), "liquids"),
    (re.compile(r"\b(hot|heat|oven|stove|fire|burn)\b", re.I), "heat"),
    (re.compile(r"\b(heavy|lift|support|hold|hang|load)\b", re.I), "gravity"),
    (re.compile(r"\b(rub|move|slide|grind|wheel|bearing)\b", re.I), "friction"),
    (re.compile(r"\b(lever|handle|gear|pulley|crank)\b", re.I), "leverage"),
    (re.compile(r"\b(seal|pressure|inflate|pump|vacuum)\b", re.I), "pressure"),
    (re.compile(r"\b(glass|ceramic|brittle|snap|crack)\b", re.I), "breakage"),
    (re.compile(r"\b(wire|battery|motor|circuit|plug|shock)\b", re.I), "electricity"),
    (re.compile(r"\b(wear|replace|maintain|service)\b", re.I), "wear"),
    (re.compile(r"\b(user|person|child|elderly|operator)\b", re.I), "humans"),
]


def common_sense(text: str) -> list[dict]:
    """Apply intuitive physics / common sense to a problem statement."""
    out, seen = [], set()
    for pattern, key in _COMMON_SENSE:
        if pattern.search(text or "") and key not in seen:
            seen.add(key)
            out.append({"rule": key, "applies": INTUITIVE_PHYSICS[key]})
    return out


def mental_simulation(action: str, context: str = "") -> dict:
    """Mental simulation: 'what happens if I do X?' — causal chain, 3 steps."""
    steps = [
        f"Do: {action}.",
        "Immediate: the direct physical/software effect happens (force moves, heat flows, code runs, data changes).",
        "Next: second-order effects — what the first effect pushes on (load shifts, temperature changes, downstream code/data).",
        "Then: what breaks or wears first — the weakest link, the seal, the edge case, the tired human.",
    ]
    risks = []
    a = (action or "").lower()
    if any(w in a for w in ("heat", "burn", "weld", "oven", "fire")):
        risks.append("Heat: burns, warping, nearby materials melting — keep a heat path / distance.")
    if any(w in a for w in ("lift", "hang", "support", "heavy")):
        risks.append("Gravity: if support fails, it falls — design the failure to fall safely, not onto a person.")
    if any(w in a for w in ("water", "liquid", "pipe", "tank")):
        risks.append("Liquids: leaks find the lowest point — contain and slope to a drain.")
    if any(w in a for w in ("delete", "remove", "cut")):
        risks.append("Deletion: something depended on the removed part — check what breaks when it is gone.")
    if any(w in a for w in ("automate", "script", "auto")):
        risks.append("Automation: automating a wrong step repeats the error at speed — verify the step first.")
    return {"action": action, "context": context, "chain": steps,
            "risks": risks or ["No specific risk flagged — check the weakest link explicitly."],
            "verdict": "Simulate before building: if the chain ends in breakage, injury, or data loss, change the action."}


def plan_with_backtracking(goal: str, steps: list[str]) -> dict:
    """Goal-directed plan with backtracking: each step has a failure branch.
    When a step fails, backtrack to the last working step and try the alternative,
    instead of pushing forward on a broken plan."""
    plan = []
    for i, step in enumerate(steps):
        plan.append({
            "step": i + 1, "do": step,
            "if_fails": f"Backtrack to step {i} (last working state). Do NOT push forward. Try the alternative: simplify the step, split it, or replace its mechanism.",
            "checkpoint": f"After this step, verify: did '{step}' actually happen? Evidence, not belief.",
        })
    return {"goal": goal, "steps": plan,
            "rule": "A plan is a chain of verifiable steps. A failed step means backtrack, not retry harder."}


def learn_from_failure(ledger_entry: dict | None) -> str:
    """Learning from failed attempts: the ledger's lessons feed the next attempt."""
    if not ledger_entry:
        return "No history — run the cheapest killing experiment first."
    attempts = ledger_entry.get("attempts", [])
    if not attempts:
        return "No attempts yet — run the cheapest killing experiment first."
    lessons = [a.get("lesson", "") for a in attempts if a.get("lesson")]
    return ("Learned from failed attempts — do not repeat: " + " | ".join(lessons)
            if lessons else "Attempts exist but no lessons recorded — record the lesson before the next attempt.")
