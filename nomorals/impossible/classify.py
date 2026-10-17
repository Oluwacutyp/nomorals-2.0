"""Stage 3 — CLASSIFY. Which of the four impossibles, and what it means."""

from __future__ import annotations

import re

from .charter import ECONOMIC, ENGINEERING, INSTITUTIONAL, PHYSICAL

# Signals are evidence, not a verdict — the feasibility gate confirms.
_PHYSICAL = [
    (re.compile(r"perpetual\s+motion|free\s+energy|infinite\s+energy|energy\s+from\s+nothing|create\s+energy", re.I),
     "Claims energy from nothing — conservation of energy."),
    (re.compile(r"faster\s+than\s+light|superluminal|warp|time\s+travel|teleport", re.I),
     "Claims faster-than-light / causality violation — special relativity."),
    (re.compile(r"100%\s*(efficient|efficiency)|over\s*unity|overunity|cop\s*>\s*1\s*without", re.I),
     "Claims >100% efficiency — second law of thermodynamics."),
    (re.compile(r"room.?temp.*superconduct|ambient\s+superconduct", re.I),
     "Ambient superconductor — no known material; would need new physics, not engineering."),
    (re.compile(r"cold\s+fusion|lenr", re.I),
     "Cold fusion — decades of attempts, no reproducible excess heat at scale."),
]
_INSTITUTIONAL = re.compile(r"certif|regulat|permit|licen[cs]e|standard|iso|food.safe|fda|nafdac|son\b|approval|compliance", re.I)
_ECONOMIC = re.compile(r"free|zero\s+cost|no\s+cost|cheapest|pennies|tenth\s+of|1/10", re.I)


def classify(formulation) -> dict:
    """Classify a formulation against the four kinds of impossible."""
    raw = getattr(formulation, "raw", "") or ""
    goal = getattr(formulation, "goal_function", "") or raw
    text = f"{raw} {goal}"
    kinds: list[dict] = []
    for pattern, reason in _PHYSICAL:
        if pattern.search(text):
            kinds.append({"kind": PHYSICAL, "reason": reason, "signal": pattern.pattern})
    if _INSTITUTIONAL.search(text):
        kinds.append({"kind": INSTITUTIONAL, "reason": "Mentions certification/regulation/standards — a path must satisfy the constraint.", "signal": "institutional keywords"})
    if _ECONOMIC.search(text):
        kinds.append({"kind": ECONOMIC, "reason": "Cost is the binding constraint — physically fine, must be made cheap enough.", "signal": "economic keywords"})
    if not kinds:
        kinds.append({"kind": ENGINEERING, "reason": "No physical exclusion detected — engineering-impossible-for-now is the default; the gate confirms.", "signal": "default"})
    else:
        # A problem can be several kinds at once (e.g. economic + institutional).
        if not any(k["kind"] == ENGINEERING for k in kinds):
            kinds.append({"kind": ENGINEERING, "reason": "Even when physics/economics/institutions bind, an engineering path usually exists for the reframed goal.", "signal": "reframe path"})
    primary = kinds[0]["kind"]
    return {"primary": primary, "kinds": kinds,
            "note": "Classification is a hypothesis — the feasibility gate confirms or overturns it with reasons."}
