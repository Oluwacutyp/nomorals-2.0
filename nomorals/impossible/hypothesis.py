"""Stage 8 — HYPOTHESIS. Falsifiable claim + cheapest killing experiment.

Every answer names its falsifier. If you can't say what would prove you
wrong, you haven't made a claim — you've made a wish.
"""

from __future__ import annotations

import re


def hypothesize(formulation, gate_result: dict, transfers: dict | None = None) -> dict:
    goal = getattr(formulation, "goal_function", "") or getattr(formulation, "raw", "")
    verdict = (gate_result or {}).get("verdict", "achievable_with_work")
    failed = (gate_result or {}).get("failed", [])
    claim = (f"If we apply the transfer moves to '{goal}', the reframed version "
             f"achieves the goal function within the stated constraints "
             f"(gate verdict: {verdict}).")
    if failed:
        claim += f" Binding checks to beat: {', '.join(failed)}."
    # Cheapest killing experiment: attack the binding check first.
    if "conservation" in failed or "thermodynamic" in failed:
        experiment = ("Measure energy in vs energy out on a bench rig for one cycle. "
                      "If out > in, the claim is dead — no further work.")
        falsifier = "A single measured cycle with energy out > energy in."
    elif "materials" in failed:
        experiment = ("Buy the candidate material, load/heat it to the design point, "
                      "and measure whether it yields/melts/fails. One coupon test kills it.")
        falsifier = "The material yields, melts, or fails at or below the design point."
    elif "cost" in failed:
        experiment = ("Price the three most expensive components from real suppliers. "
                      "If the cheapest real BOM already exceeds budget, the claim is dead.")
        falsifier = "Cheapest real BOM exceeds the budget with no substitution path."
    else:
        experiment = ("Build the smallest end-to-end version that exercises the riskiest "
                      "assumption, and measure the goal function once. If it misses by >50%, kill or reframe.")
        falsifier = "The smallest real test misses the goal function by more than 50%."
    return {"claim": claim, "falsifier": falsifier,
            "cheapest_killing_experiment": experiment,
            "status": "hypothesis",
            "note": "A claim is never promoted past its evidence: hypothesis until tested, tested until measured."}


def reframe(formulation, gate_result: dict) -> dict:
    """The reformulation product: nearest achievable formulation."""
    goal = getattr(formulation, "goal_function", "") or getattr(formulation, "raw", "")
    verdict = (gate_result or {}).get("verdict", "achievable_with_work")
    if verdict == "excluded":
        achievable = (f"The strict version of '{goal}' is excluded by physics "
                      f"({(gate_result or {}).get('why', '')}). The achievable version: "
                      f"attack the ache underneath — remove the pain (cost, heat, effort, "
                      f"failure) without the impossible property. Examples: 'infinite energy' -> "
                      f"24h storage + siting + demand shaping; 'a car that never breaks' -> "
                      f"failures predictable, cheap, scheduled.")
    elif verdict == "bounded":
        achievable = (f"'{goal}' is bounded by: {(gate_result or {}).get('why', '')}. "
                      f"The achievable version keeps the goal and relaxes the binding bound — "
                      f"cheaper material, smaller scale, longer time, or a different process.")
    else:
        achievable = (f"'{goal}' survives the gate ({verdict}). The achievable version IS the goal, "
                      f"pursued through the transfer moves with the falsifier as the first test.")
    return {"reframed_goal": achievable, "verdict": verdict,
            "ache": getattr(formulation, "pain", "")}
