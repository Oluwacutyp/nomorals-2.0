"""Stage 8 — HYPOTHESIS. Falsifiable claims + cheapest killing experiments.

Every answer names its falsifier. If you can't say what would prove you
wrong, you haven't made a claim — you've made a wish.

One hypothesis is an anchor, not a diagnosis. This stage therefore
generates MULTIPLE competing hypotheses per problem — from different
angles (mechanism, constraint, framing) — each with its own falsifier
and cheapest killing experiment, ranked by falsifiability x cheapness x
explanatory power. The attempt step tries the best one first; the
alternatives stay in the ledger, so a failed attempt moves to the next
hypothesis instead of back to guessing.
"""

from __future__ import annotations

import re

__all__ = ["hypothesize", "hypothesize_multi", "reframe", "score_hypothesis"]


def _killing_experiment_for(failed: list) -> tuple[str, str, float]:
    """Cheapest killing experiment + falsifier + cheapness score for a
    binding gate failure. Returns (experiment, falsifier, cheapness)."""
    if "conservation" in failed or "thermodynamic" in failed:
        return (
            "Measure energy in vs energy out on a bench rig for one cycle. "
            "If out > in, the claim is dead — no further work.",
            "A single measured cycle with energy out > energy in.",
            0.9,  # one bench cycle: cheap
        )
    if "materials" in failed:
        return (
            "Buy the candidate material, load/heat it to the design point, "
            "and measure whether it yields/melts/fails. One coupon test kills it.",
            "The material yields, melts, or fails at or below the design point.",
            0.9,  # one coupon test: cheap
        )
    if "cost" in failed:
        return (
            "Price the three most expensive components from real suppliers. "
            "If the cheapest real BOM already exceeds budget, the claim is dead.",
            "Cheapest real BOM exceeds the budget with no substitution path.",
            0.9,  # three price checks: cheap
        )
    return (
        "Build the smallest end-to-end version that exercises the riskiest "
        "assumption, and measure the goal function once. If it misses by >50%, kill or reframe.",
        "The smallest real test misses the goal function by more than 50%.",
        0.6,  # a build, even minimal, costs more than a check
    )


def _falsifiability(falsifier: str) -> float:
    """How checkable is the falsifier? A falsifier naming a concrete
    measurement is highly falsifiable; a vague one is not."""
    concrete = re.search(r"measur|exceed|yield|melt|fail|miss|price|BOM|cycle|test", falsifier, re.I)
    return 0.9 if concrete else 0.5


def score_hypothesis(h: dict) -> float:
    """Rank: falsifiability x cheapness x explanatory power. A hypothesis
    that cannot be killed cheaply, or explains nothing, ranks low no
    matter how appealing it sounds."""
    return round(
        float(h.get("falsifiability", 0.5))
        * float(h.get("cheapness", 0.5))
        * float(h.get("explanatory_power", 0.5)),
        4,
    )


def hypothesize_multi(formulation, gate_result: dict, transfers: dict | None = None) -> dict:
    """Generate MULTIPLE competing hypotheses (mechanism, constraint,
    framing), each with falsifier + cheapest killing experiment, ranked
    by falsifiability x cheapness x explanatory power."""
    goal = getattr(formulation, "goal_function", "") or getattr(formulation, "raw", "")
    verdict = (gate_result or {}).get("verdict", "achievable_with_work")
    failed = (gate_result or {}).get("failed", [])
    experiment, falsifier, cheapness = _killing_experiment_for(failed)

    # Angle 1 — MECHANISM: the transfer moves achieve the goal function.
    mechanism = {
        "angle": "mechanism",
        "claim": (f"If we apply the transfer moves to '{goal}', the reframed version "
                  f"achieves the goal function within the stated constraints "
                  f"(gate verdict: {verdict})."),
        "falsifier": falsifier,
        "cheapest_killing_experiment": experiment,
        "falsifiability": _falsifiability(falsifier),
        "cheapness": cheapness,
        # Explains the path, not the binding check: strong when the gate passes.
        "explanatory_power": 0.9 if verdict in ("achievable_now", "achievable_with_work") else 0.6,
    }
    # Angle 2 — CONSTRAINT: the binding gate check is beatable.
    if failed:
        constraint = {
            "angle": "constraint",
            "claim": (f"The binding check(s) {', '.join(failed)} can be beaten for '{goal}' — "
                      f"the gate's bound is relaxable (cheaper material, smaller scale, "
                      f"longer time, different process) without losing the goal function."),
            "falsifier": falsifier,
            "cheapest_killing_experiment": experiment,
            "falsifiability": _falsifiability(falsifier),
            "cheapness": cheapness,
            # Directly attacks the binding check: highest explanatory power when gated.
            "explanatory_power": 0.95,
        }
    else:
        constraint = {
            "angle": "constraint",
            "claim": (f"No gate check binds for '{goal}' — the limiting factor is execution, "
                      f"not physics/cost/materials."),
            "falsifier": "A real build attempt hits a gate check (cost, materials, scale) that blocks it.",
            "cheapest_killing_experiment": experiment,
            "falsifiability": 0.8,
            "cheapness": cheapness,
            "explanatory_power": 0.6,
        }
    # Angle 3 — FRAMING: the reframed goal achieves the underlying ache.
    framing = {
        "angle": "framing",
        "claim": (f"Even if the strict version of '{goal}' fails, the reframed version "
                  f"achieves the underlying ache (removes the pain: cost, heat, effort, "
                  f"failure) without the impossible property."),
        "falsifier": "Users reject the reframed version: it does not relieve the stated pain.",
        "cheapest_killing_experiment": ("Show the reframed version to five people with the "
                                        "stated pain and ask: does this remove it? If none say yes, kill it."),
        "falsifiability": 0.9,
        "cheapness": 0.9,  # five conversations: cheapest of all
        "explanatory_power": 0.7,
    }
    hypotheses = [mechanism, constraint, framing]
    for h in hypotheses:
        h["score"] = score_hypothesis(h)
    hypotheses.sort(key=lambda h: h["score"], reverse=True)
    for i, h in enumerate(hypotheses, 1):
        h["rank"] = i
    return {"hypotheses": hypotheses, "primary": hypotheses[0],
            "status": "hypotheses", "note": "Best first; alternatives stay in the ledger — a failed attempt moves to the next, not back to guessing."}


def hypothesize(formulation, gate_result: dict, transfers: dict | None = None) -> dict:
    """Backwards-compatible single-hypothesis shape: the best-ranked
    hypothesis, plus the full ranked list under ``hypotheses``."""
    multi = hypothesize_multi(formulation, gate_result, transfers)
    primary = multi["primary"]
    return {"claim": primary["claim"], "falsifier": primary["falsifier"],
            "cheapest_killing_experiment": primary["cheapest_killing_experiment"],
            "status": "hypothesis",
            "note": "A claim is never promoted past its evidence: hypothesis until tested, tested until measured.",
            "primary_angle": primary["angle"],
            "hypotheses": multi["hypotheses"]}


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
