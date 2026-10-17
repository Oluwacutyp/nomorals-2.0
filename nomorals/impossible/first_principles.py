"""First-principles reasoning — the benchmark this organ must outclass.

Benchmark: the first-principles problem-solving associated with Tesla/SpaceX
engineering — question every requirement, delete parts/processes, simplify,
optimize, accelerate, automate — reasoning from physics, not from analogy.
This organ applies that same rigor across hardware AND software, and must be
strongest in software (fastest iteration, full codebase access, testable in
seconds).

The order matters and is enforced: question -> delete -> simplify -> optimize
-> accelerate -> automate. Optimizing a part that should have been deleted is
the classic failure mode; automating a process that should have been
simplified is the second.

No content filters live in this module: the organ reasons freely and reports
honestly. The only controls are engineering mechanisms (sandbox for software
attempts, audit logs via the honesty ledger, confirmations for consequential
acts) — never moralizing filters, refusals, or artificial cripples.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Physics / fundamentals the decomposition bottoms out in. Every claim must
# trace to one of these, or be flagged as an assumption (convention/cost/habit).
FUNDAMENTALS = {
    "energy": "Energy is conserved; every conversion loses some as heat. Cost of energy = kWh x price.",
    "mass": "Mass must be moved / supported / heated. Mass costs energy, material, and structure.",
    "force": "Force = mass x acceleration. Structures carry force; failure is force exceeding strength.",
    "heat": "Heat flows hot -> cold. Insulation slows it, never stops it. Heat engines bounded by Carnot.",
    "information": "Information must be stored, moved, and processed. Bits are cheap; correctness is the cost.",
    "time": "Time is the one non-renewable resource. Latency, throughput, and waiting are separate costs.",
    "money": "Money is stored work. Cost = materials + energy + labour + time + risk.",
    "humans": "Humans err, tire, forget, and adapt. Any system depending on perfect humans will fail.",
}

# Requirement-questioning: is this requirement physics, or inherited?
_REQUIREMENT_KIND = [
    (re.compile(r"gravity|thermodynamic|conservation|entropy|melting|boiling|speed of light|friction|second law", re.I), "physics",
     "Physics — cannot be negotiated. Work around it, don't fight it."),
    (re.compile(r"must|required|regulation|standard|certif|permit|code|law", re.I), "institutional",
     "Institutional — satisfies a rule. Ask: does the rule exist, what does it actually require, is there a compliant path that avoids the cost?"),
    (re.compile(r"always|never|only|tradition|usual|industry standard|everyone", re.I), "convention",
     "Convention — inherited habit. Ask: who set it, when, and does the reason still hold?"),
    (re.compile(r"budget|cost|afford|expensive|cheap", re.I), "economic",
     "Economic — a price, not a law. Ask: what drives the price — material, energy, labour, scale, timing?"),
]


@dataclass
class FirstPrinciples:
    goal: str
    fundamentals: dict[str, str] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    questioned_requirements: list[dict] = field(default_factory=list)
    deletions: list[str] = field(default_factory=list)
    simplifications: list[str] = field(default_factory=list)
    optimizations: list[str] = field(default_factory=list)
    accelerations: list[str] = field(default_factory=list)
    automations: list[str] = field(default_factory=list)
    rebuilt_solution: str = ""

    def to_dict(self) -> dict:
        return {"goal": self.goal, "fundamentals": self.fundamentals,
                "assumptions": self.assumptions,
                "questioned_requirements": self.questioned_requirements,
                "deletions": self.deletions, "simplifications": self.simplifications,
                "optimizations": self.optimizations, "accelerations": self.accelerations,
                "automations": self.automations, "rebuilt_solution": self.rebuilt_solution}


def question_requirement(requirement: str) -> dict:
    """Question one requirement: physics or inherited?"""
    for pattern, kind, question in _REQUIREMENT_KIND:
        if pattern.search(requirement or ""):
            return {"requirement": requirement, "kind": kind, "question": question,
                    "negotiable": kind != "physics"}
    return {"requirement": requirement, "kind": "assumption", "negotiable": True,
            "question": "No physics marker found — treat as assumption until proven otherwise. Who requires this, and what breaks if it is removed?"}


def decompose(problem: str, requirements: list[str] | None = None,
              domain: str = "hardware") -> FirstPrinciples:
    """First-principles decomposition, in the enforced order:
    question -> delete -> simplify -> optimize -> accelerate -> automate.
    Domain 'software' gets software-specific fundamentals and deletions
    (the organ is strongest here: full codebase access, testable in seconds).
    """
    goal = (problem or "").strip()
    text = goal.lower()
    fundamentals: dict[str, str] = {}
    if domain == "software":
        fundamentals = {
            "information": FUNDAMENTALS["information"],
            "time": FUNDAMENTALS["time"],
            "correctness": "Correctness is the binding cost: a fast wrong system is worse than a slow right one. Tests are the cheapest correctness.",
            "complexity": "Complexity is the enemy: every abstraction, dependency, and branch is a future failure. Fewer moving parts = fewer failures.",
            "data": "Data must be stored, indexed, and moved. Storage is cheap; wrong indexes and N+1 queries are the usual cost.",
        }
    else:
        for key in ("energy", "mass", "force", "heat", "money", "humans"):
            fundamentals[key] = FUNDAMENTALS[key]
        if any(w in text for w in ("electric", "battery", "motor", "wire", "circuit")):
            fundamentals["charge"] = "Charge is conserved; current is charge flow; resistance turns energy into heat."
        if any(w in text for w in ("fluid", "water", "air", "pump", "pipe", "flow")):
            fundamentals["fluids"] = "Fluids flow from high pressure to low; pumps add pressure; friction in pipes costs energy."

    assumptions = []
    if "always" in text or "must" in text or "only" in text:
        assumptions.append("The wish states absolutes ('always'/'must'/'only') — each is a candidate inherited constraint, not physics.")
    if not re.search(r"\d", goal):
        assumptions.append("No numbers in the wish — quantities are assumed, not measured. Get numbers before optimizing.")

    questioned = [question_requirement(r) for r in (requirements or [])]
    # If no explicit requirements, question the implicit ones in the wish.
    if not questioned:
        questioned = [question_requirement("The current way of doing this is required")]

    deletions = _deletions(domain, text)
    simplifications = _simplifications(domain, text)
    optimizations = _optimizations(domain, text)
    accelerations = ["Remove waiting: what is idle, queued, or batched that could be continuous?",
                     "Parallelise independent steps; sequence only what truly depends."]
    automations = ["Automate only AFTER delete/simplify/optimize — automating a bad process cements it.",
                   "Automate the repetitive, verified step; keep a human in the loop for the irreversible step."]
    rebuilt = (f"Rebuilt from fundamentals for '{goal}': keep only what physics/information "
               f"requires ({', '.join(fundamentals.keys())}); delete the rest; simplify what remains; "
               f"then optimize, accelerate, automate — in that order.")
    return FirstPrinciples(goal=goal, fundamentals=fundamentals, assumptions=assumptions,
                           questioned_requirements=questioned, deletions=deletions,
                           simplifications=simplifications, optimizations=optimizations,
                           accelerations=accelerations, automations=automations,
                           rebuilt_solution=rebuilt)


def _deletions(domain: str, text: str) -> list[str]:
    if domain == "software":
        return ["Delete the feature nobody uses (check usage, not opinions).",
                "Delete the abstraction layer that adds indirection without reuse.",
                "Delete the manual step a test + script already covers.",
                "Delete the dependency that duplicates stdlib / existing code.",
                "Delete the config option that is never changed (hardcode the real value)."]
    return ["Delete the part: can the function be served by an existing part, the structure itself, or nothing?",
            "Delete the process step: does it add value, or does it exist because 'we always did it'?",
            "Delete the material: can a cheaper / lighter / local material serve the same function?",
            "Delete the fastener: can parts interlock, snap, weld, or be one piece?",
            "Delete the separate housing / cover if the structure itself protects."]


def _simplifications(domain: str, text: str) -> list[str]:
    if domain == "software":
        return ["One code path for one job — merge the three near-identical handlers.",
                "One data model, one source of truth — kill the parallel copies.",
                "Fewer states: every state is a bug surface; collapse states that behave identically.",
                "Synchronous where ordering matters; async only where parallelism pays."]
    return ["One material instead of three — fewer joints, fewer failures.",
            "One process instead of five — fewer setups, fewer tolerances stacking.",
            "Symmetry where possible — one tool, one fixture, one spare.",
            "Flat / straight where curvature adds cost without function."]


def _optimizations(domain: str, text: str) -> list[str]:
    if domain == "software":
        return ["Profile before optimizing — measure, don't guess the hotspot.",
                "Index the query that actually runs hot; cache the result that is actually recomputed.",
                "Cut allocations in the hot loop; batch I/O; avoid N+1 queries."]
    return ["Lighten the part that moves; strengthen only the part that carries load (local quality).",
            "Shorten the heat / fluid / force path — distance is loss.",
            "Match the material to the load: strong where stressed, cheap where not."]


def worked_example_car() -> dict:
    """Worked example — vehicle: decompose to physics, out-reason analogy.
    Analogy says 'make a better car like the last car'. First principles says:
    a car moves mass with energy; most mass moved is the car itself, most
    energy is lost to heat, drag, and braking. The wins are mass, drag,
    drivetrain efficiency, and braking recovery — not a bigger engine."""
    fp = decompose(
        "Make a car that uses far less energy per trip",
        requirements=["Must seat 5", "Must have 400km range", "Must pass crash standards", "Must cost under $25,000"],
        domain="hardware")
    return {"example": "car/vehicle — energy per trip", "first_principles": fp.to_dict(),
            "analogy_answer": "Copy the last car and make the engine 10% more efficient (analogy: incremental, misses the structure).",
            "first_principles_answer": (
                "Energy per trip = move mass + overcome drag + rolling resistance + braking loss. "
                "The car itself is ~80% of the mass moved; the drivetrain wastes ~60-70% of fuel as heat in an ICE. "
                "So: delete mass (lighter structure, fewer parts), simplify (electric drivetrain: ~90% efficient vs ~30% ICE, "
                "regenerative braking recovers braking loss), optimize aero (drag ~ v^2 — shape matters more than power). "
                "Result: 3-4x less energy per trip, not 10% — because the decomposition found where the energy actually goes.")}


def worked_example_software() -> dict:
    """Worked example — software: the organ's strongest ground.
    Analogy says 'add a faster server / more caching'. First principles says:
    find where time actually goes (profile), delete the work that need not
    happen, then optimize what remains."""
    fp = decompose(
        "Make the search endpoint 10x faster",
        requirements=["Must return correct results", "Must handle 1000 req/s", "Must not change the API"],
        domain="software")
    return {"example": "software — 10x faster search endpoint", "first_principles": fp.to_dict(),
            "analogy_answer": "Add more servers and a cache in front (analogy: throws resources at the symptom).",
            "first_principles_answer": (
                "Time per request = work done + waiting. Profile first: if 80% of time is one N+1 query, "
                "no server count fixes it — delete the N+1 (one indexed query), delete the serialization of fields nobody reads, "
                "simplify to one code path, then optimize the hot query. Result: 10x from deleting work, not from 10x hardware. "
                "Software wins because the loop is seconds: change -> test -> measure, not weeks of tooling.")}
