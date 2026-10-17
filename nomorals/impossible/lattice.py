"""Stage 4 — domain lattice: zones x courses x layers, known_impossible per cell.

Zones: physical / biological / digital / social / economic.
Courses: disciplines inside the zone (systematic, extensible — register_course).
Layers: theory / method / implementation / failure modes / economics.
Every cell carries: sources, methods, constraint model, evaluation criteria,
known-hard problems, and known_impossible entries with the REASON.
"""

from __future__ import annotations

from dataclasses import dataclass, field

ZONES = ("physical", "biological", "digital", "social", "economic")
LAYERS = ("theory", "method", "implementation", "failure_modes", "economics")

COURSES: dict[str, list[str]] = {
    "physical": ["materials", "thermodynamics", "fluids", "structures", "electrical",
                 "mechanisms", "manufacturing", "energy_storage", "optics", "acoustics"],
    "biological": ["human_factors", "ergonomics", "biomechanics", "nutrition_logistics",
                   "hygiene", "wound_care_logistics", "health_safety"],
    "digital": ["algorithms", "architecture", "data", "ml", "security", "performance",
                "embedded", "robotics_software", "networking"],
    "social": ["adoption", "trust", "standards", "regulation", "community", "education"],
    "economic": ["cost_models", "supply_chain", "substitution", "lifetime_cost",
                 "market_timing", "maintenance_economics"],
}


@dataclass
class KnownImpossible:
    claim: str
    reason: str
    kind: str = "physical"  # physical / economic / engineering / institutional


@dataclass
class Cell:
    zone: str
    course: str
    layer: str
    sources: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    constraint_model: str = ""
    evaluation: list[str] = field(default_factory=list)
    known_hard: list[str] = field(default_factory=list)
    known_impossible: list[KnownImpossible] = field(default_factory=list)


# Seed known-impossible entries: the single most important file in the class —
# knowing where the walls are is how you find what isn't. Each with a reason.
_KNOWN: dict[tuple[str, str], list[KnownImpossible]] = {
    ("physical", "thermodynamics"): [
        KnownImpossible("Perpetual motion machine", "Conservation of energy + second law: every cycle dissipates heat; no closed cycle outputs net work.", "physical"),
        KnownImpossible("Heat engine above Carnot efficiency", "Carnot bound eta <= 1 - T_cold/T_hot is a theorem, not a target.", "physical"),
        KnownImpossible("100% efficient energy conversion", "Second law: some energy always becomes unusable heat.", "physical"),
    ],
    ("physical", "materials"): [
        KnownImpossible("Room-temperature ambient superconductor in a garage", "No known material; requires new physics, not better engineering.", "physical"),
        KnownImpossible("Unbreakable material", "Every material has a finite fracture toughness; cracks propagate above critical stress.", "physical"),
        KnownImpossible("Weightless strong structure", "Strength scales with cross-section; square-cube law binds mass vs strength.", "physical"),
    ],
    ("physical", "energy_storage"): [
        KnownImpossible("Infinite / free energy storage", "Storage holds what you put in minus losses; capacity is bounded by chemistry/physics.", "physical"),
        KnownImpossible("Battery with energy density of petrol at battery cost", "Chemical bonds set an upper bound; petrol ~12,700 Wh/kg vs Li-ion ~250 Wh/kg system.", "physical"),
    ],
    ("physical", "structures"): [
        KnownImpossible("A house that never heats up (zero heat gain)", "Heat flows hot->cold; insulation slows it, never stops it. Goal reframes to: slow it enough that passive cooling wins.", "physical"),
    ],
    ("physical", "electrical"): [
        KnownImpossible("Wireless power at distance with no loss", "Field strength falls ~1/r^2 (far field); efficiency collapses with distance.", "physical"),
    ],
    ("digital", "algorithms"): [
        KnownImpossible("Solving NP-complete problems in polynomial time (general)", "P vs NP is open; no general polynomial algorithm is known. Specific instances are tractable.", "engineering"),
        KnownImpossible("Perfect compression of arbitrary data", "Kolmogorov / counting argument: most strings are incompressible.", "physical"),
    ],
    ("digital", "security"): [
        KnownImpossible("Perfectly secure system with zero trust assumptions", "Security rests on assumptions (keys, hardware, humans); remove all and nothing remains.", "physical"),
    ],
    ("biological", "human_factors"): [
        KnownImpossible("Humans that never need sleep", "Sleep is a biological requirement; deprivation degrades cognition and health.", "physical"),
    ],
}

# Course-level source hints (where to look for prior art / knowledge).
_SOURCES: dict[str, list[str]] = {
    "materials": ["Materials Project", "MatWeb", "Ashby charts", "PubChem"],
    "thermodynamics": ["textbooks (Cengel, Moran)", "NIST WebBook", "arXiv"],
    "manufacturing": ["supplier datasheets", "machinist handbooks", "OpenAlex"],
    "energy_storage": ["Materials Project", "Battery500", "arXiv", "PatentsView"],
    "algorithms": ["arXiv", "Semantic Scholar", "OpenAlex"],
    "ml": ["arXiv", "Semantic Scholar", "OpenAlex"],
    "security": ["CVE databases", "arXiv", "Semantic Scholar"],
    "health_safety": ["PubMed", "WHO guidance", "health/ guard_coaching boundary"],
}


def register_course(zone: str, course: str) -> None:
    COURSES.setdefault(zone, [])
    if course not in COURSES[zone]:
        COURSES[zone].append(course)


def lattice() -> dict:
    """Full lattice: zones x courses x layers, with known_impossible per cell."""
    out: dict = {"zones": {}, "layers": list(LAYERS)}
    for zone, courses in COURSES.items():
        out["zones"][zone] = {}
        for course in courses:
            cells = {}
            for layer in LAYERS:
                known = _KNOWN.get((zone, course), []) if layer in ("theory", "method") else []
                cells[layer] = {
                    "sources": _SOURCES.get(course, ["OpenAlex", "arXiv", "Semantic Scholar"]),
                    "methods": ["TRIZ", "morphological analysis", "constraint relaxation", "SCAMPER"],
                    "constraint_model": f"{course} constraints: physics, materials, cost, law, maintenance.",
                    "evaluation": ["gate results", "cost band", "falsifier test"],
                    "known_hard": [],
                    "known_impossible": [{"claim": k.claim, "reason": k.reason, "kind": k.kind} for k in known],
                }
            out["zones"][zone][course] = cells
    return out


def known_impossible_for(zone: str, course: str) -> list[dict]:
    return [{"claim": k.claim, "reason": k.reason, "kind": k.kind}
            for k in _KNOWN.get((zone, course), [])]


def find_wall(text: str) -> list[dict]:
    """Check a problem against known-impossible entries (substring match)."""
    t = (text or "").lower()
    hits = []
    for (zone, course), entries in _KNOWN.items():
        for k in entries:
            key_words = [w for w in k.claim.lower().split() if len(w) > 4]
            if any(w in t for w in key_words):
                hits.append({"zone": zone, "course": course, "claim": k.claim, "reason": k.reason, "kind": k.kind})
    return hits
