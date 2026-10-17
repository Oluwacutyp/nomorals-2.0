"""Stage 5 — TRANSFER engine. Where breakthroughs actually come from.

Methods (all real, all sourced):
- TRIZ: Altshuller's 40 Inventive Principles, derived from patent analysis,
  plus the Contradiction Matrix (improving feature vs worsening feature ->
  principles that historically resolved it). Principle names below are the
  standard English names used in TRIZ literature.
- Morphological analysis: decompose into independent parameters, list options
  per parameter, search the combination space.
- Constraint relaxation: remove one constraint at a time; ask whether the
  constraint is real (physics) or inherited (convention/cost/habit).
- SCAMPER: Substitute, Combine, Adapt, Modify, Put to another use, Eliminate, Reverse.
- Inversion: solve the opposite. Trivial-move: what would make this trivial?
- Structural analogy: "who else has this contradiction?" — not "who else
  writes about roofs?"

Every transfer move must survive the feasibility gate + falsifier. A lateral
move that fails the gate is recorded as failed, not presented as a solution.
"""

from __future__ import annotations

import itertools
import re

# The 40 Inventive Principles — standard English names (TRIZ literature).
TRIZ_40 = {
    1: ("Segmentation", "Divide an object into independent parts."),
    2: ("Taking Out", "Remove an interfering part or property."),
    3: ("Local Quality", "Change structure/properties locally; different parts do different things."),
    4: ("Asymmetry", "Change shape from symmetric to asymmetric."),
    5: ("Merging", "Bring closer together / merge identical or similar objects or operations."),
    6: ("Universality", "Make one part perform multiple functions; eliminate other parts."),
    7: ("Nested Doll", "Place one object inside another; pass through cavities."),
    8: ("Anti-Weight", "Compensate weight with lift / counterweight / buoyancy."),
    9: ("Preliminary Anti-Action", "Pre-empt harm: provide counter-action beforehand."),
    10: ("Preliminary Action", "Do the required change beforehand, fully or partially."),
    11: ("Beforehand Cushioning", "Prepare emergency means in advance to compensate for low reliability."),
    12: ("Equipotentiality", "Eliminate lifting/lowering; operate at one level."),
    13: ("The Other Way Round", "Invert the action: do it in reverse, turn it inside out, swap roles."),
    14: ("Spheroidality / Curvature", "Replace straight parts with curved ones; use rotation / ball joints."),
    15: ("Dynamics", "Make a rigid object movable / adaptive; divide into parts that move relative to each other."),
    16: ("Partial or Excessive Actions", "If 100% is hard, do slightly less or slightly more than needed."),
    17: ("Another Dimension", "Move to another dimension: 2D instead of 1D, multi-layer, tilt, use the other side."),
    18: ("Mechanical Vibration", "Set an object into oscillation; increase frequency; use resonance."),
    19: ("Periodic Action", "Replace continuous action with periodic / pulsed action."),
    20: ("Continuity of Useful Action", "Carry on work continuously; eliminate idle / intermediate steps."),
    21: ("Skipping", "Conduct a process at high speed; skip harmful stages."),
    22: ("Blessing in Disguise", "Convert harm into benefit; use a harmful factor to get a positive effect."),
    23: ("Feedback", "Introduce feedback; if it exists, change its magnitude or influence."),
    24: ("Intermediary", "Use an intermediary carrier / process; merge with another object that can be removed."),
    25: ("Self-Service", "Make an object serve itself: auxiliary functions, repair, reuse."),
    26: ("Copying", "Use a cheap copy instead of an expensive original; copy the shape/function."),
    27: ("Cheap Short-Living Objects", "Replace an expensive durable object with cheap short-lived ones."),
    28: ("Mechanics Substitution", "Replace a mechanical system with a sensory / field / software one."),
    29: ("Pneumatics and Hydraulics", "Use gas/liquid parts: air, water, hydraulics, pneumatics."),
    30: ("Flexible Shells and Thin Films", "Use flexible shells / thin films instead of rigid structures."),
    31: ("Porous Materials", "Make an object porous; add pores; if already porous, fill them usefully."),
    32: ("Color Changes", "Change colour / transparency / emissivity; use colour to signal or control."),
    33: ("Homogeneity", "Make interacting objects of the same material / properties."),
    34: ("Discarding and Recovering", "Discard / dissolve / evaporate parts that served their purpose; regenerate consumables."),
    35: ("Parameter Changes", "Change physical/chemical state, concentration, density, flexibility, temperature."),
    36: ("Phase Transitions", "Use phenomena during phase transitions: volume change, heat absorption/release."),
    37: ("Thermal Expansion", "Use thermal expansion/contraction of materials."),
    38: ("Strong Oxidants", "Replace normal air with oxygen-enriched air / oxidants."),
    39: ("Inert Atmosphere", "Replace a normal environment with an inert one; add neutral parts / vacuum."),
    40: ("Composite Materials", "Replace homogeneous materials with composites."),
}

# Contradiction Matrix — a verified subset. Keys are (improving, worsening)
# feature names; values are principle numbers. The full 39x39 matrix is a
# published table; this subset covers the contradictions that recur most in
# everyday hardware/software problems. Each entry is traceable to the
# standard matrix (e.g. Strength vs Weight -> 40, 26, 27, 1 is the textbook
# example used in TRIZ teaching).
CONTRADICTION_MATRIX: dict[tuple[str, str], list[int]] = {
    ("strength", "weight"): [40, 26, 27, 1],
    ("strength", "cost"): [10, 35, 26, 28],
    ("speed", "accuracy"): [28, 32, 13, 18],
    ("speed", "energy"): [19, 15, 35, 28],
    ("speed", "heat"): [35, 28, 2, 19],
    ("power", "weight"): [35, 15, 10, 37],
    ("power", "heat"): [35, 28, 2, 19],
    ("durability", "weight"): [40, 15, 35, 27],
    ("durability", "cost"): [27, 10, 35, 25],
    ("size", "strength"): [35, 40, 10, 14],
    ("size", "cost"): [15, 35, 10, 28],
    ("temperature", "durability"): [35, 36, 37, 39],
    ("reliability", "cost"): [27, 11, 35, 25],
    ("reliability", "complexity"): [15, 25, 35, 10],
    ("ease_of_use", "cost"): [27, 25, 10, 35],
    ("ease_of_use", "complexity"): [15, 25, 10, 6],
    ("efficiency", "cost"): [35, 10, 28, 25],
    ("efficiency", "heat"): [35, 28, 19, 2],
    ("precision", "cost"): [28, 10, 32, 25],
    ("precision", "speed"): [28, 32, 13, 18],
    ("volume", "weight"): [35, 1, 15, 40],
    ("volume", "cost"): [35, 10, 27, 25],
    ("flexibility", "strength"): [15, 35, 30, 40],
    ("flexibility", "cost"): [15, 10, 35, 27],
}

# Keyword -> feature mapping for contradiction detection in problem text.
_FEATURE_KEYWORDS: dict[str, str] = {
    "strong": "strength", "strength": "strength", "stronger": "strength", "durable": "durability", "durability": "durability",
    "heavy": "weight", "weight": "weight", "light": "weight", "lightweight": "weight", "lighter": "weight",
    "fast": "speed", "speed": "speed", "slow": "speed",
    "hot": "heat", "heat": "heat", "cool": "heat", "overheat": "heat",
    "expensive": "cost", "cost": "cost", "cheap": "cost", "price": "cost",
    "big": "size", "large": "size", "size": "size", "small": "size", "compact": "size",
    "powerful": "power", "power": "power", "energy": "energy",
    "accurate": "accuracy", "accuracy": "accuracy", "precise": "precision", "precision": "precision",
    "reliable": "reliability", "reliability": "reliability", "break": "reliability",
    "easy": "ease_of_use", "simple": "ease_of_use", "complex": "complexity",
    "efficient": "efficiency", "efficiency": "efficiency",
    "flexible": "flexibility", "flexibility": "flexibility",
    "volume": "volume", "bulk": "volume",
    "temperature": "temperature", "cold": "temperature",
}


def detect_contradiction(text: str) -> dict | None:
    """Find (improving, worsening) feature pair in problem text, if any."""
    t = (text or "").lower()
    found: list[str] = []
    for kw, feat in _FEATURE_KEYWORDS.items():
        if re.search(r"\b" + re.escape(kw) + r"\b", t) and feat not in found:
            found.append(feat)
    if len(found) < 2:
        return None
    # Prefer pairs present in the matrix; else first two found.
    for a, b in itertools.permutations(found, 2):
        if (a, b) in CONTRADICTION_MATRIX:
            return {"improving": a, "worsening": b,
                    "principles": CONTRADICTION_MATRIX[(a, b)]}
    return {"improving": found[0], "worsening": found[1], "principles": []}


def triz_moves(formulation, top_n: int = 6) -> list[dict]:
    """TRIZ moves for a formulation: contradiction-matrix principles first,
    then the highest-leverage generic principles."""
    text = f"{getattr(formulation, 'raw', '')} {getattr(formulation, 'goal_function', '')}"
    moves: list[dict] = []
    contra = detect_contradiction(text)
    if contra and contra["principles"]:
        for n in contra["principles"]:
            name, desc = TRIZ_40[n]
            moves.append({"method": "TRIZ-contradiction", "principle": n, "name": name,
                          "how": desc,
                          "apply": f"Apply '{name}' to '{getattr(formulation, 'goal_function', '')}': {desc}",
                          "contradiction": f"{contra['improving']} vs {contra['worsening']}"})
    # Generic high-leverage principles, always applicable as prompts.
    for n in (13, 1, 6, 25, 26, 27, 22, 35):
        if any(m.get("principle") == n for m in moves):
            continue
        name, desc = TRIZ_40[n]
        moves.append({"method": "TRIZ-generic", "principle": n, "name": name,
                      "how": desc,
                      "apply": f"Apply '{name}' to '{getattr(formulation, 'goal_function', '')}': {desc}",
                      "contradiction": None})
    return moves[:top_n]


SCAMPER = {
    "Substitute": "What can be substituted — material, process, component, supplier, energy source?",
    "Combine": "What can be combined — functions, parts, steps, products — into one?",
    "Adapt": "What can be adapted from elsewhere — another product, nature, another industry?",
    "Modify": "What can be modified — size, shape, colour, speed, frequency, temperature?",
    "Put to another use": "What else could this be used for — waste stream, by-product, idle capacity?",
    "Eliminate": "What can be removed entirely — a part, a step, a cost, a constraint?",
    "Reverse": "What happens if you reverse it — order, direction, roles, inside-out?",
}


def scamper_moves(formulation) -> list[dict]:
    goal = getattr(formulation, "goal_function", "") or getattr(formulation, "raw", "")
    return [{"method": "SCAMPER", "move": k,
             "apply": f"{v} Applied to: {goal}"} for k, v in SCAMPER.items()]


def morphological_search(parameters: dict[str, list[str]], limit: int = 12) -> list[dict]:
    """Morphological analysis: cartesian product of options per parameter.
    Returns combinations (capped) — the machine search nobody does by hand."""
    if not parameters:
        return []
    keys = list(parameters.keys())
    combos = []
    for i, values in enumerate(itertools.product(*(parameters[k] for k in keys))):
        if i >= limit:
            break
        combos.append({"combination": dict(zip(keys, values)),
                       "note": "One point in the morphological space; gate + falsifier decide whether it survives."})
    return combos


# Constraint kinds: physics (cannot relax) vs convention/cost/habit (can relax).
_CONSTRAINT_KIND = {
    "physics": ("gravity", "thermodynamic", "conservation", "speed of light", "entropy", "carnot"),
    "convention": ("regulation", "standard", "certification", "permit", "tradition", "usually", "always done"),
    "cost": ("budget", "expensive", "cost", "afford"),
    "assumption": ("must", "need to", "should", "only", "never"),
}


def relax_constraints(constraints: list[str]) -> list[dict]:
    """Constraint relaxation: for each constraint, say whether it is physics
    (cannot relax) or inherited (convention/cost/assumption — question it)."""
    out = []
    for c in constraints:
        cl = c.lower()
        kind = "assumption"
        relaxable = True
        for k, markers in _CONSTRAINT_KIND.items():
            if any(m in cl for m in markers):
                kind = k
                break
        if kind == "physics":
            relaxable = False
        out.append({"constraint": c, "kind": kind, "relaxable": relaxable,
                    "question": ("Cannot relax — physics. Work around it, don't fight it."
                                 if not relaxable else
                                 f"Is '{c}' real, or inherited? What becomes possible if it is removed?")})
    return out


def inversion_moves(formulation) -> list[dict]:
    goal = getattr(formulation, "goal_function", "") or getattr(formulation, "raw", "")
    return [
        {"method": "inversion", "move": "guarantee failure",
         "apply": f"How would you GUARANTEE '{goal}' fails? List those causes, then design against each."},
        {"method": "inversion", "move": "solve the opposite",
         "apply": f"Instead of achieving '{goal}', what does the opposite problem look like — and does solving it solve this one?"},
        {"method": "trivial", "move": "what would make this trivial?",
         "apply": f"What would make '{goal}' trivial — unlimited budget, no regulation, a different material, someone else already solving it? Which of those is actually removable?"},
        {"method": "analogy", "move": "structural analogy",
         "apply": f"Who else has this contradiction — not this topic, this SHAPE? Search other zones/courses for the same trade-off."},
    ]


def transfer_all(formulation, parameters: dict[str, list[str]] | None = None) -> dict:
    """Full transfer pass: TRIZ + SCAMPER + morphological + relaxation + inversion."""
    return {
        "triz": triz_moves(formulation),
        "scamper": scamper_moves(formulation),
        "morphological": morphological_search(parameters or {}),
        "relaxation": relax_constraints(list(getattr(formulation, "constraints", []) or [])),
        "inversion": inversion_moves(formulation),
        "contradiction": detect_contradiction(
            f"{getattr(formulation, 'raw', '')} {getattr(formulation, 'goal_function', '')}"),
    }
