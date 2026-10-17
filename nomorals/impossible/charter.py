"""Charter — the four kinds of impossible + the stance every stage enforces."""

from __future__ import annotations

from dataclasses import dataclass

# Kinds of "impossible" (ArenaAI §1). Genuine #1 is rare; almost every
# claim of impossible is #2, #3 or #4.
PHYSICAL = "physical"          # violates conservation / thermo / material limits
ECONOMIC = "economic"          # physically fine, costs more than the problem is worth
ENGINEERING = "engineering"     # nobody has solved it *with these constraints* yet
INSTITUTIONAL = "institutional"  # standards, regulation, adoption, human factors
KINDS = (PHYSICAL, ECONOMIC, ENGINEERING, INSTITUTIONAL)

# Gate verdicts — the gate classifies, it never refuses.
EXCLUDED = "excluded"
BOUNDED = "bounded"
ACHIEVABLE_NOW = "achievable_now"
ACHIEVABLE_WITH_WORK = "achievable_with_work"
GATE_VERDICTS = (EXCLUDED, BOUNDED, ACHIEVABLE_NOW, ACHIEVABLE_WITH_WORK)

# Claim lifecycle — a claim is never promoted past its evidence.
STATUSES = ("idea", "hypothesis", "tested", "prototype", "built", "measured")

STANCE = (
    "Nothing is refused. Everything is answered: formulation, classification, "
    "prior art, transfer, feasibility gate, falsifiable hypothesis, cheapest "
    "killing experiment, confidence band, and what would change the answer."
)


@dataclass(frozen=True)
class KindInfo:
    kind: str
    meaning: str
    what_class_can_do: str


KIND_TABLE: dict[str, KindInfo] = {
    PHYSICAL: KindInfo(PHYSICAL, "Violates conservation, thermodynamics, material limits, information bounds",
                       "Not possible as stated — but the goal is usually reframable to the achievable version."),
    ECONOMIC: KindInfo(ECONOMIC, "Physically fine; costs more than the problem is worth",
                       "Very solveable: cost models, substitution, scale, timing."),
    ENGINEERING: KindInfo(ENGINEERING, "Nobody has solved it, or nobody has solved it with these constraints",
                          "The main target. Nearly all wins live here."),
    INSTITUTIONAL: KindInfo(INSTITUTIONAL, "Standards, regulation, adoption, human factors",
                            "Designable: find the path that satisfies the constraint."),
}


def charter() -> dict:
    """The charter: kinds, stance, and the standing rules."""
    return {
        "stance": STANCE,
        "kinds": {k: {"meaning": v.meaning, "what_class_can_do": v.what_class_can_do}
                  for k, v in KIND_TABLE.items()},
        "gate_verdicts": list(GATE_VERDICTS),
        "statuses": list(STATUSES),
        "standing_rules": [
            "Never produce a confident answer with no gate results.",
            "Never refuse a question — answer at the achievable level.",
            "Never average away a contradiction — surface it.",
            "Never claim a physical result that has not been measured; hardware word is a specification, only the bench speaks for the bench.",
            "Never waste idle time on ingestion with no open question — ingest to answer, not to hoard.",
            "Every answer names its falsifier.",
            "A claim is never promoted past its evidence.",
        ],
    }
