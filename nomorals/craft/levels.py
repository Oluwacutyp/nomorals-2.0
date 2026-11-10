"""Capability levels L0–L5 as labels over the ONE proof ladder
(L5) — docs 22+23, Stage A.

Doc 22 §4 names capability levels L0–L5.  Doc 24 PG1's adoption
rule — recorded in :mod:`nomorals.core.proof_grade` itself — is
that these L-levels "must map onto this scale (a mapping table
beside the ones below) or be struck: a second ladder is how the
pre-PG1 problem happened."  This module is that mapping table.

A level is a **derived label**, read from records — the derivation
rules (the evaluation's gap 10):

* **L0 — knowledge**: a routed pack/skill for the field exists.
* **L1 — drafted**: a cell has produced the artifact class (a
  draft, labelled unproven).
* **L2 — tool-executed**: a real tool path produced the artifact
  under contract.
* **L3 — operated**: a delivery under grant is on record.
* **L4 — attestation class**: the deliverable's final act is a
  licensed human's attestation.  A *handoff class*, not a level of
  Devon's: Devon does all the preparation work and labels the
  attestation act truthfully as the license-holder's.
* **L5 — physical class**: the deliverable's final act is physical
  work by an accountable human or machine.  Also a handoff class:
  Devon coordinates, quotes, schedules, documents and verifies
  receipts.

L4/L5 therefore map to **no grade** — the row says so, following
proof_grade's own convention for semantics that do not correspond
(a ``None`` row is a finding, not an omission).  They are never
refusal tiers: the neighbouring work is done, and what is handed
off is named.  What Stage A enforced is the *claim* law:
a level-labelled claim carries at most the grade its level means
**and** at most the grade its attached evidence earns
(:func:`cap_claim`) — the no-silent-upgrade rule, applied to
levels.

Stage C P10 adds the record-reading derivation itself:
:func:`derive_level` reads a :class:`LevelFacts` — a small set of
booleans, each one the presence of a *record* (a live pack in the
filebook, claims on a contract ledger, a delivery receipt) — and
returns the highest level those records earn, with the earning
facts named.  The derivation is a pure function: it stores
nothing, trusts no stored level token, and a field's level changes
only when its underlying records change.  The gathering of facts
from the stores lives in :mod:`nomorals.craft.coverage`; this
module stays the vocabulary and the rules.  L4/L5 are never
derived: they are classes of *deliverable* (whose final act is an
attestation or physical work), not states a record set can earn.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..core.proof_grade import (
    EvidenceRef,
    MappingRow,
    ProofGrade,
    grade_of,
)

__all__ = [
    "LevelError",
    "LEVELS",
    "LEVEL_INFO",
    "LEVEL_GRADE_MAPPINGS",
    "parse_level",
    "grade_for_level",
    "is_handoff_level",
    "cap_claim",
    "LevelFacts",
    "DerivedLevel",
    "derive_level",
]


class LevelError(ValueError):
    """A capability-level token was not understood — refused, never
    guessed."""


#: The level vocabulary, in order.
LEVELS: tuple[str, ...] = ("L0", "L1", "L2", "L3", "L4", "L5")

#: What each level means: how it is earned (from records) and
#: whether it is Devon's own level or a handoff class.
LEVEL_INFO: dict[str, dict[str, str]] = {
    "L0": {"name": "knowledge",
           "earned_by": "a routed pack/skill for the field exists",
           "kind": "level"},
    "L1": {"name": "drafted",
           "earned_by": "a cell has produced the artifact class "
                        "(a draft, labelled unproven)",
           "kind": "level"},
    "L2": {"name": "tool-executed",
           "earned_by": "a real tool path produced the artifact "
                        "under contract",
           "kind": "level"},
    "L3": {"name": "operated",
           "earned_by": "a delivery under grant is on record",
           "kind": "level"},
    "L4": {"name": "attestation class",
           "earned_by": "the final act is a licensed human's "
                        "attestation; Devon prepares everything "
                        "around it",
           "kind": "handoff"},
    "L5": {"name": "physical class",
           "earned_by": "the final act is physical work by an "
                        "accountable human or machine; Devon "
                        "coordinates and verifies receipts",
           "kind": "handoff"},
}

#: The mapping table PG1's adoption rule requires: each level, the
#: grade reading its earning evidence carries, and the reason.
#: ``None`` rows are findings (see the module docstring).
LEVEL_GRADE_MAPPINGS: tuple[MappingRow, ...] = (
    MappingRow(
        "L0", ProofGrade.G1,
        "a routed pack exists only after passing the pack schema "
        "check — a logical/schema check, G1's earning type"),
    MappingRow(
        "L1", ProofGrade.G2,
        "a produced draft is a derivation from the pack's workflow "
        "— G2's earning type; it is not yet an empirical pass"),
    MappingRow(
        "L2", ProofGrade.G4,
        "a real tool path producing the artifact under contract is "
        "a primary empirical record of the work — G4's earning "
        "type"),
    MappingRow(
        "L3", ProofGrade.G5,
        "a delivery under grant on record is an operation that "
        "passed — G5's earning type"),
    MappingRow(
        "L4", None,
        "attestation is the license-holder's act, not Devon's "
        "evidence; the preparation work grades on its own evidence, "
        "the attestation carries no Devon grade"),
    MappingRow(
        "L5", None,
        "physical work is the accountable human's or machine's "
        "act; Devon's coordination grades on its own evidence, the "
        "physical act carries no Devon grade"),
)

_BY_LEVEL = {row.source_value: row for row in LEVEL_GRADE_MAPPINGS}


def parse_level(token: object) -> str:
    """Read a level token exactly (``L0``–``L5``, any case)."""
    if isinstance(token, str):
        text = token.strip().upper()
        if text in _BY_LEVEL:
            return text
    raise LevelError(f"not a capability level: {token!r} "
                     f"(known: {', '.join(LEVELS)})")


def grade_for_level(level: object) -> ProofGrade | None:
    """The grade a level's earning evidence carries — ``None`` for
    the handoff classes, whose acts are not Devon's to grade."""
    return _BY_LEVEL[parse_level(level)].grade


def is_handoff_level(level: object) -> bool:
    """True for L4/L5 — classes whose final act belongs to an
    accountable human, with Devon doing the work around it."""
    return LEVEL_INFO[parse_level(level)]["kind"] == "handoff"


def cap_claim(level: object, evidence: Iterable[EvidenceRef]) -> ProofGrade:
    """The grade a level-labelled claim may carry: at most what
    its evidence earns, and at most what its level means.

    For the handoff classes there is no level ceiling — the claim
    grades on its own (preparation) evidence alone.
    """
    ceiling = grade_for_level(level)
    derived = grade_of(evidence)
    if ceiling is None:
        return derived
    return min(derived, ceiling)


# ── the derivation (Stage C P10) ─────────────────────────────────


@dataclass(frozen=True)
class LevelFacts:
    """The record facts a level derivation reads.  Each field is
    the presence of a RECORD — never a declaration, never a stored
    level token:

    * ``live_knowledge`` — a pack/skill for the field is live
      (non-draft) in the filebook: routed knowledge exists;
    * ``draft_knowledge`` — the field's only pack/skill is an
      unproven draft (it earns NO level — the claim waits for its
      benchmark or promotion records; it is reported as a note so
      the work is visible without being dressed as competence);
    * ``produced`` — a cell has produced the artifact class: a
      delegation contract for the field carries claims on its
      ledger;
    * ``tool_executed`` — a real tool path produced the artifact
      under contract: a contract for the field completed with
      supported evidence claims carrying evidence refs;
    * ``operated`` — a delivery under grant is on record for the
      field (a delivery receipt tied to one of its contracts).
    """

    live_knowledge: bool = False
    draft_knowledge: bool = False
    produced: bool = False
    tool_executed: bool = False
    operated: bool = False


@dataclass(frozen=True)
class DerivedLevel:
    """One derivation's outcome: the level the records earn
    (``None`` when they earn none), the earning facts in ladder
    order, and honest notes about what the records do NOT earn."""

    level: str | None
    basis: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


#: Each earning fact, its level, and the basis line it contributes
#: — the gap-10 derivation rules in ladder order.
_EARNING: tuple[tuple[str, str, str], ...] = (
    ("live_knowledge", "L0",
     "knowledge: a routed pack/skill for the field is live in "
     "the filebook"),
    ("produced", "L1",
     "drafted: a cell has produced the artifact class — a "
     "delegation contract for the field carries claims on its "
     "ledger"),
    ("tool_executed", "L2",
     "tool-executed: a contract for the field completed with "
     "supported evidence carrying refs — a real tool path "
     "produced the artifact under contract"),
    ("operated", "L3",
     "operated: a delivery under grant is on record for the "
     "field"),
)


def derive_level(facts: LevelFacts) -> DerivedLevel:
    """The level a field's records earn — derived, never assigned.

    The highest earning fact present sets the level; every earning
    fact present is named in the basis.  Facts are read as they
    stand: the derivation fills in no implications (an operated
    record does not invent the tool-executed record beneath it —
    the delivery receipt IS its own earning fact).  L4/L5 are
    never returned: handoff classes describe a deliverable's final
    act, not a state of Devon's records.
    """
    basis: list[str] = []
    level: str | None = None
    for field_name, earned, line in _EARNING:
        if getattr(facts, field_name):
            basis.append(line)
            level = earned
    notes: list[str] = []
    if level is None:
        if facts.draft_knowledge:
            notes.append(
                "an unproven draft pack/skill exists for the "
                "field — work is on record, but no level is read "
                "from a draft: the claim waits for its benchmark "
                "or promotion records")
        else:
            notes.append(
                "no records earn a level for this field")
    elif facts.draft_knowledge and not facts.live_knowledge:
        notes.append(
            "an unproven draft also exists for the field; the "
            "level above rests on the other records, not on it")
    return DerivedLevel(level=level, basis=tuple(basis),
                        notes=tuple(notes))
