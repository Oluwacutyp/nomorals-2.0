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
off is named.  What this module enforces today is the *claim* law:
a level-labelled claim carries at most the grade its level means
**and** at most the grade its attached evidence earns
(:func:`cap_claim`) — the no-silent-upgrade rule, applied to
levels.  The record-reading derivation itself is the levels pass
(Stage C P10); nothing here invents a derivation Stage A has not
built.
"""

from __future__ import annotations

from collections.abc import Iterable

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
