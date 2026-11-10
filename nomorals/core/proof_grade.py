"""Proof grades — the G0–G6 vocabulary (Doc 24, Pass PG1).

One scale, in core, for naming how strongly a claim is proven. The
ladder is the doc's own (PROOF_GROUNDED_IMPOSSIBLE_AND_SHAZAM §"Proof
ladder"), quoted faithfully — this module adopts its definitions, it
does not improve them:

| Level | Evidence                                       | Use                |
|-------|------------------------------------------------|--------------------|
| G0    | raw generative thought                         | hypothesis only    |
| G1    | logical/unit/schema check                      | inexpensive filter |
| G2    | symbolic/numeric derivation                    | quantitative claim |
| G3    | simulation/property/adversarial check          | model-dependent    |
| G4    | primary empirical source / independent         | real-world claim   |
|       | replication                                    |                    |
| G5    | built artifact/experiment/operation passes     | practical proof    |
| G6    | ongoing monitoring/receipt                     | durable proof      |

The doc's governing rule, also verbatim: **"No level silently
upgrades itself. A G0 idea cannot appear as a G5 fact."** — and its
stance: *"Free imagination in search; no unsupported confidence in
delivery."*

THE NO-SILENT-UPGRADE RULE AS CODE: a grade is never a stored
opinion. :class:`ClaimGrade` stores *evidence* (typed
:class:`EvidenceRef` records) and its grade is a derived property —
:func:`grade_of` computes the grade of the strongest evidence
attached. There is no constructor, setter or deserialization path
that yields a grade above the attached evidence:
:meth:`ClaimGrade.asserting` refuses a demanded grade above the
derived one, and :meth:`ClaimGrade.from_dict` refuses a stored grade
above the evidence it is stored with (fiat on disk stays refused).
Understating is safe by construction: a stored grade below the
evidence reads back as the derived grade, because the evidence —
not the label — is the record.

THE ADOPTION RULE (recorded here so no later pass can miss it):
this module is the ONE grade scale for the whole system, adopted
once, in core. Doc 18's run ticket names the grade its completion
evidence must reach by referencing :class:`ProofGrade` here — it
does not define a scale of its own. Docs 22/23's cooperative
L-levels, when that programme reaches its levels pass, must map
onto this scale (a mapping table beside the ones below) or be
struck: a second ladder is how the pre-PG1 problem — three
different objects sharing the word "evidence" — happened.

THE MAPPING TABLES below are the layer the doc did not supply:
explicit mappings from each shipped "evidence" object onto the
scale. A mapping row gives the DEFAULT grade reading of a source
label honestly earned under that system's own rules; attached
evidence always governs via :func:`grade_of`. Where a source
object's semantics do not map, the row says so, with the reason —
honesty over completeness:

* delegation's ``required_evidence`` kinds — an OPEN vocabulary
  (contract authors name the artifacts they demand), so only the
  machine-shaped kinds shipped in contracts are tabulated; the rest
  fall under :data:`DELEGATION_UNMAPPED_REASON`;
* the outcome steward's verdicts — an OUTCOME vocabulary (what
  happened to an operation), not a proof vocabulary: no row maps,
  each row documents why;
* WS4's trend-claim kinds — a closed vocabulary whose kinds ARE
  evidence bars, so each kind maps;
* impossible's claim lifecycle — a closed vocabulary, the ladder's
  closest shipped relative; each status maps.

The source vocabularies are MIRRORED as constants here (core is L1:
it may not import the L4/L5 modules that own them); the pin tests in
``tests/test_d24pg1_proof_grades.py`` assert the mirrors equal the
real modules' values.

SHADOW ONLY (PG1): nothing in production emits grades yet. PG2
enforces promotion inside impossible's ledger; PG3 emits grades at
the claim surfaces (research_digest, WS4); doc 18's ticket consumes
the vocabulary when its envelope pass lands. This pass changes no
behaviour anywhere.

Layer: core (L1), stdlib only.
"""

from __future__ import annotations

import enum
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ValidationError

__all__ = [
    "ProofGrade",
    "ProofGradeError",
    "GradeInfo",
    "GRADE_TABLE",
    "EvidenceRef",
    "EVIDENCE_KIND_GRADES",
    "grade_of",
    "ClaimGrade",
    "MappingRow",
    "DELEGATION_KIND_MAPPINGS",
    "DELEGATION_KNOWN_KINDS",
    "DELEGATION_UNMAPPED_REASON",
    "grade_for_delegation_kind",
    "STEWARD_VERDICT_MAPPINGS",
    "STEWARD_VERDICTS",
    "STEWARD_UNKNOWN_VERDICT",
    "STEWARD_RESOLUTION_EVIDENCE",
    "grade_for_steward_verdict",
    "WS4_KIND_MAPPINGS",
    "WS4_CLAIM_KINDS",
    "grade_for_ws4_kind",
    "IMPOSSIBLE_STATUS_MAPPINGS",
    "IMPOSSIBLE_STATUSES",
    "grade_for_impossible_status",
]


class ProofGradeError(ValidationError):
    """A grade was demanded, stored or parsed above its evidence —
    or a grade/evidence token was not understood. Raised, never
    silently corrected upward: the no-silent-upgrade rule fails
    closed."""


class ProofGrade(enum.IntEnum):
    """The G0–G6 proof ladder. Ordered: a higher grade is stronger
    evidence. Definitions live in :data:`GRADE_TABLE` — the doc's
    words, not paraphrases."""

    G0 = 0
    G1 = 1
    G2 = 2
    G3 = 3
    G4 = 4
    G5 = 5
    G6 = 6

    @classmethod
    def parse(cls, token: Any) -> ProofGrade:
        """Read a grade from a ``ProofGrade``, an int 0–6, or a
        ``"G0"``–``"G6"`` token (any case). Anything else is refused:
        a grade that cannot be read exactly is not guessed."""
        if isinstance(token, ProofGrade):
            return token
        if isinstance(token, bool):
            raise ProofGradeError(f"not a proof grade: {token!r}")
        if isinstance(token, int):
            try:
                return cls(token)
            except ValueError:
                raise ProofGradeError(
                    f"proof grade out of range 0-6: {token!r}") from None
        if isinstance(token, str):
            text = token.strip().upper()
            for grade in cls:
                if text == grade.name:
                    return grade
        raise ProofGradeError(f"not a proof grade: {token!r}")


@dataclass(frozen=True)
class GradeInfo:
    """One rung of the ladder: the doc's definition of the evidence
    type, the doc's statement of what a claim at that grade may be
    used as, and the machine name of the evidence kind that earns
    it (a key of :data:`EVIDENCE_KIND_GRADES`)."""

    grade: ProofGrade
    evidence: str
    use: str
    earning_kind: str


#: The ladder, exactly as the doc defines it. ``evidence`` and
#: ``use`` are the doc's table cells, verbatim.
GRADE_TABLE: dict[ProofGrade, GradeInfo] = {
    ProofGrade.G0: GradeInfo(
        ProofGrade.G0, "raw generative thought", "hypothesis only",
        "generative_thought"),
    ProofGrade.G1: GradeInfo(
        ProofGrade.G1, "logical/unit/schema check", "inexpensive filter",
        "logical_check"),
    ProofGrade.G2: GradeInfo(
        ProofGrade.G2, "symbolic/numeric derivation",
        "quantitative claim", "derivation"),
    ProofGrade.G3: GradeInfo(
        ProofGrade.G3, "simulation/property/adversarial check",
        "model-dependent claim", "simulation_check"),
    ProofGrade.G4: GradeInfo(
        ProofGrade.G4,
        "primary empirical source / independent replication",
        "real-world claim", "empirical_source"),
    ProofGrade.G5: GradeInfo(
        ProofGrade.G5, "built artifact/experiment/operation passes",
        "practical proof", "artifact_pass"),
    ProofGrade.G6: GradeInfo(
        ProofGrade.G6, "ongoing monitoring/receipt", "durable proof",
        "monitoring_receipt"),
}

#: Evidence kinds — the machine vocabulary of *what was attached*.
#: Each kind evidences exactly the grade whose earning type it is.
#: A claim carrying only ``generative_thought`` sits at G0: recorded
#: provenance for a thought is still just a thought.
EVIDENCE_KIND_GRADES: dict[str, ProofGrade] = {
    info.earning_kind: info.grade for info in GRADE_TABLE.values()
}


@dataclass(frozen=True)
class EvidenceRef:
    """One piece of attached evidence: its kind (a key of
    :data:`EVIDENCE_KIND_GRADES`) and a ref naming where the
    evidence itself lives (a receipt id, an artifact id, a claim id,
    a source URL…). The ref is the provenance pointer; this module
    grades the *type*, and the ref is what a verifier follows to
    check the substance. An unknown kind or an empty ref is refused
    at construction — ungradeable material is not evidence."""

    kind: str
    ref: str

    def __post_init__(self) -> None:
        if self.kind not in EVIDENCE_KIND_GRADES:
            raise ProofGradeError(
                f"unknown evidence kind: {self.kind!r} (known: "
                f"{', '.join(sorted(EVIDENCE_KIND_GRADES))})")
        if not isinstance(self.ref, str) or not self.ref.strip():
            raise ProofGradeError(
                "an evidence ref must name where the evidence lives")

    @property
    def grade(self) -> ProofGrade:
        """The grade this one piece of evidence earns."""
        return EVIDENCE_KIND_GRADES[self.kind]

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "ref": self.ref}

    @classmethod
    def from_dict(cls, data: Any) -> EvidenceRef:
        if not isinstance(data, Mapping):
            raise ProofGradeError(f"evidence entry is not a record: "
                                  f"{data!r}")
        kind = data.get("kind")
        ref = data.get("ref")
        if not isinstance(kind, str) or not isinstance(ref, str):
            raise ProofGradeError(
                f"evidence entry needs string kind + ref: {data!r}")
        return cls(kind=kind, ref=ref)


def grade_of(evidence: Iterable[EvidenceRef]) -> ProofGrade:
    """The grade of the strongest evidence attached — G0 for none.

    This is THE definition of a claim's grade under the adoption
    rule: *"a claim carries the grade its evidence has reached."*
    Only :class:`EvidenceRef` records count; a bare string or dict
    is not evidence and is refused rather than interpreted (smuggled
    shapes are how fiat grades would enter)."""
    strongest = ProofGrade.G0
    for item in evidence:
        if not isinstance(item, EvidenceRef):
            raise ProofGradeError(
                f"not an EvidenceRef: {item!r} — construct evidence "
                "via EvidenceRef / ClaimGrade.from_dict")
        if item.grade > strongest:
            strongest = item.grade
    return strongest


@dataclass(frozen=True)
class ClaimGrade:
    """A claim's proof standing: the evidence attached, and (derived,
    never stored) its grade.

    There is deliberately **no grade field**. The only state is the
    evidence tuple, so a grade above the evidence is unrepresentable
    through construction, attribute assignment (frozen), or
    deserialization (:meth:`from_dict` validates the stored token
    against the evidence it travels with and refuses fiat)."""

    evidence: tuple[EvidenceRef, ...] = ()

    def __post_init__(self) -> None:
        items = tuple(self.evidence)
        for item in items:
            if not isinstance(item, EvidenceRef):
                raise ProofGradeError(
                    f"not an EvidenceRef: {item!r} — construct "
                    "evidence via EvidenceRef / ClaimGrade.from_dict")
        object.__setattr__(self, "evidence", items)

    @property
    def grade(self) -> ProofGrade:
        """Derived, every read: the grade of the strongest evidence
        attached."""
        return grade_of(self.evidence)

    @classmethod
    def asserting(cls, grade: Any,
                  evidence: Iterable[EvidenceRef]) -> ClaimGrade:
        """State a claim AT a grade, with its evidence. The demanded
        grade is a floor check, never a cap and never a source: if
        it exceeds what the evidence earns, the claim is refused
        (the no-silent-upgrade rule); if the evidence earns more,
        the record carries the derived grade."""
        demanded = ProofGrade.parse(grade)
        items = tuple(evidence)
        derived = grade_of(items)
        if demanded > derived:
            raise ProofGradeError(
                f"claim asserted at {demanded.name} but its evidence "
                f"earns {derived.name} — a grade is only ever "
                "asserted together with evidence of the earning type")
        return cls(evidence=items)

    def meets(self, required: Any) -> bool:
        """True when the attached evidence reaches ``required`` —
        the check a consumer (e.g. doc 18's ticket, naming the grade
        completion evidence must reach) applies."""
        return self.grade >= ProofGrade.parse(required)

    def require(self, required: Any) -> None:
        """Raise unless the attached evidence reaches ``required``."""
        needed = ProofGrade.parse(required)
        if self.grade < needed:
            raise ProofGradeError(
                f"claim carries {self.grade.name}; {needed.name} "
                "is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "grade": self.grade.name,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: Any) -> ClaimGrade:
        """Read a stored claim grade. The stored grade token is a
        *claim about* the derivation and is checked, never trusted:
        above the evidence it travels with → refused (fiat on disk
        stays refused); at or below → the record stands on its
        evidence, and :attr:`grade` derives from it."""
        if not isinstance(data, Mapping):
            raise ProofGradeError(
                f"claim grade payload is not a record: {data!r}")
        raw_evidence = data.get("evidence", [])
        if not isinstance(raw_evidence, (list, tuple)):
            raise ProofGradeError(
                f"claim grade evidence is not a list: {raw_evidence!r}")
        items = tuple(EvidenceRef.from_dict(entry)
                      for entry in raw_evidence)
        record = cls(evidence=items)
        stored = data.get("grade")
        if stored is not None:
            claimed = ProofGrade.parse(stored)
            if claimed > record.grade:
                raise ProofGradeError(
                    f"stored grade {claimed.name} exceeds the grade "
                    f"its attached evidence earns "
                    f"({record.grade.name}) — refused")
        return record


# ── the mapping tables ──────────────────────────────────────────────


@dataclass(frozen=True)
class MappingRow:
    """One row of a mapping table: a value in a shipped source
    vocabulary, the grade it maps to (``None`` = no mapping — the
    semantics do not correspond), and the reason, in words. A ``None``
    row is a finding, not an omission."""

    source_value: str
    grade: ProofGrade | None
    reason: str


def _rows_by_value(rows: tuple[MappingRow, ...]) -> dict[str, MappingRow]:
    return {row.source_value: row for row in rows}


def _closed_lookup(table: dict[str, MappingRow], value: str,
                   table_name: str) -> MappingRow:
    row = table.get(value)
    if row is None:
        raise ProofGradeError(
            f"{value!r} is not in the {table_name} vocabulary "
            f"(closed set: {', '.join(sorted(table))})")
    return row


# (a) Delegation required_evidence kinds ─────────────────────────────
#
# required_evidence is an OPEN vocabulary: a contract author names
# the evidence artifacts the contract demands (the specialists'
# duties in agents/manifests.py are prose phrases — "test run",
# "build output", "deploy receipt"). A kind is SATISFIED when the
# claim ledger's latest evidence claim under that kind carries a
# "supported" verdict (agents/delegation.py completion_blockers).
# Only the machine-shaped kinds actually shipped in a contract are
# tabulated; everything else falls under DELEGATION_UNMAPPED_REASON.

#: The machine-shaped kinds shipped in a production contract today
#: (agents/outcome_steward.py's investigation contracts). Mirrored:
#: no constant names this pair in the source modules.
DELEGATION_KNOWN_KINDS = ("domain-evidence", "verdict")

DELEGATION_KIND_MAPPINGS: dict[str, MappingRow] = _rows_by_value((
    MappingRow(
        "domain-evidence", ProofGrade.G4,
        "The outcome steward's worker gathers domain-evidence by "
        "read-back from the domain's own primary store, each item "
        "cited as a case receipt — a primary empirical source, "
        "which is G4's evidence type. (The ledger's 'supported' "
        "verdict is the satisfaction gate; the grade belongs to "
        "what the evidence IS.)"),
    MappingRow(
        "verdict", None,
        "A verdict is a judgment OVER evidence, not evidence: the "
        "steward's contract requires the kind so a verifier's "
        "judgment claim exists in the ledger beside the "
        "domain-evidence it judged. It maps to no grade; the proof "
        "grade rides on the evidence claims the verdict judged."),
))

#: Why an unlisted delegation kind maps to nothing: the vocabulary
#: is open by design (see above), and a satisfied kind evidences
#: that the named artifact exists in the ledger as 'supported' —
#: what grade that artifact earns depends on what the artifact IS,
#: which only the contract author knows. Rather than guess, the
#: table maps unknown kinds to no grade: attach the artifact as an
#: EvidenceRef and let grade_of() decide.
DELEGATION_UNMAPPED_REASON = (
    "open vocabulary — a satisfied kind evidences that the named "
    "artifact sits in the claim ledger with a latest verdict of "
    "'supported'; its grade depends on the artifact the kind "
    "names, which only the contract author knows"
)


def grade_for_delegation_kind(kind: str) -> ProofGrade | None:
    """The grade a SATISFIED delegation evidence kind evidences —
    or ``None`` when the kind is outside the tabulated set (the
    open-vocabulary case; see :data:`DELEGATION_UNMAPPED_REASON`).
    Never raises: an unknown kind here is a vocabulary fact, not an
    error."""
    row = DELEGATION_KIND_MAPPINGS.get(kind)
    return row.grade if row is not None else None


# (b) Outcome steward verdicts ───────────────────────────────────────
#
# RESOLVABLE_VERDICTS + the standing unknown, mirrored from
# agents/outcome_cases.py (the pin tests assert the mirror).

#: Mirrors agents/outcome_cases.RESOLVABLE_VERDICTS.
STEWARD_VERDICTS = ("succeeded", "verified", "failed", "partial",
                    "stopped", "delivered")
#: Mirrors agents/outcome_cases.UNKNOWN_VERDICT.
STEWARD_UNKNOWN_VERDICT = "outcome_unknown"

_AXIS = ("verdicts classify what happened to an operation; grades "
         "classify how strongly a claim is proven — no mapping "
         "(different axis)")

STEWARD_VERDICT_MAPPINGS: dict[str, MappingRow] = _rows_by_value((
    MappingRow(
        "succeeded", None,
        "The operation's intended effect landed — an outcome fact; "
        + _AXIS + "."),
    MappingRow(
        "verified", None,
        "The outcome was checked by read-back before resolution — "
        "still a statement about the operation, not a rung on the "
        "proof ladder; " + _AXIS + ". (If the read-back evidence "
        "is attached to a claim, grade_of() grades IT.)"),
    MappingRow(
        "failed", None,
        "The operation's effect did not land — an outcome fact; "
        + _AXIS + "."),
    MappingRow(
        "partial", None,
        "Part of the operation's effect landed — an outcome fact; "
        + _AXIS + "."),
    MappingRow(
        "stopped", None,
        "The operation was stopped before completion — an outcome "
        "fact; " + _AXIS + "."),
    MappingRow(
        "delivered", None,
        "The operation's output reached its destination — an "
        "outcome fact; " + _AXIS + "."),
    MappingRow(
        STEWARD_UNKNOWN_VERDICT, None,
        "The standing state before evidence: the steward asserts "
        "nothing yet. Explicitly NOT a grade — it is the refusal "
        "to resolve without evidence, in outcome vocabulary."),
))

#: The one true relation between steward verdicts and evidence:
#: resolution of any case requires a cited evidence receipt on that
#: very case (OutcomeCaseStore.resolve refuses by construction
#: without one), so a resolved verdict is never evidence-free. That
#: is a provenance guarantee ABOUT the verdict — recorded here —
#: and it is still not a proof grade.
STEWARD_RESOLUTION_EVIDENCE = (
    "a cited evidence receipt on the case itself "
    "(OutcomeCaseStore.resolve refuses a resolution without one)"
)


def grade_for_steward_verdict(verdict: str) -> ProofGrade | None:
    """Steward verdicts map to no proof grade — see the rows in
    :data:`STEWARD_VERDICT_MAPPINGS` for the per-verdict reasons.
    An unknown verdict string is refused (closed vocabulary)."""
    row = _closed_lookup(STEWARD_VERDICT_MAPPINGS, verdict,
                          "steward verdict")
    return row.grade


# (c) WS4 trend-claim kinds ──────────────────────────────────────────
#
# Mirrors media/explore/trends.CLAIM_KINDS. The kinds are evidence
# bars, not labels of convenience: a claim only carries a kind when
# the evaluator's bar for it was met, so the kind itself evidences
# the grade.

#: Mirrors media/explore/trends.CLAIM_KINDS.
WS4_CLAIM_KINDS = ("trend", "spike", "unverified")

WS4_KIND_MAPPINGS: dict[str, MappingRow] = _rows_by_value((
    MappingRow(
        "trend", ProofGrade.G4,
        "The trend bar IS independent replication: >=2 "
        "result-bearing observations across >=2 distinct sweeps, "
        "or cross-source-class corroboration within one sweep "
        "(media/explore/trends.py). The kind cannot exist below "
        "the bar, so the kind evidences G4's evidence type."),
    MappingRow(
        "spike", ProofGrade.G2,
        "A spike's content is a numeric derivation over the beat's "
        "own observation series — latest >=4 results and >=3x the "
        "baseline mean (trends.py _SPIKE_MIN_RESULTS / "
        "_SPIKE_RATIO). A derivation, honestly labelled as one: "
        "below the corroboration bar, so not G4."),
    MappingRow(
        "unverified", ProofGrade.G1,
        "A single recorded sighting that passed the evaluator's "
        "intake — a structured basis dict and a mandatory "
        "confidence_basis; trends.py records nothing without one. "
        "It passed the inexpensive filter and nothing more: no "
        "derivation, no corroboration."),
))


def grade_for_ws4_kind(kind: str) -> ProofGrade:
    """The grade a WS4 claim kind evidences. Unknown kinds are
    refused (closed vocabulary)."""
    return _closed_lookup(WS4_KIND_MAPPINGS, kind,
                          "WS4 claim kind").grade  # type: ignore[return-value]


# (d) Impossible claim lifecycle ─────────────────────────────────────
#
# Mirrors impossible/charter.STATUSES — the ladder's closest shipped
# relative: a lifecycle of claim maturity, unenforced in its own
# organ until PG2. The rows give each status's default grade
# reading; where an attempt's actual evidence differs in type, the
# attached evidence governs via grade_of().

#: Mirrors impossible/charter.STATUSES.
IMPOSSIBLE_STATUSES = ("idea", "hypothesis", "tested", "prototype",
                       "built", "measured")

IMPOSSIBLE_STATUS_MAPPINGS: dict[str, MappingRow] = _rows_by_value((
    MappingRow(
        "idea", ProofGrade.G0,
        "The lifecycle's entry state: a raw generative thought, "
        "not yet formalized."),
    MappingRow(
        "hypothesis", ProofGrade.G1,
        "A hypothesis here is formalized — it names its falsifier "
        "and its cheapest killing experiment "
        "(impossible/hypothesis.py). Formalization is the logical "
        "filter: G1."),
    MappingRow(
        "tested", ProofGrade.G3,
        "The cheapest killing experiment has been run against the "
        "hypothesis — an adversarial check by design, which is "
        "G3's evidence type. Where a recorded attempt was a pure "
        "derivation, the attached evidence governs via grade_of(); "
        "this row is the status's default reading."),
    MappingRow(
        "prototype", ProofGrade.G5,
        "A built artifact exists — G5's evidence type. The status "
        "alone does not assert the served operation has passed in "
        "production; 'built'/'measured' carry the rest of that "
        "claim, and attached evidence governs."),
    MappingRow(
        "built", ProofGrade.G5,
        "The artifact itself, built: G5's evidence type (built "
        "artifact / experiment / operation passes)."),
    MappingRow(
        "measured", ProofGrade.G6,
        "Measured in operation: the claim stands on ongoing "
        "measurement and receipts — G6's evidence type."),
))


def grade_for_impossible_status(status: str) -> ProofGrade:
    """The default grade reading of an impossible-lifecycle status.
    Unknown statuses are refused (closed vocabulary)."""
    return _closed_lookup(IMPOSSIBLE_STATUS_MAPPINGS, status,
                          "impossible status").grade  # type: ignore[return-value]
