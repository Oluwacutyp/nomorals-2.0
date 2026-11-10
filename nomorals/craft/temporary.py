"""Temporary competence machinery (L5) — docs 22+23, Stage C P9.

Doc 23 §1: an unknown field is a work order, not a refusal.  When
a field has no pack, Devon spawns a research cell to construct a
**temporary competence pack**, benchmarks it, and only then claims
practical capability.  This module is that composed machine, built
only from shipped parts:

    field with no live pack
      → a research cell on a delegation contract (the idea cell's
        pattern: scout produces, skeptic challenges, verifier
        checks — independence is the store's law, not a promise)
      → the findings minted as a temporary pack: a filebook
        **draft**, labelled unproven by the shipped machinery
      → the pack's declared fixtures run as its benchmark (the
        Stage B fixture evaluator, :func:`run_pack_fixtures`)
      → the benchmark — never the prose — decides the *claim*: a
        passing benchmark mints a **competence grant**, a recorded,
        scoped, expiring instrument that capability answers read.

The discipline the evaluation pins (§7): the gate binds the
**claim**, never the work.  A failing benchmark leaves everything
on record — the research contract, the draft pack, the fixtures,
the benchmark numbers — and the claim simply unmade.  A grant is
bounded three ways: its **scope** (the one field it covers), its
**expiry** (a claim past expiry reads as no competence), and its
**evidence** (the benchmark record it cites — a grant whose cited
benchmark does not pass is no grant at all, however it is stored).

Standing is *derived at read*: the grant store persists the
grant's facts (times, scope, benchmark summary) and deliberately
no standing, level or grade token — a planted ``"status":
"active"`` or ``"level": "L3"`` in the store file is never read.
The durable path stays the filebook's own: if the draft later
promotes through mission-proving, the live pack carries the claim
and the grant is moot.

The research work itself arrives from the caller — the worker
organ.  In production the caller wires the research pipeline
(``research.pipeline.run_job`` over a ``ResearchContext``) and
hands in a ``research_fn`` returning the cell's product:
``{"summary": str, "sections": {<canonical pack section>: text},
"fixtures": [{"filename": ..., "content": ...}]}`` — fixtures
the benchmark will execute, authored by the research cell as its
acceptance tests for the field.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.delegation import DelegationStore
from ..core.proof_grade import EvidenceRef
from ..skills.filebook import FileSkillLibrary, SkillDoc
from ..skills.packs import (
    PACK_SECTIONS,
    load_pack,
    pack_docs,
    validate_pack,
)
from ..storage.artifacts import ArtifactStore
from ..storage.blob import BlobStore
from .levels import cap_claim
from .pilots import run_pack_fixtures

__all__ = [
    "TemporaryCompetenceError",
    "DEFAULT_GRANT_TTL_SECONDS",
    "CompetenceGrant",
    "CompetenceGrantStore",
    "CompetenceRun",
    "acquire_competence",
    "capability_status",
    "benchmark_passed",
]

#: A temporary grant's default lifetime: long enough to work a
#: field while its draft proves itself in missions, short enough
#: that an unproven field never becomes a permanent claim by
#: neglect.  Callers may bound tighter; the read side only ever
#: trusts the recorded expiry.
DEFAULT_GRANT_TTL_SECONDS = 7 * 24 * 3600

_SCOUT = "temp-competence:scout"
_SKEPTIC = "temp-competence:skeptic"
_VERIFIER = "temp-competence:verifier"

_SLUG_RE = re.compile(r"[^a-z0-9]+")

#: Canonical section key → the heading the pack schema reads.
_SECTION_HEADINGS: dict[str, str] = {
    "terminology": "Terminology",
    "workflow": "Workflow",
    "tools": "Tools",
    "rubric": "Rubric",
    "anti_patterns": "Anti-patterns",
    "exemplars": "Exemplars",
    "evaluators": "Evaluators",
    "failure_modes": "Failure modes",
    "locale_variants": "Locale variants",
    "credential_constraints": "Credential constraints",
}


class TemporaryCompetenceError(ValueError):
    """The machinery cannot proceed: the research product is
    mis-shaped, or its recorded findings do not re-read.  Named,
    never shrugged."""


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", str(text or "").lower()).strip("-")


def benchmark_passed(benchmark: Mapping[str, Any] | None) -> bool:
    """The benchmark verdict, derived from the run's own numbers
    — the one definition, used when a grant is minted and again
    every time a stored grant is read: at least one fixture
    declared, every declared fixture ran, no file failed, and at
    least one test passed with none failing."""
    if not isinstance(benchmark, Mapping):
        return False
    declared = int(benchmark.get("declared") or 0)
    return (declared >= 1
            and int(benchmark.get("run") or 0) == declared
            and int(benchmark.get("files_failed") or 0) == 0
            and int(benchmark.get("tests_passed") or 0) >= 1
            and int(benchmark.get("tests_failed") or 0) == 0)


# ── the grant: a recorded, bounded claim instrument ──────────────


@dataclass
class CompetenceGrant:
    """One temporary competence grant, as recorded.

    The record carries facts only — scope, times, and the
    benchmark summary it stands on.  There is deliberately no
    status, level or grade field: standing derives at read from
    these facts plus the pack's current filebook state, so a
    stored token can never outlive its evidence.
    """

    pack_id: str
    domain: str
    scope: str
    granted_at: float
    expires_at: float
    benchmark_ref: str = ""
    benchmark: dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "pack_id": self.pack_id,
                "domain": self.domain, "scope": self.scope,
                "granted_at": self.granted_at,
                "expires_at": self.expires_at,
                "benchmark_ref": self.benchmark_ref,
                "benchmark": dict(self.benchmark)}

    @classmethod
    def from_dict(cls, data: Any) -> CompetenceGrant:
        """Read one stored grant.  Only the fact fields are read —
        a stored standing/level/grade token is not data this
        record carries, and is ignored rather than trusted."""
        if not isinstance(data, Mapping):
            raise TemporaryCompetenceError(
                f"grant entry is not a record: {data!r}")
        benchmark = data.get("benchmark")
        return cls(
            pack_id=str(data.get("pack_id") or ""),
            domain=str(data.get("domain") or ""),
            scope=str(data.get("scope") or ""),
            granted_at=float(data.get("granted_at") or 0.0),
            expires_at=float(data.get("expires_at") or 0.0),
            benchmark_ref=str(data.get("benchmark_ref") or ""),
            benchmark=dict(benchmark) if isinstance(
                benchmark, Mapping) else {},
            id=str(data.get("id") or ""))


def _default_grant_path() -> Path:
    home = os.environ.get("NM_HOME") or "~/.nomorals"
    return Path(os.path.expanduser(home)) / "data" \
        / "craft-competence.json"


class CompetenceGrantStore:
    """The grant records: one JSON file in the antibody-store
    idiom (a JSON document under the home data dir, written whole
    on each record).  Reads of a missing or torn file yield no
    grants — the store fails *closed*: lost records unmakes
    claims, it never invents them."""

    def __init__(self, path: Any = None) -> None:
        self.path = Path(path) if path else _default_grant_path()

    def _read(self) -> list[CompetenceGrant]:
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        rows = payload.get("grants") if isinstance(
            payload, Mapping) else None
        out: list[CompetenceGrant] = []
        for row in rows or []:
            try:
                out.append(CompetenceGrant.from_dict(row))
            except TemporaryCompetenceError:
                continue
        return out

    def record(self, grant: CompetenceGrant) -> CompetenceGrant:
        if not grant.id:
            grant = CompetenceGrant(
                **{**grant.__dict__,
                   "id": f"cg-{uuid.uuid4().hex[:12]}"})
        grants = self._read()
        grants.append(grant)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"grants": [g.to_dict() for g in grants]},
                       indent=1),
            encoding="utf-8")
        return grant

    def grants(self, pack_id: str = "") -> list[CompetenceGrant]:
        rows = self._read()
        if pack_id:
            rows = [g for g in rows if g.pack_id == pack_id]
        return rows

    def latest_for_pack(self, pack_id: str) -> CompetenceGrant | None:
        rows = self.grants(pack_id)
        if not rows:
            return None
        return max(rows, key=lambda g: g.granted_at)


# ── capability answers read pack status (+ grants) ───────────────


def capability_status(library: FileSkillLibrary, domain: str, *,
                      grants: CompetenceGrantStore | None = None,
                      now: float | None = None) -> dict[str, Any]:
    """What Devon may claim for one field, read from records now.

    * a **live pack** for the field carries the claim (level L0,
      its grade capped by the level through :func:`cap_claim`);
    * else, an **unexpired grant** whose cited benchmark passes
      carries it — temporarily, with the expiry named;
    * else the claim is **unmade**, with the reason named: no pack
      recorded, only an unproven draft, a lapsed grant, or a
      benchmark record that does not pass.

    No stored token is consulted anywhere in this read.
    """
    now = time.time() if now is None else float(now)
    domain = _slug(domain)

    packs = [doc for doc in pack_docs(library)
             if _slug(str(doc.domain or "")) == domain]
    live = [doc for doc in packs if not doc.is_draft]
    if live:
        pack = live[0]
        evidence = (EvidenceRef("logical_check", f"pack:{pack.id}"),)
        return {
            "domain": domain, "claimed": True, "temporary": False,
            "level": "L0", "grade": cap_claim("L0", evidence).name,
            "pack_id": pack.id, "pack_status": "live",
            "grant_expires_at": None,
            "basis": [f"pack {pack.id} is live in the filebook — "
                      "routed knowledge for the field, proven "
                      "through the filebook lifecycle"],
            "reason": "",
        }
    drafts = [doc for doc in packs if doc.is_draft]
    reason = ""
    for draft in drafts:
        if grants is None:
            reason = (f"pack {draft.id} exists as an unproven "
                      "draft and no grant records were read — the "
                      "claim waits for its benchmark")
            continue
        grant = grants.latest_for_pack(draft.id)
        if grant is None:
            reason = (f"pack {draft.id} exists as an unproven "
                      "draft — no competence grant is on record; "
                      "the claim waits for its benchmark")
            continue
        if not benchmark_passed(grant.benchmark):
            reason = (f"the recorded benchmark for pack {draft.id} "
                      "does not pass — its grant carries no claim")
            continue
        if now >= grant.expires_at:
            when = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(grant.expires_at))
            reason = (f"the temporary grant for pack {draft.id} "
                      f"lapsed at {when} — the pack remains an "
                      "unproven draft and the claim is unmade")
            continue
        evidence = (EvidenceRef("artifact_pass",
                                grant.benchmark_ref),) \
            if grant.benchmark_ref else ()
        return {
            "domain": domain, "claimed": True, "temporary": True,
            "level": "L0",
            "grade": cap_claim("L0", evidence).name
            if evidence else "G0",
            "pack_id": draft.id, "pack_status": "draft",
            "grant_expires_at": grant.expires_at,
            "basis": [f"temporary grant on pack {draft.id}: its "
                      "benchmark passed and is on record "
                      f"({grant.benchmark_ref or 'unreferenced'}), "
                      "and the grant is inside its recorded "
                      "lifetime"],
            "reason": "",
        }
    if not drafts:
        reason = ("no pack is recorded for this field — the field "
                  "is a work order for the research cell, not a "
                  "capability")
    return {
        "domain": domain, "claimed": False, "temporary": False,
        "level": None, "grade": None,
        "pack_id": drafts[0].id if drafts else "",
        "pack_status": "draft" if drafts else "absent",
        "grant_expires_at": None, "basis": [], "reason": reason,
    }


# ── the acquisition run ──────────────────────────────────────────


@dataclass
class CompetenceRun:
    """One acquisition run's outcome, in its records' words.

    ``stage`` names where the run ended: ``already_covered`` (a
    live pack already carries the field), ``granted`` (benchmark
    passed; a grant is on record), ``benchmark_failed``,
    ``invalid_pack``, or ``gate_refused`` (the research contract's
    own completion gate refused — its reasons are in ``detail``).
    In every stage the work products are on record; only the claim
    differs.
    """

    field: str
    domain: str
    pack_id: str
    stage: str
    claim_made: bool
    contract_id: str = ""
    validation_problems: tuple[str, ...] = ()
    benchmark: dict[str, Any] = field(default_factory=dict)
    grant: CompetenceGrant | None = None
    status: dict[str, Any] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)


def _pack_body(field_name: str, sections: Mapping[str, Any]) -> str:
    lines = [
        f"# Temporary competence pack — {field_name}",
        "",
        "Minted by the temporary-competence machinery from a "
        "research cell's findings.  A draft — labelled unproven — "
        "until its benchmark passes or the filebook's proving "
        "promotes it.",
    ]
    for key in PACK_SECTIONS:
        text = str(sections.get(key) or "").strip()
        if not text and key not in sections:
            continue
        lines.append("")
        lines.append(f"## {_SECTION_HEADINGS[key]}")
        lines.append("")
        lines.append(text)
    return "\n".join(lines) + "\n"


def _write_fixtures(library: FileSkillLibrary, pack_id: str,
                    fixtures: Any) -> list[str]:
    """Materialize the research cell's fixture files under the
    user library and return their paths (a fixture given as a path
    string is used as it stands)."""
    paths: list[str] = []
    target = library.user_dir / "fixtures" / pack_id
    for entry in fixtures or ():
        if isinstance(entry, str):
            paths.append(entry)
            continue
        if not isinstance(entry, Mapping):
            raise TemporaryCompetenceError(
                f"a fixture is a path or a filename+content record, "
                f"got {entry!r}")
        filename = Path(str(entry.get("filename") or "")).name
        if not filename:
            raise TemporaryCompetenceError(
                f"a fixture record names no filename: {entry!r}")
        target.mkdir(parents=True, exist_ok=True)
        path = target / filename
        path.write_text(str(entry.get("content") or ""),
                        encoding="utf-8")
        paths.append(str(path))
    return paths


def acquire_competence(
        db: Any, library: FileSkillLibrary, *, field: str,
        research_fn: Callable[[str], Mapping[str, Any]],
        blob_dir: Any,
        grants: CompetenceGrantStore | None = None,
        ttl_seconds: float = DEFAULT_GRANT_TTL_SECONDS,
        fixture_timeout: float = 600.0,
        now: float | None = None) -> CompetenceRun:
    """Run the temporary-competence machine for one field.

    ``research_fn`` is the research cell's worker organ: called
    with the field, it returns the findings product (see the
    module docstring for its shape).  If it raises, the failure is
    recorded on the research contract and the exception propagates
    verbatim — the machine does not soften a failed research pass
    into a minted pack.
    """
    now = time.time() if now is None else float(now)
    grants = grants if grants is not None \
        else CompetenceGrantStore()
    domain = _slug(field)
    if not domain:
        raise TemporaryCompetenceError(
            f"a field must name something, got {field!r}")
    pack_id = f"temp-{domain}-pack"

    covered = capability_status(library, domain, grants=grants,
                                now=now)
    if covered["pack_status"] == "live":
        return CompetenceRun(
            field=field, domain=domain,
            pack_id=str(covered["pack_id"]), stage="already_covered",
            claim_made=True, status=covered,
            detail={"note": "a live pack already carries this "
                            "field — no research cell was spawned"})

    store = DelegationStore(db)
    artifacts = ArtifactStore(db, BlobStore(db, Path(blob_dir)))
    contract = store.create_contract(
        goal=f"construct a temporary competence pack for the "
             f"field {field!r} and benchmark it before any "
             f"capability is claimed",
        title=f"temporary competence: {field}",
        deliverable="a temporary pack as a filebook draft, its "
                    "benchmark record, and — only if the benchmark "
                    "passes — a recorded competence grant",
        worker=_SCOUT, skeptic=_SKEPTIC, verifier=_VERIFIER,
        arbiter="owner", consequential=True,
        required_evidence=("research findings",),
        scope=(domain,), step_budget=10)
    run = CompetenceRun(field=field, domain=domain, pack_id=pack_id,
                        stage="", claim_made=False,
                        contract_id=contract.id)

    try:
        findings = research_fn(field)
    except Exception:
        store.report_failure(
            contract.id, by=_SCOUT,
            detail=f"the research cell raised for field {field!r}")
        raise
    if not isinstance(findings, Mapping) \
            or not isinstance(findings.get("sections"), Mapping):
        detail = ("the research product is mis-shaped: it must be "
                  "a mapping carrying a 'sections' mapping of pack "
                  "sections (and 'fixtures' for the benchmark)")
        store.report_failure(contract.id, by=_SCOUT, detail=detail)
        raise TemporaryCompetenceError(detail)
    sections = findings["sections"]
    summary = str(findings.get("summary") or
                  f"research findings for {field}")

    findings_artifact = artifacts.put_text(
        json.dumps({"field": field, "summary": summary,
                    "sections": {k: str(v) for k, v in
                                 sections.items()},
                    "fixtures": [str(f.get("filename") or f)
                                 if isinstance(f, Mapping) else str(f)
                                 for f in findings.get("fixtures")
                                 or ()]},
                   indent=1),
        type="text", creator=_SCOUT)
    claim = store.submit_evidence(
        contract.id, kind="research findings", claim=summary,
        producer=_SCOUT, evidence_ref=findings_artifact.id)
    recorded = artifacts.read_text(findings_artifact.id)
    try:
        recorded_sections = json.loads(recorded or "{}").get(
            "sections")
    except ValueError:
        recorded_sections = None
    if isinstance(recorded_sections, Mapping) and recorded_sections:
        store.verify_claim(claim.id, validator=_VERIFIER,
                           verdict="supported")
    else:
        store.verify_claim(claim.id, validator=_VERIFIER,
                           verdict="contradicted")
        detail = ("the recorded findings do not re-read with "
                  "their sections — the research product is not "
                  "on record in usable form")
        store.report_failure(contract.id, by=_SCOUT, detail=detail)
        raise TemporaryCompetenceError(detail)
    challenge = store.challenge(
        contract.id, challenger=_SKEPTIC,
        claim="this field is claimable only when the temporary "
              "pack minted from this research passes its "
              "benchmark — the prose is not the gate")

    fixture_paths = _write_fixtures(
        library, pack_id, findings.get("fixtures"))
    doc = SkillDoc(
        id=pack_id, title=f"Temporary competence pack — {field}",
        summary=f"Researched pack for the field {field!r}, minted "
                "as an unproven draft by the temporary-competence "
                "machinery.",
        topics=(field,), triggers=(field,), layer="craft",
        version=1, kind="pack", domain=domain,
        fixtures=tuple(fixture_paths),
        body=_pack_body(field, sections))
    library.write_draft(doc)
    view = load_pack(library, pack_id)
    problems = validate_pack(view.doc) if view is not None else [
        f"{pack_id}: the minted draft does not load back"]
    if problems:
        run.validation_problems = tuple(problems)
        run.stage = "invalid_pack"
        store.report_failure(
            contract.id, by=_SCOUT,
            detail="the temporary pack is not valid: "
                   + "; ".join(problems))
        run.status = capability_status(library, domain,
                                       grants=grants, now=now)
        return run

    benchmark = run_pack_fixtures(
        tuple(fixture_paths), timeout=fixture_timeout)
    # The evaluator's aggregate carries no failed-test total (its
    # per-fixture rows do); the grant record and the verdict both
    # read the total, so it is summed here from the same rows.
    benchmark["tests_failed"] = sum(
        int(row.get("tests_failed") or 0)
        for row in benchmark.get("per_fixture") or ())
    run.benchmark = benchmark
    benchmark_artifact = artifacts.put_text(
        json.dumps(benchmark, indent=1, default=str),
        type="text", creator=_VERIFIER)
    if not benchmark_passed(benchmark):
        store.resolve_challenge(
            challenge.id, resolver=_VERIFIER, verdict="contradicted",
            note=f"benchmark: {benchmark.get('tests_passed', 0)} "
                 f"passed, {benchmark.get('tests_failed', 0)} "
                 f"failed across {benchmark.get('declared', 0)} "
                 "fixture file(s) — the claim is unmade and the "
                 "pack stays an unproven draft")
        store.report_failure(
            contract.id, by=_SCOUT,
            detail="the benchmark did not pass — the work is on "
                   "record and the capability claim is unmade")
        run.stage = "benchmark_failed"
        run.detail = {"benchmark_ref": benchmark_artifact.id}
        run.status = capability_status(library, domain,
                                       grants=grants, now=now)
        return run

    store.resolve_challenge(
        challenge.id, resolver=_VERIFIER, verdict="supported",
        note=f"benchmark passed: {benchmark.get('tests_passed', 0)}"
             " test(s), no failures — the gate, not the prose, "
             "carries the claim")
    store.sign_off(contract.id, verifier=_VERIFIER,
                   note="temporary competence: research recorded, "
                        "pack minted, benchmark passed")
    decision = store.complete(contract.id, by=_VERIFIER)
    if not decision.ok:
        run.stage = "gate_refused"
        run.detail = {"reasons": list(decision.reasons),
                      "benchmark_ref": benchmark_artifact.id}
        run.status = capability_status(library, domain,
                                       grants=grants, now=now)
        return run
    grant = grants.record(CompetenceGrant(
        pack_id=pack_id, domain=domain, scope=f"field:{domain}",
        granted_at=now, expires_at=now + float(ttl_seconds),
        benchmark_ref=benchmark_artifact.id,
        benchmark={key: benchmark.get(key) for key in (
            "declared", "run", "files_passed", "files_failed",
            "tests_passed", "tests_failed", "wall_seconds")}))
    run.grant = grant
    run.stage = "granted"
    run.claim_made = True
    run.detail = {"benchmark_ref": benchmark_artifact.id}
    run.status = capability_status(library, domain, grants=grants,
                                   now=now)
    return run
