"""Stage B pilots (L5) — docs 22+23 P5/P6/P7.

Stage A shipped the foundation: the CraftGraph projection, role
contracts, the pack form on the filebook, and the universal receipt
schema.  A foundation nobody runs is shelfware, so Stage B runs it —
three pilots, one per domain the adoption named, each travelling
the same shape:

    pack (filebook) → role contract (manifests) → a delegation
    contract opened in the pack's workflow → the domain's shipped
    organs do the work → the pack's rubric scored from the run's
    own records → a QualityReceipt whose grade derives from the
    evidence those records carry.

* **P5, software** — the seed software pack's engineering loop over
  a real workspace task: the change is applied to real files, the
  acceptance suite really runs (twice — the skeptic's counterfactual
  demands it fail on the unpatched tree), and the pack's declared
  fixtures run as the evaluator step, which is where the fixture
  economics doc 22 guesses at get *measured*: fixtures declared,
  run, passed, and the wall time they cost.
* **P6, visual/media** — a media job through the shipped media
  substrate: a :class:`~nomorals.media_edit.jobs.JobManager` job
  running the Pillow edit chain (the deterministic offline rung),
  the deliverable landing as a versioned artifact whose provenance
  metadata carries the **rights manifest** and the **edit graph**
  (additive metadata — no Artifact schema change).  The GPU
  generation lanes are not exercised offline; the run names them
  absent-with-reason rather than pretending a Pillow resize proves
  them.
* **P7, CBT/wellbeing** — an exercise from the CBT pack run as a
  cell on the shipped health shell: the person's exercise work
  becomes a structured intervention record
  (:mod:`nomorals.health.interventions`), the skeptic scans the
  record against the shell's own no-diagnosis guard vocabulary, and
  the verifier accepts it only when the stored record re-validates
  complete.  ``guard_coaching`` and the drift crisis behavior are
  untouched.

The pilots are drivers, in the content cell's grain: they open and
discipline the contract; the work product arrives from the caller
(the worker organ — a coding agent, the media job, the person's own
exercise) and everything the cell asserts about it is checked
against real records before it is believed.  No pilot authors a
grade: receipts attach evidence refs and the grade derives, exactly
as Stage A P4 legislated.  Entry points are the three ``run_*``
functions; Stage C's composer is their future caller, and the packs
route to them through the filebook today.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agents.delegation import DelegationStore
from ..agents.manifests import manifest as manifest_for
from ..agents.role_contracts import contract_for_class
from ..core.proof_grade import EvidenceRef
from ..health.interventions import (
    InterventionError,
    InterventionStore,
    diagnostic_scan,
    render_intervention,
)
from ..media_edit.images import (
    MediaEditError,
    edit_image,
    image_probe,
    validate_ops,
)
from ..media_edit.jobs import JobManager, MediaJobRecordStore
from ..skills.filebook import FileSkillLibrary
from ..skills.packs import PackView, load_pack, validate_pack
from ..storage.artifacts import ArtifactStore
from ..storage.blob import BlobStore
from .receipts import QualityReceipt

__all__ = [
    "PilotError",
    "PilotRun",
    "run_pack_fixtures",
    "run_software_pilot",
    "run_media_pilot",
    "run_cbt_pilot",
    "SOFTWARE_PACK_ID",
    "MEDIA_PACK_ID",
    "CBT_PACK_ID",
]

SOFTWARE_PACK_ID = "software-craft-pack"
MEDIA_PACK_ID = "visual-media-craft-pack"
CBT_PACK_ID = "cbt-wellbeing-pack"


class PilotError(Exception):
    """A pilot cannot start: the pack is missing or not valid, the
    role contract does not resolve, or the task is mis-shaped.  The
    message names the problem — a pilot never starts on a shrug."""


@dataclass
class PilotRun:
    """One pilot run's outcome, in the run's own records' words."""

    domain: str
    pack_id: str
    outcome: str            # the contract's terminal word
    contract_id: str = ""
    receipt: QualityReceipt | None = None
    rubric: tuple[dict[str, Any], ...] = ()
    fixtures: dict[str, Any] = field(default_factory=dict)
    absent: dict[str, str] = field(default_factory=dict)
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def completed(self) -> bool:
        return self.outcome == "completed"


# ── shared machinery ─────────────────────────────────────────────


def _valid_pack(library: FileSkillLibrary, pack_id: str) -> PackView:
    view = load_pack(library, pack_id)
    if view is None:
        raise PilotError(
            f"pack {pack_id!r} is not in the library — a pilot "
            "runs a pack, and this one does not exist")
    problems = validate_pack(view.doc)
    if problems:
        raise PilotError(
            f"pack {pack_id!r} is not pilot-ready: "
            + "; ".join(problems))
    return view


def _independence_held(store: DelegationStore, contract: Any) -> bool:
    """The pack rubric's independence line, read off the ledger:
    worker ≠ verifier, and no claim was validated by its producer."""
    if not contract.worker or contract.worker == contract.verifier:
        return False
    for claim in store.claims(contract.id, limit=1000):
        if claim.validator and claim.validator == claim.producer:
            return False
    return True


def _receipt_for_contract(
        store: DelegationStore, contract: Any, *, source: str,
        kind: str, subject: str, actor: str, operation_ref: str = "",
        detail: dict[str, Any] | None = None) -> QualityReceipt:
    """A pilot run's receipt, derived from its contract record.

    Evidence, conservative in PG1's style (documented here, as the
    Stage A adapters document theirs):

    * the floor — ``logical_check``: the contract record exists,
      schema-shaped, in the delegation store (G1);
    * ``empirical_source`` — when at least one evidence claim
      stands *supported* with an evidence ref: an independent
      verifier marked it supported against the primary record the
      ref names (G4 — PG1's delegation mapping);
    * ``artifact_pass`` — only when the contract completed AND the
      run's own acceptance operation passed (``operation_ref``
      names its record): a built operation passing is practical
      proof (G5).

    A failed run attaches no ``artifact_pass`` and keeps whatever
    its verified evidence earned — honestly lower, never dressed.
    """
    evidence = [EvidenceRef("logical_check",
                            f"{source}:{contract.id}")]
    claims = store.claims(contract.id, limit=1000)
    supported = [c for c in claims if c.kind == "evidence"
                 and c.verdict == "supported" and c.evidence_ref]
    if supported:
        evidence.append(EvidenceRef("empirical_source",
                                    supported[0].evidence_ref))
    if contract.status == "completed" and operation_ref:
        evidence.append(EvidenceRef("artifact_pass", operation_ref))
    return QualityReceipt(
        source=source, source_id=contract.id, kind=kind,
        subject=subject, actor=actor, outcome=contract.status,
        recorded_at=contract.completed_at or contract.updated_at,
        evidence=tuple(evidence), detail=dict(detail or {}))


# ── the fixture evaluator (P5's economics instrument) ────────────

_PASSED_RE = re.compile(r"(\d+) passed")
_FAILED_RE = re.compile(r"(\d+) failed")


def _count(pattern: re.Pattern, text: str) -> int:
    found = pattern.findall(text or "")
    return int(found[-1]) if found else 0


def run_pack_fixtures(fixtures: Any, *, root: Any = None,
                      timeout: float = 600.0) -> dict[str, Any]:
    """Run a pack's declared fixtures as its evaluators run them:
    each fixture file under pytest, from the repo root, offline.

    Returns the fixture economics doc 22 §5 guesses at and P5
    measures: how many fixtures the pack declares, how many ran,
    how many files passed outright, how many tests passed in
    total, and the wall time the evaluation cost."""
    root = Path(root) if root else Path(__file__).resolve().parents[2]
    rows: list[dict[str, Any]] = []
    started = time.time()
    for fixture in fixtures or ():
        t0 = time.time()
        row: dict[str, Any] = {"fixture": str(fixture), "rc": None,
                               "tests_passed": 0, "tests_failed": 0,
                               "wall_seconds": 0.0, "error": ""}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-o", "addopts=",
                 "-q", "-p", "no:cacheprovider", str(fixture)],
                cwd=str(root), capture_output=True, text=True,
                timeout=float(timeout))
            output = (proc.stdout or "") + (proc.stderr or "")
            row["rc"] = proc.returncode
            row["tests_passed"] = _count(_PASSED_RE, output)
            row["tests_failed"] = _count(_FAILED_RE, output)
        except subprocess.TimeoutExpired:
            row["error"] = f"timed out after {timeout:g}s"
        row["wall_seconds"] = round(time.time() - t0, 2)
        rows.append(row)
    return {
        "declared": len(rows),
        "run": sum(1 for r in rows if r["rc"] is not None),
        "files_passed": sum(1 for r in rows if r["rc"] == 0),
        "files_failed": sum(1 for r in rows if r["rc"] not in (0, None)),
        "tests_passed": sum(r["tests_passed"] for r in rows),
        "wall_seconds": round(time.time() - started, 2),
        "per_fixture": rows,
    }


# ── P5: the software pilot ───────────────────────────────────────

_SOFTWARE_WORKER = "software-implementer"
_SOFTWARE_SKEPTIC = "software-skeptic"
_SOFTWARE_VERIFIER = "software-verifier"


def _write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, content in (files or {}).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _manifest_text(files: dict[str, str]) -> str:
    lines = []
    for rel in sorted(files or {}):
        data = (files or {})[rel].encode("utf-8")
        lines.append(
            f"{rel} sha256:{hashlib.sha256(data).hexdigest()[:16]}"
            f" bytes:{len(data)}")
    return "\n".join(lines)


def _clear_pycache(workspace: Path) -> None:
    """Drop stale bytecode under the workspace: a same-size,
    same-second rewrite would otherwise be served from
    ``__pycache__`` and the patched run would grade the old code."""
    for cache in workspace.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def _run_acceptance(workspace: Path, argv: list[str],
                    timeout: float) -> dict[str, Any]:
    _clear_pycache(workspace)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    try:
        proc = subprocess.run(
            argv, cwd=str(workspace), capture_output=True,
            text=True, timeout=float(timeout), env=env)
        output = (proc.stdout or "") + (proc.stderr or "")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        output = f"acceptance run timed out after {timeout:g}s"
        rc = -1
    return {"rc": rc, "output": output,
            "tests_passed": _count(_PASSED_RE, output),
            "tests_failed": _count(_FAILED_RE, output)}


def _acceptance_record(run: dict[str, Any]) -> str:
    """One acceptance run as a stored record — the verifier reads
    the verdict back out of this text, never out of memory."""
    return "\n".join([
        "acceptance run",
        f"rc: {run['rc']}",
        f"tests passed: {run['tests_passed']}",
        f"tests failed: {run['tests_failed']}",
        "--- output (tail) ---",
        (run["output"] or "")[-4000:],
    ])


_RC_RE = re.compile(r"^rc: (-?\d+)$", re.M)


def _recorded_rc(record_text: str) -> int | None:
    match = _RC_RE.search(record_text or "")
    return int(match.group(1)) if match else None


def run_software_pilot(db: Any, *, workspace: Any, blob_dir: Any,
                       task: dict[str, Any],
                       library: FileSkillLibrary | None = None,
                       fixtures: Any = None,
                       fixture_timeout: float = 600.0,
                       acceptance_timeout: float = 120.0) -> PilotRun:
    """Run the software pack's engineering loop on one real task.

    ``task`` is the worker organ's commission and product:

    * ``title`` / ``goal`` — the work, in words;
    * ``class_id`` — the governing specialist class (default
      ``backend``); its role contract supplies the output schema
      and its manifest the evidence duties;
    * ``files`` — the starting tree ({relpath: content});
    * ``change`` — the worker's change set ({relpath: content});
    * ``tests`` — the acceptance test files ({relpath: content});
    * ``acceptance`` — the acceptance argv (default: pytest over
      the test files); ``evidence_duties`` — the manifest duties
      this task commits to producing (default: the ``test run``
      duty — the duties not committed to are named absent).

    The change is applied to real files; the acceptance suite runs
    on the patched tree, and on the pristine tree first — the
    skeptic's counterfactual: a suite that also passes unpatched
    proves nothing, and its evidence claim is contradicted.
    """
    library = library or FileSkillLibrary()
    pack = _valid_pack(library, SOFTWARE_PACK_ID)
    class_id = str(task.get("class_id") or "backend")
    contract_terms = contract_for_class(class_id)
    if contract_terms is None:
        raise PilotError(
            f"no role contract for specialist class {class_id!r}")
    contract_terms.require_valid()
    spec = manifest_for(class_id)
    duties_all = tuple(spec.evidence_duties) if spec else ()
    duties = tuple(task.get("evidence_duties") or
                   tuple(d for d in duties_all if d == "test run")
                   or duties_all[:1])
    unknown = [d for d in duties if d not in duties_all]
    if unknown:
        raise PilotError(
            f"task commits to evidence duties the {class_id} "
            f"manifest does not declare: {', '.join(unknown)}")
    absent = {duty: "not produced — this task commits to "
                    f"{', '.join(duties) or 'no duties'}; the "
                    "contract required only what the task commits "
                    "to" for duty in duties_all if duty not in duties}

    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    store = DelegationStore(db)
    artifacts = ArtifactStore(db, BlobStore(db, Path(blob_dir)))
    title = str(task.get("title") or "software pilot task")
    base_files = dict(task.get("files") or {})
    change = dict(task.get("change") or {})
    tests = dict(task.get("tests") or {})
    test_paths = sorted(tests)
    argv = list(task.get("acceptance") or (
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q",
         "-p", "no:cacheprovider"] + test_paths))
    if not change or not test_paths:
        raise PilotError(
            "a software pilot task needs a change set and at "
            "least one acceptance test file")

    contract = store.create_contract(
        goal=str(task.get("goal") or title), title=title,
        deliverable="the patched workspace, its recorded "
                    "acceptance runs, and the deliverable record",
        worker=_SOFTWARE_WORKER, skeptic=_SOFTWARE_SKEPTIC,
        verifier=_SOFTWARE_VERIFIER, arbiter="owner",
        specialist_class=class_id, consequential=True,
        required_evidence=duties,
        scope=tuple(sorted(change)), step_budget=10)
    artifact_key = f"software-pilot:{contract.id}"
    store.claim_artifact(contract.id, artifact_key)
    base_artifact = artifacts.put_text(
        _manifest_text({**base_files, **tests}),
        type="text", creator=_SOFTWARE_WORKER)

    _write_tree(workspace, base_files)
    _write_tree(workspace, tests)
    pristine = _run_acceptance(workspace, argv, acceptance_timeout)
    pristine_record = artifacts.put_text(
        _acceptance_record(pristine), type="text",
        creator=_SOFTWARE_SKEPTIC)

    _write_tree(workspace, change)
    patched = _run_acceptance(workspace, argv, acceptance_timeout)
    patched_record = artifacts.put_text(
        _acceptance_record(patched), type="text",
        creator=_SOFTWARE_WORKER)
    deliverable = {
        "files_changed": sorted(change),
        "tests_run": patched["tests_passed"] + patched["tests_failed"],
        "tests_passed": patched["tests_passed"],
        "notes": f"acceptance on the patched tree: rc "
                 f"{patched['rc']} (pristine: rc {pristine['rc']})",
    }
    deliverable_artifact = artifacts.derive(
        _manifest_text({**base_files, **change, **tests}).encode(),
        from_ids=[base_artifact.id], type="text",
        creator=_SOFTWARE_WORKER, metadata={"deliverable": deliverable})
    store.write_artifact(contract.id, artifact_key, base_revision=0,
                         payload_ref=deliverable_artifact.id)

    evidence_claim = store.submit_evidence(
        contract.id, kind=duties[0] if duties else "test run",
        claim=f"acceptance run on the patched tree: "
              f"{patched['tests_passed']} passed, "
              f"{patched['tests_failed']} failed",
        producer=_SOFTWARE_WORKER,
        evidence_ref=patched_record.id)
    recorded_rc = _recorded_rc(artifacts.read_text(patched_record.id))
    patched_ok = recorded_rc == 0
    if patched_ok:
        store.verify_claim(evidence_claim.id,
                           validator=_SOFTWARE_VERIFIER,
                           verdict="supported")
    challenge = store.challenge(
        contract.id, challenger=_SOFTWARE_SKEPTIC,
        claim="a green run on the patched tree alone proves "
              "nothing — the same suite must fail on the "
              "unpatched tree",
        target_claim_id=evidence_claim.id)
    if patched_ok and pristine["rc"] == 0:
        store.resolve_challenge(
            challenge.id, resolver=_SOFTWARE_VERIFIER,
            verdict="contradicted",
            note="the suite passes on the unpatched tree too — "
                 "the change is not what makes it pass")
    elif patched_ok:
        store.resolve_challenge(
            challenge.id, resolver=_SOFTWARE_VERIFIER,
            verdict="supported",
            note=f"counterfactual recorded: pristine rc "
                 f"{pristine['rc']}, patched rc 0")

    rubric = [
        {"check": "every claim on the ledger cites evidence",
         "passed": all(c.evidence_ref for c in
                       store.claims(contract.id, limit=1000)
                       if c.kind == "evidence"),
         "detail": "evidence claims without an evidence ref would "
                   "fail this line"},
        {"check": "deliverable matches the role contract's "
                  "output schema",
         "passed": all(name in deliverable
                       for name in contract_terms.output_schema),
         "detail": "schema: " + ", ".join(
             contract_terms.output_schema)},
        {"check": "acceptance checks pass and the run is "
                  "recorded, not remembered",
         "passed": bool(patched_ok),
         "detail": f"patched acceptance rc {patched['rc']}, "
                   f"record {patched_record.id}"},
        {"check": "independence holds — no actor verifies their "
                  "own work",
         "passed": _independence_held(store, contract),
         "detail": "read off the claim ledger"},
    ]

    if patched_ok and pristine["rc"] != 0:
        store.sign_off(contract.id, verifier=_SOFTWARE_VERIFIER,
                       note="software pilot: counterfactual and "
                            "patched runs both recorded")
        decision = store.complete(contract.id, by=_SOFTWARE_VERIFIER)
        if not decision.ok:
            store.report_failure(
                contract.id, by=_SOFTWARE_WORKER,
                detail="completion gate refused: "
                       + "; ".join(decision.reasons or []))
    else:
        why = ("the acceptance suite still fails on the patched "
               f"tree (rc {patched['rc']})" if not patched_ok
               else "the acceptance suite passes without the "
                    "change — the skeptic's counterfactual "
                    "contradicts the evidence")
        store.report_failure(contract.id, by=_SOFTWARE_WORKER,
                             detail=why)
    contract = store.get_contract(contract.id) or contract

    economics = run_pack_fixtures(
        pack.fixtures if fixtures is None else fixtures,
        timeout=fixture_timeout)
    detail = {
        "pristine_rc": pristine["rc"],
        "patched_rc": patched["rc"],
        "pristine_record": pristine_record.id,
        "patched_record": patched_record.id,
        "deliverable": deliverable,
        "fixture_economics": economics,
    }
    receipt = _receipt_for_contract(
        store, contract, source="delegation_contract",
        kind="software_pilot", subject=title, actor=_SOFTWARE_WORKER,
        operation_ref=patched_record.id if patched_ok else "",
        detail=detail)
    return PilotRun(domain="software", pack_id=pack.id,
                    outcome=contract.status,
                    contract_id=contract.id, receipt=receipt,
                    rubric=tuple(rubric), fixtures=economics,
                    absent=absent, detail=detail)


# ── P6: the visual/media pilot ───────────────────────────────────

_MEDIA_WORKER = "media-editor"
_MEDIA_SKEPTIC = "media-skeptic"
_MEDIA_VERIFIER = "media-verifier"

_MIME_BY_EXT = {".png": "image/png", ".jpeg": "image/jpeg",
                ".jpg": "image/jpeg", ".webp": "image/webp",
                ".gif": "image/gif"}


def run_media_pilot(db: Any, *, workdir: Any, blob_dir: Any,
                    source_bytes: bytes, ops: list[dict[str, Any]],
                    rights: dict[str, Any],
                    library: FileSkillLibrary | None = None,
                    label: str = "media pilot edit",
                    job_timeout: float = 180.0) -> PilotRun:
    """Run the visual/media pack over one real media job.

    The job travels the shipped substrate: a durable media job
    (``media_edit.jobs.JobManager``) executes the edit graph — the
    op chain, validated against the shipped allowlist — through
    ``media_edit.images.edit_image`` (the Pillow rung), and the
    output lands as an artifact derived from the source, carrying
    in its provenance metadata the rights manifest the caller
    declared and the edit graph that produced it.

    Absent-with-reason, named in the run: the GPU generation lanes
    (SDXL/FLUX/ComfyUI) and the audio/video surfaces — this pilot
    proves the deterministic edit rung and the provenance record,
    and claims nothing for lanes it did not run.
    """
    library = library or FileSkillLibrary()
    pack = _valid_pack(library, MEDIA_PACK_ID)
    try:
        clean_ops = validate_ops(ops)
    except MediaEditError as exc:
        raise PilotError(f"the edit graph does not validate: {exc}") \
            from exc
    absent = {
        "generation lanes": "absent — SDXL/FLUX/ComfyUI generation "
                            "is not exercised offline; the pilot "
                            "runs the deterministic Pillow edit rung",
        "audio and video surfaces": "absent — this pilot's job is "
                                    "a still-image edit; the mix "
                                    "and video surfaces keep their "
                                    "own receipts",
    }
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    store = DelegationStore(db)
    artifacts = ArtifactStore(db, BlobStore(db, Path(blob_dir)))
    source_path = workdir / "source.png"
    source_path.write_bytes(source_bytes)
    source_artifact = artifacts.put(
        source_bytes, type="image", creator="owner",
        mime="image/png",
        provenance={"source_type": "user",
                    "source_id": "media pilot source"})

    contract = store.create_contract(
        goal=label,
        deliverable="the edited image as a versioned artifact, "
                    "its rights manifest and edit graph recorded "
                    "in provenance metadata",
        worker=_MEDIA_WORKER, skeptic=_MEDIA_SKEPTIC,
        verifier=_MEDIA_VERIFIER, arbiter="owner",
        consequential=True,
        required_evidence=("media-job", "validation"),
        scope=("image",), step_budget=10)

    manager = JobManager(db)

    def _job(_progress: Any) -> dict[str, Any]:
        return edit_image(source_path, clean_ops,
                          out_dir=workdir / "edited")

    job_id = manager.submit("image", label, _job,
                            input_ref=source_artifact.id)
    info = manager.wait(job_id, timeout=job_timeout)
    if info.get("status") != "done":
        store.report_failure(
            contract.id, by=_MEDIA_WORKER,
            detail=f"media job {info.get('status')}: "
                   f"{info.get('error') or 'no result recorded'}")
        contract = store.get_contract(contract.id) or contract
        detail = {"job_status": info.get("status"),
                  "job_error": info.get("error") or ""}
        receipt = _receipt_for_contract(
            store, contract, source="media_job", kind="media_pilot",
            subject=label, actor=_MEDIA_WORKER, detail=detail)
        return PilotRun(domain="media", pack_id=pack.id,
                        outcome=contract.status,
                        contract_id=contract.id, receipt=receipt,
                        absent=absent, detail=detail)

    result = info.get("result") or {}
    output_path = Path(str(result.get("output") or ""))
    probe = image_probe(output_path)
    edit_graph = [
        {"step": i + 1, "op": op["op"],
         "params": {k: v for k, v in op.items() if k != "op"}}
        for i, op in enumerate(clean_ops)]
    output_artifact = artifacts.derive(
        output_path.read_bytes(), from_ids=[source_artifact.id],
        type="image", creator=_MEDIA_WORKER,
        mime=_MIME_BY_EXT.get(output_path.suffix.lower(),
                              "image/png"),
        metadata={"rights": dict(rights or {}),
                  "edit_graph": edit_graph,
                  "edit_engine": "media_edit.images (Pillow chain)",
                  "job_ref": f"render:{job_id}",
                  "validation": {"width": probe["width"],
                                 "height": probe["height"],
                                 "format": probe["format"],
                                 "bytes": probe["bytes"]}})

    job_claim = store.submit_evidence(
        contract.id, kind="media-job",
        claim=f"media job recorded done: {label}",
        producer=_MEDIA_WORKER, evidence_ref=f"render:{job_id}")
    record = MediaJobRecordStore(db).get(job_id)
    if record is not None and record.get("status") == "done":
        job_claim = store.verify_claim(
            job_claim.id, validator=_MEDIA_VERIFIER,
            verdict="supported")
    validation_claim = store.submit_evidence(
        contract.id, kind="validation",
        claim=f"output probes as a {probe['format']} image, "
              f"{probe['width']}x{probe['height']}, "
              f"{probe['bytes']} bytes",
        producer=_MEDIA_WORKER, evidence_ref=output_artifact.id)
    reprobe = image_probe(output_path)
    if reprobe["width"] == probe["width"] \
            and reprobe["height"] == probe["height"] \
            and reprobe["bytes"] > 0:
        validation_claim = store.verify_claim(
            validation_claim.id, validator=_MEDIA_VERIFIER,
            verdict="supported")
    challenge = store.challenge(
        contract.id, challenger=_MEDIA_SKEPTIC,
        claim="the deliverable must be the declared edit graph's "
              "output and carry its rights manifest — nothing "
              "undeclared, nothing unowned",
        target_claim_id=validation_claim.id)
    stored = artifacts.get(output_artifact.id)
    graph_ok = stored is not None \
        and (stored.metadata.get("edit_graph") or []) == edit_graph \
        and bool((stored.metadata.get("rights") or {}).get("licence"))
    store.resolve_challenge(
        challenge.id, resolver=_MEDIA_VERIFIER,
        verdict="supported" if graph_ok else "contradicted",
        note="output artifact re-read: edit graph and rights "
             "manifest match the declaration"
             if graph_ok else
             "output artifact's metadata does not match the "
             "declared edit graph / rights")

    rubric = [
        {"check": "the job is recorded durably, not just run",
         "passed": bool(record and record.get("status") == "done"),
         "detail": f"media job record render:{job_id}"},
        {"check": "the output validates as the image the edit "
                  "graph implies",
         "passed": validation_claim.verdict == "supported",
         "detail": f"{probe['format']} {probe['width']}x"
                   f"{probe['height']}"},
        {"check": "rights manifest recorded on the deliverable",
         "passed": bool((stored.metadata.get("rights") or {})
                        .get("licence")) if stored else False,
         "detail": "artifact provenance metadata, additive — no "
                   "schema change"},
        {"check": "edit graph recorded on the deliverable",
         "passed": bool(graph_ok),
         "detail": f"{len(edit_graph)} op(s), in order"},
        {"check": "independence holds — no actor verifies their "
                  "own work",
         "passed": _independence_held(store, contract),
         "detail": "read off the claim ledger"},
    ]
    if graph_ok and validation_claim.verdict == "supported":
        store.sign_off(contract.id, verifier=_MEDIA_VERIFIER,
                       note="media pilot: job, validation and "
                            "provenance recorded")
        decision = store.complete(contract.id, by=_MEDIA_VERIFIER)
        if not decision.ok:
            store.report_failure(
                contract.id, by=_MEDIA_WORKER,
                detail="completion gate refused: "
                       + "; ".join(decision.reasons or []))
    else:
        store.report_failure(
            contract.id, by=_MEDIA_WORKER,
            detail="the deliverable did not validate against the "
                   "declared edit graph / rights manifest")
    contract = store.get_contract(contract.id) or contract
    detail = {"job_ref": f"render:{job_id}",
              "output_artifact": output_artifact.id,
              "source_artifact": source_artifact.id,
              "probe": {"width": probe["width"],
                        "height": probe["height"],
                        "format": probe["format"],
                        "bytes": probe["bytes"]}}
    receipt = _receipt_for_contract(
        store, contract, source="media_job", kind="media_pilot",
        subject=label, actor=_MEDIA_WORKER,
        operation_ref=output_artifact.id, detail=detail)
    return PilotRun(domain="media", pack_id=pack.id,
                    outcome=contract.status,
                    contract_id=contract.id, receipt=receipt,
                    rubric=tuple(rubric), absent=absent,
                    detail=detail)


# ── P7: the CBT/wellbeing pilot ──────────────────────────────────

_CBT_WORKER = "cbt-coach"
_CBT_SKEPTIC = "cbt-skeptic"
_CBT_VERIFIER = "cbt-verifier"


def run_cbt_pilot(db: Any, *, exercise: dict[str, Any],
                  interventions_path: Any = None,
                  library: FileSkillLibrary | None = None) -> PilotRun:
    """Run one CBT-pack exercise as a cell on the health shell.

    ``exercise`` is the person's own work: ``{"kind": <one of the
    intervention kinds>, "fields": {...}}``.  The cell structures
    it into a stored intervention record, the skeptic scans the
    record's rendered text against the shell's no-diagnosis guard
    vocabulary, and the verifier accepts the record only when the
    stored record re-validates complete against its kind's schema.

    An exercise whose fields do not meet the schema is refused by
    the store with the problems named, and the run records that
    refusal as its failure — the record is never half-stored.  A
    record carrying diagnostic language is stored, blocked by the
    skeptic, and the run fails with the record marked blocked:
    the exercise happened, its delivery did not.
    """
    library = library or FileSkillLibrary()
    pack = _valid_pack(library, CBT_PACK_ID)
    kind = str(exercise.get("kind") or "")
    fields = dict(exercise.get("fields") or {})
    store = DelegationStore(db)
    interventions = InterventionStore(interventions_path)
    subject = f"{kind.replace('_', ' ')} exercise" if kind \
        else "CBT exercise"
    contract = store.create_contract(
        goal=f"structure one {subject} into a complete "
             "intervention record on the health shell",
        deliverable="a stored intervention record that "
                    "re-validates complete and carries no "
                    "diagnostic language",
        worker=_CBT_WORKER, skeptic=_CBT_SKEPTIC,
        verifier=_CBT_VERIFIER, arbiter="owner",
        consequential=True,
        required_evidence=("intervention-record",),
        scope=("health",), step_budget=10)
    absent = {
        "crisis routing": "absent — the shell's mood-triggered "
                          "crisis resources (health/drift) are "
                          "shipped behavior an exercise run does "
                          "not exercise, and this pass does not "
                          "change them",
    }
    detail: dict[str, Any] = {"kind": kind}
    try:
        record = interventions.record(kind, fields,
                                      contract_id=contract.id)
    except InterventionError as exc:
        store.report_failure(contract.id, by=_CBT_WORKER,
                             detail=str(exc))
        contract = store.get_contract(contract.id) or contract
        detail["refusal"] = str(exc)
        receipt = _receipt_for_contract(
            store, contract, source="cbt_intervention",
            kind="cbt_pilot", subject=subject, actor=_CBT_WORKER,
            detail=detail)
        rubric = (
            {"check": "the record meets its kind's schema",
             "passed": False, "detail": str(exc)},
            {"check": "independence holds — no actor verifies "
                      "their own work",
             "passed": _independence_held(store, contract),
             "detail": "read off the claim ledger"},
        )
        return PilotRun(domain="cbt", pack_id=pack.id,
                        outcome=contract.status,
                        contract_id=contract.id, receipt=receipt,
                        rubric=rubric, absent=absent, detail=detail)

    detail["record_id"] = record["id"]
    claim = store.submit_evidence(
        contract.id, kind="intervention-record",
        claim=f"{subject} stored as a structured intervention "
              "record",
        producer=_CBT_WORKER,
        evidence_ref=f"intervention:{record['id']}")
    problems = interventions.validate_record(record["id"])
    if not problems:
        store.verify_claim(claim.id, validator=_CBT_VERIFIER,
                           verdict="supported")
    rendered = render_intervention(kind, fields)
    hits = diagnostic_scan(rendered)
    challenge = store.challenge(
        contract.id, challenger=_CBT_SKEPTIC,
        claim="this record must describe the person's own "
              "exercise work only — no diagnostic or "
              "prescriptive language, nothing beyond the "
              "exercise's structure",
        target_claim_id=claim.id)
    if hits:
        interventions.set_status(record["id"], "blocked")
        store.resolve_challenge(
            challenge.id, resolver=_CBT_VERIFIER,
            verdict="contradicted",
            note="diagnostic language found in the record: "
                 + ", ".join(repr(h) for h in hits))
        store.report_failure(
            contract.id, by=_CBT_WORKER,
            detail="the skeptic blocked the record: diagnostic "
                   "language " + ", ".join(repr(h) for h in hits))
    else:
        store.resolve_challenge(
            challenge.id, resolver=_CBT_VERIFIER,
            verdict="supported",
            note="the record's rendered text scans clean "
                 "against the guard vocabulary")
    rubric = [
        {"check": "the record is stored on the health shell",
         "passed": True,
         "detail": f"intervention:{record['id']}"},
        {"check": "the stored record re-validates complete "
                  "against its kind's schema",
         "passed": not problems,
         "detail": "; ".join(problems) or "schema holds on "
                                          "read-back"},
        {"check": "no diagnostic or prescriptive language in "
                  "the record",
         "passed": not hits,
         "detail": "guard vocabulary scan of the rendered "
                   "record"},
        {"check": "independence holds — no actor verifies "
                  "their own work",
         "passed": _independence_held(store, contract),
         "detail": "read off the claim ledger"},
    ]
    contract = store.get_contract(contract.id) or contract
    if contract.status != "failed":
        store.sign_off(contract.id, verifier=_CBT_VERIFIER,
                       note="CBT pilot: record stored, complete "
                            "on read-back, scans clean")
        decision = store.complete(contract.id, by=_CBT_VERIFIER)
        if not decision.ok:
            store.report_failure(
                contract.id, by=_CBT_WORKER,
                detail="completion gate refused: "
                       + "; ".join(decision.reasons or []))
        contract = store.get_contract(contract.id) or contract
    receipt = _receipt_for_contract(
        store, contract, source="cbt_intervention",
        kind="cbt_pilot", subject=subject, actor=_CBT_WORKER,
        operation_ref=f"intervention:{record['id']}",
        detail=detail)
    return PilotRun(domain="cbt", pack_id=pack.id,
                    outcome=contract.status,
                    contract_id=contract.id, receipt=receipt,
                    rubric=tuple(rubric), absent=absent,
                    detail=detail)
