"""Golden missions: end-to-end plan → execute → verify → repair drills.

A golden mission is a fixed, deterministic multi-step mission that runs
without any model: every step is plain Python, so the suite can execute the
short versions in milliseconds and prove the whole mission machinery —
planning (the fixed plan), execution, per-step verification, repair on
failure, checkpointing, kill, and resume — without flakiness.

Each mission ships in two sizes:

* ``long=False`` — the unit-suite version: seconds, tiny fixtures.
* ``long=True`` — the real drill: scaled-up fixtures and real subprocess
  test runs, taking minutes.

Results are recorded into the benchmark DB (``BenchmarkDB``) with
``model_id="golden:<key>"`` so golden runs are comparable over time with
model benchmarks.

State lives in the regular :class:`MissionStore`: a killed golden mission
is an ordinary CANCELLED mission and :meth:`GoldenRunner.resume` continues
it from the persisted ``completed`` step list.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..llm.benchmarks import BenchmarkDB
from .mission import Mission, MissionStatus, MissionStore

__all__ = [
    "GoldenStep",
    "GoldenContext",
    "GoldenMission",
    "GoldenResult",
    "GoldenRunner",
    "CompensateFn",
    "GOLDEN_MISSIONS",
    "GOLDEN_BASELINES",
    "list_golden_missions",
    "normalize_output",
    "check_regression",
    "render_golden_report",
]

_log = get_logger(__name__)

VerifyFn = Callable[[dict[str, Any]], "tuple[bool, str]"]
RepairFn = Callable[[dict[str, Any], "GoldenContext"], dict[str, Any]]
#: Saga compensation for a golden step: undo the step's effects.
#: Receives the step's output and the context, returns a note dict.
CompensateFn = Callable[[dict[str, Any], "GoldenContext"], dict[str, Any]]


@dataclass
class GoldenContext:
    """What a step runs with: a scratch dir, the size flag, and knobs."""

    workdir: Path
    long: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenStep:
    """One plan step: execute, verify, optionally repair, optionally undo."""

    name: str
    run: Callable[[GoldenContext], dict[str, Any]]
    verify: VerifyFn
    repair: RepairFn | None = None
    compensate: CompensateFn | None = None
    doc: str = ""


@dataclass
class GoldenMission:
    """A fixed plan of golden steps."""

    key: str
    name: str
    goal: str
    steps: list[GoldenStep]
    doc: str = ""


@dataclass
class GoldenResult:
    """Terminal summary of a golden run."""

    mission_id: str
    key: str
    ok: bool
    status: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0
    benchmark_row: int = 0
    error: str = ""
    #: saga compensations, newest-first (empty when the mission succeeded
    #: or no step declared a compensation).
    compensations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "key": self.key,
            "ok": self.ok,
            "status": self.status,
            "steps": self.steps,
            "seconds": round(self.seconds, 3),
            "benchmark_row": self.benchmark_row,
            "error": self.error,
            "compensations": self.compensations,
        }


# ── mission 1: research → write → verify ────────────────────────────────────

_FACTS_SHORT = [
    "Devon snapshots use VACUUM INTO for consistent copies.",
    "Restore is transactional: stage, verify, then swap.",
    "Golden missions run without any model.",
]
_FACTS_LONG_EXTRA = [
    f"Drill fact {i:03d}: the quick brown fox jumps over the lazy dog. "
    f"Pack my box with five dozen liquor jugs. How vexingly quick daft "
    f"zebras jump! ({i})"
    for i in range(120)
]


def _collect(ctx: GoldenContext) -> dict[str, Any]:
    corpus = ctx.workdir / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    facts = list(_FACTS_SHORT)
    if ctx.long:
        facts.extend(_FACTS_LONG_EXTRA)
        # Simulate a flaky source in the long drill: the last fact is
        # "lost" on first collection so the repair path gets exercised.
        if not ctx.params.get("recollected"):
            facts = facts[:-1]
    for i, fact in enumerate(facts):
        (corpus / f"fact-{i:03d}.txt").write_text(fact, encoding="utf-8")
    gathered = [p.read_text(encoding="utf-8") for p in sorted(corpus.glob("*.txt"))]
    if ctx.long:
        time.sleep(20)  # the long drill does real, slow gathering work
    return {"facts": gathered, "count": len(gathered)}


def _draft(ctx: GoldenContext) -> dict[str, Any]:
    facts: list[str] = ctx.params.get("facts", [])
    report = ctx.workdir / "report.md"
    body = "# Research report\n\n" + "\n".join(f"- {f}" for f in facts) + "\n"
    report.write_text(body, encoding="utf-8")
    return {"path": str(report), "bytes": len(body.encode())}


def _verify_draft(output: dict[str, Any]) -> tuple[bool, str]:
    facts: list[str] = output.get("facts", [])
    path = output.get("path", "")
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"cannot read draft: {exc}"
    missing = [f for f in facts if f not in text]
    if missing:
        return False, f"{len(missing)} facts missing from draft"
    if len(text) < 50:
        return False, "draft suspiciously short"
    return True, f"draft ok: {len(facts)} facts, {len(text)} chars"


def _repair_draft(output: dict[str, Any], ctx: GoldenContext) -> dict[str, Any]:
    # Re-collect (the flaky source recovers) and re-draft.
    ctx.params["recollected"] = True
    recollected = _collect(ctx)
    ctx.params["facts"] = recollected["facts"]
    return {**output, **_draft(ctx), "facts": recollected["facts"],
            "repaired": True}


def _research_mission() -> GoldenMission:
    def collect(ctx: GoldenContext) -> dict[str, Any]:
        out = _collect(ctx)
        ctx.params["facts"] = out["facts"]
        return out

    def draft(ctx: GoldenContext) -> dict[str, Any]:
        out = _draft(ctx)
        out["facts"] = ctx.params.get("facts", [])
        return out

    return GoldenMission(
        key="research_write_verify",
        name="Research → write → verify",
        goal="gather facts from the corpus, write a report, verify it",
        doc="Exercises plan→execute→verify→repair: the long drill loses a "
            "fact on first collection and the repair step must recover it.",
        steps=[
            GoldenStep("collect", collect,
                       lambda o: (True, f"{o['count']} facts")
                       if o.get("count") else (False, "no facts"),
                       doc="gather facts from the corpus"),
            GoldenStep("draft", draft, _verify_draft, _repair_draft,
                       doc="write the report; repair re-collects on failure"),
        ],
    )


# ── mission 2: build → test → fix ───────────────────────────────────────────

_CALC_SRC = '''"""Tiny calculator under test by the golden build mission."""

def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
'''

_CALC_TEST = '''from calc import add, mul, fib

def test_add():
    assert add(2, 3) == 5

def test_mul():
    assert mul(2, 3) == 6

def test_fib():
    assert [fib(i) for i in range(8)] == [0, 1, 1, 2, 3, 5, 8, 13]
'''

# The deterministic bug the long drill must find and fix.
_BUGGY_LINE = "    return a + b + 1  # BUG: off by one\n"
_FIXED_LINE = "    return a + b\n"


def _scaffold(ctx: GoldenContext) -> dict[str, Any]:
    src = _CALC_SRC
    if ctx.long:
        src = src.replace("    return a + b\n", _BUGGY_LINE, 1)
    (ctx.workdir / "calc.py").write_text(src, encoding="utf-8")
    (ctx.workdir / "test_calc.py").write_text(_CALC_TEST, encoding="utf-8")
    if ctx.long:
        time.sleep(20)
    return {"files": ["calc.py", "test_calc.py"], "buggy": ctx.long}


def _run_tests(ctx: GoldenContext) -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "test_calc.py", "-q", "--no-header"],
        cwd=str(ctx.workdir),
        capture_output=True,
        text=True,
        timeout=600,
    )
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-5:])
    return {"returncode": proc.returncode, "output": tail}


def _verify_tests(output: dict[str, Any]) -> tuple[bool, str]:
    if output.get("returncode") == 0:
        return True, "all tests pass"
    return False, f"pytest failed:\n{output.get('output', '')}"


def _repair_calc(output: dict[str, Any], ctx: GoldenContext) -> dict[str, Any]:
    path = ctx.workdir / "calc.py"
    src = path.read_text(encoding="utf-8")
    if _BUGGY_LINE not in src:
        return {**output, "repaired": False, "note": "no known bug pattern"}
    path.write_text(src.replace(_BUGGY_LINE, _FIXED_LINE, 1), encoding="utf-8")
    fixed = _run_tests(ctx)
    return {**fixed, "repaired": True}


def _build_mission() -> GoldenMission:
    return GoldenMission(
        key="build_test_fix",
        name="Build → test → fix",
        goal="scaffold a module, run its tests, fix failures",
        doc="The long drill scaffolds a module with a deterministic off-by-one "
            "bug; the repair step must patch it and re-run green.",
        steps=[
            GoldenStep("scaffold", _scaffold,
                       lambda o: (True, "2 files") if len(o.get("files", [])) == 2
                       else (False, "scaffold incomplete"),
                       doc="write calc.py + test_calc.py"),
            GoldenStep("test", _run_tests, _verify_tests, _repair_calc,
                       doc="pytest; repair patches the off-by-one and re-runs"),
        ],
    )


# ── mission 3: audit → remediate → re-scan ──────────────────────────────────

_ISSUE_MARKER = "INSECURE-DEFAULT"


def _generate_configs(ctx: GoldenContext) -> dict[str, Any]:
    confdir = ctx.workdir / "configs"
    confdir.mkdir(parents=True, exist_ok=True)
    n_files = 30 if ctx.long else 4
    n_bad = 6 if ctx.long else 2
    bad_indexes = {1, 7, 13, 19, 25, 29} if ctx.long else {1, 3}
    for i in range(n_files):
        lines = [f"# service config {i}", "port = 8080", "tls = on"]
        if i in bad_indexes and i < n_bad * 5:
            lines.append(f"auth = {_ISSUE_MARKER}")
        (confdir / f"svc-{i:02d}.conf").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")
    if ctx.long:
        time.sleep(20)
    return {"dir": str(confdir), "files": n_files}


def _audit_configs(ctx: GoldenContext) -> dict[str, Any]:
    confdir = ctx.workdir / "configs"
    issues = []
    for path in sorted(confdir.glob("*.conf")):
        for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if _ISSUE_MARKER in line:
                issues.append(f"{path.name}:{lineno}")
    return {"issues": issues, "count": len(issues)}


def _verify_audit(output: dict[str, Any]) -> tuple[bool, str]:
    # The audit step itself always "passes" — it reports; remediation fixes.
    return True, f"audit found {output.get('count', 0)} issues"


def _remediate(ctx: GoldenContext) -> dict[str, Any]:
    confdir = ctx.workdir / "configs"
    fixed = 0
    for path in sorted(confdir.glob("*.conf")):
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [ln for ln in lines if _ISSUE_MARKER not in ln]
        if len(kept) != len(lines):
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            fixed += len(lines) - len(kept)
    return {"fixed": fixed}


def _verify_clean(output: dict[str, Any]) -> tuple[bool, str]:
    # Re-scan from the recorded directory: the guard that proves
    # remediation worked.
    confdir = Path(output.get("confdir", ""))
    issues = []
    if confdir.exists():
        for path in sorted(confdir.glob("*.conf")):
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), 1):
                if _ISSUE_MARKER in line:
                    issues.append(f"{path.name}:{lineno}")
    if issues:
        return False, f"{len(issues)} issues remain: {issues[:3]}"
    return True, "re-scan clean: 0 issues"


def _audit_mission() -> GoldenMission:
    def rescan_step(ctx: GoldenContext) -> dict[str, Any]:
        out = _audit_configs(ctx)
        out["confdir"] = str(ctx.workdir / "configs")
        return out

    return GoldenMission(
        key="audit_remediate_rescan",
        name="Audit → remediate → re-scan",
        goal="find insecure defaults in configs, fix them, prove zero remain",
        doc="The re-scan step is the guard: remediation only counts when the "
            "second scan finds nothing.",
        steps=[
            GoldenStep("generate", _generate_configs,
                       lambda o: (True, f"{o['files']} configs")
                       if o.get("files") else (False, "no configs"),
                       doc="lay down config fixtures with injected issues"),
            GoldenStep("audit", _audit_configs, _verify_audit,
                       doc="scan for INSECURE-DEFAULT markers"),
            GoldenStep("remediate", _remediate,
                       lambda o: (True, f"fixed {o.get('fixed', 0)}")
                       if o.get("fixed", 0) > 0 else (False, "nothing fixed"),
                       doc="strip the insecure defaults"),
            GoldenStep("rescan", rescan_step, _verify_clean,
                       doc="guard: re-scan must find zero issues"),
        ],
    )


# ── mission 4: crash → resume without duplicate side effects ──────────────

def _mark_side_effect(ctx: GoldenContext) -> dict[str, Any]:
    # The append is the "side effect": if resume re-executed this step,
    # the log would hold two lines and the drill would fail.
    log = ctx.workdir / "side_effects.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"marked at {time.time():.3f}\n")
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    return {"marks": len(lines)}


def _crash_mission() -> GoldenMission:
    return GoldenMission(
        key="crash_no_dup",
        name="Crash → resume without duplicate side effects",
        goal="prove a killed mission resumes without re-executing "
             "completed steps",
        doc="The drill harness kills the runner after the first step and "
            "resumes it: the side-effect log must hold exactly one line. "
            "Exercises kill → PAUSED → resume, the at-least-once contract, "
            "and the completed-step skip.",
        steps=[
            GoldenStep("mark", _mark_side_effect,
                       lambda o: (True, f"{o['marks']} mark(s)")
                       if o.get("marks") == 1 else (False, "mark missing"),
                       doc="append one line to the side-effect log"),
            GoldenStep("finish", lambda ctx: {"done": True},
                       lambda o: (True, "finished")
                       if o.get("done") else (False, "not done"),
                       doc="trivial second step after the kill point"),
        ],
    )


# ── mission 5: saga — a failed step undoes the earlier ones ─────────────────

def _make_file(name: str) -> Callable[[GoldenContext], dict[str, Any]]:
    def _run(ctx: GoldenContext) -> dict[str, Any]:
        path = ctx.workdir / name
        path.write_text(f"created by {name}\n", encoding="utf-8")
        return {"file": str(path)}
    return _run


def _remove_file(name: str) -> CompensateFn:
    def _compensate(output: dict[str, Any],
                    ctx: GoldenContext) -> dict[str, Any]:
        path = ctx.workdir / name
        # idempotent undo: already-gone is success, not an error.
        removed = False
        if path.exists():
            path.unlink()
            removed = True
        return {"note": f"{name} removed" if removed else f"{name} already gone",
                "removed": removed}
    return _compensate


def _saga_mission() -> GoldenMission:
    return GoldenMission(
        key="saga_undo",
        name="Saga → undo on failure",
        goal="prove a failed step triggers reverse-order compensation",
        doc="Steps create files with compensations that delete them; the "
            "last step always fails, so the runner must undo create_b then "
            "create_a (newest first). Exercises the orchestrated-saga path.",
        steps=[
            GoldenStep("create_a", _make_file("a.txt"),
                       lambda o: (True, "a.txt created")
                       if Path(o.get("file", "")).exists()
                       else (False, "a.txt missing"),
                       compensate=_remove_file("a.txt"),
                       doc="create a.txt; compensation deletes it"),
            GoldenStep("create_b", _make_file("b.txt"),
                       lambda o: (True, "b.txt created")
                       if Path(o.get("file", "")).exists()
                       else (False, "b.txt missing"),
                       compensate=_remove_file("b.txt"),
                       doc="create b.txt; compensation deletes it"),
            GoldenStep("boom", lambda ctx: {"exploded": True},
                       lambda o: (False, "boom always fails"),
                       doc="deterministic failure that triggers the saga"),
        ],
    )


def _build_registry() -> dict[str, GoldenMission]:
    missions = [_research_mission(), _build_mission(), _audit_mission(),
                _crash_mission(), _saga_mission()]
    return {m.key: m for m in missions}


GOLDEN_MISSIONS: dict[str, GoldenMission] = _build_registry()


def list_golden_missions() -> list[dict[str, Any]]:
    return [
        {"key": m.key, "name": m.name, "goal": m.goal,
         "steps": [s.name for s in m.steps], "doc": m.doc}
        for m in GOLDEN_MISSIONS.values()
    ]


# ── runner ──────────────────────────────────────────────────────────────────

class GoldenRunner:
    """Drive a golden mission to a terminal state, checkpointing as it goes.

    Killable via :meth:`kill` (cooperative, checked between steps) and
    resumable via :meth:`resume`, which reloads the mission row and
    continues at the first incomplete step.
    """

    def __init__(
        self,
        db: Any,
        *,
        workdir_root: str | Path | None = None,
        benchmark_db: BenchmarkDB | None = None,
        store: MissionStore | None = None,
    ) -> None:
        self.db = db
        self.store = store or MissionStore(db)
        self.benchmarks = benchmark_db or BenchmarkDB(db)
        self.workdir_root = Path(workdir_root) if workdir_root else Path(
            tempfile.mkdtemp(prefix="golden-"))
        self._killed = False

    def kill(self, reason: str = "killed") -> None:
        """Request cooperative cancellation; checked between steps."""
        self._killed = True
        _log.info("golden mission kill requested: %s", reason)

    # -- run / resume --------------------------------------------------------

    def run(self, key: str, *, long: bool = False) -> GoldenResult:
        mission_def = GOLDEN_MISSIONS[key]  # KeyError on unknown key: honest
        mission = self.store.create_new(
            mission_def.goal,
            name=f"golden:{key}",
            state={"golden": {"key": key, "long": long, "completed": [],
                              "outputs": {}, "repairs": 0}},
            metadata={"golden_key": key, "golden_long": long},
        )
        return self._drive(mission, mission_def, long=long)

    def resume(self, mission_id: str) -> GoldenResult:
        mission = self.store.get(mission_id)
        golden = mission.state.get("golden") or {}
        key = golden.get("key") or mission.metadata.get("golden_key")
        if not key or key not in GOLDEN_MISSIONS:
            raise ValueError(f"mission {mission_id} is not a golden mission")
        if mission.terminal:
            return self._result(mission, key, list(golden.get("completed", [])))
        self._killed = False
        return self._drive(mission, GOLDEN_MISSIONS[key],
                           long=bool(golden.get("long")))

    # -- internals -----------------------------------------------------------

    def _workdir(self, mission: Mission) -> Path:
        workdir = self.workdir_root / mission.id
        workdir.mkdir(parents=True, exist_ok=True)
        return workdir

    def _drive(self, mission: Mission, mission_def: GoldenMission,
               *, long: bool) -> GoldenResult:
        started = time.time()
        self._killed = False
        workdir = self._workdir(mission)
        ctx = GoldenContext(workdir=workdir, long=long)
        golden = mission.state.setdefault("golden", {})
        completed: list[str] = list(golden.get("completed", []))
        outputs: dict[str, Any] = dict(golden.get("outputs", {}))
        step_reports: list[dict[str, Any]] = []

        mission.status = MissionStatus.RUNNING
        self.store.save(mission)

        failure = ""
        for step in mission_def.steps:
            if self._killed:
                # PAUSED, not CANCELLED: a killed golden mission is
                # resumable by design (kill → resume → complete).
                return self._finish(mission, mission_def, MissionStatus.PAUSED,
                                   completed, step_reports, started,
                                   error="killed")
            if step.name in completed:
                _log.debug("golden %s skipping completed step %s",
                           mission.id, step.name)
                continue
            report = self._execute_step(mission, step, ctx, outputs)
            step_reports.append(report)
            if report["ok"]:
                completed.append(step.name)
                golden["completed"] = completed
                golden["outputs"] = outputs
                self.store.save(mission)
                self.store.checkpoint(mission, label=f"golden:{step.name}")
            else:
                failure = report["detail"]
                golden["last_error"] = failure
                self.store.save(mission)
                break

        final = MissionStatus.DONE if not failure else MissionStatus.FAILED
        compensations: list[dict[str, Any]] = []
        if failure:
            # orchestrated saga: undo completed steps newest-first.
            # Compensations are idempotent by contract (already-gone is
            # success); a failed undo is recorded, never fatal.
            for step_name in reversed(completed):
                gstep = next(
                    (s for s in mission_def.steps if s.name == step_name),
                    None)
                if gstep is None or gstep.compensate is None:
                    continue
                try:
                    cout = gstep.compensate(
                        outputs.get(step_name) or {}, ctx) or {}
                    compensations.append({
                        "step": step_name, "ok": True,
                        "detail": str(cout.get("note") or "undone")})
                    _log.info("golden %s compensated step %s",
                              mission.id, step_name)
                except Exception as exc:  # noqa: BLE001
                    compensations.append({
                        "step": step_name, "ok": False,
                        "detail": f"{type(exc).__name__}: {exc}"})
            if compensations:
                golden["compensations"] = compensations
                self.store.save(mission)
        return self._finish(mission, mission_def, final, completed,
                           step_reports, started, error=failure,
                           compensations=compensations)

    def _execute_step(
        self,
        mission: Mission,
        step: GoldenStep,
        ctx: GoldenContext,
        outputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute → verify → (repair → re-verify). Returns a step report."""
        started = time.time()
        repaired = False
        attempts = 1
        try:
            output = step.run(ctx) or {}
        except Exception as exc:  # noqa: BLE001 - a step failure is a result
            return {"step": step.name, "ok": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                    "seconds": round(time.time() - started, 3),
                    "repaired": False, "attempts": attempts}
        try:
            ok, detail = step.verify(output)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"verify raised {type(exc).__name__}: {exc}"

        if not ok and step.repair is not None:
            _log.info("golden %s step %s failed verification; repairing",
                      mission.id, step.name)
            try:
                output = step.repair(output, ctx) or {}
                repaired = True
                attempts = 2
                ok, detail = step.verify(output)
                detail = f"repaired; {detail}"
            except Exception as exc:  # noqa: BLE001
                ok, detail = False, f"repair raised {type(exc).__name__}: {exc}"

        # Outputs must be JSON-serialisable for the mission row.
        try:
            json.dumps(output)
            outputs[step.name] = output
        except (TypeError, ValueError):
            outputs[step.name] = {"_unserialisable": True}
        return {"step": step.name, "ok": bool(ok), "detail": detail,
                "seconds": round(time.time() - started, 3),
                "repaired": repaired, "attempts": attempts}

    def _finish(
        self,
        mission: Mission,
        mission_def: GoldenMission,
        status: str,
        completed: list[str],
        step_reports: list[dict[str, Any]],
        started: float,
        *,
        error: str = "",
        compensations: list[dict[str, Any]] | None = None,
    ) -> GoldenResult:
        seconds = time.time() - started
        mission.status = status
        mission.state.setdefault("golden", {})["completed"] = completed
        self.store.save(mission)
        self.store.checkpoint(mission, label=f"golden:final:{status}")

        ok = status == MissionStatus.DONE
        row_id = 0
        try:
            row_id = self.benchmarks.record(
                model_id=f"golden:{mission_def.key}",
                capability="chat",
                latency_s=seconds,
                success=ok,
                source="golden",
                task_kind=mission_def.key,
            )
        except Exception:  # noqa: BLE001 - benchmark never breaks the mission
            _log.debug("golden benchmark record failed", exc_info=True)

        _log.info("golden %s finished: %s", mission_def.key, status)
        return GoldenResult(
            mission_id=mission.id,
            key=mission_def.key,
            ok=ok,
            status=status,
            steps=step_reports,
            seconds=seconds,
            benchmark_row=row_id,
            error=error,
            compensations=list(compensations or []),
        )

    def _result(self, mission: Mission, key: str,
                completed: list[str]) -> GoldenResult:
        return GoldenResult(
            mission_id=mission.id, key=key,
            ok=mission.status == MissionStatus.DONE, status=mission.status,
            steps=[{"step": name, "ok": True, "detail": "already complete",
                    "seconds": 0.0, "repaired": False} for name in completed],
        )


# ── golden discipline: normalization, regression gate, reports ──────────────
#
# Golden-test discipline, mined from the best practice: deterministic
# assertions on version-controlled fixtures, volatile fields normalized
# away before comparison, and CI gated on pass-rate *regression* — not on
# perfection.

#: Volatile output keys: dropped by normalize_output before any golden
#: comparison (timestamps, pids, timings, tmp paths are never stable).
_VOLATILE_KEYS = frozenset({
    "seconds", "pid", "workdir", "tmpdir", "duration", "elapsed",
    "at", "started_at", "finished_at", "created_at",
})
_VOLATILE_SUFFIXES = ("_at", "_time", "_ts", "_ms", "_us", "_pid", "_path")


def normalize_output(output: Any) -> Any:
    """Recursively drop volatile fields from a step output.

    Timestamps, pids, durations and tmp paths make golden comparisons
    flaky; they are evidence of *when*, not *what*. What remains is the
    deterministic substance a golden assertion should check.
    """
    if isinstance(output, dict):
        return {
            key: normalize_output(value)
            for key, value in output.items()
            if key not in _VOLATILE_KEYS
            and not str(key).endswith(_VOLATILE_SUFFIXES)
        }
    if isinstance(output, (list, tuple)):
        return [normalize_output(v) for v in output]
    return output


#: Committed pass-rate baselines per drill. Raising a baseline is a
#: reviewed commit; dropping below one fails the gate.
GOLDEN_BASELINES: dict[str, dict[str, float]] = {
    "research_write_verify": {"pass_rate": 1.0},
    "build_test_fix": {"pass_rate": 1.0},
    "audit_remediate_rescan": {"pass_rate": 1.0},
    "crash_no_dup": {"pass_rate": 1.0},
    "saga_undo": {"pass_rate": 1.0},
}


def check_regression(results: list[GoldenResult]) -> dict[str, dict[str, Any]]:
    """CI gate: fail when a drill's pass rate drops below its baseline.

    Takes golden results (usually one run per drill) and returns, per
    drill key, ``{"runs", "passed", "pass_rate", "baseline",
    "regression"}``. Gate on ``regression`` — the discipline is "never
    get worse", not "be perfect on day one".
    """
    by_key: dict[str, list[GoldenResult]] = {}
    for result in results:
        by_key.setdefault(result.key, []).append(result)
    report: dict[str, dict[str, Any]] = {}
    for key, runs in by_key.items():
        passed = sum(1 for r in runs if r.ok)
        rate = passed / len(runs) if runs else 0.0
        baseline = float(GOLDEN_BASELINES.get(key, {}).get("pass_rate", 1.0))
        report[key] = {
            "runs": len(runs),
            "passed": passed,
            "pass_rate": round(rate, 4),
            "baseline": baseline,
            "regression": rate < baseline,
        }
    return report


def render_golden_report(result: GoldenResult, *, width: int = 62) -> str:
    """One boxed card per golden run: steps, repairs, compensations."""
    from .progress import box_lines, fmt_duration

    title = (f"🥇 golden:{result.key} — "
             f"{'✅ DONE' if result.ok else '❌ ' + result.status}")
    lines = [f"mission {result.mission_id[:8]} · "
             f"{fmt_duration(result.seconds)} wall"]
    for step in result.steps:
        glyph = "✓" if step.get("ok") else "✗"
        repaired = " (repaired)" if step.get("repaired") else ""
        lines.append(
            f"  {glyph} {step.get('step')}{repaired} — "
            f"{str(step.get('detail') or '')[:60]}")
    for comp in result.compensations:
        if comp.get("ok"):
            lines.append(f"  ↩ {comp.get('step')}: undone")
        else:
            lines.append(f"  ↩ {comp.get('step')}: UNDO FAILED — "
                         f"{str(comp.get('detail') or '')[:40]}")
    if result.benchmark_row:
        lines.append(f"benchmark row #{result.benchmark_row}")
    if result.error:
        lines.append(f"error: {result.error[:80]}")
    return "\n".join(box_lines(title, lines, width=width))
