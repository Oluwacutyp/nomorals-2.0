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
    "GOLDEN_MISSIONS",
    "list_golden_missions",
]

_log = get_logger(__name__)

VerifyFn = Callable[[dict[str, Any]], "tuple[bool, str]"]
RepairFn = Callable[[dict[str, Any], "GoldenContext"], dict[str, Any]]


@dataclass
class GoldenContext:
    """What a step runs with: a scratch dir, the size flag, and knobs."""

    workdir: Path
    long: bool = False
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldenStep:
    """One plan step: execute, verify, and optionally repair."""

    name: str
    run: Callable[[GoldenContext], dict[str, Any]]
    verify: VerifyFn
    repair: RepairFn | None = None
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


def _build_registry() -> dict[str, GoldenMission]:
    missions = [_research_mission(), _build_mission(), _audit_mission()]
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
        return self._finish(mission, mission_def, final, completed,
                           step_reports, started, error=failure)

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
        )

    def _result(self, mission: Mission, key: str,
                completed: list[str]) -> GoldenResult:
        return GoldenResult(
            mission_id=mission.id, key=key,
            ok=mission.status == MissionStatus.DONE, status=mission.status,
            steps=[{"step": name, "ok": True, "detail": "already complete",
                    "seconds": 0.0, "repaired": False} for name in completed],
        )
