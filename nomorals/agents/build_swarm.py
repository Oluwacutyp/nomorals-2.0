"""Multi-worker BUILD swarm: split a build job into single-job workers,
verify every worker's output by running its tests, merge into one report.

Sibling of ``research_swarm`` (research-only: parallel researchers,
conflict-aware synthesis) and of ``swarm.py`` (investigation legs that
produce prose digests, fused by a model). Neither builds code; this one
does:

* **workers build, in a language rung** — each worker gets one role, one
  task, one language (``python`` | ``rust`` | ``go``) and its own work
  dir under ``<home>/data/swarm_builds/<run_id>/<worker>/``. Per-job,
  per-swarm — there is no central shared queue, no dispatcher, nothing
  shared between runs. The default worker engine is the real CodingAgent
  rooted at the worker dir; with no model available the leg fails
  honestly instead of inventing a digest.
* **a VERIFIER agent runs real tests** — for every leg it probes the
  rung toolchain, runs the rung's test command in the worker dir
  (``pytest`` / ``cargo test`` / ``go test``), falls back to the rung's
  compile/syntax check when the worker wrote no test files, and reviews
  the diff (files actually on disk, non-empty). ACCEPT or REJECT, always
  with concrete reasons. A rejected leg gets exactly ONE retry with the
  verifier's reasons appended to its brief; a second rejection is final.
* **honest failure** — a crashed worker, an empty diff, un-runnable
  tests, a missing toolchain all land as REJECTED with the reason in the
  merged report. Nothing is ever reported accepted without evidence.

``/swarm build <job>`` from chat, ``nm swarm build`` from the CLI, the
``build_swarm`` brain tool, NL routing in the agent loop, and the
``swarm_build`` workflow step all drive this module.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.ids import new_short_id
from ..core.logging_setup import get_logger
from ..core.tool_render import describe_exc, emit_progress
from ..llm.brain import brain_for
from .swarm_board import worker_name_for

_log = get_logger(__name__)

__all__ = [
    "LANGUAGE_RUNGS",
    "BuildTask",
    "BuildLeg",
    "BuildSwarmResult",
    "BuildSwarm",
    "parse_build_swarm_request",
    "render_build_swarm",
    "register",
]

#: Max workers per run — same cap as the investigation swarm: the box
#: and the model budget bind it, not the design.
MAX_WORKERS = 5

#: Language rungs. ``probe`` is the toolchain binary that must be on
#: PATH; ``test_cmd`` runs the worker's tests; ``check_cmd`` is the
#: compile/syntax check the verifier falls back to when the worker wrote
#: no test files; ``main_file`` seeds the worker's entry point.
LANGUAGE_RUNGS: dict[str, dict[str, Any]] = {
    "python": {
        "probe": "python3",
        "test_cmd": ["python3", "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        "check_cmd": ["python3", "-m", "py_compile"],
        "extensions": (".py",),
        "main_file": "main.py",
    },
    "rust": {
        "probe": "cargo",
        "test_cmd": ["cargo", "test", "--quiet"],
        "check_cmd": ["cargo", "check", "--quiet"],
        "extensions": (".rs",),
        "main_file": "src/main.rs",
    },
    "go": {
        "probe": "go",
        "test_cmd": ["go", "test", "./..."],
        "check_cmd": ["go", "vet", "./..."],
        "extensions": (".go",),
        "main_file": "main.go",
    },
}

#: Roles a heuristic decomposition hands out, in order.
_ROLES = ("builder", "tester", "documenter")

#: Dirs the verifier never counts as worker output.
_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", "target",
              "node_modules", ".venv", "venv"}

#: How much of a test run's output the report keeps.
_TEST_TAIL_CHARS = 2000

#: Test-command wall budget (seconds) when the run has more left.
_TEST_TIMEOUT_S = 120.0


def _is_test_file(rel: str, language: str) -> bool:
    """Is ``rel`` (repo-relative) a test file for this rung?"""
    name = rel.replace("\\", "/")
    base = name.rsplit("/", 1)[-1]
    if language == "python":
        return (base.startswith("test_") and base.endswith(".py")) or \
            base.endswith("_test.py") or name.startswith("tests/")
    if language == "rust":
        return base.endswith("_test.rs") or name.startswith("tests/")
    if language == "go":
        return base.endswith("_test.go")
    return False


@dataclass
class BuildTask:
    """One worker's single job: a role, a task, a language rung."""

    role: str
    task: str
    language: str
    name: str = ""


@dataclass
class BuildLeg:
    """One worker's outcome, after verification."""

    task: str
    role: str
    language: str
    name: str
    work_dir: Path
    #: True when the worker itself reported success (files written,
    #: no crash). The verifier may still reject the leg.
    ok: bool = True
    #: Worker-reported files (relative). The verifier re-lists the dir
    #: itself; this is the worker's claim, not the evidence.
    files: list[str] = field(default_factory=list)
    #: Verifier verdict: "accepted" | "rejected" | "unverified".
    verdict: str = "unverified"
    #: Verifier reasons — always non-empty for a rejection.
    reasons: list[str] = field(default_factory=list)
    #: The verifier's test run: return code (None when nothing ran)
    #: and the kept tail of its output.
    test_rc: int | None = None
    test_output: str = ""
    seconds: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.verdict == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "role": self.role,
            "language": self.language,
            "name": self.name,
            "work_dir": str(self.work_dir),
            "ok": self.ok,
            "accepted": self.accepted,
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "files": list(self.files),
            "test_rc": self.test_rc,
            "test_output": self.test_output,
            "seconds": round(self.seconds, 1),
        }


@dataclass
class BuildSwarmResult:
    job: str
    language: str
    tasks: list[BuildTask] = field(default_factory=list)
    legs: list[BuildLeg] = field(default_factory=list)
    seconds: float = 0.0
    run_id: str = field(default_factory=lambda: new_short_id("buildswarm"))

    @property
    def ok(self) -> bool:
        return any(leg.accepted for leg in self.legs)

    @property
    def summary(self) -> str:
        done = sum(1 for leg in self.legs if leg.accepted)
        parts = [f"{done}/{len(self.legs)} workers accepted"]
        for leg in self.legs:
            if not leg.accepted:
                why = "; ".join(leg.reasons[:1]) or "no reason recorded"
                parts.append(f"{leg.name} rejected: {why}")
        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job": self.job,
            "language": self.language,
            "run_id": self.run_id,
            "ok": self.ok,
            "summary": self.summary,
            "seconds": round(self.seconds, 2),
            "tasks": [{"role": t.role, "task": t.task,
                       "language": t.language, "name": t.name}
                      for t in self.tasks],
            "legs": [leg.to_dict() for leg in self.legs],
        }


class BuildSwarm:
    """Plan → parallel build workers → verify each leg → merge.

    ``worker_fn(task, work_dir, deadline, attempt) -> dict`` runs one
    worker; inject it in tests to avoid the model. ``run_tests(cmd, cwd,
    timeout) -> (rc, output)`` runs one test/check command; inject it
    to stub the toolchain. The defaults are real: CodingAgent for the
    worker, ``subprocess`` for the verifier.
    """

    def __init__(
        self,
        context: Any,
        *,
        brain: Any = None,
        gateway: Any = None,
        wall_seconds: float = 600.0,
        work_root: str | Path | None = None,
        worker_fn: Callable[..., dict[str, Any]] | None = None,
        run_tests: Callable[..., tuple[int, str]] | None = None,
    ) -> None:
        self.context = context
        self.brain = brain
        self.gateway = gateway
        self.wall_seconds = float(wall_seconds)
        self._work_root = Path(work_root) if work_root is not None \
            else self._default_work_root(context)
        self._worker_fn = worker_fn or self._default_worker
        self._run_tests = run_tests or self._run_command

    # ── paths ────────────────────────────────────────────────────────────
    @staticmethod
    def _default_work_root(context: Any) -> Path:
        """Per-run work dirs live under the runtime home when known,
        else under the repo checkout — never /tmp (ephemeral) and never
        a shared queue dir."""
        home = getattr(getattr(context, "settings", None), "home", None)
        if home is None:
            home = getattr(context, "home", None)
        if home:
            return Path(home) / "data" / "swarm_builds"
        here = Path(__file__).resolve()
        for parent in here.parents:
            if (parent / ".git").exists() and (parent / "nomorals").is_dir():
                return parent / "swarm_builds"
        return Path.cwd() / "swarm_builds"

    # ── model availability ───────────────────────────────────────────────
    def _model_available(self) -> bool:
        router = getattr(self.context, "router", None)
        snapshot = getattr(router, "stats_snapshot", None)
        if snapshot is None:
            return False
        try:
            snap = snapshot()
        except Exception:  # noqa: BLE001
            return False
        active = str(snap.get("active") or "")
        return bool(active) and active not in {"mock", "offline", "test", ""}

    # ── decomposition ────────────────────────────────────────────────────
    def decompose(self, job: str, workers: int) -> tuple[list[BuildTask], str]:
        """→ (tasks, planner name). Model first, heuristic fallback."""
        workers = max(1, min(int(workers), MAX_WORKERS))
        if self._model_available():
            try:
                from ..llm.base import Message, SamplingParams

                response = brain_for(self.context).chat(
                    [
                        Message.system(
                            f"Split this build job into at most {workers} "
                            "parallel subtasks that independent builders can "
                            "each complete without talking to each other. "
                            "Each subtask gets one role: builder (implement "
                            "it), tester (write tests and verify), or "
                            "documenter (usage docs). Reply with ONLY a "
                            "JSON array of objects with keys role and task."
                        ),
                        Message.user(job),
                    ],
                    SamplingParams(temperature=0.2, max_tokens=300),
                    task_kind="chat")
                text = (response.text or "").strip()
                start, end = text.find("["), text.rfind("]")
                if start != -1 and end > start:
                    raw = json.loads(text[start:end + 1])
                    tasks: list[BuildTask] = []
                    taken: set[str] = set()
                    for i, item in enumerate(raw):
                        if isinstance(item, dict):
                            role = str(item.get("role") or _ROLES[i % 3])
                            task = str(item.get("task") or "").strip()
                        else:
                            role, task = _ROLES[i % 3], str(item).strip()
                        if not task:
                            continue
                        tasks.append(BuildTask(
                            role=role if role in _ROLES else "builder",
                            task=task, language="",
                            name=worker_name_for(task, i, taken)))
                        taken.add(tasks[-1].name)
                    if tasks:
                        return tasks[:workers], "llm"
            except Exception as exc:  # noqa: BLE001 - fall through
                _log.debug("build swarm decomposition failed: %s", exc)
        return self._heuristic_decompose(job, workers), "heuristic"

    def _heuristic_decompose(self, job: str, workers: int) -> list[BuildTask]:
        """Explicit conjunctions first; else numbered items; else the
        role split (builder → tester → documenter) for the one job."""
        parts = [p.strip() for p in re.split(
            r"\s+(?:and|then|,)\s+(?=[a-z])", job, flags=re.IGNORECASE)
            if p.strip()]
        if len(parts) < 2:
            numbered = re.split(r"(?:^|\n)\s*\d+[.)]\s*", job)
            parts = [p.strip() for p in numbered if p.strip()]
            if len(parts) >= 2 and not re.search(
                    r"\d+[.)]", job.split("\n")[0].strip()[:3]):
                parts = parts[1:]
        taken: set[str] = set()
        tasks: list[BuildTask] = []
        if len(parts) >= 2:
            for i, part in enumerate(parts[:workers]):
                name = worker_name_for(part, i, taken)
                taken.add(name)
                tasks.append(BuildTask(role=_ROLES[i % len(_ROLES)],
                                       task=part, language="", name=name))
            return tasks
        for i in range(workers):
            role = _ROLES[i % len(_ROLES)]
            if role == "builder":
                task = f"implement: {job}"
            elif role == "tester":
                task = f"write tests for and verify: {job}"
            else:
                task = f"write usage docs for: {job}"
            name = worker_name_for(task, i, taken)
            taken.add(name)
            tasks.append(BuildTask(role=role, task=task,
                                   language="", name=name))
        return tasks

    # ── the worker ───────────────────────────────────────────────────────
    def _worker_brief(self, task: BuildTask, work_dir: Path,
                      notes: list[str] | None = None) -> str:
        lines = [
            f"You are the {task.role} of a build swarm. Your one job:",
            task.task,
            "",
            f"Language: {task.language}. Work ONLY inside {work_dir}.",
        ]
        if task.language == "rust":
            lines.append("Write the implementation as a Cargo project "
                         "(Cargo.toml + src/main.rs, tests under tests/).")
        elif task.language == "go":
            lines.append("Write the implementation as a go module "
                         "(go.mod + main.go, tests *_test.go).")
        else:
            lines.append("Write the implementation as a Python module "
                         "(a main module + test_*.py tests).")
        lines.append("Write tests for what you build and run them yourself. "
                     "Do not touch anything outside your work dir. "
                     "When done, report the files you created or changed.")
        if notes:
            lines.append("")
            lines.append("The verifier rejected your last attempt — fix "
                         "these, do not repeat them:")
            lines.extend(f"- {n}" for n in notes)
        return "\n".join(lines)

    def _default_worker(self, task: BuildTask, work_dir: Path,
                        deadline: float, attempt: int = 0) -> dict[str, Any]:
        """The real worker: CodingAgent rooted at the worker dir.

        No model → an honest failure dict, never a fake digest."""
        if not self._model_available():
            return {"ok": False, "files": [], "output": "",
                    "error": "no model available — worker cannot build "
                             "without a live model"}
        from .coding import CodingAgent

        remaining = max(30.0, deadline - time.time())
        rung = LANGUAGE_RUNGS[task.language]
        agent = CodingAgent(self.context, root=str(work_dir))
        result = agent.run(
            self._worker_brief(task, work_dir),
            filename=rung["main_file"], timeout=remaining)
        return {"ok": bool(result.ok), "files": list(result.files or []),
                "output": str(result.output or "")[:2000],
                "error": str(result.error or "")}

    def _run_worker(self, task: BuildTask, work_dir: Path, deadline: float,
                    *, attempt: int = 0,
                    notes: list[str] | None = None,
                    on_progress: Callable[[int, int, str, str], None] | None = None,
                    index: int = 0, total: int = 1) -> BuildLeg:
        started = time.time()
        work_dir.mkdir(parents=True, exist_ok=True)
        emit_progress(on_progress, index + 1, total,
                      f"buildswarm {task.name}", "start")
        try:
            outcome = self._worker_fn(task, work_dir, deadline, attempt)
            ok = bool(outcome.get("ok", False))
            files = [str(f) for f in (outcome.get("files") or [])]
            error = str(outcome.get("error") or "")
            leg = BuildLeg(task=task.task, role=task.role,
                           language=task.language, name=task.name,
                           work_dir=work_dir, ok=ok, files=files)
            if not ok:
                leg.verdict = "rejected"
                leg.reasons = [error or "worker reported failure"]
        except Exception as exc:  # noqa: BLE001 - a dead worker is a leg
            _log.exception("build swarm worker failed: %r", task.task)
            leg = BuildLeg(task=task.task, role=task.role,
                           language=task.language, name=task.name,
                           work_dir=work_dir, ok=False, files=[])
            leg.verdict = "rejected"
            leg.reasons = [f"worker crashed: {describe_exc(exc)}"]
        emit_progress(on_progress, index + 1, total,
                      f"buildswarm {task.name}", "done")
        leg.seconds = round(time.time() - started, 1)
        return leg

    # ── the verifier ─────────────────────────────────────────────────────
    def _list_files(self, work_dir: Path) -> list[str]:
        """Files the worker actually left on disk (relative, sorted).
        Build caches and VCS state are never worker output."""
        found: list[str] = []
        if not work_dir.is_dir():
            return found
        for path in sorted(work_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(work_dir)
            if any(part in _SKIP_DIRS for part in rel.parts):
                continue
            found.append(str(rel))
        return found

    @staticmethod
    def _run_command(cmd: list[str], *, cwd: Path,
                     timeout: float) -> tuple[int, str]:
        """Run one command, capturing combined output. Never raises:
        a timeout is rc 124 with the reason attached."""
        try:
            proc = subprocess.run(
                [str(c) for c in cmd], cwd=str(cwd), timeout=timeout,
                capture_output=True, text=True)
            out = (proc.stdout or "") + (proc.stderr or "")
            return proc.returncode, out[-_TEST_TAIL_CHARS:]
        except subprocess.TimeoutExpired as exc:
            out = b""
            if exc.stdout:
                out += exc.stdout if isinstance(exc.stdout, bytes) \
                    else exc.stdout.encode()
            if exc.stderr:
                out += exc.stderr if isinstance(exc.stderr, bytes) \
                    else exc.stderr.encode()
            text = out.decode("utf-8", "replace")[-_TEST_TAIL_CHARS:]
            return 124, f"command timed out after {timeout:.0f}s\n{text}"
        except OSError as exc:
            return 127, f"could not run {' '.join(map(str, cmd))}: {exc}"

    def _verify_leg(self, leg: BuildLeg,
                    *, timeout: float = _TEST_TIMEOUT_S) -> BuildLeg:
        """The VERIFIER agent: toolchain probe → real test run (or the
        rung's compile/syntax check when no test files exist) → diff
        review. ACCEPT or REJECT with concrete reasons. Never raises:
        a verifier that cannot run reports WHY."""
        rung = LANGUAGE_RUNGS[leg.language]
        probe = str(rung["probe"])
        if shutil.which(probe) is None:
            leg.verdict = "rejected"
            leg.reasons = [f"toolchain {probe!r} not found on PATH — "
                           f"install it or pick another rung"]
            return leg
        files = self._list_files(leg.work_dir)
        if not files:
            leg.verdict = "rejected"
            leg.reasons = [f"worker produced no files in {leg.work_dir}"]
            return leg
        leg.files = files
        exts = tuple(rung["extensions"])
        sources = [f for f in files if f.endswith(exts)]
        if not sources:
            leg.verdict = "rejected"
            leg.reasons = [f"no {leg.language} source files to verify "
                           f"(found: {', '.join(files[:5])})"]
            return leg
        test_files = [f for f in files
                      if _is_test_file(f, leg.language)]
        if test_files:
            cmd = list(rung["test_cmd"])
            rc, out = self._run_tests(cmd, cwd=leg.work_dir, timeout=timeout)
            leg.test_rc, leg.test_output = rc, out
            if rc == 0:
                leg.verdict = "accepted"
                leg.reasons = [f"tests passed ({len(test_files)} test "
                               f"file(s), rc 0)"]
            else:
                leg.verdict = "rejected"
                leg.reasons = [f"tests failed (rc {rc}): "
                               + (out.strip().splitlines()[-1]
                                  if out.strip() else "no output")]
            return leg
        # No tests: the rung's compile/syntax check is the evidence.
        if leg.language == "python":
            cmd = list(rung["check_cmd"]) + sources
        elif leg.language == "rust":
            if not (leg.work_dir / "Cargo.toml").exists():
                leg.verdict = "rejected"
                leg.reasons = ["no Cargo.toml — not a Rust project"]
                return leg
            cmd = list(rung["check_cmd"])
        else:  # go
            if not (leg.work_dir / "go.mod").exists():
                leg.verdict = "rejected"
                leg.reasons = ["no go.mod — not a Go module"]
                return leg
            cmd = list(rung["check_cmd"])
        rc, out = self._run_tests(cmd, cwd=leg.work_dir, timeout=timeout)
        leg.test_rc, leg.test_output = rc, out
        if rc == 0:
            leg.verdict = "accepted"
            leg.reasons = ["no tests — compile/syntax check passed"]
        else:
            leg.verdict = "rejected"
            leg.reasons = [f"compile/syntax check failed (rc {rc}): "
                           + (out.strip().splitlines()[-1]
                              if out.strip() else "no output")]
        return leg

    # ── the run ──────────────────────────────────────────────────────────
    def run(self, job: str, workers: int = 3, language: str = "python",
            on_progress: Callable[[int, int, str, str], None] | None = None
            ) -> BuildSwarmResult:
        """Run the build swarm. One retry per rejected leg, then the
        merged report — rejected legs stay rejected with reasons."""
        job = (job or "").strip()
        if not job:
            raise ValueError("build swarm needs a job")
        language = str(language or "python").lower()
        if language not in LANGUAGE_RUNGS:
            raise ValueError(
                f"unknown language rung {language!r} — want one of "
                f"{sorted(LANGUAGE_RUNGS)}")
        workers = max(1, min(int(workers), MAX_WORKERS))
        started = time.time()
        deadline = started + self.wall_seconds

        tasks, _planner = self.decompose(job, workers)
        for task in tasks:
            task.language = language
        result = BuildSwarmResult(job=job, language=language, tasks=tasks)
        run_dir = self._work_root / result.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        pool = ThreadPoolExecutor(max_workers=len(tasks),
                                  thread_name_prefix="buildswarm")
        futures = [pool.submit(
            self._run_worker, task, run_dir / task.name, deadline,
            attempt=0, on_progress=on_progress, index=i,
            total=len(tasks))
            for i, task in enumerate(tasks)]
        legs: list[BuildLeg] = []
        for i, future in enumerate(futures):
            try:
                leg = future.result(
                    timeout=max(10.0, deadline - time.time()))
            except Exception as exc:  # noqa: BLE001 - a late leg is a leg
                leg = BuildLeg(task=tasks[i].task, role=tasks[i].role,
                               language=language, name=tasks[i].name,
                               work_dir=run_dir / tasks[i].name,
                               ok=False, files=[])
                leg.verdict = "rejected"
                leg.reasons = [f"worker timed out: {describe_exc(exc)}"]
            legs.append(leg)
        pool.shutdown(wait=False, cancel_futures=True)

        # ── verifier phase: verify, one retry per rejection, re-verify ──
        for i, leg in enumerate(legs):
            if leg.ok:
                self._verify_leg(leg, timeout=min(
                    _TEST_TIMEOUT_S, max(10.0, deadline - time.time())))
            if leg.accepted:
                continue
            # exactly one retry, with the verifier's reasons attached
            retry = self._run_worker(
                tasks[i], run_dir / tasks[i].name, deadline, attempt=1,
                notes=list(leg.reasons), on_progress=on_progress,
                index=i, total=len(tasks))
            if retry.ok:
                self._verify_leg(retry, timeout=min(
                    _TEST_TIMEOUT_S, max(10.0, deadline - time.time())))
            legs[i] = retry
            _log.info("buildswarm %s leg %s: %s (%s)", result.run_id,
                      retry.name, retry.verdict,
                      "; ".join(retry.reasons[:1]))

        result.legs = legs
        result.seconds = time.time() - started
        _log.info("buildswarm %s done: %d/%d legs accepted, %.0fs",
                  result.run_id,
                  sum(1 for leg in legs if leg.accepted), len(legs),
                  result.seconds)
        return result


# ── the NL request parser (agent loop) ────────────────────────────────────

_RE_SWARM_BUILD = re.compile(
    r"^\s*(?:(?:please|run|start)\s+(?:a\s+)?)?"
    r"(?:swarm\s+build|build\s+swarm|build-swarm)\b[ :]*?(.*)$",
    re.IGNORECASE)
_RE_LANG = re.compile(r"\b(?:in|using)\s+(python|rust|go)\b", re.IGNORECASE)
_RE_WORKERS = re.compile(r"\b(?:with\s+)?(\d+)\s+workers?\b", re.IGNORECASE)
_RE_TRAILING_NUM = re.compile(r"\s+(\d+)\s*$")
_RE_LEADING_FOR = re.compile(r"^for\s+", re.IGNORECASE)


def parse_build_swarm_request(text: str) -> dict[str, Any] | None:
    """Parse an owner-DM request for a build swarm.

    ``swarm build <job> [in rust|go|python] [with N workers|N]`` or
    ``build swarm <job>``. Returns ``{job, workers, language}`` or None
    when the text is not a build-swarm request."""
    match = _RE_SWARM_BUILD.match(text or "")
    if match is None:
        return None
    rest = (match.group(1) or "").strip()
    if not rest:
        return None
    language = "python"
    lang = _RE_LANG.search(rest)
    if lang is not None:
        language = lang.group(1).lower()
        rest = (_RE_LANG.sub("", rest)).strip()
    workers = 3
    count = _RE_WORKERS.search(rest)
    if count is not None:
        workers = max(1, min(int(count.group(1)), MAX_WORKERS))
        rest = _RE_WORKERS.sub("", rest).strip()
    else:
        trailing = _RE_TRAILING_NUM.search(rest)
        if trailing is not None:
            workers = max(1, min(int(trailing.group(1)), MAX_WORKERS))
            rest = _RE_TRAILING_NUM.sub("", rest).strip()
    job = " ".join(rest.split())
    job = _RE_LEADING_FOR.sub("", job).strip()
    if not job:
        return None
    return {"job": job, "workers": workers, "language": language}


# ── presentation ──────────────────────────────────────────────────────────

def render_build_swarm(result: BuildSwarmResult) -> str:
    """Render a build-swarm run as a mission report. Never raises."""
    from .render import ICONS, banner, bar, kv, section, table, truncate

    try:
        legs = result.legs or []
        done = sum(1 for leg in legs if leg.accepted)
        head = {
            "job": truncate(result.job, 80),
            "legs": f"{done}/{len(legs)} accepted",
            "rung": result.language,
            "time": f"{result.seconds:.1f}s",
            "run": result.run_id,
        }
        lines = [banner("Build swarm", ICONS["thinking"]),
                 kv(head.items()), "",
                 bar(done / len(legs) if legs else 0.0)]
        if legs:
            rows = []
            for i, leg in enumerate(legs, 1):
                icon = ICONS["ok"] if leg.accepted else ICONS["fail"]
                why = "; ".join(leg.reasons[:1])
                rows.append([f"{icon} {i}",
                             truncate(f"{leg.role} · {leg.task}", 36),
                             leg.verdict,
                             truncate(why, 60),
                             f"{leg.seconds:.0f}s"])
            lines.append("")
            lines.append(table(["#", "worker", "verdict", "why", "time"],
                               rows))
            for leg in legs:
                if leg.accepted:
                    continue
                lines.append("")
                lines.append(section(
                    f"rejected: {leg.name}",
                    "\n".join(f"- {r}" for r in leg.reasons) or "(no reason)",
                    ICONS["fail"]))
        lines.append("")
        lines.append(section("Merged report",
                             truncate(result.summary, 1200), ICONS["star"]))
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — rendering never breaks callers
        return f"buildswarm {getattr(result, 'run_id', '?')} (render failed)"


# ── tool registration ─────────────────────────────────────────────────────

def register(registry: Any) -> None:
    """Register the ``build_swarm`` brain tool (mirrors research_swarm)."""
    from ..core.policy import Capability

    context = registry.context

    @registry.register(
        "build_swarm",
        description=(
            "multi-worker build swarm: split a build job into single-job "
            "workers (each with a role + a language rung: python, rust, "
            "go), build in parallel, then a verifier agent runs every "
            "worker's tests, reviews the diff, and accepts or rejects "
            "with reasons. Merged report, one retry per rejected worker. "
            "Use when a build is bigger than one worker's job."
        ),
        capability=Capability.EXEC_CODE,
        parameters={
            "goal": "str — the build job",
            "workers": "int (optional, 1-5) — parallel builders",
            "language": "str (optional) — python | rust | go",
        },
    )
    def build_swarm(
        goal: str,
        workers: int = 3,
        language: str = "python",
        **_: Any,
    ) -> dict[str, Any]:
        swarm = BuildSwarm(context)
        result = swarm.run(goal, workers=workers or 3,
                           language=language or "python")
        return result.to_dict()
