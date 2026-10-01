"""Agent benchmark — measures the SYSTEM, not just the model.

Four dimensions, each a fixed task set with predicates, all hermetic
(no network; code tasks run in the local sandbox):

  reasoning        the reasoning engine on fixed tasks (reasoning_eval)
  planning         the orchestrator's planner on real goals — steps must
                   exist, cover the required sub-problems, and be ordered
  tool_use         planning tool-call sequences against a hermetic fake
                   tool registry — valid names, valid args, right order,
                   no hallucinated tools
  self_correction  given BROKEN code and its REAL error output (the
                   benchmark runs the broken code to capture the actual
                   traceback), the system must produce a fix that runs

Why the system, not the model: an evolution changes prompts, tool
catalogs, and agent logic — the score moves when the SYSTEM gets better
or worse. The evolution promotion gate consumes this as a regression
check: it records a baseline on first measurable run and blocks
promotions that regress it (with tolerance).

Unmeasurable = honest: with the mock provider, offline mode, or no
router, the report says ``measurable: False`` and the gate skips — a
benchmark that runs on a stand-in model must never block real work.
"""
from __future__ import annotations

import json
import re
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..llm.base import Message, SamplingParams
from .reasoning import _extract_json, reasoning_eval

_log = get_logger(__name__)

__all__ = ["BenchmarkReport", "DimensionScore", "run_benchmark", "register"]

MAX_TASKS_PER_DIMENSION = 3
_TOLERANCE = 0.05


@dataclass
class DimensionScore:
    name: str
    score: Optional[float]      # None = unmeasurable
    passed: int = 0
    total: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": None if self.score is None else round(self.score, 3),
            "passed": self.passed, "total": self.total,
            "details": self.details,
        }


@dataclass
class BenchmarkReport:
    scores: dict[str, DimensionScore]
    overall: Optional[float]
    measurable: bool
    provider: str
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": None if self.overall is None else round(self.overall, 3),
            "measurable": self.measurable,
            "provider": self.provider,
            "seconds": round(self.seconds, 1),
            "dimensions": {k: v.as_dict() for k, v in self.scores.items()},
        }


# ── measurability ────────────────────────────────────────────────────────────


def _provider_name(context: Any) -> str:
    try:
        return str(context.settings.llm.provider or "").strip().lower()
    except Exception:  # noqa: BLE001
        return ""


def measurable(context: Any) -> bool:
    """Can we actually measure anything with the active provider?"""
    if getattr(context, "router", None) is None:
        return False
    settings = getattr(context, "settings", None)
    if settings is not None and getattr(settings, "offline", False):
        return False
    return _provider_name(context) not in {"", "mock", "none"}


def _llm(context: Any, system: str, user: str,
         *, temperature: float = 0.1) -> Optional[str]:
    response = context.router.chat(
        [Message.system(system), Message.user(user)],
        SamplingParams(temperature=temperature, max_tokens=1200),
    )
    if not getattr(response, "ok", False):
        return None
    text = (getattr(response, "text", "") or "").strip()
    return text or None


# ── dimension: reasoning ─────────────────────────────────────────────────────


def _dim_reasoning(context: Any, limit: int) -> DimensionScore:
    limit = limit or MAX_TASKS_PER_DIMENSION
    report = reasoning_eval(context, limit=limit)
    details = [{"task": t["goal"], "pass": t["pass"],
                "detail": t["detail"]} for t in report["tasks"]]
    return DimensionScore("reasoning", report["score"],
                          report["passed"], report["total"], details)


# ── dimension: planning ──────────────────────────────────────────────────────


def _transitively_depends_on(plan: Any, a_name: str, b_name: str) -> bool:
    """True if step a depends on step b (directly or through the graph).

    A step never counts as depending on itself.
    """
    if a_name == b_name:
        return False
    deps = {s.name: set(s.depends_on) for s in plan.steps}
    seen: set[str] = set()
    stack = [a_name]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(deps.get(n, ()))
    return b_name in seen


def _judge_plan(
    plan: Any,
    *,
    min_steps: int = 1,
    must_mention: list[str] = (),
    coverage: list[tuple[str, ...]] = (),
    order: tuple[str, str] = (),   # (pattern_of_later, pattern_of_earlier)
) -> tuple[bool, str]:
    """Pure judgment of a plan against a task spec. (bool, detail)"""
    problems: list[str] = []
    if len(plan.steps) < min_steps:
        problems.append(f"only {len(plan.steps)} steps (need >= {min_steps})")
    goals_text = " ".join(s.goal.lower() for s in plan.steps)
    for token in must_mention:
        if not re.search(token, goals_text):
            problems.append(f"no step mentions /{token}/")
    for alternatives in coverage:
        if not any(re.search(a, goals_text) for a in alternatives):
            problems.append("covers none of " + " | ".join(alternatives))
    if order:
        later, earlier = order
        later_steps = [s for s in plan.steps
                       if later in (s.role, s.name) or re.search(later, s.goal)]
        earlier_steps = [s for s in plan.steps
                         if earlier in (s.role, s.name)
                         or re.search(earlier, s.goal)]
        if not later_steps or not earlier_steps:
            problems.append(f"missing a step matching {later!r} or {earlier!r}")
        else:
            ok = any(
                _transitively_depends_on(plan, l.name, e.name)
                for l in later_steps for e in earlier_steps)
            if not ok:
                problems.append(f"{later!r} does not depend on {earlier!r}")
    if problems:
        return False, "; ".join(problems)[:200]
    return True, f"{len(plan.steps)} steps: " + \
        " → ".join(s.name for s in plan.steps)[:160]


def _planning_check(
    context: Any,
    goal: str,
    *,
    min_steps: int = 1,
    must_mention: list[str] = (),
    coverage: list[tuple[str, ...]] = (),
    order: tuple[str, str] = (),
) -> tuple[bool, str]:
    """Run the orchestrator planner on one goal and judge the plan."""
    from .orchestrator import MasterOrchestrator

    orch = MasterOrchestrator(context)
    plan = orch.plan(goal)
    return _judge_plan(plan, min_steps=min_steps, must_mention=must_mention,
                       coverage=coverage, order=order)


_PLANNING_TASKS: list[dict[str, Any]] = [
    {
        "goal": ("Write a script that backs up the data/ directory to a "
                 "remote server, then verify the backup is restorable."),
        "min_steps": 2,
        "must_mention": [r"backup|remote|data/"],
        "order": (r"verif|restor", r"backup|copy|transfer"),
    },
    {
        "goal": ("Investigate why the API is slow: check the database "
                 "queries, the network path, and the server load — then fix "
                 "the bottleneck you find."),
        "min_steps": 3,
        "coverage": [(r"database|quer",), (r"networ",), (r"load|server|cpu")],
    },
    {
        "goal": ("Ship a new CLI command: implement it, add tests that "
                 "cover it, and document it in the help output."),
        "min_steps": 3,
        "must_mention": [r"test", r"doc|help|readme"],
        "order": (r"test", r"implement|writ|code"),
    },
]


def _dim_planning(context: Any, limit: int) -> DimensionScore:
    tasks = _PLANNING_TASKS[: (limit or MAX_TASKS_PER_DIMENSION)]
    passed = 0
    details: list[dict[str, Any]] = []
    for task in tasks:
        try:
            ok, detail = _planning_check(context, task["goal"],
                                          min_steps=task.get("min_steps", 1),
                                          must_mention=task.get("must_mention", []),
                                          coverage=task.get("coverage", []),
                                          order=task.get("order", ()))
        except Exception as exc:  # noqa: BLE001 — one bad task, on
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        passed += bool(ok)
        details.append({"task": task["goal"][:100], "pass": ok,
                        "detail": detail})
    score = passed / len(tasks) if tasks else 0.0
    return DimensionScore("planning", score, passed, len(tasks), details)


# ── dimension: tool use ──────────────────────────────────────────────────────


# a hermetic, deterministic tool registry the model must plan against
_FAKE_TOOLS: dict[str, dict[str, bool]] = {
    "read_file": {"path": True},
    "grep": {"pattern": True, "path": False},
    "run_tests": {"pattern": False},
    "http_get": {"url": True},
    "db_query": {"sql": True},
    "notify": {"message": True},
}


def _tool_use_check(context: Any, task: dict[str, Any]) -> tuple[bool, str]:
    """Ask the model (in the devon planner's own prompt shape) for a
    tool-call sequence; validate it against the fake registry."""
    def _fmt_args(args: dict[str, bool]) -> str:
        return ", ".join(
            a + (" (required)" if req else "") for a, req in args.items())

    catalog = "\n".join(
        f"- {name}: {_fmt_args(args)}" for name, args in _FAKE_TOOLS.items())
    system = (
        "You are an autonomous agent. Plan a short sequence of tool calls "
        "to complete the task, in the order you'd run them. Pick only tools "
        "that exist in the list, with the arguments they need. Reply with "
        'ONLY JSON of the form {"steps":[{"tool":"<name>","args":{...},'
        '"why":"<short>"}]}. No prose outside the JSON.')
    user = f"Available tools:\n{catalog}\n\nTask: {task['goal']}"
    text = _llm(context, system, user)
    data = _extract_json(text) if text else None
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list) or not steps:
        return False, "no usable plan returned"
    problems: list[str] = []
    seen_order: list[str] = []
    for item in steps:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        seen_order.append(tool)
        if tool not in _FAKE_TOOLS:
            problems.append(f"unknown tool {tool!r}")
            continue
        args = item.get("args")
        args = args if isinstance(args, dict) else {}
        for arg_name, required in _FAKE_TOOLS[tool].items():
            value = args.get(arg_name, "")
            if required and not str(value).strip():
                problems.append(f"{tool} missing required arg {arg_name!r}")
    required_tools = task.get("requires", [])
    for tool in required_tools:
        if tool not in seen_order:
            problems.append(f"never plans {tool!r}")
    for later, earlier in task.get("order", []):
        if later in seen_order and earlier in seen_order:
            if seen_order.index(later) < seen_order.index(earlier):
                problems.append(f"{later!r} before {earlier!r}")
        elif later not in seen_order or earlier not in seen_order:
            problems.append(f"cannot check {later!r}→{earlier!r}")
    if problems:
        return False, "; ".join(dict.fromkeys(problems))[:200]
    return True, " → ".join(seen_order)[:160]


_TOOL_USE_TASKS: list[dict[str, Any]] = [
    {
        "goal": ("Find where the login endpoint is defined, read the file, "
                 "and run the tests that cover it."),
        "requires": ["grep", "read_file", "run_tests"],
        "order": [("run_tests", "grep"), ("read_file", "grep")],
    },
    {
        "goal": ("Check that https://api.example.com/health responds, query "
                 "the local database for the latest error row, and notify "
                 "the owner summarizing both results."),
        "requires": ["http_get", "db_query", "notify"],
        "order": [("notify", "http_get"), ("notify", "db_query")],
    },
    {
        "goal": ("Deploy the application to the production Kubernetes "
                 "cluster and open a pull request for the changes."),
        # no deploy/k8s/PR tools exist — a competent plan must not invent
        # them; every planned tool must exist in the registry
        "requires": [],
        "order": [],
    },
]


def _dim_tool_use(context: Any, limit: int) -> DimensionScore:
    tasks = _TOOL_USE_TASKS[: (limit or MAX_TASKS_PER_DIMENSION)]
    passed = 0
    details: list[dict[str, Any]] = []
    for task in tasks:
        try:
            ok, detail = _tool_use_check(context, task)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        passed += bool(ok)
        details.append({"task": task["goal"][:100], "pass": ok,
                        "detail": detail})
    score = passed / len(tasks) if tasks else 0.0
    return DimensionScore("tool_use", score, passed, len(tasks), details)


# ── dimension: self-correction ──────────────────────────────────────────────


def _run_python(code: str, timeout: float = 15.0) -> dict[str, Any]:
    """Run python in the same sandbox the coding agent uses. Hermetic:
    a plain script with no network access needs from it."""
    from ..tools.shell import SandboxLimits, run_sandboxed

    tmp = Path(tempfile.mkdtemp(prefix="nm-bench-"))
    target = tmp / "script.py"
    target.write_text(code, encoding="utf-8")
    import sys as _sys

    env = {"PATH": f"{str(Path(_sys.executable).parent)}:/usr/local/bin:"
                   f"/usr/bin:/bin"}
    try:
        return run_sandboxed(
            f"{_sys.executable} \"{target}\"",
            cwd=str(tmp), timeout=timeout, env=env,
            limits=SandboxLimits(cpu_seconds=int(min(max(timeout, 5.0), 60.0))),
        )
    finally:
        try:
            target.unlink(missing_ok=True)
            tmp.rmdir()
        except OSError:  # noqa: E103 - temp cleanup is best-effort
            pass


def _self_correction_check(context: Any, task: dict[str, Any]
                           ) -> tuple[bool, str]:
    """Run the BROKEN code to capture its real error, ask the system to
    fix it, and run the fix. Pass = the fix runs and prints the expected
    output."""
    from .coding import extract_code_block

    broken = task["code"]
    crashed = _run_python(broken)
    if crashed["exit_code"] == 0:
        # the fixture is wrong, not the system — don't score it
        return True, "fixture did not crash (skipped as pass)"
    error = (crashed["stderr"] or crashed["stdout"] or "non-zero exit")[-1500:]
    system = (
        "You are a coding agent. Write complete, runnable Python for the "
        "file 'script.py'. Respond with EXACTLY ONE fenced ```python code "
        "block containing the whole file and nothing else — no prose "
        "outside the block.")
    user = (
        f"Task: {task['task']}\n\nAttempt 1.\n\n"
        f"Previous code:\n```\n{broken}\n```\n\n"
        f"It was run and FAILED with this exact output:\n```\n{error}\n```\n"
        "Fix the code so the run succeeds.")
    text = _llm(context, system, user)
    fixed = extract_code_block(text) if text else ""
    if not fixed.strip():
        return False, "no code block in the fix"
    ran = _run_python(fixed)
    if ran["exit_code"] != 0 or ran.get("timed_out"):
        return False, "fixed code still fails: " + \
            (ran["stderr"] or ran["stdout"] or "")[:120]
    if task.get("expect") and task["expect"] not in (ran["stdout"] or ""):
        return False, f"output missing {task['expect']!r}: " + \
            (ran["stdout"] or "")[:120]
    return True, "fixed + ran: " + (ran["stdout"] or "").strip()[:80]


_SELF_CORRECTION_TASKS: list[dict[str, Any]] = [
    {
        "task": "print the name of the last person in the list",
        "code": (
            "def last_name(people):\n"
            "    return people[len(people)][\"name\"]\n\n"
            "print(last_name([{\"name\": \"ada\"}]))\n"),
        "expect": "ada",
    },
    {
        "task": "print the average of the values; an empty list averages to 0",
        "code": (
            "def average(values):\n"
            "    return sum(values) / len(values)\n\n"
            "print(average([]))\n"),
        "expect": "0",
    },
    {
        "task": "print the top k scores, highest first",
        "code": (
            "def top_scores(scores, k=3):\n"
            "    return sorted(scores, reverse=True)[:k]\n\n"
            "print(top_scores([90, 75, 88], top_k))\n"),
        "expect": "[90, 88, 75]",
    },
]


def _dim_self_correction(context: Any, limit: int) -> DimensionScore:
    tasks = _SELF_CORRECTION_TASKS[: (limit or MAX_TASKS_PER_DIMENSION)]
    passed = 0
    details: list[dict[str, Any]] = []
    for task in tasks:
        try:
            ok, detail = _self_correction_check(context, task)
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        passed += bool(ok)
        details.append({"task": task["task"][:100], "pass": ok,
                        "detail": detail})
    score = passed / len(tasks) if tasks else 0.0
    return DimensionScore("self_correction", score, passed, len(tasks),
                          details)


# ── runner ───────────────────────────────────────────────────────────────────

_DIMENSIONS: dict[str, Callable[[Any, int], DimensionScore]] = {
    "reasoning": _dim_reasoning,
    "planning": _dim_planning,
    "tool_use": _dim_tool_use,
    "self_correction": _dim_self_correction,
}


def run_benchmark(context: Any, *,
                  dimensions: list[str] | None = None,
                  limit: int = 0) -> BenchmarkReport:
    """Run the agent benchmark against the live system.

    ``dimensions``: subset to run (default: all four). ``limit``: cap
    tasks per dimension (default: MAX_TASKS_PER_DIMENSION).
    """
    started = time.monotonic()
    provider = _provider_name(context)
    is_measurable = measurable(context)
    names = [d for d in (dimensions or list(_DIMENSIONS))
             if d in _DIMENSIONS]
    scores: dict[str, DimensionScore] = {}
    for name in names:
        if not is_measurable:
            scores[name] = DimensionScore(name, None)
            continue
        scores[name] = _DIMENSIONS[name](context, limit)
    measured = [s.score for s in scores.values() if s.score is not None]
    overall = sum(measured) / len(measured) if measured else None
    return BenchmarkReport(
        scores=scores,
        overall=overall,
        measurable=is_measurable,
        provider=provider,
        seconds=time.monotonic() - started,
    )


# ── registry ─────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "benchmark",
        description=(
            "Run the agent benchmark: reasoning + planning + tool use + "
            "self-correction dimensions, scored 0-1, hermetic. This is what "
            "the evolution promotion gate uses to block intelligence "
            "regressions."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "dimensions": "str (optional) — comma list from reasoning,planning,tool_use,self_correction",
            "limit": "int (optional) — max tasks per dimension",
        },
    )
    def benchmark(dimensions: str = "", limit: str = "") -> dict[str, Any]:
        dims = [d.strip() for d in dimensions.split(",") if d.strip()] or None
        try:
            n = int(limit or 0)
        except ValueError:
            n = 0
        report = run_benchmark(context, dimensions=dims, limit=n)
        return report.as_dict()
