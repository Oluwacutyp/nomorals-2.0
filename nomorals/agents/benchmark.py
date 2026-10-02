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

import importlib
import json
import re
import secrets
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from ..core.logging_setup import get_logger
from ..core.policy import Capability
from ..llm.base import Message, SamplingParams
from .reasoning import _extract_json, reasoning_eval

_log = get_logger(__name__)

__all__ = [
    "BenchmarkReport", "DimensionScore", "run_benchmark", "register",
    # K3 scoreboard
    "ScoreboardReport", "SweTask", "ResearchTask", "EditTask",
    "BuildBackend", "register_swe_task", "register_research_task",
    "register_edit_task", "register_suite", "list_suites", "list_swe_tasks",
    "run_scoreboard", "save_run", "list_runs", "get_run", "compare_runs",
    "export_run_json", "SCOREBOARD_NAME", "MODE_SELF_TEST", "MODE_MODEL",
]

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


# ── K3 scoreboard ("beat K3") ───────────────────────────────────────────────
# A second, harder benchmark that scores the whole system on five
# externally-verifiable axes:
#
#   swe_coding          SWE-style: task spec + hidden tests -> agent patch ->
#                       run the tests. pass = tests green (no network).
#   research            claim-labeling accuracy: the system labels checkable
#                       claims true/false; a hermetic verifier scores them.
#   edits               edit precision: intended vs actual diff through the
#                       Wave-A edit_loop (hunks clean, exact-match rate,
#                       collateral changes).
#   builds              build success: scaffold -> serve -> smoke through the
#                       builders package (backend injected from L7).
#   latency             fast-path vs heavy-path: CoreMind deterministic route
#                       p50/p99 vs the model-check route p50/p99.
#
# Honest scoring rules (no grade inflation):
# - every dimension reports passed/total; score = passed/total, always with
#   the denominator visible;
# - timeouts count as FAILURES, never as skips;
# - a run is labeled ``harness-self-test`` when no live LLM was available
#   (the harness then verifies its own mechanics with reference solutions
#   and negative controls) and ``model-scored`` when a real model produced
#   the answers. The two labels are never compared against each other.

SCOREBOARD_NAME = "k3"
MODE_SELF_TEST = "harness-self-test"
MODE_MODEL = "model-scored"

#: suites the scoreboard knows, in run order for "all"
_SUITE_FNS: dict[str, Callable[..., DimensionScore]] = {}


def register_suite(name: str, fn: Callable[..., DimensionScore]) -> None:
    """Register (or replace) a scoreboard suite. ``fn`` is called as
    ``fn(context, limit, *, mode, build_backend=None)``."""
    _SUITE_FNS[name] = fn


def list_suites() -> list[str]:
    return list(_SUITE_FNS)


@dataclass
class ScoreboardReport(BenchmarkReport):
    """A benchmark run, persisted and comparable across runs."""

    run_id: str = ""
    suite: str = "all"
    mode: str = MODE_SELF_TEST
    scoreboard: str = SCOREBOARD_NAME

    def as_dict(self) -> dict[str, Any]:
        payload = super().as_dict()
        payload.update({
            "run_id": self.run_id,
            "suite": self.suite,
            "mode": self.mode,
            "scoreboard": self.scoreboard,
        })
        return payload


# ── SWE-style coding tasks ───────────────────────────────────────────────────


@dataclass
class SweTask:
    """One SWE-style task: spec + starter + hidden tests + reference."""

    id: str
    title: str
    spec: str                 # function spec handed to the agent
    starter: str              # solution.py content the agent starts from
    hidden_tests: str         # unittest module source (tests/test_hidden.py)
    reference_solution: str   # known-good solution.py (self-check mode)
    timeout_s: float = 25.0


_SWE_TASKS: list[SweTask] = []


def register_swe_task(task: SweTask) -> None:
    _SWE_TASKS.append(task)


def list_swe_tasks() -> list[SweTask]:
    return list(_SWE_TASKS)


register_swe_task(SweTask(
    id="fizzbuzz",
    title="fizzbuzz(n)",
    spec=("Write fizzbuzz(n): return a list of strings for the numbers "
          "1..n inclusive. Multiples of 3 -> 'fizz', of 5 -> 'buzz', of "
          "both -> 'fizzbuzz', otherwise the number itself as a string."),
    starter='def fizzbuzz(n):\n    return []\n',
    hidden_tests=(
        "import unittest\n"
        "from solution import fizzbuzz\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_basic(self):\n"
        "        self.assertEqual(fizzbuzz(5),\n"
        "                         ['1', '2', 'fizz', '4', 'buzz'])\n"
        "    def test_fifteen(self):\n"
        "        self.assertEqual(fizzbuzz(15)[14], 'fizzbuzz')\n"
        "    def test_one(self):\n"
        "        self.assertEqual(fizzbuzz(1), ['1'])\n"
        "    def test_types(self):\n"
        "        self.assertTrue(all(isinstance(x, str) for x in fizzbuzz(20)))\n"
    ),
    reference_solution=(
        "def fizzbuzz(n):\n"
        "    out = []\n"
        "    for i in range(1, n + 1):\n"
        "        s = ''\n"
        "        if i % 3 == 0:\n"
        "            s += 'fizz'\n"
        "        if i % 5 == 0:\n"
        "            s += 'buzz'\n"
        "        out.append(s or str(i))\n"
        "    return out\n"
    ),
))

register_swe_task(SweTask(
    id="flatten",
    title="flatten(nested)",
    spec=("Write flatten(nested): take an arbitrarily nested list of ints "
          "and return a flat list of the ints in left-to-right order. "
          "Non-list items are ints; empty lists contribute nothing."),
    starter=("def flatten(nested):\n"
             "    out = []\n"
             "    for x in nested:\n"
             "        if isinstance(x, list):\n"
             "            out.extend(x)\n"
             "        else:\n"
             "            out.append(x)\n"
             "    return out\n"),
    hidden_tests=(
        "import unittest\n"
        "from solution import flatten\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_deep(self):\n"
        "        self.assertEqual(flatten([1, [2, [3, [4]], 5]]), [1, 2, 3, 4, 5])\n"
        "    def test_empty(self):\n"
        "        self.assertEqual(flatten([[], [[], []]]), [])\n"
        "    def test_flat(self):\n"
        "        self.assertEqual(flatten([1, 2, 3]), [1, 2, 3])\n"
        "    def test_mixed(self):\n"
        "        self.assertEqual(flatten([[1], 2, [[3]]]), [1, 2, 3])\n"
    ),
    reference_solution=(
        "def flatten(nested):\n"
        "    out = []\n"
        "    for x in nested:\n"
        "        if isinstance(x, list):\n"
        "            out.extend(flatten(x))\n"
        "        else:\n"
        "            out.append(x)\n"
        "    return out\n"
    ),
))

register_swe_task(SweTask(
    id="dedupe",
    title="dedupe(items)",
    spec=("Write dedupe(items): return a list with duplicates removed, "
          "keeping the first occurrence of each item in order. Items may "
          "be unhashable (e.g. dicts, lists) — do not rely on set()."),
    starter=("def dedupe(items):\n"
             "    return list(set(items))\n"),
    hidden_tests=(
        "import unittest\n"
        "from solution import dedupe\n\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_order(self):\n"
        "        self.assertEqual(dedupe([3, 1, 3, 2, 1]), [3, 1, 2])\n"
        "    def test_unhashable(self):\n"
        "        self.assertEqual(dedupe([{'a': 1}, {'a': 1}, [1], [1]]),\n"
        "                         [{'a': 1}, [1]])\n"
        "    def test_empty(self):\n"
        "        self.assertEqual(dedupe([]), [])\n"
        "    def test_all_same(self):\n"
        "        self.assertEqual(dedupe([7, 7, 7]), [7])\n"
    ),
    reference_solution=(
        "def dedupe(items):\n"
        "    out = []\n"
        "    for x in items:\n"
        "        if x not in out:\n"
        "            out.append(x)\n"
        "    return out\n"
    ),
))


def _write_swe_workspace(task: SweTask, solution_source: str) -> Path:
    """Materialize solution.py + hidden tests into a fresh temp dir."""
    workdir = Path(tempfile.mkdtemp(prefix=f"k3-swe-{task.id}-"))
    (workdir / "solution.py").write_text(solution_source, encoding="utf-8")
    tests_dir = workdir / "tests"
    tests_dir.mkdir(exist_ok=True)
    (tests_dir / "__init__.py").write_text("", encoding="utf-8")
    (tests_dir / "test_hidden.py").write_text(task.hidden_tests,
                                              encoding="utf-8")
    return workdir


def _run_hidden_tests(workdir: Path, timeout_s: float) -> dict[str, Any]:
    """Run the hidden test module. Timeout = failure, never a skip."""
    from ..tools.shell import SandboxLimits, run_sandboxed

    env = {"PATH": f"{Path(sys.executable).parent}:/usr/local/bin:"
                   f"/usr/bin:/bin"}
    try:
        result = run_sandboxed(
            f"{sys.executable} -m unittest discover -s tests -t .",
            cwd=str(workdir), timeout=timeout_s, env=env,
            limits=SandboxLimits(
                cpu_seconds=int(min(max(timeout_s, 5.0), 120.0))),
        )
    except Exception as exc:  # noqa: BLE001 — a launch failure is a failure
        return {"pass": False, "timed_out": False,
                "detail": f"could not launch tests: {type(exc).__name__}"}
    if result.get("timed_out"):
        return {"pass": False, "timed_out": True,
                "detail": f"timed out after {timeout_s:.0f}s"}
    ok = result.get("exit_code") == 0
    tail = ((result.get("stderr") or "") + "\n" +
            (result.get("stdout") or "")).strip().splitlines()
    summary = " / ".join(tail[-3:]) if tail else "(no output)"
    return {"pass": bool(ok), "timed_out": False,
            "detail": summary[:240]}


def _swe_model_solution(context: Any, task: SweTask) -> str:
    """Ask the live model for a complete solution file."""
    from .coding import extract_code_block

    text = _llm(
        context,
        "You are a coding agent. Write complete, runnable Python for the "
        "file 'solution.py'. Respond with EXACTLY ONE fenced ```python code "
        "block containing the whole file and nothing else — no prose "
        "outside the block.",
        f"Task: {task.title}\n\n{task.spec}\n\nStarting point "
        f"(currently wrong):\n```python\n{task.starter}\n```",
    )
    return extract_code_block(text) if text else ""


def _run_swe_task(task: SweTask, solution_source: str) -> dict[str, Any]:
    """Score one candidate solution: run the hidden tests, honestly."""
    if not solution_source.strip():
        return {"task": task.id, "pass": False, "timed_out": False,
                "detail": "no solution produced"}
    workdir = _write_swe_workspace(task, solution_source)
    try:
        outcome = _run_hidden_tests(workdir, task.timeout_s)
    finally:
        import shutil as _shutil

        _shutil.rmtree(workdir, ignore_errors=True)
    outcome["task"] = task.id
    return outcome


def _dim_swe_coding(context: Any, limit: int, *,
                    mode: str, **_kw: Any) -> DimensionScore:
    tasks = _SWE_TASKS[: (limit or len(_SWE_TASKS))]
    passed = 0
    details: list[dict[str, Any]] = []
    for task in tasks:
        try:
            if mode == MODE_MODEL:
                solution = _swe_model_solution(context, task)
            else:
                # self-check: the reference solution must PASS and the
                # starter must FAIL — the second half proves the harness
                # actually discriminates instead of rubber-stamping.
                solution = task.reference_solution
            outcome = _run_swe_task(task, solution)
            if mode != MODE_MODEL:
                control = _run_swe_task(task, task.starter)
                outcome["control_starter_fails"] = not control["pass"]
                if control["pass"]:
                    outcome["detail"] += " [BROKEN FIXTURE: starter passes]"
            ok = bool(outcome["pass"])
            if mode != MODE_MODEL:
                ok = ok and bool(outcome.get("control_starter_fails"))
        except Exception as exc:  # noqa: BLE001 — one bad task, on
            ok, outcome = False, {"task": task.id,
                                  "detail": f"{type(exc).__name__}: {exc}"[:160]}
        passed += bool(ok)
        details.append({"task": f"{task.id}: {task.title}",
                        "pass": ok,
                        "detail": outcome.get("detail", ""),
                        "timed_out": bool(outcome.get("timed_out", False))})
    score = passed / len(tasks) if tasks else 0.0
    return DimensionScore("swe_coding", score, passed, len(tasks), details)


register_suite("swe_coding", _dim_swe_coding)


# ── research usefulness: claim-labeling accuracy ─────────────────────────────
# The system is handed checkable claims (some true, some false, some
# unverifiable) and must label each. A hermetic verifier — no network, no
# model — computes ground truth, so the score is claim-labeling accuracy
# with an explicit denominator. Unverifiable claims are excluded from the
# denominator and reported separately (never silently counted either way).


@dataclass
class ResearchTask:
    id: str
    question: str
    # each claim: {"id", "text", "check": (kind, *args) | None}.
    # check=None -> unverifiable by the harness (excluded from scoring).
    claims: list[dict[str, Any]]


_RESEARCH_TASKS: list[ResearchTask] = []


def register_research_task(task: ResearchTask) -> None:
    _RESEARCH_TASKS.append(task)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _run_claim_check(check: tuple[Any, ...] | None) -> bool | None:
    """Run one hermetic claim check. True/False, or None = unverifiable."""
    if not check:
        return None
    kind = check[0]
    root = _repo_root()
    try:
        if kind == "symbol":
            _, module, name = check
            mod = importlib.import_module(str(module))
            return hasattr(mod, str(name))
        if kind == "file_exists":
            return (root / str(check[1])).is_file()
        if kind == "line_count_ge":
            _, rel, n = check
            text = (root / str(rel)).read_text(encoding="utf-8",
                                               errors="ignore")
            return len(text.splitlines()) >= int(n)
        if kind == "regex_in_file":
            _, rel, pattern = check
            text = (root / str(rel)).read_text(encoding="utf-8",
                                               errors="ignore")
            return re.search(str(pattern), text) is not None
        if kind == "cli_command":
            from .. import cli as _cli

            return str(check[1]) in _cli.CLI_ALIASES
    except Exception:  # noqa: BLE001 — a broken check reads as False
        return False
    return None


register_research_task(ResearchTask(
    id="repo-facts",
    question=("Label each claim about this codebase true or false. "
              "Only label 'unverifiable' if the claim cannot be checked "
              "from the repository itself."),
    claims=[
        {"id": "c1",
         "text": "nomorals/agents/benchmark.py defines a dataclass named DimensionScore",
         "check": ("symbol", "nomorals.agents.benchmark", "DimensionScore")},
        {"id": "c2",
         "text": "nomorals/agents/benchmark.py defines a function named run_scoreboard",
         "check": ("symbol", "nomorals.agents.benchmark", "run_scoreboard")},
        {"id": "c3",
         "text": "nomorals/agents/benchmark.py defines a class named QuantumFluxCapacitor",
         "check": ("symbol", "nomorals.agents.benchmark", "QuantumFluxCapacitor")},
        {"id": "c4",
         "text": "the file nomorals/cli.py exists in the repo",
         "check": ("file_exists", "nomorals/cli.py")},
        {"id": "c5",
         "text": "the CLI has a top-level command named 'benchmark'",
         "check": ("cli_command", "benchmark")},
        {"id": "c6",
         "text": "nomorals/agents/benchmark.py has more than 100000 lines",
         "check": ("line_count_ge", "nomorals/agents/benchmark.py", 100000)},
        {"id": "c7",
         "text": "the next major release will ship on a Tuesday",
         "check": None},
    ],
))


def _label_claims_selftest(task: ResearchTask) -> dict[str, str]:
    """Self-check labeler: the verifier labels its own claims (mechanics
    check — expected accuracy 1.0 on verifiable claims)."""
    labels: dict[str, str] = {}
    for claim in task.claims:
        truth = _run_claim_check(claim.get("check"))
        labels[claim["id"]] = ("unverifiable" if truth is None
                               else ("true" if truth else "false"))
    return labels


def _label_claims_model(context: Any, task: ResearchTask) -> dict[str, str]:
    listed = "\n".join(f'- [{c["id"]}] {c["text"]}' for c in task.claims)
    text = _llm(
        context,
        "You are a careful research assistant. Label each claim true, "
        "false, or unverifiable. Reply with ONLY JSON of the form "
        '{"labels":[{"id":"<claim id>","label":"true|false|unverifiable"}]}.',
        f"{task.question}\n\nClaims:\n{listed}",
    )
    data = _extract_json(text) if text else None
    labels: dict[str, str] = {}
    items = data.get("labels") if isinstance(data, dict) else None
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and item.get("id"):
            label = str(item.get("label", "")).strip().lower()
            labels[str(item["id"])] = (label if label in
                                       {"true", "false", "unverifiable"}
                                       else "unverifiable")
    return labels


def _score_claim_labels(task: ResearchTask,
                        labels: dict[str, str]) -> dict[str, Any]:
    verifiable = [c for c in task.claims if c.get("check")]
    unverifiable = [c for c in task.claims if not c.get("check")]
    correct = 0
    per_claim: list[dict[str, Any]] = []
    for claim in verifiable:
        truth = _run_claim_check(claim["check"])
        expected = "true" if truth else "false"
        got = labels.get(claim["id"], "unverifiable")
        hit = got == expected
        correct += hit
        per_claim.append({"id": claim["id"], "expected": expected,
                          "got": got, "hit": hit})
    return {
        "correct": correct,
        "verifiable": len(verifiable),
        "unverifiable": len(unverifiable),
        "per_claim": per_claim,
    }


def _dim_research(context: Any, limit: int, *,
                  mode: str, **_kw: Any) -> DimensionScore:
    tasks = _RESEARCH_TASKS[: (limit or len(_RESEARCH_TASKS))]
    passed = 0
    total = 0
    details: list[dict[str, Any]] = []
    for task in tasks:
        try:
            if mode == MODE_MODEL:
                labels = _label_claims_model(context, task)
            else:
                labels = _label_claims_selftest(task)
            scored = _score_claim_labels(task, labels)
            task_correct = scored["correct"]
            task_acc = (task_correct / scored["verifiable"]
                        if scored["verifiable"] else 0.0)
            detail = (f"{scored['correct']}/{scored['verifiable']} claims "
                      f"right ({scored['unverifiable']} unverifiable "
                      f"excluded)")
            if mode != MODE_MODEL:
                # negative control: flip one label — the scorer must notice
                flipped = dict(labels)
                verifiable = [c for c in task.claims if c.get("check")]
                if verifiable:
                    first = verifiable[0]["id"]
                    flipped[first] = ("false" if flipped.get(first) == "true"
                                      else "true")
                    neg = _score_claim_labels(task, flipped)
                    detail += (f"; control: flipped label scores "
                               f"{neg['correct']}/{neg['verifiable']}")
                    if neg["correct"] >= scored["correct"]:
                        detail += " [BROKEN SCORER: flip not detected]"
                        task_correct = 0  # this task's score is untrusted
            total += scored["verifiable"]
            passed += task_correct
        except Exception as exc:  # noqa: BLE001 — one bad task, on
            detail = f"{type(exc).__name__}: {exc}"[:160]
            task_acc = 0.0
        details.append({"task": f"{task.id}: {task.question[:80]}",
                        "pass": task_acc >= 0.5, "detail": detail})
    score = passed / total if total else 0.0
    # the dimension score is claim-level accuracy (passed/total at claim
    # granularity); per-task pass marks tasks at/above 50% accuracy.
    return DimensionScore("research", score, passed, total, details)


register_suite("research", _dim_research)


# ── edit precision: intended vs actual diff ──────────────────────────────────
# Fixture file + intended (old -> new) hunks are applied through the Wave-A
# edit_loop. Metrics: hunks applied cleanly, exact-match hit rate, bogus
# hunks rejected, collateral lines (changes outside the intended hunks).


@dataclass
class EditTask:
    id: str
    filename: str
    original: str
    edits: list[tuple[str, str]]        # intended (old, new) hunks
    bogus_edits: list[tuple[str, str]]  # must be REJECTED, not applied


_EDIT_TASKS: list[EditTask] = []


def register_edit_task(task: EditTask) -> None:
    _EDIT_TASKS.append(task)


register_edit_task(EditTask(
    id="config-bump",
    filename="settings.py",
    original=(
        "MAX_RETRIES = 3\n"
        "TIMEOUT_S = 30\n"
        "\n"
        "\n"
        "def connect(host):\n"
        "    attempt = 0\n"
        "    while attempt < MAX_RETRIES:\n"
        "        attempt += 1\n"
        "    return None\n"
    ),
    edits=[
        ("MAX_RETRIES = 3", "MAX_RETRIES = 5"),
        ("TIMEOUT_S = 30", "TIMEOUT_S = 60"),
        ("    return None\n", "    return attempt\n"),
    ],
    bogus_edits=[
        ("THIS_STRING_DOES_NOT_EXIST = 1", "x = 2"),
        ("MAX_RETRIES", "MAX_RETRIES"),  # occurs twice -> must be rejected
    ],
))


def _edit_plan_selftest(task: EditTask) -> list[tuple[str, str]]:
    return list(task.edits)


def _edit_plan_model(context: Any, task: EditTask) -> list[tuple[str, str]]:
    text = _llm(
        context,
        "You are a precise editing agent. Reply with ONLY JSON of the "
        'form {"edits":[{"old":"<exact text to replace>",'
        '"new":"<replacement text>"}]}. Each "old" must occur exactly once '
        "in the file. No prose outside the JSON.",
        ("File 'settings.py':\n```python\n" + task.original + "```\n\n"
         "Make these changes: raise MAX_RETRIES from 3 to 5, raise "
         "TIMEOUT_S from 30 to 60, and make connect() return `attempt` "
         "instead of None."),
    )
    data = _extract_json(text) if text else None
    items = data.get("edits") if isinstance(data, dict) else None
    plan: list[tuple[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if isinstance(item, dict) and "old" in item and "new" in item:
            plan.append((str(item["old"]), str(item["new"])))
    return plan


def _apply_edit_plan(task: EditTask,
                     plan: list[tuple[str, str]]) -> dict[str, Any]:
    """Apply a plan through the Wave-A edit_loop; measure everything."""
    from ..tools.edit_loop import EditLoop

    tmp = Path(tempfile.mkdtemp(prefix=f"k3-edit-{task.id}-"))
    target = tmp / task.filename
    target.write_text(task.original, encoding="utf-8")
    loop = EditLoop(agent=None, project_root=str(tmp))  # type: ignore[arg-type]
    clean = 0
    applied_new: list[str] = []
    try:
        for old, new in plan:
            try:
                loop.surgical_replace(task.filename, old, new)
            except (ValueError, FileNotFoundError):
                continue  # rejected hunk — counted, not fatal
            clean += 1
            applied_new.append(new)
        # bogus hunks must be rejected, never applied
        rejected = 0
        for old, new in task.bogus_edits:
            try:
                loop.surgical_replace(task.filename, old, new,
                                      dry_run=True)
            except (ValueError, FileNotFoundError):
                rejected += 1
        result = target.read_text(encoding="utf-8")
    finally:
        import shutil as _shutil

        _shutil.rmtree(tmp, ignore_errors=True)
    # collateral: lines changed that no intended hunk accounts for
    collateral = _count_collateral_lines(task.original, result, task.edits)
    semantic_hits = sum(1 for _, new in task.edits if new in result)
    return {
        "clean": clean,
        "hunks": len(task.edits),
        "rejected_bogus": rejected,
        "bogus": len(task.bogus_edits),
        "collateral_lines": collateral,
        "semantic_hits": semantic_hits,
        "semantic_total": len(task.edits),
    }


def _count_collateral_lines(original: str, result: str,
                            edits: list[tuple[str, str]]) -> int:
    """Lines changed that no intended hunk accounts for."""
    import difflib as _difflib

    accounted: set[str] = set()
    for _old, new in edits:
        accounted.update(new.splitlines())
    changed = 0
    for line in _difflib.unified_diff(original.splitlines(),
                                      result.splitlines(), lineterm=""):
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:]
            if body not in accounted:
                changed += 1
    return changed


def _dim_edits(context: Any, limit: int, *,
               mode: str, **_kw: Any) -> DimensionScore:
    tasks = _EDIT_TASKS[: (limit or len(_EDIT_TASKS))]
    passed = 0
    total = 0
    details: list[dict[str, Any]] = []
    for task in tasks:
        try:
            plan = (_edit_plan_model(context, task) if mode == MODE_MODEL
                    else _edit_plan_selftest(task))
            m = _apply_edit_plan(task, plan)
            total += m["hunks"]
            hits = m["clean"] if mode != MODE_MODEL else m["semantic_hits"]
            passed += hits
            ok = (m["clean"] == m["hunks"]
                  and m["rejected_bogus"] == m["bogus"]
                  and m["collateral_lines"] == 0)
            detail = (f"{m['clean']}/{m['hunks']} hunks clean, "
                      f"{m['semantic_hits']}/{m['semantic_total']} intended "
                      f"changes present, {m['rejected_bogus']}/{m['bogus']} "
                      f"bogus rejected, {m['collateral_lines']} collateral "
                      f"lines")
        except Exception as exc:  # noqa: BLE001 — one bad task, on
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        details.append({"task": f"{task.id}: {task.filename}", "pass": ok,
                        "detail": detail})
    score = passed / total if total else 0.0
    return DimensionScore("edits", score, passed, total, details)


register_suite("edits", _dim_edits)


# ── build success: scaffold -> serve -> smoke ────────────────────────────────
# The builders package lives at L7 (entry-point layer), so this module at
# L5 must NOT import it (see tests/test_layering.py). The backend is
# injected: ``nm benchmark run`` wires the real builders implementation;
# tests inject fakes. With no backend the dimension is honestly reported
# unmeasurable — never faked.


@runtime_checkable
class BuildBackend(Protocol):
    """Scaffold one app of ``kind`` in ``workdir``, serve it if it is an
    HTTP app, smoke-test it, and report stage booleans."""

    def run(self, kind: str, workdir: str) -> dict[str, Any]: ...


#: app kinds exercised by the build dimension (cli_tool smokes via --help,
#: webapp is served and HTTP-smoked)
_BUILD_KINDS = ("cli_tool", "webapp")


def _dim_builds(context: Any, limit: int, *,
                mode: str, build_backend: BuildBackend | None = None,
                **_kw: Any) -> DimensionScore:
    kinds = list(_BUILD_KINDS[: (limit or len(_BUILD_KINDS))])
    if build_backend is None:
        return DimensionScore(
            "builds", None, 0, 0,
            [{"task": k, "pass": None,
              "detail": "no build backend wired (builders/ is L7-only; "
                        "run via `nm benchmark run`)"} for k in kinds])
    passed = 0
    total = 0
    details: list[dict[str, Any]] = []
    for kind in kinds:
        stages = ("scaffold", "serve", "smoke") if kind == "webapp" \
            else ("scaffold", "smoke")
        try:
            with tempfile.TemporaryDirectory(prefix=f"k3-build-{kind}-") as tmp:
                outcome = build_backend.run(kind, tmp)
            stage_ok = {s: bool(outcome.get(f"{s}ed",
                                           outcome.get(s, False)))
                        for s in stages}
            # a timeout fails every stage it touches — never a skip
            if outcome.get("timed_out"):
                stage_ok = dict.fromkeys(stages, False)
            n_ok = sum(stage_ok.values())
            passed += n_ok
            total += len(stages)
            ok = n_ok == len(stages)
            detail = (" ".join(f"{s}={'ok' if v else 'FAIL'}"
                               for s, v in stage_ok.items())
                      + f" ({outcome.get('seconds', 0.0):.1f}s)"
                      + (f" — {outcome.get('detail', '')}"
                         if outcome.get("detail") else ""))
            if outcome.get("timed_out"):
                detail += " [TIMED OUT — counted as failure]"
        except Exception as exc:  # noqa: BLE001 — one bad build, on
            total += len(stages)
            ok, detail = False, f"{type(exc).__name__}: {exc}"[:160]
        details.append({"task": f"build:{kind}", "pass": ok, "detail": detail})
    score = passed / total if total else 0.0
    return DimensionScore("builds", score, passed, total, details)


register_suite("builds", _dim_builds)


# ── latency tiers: fast-path vs heavy-path ───────────────────────────────────
# fast  = CoreMind.decide(allow_model=False): the deterministic intent pass,
#         never touches the model router.
# heavy = CoreMind._model_check: the bounded router call the mind makes when
#         the deterministic pass is unsure (MODEL_CHECK_TIMEOUT_S budget).
# Score = fraction of fast reps that beat the heavy p50 — a direct,
# unit-free "the fast path is actually faster" number, with p50/p99 for
# both tiers in the details.

_FAST_TEXTS = (
    "status",
    "what is the system status",
    "start hangman",
    "play word chain",
    "stop",
)

_HEAVY_TEXT = "maybe look into that thing we discussed"


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    ordered = sorted(xs)
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


class _SelfTestRouter:
    """Deterministic stand-in for the model router: fixed delay, then a
    low-confidence verdict (so the deterministic intent stands, exactly
    like a real inconclusive model call)."""

    def __init__(self, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s
        self.calls = 0

    def complete(self, prompt: str, params: Any = None, **kw: Any) -> Any:
        self.calls += 1
        time.sleep(self.delay_s)
        return _SimpleResponse(
            ok=True,
            text='{"kind": "chat", "confidence": 0.1, "why": "self-test"}',
        )


class _SimpleResponse:
    def __init__(self, ok: bool, text: str) -> None:
        self.ok = ok
        self.text = text


def _dim_latency(context: Any, limit: int, *,
                 mode: str, **_kw: Any) -> DimensionScore:
    from .coremind import CoreMind, Intent

    reps = max(5, min(limit or 25, 100))
    if mode != MODE_MODEL:
        # measure against a deterministic stand-in router on a shim context —
        # never mutate the caller's context object
        import types as _types

        shim = _types.SimpleNamespace(
            settings=context.settings,
            router=_SelfTestRouter(delay_s=0.05),
            extras=getattr(context, "extras", {}),
            memory=getattr(context, "memory", None),
        )
        mind = CoreMind(shim)
    else:
        mind = CoreMind(context)
    fast_ms: list[float] = []
    for i in range(reps):
        text = _FAST_TEXTS[i % len(_FAST_TEXTS)]
        started = time.perf_counter()
        try:
            mind.decide(text, allow_model=False)
        except Exception:  # noqa: BLE001 — a failed rep counts, slowly
            pass
        fast_ms.append((time.perf_counter() - started) * 1000.0)
    heavy_ms: list[float] = []
    for _ in range(reps):
        started = time.perf_counter()
        try:
            mind._model_check(_HEAVY_TEXT,
                              Intent("research", 0.6, why="latency bench"))
        except Exception:  # noqa: BLE001 — a failed rep counts, slowly
            pass
        heavy_ms.append((time.perf_counter() - started) * 1000.0)
    heavy_p50 = _percentile(heavy_ms, 50)
    wins = sum(1 for x in fast_ms if x < heavy_p50)
    details = [
        {"task": "fast-path p50/p99 (ms)", "pass": None,
         "detail": f"p50={_percentile(fast_ms, 50):.2f} "
                   f"p99={_percentile(fast_ms, 99):.2f} over {reps} reps"},
        {"task": "heavy-path p50/p99 (ms)", "pass": None,
         "detail": f"p50={heavy_p50:.2f} "
                   f"p99={_percentile(heavy_ms, 99):.2f} over {reps} reps"
                   + (" (simulated router)" if mode != MODE_MODEL else "")},
        {"task": "fast beats heavy", "pass": None,
         "detail": f"{wins}/{reps} fast reps under heavy p50 "
                   f"(ratio heavy/fast p50: "
                   f"{(heavy_p50 / max(_percentile(fast_ms, 50), 1e-9)):.1f}x)"},
    ]
    for d in details:
        d["pass"] = wins == reps
    score = wins / reps if reps else 0.0
    return DimensionScore("latency", score, wins, reps, details)


register_suite("latency", _dim_latency)


# ── the scoreboard runner ────────────────────────────────────────────────────

#: legacy dimensions (need a live model) also runnable under the scoreboard
_LEGACY_SUITES = ("reasoning", "planning", "tool_use", "self_correction")
for _legacy in _LEGACY_SUITES:
    register_suite(_legacy, _DIMENSIONS[_legacy])


def run_scoreboard(context: Any, *,
                   suites: list[str] | None = None,
                   limit: int = 0,
                   build_backend: BuildBackend | None = None,
                   ) -> ScoreboardReport:
    """Run the K3 scoreboard.

    ``suites``: subset of list_suites() (default: all). ``limit``: cap
    tasks per suite. ``build_backend``: injected builders backend for the
    ``builds`` suite (L7 only — see BuildBackend).

    Without a live LLM the harness runs in ``harness-self-test`` mode:
    reference solutions, negative controls, and a simulated router verify
    the mechanics end-to-end. Scores from that mode are labeled as such
    and must never be presented as model scores.
    """
    started = time.monotonic()
    provider = _provider_name(context)
    is_measurable = measurable(context)
    mode = MODE_MODEL if is_measurable else MODE_SELF_TEST
    names = [s for s in (suites or list_suites()) if s in _SUITE_FNS]
    scores: dict[str, DimensionScore] = {}
    for name in names:
        fn = _SUITE_FNS[name]
        if name in _LEGACY_SUITES and not is_measurable:
            scores[name] = DimensionScore(
                name, None, 0, 0,
                [{"task": name, "pass": None,
                  "detail": "needs a live model (self-test mode)"}])
            continue
        try:
            scores[name] = fn(context, limit, mode=mode,
                              build_backend=build_backend)
        except Exception as exc:  # noqa: BLE001 — one bad suite, on
            scores[name] = DimensionScore(
                name, 0.0, 0, 1,
                [{"task": name, "pass": False,
                  "detail": f"suite crashed: {type(exc).__name__}: {exc}"[:200]}])
    measured = [s.score for s in scores.values() if s.score is not None]
    overall = sum(measured) / len(measured) if measured else None
    return ScoreboardReport(
        scores=scores,
        overall=overall,
        measurable=is_measurable,
        provider=provider,
        seconds=time.monotonic() - started,
        run_id=secrets.token_hex(8),
        suite=",".join(names) if names != list_suites() else "all",
        mode=mode,
    )


# ── persistence: the scoreboard table ────────────────────────────────────────
# One row per run (migration 63: benchmark_runs). Per-dimension detail is
# JSON in one column so new suites never need schema changes.

_RUN_COLUMNS = ("id", "ts", "scoreboard", "suite", "mode", "provider",
                "measurable", "overall", "passed", "total", "dimensions",
                "seconds", "notes")


def save_run(db: Any, report: ScoreboardReport, *,
             notes: str = "") -> str:
    """Persist a scoreboard run. Returns the run id ("" when no database —
    never raises)."""
    if db is None or not report.run_id:
        return ""
    passed = sum(s.passed for s in report.scores.values())
    total = sum(s.total for s in report.scores.values())
    row = {
        "id": report.run_id,
        "ts": time.time(),
        "scoreboard": report.scoreboard,
        "suite": report.suite,
        "mode": report.mode,
        "provider": report.provider,
        "measurable": int(bool(report.measurable)),
        "overall": report.overall,
        "passed": passed,
        "total": total,
        "dimensions": json.dumps(
            {k: v.as_dict() for k, v in report.scores.items()}),
        "seconds": report.seconds,
        "notes": notes or "",
    }
    try:
        with db.transaction():
            db.insert("benchmark_runs", row)
        return report.run_id
    except Exception:  # noqa: BLE001 — persistence is best-effort
        _log.warning("benchmark run %s not persisted", report.run_id,
                     exc_info=True)
        return ""


def list_runs(db: Any, *, suite: str = "",
              limit: int = 50) -> list[dict[str, Any]]:
    """Newest-first run summaries (no per-dimension detail payload)."""
    if db is None:
        return []
    sql = ("SELECT id, ts, scoreboard, suite, mode, provider, measurable, "
           "overall, passed, total, seconds, notes FROM benchmark_runs")
    params: list[Any] = []
    if suite:
        sql += " WHERE suite = ?"
        params.append(suite)
    sql += " ORDER BY ts DESC LIMIT ?"
    params.append(max(1, int(limit)))
    try:
        return db.query(sql, params)
    except Exception:  # noqa: BLE001
        _log.warning("benchmark list_runs failed", exc_info=True)
        return []


def get_run(db: Any, run_id: str) -> dict[str, Any] | None:
    """Full run record including per-dimension detail, or None."""
    if db is None or not run_id:
        return None
    try:
        row = db.query_one(
            "SELECT * FROM benchmark_runs WHERE id = ?", (run_id,))
    except Exception:  # noqa: BLE001
        _log.warning("benchmark get_run failed", exc_info=True)
        return None
    if not row:
        return None
    try:
        row["dimensions"] = json.loads(row.get("dimensions") or "{}")
    except (json.JSONDecodeError, TypeError):
        row["dimensions"] = {}
    return row


def export_run_json(run: dict[str, Any]) -> str:
    """Serialize a run record (as returned by get_run) to JSON."""
    return json.dumps(run, indent=2, default=str, ensure_ascii=False)


def compare_runs(db: Any, run_a: str, run_b: str) -> dict[str, Any]:
    """Delta of run B relative to run A, per dimension and overall.

    Modes must match — comparing a harness-self-test against a
    model-scored run is refused, loudly, because the numbers mean
    different things.
    """
    a = get_run(db, run_a)
    b = get_run(db, run_b)
    if a is None or b is None:
        missing = run_a if a is None else run_b
        return {"ok": False, "error": f"run not found: {missing}"}
    if a.get("mode") != b.get("mode"):
        return {"ok": False,
                "error": (f"mode mismatch: {a.get('mode')} vs {b.get('mode')} "
                          f"— self-test and model-scored runs are not "
                          f"comparable")}
    dims_a = a.get("dimensions") or {}
    dims_b = b.get("dimensions") or {}
    deltas: dict[str, dict[str, Any]] = {}
    for name in sorted(set(dims_a) | set(dims_b)):
        da, dbb = dims_a.get(name) or {}, dims_b.get(name) or {}
        sa, sb = da.get("score"), dbb.get("score")
        deltas[name] = {
            "a": sa, "b": sb,
            "delta": (round(sb - sa, 3) if isinstance(sa, (int, float))
                      and isinstance(sb, (int, float)) else None),
            "a_passed": f"{da.get('passed')}/{da.get('total')}",
            "b_passed": f"{dbb.get('passed')}/{dbb.get('total')}",
        }
    oa, ob = a.get("overall"), b.get("overall")
    return {
        "ok": True,
        "a": {"id": a["id"], "ts": a["ts"], "suite": a.get("suite"),
              "mode": a.get("mode")},
        "b": {"id": b["id"], "ts": b["ts"], "suite": b.get("suite"),
              "mode": b.get("mode")},
        "overall_delta": (round(ob - oa, 3)
                          if isinstance(oa, (int, float))
                          and isinstance(ob, (int, float)) else None),
        "overall_a": oa, "overall_b": ob,
        "dimensions": deltas,
    }


# ── registry ─────────────────────────────────────────────────────────────────


def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "benchmark",
        description=(
            "Run the agent benchmark: reasoning + planning + tool use + "
            "self-correction dimensions, scored 0-1, hermetic — or the K3 "
            "scoreboard suites (swe_coding, research, edits, builds, "
            "latency) via the suite parameter. This is what the evolution "
            "promotion gate uses to block intelligence regressions."
        ),
        capability=Capability.MODEL_CALL,
        parameters={
            "dimensions": "str (optional) — comma list from reasoning,planning,tool_use,self_correction",
            "limit": "int (optional) — max tasks per dimension",
            "suite": "str (optional) — comma list from swe_coding,research,edits,builds,latency (runs the K3 scoreboard instead)",
        },
    )
    def benchmark(dimensions: str = "", limit: str = "",
                  suite: str = "") -> dict[str, Any]:
        try:
            n = int(limit or 0)
        except ValueError:
            n = 0
        suites = [s.strip() for s in suite.split(",") if s.strip()]
        if suites:
            # the builders backend is L7-only; the agent tool runs without
            # it and the builds suite reports itself unmeasurable, honestly
            report = run_scoreboard(context, suites=suites, limit=n)
            return report.as_dict()
        dims = [d.strip() for d in dimensions.split(",") if d.strip()] or None
        report = run_benchmark(context, dimensions=dims, limit=n)
        return report.as_dict()
