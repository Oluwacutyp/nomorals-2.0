"""Shell-fix pass: CPU budgets on execution paths derive from the caller timeout.

The ceiling fix (82df789) removed the coding loop's 600s verify/accept
clamps.  The same defect class survived on the other execution paths:
a caller-supplied wall-clock budget paired with a CPU allowance that a
fixed constant (or an un-derived default) silently held below it — a
CPU-bound execution is killed by RLIMIT_CPU long before the wall
budget the caller actually granted:

- ``shell_run`` / ``python_run`` (tools/shell.py): cpu_seconds was
  ``int(min(timeout, 900))`` while run_sandboxed got the raw timeout
  as wall clock — timeout=1800 meant a CPU kill at 900.
- ``benchmark._run_python``: ``int(min(max(timeout, 5.0), 60.0))``.
- ``benchmark._run_hidden_tests``: ``int(min(max(timeout_s, 5.0),
  120.0))`` against the raw timeout_s wall.
- ``simulation.SandboxSimulator.run``: explicit ``SandboxLimits()``
  — the default cpu_seconds=120 no matter what timeout the caller
  passed (default timeout is itself 120, so any larger budget bound).
- ``execbox.CodeRunner.run`` / ``._run_cmd``: no limits at all, so
  DEFAULT_LIMITS cpu_seconds=120 applied while the wall could be 300
  (run, after its named MAX_TIMEOUT clamp) or 180 (run_project_
  until_green's DEFAULT timeout into _run_cmd — bound by default).

The fix derives the CPU allowance from the caller's timeout at every
one of those sites — ``int(timeout)`` where the site had no floor,
``int(max(timeout, 5.0))`` where the site's derivation already carried
the 5s floor (benchmark, mirroring the ceiling fix in coding.py).
Boundary: for every timeout at or below the old clamp the computed
value is IDENTICAL to the pre-fix code; only larger budgets change.
No caller default moved, no replacement ceiling introduced.

These tests spy ``run_sandboxed`` (the way the existing shell tests
fake it) and pin the contract per site:

(a) timeout=1800 (or the site's large-budget analogue) -> the CPU
    allowance follows the budget, not the old clamp/default;
(b) defaults and small timeouts compute exactly the pre-fix values;
(c) the wall timeout passes through to run_sandboxed untouched.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.tools import shell as shell_module


def _result_dict(**overrides):
    result = {
        "exit_code": 0,
        "stdout": "",
        "stderr": "",
        "truncated": False,
        "timed_out": False,
        "seconds": 0.01,
        "backend": "spy",
        "network": False,
        "network_enforced": False,
        "fs_confined": False,
        "cwd": "",
    }
    result.update(overrides)
    return result


class _Recorder:
    """Stand-in for run_sandboxed: records calls, runs nothing."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, command, **kwargs):
        self.calls.append({"command": command, **kwargs})
        return _result_dict(cwd=str(kwargs.get("cwd", "")))

    @property
    def last(self) -> dict:
        return self.calls[-1]


class _Registry:
    """Minimal registry double capturing what shell.register attaches."""

    def __init__(self, context) -> None:
        self.context = context
        self.funcs: dict = {}

    def register(self, name, description=None, capability=None,
                 parameters=None):
        def deco(fn):
            self.funcs[name] = fn
            return fn
        return deco


class _ShellToolCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-shellfix-")
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=self.tmp, tools=None))
        registry = _Registry(context)
        shell_module.register(registry)
        self.funcs = registry.funcs
        self.recorder = _Recorder()
        patcher = mock.patch.object(
            shell_module, "run_sandboxed", self.recorder)
        patcher.start()
        self.addCleanup(patcher.stop)


class ShellRunCpuBudgetTests(_ShellToolCase):
    def test_1800_cpu_follows_budget_not_900_clamp(self):
        self.funcs["shell_run"]("echo hi", timeout=1800.0)
        call = self.recorder.last
        self.assertEqual(call["limits"].cpu_seconds, 1800)
        self.assertEqual(call["timeout"], 1800.0)  # wall untouched

    def test_default_and_small_timeouts_unchanged(self):
        self.funcs["shell_run"]("echo hi")
        self.assertEqual(self.recorder.last["limits"].cpu_seconds, 120)
        self.assertEqual(self.recorder.last["timeout"], 120.0)
        for timeout, expected in ((120.0, 120), (60.0, 60), (2.0, 2),
                                  (900.0, 900)):
            self.funcs["shell_run"]("echo hi", timeout=timeout)
            self.assertEqual(
                self.recorder.last["limits"].cpu_seconds, expected)
            self.assertEqual(self.recorder.last["timeout"], timeout)


class PythonRunCpuBudgetTests(_ShellToolCase):
    def test_1800_cpu_follows_budget_not_900_clamp(self):
        self.funcs["python_run"]("print(1)", timeout=1800.0)
        call = self.recorder.last
        self.assertEqual(call["limits"].cpu_seconds, 1800)
        self.assertEqual(call["timeout"], 1800.0)  # wall untouched

    def test_default_and_small_timeouts_unchanged(self):
        self.funcs["python_run"]("print(1)")
        self.assertEqual(self.recorder.last["limits"].cpu_seconds, 60)
        self.assertEqual(self.recorder.last["timeout"], 60.0)
        for timeout, expected in ((120.0, 120), (60.0, 60), (2.0, 2),
                                  (900.0, 900)):
            self.funcs["python_run"]("print(1)", timeout=timeout)
            self.assertEqual(
                self.recorder.last["limits"].cpu_seconds, expected)
            self.assertEqual(self.recorder.last["timeout"], timeout)


class _SandboxedSpyCase(unittest.TestCase):
    """Base for callers that import run_sandboxed from tools.shell."""

    def setUp(self):
        self.recorder = _Recorder()
        patcher = mock.patch.object(
            shell_module, "run_sandboxed", self.recorder)
        patcher.start()
        self.addCleanup(patcher.stop)


class BenchmarkCpuBudgetTests(_SandboxedSpyCase):
    def test_run_python_large_budget_not_clamped_to_60(self):
        from nomorals.agents import benchmark

        benchmark._run_python("x = 1", timeout=1800.0)
        call = self.recorder.last
        self.assertEqual(call["limits"].cpu_seconds, 1800)
        self.assertEqual(call["timeout"], 1800.0)

    def test_run_python_default_unchanged(self):
        from nomorals.agents import benchmark

        benchmark._run_python("x = 1")
        self.assertEqual(self.recorder.last["limits"].cpu_seconds, 15)
        self.assertEqual(self.recorder.last["timeout"], 15.0)

    def test_hidden_tests_large_budget_not_clamped_to_120(self):
        from nomorals.agents import benchmark

        workdir = Path(tempfile.mkdtemp(prefix="nm-shellfix-swe-"))
        outcome = benchmark._run_hidden_tests(workdir, 1800.0)
        call = self.recorder.last
        self.assertEqual(call["limits"].cpu_seconds, 1800)
        self.assertEqual(call["timeout"], 1800.0)
        self.assertTrue(outcome["pass"])

    def test_hidden_tests_default_task_budget_unchanged(self):
        from nomorals.agents import benchmark

        workdir = Path(tempfile.mkdtemp(prefix="nm-shellfix-swe-"))
        benchmark._run_hidden_tests(workdir, 25.0)
        self.assertEqual(self.recorder.last["limits"].cpu_seconds, 25)


class SimulationCpuBudgetTests(_SandboxedSpyCase):
    def _simulator(self):
        from nomorals.agents.simulation import SandboxSimulator

        tmp = tempfile.mkdtemp(prefix="nm-shellfix-sim-")
        context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=tmp, tools=None))
        return SandboxSimulator(context)

    def test_large_budget_not_held_at_default_120(self):
        sim = self._simulator()
        sim.run("echo hi", timeout=1800.0)
        call = self.recorder.last
        self.assertEqual(call["limits"].cpu_seconds, 1800)
        self.assertEqual(call["timeout"], 1800.0)  # wall untouched

    def test_default_budget_unchanged(self):
        sim = self._simulator()
        sim.run("echo hi")
        self.assertEqual(self.recorder.last["limits"].cpu_seconds, 120)
        self.assertEqual(self.recorder.last["timeout"], 120.0)


class ExecboxCpuBudgetTests(_SandboxedSpyCase):
    def _runner(self):
        from nomorals.execbox import CodeRunner

        return CodeRunner(SimpleNamespace())

    def test_run_passes_limits_derived_from_budget(self):
        runner = self._runner()
        runner.run("print(1)", lang="python", timeout=240.0)
        call = self.recorder.last
        self.assertIsNotNone(call.get("limits"))
        self.assertEqual(call["limits"].cpu_seconds, 240)
        self.assertEqual(call["timeout"], 240.0)  # wall untouched

    def test_run_default_budget(self):
        runner = self._runner()
        runner.run("print(1)", lang="python")
        call = self.recorder.last
        self.assertIsNotNone(call.get("limits"))
        self.assertEqual(call["limits"].cpu_seconds, 30)
        self.assertEqual(call["timeout"], 30.0)

    def test_run_cmd_project_default_budget_not_held_at_120(self):
        # run_project_until_green's default timeout is 180 — above the
        # old DEFAULT_LIMITS cpu_seconds=120, so this bound by default.
        runner = self._runner()
        tmp = tempfile.mkdtemp(prefix="nm-shellfix-proj-")
        runner._run_cmd(["echo", "hi"], cwd=tmp, timeout=180.0)
        call = self.recorder.last
        self.assertIsNotNone(call.get("limits"))
        self.assertEqual(call["limits"].cpu_seconds, 180)
        self.assertEqual(call["timeout"], 180.0)  # wall untouched


if __name__ == "__main__":
    unittest.main()
