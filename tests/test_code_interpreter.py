"""Code interpreter: sandboxed python with sessions, last-expression capture.

Covers the contract the /py command and the ``code`` tool promise:

* ``print`` lands in stdout, a trailing expression lands in ``result``
* a named session keeps variables between runs, ``reset`` clears them
* exceptions come back as exit=1 + stderr, never as a raised traceback
* files created in the run are listed relative to the session workdir
* unsupported languages fail fast with a readable error
* the registry exposes ``code`` / ``code_session`` behind EXEC_CODE
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nomorals.core.policy import Capability, CapabilitySet
from nomorals.tools.sandbox_code import CodeInterpreter


def _interp(root: Path) -> CodeInterpreter:
    return CodeInterpreter(root=root)


class TestCodeInterpreter(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-ci-test-")
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_print_goes_to_stdout(self) -> None:
        r = _interp(self.tmp).run("print('hi from sandbox')\nprint(6*7)")
        self.assertTrue(r["ok"])
        self.assertEqual(r["exit_code"], 0)
        self.assertEqual(r["stdout"], "hi from sandbox\n42\n")

    def test_trailing_expression_captured_as_result(self) -> None:
        r = _interp(self.tmp).run("sum(range(10))")
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"], "45")
        self.assertEqual(r["stdout"], "")

    def test_assignment_yields_no_result(self) -> None:
        r = _interp(self.tmp).run("x = 5")
        self.assertTrue(r["ok"])
        self.assertIsNone(r["result"])

    def test_trailing_none_expression_maps_to_none(self) -> None:
        r = _interp(self.tmp).run("None if False else None")
        self.assertTrue(r["ok"])
        self.assertIsNone(r["result"])

    def test_result_serializes_structures(self) -> None:
        r = _interp(self.tmp).run("{'a': [1, 2, 3], 'b': None}")
        self.assertTrue(r["ok"])
        self.assertEqual(r["result"], "{'a': [1, 2, 3], 'b': None}")

    def test_session_persists_variables_between_runs(self) -> None:
        ci = _interp(self.tmp)
        first = ci.run("x = 10\ny = 'alpha'", session="work")
        second = ci.run("x * 2, y.upper()", session="work")
        self.assertEqual(first["ok"], True)
        self.assertEqual(second["result"], "(20, 'ALPHA')")

    def test_sessions_are_isolated(self) -> None:
        ci = _interp(self.tmp)
        ci.run("shared = 1", session="a")
        r = ci.run("shared", session="b")
        self.assertEqual(r["exit_code"], 1)
        self.assertIn("NameError", r["stderr"])

    def test_reset_clears_the_session(self) -> None:
        ci = _interp(self.tmp)
        ci.run("x = 1", session="s")
        ci.run("", session="s", reset=True)
        r = ci.run("x", session="s")
        self.assertEqual(r["exit_code"], 1)
        self.assertIn("NameError", r["stderr"])

    def test_exception_returns_exit_one_with_traceback(self) -> None:
        r = _interp(self.tmp).run("1/0")
        self.assertFalse(r["ok"])
        self.assertEqual(r["exit_code"], 1)
        self.assertIn("ZeroDivisionError", r["stderr"])

    def test_files_created_are_listed(self) -> None:
        r = _interp(self.tmp).run('open("out.txt", "w").write("hello")\n"done"', session="fs")
        self.assertTrue(r["ok"])
        self.assertIn("out.txt", r["files"])
        written = (Path(r["workdir"]) / "out.txt").read_text()
        self.assertEqual(written, "hello")

    def test_unsupported_language_fails_fast(self) -> None:
        r = _interp(self.tmp).run("fn main() {}", language="rust")
        self.assertFalse(r["ok"])
        self.assertIn("not supported", r["error"])

    def test_timeout_kills_the_run(self) -> None:
        r = _interp(self.tmp).run("import time\ntime.sleep(30)", timeout=2)
        self.assertFalse(r["ok"])
        self.assertTrue(r["timed_out"])

    def test_workdir_is_confined_to_sandbox_root(self) -> None:
        r = _interp(self.tmp).run("1", session="confine")
        self.assertTrue(Path(r["workdir"]).resolve().is_relative_to(self.tmp.resolve()))


class TestCodeToolRegistration(unittest.TestCase):
    """The registry exposes the interpreter behind EXEC_CODE."""

    def test_code_tool_registered_and_callable(self) -> None:
        import os

        from nomorals.agents.context import build_context
        from nomorals.storage.db import Database

        with tempfile.TemporaryDirectory(prefix="nm-ci-reg-") as d:
            os.environ["NM_SANDBOX_ROOT"] = d
            try:
                ctx = build_context(
                    db=Database(":memory:"),
                    with_router=False, with_memory=False,
                    with_executor=False, with_tools=True,
                )
                reg = ctx.tools
                self.assertIn("code", reg.names())
                self.assertIn("code_session", reg.names())

                # granted: runs
                ok = reg.call("code", code="7*6")
                self.assertTrue(ok.ok)
                self.assertEqual(ok.value["result"], "42")

                # not granted: denied, never runs
                denied = reg.call(
                    "code", code="1",
                    capabilities=CapabilitySet.of("fs.read"),
                )
                self.assertFalse(denied.ok)

                schema = next(s for s in reg.schemas() if s["name"] == "code")
                self.assertEqual(schema["capability"], Capability.EXEC_CODE)
            finally:
                os.environ.pop("NM_SANDBOX_ROOT", None)


if __name__ == "__main__":
    unittest.main()
