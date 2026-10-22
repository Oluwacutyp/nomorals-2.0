"""Pass #69: the coding run's STEP ACCOUNT renders on owner surfaces.

The defect, from the surface survey: since pass #60 a CodingResult
carries a rich per-step account (steps with status / reason /
attempts, #68's per-step verify records, #63's review verdict,
#65's usage + capped state, #66's scope denials) — and no owner
surface showed any of it:

- chat ``/code`` (partner/runtime_build._control_code): one headline
  line + raw output/error excerpt;
- chat NL build (coremind job replies) and /task directives
  (directives._code): the same flat one-liner shape;
- ``nm code`` (cmdline/commands/code.py): a stats line + the #65
  tokens suffix, then OK/FAILED;
- mission coding steps (roles.MissionCodingAgent._settle): the
  settlement dict carried steps/review/usage/scope_denials as data,
  but the owner's rendering of it — ``/mission results`` via
  missions/progress.payload_text — keys on the text-like "output"
  field, so the owner saw only the loop's raw output tail.

The fix: ONE shared renderer, ``render_step_account`` in
agents/coding.py beside CodingResult, called from every surface
above (the mission settlement carries its output as a pre-rendered
``step_account`` string, which payload_text leads with).  A result
with no steps renders "" — every surface then produces exactly the
bytes it produced before this pass.
"""

from __future__ import annotations

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.coding import (
    STEP_ACCOUNT_FULL_LIMIT,
    CodingResult,
    render_step_account,
)


# ── fabricated step / verify shapes (mirrors passes #60/#68) ─────────

def _verify_ok(tests=("tests/test_a.py::test_a",)):
    return {"ran": True, "ok": True, "tests": list(tests), "failed": [],
            "checks": 1, "executed_total": len(tests), "note": ""}


def _verify_failed(failed, tests=("tests/test_a.py::test_a",)):
    return {"ran": True, "ok": False, "tests": list(tests),
            "failed": list(failed), "checks": 2,
            "executed_total": len(tests), "note": ""}


def _verify_none(note="no tests cover this step's files — the step "
                      "is unverified at this step"):
    return {"ran": False, "ok": None, "tests": [], "failed": [],
            "checks": 0, "executed_total": 0, "note": note}


def _step(sid, status, *, instructions="", path="", attempts=1,
          reason="", verify="absent", depends_on=None):
    d = {"id": sid, "path": path, "files": [path] if path else [],
         "status": status, "reason": reason, "attempts": attempts,
         "instructions": instructions, "acceptance": "",
         "depends_on": list(depends_on or [])}
    if verify != "absent":
        d["verify"] = verify
    return d


def _mixed_result(**overrides):
    """A 5-step run: 3 done, 1 failed (named failing test), 1 skipped."""
    steps = [
        _step("s1", "done", instructions="add the config parser",
              path="calc.py", verify=_verify_ok()),
        _step("s2", "done", instructions="wire the parser into main",
              path="main.py", verify=_verify_ok()),
        _step("s3", "done", instructions="add scratch notes module",
              path="notes.py", verify=_verify_none()),
        _step("s4", "failed", instructions="render the report title",
              path="report.py", attempts=3,
              reason="verify failed: tests/test_report.py::test_title",
              verify=_verify_failed(
                  ["tests/test_report.py::test_title"],
                  tests=("tests/test_report.py::test_title",))),
        _step("s5", "skipped", instructions="publish the report",
              path="publish.py", reason="dependency s4 failed",
              depends_on=["s4"], verify=_verify_none(
                  "no isolated check ran for this step")),
    ]
    kw = dict(ok=False, iterations=4, files=["calc.py", "main.py"],
              output="some output", error="run failed", steps=steps)
    kw.update(overrides)
    return CodingResult(**kw)


class RendererUnitTests(unittest.TestCase):

    def test_all_done_short_plan_every_step_shown(self):
        steps = [
            _step("s1", "done", instructions="add the config parser",
                  path="calc.py", verify=_verify_ok()),
            _step("s2", "done", instructions="wire the parser into main",
                  path="main.py", verify=_verify_ok()),
            _step("s3", "done", instructions="document the flags",
                  path="docs.py", verify=_verify_ok()),
        ]
        out = render_step_account(CodingResult(
            ok=True, iterations=2, steps=steps))
        self.assertIn("steps: 3 done", out)
        for sid, gist in (("s1", "add the config parser"),
                          ("s2", "wire the parser into main"),
                          ("s3", "document the flags")):
            self.assertIn(sid, out)
            self.assertIn(gist, out)
        # every step line carries its green verify state
        self.assertEqual(out.count("tests ✓"), 3)

    def test_failed_step_names_failing_tests_and_reason(self):
        out = render_step_account(_mixed_result())
        self.assertIn("1 failed", out)
        self.assertIn("s4", out)
        self.assertIn("tests/test_report.py::test_title", out)
        self.assertIn("verify failed", out)  # the step's reason
        self.assertIn("3 attempts", out)

    def test_skipped_dependent_shows_reason(self):
        out = render_step_account(_mixed_result())
        self.assertIn("1 skipped", out)
        self.assertIn("s5", out)
        self.assertIn("dependency s4 failed", out)

    def test_unverified_step_is_named_not_dressed_up(self):
        out = render_step_account(_mixed_result())
        self.assertIn("s3", out)
        self.assertIn("no tests cover", out)
        # the unverified step must not read as verified
        s3_line = next(ln for ln in out.splitlines() if "s3" in ln)
        self.assertNotIn("tests ✓", s3_line.replace("not verified", ""))

    def test_step_lines_stay_one_line(self):
        long_reason = "boom " * 80
        steps = [_step("s1", "failed", instructions="x " * 60,
                       path="a.py", reason=long_reason,
                       verify=_verify_failed(["tests/test_a.py::test_a"]))]
        out = render_step_account(CodingResult(
            ok=False, iterations=1, steps=steps))
        for line in out.splitlines():
            self.assertLessEqual(len(line), 160, line)

    def test_capped_run_line(self):
        result = _mixed_result(usage={
            "prompt_tokens": 4000, "completion_tokens": 1123,
            "total_tokens": 5123, "calls": 9, "usage_unreported": 0,
            "by_phase": {}, "cap": 5000, "capped": True})
        out = render_step_account(result)
        self.assertIn("cap hit", out)
        self.assertIn("5123", out)
        self.assertIn("5000", out)

    def test_uncapped_run_has_no_cap_line(self):
        result = _mixed_result(usage={
            "prompt_tokens": 10, "completion_tokens": 5,
            "total_tokens": 15, "calls": 1, "usage_unreported": 0,
            "by_phase": {}})
        self.assertNotIn("cap hit", render_step_account(result))

    def test_scope_denials_named(self):
        result = _mixed_result(scope_denials=[
            {"tool": "fs_write", "reason": "outside the role scope"},
            {"tool": "shell_run", "reason": "not in the allowlist"},
            {"tool": "fs_read", "reason": "path escapes the root"},
        ])
        out = render_step_account(result)
        self.assertIn("denied", out)
        self.assertIn("fs_write", out)
        self.assertIn("outside the role scope", out)
        self.assertIn("shell_run", out)
        self.assertIn("+1 more", out)  # the third denial is counted

    def test_review_unavailable_line(self):
        result = _mixed_result(review={
            "passed": False, "available": False, "rounds": 0,
            "objections": [], "depth": "focused",
            "notes": ["reviewer model unreachable"]})
        out = render_step_account(result)
        self.assertIn("review", out)
        self.assertIn("unavailable", out)
        self.assertIn("reviewer model unreachable", out)

    def test_review_failed_line_names_objections(self):
        result = _mixed_result(review={
            "passed": False, "available": True, "rounds": 1,
            "objections": ["the parser drops unknown keys silently"],
            "depth": "standard", "notes": []})
        out = render_step_account(result)
        self.assertIn("review", out)
        self.assertIn("the parser drops unknown keys silently", out)

    def test_clean_review_adds_no_line(self):
        result = _mixed_result(review={
            "passed": True, "available": True, "rounds": 1,
            "objections": [], "depth": "standard", "notes": []})
        self.assertNotIn("review", render_step_account(result))

    def test_long_plan_collapses_but_never_hides_non_done(self):
        steps = []
        for i in range(1, STEP_ACCOUNT_FULL_LIMIT + 3):
            steps.append(_step(f"s{i}", "done",
                               instructions=f"done step number {i}",
                               path=f"f{i}.py", verify=_verify_ok()))
        steps.append(_step("sbad", "failed", instructions="the bad step",
                           path="bad.py", reason="verify failed",
                           verify=_verify_failed(
                               ["tests/test_bad.py::test_bad"])))
        steps.append(_step("sskip", "skipped", instructions="the skipped",
                           path="skip.py", reason="dependency sbad failed",
                           verify=_verify_none(
                               "no isolated check ran for this step")))
        steps.append(_step("sunv", "done", instructions="unverified done",
                           path="unv.py", verify=_verify_none()))
        total = len(steps)
        self.assertGreater(total, STEP_ACCOUNT_FULL_LIMIT)
        out = render_step_account(CodingResult(
            ok=False, iterations=3, steps=steps))
        # counts tell the whole story up front
        self.assertIn(f"{total - 2} done", out)
        self.assertIn("1 failed", out)
        self.assertIn("1 skipped", out)
        # every non-done / unverified step is individually named
        self.assertIn("sbad", out)
        self.assertIn("tests/test_bad.py::test_bad", out)
        self.assertIn("sskip", out)
        self.assertIn("sunv", out)
        # the verified done steps collapse into one summary line
        self.assertIn("more done steps", out)
        self.assertNotIn("done step number 1", out)
        self.assertNotIn(
            f"done step number {STEP_ACCOUNT_FULL_LIMIT + 2}", out)

    def test_replans_noted_in_header(self):
        out = render_step_account(_mixed_result(replans=1))
        self.assertIn("re-plan", out)

    def test_stepsless_result_renders_empty(self):
        self.assertEqual(render_step_account(
            CodingResult(ok=True, iterations=1)), "")
        self.assertEqual(render_step_account({}), "")
        self.assertEqual(render_step_account(None), "")
        # steps present but empty list -> still the legacy rendering
        self.assertEqual(render_step_account(
            CodingResult(ok=True, iterations=1, steps=[])), "")

    def test_dict_form_renders_like_the_object(self):
        result = _mixed_result()
        self.assertEqual(render_step_account(result.to_dict()),
                         render_step_account(result))


# ── integration: the real chat /code surface ─────────────────────────

class _FakeCodingAgent:
    result = None

    def __init__(self, context, **kwargs):
        self.context = context

    def run(self, task, **kwargs):
        return _FakeCodingAgent.result


class _StubGateway:
    def __init__(self):
        self.sent = []

    def send(self, platform, chat, text):
        self.sent.append(text)


class ChatCodeSurfaceTests(unittest.TestCase):

    def _drive(self, result):
        from nomorals.agents.partner.runtime_build import RuntimeBuildMixin

        class Stub(RuntimeBuildMixin):
            def __init__(self):
                self.context = SimpleNamespace()
                self.gateway = _StubGateway()

            def _ref_from_key(self, chat_key):
                return SimpleNamespace(platform="telegram", chat_id="1")

            def _send_long_checked(self, platform, chat, text):
                return text

        _FakeCodingAgent.result = result
        with mock.patch("nomorals.agents.coding.CodingAgent",
                        _FakeCodingAgent):
            return Stub()._control_code("build the thing",
                                        chat_key="telegram:1")

    def test_chat_code_failure_shows_the_step_account(self):
        text = self._drive(_mixed_result())
        self.assertTrue(text.startswith("❌ coding bot gave up"), text)
        self.assertIn("s4", text)
        self.assertIn("tests/test_report.py::test_title", text)
        self.assertIn("dependency s4 failed", text)

    def test_chat_code_success_shows_the_step_account(self):
        steps = [
            _step("s1", "done", instructions="add the config parser",
                  path="calc.py", verify=_verify_ok()),
            _step("s2", "done", instructions="wire the parser into main",
                  path="main.py", verify=_verify_ok()),
        ]
        text = self._drive(CodingResult(
            ok=True, iterations=2, files=["calc.py", "main.py"],
            output="ran fine", steps=steps))
        self.assertTrue(text.startswith("✅ code done"), text)
        self.assertIn("s1", text)
        self.assertIn("add the config parser", text)

    def test_chat_code_stepsless_result_is_byte_identical_to_legacy(self):
        result = CodingResult(ok=True, iterations=1, files=["a.py"],
                              output="hello")
        text = self._drive(result)
        self.assertTrue(text.startswith(
            "✅ code done in 1 iteration(s),"), text)
        self.assertIn("— a.py\nrun output:\nhello", text)
        self.assertNotIn("steps:", text)


# ── integration: mission settlement -> /mission results ─────────────

class MissionResultsSurfaceTests(unittest.TestCase):

    def test_settlement_carries_the_account_and_results_render_it(self):
        from nomorals.agents.roles import MissionCodingAgent
        from nomorals.missions.progress import render_results_text

        settlement = MissionCodingAgent._settle("build the thing",
                                                _mixed_result())
        self.assertIn("step_account", settlement)
        self.assertIn("s4", settlement["step_account"])

        mission = SimpleNamespace(
            name="build-it", status="failed", terminal=True,
            state={"outputs": {"build": settlement}})
        text = render_results_text(mission)
        self.assertIn("s4", text)
        self.assertIn("tests/test_report.py::test_title", text)
        self.assertIn("dependency s4 failed", text)
        # the loop's own output still shows (the account leads, the
        # evidence is not destroyed)
        self.assertIn("some output", text)

    def test_stepsless_settlement_has_no_account_key(self):
        from nomorals.agents.roles import MissionCodingAgent

        settlement = MissionCodingAgent._settle(
            "tiny", CodingResult(ok=True, iterations=1, output="ok"))
        self.assertNotIn("step_account", settlement)


# ── integration: the nm code CLI ─────────────────────────────────────

class CliCodeSurfaceTests(unittest.TestCase):

    def test_nm_code_run_prints_the_step_account(self):
        from nomorals.cmdline.commands.code import _cmd_code_run

        _FakeCodingAgent.result = _mixed_result()
        args = SimpleNamespace(file=None, accept=None, max_iters=5,
                               plan_mode="never", root=".", json=False)
        buf = io.StringIO()
        with mock.patch("nomorals.agents.coding.CodingAgent",
                        _FakeCodingAgent), \
                contextlib.redirect_stdout(buf):
            rc = _cmd_code_run(args, SimpleNamespace(), "build the thing")
        out = buf.getvalue()
        self.assertEqual(rc, 1)  # the fabricated run failed
        self.assertIn("s4", out)
        self.assertIn("tests/test_report.py::test_title", out)
        self.assertIn("FAILED: run failed", out)


if __name__ == "__main__":
    unittest.main()
