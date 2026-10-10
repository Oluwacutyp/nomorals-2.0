"""Sweep tests for nomorals/tools/ — the 2026-10-10 tools sweep delta.

Covers the new/changed behavior introduced by the sweep:
registry (annotations, exact schemas, idempotency, close-name errors),
edit_loop (flexible matching, SEARCH/REPLACE blocks, laziness detection),
code_executor (plan-and-execute replanning, loop detection),
captcha (SolveBudget, AutoPipelineBackend), osint (CT subdomain sources,
passive/active modes), proxylab (response classification, identity,
subnet tracking, tiers), web (extraction ladder, markdown),
repo_index (render_repo_map), macros (checkpoints, export/import, continue).
"""

import json
import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.tools.registry import ToolRegistry, ToolSpec, ToolError


# ── fakes ────────────────────────────────────────────────────────────────


class FakeContext:
    def __init__(self):
        self.settings = SimpleNamespace(
            workspace_dir=tempfile.mkdtemp(prefix="tools_sweep_"))


class FakeOutcome:
    def __init__(self, ok=True, value=None, error=None):
        self.ok = ok
        self.value = value
        self.error = error


class FakeRegistry:
    def __init__(self):
        self.context = FakeContext()
        self.tools: dict = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = (fn, kwargs)
            return fn
        return deco

    def unregister(self, name):
        self.tools.pop(name, None)

    def names(self):
        return set(self.tools)

    def call(self, name, actor=None, **kwargs):
        if name not in self.tools:
            raise ToolError(f"unknown tool {name!r}")
        fn, _ = self.tools[name]
        return FakeOutcome(ok=True, value=fn(**kwargs))


# ── registry ─────────────────────────────────────────────────────────────


class RegistrySweepTests(unittest.TestCase):
    def test_annotations_defaults_per_kind(self):
        spec = ToolSpec(name="t", fn=lambda: None, kind="query")
        ann = spec.resolved_annotations()
        self.assertTrue(ann["read_only"])
        self.assertFalse(ann["destructive"])
        spec2 = ToolSpec(name="t", fn=lambda: None, kind="delete")
        ann2 = spec2.resolved_annotations()
        self.assertTrue(ann2["destructive"])
        self.assertFalse(ann2["idempotent"])

    def test_schema_is_exact(self):
        spec = ToolSpec(name="t", fn=lambda: None,
                        parameters={"a": "str", "b": "int (optional)"})
        spec = ToolSpec(name="t", fn=lambda: None,
                        parameters={"a": {"type": "str", "required": True},
                                    "b": {"type": "int"}})
        schema = spec.schema()
        self.assertEqual(schema["parameters"]["required"], ["a"])
        self.assertFalse(schema["parameters"]["additionalProperties"])

    def test_title_and_examples(self):
        spec = ToolSpec(name="t", fn=lambda: None, title="My Tool",
                        examples=['{"a": "1"}'])
        self.assertEqual(spec.title, "My Tool")
        self.assertEqual(spec.examples, ['{"a": "1"}'])

    def test_close_name_suggestion(self):
        reg = ToolRegistry(FakeContext())
        reg.register("send_email", description="x",
                     parameters={})(lambda: None)
        outcome = reg.call("send_emial", actor="t")
        self.assertFalse(outcome.ok)
        self.assertIn("send_email", str(outcome.error))

    def test_idempotency_key_replays_result(self):
        reg = ToolRegistry(FakeContext())
        calls = []
        reg.register("flip", description="x", parameters={})(lambda: calls.append(1))
        reg.call("flip", actor="t", idempotency_key="k1")
        reg.call("flip", actor="t", idempotency_key="k1")
        self.assertEqual(len(calls), 1)
        reg.call("flip", actor="t", idempotency_key="k2")
        self.assertEqual(len(calls), 2)


# ── edit_loop ────────────────────────────────────────────────────────────


class EditLoopSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="edit_sweep_")

    def _f(self, name, text):
        path = os.path.join(self.tmp, name)
        with open(path, "w") as fh:
            fh.write(text)
        return path

    def test_flexible_match_ignores_whitespace(self):
        from nomorals.tools.edit_loop import _locate_flexible
        text = "def f():\n    x   =   1\n    return x\n"
        start, end, strategy = _locate_flexible(text, "x = 1")
        self.assertEqual(strategy, "whitespace")
        self.assertIn("x", text[start:end])

    def test_flexible_match_exact_first(self):
        from nomorals.tools.edit_loop import _locate_flexible
        start, end, strategy = _locate_flexible("a = 1\n", "a = 1")
        self.assertEqual(strategy, "exact")

    def test_parse_blocks(self):
        from nomorals.tools.edit_loop import parse_search_replace_blocks
        text = ("a.py\n```\n<<<<<<< SEARCH\nold\n=======\nnew\n>>>>>>> REPLACE\n"
                "```\nnew.py\n```\n<<<<<<< SEARCH\n=======\nbrand new file\n"
                ">>>>>>> REPLACE\n```")
        blocks = parse_search_replace_blocks(text)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0]["path"], "a.py")
        self.assertEqual(blocks[0]["search"].strip(), "old")
        self.assertEqual(blocks[1]["search"], "")

    def test_apply_blocks_creates_file(self):
        from nomorals.tools.edit_loop import apply_search_replace_blocks
        text = ("new.py\n```\n<<<<<<< SEARCH\n=======\nprint('hi')\n"
                ">>>>>>> REPLACE\n```")
        result = apply_search_replace_blocks(text, self.tmp)
        self.assertIn("new.py", result["changed"])
        with open(os.path.join(self.tmp, "new.py")) as fh:
            self.assertIn("print('hi')", fh.read())

    def test_apply_blocks_atomic_per_file(self):
        from nomorals.tools.edit_loop import (
            apply_search_replace_blocks, EditConflictError)
        self._f("b.py", "original = True\n")
        text = ("b.py\n```\n<<<<<<< SEARCH\noriginal = True\n=======\n"
                "changed = True\n>>>>>>> REPLACE\n```\n"
                "b.py\n```\n<<<<<<< SEARCH\nmissing = 1\n=======\nx\n"
                ">>>>>>> REPLACE\n```")
        with self.assertRaises(EditConflictError):
            apply_search_replace_blocks(text, self.tmp)
        with open(os.path.join(self.tmp, "b.py")) as fh:
            self.assertEqual(fh.read(), "original = True\n")

    def test_laziness_rejected(self):
        from nomorals.tools.edit_loop import detect_placeholders
        bad = detect_placeholders("# ... rest of the code\nx = 1\n...\n")
        self.assertTrue(bad)
        self.assertFalse(detect_placeholders("x = 1\nreturn x\n"))


# ── code_executor ────────────────────────────────────────────────────────


class FakeAgent:
    def __init__(self, replies):
        self.replies = list(replies)

    def ask(self, prompt):
        return self.replies.pop(0)


class CodeExecutorSweepTests(unittest.TestCase):
    def _agent(self, replies):
        class _A:
            def __init__(self, rs):
                self.rs = list(rs)

            def chat(self, prompt):
                return SimpleNamespace(content=self.rs.pop(0))
        return _A(replies)

    def _step_result(self, step, ok, error=""):
        from nomorals.tools.code_executor import StepResult
        return StepResult(step_id=step.step_id, action=step.action,
                          target=step.target, description=step.description,
                          success=ok, error=error)

    def _run(self, ex, plan, **kw):
        import asyncio
        return asyncio.run(ex.execute_plan(plan, **kw))

    def test_replan_recovers_and_succeeds(self):
        from unittest.mock import AsyncMock
        from nomorals.tools.code_executor import (
            CodeExecutor, PlanStep, ExecutionPlan)
        step = PlanStep(step_id="s1", action="run_command",
                        target="false", description="fails")
        plan = ExecutionPlan(plan_id="p1", goal="demo", summary="s", steps=[step])
        agent = self._agent([json.dumps({
            "action": "replan", "reasoning": "try a working command",
            "revised_steps": [{"action": "run_command", "target": "true",
                               "description": "works",
                               "estimated_risk": "low"}]})])
        ex = CodeExecutor(agent)
        ex._execute_step = AsyncMock(side_effect=[
            self._step_result(step, False, "exit 1"),
            self._step_result(step, True),
        ])
        result = self._run(ex, plan)
        self.assertTrue(result.ok if hasattr(result, "ok") else result.success)
        self.assertTrue(result.success)
        self.assertEqual(result.replans, 1)

    def test_loop_detection_stops(self):
        from unittest.mock import AsyncMock
        from nomorals.tools.code_executor import (
            CodeExecutor, PlanStep, ExecutionPlan)
        step = PlanStep(step_id="s1", action="run_command",
                        target="false", description="always fails")
        plan = ExecutionPlan(plan_id="p2", goal="loop", summary="s", steps=[step])
        same = {"action": "run_command", "target": "false",
                "description": "retry", "estimated_risk": "low"}
        agent = self._agent([json.dumps(
            {"action": "replan", "reasoning": "retry", "revised_steps": [same]}
        )] * 6)
        ex = CodeExecutor(agent)

        async def fail(s, p):
            return self._step_result(s, False, "boom")
        ex._execute_step = AsyncMock(side_effect=fail)
        result = self._run(ex, plan, max_replans=6)
        self.assertFalse(result.success)
        self.assertIn("loop", (result.error or "").lower())

    def test_final_failure_marks_plan_failed(self):
        from unittest.mock import AsyncMock
        from nomorals.tools.code_executor import (
            CodeExecutor, PlanStep, ExecutionPlan)
        step = PlanStep(step_id="s1", action="run_command",
                        target="false", description="fails")
        plan = ExecutionPlan(plan_id="p3", goal="x", summary="s", steps=[step])
        agent = self._agent([json.dumps(
            {"action": "finish", "reasoning": "unreachable",
             "final_answer": "giving up"})])

        async def fail(s, p):
            return self._step_result(s, False, "boom")
        ex = CodeExecutor(agent)
        ex._execute_step = AsyncMock(side_effect=fail)
        result = self._run(ex, plan)
        self.assertFalse(result.success)
        self.assertIn("giving up", result.error or "")


# ── captcha ──────────────────────────────────────────────────────────────


class CaptchaSweepTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="captcha_budget_")
        self._patcher = mock.patch.dict(
            os.environ, {"NOMORALS_CAPTCHA_DIR": self.tmpdir})
        self._patcher.start()
        self.addCleanup(self._patcher.stop)
    def test_budget_default_deny(self):
        from nomorals.tools.captcha import SolveBudget
        bud = SolveBudget(daily_cap_usd=0.0, monthly_cap_usd=0.0)
        ok, reason = bud.can_spend()
        self.assertFalse(ok)
        self.assertIn("cap", reason.lower())

    def test_budget_caps_and_reset(self):
        from nomorals.tools.captcha import SolveBudget
        bud = SolveBudget(daily_cap_usd=0.01, monthly_cap_usd=1.0)
        self.assertTrue(bud.can_spend()[0])
        bud.record_spend(0.02, "t1", "turnstile")
        self.assertFalse(bud.can_spend()[0])
        st = bud.budget_status()
        self.assertGreater(st["daily_spent_usd"], 0)
        self.assertIn("daily_remaining_usd", st)

    def test_pipeline_falls_back_to_takeover_without_key(self):
        from nomorals.tools.captcha import (
            backend_for, CaptchaChallenge, CaptchaKind)
        b = backend_for("pipeline")
        self.assertEqual(b.name, "pipeline")
        ch = CaptchaChallenge(kind=CaptchaKind.TURNSTILE, sitekey="k",
                              page_url="https://example.com/")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CAPTCHA_API_KEY", None)
            result = b.solve(ch)
        self.assertTrue(result.takeover)
        self.assertIn("layer1", result.detail)

    def test_pipeline_registered(self):
        from nomorals.tools.captcha import _BACKENDS
        self.assertIn("pipeline", _BACKENDS)

    def test_report_solve_methods(self):
        from nomorals.tools.captcha import ServiceBackend, SolveBudget
        be = ServiceBackend(settings=None, api_key="k")
        self.assertTrue(hasattr(be, "report_solve"))


# ── osint ────────────────────────────────────────────────────────────────


class OsintSweepTests(unittest.TestCase):
    def test_crtsh_parses_subdomains(self):
        from nomorals.tools import osint
        payload = [{"name_value": "a.example.com\nexample.com"},
                   {"name_value": "*.example.com"}]
        resp = SimpleNamespace(ok=True, json=lambda: payload)
        with mock.patch.object(osint, "HttpClient") as hc:
            hc.return_value.get.return_value = resp
            out = osint._crtsh_subdomains("example.com", 5.0)
        self.assertIn("a.example.com", out["subdomains"])
        self.assertNotIn("*.example.com", out["subdomains"])

    def test_certspotter_parses(self):
        from nomorals.tools import osint
        payload = [{"dns_names": ["x.example.com", "*.example.com"]}]
        resp = SimpleNamespace(ok=True, json=lambda: payload)
        with mock.patch.object(osint, "HttpClient") as hc:
            hc.return_value.get.return_value = resp
            out = osint._certspotter_subdomains("example.com", 5.0)
        self.assertEqual(out["subdomains"], ["x.example.com"])

    def test_rapiddns_parses_html(self):
        from nomorals.tools import osint
        html = '<table><tr><td>sub.example.com</td></tr></table>'
        resp = SimpleNamespace(ok=True, text=html)
        with mock.patch.object(osint, "HttpClient") as hc:
            hc.return_value.get.return_value = resp
            out = osint._rapiddns_subdomains("example.com", 5.0)
        self.assertEqual(out["subdomains"], ["sub.example.com"])

    def test_sweep_rejects_bad_mode(self):
        from nomorals.tools import osint
        ctx = SimpleNamespace(settings=SimpleNamespace(
            osint_timeout=1.0, osint_abuse_key="", osint_shodan_key="",
            osint_history_days=30))
        with self.assertRaises(Exception):
            osint.osint_sweep(ctx, "example.com", mode="nuclear")

    def test_verify_hosts_live(self):
        from nomorals.tools import osint
        with mock.patch("nomorals.tools.network.dns_query") as dq:
            dq.side_effect = [["1.2.3.4"], Exception("nx")]
            out = osint._verify_hosts_live(["a.example.com", "b.example.com"],
                                           5.0)
        self.assertEqual(out["live_count"], 1)
        self.assertEqual(out["dead_count"], 1)


# ── proxylab ─────────────────────────────────────────────────────────────


class ProxylabSweepTests(unittest.TestCase):
    def test_classify_response(self):
        from nomorals.tools.proxylab import classify_response
        self.assertEqual(classify_response(429)[0], "throttle")
        self.assertEqual(classify_response(403, "cloudflare blocked")[0],
                         "block")
        self.assertEqual(classify_response(200, "g-recaptcha here")[0],
                         "challenge")
        self.assertEqual(classify_response(200, "normal")[0], "data")

    def test_retry_after(self):
        from nomorals.tools.proxylab import retry_after_delay
        self.assertEqual(retry_after_delay({"Retry-After": "120"}), 120.0)
        self.assertEqual(retry_after_delay({}), 60.0)
        self.assertEqual(retry_after_delay({"retry-after": "bad"}), 60.0)

    def test_identity_binding(self):
        from nomorals.tools.proxylab import ProxyIdentity
        ident = ProxyIdentity(proxy_url="http://h:8080", user_agent="UA",
                              cookies={"s": "1"})
        self.assertTrue(ident.session_id)
        headers = ident.headers()
        self.assertEqual(headers["User-Agent"], "UA")
        self.assertIn("s=1", headers["Cookie"])

    def test_subnet_tracker_burn(self):
        from nomorals.tools.proxylab import SubnetTracker
        tr = SubnetTracker(burn_threshold=0.5, min_samples=3)
        for _ in range(4):
            tr.note("10.0.0.5", False)
        for _ in range(4):
            tr.note("10.9.9.9", True)
        burned = tr.burned_subnets()
        self.assertEqual(len(burned), 1)
        self.assertTrue(burned[0]["subnet"].startswith("10.0.0."))

    def test_pool_tiers(self):
        from nomorals.tools.proxylab import pool_tiers, Proxy
        now = time.time()
        gold = Proxy(host="a", port=1, alive=True, anonymity="elite")
        gold.last_success = now
        gold.latency_ms = 50
        dead = Proxy(host="b", port=1, alive=False)
        cooling = Proxy(host="c", port=1, alive=True)
        cooling.backoff_until = now + 600
        tiers = pool_tiers([gold, dead, cooling], at=now)
        self.assertIn(gold.url, tiers["gold"])
        self.assertIn(dead.url, tiers["dead"])
        self.assertIn(cooling.url, tiers["cooling"])


# ── web ──────────────────────────────────────────────────────────────────


class WebSweepTests(unittest.TestCase):
    def test_extraction_ladder_names_engine(self):
        from nomorals.tools.web import readability_extract
        r = readability_extract("<html><body>short</body></html>")
        self.assertIn(r["engine"], ("builtin", "trafilatura"))
        self.assertIn("confidence", r)

    def test_markdown_output(self):
        from nomorals.tools.web import readability_extract
        long_p = ("word " * 200)
        html = (f"<html><head><title>T</title></head><body>"
                f"<article><h2>H</h2><p>{long_p}</p></article></body></html>")
        r = readability_extract(html, output_format="markdown")
        self.assertIn("#", r["text"])
        self.assertEqual(r["confidence"], "medium")

    def test_html_to_markdown_fallback(self):
        from nomorals.tools.web import _html_to_markdown
        out = _html_to_markdown(
            '<h2>T</h2><p>see <a href="http://x">y</a></p>')
        self.assertIn("## T", out)
        self.assertIn("[y](http://x)", out)

    def test_trafilatura_absent_falls_back(self):
        from nomorals.tools.web import _trafilatura_extract
        with mock.patch.dict("sys.modules", {"trafilatura": None}):
            self.assertIsNone(_trafilatura_extract("<p>x</p>"))


# ── repo_index ───────────────────────────────────────────────────────────


class RepoMapSweepTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="repomap_")
        files = {
            "a.py": '"""A."""\ndef alpha(x, y):\n    """Add."""\n    return x\n',
            "b.py": "import a\n\n\nclass Beta:\n    def run(self):\n        return a.alpha(1, 2)\n",
        }
        for name, text in files.items():
            with open(os.path.join(self.root, name), "w") as fh:
                fh.write(text)

    def test_render_repo_map_budgeted(self):
        from nomorals.tools.repo_index import render_repo_map
        out = render_repo_map(self.root, max_tokens=100)
        self.assertIn("repo map", out)
        self.assertIn("a.py", out)
        self.assertLessEqual(len(out), 100 * 4 + 500)

    def test_render_repo_map_signatures(self):
        from nomorals.tools.repo_index import render_repo_map
        out = render_repo_map(self.root, max_tokens=500)
        self.assertIn("def alpha(x, y):", out)
        self.assertIn("class Beta:", out)

    def test_focus_boost_mentions_focus(self):
        from nomorals.tools.repo_index import render_repo_map
        out = render_repo_map(self.root, max_tokens=500,
                              focus_files=["a.py"])
        self.assertIn("focused", out)


# ── macros ───────────────────────────────────────────────────────────────


class MacrosSweepTests(unittest.TestCase):
    def setUp(self):
        import sqlite3
        self.db_path = os.path.join(tempfile.mkdtemp(), "m.db")
        conn = sqlite3.connect(self.db_path)
        conn.execute("CREATE TABLE macros (id TEXT PRIMARY KEY, name TEXT, "
                     "description TEXT, steps TEXT, created_at REAL, runs INT, "
                     "last_run REAL, last_result TEXT)")
        conn.commit()
        conn.close()
        from nomorals.tools import macros
        self.macros = macros
        self.reg = FakeRegistry()
        self.reg.tools["echo_tool"] = (lambda **kw: "echo", {})

        class DB:
            def __init__(self, path):
                self.path = path

            def _c(self):
                import sqlite3 as s
                c = s.connect(self.path)
                c.row_factory = s.Row
                return c

            def query_one(self, q, params=()):
                c = self._c()
                try:
                    return c.execute(q, params).fetchone()
                finally:
                    c.close()

            def execute(self, q, params=()):
                c = self._c()
                try:
                    c.execute(q, params)
                    c.commit()
                finally:
                    c.close()

            def transaction(self):
                from contextlib import contextmanager

                @contextmanager
                def _t():
                    yield self
                return _t()

        self.ctx = SimpleNamespace(db=DB(self.db_path),
                                   settings=SimpleNamespace())

    def test_checkpoint_replay_verified(self):
        m = self.macros
        m.record_start("cp1")
        m.record_checkpoint("page says hi", tool="echo_tool",
                            expect_contains="hi")
        m.record_stop(registry=self.reg, context=self.ctx)
        self.reg.tools["echo_tool"] = (lambda: "well hi there", {})
        res = m.run_macro(self.ctx, self.reg, "cp1")
        self.assertTrue(res["ok"])
        self.assertTrue(any(r["tool"] == "__checkpoint__"
                            for r in res["steps"]))

    def test_checkpoint_failure_stops_run(self):
        m = self.macros
        m.record_start("cp2")
        m.record_checkpoint("says bye", tool="echo_tool",
                            expect_contains="bye")
        m.record_stop(registry=self.reg, context=self.ctx)
        self.reg.tools["echo_tool"] = (lambda: "hello", {})
        res = m.run_macro(self.ctx, self.reg, "cp2")
        self.assertFalse(res["ok"])

    def test_export_import_roundtrip(self):
        m = self.macros
        m.record_start("exp1", "demo")
        m.record_step("echo_tool", {})
        m.record_stop(registry=self.reg, context=self.ctx)
        exported = m.export_macro(self.ctx, "exp1")
        self.assertEqual(exported["format"], "json")
        m.delete_macro(self.ctx, "exp1")
        imported = m.import_macro(self.ctx, self.reg, exported["text"])
        self.assertEqual(imported["saved"], "exp1")
        self.assertEqual(imported["steps"], 1)

    def test_record_continue_prefixes_steps(self):
        m = self.macros
        m.record_start("cont1")
        m.record_step("echo_tool", {"a": 1})
        m.record_stop(registry=self.reg, context=self.ctx)
        out = m.record_continue(self.ctx, "cont1")
        self.assertEqual(out["steps"], 1)
        m.record_step("echo_tool", {"a": 2})
        m.record_stop(registry=self.reg, context=self.ctx)
        shown = m.show_macro(self.ctx, "cont1")
        self.assertEqual(len(shown["steps"]), 2)

    def test_idle_auto_stop(self):
        m = self.macros
        m.record_start("idle1", idle_timeout_s=60)
        m.record_step("echo_tool", {})
        with m._recorder_lock:
            m._active["current"]["last_step_at"] = time.time() - 5000
        fired = m._auto_stop_if_idle(registry=self.reg, context=self.ctx)
        self.assertTrue(fired and fired.get("auto_stopped"))


if __name__ == "__main__":
    unittest.main()
