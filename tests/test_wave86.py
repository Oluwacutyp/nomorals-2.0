"""Wave 86 — profile-aware runtime, browser v2, research swarm, command
center.

Hermetic: fake context/tools, canned HTML, canned search results. The
network is only touched when a test says so (none do).
"""

from __future__ import annotations

import dataclasses
import http.cookiejar
import json
import random
import tempfile
import time
import unittest
from types import SimpleNamespace

from nomorals.agents.context import build_context
from nomorals.core.config import Settings, load_settings
from nomorals.core.runtune import RuntimeTune, build_tune
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.tools.browser import BrowserSession, parse_html
from nomorals.workspace.profile import EnvironmentProfile

TERMUX = EnvironmentProfile(kind="termux", min_vcpus=1, target_vcpus=2,
                            max_vcpus=3, cpu=8, memory_mb=3072,
                            detail="test phone", detected=True)
VPS = EnvironmentProfile(kind="vps", min_vcpus=2, target_vcpus=4,
                         max_vcpus=8, cpu=2, memory_mb=3072,
                         detail="test box", detected=True)
WORKSTATION = EnvironmentProfile(kind="workstation", min_vcpus=4,
                                 target_vcpus=8, max_vcpus=16, cpu=32,
                                 memory_mb=131072, detail="test tower",
                                 detected=True)


class RuntuneTests(unittest.TestCase):
    def test_termux_gets_phone_tuning(self) -> None:
        t = build_tune(Settings(), profile=TERMUX)
        self.assertFalse(t.use_processes, "no fork() on Android")
        self.assertEqual(t.max_parallel_chats, 2)
        self.assertLessEqual(t.max_concurrent_downloads, 1)
        self.assertLessEqual(t.max_download_mb, 25.0)
        self.assertEqual(t.memory_pressure, "aggressive")
        self.assertEqual(t.model_pref, "small_local")
        self.assertLessEqual(t.mission_max_concurrent, 1)
        # vcpus pinned to the profile envelope
        self.assertEqual((t.vcpu_min, t.vcpu_target, t.vcpu_max), (1, 2, 3))

    def test_workstation_gets_desktop_tuning(self) -> None:
        t = build_tune(Settings(), profile=WORKSTATION)
        self.assertTrue(t.use_processes)
        self.assertGreaterEqual(t.threads, 16)
        self.assertGreater(t.max_download_mb, 300.0)
        self.assertEqual((t.vcpu_min, t.vcpu_target, t.vcpu_max), (4, 8, 16))
        self.assertGreaterEqual(t.mission_max_concurrent, 4)

    def test_explicit_runtime_override_wins(self) -> None:
        s = dataclasses.replace(Settings(),
                                runtime=dataclasses.replace(
                                    Settings().runtime, threads=3,
                                    max_download_mb=77.0))
        t = build_tune(s, profile=VPS)
        self.assertEqual(t.threads, 3)
        self.assertEqual(t.max_download_mb, 77.0)
        self.assertTrue(any("explicit runtime.threads" in n for n in t.notes))

    def test_named_profile_wins_over_detection(self) -> None:
        tmp = tempfile.mkdtemp()
        s = load_settings(overrides={"home": tmp}, env={"NM_PROFILE": "termux"})
        t = build_tune(s, profile=WORKSTATION)
        # the named termux profile set these deliberately
        self.assertEqual(t.max_parallel_chats, 2)
        self.assertEqual(t.context_budget_tokens, 4000)
        self.assertFalse(t.use_processes)
        # ...including its VCPU envelope, which pins the farm to phone-sized
        self.assertEqual((t.vcpu_min, t.vcpu_target, t.vcpu_max), (1, 2, 3))

    def test_provenance_and_json(self) -> None:
        t = build_tune(Settings(), profile=VPS)
        self.assertTrue(all("=" in n for n in t.notes))
        json.dumps(t.to_dict())  # must be serializable for --json

    def test_memory_pressure_follows_ram(self) -> None:
        tiny = dataclasses.replace(VPS, memory_mb=1500)
        self.assertEqual(build_tune(Settings(), profile=tiny).memory_pressure, "aggressive")
        big = dataclasses.replace(VPS, memory_mb=32 * 1024)
        self.assertEqual(build_tune(Settings(), profile=big).memory_pressure, "relaxed")


class ContextTuneTests(unittest.TestCase):
    def test_context_carries_tune(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-w86-")
        settings = load_settings(overrides={"home": tmp.name,
                                            "partner.platforms": "local",
                                            "chat.local_enabled": "true"})
        context = build_context(settings, with_executor=True, with_tools=False)
        try:
            tune = context.extras.get("tune")
            self.assertIsInstance(tune, RuntimeTune)
            # the executor follows the tune
            self.assertEqual(context.executor.threads, tune.threads)
        finally:
            try:
                context.close()
            except Exception:  # noqa: BLE001
                pass
            tmp.cleanup()


class PartnerPoolTuneTests(unittest.TestCase):
    def _runtime(self, **overrides):
        from nomorals.agents.partner_runtime import PartnerRuntime

        tmp = tempfile.TemporaryDirectory(prefix="nm-w86-")
        settings = load_settings(
            overrides={"home": tmp.name, "partner.platforms": "local",
                       "chat.local_enabled": "true", **overrides})
        context = build_context(settings, with_executor=False, with_tools=False)
        self._context = context
        self._tmp = tmp
        from tests.test_w85c import RecordingAdapter, _ScriptedRouter

        context.router = _ScriptedRouter()
        gateway = None
        runtime = PartnerRuntime(context, gateway=gateway)
        return runtime

    def tearDown(self) -> None:
        try:
            self._context.close()
        except Exception:  # noqa: BLE001
            pass
        self._tmp.cleanup()

    def test_pool_follows_tune(self) -> None:
        runtime = self._runtime()
        try:
            tune = runtime.context.extras["tune"]
            self.assertEqual(runtime._pool._max_workers, tune.max_parallel_chats)
        finally:
            runtime.stop()

    def test_explicit_chat_cap_wins(self) -> None:
        runtime = self._runtime(**{"partner.max_parallel_chats": "5"})
        try:
            self.assertEqual(runtime._pool._max_workers, 5)
        finally:
            runtime.stop()


class AutonomyCapsTuneTests(unittest.TestCase):
    """Profile aggressiveness scales the daily proactive-volume caps."""

    def _stub(self, tune, dm=6, group=2):
        from nomorals.agents.partner_runtime import PartnerRuntime

        partner = dataclasses.replace(
            Settings().partner,
            max_proactive_dm_per_day=dm, max_group_posts_per_day=group)
        stub = SimpleNamespace(
            settings=SimpleNamespace(partner=partner),
            context=SimpleNamespace(extras={"tune": tune}),
        )
        return stub, PartnerRuntime._tuned_autonomy_caps

    def test_phone_halves_the_defaults(self) -> None:
        tune = build_tune(Settings(), profile=TERMUX)
        stub, fn = self._stub(tune)
        self.assertEqual(fn(stub), (3, 1))  # ceil(6*0.5), ceil(2*0.5)

    def test_full_power_keeps_defaults(self) -> None:
        tune = build_tune(Settings(), profile=WORKSTATION)
        stub, fn = self._stub(tune)
        self.assertEqual(fn(stub), (6, 2))

    def test_explicit_zero_unlimited_never_scaled(self) -> None:
        tune = build_tune(Settings(), profile=TERMUX)
        stub, fn = self._stub(tune, dm=0, group=2)
        self.assertEqual(fn(stub), (0, 1))

    def test_non_default_caps_respected_as_is(self) -> None:
        tune = build_tune(Settings(), profile=TERMUX)
        stub, fn = self._stub(tune, dm=10, group=2)
        self.assertEqual(fn(stub), (10, 1))


PAGE = """
<html><head>
<title>Test Page</title>
<meta name="description" content="a description about widgets">
<meta property="og:title" content="Widget News">
<link rel="canonical" href="https://example.com/canon">
</head><body>
<h1>Big Widget Story</h1>
<h2>Sub widget details</h2>
<p>Some text about the widget economy and its future.</p>
<table><tr><th>Name</th><th>Price</th></tr>
<tr><td>widget-a</td><td>5</td></tr><tr><td>widget-b</td><td>9</td></tr></table>
<form id="f0" action="/search" method="post">
  <input type="text" name="q" placeholder="query">
  <input type="submit" name="go" value="Go">
</form>
<a href="/next">next page</a>
<a href="https://other.example/away">away</a>
</body></html>
"""


class BrowserV2Tests(unittest.TestCase):
    def _session(self, **kw) -> BrowserSession:
        s = BrowserSession(name=f"t{time.time_ns()}", **kw)
        s.url = "https://example.com/page"
        s._raw = PAGE
        s.dom = parse_html(PAGE)
        s.title = "Test Page"
        return s

    # ── structured extraction ──────────────────────────────────────────────
    def test_extract_headings(self) -> None:
        out = self._session().do("extract", kind="headings")
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["items"][0], {"level": 1, "text": "Big Widget Story"})
        self.assertEqual(out["items"][1]["level"], 2)

    def test_extract_tables(self) -> None:
        out = self._session().do("extract", kind="tables")
        self.assertEqual(out["count"], 1)
        table = out["tables"][0]
        self.assertEqual(table["headers"], ["Name", "Price"])
        self.assertEqual(table["rows"], [["widget-a", "5"], ["widget-b", "9"]])
        self.assertEqual(table["row_count"], 3)

    def test_extract_forms(self) -> None:
        out = self._session().do("extract", kind="forms")
        self.assertEqual(out["count"], 1)
        form = out["forms"][0]
        self.assertEqual(form["id"], "f0")
        self.assertEqual(form["method"], "POST")
        names = {f["name"] for f in form["fields"]}
        self.assertIn("q", names)

    def test_extract_meta(self) -> None:
        out = self._session().do("extract", kind="meta")
        self.assertEqual(out["title"], "Test Page")
        self.assertEqual(out["description"], "a description about widgets")
        self.assertEqual(out["canonical"], "https://example.com/canon")
        self.assertEqual(out["og"]["og:title"], "Widget News")

    def test_extract_nav(self) -> None:
        out = self._session().do("extract", kind="nav")
        urls = [l["url"] for l in out["links"]]
        self.assertIn("https://example.com/next", urls)
        self.assertIn("https://other.example/away", urls)

    def test_unknown_extract_kind_errors(self) -> None:
        from nomorals.core.errors import ToolError

        with self.assertRaises(ToolError):
            self._session().do("extract", kind="nope")

    # ── multi-step task ────────────────────────────────────────────────────
    def test_task_runs_and_reports(self) -> None:
        out = self._session().do("task", steps=[
            {"act": "extract", "kind": "headings"},
            {"act": "wait", "seconds": 0.01},
            {"act": "stop", "note": "done"},
        ])
        self.assertTrue(out["ok"])
        self.assertEqual(out["steps_done"], 3)
        self.assertTrue(out["steps"][0]["ok"])
        self.assertEqual(out["steps"][0]["data"]["count"], 2)
        self.assertEqual(out["steps"][2]["note"], "done")

    def test_task_stops_on_error(self) -> None:
        out = self._session().do("task", steps=[
            {"act": "extract", "kind": "nope"},
            {"act": "stop"},
        ])
        self.assertFalse(out["ok"])
        self.assertEqual(out["steps_done"], 1)
        self.assertIn("unknown extract kind", out["steps"][0]["error"])

    def test_task_continues_when_allowed(self) -> None:
        out = self._session().do("task",
                                 steps=[{"act": "extract", "kind": "nope"},
                                        {"act": "extract", "kind": "meta"}],
                                 stop_on_error=False)
        self.assertTrue(out["ok"])
        self.assertFalse(out["steps"][0]["ok"])
        self.assertTrue(out["steps"][1]["ok"])

    def test_task_rejects_unknown_act_and_bad_shapes(self) -> None:
        from nomorals.core.errors import ToolError

        with self.assertRaises(ToolError):
            self._session().do("task", steps=[{"act": "fly"}])
        with self.assertRaises(ToolError):
            self._session().do("task", steps="not json")
        with self.assertRaises(ToolError):
            self._session().do("task", steps=[])

    def test_task_respects_profile_step_cap(self) -> None:
        from nomorals.core.errors import ToolError

        s = self._session()
        s.max_task_steps = 2
        with self.assertRaises(ToolError):
            s.do("task", steps=[{"act": "stop"}, {"act": "stop"}, {"act": "stop"}])
        # JSON-string steps work too
        out = s.do("task", steps=json.dumps([{"act": "stop"}]))
        self.assertTrue(out["ok"])

    # ── cookie persistence ─────────────────────────────────────────────────
    def test_cookies_persist_across_sessions(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="nm-w86-cookies-")
        try:
            name = f"persist{time.time_ns()}"
            s1 = BrowserSession(name=name, session_dir=tmp.name)
            # version, name, value, port, port_specified, domain,
            # domain_specified, domain_initial_dot, path, path_specified,
            # secure, expires, discard, comment, comment_url, rest, rfc2109
            c = http.cookiejar.Cookie(
                0, "sid", "abc123", None, False, "example.com", False, False,
                "/", False, True, None, False, None, None, {}, False)
            s1.cookie_jar.set_cookie(c)
            s1._save_cookies()
            s2 = BrowserSession(name=name, session_dir=tmp.name)
            self.assertEqual(len(s2.cookie_jar), 1)
            self.assertEqual(list(s2.cookie_jar)[0].value, "abc123")
        finally:
            tmp.cleanup()

    def test_no_session_dir_means_no_persistence(self) -> None:
        s = self._session()
        self.assertEqual(s._session_file(), "")
        s._save_cookies()  # must be a silent no-op


class _FakeTools:
    """Tool registry stand-in answering web_search with canned results."""

    def __init__(self, results):
        self._results = results

    def call(self, name, **kw):
        if name == "web_search":
            return SimpleNamespace(ok=True, value={"results": self._results},
                                   error=None,
                                   unwrap=lambda: {"results": self._results})
        return SimpleNamespace(ok=False, value=None,
                               unwrap=lambda: {},
                               error=SimpleNamespace(message="unknown tool"))


class _FakeContext:
    def __init__(self, results, memory=None):
        self.settings = Settings()
        self.tools = _FakeTools(results)
        self.router = None
        self.memory = memory
        self.extras = {"tune": build_tune(self.settings, profile=WORKSTATION)}


class _FakeMemory:
    def __init__(self):
        self.items = []

    def remember(self, content, *, kind="", importance=0.5, source=""):
        self.items.append((content, kind, source))
        return f"m{len(self.items)}"


def _results():
    return [
        {"url": "https://a.example/one", "title": "Widgets are booming",
         "snippet": "The widget economy is booming this year, and analysts "
                    "say widget production grew forty percent over last year.",
         "score": "0.9"},
        {"url": "https://b.example/two", "title": "Widget boom disputed",
         "snippet": "Contrary to the hype, the widget economy is not growing; "
                    "independent auditors found widget production fell, not rose.",
         "score": "0.8"},
        {"url": "https://c.example/three", "title": "How to build widgets",
         "snippet": "To build a widget you need a frame, a spring, and patience; "
                    "the practical guide walks through each part of a widget.",
         "score": "0.7"},
    ]


class ResearchSwarmTests(unittest.TestCase):
    def _swarm(self, **kw):
        from nomorals.agents.research_swarm import ResearchSwarm

        return ResearchSwarm(_FakeContext(_results()), **kw)

    def test_angles_are_distinct_and_capped(self) -> None:
        s = self._swarm(workers=3)
        angles = s.angles_for("is the widget economy real")
        self.assertEqual(len(angles), 3)
        self.assertEqual(angles[0], "is the widget economy real")
        self.assertEqual(len(set(angles)), len(angles))

    def test_run_produces_findings_sources_and_conflicts(self) -> None:
        s = self._swarm(workers=2, read_pages=False)
        report = s.run("is the widget economy real")
        self.assertGreaterEqual(len(report.findings), 3)
        self.assertGreaterEqual(len(report.sources), 2)
        # sources de-duplicated across angles
        urls = [x["url"] for x in report.sources]
        self.assertEqual(len(urls), len(set(urls)))
        # the canned opposing snippets must be flagged
        self.assertTrue(report.conflicts, "conflict was not detected")
        self.assertTrue(any("disagree" in c for c in report.conflicts))
        # deterministic synthesis (no router on the fake context)
        self.assertIn("deterministic synthesis", report.synthesis)
        self.assertTrue(report.to_text().strip())
        json.dumps(report.to_dict())

    def test_explicit_angles_override_decomposition(self) -> None:
        s = self._swarm(workers=4, read_pages=False)
        report = s.run("widget economy", angles=["widgets in 1890", "widget safety"])
        self.assertEqual(report.angles, ["widgets in 1890", "widget safety"])

    def test_failed_worker_is_not_fatal(self) -> None:
        from nomorals.agents.research_swarm import ResearchSwarm
        from nomorals.agents.search.engine import ToolError

        ctx = _FakeContext(_results())

        def boom(self, angle, i):
            raise ToolError("search failed: no network")

        s = ResearchSwarm(ctx, workers=2, read_pages=False)
        s._research_angle = boom.__get__(s)  # type: ignore[method-assign]
        report = s.run("widget economy")
        self.assertEqual(len(report.failed_angles), 2)
        self.assertEqual(report.findings, [])
        # still a valid report
        self.assertIn("failed", report.to_text())

    def test_conflict_detection_direct(self) -> None:
        from nomorals.agents.research_swarm import SwarmFinding, ResearchSwarm

        a = SwarmFinding(angle="x", claim="the widget engine is not thread-safe",
                         sources=[{"url": "u1", "title": "t", "trust": 0.7}],
                         confidence=0.6)
        b = SwarmFinding(angle="y", claim="the widget engine is thread-safe and fast",
                         sources=[{"url": "u2", "title": "t", "trust": 0.8}],
                         confidence=0.7)
        conflicts = ResearchSwarm._detect_conflicts([a, b])
        self.assertEqual(len(conflicts), 1)
        # no negation on either side → no false alarm
        c = SwarmFinding(angle="y", claim="the widget engine is fast",
                         sources=[], confidence=0.5)
        self.assertEqual(ResearchSwarm._detect_conflicts([c, SwarmFinding(
            angle="z", claim="the widget engine is fast too", sources=[],
            confidence=0.5)]), [])

    def test_to_memory_files_report(self) -> None:
        memory = _FakeMemory()
        s = self._swarm(workers=2, read_pages=False)
        s.context.memory = memory
        report = s.run("widget economy", save_memory=True)
        self.assertGreaterEqual(len(memory.items), 2)
        self.assertIn("episode", [k for _c, k, _s in memory.items])
        self.assertIn("fact", [k for _c, k, _s in memory.items])

    def test_claim_mining_prefers_angle_overlap(self) -> None:
        from nomorals.agents.research_swarm import ResearchSwarm

        words = ResearchSwarm._best_sentence(
            "Unrelated sentence about cats. The widget economy grew forty "
            "percent and the widget economy is expected to keep growing.",
            {"widget", "economy"})
        self.assertIn("widget economy grew", words)


class CommandCenterTests(unittest.TestCase):
    def test_registry_is_complete(self) -> None:
        from nomorals.social.chat.control import (CONTROL_COMMANDS,
                                                  COMMAND_DETAILS, LIST_GROUPS,
                                                  LIST_ONELINERS)

        aliases = {"commands", "menu", "list"}
        missing_details = [c for c in CONTROL_COMMANDS
                           if c not in COMMAND_DETAILS and c not in aliases]
        self.assertEqual(missing_details, [])
        grouped: set[str] = set()
        for _name, kinds in LIST_GROUPS:
            grouped.update(kinds)
        ungrouped = [c for c in CONTROL_COMMANDS if c not in grouped]
        self.assertEqual(ungrouped, [])
        missing_oneliners = [c for c in CONTROL_COMMANDS
                             if c not in LIST_ONELINERS and c not in aliases]
        self.assertEqual(missing_oneliners, [])

    def test_list_shows_new_commands(self) -> None:
        from nomorals.social.chat.control import list_catalog

        catalog = list_catalog("")
        for cmd in ("/profile", "/swarm", "/decode", "/structure", "/cipher"):
            self.assertIn(cmd, catalog)
        # group filter still works
        self.assertIn("/status", list_catalog("status"))

    def test_swarm_help_mentions_research(self) -> None:
        from nomorals.social.chat.control import detailed_help

        self.assertIn("research", detailed_help("swarm"))
        self.assertIn("profile", detailed_help("profile"))


class ChatCommandTuneTests(unittest.TestCase):
    """/profile and /list through the real control dispatcher."""

    def setUp(self) -> None:
        from tests.test_w85c import _RuntimeFixture

        self.fx = _RuntimeFixture()
        self.runtime, self.adapter, self.settings = self.fx._make()

    def tearDown(self) -> None:
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            pass
        self.fx.close()

    def test_profile_command_in_chat(self) -> None:
        reply = self.runtime.handle_control("/profile", "local:console")
        self.assertIn("running on:", reply)
        self.assertIn("threads", reply)
        self.assertIn("override:", reply)

    def test_list_command_in_chat(self) -> None:
        reply = self.runtime.handle_control("/list", "local:console")
        self.assertIn("executable commands", reply)
        self.assertIn("/profile", reply)

    def test_help_command_in_chat(self) -> None:
        reply = self.runtime.handle_control("/help swarm", "local:console")
        self.assertIn("research", reply)


if __name__ == "__main__":
    unittest.main()
