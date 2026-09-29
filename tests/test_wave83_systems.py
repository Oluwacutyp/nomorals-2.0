"""Wave 83 systems: self-heal before paging, memory hardening, the
nm watch loop, planner self-tuning — plus the model-chain guard and
the proxy source registry + internet-wide discovery.  All hermetic:
no model, no network (fake fetchers)."""

from __future__ import annotations

import json
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.orchestrator import MasterOrchestrator
from nomorals.agents.skills import SkillLibrary
from nomorals.core.config import Settings
from nomorals.missions import (MissionRunner, MissionStatus, MissionStore)


class _Base(unittest.TestCase):
    def setUp(self):
        self.home = tempfile.mkdtemp(prefix="nm-w83-")
        self.context = build_context(Settings(home=self.home,
                                              reasoning_mode="off"))
        self.context.__enter__()

    def tearDown(self):
        self.context.__exit__(None, None, None)

    def _make_stuck(self, goal="stuck worker"):
        store = MissionStore(self.context.db)
        mission = store.create_new(goal)
        mission.status = MissionStatus.RUNNING
        store.save(mission)
        self.context.db.execute(
            "UPDATE missions SET updated_at = ? WHERE id = ?",
            (time.time() - MissionRunner.STUCK_AFTER_SECONDS - 60,
             mission.id))
        return mission


# ── model chain guard ────────────────────────────────────────────────────────

class RouterChatGuardTests(unittest.TestCase):
    def _failing_chat_provider(self, name, error):
        from nomorals.llm.base import LLMProvider

        class P(LLMProvider):
            def __init__(self):
                super().__init__()
                self.name = name

            @property
            def model_id(self):
                return f"{self.name}-model"

            @property
            def capabilities(self):
                return {"chat", "complete"}

            def chat(self, messages, params=None, **kw):
                raise RuntimeError(error)

            def health(self):
                return True

        return P()

    def test_chat_never_reaches_the_ocr_provider(self):
        from nomorals.llm.providers.ocr import OCRProvider
        from nomorals.llm.router import LLMRouter

        router = LLMRouter()
        router.add(self._failing_chat_provider(
            "groq", "connection failed [Errno 7]"), primary=True)
        router.add(OCRProvider())  # the vision floor, registered last

        response = router.chat([types.SimpleNamespace(role="user",
                                                      content="hi")])
        self.assertFalse(response.ok)
        # the ocr provider must never be attempted for chat
        self.assertNotIn("ocr", response.error)
        self.assertNotIn("reads images", response.error)

    def test_exhausted_chain_reports_every_attempt(self):
        from nomorals.llm.providers.ocr import OCRProvider
        from nomorals.llm.router import LLMRouter

        router = LLMRouter()
        router.add(self._failing_chat_provider(
            "groq", "connection failed [Errno 7]"), primary=True)
        router.add(self._failing_chat_provider("hf", "HTTP 400"),
                   primary=False)
        router.add(OCRProvider())

        response = router.chat([types.SimpleNamespace(role="user",
                                                      content="hi")])
        self.assertIn("all chat providers failed", response.error)
        self.assertIn("groq: unhandled.runtimeerror: "
                      "connection failed [Errno 7]", response.error)
        self.assertIn("hf: unhandled.runtimeerror: HTTP 400",
                      response.error)
        self.assertNotIn("reads images", response.error)


# ── proxy source registry + discovery ────────────────────────────────────────

class ProxySourceRegistryTests(unittest.TestCase):
    def test_dead_source_retires_and_reinstates(self):
        from nomorals.tools.proxysources import SourceRegistry
        tmp = Path(tempfile.mkdtemp(prefix="nm-w83-src-"))
        reg = SourceRegistry(tmp / "sources.json")
        self.assertIn("thespeedx-http",
                      [n for n, _, _ in reg.sources()])
        for _ in range(3):
            reg.record("thespeedx-https", ok=False, error="HTTP 404")
        disabled = {d["name"] for d in reg.disabled()}
        self.assertIn("thespeedx-https", disabled)
        self.assertNotIn("thespeedx-https",
                         [n for n, _, _ in reg.sources()])
        reg.record("thespeedx-https", ok=True, found=20)
        self.assertEqual(reg.disabled(), [])
        self.assertEqual(reg.stats()["active"],
                         len(reg.sources()))

    def test_registry_persists_across_instances(self):
        from nomorals.tools.proxysources import SourceRegistry
        tmp = Path(tempfile.mkdtemp(prefix="nm-w83-src-"))
        reg = SourceRegistry(tmp / "sources.json")
        reg.sources()
        reg.record("monosans-http", ok=True, found=42)
        reg2 = SourceRegistry(tmp / "sources.json")
        row = [h for h in reg2.health() if h["name"] == "monosans-http"][0]
        self.assertEqual(row["last_found"], 42)

    def test_discovered_source_joins_the_catalog(self):
        from nomorals.tools.proxysources import SourceRegistry
        tmp = Path(tempfile.mkdtemp(prefix="nm-w83-src-"))
        reg = SourceRegistry(tmp / "sources.json")
        reg.sources()
        reg.add_discovered("found-team-alpha-http",
                           "https://example.com/alpha.txt", "list",
                           seed="https://github.com/topics/proxies",
                           found=30)
        names = [n for n, _, _ in reg.sources()]
        self.assertIn("found-team-alpha-http", names)
        self.assertEqual(reg.stats()["discovered"], 1)


class ProxyDiscoverTests(unittest.TestCase):
    def test_discovery_registers_working_endpoints_only(self):
        from nomorals.tools.proxylab import ProxyScraper

        seed = (b'<a href="https://github.com/owner1/proxy-listing">x</a> '
                b'<a href="https://github.com/owner1/empty-repo">y</a> '
                b"linked 'https://lists.example.com/fresh.txt' on the page")
        api = json.dumps([{"name": "socks5.txt"},
                          {"name": "README.md"}]).encode()
        good = "\n".join(f"10.9.0.{i}:1080" for i in range(10)).encode()
        bad = b"nothing proxy-shaped in here\n"

        def fake(url):
            if "topics" in url:
                return seed
            if url.endswith("contents/"):
                return api if "owner1/proxy-listing" in url else api
            if "raw.githubusercontent" in url:
                if "README" in url:
                    return bad
                if "empty-repo" in url:
                    return bad
                return good
            if "lists.example.com" in url:
                return good
            raise ConnectionError("off the map")

        scraper = ProxyScraper(fetcher=fake)
        report = scraper.discover(["https://github.com/topics/proxies"],
                                  min_proxies=5)
        names = [r["name"] for r in report["registered"]]
        self.assertIn("found-owner1-socks5", names)
        self.assertIn("found-lists-example-com", names)
        self.assertEqual(len(names), 2)  # README + empty repo rejected

    def test_auto_parse_picks_the_right_kind(self):
        from nomorals.tools.proxylab import ProxyScraper
        text_proto = "socks5://1.2.3.4:1080\nhttp://5.6.7.8:8080\n"
        text_list = "9.9.9.9:80\n8.8.8.8:3128\n"
        text_html = "<table><tr><td>7.7.7.7</td><td>8080</td></tr></table>"
        self.assertEqual(ProxyScraper._auto_parse(text_proto)[1], "protocol")
        self.assertEqual(ProxyScraper._auto_parse(text_list)[1], "list")
        self.assertEqual(ProxyScraper._auto_parse(text_html)[1], "html")
        self.assertEqual(ProxyScraper._auto_parse("words only")[0], [])


# ── self-heal before paging ──────────────────────────────────────────────────

class SelfHealTests(_Base):
    def _scan(self, **kw):
        from nomorals.agents.ops_alerts import ops_alerts
        return ops_alerts(self.context, heal_background=False, **kw)

    def test_first_stuck_scan_self_heals_without_paging(self):
        self._make_stuck()
        report = self._scan()
        self.assertEqual(report["sent"], [])
        self.assertEqual(len(report["self_healed"]), 1)
        # the heal attempt ran the mission once (bounded) — it is now
        # in a terminal state, no longer stuck
        mission = MissionStore(self.context.db).list(limit=1)[0]
        self.assertIn(mission.status, MissionStatus.TERMINAL)

    def test_already_healed_mission_pages(self):
        mission = self._make_stuck()
        from nomorals.agents.ops_alerts import _kv_set
        _kv_set(self.context.db,
                f"opsselfheal:stuck:{mission.id}", time.time())
        report = self._scan()
        self.assertEqual(len(report["sent"]), 1)
        self.assertEqual(report["sent"][0]["kind"], "stuck_mission")
        self.assertIn("still stuck after self-heal",
                      report["sent"][0]["title"])

    def test_force_bypasses_the_heal_step(self):
        mission = self._make_stuck()
        report = self._scan(force=True)
        self.assertEqual(len(report["sent"]), 1)  # raw truth, no heal
        kv = self.context.db.query_one(
            "SELECT value FROM kv_store WHERE key=?",
            (f"opsselfheal:stuck:{mission.id}",))
        self.assertIsNone(kv)


# ── memory hardening ─────────────────────────────────────────────────────────

class MemoryHardeningTests(_Base):
    def _abort(self, text):
        from nomorals.agents.reasoning import ReasoningAgent
        return ReasoningAgent(self.context).mid_task_check(
            text, hard_risks=["hard risk: untested code"], log=True)

    def test_advisory_below_threshold(self):
        for _ in range(2):
            out = self._abort("wipe the cache tables")
            self.assertFalse(out["proceed"])
        from nomorals.agents.reasoning import ReasoningAgent
        out = ReasoningAgent(self.context).mid_task_check(
            "wipe the cache tables now", log=False)
        self.assertTrue(any(r.startswith("repeated abort:")
                            for r in out["risks"]))
        self.assertFalse(any(r.startswith("hard risk: repeated abort")
                             for r in out["risks"]))
        self.assertTrue(out["proceed"])

    def test_hard_abort_at_threshold(self):
        for _ in range(5):
            self._abort("format the disk drive")
        from nomorals.agents.reasoning import ReasoningAgent
        out = ReasoningAgent(self.context).mid_task_check(
            "format the disk drive now", log=False)
        self.assertTrue(any(r.startswith("hard risk: repeated abort (5x)")
                            for r in out["risks"]))
        self.assertFalse(out["proceed"])
        # the mined skill carries the hard-tier marker
        from nomorals.agents.skills import SkillLibrary
        skills = SkillLibrary(self.context.db).list(kind="prevention",
                                                    limit=20)
        self.assertTrue(any("HARD TIER" in s.description for s in skills))


# ── watch loop ───────────────────────────────────────────────────────────────

class WatchLoopTests(_Base):
    def test_quiet_tick(self):
        from nomorals.agents.watcher import WatchLoop
        loop = WatchLoop(self.context, interval=1)
        result = loop.tick()
        self.assertEqual(result["paged"], [])
        self.assertEqual(result["findings"], 0)
        self.assertIn("all quiet", WatchLoop.format(result))

    def test_once_run_exits(self):
        from nomorals.agents.watcher import WatchLoop
        loop = WatchLoop(self.context, interval=1)
        rc = loop.run(once=True)
        self.assertEqual(rc, 0)
        self.assertEqual(loop.ticks, 1)

    def test_tick_pages_a_stuck_mission(self):
        self._make_stuck()
        from nomorals.agents.watcher import WatchLoop
        loop = WatchLoop(self.context, interval=1)
        # first tick: self-heal, no page; force-second tick pages it
        first = loop.tick()
        self.assertEqual(first["paged"], [])
        self.assertEqual(len(first["self_healed"]), 1)


# ── planner self-tuning ──────────────────────────────────────────────────────

class PlannerTuningTests(_Base):
    def _learn(self, goal, ok=True, pivots=0):
        orch = MasterOrchestrator(self.context, max_steps=4)
        orch._learn_from_run(goal, None, [], ok=ok, pivots=pivots)

    def test_clean_family_earns_skip_research(self):
        for _ in range(3):
            self._learn("count the files in the folder")
        from nomorals.agents.skills import SkillLibrary
        lib = SkillLibrary(self.context.db)
        skill = lib.list(kind="plantmpl", limit=10)
        body = json.loads(skill[0].body)
        self.assertTrue(body["skip_research"])

    def test_pivot_cancels_the_tuning(self):
        self._learn("count the files in the folder")
        self._learn("count the files in the folder")
        self._learn("count the files in the folder", ok=False, pivots=1)
        from nomorals.agents.skills import SkillLibrary
        lib = SkillLibrary(self.context.db)
        body = json.loads(lib.list(kind="plantmpl", limit=10)[0].body)
        self.assertFalse(body["skip_research"])

    def test_plan_drops_research_for_tuned_family(self):
        for _ in range(3):
            self._learn("count the files in the folder")
        orch = MasterOrchestrator(self.context, max_steps=4)
        plan = orch.plan("count the files in the folder")
        roles = [s.role for s in plan.steps]
        self.assertNotIn("research", roles)
        self.assertIn("self-tuning", plan.rationale)

    def test_unlearned_family_keeps_research(self):
        orch = MasterOrchestrator(self.context, max_steps=4)
        plan = orch.plan("brand new goal nobody has run before")
        self.assertEqual(plan.steps[0].role, "research")


if __name__ == "__main__":
    unittest.main()
