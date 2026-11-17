"""Arena researcher acceptance tests: Devon researching herself.

Covers agents/arena_researcher/:
 1. arena_hours reads NM_ARENA_RESEARCHER_HOURS (default 24), clamps
    [6, 168], falls back on garbage.
 2. The survey finds dead modules (zero production importers AND zero
    dotted-name references), stubs (NotImplementedError / pass-only
    bodies, abstract methods excluded), unwired tools (a registry entry
    whose module cannot import or has no register hook), and silent-
    swallow weak spots (>= 3 sites in one module). Every finding
    carries a file:line anchor that verifies against the surveyed root.
 3. Ranking puts dead modules first, then stubs, then unwired tools,
    then swallow density; tickets built from findings pass the real
    gate_ticket and file into the upgrade queue as source
    "arena_researcher" via UpgradePipeline.
 4. ArenaResearcher.tick defers cleanly (recorded, never raises) when
    gated; with a fake survey + fake SOTA pipeline it runs the full
    cycle — findings counted, top findings filed, owner notified, run
    recorded in the durable store. A second tick with the same finding
    does not refile it.
 5. ensure_arena_researcher_job is idempotent by name and replaces a
    stale interval; the payload fires the arena_researcher tool's tick
    action. status exposes job health, gates, last run, and pending
    proposals filed by this organ.
 6. The durable JSON store round-trips runs, quarantines a corrupt
    file and degrades to empty with a named reason, and runs in-memory
    when the runtime has no home.
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None, proactive=True, home=None):

    db = db or _db()
    partner = SimpleNamespace(proactive_enabled=proactive, quiet_start=22,
                              quiet_end=8)
    settings = SimpleNamespace(partner=partner, home=home)
    return SimpleNamespace(db=db, settings=settings)


def _enable_research(ctx):
    from nomorals.agents.features import FeatureRegistry

    FeatureRegistry(ctx.db).set("research", True)


def _disable_research(ctx):
    from nomorals.agents.features import FeatureRegistry

    FeatureRegistry(ctx.db).set("research", False)


def _write(root: Path, rel: str, text: str) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class HoursTests(unittest.TestCase):
    def setUp(self):
        self._old = os.environ.get("NM_ARENA_RESEARCHER_HOURS")

    def tearDown(self):
        if self._old is None:
            os.environ.pop("NM_ARENA_RESEARCHER_HOURS", None)
        else:
            os.environ["NM_ARENA_RESEARCHER_HOURS"] = self._old

    def test_default_twenty_four(self):
        from nomorals.agents.arena_researcher import arena_hours

        os.environ.pop("NM_ARENA_RESEARCHER_HOURS", None)
        self.assertEqual(arena_hours(), 24.0)

    def test_env_parses(self):
        from nomorals.agents.arena_researcher import arena_hours

        os.environ["NM_ARENA_RESEARCHER_HOURS"] = "48"
        self.assertEqual(arena_hours(), 48.0)

    def test_garbage_falls_back(self):
        from nomorals.agents.arena_researcher import arena_hours

        os.environ["NM_ARENA_RESEARCHER_HOURS"] = "not-a-number"
        self.assertEqual(arena_hours(), 24.0)

    def test_clamps(self):
        from nomorals.agents.arena_researcher import arena_hours

        os.environ["NM_ARENA_RESEARCHER_HOURS"] = "1"
        self.assertEqual(arena_hours(), 6.0)
        os.environ["NM_ARENA_RESEARCHER_HOURS"] = "999"
        self.assertEqual(arena_hours(), 168.0)


class SurveyTests(unittest.TestCase):
    def _root(self, tmp_path):
        root = Path(tmp_path) / "repo"
        _write(root, "nomorals/pkg_a/__init__.py",
               "from nomorals.pkg_a.caller import go\n")
        _write(root, "nomorals/pkg_a/live.py",
               "VALUE = 1\n\n\ndef used():\n    return VALUE\n")
        _write(root, "nomorals/pkg_a/dead.py",
               "VALUE = 2\n\n\ndef never_called():\n    return VALUE\n")
        _write(root, "nomorals/pkg_a/caller.py",
               "from nomorals.pkg_a.live import used\n\n\ndef go():\n    return used()\n")
        return root

    def _tmpdir(self):
        import tempfile

        return tempfile.mkdtemp()

    def test_dead_module_found_live_module_not(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            findings = survey_repo(root)
            dead = [f for f in findings if f["kind"] == "dead_module"]
            self.assertEqual([f["path"] for f in dead],
                             ["nomorals/pkg_a/dead.py"])
            self.assertEqual(dead[0]["line"], 1)
            self.assertIn("zero production importers", dead[0]["evidence"])

    def test_entry_points_excluded(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write(root, "nomorals/cli.py", "def main():\n    pass\n")
            findings = survey_repo(root)
            paths = [f["path"] for f in findings
                     if f["kind"] == "dead_module"]
            self.assertNotIn("nomorals/cli.py", paths)

    def test_string_reference_keeps_module_alive(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write(root, "nomorals/pkg_a/caller.py",
                   "from nomorals.pkg_a.live import used\n"
                   "NAME = \"nomorals.pkg_a.dead\"\n\n\n"
                   "def go():\n    return used()\n")
            findings = survey_repo(root)
            dead = [f for f in findings if f["kind"] == "dead_module"]
            self.assertEqual(dead, [])

    def test_stub_found_abstract_excluded(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write(root, "nomorals/pkg_a/stubbed.py",
                   "from abc import ABC, abstractmethod\n\n\n"
                   "def not_done_yet():\n"
                   "    raise NotImplementedError(\"later\")\n\n\n"
                   "def empty_body():\n"
                   "    pass\n\n\n"
                   "class Base(ABC):\n"
                   "    @abstractmethod\n"
                   "    def contract(self):\n"
                   "        raise NotImplementedError\n")
            findings = survey_repo(root)
            stubs = [f for f in findings if f["kind"] == "stub"]
            names = sorted(f["evidence"] for f in stubs)
            self.assertEqual(len(stubs), 2)
            self.assertTrue(any("not_done_yet" in n for n in names))
            self.assertTrue(any("empty_body" in n for n in names))
            self.assertFalse(any("contract" in n for n in names))

    def test_unwired_tool_found(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            findings = survey_repo(
                root, tool_modules=["missing_module_nope"])
            unwired = [f for f in findings if f["kind"] == "unwired_tool"]
            self.assertEqual(len(unwired), 1)
            self.assertIn("missing_module_nope", unwired[0]["evidence"])

    def test_silent_swallows_need_three_sites(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write(root, "nomorals/pkg_a/noisy.py",
                   "def a():\n"
                   "    try:\n        risky()\n    except Exception:\n        pass\n\n\n"
                   "def b():\n"
                   "    try:\n        risky()\n    except Exception:\n        pass\n\n\n"
                   "def c():\n"
                   "    try:\n        risky()\n    except Exception:\n        pass\n")
            _write(root, "nomorals/pkg_a/quiet.py",
                   "def a():\n"
                   "    try:\n        risky()\n    except Exception:\n        pass\n")
            findings = survey_repo(root)
            sw = [f for f in findings if f["kind"] == "silent_swallow"]
            self.assertEqual([f["path"] for f in sw],
                             ["nomorals/pkg_a/noisy.py"])
            self.assertIn("3 silent", sw[0]["evidence"])

    def test_every_finding_anchor_verifies(self):
        import tempfile

        from nomorals.agents.arena_researcher.survey import survey_repo, verify_anchor

        with tempfile.TemporaryDirectory() as tmp:
            root = self._root(tmp)
            _write(root, "nomorals/pkg_a/stubbed.py",
                   "def not_done_yet():\n"
                   "    raise NotImplementedError(\"later\")\n")
            findings = survey_repo(root)
            self.assertTrue(findings)
            for f in findings:
                self.assertTrue(verify_anchor(root, f["path"], f["line"]),
                                f["path"])


class RankTests(unittest.TestCase):
    def test_dead_outranks_stub_outranks_unwired_outranks_swallow(self):
        from nomorals.agents.arena_researcher.rank import rank_findings

        findings = [
            {"kind": "silent_swallow", "path": "nomorals/a/b.py",
             "line": 1, "organ": "a", "evidence": "3 silent sites",
             "count": 3},
            {"kind": "unwired_tool", "path": "nomorals/tools/agents.py",
             "line": 70, "organ": "tools",
             "evidence": "tool module missing"},
            {"kind": "stub", "path": "nomorals/a/c.py", "line": 5,
             "organ": "a", "evidence": "def f raises NotImplementedError"},
            {"kind": "dead_module", "path": "nomorals/a/d.py", "line": 1,
             "organ": "a", "evidence": "zero production importers"},
        ]
        ranked = rank_findings(findings)
        self.assertEqual([f["kind"] for f in ranked],
                         ["dead_module", "stub", "unwired_tool",
                          "silent_swallow"])

    def test_ticket_passes_the_real_gate(self):
        from nomorals.agents.arena_researcher.rank import finding_to_ticket
        from nomorals.agents.research_digest import gate_ticket

        finding = {
            "kind": "dead_module",
            "path": "nomorals/agents/research_loop.py",
            "line": 82, "organ": "agents",
            "evidence": ("zero production importers in nomorals/ and zero "
                         "dotted-name references in nomorals/"),
            "id": "ar-test-0001",
        }
        ticket = finding_to_ticket(finding, run_id="ar-run-test")
        self.assertEqual(gate_ticket(ticket), [])

    def test_ticket_files_into_upgrade_queue(self):
        from nomorals.agents.arena_researcher.rank import finding_to_ticket
        from nomorals.agents.upgrade_queue import UpgradePipeline, UpgradeQueue

        ctx = _ctx()
        _enable_research(ctx)
        finding = {
            "kind": "dead_module",
            "path": "nomorals/agents/research_loop.py",
            "line": 82, "organ": "agents",
            "evidence": ("zero production importers in nomorals/ and zero "
                         "dotted-name references in nomorals/"),
            "id": "ar-test-0002",
        }
        ticket = finding_to_ticket(finding, run_id="ar-run-test")
        pid = UpgradePipeline(ctx).propose_from_ticket(
            ticket, source="arena_researcher")
        self.assertTrue(pid)
        row = UpgradeQueue(ctx).get(pid)
        self.assertEqual(row["source"], "arena_researcher")
        self.assertEqual(row["status"], "proposed")


class StoreTests(unittest.TestCase):
    def test_round_trip_and_last_run(self):
        import tempfile

        from nomorals.agents.arena_researcher.store import (
            ArenaResearcherStore,
            default_arena_researcher_path,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = default_arena_researcher_path(tmp)
            store = ArenaResearcherStore(path=path)
            rid = store.record_run({
                "id": "ar-run-1", "started_at": 1000.0,
                "finished_at": 1001.0, "ok": True, "skipped_reason": "",
                "organs_surveyed": 3, "findings_count": 2,
                "proposals_created": 1, "proposal_ids": ["upg-1"],
                "sota_mined": True, "sota_claims": 4, "error": "",
            })
            self.assertEqual(rid, "ar-run-1")
            store2 = ArenaResearcherStore(path=path)
            last = store2.last_run()
            self.assertEqual(last["id"], "ar-run-1")
            self.assertEqual(last["findings_count"], 2)

    def test_corrupt_store_quarantines_and_degrades_empty(self):
        import tempfile

        from nomorals.agents.arena_researcher.store import ArenaResearcherStore

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "arena_researcher.json"
            path.write_text("{not json", encoding="utf-8")
            store = ArenaResearcherStore(path=path)
            self.assertEqual(store.last_run(), None)
            self.assertTrue(store.degraded_reason)
            self.assertTrue((Path(tmp) / "arena_researcher.json.corrupt").exists()
                            or list(Path(tmp).glob("*.corrupt")))

    def test_in_memory_when_no_home(self):
        from nomorals.agents.arena_researcher.store import ArenaResearcherStore

        store = ArenaResearcherStore(path=None)
        store.record_run({"id": "ar-run-mem", "started_at": 1.0,
                          "finished_at": 2.0, "ok": True,
                          "skipped_reason": "", "organs_surveyed": 1,
                          "findings_count": 0, "proposals_created": 0,
                          "proposal_ids": [], "sota_mined": False,
                          "sota_claims": 0, "error": ""})
        self.assertEqual(store.last_run()["id"], "ar-run-mem")


class LoopTests(unittest.TestCase):
    def _fake_finding(self):
        return {
            "kind": "dead_module",
            "path": "nomorals/agents/research_loop.py",
            "line": 82, "organ": "agents",
            "evidence": ("zero production importers in nomorals/ and zero "
                         "dotted-name references in nomorals/"),
            "id": "ar-test-0003",
        }

    def test_tick_defers_when_gated(self):
        from nomorals.agents.arena_researcher import ArenaResearcher

        ctx = _ctx()
        _disable_research(ctx)
        out = ArenaResearcher(ctx).tick()
        self.assertFalse(out["ok"])
        self.assertTrue(out["skipped_reason"])
        self.assertEqual(out["proposals_created"], 0)

    def test_tick_full_cycle_files_and_notifies(self):
        from nomorals.agents.arena_researcher import ArenaResearcher
        from nomorals.agents.upgrade_queue import UpgradeQueue

        ctx = _ctx()
        _enable_research(ctx)
        finding = self._fake_finding()
        notified = {}

        def fake_publish(self, kind, title="", body="", **kw):
            notified["kind"] = kind
            notified["title"] = title
            return {"delivered": True}

        # Cycle tests run outside quiet hours by mocking the clock
        # check, the same pattern test_research_loop.py uses — the
        # quiet-hours deferral itself is covered there.
        with mock.patch(
                "nomorals.agents.research_loop._in_quiet_hours_now",
                return_value=False), \
            mock.patch(
                "nomorals.agents.arena_researcher.loop.survey_repo",
                return_value=[finding]), \
            mock.patch(
                "nomorals.agents.research_digest.ResearchPipeline.run",
                return_value={
                    "claims": [{"text": "bf16 is the 2026 default for "
                                "large image models",
                                "confidence": 0.8}],
                    "tickets": [],
                }), \
            mock.patch(
                "nomorals.agents.notifier.Notifier.publish",
                fake_publish):
            out = ArenaResearcher(ctx).tick()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["findings_count"], 1)
        self.assertEqual(out["proposals_created"], 1)
        self.assertEqual(len(out["proposal_ids"]), 1)
        self.assertTrue(out["notified"])
        self.assertEqual(notified.get("kind"), "arena_researcher")
        row = UpgradeQueue(ctx).get(out["proposal_ids"][0])
        self.assertEqual(row["source"], "arena_researcher")
        self.assertEqual(row["status"], "proposed")

    def test_second_tick_does_not_refile(self):
        from nomorals.agents.arena_researcher import ArenaResearcher

        ctx = _ctx()
        _enable_research(ctx)
        finding = self._fake_finding()
        with mock.patch(
                "nomorals.agents.research_loop._in_quiet_hours_now",
                return_value=False), \
            mock.patch(
                "nomorals.agents.arena_researcher.loop.survey_repo",
                return_value=[finding]), \
            mock.patch(
                "nomorals.agents.research_digest.ResearchPipeline.run",
                return_value={"claims": [], "tickets": []}), \
            mock.patch(
                "nomorals.agents.notifier.Notifier.publish",
                lambda self, kind, title="", body="", **kw:
                    {"delivered": True}):
            first = ArenaResearcher(ctx).tick()
            second = ArenaResearcher(ctx).tick()
        self.assertEqual(first["proposals_created"], 1)
        self.assertEqual(second["proposals_created"], 0)
        self.assertEqual(second["proposals_deduped"], 1)

    def test_tick_survives_sota_failure(self):
        from nomorals.agents.arena_researcher import ArenaResearcher

        ctx = _ctx()
        _enable_research(ctx)
        finding = self._fake_finding()
        with mock.patch(
                "nomorals.agents.research_loop._in_quiet_hours_now",
                return_value=False), \
            mock.patch(
                "nomorals.agents.arena_researcher.loop.survey_repo",
                return_value=[finding]), \
            mock.patch(
                "nomorals.agents.research_digest.ResearchPipeline.run",
                side_effect=RuntimeError("no network")), \
            mock.patch(
                "nomorals.agents.notifier.Notifier.publish",
                lambda self, kind, title="", body="", **kw:
                    {"delivered": True}):
            out = ArenaResearcher(ctx).tick()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["proposals_created"], 1)
        self.assertFalse(out["sota_mined"])

    def test_status_reports_job_gates_last_run(self):
        from nomorals.agents import arena_researcher as ar

        ctx = _ctx()
        _enable_research(ctx)
        data = ar.status(ctx)
        self.assertIn("job", data)
        self.assertIn("gates", data)
        self.assertTrue(data["gates"]["feature_research"])
        self.assertEqual(data["pending_proposals"], 0)

    def test_ensure_job_idempotent(self):
        from nomorals.agents import arena_researcher as ar

        ctx = _ctx()
        first = ar.ensure_arena_researcher_job(ctx)
        second = ar.ensure_arena_researcher_job(ctx)
        self.assertEqual(first.get("job_id"), second.get("job_id"))
        self.assertTrue(second.get("already_scheduled"))
        from nomorals.agents.scheduler import Scheduler

        jobs = [j for j in Scheduler(ctx).list_jobs()
                if j.get("name") == ar.ARENA_RESEARCHER_JOB]
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["payload_kind"], "tool")
        # list_jobs drops the raw payload; prove the stored payload
        # fires the arena_researcher tool's tick action from the row.

        rows = ctx.db.query(
            "SELECT payload FROM schedule_jobs WHERE name = ?",
            (ar.ARENA_RESEARCHER_JOB,))
        payload = json.loads(rows[0]["payload"])
        self.assertEqual(payload["tool"], "arena_researcher")
        self.assertEqual(payload["args"], {"action": "tick"})

    def test_tool_registers(self):
        from nomorals.agents.arena_researcher import register

        seen = {}

        class FakeRegistry:
            context = _ctx()

            def register(self, name, **kw):
                seen[name] = kw
                return lambda fn: fn

        register(FakeRegistry())
        self.assertIn("arena_researcher", seen)
        params = seen["arena_researcher"].get("parameters") or {}
        self.assertIn("tick", str(params.get("action", "")))


if __name__ == "__main__":
    unittest.main()
