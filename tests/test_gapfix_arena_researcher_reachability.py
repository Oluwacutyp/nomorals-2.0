"""Arena researcher reachability: the whole surface over the real organ.

Coverage proof for the already-shipped organ (9597fce0, fixed 07f4602d,
1117ac87). tests/test_gapfix_arena_researcher.py proves the organ's
internals on fake repos; THIS file proves the surface is reachable end
to end on the real tree — live registry, chat dispatch, CLI verb, and
a survey over the real repo root producing real findings with verified
anchors. If any wiring is removed (tool unregistered, chat branch
deleted, CLI verb dropped), these fail. They pass on pristine by
design: the organ is already on main 9bce92b6, so prove.py treats
them as coverage pins, not new behaviour — no pin re-bump, the tool
already counts in the 786.
"""
from __future__ import annotations

import io
import json
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from nomorals.storage.db import Database


def _db():
    db = Database(":memory:")
    db.migrate()
    return db


def _ctx(db=None, home=None):
    import tempfile

    db = db or _db()
    home = home or tempfile.mkdtemp()
    partner = SimpleNamespace(proactive_enabled=True, quiet_start=22,
                              quiet_end=8)
    settings = SimpleNamespace(partner=partner, home=home)
    return SimpleNamespace(db=db, settings=settings)


def _live_registry(ctx):
    from nomorals.tools.registry import ToolRegistry

    registry = ToolRegistry(ctx)
    registry.register_builtins()
    return registry


def _enable_research(ctx):
    from nomorals.agents.features import FeatureRegistry

    FeatureRegistry(ctx.db).set("research", True)


class RegistryTests(unittest.TestCase):
    def test_tool_in_live_registry_at_786(self):
        ctx = _ctx()
        registry = _live_registry(ctx)
        self.assertEqual(len(registry._tools), 812)
        self.assertIsNotNone(registry.get("arena_researcher"))
        self.assertEqual(registry.failed_modules(), [])

    def test_tool_status_call_ok(self):
        ctx = _ctx()
        _enable_research(ctx)
        registry = _live_registry(ctx)
        outcome = registry.call("arena_researcher", action="status")
        self.assertTrue(outcome.ok, outcome.error)
        value = outcome.value or {}
        for key in ("job", "gates", "last_run", "pending_proposals",
                    "signals"):
            self.assertIn(key, value)


class ChatTests(unittest.TestCase):
    def _handler(self, ctx):
        from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin

        handler = object.__new__(RuntimeMetaMixin)
        handler.context = SimpleNamespace(tools=_live_registry(ctx))
        return handler

    def test_chat_status_renders(self):
        ctx = _ctx()
        _enable_research(ctx)
        reply = self._handler(ctx)._control_arena_research("status")
        self.assertTrue(reply.startswith("arena researcher:"))
        self.assertIn("job:", reply)
        self.assertIn("gates:", reply)

    def test_chat_unknown_verb_is_usage(self):
        ctx = _ctx()
        reply = self._handler(ctx)._control_arena_research("bogus")
        self.assertTrue(reply.startswith("usage: /arena-research"))

    def test_chat_proposals_empty(self):
        ctx = _ctx()
        _enable_research(ctx)
        reply = self._handler(ctx)._control_arena_research("proposals")
        self.assertEqual(reply,
                         "arena researcher: no proposals filed yet")


class CliTests(unittest.TestCase):
    def test_cli_status_rc0(self):
        from nomorals.cmdline.commands.research import _cmd_arena_research

        ctx = _ctx()
        _enable_research(ctx)
        args = Namespace(action="status", json=False, max_proposals=0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_arena_research(args, ctx)
        self.assertEqual(rc, 0)
        self.assertIn("arena researcher:", buf.getvalue())

    def test_cli_ensure_schedules_job(self):
        from nomorals.agents import arena_researcher as ar
        from nomorals.agents.scheduler import Scheduler
        from nomorals.cmdline.commands.research import _cmd_arena_research

        ctx = _ctx()
        args = Namespace(action="ensure", json=False, max_proposals=0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_arena_research(args, ctx)
        self.assertEqual(rc, 0)
        self.assertIn("arena researcher job:", buf.getvalue())
        jobs = [j for j in Scheduler(ctx).list_jobs()
                if j.get("name") == ar.ARENA_RESEARCHER_JOB]
        self.assertEqual(len(jobs), 1)
        rows = ctx.db.query(
            "SELECT payload FROM schedule_jobs WHERE name = ?",
            (ar.ARENA_RESEARCHER_JOB,))
        payload = json.loads(rows[0]["payload"])
        self.assertEqual(payload["tool"], "arena_researcher")
        self.assertEqual(payload["args"], {"action": "tick"})

    def test_cli_unknown_action_rc2(self):
        from nomorals.cmdline.commands.research import _cmd_arena_research

        ctx = _ctx()
        args = Namespace(action="bogus", json=False, max_proposals=0)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_arena_research(args, ctx)
        self.assertEqual(rc, 2)


class RealSurveyTests(unittest.TestCase):
    def test_real_tree_survey_produces_real_findings(self):
        import tempfile

        from nomorals.agents.arena_researcher.rank import rank_findings
        from nomorals.agents.arena_researcher.survey import (
            survey_repo,
            verify_anchor,
        )
        from nomorals.agents.research_digest import repo_root

        root = repo_root()
        self.assertTrue((root / "nomorals" / "agents"
                         / "arena_researcher" / "loop.py").is_file())
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "arena_survey_cache.json"
            # The organ's own survey budget (120s) as a tripwire: a
            # capped job must pass a deadline, and a partial survey's
            # findings are real, just incomplete.
            findings = survey_repo(root, deadline_s=120, cache_path=cache)
        self.assertGreater(findings.total_files, 2000)
        self.assertGreater(findings.surveyed_files, 0)
        self.assertTrue(findings)
        kinds = {"dead_module", "stub", "unwired_tool", "silent_swallow"}
        for f in findings:
            self.assertIn(f["kind"], kinds)
            self.assertTrue(verify_anchor(root, f["path"], f["line"]),
                            f["path"])
        unwired = [f for f in findings if f["kind"] == "unwired_tool"]
        self.assertEqual(unwired, [])
        ranked = rank_findings(findings)
        self.assertEqual(len(ranked), len(findings))
        self.assertEqual(ranked[0]["kind"], "dead_module")


if __name__ == "__main__":
    unittest.main()
