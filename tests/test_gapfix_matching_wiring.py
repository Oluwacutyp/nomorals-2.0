"""Reachability for the matching wiring (doc 14, Tier 0 #3 + #4).

The three documented surfaces (gigs, community, mentoring) live in
``nomorals.matching`` and were already served in chat by ``/match``.
This file proves the wiring that was missing: every surface is
reachable end-to-end — command → tool/store → reply — from all three
entry points, over the SAME stores, so a candidate added in one place
shows up in the others:

* chat: ``/match`` (``control_match``) — daily / add / like / why /
  stats / questionnaire / run, per surface.
* CLI: ``nm matching`` (``_cmd_matching``) — surface-first forms
  (``gigs`` / ``community`` / ``mentoring``) and bare verbs.
* tool: the ``matching`` brain tool, through ``registry.call``.

It also pins the two organ behaviours this lane fixed: a bare number
after ``daily`` is a count (not a surface name), and the doc's surface
names (``gigs``, ``mentoring``) canonicalise onto the store's names
(``gig``, ``learning``) instead of splitting one surface in two.

All stores point at tmp paths — nothing here touches ~/.nomorals.
"""

import types
import unittest

from nomorals.cmdline.commands.matching import _cmd_matching
from nomorals.matching import (
    DailyBatchStore,
    Questionnaire,
)
from nomorals.matching.chat import control_match


def _ctx(tmp_path, surface="gig"):
    """A context whose matching stores live under tmp_path."""
    return types.SimpleNamespace(
        matching_batch_store=DailyBatchStore(
            db_path=str(tmp_path / "b.db")),
        matching_questionnaire=Questionnaire(
            owner="owner", db_path=str(tmp_path / "q.db")),
        matching_surface=surface,
    )


def _seed(ctx, surface, cid, title):
    """Register a candidate on the named surface.

    ``add`` uses the context surface (``_surface(context)``), so the
    seed sets ``ctx.matching_surface`` for the add and restores it
    afterwards — without that the candidate lands on whatever surface
    the context happened to carry, and the daily for the named
    surface comes back empty.
    """
    prev = getattr(ctx, "matching_surface", None)
    ctx.matching_surface = surface
    try:
        out = control_match(
            f"add {cid} | {title} | python | quality=0.9", context=ctx)
    finally:
        if prev is None:
            delattr(ctx, "matching_surface")
        else:
            ctx.matching_surface = prev
    assert "registered for" in out, out


class ChatReachabilityTests(unittest.TestCase):
    def setUp(self):
        import pathlib
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_gig_surface_chat_end_to_end(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "gig", "g-1", "gig pick")
        out = control_match("daily gig", context=ctx)
        self.assertIn("gig pick", out)
        self.assertIn("liked", control_match("like g-1", context=ctx))
        why = control_match("why g-1", context=ctx)
        # explain_pick names only MATCHED preference tags; with no
        # questionnaire answers there are none, so the candidate's
        # own tag never appears in the why. What the why does name
        # is the pick, its score, and its quality — 0.50, because
        # the seed's quality=0.9 is an attribute and add never
        # passes quality= to Candidate (it defaults to 0.5).
        self.assertIn("gig pick", why)
        self.assertIn("score", why)
        self.assertIn("quality 0.50", why)
        self.assertIn("prefs 0.00", why)
        self.assertIn("g-1", control_match("stats", context=ctx))

    def test_community_surface_chat_end_to_end(self):
        ctx = _ctx(self._tmp, "community")
        _seed(ctx, "community", "c-1", "community pick")
        out = control_match("daily community", context=ctx)
        self.assertIn("community pick", out)
        self.assertIn("liked", control_match("like c-1", context=ctx))
        self.assertIn("c-1", control_match("stats", context=ctx))

    def test_mentoring_surface_chat_end_to_end(self):
        ctx = _ctx(self._tmp, "learning")
        _seed(ctx, "learning", "m-1", "mentor pick")
        out = control_match("daily learning", context=ctx)
        self.assertIn("mentor pick", out)
        self.assertIn("liked", control_match("like m-1", context=ctx))

    def test_surfaces_are_isolated(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "gig", "g-1", "gig pick")
        _seed(ctx, "community", "c-1", "community pick")
        self.assertIn("gig pick", control_match("daily gig", context=ctx))
        self.assertNotIn("gig pick",
                         control_match("daily community", context=ctx))
        self.assertIn("community pick",
                      control_match("daily community", context=ctx))

    def test_daily_bare_number_is_a_count(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "gig", "g-1", "gig pick")
        out = control_match("daily 5", context=ctx)
        self.assertIn("gig pick", out)
        self.assertNotIn("'5'", out)

    def test_mentoring_alias_reaches_learning(self):
        ctx = _ctx(self._tmp, "learning")
        _seed(ctx, "learning", "m-1", "mentor pick")
        self.assertIn("mentor pick",
                      control_match("daily mentoring", context=ctx))

    def test_gigs_alias_reaches_gig(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "gig", "g-1", "gig pick")
        self.assertIn("gig pick", control_match("daily gigs", context=ctx))

    def test_questionnaire_and_stable_run(self):
        ctx = _ctx(self._tmp, "learning")
        self.assertIn("answer more",
                      control_match("questionnaire learning", context=ctx))
        control_match("rank proposer alice m1,m2", context=ctx)
        control_match("rank reviewer mentor m1,m2", context=ctx)
        out = control_match("run", context=ctx)
        self.assertIn("verified: no blocking pairs", out)


class CliReachabilityTests(unittest.TestCase):
    def setUp(self):
        import pathlib
        import tempfile

        self._tmpdir = tempfile.TemporaryDirectory()
        self._tmp = pathlib.Path(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def _run(self, ctx, task):
        import io
        from contextlib import redirect_stdout

        args = types.SimpleNamespace(task=task)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = _cmd_matching(args, ctx)
        return rc, buf.getvalue()

    def test_cli_surface_first_daily(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "gig", "g-1", "gig pick")
        rc, out = self._run(ctx, ["gigs"])
        self.assertEqual(rc, 0)
        self.assertIn("gig pick", out)

    def test_cli_community_and_mentoring(self):
        ctx = _ctx(self._tmp, "community")
        _seed(ctx, "community", "c-1", "community pick")
        rc, out = self._run(ctx, ["community", "daily"])
        self.assertEqual(rc, 0)
        self.assertIn("community pick", out)
        ctx2 = _ctx(self._tmp, "learning")
        _seed(ctx2, "learning", "m-1", "mentor pick")
        rc, out = self._run(ctx2, ["mentoring"])
        self.assertEqual(rc, 0)
        self.assertIn("mentor pick", out)

    def test_cli_add_then_daily_same_store(self):
        ctx = _ctx(self._tmp, "gig")
        rc, out = self._run(
            ctx, ["gigs", "add", "g-9 | cli gig | python | quality=0.9"])
        self.assertEqual(rc, 0)
        self.assertIn("registered for", out)
        rc, out = self._run(ctx, ["gigs", "daily"])
        self.assertIn("cli gig", out)

    def test_cli_bare_verb_passthrough(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "gig", "g-1", "gig pick")
        rc, out = self._run(ctx, ["daily", "gig"])
        self.assertEqual(rc, 0)
        self.assertIn("gig pick", out)

    def test_cli_no_words_is_usage(self):
        ctx = _ctx(self._tmp, "gig")
        rc, _ = self._run(ctx, [])
        self.assertEqual(rc, 2)

    def test_cli_restores_context_surface(self):
        ctx = _ctx(self._tmp, "gig")
        _seed(ctx, "community", "c-1", "community pick")
        rc, out = self._run(ctx, ["community", "daily"])
        self.assertEqual(rc, 0)
        # the seed landed on community, not the context's gig
        # surface — the daily for community lists it.
        self.assertIn("community pick", out)
        self.assertEqual(ctx.matching_surface, "gig")


class ToolReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nomorals.tools.registry import ToolRegistry

        cls._ns = types.SimpleNamespace()
        cls.registry = ToolRegistry(cls._ns)
        cls.registry.register_builtins()

    def _tool_ctx(self, tmp_path, surface="gig"):
        self._ns.matching_batch_store = DailyBatchStore(
            db_path=str(tmp_path / "b.db"))
        self._ns.matching_questionnaire = Questionnaire(
            owner="owner", db_path=str(tmp_path / "q.db"))
        self._ns.matching_surface = surface
        return self._ns

    def _call(self, **kwargs):
        from nomorals.core.policy import CapabilitySet

        return self.registry.call(
            "matching", actor="test",
            capabilities=CapabilitySet.all(), **kwargs)

    def test_tool_is_registered_and_count_is_806(self):
        # 748 after the auth-plugin wiring added 5 tools (session_forge,
        # session_verify, payload_png_bomb, payload_svg_bomb,
        # payload_keylogger) on top of the bughunter port (lane 121).
        # 749 after the mocap port (lane 125) added the mocap tool.
        # 774 after the secprobe wire-up added 25 tools (scope_load,
        # scope_check, param_mine, param_confirm, inject_probe,
        # jwt_attack, graphql_probe, graphql_amplify, race_wave,
        # cors_probe, csrf_poc, plan_next, plan_add, plan_resolve,
        # leaks_harvest, dossier_build, ban_report, ban_state,
        # ban_set_state, ban_appeal, report_capture, report_amplify,
        # replay_run, replay_diagnose, sec_chain). 775 after this lane
        # (B.19) added scene_spawn. 776 after this lane (B.20) added
        # cloudcam. 777 after this lane (B.22) added show. 778 after
        # this lane (B.26) added scene_project. 779 after this lane
        # (B.23) added facemcap. 780 after this lane (B.25) added
        # facemesh. 781 after this lane (B.27) added vtuber. 782 after
        # this lane (C.2) registered tonal_g2p; 783 after this lane
        # (C.1) added smarthome; 784 after this lane (C.4) added
        # camwall; 785 after this lane (C.8) added robot; 786 after
        # this lane (audit N4) added build_swarm; 806 after gapfix119
        # added matching.
        self.assertIn("matching", self.registry._tools)
        self.assertEqual(len(self.registry._tools), 808)

    def test_tool_daily_per_surface(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib

            tmp = pathlib.Path(tmp)
            self._tool_ctx(tmp, "community")
            ctx = self._ns
            _seed(ctx, "community", "c-1", "community pick")
            outcome = self._call(action="daily", surface="community")
            self.assertTrue(outcome.ok, outcome)
            value = outcome.unwrap()
            self.assertTrue(value["ok"], value)
            self.assertIn("community pick", value["reply"])
            self.assertEqual(value["surface"], "community")

    def test_tool_add_then_daily_same_store(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib

            tmp = pathlib.Path(tmp)
            self._tool_ctx(tmp, "gig")
            outcome = self._call(
                action="add", surface="gig",
                args="g-7 | tool gig | python | quality=0.9")
            self.assertTrue(outcome.ok, outcome)
            self.assertIn("registered for", outcome.unwrap()["reply"])
            outcome = self._call(action="daily", surface="gig")
            self.assertIn("tool gig", outcome.unwrap()["reply"])

    def test_tool_mentoring_alias(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib

            tmp = pathlib.Path(tmp)
            self._tool_ctx(tmp, "learning")
            ctx = self._ns
            _seed(ctx, "learning", "m-1", "mentor pick")
            outcome = self._call(action="daily", surface="mentoring")
            self.assertTrue(outcome.ok, outcome)
            value = outcome.unwrap()
            self.assertIn("mentor pick", value["reply"])
            self.assertEqual(value["surface"], "learning")

    def test_tool_unknown_action_is_an_error(self):
        outcome = self._call(action="explode", surface="gig")
        self.assertFalse(outcome.ok)

    def test_tool_restores_context_surface(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            import pathlib

            tmp = pathlib.Path(tmp)
            self._tool_ctx(tmp, "gig")
            self._call(action="stats", surface="community")
            self.assertEqual(self._ns.matching_surface, "gig")


if __name__ == "__main__":
    unittest.main()
