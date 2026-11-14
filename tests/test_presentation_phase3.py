"""Presentation Phase 3 — the raw-reply→Card sweep.

One pin per converted surface: the Card frame is present (title,
titled sections, stamp where the class is a presented class) AND
every fact the old hand-joined text carried survives verbatim.
The conversions render in the canonical house dialect
(``card_text``) — the gateway converts per platform at send —
so substring pins from the old text hold without re-pinning.
"""

from __future__ import annotations

import re
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import power as power_mod
from nomorals.agents.partner import failures as failures_mod
from nomorals.agents.partner.runtime_meta import RuntimeMetaMixin
from nomorals.goals import GoalTracker, surface
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    detailed_help,
    list_catalog,
)
from nomorals.storage.db import Database

UID = "owner"


def _status_instance():
    """A bare RuntimeMetaMixin, as test_partner_failures builds it."""
    inst = object.__new__(RuntimeMetaMixin)
    mood = SimpleNamespace(label="warm", values={"affection": 50,
                                                 "energy": 60,
                                                 "frustration": 5})
    rel = SimpleNamespace(stage="close", trust=80, milestones=[1, 2],
                          fights=[])
    inst.brain = SimpleNamespace(mood=SimpleNamespace(current=lambda: mood),
                                 relationship=rel)
    inst._autonomy = None
    inst.gateway = SimpleNamespace(status=lambda: {})
    inst.context = SimpleNamespace()
    inst._stats_snapshot = lambda: {"messages": 3, "replies": 2,
                                    "errors": 0, "controls": 1}
    inst._degradation_status = lambda: {"degraded": []}
    inst._model_status_line = lambda: "model: test"
    return inst


class StatusCardTests(unittest.TestCase):
    def test_frame_and_facts(self):
        # hermetic: power mode on, failure ledger empty
        with mock.patch.object(
                power_mod, "power_mode_for",
                lambda ctx: SimpleNamespace(active=True)), \
                mock.patch.object(failures_mod, "failure_count",
                                  lambda: 0):
            out = _status_instance()._control_status()
        # the Card frame: title + titled sections
        self.assertIn("**Devon — status**", out)
        self.assertIn("**State**", out)
        self.assertIn("**Traffic**", out)
        # every fact the old eight-line dump carried
        self.assertIn("warm (affection 50, energy 60, frustration 5)", out)
        self.assertIn("close, trust 80, 2 milestones, 0 fights logged", out)
        self.assertIn("off, 0 pending", out)
        self.assertIn("model: test", out)
        self.assertIn("platforms running", out)
        self.assertIn("active", out)
        self.assertIn("messages", out)
        self.assertIn("replies", out)
        self.assertIn("errors", out)
        self.assertIn("commands", out)
        # healthy: no degradation, no failure ledger
        self.assertNotIn("degradation:", out)
        self.assertNotIn("partner failures", out)
        # settings is not a presentation class — no stamp
        self.assertNotIn("Powered by peacethefirst", out)

    def test_degradation_and_failures_surface_only_when_real(self):
        inst = _status_instance()
        inst._degradation_status = lambda: {
            "degraded": ["llm"],
            "ladders": {"llm": {"rung": 2, "rung_name": "fallback"}}}
        with mock.patch.object(
                power_mod, "power_mode_for",
                lambda ctx: SimpleNamespace(active=True)), \
                mock.patch.object(failures_mod, "failure_count",
                                  lambda: 1), \
                mock.patch.object(
                    failures_mod, "format_failures",
                    lambda limit=5: (
                        "⚠️ partner failures (1 total, "
                        "showing 1 newest):\n"
                        "• 11-11 12:00 runtime.test: ValueError: boom")):
            out = inst._control_status()
        self.assertIn("degradation: llm on rung 2 (fallback)", out)
        self.assertIn("partner failures (1 total", out)
        self.assertIn("runtime.test: ValueError: boom", out)


class CatalogCardTests(unittest.TestCase):
    def test_frame_and_facts(self):
        catalog = list_catalog()
        self.assertIn("**Devon — commands**", catalog)
        self.assertIn("tap any", catalog)
        self.assertIn("**🔍 search & research**", catalog)
        for kind in CONTROL_COMMANDS:
            self.assertIn(f"/{kind}", catalog)
        m = re.search(r"(\d+) commands", catalog)
        self.assertIsNotNone(m, "no command count")
        assert m is not None
        self.assertEqual(int(m.group(1)), len(CONTROL_COMMANDS))
        # every command row is still scannable
        rows = [ln for ln in catalog.splitlines()
                if ln.startswith("  <code>/") and "&lt;" not in ln]
        self.assertTrue(rows, "no command rows")
        for ln in rows:
            self.assertRegex(ln, r"<code>/[a-z0-9_]+</code> — .+")
        # a menu-class Card carries the stamp
        self.assertIn("_𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁_", catalog)

    def test_no_group_stays_plain(self):
        out = list_catalog("no-such-group-xyz")
        self.assertTrue(out.startswith("no group 'no-such-group-xyz'"))
        self.assertIn("available:", out)
        self.assertNotIn("Powered by peacethefirst", out)

    def test_public_catalog_excludes_owner_commands(self):
        text = list_catalog("", public_only=True)
        for kind in ("/update", "/exness", "/status", "/quit"):
            self.assertNotIn(kind, text, kind)
        self.assertIn("/search", text)


class HelpOverviewCardTests(unittest.TestCase):
    def test_frame_and_facts(self):
        out = detailed_help("")
        self.assertIn("**Devon — help**", out)
        self.assertIn("the full catalog", out)
        self.assertIn("topic pages:", out)
        self.assertIn("**her day**", out)
        self.assertIn("**search & research**", out)
        self.assertIn("**everything else**", out)
        self.assertIn("how she works", out)
        self.assertIn("console:", out)
        self.assertIn("_𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁_", out)

    def test_public_overview(self):
        out = detailed_help("", public_only=True)
        self.assertIn("the public catalog", out)
        self.assertNotIn("how she works", out)


class GoalsBoardCardTests(unittest.TestCase):
    def setUp(self):
        self.tracker = GoalTracker(Database(":memory:"))

    def test_board_frame_and_facts(self):
        goal, _, _ = surface.create_goal_filed(
            self.tracker, UID, "Save for a house", "finance")
        surface.run_sync(self.tracker.update_progress(goal.goal_id, 40.0))
        board = surface.render_class_board(self.tracker, UID)
        self.assertIn("**Goals**", board)
        self.assertIn("_𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁_", board)
        # the old pins, verbatim
        self.assertIn("**Finance** (`finance`)", board)
        self.assertIn("1 goal(s) · avg 40%", board)
        self.assertIn("Save for a house — 40% (self-reported)", board)
        self.assertIn("**Careers** (`careers`)", board)
        self.assertIn("🗒 nothing here yet", board)
        self.assertIn("/goals add <title> in careers", board)
        for descriptor in surface.list_classes():
            self.assertEqual(board.count(f"**{descriptor.title}**"), 1)

    def test_board_empty_user(self):
        board = surface.render_class_board(self.tracker, UID)
        self.assertIn("**Goals**", board)
        self.assertIn("0 goal(s) on file", board)
        self.assertIn("🗒 nothing here yet", board)

    def test_focus_card(self):
        surface.create_goal_filed(self.tracker, UID, "Save for a house",
                                  "finance")
        out = surface.render_focus(self.tracker, UID)
        self.assertIn("**Focus**", out)
        self.assertIn("Focus now", out)
        self.assertIn("Save for a house", out)
        self.assertIn("(self-reported)", out)

    def test_progress_card(self):
        surface.create_goal_filed(self.tracker, UID, "Save for a house",
                                  "finance")
        out = surface.render_goal_progress(self.tracker, UID, "house")
        self.assertIn("**Goal progress**", out)
        self.assertIn("Save for a house", out)
        self.assertIn("progress:", out)
        self.assertIn("goal id:", out)

    def test_classes_card(self):
        out = surface.render_classes()
        self.assertIn("**Goal classes**", out)
        self.assertIn("file a goal: /goals add <title> in <class>", out)
        self.assertIn("move a goal: /goals class <goal> <class>", out)


class CanonicalStampTelegramTests(unittest.TestCase):
    """The gateway's F1 conversion must render the canonical
    stamp line as real Telegram HTML — otherwise every
    card_text surface prints ``_Powered by peacethefirst_``
    literally on Telegram (owner screenshot 2026-11-06)."""

    def test_to_telegram_converts_trailing_canonical_stamp(self):
        from nomorals.social.chat.platforms import to_telegram

        out = to_telegram("**Title**\n\n_Powered by peacethefirst_")
        self.assertEqual(
            out, "<b>Title</b>\n\n<i>𝗣𝗼𝘄𝗲𝗿𝗲𝗱 𝗯𝘆 𝗽𝗲𝗮𝗰𝗲𝘁𝗵𝗲𝗳𝗶𝗿𝘀𝘁</i>")

    def test_to_telegram_leaves_other_underscores_literal(self):
        from nomorals.social.chat.platforms import to_telegram

        out = to_telegram("see snake_case_name and _not_a_stamp_ here")
        self.assertIn("snake_case_name", out)
        self.assertIn("_not_a_stamp_", out)
        self.assertNotIn("<i>", out)

    def test_to_telegram_stamp_only_at_the_trailing_footer(self):
        from nomorals.social.chat.platforms import to_telegram

        out = to_telegram("_Powered by peacethefirst_ is our mark\n"
                          "and the real line follows")
        self.assertIn("_Powered by peacethefirst_", out)
        self.assertNotIn("<i>Powered by peacethefirst</i>", out)


if __name__ == "__main__":
    unittest.main()
