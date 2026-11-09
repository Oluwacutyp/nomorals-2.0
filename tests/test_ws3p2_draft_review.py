"""WS3 P2 — the draft review surface, wired.

Pins:

* catalog: ``drafts`` registered everywhere a command must be
  (CONTROL_COMMANDS / COMMAND_DETAILS / LIST_ONELINERS / exactly
  one LIST_GROUPS slot), owner-only, never public;
* the tap route: a ``/draft:<action>:<id>`` text through the REAL
  partner-runtime dispatch (handle_control, the cardtap seam)
  reaches handle_draft_callback and acts on the NAMED draft — a
  card approval posts that draft, not the oldest/first pending one;
* the owner gate on taps: a non-owner tap is refused in words and
  nothing moves;
* state normalization: a scheduled weekly draft's "post" tap
  re-presents it for review and posts it;
* the /drafts board: the review batch with grid-packed per-draft
  buttons, per-draft cards (review_card for in-review drafts, the
  propose_post standing question for unproposed ones — opening one
  proposes it), and a COMPLETE text path (show/post/discard by
  board number, no buttons involved);
* the morning briefing's draft section: the same batch numbering,
  present when drafts wait, absent when none do;
* the weekly content tool's report carries the review batch.
"""

from __future__ import annotations

import contextlib
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.morning_briefing import BriefingComposer
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.core.policy import CapabilitySet
from nomorals.social import drafts as drafts_mod
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    detailed_help,
    parse_control,
)
from nomorals.social.drafts import (
    DISCARDED,
    DRAFT,
    PENDING_REVIEW,
    POSTED,
    SCHEDULED,
    DraftQueue,
)
from nomorals.storage.db import Database
from nomorals.tools.registry import ToolRegistry

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

_REAL_QUEUE = DraftQueue


class _FakeSocialManager:
    """The publisher seam: records publishes, posts to one platform."""

    instances: list[_FakeSocialManager] = []

    def __init__(self, context: Any) -> None:
        self.context = context
        self.published: list[tuple[str, Any]] = []

    def register_builtins(self) -> _FakeSocialManager:
        _FakeSocialManager.instances.append(self)
        return self

    def publish(self, content: str, platforms: Any = None,
                actor: str = "user", confirmation: str = "") -> Any:
        self.published.append((content, platforms))
        return SimpleNamespace(
            posted=["x"], failed=[], error="",
            results=[{"platform": "x", "ok": True}])


class _RuntimeHarness(unittest.TestCase):
    """PartnerRuntime over a tmp home, with the DraftQueue factory
    and the SocialManager pointed at test doubles — the dispatch,
    routing, queue semantics and callback router stay real."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-ws3p2-")
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        })
        self.context = build_context(settings, with_executor=False,
                                     with_tools=False)

        class _Router:
            def chat(self, *a: Any, **k: Any) -> Any:
                raise AssertionError("test must not call the model")

        self.context.router = _Router()
        self.context.tools = _FakeTools()
        self.runtime = PartnerRuntime(self.context,
                                      gateway=_RecordingGateway())
        _FakeSocialManager.instances = []
        self._patchers = [
            mock.patch.object(
                drafts_mod, "DraftQueue",
                lambda *a, **k: _REAL_QUEUE(db_path=self.db_path)),
            mock.patch("nomorals.social.SocialManager",
                       _FakeSocialManager),
        ]
        for patcher in self._patchers:
            patcher.start()

    def tearDown(self) -> None:
        for patcher in self._patchers:
            patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    # — seeding / reading the one real queue file ──────────────────
    def _seed(self, content: str, *, propose: bool = True,
              schedule: bool = False) -> str:
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            draft = queue.create_draft(content, ["x"])
            if propose:
                queue.propose(draft.id)
            if schedule:
                queue.schedule(draft.id, 1999999999.0)
            return draft.id
        finally:
            queue.close()

    def _status(self, draft_id: str) -> str:
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            draft = queue.get(draft_id)
            assert draft is not None
            return draft.status
        finally:
            queue.close()

    def _run(self, text: str, *, owner: bool = True) -> Any:
        return self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999", owner=owner))

    @staticmethod
    def _text(reply: Any) -> str:
        return reply.text if hasattr(reply, "text") else str(reply)

    def _board_number(self, token: str) -> int:
        """The board number naming the draft whose preview carries
        ``token`` — read off the surface itself, never assumed."""
        reply = self._run("/drafts")
        for line in self._text(reply).splitlines():
            m = re.match(r"(\d+)\. \[.*\] .*", line)
            if m and token in line:
                return int(m.group(1))
        raise AssertionError(f"no board line carries {token!r}")


class CatalogTests(unittest.TestCase):
    def test_drafts_registered_everywhere(self) -> None:
        self.assertIn("drafts", CONTROL_COMMANDS)
        self.assertIn("drafts", COMMAND_DETAILS)
        self.assertIn("drafts", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("drafts"), 1)
        self.assertIn("/drafts show", detailed_help("drafts"))

    def test_drafts_owner_only(self) -> None:
        self.assertNotIn("drafts", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("drafts", False))
        self.assertTrue(caller_may_run("drafts", True))
        parsed = parse_control("/drafts show 2")
        assert parsed is not None
        self.assertEqual(parsed.kind, "drafts")


class CallbackRouteTests(_RuntimeHarness):
    def test_tap_posts_the_named_draft_not_the_other(self) -> None:
        first = self._seed("alpha draft about patience")
        second = self._seed("beta draft about momentum")
        reply = self._run(f"/draft:post:{second}")
        self.assertIn("posted to 1", self._text(reply))
        self.assertEqual(self._status(second), POSTED)
        self.assertEqual(self._status(first), PENDING_REVIEW)
        manager = _FakeSocialManager.instances[-1]
        self.assertEqual(len(manager.published), 1)
        self.assertEqual(manager.published[0][0],
                         "beta draft about momentum")

    def test_tap_discard_and_unknown(self) -> None:
        draft_id = self._seed("gamma draft to drop")
        reply = self._run(f"/draft:discard:{draft_id}")
        self.assertIn("discarded", self._text(reply))
        self.assertEqual(self._status(draft_id), DISCARDED)
        reply = self._run("/draft:post:missingid")
        self.assertIn("gone", self._text(reply))

    def test_tap_from_non_owner_refused(self) -> None:
        draft_id = self._seed("owner only delta draft")
        reply = self._run(f"/draft:post:{draft_id}", owner=False)
        self.assertEqual(self._text(reply),
                         "that one's just for the owner.")
        self.assertEqual(self._status(draft_id), PENDING_REVIEW)
        self.assertEqual(_FakeSocialManager.instances, [])

    def test_scheduled_draft_tap_posts_it(self) -> None:
        draft_id = self._seed("epsilon weekly draft", schedule=True)
        self.assertEqual(self._status(draft_id), SCHEDULED)
        reply = self._run(f"/draft:post:{draft_id}")
        self.assertIn("posted to 1", self._text(reply))
        self.assertEqual(self._status(draft_id), POSTED)


class BoardTests(_RuntimeHarness):
    def test_board_batch_with_grid_buttons(self) -> None:
        self._seed("zeta draft one for the board")
        self._seed("eta draft two for the board")
        reply = self._run("/drafts")
        self.assertTrue(hasattr(reply, "buttons"))
        text = self._text(reply)
        self.assertIn("2 DRAFTS NEED REVIEW", text)
        self.assertIn("/drafts post <n>", text)
        rows = reply.buttons
        self.assertTrue(all(len(row) <= 2 for row in rows))
        flat = [data for row in rows for _, data in row]
        self.assertIn("drafts show 1", flat)
        self.assertIn("drafts show 2", flat)

    def test_show_card_carries_named_draft_buttons(self) -> None:
        self._seed("theta draft first in line")
        second = self._seed("iota draft second in line")
        n = self._board_number("iota draft")
        reply = self._run(f"/drafts show {n}")
        text = self._text(reply)
        self.assertIn("iota draft second in line", text)
        self.assertIn("Virality", text)
        self.assertIn(f"/drafts post {n}", text)
        flat = [data for row in reply.buttons for _, data in row]
        self.assertIn(f"draft:post:{second}", flat)
        self.assertIn(f"draft:discard:{second}", flat)
        self.assertTrue(all(len(row) <= 2 for row in reply.buttons))

    def test_text_path_posts_and_discards_by_number(self) -> None:
        first = self._seed("kappa draft stays or goes")
        second = self._seed("lambda draft goes out")
        n = self._board_number("lambda draft")
        reply = self._run(f"/drafts post {n}")
        self.assertIn("posted to 1", self._text(reply))
        self.assertEqual(self._status(second), POSTED)
        self.assertEqual(self._status(first), PENDING_REVIEW)
        n = self._board_number("kappa draft")
        reply = self._run(f"/drafts discard {n}")
        self.assertIn("discarded", self._text(reply))
        self.assertEqual(self._status(first), DISCARDED)
        reply = self._run("/drafts")
        self.assertIn("no drafts waiting", self._text(reply))

    def test_show_unproposed_draft_proposes_it(self) -> None:
        draft_id = self._seed("mu draft never proposed", propose=False)
        self.assertEqual(self._status(draft_id), DRAFT)
        reply = self._run("/drafts show 1")
        text = self._text(reply)
        self.assertIn("Post directly or send for review?", text)
        self.assertEqual(self._status(draft_id), PENDING_REVIEW)
        flat = [data for row in reply.buttons for _, data in row]
        self.assertIn(f"draft:post:{draft_id}", flat)

    def test_board_refused_for_non_owner(self) -> None:
        self._seed("nu draft behind the gate")
        reply = self._run("/drafts", owner=False)
        self.assertNotIn("need review", self._text(reply))


class BriefingSectionTests(_RuntimeHarness):
    def test_section_present_with_pending_drafts(self) -> None:
        self._seed("xi draft for the morning")
        briefing = BriefingComposer().compose(
            SimpleNamespace(db=Database(":memory:")), "2026-11-09")
        section = next((s for s in briefing.sections
                        if s.name == "draft_review"), None)
        self.assertIsNotNone(section)
        assert section is not None
        self.assertEqual(section.source, "draft-queue")
        joined = "\n".join(section.lines)
        self.assertIn("xi draft for the morning", joined)
        self.assertIn("/drafts", joined)

    def test_section_absent_when_nothing_waits(self) -> None:
        briefing = BriefingComposer().compose(
            SimpleNamespace(db=Database(":memory:")), "2026-11-09")
        self.assertNotIn("draft_review",
                         [s.name for s in briefing.sections])


class WeeklyReportTests(_RuntimeHarness):
    def test_weekly_tool_report_carries_review_batch(self) -> None:
        registry = ToolRegistry(self.context)
        from nomorals.tools import social as social_tools
        social_tools.register(registry)
        fake_brain = SimpleNamespace(
            complete=lambda prompt, task_kind="": SimpleNamespace(
                text=""))
        with mock.patch("nomorals.llm.brain.brain_for",
                        lambda ctx: fake_brain):
            outcome = registry.call(
                "weekly_content", capabilities=CapabilitySet.all(),
                niche="testing")
        self.assertTrue(outcome.ok)
        text = str(outcome.value["text"])
        self.assertIn("DRAFTS NEED REVIEW", text)
        self.assertIn("tap a draft to review it", text)
        self.assertIn("/drafts show <n>", text)
        queue = _REAL_QUEUE(db_path=self.db_path)
        try:
            self.assertGreaterEqual(len(queue.pending()), 1)
        finally:
            queue.close()


if __name__ == "__main__":
    unittest.main()
