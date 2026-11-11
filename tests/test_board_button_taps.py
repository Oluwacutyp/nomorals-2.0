"""Button-fix lane 5 — the thirteen buttonless boards, tap-tested
BY EFFECT.

The button-effect audit (~/workspace/button-audit/AUDIT.md §4)
mechanically verified thirteen boards whose handlers returned
plain text with no buttons at all — careers, ideas, day, calendar,
repurpose, engage, cases, runs, worldmodel, craft, glossary,
notify decisions, explore — while their boards presented per-item
typed actions in text. Lane 5 adjudicated each board: a button
ships only where a tap performs the board's real operation (the
typed verb for the same act, identity carried in the payload);
views whose honest interaction is display-only keep NO buttons
and are pinned buttonless here.

This suite is the audit §5 harness for those boards:

* **Harvest** — each board's real emitted payloads, collected
  from the board render itself (never hand-written copies).
* **Drive the real entry point** — the Telegram Bot API personal
  bot's signed callback path (adapter → synthetic text →
  PartnerRuntime.handle_control), the same chain a handset tap
  travels.
* **Assert STATE, pairwise** — the tap path and the typed path
  run against twin fixtures and must land identical end state
  (an application moved, an item ignored, a case acknowledged,
  a ticket stopped with its receipt, an entry deleted). Reply
  text alone never counts as proof.
"""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social import drafts as drafts_mod
from nomorals.social.chat.telegram import TelegramBotAdapter
from nomorals.social.drafts import DraftQueue

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402

_OWNER_ID = "5478650254"


def _personal_bot() -> TelegramBotAdapter:
    """The personal bot's shape (owner_only) — the surface the
    owner's taps actually arrive on."""
    return TelegramBotAdapter(
        token="<redacted>",
        chat_allow="42", adapter_name="telegram-personal",
        owner_only=True, owner_sender_ids=(_OWNER_ID,))


def _bot_query(adapter: TelegramBotAdapter, data: str) -> dict:
    return {
        "id": "q1",
        "from": {"id": int(_OWNER_ID), "is_bot": False,
                 "first_name": "Owner"},
        "message": {"chat": {"id": 42, "type": "private"},
                    "message_id": 5},
        "data": adapter._sign_callback(data),
    }


class _Rig(unittest.TestCase):
    """PartnerRuntime over a tmp home (the lane-1 rig shape) with
    the DraftQueue factory doubled to a tmp file — dispatch,
    routing and every board's own store stay real."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-btnfix5-")
        self.db_path = str(Path(self.tmp.name) / "drafts.db")
        settings = load_settings(overrides={
            "home": self.tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
            "partner.owner_chats": _OWNER_ID,
            "partner.owner_sender_ids": _OWNER_ID,
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
        self._patcher = mock.patch.object(
            drafts_mod, "DraftQueue",
            lambda *a, **k: DraftQueue(db_path=self.db_path))
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    # — driving ────────────────────────────────────────────────
    def _run(self, text: str, chat_key: str = "telegram:999") -> Any:
        return self.runtime.handle_control(
            text, chat_key,
            message=_msg(text, chat_key=chat_key, owner=True))

    def _tap(self, data: str) -> list:
        """One handset tap: signed Bot API callback → adapter →
        runtime. Returns the replies the runtime produced."""
        adapter = _personal_bot()
        replies: list = []

        def handler(msg: Any) -> None:
            replies.append(self.runtime.handle_control(
                msg.text, msg.chat.key, message=msg))

        adapter._handle_callback(_bot_query(adapter, data), handler)
        return replies

    # — reading replies ────────────────────────────────────────
    @staticmethod
    def _text(reply: Any) -> str:
        return reply.text if hasattr(reply, "text") else str(reply)

    @staticmethod
    def _payloads(reply: Any) -> list[str]:
        buttons = getattr(reply, "buttons", None) or []
        return [data for row in buttons for _label, data in row]

    def _queue(self) -> DraftQueue:
        return DraftQueue(db_path=self.db_path)


# ── careers ────────────────────────────────────────────────────


class CareersTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.careers.store import CareerStore

        self.store = CareerStore(
            Path(self.tmp.name) / "careers" / "applications.json")
        self.runtime._careers_store = lambda: self.store
        self.a = self.store.add("Acme", "Senior Dev")
        self.b = self.store.add("Beta", "QA Engineer")
        self.c = self.store.add("Ceta", "PM")
        self.d = self.store.add("Delta", "Ops Lead")

    def test_board_harvest(self) -> None:
        payloads = self._payloads(self._run("/careers"))
        self.assertEqual(
            payloads,
            ["careers stages"]
            + [f"careers move {app.id}"
               for app in (self.a, self.b, self.c, self.d)])

    def test_move_tap_lands_typed_state(self) -> None:
        picker = self._tap(f"careers move {self.a.id}")
        self.assertEqual(len(picker), 1)
        stage_payloads = self._payloads(picker[0])
        self.assertIn(f"careers move {self.a.id} applied",
                      stage_payloads)
        self.assertIn(f"careers remove {self.a.id}", stage_payloads)
        self._tap(f"careers move {self.a.id} applied")
        self._run(f"/careers move {self.b.id} applied")
        self.assertEqual(self.store.get(self.a.id).stage, "applied")
        self.assertEqual(self.store.get(self.a.id).stage,
                         self.store.get(self.b.id).stage)

    def test_remove_tap_lands_typed_state(self) -> None:
        picker = self._run(f"/careers move {self.c.id}")
        self.assertIn(f"careers remove {self.c.id}",
                      self._payloads(picker))
        self._tap(f"careers remove {self.c.id}")
        self._run(f"/careers remove {self.d.id}")
        self.assertIsNone(self.store.get(self.c.id))
        self.assertIsNone(self.store.get(self.d.id))


# ── ideas ──────────────────────────────────────────────────────


class IdeasTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.goals import IdeaTracker
        from nomorals.goals.surface import OWNER_USER_ID, run_sync

        self.tracker = IdeaTracker(self.context.db)
        self.user = OWNER_USER_ID
        self.i1 = run_sync(self.tracker.create(
            self.user, "first raw idea", source="chat"))
        self.i2 = run_sync(self.tracker.create(
            self.user, "second raw idea", source="chat"))

    def test_board_harvest(self) -> None:
        self.assertEqual(
            self._payloads(self._run("/ideas")),
            ["ideas show 1", "ideas develop 1",
             "ideas show 2", "ideas develop 2"])

    def test_show_tap_matches_typed(self) -> None:
        tapped = self._tap("ideas show 1")
        typed = self._run("/ideas show 1")
        self.assertEqual(self._text(tapped[0]), self._text(typed))
        self.assertEqual(self._payloads(tapped[0]),
                         ["ideas develop 1"])

    def test_develop_tap_lands_a_record_like_typed(self) -> None:
        from nomorals.agents.idea_development import IdeaDevelopment

        cell = IdeaDevelopment(self.context.db)
        self.assertIsNone(cell.latest_for(self.i1.idea_id))
        self._tap("ideas develop 1")
        self._run("/ideas develop 2")
        rec1 = cell.latest_for(self.i1.idea_id)
        rec2 = cell.latest_for(self.i2.idea_id)
        self.assertIsNotNone(rec1)
        self.assertIsNotNone(rec2)
        assert rec1 is not None and rec2 is not None
        self.assertTrue((rec1.get("verdict") or {}).get("exit"))
        self.assertTrue((rec2.get("verdict") or {}).get("exit"))
        # The raw captures never move.
        from nomorals.goals.surface import run_sync
        ideas = run_sync(self.tracker.list_ideas(self.user, limit=10))
        self.assertEqual({i.title for i in ideas},
                         {"first raw idea", "second raw idea"})


# ── day ────────────────────────────────────────────────────────


class DayTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.agents.dayrecord import DayStore, today_key, yesterday_key

        self.store = DayStore(self.context.db)
        self.today = today_key()
        self.yesterday = yesterday_key()
        self.store.add_fragment(text="did a thing today",
                                source="chat", day=self.today)
        self.store.add_fragment(text="did a thing yesterday",
                                source="chat", day=self.yesterday)

    def test_board_harvest(self) -> None:
        self.assertEqual(
            self._payloads(self._run("/day")),
            [f"day make {self.today}", f"day skip {self.today}",
             f"day export {self.today}", "day yesterday"])

    def test_skip_tap_and_typed_alternate_the_same_state(self) -> None:
        self.assertFalse(self.store.skipped(self.today))
        self._tap(f"day skip {self.today}")
        self.assertTrue(self.store.skipped(self.today))
        # The re-rendered board now offers Unskip — the toggle
        # follows the day's recorded state.
        self.assertIn(f"day unskip {self.today}",
                      self._payloads(self._run("/day")))
        self._run(f"/day unskip {self.today}")
        self.assertFalse(self.store.skipped(self.today))
        self._run(f"/day skip {self.today}")
        self.assertTrue(self.store.skipped(self.today))
        self._tap(f"day unskip {self.today}")
        self.assertFalse(self.store.skipped(self.today))

    def test_make_tap_and_typed_derive_the_same_way(self) -> None:
        from nomorals.social.drafts import PENDING_REVIEW

        def pending() -> int:
            queue = self._queue()
            try:
                return len(queue.list_by_status(
                    PENDING_REVIEW, limit=100))
            finally:
                queue.close()

        before = pending()
        tapped = self._tap(f"day make {self.yesterday}")
        tap_delta = pending() - before
        mid = pending()
        self._run(f"/day make {self.today}")
        typed_delta = pending() - mid
        self.assertEqual(tap_delta, typed_delta)
        self.assertIn("draft", self._text(tapped[0]).lower())


# ── calendar ───────────────────────────────────────────────────


class CalendarTapTests(_Rig):
    def _seed(self) -> None:
        from nomorals.social.series import SeriesStore

        store = SeriesStore(self.context.db)
        store.create_series("Alpha", objective="grow the channel")
        store.create_series("Beta")

    def test_empty_board_is_buttonless(self) -> None:
        reply = self._run("/calendar")
        self.assertIsInstance(reply, str)
        self.assertFalse(self._payloads(reply))

    def test_board_and_series_list_harvest(self) -> None:
        self._seed()
        self.assertEqual(self._payloads(self._run("/calendar")),
                         ["calendar series Alpha",
                          "calendar series Beta"])
        self.assertEqual(self._payloads(self._run("/calendar series")),
                         ["calendar series Alpha",
                          "calendar series Beta"])

    def test_open_tap_matches_typed(self) -> None:
        self._seed()
        tapped = self._tap("calendar series Alpha")
        typed = self._run("/calendar series Alpha")
        self.assertEqual(self._text(tapped[0]), self._text(typed))

    def test_series_detail_is_buttonless(self) -> None:
        # Adjudicated display-only: the detail view lists members;
        # attaching needs piece numbers from other boards, so no
        # button is shipped on it.
        self._seed()
        reply = self._run("/calendar series Alpha")
        self.assertIsInstance(reply, str)
        self.assertFalse(self._payloads(reply))


# ── repurpose ──────────────────────────────────────────────────


class RepurposeTapTests(_Rig):
    def _seed_posted(self, text: str) -> str:
        queue = self._queue()
        try:
            draft = queue.create_draft(text, ["x"])
            queue.mark_posted(draft.id)
            return draft.id
        finally:
            queue.close()

    def _pending(self) -> int:
        from nomorals.social.drafts import PENDING_REVIEW

        queue = self._queue()
        try:
            return len(queue.list_by_status(PENDING_REVIEW,
                                            limit=100))
        finally:
            queue.close()

    def test_empty_board_is_buttonless(self) -> None:
        reply = self._run("/repurpose")
        self.assertIsInstance(reply, str)
        self.assertFalse(self._payloads(reply))

    def test_board_harvest(self) -> None:
        self._seed_posted("a posted thought about gold")
        self._seed_posted("a posted thought about rain")
        self.assertEqual(
            self._payloads(self._run("/repurpose")),
            ["repurpose post 1 short", "repurpose post 1 thread",
             "repurpose post 2 short", "repurpose post 2 thread"])

    def test_repurpose_tap_lands_typed_state(self) -> None:
        self._seed_posted("a posted thought about gold")
        self._seed_posted("a posted thought about rain")
        before = self._pending()
        self._tap("repurpose post 1 short")
        self.assertEqual(self._pending(), before + 1)
        self._run("/repurpose post 2 short")
        self.assertEqual(self._pending(), before + 2)


# ── engage ─────────────────────────────────────────────────────


class EngageTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.social.engagement import EngagementStore

        self.store = EngagementStore(self.context.db)
        self.ids: list[str] = []
        for n in range(1, 9):
            self.store.add_item(
                platform="x", post_id=f"p{n}",
                post_external_id=f"pe{n}", account_id="acct",
                item_external_id=f"i{n}", author=f"Author {n}",
                text=f"a reply needing an answer {n}", kind="reply",
                arrived_at=1700000000.0 + n)
        from nomorals.social.engagement import board_items
        self.items = board_items(self.store)
        self.assertEqual(len(self.items), 8)

    def _state(self, idx: int) -> str:
        row = self.store.get(self.items[idx - 1]["id"])
        assert row is not None
        return str(row["state"])

    def _pos(self, idx: int) -> int:
        """The item's CURRENT board position — settled items
        leave the board, so positions are always resolved fresh
        by identity, exactly as a user re-reading the board
        would."""
        from nomorals.social.engagement import board_items

        target = self.items[idx - 1]["id"]
        for pos, row in enumerate(board_items(self.store), 1):
            if row["id"] == target:
                return pos
        raise AssertionError(f"item {idx} is off the board")

    def test_board_harvest(self) -> None:
        payloads = self._payloads(self._run("/engage"))
        self.assertEqual(len(payloads), 32)
        for idx in range(1, 9):
            for verb in ("show", "reply", "ignore", "done"):
                self.assertIn(f"engage {verb} {idx}", payloads)

    def test_ignore_and_done_taps_land_typed_states(self) -> None:
        from nomorals.social.engagement import STATE_DONE, STATE_IGNORED

        self._tap(f"engage ignore {self._pos(1)}")
        self._run(f"/engage ignore {self._pos(2)}")
        self.assertEqual(self._state(1), STATE_IGNORED)
        self.assertEqual(self._state(1), self._state(2))
        self._tap(f"engage done {self._pos(3)}")
        self._run(f"/engage done {self._pos(4)}")
        self.assertEqual(self._state(3), STATE_DONE)
        self.assertEqual(self._state(3), self._state(4))

    def test_show_tap_triages_like_typed(self) -> None:
        from nomorals.social.engagement import STATE_TRIAGED

        self._tap(f"engage show {self._pos(5)}")
        self._run(f"/engage show {self._pos(6)}")
        self.assertEqual(self._state(5), STATE_TRIAGED)
        self.assertEqual(self._state(5), self._state(6))

    def test_reply_tap_and_typed_both_draft(self) -> None:
        before = self._pending_count()
        self._tap(f"engage reply {self._pos(7)}")
        mid = self._pending_count()
        self._run(f"/engage reply {self._pos(8)}")
        after = self._pending_count()
        self.assertEqual(mid - before, after - mid)
        self.assertGreater(mid, before)

    def _pending_count(self) -> int:
        from nomorals.social.drafts import PENDING_REVIEW

        queue = self._queue()
        try:
            return len(queue.list_by_status(PENDING_REVIEW,
                                            limit=100))
        finally:
            queue.close()


# ── cases ──────────────────────────────────────────────────────


class CasesTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.agents.outcome_cases import OutcomeCaseStore

        self.store = OutcomeCaseStore(self.context.db)
        self.c1, _ = self.store.open_case(
            domain="missions", kind="job", subject="forgotten export",
            subject_key="job-1", reason="stalled")
        self.c2, _ = self.store.open_case(
            domain="missions", kind="job", subject="forgotten import",
            subject_key="job-2", reason="stalled")

    def test_board_harvest(self) -> None:
        self.assertEqual(
            self._payloads(self._run("/cases")),
            ["cases 1", "cases verify 1", "cases ack 1",
             "cases 2", "cases verify 2", "cases ack 2"])

    def test_open_tap_matches_typed(self) -> None:
        tapped = self._tap("cases 1")
        typed = self._run("/cases 1")
        self.assertEqual(self._text(tapped[0]), self._text(typed))

    def _pos(self, case: Any) -> int:
        """The case's CURRENT board position — acknowledged cases
        go quiet and leave the board, so positions resolve fresh
        by identity."""
        from nomorals.agents.outcome_steward import case_board

        for pos, row in enumerate(case_board(self.store), 1):
            if row.id == case.id:
                return pos
        raise AssertionError("case is off the board")

    def test_ack_tap_lands_typed_state(self) -> None:
        from nomorals.agents.outcome_steward import ack_snapshot

        self.assertIsNone(ack_snapshot(self.store.get(self.c1.id)))
        self._tap(f"cases ack {self._pos(self.c1)}")
        self._run(f"/cases ack {self._pos(self.c2)}")
        self.assertIsNotNone(ack_snapshot(self.store.get(self.c1.id)))
        self.assertIsNotNone(ack_snapshot(self.store.get(self.c2.id)))
        # Acknowledgement never resolves — the state stands.
        self.assertEqual(self.store.get(self.c1.id).state,
                         self.store.get(self.c2.id).state)

    def test_verify_tap_runs_the_stewards_cycle(self) -> None:
        before1 = len(self.store.receipts(self.c1.id))
        before2 = len(self.store.receipts(self.c2.id))
        tapped = self._tap(f"cases verify {self._pos(self.c1)}")
        self._run(f"/cases verify {self._pos(self.c2)}")
        self.assertGreater(len(self.store.receipts(self.c1.id)),
                           before1)
        self.assertGreater(len(self.store.receipts(self.c2.id)),
                           before2)
        self.assertEqual(self.store.get(self.c1.id).state,
                         self.store.get(self.c2.id).state)
        self.assertIn("re-check", self._text(tapped[0]))


# ── runs ───────────────────────────────────────────────────────


class RunsTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.agents.delegation import DelegationStore
        from nomorals.missions.mission import Mission, MissionStatus, MissionStore

        self.contract = DelegationStore(self.context.db) \
            .create_contract(goal="wire the cluster",
                             worker="builder", verifier="checker")
        now = time.time()
        self.mission = MissionStore(self.context.db).create(Mission(
            goal="keep the board honest", name="board-keeper",
            status=MissionStatus.RUNNING,
            created_at=now, updated_at=now))

    def _ordinal(self, board_text: str, marker: str) -> int:
        for line in board_text.splitlines():
            if marker in line:
                head = line.split(".", 1)[0].strip()
                if head.isdigit():
                    return int(head)
        raise AssertionError(f"no board row carries {marker!r}")

    def test_board_harvest(self) -> None:
        reply = self._run("/runs")
        payloads = self._payloads(reply)
        self.assertIn("runs coverage", payloads)
        text = self._text(reply)
        m_ord = self._ordinal(text, "board-keeper")
        c_ord = self._ordinal(text, "wire the cluster")
        self.assertIn(f"runs {m_ord}", payloads)
        self.assertIn(f"runs stop {m_ord}", payloads)
        self.assertIn(f"runs stop {c_ord}", payloads)

    def test_stop_tap_lands_typed_state_with_receipt(self) -> None:
        from nomorals.agents.delegation import DelegationStore
        from nomorals.core.cancellation import receipts_for_ref
        from nomorals.missions.mission import MissionStore

        board = self._run("/runs")
        m_ord = self._ordinal(self._text(board), "board-keeper")
        self._tap(f"runs stop {m_ord}")
        # The stopped row settles off the live list — resolve the
        # contract's ordinal from the re-rendered board.
        board = self._run("/runs")
        c_ord = self._ordinal(self._text(board), "wire the cluster")
        self._run(f"/runs stop {c_ord}")
        mission = MissionStore(self.context.db).get(self.mission.id)
        contract = DelegationStore(self.context.db).get_contract(
            self.contract.id)
        self.assertEqual(str(mission.status), "cancelled")
        self.assertEqual(contract.status, "cancelled")
        self.assertEqual(
            len(receipts_for_ref(
                self.context.db, f"mission:{self.mission.id}")), 1)
        self.assertEqual(
            len(receipts_for_ref(
                self.context.db, f"contract:{self.contract.id}")), 1)

    def test_open_tap_matches_typed(self) -> None:
        board = self._run("/runs")
        m_ord = self._ordinal(self._text(board), "board-keeper")
        tapped = self._tap(f"runs {m_ord}")
        typed = self._run(f"/runs {m_ord}")
        self.assertEqual(self._text(tapped[0]), self._text(typed))


# ── worldmodel ─────────────────────────────────────────────────


class WorldmodelTapTests(_Rig):
    def test_empty_board_is_buttonless(self) -> None:
        reply = self._run("/worldmodel")
        self.assertIsInstance(reply, str)
        self.assertFalse(self._payloads(reply))

    def test_area_tap_matches_typed(self) -> None:
        from nomorals.agents.worldmodel import default_world

        world = default_world(self.context)
        world.assert_fact("runtime:host", "ec2", kind="observed",
                          source="probe")
        world.assert_fact("codebase:lang", "python", kind="observed",
                          source="probe")
        self.assertEqual(self._payloads(self._run("/worldmodel")),
                         ["worldmodel codebase", "worldmodel runtime"])
        tapped = self._tap("worldmodel runtime")
        typed = self._run("/worldmodel runtime")
        self.assertEqual(self._text(tapped[0]), self._text(typed))


# ── craft ──────────────────────────────────────────────────────


class CraftTapTests(_Rig):
    def test_board_harvest(self) -> None:
        payloads = self._payloads(self._run("/craft"))
        self.assertEqual(payloads[0], "craft coverage")
        self.assertIn("craft status software", payloads)

    def test_status_and_coverage_taps_match_typed(self) -> None:
        tapped = self._tap("craft status software")
        typed = self._run("/craft status software")
        self.assertEqual(self._text(tapped[0]), self._text(typed))
        tapped = self._tap("craft coverage")
        typed = self._run("/craft coverage")
        self.assertEqual(self._text(tapped[0]), self._text(typed))


# ── glossary ───────────────────────────────────────────────────


class GlossaryTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        from nomorals.llm.glossary import GlossaryStore

        self.store = GlossaryStore(self.context.db, self.tmp.name)
        self.store.put_term("legal", "bank", "Banco", scope="owner")
        self.store.put_term("legal", "court", "Corte", scope="domain")
        self.store.put_term("home", "house", "Casa", scope="global")
        self.runtime._glossary_store_instance = self.store

    def _terms(self, gid: str) -> set[str]:
        return {r.source_term for r in self.store.entries(gid)}

    def test_board_harvest(self) -> None:
        payloads = self._payloads(self._run("/glossary"))
        self.assertEqual(sorted(payloads),
                         ["glossary list home", "glossary list legal"])

    def test_list_tap_matches_typed_and_carries_del(self) -> None:
        tapped = self._tap("glossary list legal")
        typed = self._run("/glossary list legal")
        self.assertEqual(self._text(tapped[0]), self._text(typed))
        self.assertEqual(self._payloads(tapped[0]),
                         ["glossary del legal owner bank",
                          "glossary del legal domain court"])

    def test_del_tap_lands_typed_state(self) -> None:
        self._tap("glossary del legal owner bank")
        self._run("/glossary del legal domain court")
        self.assertEqual(self._terms("legal"), set())
        self.assertEqual(self._terms("home"), {"house"})


# ── notify decisions ───────────────────────────────────────────


class NotifyDecisionsTapTests(_Rig):
    def _seed_decision(self, subject: str) -> None:
        from nomorals.agents.notifier import Notifier

        notifier = Notifier(self.context, _RecordingGateway())
        meta = {"subject": subject, "prev_revision": "hash-a",
                "current_revision": "hash-a"}
        res = notifier.publish("watch", f"{subject} — no change",
                               "still", output_meta=dict(meta))
        assert res["suppressed"] is True

    def test_empty_decisions_board_is_buttonless(self) -> None:
        reply = self._run("/notify decisions")
        self.assertIsInstance(reply, str)
        self.assertFalse(self._payloads(reply))

    def test_why_tap_matches_typed(self) -> None:
        self._seed_decision("watch:one.example")
        self._seed_decision("watch:two.example")
        self.assertEqual(self._payloads(self._run("/notify decisions")),
                         ["notify why decision 1",
                          "notify why decision 2"])
        tapped = self._tap("notify why decision 1")
        typed = self._run("/notify why decision 1")
        self.assertEqual(self._text(tapped[0]), self._text(typed))
        self.assertIn("decision", self._text(tapped[0]).lower())


# ── explore ────────────────────────────────────────────────────


class ExploreTapTests(_Rig):
    def setUp(self) -> None:
        super().setUp()
        self._env = mock.patch.dict(
            os.environ,
            {"NM_NICHE_DRAFTS_DIR": str(
                Path(self.tmp.name) / "niche-drafts")})
        self._env.start()
        from nomorals.media.explore import board as explore_board

        self.store = explore_board.resolve_store(self.context)
        self._proposals_snapshot = list(self.store._niche_proposals)

    def tearDown(self) -> None:
        # The explore store is a process-wide singleton — restore
        # its proposal list so nothing seeded here leaks into
        # other suites sharing the process.
        self.store._niche_proposals = self._proposals_snapshot
        self._env.stop()
        super().tearDown()

    def _proposal(self, slug: str, score: float) -> dict[str, Any]:
        from nomorals.media.explore.niches import NICHE_PROPOSED

        return {
            "slug": slug, "status": NICHE_PROPOSED,
            "thesis": f"the {slug} field is moving", "cadence": 1.0,
            "members": [{"name": "Chan A"}, {"name": "Chan B"}],
            "evidence_pack": {"score": score, "sightings": 3,
                              "sweeps": ["s1", "s2"]},
            "basis": "two sweeps corroborate",
        }

    def test_board_harvest(self) -> None:
        self.assertEqual(
            self._payloads(self._run("/explore")),
            ["explore run", "explore trends", "explore candidates",
             "explore niches"])

    def test_view_taps_match_typed(self) -> None:
        for payload, typed in (
                ("explore trends", "/explore trends"),
                ("explore candidates", "/explore candidates"),
                ("explore niches", "/explore niches")):
            tapped = self._tap(payload)
            self.assertEqual(self._text(tapped[0]),
                             self._text(self._run(typed)))

    def test_empty_niches_view_is_buttonless(self) -> None:
        reply = self._run("/explore niches")
        self.assertIsInstance(reply, str)
        self.assertFalse(self._payloads(reply))

    def test_adopt_and_dismiss_taps_land_typed_states(self) -> None:
        from nomorals.media.explore.niches import NICHE_ADOPTED, NICHE_DISMISSED

        self.store.record_niche_proposal(
            self._proposal("gold_trading", 9.0))
        self.store.record_niche_proposal(
            self._proposal("silver_trading", 5.0))
        self.assertEqual(
            self._payloads(self._run("/explore niches")),
            ["explore adopt 1", "explore dismiss 1",
             "explore adopt 2", "explore dismiss 2"])
        self._tap("explore adopt 1")
        # Adoption settles the gold proposal off the proposed
        # list — silver is re-read at its fresh board position.
        fresh = self._payloads(self._run("/explore niches"))
        dismiss = [p for p in fresh if p.startswith("explore dismiss")]
        self.assertEqual(len(dismiss), 1)
        self._run("/" + dismiss[0])
        gold = self.store.niche_proposal("gold_trading")
        silver = self.store.niche_proposal("silver_trading")
        assert gold is not None and silver is not None
        self.assertEqual(gold["status"], NICHE_ADOPTED)
        self.assertEqual(silver["status"], NICHE_DISMISSED)
        self.assertTrue(gold.get("draft_path"))


# ── manifest rows ──────────────────────────────────────────────


class ManifestRowTests(unittest.TestCase):
    def test_all_thirteen_boards_are_registered(self) -> None:
        from nomorals.social.chat import surface_manifest as sm

        ids = {s.id for s in sm.SURFACES}
        for board in ("careers", "ideas", "day", "calendar",
                      "repurpose", "engage", "cases", "runs",
                      "worldmodel", "craft", "glossary",
                      "notify-decisions", "explore"):
            self.assertIn(board, ids)


if __name__ == "__main__":
    unittest.main()
