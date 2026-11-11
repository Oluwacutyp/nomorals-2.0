"""Presentation Phase 3 remainder — the runtime_system boards.

Phase 3 (4bee8951) converted /status, /help and /goals to Card
renders in the canonical dialect.  This suite pins the rest:
drafts, day, ideas, calendar, engage, cases, runs, worldmodel,
craft, glossary, explore, repurpose — plus the /goals
briefing/review (tracker-owned formatting, framed here).

The contract, per board: the Card frame is present (title +
stamp) AND every non-empty line of the pre-conversion reply
rides verbatim inside it — the wrap changes framing only.
Where the pre-conversion text is a helper's own render, the
test calls that helper directly on the same store and compares
line for line.  Replies that genuinely have no card shape stay
plain, pinned as plain: empty-board honest lines, usage lines,
errors, act outcomes.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.dayrecord import render_day, today_key
from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
from nomorals.agents.partner.runtime_system import RuntimeSystemMixin
from nomorals.craft.temporary import CompetenceGrantStore
from nomorals.goals import GoalTracker
from nomorals.goals.surface import (
    OWNER_USER_ID,
    render_briefing,
    render_review,
    run_sync,
)
from nomorals.skills.filebook import FileSkillLibrary
from nomorals.social.drafts import DraftQueue, render_review_batch
from nomorals.social.engagement import (
    EngagementStore,
)
from nomorals.social.engagement import (
    render_board as render_engage_board,
)
from nomorals.social.engagement import (
    render_item as render_engage_item,
)
from nomorals.social.series import SeriesStore, render_calendar
from nomorals.storage.db import Database


def _tmp(testcase: unittest.TestCase) -> Path:
    tmp = Path(tempfile.mkdtemp(dir=os.environ.get("NM_TEST_TMPDIR")))
    testcase.addCleanup(
        lambda: shutil.rmtree(tmp, ignore_errors=True))
    return tmp


def _tmp_db(testcase: unittest.TestCase) -> Database:
    db = Database(str(_tmp(testcase) / "test.db"))
    db.migrate()
    return db


class _Rt(RuntimeSystemMixin, RuntimeMediaMixin):
    """The system + media mixins, with per-test stores.

    The draft queue is a per-test file (never the live home);
    the craft stores are an empty per-test library (the board
    still renders: greenfield domains, named absences).
    """

    def __init__(self, db, tmp):
        self.context = SimpleNamespace(db=db)
        self._tmp = tmp
        self._queue = DraftQueue(tmp / "drafts.db")
        (tmp / "seed").mkdir(parents=True, exist_ok=True)
        self._craft_library = FileSkillLibrary(
            seed_dir=tmp / "seed", user_dir=tmp / "user")
        self._craft_grants = CompetenceGrantStore(tmp / "grants.json")

    def _draft_queue(self):
        # Production semantics: a fresh queue per call — the
        # handlers close it in a finally, so a shared instance
        # would die on first use (closed-database).  The
        # seeding handle above is a separate instance on the
        # same file, for the tests' own reads.
        return DraftQueue(self._tmp / "drafts.db")


def _rt(testcase: unittest.TestCase):
    tmp = _tmp(testcase)
    db = _tmp_db(testcase)
    return _Rt(db, tmp), db, tmp


def _text(reply):
    """A board render's text — boards return actionable replies
    since button lane 5; read the text half here."""
    return reply.text if hasattr(reply, "text") else reply


def _assert_card(testcase, wrapped, title):
    """The canonical frame: title + house stamp."""
    testcase.assertIn("**" + title + "**", wrapped)
    testcase.assertIn("_Powered by peacethefirst_", wrapped)


def _assert_lines_verbatim(testcase, raw, wrapped):
    """Every non-empty line of the pre-conversion text rides
    verbatim inside the Card."""
    for line in raw.splitlines():
        if line.strip():
            testcase.assertIn(line, wrapped)


# ── the shared wrap ──────────────────────────────────────────────


class CardForTextTests(unittest.TestCase):

    def test_frames_paragraphs_and_keeps_every_line(self):
        from nomorals.social.chat.typography import card_for_text

        raw = ("🌅 your day — 2026-11-11 (Africa/Lagos)\n"
               "\n"
               "🗣 your words\n"
               "  shipped three lanes\n"
               "\n"
               "🎯 goals — 1 goal(s)\n"
               "  Save for a house — 40% (self-reported)")
        out = card_for_text(raw, "briefing", title="Day — 2026-11-11")
        _assert_card(self, out, "Day — 2026-11-11")
        _assert_lines_verbatim(self, raw, out)

    def test_empty_stays_empty(self):
        from nomorals.social.chat.typography import card_for_text

        self.assertEqual(card_for_text("", "briefing", title="Day"),
                         "")
        self.assertEqual(card_for_text("  \n ", "briefing",
                                       title="Day"), "  \n ")


# ── ideas ────────────────────────────────────────────────────────


class IdeasTests(unittest.TestCase):

    def test_board_frames_the_list_verbatim(self):
        rt, _db, _tmp = _rt(self)
        rt._control_ideas("add Test idea | a description to grow")
        board = _text(rt._control_ideas(""))
        _assert_card(self, board, "Ideas")
        # The board lists titles only; the description lives
        # on the show view, pinned there.
        self.assertIn("Test idea", board)
        self.assertIn("grow one: /ideas develop <#> — the raw "
                      "words never change", board)

    def test_show_frames_the_record(self):
        rt, _db, _tmp = _rt(self)
        rt._control_ideas("add Test idea | a description to grow")
        shown = _text(rt._control_ideas("show 1"))
        _assert_card(self, shown, "Idea #1")
        self.assertIn("Test idea", shown)
        self.assertIn("not developed yet", shown)

    def test_develop_frames_the_record(self):
        rt, _db, _tmp = _rt(self)
        rt._control_ideas("add Test idea | a description to grow")
        grown = _text(rt._control_ideas("develop 1"))
        _assert_card(self, grown, "Idea development")
        self.assertIn("Test idea", grown)


# ── drafts ───────────────────────────────────────────────────────


class DraftsTests(unittest.TestCase):

    def test_list_frames_the_batch_verbatim(self):
        rt, _db, _tmp = _rt(self)
        rt._queue.create_draft("hello world draft content here",
                               platforms=["telegram"])
        listed = _text(rt._control_drafts(""))
        _assert_card(self, listed, "Drafts")
        raw = render_review_batch(rt._queue.pending())
        _assert_lines_verbatim(self, raw, listed)
        self.assertIn("open one: /drafts show <n>", listed)

    def test_show_frames_the_review_card(self):
        rt, _db, _tmp = _rt(self)
        draft = rt._queue.create_draft(
            "hello world draft content here",
            platforms=["telegram"])
        # A fresh draft is proposed for posting first; the
        # review card is the pending-review view.
        rt._queue.propose(draft.id)
        shown = _text(rt._control_drafts("show 1"))
        _assert_card(self, shown, "Draft #1")
        self.assertIn("hello world draft content here", shown)
        self.assertIn("Virality", shown)

    def test_empty_list_stays_a_plain_honest_line(self):
        rt, _db, _tmp = _rt(self)
        listed = _text(rt._control_drafts(""))
        self.assertEqual(listed, "📝 no drafts waiting for review.")
        self.assertNotIn("_Powered by peacethefirst_", listed)


# ── day ──────────────────────────────────────────────────────────


class DayTests(unittest.TestCase):

    def test_day_frames_the_record_verbatim(self):
        rt, db, _tmp = _rt(self)
        rt._control_day("tell shipped three lanes")
        day = today_key()
        shown = _text(rt._control_day(""))
        _assert_card(self, shown, "Day — " + day)
        raw = render_day(rt.context, day,
                         tracker=rt._goal_tracker(),
                         artifact_store=rt._artifact_store())
        _assert_lines_verbatim(self, raw, shown)
        self.assertIn("shipped three lanes", shown)
        self.assertRegex(shown, r"🌅 your day — \d{4}-\d{2}-\d{2}")


# ── calendar ─────────────────────────────────────────────────────


class CalendarTests(unittest.TestCase):

    def test_list_frames_the_calendar_verbatim(self):
        rt, db, _tmp = _rt(self)
        listed = _text(rt._control_calendar(""))
        _assert_card(self, listed, "Calendar")
        raw = render_calendar(db, rt._queue, shorts=[],
                              store=SeriesStore(db))
        _assert_lines_verbatim(self, raw, listed)

    def test_series_list_and_detail_frame_their_lines(self):
        rt, _db, _tmp = _rt(self)
        rt._control_calendar("new Test series | an objective")
        listed = _text(rt._control_calendar("series"))
        _assert_card(self, listed, "Series")
        self.assertIn("Test series", listed)
        self.assertIn("0 piece(s)", listed)
        detail = _text(rt._control_calendar("series Test series"))
        _assert_card(self, detail, "Series — Test series")
        self.assertIn("no pieces yet", detail)

    def test_empty_series_list_stays_a_plain_honest_line(self):
        rt, _db, _tmp = _rt(self)
        listed = _text(rt._control_calendar("series"))
        self.assertIn("no series yet", listed)
        self.assertNotIn("_Powered by peacethefirst_", listed)


# ── repurpose ────────────────────────────────────────────────────


class RepurposeTests(unittest.TestCase):

    def test_list_frames_the_board_verbatim(self):
        rt, _db, _tmp = _rt(self)
        listed = _text(rt._control_repurpose(""))
        _assert_card(self, listed, "Repurpose")
        self.assertIn("♻️ **repurpose — one asset, other forms**",
                      listed)
        self.assertIn("📤 **posted drafts** (→ short script"
                      " or thread)", listed)
        self.assertIn("  (nothing posted yet)", listed)
        self.assertIn("🎬 **shorts** (→ text post)", listed)
        self.assertIn("  (no shorts planned yet)", listed)
        self.assertIn("📔 **day record**", listed)
        self.assertIn("verbs: /repurpose post <n>", listed)


# ── engage ───────────────────────────────────────────────────────


class EngageTests(unittest.TestCase):

    def _seed_item(self, db):
        store = EngagementStore(db)
        store.add_item(
            platform="telegram", post_id="post-1",
            post_external_id="ext-1", account_id="acct-1",
            item_external_id="item-1", author="Ada",
            text="hello from Ada", kind="comment",
            arrived_at=1760000000.0)
        return store

    def test_board_frames_the_store_verbatim(self):
        rt, db, _tmp = _rt(self)
        store = self._seed_item(db)
        board = _text(rt._control_engage(""))
        _assert_card(self, board, "Engage")
        raw = render_engage_board(store, truth=None)
        _assert_lines_verbatim(self, raw, board)
        self.assertIn("Ada", board)

    def test_show_frames_the_item_verbatim(self):
        rt, db, _tmp = _rt(self)
        store = self._seed_item(db)
        shown = _text(rt._control_engage("show 1"))
        _assert_card(self, shown, "Engagement #1")
        from nomorals.social.engagement import board_items

        raw = render_engage_item(board_items(store)[0], 1)
        _assert_lines_verbatim(self, raw, shown)
        self.assertIn("hello from Ada", shown)


# ── cases ────────────────────────────────────────────────────────


class CasesTests(unittest.TestCase):

    def _seed_case(self, db):
        from nomorals.agents.outcome_cases import OutcomeCaseStore

        store = OutcomeCaseStore(db)
        case, _created = store.open_case(
            domain="delegation", kind="missing_outcome",
            subject="test case", subject_key="test-case-1",
            reason="unknown-outcome")
        return store, case

    def test_board_frames_the_store_verbatim(self):
        rt, db, _tmp = _rt(self)
        store, case = self._seed_case(db)
        board = _text(rt._control_cases(""))
        _assert_card(self, board, "Cases")
        from nomorals.agents.outcome_steward import render_case_board

        raw = render_case_board(store)
        _assert_lines_verbatim(self, raw, board)
        self.assertIn("test case", board)

    def test_show_frames_the_detail_verbatim(self):
        rt, db, _tmp = _rt(self)
        store, case = self._seed_case(db)
        shown = _text(rt._control_cases("show 1"))
        _assert_card(self, shown, "Case — test case")
        from nomorals.agents.outcome_steward import render_case_detail

        raw = render_case_detail(
            case, store.receipts(case.id), None)
        _assert_lines_verbatim(self, raw, shown)

    def test_verify_frames_verdict_plus_detail(self):
        rt, db, _tmp = _rt(self)
        store, case = self._seed_case(db)
        shown = _text(rt._control_cases("verify 1"))
        _assert_card(self, shown, "Case — test case")
        self.assertIn("re-check ran", shown)
        self.assertIn("test case", shown)


# ── runs ─────────────────────────────────────────────────────────


class RunsTests(unittest.TestCase):

    def test_board_frames_the_projection_verbatim(self):
        rt, _db, _tmp = _rt(self)
        board = _text(rt._control_runs(""))
        _assert_card(self, board, "Runs")
        from nomorals.agents.ticket_board import render_board

        raw = render_board(rt._runs_board())
        _assert_lines_verbatim(self, raw, board)

    def test_coverage_frames_the_rollup_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_runs("coverage"))
        _assert_card(self, shown, "Runs coverage")
        from nomorals.agents.ticket_board import render_coverage

        raw = render_coverage(rt._runs_board())
        _assert_lines_verbatim(self, raw, shown)

    def test_show_frames_the_ticket_or_says_so_plainly(self):
        rt, _db, _tmp = _rt(self)
        rows = rt._runs_board()["rows"]
        if not rows:
            shown = _text(rt._control_runs("1"))
            self.assertIn("isn't on the board", shown)
            self.assertNotIn("_Powered by peacethefirst_", shown)
            return
        ordinal = rows[0]["ordinal"]
        shown = _text(rt._control_runs(str(ordinal)))
        _assert_card(self, shown, f"Ticket #{ordinal}")
        from nomorals.agents.ticket_board import render_ticket

        raw = render_ticket(rows[0])
        _assert_lines_verbatim(self, raw, shown)

    def test_stop_receipt_stays_a_plain_outcome_line(self):
        # A stop is an act's receipt, not a board — it stays
        # plain by design (named in the Phase 3 remainder).
        rt, _db, _tmp = _rt(self)
        rows = [r for r in rt._runs_board()["rows"]
                if r["family"] == "run"]
        if not rows:
            self.skipTest("no stoppable run ticket on this board")
        shown = _text(rt._control_runs(
            f"stop {rows[0]['ordinal']} test"))
        self.assertIn("stop", shown.lower())
        self.assertNotIn("_Powered by peacethefirst_", shown)


# ── worldmodel ───────────────────────────────────────────────────


class WorldmodelTests(unittest.TestCase):

    def test_board_frames_the_world_verbatim(self):
        rt, _db, _tmp = _rt(self)
        board = _text(rt._control_worldmodel(""))
        _assert_card(self, board, "World model")
        from nomorals.agents.world_board import render_world
        from nomorals.agents.worldmodel import default_world

        raw = render_world(default_world(rt.context))
        _assert_lines_verbatim(self, raw, board)

    def test_topic_frames_the_view_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_worldmodel("general"))
        _assert_card(self, shown, "World model — general")
        from nomorals.agents.world_board import render_topic
        from nomorals.agents.worldmodel import default_world

        raw = render_topic(default_world(rt.context), "general")
        _assert_lines_verbatim(self, raw, shown)


# ── craft ────────────────────────────────────────────────────────


class CraftTests(unittest.TestCase):

    def test_board_frames_the_render_verbatim(self):
        rt, db, _tmp = _rt(self)
        board = _text(rt._control_craft(""))
        _assert_card(self, board, "Craft")
        from nomorals.craft import coverage_report, render_board

        raw = render_board(coverage_report(
            db, library=rt._craft_library, grants=rt._craft_grants))
        _assert_lines_verbatim(self, raw, board)
        self.assertEqual(board, _text(rt._control_craft("board")))

    def test_coverage_frames_the_rollup_verbatim(self):
        rt, db, _tmp = _rt(self)
        shown = _text(rt._control_craft("coverage"))
        _assert_card(self, shown, "Craft coverage")
        from nomorals.craft import coverage_report, render_coverage

        raw = render_coverage(coverage_report(
            db, library=rt._craft_library, grants=rt._craft_grants))
        _assert_lines_verbatim(self, raw, shown)

    def test_status_frames_the_claim_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_craft("status software"))
        _assert_card(self, shown, "Craft — software")
        from nomorals.craft import (
            capability_status,
            render_capability_status,
        )

        raw = render_capability_status(capability_status(
            rt._craft_library, "software",
            grants=rt._craft_grants))
        _assert_lines_verbatim(self, raw, shown)

    def test_intake_frames_the_preview_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_craft("intake fix the add function"))
        _assert_card(self, shown, "Craft intake")
        self.assertIn("intake — fix the add function", shown)
        self.assertIn("preview only", shown)


# ── glossary ─────────────────────────────────────────────────────


class GlossaryTests(unittest.TestCase):

    def _seed(self, rt):
        rt._control_glossary("set test global hello = hola")

    def test_board_frames_the_store_verbatim(self):
        rt, _db, _tmp = _rt(self)
        self._seed(rt)
        board = _text(rt._control_glossary(""))
        _assert_card(self, board, "Translation glossaries")
        raw = rt._glossary_board(rt._glossary_store())
        _assert_lines_verbatim(self, raw, board)
        self.assertIn("test", board)

    def test_list_frames_the_entries_verbatim(self):
        rt, _db, _tmp = _rt(self)
        self._seed(rt)
        listed = _text(rt._control_glossary("list"))
        _assert_card(self, listed, "Glossary entries")
        self.assertIn("hello → hola", listed)


# ── explore ──────────────────────────────────────────────────────


class ExploreTests(unittest.TestCase):

    def test_board_frames_the_store_verbatim(self):
        rt, _db, _tmp = _rt(self)
        board = _text(rt._control_explore(""))
        _assert_card(self, board, "Explore")
        from nomorals.media.explore import board as explore_board

        raw = explore_board.render_explore_board(
            rt.context, scheduler=None)
        _assert_lines_verbatim(self, raw, board)

    def test_trends_frames_the_claims_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_explore("trends"))
        _assert_card(self, shown, "Explore trends")
        from nomorals.media.explore import board as explore_board

        raw = explore_board.render_explore_trends(rt.context)
        _assert_lines_verbatim(self, raw, shown)

    def test_candidates_frames_the_pool_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_explore("candidates"))
        _assert_card(self, shown, "Explore candidates")
        from nomorals.media.explore import board as explore_board

        raw = explore_board.render_explore_candidates(rt.context)
        _assert_lines_verbatim(self, raw, shown)

    def test_niches_frames_the_registry_verbatim(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_explore("niches"))
        _assert_card(self, shown, "Explore niches")
        from nomorals.media.explore import board as explore_board

        raw = explore_board.render_explore_niches(rt.context)
        _assert_lines_verbatim(self, raw, shown)

    def test_run_frames_the_sweep_report(self):
        rt, _db, _tmp = _rt(self)
        shown = _text(rt._control_explore("run"))
        _assert_card(self, shown, "Explore sweep")
        self.assertIn("🔭 **Exploration sweep complete**", shown)
        self.assertIn("metadata only — no exemplar files were "
                      "fetched", shown)


# ── goals briefing / review ──────────────────────────────────────


class GoalsSurfaceTests(unittest.TestCase):

    def _seed_goal(self, db):
        tracker = GoalTracker(db)
        run_sync(tracker.create(
            OWNER_USER_ID, "Save for a house",
            description="a deposit for our own place",
            goal_class="finance"))
        return tracker

    def test_briefing_frames_the_tracker_verbatim(self):
        _rt_, db, _tmp = _rt(self)
        tracker = self._seed_goal(db)
        shown = render_briefing(tracker, OWNER_USER_ID)
        _assert_card(self, shown, "Goals — progress briefing")
        raw = run_sync(tracker.get_briefing(OWNER_USER_ID))
        _assert_lines_verbatim(self, raw, shown)
        self.assertIn("Save for a house", shown)
        # The same framing rides the /goals briefing verb.
        via_command = _text(_rt_._control_goals("briefing"))
        self.assertEqual(via_command, shown)

    def test_review_frames_the_tracker_verbatim(self):
        _rt_, db, _tmp = _rt(self)
        tracker = self._seed_goal(db)
        shown = render_review(tracker, OWNER_USER_ID)
        _assert_card(self, shown, "Goals — weekly review")
        raw = tracker.format_weekly_review(
            run_sync(tracker.weekly_review(OWNER_USER_ID)))
        _assert_lines_verbatim(self, raw, shown)
        # A fresh active goal appears nowhere in a weekly
        # review — only completed/overdue/due-soon/stale/
        # blocked/ideas/actions do — so the pin is the
        # review's own counts, not the title.
        self.assertIn("✅ **Completed this week:** 0", shown)
        via_command = _text(_rt_._control_goals("review"))
        self.assertEqual(via_command, shown)

    def test_empty_briefing_stays_a_plain_honest_line(self):
        _rt_, db, _tmp = _rt(self)
        tracker = GoalTracker(db)
        shown = render_briefing(tracker, OWNER_USER_ID)
        self.assertEqual(shown, "No active goals.")
        self.assertNotIn("_Powered by peacethefirst_", shown)


if __name__ == "__main__":
    unittest.main()
