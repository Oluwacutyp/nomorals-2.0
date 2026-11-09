"""WS2 P4 — the careers organ: pipeline store, provider, surfaces.

Pins:

* the pipeline store (``nomorals/careers/store.py``): CRUD with an
  append-only stage history; unknown stages and moves out of a
  terminal stage refused in words; next actions with due dates;
  stale = untouched 14d+ with nothing scheduled, called out — a
  future due date is a plan, not staleness;
* the careers provider (``application_pipeline``): a
  ``career-app:<id>`` link measures the linked application's stage
  weight exactly; a ``career-target:<n>`` link counts applications
  submitted since the goal was filed (stage history reached
  ``applied`` — a withdrawn-after-applying application WAS
  submitted, a saved one was not); stale deliberate links and
  unlinked goals measure NOTHING (None → self-reported fallback);
* production wiring: importing the goals package registers the
  careers provider — proven in-process AND in a fresh interpreter;
* surfaces: /careers add → move → board shows the stage, the
  attention call-outs fire, /goals list careers carries the
  measured reading + the pipeline drill-down, attach records a
  real artifact id, and the morning briefing's goal-classes
  section carries overdue careers actions (the reminders seam —
  no new scheduler);
* the catalog: /careers registered everywhere /goals is, and
  owner-only like it.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.morning_briefing import BriefingComposer
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.careers import surface as careers_surface
from nomorals.careers.store import (
    STAGE_PROGRESS,
    CareerError,
    CareerStore,
)
from nomorals.core.config import load_settings
from nomorals.goals import GoalTracker, providers, surface
from nomorals.goals.classes import registered_provider
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    LIST_GROUPS,
    LIST_ONELINERS,
    PUBLIC_COMMANDS,
    caller_may_run,
    parse_control,
)
from nomorals.storage.db import Database

UID = "owner"
REPO_ROOT = Path(__file__).resolve().parent.parent
DAY = 86400.0

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_artifact_command import _FakeTools, _msg, _RecordingGateway  # noqa: E402


def _goal(**kwargs: Any) -> SimpleNamespace:
    base = dict(goal_id="goal_test", user_id=UID, title="",
                tags=[], created_at=0.0, goal_class="careers",
                progress=0.0, status="active")
    base.update(kwargs)
    return SimpleNamespace(**base)


class StoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.store = CareerStore(Path(self._tmp.name) / "applications.json")
        self.t0 = 1_700_000_000.0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_add_defaults_and_history_opens(self) -> None:
        app = self.store.add("Acme", "Senior Dev", source="linkedin",
                             now=self.t0)
        self.assertTrue(app.id.startswith("app_"))
        self.assertEqual(app.stage, "saved")
        self.assertEqual(app.history,
                         [{"stage": "saved", "at": self.t0, "note": "filed"}])
        self.assertEqual(app.created_at, self.t0)

    def test_add_requires_company_and_role(self) -> None:
        with self.assertRaises(CareerError):
            self.store.add("", "Dev")
        with self.assertRaises(CareerError):
            self.store.add("Acme", " ")

    def test_add_unknown_stage_refused_in_words(self) -> None:
        with self.assertRaises(CareerError) as cm:
            self.store.add("Acme", "Dev", stage="limbo")
        self.assertIn("unknown stage 'limbo'", str(cm.exception))
        self.assertIn("interview", str(cm.exception))

    def test_move_appends_history_and_stamps(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        moved = self.store.move(app.id, "applied", note="via site",
                                now=self.t0 + DAY)
        self.assertEqual(moved.stage, "applied")
        self.assertEqual([h["stage"] for h in moved.history],
                         ["saved", "applied"])
        self.assertEqual(moved.history[-1]["note"], "via site")
        self.assertEqual(moved.updated_at, self.t0 + DAY)
        self.assertEqual(moved.stage_since(), self.t0 + DAY)

    def test_move_unknown_stage_and_same_stage_refused(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        with self.assertRaises(CareerError):
            self.store.move(app.id, "limbo")
        with self.assertRaises(CareerError) as cm:
            self.store.move(app.id, "saved")
        self.assertIn("already at saved", str(cm.exception))

    def test_terminal_application_never_moves(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.store.move(app.id, "applied", now=self.t0 + DAY)
        closed = self.store.move(app.id, "rejected", now=self.t0 + 2 * DAY)
        self.assertTrue(closed.closed)
        with self.assertRaises(CareerError) as cm:
            self.store.move(app.id, "interview")
        self.assertIn("closed (rejected)", str(cm.exception))
        self.assertIn("new application", str(cm.exception))

    def test_closing_clears_next_action(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.store.set_next_action(app.id, "follow up",
                                   due=self.t0 + DAY, now=self.t0)
        closed = self.store.move(app.id, "withdrawn", now=self.t0 + DAY)
        self.assertEqual(closed.next_action, "")
        self.assertIsNone(closed.next_action_due)

    def test_reached_uses_history_not_current_stage(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.store.move(app.id, "applied", now=self.t0 + DAY)
        self.store.move(app.id, "interview", now=self.t0 + 2 * DAY)
        closed = self.store.move(app.id, "rejected", now=self.t0 + 3 * DAY)
        self.assertTrue(closed.reached("applied"))
        self.assertTrue(closed.reached("interview"))
        self.assertFalse(closed.reached("offer"))

    def test_submitted_since_counts_submissions_not_intent(self) -> None:
        saved_only = self.store.add("A", "Dev", now=self.t0)
        applied = self.store.add("B", "Dev", now=self.t0)
        self.store.move(applied.id, "applied", now=self.t0 + 1)
        withdrawn = self.store.add("C", "Dev", now=self.t0)
        self.store.move(withdrawn.id, "applied", now=self.t0 + 1)
        self.store.move(withdrawn.id, "withdrawn", now=self.t0 + 2)
        self.store.add("D", "Dev", now=self.t0 - 10 * DAY)  # pre-goal
        self.assertFalse(saved_only.reached("applied"))
        # B submitted, C submitted-then-withdrawn counts; A (saved
        # only) and D (filed before the goal) do not.
        self.assertEqual(self.store.submitted_since(self.t0), 2)

    def test_overdue_and_stale_flagging(self) -> None:
        now = self.t0 + 20 * DAY
        overdue = self.store.add("A", "Dev", now=self.t0)
        self.store.set_next_action(overdue.id, "chase recruiter",
                                   due=self.t0 + DAY, now=self.t0)
        planned = self.store.add("B", "Dev", now=self.t0)
        self.store.set_next_action(planned.id, "prep",
                                   due=now + DAY, now=self.t0)
        stale = self.store.add("C", "Dev", now=self.t0)
        fresh = self.store.add("D", "Dev", now=now - DAY)
        self.assertEqual([a.id for a in self.store.overdue_actions(now=now)],
                         [overdue.id])
        stale_ids = [a.id for a in self.store.stale(now=now)]
        self.assertEqual(stale_ids, [stale.id])
        self.assertNotIn(fresh.id, stale_ids)

    def test_notes_attach_and_persistence(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.store.add_note(app.id, "recruiter is Ada", now=self.t0 + 1)
        self.store.attach_cv(app.id, "art_123", now=self.t0 + 2)
        reopened = CareerStore(self.store.path).get(app.id)
        assert reopened is not None
        self.assertEqual(reopened.notes[0]["text"], "recruiter is Ada")
        self.assertEqual(reopened.cv_artifact_id, "art_123")
        self.assertEqual(reopened.updated_at, self.t0 + 2)

    def test_by_company_groups_employer_records(self) -> None:
        self.store.add("Acme", "Dev", now=self.t0)
        self.store.add("acme", "Designer", now=self.t0 + 1)
        self.store.add("Beta", "Dev", now=self.t0 + 2)
        groups = self.store.by_company()
        self.assertEqual(sorted(groups), ["acme", "beta"])
        self.assertEqual(len(groups["acme"]), 2)

    def test_remove(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.assertTrue(self.store.remove(app.id))
        self.assertIsNone(self.store.get(app.id))
        self.assertFalse(self.store.remove(app.id))


class _ProviderCase(unittest.TestCase):
    """Tmp home + the providers' settings seam pointed at it."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.home = Path(self._tmp.name)
        self._real_get_settings = providers.get_settings
        providers.get_settings = lambda: SimpleNamespace(home_path=self.home)
        self.store = CareerStore(self.home / "careers" / "applications.json")
        self.t0 = 1_700_000_000.0

    def tearDown(self) -> None:
        providers.get_settings = self._real_get_settings
        self._tmp.cleanup()


class ProviderTests(_ProviderCase):
    def test_linked_application_measures_stage_weight_exactly(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.store.move(app.id, "applied", now=self.t0 + DAY)
        self.store.move(app.id, "interview", now=self.t0 + 2 * DAY)
        goal = _goal(tags=[f"career-app:{app.id}"], created_at=self.t0)
        reading = providers.careers_progress(None, goal)
        assert reading is not None
        self.assertEqual(reading.percent, STAGE_PROGRESS["interview"])
        self.assertEqual(reading.percent, 70.0)
        self.assertEqual(reading.source, "application pipeline")
        self.assertEqual(reading.evidence["stage"], "interview")
        self.assertEqual(reading.evidence["stages_visited"],
                         ["saved", "applied", "interview"])

    def test_linked_accepted_is_100_and_rejected_is_0_measured(self) -> None:
        won = self.store.add("Acme", "Dev", now=self.t0)
        self.store.move(won.id, "accepted", now=self.t0 + DAY)
        reading = providers.careers_progress(
            None, _goal(tags=[f"career-app:{won.id}"]))
        assert reading is not None
        self.assertEqual(reading.percent, 100.0)

        lost = self.store.add("Beta", "Dev", now=self.t0)
        self.store.move(lost.id, "rejected", now=self.t0 + DAY)
        reading = providers.careers_progress(
            None, _goal(tags=[f"career-app:{lost.id}"]))
        assert reading is not None
        self.assertEqual(reading.percent, 0.0)
        self.assertIn("closed (rejected)", reading.headline)

    def test_target_counts_submissions_since_the_goal(self) -> None:
        for company in ("A", "B"):
            app = self.store.add(company, "Dev", now=self.t0 + DAY)
            self.store.move(app.id, "applied", now=self.t0 + DAY)
        self.store.add("C", "Dev", now=self.t0 + DAY)  # saved only
        goal = _goal(tags=["career-target:4"], created_at=self.t0)
        reading = providers.careers_progress(None, goal)
        assert reading is not None
        self.assertEqual(reading.percent, 50.0)  # 2 of 4, exactly
        self.assertEqual(reading.evidence["submitted"], 2)
        self.assertEqual(reading.evidence["target_applications"], 4)

    def test_unlinked_and_invalid_measure_nothing(self) -> None:
        self.assertIsNone(providers.careers_progress(None, _goal()))
        self.assertIsNone(providers.careers_progress(
            None, _goal(tags=["career-target:0"])))
        self.assertIsNone(providers.careers_progress(
            None, _goal(tags=["career-target:many"])))

    def test_stale_link_never_falls_through_to_target(self) -> None:
        app = self.store.add("Acme", "Dev", now=self.t0)
        self.store.move(app.id, "applied", now=self.t0)
        goal = _goal(tags=["career-app:app_gone", "career-target:1"],
                     created_at=self.t0)
        # The deliberate link is broken: no measurement, even though
        # the target tag alone would have measured 100.
        self.assertIsNone(providers.careers_progress(None, goal))

    def test_broken_store_is_contained_to_none(self) -> None:
        def _boom() -> Any:
            raise RuntimeError("settings exploded")

        providers.get_settings = _boom
        goal = _goal(tags=["career-target:2"], created_at=self.t0)
        with self.assertLogs(providers._log, level="WARNING"):
            self.assertIsNone(providers.careers_progress(None, goal))


class WiringTests(unittest.TestCase):
    def test_provider_registered_in_process(self) -> None:
        self.assertIs(registered_provider("careers"),
                      providers.careers_progress)

    def test_fresh_interpreter_wiring(self) -> None:
        """A bare `import nomorals.goals` — how the runtime/CLI
        start — leaves the careers provider registered."""
        code = (
            "import nomorals.goals\n"
            "from nomorals.goals.classes import registered_provider\n"
            "from nomorals.goals import providers\n"
            "assert registered_provider('careers') is providers.careers_progress\n"
            "print('wired')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=180)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("wired", proc.stdout)


class CatalogTests(unittest.TestCase):
    def test_registered_everywhere(self) -> None:
        self.assertIn("careers", CONTROL_COMMANDS)
        self.assertIn("careers", COMMAND_DETAILS)
        self.assertIn("careers", LIST_ONELINERS)
        grouped = [c for _, cs in LIST_GROUPS for c in cs]
        self.assertEqual(grouped.count("careers"), 1)
        self.assertIn("/careers", COMMAND_DETAILS["careers"]["usage"])

    def test_owner_only(self) -> None:
        self.assertNotIn("careers", PUBLIC_COMMANDS)
        self.assertFalse(caller_may_run("careers", False))
        self.assertTrue(caller_may_run("careers", True))

    def test_parse_control(self) -> None:
        cmd = parse_control("/careers add Acme | Dev")
        self.assertIsNotNone(cmd)
        assert cmd is not None
        self.assertEqual(cmd.kind, "careers")
        self.assertEqual(cmd.arg, "add")


class BoardRenderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.store = CareerStore(Path(self._tmp.name) / "applications.json")
        self.t0 = 1_700_000_000.0

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_empty_board_is_a_designed_state(self) -> None:
        board = careers_surface.render_pipeline_board(
            self.store, now=self.t0)
        self.assertIn("no applications yet", board)
        self.assertIn("/careers add", board)

    def test_board_groups_stages_and_calls_out_attention(self) -> None:
        now = self.t0 + 20 * DAY
        hot = self.store.add("Acme", "Dev", now=self.t0)
        self.store.move(hot.id, "interview", now=self.t0 + DAY)
        self.store.set_next_action(hot.id, "send prep notes",
                                   due=self.t0 + 2 * DAY, now=self.t0 + DAY)
        self.store.add("Beta", "Dev", now=self.t0)  # stale by `now`
        board = careers_surface.render_pipeline_board(self.store, now=now)
        self.assertIn("**Interview**", board)
        self.assertIn("Needs attention", board)
        self.assertIn("⛔ Acme — Dev: next action", board)
        self.assertIn("⚠ Beta — Dev: untouched for 20d", board)
        self.assertIn("OVERDUE", board)


class BriefingSeamTests(unittest.TestCase):
    """Reminders: careers next actions ride the morning briefing's
    existing goal-classes section — no new scheduler."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.path = Path(self._tmp.name) / "applications.json"
        import nomorals.careers.store as store_mod
        self._real_path = store_mod.careers_path
        store_mod.careers_path = lambda settings=None: self.path

    def tearDown(self) -> None:
        import nomorals.careers.store as store_mod
        store_mod.careers_path = self._real_path
        self._tmp.cleanup()

    def test_overdue_action_rides_the_goal_classes_section(self) -> None:
        store = CareerStore(self.path)
        app = store.add("Acme", "Dev", now=time.time() - 3 * DAY)
        store.set_next_action(app.id, "chase the recruiter",
                              due=time.time() - DAY, now=time.time() - DAY)
        ctx = SimpleNamespace(db=Database(":memory:"))
        briefing = BriefingComposer().compose(ctx, "2026-11-09")
        section = next((s for s in briefing.sections
                        if s.name == "goal_classes"), None)
        self.assertIsNotNone(section)
        assert section is not None
        self.assertTrue(any("Acme" in ln and "⛔" in ln
                            for ln in section.lines))

    def test_no_actions_no_section(self) -> None:
        ctx = SimpleNamespace(db=Database(":memory:"))
        briefing = BriefingComposer().compose(ctx, "2026-11-09")
        self.assertNotIn("goal_classes",
                         [s.name for s in briefing.sections])


class ChatSurfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-careers-p4-")
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
        self.store = CareerStore(
            Path(self.tmp.name) / "careers" / "applications.json")
        self.runtime._careers_store = lambda: self.store
        # The careers provider resolves the same file through the
        # process-settings seam (the P3 pattern) — point it at the
        # same tmp home so surface and measurement read one store.
        self._real_get_settings = providers.get_settings
        providers.get_settings = lambda: SimpleNamespace(
            home_path=Path(self.tmp.name))

    def tearDown(self) -> None:
        providers.get_settings = self._real_get_settings
        with contextlib.suppress(Exception):
            self.runtime.stop()
        self.context.close()
        self.tmp.cleanup()

    def _run(self, text: str) -> str:
        reply = self.runtime.handle_control(
            text, "telegram:999",
            message=_msg(text, chat_key="telegram:999"))
        return reply.text if hasattr(reply, "text") else reply

    def test_add_move_board_end_to_end(self) -> None:
        out = self._run("/careers add Acme | Senior Dev | linkedin")
        self.assertIn("filed **Acme — Senior Dev** at saved", out)
        app = self.store.list()[0]
        out = self._run(f"/careers move {app.id} applied via the site")
        self.assertIn("moved **Acme — Senior Dev** to applied", out)
        board = self._run("/careers")
        self.assertIn("**Applied**", board)
        self.assertIn("Acme", board)
        self.assertEqual([h["stage"] for h in
                          self.store.get(app.id).history],
                         ["saved", "applied"])

    def test_move_unknown_stage_and_closed_refusals_in_words(self) -> None:
        self._run("/careers add Acme | Dev")
        app = self.store.list()[0]
        out = self._run(f"/careers move {app.id} limbo")
        self.assertIn("usage: /careers move", out)
        self._run(f"/careers move {app.id} rejected")
        out = self._run(f"/careers move {app.id} interview")
        self.assertIn("closed (rejected)", out)

    def test_next_and_note_verbs(self) -> None:
        self._run("/careers add Acme | Dev")
        app = self.store.list()[0]
        out = self._run(
            f"/careers next {app.id} | send follow-up by 2026-12-01")
        self.assertIn("next for **Acme — Dev**: send follow-up", out)
        self.assertIn("due 2026-12-01", out)
        stored = self.store.get(app.id)
        self.assertEqual(stored.next_action, "send follow-up")
        self.assertIsNotNone(stored.next_action_due)
        out = self._run(f"/careers note {app.id} | recruiter is Ada")
        self.assertIn("noted on", out)
        self.assertEqual(self.store.get(app.id).notes[0]["text"],
                         "recruiter is Ada")

    def test_attach_records_a_real_artifact(self) -> None:
        self._run("/careers add Acme | Dev")
        app = self.store.list()[0]
        artifacts = self.runtime._artifact_store()
        art = artifacts.put(b"my cv", type="document",
                            mime="text/plain")
        out = self._run(f"/careers attach {app.id} {art.id}")
        self.assertIn("attached 📎", out)
        self.assertEqual(self.store.get(app.id).cv_artifact_id, art.id)
        out = self._run(f"/careers attach {app.id} art_missing")
        self.assertIn("unknown artifact", out)

    def test_goals_careers_drilldown_and_measured_flip(self) -> None:
        self._run("/careers add Acme | Senior Dev")
        app = self.store.list()[0]
        self._run(f"/careers move {app.id} interview")
        self._run("/goals add Land the Acme role in careers")
        tracker = GoalTracker(self.context.db)
        goal = surface.run_sync(
            tracker.list_goals(UID, status="active"))[0]
        surface.run_sync(tracker.update(
            goal.goal_id, tags=[f"career-app:{app.id}"]))
        board = self._run("/goals list careers")
        self.assertIn("70% (measured · application pipeline)", board)
        # The pipeline drill-down rides under the class board.
        self.assertIn("Career pipeline", board)
        self.assertIn("**Interview**", board)


if __name__ == "__main__":
    unittest.main()
