"""WS2 P5 — the productivity + interests measured-progress providers.

Pins, in the P3/P4 style:

* productivity (``commitment_reliability``): kept ÷ due over the
  goal's window, computed from the commitments ledger's OWN
  replayed state (the store is event-sourced: the file holds
  ledger entries, commitments are derived).  Cancelled and
  superseded commitments are not misses and land in the evidence
  as exclusions, with no-deadline / pre-window / not-yet-due /
  other-promisor rows; unreadable ledger entries are counted, not
  absorbed.  Nothing due → None (a 0/0 rate would be fabricated).
* the focus measure: a ``focus-target:<minutes>`` tag switches
  the class measure to completed focus minutes (done sessions
  only — an in-progress session's length depends on when you
  look) from the focus timer's file; an invalid target is a
  broken deliberate declaration → None, never a fall-through.
* interests (``watch_coverage``): delivered ÷ flowed (or ÷ a
  ``watch-target:<n>``) over the news store's delivery states,
  matched deterministically (every content word of a topic in the
  item text).  Scope-but-no-flow is a measured 0%; no usable scope
  is None — the distinction is pinned both ways.
* beat registration (the census's settled behaviour change):
  creating an interests goal files its topics as NewsAgent beats
  (seeds materialised, no duplicates, briefing topics respected);
  re-filing out un-files them with seed/briefing/other-goal
  protections; the NewsAgent itself sees the new beat.
* failure containment: a broken store reads as self-reported with
  a named warning in the log — never an exception in a surface.
* wiring: a fresh interpreter importing the goals package finds
  all five WS2 providers registered; the seed-beat and
  item-column mirrors (agents.news is L5, unimportable from
  goals' L4) are pinned against the real constants.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.goals import GoalTracker, providers, surface
from nomorals.goals.classes import goal_progress, registered_provider
from nomorals.storage import Database

NOW = 1_800_000_000.0
SINCE = NOW - 7 * 86400

_REAL_GET_SETTINGS = providers.get_settings


def _harness(case: unittest.TestCase) -> tuple[Path, GoalTracker]:
    """A tmp nomorals home + an in-memory tracker; providers'
    settings seam pointed at the home (the P3/P4 pattern)."""
    tmp = tempfile.TemporaryDirectory(
        dir=os.environ.get("NM_TEST_TMPDIR"))
    case.addCleanup(tmp.cleanup)
    home = Path(tmp.name)
    (home / "data").mkdir(parents=True)
    settings = SimpleNamespace(
        home_path=home, briefing=SimpleNamespace(topics=[]))
    providers.get_settings = lambda: settings
    case.addCleanup(setattr, providers, "get_settings",
                    _REAL_GET_SETTINGS)
    return home, GoalTracker(Database(":memory:"))


def _goal(title: str, *, tags=(), created: float = SINCE,
          goal_class: str = "productivity", progress: float = 0.0):
    return SimpleNamespace(
        goal_id="g-test", title=title, tags=list(tags),
        created_at=created, goal_class=goal_class, progress=progress)


# ── commitments ledger seeding (the store's real event shape) ────────

_seq = [0]


def _ev(kind: str, cid: str, ts: float, payload: dict | None = None) -> dict:
    _seq[0] += 1
    return {"seq": _seq[0], "ts": ts, "kind": kind,
            "commitment_id": cid, "detail": "", "payload": payload or {}}


def _created(cid: str, deadline, *, promisor: str = "owner",
             at: float | None = None) -> dict:
    at = SINCE + 10 if at is None else at
    return _ev("created", cid, at,
               {"promisor": promisor, "promisee": "someone",
                "what": f"do {cid}", "deadline": deadline,
                "created_at": at})


def _write_ledger(home: Path, entries: list[dict]) -> None:
    (home / "data" / "commitments.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8")


def _full_ledger() -> list[dict]:
    """Two kept, one broken, one overdue-open due in the window;
    one of every exclusion; one unreadable entry row."""
    return [
        _created("k1", SINCE + 100), _ev("kept", "k1", SINCE + 90),
        _created("k2", SINCE + 200), _ev("kept", "k2", SINCE + 190),
        _created("b1", SINCE + 300), _ev("broken", "b1", SINCE + 290),
        _created("o1", SINCE + 400),
        _created("c1", SINCE + 500), _ev("cancelled", "c1", SINCE + 490),
        _created("s1", SINCE + 600), _ev("superseded", "s1", SINCE + 590),
        _created("n1", None),
        _created("f1", NOW + 9999),
        _created("p1", SINCE - 5000, at=SINCE - 6000),
        _ev("kept", "p1", SINCE - 4000),
        _created("x1", SINCE + 700, promisor="them"),
        _ev("kept", "x1", SINCE + 690),
        {"seq": "not-a-number", "ts": 0, "kind": "created",
         "commitment_id": "bad", "detail": "", "payload": {}},
    ]


def _session(sid: str, status: str, started: float,
             ended: float | None = None) -> dict:
    return {"id": sid, "label": "deep work", "status": status,
            "started_at": started, "ended_at": ended, "pauses": [],
            "interruptions": 0, "notes": []}


def _write_focus(home: Path, sessions: list[dict]) -> None:
    (home / "data" / "focus.json").write_text(
        json.dumps({"sessions": sessions}), encoding="utf-8")


def _seed_items(db: Database, rows: list[tuple]) -> None:
    db.execute(
        "CREATE TABLE IF NOT EXISTS news_items ("
        "id INTEGER PRIMARY KEY, title TEXT, summary TEXT, "
        "source TEXT, publisher TEXT, delivery_state TEXT, "
        "delivered_at REAL, delivery_channel TEXT, interest TEXT, "
        "created_at REAL)")
    for title, state, created in rows:
        db.execute(
            "INSERT INTO news_items (title, summary, source, publisher,"
            " delivery_state, created_at) VALUES (?, '', 'wire', '', ?, ?)",
            (title, state, created))


class TestProductivityReliability(unittest.TestCase):

    def test_exact_reliability_math_and_exclusions(self):
        home, _tracker = _harness(self)
        _write_ledger(home, _full_ledger())
        reading = providers.productivity_progress(
            None, _goal("Get things done"), now=NOW)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.source, "commitment ledger")
        self.assertAlmostEqual(reading.percent, 50.0)
        ev = reading.evidence
        self.assertEqual(ev["kept"], 2)
        self.assertEqual(ev["broken"], 1)
        self.assertEqual(ev["overdue_open"], 1)
        self.assertEqual(ev["due"], 4)
        self.assertEqual(ev["excluded"], {
            "cancelled": 1, "superseded": 1, "no_deadline": 1,
            "due_before_window": 1, "not_yet_due": 1,
            "other_promisor": 1})
        self.assertEqual(ev["unreadable_ledger_entries"], 1)
        self.assertIn("kept 2 of 4", reading.headline)
        self.assertIn("1 broken, 1 overdue", reading.headline)

    def test_cancelled_and_superseded_are_not_misses(self):
        home, _tracker = _harness(self)
        _write_ledger(home, [
            _created("k1", SINCE + 100), _ev("kept", "k1", SINCE + 90),
            _created("c1", SINCE + 200), _ev("cancelled", "c1", SINCE + 190),
            _created("s1", SINCE + 300),
            _ev("superseded", "s1", SINCE + 290),
        ])
        reading = providers.productivity_progress(
            None, _goal("Reliable"), now=NOW)
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading.percent, 100.0)
        self.assertEqual(reading.evidence["due"], 1)

    def test_nothing_due_measures_nothing(self):
        home, _tracker = _harness(self)
        _write_ledger(home, [
            _created("c1", SINCE + 100), _ev("cancelled", "c1", SINCE + 90),
            _created("f1", NOW + 9999),
        ])
        goal = _goal("Nothing due", progress=35.0)
        self.assertIsNone(
            providers.productivity_progress(None, goal, now=NOW))
        fallback = goal_progress(None, goal, now=NOW)
        self.assertEqual(fallback.basis, "self-reported")
        self.assertAlmostEqual(fallback.percent, 35.0)

    def test_missing_ledger_measures_nothing(self):
        _harness(self)
        self.assertIsNone(providers.productivity_progress(
            None, _goal("No organ yet"), now=NOW))

    def test_corrupt_ledger_file_fails_contained(self):
        home, _tracker = _harness(self)
        (home / "data" / "commitments.json").write_text(
            "{not json", encoding="utf-8")
        with self.assertLogs("nomorals.goals.providers", level="WARNING") as cm:
            reading = providers.productivity_progress(
                None, _goal("Broken store"), now=NOW)
        self.assertIsNone(reading)
        self.assertTrue(any(
            "productivity progress provider failed" in line
            for line in cm.output), cm.output)

    def test_through_tracker_class_progress(self):
        home, tracker = _harness(self)
        goal = surface.run_sync(tracker.create(
            "owner", "Ship the release plan", goal_class="productivity"))
        _write_ledger(home, [
            _created("k1", goal.created_at + 60),
            _ev("kept", "k1", goal.created_at + 50),
            _created("b1", goal.created_at + 120),
            _ev("broken", "b1", goal.created_at + 110),
        ])
        reading = tracker.class_progress(
            goal, now=goal.created_at + 3600)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.source, "commitment ledger")
        self.assertAlmostEqual(reading.percent or 0.0, 50.0)


class TestProductivityFocus(unittest.TestCase):

    def _seeded_home(self) -> Path:
        home, _tracker = _harness(self)
        _write_focus(home, [
            _session("a", "done", SINCE + 100, SINCE + 100 + 1800),
            _session("b", "done", SINCE + 200, SINCE + 200 + 2700),
            _session("c", "active", SINCE + 300),
            _session("d", "done", SINCE - 9000, SINCE - 9000 + 3600),
        ])
        return home

    def test_focus_minutes_exact(self):
        self._seeded_home()
        reading = providers.productivity_progress(
            None, _goal("Deep work", tags=["focus-target:150"]), now=NOW)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.source, "focus sessions")
        self.assertAlmostEqual(reading.percent, 50.0)
        ev = reading.evidence
        self.assertEqual(ev["focus_minutes"], 75.0)
        self.assertEqual(ev["completed_sessions"], 2)
        self.assertEqual(ev["sessions_in_progress"], 1)
        self.assertEqual(ev["sessions_predating_goal"], 1)
        self.assertIn("focused 75 of 150", reading.headline)

    def test_focus_target_wins_over_reliability(self):
        home = self._seeded_home()
        _write_ledger(home, _full_ledger())
        reading = providers.productivity_progress(
            None, _goal("Deep work", tags=["focus-target:150"]), now=NOW)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.evidence["measure"], "focus_minutes")

    def test_invalid_focus_target_is_none(self):
        self._seeded_home()
        for tag in ("focus-target:soon", "focus-target:0",
                    "focus-target:-30"):
            self.assertIsNone(providers.productivity_progress(
                None, _goal("Deep work", tags=[tag]), now=NOW), tag)

    def test_missing_focus_file_is_measured_zero(self):
        _harness(self)
        reading = providers.productivity_progress(
            None, _goal("Deep work", tags=["focus-target:120"]), now=NOW)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.basis, "measured")
        self.assertAlmostEqual(reading.percent, 0.0)
        self.assertIn("no completed focus sessions", reading.headline)


class TestInterestBeats(unittest.TestCase):

    def test_create_registers_beat_and_note_states_it(self):
        _home, tracker = _harness(self)
        goal, class_id, _suggested = surface.create_goal_filed(
            tracker, "owner", "Ekiti startup ecosystem", "interests")
        self.assertEqual(class_id, "interests")
        profile = providers._load_news_profile(tracker.db)
        topics = [b["topic"] for b in profile["beats"]]
        self.assertIn("Ekiti startup ecosystem", topics)
        # The seeds materialised alongside (the NewsAgent's profile
        # shape), so the organ sees one coherent beat list.
        self.assertIn("XAUUSD and gold trading", topics)
        note = surface.interest_watch_note(tracker, goal)
        self.assertIn("Ekiti startup ecosystem", note)
        self.assertIn("news beats", note)

    def test_news_agent_sees_the_registered_beat(self):
        _home, tracker = _harness(self)
        surface.create_goal_filed(
            tracker, "owner", "Ekiti startup ecosystem", "interests")
        from nomorals.agents.news import NewsAgent
        agent = NewsAgent(SimpleNamespace(db=tracker.db, settings=None))
        beats = {b["topic"] for b in agent.interest_profile()["beats"]}
        self.assertIn("Ekiti startup ecosystem", beats)

    def test_suggested_interests_filing_registers_title(self):
        # The deterministic suggester files interests when the title
        # names the class (P2's matcher); registration must run for
        # suggested filings exactly as for explicit ones.
        _home, tracker = _harness(self)
        goal, class_id, suggested = surface.create_goal_filed(
            tracker, "owner", "Interests: Ekiti startups", None)
        self.assertEqual(class_id, "interests")
        self.assertTrue(suggested)
        # Registration already ran inside the filing: a second call
        # adds nothing.
        self.assertEqual(
            providers.register_interest_beats(tracker.db, goal), [])
        topics = [b["topic"] for b in
                  providers._load_news_profile(tracker.db)["beats"]]
        self.assertIn("Interests: Ekiti startups", topics)

    def test_no_duplicate_registration(self):
        _home, tracker = _harness(self)
        goal, _cid, _s = surface.create_goal_filed(
            tracker, "owner", "First watch", "interests")
        again = providers.register_interest_beats(tracker.db, goal)
        self.assertEqual(again, [])
        topics = [b["topic"] for b in
                  providers._load_news_profile(tracker.db)["beats"]]
        self.assertEqual(topics.count("First watch"), 1)

    def test_seed_covered_topic_registers_nothing_but_watch_is_live(self):
        _home, tracker = _harness(self)
        goal, _cid, _s = surface.create_goal_filed(
            tracker, "owner", "AI and model work", "interests")
        self.assertEqual(
            providers.register_interest_beats(tracker.db, goal), [])
        self.assertIn("AI and model work",
                      surface.interest_watch_note(tracker, goal))

    def test_watch_tags_are_the_registered_topics(self):
        _home, tracker = _harness(self)
        goal = surface.run_sync(tracker.create(
            "owner", "My reading list", goal_class="interests",
            tags=["watch:Ekiti startups", "watch:Lagos fintech"]))
        added = providers.register_interest_beats(tracker.db, goal)
        self.assertEqual(added, ["Ekiti startups", "Lagos fintech"])

    def test_refile_in_registers_and_refile_out_removes(self):
        _home, tracker = _harness(self)
        goal, _cid, _s = surface.create_goal_filed(
            tracker, "owner", "Ekiti startup ecosystem", None)
        text = surface.refile_goal(
            tracker, "owner", goal.goal_id, "interests")
        self.assertIn("registered as news beat(s)", text)
        self.assertIn("Ekiti startup ecosystem", text)
        topics = [b["topic"] for b in
                  providers._load_news_profile(tracker.db)["beats"]]
        self.assertIn("Ekiti startup ecosystem", topics)
        text = surface.refile_goal(
            tracker, "owner", goal.goal_id, "general")
        self.assertIn("stopped watching", text)
        self.assertIn("Ekiti startup ecosystem", text)
        topics = [b["topic"] for b in
                  providers._load_news_profile(tracker.db)["beats"]]
        self.assertNotIn("Ekiti startup ecosystem", topics)
        self.assertIn("XAUUSD and gold trading", topics)

    def test_refile_out_keeps_topic_another_goal_claims(self):
        _home, tracker = _harness(self)
        first, _c, _s = surface.create_goal_filed(
            tracker, "owner", "Shared watch", "interests")
        surface.create_goal_filed(
            tracker, "owner", "Shared watch", "interests")
        text = surface.refile_goal(
            tracker, "owner", first.goal_id, "general")
        self.assertNotIn("stopped watching", text)
        topics = [b["topic"] for b in
                  providers._load_news_profile(tracker.db)["beats"]]
        self.assertIn("Shared watch", topics)

    def test_refile_out_never_removes_a_seed_beat(self):
        _home, tracker = _harness(self)
        goal, _c, _s = surface.create_goal_filed(
            tracker, "owner", "AI and model work", "interests")
        surface.refile_goal(tracker, "owner", goal.goal_id, "general")
        topics = [b["topic"] for b in
                  providers._load_news_profile(tracker.db)["beats"]]
        self.assertIn("AI and model work", topics)


class TestInterestsCoverage(unittest.TestCase):

    def _tracker_with_goal(self, title="Ekiti startups", tags=()):
        _home, tracker = _harness(self)
        goal = surface.run_sync(tracker.create(
            "owner", title, goal_class="interests", tags=list(tags)))
        return tracker, goal

    def test_coverage_math_exact(self):
        tracker, goal = self._tracker_with_goal()
        at = goal.created_at
        _seed_items(tracker.db, [
            ("Ekiti startups raise seed round", "pushed", at + 10),
            ("Ekiti startups win grant", "undelivered", at + 20),
            ("Ekiti startups archive piece", "archived", at + 30),
            ("Ekiti startups legacy story", None, at + 40),
            ("Gold market update", "pushed", at + 50),
            ("Ekiti startups before the goal", "pushed", at - 5000),
        ])
        reading = providers.interests_progress(
            tracker.db, goal, now=at + 100000)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.basis, "measured")
        self.assertEqual(reading.source, "news watch")
        self.assertEqual(reading.evidence["flowed"], 4)
        self.assertEqual(reading.evidence["delivered"], 2)
        self.assertAlmostEqual(reading.percent, 50.0)
        self.assertIn("2 of 4 findings", reading.headline)

    def test_watch_target_math(self):
        tracker, goal = self._tracker_with_goal(
            title="Lagos fintech watch",
            tags=["watch:Lagos fintech", "watch-target:4"])
        _seed_items(tracker.db, [
            ("Lagos fintech app launches", "digested",
             goal.created_at + 5),
        ])
        reading = providers.interests_progress(
            tracker.db, goal, now=goal.created_at + 100000)
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading.percent, 25.0)
        self.assertEqual(reading.evidence["target_findings"], 4)
        self.assertIn("1 of 4 watched findings delivered",
                      reading.headline)

    def test_invalid_watch_target_is_none(self):
        tracker, goal = self._tracker_with_goal(
            title="Lagos fintech watch",
            tags=["watch:Lagos fintech", "watch-target:many"])
        self.assertIsNone(providers.interests_progress(
            tracker.db, goal, now=goal.created_at + 1000))

    def test_scope_but_no_flow_is_measured_zero(self):
        tracker, goal = self._tracker_with_goal(title="Quantum sensors")
        reading = providers.interests_progress(
            tracker.db, goal, now=goal.created_at + 1000)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.basis, "measured")
        self.assertAlmostEqual(reading.percent, 0.0)
        self.assertIn("no findings on this watch yet", reading.headline)

    def test_no_scope_is_none(self):
        _home, tracker = _harness(self)
        goal = _goal("the", goal_class="interests")
        self.assertIsNone(providers.interests_progress(
            tracker.db, goal, now=NOW))
        fallback = goal_progress(tracker.db, goal, now=NOW)
        self.assertEqual(fallback.basis, "self-reported")

    def test_missing_news_table_reads_as_no_flow(self):
        _home, tracker = _harness(self)
        goal = _goal("Quantum sensors", goal_class="interests")
        reading = providers.interests_progress(
            tracker.db, goal, now=NOW)
        self.assertIsNotNone(reading)
        self.assertAlmostEqual(reading.percent, 0.0)

    def test_no_database_is_none(self):
        goal = _goal("Quantum sensors", goal_class="interests")
        self.assertIsNone(providers.interests_progress(None, goal, now=NOW))

    def test_unreadable_item_rows_counted(self):
        tracker, goal = self._tracker_with_goal()
        _seed_items(tracker.db, [
            ("Ekiti startups raise seed round", "pushed",
             goal.created_at + 10),
            ("Ekiti startups broken clock", "pushed", "not-a-time"),
        ])
        reading = providers.interests_progress(
            tracker.db, goal, now=goal.created_at + 100000)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.evidence["flowed"], 1)
        self.assertEqual(reading.evidence["unreadable_item_rows"], 1)

    def test_watch_tags_scope_not_title(self):
        tracker, goal = self._tracker_with_goal(
            title="My reading list", tags=["watch:Ekiti startups"])
        _seed_items(tracker.db, [
            ("Ekiti startups raise seed round", "pushed",
             goal.created_at + 10),
            ("My reading list roundup", "pushed", goal.created_at + 20),
        ])
        reading = providers.interests_progress(
            tracker.db, goal, now=goal.created_at + 100000)
        self.assertIsNotNone(reading)
        self.assertEqual(reading.evidence["topics"], ["Ekiti startups"])
        self.assertEqual(reading.evidence["flowed"], 1)


class TestSurfaceFlip(unittest.TestCase):

    def test_board_renders_measured_for_both_classes(self):
        home, tracker = _harness(self)
        prod, _c, _s = surface.create_goal_filed(
            tracker, "owner", "Ship the release plan", "productivity")
        _write_ledger(home, [
            _created("k1", prod.created_at + 60, at=prod.created_at),
            _ev("kept", "k1", prod.created_at + 50),
            _created("b1", prod.created_at + 120, at=prod.created_at),
            _ev("broken", "b1", prod.created_at + 110),
        ])
        surface.create_goal_filed(
            tracker, "owner", "Ekiti startup ecosystem", "interests")
        board = surface.render_class_board(
            tracker, "owner", now=prod.created_at + 3600)
        self.assertIn("measured · commitment ledger", board)
        self.assertIn("measured · news watch", board)
        progress = surface.render_goal_progress(
            tracker, "owner", prod.goal_id, now=prod.created_at + 3600)
        self.assertIn("measured · commitment ledger", progress)
        self.assertIn("kept 1 of 2", progress)


class TestFailureContainment(unittest.TestCase):

    def test_interests_store_error_fails_contained(self):
        _harness(self)

        class ExplodingDb:
            def query(self, *args, **kwargs):
                raise RuntimeError("store exploded")

            def query_one(self, *args, **kwargs):
                raise RuntimeError("store exploded")

        goal = _goal("Quantum sensors", goal_class="interests")
        with self.assertLogs("nomorals.goals.providers", level="WARNING") as cm:
            reading = providers.interests_progress(
                ExplodingDb(), goal, now=NOW)
        self.assertIsNone(reading)
        self.assertTrue(any(
            "interests progress provider failed" in line
            for line in cm.output), cm.output)

    def test_productivity_store_path_error_fails_contained(self):
        home, _tracker = _harness(self)
        # A directory where the ledger file belongs: read fails.
        (home / "data" / "commitments.json").mkdir()
        with self.assertLogs("nomorals.goals.providers", level="WARNING"):
            reading = providers.productivity_progress(
                None, _goal("Blocked store"), now=NOW)
        self.assertIsNone(reading)


class TestWiringAndMirrors(unittest.TestCase):

    def test_all_five_providers_registered(self):
        for class_id, fn in (
                ("finance", providers.finance_progress),
                ("trading", providers.trading_progress),
                ("careers", providers.careers_progress),
                ("productivity", providers.productivity_progress),
                ("interests", providers.interests_progress)):
            self.assertIs(registered_provider(class_id), fn, class_id)

    def test_seed_beats_mirror_matches_news_agent(self):
        from nomorals.agents.news import INTEREST_SEED_BEATS
        self.assertEqual(providers._SEED_BEATS, INTEREST_SEED_BEATS)

    def test_item_extra_mirror_matches_news_agent(self):
        from nomorals.agents.news import NewsAgent
        self.assertEqual(providers._ITEM_EXTRA, NewsAgent._ITEM_EXTRA)

    def test_fresh_interpreter_wiring(self):
        code = (
            "import nomorals.goals\n"
            "from nomorals.goals.classes import registered_provider\n"
            "ids = ['finance', 'trading', 'careers', 'productivity',"
            " 'interests']\n"
            "missing = [c for c in ids if registered_provider(c) is None]\n"
            "assert not missing, missing\n"
            "print('wiring ok:', ', '.join(ids))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            cwd=".", timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("wiring ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
