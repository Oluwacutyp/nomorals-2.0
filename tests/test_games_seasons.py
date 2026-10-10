"""Seasonal events: deterministic calendar, multipliers, engine hooks."""
from __future__ import annotations

import time
import unittest
from datetime import datetime

from nomorals.games import seasons
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.storage.db import Database


class Ctx:
    def __init__(self, db):
        self.db = db


def make_engine():
    db = Database(":memory:")
    db.migrate()
    sent: list[str] = []
    engine = GameEngine(Ctx(db), send=lambda chat, text: sent.append(text))
    return engine, db, sent


ADA = Player.from_sender("telegram", "111", "Ada")


def ts_for(year: int, week: int) -> float:
    # a timestamp inside the given ISO week (Monday noon)
    return datetime.fromisocalendar(year, week, 1).timestamp() + 43200


class CalendarTests(unittest.TestCase):
    def test_deterministic(self):
        a = seasons.event_for_week(2026, 40)
        b = seasons.event_for_week(2026, 40)
        self.assertEqual(a["id"], b["id"])

    def test_rotates(self):
        ids = {seasons.event_for_week(2026, w)["id"] for w in range(1, 13)}
        self.assertGreater(len(ids), 1)

    def test_all_events_valid(self):
        for e in seasons.SEASON_ROSTER:
            self.assertTrue(e["id"] and e["name"])
            self.assertGreaterEqual(e["coin_mult"], 1.0)
            self.assertGreaterEqual(e["xp_mult"], 1.0)

    def test_active_event_uses_calendar(self):
        now = ts_for(2026, 40)
        ev = seasons.active_event(None, now)
        dt = datetime.fromtimestamp(now)
        y, w, _ = dt.isocalendar()
        self.assertEqual(ev["id"], seasons.event_for_week(y, w)["id"])


class ApplyTests(unittest.TestCase):
    def _event_week(self, event_id: str) -> float:
        for w in range(1, 54):
            if seasons.event_for_week(2026, w)["id"] == event_id:
                return ts_for(2026, w)
        raise AssertionError(f"no week for {event_id}")

    def test_coin_mult_applies_to_touched_game(self):
        now = self._event_week("golden_week")
        new, note = seasons.apply_event(None, "slots", "coins", 100, now)
        self.assertEqual(new, 150)
        self.assertIn("Golden Week", note)

    def test_no_mult_for_untouched_game(self):
        now = self._event_week("golden_week")
        new, note = seasons.apply_event(None, "arena", "coins", 100, now)
        self.assertEqual(new, 100)
        self.assertEqual(note, "")

    def test_quiet_week_is_neutral(self):
        now = self._event_week("quiet")
        new, note = seasons.apply_event(None, "arena", "coins", 100, now)
        self.assertEqual(new, 100)
        self.assertEqual(note, "")

    def test_blurb_only_for_touched_games(self):
        now = self._event_week("mind_games")
        self.assertIsNotNone(seasons.event_blurb(None, "sudoku", now))
        self.assertIsNone(seasons.event_blurb(None, "arena", now))

    def test_describe_lists_current_and_next(self):
        now = ts_for(2026, 40)
        text = seasons.describe_seasons(None, now)
        self.assertIn("now:", text)
        self.assertIn("next:", text)


class EngineHookTests(unittest.TestCase):
    def _event_week(self, event_id: str) -> float:
        for w in range(1, 54):
            if seasons.event_for_week(2026, w)["id"] == event_id:
                return ts_for(2026, w)
        raise AssertionError(f"no week for {event_id}")

    def test_season_blurb_at_table_open(self):
        # force golden week via override, then open a casino table
        engine, db, sent = make_engine()
        db.execute(
            "CREATE TABLE IF NOT EXISTS season_override ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "event_id TEXT NOT NULL, until REAL NOT NULL DEFAULT 0)")
        db.execute(
            "INSERT OR REPLACE INTO season_override (id, event_id, until) "
            "VALUES (1, 'golden_week', ?)", (time.time() + 3600,))
        room, msgs = engine.start("test:sl", "slots", ADA, kind="dm")
        self.assertTrue(any("Golden Week" in m for m in msgs))
        engine.quit("test:sl")

    def test_no_blurb_for_untouched_game(self):
        engine, db, sent = make_engine()
        db.execute(
            "CREATE TABLE IF NOT EXISTS season_override ("
            "id INTEGER PRIMARY KEY CHECK (id = 1), "
            "event_id TEXT NOT NULL, until REAL NOT NULL DEFAULT 0)")
        db.execute(
            "INSERT OR REPLACE INTO season_override (id, event_id, until) "
            "VALUES (1, 'golden_week', ?)", (time.time() + 3600,))
        room, msgs = engine.start("test:ar", "arena", ADA, kind="dm")
        self.assertFalse(any("Golden Week" in m for m in msgs))
        engine.quit("test:ar")


if __name__ == "__main__":
    unittest.main()
