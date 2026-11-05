"""Queue cron parity — storage.queue delegates to the canonical engine.

Wave-5 follow-up (scheduler/PARITY.md ranked gap #1): queue.py's
private cron engine (AND semantics for dom+dow, local-time only,
minute-stepping capped at a year, silent ``None`` → stored
``next_fire_at = 0``) is retired. These tests pin:

* queue scheduling AGREES with ``nomorals.scheduler.recurrence`` on a
  shared expression corpus (the same engine every surface uses);
* the deliberate deltas hold — dom+dow OR semantics, an explicit
  timezone parameter, named errors instead of the silent fallback;
* legacy stored rows are healed by ``tick_recurring`` (dead-zero rows
  recomputed or reported by name; AND-computed next fires tightened
  to the canonical OR value).
"""

from __future__ import annotations

import calendar as _calendar
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.scheduler.recurrence import (  # noqa: E402
    CronNeverMatchesError,
    CronTimezoneError,
    next_cron,
)
from nomorals.storage.db import Database  # noqa: E402
from nomorals.storage.failures import recent_failures  # noqa: E402
from nomorals.storage.queue import (  # noqa: E402
    WorkQueue,
    next_cron_fire,
    parse_cron,
)


def _old_engine_next(expr: str, after_ts: float) -> float | None:
    """The retired queue engine's semantics, for divergence proofs.

    AND semantics across all five fields, system-local wall time,
    day-stepping (equivalent to its minute-stepping for the fixed
    HH:MM expressions used here), ``None`` past a 366-day cap.
    """
    minute_s, hour_s, dom_s, month_s, dow_s = expr.split()
    minute, hour = int(minute_s), int(hour_s)
    doms = set(range(1, 32)) if dom_s == "*" else {int(d) for d in dom_s.split(",")}
    months = set(range(1, 13)) if month_s == "*" else {int(m) for m in month_s.split(",")}
    dows = set(range(7)) if dow_s == "*" else {int(d) for d in dow_s.split(",")}
    day = datetime.fromtimestamp(after_ts).date()
    for offset in range(367):
        d = day + timedelta(days=offset)
        cron_dow = (d.weekday() + 1) % 7  # cron: 0 = Sunday
        if d.day in doms and d.month in months and cron_dow in dows:
            cand = datetime(d.year, d.month, d.day, hour, minute).timestamp()
            if cand > after_ts:
                return cand
    return None


def _first_monday_not_first() -> datetime:
    """A Monday (08:00 local) that is not the 1st of its month."""
    day = datetime(2026, 1, 1, 8, 0)
    while day.weekday() != 0 or day.day == 1:
        day += timedelta(days=1)
    return day


class CorpusParityTests(unittest.TestCase):
    """queue.next_cron_fire == recurrence.next_cron on shared cases."""

    CASES = [
        ("* * * * *", datetime(2026, 3, 4, 5, 6, 30)),
        ("*/15 9-17 * * 1-5", datetime(2026, 3, 6, 16, 50)),   # Friday
        ("*/15 9-17 * * 1-5", datetime(2026, 3, 7, 10, 0)),    # Saturday
        ("0 0 29 2 *", datetime(2025, 3, 1, 0, 0)),            # → 2028
        ("0 0 29 2 *", datetime(2096, 3, 1, 0, 0)),            # → 2104
        ("30 4 1,15 * *", datetime(2026, 6, 15, 4, 30)),
        ("0 9 1 * 1", datetime(2026, 1, 1, 8, 0)),             # dom+dow OR
        ("0 0 L * *", datetime(2026, 2, 10, 0, 0)),
        ("0 12 * * 0", datetime(2026, 1, 2, 12, 0)),
        ("0 0 1 1 *", datetime(2026, 12, 31, 23, 59)),
        ("5/20 * * * *", datetime(2026, 5, 5, 5, 5)),          # Vixie a/n
    ]

    def test_queue_agrees_with_canonical_engine(self):
        for expr, pivot in self.CASES:
            with self.subTest(expr=expr, pivot=pivot):
                self.assertEqual(
                    next_cron_fire(expr, after=pivot.timestamp()),
                    next_cron(expr, pivot),
                )

    def test_explicit_timezone_agrees_with_canonical_engine(self):
        pivot = datetime(2026, 1, 1, 0, 0)
        for zone in ("Africa/Lagos", "America/Denver", "UTC"):
            with self.subTest(zone=zone):
                self.assertEqual(
                    next_cron_fire("0 6 * * *", after=pivot.timestamp(),
                                   timezone=zone),
                    next_cron("0 6 * * *", pivot, zone),
                )

    def test_lagos_anchor_fires_at_0600_wat(self):
        nxt = next_cron_fire("0 6 * * *", after=datetime(2026, 1, 1).timestamp(),
                             timezone="Africa/Lagos")
        wall = datetime.fromtimestamp(nxt, ZoneInfo("Africa/Lagos"))
        self.assertEqual((wall.hour, wall.minute), (6, 0))


class DeltaTests(unittest.TestCase):
    """The deliberate behaviour deltas vs the retired engine."""

    def test_dom_dow_or_semantics_pinned(self):
        monday = _first_monday_not_first()          # a Monday, not the 1st
        pivot_ts = monday.timestamp()
        got = next_cron_fire("0 9 1 * 1", after=pivot_ts)
        # OR: fires that same Monday at 09:00.
        self.assertEqual(got, monday.replace(hour=9, minute=0).timestamp())
        # The retired AND engine waited for a day that is BOTH the 1st
        # and a Monday — strictly later.
        and_next = _old_engine_next("0 9 1 * 1", pivot_ts)
        self.assertIsNotNone(and_next)
        self.assertLess(got, and_next)

    def test_feb29_beyond_the_old_one_year_cap(self):
        pivot = datetime(2025, 3, 1, 0, 0)
        # The retired engine gave up after 366 days -> None.
        self.assertIsNone(_old_engine_next("0 0 29 2 *", pivot.timestamp()))
        got = next_cron_fire("0 0 29 2 *", after=pivot.timestamp())
        self.assertEqual(got, datetime(2028, 2, 29, 0, 0).timestamp())

    def test_never_matches_raises_by_name(self):
        with self.assertRaises(CronNeverMatchesError):
            next_cron_fire("0 0 30 2 *")
        # ... and it is a ValueError, so legacy `except ValueError`
        # callers keep working.
        with self.assertRaises(ValueError):
            next_cron_fire("0 0 31 4 *")

    def test_unknown_timezone_raises_by_name(self):
        with self.assertRaises(CronTimezoneError):
            next_cron_fire("0 0 * * *", timezone="Not/AZone")

    def test_grammar_widened_to_the_canonical_one(self):
        # Day names (the retired parser was numbers-only).
        spec = parse_cron("0 9 * * MON")
        monday = _first_monday_not_first().replace(hour=9, minute=0)
        self.assertTrue(spec.matches(monday))
        # dow 7 = Sunday (the retired parser's bounds were 0-6).
        pivot = datetime(2026, 1, 2, 12, 0)  # a Friday
        self.assertEqual(
            next_cron_fire("0 0 * * 7", after=pivot.timestamp()),
            next_cron("0 0 * * 7", pivot),
        )
        # L = last day of month.
        got = next_cron_fire("0 0 L * *", after=datetime(2026, 2, 10).timestamp())
        wall = datetime.fromtimestamp(got)
        self.assertEqual(
            wall.day, _calendar.monthrange(wall.year, wall.month)[1])


class ScheduleTimeHonestyTests(unittest.TestCase):
    """schedule_recurring refuses unschedulable expressions by name."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.addCleanup(self.db.close)
        self.q = WorkQueue(self.db)

    def test_impossible_expression_is_refused_not_stored_dead(self):
        name = f"never-{time.time_ns()}"
        with self.assertRaises(CronNeverMatchesError):
            self.q.schedule_recurring(name, "topic-x", "0 0 30 2 *")
        self.assertNotIn(name, [r["name"] for r in self.q.list_recurring()])

    def test_unknown_timezone_is_refused_not_stored(self):
        name = f"badzone-{time.time_ns()}"
        with self.assertRaises(CronTimezoneError):
            self.q.schedule_recurring(name, "topic-x", "0 6 * * *",
                                      timezone="Not/AZone")
        self.assertNotIn(name, [r["name"] for r in self.q.list_recurring()])

    def test_schedule_with_explicit_timezone(self):
        name = f"lagos-{time.time_ns()}"
        nxt = self.q.schedule_recurring(name, "topic-x", "0 6 * * *",
                                        timezone="Africa/Lagos")
        wall = datetime.fromtimestamp(nxt, ZoneInfo("Africa/Lagos"))
        self.assertEqual((wall.hour, wall.minute), (6, 0))
        self.assertGreater(nxt, time.time())


class LegacyRowHealingTests(unittest.TestCase):
    """tick_recurring heals rows written by the retired engine."""

    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.addCleanup(self.db.close)
        self.q = WorkQueue(self.db)
        self.q._ensure_recurring_table()

    def _insert_legacy(self, name, cron, next_fire_at, created_at,
                       last_fired_at=0.0):
        topic = f"legacy-{name}"
        self.db.execute(
            'INSERT INTO "recurring_jobs" '
            "(name, topic, payload, cron, priority, max_attempts, "
            " queueing_lock, last_fired_at, next_fire_at, created_at, "
            " enabled) VALUES (?, ?, '{}', ?, 0, 5, '', ?, ?, ?, 1)",
            (name, topic, cron, last_fired_at, next_fire_at, created_at),
        )
        return topic

    def _row(self, name):
        rows = [r for r in self.q.list_recurring() if r["name"] == name]
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_dead_zero_row_is_recomputed(self):
        name = f"feb29-{time.time_ns()}"
        created = datetime(2025, 1, 1, 0, 0).timestamp()
        self._insert_legacy(name, "0 0 29 2 *", 0.0, created)
        fired = self.q.tick_recurring(now=datetime(2026, 6, 1).timestamp())
        self.assertNotIn(name, fired)
        self.assertEqual(self._row(name)["next_fire_at"],
                         datetime(2028, 2, 29, 0, 0).timestamp())

    def test_dead_zero_impossible_row_is_reported_by_name(self):
        name = f"feb30-{time.time_ns()}"
        created = datetime(2025, 1, 1, 0, 0).timestamp()
        self._insert_legacy(name, "0 0 30 2 *", 0.0, created)
        fired = self.q.tick_recurring(now=datetime(2026, 6, 1).timestamp())
        self.assertNotIn(name, fired)
        self.assertEqual(self._row(name)["next_fire_at"], 0.0)  # stays dead
        sites = [f.site for f in recent_failures(100)]
        self.assertIn(f"queue.WorkQueue.tick_recurring:{name}", sites)

    def test_and_computed_next_fire_is_tightened_to_or(self):
        monday = _first_monday_not_first()      # Monday 08:00, not the 1st
        anchor = monday.timestamp()
        and_next = _old_engine_next("0 9 1 * 1", anchor)
        self.assertIsNotNone(and_next)
        name = f"tighten-{time.time_ns()}"
        topic = self._insert_legacy(name, "0 9 1 * 1", and_next, anchor,
                                    last_fired_at=anchor)
        # Tick before the OR fire: value tightened, nothing fired yet.
        fired = self.q.tick_recurring(now=anchor + 1800)
        self.assertNotIn(name, fired)
        or_next = monday.replace(hour=9, minute=0).timestamp()
        self.assertEqual(self._row(name)["next_fire_at"], or_next)
        # Tick at the OR fire time: it fires on the canonical schedule.
        fired = self.q.tick_recurring(now=or_next)
        self.assertIn(name, fired)
        self.assertEqual(self.q.pending(topic), 1)


if __name__ == "__main__":
    unittest.main()
