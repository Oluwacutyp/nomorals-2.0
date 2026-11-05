"""Exhaustive cron proof suite — Echelon Wave 5 (crontab.rs standard).

The donor crate's bar was an *exhaustive* next-event proof, not spot
examples.  This suite holds ``nomorals.scheduler.recurrence`` to it:

* an INDEPENDENT reference matcher, written here from the documented
  cron semantics (classic 5-field + the Quartz extras the module
  docstring promises) using only ``calendar`` — it shares no code
  with ``CronSpec`` / ``_DomMatcher`` / ``_DowMatcher``;
* two independent oracles built on it — brute-force minute stepping
  over *instants* (DST-exact by construction) and an analytic
  day-granularity constructor — cross-checked against each other and
  against the engine's ``next_cron`` / ``prev_cron``;
* a generated grammar sweep (every field grammar × representative
  values), a DST battery (gap / overlap, incl. the 30-minute
  Lord Howe shift), leap-year and month-end edges (incl. Feb 29
  across the non-leap year 2100), impossible dates refused BY NAME,
  and unknown timezones refused BY NAME.

Pinned DST semantics (the engine's contract): a wall time that never
occurs (spring-forward gap) is skipped — no firing that day; a wall
time that occurs twice (fall-back overlap) fires at the first
occurrence, unless the pivot already sits between the two, in which
case the second occurrence is the next firing.  The returned instant,
localized, always satisfies the expression.
"""

from __future__ import annotations

import calendar as _cal
import re
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from nomorals.scheduler.recurrence import (
    CronNeverMatchesError,
    CronSpec,
    CronTimezoneError,
    next_cron,
    prev_cron,
    validate_cron,
)

UTC = timezone.utc
DENVER = "America/Denver"
LAGOS = "Africa/Lagos"
LORD_HOWE = "Australia/Lord_Howe"

# ── independent reference implementation (from the spec text) ────────

_REF_DOW_NAMES = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3,
                  "THU": 4, "FRI": 5, "SAT": 6}


def _ref_simple(field: str, lo: int, hi: int) -> set[int]:
    """Classic cron field → set (``*``, ``*/n``, ``a``, ``a-b``,
    ``a-b/n``, ``a/n`` = a..max step n, comma lists)."""
    field = field.strip()
    if field in ("*", "?"):
        return set(range(lo, hi + 1))
    out: set[int] = set()
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        stepped = "/" in part
        if stepped:
            part, step_s = part.split("/", 1)
            step = int(step_s)
        if part in ("*", ""):
            a, b = lo, hi
        elif "-" in part:
            a_s, b_s = part.split("-", 1)
            a, b = int(a_s), int(b_s)
        elif stepped:
            a, b = int(part), hi
        else:
            a = b = int(part)
        out.update(range(a, b + 1, step))
    return out


def _ref_last_weekday(y: int, m: int) -> int:
    last = _cal.monthrange(y, m)[1]
    d = last
    while datetime(y, m, d).weekday() >= 5:
        d -= 1
    return d


def _ref_nearest_weekday(y: int, m: int, n: int) -> int | None:
    """Quartz ``nW`` per the documented rule, from calendar directly."""
    last = _cal.monthrange(y, m)[1]
    if n > last:
        return None
    wd = datetime(y, m, n).weekday()  # Mon=0 … Sun=6
    if wd < 5:
        return n
    if wd == 5:  # Saturday → Friday before, except the 1st → Monday 3rd
        return n - 1 if n > 1 else n + 2
    # Sunday → Monday after, except month-end → Friday before
    return n + 1 if n < last else n - 2


def _ref_dow_column(y: int, m: int, cron_dow: int) -> list[int]:
    """All days of month ``y-m`` falling on ``cron_dow`` (Sun=0)."""
    py_wd = (cron_dow + 6) % 7  # to Mon=0 convention
    weeks = _cal.monthcalendar(y, m)  # weeks start Monday
    return [week[py_wd] for week in weeks if week[py_wd] != 0]


def _ref_dom_match(y: int, m: int, d: int, field: str) -> tuple[bool, bool]:
    """(matches, restricted) for the day-of-month field string."""
    field = field.strip().upper()
    if field in ("*", "?"):
        return True, False
    last = _cal.monthrange(y, m)[1]
    for part in field.split(","):
        part = part.strip()
        if part == "L":
            if d == last:
                return True, True
        elif part == "LW":
            if d == _ref_last_weekday(y, m):
                return True, True
        elif part.endswith("W") and part[:-1].isdigit():
            if _ref_nearest_weekday(y, m, int(part[:-1])) == d:
                return True, True
        else:
            if d in _ref_simple(part, 1, 31):
                return True, True
    return False, True


def _ref_dow_match(y: int, m: int, d: int, field: str) -> tuple[bool, bool]:
    """(matches, restricted) for the day-of-week field string."""
    raw = field.strip().upper()
    if raw in ("*", "?"):
        return True, False
    for name, num in _REF_DOW_NAMES.items():
        raw = re.sub(rf"\b{name}\b", str(num), raw)
    cron_dow = (datetime(y, m, d).weekday() + 1) % 7  # Sun=0
    for part in raw.split(","):
        part = part.strip()
        mm = re.fullmatch(r"([0-7])#([1-5])", part)
        if mm:
            col = _ref_dow_column(y, m, int(mm.group(1)) % 7)
            k = int(mm.group(2))
            if k <= len(col) and col[k - 1] == d:
                return True, True
            continue
        mm = re.fullmatch(r"([0-7])L", part)
        if mm:
            col = _ref_dow_column(y, m, int(mm.group(1)) % 7)
            if col and col[-1] == d:
                return True, True
            continue
        vals = {v % 7 for v in _ref_simple(part, 0, 7)}
        if cron_dow in vals:
            return True, True
    return False, True


def ref_matches(dt: datetime, expr: str) -> bool:
    """Independent matcher: does aware/naive ``dt`` satisfy ``expr``?"""
    minute_s, hour_s, dom_s, month_s, dow_s = expr.strip().split()
    if dt.minute not in _ref_simple(minute_s, 0, 59):
        return False
    if dt.hour not in _ref_simple(hour_s, 0, 23):
        return False
    if dt.month not in _ref_simple(month_s, 1, 12):
        return False
    dom_ok, dom_res = _ref_dom_match(dt.year, dt.month, dt.day, dom_s)
    dow_ok, dow_res = _ref_dow_match(dt.year, dt.month, dt.day, dow_s)
    if dom_res and dow_res:
        return dom_ok or dow_ok  # classic cron OR semantics
    return dom_ok and dow_ok


# ── oracle 1: brute-force minute stepping over instants ──────────────

def oracle_step_next(expr: str, after: datetime, tz_name: str = "",
                     max_days: int = 400) -> float | None:
    """Step minute-aligned instants forward; first whose localized
    wall time satisfies the reference matcher wins.  Instant stepping
    is DST-exact: gap wall times never occur, overlap wall times
    occur at both of their instants."""
    tz = ZoneInfo(tz_name) if tz_name else after.tzinfo
    start = after.timestamp()
    cursor = int(start // 60) * 60 + 60
    limit = start + max_days * 86400
    while cursor <= limit:
        loc = datetime.fromtimestamp(cursor, tz)
        if ref_matches(loc, expr):
            return float(cursor)
        cursor += 60
    return None


def oracle_step_prev(expr: str, before: datetime, tz_name: str = "",
                     max_days: int = 400) -> float | None:
    tz = ZoneInfo(tz_name) if tz_name else before.tzinfo
    start = before.timestamp()
    cursor = int(start // 60) * 60
    if cursor >= start:
        cursor -= 60
    limit = start - max_days * 86400
    while cursor >= limit:
        loc = datetime.fromtimestamp(cursor, tz)
        if ref_matches(loc, expr):
            return float(cursor)
        cursor -= 60
    return None


# ── oracle 2: analytic day-granularity construction ──────────────────

def _oracle_wall_instants(y: int, m: int, d: int, hh: int, mm: int,
                          tz: ZoneInfo | None) -> list[float]:
    naive = datetime(y, m, d, hh, mm)
    if tz is None:
        return [naive.timestamp()]
    out: list[float] = []
    for fold in (0, 1):
        cand = naive.replace(tzinfo=tz, fold=fold)
        ts = cand.timestamp()
        back = datetime.fromtimestamp(ts, tz)
        if (back.year, back.month, back.day, back.hour, back.minute) == \
                (y, m, d, hh, mm) and ts not in out:
            out.append(ts)
    return sorted(out)


def oracle_analytic_next(expr: str, after: datetime,
                         tz_name: str = "") -> float | None:
    tz = ZoneInfo(tz_name) if tz_name else after.tzinfo
    minute_s, hour_s, dom_s, month_s, dow_s = expr.strip().split()
    minutes = _ref_simple(minute_s, 0, 59)
    hours = _ref_simple(hour_s, 0, 23)
    months = _ref_simple(month_s, 1, 12)
    local = after.astimezone(tz) if tz is not None else after
    start_ts = after.timestamp()
    day = local.date()
    for _ in range(2931):
        if day.month in months:
            probe = datetime(day.year, day.month, day.day)
            dom_ok, dom_res = _ref_dom_match(
                day.year, day.month, day.day, dom_s)
            dow_ok, dow_res = _ref_dow_match(
                day.year, day.month, day.day, dow_s)
            day_ok = (dom_ok or dow_ok) if (dom_res and dow_res) \
                else (dom_ok and dow_ok)
            if day_ok:
                cands: list[float] = []
                for hh in hours:
                    for mm in minutes:
                        cands.extend(_oracle_wall_instants(
                            day.year, day.month, day.day, hh, mm, tz))
                for ts in sorted(cands):
                    if ts > start_ts:
                        return ts
        day += timedelta(days=1)
    return None


def oracle_analytic_prev(expr: str, before: datetime,
                         tz_name: str = "") -> float | None:
    tz = ZoneInfo(tz_name) if tz_name else before.tzinfo
    minute_s, hour_s, dom_s, month_s, dow_s = expr.strip().split()
    minutes = _ref_simple(minute_s, 0, 59)
    hours = _ref_simple(hour_s, 0, 23)
    months = _ref_simple(month_s, 1, 12)
    local = before.astimezone(tz) if tz is not None else before
    start_ts = before.timestamp()
    day = local.date()
    for _ in range(2931):
        if day.month in months:
            dom_ok, dom_res = _ref_dom_match(
                day.year, day.month, day.day, dom_s)
            dow_ok, dow_res = _ref_dow_match(
                day.year, day.month, day.day, dow_s)
            day_ok = (dom_ok or dow_ok) if (dom_res and dow_res) \
                else (dom_ok and dow_ok)
            if day_ok:
                cands: list[float] = []
                for hh in hours:
                    for mm in minutes:
                        cands.extend(_oracle_wall_instants(
                            day.year, day.month, day.day, hh, mm, tz))
                for ts in sorted(cands, reverse=True):
                    if ts < start_ts:
                        return ts
        day -= timedelta(days=1)
    return None


def assert_engine_next(tc: unittest.TestCase, expr: str,
                       after: datetime, tz_name: str = "",
                       oracle: str = "analytic") -> float:
    """Engine next == oracle next, and the invariant holds."""
    got = next_cron(expr, after, tz_name)
    want = (oracle_analytic_next if oracle == "analytic"
            else oracle_step_next)(expr, after, tz_name)
    tc.assertIsNotNone(want, f"oracle found no firing for {expr!r}")
    tc.assertEqual(got, want, f"next_cron({expr!r}) after {after} {tz_name}")
    tz = ZoneInfo(tz_name) if tz_name else after.tzinfo
    loc = datetime.fromtimestamp(got, tz)
    tc.assertTrue(CronSpec(expr).matches(loc),
                  f"returned instant {loc} does not itself match {expr!r}")
    tc.assertTrue(ref_matches(loc, expr))
    tc.assertGreater(got, after.timestamp())
    return got


def assert_engine_prev(tc: unittest.TestCase, expr: str,
                       before: datetime, tz_name: str = "",
                       oracle: str = "analytic") -> float:
    got = prev_cron(expr, before, tz_name)
    want = (oracle_analytic_prev if oracle == "analytic"
            else oracle_step_prev)(expr, before, tz_name)
    tc.assertIsNotNone(want, f"oracle found no firing for {expr!r}")
    tc.assertEqual(got, want, f"prev_cron({expr!r}) before {before} {tz_name}")
    tz = ZoneInfo(tz_name) if tz_name else before.tzinfo
    loc = datetime.fromtimestamp(got, tz)
    tc.assertTrue(CronSpec(expr).matches(loc))
    tc.assertTrue(ref_matches(loc, expr))
    tc.assertLess(got, before.timestamp())
    return got


# ── 1. generated grammar sweep vs the analytic oracle ────────────────

_MINUTES = ["*", "*/1", "*/7", "*/15", "0", "59", "0,30", "5-10",
            "10-50/10", "*/59", "5/15", "58,59"]
_HOURS = ["*", "*/6", "0", "23", "9-17", "0,12", "22-23", "3/8"]
_DOMS = ["*", "1", "28", "31", "15,31", "1-7", "*/10", "10-20/5"]
_MONTHS = ["*", "1", "2", "12", "1,6,12", "3-5", "2/3"]
_DOWS = ["*", "0", "7", "1", "1-5", "5", "0,6", "MON-FRI", "SUN",
         "*/2", "2/2"]
_PIVOTS = [
    datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
    datetime(2026, 2, 27, 23, 59, tzinfo=UTC),
    datetime(2026, 12, 31, 23, 58, tzinfo=UTC),
    datetime(2027, 6, 30, 0, 0, tzinfo=UTC),
]


def _sweep_exprs() -> list[str]:
    exprs: list[str] = []
    for mi in _MINUTES:
        for hh in _HOURS:
            exprs.append(f"{mi} {hh} * * *")
    for dom in _DOMS:
        for mi, hh in (("0", "0"), ("30", "9"), ("*/20", "*/5")):
            exprs.append(f"{mi} {hh} {dom} * *")
    for mon in _MONTHS:
        exprs.append(f"15 4 10 {mon} *")
        exprs.append(f"15 4 29 {mon} *")
    for dow in _DOWS:
        for mi, hh in (("0", "9"), ("45", "23")):
            exprs.append(f"{mi} {hh} * * {dow}")
    # dom AND dow both restricted → OR semantics, exercised against
    # one-restricted controls in the dedicated class below as well.
    exprs += ["0 0 13 * 5", "0 0 1 * 1", "30 6 15 3 2", "0 0 29 2 4"]
    return exprs


class GrammarSweepTests(unittest.TestCase):
    """Every field grammar × representatives, next and prev, vs oracle."""

    def test_next_matches_oracle(self):
        for expr in _sweep_exprs():
            for pivot in _PIVOTS:
                with self.subTest(expr=expr, pivot=pivot):
                    assert_engine_next(self, expr, pivot)

    def test_prev_matches_oracle(self):
        for expr in _sweep_exprs():
            for pivot in _PIVOTS[::2]:
                with self.subTest(expr=expr, pivot=pivot):
                    assert_engine_prev(self, expr, pivot)


class SteppingOracleCrossCheckTests(unittest.TestCase):
    """Brute-force instant stepping agrees on dense/boundary cases."""

    CASES = [
        ("*/15 * * * *", datetime(2026, 3, 1, 0, 0, tzinfo=UTC)),
        ("*/15 * * * *", datetime(2026, 2, 28, 23, 59, 30, tzinfo=UTC)),
        ("0 * * * *", datetime(2026, 1, 31, 23, 0, tzinfo=UTC)),
        ("59 23 * * *", datetime(2026, 12, 31, 23, 59, tzinfo=UTC)),
        ("0 0 1 * *", datetime(2026, 1, 1, 0, 0, tzinfo=UTC)),
        ("0 0 29 * *", datetime(2026, 1, 30, 12, 0, tzinfo=UTC)),
        ("30 4 1,15 * *", datetime(2026, 4, 15, 4, 30, tzinfo=UTC)),
        ("0 9 * * MON-FRI", datetime(2026, 1, 2, 9, 0, tzinfo=UTC)),
        ("0 9 * * MON-FRI", datetime(2026, 1, 3, 10, 0, tzinfo=UTC)),
        ("5/15 * * * *", datetime(2026, 5, 5, 5, 5, tzinfo=UTC)),
        ("0 0 * * */2", datetime(2026, 5, 5, 0, 0, tzinfo=UTC)),
        ("0 0 */2 * *", datetime(2026, 5, 5, 0, 0, tzinfo=UTC)),
    ]

    def test_next_stepping(self):
        for expr, pivot in self.CASES:
            with self.subTest(expr=expr, pivot=pivot):
                assert_engine_next(self, expr, pivot, oracle="step")

    def test_prev_stepping(self):
        for expr, pivot in self.CASES:
            with self.subTest(expr=expr, pivot=pivot):
                assert_engine_prev(self, expr, pivot, oracle="step")

    def test_oracles_agree_with_each_other(self):
        for expr, pivot in self.CASES:
            with self.subTest(expr=expr, pivot=pivot):
                self.assertEqual(oracle_step_next(expr, pivot),
                                 oracle_analytic_next(expr, pivot))


# ── 2. DST battery ────────────────────────────────────────────────────

class DstTests(unittest.TestCase):
    """Gap times skipped; overlap times first-occurrence (or second
    when the pivot already sits between the two occurrences)."""

    def test_denver_spring_forward_gap_day_skipped(self):
        # 2026-03-08 02:30 never happens in Denver.
        pivot = datetime(2026, 3, 7, 12, 0, tzinfo=ZoneInfo(DENVER))
        got = assert_engine_next(self, "30 2 * * *", pivot, DENVER,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(DENVER))
        self.assertEqual((loc.month, loc.day, loc.hour, loc.minute),
                         (3, 9, 2, 30))

    def test_denver_spring_forward_minutely_series_unbroken(self):
        pivot = datetime(2026, 3, 8, 0, 30, tzinfo=ZoneInfo(DENVER))
        got = assert_engine_next(self, "*/30 * * * *", pivot, DENVER,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(DENVER))
        self.assertEqual((loc.hour, loc.minute), (1, 0))

    def test_denver_fall_back_first_occurrence(self):
        pivot = datetime(2026, 10, 31, 12, 0, tzinfo=ZoneInfo(DENVER))
        got = assert_engine_next(self, "30 1 * * *", pivot, DENVER,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(DENVER))
        self.assertEqual((loc.hour, loc.minute, loc.fold), (1, 30, 0))
        self.assertEqual(loc.utcoffset(), timedelta(hours=-6))  # MDT

    def test_denver_fall_back_pivot_between_occurrences(self):
        # Pivot sits in the SECOND pass (MST) before 01:30: the next
        # firing is the second occurrence — never an instant in the
        # past (the incumbent stepping engine returned fold=0 here).
        pivot = datetime(2026, 11, 1, 1, 15,
                         tzinfo=ZoneInfo(DENVER)).replace(fold=1)
        got = assert_engine_next(self, "30 1 * * *", pivot, DENVER,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(DENVER))
        self.assertEqual((loc.hour, loc.minute, loc.fold), (1, 30, 1))
        self.assertEqual(loc.utcoffset(), timedelta(hours=-7))  # MST

    def test_denver_fall_back_prev_between_occurrences(self):
        pivot = datetime(2026, 11, 1, 1, 45,
                         tzinfo=ZoneInfo(DENVER)).replace(fold=1)
        got = assert_engine_prev(self, "30 1 * * *", pivot, DENVER,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(DENVER))
        self.assertEqual((loc.hour, loc.minute, loc.fold), (1, 30, 1))

    def test_denver_gap_prev_skips_gap_day(self):
        pivot = datetime(2026, 3, 9, 1, 0, tzinfo=ZoneInfo(DENVER))
        got = assert_engine_prev(self, "30 2 * * *", pivot, DENVER,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(DENVER))
        self.assertEqual((loc.month, loc.day), (3, 7))

    def test_lord_howe_half_hour_shift(self):
        # Lord Howe shifts only 30 minutes: spring 2026-10-04
        # 02:00 → 02:30 (+10:30 → +11:00); 02:15 never happens.
        pivot = datetime(2026, 10, 3, 12, 0, tzinfo=ZoneInfo(LORD_HOWE))
        got = assert_engine_next(self, "15 2 * * *", pivot, LORD_HOWE,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(LORD_HOWE))
        self.assertEqual((loc.day, loc.hour, loc.minute), (5, 2, 15))
        # Fall: 2026-04-05 02:00 → 01:30 (+11 → +10:30); 01:45 occurs
        # twice, 30 minutes apart.
        pivot = datetime(2026, 4, 4, 12, 0, tzinfo=ZoneInfo(LORD_HOWE))
        got = assert_engine_next(self, "45 1 * * *", pivot, LORD_HOWE,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(LORD_HOWE))
        self.assertEqual((loc.day, loc.hour, loc.minute, loc.fold),
                         (5, 1, 45, 0))
        pivot2 = datetime(2026, 4, 5, 1, 50,
                          tzinfo=ZoneInfo(LORD_HOWE)).replace(fold=1)
        got2 = assert_engine_next(self, "45 1 * * *", pivot2, LORD_HOWE,
                                  oracle="step")
        loc2 = datetime.fromtimestamp(got2, ZoneInfo(LORD_HOWE))
        self.assertEqual((loc2.day, loc2.hour, loc2.minute), (6, 1, 45))

    def test_lagos_no_dst_daily_anchor(self):
        # The 06:00 WAT briefing anchor: no DST, exact daily cadence
        # across a month boundary.
        pivot = datetime(2026, 1, 30, 12, 0, tzinfo=ZoneInfo(LAGOS))
        got = assert_engine_next(self, "0 6 * * *", pivot, LAGOS,
                                 oracle="step")
        loc = datetime.fromtimestamp(got, ZoneInfo(LAGOS))
        self.assertEqual((loc.day, loc.hour), (31, 6))
        got2 = next_cron("0 6 * * *",
                         datetime.fromtimestamp(got, ZoneInfo(LAGOS)),
                         LAGOS)
        loc2 = datetime.fromtimestamp(got2, ZoneInfo(LAGOS))
        self.assertEqual((loc2.month, loc2.day, loc2.hour), (2, 1, 6))
        self.assertEqual(got2 - got, 86400.0)


# ── 3. leap years, month-ends, extended fields ───────────────────────

class LeapAndMonthEndTests(unittest.TestCase):
    def test_feb29_next_leap(self):
        got = next_cron("0 0 29 2 *", datetime(2025, 3, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2028, 2, 29, tzinfo=UTC))

    def test_feb29_across_non_leap_century(self):
        # 2100 is NOT a leap year: 2096 → 2104 is an 8-year gap.
        got = next_cron("0 0 29 2 *", datetime(2096, 3, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2104, 2, 29, tzinfo=UTC))
        got = prev_cron("0 0 29 2 *", datetime(2103, 3, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2096, 2, 29, tzinfo=UTC))

    def test_feb29_prev(self):
        got = prev_cron("0 0 29 2 *", datetime(2025, 3, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2024, 2, 29, tzinfo=UTC))

    def test_last_day_of_month(self):
        cases = [
            (datetime(2026, 1, 15, tzinfo=UTC), datetime(2026, 1, 31, 12, tzinfo=UTC)),
            (datetime(2026, 2, 1, tzinfo=UTC), datetime(2026, 2, 28, 12, tzinfo=UTC)),
            (datetime(2023, 12, 1, tzinfo=UTC), datetime(2024, 2, 29, 12, tzinfo=UTC)),
            (datetime(2026, 4, 1, tzinfo=UTC), datetime(2026, 4, 30, 12, tzinfo=UTC)),
        ]
        # "0 12 L 2 *"-style scoping uses one month; here L in Feb only
        # for the Feb cases — use per-case expressions instead.
        for pivot, want in cases:
            expr = f"0 12 L {want.month} *"
            with self.subTest(expr=expr, pivot=pivot):
                self.assertEqual(
                    datetime.fromtimestamp(next_cron(expr, pivot), UTC),
                    want)

    def test_31st_skips_short_months(self):
        got = next_cron("0 9 31 * *", datetime(2026, 4, 15, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 5, 31, 9, tzinfo=UTC))

    def test_last_weekday_of_month(self):
        # Feb 2026: the 28th is a Saturday → last weekday is Fri 27th.
        got = next_cron("0 9 LW * *", datetime(2026, 2, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 2, 27, 9, tzinfo=UTC))

    def test_nearest_weekday(self):
        # 15 Aug 2026 is a Saturday → nearest weekday is Fri 14th.
        got = next_cron("0 9 15W * *", datetime(2026, 8, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 8, 14, 9, tzinfo=UTC))
        # 15 Feb 2026 is a Sunday → Monday 16th.
        got = next_cron("0 9 15W * *", datetime(2026, 2, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 2, 16, 9, tzinfo=UTC))

    def test_nth_and_last_weekday(self):
        # Jan 2026 Fridays: 2, 9, 16, 23, 30.
        got = next_cron("0 9 * * 5#3", datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 16, 9, tzinfo=UTC))
        got = next_cron("0 9 * * 5L", datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 30, 9, tzinfo=UTC))
        got = next_cron("0 9 * * FRI#3", datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 16, 9, tzinfo=UTC))

    def test_extended_fields_vs_oracle(self):
        exprs = ["0 9 L * *", "0 9 LW * *", "0 9 1W * *", "0 9 15W * *",
                 "0 9 31W * *", "0 9 * * 1#2", "0 9 * * 6#5",
                 "0 9 * * 0L", "0 9 * * 3L", "15 3 L 2 *"]
        for expr in exprs:
            for pivot in _PIVOTS:
                with self.subTest(expr=expr, pivot=pivot):
                    assert_engine_next(self, expr, pivot)
                    assert_engine_prev(self, expr, pivot)

    def test_dom_dow_or_semantics(self):
        # "13th OR Friday": from Sat Jan 10 2026 the next hit is
        # Tue Jan 13 (the 13th), not the next Friday (Jan 16).
        got = next_cron("0 0 13 * 5", datetime(2026, 1, 10, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 13, tzinfo=UTC))
        # From Jan 3: Fri Jan 9 comes before the 13th.
        got = next_cron("0 0 13 * 5", datetime(2026, 1, 3, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 9, tzinfo=UTC))

    def test_year_rollover(self):
        got = next_cron("59 23 31 12 *",
                        datetime(2026, 12, 31, 23, 59, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2027, 12, 31, 23, 59, tzinfo=UTC))
        got = prev_cron("0 0 1 1 *", datetime(2026, 1, 1, tzinfo=UTC))
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2025, 1, 1, tzinfo=UTC))

    def test_sunday_aliases_agree(self):
        for pivot in _PIVOTS:
            a = next_cron("0 0 * * 0", pivot)
            b = next_cron("0 0 * * 7", pivot)
            c = next_cron("0 0 * * SUN", pivot)
            self.assertEqual(a, b)
            self.assertEqual(a, c)


# ── 4. impossible dates + named refusals ─────────────────────────────

class ImpossibleAndRefusalTests(unittest.TestCase):
    IMPOSSIBLE = ["0 0 31 2 *", "0 0 30 2 *", "0 0 31 4 *",
                  "0 0 31 6 *", "0 0 31 9 *", "0 0 31 11 *",
                  "0 0 31 4,6,9,11 *", "0 0 30,31 2 *"]

    def test_impossible_refused_by_name(self):
        pivot = datetime(2026, 1, 1, tzinfo=UTC)
        for expr in self.IMPOSSIBLE:
            with self.subTest(expr=expr):
                with self.assertRaises(CronNeverMatchesError):
                    next_cron(expr, pivot)
                with self.assertRaises(CronNeverMatchesError):
                    prev_cron(expr, pivot)
                with self.assertRaises(CronNeverMatchesError):
                    validate_cron(expr)
                # …and it is still a ValueError for legacy callers.
                with self.assertRaises(ValueError):
                    next_cron(expr, pivot)

    def test_or_semantics_rescue_otherwise_impossible_dom(self):
        # Feb 31 never exists, but OR with Sunday (dow restricted)
        # fires on February Sundays.
        got = next_cron("0 0 31 2 0", datetime(2026, 1, 1, tzinfo=UTC))
        loc = datetime.fromtimestamp(got, UTC)
        self.assertEqual((loc.month, loc.day), (2, 1))  # Sun 1 Feb 2026
        self.assertEqual(loc.weekday(), 6)

    def test_unknown_timezone_refused_by_name(self):
        pivot = datetime(2026, 1, 1, tzinfo=UTC)
        for bad in ("America/Denverr", "Nowhere/Place", "GMT+9x"):
            with self.subTest(tz=bad):
                with self.assertRaises(CronTimezoneError) as cm:
                    next_cron("0 9 * * *", pivot, bad)
                self.assertIn(bad, str(cm.exception))
                with self.assertRaises(CronTimezoneError):
                    prev_cron("0 9 * * *", pivot, bad)
                with self.assertRaises(CronTimezoneError):
                    validate_cron("0 9 * * *", bad)

    def test_garbage_is_parse_error_not_never_matches(self):
        pivot = datetime(2026, 1, 1, tzinfo=UTC)
        for bad in ("", "not a cron", "61 * * * *", "* * * *",
                    "0 0 * JAN *", "0 0 * * 8", "*/0 * * * *",
                    "0 0 32 * *", "0 24 * * *"):
            with self.subTest(expr=bad):
                with self.assertRaises(ValueError) as cm:
                    next_cron(bad, pivot)
                self.assertNotIsInstance(cm.exception,
                                         CronNeverMatchesError)

    def test_validate_cron_accepts_satisfiable(self):
        for expr in ("0 0 29 2 *", "0 9 * * MON-FRI", "*/15 * * * *",
                     "0 9 L * *", "0 0 31 2 0"):
            with self.subTest(expr=expr):
                spec = validate_cron(expr, DENVER)
                self.assertIsInstance(spec, CronSpec)


# ── 5. strictness + round-trip ────────────────────────────────────────

class StrictnessAndRoundTripTests(unittest.TestCase):
    def test_strictly_after_on_exact_firing(self):
        pivot = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        got = next_cron("0 9 * * *", pivot)
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 2, 9, 0, tzinfo=UTC))

    def test_strictly_before_on_exact_firing(self):
        pivot = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
        got = prev_cron("0 9 * * *", pivot)
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2025, 12, 31, 9, 0, tzinfo=UTC))

    def test_pivot_with_seconds(self):
        pivot = datetime(2026, 1, 1, 8, 59, 30, tzinfo=UTC)
        got = next_cron("0 9 * * *", pivot)
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 1, 9, 0, tzinfo=UTC))
        pivot = datetime(2026, 1, 1, 9, 0, 30, tzinfo=UTC)
        got = prev_cron("0 9 * * *", pivot)
        self.assertEqual(datetime.fromtimestamp(got, UTC),
                         datetime(2026, 1, 1, 9, 0, tzinfo=UTC))

    def test_naive_pivot_keeps_naive_footing(self):
        pivot = datetime(2026, 1, 1, 8, 0)  # naive, caller's footing
        got = next_cron("0 9 * * *", pivot)
        self.assertEqual(datetime.fromtimestamp(got),
                         datetime(2026, 1, 1, 9, 0))

    def test_next_prev_round_trip(self):
        exprs = ["*/15 * * * *", "0 9 * * MON-FRI", "0 0 29 2 *",
                 "0 12 L * *", "30 2 * * *", "0 0 13 * 5"]
        for expr in exprs:
            for pivot in _PIVOTS:
                with self.subTest(expr=expr, pivot=pivot):
                    f1 = next_cron(expr, pivot, DENVER)
                    f0 = prev_cron(expr, datetime.fromtimestamp(f1, UTC),
                                   DENVER)
                    # prev(f1) is a firing < f1, and next of it is f1
                    self.assertLess(f0, f1)
                    again = next_cron(
                        expr, datetime.fromtimestamp(f0, UTC), DENVER)
                    self.assertEqual(again, f1)


if __name__ == "__main__":
    unittest.main()
