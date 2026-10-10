"""Calendar-grade recurrence math, shared by both schedulers.

Stdlib only (``datetime`` / ``zoneinfo`` / ``calendar``).  Two engines:

1. **Extended cron** — classic 5-field cron plus the Quartz-flavored
   extras that make cron calendar-capable::

       L    day-of-month → last day of the month
       LW   day-of-month → last weekday (Mon–Fri) of the month
       nW   day-of-month → weekday nearest day *n* (Quartz rule:
            Saturday → Friday before, Sunday → Monday after; the 1st on a
            Saturday moves to Monday the 3rd, month-end Sunday to Friday)
       n#k  day-of-week  → k-th *n* weekday of the month (``5#3`` = 3rd Friday)
       nL   day-of-week  → last *n* weekday of the month (``5L`` = last Friday)
       ?    day-of-month or day-of-week → "no specific value" (like ``*``)

   Everything else is classic cron: ``*``, ``*/n``, ``a-b``, ``a-b/n``,
   ``a,b,c``.  Weekday 0 and 7 both mean Sunday.  When day-of-month *and*
   day-of-week are both restricted they keep classic cron OR semantics.

2. **RFC 5545 RRULE** (subset) — the recurrence language of Google
   Calendar and Outlook::

       FREQ=MINUTELY|HOURLY|DAILY|WEEKLY|MONTHLY|YEARLY
       INTERVAL=n  COUNT=n  UNTIL=<date/time>
       BYDAY=MO,TU,2TU,-1FR   (ordinals valid for MONTHLY/YEARLY)
       BYMONTHDAY=1,15,-1     (negative counts back from month end)
       BYMONTH=1,6,12
       BYSETPOS=1,-1          (e.g. last Friday: FREQ=MONTHLY;BYDAY=FR;BYSETPOS=-1)
       WKST=MO                (week start for WEEKLY)

   "every 2nd Tuesday" = ``FREQ=MONTHLY;BYDAY=2TU``.
   "last weekday of the month" = ``FREQ=MONTHLY;BYDAY=MO,TU,WE,TH,FR;BYSETPOS=-1``.

Layering: L4, pure math — no imports from anywhere else in nomorals.
"""

from __future__ import annotations

import calendar as _calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterator
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

__all__ = [
    "CronSpec",
    "parse_cron",
    "cron_matches",
    "next_cron",
    "RRule",
    "parse_rrule",
    "WEEKDAYS",
]

#: Monday=0 … Sunday=6, the ``datetime.weekday()`` convention.
WEEKDAYS = {"MO": 0, "TU": 1, "WE": 2, "TH": 3, "FR": 4, "SA": 5, "SU": 6}

_CRON_RE = re.compile(r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")


# ── extended cron ────────────────────────────────────────────────────────────

def _parse_simple_set(field: str, lo: int, hi: int) -> set[int]:
    """Classic cron field → set of ints (``*``, ``*/n``, ``a-b``, ``a-b/n``, ``a,b``)."""
    field = field.strip()
    out: set[int] = set()
    if field in ("*", "?"):
        return set(range(lo, hi + 1))
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            part, step_s = part.split("/", 1)
            step = int(step_s)
            if step < 1:
                raise ValueError(f"bad cron step in {field!r}")
        if part in ("*", "", "?"):
            lo_p, hi_p = lo, hi
        elif "-" in part:
            a_s, b_s = part.split("-", 1)
            lo_p, hi_p = int(a_s), int(b_s)
        else:
            lo_p = hi_p = int(part)
        if lo_p < lo or hi_p > hi or lo_p > hi_p:
            raise ValueError(f"cron value out of range in {field!r}")
        out.update(range(lo_p, hi_p + 1, step))
    if not out:
        raise ValueError(f"empty cron field {field!r}")
    return out


class _DomMatcher:
    """Day-of-month matcher supporting ``L``, ``LW``, ``nW``."""

    def __init__(self, field: str) -> None:
        field = field.strip()
        self.any = field in ("*", "?")
        self.days: set[int] = set()
        self.last_day = False
        self.last_weekday = False
        self.nearest: set[int] = set()  # days with W suffix
        if self.any:
            return
        for part in field.split(","):
            part = part.strip().upper()
            if not part:
                continue
            if part == "L":
                self.last_day = True
            elif part == "LW":
                self.last_weekday = True
            elif part.endswith("W") and part[:-1].isdigit():
                day = int(part[:-1])
                if not 1 <= day <= 31:
                    raise ValueError(f"bad W day in {field!r}")
                self.nearest.add(day)
            elif re.fullmatch(r"\d+(-\d+)?(/\d+)?", part):
                self.days.update(_parse_simple_set(part, 1, 31))
            else:
                raise ValueError(f"bad day-of-month value {part!r} in {field!r}")
        if not (self.days or self.last_day or self.last_weekday or self.nearest):
            raise ValueError(f"empty day-of-month field {field!r}")

    def matches(self, dt: datetime) -> bool:
        if self.any:
            return True
        year, month, day = dt.year, dt.month, dt.day
        if day in self.days:
            return True
        if self.last_day and day == _calendar.monthrange(year, month)[1]:
            return True
        if self.last_weekday and day == _last_weekday_of_month(year, month):
            return True
        if self.nearest and day in _nearest_weekdays(year, month, self.nearest):
            return True
        return False

    @property
    def restricted(self) -> bool:
        return not self.any


#: Day-name → cron weekday number (Sunday=0).
_DOW_NAMES = {"SUN": 0, "MON": 1, "TUE": 2, "WED": 3, "THU": 4, "FRI": 5, "SAT": 6}


def _pos_int(val: str, what: str) -> int:
    try:
        n = int(val)
    except (TypeError, ValueError):
        raise ValueError(f"bad {what}={val!r}")
    if n < 1:
        raise ValueError(f"{what} must be >= 1, got {val!r}")
    return n


def _parse_until(val: str) -> datetime:
    """Parse an RRULE UNTIL value (date or datetime, optional Z suffix)."""
    text = val.strip()
    utc = text.endswith("Z")
    if utc:
        text = text[:-1]
    for fmt in ("%Y%m%dT%H%M%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S",
                "%Y%m%d", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, fmt)
            if utc:
                from datetime import timezone as _tz
                parsed = parsed.replace(tzinfo=_tz.utc)
            return parsed
        except ValueError:
            continue
    raise ValueError(f"bad UNTIL value {val!r}")


def _dow_name_to_num(token: str) -> str:
    """Rewrite day names (MON, TUE, …) to cron weekday numbers."""
    for name, num in _DOW_NAMES.items():
        token = re.sub(rf"\b{name}\b", str(num), token)
    return token


class _DowMatcher:
    """Day-of-week matcher supporting ``n#k`` and ``nL``."""

    def __init__(self, field: str) -> None:
        field = _dow_name_to_num(field.strip().upper())
        self.any = field in ("*", "?")
        self.days: set[int] = set()       # plain weekday numbers (0/7 = Sunday)
        self.nth: dict[int, int] = {}     # weekday -> k  ("5#3")
        self.last: set[int] = set()       # weekdays with L ("5L")
        if self.any:
            return
        for part in field.split(","):
            part = part.strip().upper()
            if not part:
                continue
            m = re.fullmatch(r"([0-7])#([1-5])", part)
            if m:
                self.nth[int(m.group(1)) % 7] = int(m.group(2))
                continue
            m = re.fullmatch(r"([0-7])L", part)
            if m:
                self.last.add(int(m.group(1)) % 7)
                continue
            if re.fullmatch(r"[0-7](-[0-7])?(/[0-9]+)?", part):
                for v in _parse_simple_set(part, 0, 7):
                    self.days.add(v % 7)
                continue
            raise ValueError(f"bad day-of-week value {part!r} in {field!r}")
        if not (self.days or self.nth or self.last):
            raise ValueError(f"empty day-of-week field {field!r}")

    def matches(self, dt: datetime) -> bool:
        if self.any:
            return True
        py_dow = dt.weekday()          # Monday=0 … Sunday=6
        cron_dow = (py_dow + 1) % 7    # Sunday=0 … Saturday=6
        if cron_dow in self.days:
            return True
        if self.nth and cron_dow in self.nth:
            if _nth_weekday_of_month(dt.year, dt.month, cron_dow,
                                     self.nth[cron_dow]) == dt.day:
                return True
        if self.last and cron_dow in self.last:
            if _last_dow_of_month(dt.year, dt.month, cron_dow) == dt.day:
                return True
        return False

    @property
    def restricted(self) -> bool:
        return not self.any


def _last_weekday_of_month(year: int, month: int) -> int:
    """Day number of the last Monday–Friday of the month."""
    last = _calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last)
    while dt.weekday() >= 5:  # Saturday/Sunday → step back
        dt -= timedelta(days=1)
    return dt.day


def _nearest_weekdays(year: int, month: int, days: set[int]) -> set[int]:
    """Quartz ``nW``: weekday nearest each listed day-of-month."""
    last = _calendar.monthrange(year, month)[1]
    out: set[int] = set()
    for day in days:
        if day > last:
            continue
        wd = datetime(year, month, day).weekday()
        if wd < 5:
            out.add(day)
        elif wd == 5:  # Saturday → Friday before…
            out.add(day - 1 if day > 1 else day + 2)
        else:          # Sunday → Monday after…
            out.add(day + 1 if day < last else day - 2)
    return out


def _nth_weekday_of_month(year: int, month: int, cron_dow: int, n: int) -> int | None:
    """Day number of the n-th ``cron_dow`` (Sun=0) in the month, or None."""
    py_dow = (cron_dow + 6) % 7  # back to Monday=0
    first = datetime(year, month, 1)
    delta = (py_dow - first.weekday()) % 7
    day = 1 + delta + (n - 1) * 7
    return day if day <= _calendar.monthrange(year, month)[1] else None


def _last_dow_of_month(year: int, month: int, cron_dow: int) -> int:
    """Day number of the last ``cron_dow`` (Sun=0) in the month."""
    py_dow = (cron_dow + 6) % 7
    last = _calendar.monthrange(year, month)[1]
    dt = datetime(year, month, last)
    while dt.weekday() != py_dow:
        dt -= timedelta(days=1)
    return dt.day


class CronSpec:
    """A parsed 5-field extended-cron expression.

    Parsed ONCE — :func:`next_cron` reuses the matchers instead of
    re-parsing the expression on every probed minute (the old code
    re-ran the regex + five field parses per minute probed, which made a
    yearly cron scan ~500k redundant parses).
    """

    def __init__(self, expr: str) -> None:
        m = _CRON_RE.match(expr.strip())
        if not m:
            raise ValueError(f"not a 5-field cron expression: {expr!r}")
        minute_s, hour_s, dom_s, month_s, dow_s = m.groups()
        self.minute = _parse_simple_set(minute_s, 0, 59)
        self.hour = _parse_simple_set(hour_s, 0, 23)
        self.dom = _DomMatcher(dom_s)
        self.month = _parse_simple_set(month_s, 1, 12)
        self.dow = _DowMatcher(dow_s)
        # canonical form: normalized fields joined (extras preserved)
        self.expression = (
            f"{minute_s.strip()} {hour_s.strip()} {dom_s.strip().upper()} "
            f"{month_s.strip()} {dow_s.strip().upper()}"
        )

    def matches(self, dt: datetime) -> bool:
        """True if ``dt`` (minute precision) matches."""
        if dt.minute not in self.minute:
            return False
        if dt.hour not in self.hour:
            return False
        if dt.month not in self.month:
            return False
        # classic cron semantics: dom AND dow both restricted → OR them;
        # otherwise each restricted field must match.
        if self.dom.restricted and self.dow.restricted:
            return self.dom.matches(dt) or self.dow.matches(dt)
        return self.dom.matches(dt) and self.dow.matches(dt)


def parse_cron(expr: str) -> CronSpec:
    """Parse (and validate) an extended 5-field cron expression."""
    return CronSpec(expr)


def cron_matches(expr: str, dt: datetime) -> bool:
    """True if ``dt`` matches the cron expression (minute precision)."""
    return CronSpec(expr).matches(dt)


def next_cron(expr: str, after: datetime, timezone: str = "") -> float:
    """Next minute strictly after ``after`` matching the cron expression.

    ``timezone`` is an IANA name — the wall-clock scan runs in that zone
    (DST-safe).  Scans forward minute-by-minute, cap 366 days.  Returns a
    unix timestamp.
    """
    spec = CronSpec(expr)  # parsed once
    tz: Any = None
    if (timezone or "").strip():
        try:
            tz = ZoneInfo(timezone.strip())
        except ZoneInfoNotFoundError:
            tz = None
    probe = after.astimezone(tz) if tz else after
    # strictly after: start at the next minute boundary
    probe = probe.replace(second=0, microsecond=0) + timedelta(minutes=1)
    limit = probe + timedelta(days=366)
    while probe <= limit:
        if spec.matches(probe):
            return probe.timestamp()
        probe += timedelta(minutes=1)
    raise ValueError(f"cron expression never matches within a year: {expr!r}")


# ── RFC 5545 RRULE (subset) ────────────────────────────────────────────────

_FREQS = ("MINUTELY", "HOURLY", "DAILY", "WEEKLY", "MONTHLY", "YEARLY")
_RRULE_PART = re.compile(r"^([A-Z]+)=([^;]+)$")


@dataclass
class RRule:
    """A parsed RFC 5545 recurrence rule (practical subset).

    ``dtstart`` anchors the series (time-of-day for DAILY and coarser,
    exact instant for MINUTELY/HOURLY).  :meth:`after` returns the next
    occurrence strictly after a datetime, or None when the series is
    exhausted (COUNT reached / UNTIL passed / search cap hit).
    """

    freq: str
    dtstart: datetime
    interval: int = 1
    count: int | None = None
    until: datetime | None = None
    byday: list[tuple[int | None, int]] = field(default_factory=list)
    bymonthday: list[int] = field(default_factory=list)
    bymonth: list[int] = field(default_factory=list)
    bysetpos: list[int] = field(default_factory=list)
    wkst: int = 0  # Monday

    # ── parsing ──
    @classmethod
    def parse(cls, text: str, dtstart: datetime | None = None) -> "RRule":
        """Parse ``FREQ=...;INTERVAL=...`` → RRule.  Raises ValueError."""
        text = (text or "").strip()
        if not text:
            raise ValueError("empty RRULE")
        # tolerate a leading "RRULE:" (iCalendar property form)
        if text.upper().startswith("RRULE:"):
            text = text[6:]
        kwargs: dict[str, Any] = {}
        byday: list[tuple[int | None, int]] = []
        bymonthday: list[int] = []
        bymonth: list[int] = []
        bysetpos: list[int] = []
        wkst = 0
        for raw in text.split(";"):
            raw = raw.strip()
            if not raw:
                continue
            m = _RRULE_PART.match(raw.upper())
            if not m:
                raise ValueError(f"bad RRULE part: {raw!r}")
            key, val = m.group(1), m.group(2)
            if key == "FREQ":
                if val not in _FREQS:
                    raise ValueError(f"unsupported FREQ={val!r}")
                kwargs["freq"] = val
            elif key == "INTERVAL":
                kwargs["interval"] = _pos_int(val, "INTERVAL")
            elif key == "COUNT":
                kwargs["count"] = _pos_int(val, "COUNT")
            elif key == "UNTIL":
                kwargs["until"] = _parse_until(val)
            elif key == "BYDAY":
                for tok in val.split(","):
                    tok = tok.strip()
                    dm = re.fullmatch(r"([+-]?\d+)?([A-Z]{2})", tok)
                    if not dm or dm.group(2) not in WEEKDAYS:
                        raise ValueError(f"bad BYDAY token {tok!r}")
                    ordinal = int(dm.group(1)) if dm.group(1) else None
                    if ordinal is not None and not -53 <= ordinal <= 53:
                        raise ValueError(f"BYDAY ordinal out of range: {tok!r}")
                    byday.append((ordinal, WEEKDAYS[dm.group(2)]))
            elif key == "BYMONTHDAY":
                for tok in val.split(","):
                    day = int(tok.strip())
                    if day == 0 or not -31 <= day <= 31:
                        raise ValueError(f"bad BYMONTHDAY value {tok!r}")
                    bymonthday.append(day)
            elif key == "BYMONTH":
                for tok in val.split(","):
                    month = int(tok.strip())
                    if not 1 <= month <= 12:
                        raise ValueError(f"bad BYMONTH value {tok!r}")
                    bymonth.append(month)
            elif key == "BYSETPOS":
                for tok in val.split(","):
                    pos = int(tok.strip())
                    if pos == 0 or not -366 <= pos <= 366:
                        raise ValueError(f"bad BYSETPOS value {tok!r}")
                    bysetpos.append(pos)
            elif key == "WKST":
                if val not in WEEKDAYS:
                    raise ValueError(f"bad WKST={val!r}")
                wkst = WEEKDAYS[val]
            else:
                raise ValueError(f"unsupported RRULE part: {key}")
        if "freq" not in kwargs:
            raise ValueError("RRULE needs FREQ")
        if kwargs["freq"] in ("MINUTELY", "HOURLY") and (
                byday or bymonthday or bymonth or bysetpos):
            raise ValueError("BYDAY/BYMONTHDAY/BYMONTH/BYSETPOS need "
                             "DAILY or coarser FREQ")
        if kwargs["freq"] in ("MINUTELY", "HOURLY", "DAILY", "WEEKLY"):
            if any(o is not None for o, _ in byday):
                raise ValueError("BYDAY ordinals (2TU, -1FR) need "
                                 "MONTHLY or YEARLY FREQ")
        dtstart = dtstart or datetime.now()
        until = kwargs.get("until")
        if until is not None:
            # naive and aware must not meet in comparisons: put UNTIL on
            # the same footing as dtstart.
            if (until.tzinfo is None) != (dtstart.tzinfo is None):
                if dtstart.tzinfo is None:
                    until = until.replace(tzinfo=None)
                else:
                    until = until.replace(tzinfo=dtstart.tzinfo)
            kwargs["until"] = until
        return cls(
            dtstart=dtstart,
            byday=byday, bymonthday=bymonthday, bymonth=bymonth,
            bysetpos=bysetpos, wkst=wkst, **kwargs,
        )

    # ── occurrence generation ──
    def _day_ok(self, day: datetime) -> bool:
        """BYMONTH / BYMONTHDAY / BYDAY filters for one calendar day."""
        if self.bymonth and day.month not in self.bymonth:
            return False
        if self.bymonthday:
            last = _calendar.monthrange(day.year, day.month)[1]
            wanted = {d if d > 0 else last + 1 + d for d in self.bymonthday}
            if day.day not in wanted:
                return False
        if self.byday:
            # plain weekday tokens and ordinal tokens UNION (RFC 5545)
            plains = {wd for o, wd in self.byday if o is None}
            ords = [(o, wd) for o, wd in self.byday if o is not None]
            if plains and day.weekday() in plains:
                return True
            for ordinal, wd in ords:
                if wd != day.weekday():
                    continue
                if ordinal > 0:
                    hit = _nth_weekday_of_month(
                        day.year, day.month, (wd + 1) % 7,
                        ordinal) == day.day
                else:
                    hit = self._neg_ordinal_day(
                        day.year, day.month, wd, ordinal) == day.day
                if hit:
                    return True
            return False
        return True

    @staticmethod
    def _neg_ordinal_day(year: int, month: int, wd: int, ordinal: int) -> int | None:
        """Day number of the -ordinal-th ``wd`` (Mon=0) from month end."""
        last = _calendar.monthrange(year, month)[1]
        dt = datetime(year, month, last)
        while dt.weekday() != wd:
            dt -= timedelta(days=1)
        dt -= timedelta(days=7 * (-ordinal - 1))
        return dt.day if dt.month == month else None

    def _period_starts(self) -> Iterator[datetime]:
        """Yield period anchor datetimes from dtstart, stepping INTERVAL."""
        step = self.interval
        if self.freq == "MINUTELY":
            cur = self.dtstart.replace(second=0, microsecond=0)
            while True:
                yield cur
                cur += timedelta(minutes=step)
        elif self.freq == "HOURLY":
            cur = self.dtstart.replace(minute=0, second=0, microsecond=0)
            while True:
                yield cur
                cur += timedelta(hours=step)
        elif self.freq == "DAILY":
            cur = self.dtstart.replace(hour=0, minute=0, second=0, microsecond=0)
            while True:
                yield cur
                cur += timedelta(days=step)
        elif self.freq == "WEEKLY":
            # anchor on WKST of dtstart's week
            base = self.dtstart.replace(hour=0, minute=0, second=0, microsecond=0)
            delta = (base.weekday() - self.wkst) % 7
            cur = base - timedelta(days=delta)
            while True:
                yield cur
                cur += timedelta(weeks=step)
        elif self.freq == "MONTHLY":
            y, m = self.dtstart.year, self.dtstart.month
            while True:
                yield datetime(y, m, 1)
                m += step
                y += (m - 1) // 12
                m = (m - 1) % 12 + 1
        else:  # YEARLY
            y = self.dtstart.year
            while True:
                yield datetime(y, 1, 1)
                y += step

    def _period_candidates(self, anchor: datetime) -> list[datetime]:
        """All occurrence datetimes inside one period, sorted."""
        tod = (self.dtstart.hour, self.dtstart.minute,
               self.dtstart.second, self.dtstart.microsecond)

        def at(day: datetime) -> datetime:
            out = day.replace(hour=tod[0], minute=tod[1],
                              second=tod[2], microsecond=tod[3])
            if out.tzinfo is None and self.dtstart.tzinfo is not None:
                out = out.replace(tzinfo=self.dtstart.tzinfo)
            return out

        if self.freq == "MINUTELY":
            return [anchor]
        if self.freq == "HOURLY":
            return [anchor]
        if self.freq == "DAILY":
            return [at(anchor)] if self._day_ok(anchor) else []
        if self.freq == "WEEKLY":
            days = [anchor + timedelta(days=i) for i in range(7)]
            want = ({wd for _, wd in self.byday}
                    if self.byday else {self.dtstart.weekday()})
            return sorted(at(d) for d in days
                          if d.weekday() in want and self._day_ok(d))
        if self.freq == "MONTHLY":
            last = _calendar.monthrange(anchor.year, anchor.month)[1]
            days = [datetime(anchor.year, anchor.month, d)
                    for d in range(1, last + 1)]
            if not (self.byday or self.bymonthday or self.bymonth):
                # default: same day-of-month as dtstart (clamped)
                day = min(self.dtstart.day, last)
                days = [datetime(anchor.year, anchor.month, day)]
            cands = sorted(at(d) for d in days if self._day_ok(d))
            return self._apply_bysetpos(cands)
        # YEARLY
        months = self.bymonth or [self.dtstart.month]
        cands: list[datetime] = []
        for month in sorted(set(months)):
            last = _calendar.monthrange(anchor.year, month)[1]
            if not (self.byday or self.bymonthday):
                day = min(self.dtstart.day, last)
                mdays = [datetime(anchor.year, month, day)]
            else:
                mdays = [datetime(anchor.year, month, d)
                         for d in range(1, last + 1)]
            cands.extend(at(d) for d in mdays if self._day_ok(d))
        cands.sort()
        return self._apply_bysetpos(cands)

    def _apply_bysetpos(self, cands: list[datetime]) -> list[datetime]:
        if not self.bysetpos or not cands:
            return cands
        n = len(cands)
        picked = set()
        for pos in self.bysetpos:
            idx = pos - 1 if pos > 0 else n + pos
            if 0 <= idx < n:
                picked.add(cands[idx])
        return sorted(picked)

    def occurrences(self, limit: int = 100_000) -> Iterator[datetime]:
        """Yield occurrences from dtstart (bounded by COUNT/UNTIL/limit)."""
        seen = 0
        for anchor in self._period_starts():
            for occ in self._period_candidates(anchor):
                if occ < self.dtstart:
                    continue
                if self.until is not None and occ > self.until:
                    return
                yield occ
                seen += 1
                if self.count is not None and seen >= self.count:
                    return
                if seen >= limit:
                    return

    def _first_after(self, after: datetime, inclusive: bool) -> datetime | None:
        for occ in self.occurrences():
            if occ > after or (inclusive and occ == after):
                return occ
            # safety: don't scan absurdly far for a far-future `after`
            if occ > after + timedelta(days=366 * 5):
                break
        return None

    def after(self, after: datetime, *, inclusive: bool = False) -> datetime | None:
        """Next occurrence strictly after ``after`` (``>=`` if inclusive).

        Naive/tz-aware datetimes compare against dtstart as given; mixing
        naive and aware raises TypeError from the comparison — keep both
        on the same footing.
        """
        return self._first_after(after, inclusive)

    # ── convenience ──
    def next_timestamp(self, after_ts: float, tz: str = "") -> float | None:
        """Next occurrence after a unix timestamp, in an IANA zone."""
        zone = None
        if (tz or "").strip():
            try:
                zone = ZoneInfo(tz.strip())
            except ZoneInfoNotFoundError:
                zone = None
        after = datetime.fromtimestamp(after_ts, tz=zone)
        nxt = self.after(after)
        return nxt.timestamp() if nxt is not None else None


def parse_rrule(text: str, dtstart: datetime | None = None) -> RRule:
    """Parse and validate an RRULE string → :class:`RRule`."""
    return RRule.parse(text, dtstart)