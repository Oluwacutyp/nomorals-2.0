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
    "prev_cron",
    "describe_cron",
    "describe_rrule",
    "parse_natural_schedule",
    "parse_natural_datetime",
    "parse_rrule_set",
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
    #: Datetimes excluded even when the rule matches them (dateutil
    #: ``rruleset.exdate`` semantics — "delete this instance", the Google
    #: Calendar model).
    exdates: tuple[datetime, ...] = ()
    #: Extra datetimes merged into the stream in order (dateutil
    #: ``rruleset.rdate`` semantics — one-off additions to a series).
    rdates: tuple[datetime, ...] = ()

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
        """Yield occurrences from dtstart (bounded by COUNT/UNTIL/limit).

        ``exdates`` are skipped even when the rule matches them and
        ``rdates`` are merged into the stream in chronological order
        (dateutil ``rruleset`` semantics).  COUNT / UNTIL / ``limit``
        apply to the merged stream.
        """
        exset = set(self.exdates)
        extras = sorted(
            d for d in set(self.rdates)
            if d >= self.dtstart and d not in exset
            and (self.until is None or d <= self.until))

        def base() -> Iterator[datetime]:
            for anchor in self._period_starts():
                for occ in self._period_candidates(anchor):
                    if occ < self.dtstart:
                        continue
                    if self.until is not None and occ > self.until:
                        return
                    if occ in exset:
                        continue
                    yield occ

        stream = base()
        nxt = next(stream, None)
        idx = 0
        seen = 0
        while True:
            extra = extras[idx] if idx < len(extras) else None
            if nxt is None and extra is None:
                return
            if extra is not None and (nxt is None or extra <= nxt):
                occ = extra
                idx += 1
                if nxt == occ:  # rdate duplicates a rule occurrence: once
                    nxt = next(stream, None)
            else:
                occ = nxt
                nxt = next(stream, None)
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

    def between(self, after: datetime, before: datetime,
                *, inclusive: bool = False) -> list[datetime]:
        """Occurrences in the open interval ``(after, before)``.

        With ``inclusive=True`` the endpoints are included.  The scan
        stops at ``before``, so infinite rules stay cheap.
        """
        out: list[datetime] = []
        for occ in self.occurrences():
            if occ > before or (occ == before and not inclusive):
                break
            if occ > after or (occ == after and inclusive):
                out.append(occ)
        return out

    def next_n(self, n: int, after: datetime | None = None) -> list[datetime]:
        """Next ``n`` occurrences strictly after ``after`` (default: now)."""
        if n <= 0:
            return []
        base = (after if after is not None
                else datetime.now(tz=self.dtstart.tzinfo))
        out: list[datetime] = []
        for occ in self.occurrences():
            if occ > base:
                out.append(occ)
                if len(out) >= n:
                    break
        return out

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


def _align_footing(dt: datetime, ref: datetime) -> datetime:
    """Put ``dt`` on ``ref``'s naive/aware footing (never raises)."""
    if (dt.tzinfo is None) == (ref.tzinfo is None):
        return dt
    if ref.tzinfo is None:
        return dt.replace(tzinfo=None)
    return dt.replace(tzinfo=ref.tzinfo)


def parse_rrule_set(text: str, dtstart: datetime | None = None) -> RRule:
    """Parse a multi-line iCalendar-ish recurrence set.

    Understands ``DTSTART:``, ``RRULE:`` (or a bare ``FREQ=…`` line),
    ``EXDATE:`` and ``RDATE:`` lines with comma-separated values, e.g.::

        DTSTART:20260105T090000
        RRULE:FREQ=WEEKLY;BYDAY=MO,WE,FR
        EXDATE:20260112T090000

    Returns an :class:`RRule` with ``exdates``/``rdates`` populated.
    Raises ValueError on garbage.
    """
    dt = dtstart
    rule_text: str | None = None
    exdates: list[datetime] = []
    rdates: list[datetime] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("DTSTART:"):
            dt = _parse_until(line[8:])
        elif upper.startswith("RRULE:"):
            rule_text = line[6:]
        elif upper.startswith("EXDATE:"):
            exdates.extend(_parse_until(v)
                           for v in line[7:].split(",") if v.strip())
        elif upper.startswith("RDATE:"):
            rdates.extend(_parse_until(v)
                          for v in line[6:].split(",") if v.strip())
        elif "=" in line and rule_text is None:
            rule_text = line  # bare "FREQ=…;…" value on its own line
        else:
            raise ValueError(f"bad recurrence-set line: {raw_line!r}")
    if rule_text is None:
        raise ValueError("recurrence set needs an RRULE")
    rule = RRule.parse(rule_text, dt)
    ref = rule.dtstart
    rule.exdates = tuple(_align_footing(d, ref) for d in exdates)
    rule.rdates = tuple(_align_footing(d, ref) for d in rdates)
    return rule


def prev_cron(expr: str, before: datetime, timezone: str = "") -> float:
    """Most recent minute strictly before ``before`` matching the cron.

    Mirror of :func:`next_cron`, scanning backwards (cap 366 days).
    ``timezone`` is an IANA name — the wall-clock scan runs in that zone.
    Returns a unix timestamp.
    """
    spec = CronSpec(expr)  # parsed once; raises on garbage
    tz: Any = None
    if (timezone or "").strip():
        try:
            tz = ZoneInfo(timezone.strip())
        except ZoneInfoNotFoundError:
            tz = None
    probe = before.astimezone(tz) if tz else before
    probe = probe.replace(second=0, microsecond=0)
    if probe == before:
        # exactly on a minute boundary → strictly before
        probe -= timedelta(minutes=1)
    limit = probe - timedelta(days=366)
    while probe >= limit:
        if spec.matches(probe):
            return probe.timestamp()
        probe -= timedelta(minutes=1)
    raise ValueError(f"cron expression never matches within a year: {expr!r}")


# ── human-readable schedule descriptions ───────────────────────────────────
# crontab.guru / cron-descriptor style: never show a raw cron string to a
# user when a sentence will do.  Implemented natively (stdlib only).

_MONTH_NAMES = ["", "January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November",
                "December"]
_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
                  "Saturday", "Sunday"]                      # Monday=0
_CRON_DOW_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday",
                   "Friday", "Saturday"]                     # cron: 0=Sunday


def _t12(hour: int, minute: int) -> str:
    """'9:00 AM' style 12-hour clock."""
    suffix = "AM" if hour < 12 else "PM"
    return f"{hour % 12 or 12}:{minute:02d} {suffix}"


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _join_names(names: list[str]) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return f"{', '.join(names[:-1])} and {names[-1]}"


def _detect_step(vals: set[int], lo: int, hi: int) -> int | None:
    """Step ``n`` when ``vals`` is exactly ``range(lo, hi+1, n)``, else None."""
    s = sorted(vals)
    if len(s) < 2:
        return None
    step = s[1] - s[0]
    if step < 2:
        return None
    return step if s == list(range(lo, hi + 1, step)) else None


def _is_contiguous(vals: list[int]) -> bool:
    s = sorted(vals)
    return len(s) > 1 and s == list(range(s[0], s[-1] + 1))


def _describe_dom(dom: _DomMatcher) -> str:
    bits: list[str] = []
    if dom.days:
        days = sorted(dom.days)
        label = _join_names([_ordinal(d) for d in days])
        bits.append(f"on day {label} of the month" if len(days) == 1
                    else f"on days {label} of the month")
    if dom.last_day:
        bits.append("on the last day of the month")
    if dom.last_weekday:
        bits.append("on the last weekday of the month")
    if dom.nearest:
        days = sorted(dom.nearest)
        label = _join_names([f"day {d}" for d in days])
        bits.append(f"on the weekday nearest {label}")
    return " or ".join(bits)


def _describe_dow(dow: _DowMatcher) -> str:
    bits: list[str] = []
    if dow.days:
        days = sorted(dow.days)
        names = [_CRON_DOW_NAMES[d] for d in days]
        if _is_contiguous(days):
            bits.append(f"on {names[0]} through {names[-1]}")
        else:
            bits.append(f"on {_join_names(names)}")
    for cron_dow in sorted(dow.nth):
        bits.append(f"on the {_ordinal(dow.nth[cron_dow])} "
                    f"{_CRON_DOW_NAMES[cron_dow]} of the month")
    for cron_dow in sorted(dow.last):
        bits.append(f"on the last {_CRON_DOW_NAMES[cron_dow]} of the month")
    return " or ".join(bits)


def _describe_months(months: list[int]) -> str:
    names = [_MONTH_NAMES[m] for m in months]
    if _is_contiguous(months):
        return f"in {names[0]} through {names[-1]}"
    return f"only in {_join_names(names)}"


def describe_cron(expr: str) -> str:
    """Human-readable description of a 5-field cron expression.

    ``"30 11 * * 1-5"`` → ``"At 11:30 AM, on Monday through Friday"``.
    Extended fields are described too (``L`` → "last day of the month",
    ``5#3`` → "3rd Friday of the month").  Raises ValueError on garbage.
    """
    spec = parse_cron(expr)
    minutes = sorted(spec.minute)
    hours = sorted(spec.hour)

    if minutes == list(range(60)) and hours == list(range(24)):
        time_s = "Every minute"
    elif minutes == [0] and hours == list(range(24)):
        time_s = "Every hour"
    else:
        mstep = _detect_step(spec.minute, 0, 59)
        hstep = _detect_step(spec.hour, 0, 23)
        if mstep and hours == list(range(24)):
            time_s = f"Every {mstep} minutes"
        elif hstep and minutes == [0]:
            time_s = f"Every {hstep} hours"
        elif mstep and len(hours) == 1:
            time_s = (f"Every {mstep} minutes between "
                      f"{_t12(hours[0], 0)} and {_t12(hours[0], 59)}")
        elif len(minutes) == 1 and len(hours) == 1:
            time_s = f"At {_t12(hours[0], minutes[0])}"
        elif len(minutes) * len(hours) <= 6:
            times = ", ".join(_t12(h, m) for h in hours for m in minutes)
            time_s = f"At {times}"
        else:
            time_s = (f"At minutes {', '.join(map(str, minutes))} "
                      f"past hours {', '.join(map(str, hours))}")

    day_s = ""
    if spec.dom.restricted and spec.dow.restricted:
        # classic cron OR semantics between the two day fields
        day_s = f", {_describe_dom(spec.dom)} or {_describe_dow(spec.dow)}"
    elif spec.dom.restricted:
        day_s = f", {_describe_dom(spec.dom)}"
    elif spec.dow.restricted:
        day_s = f", {_describe_dow(spec.dow)}"

    month_s = ""
    if spec.month != set(range(1, 13)):
        month_s = f", {_describe_months(sorted(spec.month))}"

    return time_s + day_s + month_s


def _describe_rrule_monthday(rule: "RRule") -> str:
    if rule.bymonthday:
        pos = sorted(d for d in rule.bymonthday if d > 0)
        neg = sorted(d for d in rule.bymonthday if d < 0)
        bits = []
        if pos:
            label = _join_names([_ordinal(d) for d in pos])
            bits.append(f"on day {label}" if len(pos) == 1
                        else f"on days {label}")
        if neg:
            bits.append("on the last day of the month" if neg == [-1]
                        else f"on {_join_names([_ordinal(-d) for d in neg])} "
                             "days before month end")
        return " " + " and ".join(bits)
    if rule.byday:
        ords = [(o, wd) for o, wd in rule.byday if o is not None]
        plains = sorted({wd for o, wd in rule.byday if o is None})
        if ords and not plains:
            bits = []
            for ordinal, wd in ords:
                name = _WEEKDAY_NAMES[wd]
                bits.append(f"on the last {name}" if ordinal == -1
                            else f"on the {_ordinal(ordinal)} {name}")
            return " " + _join_names(bits)
        if plains and rule.bysetpos:
            names = [_WEEKDAY_NAMES[w] for w in plains]
            pos = "last" if -1 in rule.bysetpos else _ordinal(rule.bysetpos[0])
            return f" on the {pos} {_join_names(names)}"
        if plains:
            return f" on {_join_names([_WEEKDAY_NAMES[w] for w in plains])}"
    return f" on day {rule.dtstart.day}"


def describe_rrule(rule_text: str) -> str:
    """Human-readable description of an RRULE.

    ``"FREQ=MONTHLY;BYDAY=2TU"`` → ``"Every month on the 2nd Tuesday"``.
    Raises ValueError on garbage.
    """
    rule = parse_rrule(rule_text)
    n = rule.interval

    def every(unit: str) -> str:
        return f"Every {unit}" if n == 1 else f"Every {n} {unit}s"

    tod = _t12(rule.dtstart.hour, rule.dtstart.minute)
    if rule.freq == "MINUTELY":
        s = every("minute")
    elif rule.freq == "HOURLY":
        s = every("hour")
    elif rule.freq == "DAILY":
        s = every("day") + f" at {tod}"
    elif rule.freq == "WEEKLY":
        s = every("week")
        if rule.byday:
            days = sorted({wd for _, wd in rule.byday},
                          key=lambda w: (w - rule.wkst) % 7)
        else:
            days = [rule.dtstart.weekday()]
        s += f" on {_join_names([_WEEKDAY_NAMES[w] for w in days])} at {tod}"
    elif rule.freq == "MONTHLY":
        s = every("month") + _describe_rrule_monthday(rule) + f" at {tod}"
    else:  # YEARLY
        s = every("year")
        if rule.bymonth:
            s += (" in " + _join_names(
                [_MONTH_NAMES[m] for m in sorted(set(rule.bymonth))]))
        s += _describe_rrule_monthday(rule) + f" at {tod}"
    if rule.count:
        s += f", {rule.count} time{'s' if rule.count != 1 else ''}"
    if rule.until:
        s += f", until {rule.until.strftime('%B %d, %Y')}"
    return s


# ── natural-language schedule / datetime parsing ────────────────────────────
# stdlib-only take on the parsedatetime/dateparser pattern set: the phrases
# users actually type ("in 20 minutes", "every weekday at 9am").  Deliberately
# a curated pattern set rather than full NLP — every accepted phrase is
# tested, every miss raises a helpful ValueError.

_WEEKDAY_WORDS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}
_MONTH_WORDS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12,
    "dec": 12,
}
_ORDINAL_WORDS = {
    "1st": 1, "first": 1, "2nd": 2, "second": 2, "3rd": 3, "third": 3,
    "4th": 4, "fourth": 4, "5th": 5, "fifth": 5, "last": -1,
}
_TIME_RE = re.compile(
    r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$", re.IGNORECASE)


def _parse_time_token(token: str) -> tuple[int, int]:
    """'9am' / '9:30pm' / '21:15' / '9' → (hour, minute).  Raises ValueError."""
    m = _TIME_RE.match(token or "")
    if not m:
        raise ValueError(f"not a time: {token!r}")
    hour, minute_s, meridiem = int(m.group(1)), m.group(2), m.group(3)
    minute = int(minute_s) if minute_s else 0
    if not (0 <= minute <= 59):
        raise ValueError(f"bad minutes in {token!r}")
    if meridiem:
        if not 1 <= hour <= 12:
            raise ValueError(f"bad hour in {token!r}")
        pm = meridiem[0].lower() == "p"
        hour = (hour % 12) + (12 if pm else 0)
    elif not 0 <= hour <= 23:
        raise ValueError(f"bad hour in {token!r}")
    return hour, minute


def _split_at_time(text: str) -> tuple[str, str | None]:
    """Split 'tomorrow at 8am' → ('tomorrow', '8am'); 'at 9' → ('', '9')."""
    m = re.match(r"^(?:(.*?)\s+)?at\s+(.+)$", text.strip(), re.IGNORECASE)
    if m:
        return (m.group(1) or "").strip(), m.group(2).strip()
    return text.strip(), None


def parse_natural_datetime(text: str, *,
                           now: datetime | None = None,
                           tz: str | None = None) -> datetime:
    """Parse a natural-language datetime → :class:`datetime`.

    Understands (case-insensitive): ``now``; ``in 20 minutes`` / ``in an
    hour`` / ``in 3 days`` / ``in 2 weeks``; ``tomorrow`` / ``today`` /
    ``tonight`` (each optionally ``at 8am``); ``next monday`` / ``this fri``
    / bare ``wednesday`` (optionally ``at …``); ``noon`` / ``midnight``;
    ``at 9:30pm`` / bare ``9:30pm`` (today if future else tomorrow);
    ``oct 15`` / ``15 oct`` / ``2026-10-15`` / ``2026-10-15 14:30``.

    ``now`` anchors relative phrases (default: current time).  ``tz`` is an
    optional IANA name — the result is aware in that zone; without it the
    result is naive local time.  Raises ValueError when nothing matches.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty datetime text")
    zone = None
    if (tz or "").strip():
        try:
            zone = ZoneInfo(tz.strip())
        except ZoneInfoNotFoundError:
            raise ValueError(f"unknown timezone: {tz!r}")
    base = now or datetime.now(tz=zone)

    def aware(dt: datetime) -> datetime:
        if zone is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=zone)
        return dt

    lowered = raw.lower()

    if lowered == "now":
        return base

    # ISO first ("2026-10-15", "2026-10-15 14:30")
    try:
        return aware(datetime.fromisoformat(raw))
    except ValueError:
        pass

    # "in N <unit>" / "in a minute"
    m = re.fullmatch(
        r"in\s+(a|an|\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|"
        r"days?|weeks?)\b\s*(.*)", lowered)
    if m:
        qty_s, unit, rest = m.group(1), m.group(2), m.group(3).strip()
        qty = 1 if qty_s in ("a", "an") else int(qty_s)
        if rest:
            raise ValueError(
                f"don't know how to handle {raw!r} — try 'in {qty_s} {unit}'")
        kwargs: dict[str, float] = {}
        if unit.startswith("sec"):
            kwargs["seconds"] = qty
        elif unit.startswith("min"):
            kwargs["minutes"] = qty
        elif unit.startswith("hour") or unit.startswith("hr"):
            kwargs["hours"] = qty
        elif unit.startswith("day"):
            kwargs["days"] = qty
        else:
            kwargs["weeks"] = qty
        return aware(base + timedelta(**kwargs))

    date_part, time_part = _split_at_time(raw)
    d = date_part.lower()

    def at_time(default_h: int, default_m: int = 0) -> tuple[int, int]:
        if time_part:
            return _parse_time_token(time_part)
        return default_h, default_m

    # relative day words
    if d in ("today",):
        h, mi = at_time(base.hour, base.minute)
        if time_part:
            dt = base.replace(hour=h, minute=mi, second=0, microsecond=0)
            return aware(dt)
        return base
    if d == "tonight":
        h, mi = at_time(20)
        return aware(base.replace(hour=h, minute=mi, second=0, microsecond=0))
    if d == "tomorrow":
        h, mi = at_time(9)
        day = (base + timedelta(days=1)).replace(
            hour=h, minute=mi, second=0, microsecond=0)
        return aware(day)
    if d in ("noon", "midday"):
        dt = base.replace(hour=12, minute=0, second=0, microsecond=0)
        if dt <= base:
            dt += timedelta(days=1)
        return aware(dt)
    if d == "midnight":
        dt = (base + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        return aware(dt)

    # weekday words: "next monday", "this fri", "wednesday"
    m = re.fullmatch(r"(next|this)?\s*([a-z]+)", d)
    if m and m.group(2) in _WEEKDAY_WORDS:
        which, target = m.group(1), _WEEKDAY_WORDS[m.group(2)]
        h, mi = at_time(9)
        if which == "next":
            delta = (target - base.weekday()) % 7 + 7
        elif which == "this":
            delta = (target - base.weekday()) % 7
        else:
            delta = (target - base.weekday()) % 7
            if delta == 0 and time_part:
                probe = base.replace(hour=h, minute=mi,
                                     second=0, microsecond=0)
                if probe <= base:
                    delta = 7
        dt = (base + timedelta(days=delta)).replace(
            hour=h, minute=mi, second=0, microsecond=0)
        return aware(dt)

    # month-day words: "oct 15", "15 october", "oct 15 2027"
    m = re.fullmatch(
        r"([a-z]+)\s+(\d{1,2})(?:\s*,?\s*(\d{4}))?", d)
    m2 = re.fullmatch(
        r"(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)(?:\s*,?\s*(\d{4}))?", d)
    mm = m or m2
    if mm:
        if m:
            mon_w, day_s, year_s = mm.group(1), mm.group(2), mm.group(3)
        else:
            day_s, mon_w, year_s = mm.group(1), mm.group(2), mm.group(3)
        if mon_w in _MONTH_WORDS:
            month, day = _MONTH_WORDS[mon_w], int(day_s)
            year = int(year_s) if year_s else base.year
            h, mi = at_time(9)
            try:
                dt = base.replace(year=year, month=month, day=day,
                                  hour=h, minute=mi, second=0, microsecond=0)
            except ValueError:
                raise ValueError(f"not a real date: {raw!r}")
            if year_s is None and dt <= base:
                try:
                    dt = dt.replace(year=year + 1)
                except ValueError:
                    raise ValueError(f"not a real date: {raw!r}")
            return aware(dt)

    # bare time: "9:30pm" / "at 9" → today if future else tomorrow
    time_token: str | None = None
    if time_part:
        time_token = time_part
    elif not d or d == "at":
        time_token = ""
    else:
        try:
            _parse_time_token(d)
            time_token = d
        except ValueError:
            time_token = None
    if time_token is not None:
        try:
            h, mi = _parse_time_token(time_token)
        except ValueError:
            raise ValueError(
                f"don't understand {raw!r} — try 'in 20 minutes', "
                "'tomorrow at 8am', 'next monday at 9', 'oct 15'")
        dt = base.replace(hour=h, minute=mi, second=0, microsecond=0)
        if dt <= base:
            dt += timedelta(days=1)
        return aware(dt)

    raise ValueError(
        f"don't understand {raw!r} — try 'in 20 minutes', 'tomorrow at 8am', "
        "'next monday at 9', 'oct 15'")


def parse_natural_schedule(text: str) -> tuple[str, str]:
    """Parse a natural-language schedule → ``(kind, expression)``.

    ``kind`` is ``"cron"`` (5-field expression) or ``"rrule"`` (RFC 5545).
    Understands: ``every minute``; ``every 5 minutes``; ``hourly`` / ``every
    2 hours``; ``daily at 9am`` / ``every day at 18:30``; ``every weekday at
    9am``; ``weekends at 10am``; ``weekly on monday at 8am`` / ``every friday
    at 5pm``; ``monthly on the 15th`` / ``monthly``; ``every 3 days at 9am``;
    ``every 2nd tuesday``; ``last friday of the month``; ``yearly on jan 1``.

    A missing time defaults to 9:00 AM.  Raises ValueError when the text
    doesn't match a known pattern.
    """
    raw = (text or "").strip()
    if not raw:
        raise ValueError("empty schedule text")
    lowered = re.sub(r"\s+", " ", raw.lower())

    # optional trailing "at <time>" (default 9:00)
    time_part: str | None = None
    m = re.match(r"^(.*?)\s+at\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)$",
                 lowered)
    if m:
        lowered, time_part = m.group(1).strip(), m.group(2).strip()
    hour, minute = _parse_time_token(time_part) if time_part else (9, 0)

    if lowered in ("every minute", "each minute"):
        return "cron", "* * * * *"

    m = re.fullmatch(r"every (\d+) minutes?", lowered)
    if m:
        return "cron", f"*/{int(m.group(1))} * * * *"

    if lowered in ("hourly", "every hour"):
        return "cron", "0 * * * *"
    m = re.fullmatch(r"every (\d+) hours?", lowered)
    if m:
        return "cron", f"{minute} */{int(m.group(1))} * * *"

    if lowered in ("daily", "every day", "each day"):
        return "cron", f"{minute} {hour} * * *"

    if lowered in ("weekdays", "every weekday", "weekdays only",
                   "monday to friday", "mon-fri"):
        return "cron", f"{minute} {hour} * * MON-FRI"
    if lowered in ("weekends", "every weekend", "weekend"):
        return "cron", f"{minute} {hour} * * SAT,SUN"

    m = re.fullmatch(r"(?:weekly|every week)(?: on ([a-z]+))?", lowered)
    if m:
        wd = m.group(1)
        if wd:
            if wd not in _WEEKDAY_WORDS:
                raise ValueError(f"unknown weekday: {wd!r}")
            cron_dow = (_WEEKDAY_WORDS[wd] + 1) % 7  # cron: Sunday=0
            return "cron", f"{minute} {hour} * * {cron_dow}"
        return "cron", f"{minute} {hour} * * 1"  # bare "weekly" → Monday
    m = re.fullmatch(r"every ([a-z]+)", lowered)
    if m and m.group(1) in _WEEKDAY_WORDS:
        cron_dow = (_WEEKDAY_WORDS[m.group(1)] + 1) % 7
        return "cron", f"{minute} {hour} * * {cron_dow}"

    m = re.fullmatch(r"monthly(?: on the (\d{1,2})(?:st|nd|rd|th)?)?", lowered)
    if m:
        day = int(m.group(1)) if m.group(1) else 1
        if not 1 <= day <= 31:
            raise ValueError(f"bad day of month: {m.group(1)!r}")
        return "cron", f"{minute} {hour} {day} * *"

    m = re.fullmatch(r"every (\d+) days?", lowered)
    if m:
        return "rrule", f"FREQ=DAILY;INTERVAL={int(m.group(1))}"

    m = re.fullmatch(r"every (1st|2nd|3rd|4th|5th|first|second|third|fourth|"
                     r"fifth|last) ([a-z]+)", lowered)
    if m and m.group(2) in _WEEKDAY_WORDS:
        ordinal, wd = _ORDINAL_WORDS[m.group(1)], m.group(2).upper()[:2]
        if ordinal == -1:
            return "rrule", f"FREQ=MONTHLY;BYDAY={wd};BYSETPOS=-1"
        return "rrule", f"FREQ=MONTHLY;BYDAY={ordinal}{wd}"

    m = re.fullmatch(r"(1st|2nd|3rd|4th|5th|first|second|third|fourth|fifth|"
                     r"last) ([a-z]+) of the month", lowered)
    if m and m.group(2) in _WEEKDAY_WORDS:
        ordinal, wd = _ORDINAL_WORDS[m.group(1)], m.group(2).upper()[:2]
        if ordinal == -1:
            return "rrule", f"FREQ=MONTHLY;BYDAY={wd};BYSETPOS=-1"
        return "rrule", f"FREQ=MONTHLY;BYDAY={ordinal}{wd}"

    m = re.fullmatch(r"yearly(?: on ([a-z]+) (\d{1,2})(?:st|nd|rd|th)?)?",
                     lowered)
    if m:
        if m.group(1):
            if m.group(1) not in _MONTH_WORDS:
                raise ValueError(f"unknown month: {m.group(1)!r}")
            month, day = _MONTH_WORDS[m.group(1)], int(m.group(2))
            return "cron", f"{minute} {hour} {day} {month} *"
        return "cron", f"{minute} {hour} 1 1 *"

    raise ValueError(
        f"don't understand schedule {raw!r} — try 'every weekday at 9am', "
        "'daily at 6pm', 'every 2nd tuesday', 'monthly on the 15th'")


def parse_rrule(text: str, dtstart: datetime | None = None) -> RRule:
    """Parse and validate an RRULE string → :class:`RRule`."""
    return RRule.parse(text, dtstart)