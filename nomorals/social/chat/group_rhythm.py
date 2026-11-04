"""Group rhythm: the room's own clock (DESIGN §30/§31/§26, P3).

Every group lives on a rhythm: quiet hours, a morning shape, a
memory of its own firsts.  This module is the pure half of that
layer — the settings block, the group-local clock math and the
§26 contribution ensemble — while :class:`GroupSurface` owns the
state (the store's per-group ``extras``) and the cycle hooks.

The pieces:

- **Settings** (``RHYTHM_DEFAULTS``): silent hours with a
  per-group UTC offset (groups carry no timezone of their own —
  the offset IS the honest stand-in, set by the group's admins),
  the nightshield, the group briefing, reminders, the poll
  defaults and the §26 contribution-review mode.
- **The clock**: group-local day/minute math over plain epoch
  seconds.  Windows are half-open ``[start, end)`` and wrap past
  midnight (23:00–07:00 is the normal case, not an edge case).
- **The §26 ensemble**: a deterministic first read on a candidate
  contribution — length, structure, sources, duplication against
  the author's own recent posts, copy-paste against the room's
  recent traffic.  It returns a score and plain-word reasons; the
  review MODE (off / auto / assisted / manual) decides what the
  score means, in the surface.  No model, no guessing: the same
  post scores the same way every time.
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable

__all__ = [
    "RHYTHM_DEFAULTS", "REVIEW_MODES",
    "parse_hhmm", "parse_offset", "format_offset",
    "local_parts", "in_window", "day_window",
    "CONTRIBUTION_MIN_CHARS", "CLEAR_SCORE", "BORDERLINE_SCORE",
    "is_contribution_candidate", "normalize_text",
    "score_contribution",
    "GROUP_MESSAGE_MILESTONES", "MEMBER_MESSAGE_MILESTONES",
    "POLL_DEFAULT_DURATION_S", "HELD_RING", "CONTRIBUTION_RING",
]


# ── settings ────────────────────────────────────────────────────

#: The per-group rhythm block (DESIGN §30: the console's rhythm
#: section joins when rhythm lands — this build).  Everything is
#: inert until a group turns it on, except the briefing/reminder
#: machinery which carries its own on/off.
RHYTHM_DEFAULTS: dict[str, Any] = {
    # Group-local time: minutes east of UTC.  Groups carry no
    # timezone today; the offset is set by the group's admins and
    # defaults to UTC, stated plainly wherever it shows.
    "tz_offset_min": 0,
    # Silent hours: non-urgent SYSTEM output is held and released
    # as one morning digest.  Moderation is never held.
    "silent_enabled": False,
    "silent_start": "23:00",
    "silent_end": "07:00",
    # Nightshield: during the same window, acceptance tightens to
    # the §28 raid posture and drop/war cycle events pause.
    "nightshield": False,
    # The group briefing (NOT the owner's morning pulse — a
    # different organ): yesterday's shape, posted in-group at the
    # group's local briefing hour.
    "briefing_enabled": False,
    "briefing_hour": 8,
    # Reminders (§31 cards on the rhythm tick): daily or
    # weekday-bound one-line cards, plus a rules card kind.
    "reminders_enabled": True,
    "reminders": [],  # {id, text, time "HH:MM", days [0..6], kind}
    # Executing polls: a treasury poll needs at least this many
    # votes cast before its result can move money.
    "poll_quorum": 3,
    "poll_duration_s": 86400,
    # §26 contribution review: off | auto | assisted | manual.
    "review": {"mode": "off"},
}

REVIEW_MODES = ("off", "auto", "assisted", "manual")


# ── the group-local clock ───────────────────────────────────────

_HHMM = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_hhmm(value: Any) -> int | None:
    """``"23:30"`` → minutes after local midnight; None if bad."""
    m = _HHMM.match(str(value or "").strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute


def parse_offset(text: Any) -> int | None:
    """Parse a UTC offset: ``+1``, ``-5``, ``+01:30``, ``330``.

    A bare signed number is HOURS (``+1`` = +60); ``H:MM`` is
    exact; a bare unsigned number of 15+ is read as minutes (the
    way offsets are stored).  Returns minutes or None."""
    raw = str(text or "").strip().upper().replace("UTC", "")
    if not raw:
        return None
    sign = 1
    if raw.startswith("+"):
        raw = raw[1:]
    elif raw.startswith("-"):
        sign = -1
        raw = raw[1:]
    if ":" in raw:
        hour, _, minute = raw.partition(":")
        try:
            return sign * (int(hour) * 60 + int(minute))
        except ValueError:
            return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if abs(value) >= 15 and ("+" not in str(text) and "-" not in
                             str(text)):
        return value  # already minutes
    return sign * value * 60 if abs(value) <= 14 else sign * value


def format_offset(minutes: int) -> str:
    """``60`` → ``"UTC+01:00"``; ``0`` → ``"UTC"``."""
    minutes = int(minutes or 0)
    if minutes == 0:
        return "UTC"
    sign = "+" if minutes > 0 else "-"
    minutes = abs(minutes)
    return f"UTC{sign}{minutes // 60:02d}:{minutes % 60:02d}"


def local_parts(ts: float, offset_min: int) -> dict[str, Any]:
    """The group-local reading of an epoch second: day key,
    minute of day, hour, weekday (Mon=0) and the date tuple."""
    tm = time.gmtime(float(ts) + int(offset_min or 0) * 60)
    return {
        "day": time.strftime("%Y-%m-%d", tm),
        "minute": tm.tm_hour * 60 + tm.tm_min,
        "hour": tm.tm_hour,
        "weekday": tm.tm_wday,
        "ymd": (tm.tm_year, tm.tm_mon, tm.tm_mday),
    }


def in_window(minute: int, start: int | None,
              end: int | None) -> bool:
    """Half-open window membership, wrapping past midnight.

    ``start == end`` reads as an empty window (never active) —
    a zero-length quiet is no quiet at all."""
    if start is None or end is None or start == end:
        return False
    if start < end:
        return start <= minute < end
    return minute >= start or minute < end


def day_window(day: str, offset_min: int) -> tuple[float, float]:
    """UTC epoch bounds ``[start, end)`` of one group-local day."""
    year, month, dom = (int(p) for p in str(day).split("-"))
    import calendar
    start = calendar.timegm((year, month, dom, 0, 0, 0, 0, 0, 0))
    start -= int(offset_min or 0) * 60
    return float(start), float(start + 86400)


# ── §26: the contribution ensemble ──────────────────────────────

#: A message becomes a contribution CANDIDATE at this length (or
#: by carrying media/a link).  Below it, the message still counts
#: on the raw activity counter — it just isn't a "contribution"
#: the reviewed ranker weighs (DESIGN §26).
CONTRIBUTION_MIN_CHARS = 40

#: Ensemble verdict bands: at/above CLEAR the post counts on the
#: ensemble's own word; at BORDERLINE the call is close enough
#: that assisted mode asks a human; below, the ensemble says no.
CLEAR_SCORE = 3
BORDERLINE_SCORE = 2

_WS = re.compile(r"\s+")
_SENTENCE = re.compile(r"[.!?](\s|$)")


def normalize_text(text: Any) -> str:
    return _WS.sub(" ", str(text or "").strip().lower())


def is_contribution_candidate(text: Any, *, has_media: bool = False,
                              has_link: bool = False) -> bool:
    if has_media or has_link:
        return True
    return len(str(text or "").strip()) >= CONTRIBUTION_MIN_CHARS


def score_contribution(text: Any, *, own_texts: Iterable[str] = (),
                       recent_texts: Iterable[str] = (),
                       has_media: bool = False,
                       has_link: bool = False,
                       ) -> tuple[int, list[str]]:
    """The deterministic §26 first read: ``(score, reasons)``.

    Reasons are plain words a member can be shown — the design's
    transparency rule (counted / not counted / why) starts here.
    """
    body = str(text or "")
    stripped = body.strip()
    score = 0
    reasons: list[str] = []
    n = len(stripped)
    if n >= 400:
        score += 3
        reasons.append("a long, detailed post")
    elif n >= 120:
        score += 2
        reasons.append("a substantial post")
    elif n >= CONTRIBUTION_MIN_CHARS:
        score += 1
        reasons.append("a short post")
    if len(_SENTENCE.findall(stripped)) >= 2 or "\n" in stripped:
        score += 1
        reasons.append("more than one point, structured")
    if has_link:
        score += 1
        reasons.append("brings a source or link")
    if has_media:
        score += 1
        reasons.append("brings media")
    norm = normalize_text(stripped)
    if norm:
        for prev in own_texts:
            other = normalize_text(prev)
            if not other:
                continue
            if other == norm:
                score -= 3
                reasons.append("repeats your own earlier post")
                break
            if len(norm) >= 80 and other[:80] == norm[:80]:
                score -= 2
                reasons.append("near-copy of your own earlier post")
                break
        else:
            for prev in recent_texts:
                if normalize_text(prev) == norm:
                    score -= 2
                    reasons.append(
                        "copy-paste of a recent message in the room")
                    break
    return max(-5, min(6, score)), reasons


# ── milestones + rings ──────────────────────────────────────────

#: Message-count milestones (group totals and per-member).
GROUP_MESSAGE_MILESTONES = (100, 500, 1000, 5000, 10000, 25000,
                            50000, 100000)
MEMBER_MESSAGE_MILESTONES = (100, 500, 1000, 5000, 10000)

POLL_DEFAULT_DURATION_S = 86400

#: Held system output waiting out the silent hours (oldest drops
#: first past the cap — the digest is a morning read, not an
#: archive).
HELD_RING = 60

#: Contribution events kept per group for the rankers, the
#: briefing's yesterday window and the review queue.
CONTRIBUTION_RING = 400
