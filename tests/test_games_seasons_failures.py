"""Tail-114 pins: nomorals/games/seasons.py census sites (143, 158, 174).

All three guard the same call — ``active_event(db, now)`` — and are
held as exclusions. The reachable trigger is a garbage ``now``:
``_override`` has its own bind+log guard, so ``active_event`` only
raises downstream (``datetime.fromtimestamp`` on a non-timestamp).

143 — ``event_blurb`` -> None, byte-identical to the function's
     legitimate "season doesn't touch this game" answer.
158 — ``apply_event`` -> (amount, ""): the payout passes through
     unmultiplied, identical to the no-season / multiplier-1 answer.
     The failure direction never invents a multiplier.
174 — ``describe_seasons`` -> "the seasons are unreadable right now."
     The failure is NAMED in-band on the /game events surface.
"""

from __future__ import annotations

import logging

from nomorals.games import seasons

GARBAGE_NOW = "not-a-timestamp"


def _blood_moon_ts() -> float:
    base = 1_700_000_000.0
    for d in range(400):
        cand = base + d * 86400
        if seasons.active_event(None, cand)["id"] == "blood_moon":
            return cand
    raise AssertionError("no Blood Moon week found in scan window")


def test_143_blurb_crash_returns_none_like_untouched(caplog):
    with caplog.at_level(logging.DEBUG, logger="nomorals.games.seasons"):
        assert seasons.event_blurb(None, "arena", now=GARBAGE_NOW) is None
    # the guard itself records nothing
    assert caplog.records == []


def test_143_control_blurb_renders_for_touched_game():
    ts = _blood_moon_ts()
    blurb = seasons.event_blurb(None, "arena", now=ts)
    assert blurb is not None and "Blood Moon" in blurb


def test_143_control_untouched_game_also_returns_none():
    ts = _blood_moon_ts()
    assert seasons.event_blurb(None, "slots", now=ts) is None


def test_158_apply_crash_passes_amount_through():
    assert seasons.apply_event(None, "arena", "coins", 100,
                               now=GARBAGE_NOW) == (100, "")


def test_158_control_multiplier_applies():
    ts = _blood_moon_ts()
    amount, note = seasons.apply_event(None, "arena", "coins", 100, now=ts)
    assert amount == 125
    assert "x1.25" in note


def test_174_describe_crash_names_the_failure():
    assert seasons.describe_seasons(None, now=GARBAGE_NOW) == \
        "the seasons are unreadable right now."


def test_174_control_describe_renders_calendar():
    ts = _blood_moon_ts()
    text = seasons.describe_seasons(None, now=ts)
    assert text.startswith("🗓️")
    assert "Blood Moon" in text
