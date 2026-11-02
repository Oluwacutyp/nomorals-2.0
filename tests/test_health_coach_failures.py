"""Tail-88 pins: nomorals/health/coach.py census sites.

The coach's posture is printed in its own docstrings: the
pure helpers "never raise", and the module's positioning law
is that numbers are never fabricated — a coach answer degrades
to its documented base value or an honest failure string,
never to invented data. Every site is held as an exclusion:

293 — ``progress_bar`` crash -> "". Pure, never raises; the
sole consumer (Readiness.format) prints the numeric score
right beside the bar, so an empty bar is a cosmetic absence,
never a falsified reading.
312 — ``sleep_need`` crash -> 8.0. 8.0 is the function's own
documented base need before adjustments — the unadjusted
answer, not an invented one. Pure, never raises.
329 — ``sleep_score_vs_need`` crash -> 0.0. The failure
direction is pessimistic: zero sleep credit can only lower a
readiness score, never inflate it — the safe direction on a
surface that must not fabricate. The sole caller passes
sanitized floats (a non-None _sleep_hours result and
sleep_need's own output), so the guard is defense under the
documented never-raises pure-helper contract. Tension stated
plainly: 0.0 is byte-identical to a genuine zero-hour score
in the contributors map.
398 — ``Readiness.explain`` crash -> "couldn't break this
down right now." An honest failure string, distinguishable
from every success output; the score itself is untouched.
828 — ``readiness_trend`` crash -> "couldn't build the
readiness trend." Same shape: an honest failure string, and
the history read it wraps has its own bind+debug guards.
"""
from __future__ import annotations

from types import SimpleNamespace

from nomorals.health import coach as C


def _raise_boom(*args, **kwargs):
    raise RuntimeError("boom")


# ── 293: progress bar degrades to empty ─────────────────────────

def test_293_progress_bar_crash_returns_empty():
    assert C.progress_bar(object()) == ""
    # control: a mid score fills half the bar
    assert C.progress_bar(50, width=10) == "█████░░░░░"


# ── 312: sleep need degrades to the base need ───────────────────

def test_312_sleep_need_crash_returns_base():
    assert C.sleep_need(workouts_48h="x") == 8.0
    # controls: the base need, and the short-average repayment
    assert C.sleep_need() == 8.0
    assert C.sleep_need(recent_avg_hours=6.0) == 9.0


# ── 329: sleep score degrades to zero credit ────────────────────

def test_329_sleep_score_crash_returns_zero():
    assert C.sleep_score_vs_need("x", 8.0) == 0.0
    # controls: full marks at need, half marks at half of need
    assert C.sleep_score_vs_need(8.0, 8.0) == 100.0
    assert C.sleep_score_vs_need(4.0, 8.0) == 50.0


# ── 398: explain degrades to an honest failure string ───────────

def test_398_explain_crash_returns_failure_string():
    r = C.Readiness(level="high", score=80.0, has_data=True,
                    contributors={"hrv": "abc"})
    assert r.explain() == "couldn't break this down right now."
    # control: healthy contributors produce the breakdown
    r2 = C.Readiness(level="high", score=80.0, has_data=True,
                     contributors={"hrv": 90.0})
    assert "what moved your score" in r2.explain()


# ── 828: trend degrades to an honest failure string ─────────────

def test_828_trend_crash_returns_failure_string(monkeypatch):
    coach = C.HealthCoach(source=SimpleNamespace(has_data=lambda: True))
    monkeypatch.setattr(coach, "readiness_history", _raise_boom)
    assert coach.readiness_trend() == "couldn't build the readiness trend."
    # control: a rising history renders the trend card
    pts = [C.ReadinessPoint(ts=float(i), level="high",
                            score=50.0 + 10 * i) for i in range(4)]
    monkeypatch.setattr(coach, "readiness_history",
                        lambda days=30: pts)
    text = coach.readiness_trend()
    assert "readiness trend" in text and "↑" in text
