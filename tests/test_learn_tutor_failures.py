"""Tail-90 pins: nomorals/learn/tutor.py census sites.

Module posture: syllabus grounding (#48) is explicitly
"additive", the session report is a reflection surface whose
lines are individually optional, and the mistake notebook's
external scheduler is documented as an optional back-compat
feed — "the native FSRS-lite state on each card is the source
of truth... a broken scheduler must not break tutoring"
(flashcards.MistakeNotebook). Every site degrades to omission
or to the documented fallback; none fabricates a claim:

675 — ``start()`` syllabus ``find_topic`` crash -> pass. The
session starts ungrounded: the syllabus stays the raw string,
and because the code-line block requires a resolved topic,
NO syllabus claim is rendered. find_topic itself is
documented "never raises", so the guard fires only on the
import or a pathological syllabus object — additive
grounding lost for one session, nothing falsified.
682 — ``start()`` ``syllabus_code`` crash -> pass. Pure
display decoration: the "📖 WAEC …" line is omitted; the
resolved topic, the teaching and the question are untouched.
862 — ``report()`` ``mastery.weakest()`` ValueError -> pass.
weakest() is ``min()`` over the skills dict and raises only
when it is empty; the "weakest skill" line is then omitted
from an otherwise complete, truthful report.
870 — ``report()`` ``notebook.stats()`` crash -> pass. The
notebook summary line is omitted; the report's own numbers
(answers, mastery, time) are computed independently and stay.
1019 — ``get_notebook`` scheduler-equipped load crash ->
load WITHOUT a RepetitionScheduler. The fallback notebook is
fully functional: record/due/stats run on the native
FSRS-lite card state, which the class documents as the
source of truth; the scheduler is a parallel back-compat
feed ("fed in parallel"), so its absence loses only that
feed. Consumers (learn/chat.py, partner/runtime_memory.py)
use the notebook through the same API either way.
"""
from __future__ import annotations

import nomorals.memory.repetition as repetition_mod
from nomorals.agency import failures as agency_failures
from nomorals.learn import curriculum as C
from nomorals.learn import tutor as T


def _not_recorded(before: int) -> None:
    assert agency_failures.failure_count() == before


def _boom(*a, **k):
    raise RuntimeError("probe boom")


# ── 675: syllabus resolution crash -> ungrounded start ──────────

def test_675_find_topic_crash_starts_ungrounded(monkeypatch):
    before = agency_failures.failure_count()
    monkeypatch.setattr(C, "find_topic", _boom)
    s = T.TutorSession("Waves", syllabus="waves")
    turn = s.start()
    assert turn.prompt
    # no grounding happened and none is claimed
    assert s.syllabus == "waves"
    assert "📖" not in turn.feedback and "📖" not in turn.prompt
    # control: resolution lands and the code line is rendered
    monkeypatch.undo()
    ctl = T.TutorSession("Waves", syllabus="waves")
    ctl_turn = ctl.start()
    assert not isinstance(ctl.syllabus, str)
    assert "📖" in ctl_turn.feedback
    _not_recorded(before)


# ── 682: syllabus code crash -> decoration omitted ──────────────

def test_682_syllabus_code_crash_omits_code_line(monkeypatch):
    before = agency_failures.failure_count()
    monkeypatch.setattr(C, "syllabus_code", _boom)
    s = T.TutorSession("Waves", syllabus="waves")
    turn = s.start()
    assert not isinstance(s.syllabus, str)  # resolution landed
    assert "📖" not in turn.feedback and turn.prompt
    _not_recorded(before)


# ── 862: weakest() on empty mastery -> line omitted ─────────────

def test_862_empty_mastery_omits_weakest_line():
    before = agency_failures.failure_count()
    s = T.TutorSession("Fractions")
    s.start()
    s.mastery.skills.clear()
    s._mastery_start = {}
    rep = s.report()
    assert "Session report" in rep and "weakest skill" not in rep
    # control: a normal session names its weakest skill
    ctl = T.TutorSession("Fractions")
    ctl.start()
    assert "weakest skill" in ctl.report()
    _not_recorded(before)


# ── 870: notebook.stats() crash -> line omitted ─────────────────

class _BoomNotebook:
    def stats(self):
        raise RuntimeError("stats boom")


class _GoodNotebook:
    def stats(self):
        return {"total": 3, "due": 1}


def test_870_stats_crash_omits_notebook_line():
    before = agency_failures.failure_count()
    s = T.TutorSession("Fractions", notebook=_BoomNotebook())
    s.start()
    rep = s.report()
    assert "Session report" in rep and "mistake notebook" not in rep
    # control: healthy stats render the summary line
    ctl = T.TutorSession("Fractions", notebook=_GoodNotebook())
    ctl.start()
    assert "mistake notebook: 3 cards, 1 due" in ctl.report()
    _not_recorded(before)


# ── 1019: scheduler crash -> scheduler-less fallback notebook ───

class _BoomScheduler:
    def __init__(self, *a, **k):
        raise RuntimeError("scheduler boom")


def test_1019_scheduler_crash_falls_back_to_native_notebook(monkeypatch):
    before = agency_failures.failure_count()
    key, ctl_key = "tail90-probe", "tail90-probe-ctl"
    T._NOTEBOOKS.pop(key, None)
    T._NOTEBOOKS.pop(ctl_key, None)
    try:
        monkeypatch.setattr(repetition_mod, "RepetitionScheduler",
                            _BoomScheduler)
        nb = T.get_notebook(key)
        assert nb.scheduler is None
        # the fallback notebook works on its native FSRS state
        assert "total" in nb.stats()
        assert nb.due_cards(limit=1) == []
        # control: scheduler constructs -> the notebook carries it
        monkeypatch.undo()
        ctl = T.get_notebook(ctl_key)
        assert ctl.scheduler is not None
    finally:
        T._NOTEBOOKS.pop(key, None)
        T._NOTEBOOKS.pop(ctl_key, None)
    _not_recorded(before)
