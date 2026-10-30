"""Tail-59 pins for nomorals/health/previsit.py —
1 convert (672) / 6 exclusions (354, 382, 400, 405, 447, 561).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers, so it adds zero
census sites.

previsit.py calls no failures-ledger record function anywhere
(grep-proven; its sibling handlers log via _log instead), so
NOT-recording is pinned against the health package ledger
(nomorals/health/failures.py): recent_failures() snapshot
before/after every trigger. The convert's surfacing mechanism
is the module's own _log (stdlib logging), asserted via caplog.

Verdict notes (full grounds in the tail-59 SURVEY.md):
- 354: clarify_questions body failure -> [_CLARIFY_GENERAL];
  the general red-flag probe is the safety-critical question
  and is exactly the fallback; "pure; never raises" by design.
- 382: answer_clarifications body failure -> the original
  symptoms, literally true and unchanged; re-routing proceeds
  on them, no claim is made about the answers.
- 400: route_with_context HealthTimeline() failure -> plain
  triage_route result; no reason asserts anything about the
  history, and no production caller uses this path (chat
  constructs the timeline itself and surfaces its failure).
- 405: _symptom_key import failure -> recurrence correlation
  skipped, plain route returned; mirror of 400.
- 447: handover_report HealthTimeline() failure -> report
  built without history sections; it asserts no history
  exists/doesn't, and the missing sections are visible.
- 561: prepare_visit HealthTimeline() failure -> the packet
  renders the explicit honest string "timeline unavailable."
- 672: CONVERTED — summarize_visit HealthTimeline() failure
  left logged_event_id="" while format_recap rendered "full
  notes saved to your timeline", a false success claim on an
  anticipated storage/IO failure. Fix: the handler binds the
  error and debug-logs it (the module's own mechanism, as its
  sibling handlers do), and format_recap renders the saved
  claim only when an event id exists, otherwise an honest
  could-NOT-be-saved line.
"""

import logging
import sys
import tempfile
import types as _types
from pathlib import Path
from unittest import mock

import nomorals.health.previsit as P
from nomorals.health.failures import recent_failures
from nomorals.health.timeline import HealthTimeline


def _snap():
    return list(recent_failures(200))


def test_354_clarify_questions_failure_returns_general_probe():
    before = _snap()

    class BadIter:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            raise RuntimeError("iter boom")

        def __bool__(self):
            return True

    bad = BadIter()
    assert P.clarify_questions(bad) == [P._CLARIFY_GENERAL]
    assert bad.calls == 1
    qs = P.clarify_questions(["headache"])
    assert len(qs) > 1 and P._CLARIFY_GENERAL in qs
    assert _snap() == before


def test_382_answer_clarifications_failure_returns_symptoms():
    before = _snap()

    class BadDict(dict):
        def __init__(self):
            self.calls = 0

        def items(self):
            self.calls += 1
            raise RuntimeError("items boom")

    bd = BadDict()
    bd["q"] = "yes"  # non-empty: (answers or {}) must reach .items()
    assert P.answer_clarifications(["headache"], bd) == ["headache"]
    assert bd.calls == 1
    qs = P.clarify_questions(["cough"])
    dur_q = next(q for q in qs if "how long" in q)
    merged = P.answer_clarifications(["cough"], {dur_q: "3 days"})
    assert any("3 days" in s for s in merged)
    assert _snap() == before


class _Ev:
    def __init__(self, text):
        self.text = text


class _FakeTL:
    def timeline(self, since=None, event_type=None, limit=500):
        return [_Ev("headache, mild")] * 3


def test_400_route_with_context_timeline_open_failure_plain_route():
    before = _snap()
    ctor_calls = []

    def bad_tl(*a, **k):
        ctor_calls.append(1)
        raise RuntimeError("db boom")

    with mock.patch.object(P, "HealthTimeline", bad_tl):
        r = P.route_with_context(["mild headache"])
    assert len(ctor_calls) == 1
    assert r.level == "self_care"
    assert not any("logged this" in x for x in r.reasons)
    r2 = P.route_with_context(["mild headache"], timeline=_FakeTL())
    assert r2.level == "routine_care"  # control: recurrence bump
    assert any("logged this 3" in x for x in r2.reasons)
    assert _snap() == before


def test_405_symptom_key_import_failure_skips_recurrence():
    before = _snap()
    stub = _types.ModuleType("nomorals.health.timeline")
    stub_calls = []

    def _stub_getattr(name):
        stub_calls.append(name)
        raise AttributeError(name)

    stub.__getattr__ = _stub_getattr
    with mock.patch.dict(sys.modules, {"nomorals.health.timeline": stub}):
        r = P.route_with_context(["mild headache"], timeline=_FakeTL())
    assert "_symptom_key" in stub_calls  # import attempted
    assert r.level == "self_care"  # no bump without the grouping key
    r2 = P.route_with_context(["mild headache"], timeline=_FakeTL())
    assert r2.level == "routine_care"  # control: real module -> bump
    assert _snap() == before


def test_447_handover_timeline_open_failure_report_without_history():
    before = _snap()
    ctor_calls = []

    def bad_tl(*a, **k):
        ctor_calls.append(1)
        raise RuntimeError("db boom")

    with mock.patch.object(P, "HealthTimeline", bad_tl):
        rep = P.handover_report(["headache"])
    assert len(ctor_calls) == 1
    assert "# Care handover" in rep and "headache" in rep
    assert "Symptom history" not in rep
    tmpdb = str(Path(tempfile.mkdtemp()) / "h.db")
    realtl = HealthTimeline(tmpdb)
    realtl.log("symptom", "headache", severity=3)
    with mock.patch.object(P, "HealthTimeline", lambda *a, **k: realtl):
        rep2 = P.handover_report(["headache"])
    assert "Symptom history" in rep2  # control
    realtl.close()
    assert _snap() == before


def test_561_prepare_visit_timeline_open_failure_honest_string():
    before = _snap()
    ctor_calls = []

    def bad_tl(*a, **k):
        ctor_calls.append(1)
        raise RuntimeError("db boom")

    with mock.patch.object(P, "HealthTimeline", bad_tl):
        out = P.prepare_visit(["knee pain"])
    assert len(ctor_calls) == 1
    assert "timeline unavailable." in out and "knee pain" in out
    tmpdb = str(Path(tempfile.mkdtemp()) / "h2.db")
    realtl = HealthTimeline(tmpdb)
    realtl.log("symptom", "knee pain since tuesday", severity=2)
    with mock.patch.object(P, "HealthTimeline", lambda *a, **k: realtl):
        out2 = P.prepare_visit(["knee pain"])
    assert "knee pain since tuesday" in out2  # control
    realtl.close()
    assert _snap() == before


def test_672_summarize_visit_timeline_open_failure_no_saved_claim(caplog):
    before = _snap()
    ctor_calls = []

    def bad_tl(*a, **k):
        ctor_calls.append(1)
        raise RuntimeError("db boom")

    with caplog.at_level(logging.DEBUG, logger="nomorals.health.previsit"):
        with mock.patch.object(P, "HealthTimeline", bad_tl):
            recap = P.summarize_visit("doctor said rest, come back next week")
    assert len(ctor_calls) == 1
    assert recap.logged_event_id == ""
    assert any("timeline open failed" in r.message for r in caplog.records)
    rendered = P.format_recap(recap)
    assert "full notes saved to your timeline" not in rendered
    assert "could NOT be saved" in rendered
    tmpdb = str(Path(tempfile.mkdtemp()) / "h3.db")
    realtl = HealthTimeline(tmpdb)
    recap2 = P.summarize_visit("doctor said rest, come back next week",
                               timeline=realtl)
    assert recap2.logged_event_id  # control: really logged
    assert "full notes saved to your timeline" in P.format_recap(recap2)
    realtl.close()
    assert _snap() == before
