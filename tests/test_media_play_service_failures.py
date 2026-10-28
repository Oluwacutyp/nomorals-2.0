"""Pins for nomorals/media/play_service.py — tail-15 (pins-only, 0 converts / 10 exclusions).

Module contract: "Never raises out of the public functions", and the
PlayReport invariant — the report ALWAYS carries what was tried and why
each attempt failed, never a dead end. Every site below maps a failure
to a documented miss value, an honest failure note, a recorded attempt,
or a safe default; nothing is fabricated and no outcome is misreported.

NOT-recording is pinned via the media failures ledger snapshot (tail-8
method): play_service.py records nothing into it, so recent_failures()
must be unchanged after each trigger. (Site 471 logs the dead provider
to _log at info level — a log line, not a ledger record.)

The pins file itself contains no except handlers (AST-checked).
"""
import sys

from nomorals.media import failures
from nomorals.media.play_service import (
    PlayService, merge_charts, quality_licence_rank)


def _snapshot():
    return failures.recent_failures()


def _assert_unrecorded(before):
    assert failures.recent_failures() == before


class _BoomLexicon:
    def normalize_query(self, q):
        raise RuntimeError("lexicon boom")


class _BoomLLM:
    def complete(self, prompt):
        raise RuntimeError("llm boom")


class _BadRes:
    @property
    def kind(self):
        raise RuntimeError("kind boom")


class _BadResolution:
    @property
    def attempts(self):
        raise RuntimeError("attempts boom")


def _boom_provider(nation):
    raise RuntimeError("provider boom")


def test_154_sort_failure_returns_input_order():
    before = _snapshot()
    cands = [{"quality": "audio", "licence": "licensed", "score": "not-a-number"},
             {"quality": "lossless", "licence": "licensed", "score": 9.0}]
    assert quality_licence_rank(cands) == cands
    _assert_unrecorded(before)


def test_201_merge_failure_honest_note_and_empty():
    before = _snapshot()
    merged = merge_charts({"p": ([{"title": "T", "artist": "A", "rank": 1}], "bad")})
    assert merged["entries"] == [] and merged["note"] == "chart merge failed"
    _assert_unrecorded(before)


def test_261_understand_failure_keeps_verbatim_query():
    before = _snapshot()
    svc = PlayService()
    svc._lexicon = _BoomLexicon()
    intent = svc.understand("hello world")
    assert intent.original == "hello world"
    assert intent.search_query == "hello world"
    assert intent.understood_by == "deterministic"
    _assert_unrecorded(before)


def test_270_llm_parse_failure_returns_none():
    before = _snapshot()
    assert PlayService._llm_parse(_BoomLLM(), "x") is None
    _assert_unrecorded(before)


def test_303_resolver_import_failure_recorded_as_unavailable(monkeypatch):
    before = _snapshot()
    monkeypatch.setitem(sys.modules, "nomorals.media.resolver", None)
    report = PlayService().resolve("some song")
    assert report.ok is False
    assert any(a.strategy == "resolver" and "unavailable:" in a.detail
               for a in report.tried)
    _assert_unrecorded(before)


def test_428_licence_failure_returns_unknown():
    before = _snapshot()
    assert PlayService()._licence_for(_BadRes()) == "unknown"
    _assert_unrecorded(before)


def test_446_explain_failure_honest_fallback():
    before = _snapshot()
    assert (PlayService().explain_failure(_BadResolution())
            == "tried: (no attempt log)")
    _assert_unrecorded(before)


def test_471_dead_chart_provider_skipped_with_honest_note():
    before = _snapshot()
    charts = PlayService(chart_providers={"bad": _boom_provider}).charts("NG")
    assert charts["entries"] == []
    assert "no chart fabricated" in charts["note"]
    _assert_unrecorded(before)


def test_513_curve_failure_keeps_computed_target(monkeypatch):
    before = _snapshot()
    monkeypatch.setitem(sys.modules, "nomorals.media.dj_engine", None)
    plan = PlayService().dj_continuity(
        feedback_rows=[{"event": "kept"}, {"event": "kept"}, {"event": "kept"},
                       {"event": "liked"}],
        played_ids=["t1"], current_energy=0.6)
    # Unsnapped pushing target (0.6 + 0.10); the snapped value is 0.85.
    assert plan.next_energy_target == 0.7
    assert plan.avoid_ids == ["t1"]
    _assert_unrecorded(before)


def test_521_outer_failure_holds_current_energy():
    before = _snapshot()
    plan = PlayService().dj_continuity(feedback_rows=[object()],
                                       played_ids=["t1"], current_energy=0.6)
    assert plan.next_energy_target == 0.6
    assert plan.avoid_ids == ["t1"]
    assert plan.read_the_room == ""
    _assert_unrecorded(before)
