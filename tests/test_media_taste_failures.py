"""Tail-16 pins for nomorals/media/taste.py — 0 converts / 10 exclusions.

Every site degrades to a documented miss value under the module's
Never-raises contract; nothing here fabricates success, moves money,
publishes, or writes accounts. NOT-recording is pinned via the media
failures ledger snapshot (tail-8 method): taste.py records nothing into
it, so recent_failures() must be unchanged across each trigger. (Its
own _log lines are logs, not ledger records.)

No except handlers appear in this file (AST-checked).
"""
import os
import sys
import tempfile
from types import SimpleNamespace

from nomorals.media import failures
from nomorals.media.taste import (
    TasteProfile as TP,
)
from nomorals.media.taste import (
    TasteStore,
    _taste_path,
    detect_feedback,
    detect_produce_intent,
    last_production_profile,
    modify_profile,
    mood_hint,
)


def _snap():
    return failures.recent_failures()


class _RaisingLower:
    def __bool__(self):
        return True

    def lower(self):
        raise RuntimeError("pin: lower failed")


class _RaisingGet(dict):
    def get(self, *a, **k):
        raise RuntimeError("pin: get failed")


class _HostileStr:
    def __str__(self):
        raise RuntimeError("pin: str failed")


def test_073_path_failure_falls_back_to_home_default():
    before = _snap()
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=_HostileStr()))
    p = _taste_path(ctx)
    assert str(p).endswith(os.path.join(".nomorals", "taste.json"))
    assert failures.recent_failures() == before


def test_126_bad_track_skipped_good_kept():
    before = _snap()
    prof = TP.from_dict({"tracks": [
        {"description": "good", "bpm": 120.0},
        {"description": "bad", "bpm": "not-a-number"},
    ]})
    assert [t.description for t in prof.tracks] == ["good"]
    assert failures.recent_failures() == before


def test_133_bad_tempo_skipped_good_kept():
    before = _snap()
    prof = TP.from_dict({"preferred_tempos": [[100.0, 120.0], ["x", "y"]]})
    assert prof.preferred_tempos == [[100.0, 120.0]]
    assert failures.recent_failures() == before


def test_145_corrupt_scalar_yields_fresh_profile():
    # Documented corrupt -> fresh contract (module docstring + comment).
    # The granularity wart (one scalar discards all parsed tracks) is
    # documented in SURVEY.md for parent adjudication, not pinned as ideal.
    before = _snap()
    prof = TP.from_dict({"tracks": [{"description": "good", "bpm": 120.0}],
                         "production_count": "abc"})
    assert prof.tracks == [] and prof.production_count == 0
    assert failures.recent_failures() == before


def test_277_producer_import_failure_yields_none():
    before = _snap()
    store = TasteStore.__new__(TasteStore)
    store.context = None
    store.profile = TP(tracks=[], production_count=3,
                       energy_weights={"driving": 2.0},
                       preferred_tempos=[[100.0, 120.0]])
    store.path = None
    saved = sys.modules.pop("nomorals.media.producer", None)

    class _Stub:
        pass

    sys.modules["nomorals.media.producer"] = _Stub()
    try:
        assert store.suggest_profile() is None
    finally:
        sys.modules.pop("nomorals.media.producer", None)
        if saved is not None:
            sys.modules["nomorals.media.producer"] = saved
    assert store.suggest_profile() is not None
    assert failures.recent_failures() == before


def test_363_modify_failure_returns_input_unchanged():
    before = _snap()
    bad = SimpleNamespace(artist="a", bpm=120.0, key="A", mode="minor",
                          energy_words=("driving",), genre="edm")
    assert modify_profile(bad, "harder") is bad
    good = SimpleNamespace(title="t", artist="a", bpm=120.0, key="A",
                           mode="minor", energy_words=("driving",),
                           genre="edm")
    out = modify_profile(good, "harder")
    assert out is not good and out.bpm == 128.0
    assert failures.recent_failures() == before


def test_429_detector_failure_yields_none():
    before = _snap()
    assert detect_produce_intent(_RaisingLower()) is None
    assert detect_produce_intent("make me something") is not None
    assert failures.recent_failures() == before


def test_444_mood_hint_failure_yields_empty():
    before = _snap()
    assert mood_hint(_RaisingLower()) == ()
    assert "aggressive" in mood_hint("make me something for the gym")
    assert failures.recent_failures() == before


def test_478_feedback_detector_failure_yields_none():
    before = _snap()
    assert detect_feedback(_RaisingLower()) is None
    assert detect_feedback("I love this") == (True, "I love this")
    assert failures.recent_failures() == before


def test_499_last_production_failure_yields_none():
    before = _snap()
    store = TasteStore.__new__(TasteStore)
    store.profile = TP(last_production=_RaisingGet())
    assert last_production_profile(store) is None
    store2 = TasteStore.__new__(TasteStore)
    store2.profile = TP(last_production={"title": "t", "bpm": 120.0, "key": "A",
                                         "mode": "minor",
                                         "energy_words": ["driving"],
                                         "genre": "edm"})
    rebuilt = last_production_profile(store2)
    assert rebuilt is not None and rebuilt.bpm == 120.0
    assert failures.recent_failures() == before
