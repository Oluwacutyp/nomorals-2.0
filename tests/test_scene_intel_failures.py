"""Pins for Pass #107: scene_intel failures stop vanishing.

Every converted site keeps its fallback AND records the failure in the
scene_intel ledger. The ledger itself is stdlib-only, capped, and
never raises.
"""

import sys
import types
from unittest import mock

import pytest

from nomorals.media import scene_intel
from nomorals.media.scene_intel import failures
from nomorals.media.scene_intel import score as score_mod
from nomorals.media.scene_intel import segment as segment_mod
from nomorals.tools.error_scan import scan


@pytest.fixture(autouse=True)
def _clean_ledger():
    failures.clear_failures()
    yield
    failures.clear_failures()


def _sites():
    return [f.site for f in failures.recent_failures(500)]


def test_ledger_basics():
    failures.record_failure("score.test_site", ValueError("boom"))
    assert failures.failure_count() == 1
    recent = failures.recent_failures()
    assert len(recent) == 1
    assert recent[0].site == "score.test_site"
    assert recent[0].exc_type == "ValueError"
    assert recent[0].message == "boom"


def test_ledger_bounded():
    for _ in range(250):
        failures.record_failure("score.flood", RuntimeError("x"))
    assert failures.failure_count() == 250
    assert len(failures.recent_failures(500)) == 200


def test_ledger_unprintable_exc():
    class BadExc(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    failures.record_failure("score.bad_str", BadExc())
    assert failures.recent_failures()[0].message.startswith("<unprintable ")


def test_format_render_and_empty():
    assert failures.format_failures() == "no scene_intel failures recorded."
    failures.record_failure("score.fmt", OSError("disk gone"))
    rendered = failures.format_failures()
    assert "scene_intel failures (1 total" in rendered
    assert "score.fmt: OSError: disk gone" in rendered


def test_package_all_unchanged():
    # #102 lesson: do not widen a declared package surface without need.
    # The ledger is a direct import, so __all__ must be untouched.
    assert scene_intel.__all__ == [
        "analyze_movie", "MovieAnalysis",
        "Segmentation", "CharacterTrack", "ScoredSegment", "CharacterProfile",
        "ModelUnavailable", "character_reel",
    ]


def test_audio_curve_failure_records_and_falls_back(monkeypatch):
    monkeypatch.setattr(score_mod, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(score_mod.subprocess, "run",
                        mock.Mock(side_effect=RuntimeError("ffmpeg boom")))
    assert score_mod._audio_energy_curve("nope.mp4") == []
    assert "score._audio_energy_curve" in _sites()


def test_audio_curve_cleanup_failure_records(monkeypatch):
    monkeypatch.setattr(score_mod, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(score_mod.subprocess, "run",
                        mock.Mock(side_effect=RuntimeError("ffmpeg boom")))
    monkeypatch.setattr(score_mod.os, "unlink",
                        mock.Mock(side_effect=OSError("unlink boom")))
    assert score_mod._audio_energy_curve("nope.mp4") == []
    sites = _sites()
    assert "score._audio_energy_curve" in sites
    assert "score._audio_energy_curve:cleanup" in sites


def test_dialogue_import_failure_records():
    with mock.patch.dict(sys.modules, {"nomorals.voice.stt": None}):
        assert score_mod._dialogue_curve("nope.mp4") == []
    assert "score._dialogue_curve:import" in _sites()


def test_dialogue_transcribe_failure_records():
    fake = types.ModuleType("nomorals.voice.stt")
    fake.transcribe = mock.Mock(side_effect=RuntimeError("stt boom"))
    with mock.patch.dict(sys.modules, {"nomorals.voice.stt": fake}):
        assert score_mod._dialogue_curve("nope.mp4") == []
    assert "score._dialogue_curve:transcribe" in _sites()


def test_excitement_events_failure_records(monkeypatch):
    monkeypatch.setattr(score_mod, "_audio_energy_curve",
                        mock.Mock(side_effect=RuntimeError("curve boom")))
    assert score_mod.excitement_events("nope.mp4") == []
    assert "score.excitement_events" in _sites()


def test_learned_model_load_failure_records(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"\x00" * 16)
    fake = types.ModuleType("nomorals.media.highlight_model.model")
    fake.default_weights_path = lambda: str(weights)
    fake.HighlightMLP = mock.Mock(side_effect=RuntimeError("mlp boom"))
    with mock.patch.dict(sys.modules,
                         {"nomorals.media.highlight_model.model": fake}):
        assert score_mod.score_segments("nope.mp4", [(0.0, 10.0)]) == []
    assert "score.score_segments:learned_model" in _sites()
    assert "score.score_segments:learned_features" not in _sites()


def test_learned_features_failure_records(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"\x00" * 16)
    model = types.ModuleType("nomorals.media.highlight_model.model")
    model.default_weights_path = lambda: str(weights)
    model.HighlightMLP = mock.Mock(return_value=object())
    feats = types.ModuleType("nomorals.media.highlight_model.features")
    feats.extract_features = mock.Mock(
        side_effect=RuntimeError("features boom"))
    with mock.patch.dict(sys.modules, {
            "nomorals.media.highlight_model.model": model,
            "nomorals.media.highlight_model.features": feats}):
        assert score_mod.score_segments("nope.mp4", [(0.0, 10.0)]) == []
    assert "score.score_segments:learned_features" in _sites()
    assert "score.score_segments:learned_model" not in _sites()


def test_pyscenedetect_failure_records():
    fake = types.ModuleType("scenedetect")
    fake.detect = mock.Mock(side_effect=RuntimeError("detect boom"))
    fake.AdaptiveDetector = mock.Mock(return_value=object())
    with mock.patch.dict(sys.modules, {"scenedetect": fake}):
        assert segment_mod._pyscenedetect_segments("nope.mp4") is None
    assert "segment._pyscenedetect_segments" in _sites()


def test_subsystem_scan_clean():
    rep = scan(["nomorals/media/scene_intel"])
    assert rep.findings == []
