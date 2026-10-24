"""Pass #113 pins: edit_engine failure ledger + all 10 converted sites."""
import sys
from importlib import import_module
from types import ModuleType, SimpleNamespace

import pytest

from nomorals.media.edit_engine import __all__ as edit_engine_all
from nomorals.media.edit_engine import failures
from nomorals.media.edit_engine.document import Document
from nomorals.media.edit_engine.quality_gate import _aesthetic_score, auto_fix
from nomorals.media.edit_engine.render import (
    RenderReport, _perf_for, render_timeline,
)
# NB: the package's __init__ binds the name `render` to spec.render (a
# function), shadowing the submodule attribute — import_module by dotted
# path returns the real render module for monkeypatching.
render_mod = import_module("nomorals.media.edit_engine.render")
from nomorals.media.edit_engine.text import TextLayer
from nomorals.media.edit_engine.video_four import auto_broll


@pytest.fixture(autouse=True)
def _clear():
    failures.clear_failures()
    yield
    failures.clear_failures()


def _sites():
    return [f.site for f in failures.recent_failures(100)]


# --- ledger basics -------------------------------------------------------

def test_ledger_basics_and_bound():
    assert failures.format_failures() == "no edit_engine failures recorded."
    assert failures.failure_count() == 0
    for _ in range(250):
        failures.record_failure("pin.site", ValueError("boom"))
    assert failures.failure_count() == 250
    assert len(failures.recent_failures(1000)) == 200
    assert failures.recent_failures(1)[0].site == "pin.site"
    assert "pin.site" in failures.format_failures()
    assert "250 total" in failures.format_failures()


def test_ledger_unprintable_exc():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    failures.record_failure("pin.bad", BadStr())
    entry = failures.recent_failures(1)[0]
    assert entry.exc_type == "BadStr"
    assert "unprintable" in entry.message


def test_direct_import_surface_and_all_unchanged():
    from nomorals.media.edit_engine.failures import (
        Failure, clear_failures, failure_count, format_failures,
        recent_failures, record_failure,
    )
    assert callable(record_failure) and callable(recent_failures)
    assert callable(failure_count) and callable(clear_failures)
    assert callable(format_failures) and Failure is not None
    # Deliberately NO package export: __init__ untouched (pre-change len 94).
    assert "failures" not in edit_engine_all
    assert len(edit_engine_all) == 94


# --- document.duration / timeline.duration --------------------------------

def _bad_clip():
    clip = SimpleNamespace(at=1.0)
    clip.play_duration = lambda: (_ for _ in ()).throw(ValueError("pin bad clip"))
    return clip


def _good_clip(at=0.0, dur=3.0):
    return SimpleNamespace(at=at, play_duration=lambda: dur)


def test_document_duration_skips_bad_clip_and_records():
    doc = Document.__new__(Document)
    doc.tracks = [SimpleNamespace(kind="video",
                                   clips=[_good_clip(0.0, 3.0), _bad_clip()])]
    assert doc.duration == 3.0
    assert _sites() == ["document.duration"]


def test_document_duration_healthy_records_nothing():
    doc = Document.__new__(Document)
    doc.tracks = [SimpleNamespace(kind="video",
                                   clips=[_good_clip(0.0, 3.0), _good_clip(3.0, 2.0)])]
    assert doc.duration == 5.0
    assert _sites() == []


def test_timeline_duration_skips_bad_clip_and_records():
    from nomorals.media.edit_engine.timeline import Timeline
    tl = Timeline.__new__(Timeline)
    tl.video = SimpleNamespace(clips=[_good_clip(0.0, 4.0), _bad_clip()])
    assert tl.duration == 4.0
    assert _sites() == ["timeline.duration"]


# --- quality_gate: learned model + critique -------------------------------

def test_aesthetic_learned_model_failure_falls_back_and_records():
    boom = ModuleType("nomorals.media.highlight_model.model")

    class BoomMLP:
        def __init__(self, *a, **k):
            raise RuntimeError("pin: model load fails")

    boom.HighlightMLP = BoomMLP
    boom.default_weights_path = lambda: "/nonexistent/weights"
    key = "nomorals.media.highlight_model.model"
    original = sys.modules.get(key)
    sys.modules[key] = boom
    try:
        result = _aesthetic_score({"highlight_features": [[0.1] * 8]})
    finally:
        if original is None:
            sys.modules.pop(key, None)
        else:
            sys.modules[key] = original
    assert abs(result - 0.1) < 1e-9  # heuristic fallback held
    assert _sites() == ["quality_gate._aesthetic_score:learned_model"]


def test_aesthetic_critique_failure_returns_none_and_records():
    result = _aesthetic_score(
        {"image_features": {"exposure": 0.5, "saturation": "bad"}})
    assert result is None
    assert _sites() == ["quality_gate._aesthetic_score:critique"]


def test_aesthetic_critique_healthy_records_nothing():
    result = _aesthetic_score(
        {"image_features": {"exposure": 0.5, "saturation": 0.45}})
    assert abs(result - 0.7) < 1e-9
    assert _sites() == []


# --- quality_gate: auto_fix caption shifts --------------------------------

class _StartLocked:
    words = []
    end = None

    def __init__(self):
        self._start = 1.0

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        raise AttributeError("pin: start locked")


class _EndLocked:
    words = []
    start = 1.0

    def __init__(self):
        self._end = 5.0

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        raise AttributeError("pin: end locked")


class _CopyDoc(Document):
    """Document subclass whose copy() returns stub layers + seal()."""

    def __init__(self, layers):
        self._stub_layers = layers

    def copy(self):
        return SimpleNamespace(text_layers=self._stub_layers,
                               loudness={}, seal=lambda: {"stub": True})


def _drift_report():
    return {"axes": {"caption_sync_drift": {"status": "fail"}}}


def test_auto_fix_shift_start_failure_recorded():
    out = auto_fix(_CopyDoc([_StartLocked()]), _drift_report(),
                   {"caption_drift_s": 0.5})
    assert "captions shifted" in " ".join(out["applied"])
    assert _sites() == ["quality_gate.auto_fix:shift_start"]


def test_auto_fix_shift_end_failure_recorded():
    out = auto_fix(_CopyDoc([_EndLocked()]), _drift_report(),
                   {"caption_drift_s": 0.5})
    assert "captions shifted" in " ".join(out["applied"])
    assert _sites() == ["quality_gate.auto_fix:shift_end"]


def test_auto_fix_shift_healthy_shifts_and_records_nothing():
    doc = Document(text_layers=[TextLayer(text="hello", start=2.0, end=5.0)])
    out = auto_fix(doc, _drift_report(), {"caption_drift_s": 0.5})
    sealed = out["document"]
    assert sealed.text_layers[0].start == 1.5
    assert sealed.text_layers[0].end == 4.5
    assert _sites() == []


# --- render: fps parse, profile detect, hash ------------------------------

def test_self_review_fps_parse_failure_recorded(tmp_path):
    probe = {"streams": [{"codec_type": "video", "codec_name": "h264",
                           "pix_fmt": "yuv420p", "width": 1080, "height": 1920,
                           "avg_frame_rate": "banana"}]}
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x" * 64)
    original = render_mod.video_probe
    render_mod.video_probe = lambda p: probe
    try:
        report = RenderReport(output=str(target), bytes=64)
        report.self_review()
    finally:
        render_mod.video_probe = original
    assert report.fps == 0.0
    assert _sites() == ["render.self_review:fps"]


def test_perf_for_detect_failure_defaults_pc_and_records():
    original = render_mod.detect_profile
    render_mod.detect_profile = lambda: (_ for _ in ()).throw(
        RuntimeError("pin: no profiler"))
    try:
        result = _perf_for(None)
    finally:
        render_mod.detect_profile = original
    assert result["profile"] == "pc"
    assert _sites() == ["render._perf_for"]
    assert _perf_for("pc")["profile"] == "pc"
    assert _sites() == ["render._perf_for"]  # healthy path adds nothing


def test_render_timeline_hash_failure_recorded(tmp_path):
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    timeline = SimpleNamespace(
        video=SimpleNamespace(clips=[SimpleNamespace(path="x.mp4")]),
        width=1080, height=1920, fps=30, seed=0,
        audio_mix=None, text_layers=[])
    original = render_mod.assemble_segments
    render_mod.assemble_segments = lambda *a, **k: {
        "output": str(outdir), "duration": 2.5, "profile": "pc"}
    try:
        result = render_timeline(timeline, out=str(outdir), crf=21,
                                 workdir=str(tmp_path))
    finally:
        render_mod.assemble_segments = original
    assert result["duration"] == 2.5
    assert result["profile"] == "pc"
    assert result["render_report"].sha256 == ""  # hash failed, pass kept
    assert _sites() == ["render.render_timeline:sha256"]


# --- video_four: index query ----------------------------------------------

class _BoomIndex:
    def query(self, topic):
        raise RuntimeError("pin: index down")


def test_auto_broll_index_failure_recorded():
    picks = auto_broll([], _BoomIndex(), topics=["mountain"])
    assert picks == []
    assert _sites() == ["video_four.auto_broll:index_query"]


def test_auto_broll_healthy_records_nothing():
    class OkIndex:
        def query(self, topic):
            return [{"clip_id": "c1", "start": 0}]

    picks = auto_broll([], OkIndex(), topics=["mountain"])
    assert len(picks) == 1 and picks[0]["topic"] == "mountain"
    assert _sites() == []


# --- subsystem remainder = exactly the deliberate _is_still probe ----------

def test_subsystem_remainder_is_only_is_still_probe(tmp_path):
    from nomorals.tools import error_scan as es
    # Suppression-off: the deliberate site is noqa-tagged, so only the
    # true-estate scan sees it. Assert it is the SOLE remaining site.
    original = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan(["/home/hatch/workspace/nm113-tree/nomorals/media/edit_engine"])
    finally:
        es._noqa_suppresses = original
    mine = [(f.file.split("/")[-1], f.rule, f.context)
            for f in report.findings]
    assert mine == [("render.py", "E104",
                     "except Exception:  # noqa: BLE001 — not an image, or PIL disagrees")]
    # Functional proof the deliberate site is a probe, not a failure:
    # a non-image answers False by design, with no ledger entry.
    from nomorals.media.edit_engine.render import _is_still
    text_file = tmp_path / "notes.txt"
    text_file.write_text("not an image")
    assert _is_still(str(text_file)) is False
    assert _sites() == []
