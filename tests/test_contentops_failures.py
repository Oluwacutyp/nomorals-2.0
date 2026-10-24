"""Pass #109 pins: the contentops failure ledger records every converted site.

Covers the ledger contract (basics, 200-cap, unprintable exceptions,
render, empty state), the direct-import surface, and one real drive per
converted site: the fallback/default is held AND the site is recorded.
"""

import array
import sys
import types
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.contentops import failures as F
from nomorals.media.contentops.failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)


def sites():
    return [f.site for f in recent_failures(500)]


@pytest.fixture(autouse=True)
def _clean():
    clear_failures()
    yield
    clear_failures()


def test_ledger_basics_and_render():
    record_failure("contentops.pin", ValueError("boom"))
    assert failure_count() == 1
    (entry,) = recent_failures()
    assert entry.site == "contentops.pin"
    assert entry.exc_type == "ValueError"
    assert entry.message == "boom"
    rendered = format_failures()
    assert "contentops.pin" in rendered and "ValueError: boom" in rendered


def test_ledger_bound_and_empty_state():
    assert format_failures() == "no contentops failures recorded."
    for i in range(250):
        record_failure(f"contentops.pin#{i}", RuntimeError("x"))
    assert failure_count() == 250
    assert len(recent_failures(500)) == 200  # ring cap holds
    assert recent_failures(1)[0].site == "contentops.pin#249"  # newest first


def test_ledger_unprintable_exc():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("nope")

    record_failure("contentops.pin", BadStr())
    (entry,) = recent_failures()
    assert entry.message.startswith("<unprintable ")


def test_direct_import_surface_and_package_still_imports():
    import nomorals.media.contentops  # noqa: F401 — must stay importable

    assert F.__all__ == ["Failure", "record_failure", "recent_failures",
                         "failure_count", "clear_failures", "format_failures"]


def _raising_catalogue():
    class Catalogue:
        @property
        def voices(self):
            raise RuntimeError("catalogue unreadable")

    return Catalogue()


def test_shims_resolve_records_and_falls_back():
    from nomorals.media.contentops._shims import _ShimVoiceSpec

    assert _ShimVoiceSpec().resolve(_raising_catalogue()) == ""
    assert "_shims.resolve" in sites()


def _synth_wav(path: Path, seconds: float = 1.0) -> None:
    """Mono 22050 Hz wav: loud bursts on silence, so onset pickers fire."""
    sr = 22050
    frames = array.array("h", [0]) * int(sr * seconds)
    for start in (0.1, 0.5):
        base = int(start * sr)
        for i in range(2000):
            frames[base + i] = 12000 if i % 2 else -12000
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(frames.tobytes())


def test_shims_detect_beats_holds_and_records_unlink_failure(tmp_path):
    from nomorals.media.contentops import _shims

    wav = tmp_path / "tone.wav"
    _synth_wav(wav)
    onsets = _shims.detect_beats(str(wav))
    assert isinstance(onsets, list)  # the `e` loop var still works post-edit
    assert "_shims.detect_beats" not in sites()  # healthy run: silent ledger
    with mock.patch.object(_shims.os, "unlink", side_effect=OSError("gone")):
        onsets2 = _shims.detect_beats(str(wav))
    assert onsets2 == onsets  # fallback (returned beats) untouched
    assert "_shims.detect_beats" in sites()


def test_beats_decode_mono_holds_and_records_unlink_failure(tmp_path):
    from nomorals.media.contentops import beats

    wav = tmp_path / "tone.wav"
    _synth_wav(wav)
    arr = beats._decode_mono(wav)
    assert arr.shape[0] > 0
    assert "beats._decode_mono" not in sites()
    with mock.patch.object(beats.os, "unlink", side_effect=OSError("gone")):
        arr2 = beats._decode_mono(wav)
    assert arr2.shape == arr.shape
    assert "beats._decode_mono" in sites()


def test_niches_base_resolve_records_and_falls_back():
    from nomorals.media.contentops.niches.base import VoiceSpec

    assert VoiceSpec().resolve(_raising_catalogue()) == ""
    assert "niches.base.resolve" in sites()


def _concrete_niche():
    from nomorals.media.contentops.niches.base import NichePlugin

    class PinNiche(NichePlugin):
        name = "pin"

        def script_prompt(self, topic: str) -> str:
            return f"Write a script about {topic}."

        def visual_strategy(self, script: str):
            raise AssertionError("not driven in this pin")

    return PinNiche()


def test_niches_base_generate_script_records_and_returns_empty():
    class BoomBrain:
        def complete(self, prompt):
            raise RuntimeError("brain down")

    result = _concrete_niche().generate_script(BoomBrain(), "test topic")
    assert result.text == "" and result.word_count == 0
    assert result.topic == "test topic" and result.niche == "pin"
    assert "niches.base.generate_script" in sites()


def _run_bootstrap_with(modules):
    from nomorals.media.contentops.niches import registry

    saved_bootstrapped = registry._BOOTSTRAPPED
    saved_modules = registry._SHIPPED_MODULES
    registry._BOOTSTRAPPED = False
    registry._SHIPPED_MODULES = modules
    try:
        registry._bootstrap()
    finally:
        registry._BOOTSTRAPPED = saved_bootstrapped
        registry._SHIPPED_MODULES = saved_modules


def test_registry_bootstrap_import_failure_recorded():
    _run_bootstrap_with(("_bogus_nope_xyz",))
    assert "niches.registry._bootstrap" in sites()


def test_registry_bootstrap_register_failure_recorded():
    from nomorals.media.contentops.niches import registry
    from nomorals.media.contentops.niches.base import NichePlugin
    from nomorals.media.contentops.niches.registry import NicheError

    fake_name = "nomorals.media.contentops.niches._pinboom"
    mod = types.ModuleType(fake_name)

    class Boom(NichePlugin):
        def __init__(self):
            raise NicheError("boom at construction")

        def script_prompt(self, topic: str) -> str:
            return "x"

        def visual_strategy(self, script: str):
            raise AssertionError("not driven")

    Boom.__module__ = fake_name
    mod.Boom = Boom
    sys.modules[fake_name] = mod
    try:
        _run_bootstrap_with(("_pinboom",))
    finally:
        sys.modules.pop(fake_name, None)
    assert "niches.registry._bootstrap#2" in sites()
    assert registry._REGISTRY.get("pin") is None  # nothing half-registered


def test_registry_get_niche_miss_records_and_falls_back_to_shim():
    from nomorals.media.contentops.niches.registry import get_niche

    plugin = get_niche("definitely-not-a-niche-xyz")
    assert plugin is not None
    assert "niches.registry.get_niche" in sites()


def test_pipeline_trail_records_and_never_raises():
    from nomorals.media.contentops.pipeline import RunResult

    result = RunResult(ok=True, run_id="r", job_id="j")

    class RaisingList(list):
        def append(self, item):
            raise OSError("trail store down")

    result.decision_trail = RaisingList()
    assert result.trail("picked voice", "test") is None
    assert "pipeline.trail" in sites()


def _pipeline(tmp_path):
    from nomorals.media.contentops.pipeline import ShortPipeline

    return ShortPipeline(tmp_path)


def test_pipeline_voice_candidates_resolve_failure_recorded(tmp_path):
    pipe = _pipeline(tmp_path)

    class BadSpec:
        def resolve(self, tts):
            raise RuntimeError("foreign spec exploded")

        def candidates(self):
            return ["voice-a"]

        preferred_voice = ""
        fallback_voice = ""

    niche = SimpleNamespace(voice_spec=BadSpec(), voice_style="")
    out = pipe._voice_candidates(niche)
    assert "voice-a" in out and out[-1] == "narrator"
    assert "pipeline._voice_candidates" in sites()


def test_pipeline_voice_candidates_candidates_failure_recorded(tmp_path):
    pipe = _pipeline(tmp_path)

    class BadCandidates:
        def resolve(self, tts):
            return ""

        def candidates(self):
            raise RuntimeError("candidates exploded")

        preferred_voice = "pref"
        fallback_voice = ""

    niche = SimpleNamespace(voice_spec=BadCandidates(), voice_style="")
    out = pipe._voice_candidates(niche)
    assert "pref" in out and out[-1] == "narrator"
    assert "pipeline._voice_candidates#2" in sites()


def test_pipeline_resolve_style_records_and_falls_back_to_phonk(tmp_path):
    pipe = _pipeline(tmp_path)
    pipe._get_niche = lambda name: (_ for _ in ()).throw(
        RuntimeError("niche store down"))
    job = SimpleNamespace(style="", niche="some-niche")
    assert pipe._resolve_style({"job": job}) == "phonk"
    assert "pipeline._resolve_style" in sites()


def _resp(*, status=200, ok=True, json_exc=None, headers=None):
    class Resp:
        def __init__(self):
            self.status = status
            self.ok = ok
            self.text = ""
            self._headers = headers

        def json(self):
            if json_exc is not None:
                raise json_exc
            return {}

        @property
        def headers(self):
            if isinstance(self._headers, Exception):
                raise self._headers
            return self._headers

    return Resp()


def test_publish_meta_json_failure_recorded(tmp_path):
    from nomorals.media.contentops.publish.meta import (
        CapabilityUnavailable,
        MetaPublisher,
    )

    pub = MetaPublisher.__new__(MetaPublisher)
    pub.http = SimpleNamespace(
        post_json=lambda *a, **k: _resp(
            status=200, ok=True, json_exc=ValueError("not json")))
    with pytest.raises(CapabilityUnavailable):
        pub.publish_facebook_video(
            "page1", video_url="https://example.com/v.mp4",
            description="d", page_access_token="test-token",
            confirmed=True)
    assert "publish.meta.publish_facebook_video" in sites()


def test_publish_youtube_session_uri_failure_recorded():
    from nomorals.media.contentops.publish.youtube import (
        YouTubeError,
        YouTubePublisher,
    )

    pub = YouTubePublisher.__new__(YouTubePublisher)
    pub.http = SimpleNamespace(
        request=lambda *a, **k: _resp(
            status=200, headers=RuntimeError("headers unreadable")))
    pub._google_access_token = lambda: "tok"
    with pytest.raises(YouTubeError, match="no resumable session URI"):
        pub.start_resumable_session({}, 1234)
    assert "publish.youtube.start_resumable_session" in sites()


def test_publish_youtube_offset_failure_recorded_as_zero():
    from nomorals.media.contentops.publish.youtube import YouTubePublisher

    pub = YouTubePublisher.__new__(YouTubePublisher)
    pub.http = SimpleNamespace(
        request=lambda *a, **k: _resp(
            status=308, headers=RuntimeError("headers unreadable")))
    pub._google_access_token = lambda: "tok"
    assert pub.query_upload_offset("session-uri", 1234) == 0
    assert "publish.youtube.query_upload_offset" in sites()


def test_contentops_subsystem_scan_clean():
    from nomorals.tools.error_scan import scan

    root = Path(__file__).resolve().parents[1]
    report = scan([str(root / "nomorals" / "media" / "contentops")])
    assert report.findings == []
