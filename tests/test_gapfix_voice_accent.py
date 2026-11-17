"""Voice-direction accent: the accent field finally reaches a tier.

Before this lane, ``render_emotional`` never read ``direction.accent``:
emotion, delivery and pace were realized on the DSP tier and the RVC /
native tiers rendered the mood, but the accent went nowhere — and
``speak_rich`` was ``if direction: ... elif accent: ...``, so passing
both silently dropped the accent. These tests pin the wiring: the
accent rides the source render's language hint (``perform()`` /
``speak()`` already plumb ``language`` through to multilingual
backends), RVC preserves the style while swapping identity, the result
names the accent honestly, and ``speak_rich`` combines direction +
accent instead of choosing.
"""

import math
import wave
from array import array
from pathlib import Path

import pytest

from nomorals.voice.neural_emotion import render_emotional
from nomorals.voice.nl_director import Direction, parse_direction
from nomorals.voice.tts import UniversalTTS


def _read_wav_samples(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.getframerate(), list(
            array("h", w.readframes(w.getnframes())))


class _FakeModel:
    def __init__(self, name):
        self.name = name


class _FakeTTS:
    """perform()/speak() -> tiny wav; records every call incl. language.

    Mirrors the voice-direction fake, plus the language hint and the
    ``native_opts`` a real speak() reports (which backends took the
    hint natively). ``honor_language=False`` simulates a backend that
    drops the hint: the call still succeeds, the report says so.
    """

    def __init__(self, tmp_path, sr=8000, honor_language=True):
        self.sr = sr
        self.tmp_path = Path(tmp_path)
        self.honor_language = honor_language
        self.perform_calls: list[dict] = []
        self.speak_calls: list[dict] = []

    def _write(self, samples):
        n = len(self.perform_calls) + len(self.speak_calls) + 1
        path = str(self.tmp_path / f"fake{n}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(array("h", samples).tobytes())
        return path

    def _samples(self):
        base = (len(self.perform_calls) + len(self.speak_calls) + 1) * 1000
        return [base + i + 1 for i in range(4)]

    def perform(self, text, voice_name=None, mood="neutral", intensity=3,
                language="", **kw):
        samples = self._samples()
        path = self._write(samples)
        self.perform_calls.append(
            {"text": text, "mood": mood, "intensity": intensity,
             "language": language, "path": path, "samples": samples})
        return {"path": path, "backend": "fake"}

    def speak(self, text, voice_name=None, mood="", language="", **kw):
        # a 0.3s tone: the DSP tier shapes real frames, not 4 samples
        samples = [int(8000 * math.sin(2 * math.pi * 440 * i / self.sr))
                   for i in range(int(0.3 * self.sr))]
        path = self._write(samples)
        self.speak_calls.append(
            {"text": text, "language": language, "path": path,
             "samples": samples})
        return {"path": path, "backend": "fake",
                "native_opts": ["language"] if language
                and self.honor_language else []}


@pytest.fixture
def rvc_up(monkeypatch, tmp_path):
    """RVC present with a model for voice 'zara'; records converts."""
    import nomorals.voice.neural_emotion as ne

    calls: list[dict] = []

    def _convert(src_path, model):
        path = str(tmp_path / f"rvc{len(calls) + 1}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(array("h", [9000, 9001]).tobytes())
        calls.append({"src": src_path, "model": model.name, "path": path})
        return {"path": path}

    monkeypatch.setattr(ne.rvc_bridge, "detect_rvc",
                        lambda: {"ok": True})
    monkeypatch.setattr(ne.rvc_bridge, "list_models",
                        lambda: [_FakeModel("zara")])
    monkeypatch.setattr(ne.rvc_bridge, "convert", _convert)
    return calls


@pytest.fixture
def rvc_down(monkeypatch):
    """RVC absent — the tiers below it must carry the accent instead."""
    import nomorals.voice.neural_emotion as ne

    monkeypatch.setattr(ne.rvc_bridge, "detect_rvc",
                        lambda: {"ok": False,
                                 "detail": "rvc not installed"})
    monkeypatch.setattr(ne.rvc_bridge, "list_models", lambda: [])


# ── the hint itself ──────────────────────────────────────────────────────────

class TestAccentHint:
    def test_hint_known_accents(self):
        # function-level import: the helper is new in this lane, so a
        # module-level import would break collection on the base
        from nomorals.voice.accent import accent_language_hint

        assert accent_language_hint("british") == "en-GB"
        assert accent_language_hint("nigerian") == "en-NG"
        assert accent_language_hint("yoruba") == "yo"
        assert accent_language_hint("french") == "fr-FR"
        assert accent_language_hint("pidgin") == "en-NG"

    def test_hint_unknown_is_empty(self):
        from nomorals.voice.accent import accent_language_hint

        assert accent_language_hint("klingon") == ""
        assert accent_language_hint("") == ""

    def test_hint_normalizes_free_text(self):
        from nomorals.voice.accent import accent_language_hint

        assert accent_language_hint("British Accent") == "en-GB"
        assert accent_language_hint("nigerian pidgin accent") == "en-NG"

    def test_empty_input_normalizes_to_empty(self):
        # regression: "" is a substring of every accent name, so the
        # substring fallback used to normalize empty input to
        # south_african — every accent-less render carried en-ZA
        from nomorals.voice.accent import normalize_accent

        assert normalize_accent("") == ""
        assert normalize_accent("   ") == ""
        assert normalize_accent(None) == ""  # type: ignore[arg-type]


# ── render_emotional: the accent reaches a tier ─────────────────────────────

class TestRenderEmotionalAccent:
    def test_rvc_tier_carries_accent_hint(self, tmp_path, rvc_up):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]")
        # the expressive source render is acted in the accent's
        # language, and RVC converts that render to the target voice
        assert len(fake.perform_calls) == 1
        src = fake.perform_calls[0]
        assert src["mood"] == "angry"
        assert src["language"] == "en-GB"
        assert len(rvc_up) == 1
        assert rvc_up[0]["src"] == src["path"]
        assert rvc_up[0]["model"] == "zara"
        assert out["tier"] == "rvc"
        assert out["accent"] == "british"
        assert out["path"] == rvc_up[0]["path"]

    def test_native_tier_carries_accent_hint(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]")
        assert len(fake.perform_calls) == 1
        assert fake.perform_calls[0]["language"] == "en-GB"
        assert out["tier"] == "native"
        assert out["accent"] == "british"

    def test_dsp_tier_carries_accent_hint(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]",
            tiers=("dsp",))
        assert len(fake.speak_calls) == 1
        assert fake.speak_calls[0]["language"] == "en-GB"
        assert out["tier"] == "dsp"
        assert out["accent"] == "british"
        # speak() reports the hint was honored natively here
        assert out["native_language"] is True

    def test_dsp_tier_reports_dropped_hint(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path, honor_language=False)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]",
            tiers=("dsp",))
        assert fake.speak_calls[0]["language"] == "en-GB"
        assert out["accent"] == "british"
        # the backend dropped the hint — the result says so instead
        # of claiming a native accent
        assert out["native_language"] is False

    def test_no_accent_means_no_hint_and_no_key(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("hello world", fake,
                               direction="[said angrily]")
        assert fake.perform_calls[0]["language"] == ""
        assert "accent" not in out
        # the flat native result keeps its old keys exactly
        assert set(out) == {"ok", "path", "tier", "backend"}

    def test_unknown_accent_fails_open(self, tmp_path, rvc_down):
        assert parse_direction(
            "[said angrily in Klingon accent]").accent == ""
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake,
            direction="[said angrily in Klingon accent]")
        assert fake.perform_calls[0]["language"] == ""
        assert fake.perform_calls[0]["mood"] == "angry"
        assert "accent" not in out

    def test_no_accent_works_on_old_call_shape(self, tmp_path, rvc_down):
        # the hint only rides the call when one exists: an engine
        # implementing the old perform()/speak() signature (no
        # language kwarg) still renders an accent-less line exactly
        # as before
        class _StrictTTS:
            def perform(self, text, voice_name=None, mood="neutral",
                        intensity=3):
                return {"path": fake_path, "backend": "strict"}

            def speak(self, text, voice_name=None):
                return {"path": fake_path, "backend": "strict"}

        fake_path = str(Path(tmp_path) / "strict.wav")
        with wave.open(fake_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(array("h", [1, 2, 3, 4]).tobytes())
        out = render_emotional("hello world", _StrictTTS(),
                               direction="[said angrily]")
        assert out["ok"] is True
        assert out["tier"] == "native"
        assert "accent" not in out

    def test_untagged_line_with_accent_renders_flat(self, tmp_path,
                                                    rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake,
            direction="[said angrily in British accent]")
        assert len(fake.perform_calls) == 1
        assert fake.perform_calls[0]["text"] == "hello world"
        assert fake.perform_calls[0]["language"] == "en-GB"
        assert "directed" not in out
        assert out["path"] == fake.perform_calls[0]["path"]
        assert out["accent"] == "british"

    def test_per_segment_accent_is_per_segment(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        line = ("[said angrily in British accent] hello there "
                "[whispers fearfully] good night")
        out = render_emotional(line, fake, direction="[said angrily]")
        assert out["directed"] == "per-segment"
        assert out["segments"] == 2
        # the accent rides only the segment whose cue names it
        assert [(c["text"], c["mood"], c["language"])
                for c in fake.perform_calls] == [
            ("hello there", "angry", "en-GB"),
            ("good night", "fearful", "")]
        assert out["track"]["cues"][0]["accent"] == "british"
        assert out["track"]["cues"][1]["accent"] == ""

    def test_whole_line_accent_falls_back_to_untagged_segments(
            self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello [said angrily] world", fake,
            direction=Direction(emotion="sad", accent="nigerian"))
        assert out["directed"] == "per-segment"
        # both segments inherit the whole-line accent (the angry cue
        # names no accent of its own, so the fallback applies)
        assert [c["language"] for c in fake.perform_calls] == [
            "en-NG", "en-NG"]
        assert out["accent"] == "nigerian"


# ── speak_rich: direction + accent combine ───────────────────────────────────

class TestSpeakRichCombines:
    def _rich(self, fake, text, **kw):
        # the real speak_rich, unbound, over a fake engine — no
        # backend loads, no post-processing (master_audio off)
        return UniversalTTS.speak_rich(fake, text, master_audio=False,
                                       **kw)

    def test_direction_plus_accent_applies_both(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = self._rich(fake, "hello world",
                         direction="[said angrily]", accent="nigerian")
        assert len(fake.perform_calls) == 1
        call = fake.perform_calls[0]
        assert call["mood"] == "angry"
        assert call["language"] == "en-NG"
        assert out["accent"] == "nigerian"

    def test_explicit_accent_wins_over_tag_accent(self, tmp_path,
                                                  rvc_down):
        fake = _FakeTTS(tmp_path)
        out = self._rich(
            fake, "hello world",
            direction="[said angrily in British accent]",
            accent="french")
        assert fake.perform_calls[0]["language"] == "fr-FR"
        assert out["accent"] == "french"

    def test_direction_tag_accent_without_param(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = self._rich(
            fake, "hello world",
            direction="[said angrily in British accent]")
        assert fake.perform_calls[0]["language"] == "en-GB"
        assert out["accent"] == "british"

    def test_direction_only_unchanged(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = self._rich(fake, "hello world", direction="[said angrily]")
        assert fake.perform_calls[0]["language"] == ""
        assert fake.perform_calls[0]["mood"] == "angry"
        assert "accent" not in out

    def test_accent_only_keeps_convert_accent_path(self, tmp_path,
                                                   rvc_down):
        fake = _FakeTTS(tmp_path)
        out = self._rich(fake, "hello there", accent="nigerian")
        # no direction, so the accent-first render still runs; the
        # tier is accent-only, +prosody when the nudge lands
        assert out["tier"] in ("accent-only", "accent-only+prosody")
        assert out["accent"] == "nigerian"
        assert fake.speak_calls[-1]["language"] == "en-NG"

    def test_unknown_accent_param_fails_open(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = self._rich(fake, "hello world",
                         direction="[said angrily]", accent="klingon")
        assert fake.perform_calls[0]["language"] == ""
        assert fake.perform_calls[0]["mood"] == "angry"
        assert "accent" not in out


# ── voice_rich tool: both arguments reach speak_rich ─────────────────────────

class TestVoiceRichTool:
    def _voice_rich_fn(self):
        # voice_rich is a closure inside tools.audio.register(), not a
        # module function — capture it with a stub registry
        from nomorals.tools import audio as audio_tools

        class _CaptureRegistry:
            def __init__(self):
                self.context = None
                self.tools: dict = {}

            def register(self, name, **kw):
                def deco(fn):
                    self.tools[name] = fn
                    return fn

                return deco

        reg = _CaptureRegistry()
        audio_tools.register(reg)
        return reg.tools["voice_rich"]

    def test_tool_passes_direction_and_accent(self, monkeypatch):
        voice_rich = self._voice_rich_fn()
        seen: dict = {}

        class _Engine:
            def __init__(self, **kw):
                pass

            def speak_rich(self, text, voice_name=None, out_path="",
                           direction="", accent="", ambient="",
                           room=""):
                seen.update({"text": text, "direction": direction,
                             "accent": accent})
                return {"ok": True, "path": "/tmp/rich.wav",
                        "tier": "native"}

        monkeypatch.setattr("nomorals.voice.tts.UniversalTTS", _Engine)
        out = voice_rich("hello world", direction="[said angrily]",
                         accent="nigerian")
        assert out["ok"] is True
        assert seen["direction"] == "[said angrily]"
        assert seen["accent"] == "nigerian"

    def test_tool_needs_text(self):
        from nomorals.tools.registry import ToolError

        voice_rich = self._voice_rich_fn()
        with pytest.raises(ToolError):
            voice_rich("   ")
