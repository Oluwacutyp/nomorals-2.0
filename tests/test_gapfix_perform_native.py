"""perform() reports native_opts, so the RVC/native accent tiers can
say whether the backend honored the accent hint.

Gap: the accent ship (2020eed1) realizes accent through the backend's
language hint, but only the DSP tier could report whether the hint
was honored — speak() returns ``native_opts`` while perform() did
not, so the RVC/native tiers carried the accent key with no honesty
field. This lane adds ``native_opts`` to perform()'s return
(unconditional, mirroring speak()) and has both tiers report
``native_language`` the way the DSP tier does — including the
per-segment stitch, which carries it at the top when every accented
segment agrees.
"""

import math
import os
import tempfile
import unittest
from array import array
from pathlib import Path
from unittest import mock

import pytest

from nomorals.voice.neural_emotion import render_emotional


class _FakeModel:
    def __init__(self, name):
        self.name = name


class _FakeTTS:
    """perform()/speak() -> tiny wav; perform reports native_opts.

    Mirrors the voice-accent fake, plus the ``native_opts`` a real
    perform() reports after this lane. ``honor_language=False``
    simulates a backend that drops the hint: the call still
    succeeds, the report says so.
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
        import wave

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
        return {"path": path, "backend": "fake",
                "native_opts": ["language"] if language
                and self.honor_language else []}

    def speak(self, text, voice_name=None, mood="", language="", **kw):
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
    import wave

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


# ── perform() itself ─────────────────────────────────────────────────────

class TestPerformNativeOpts(unittest.TestCase):
    def test_perform_native_opts_honored(self):
        import numpy as np

        from nomorals.voice import tts as tts_mod

        class StubBackend:
            name = "cosyvoice"
            sample_rate = 22050
            _supports_opts = ("language", "exaggeration")

            def synthesize(self, text, voice, *, instruct="", **kw):
                return np.zeros(2205, dtype=np.float32)

        stub = StubBackend()
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_mod.UniversalTTS(backend="cosyvoice",
                                          voices_dir=tmp)
            with mock.patch.object(tts_mod, "_BACKENDS",
                                   {"cosyvoice": lambda: stub}), mock.patch.object(tts_mod, "_spec",
                                   lambda name: True):
                out = engine.perform(
                    "haha great news!", mood="happy", intensity=5,
                    seed=9, language="en-GB", exaggeration=0.7,
                    out_path=os.path.join(tmp, "o.wav"))
            # sorted, mirroring speak(): both honored opts, language
            # first alphabetically only because it sorts that way
            self.assertEqual(out["native_opts"],
                             ["exaggeration", "language"])
            self.assertTrue(os.path.exists(out["path"]))

    def test_perform_native_opts_dropped(self):
        import numpy as np

        from nomorals.voice import tts as tts_mod

        class StubBackend:
            name = "cosyvoice"
            sample_rate = 22050
            _supports_opts = ("speed",)

            def synthesize(self, text, voice, *, instruct="", **kw):
                return np.zeros(2205, dtype=np.float32)

        stub = StubBackend()
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_mod.UniversalTTS(backend="cosyvoice",
                                          voices_dir=tmp)
            with mock.patch.object(tts_mod, "_BACKENDS",
                                   {"cosyvoice": lambda: stub}), mock.patch.object(tts_mod, "_spec",
                                   lambda name: True):
                out = engine.perform(
                    "haha great news!", mood="happy", intensity=5,
                    seed=9, language="en-GB",
                    out_path=os.path.join(tmp, "o.wav"))
            # the backend took no per-call opts: the hint was dropped
            self.assertEqual(out["native_opts"], [])

    def test_perform_native_opts_empty_without_hint(self):
        import numpy as np

        from nomorals.voice import tts as tts_mod

        class StubBackend:
            name = "cosyvoice"
            sample_rate = 22050
            _supports_opts = ("language",)

            def synthesize(self, text, voice, *, instruct="", **kw):
                return np.zeros(2205, dtype=np.float32)

        stub = StubBackend()
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_mod.UniversalTTS(backend="cosyvoice",
                                          voices_dir=tmp)
            with mock.patch.object(tts_mod, "_BACKENDS",
                                   {"cosyvoice": lambda: stub}), mock.patch.object(tts_mod, "_spec",
                                   lambda name: True):
                out = engine.perform(
                    "haha great news!", mood="happy", intensity=5,
                    seed=9, out_path=os.path.join(tmp, "o.wav"))
            # no hint requested, no opts taken — the key still rides,
            # exactly like speak()
            self.assertEqual(out["native_opts"], [])

    def test_perform_native_opts_mirrors_speak(self):
        # same backend, same hint: perform() and speak() must agree
        # on which opts the backend took natively
        import numpy as np

        from nomorals.voice import tts as tts_mod

        class StubBackend:
            name = "cosyvoice"
            sample_rate = 22050
            _supports_opts = ("language",)

            def synthesize(self, text, voice, *, instruct="", **kw):
                return np.zeros(2205, dtype=np.float32)

        stub = StubBackend()
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_mod.UniversalTTS(backend="cosyvoice",
                                          voices_dir=tmp)
            with mock.patch.object(tts_mod, "_BACKENDS",
                                   {"cosyvoice": lambda: stub}), mock.patch.object(tts_mod, "_spec",
                                   lambda name: True):
                performed = engine.perform(
                    "haha great news!", mood="happy", intensity=5,
                    seed=9, language="en-GB",
                    out_path=os.path.join(tmp, "p.wav"))
                spoken = engine.speak(
                    "haha great news!", language="en-GB",
                    out_path=os.path.join(tmp, "s.wav"))
            self.assertEqual(performed["native_opts"], ["language"])
            self.assertEqual(spoken["native_opts"], ["language"])
            self.assertEqual(performed["native_opts"],
                             spoken["native_opts"])

    def test_cached_speak_keeps_native_opts(self):
        # regression: the segment-cache-hit path in _speak_impl used
        # to return without native_opts, so a repeated line could not
        # say whether its hint was honored — the second speak of the
        # same text must report the same fields as the first
        import numpy as np

        from nomorals.voice import tts as tts_mod

        class StubBackend:
            name = "cosyvoice"
            sample_rate = 22050
            _supports_opts = ("language",)

            def synthesize(self, text, voice, *, instruct="", **kw):
                return np.zeros(2205, dtype=np.float32)

        stub = StubBackend()
        with tempfile.TemporaryDirectory() as tmp:
            engine = tts_mod.UniversalTTS(backend="cosyvoice",
                                          voices_dir=tmp)
            with mock.patch.object(tts_mod, "_BACKENDS",
                                   {"cosyvoice": lambda: stub}), mock.patch.object(tts_mod, "_spec",
                                   lambda name: True):
                first = engine.speak(
                    "haha great news!", language="en-GB",
                    out_path=os.path.join(tmp, "s1.wav"))
                second = engine.speak(
                    "haha great news!", language="en-GB",
                    out_path=os.path.join(tmp, "s2.wav"))
            self.assertEqual(first["native_opts"], ["language"])
            self.assertTrue(second.get("cached"))
            self.assertEqual(second["native_opts"], ["language"])


# ── the tiers report it ──────────────────────────────────────────────────

class TestTierNativeLanguage:
    def test_rvc_tier_reports_honored_hint(self, tmp_path, rvc_up):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]")
        assert out["tier"] == "rvc"
        assert out["accent"] == "british"
        assert out["native_language"] is True

    def test_rvc_tier_reports_dropped_hint(self, tmp_path, rvc_up):
        fake = _FakeTTS(tmp_path, honor_language=False)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]")
        assert out["tier"] == "rvc"
        assert out["accent"] == "british"
        # the backend dropped the hint — the result says so instead
        # of claiming a native accent
        assert out["native_language"] is False

    def test_native_tier_reports_honored_hint(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]")
        assert out["tier"] == "native"
        assert out["accent"] == "british"
        assert out["native_language"] is True

    def test_native_tier_reports_dropped_hint(self, tmp_path, rvc_down):
        fake = _FakeTTS(tmp_path, honor_language=False)
        out = render_emotional(
            "hello world", fake, voice_name="zara",
            direction="[said angrily in British accent]")
        assert out["tier"] == "native"
        assert out["accent"] == "british"
        assert out["native_language"] is False

    def test_no_accent_means_no_native_language_key(self, tmp_path,
                                                    rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("hello world", fake,
                               direction="[said angrily]")
        assert fake.perform_calls[0]["language"] == ""
        assert "accent" not in out
        # the flat native result keeps its old keys exactly — the
        # honesty field rides only with an accent, like the DSP tier
        assert set(out) == {"ok", "path", "tier", "backend"}

    def test_rvc_tier_no_accent_keeps_old_keys(self, tmp_path, rvc_up):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("hello world", fake, voice_name="zara",
                               direction="[said angrily]")
        assert out["tier"] == "rvc"
        assert "accent" not in out
        assert "native_language" not in out
        assert set(out) == {"ok", "path", "tier", "model", "emotion",
                            "delivery"}

    def test_perform_without_native_opts_reports_dropped(self, tmp_path,
                                                         rvc_down):
        # an engine predating this lane returns no native_opts at
        # all: the tier reads .get() and reports dropped, never crashes
        import wave

        class _OldTTS:
            def perform(self, text, voice_name=None, mood="neutral",
                        intensity=3, language="", **kw):
                path = str(tmp_path / "old.wav")
                with wave.open(path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(8000)
                    w.writeframes(array("h", [1, 2, 3, 4]).tobytes())
                return {"path": path, "backend": "old"}

            def speak(self, text, voice_name=None, **kw):
                return self.perform(text)

        out = render_emotional(
            "hello world", _OldTTS(),
            direction="[said angrily in British accent]")
        # the old engine's native tier drops the hint (it never saw
        # native_opts); the render still lands
        assert out["tier"] == "native"
        assert out["accent"] == "british"
        assert out.get("native_language") is False

    def test_per_segment_stitch_carries_agreement(self, tmp_path,
                                                  rvc_down):
        # a directed line with tags takes the per-segment path; both
        # segments inherit the whole-line accent, both agree the hint
        # was honored, so the stitched line says so at the top
        from nomorals.voice.nl_director import Direction

        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "hello [said angrily] world", fake,
            direction=Direction(emotion="sad", accent="nigerian"))
        assert out["directed"] == "per-segment"
        assert out["accent"] == "nigerian"
        assert [c["language"] for c in fake.perform_calls] == [
            "en-NG", "en-NG"]
        assert out["native_language"] is True

    def test_per_segment_stitch_carries_drop(self, tmp_path, rvc_down):
        from nomorals.voice.nl_director import Direction

        fake = _FakeTTS(tmp_path, honor_language=False)
        out = render_emotional(
            "hello [said angrily] world", fake,
            direction=Direction(emotion="sad", accent="nigerian"))
        assert out["directed"] == "per-segment"
        assert out["accent"] == "nigerian"
        assert out["native_language"] is False

    def test_per_segment_stitch_silent_on_disagreement(self, tmp_path,
                                                       rvc_down):
        # segments that disagree stay silent rather than claiming
        # one answer — exercised with a stateful fake that honors
        # the hint on the first segment's render and drops it on
        # the second

        from nomorals.voice.nl_director import Direction

        class _SplitTTS(_FakeTTS):
            def perform(self, text, voice_name=None, mood="neutral",
                        intensity=3, language="", **kw):
                honored = len(self.perform_calls) == 0
                res = super().perform(text, voice_name, mood,
                                      intensity, language, **kw)
                res["native_opts"] = (["language"] if honored
                                      and language else [])
                return res

        fake = _SplitTTS(tmp_path)
        out = render_emotional(
            "hello [said angrily] world", fake,
            direction=Direction(emotion="sad", accent="nigerian"))
        assert out["directed"] == "per-segment"
        assert out["accent"] == "nigerian"
        assert "native_language" not in out

    def test_per_segment_stitch_silent_without_accent(self, tmp_path,
                                                      rvc_down):
        fake = _FakeTTS(tmp_path)
        out = render_emotional(
            "[said angrily] hello there [whispers fearfully] good night",
            fake, direction="[said angrily]")
        assert out["directed"] == "per-segment"
        assert "accent" not in out
        assert "native_language" not in out


if __name__ == "__main__":
    unittest.main()
