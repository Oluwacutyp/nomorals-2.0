"""§1.4 TTS-side timing feed: synthesis-side word timings, labelled.

``UniversalTTS.speak_aligned`` is the feed: backends that can time
their own words (kokoro's token timestamps) return them labelled
``measured``; every other backend *says so* — ``timing_source:
"unavailable"`` with the reason named — instead of letting a caller
believe timings exist. Estimates are never manufactured here.

All backends are fakes; no models load.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice.tts import KokoroBackend, PiperBackend, UniversalTTS


class _AlignBackend:
    """Declares token timestamps and produces usable words."""

    name = "fakealign"
    sample_rate = 16000
    supports_native_tags = False
    supports_streaming = False
    word_alignment = "token_timestamps"

    def synthesize(self, text, voice, *, instruct=""):
        return [0.0] * 160

    def synthesize_aligned(self, text, voice, *, instruct=""):
        words = [{"text": tok, "start": round(0.3 * i, 4),
                  "end": round(0.3 * i + 0.22, 4)}
                 for i, tok in enumerate(text.split())]
        return [0.0] * 1600, words


class _PlainBackend:
    name = "fakeplain"
    sample_rate = 16000
    supports_native_tags = False
    supports_streaming = False

    def synthesize(self, text, voice, *, instruct=""):
        return [0.0] * 160


class _BrokenAlignBackend(_AlignBackend):
    """Claims alignment, produces garbage — must not be trusted."""

    name = "fakebroken"

    def synthesize_aligned(self, text, voice, *, instruct=""):
        return [0.0] * 160, [{"text": "x", "start": "soon", "end": None}]


def _engine_with(backend) -> UniversalTTS:
    eng = UniversalTTS(backend="bark")
    eng._impl = backend
    eng._loaded = True
    return eng


def test_aligned_backend_returns_measured_words(tmp_path):
    out = _engine_with(_AlignBackend()).speak_aligned(
        "hello brave world", out_path=str(tmp_path / "a.wav"))
    assert out["timing_source"] == "measured"
    assert out["alignment"] == "token_timestamps"
    assert [w["text"] for w in out["words"]] == \
        ["hello", "brave", "world"]
    assert out["words"][1]["start"] == 0.3
    assert os.path.isfile(out["path"]) and out["bytes"] > 0


def test_plain_backend_says_unavailable_and_still_speaks(tmp_path):
    out = _engine_with(_PlainBackend()).speak_aligned(
        "hello world", out_path=str(tmp_path / "b.wav"))
    assert out["timing_source"] == "unavailable"
    assert out["alignment"] == "none"
    assert out["words"] == []
    assert "fakeplain" in out["alignment_reason"]
    assert os.path.isfile(out["path"])


def test_broken_alignment_claim_is_reported_not_trusted(tmp_path):
    out = _engine_with(_BrokenAlignBackend()).speak_aligned(
        "hello world", out_path=str(tmp_path / "c.wav"))
    assert out["timing_source"] == "unavailable"
    assert out["words"] == []
    assert "fakebroken" in out["alignment_reason"]
    assert os.path.isfile(out["path"])


def test_speed_adjustment_reports_unavailable(tmp_path):
    out = _engine_with(_AlignBackend()).speak_aligned(
        "hello world", out_path=str(tmp_path / "d.wav"), speed=1.5)
    assert out["timing_source"] == "unavailable"
    assert "speed" in out["alignment_reason"]
    assert out["words"] == []


def test_to_int16_treats_int16_arrays_as_pcm_not_amplitudes():
    """Regression (found by the §1.4 corpus's warm-cache run): the
    SegmentCache hands back stdlib array('h') int16 samples; the
    old _to_int16 read them as float amplitudes and clipped every
    nonzero sample to ±32767 — a square wave on every cache hit."""
    import struct
    from array import array

    from nomorals.voice.tts import _to_int16

    pcm = _to_int16(array("h", [0, 8192, -8192, 16384]), 16000)
    vals = struct.unpack("<4h", pcm)
    assert vals[0] == 0
    assert abs(vals[1] - 8192) <= 1
    assert abs(vals[2] + 8192) <= 1
    assert abs(vals[3] - 16384) <= 1
    # float input keeps its meaning
    pcm_f = _to_int16([0.0, 0.25, -0.25, 0.5], 16000)
    vals_f = struct.unpack("<4h", pcm_f)
    assert abs(vals_f[1] - 8192) <= 1 and abs(vals_f[3] - 16384) <= 1


def test_backend_info_reports_word_alignment_comparably():
    eng = UniversalTTS(backend="bark")
    assert eng.backend_info("kokoro")["word_alignment"] == \
        "token_timestamps"
    assert eng.backend_info("piper")["word_alignment"] == \
        "phoneme_alignments"
    infos = {i["name"]: i["word_alignment"]
             for i in eng.list_backend_info()}
    assert infos["kokoro"] == "token_timestamps"
    assert infos["piper"] == "phoneme_alignments"
    assert set(infos.values()) <= \
        {"token_timestamps", "phoneme_alignments", "none"}


# ---------------------------------------------------------------------------
# the kokoro rung itself: token extraction against a fake pipeline
# ---------------------------------------------------------------------------

class _Tok:
    def __init__(self, text, start_ts, end_ts):
        self.text = text
        self.start_ts = start_ts
        self.end_ts = end_ts


class _Result:
    def __init__(self, audio, tokens):
        self.audio = audio
        self.tokens = tokens


def _kokoro_with_pipeline(results) -> KokoroBackend:
    be = KokoroBackend.__new__(KokoroBackend)
    be.pipeline = lambda text, voice=None: iter(results)
    return be


def test_kokoro_aligned_extracts_token_timestamps():
    # 24000 samples at 24000 Hz = 1.0 s per result chunk
    results = [
        _Result([0.0] * 24000,
                [_Tok("hello", 0.02, 0.40), _Tok("brave", 0.45, 0.90),
                 _Tok("", 0.95, 0.99), _Tok("skip", None, None)]),
        _Result([0.0] * 24000, [_Tok("world", 0.05, 0.50)]),
    ]
    audio, words = _kokoro_with_pipeline(results).synthesize_aligned(
        "hello brave world", None)
    assert len(audio) == 48000
    assert [w["text"] for w in words] == ["hello", "brave", "world"]
    # second chunk's tokens are offset by the first chunk's duration
    assert words[-1]["start"] == 1.05 and words[-1]["end"] == 1.50


def test_kokoro_aligned_without_tokens_yields_no_words():
    results = [_Result([0.0] * 2400, [])]
    audio, words = _kokoro_with_pipeline(results).synthesize_aligned(
        "hello", None)
    assert len(audio) == 2400
    assert words == []


# ---------------------------------------------------------------------------
# the piper rung: current-API synthesis + phoneme-alignment grouping
# ---------------------------------------------------------------------------

class _Chunk:
    """piper1-gpl AudioChunk shape, fake."""

    def __init__(self, n_samples, *, counts=None, entries=None,
                 rate=1000):
        import struct as _s
        self.audio_int16_array = [0] * n_samples
        self.audio_int16_bytes = _s.pack(f"<{n_samples}h",
                                         *([0] * n_samples))
        self.sample_rate = rate
        self.phoneme_id_samples = counts
        self.phoneme_alignments = entries


class _Entry:
    def __init__(self, start_time, end_time):
        self.start_time = start_time
        self.end_time = end_time


class _FakePiperVoice:
    """phonemize: 'ab cd' → a b ' ' c d (one boundary phoneme)."""

    def __init__(self, chunks):
        self._chunks = chunks

    def phonemize(self, text):
        out = []
        for i, tok in enumerate(text.split()):
            if i:
                out.append(" ")
            out.extend(list(tok))
        return [out]

    def synthesize(self, text, syn_config=None, include_alignments=False):
        return iter(self._chunks)


def _piper_with(voice) -> PiperBackend:
    # Constructed by hand with a pre-loaded "default" voice — through
    # the lifecycle manager that governs loaded sessions (Wave 3a),
    # which replaced the backend's old ad-hoc _voices dict.
    from nomorals.voice.lifecycle import (
        LifecyclePolicy,
        ModelLifecycleManager,
    )

    be = PiperBackend.__new__(PiperBackend)
    be._PiperVoice = None
    be._lifecycle = ModelLifecycleManager(LifecyclePolicy())
    be._lifecycle.acquire("default", lambda: voice)
    be.voice_path = "fake.onnx"
    be.sample_rate = 1000
    return be


def test_piper_synthesize_current_api():
    be = _piper_with(_FakePiperVoice([_Chunk(500)]))
    out = be.synthesize("ab cd", None)
    assert len(out) == 500


def test_piper_aligned_groups_phoneme_counts_into_words():
    # rate 1000: counts are milliseconds; phonemes a b ' ' c d
    chunk = _Chunk(1000, counts=[100, 200, 50, 300, 350])
    audio, words = _piper_with(_FakePiperVoice([chunk])) \
        .synthesize_aligned("ab cd", None)
    assert len(audio) == 1000
    assert words == [{"text": "ab", "start": 0.0, "end": 0.3},
                     {"text": "cd", "start": 0.35, "end": 1.0}]


def test_piper_aligned_groups_timestamp_entries_into_words():
    entries = [_Entry(0.0, 0.1), _Entry(0.1, 0.3), _Entry(0.3, 0.35),
               _Entry(0.35, 0.65), _Entry(0.65, 1.0)]
    chunk = _Chunk(1000, entries=entries)
    _audio, words = _piper_with(_FakePiperVoice([chunk])) \
        .synthesize_aligned("ab cd", None)
    assert words == [{"text": "ab", "start": 0.0, "end": 0.3},
                     {"text": "cd", "start": 0.35, "end": 1.0}]


def test_piper_aligned_unpopulated_voice_yields_no_words():
    # the standard voices' shape on current piper-tts: chunks carry
    # neither entries nor counts → no words, never reconstructed
    chunk = _Chunk(1000)
    audio, words = _piper_with(_FakePiperVoice([chunk])) \
        .synthesize_aligned("ab cd", None)
    assert len(audio) == 1000
    assert words == []
