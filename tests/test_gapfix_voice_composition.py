"""Failing-first tests for gapfix R3-52 — voice composition organs.

Two wirings, both failing before the fix:
  (a) audiobook `_default_tts_fn` must render a chapter through
      `synthesize_long` (sentence-boundary chunks, one voice anchor,
      degenerate-chunk regenerate) instead of its naive 400-word loop;
  (b) `UniversalTTS.speak()` must branch multi-speaker `[S1]`/`[S2]`
      scripts to `render_dialogue`, while plain text and Dia-native
      scripts keep the old path.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import wave
from array import array
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.voice.longform import LongFormSynthesizer  # noqa: E402
from nomorals.voice.tts import UniversalTTS  # noqa: E402

SR = 24000


def _wav(path, secs=0.4, freq=440.0, amp=12000, sr=SR):
    n = int(sr * secs)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        for i in range(n):
            v = int(amp * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<h", v))
    return path


def _rms(path):
    with wave.open(path, "rb") as w:
        raw = w.readframes(w.getnframes())
    samples = array("h", raw)
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples))


# ── fakes ──────────────────────────────────────────────────────────────────


class _FakeVoices:
    def __init__(self):
        self.uploads = []

    def upload_voice(self, name, ref, language="en"):
        self.uploads.append((name, ref, language))
        return {"ok": True, "name": name}


class _FakePerformTTS:
    """Stands in for UniversalTTS in the audiobook default path.

    Deliberately has NO `speak` method: if `_default_tts_fn` still
    runs its old 400-word `speak` loop, it fails loudly instead of
    silently passing through a `speak` shim.
    """

    def __init__(self, backend="auto", voices_dir="", audience="private"):
        self.calls = []  # (text, voice_name, mood)
        self.voices = _FakeVoices()
        self._n = 0

    def perform(self, text, voice_name=None, mood="neutral"):
        self.calls.append((text, voice_name, mood))
        self._n += 1
        p = os.path.join(tempfile.mkdtemp(), f"chunk{self._n}.wav")
        _wav(p, secs=0.4)
        return {"path": p}


class _DegenerateFakeTTS(_FakePerformTTS):
    """First render of the SENTINEL chunk is silence; the retry is healthy."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._silent_done = set()

    def perform(self, text, voice_name=None, mood="neutral"):
        if "SENTINEL" in text and text not in self._silent_done:
            self._silent_done.add(text)
            self.calls.append((text, voice_name, mood))
            self._n += 1
            p = os.path.join(tempfile.mkdtemp(), f"chunk{self._n}.wav")
            _wav(p, secs=0.4, amp=0)
            return {"path": p}
        return super().perform(text, voice_name, mood)


class _FakeBackend:
    """Records synthesize() calls; returns 0.1 s of silence (floats)."""

    sample_rate = SR

    def __init__(self, name, native=False):
        self.name = name
        self.supports_native_tags = native
        self.calls = []  # (text, voice, instruct)

    def synthesize(self, text, voice, *, instruct=""):
        self.calls.append((text, voice, instruct))
        return [0.0] * 2400


class _NoCache:
    def get(self, key):
        return None

    def put(self, key, samples, sample_rate):
        return None


def _engine(backend, **kw):
    tmp = tempfile.mkdtemp()
    eng = UniversalTTS(backend="bark", voices_dir=tmp, **kw)
    eng._impl = backend
    eng._loaded = True
    eng._seg_cache = _NoCache()
    return eng


CHAPTER = (
    "The old lighthouse stood against the grey Atlantic wind for a hundred years. "
    "Mara climbed the iron stairs each morning before the town woke up. "
    "She polished the great lens until it caught the first pale light. "
    "Below her the fishing boats slipped out past the harbour wall. "
    "Gulls circled the tower and cried out over the cold dark water. "
    "She kept the log in her father's handwriting, page after careful page. "
    "Every storm that rolled in from the west tested the old stone walls. "
    "She trimmed the wick and checked the oil before the sun went down. "
    "The light swept the black sea in slow and patient circles all night. "
    "When dawn broke she wrote one more line and closed the heavy book."
)

DIALOGUE = "[S1: Zara] Hey there! [S2: Kilo] Yeah, it's fire."


# ── (a) audiobook → synthesize_long ────────────────────────────────────────


def test_audiobook_chapter_renders_through_synthesize_long():
    from nomorals.audio.audiobook import _default_tts_fn
    from nomorals.voice import tts as tts_mod

    tmp = tempfile.mkdtemp()
    ref = _wav(os.path.join(tmp, "voice.wav"))
    fake = _FakePerformTTS()
    with mock.patch.object(tts_mod, "UniversalTTS", lambda **kw: fake):
        out = _default_tts_fn(CHAPTER, ref, lang="en")
    assert out and os.path.exists(out), "chapter must render to a wav"
    assert _rms(out) > 1000, "stitched chapter must not be silence"
    assert len(fake.calls) >= 2, \
        f"chapter must chunk through synthesize_long, got {len(fake.calls)} call(s)"
    # one voice anchor for every chunk
    voices = {c[1] for c in fake.calls}
    assert voices == {"audiobook-narrator"}, f"voice drifted: {voices}"
    # sentence-boundary chunks: re-joining reproduces the chapter, and
    # every chunk ends on sentence punctuation (never mid-sentence)
    joined = " ".join(c[0] for c in fake.calls)
    assert joined == " ".join(CHAPTER.split()), "chunks must cover the text"
    for text, _, _ in fake.calls:
        assert text.rstrip()[-1] in ".!?…", \
            f"chunk split mid-sentence: {text[-40:]!r}"


def test_audiobook_degenerate_chunk_regenerates_never_ships_silence():
    from nomorals.audio.audiobook import _default_tts_fn
    from nomorals.voice import tts as tts_mod

    chapter = (
        "The harbour bell rang out across the still morning water. "
        "SENTINEL sailors gathered on the quay to watch the tide turn. "
        "Ropes creaked and gulls circled above the tall masts. "
        "The captain raised his hand and the whole crew fell silent. "
        "Somewhere far below deck a kettle sang its thin bright note. "
        "By noon the little fleet was gone beyond the harbour wall. "
        "The lighthouse keeper watched them go from his high cold tower. "
        "He trimmed the great lamp and polished the glass with care. "
        "Night fell softly over the harbour and the stars came out. "
        "The bell rang once more before the town went dark and quiet."
    )
    tmp = tempfile.mkdtemp()
    ref = _wav(os.path.join(tmp, "voice.wav"))
    fake = _DegenerateFakeTTS()
    with mock.patch.object(tts_mod, "UniversalTTS", lambda **kw: fake):
        out = _default_tts_fn(chapter, ref, lang="en")
    assert out and os.path.exists(out), "chapter must still render"
    assert _rms(out) > 1000, "degenerate chunk must not ship as silence"
    sentinel_calls = [c for c in fake.calls if "SENTINEL" in c[0]]
    assert len(sentinel_calls) == 2, \
        f"degenerate chunk must regenerate exactly once, got {len(sentinel_calls)}"


def test_audiobook_short_chapter_single_chunk():
    from nomorals.audio.audiobook import _default_tts_fn
    from nomorals.voice import tts as tts_mod

    tmp = tempfile.mkdtemp()
    ref = _wav(os.path.join(tmp, "voice.wav"))
    fake = _FakePerformTTS()
    with mock.patch.object(tts_mod, "UniversalTTS", lambda **kw: fake):
        out = _default_tts_fn("Hello world.", ref, lang="en")
    assert out and os.path.exists(out)
    assert len(fake.calls) == 1, "short text is one chunk"
    assert fake.calls[0][1] == "audiobook-narrator"


def test_audiobook_tts_failure_still_refuses():
    from nomorals.audio.audiobook import _default_tts_fn
    from nomorals.voice import tts as tts_mod

    class _Boom(_FakePerformTTS):
        def perform(self, text, voice_name=None, mood="neutral"):
            raise RuntimeError("no backend")

    tmp = tempfile.mkdtemp()
    ref = _wav(os.path.join(tmp, "voice.wav"))
    with mock.patch.object(tts_mod, "UniversalTTS", lambda **kw: _Boom()):
        assert _default_tts_fn("Hello world.", ref, lang="en") is None


def test_synthesize_long_keeps_one_voice_and_adapts_hints():
    fake = _FakePerformTTS()
    synth = LongFormSynthesizer(fake, "narrator-voice", "neutral")
    res = synth.synthesize(CHAPTER, os.path.join(tempfile.mkdtemp(), "o.wav"))
    assert res["ok"] is True
    assert {c[1] for c in fake.calls} == {"narrator-voice"}
    # the rolling context is live: the energy hint moved off its 1.0 seed
    # toward the measured chunk energy (loud sine → clamped 1.3 target)
    assert synth._energy_hint != 1.0, "hints must adapt across chunks"


# ── (b) speak() dialogue branch ────────────────────────────────────────────


def test_speak_dialogue_splits_turns_per_speaker():
    backend = _FakeBackend("xtts")
    eng = _engine(backend, audience="private")
    res = eng.speak(DIALOGUE)
    assert res.get("ok") is True, f"dialogue render failed: {res}"
    assert res.get("dialogue") is True, "dialogue render must say so"
    assert res.get("turns") == 2, f"expected 2 turns, got {res.get('turns')}"
    assert len(res.get("voices", [])) == 2
    assert res.get("path") and os.path.exists(res["path"])
    assert len(backend.calls) == 2, \
        f"each turn must synthesize separately, got {len(backend.calls)}"
    assert "Hey there" in backend.calls[0][0]
    assert "[S1" not in backend.calls[0][0], "markers must not be spoken"
    assert "Yeah" in backend.calls[1][0]
    assert "[S2" not in backend.calls[1][0]


def test_speak_dialogue_uses_voice_map():
    backend = _FakeBackend("xtts")
    eng = _engine(backend, audience="private")
    res = eng.speak(DIALOGUE, voice_map={"1": "zara", "2": "kilo"})
    assert res.get("ok") is True
    assert res.get("voices") == ["zara", "kilo"], \
        f"voice map must pick per-speaker voices, got {res.get('voices')}"


def test_speak_dia_passes_script_through_unsplit():
    backend = _FakeBackend("dia", native=True)
    eng = _engine(backend, audience="private")
    res = eng.speak(DIALOGUE)
    assert res.get("path") and os.path.exists(res["path"])
    assert not res.get("dialogue"), "Dia handles [S1] natively — no split"
    assert len(backend.calls) == 1, \
        f"Dia script must reach the backend whole, got {len(backend.calls)}"
    assert "Hey there" in backend.calls[0][0]
    assert "Yeah" in backend.calls[0][0]


def test_speak_plain_text_unchanged():
    backend = _FakeBackend("xtts")
    eng = _engine(backend, audience="private")
    res = eng.speak("Just a plain sentence, nothing scripted.")
    assert res.get("path") and os.path.exists(res["path"])
    assert not res.get("dialogue")
    assert "segments" in res, "plain text keeps the old speak shape"
    assert len(backend.calls) == 1
    assert "plain sentence" in backend.calls[0][0]


def test_speak_lone_marker_unchanged():
    backend = _FakeBackend("xtts")
    eng = _engine(backend, audience="private")
    res = eng.speak("[S1] only one turn here, not a dialogue.")
    assert res.get("path") and os.path.exists(res["path"])
    assert not res.get("dialogue"), "a lone marker is not a dialogue"
    assert len(backend.calls) == 1


def test_speak_dialogue_honors_out_path():
    backend = _FakeBackend("xtts")
    eng = _engine(backend, audience="private")
    dest = os.path.join(tempfile.mkdtemp(), "dlg.wav")
    res = eng.speak(DIALOGUE, out_path=dest)
    assert res.get("ok") is True
    assert res.get("path") == dest, "dialogue render must honor out_path"
    assert os.path.exists(dest)


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
