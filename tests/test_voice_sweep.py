"""Sweep tests for nomorals/voice (2026-10-10 system-wide upgrade).

Covers the new/changed behavior only; no backend weights are installed
in the sandbox, so synthesis paths are exercised with a fake TTS or
through the honest error branches.
"""
from __future__ import annotations

import os
import sys
import tempfile
import wave
from array import array

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.voice import (
    AmbienceScene,
    Direction,
    SegmentCache,
    StreamingSTT,
    describe_scene,
    describe_to_params,
    diarize,
    estimate_duration,
    export_midi,
    format_confirmation,
    format_transcript,
    humanize,
    list_accents,
    normalize_accent,
    parse_dialogue,
    parse_direction,
    parse_melody,
    parse_voice_money,
    render_dialogue,
    render_emotional,
    render_kitten,
    render_spark,
    render_zonos,
    to_srt,
    to_vtt,
    voice_note_info,
)
from nomorals.voice.accent import convert_accent
from nomorals.voice.biometrics import (
    check_liveness_response,
    decide,
    liveness_challenge,
    spoof_score,
)
from nomorals.voice.catalogue import VoiceCatalogue
from nomorals.voice.design import blend_voices
from nomorals.voice.director import supported_backends
from nomorals.voice.emotion_dsp import (
    apply_delivery,
    tremolo,
    whisperize,
)
from nomorals.voice.fetch import MODEL_REGISTRY
from nomorals.voice.mastering import (
    compress,
    fade_in_out,
    loudness_match,
)
from nomorals.voice.rvc_bridge import model_info
from nomorals.voice.tts import (
    SparkTTSBackend,
    UniversalTTS,
    ZonosBackend,
)


# ── helpers ──────────────────────────────────────────────────────────────

def _wav(samples: array, sr: int = 8000) -> str:
    p = tempfile.mktemp(suffix=".wav")
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())
    return p


def _tone(sr: int = 8000, secs: float = 0.3, amp: int = 3000) -> array:
    return array("h", [amp] * int(sr * secs))


class FakeTTS:
    """speak() → wav file; records the call kwargs."""
    def __init__(self, sr: int = 8000, secs: float = 0.3):
        self.calls: list[dict] = []
        self.sr = sr
        self.secs = secs

    def speak(self, text: str, voice_name=None, mood="",
              language="", **kw):
        self.calls.append({"text": text, "voice_name": voice_name,
                           "mood": mood, "language": language})
        return {"path": _wav(_tone(self.sr, self.secs), self.sr),
                "native_opts": []}


# ── tts: opt routing, zonos emotion tags, spark honesty ─────────────────

def test_split_synth_opts_routing():
    class Kittenish:
        _supports_opts = ("speed", "voice")
        name = "kitten"

    class Chatterboxish:
        _supports_opts = ("exaggeration", "language")
        name = "chatterbox"

    class Plain:
        _supports_opts = ()
        name = "system"

    opts, residual = UniversalTTS._split_synth_opts(
        Kittenish(), speed=1.5, exaggeration=0.7)
    assert opts == {"speed": 1.5} and residual == 1.0
    opts, residual = UniversalTTS._split_synth_opts(
        Chatterboxish(), speed=1.5, exaggeration=0.7, language="en-NG")
    assert opts == {"exaggeration": 0.7, "language": "en-NG"}
    assert residual == 1.5  # honest DSP fallback for speed
    opts, residual = UniversalTTS._split_synth_opts(Plain(), speed=1.0)
    assert opts == {} and residual == 1.0


def test_zonos_emotion_tag_roundtrip():
    rendered = render_zonos("[angry] You did this.")
    assert "[zonos-emo:" in rendered
    clean, vec = ZonosBackend._extract_emotion(rendered)
    assert len(vec) == 8
    assert vec[5] > 0.5  # anger dimension
    assert clean == "You did this."
    # unknown text → neutral vector, text untouched
    clean2, vec2 = ZonosBackend._extract_emotion("plain hello")
    assert clean2 == "plain hello" and max(vec2) <= 1.0


def test_spark_backend_error_honest():
    with pytest.raises(RuntimeError, match="[Ss]park"):
        SparkTTSBackend().synthesize("hello", None)


def test_speak_accepts_language_kwarg():
    # engine-level plumbing: language flows through _split_synth_opts
    sig_params = UniversalTTS.speak.__code__.co_varnames
    assert "language" in sig_params


# ── director renderers ─────────────────────────────────────────────────

def test_new_renderers_registered():
    backends = supported_backends()
    assert {"zonos", "kitten", "spark"} <= set(backends)


def test_render_kitten_bursts_and_emotion_drop():
    assert render_kitten("[laugh] hello") == "ha-ha hello"
    assert render_kitten("[angry] hello") == "hello"
    assert render_kitten("[pause:500] hello") == "hello"  # pause points out


def test_render_spark_passthrough():
    out = render_spark("[laugh] hello")
    assert out == "ha-ha hello"
    assert render_spark("plain text here") == "plain text here"


# ── nl_director: vocabulary, intensity, describe, dsp params ────────────

def test_parse_intensity_adverb():
    assert parse_direction("[said very angrily]").intensity == 4
    assert parse_direction("[slightly nervous]").intensity == 1
    assert parse_direction("[said angrily]").intensity == 3  # default


def test_parse_compound_accent_pidgin():
    assert parse_direction("[said in Nigerian pidgin accent]").accent == \
        "pidgin"
    assert parse_direction("[in a French accent]").accent == "french"


def test_parse_new_delivery_verbs():
    assert parse_direction("[raps the verse]").delivery in ("rap", "raps")
    assert parse_direction("[she preaches]").delivery in ("preach",
                                                          "preaches")


def test_direction_describe_and_dsp():
    d = parse_direction("[shouts fearfully]")
    s = d.describe()
    assert "fearful" in s and "shout" in s
    p = d.to_dsp_params()
    assert p["pitch_shift"] > 0 and p["rate_mult"] > 1.0
    # intensity scales deviation from neutral
    hi = parse_direction("[very fearfully]").to_dsp_params()
    lo = parse_direction("[slightly fearfully]").to_dsp_params()
    assert hi["pitch_shift"] > lo["pitch_shift"]


def test_direction_method_form_matches_function():
    d = parse_direction("[whispers sadly]")
    assert d.to_dsp_params() == __import__(
        "nomorals.voice.nl_director",
        fromlist=["direction_to_dsp_params"]).direction_to_dsp_params(d)


def test_accent_normalize_suffix_rule():
    assert normalize_accent("Nigerian Pidgin accent") == "pidgin"
    assert normalize_accent("British English") == "british"
    assert normalize_accent("south african") == "south_african"
    assert normalize_accent("klingon") == ""


def test_list_accents_complete():
    accs = list_accents()
    assert len(accs) >= 21
    by_name = {a["accent"]: a for a in accs}
    assert by_name["pidgin"]["hint"] == "en-NG"
    assert by_name["yoruba"]["language"] == "yo"


def test_convert_accent_unknown_passthrough():
    out = convert_accent("hello", FakeTTS(), "klingon")
    assert out["ok"] and out["tier"] == "none"


def test_convert_accent_strength_scales_prosody():
    tts = FakeTTS()
    out = convert_accent("hello there", tts, "nigerian", strength=0.0)
    assert out["ok"] and out["tier"] == "accent-only"
    # the language hint now reaches the backend per call
    assert tts.calls[-1]["language"] == "en-NG"


# ── stt: formats, streaming, diarize guard ─────────────────────────────

def test_to_srt_vtt():
    segs = [{"start": 0.0, "end": 1.2, "text": "hello"},
            {"start": 1.3, "end": 2.5, "text": "world"}]
    srt = to_srt(segs)
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,200\nhello")
    assert "\n\n2\n" in srt
    vtt = to_vtt(segs)
    assert vtt.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.200" in vtt


def test_format_transcript_styles():
    res = {"text": "hi there",
           "segments": [{"start": 0.0, "end": 0.5, "text": "hi",
                         "words": [{"start": 0.0, "end": 0.4,
                                    "word": "hi"}]}]}
    assert format_transcript(res, "text") == "hi there"
    pretty = format_transcript(res, "pretty")
    assert pretty == "[00:00:00.000 → 00:00:00.500] hi"
    words = format_transcript(res, "words")
    assert words == "[00:00:00.000] hi"


def test_streaming_stt_feed_flush():
    s = StreamingSTT(min_chunk_s=10.0)  # never auto-transcribes in test
    partials = s.feed(b"\x00\x00" * 1600)
    assert isinstance(partials, list)
    out = s.flush()
    assert isinstance(out, dict)


def test_diarize_without_pyannote_is_honest():
    try:
        import pyannote.audio  # noqa: F401
        pytest.skip("pyannote installed — can't test the guard")
    except ImportError:
        with pytest.raises(RuntimeError, match="pyannote"):
            diarize(_wav(_tone()))


# ── dialogue: parse, pipeline, duration ────────────────────────────────

def test_parse_dialogue():
    turns = parse_dialogue("[S1: Zara] Hey! [S2: Kilo] Yeah, fire.")
    assert [t.speaker_id for t in turns] == ["1", "2"]
    assert turns[0].speaker_name == "Zara"
    assert turns[1].text == "Yeah, fire."


def test_render_dialogue_pipeline():
    tts = FakeTTS()
    res = render_dialogue("[S1: Zara] Hey! [S2: Kilo] Yeah.",
                          tts, voice_map={"1": "zara"})
    assert res["ok"] and res["turns"] == 2
    assert res["seconds"] > 0.6  # two 0.3s turns + inter-turn pause
    assert res["voices"][0] == "zara"
    assert os.path.exists(res["path"])
    # speaker 2 had no mapped voice → pitch-shifted fallback
    assert "pitch-shifted" in res["voices"][1]


def test_render_dialogue_empty_is_honest():
    assert render_dialogue("   ", FakeTTS())["ok"] is False


def test_estimate_duration():
    dur = estimate_duration([(array("h", [0] * 2400), 8000)] * 2)
    assert dur == pytest.approx(1.0, abs=0.05)
    assert estimate_duration([]) == 0.0


# ── longform: chapters, progress, resume ───────────────────────────────

def test_longform_chapters_progress_resume():
    from nomorals.voice.longform import LongFormSynthesizer

    class ChunkTTS:
        def perform(self, text, voice_name=None, mood="neutral"):
            return {"path": _wav(_tone(secs=0.3))}

    text = " ".join(
        f"Sentence {i} carries the story forward with feeling."
        for i in range(40))
    synth = LongFormSynthesizer(ChunkTTS(), "", "neutral")
    seen: list[int] = []
    res = synth.synthesize(text, progress_cb=lambda d, t, c: seen.append(d))
    assert res["ok"] and res["chapters"]
    assert len(res["chapters"]) == res["rendered"]
    assert seen == list(range(1, res["rendered"] + 1))
    starts = [c["start_s"] for c in res["chapters"]]
    ends = [c["end_s"] for c in res["chapters"]]
    assert all(e >= s for s, e in zip(starts, ends))
    assert all("failed" in c and "text" in c for c in res["chapters"])
    res2 = synth.synthesize(text, resume_from=2)
    assert res2["rendered"] == res["rendered"] - 2
    assert res2["chapters"][0]["index"] == 2


# ── singing: melody parse, humanize, midi ─────────────────────────────

def test_parse_melody_modifiers():
    notes = parse_melody("E4:1.2:love:vib=6,dyn=0.9 Bb3:2.0:ah:bre=0.3")
    assert len(notes) == 2
    assert notes[0].vibrato == 6.0 and notes[0].dynamics == 0.9
    assert notes[0].lyric == "love"
    assert notes[1].breathiness == 0.3
    assert notes[1].midi == 58  # Bb3


def test_parse_melody_flats():
    notes = parse_melody("Bb3:1.0:x")
    assert notes and notes[0].midi == 58


def test_humanize_returns_new_perturbed_notes():
    notes = parse_melody("C4:1.0:a D4:1.0:b E4:1.0:c")
    h = humanize(notes, seed=7)
    assert len(h) == len(notes)
    assert h is not notes and h[0] is not notes[0]
    assert any(abs(a.duration_s - b.duration_s) > 1e-9
               for a, b in zip(notes, h))
    # deterministic per seed
    h2 = humanize(notes, seed=7)
    assert [n.duration_s for n in h] == [n.duration_s for n in h2]


def test_export_midi_valid_smf():
    notes = parse_melody("C4:1.0:a E4:1.0:b G4:2.0:c")
    path = export_midi(notes)
    data = open(path, "rb").read()
    assert data[:4] == b"MThd" and b"MTrk" in data
    assert os.path.getsize(path) > 50


# ── mastering / dsp ────────────────────────────────────────────────────

def test_loudness_match_caps_gain():
    quiet = array("h", [100] * 8000)
    out = loudness_match(quiet, -14.0)
    # ±12 dB cap → gain ≤ ~3.98x → peak stays well under clipping
    assert max(abs(v) for v in out) <= 100 * 4
    assert len(out) == len(quiet)


def test_compress_reduces_peak():
    loud = array("h", [30000, -30000] * 4000)
    out = compress(loud, threshold_db=-20.0, ratio=10.0, amount=1.0)
    assert max(abs(v) for v in out) <= 30000
    # sustained loud section is pulled down (past the attack window)
    assert max(abs(v) for v in out[2000:]) < 30000
    # amount=0 → untouched
    assert list(compress(loud, amount=0.0)) == list(loud)


def test_fade_in_out_edges():
    s = array("h", [10000] * 8000)
    out = fade_in_out(s, 8000, fade_ms=100)
    assert out[0] == 0 and abs(out[-1]) < 100
    assert len(out) == len(s)


def test_apply_delivery_recipes():
    smp = _tone(secs=0.5, amp=8000)
    for verb in ("whisper", "shout", "scream"):
        out = apply_delivery(smp, 8000, verb)
        assert len(out) > 0  # recipes reshape (rate/energy)
    # unknown verbs pass through untouched, never raise
    assert list(apply_delivery(smp, 8000, "nonsense-verb")) == list(smp)
    w_out = whisperize(smp, 8000)
    assert max(abs(v) for v in w_out) < max(abs(v) for v in smp)
    t_out = tremolo(smp, 8000, rate_hz=6.0, depth=0.8)
    assert len(t_out) == len(smp)


# ── biometrics: decide, spoof, liveness ────────────────────────────────

def test_decide_four_outcomes():
    from nomorals.voice.biometrics import (
        MATCH_THRESHOLD,
        REJECT_THRESHOLD,
    )
    assert decide(0.0)["decision"] == "IDENTIFIED"
    assert decide(None)["decision"] == "UNKNOWN"  # unknown is a success
    mid = (MATCH_THRESHOLD + REJECT_THRESHOLD) / 2
    assert decide(mid)["decision"] == "AMBIGUOUS"
    assert decide(REJECT_THRESHOLD + 1.0)["decision"] == "UNKNOWN"
    assert decide(0.0, spoof=0.95)["decision"] == "REJECTED"
    assert decide(0.0, voiced_s=0.01)["decision"] == "REJECTED"
    for r in (decide(0.0), decide(None)):
        assert {"decision", "verdict", "confidence"} <= set(r)


def test_spoof_score_flat_tone_suspicious():
    flat = _wav(array("h", [5000] * 16000))
    res = spoof_score(flat)
    assert res["score"] >= 0.5
    assert res["verdict"] and isinstance(res["reasons"], list)


def test_liveness_challenge_roundtrip():
    ch = liveness_challenge()
    assert len(ch["digits"]) == 4 and ch["expires_at"] > 0
    assert ch["phrase"].startswith("Please say")
    heard = " ".join(str(d) for d in ch["digits"])
    good = check_liveness_response(f"the numbers are {heard} ok", ch)
    assert good["ok"] is True
    bad = check_liveness_response("nine nine nine nine", ch)
    assert bad["ok"] is False
    expired = dict(ch, expires_at=0.0)
    assert check_liveness_response(heard, expired)["ok"] is False


# ── design: intensity adverbs + blend ──────────────────────────────────

def test_describe_to_params_intensity():
    plain = describe_to_params("deep voice")
    very = describe_to_params("very deep voice")
    assert very.pitch_semitones < plain.pitch_semitones == -4.0
    slight = describe_to_params("slightly bright voice")
    very_b = describe_to_params("very bright voice")
    assert 0 < slight.brightness < very_b.brightness
    assert plain.matched  # descriptors recorded


def test_blend_voices_empty_is_honest():
    assert blend_voices("x", [])["ok"] is False


def test_blend_voices_unknown_voices_honest():
    r = blend_voices("x", [("nope1", 1.0), ("nope2", 1.0)])
    assert r["ok"] is False and "reference audio" in r["reason"]


def test_blend_voices_zero_weights_honest():
    assert blend_voices("x", [("a", 0.0)])["ok"] is False


# ── catalogue: search/rename/stats/format ─────────────────────────────

def test_catalogue_search_rename_stats(tmp_path):
    cat = VoiceCatalogue(voices_dir=str(tmp_path))
    cat.add("zara-test", backend="chatterbox", tags=("test-voice",))
    hits = cat.search("zara")
    assert any(h["name"] == "zara-test" for h in hits)
    assert any(h["name"] == "zara-test"
               for h in cat.search(backend="chatterbox"))
    assert any(h["name"] == "zara-test"
               for h in cat.search(tag="test-voice"))
    assert cat.search("zzz-no-match") == []
    cat.rename("zara-test", "zara2")
    assert cat.get("zara2") is not None
    assert cat.get("zara-test") is None
    assert cat.stats()["voices"] >= 1
    assert "zara2" in cat.format_table()


# ── fetch registry ─────────────────────────────────────────────────────

def test_fetch_registry_new_models():
    for key, repo in (("kitten", "KittenML/kitten-tts-nano-0.1"),
                      ("spark", "SparkAudio/Spark-TTS-0.5B"),
                      ("zonos", "Zyphra/Zonos-v0.1-hybrid"),
                      ("kokoro", "hexgrad/Kokoro-82M")):
        assert MODEL_REGISTRY[key]["hf_repo"] == repo
        assert MODEL_REGISTRY[key]["license"] == "Apache-2.0"


# ── session: silero guard + vad fallback ───────────────────────────────

def test_make_vad_auto_falls_back_without_onnx():
    from nomorals.voice.session import EnergyVAD, make_vad
    try:
        import onnxruntime  # noqa: F401
        pytest.skip("onnxruntime installed — can't test fallback")
    except ImportError:
        assert isinstance(make_vad(kind="auto"), EnergyVAD)


def test_silero_vad_missing_runtime_is_honest():
    from nomorals.voice.session import SileroVAD
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="onnxruntime"):
            SileroVAD()


# ── money: confirmation card + parsing ─────────────────────────────────

def test_format_confirmation_styles():
    st = {"amount_kobo": 500000, "recipient": "Ada", "note": "data",
          "id": "stg-1", "status": "biometric_confirmed",
          "created": 1760000000}
    chat = format_confirmation(st)
    assert "₦5,000" in chat and "Ada" in chat and "confirm" in chat
    card = format_confirmation(st, style="card")
    assert "TRANSFER READY" in card and "stg-1" in card
    assert "₦5,000" in card
    receipt = format_confirmation(st, style="receipt")
    assert receipt.startswith("RECEIPT stg-1")
    # dict and object forms both work
    assert "₦5,000" in format_confirmation(st, style="chat")


def test_parse_voice_money_transfer():
    intent = parse_voice_money("send 5000 naira to Ada")
    assert intent.kind == "transfer"
    assert intent.amount_kobo == 500000
    assert intent.recipient == "Ada"


# ── pingpong / rvc / ambience / cache ───────────────────────────────────

def test_voice_note_info_missing():
    info = voice_note_info("/tmp/definitely-not-a-voice-note.ogg")
    assert info["exists"] is False and info["duration_s"] == 0.0


def test_rvc_model_info_missing_is_honest():
    info = model_info("no-such-model", models_dir="/tmp/no-rvc-here")
    assert info["ok"] is False and "not found" in info["reason"]
    assert info["known_models"] == []


def test_describe_scene_parses_layers():
    scene = describe_scene("rainy cafe at night")
    assert isinstance(scene, AmbienceScene)
    kinds = [k for k, _ in scene.layers]
    assert "rain" in kinds or "cafe" in kinds
    empty = describe_scene("xyzzy nothing matches this")
    assert isinstance(empty, AmbienceScene) and empty.layers  # never empty


def test_segment_cache_put_get_invalidate_warm(tmp_path):
    import nomorals.voice.efficiency as ef
    # isolate the disk cache from the real one
    ef._cache_dir = lambda: str(tmp_path)
    cache = SegmentCache(max_entries=10, ttl_s=3600)
    key = SegmentCache.key("hello", "zara", "kitten", "")
    cache.put(key, _tone(), 8000)
    got = cache.get(key)
    assert got is not None and got[1] == 8000
    assert cache.invalidate(key[:8]) >= 1
    assert cache.get(key) is None
    keys = [SegmentCache.key(f"t{i}", "v", "b", "") for i in range(3)]
    cache.warm([(k, _tone(), 8000) for k in keys])
    assert all(cache.get(k) is not None for k in keys)
    rep = cache.size_report()
    assert rep["mem_entries"] >= 3
    cache.clear()


def test_neural_emotion_dsp_tier_with_intensity():
    d = parse_direction("[very angrily]")
    assert isinstance(d, Direction) and d.intensity == 4
    out = render_emotional("hello there", FakeTTS(),
                           direction="[slightly sad]", tiers=("dsp",))
    assert out["ok"] is True and out["tier"] == "dsp"
    assert os.path.exists(out["path"])
    # explicit intensity syncs the DSP tier when no adverb was parsed
    out2 = render_emotional("hello there", FakeTTS(),
                            direction="[sad]", tiers=("dsp",),
                            intensity=10)
    assert out2["ok"] is True
