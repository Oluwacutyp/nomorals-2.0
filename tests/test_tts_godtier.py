"""Tests for God-tier TTS: NL direction, emotion DSP, dialogue, efficiency."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from array import array


def test_nl_direction_basic():
    from nomorals.voice.nl_director import parse_direction
    d = parse_direction("[said angrily in British accent]")
    assert d.emotion == "angry", f"got {d.emotion}"
    assert d.accent == "british", f"got {d.accent}"


def test_nl_direction_whisper():
    from nomorals.voice.nl_director import parse_direction
    d = parse_direction("[whispers fearfully]")
    assert d.delivery in ("whisper", "whispers"), f"got {d.delivery}"
    assert d.emotion in ("fearful", "fear"), f"got {d.emotion}"


def test_nl_direction_ambient():
    from nomorals.voice.nl_director import parse_direction
    d = parse_direction("[light rain]")
    assert d.ambient == "light rain", f"got {d.ambient}"


def test_nl_direction_pace():
    from nomorals.voice.nl_director import parse_direction
    d = parse_direction("[said slowly]")
    assert d.pace == "slow", f"got {d.pace}"


def test_dsp_params():
    from nomorals.voice.nl_director import parse_direction, direction_to_dsp_params
    d = parse_direction("[excited]")
    p = direction_to_dsp_params(d)
    assert p["pitch_shift"] > 0, "excited should raise pitch"
    assert p["rate_mult"] > 1.0, "excited should speed up"


def test_emotion_dsp_shapes():
    from nomorals.voice.emotion_dsp import shape_emotion
    # 1s of 440Hz sine at 22050Hz
    import math
    sr = 22050
    samples = array("h", [int(10000 * math.sin(2 * math.pi * 440 * i / sr))
                          for i in range(sr)])
    # Sad: lower pitch, slower
    out = shape_emotion(samples, sr, pitch_shift_st=-2.0, rate_mult=0.85)
    assert len(out) > len(samples), "slower rate = more samples"
    # Happy: higher pitch
    out2 = shape_emotion(samples, sr, pitch_shift_st=2.0)
    assert len(out2) < len(samples), "higher pitch = fewer samples"


def test_emotion_dsp_energy():
    from nomorals.voice.emotion_dsp import apply_energy
    samples = array("h", [1000] * 100)
    out = apply_energy(samples, 0.5)
    assert out[0] == 500, f"got {out[0]}"


def test_dialogue_parse():
    from nomorals.voice.dialogue import parse_dialogue
    turns = parse_dialogue(
        "[S1: Zara] Hey there. [S2: Kilo] [laughs] Yeah!")
    assert len(turns) == 2
    assert turns[0].speaker_name == "Zara"
    assert turns[1].speaker_name == "Kilo"
    assert "laughs" in turns[1].text


def test_dialogue_no_markers():
    from nomorals.voice.dialogue import parse_dialogue
    turns = parse_dialogue("Just plain text.")
    assert len(turns) == 1
    assert turns[0].speaker_id == "1"


def test_segment_cache():
    from nomorals.voice.efficiency import SegmentCache
    import tempfile
    # Isolated cache dir per test run
    c = SegmentCache(max_entries=10)
    c.dir = tempfile.mkdtemp()
    samples = array("h", [100] * 50)
    import time
    key = SegmentCache.key(f"hello-{time.time_ns()}", "voice1", "piper", "")
    assert c.get(key) is None
    c.put(key, samples, 22050)
    got = c.get(key)
    assert got is not None
    assert got[1] == 22050
    assert list(got[0]) == list(samples)
    stats = c.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1


def test_latency_table():
    from nomorals.voice.efficiency import LatencyTable
    lt = LatencyTable()
    lt.record("test-backend", 150.0, 0.5)
    lt.record("test-backend", 170.0, 0.5)
    avg = lt.get_avg("test-backend")
    assert avg is not None
    assert 150 <= avg <= 170, f"got {avg}"
    best = lt.fastest(["test-backend", "unknown"], max_ms=300)
    assert best == "test-backend"


def test_phone_backends():
    from nomorals.voice.efficiency import phone_viable_backends
    phones = phone_viable_backends()
    assert "chatterbox-nano" in phones
    assert "piper" in phones
    assert "dia" not in phones  # GPU only


def test_nl_tag_parsing_integration():
    """NL tags flow through the tag processor."""
    from nomorals.voice.tts import TagProcessor
    tp = TagProcessor()
    segs = tp.parse("[said angrily in British accent] Hello there.")
    # Should produce segments with parsed emotion/accent tags
    assert len(segs) >= 1
    all_tags = [t for s in segs for t in s.tags]
    assert "angry" in all_tags or "accent:british" in all_tags, \
        f"tags: {all_tags}"


if __name__ == "__main__":
    test_nl_direction_basic()
    test_nl_direction_whisper()
    test_nl_direction_ambient()
    test_nl_direction_pace()
    test_dsp_params()
    test_emotion_dsp_shapes()
    test_emotion_dsp_energy()
    test_dialogue_parse()
    test_dialogue_no_markers()
    test_segment_cache()
    test_latency_table()
    test_phone_backends()
    test_nl_tag_parsing_integration()
    print("All 13 God-tier TTS tests passed.")
