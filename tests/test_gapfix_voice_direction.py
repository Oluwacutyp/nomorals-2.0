"""Voice direction track: Direction -> track builder + per-segment render.

The organ (nomorals/voice/direction.py) was complete but had zero
production callers: direction was parsed live by TagProcessor into
flat string tags and a directed line rendered with one flat mood
(render_emotional over the whole text). These tests pin the wiring:
the builder anchors cues to word ranges following TagProcessor's
segment boundaries, and render_emotional renders a directed line
per-segment through the track — while an untagged line renders
exactly as before.
"""

import wave
from array import array
from pathlib import Path

import pytest

from nomorals.voice.direction import (
    DirectionCue,
    segments,
    track_from_segments,
    track_from_text,
)
from nomorals.voice.neural_emotion import render_emotional
from nomorals.voice.nl_director import Direction
from nomorals.voice.tts import TagProcessor


def _read_wav_samples(path):
    with wave.open(path, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2
        return w.getframerate(), list(
            array("h", w.readframes(w.getnframes())))


class _FakeTTS:
    """perform()/speak() -> tiny wav; records every call.

    Each call writes 4 distinct non-zero samples so the stitched
    output proves segment order (and pause silence shows up as zeros
    between them).
    """

    def __init__(self, tmp_path, sr=8000):
        self.sr = sr
        self.tmp_path = Path(tmp_path)
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
                **kw):
        samples = self._samples()
        path = self._write(samples)
        self.perform_calls.append(
            {"text": text, "mood": mood, "intensity": intensity,
             "path": path, "samples": samples})
        return {"path": path, "backend": "fake"}

    def speak(self, text, voice_name=None, **kw):
        # a 0.3s tone: the DSP tier shapes real frames, not 4 samples
        import math

        samples = [int(8000 * math.sin(2 * math.pi * 440 * i / self.sr))
                   for i in range(int(0.3 * self.sr))]
        path = self._write(samples)
        self.speak_calls.append(
            {"text": text, "path": path, "samples": samples})
        return {"path": path, "backend": "fake"}


# ── builder: Direction -> track ─────────────────────────────────────────────

class TestTrackBuilder:
    def test_track_anchors_cues_to_word_ranges(self):
        line = ("[said angrily in British accent] hello there "
                "[whispers fearfully] good night")
        dialogue, track = track_from_text(line)
        assert dialogue == "hello there good night"
        assert track is not None
        assert [(c.word_start, c.word_end) for c in track.cues] == [
            (0, 2), (2, 4)]
        first, second = track.cues
        assert first.emotion == "angry" and first.accent == "british"
        assert second.emotion == "fearful" and second.delivery == "whispers"
        # word lookup follows the anchors
        assert track.cue_for_word(0) is first
        assert track.cue_for_word(1) is first
        assert track.cue_for_word(2) is second
        assert track.cue_for_word(3) is second

    def test_track_change_mid_line_leaves_earlier_words_undirected(self):
        dialogue, track = track_from_text(
            "we came in peace [said angrily] leave now")
        assert dialogue == "we came in peace leave now"
        assert track is not None
        assert len(track.cues) == 1
        cue = track.cues[0]
        assert (cue.word_start, cue.word_end) == (4, 6)
        assert cue.emotion == "angry"
        assert track.cue_for_word(0) is None
        assert track.cue_for_word(3) is None
        assert track.cue_for_word(4) is cue

    def test_track_none_for_untagged_line(self):
        dialogue, track = track_from_text("hello world")
        assert dialogue == "hello world"
        assert track is None

    def test_track_malformed_tag_degrades_to_surrounding_direction(self):
        # TagProcessor ignores "[said blah blah]" (no recognized word)
        # and the previous direction persists — the builder inherits
        # that, it never re-parses a tag.
        dialogue, track = track_from_text(
            "[said angrily] hello [said blah blah] world")
        assert dialogue == "hello world"
        assert track is not None
        assert [c.emotion for c in track.cues] == ["angry", "angry"]
        assert [(c.word_start, c.word_end) for c in track.cues] == [
            (0, 1), (1, 2)]

    def test_track_skips_sound_segments_in_word_ranges(self):
        dialogue, track = track_from_text(
            "[said angrily] hello [laughs] world")
        # the laugh is not a dialogue word: it takes no word range
        assert dialogue == "hello world"
        assert [(c.word_start, c.word_end) for c in track.cues] == [
            (0, 1), (1, 2)]

    def test_track_carries_pause_after(self):
        _, track = track_from_text("[happy] hello [pause:300] world")
        assert track is not None
        assert track.cues[0].pause_after_s == pytest.approx(0.3)
        assert track.cues[1].pause_after_s == 0.0

    def test_track_default_fills_untagged_segments(self):
        dialogue, track = track_from_text(
            "hello [said angrily] world",
            default=Direction(emotion="sad"))
        assert dialogue == "hello world"
        assert [c.emotion for c in track.cues] == ["sad", "angry"]

    def test_track_from_segments_accepts_parsed_segments(self):
        parsed = TagProcessor().parse(
            "[said happily] good morning [said sadly] good night")
        dialogue, track = track_from_segments(parsed)
        assert dialogue == "good morning good night"
        assert [c.emotion for c in track.cues] == ["happy", "sad"]
        assert [(c.word_start, c.word_end) for c in track.cues] == [
            (0, 2), (2, 4)]

    def test_segments_surface_accent_delivery_pace(self):
        _, track = track_from_text(
            "[said angrily in British accent] hello there "
            "[whispers fearfully] good night")
        segs = segments("hello there good night", track)
        assert [s["text"] for s in segs] == ["hello there", "good night"]
        assert segs[0]["accent"] == "british"
        assert segs[0]["emotion"] == "angry"
        assert segs[1]["delivery"] == "whispers"
        assert segs[1]["emotion"] == "fearful"

    def test_cue_round_trips_with_new_fields(self):
        cue = DirectionCue(word_start=0, word_end=2, emotion="angry",
                           accent="british", delivery="whispers",
                           pace="fast")
        assert DirectionCue.from_dict(cue.to_dict()).to_dict() == \
            cue.to_dict()


# ── wiring: render_emotional through the track ───────────────────────────────

class TestDirectedRender:
    def test_directed_line_renders_per_segment(self, tmp_path):
        fake = _FakeTTS(tmp_path)
        line = ("[said angrily in British accent] hello there "
                "[whispers fearfully] good night")
        out = render_emotional(line, fake, direction="[said angrily]")
        assert out["ok"] is True
        assert out["directed"] == "per-segment"
        assert out["segments"] == 2
        # each segment's direction reached its own backend call
        assert [(c["text"], c["mood"]) for c in fake.perform_calls] == [
            ("hello there", "angry"), ("good night", "fearful")]
        # the flat mood is still derivable: whole-line direction
        assert out["emotion"] == "angry"
        assert out["tier"] == "native"
        assert out["segment_tiers"] == ["native", "native"]
        # the track rode along
        assert out["track"]["cues"][0]["accent"] == "british"
        assert out["track"]["cues"][1]["delivery"] == "whispers"
        # stitched wav is the two segment wavs concatenated, in order
        _, samples = _read_wav_samples(out["path"])
        expect = (fake.perform_calls[0]["samples"]
                  + fake.perform_calls[1]["samples"])
        assert samples == expect

    def test_untagged_line_renders_flat_and_unchanged(self, tmp_path):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("hello world", fake,
                               direction="[said angrily]")
        # one backend call, whole text, flat mood — the old path
        assert len(fake.perform_calls) == 1
        assert fake.perform_calls[0]["text"] == "hello world"
        assert fake.perform_calls[0]["mood"] == "angry"
        assert "directed" not in out
        assert out["path"] == fake.perform_calls[0]["path"]
        assert out["tier"] == "native"
        # the flat tier dict carries no "emotion" key (only the RVC
        # tier does) — the per-segment result adds one, the flat path
        # is byte-identical to before, keys included

    def test_single_direction_line_renders_one_segment(self, tmp_path):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("[said angrily] hello world", fake,
                               direction="[said angrily]")
        assert out["directed"] == "per-segment"
        assert out["segments"] == 1
        assert [(c["text"], c["mood"]) for c in fake.perform_calls] == [
            ("hello world", "angry")]

    def test_sound_segment_survives_directed_render(self, tmp_path):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("[said angrily] hello [laughs] world",
                               fake, direction="[said angrily]")
        assert out["segments"] == 3
        # the laugh renders (neutral — nothing to direct) between the
        # two angry words; nothing is dropped
        assert [(c["text"], c["mood"]) for c in fake.perform_calls] == [
            ("hello", "angry"), ("[laughs]", "neutral"),
            ("world", "angry")]
        _, samples = _read_wav_samples(out["path"])
        expect = [s for c in fake.perform_calls for s in c["samples"]]
        assert samples == expect

    def test_pause_after_inserts_silence(self, tmp_path):
        fake = _FakeTTS(tmp_path)
        out = render_emotional("[happy] hello [pause:300] world", fake,
                               direction="[happy]")
        assert out["directed"] == "per-segment"
        _, samples = _read_wav_samples(out["path"])
        first = fake.perform_calls[0]["samples"]
        second = fake.perform_calls[1]["samples"]
        # 300ms of silence at 8kHz between the two segments
        assert samples == first + [0] * 2400 + second

    def test_directed_line_renders_per_segment_on_dsp_tier(self, tmp_path):
        fake = _FakeTTS(tmp_path)
        line = ("[said angrily] hello there "
                "[said sadly] good night")
        out = render_emotional(line, fake, direction="[said angrily]",
                               tiers=("dsp",))
        assert out["ok"] is True
        assert out["directed"] == "per-segment"
        assert out["tier"] == "dsp"
        # the DSP tier shapes each segment with its own direction:
        # speak() ran once per spoken segment, in order
        assert [c["text"] for c in fake.speak_calls] == [
            "hello there", "good night"]
