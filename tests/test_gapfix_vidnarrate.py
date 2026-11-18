"""Tests for the code-video narration rung — the video speaks.

The Anthropic code-video style as the owner asked it: text in, polished
video out, NO video model — "text to speech is the bastard I'm talking
about". The code backend rendered real but SILENT MP4s; every code video
was silent and none was ever delivered in chat. This rung speaks the
scene's own script through Devon's existing TTS ladder (UniversalTTS)
and muxes the voice into the MP4.

All speech in these tests is REAL piper synthesis (piper-tts + the
en_US-lessac-medium voice, installed in the verify venv — see the
battery command in VIDNARRATE_SURVEY.md). No placeholder tones, no
fixture wavs standing in for a voice: the e2e test probes the MP4's
audio stream and measures real energy in it.
"""
from __future__ import annotations

import math
import os
import struct
import wave

from nomorals.media.videogen import generate
from nomorals.media.videogen.codegen import render_code_scene
from nomorals.media.videogen.formats import build_format, render_format

DIAGRAM = {
    "engine": "diagram",
    "kind": "diagram",
    "title": "Lagos to Abuja",
    "nodes": [
        {"id": "lagos", "label": "Lagos", "x": 0.15, "y": 0.5,
         "shape": "circle", "step": 0},
        {"id": "abuja", "label": "Abuja", "x": 0.85, "y": 0.5,
         "shape": "circle", "step": 1},
    ],
    "edges": [{"from": "lagos", "to": "abuja", "label": "750 km",
               "step": 1}],
}

SIZE = (320, 180)


# -- helpers -------------------------------------------------------------------

def _wav_rms(path: str, t0: float, t1: float) -> float:
    """RMS energy of a wav slice — proves a region is voiced or silent."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    fmt = f"<{len(raw) // 2}h"
    samples = struct.unpack(fmt, raw)
    a = max(0, int(t0 * sr))
    b = min(len(samples), int(t1 * sr))
    if b <= a:
        return 0.0
    seg = samples[a:b]
    return math.sqrt(sum(s * s for s in seg) / len(seg)) / 32768.0


def _probe_streams(path: str) -> list[dict]:
    """ffprobe stream list for an MP4 (``video_probe`` shape: streams
    carry ``type``/``codec``/``duration``, not ffprobe's raw keys)."""
    from nomorals.media_edit.videos import video_probe

    probe = video_probe(path)
    return probe.get("streams", [])


def _has_audio(path: str) -> bool:
    return any(s.get("type") == "audio" for s in _probe_streams(path))


def _audio_duration(path: str) -> float:
    """Seconds of audio actually in the MP4 — decoded and measured,
    never the container's word for it."""
    from nomorals.media.motion_studio._core import decode_audio_mono

    data = decode_audio_mono(path, sr=22050)
    return len(data) / 22050.0


# -- narration derivation: the scene's own script ------------------------------

def test_scene_narration_derives_steps_from_map_route():
    from nomorals.media.videogen.narration import scene_narration

    design = build_format("map-route", route="Lagos to Abuja")
    segments = scene_narration(design, duration=5.0)
    assert len(segments) >= 2
    texts = [s["text"] for s in segments]
    assert any("Lagos" in t for t in texts)
    assert any("Abuja" in t for t in texts)
    starts = [s["start"] for s in segments]
    assert starts == sorted(starts)
    assert all(s["end"] > s["start"] for s in segments)
    assert all(s["text"].strip() for s in segments)


def test_scene_narration_headline_lines():
    from nomorals.media.videogen.narration import scene_narration

    design = build_format("headline", headline="Gold hits record high",
                          lines="spot up two percent|traders cheer")
    segments = scene_narration(design, duration=6.0)
    texts = " ".join(s["text"] for s in segments)
    assert "Gold hits record high" in texts
    assert "spot up two percent" in texts
    assert "traders cheer" in texts


def test_scene_narration_html_title():
    from nomorals.media.videogen.narration import scene_narration

    segments = scene_narration({"html": "<h1>hi</h1>",
                                "title": "Welcome to Devon"},
                               duration=4.0)
    assert len(segments) == 1
    assert segments[0]["text"] == "Welcome to Devon"
    assert scene_narration({"html": "<h1>hi</h1>"}, duration=4.0) == []


def test_scene_narration_teardown_parts():
    from nomorals.media.videogen.narration import scene_narration

    design = build_format("teardown", thing="keyboard",
                          parts=["keycap", "switch", "plate"])
    segments = scene_narration(design, duration=6.0)
    texts = " ".join(s["text"] for s in segments)
    assert "keycap" in texts
    assert "switch" in texts
    assert "plate" in texts


# -- synthesis + layout: real speech, timed, no bleed --------------------------

def test_synthesize_lays_segments_on_timeline(tmp_path):
    """Real piper speech, laid out so segment 2 starts after segment 1
    ends — the layout, not a tone, proves the timing."""
    from nomorals.media.videogen.narration import (
        synthesize_narration,
    )

    segments = [
        {"text": "Lagos.", "start": 0.0, "end": 2.0},
        {"text": "Abuja.", "start": 2.0, "end": 4.0},
    ]
    res = synthesize_narration(segments, duration=4.0,
                               out_dir=str(tmp_path))
    assert res["ok"] is True, res
    assert res["backend"] == "piper"
    wav = res["path"]
    assert os.path.isfile(wav)
    # segment 1 voiced at its start, silent in the gap, segment 2 voiced
    # after segment 1's speech has ended.
    assert _wav_rms(wav, 0.1, 0.5) > 0.01
    gap_start = res["segments"][0]["spoken_end"]
    gap_end = res["segments"][1]["spoken_start"]
    assert gap_end > gap_start
    mid = (gap_start + gap_end) / 2
    assert _wav_rms(wav, mid - 0.05, mid + 0.05) < 0.005
    assert _wav_rms(wav, res["segments"][1]["spoken_start"] + 0.1,
                    res["segments"][1]["spoken_start"] + 0.5) > 0.01


def test_synthesize_missing_tts_reports_honestly(tmp_path, monkeypatch):
    """No TTS backend on the box: an honest missing, never a silent file
    passed off as narrated, never a crash."""
    import nomorals.media.videogen.narration as narr

    monkeypatch.setattr(narr, "available_backends", lambda: [])
    res = narr.synthesize_narration(
        [{"text": "Hello.", "start": 0.0, "end": 2.0}],
        duration=2.0, out_dir=str(tmp_path))
    assert res["ok"] is False
    assert "no TTS backend" in res["reason"]


# -- end to end: the MP4 has a voice -------------------------------------------

def test_render_code_scene_narrates_by_default(tmp_path):
    out = tmp_path / "route.mp4"
    rec = render_code_scene(DIAGRAM, out, duration=5.0, size=SIZE)
    narr = rec["narration"]
    assert narr["status"] == "spoken", narr
    assert narr["backend"] == "piper"
    assert narr["segments"] >= 2
    assert os.path.isfile(out)
    assert os.path.isfile(narr["audio"])
    streams = _probe_streams(str(out))
    assert _has_audio(str(out)), streams
    assert _audio_duration(str(out)) >= 1.0
    # the audio track is real speech, not encoded silence: decode it and
    # measure energy in the first spoken second.
    from nomorals.media.motion_studio._core import decode_audio_mono

    data = decode_audio_mono(str(out), sr=22050)
    assert len(data) > 22050
    voiced = data[int(0.1 * 22050):int(0.6 * 22050)]
    assert (sum(float(x) * float(x) for x in voiced) / len(voiced)) ** 0.5 \
        > 0.01


def test_render_code_scene_silent_flag_produces_no_audio(tmp_path):
    out = tmp_path / "silent.mp4"
    rec = render_code_scene(DIAGRAM, out, duration=5.0, size=SIZE,
                            narrate=False)
    assert rec["narration"]["status"] == "off"
    assert rec["narration"]["audio"] == ""
    assert not _has_audio(str(out))


def test_render_code_scene_missing_tts_is_honest(tmp_path, monkeypatch):
    import nomorals.media.videogen.narration as narr

    monkeypatch.setattr(narr, "available_backends", lambda: [])
    out = tmp_path / "novoice.mp4"
    rec = render_code_scene(DIAGRAM, out, duration=5.0, size=SIZE)
    assert rec["narration"]["status"] == "missing"
    assert "no TTS backend" in rec["narration"]["reason"]
    assert os.path.isfile(out)
    assert not _has_audio(str(out))


def test_render_format_speaks(tmp_path):
    res = render_format("headline", headline="Gold hits record high",
                        lines="spot up two percent",
                        duration_s=5.0, size=SIZE,
                        out=str(tmp_path / "headline.mp4"))
    assert res.meta["narration"]["status"] == "spoken", res.meta
    assert _has_audio(res.path)
    assert "narrat" in res.message().lower()


def test_generate_code_threads_narration(tmp_path):
    spoken = generate("", backend="code", scene=DIAGRAM, duration_s=5.0,
                      size=SIZE, out=str(tmp_path / "gen.mp4"))
    assert spoken.meta["narration"]["status"] == "spoken"
    silent = generate("", backend="code", scene=DIAGRAM, duration_s=5.0,
                      size=SIZE, out=str(tmp_path / "gen-silent.mp4"),
                      narrate=False)
    assert silent.meta["narration"]["status"] == "off"
    assert not _has_audio(silent.path)


def test_narration_stretches_duration_to_fit_speech(tmp_path):
    """A short video with long narration grows to fit the voice instead
    of clipping it."""
    out = tmp_path / "long.mp4"
    design = build_format(
        "headline", headline="Gold hits a record high in Lagos trading",
        lines="spot up two percent|traders cheer the move|analysts watch")
    rec = render_code_scene(design, out, duration=2.0, size=SIZE)
    assert rec["narration"]["status"] == "spoken"
    assert rec["duration"] > 2.0
    assert _audio_duration(str(out)) >= 1.0
