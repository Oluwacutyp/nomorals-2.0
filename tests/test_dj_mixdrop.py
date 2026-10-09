"""Tests for DJ mix-drop mode (nomorals/media/dj_mixdrop.py)."""

import math
import os
import struct
import wave
from array import array
from pathlib import Path

import pytest

from nomorals.media import dj_mixdrop as md
from nomorals.media.dj_mixdrop import (
    MixFetch,
    _read_segment,
    _write_wav,
    render_mixdrop,
    control_dj_mix,
)

SR = 22050


def _click_track(bpm: float, seconds: float = 30.0,
                 freq: float = 180.0) -> array:
    n = int(SR * seconds)
    out = array("d", [0.0]) * n
    beat_n = int(SR * 60.0 / bpm)
    for start in range(0, n, beat_n):
        for i in range(min(2205, n - start)):
            t = i / SR
            out[start + i] += math.exp(-t * 60.0) * math.sin(
                2 * math.pi * freq * t)
    return out


def _write_test_wav(path: str, samples: array) -> str:
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(struct.pack("<%dh" % len(samples),
                                 *(int(s * 20000) for s in samples)))
    return path


@pytest.fixture()
def two_tracks(tmp_path):
    p1 = _write_test_wav(str(tmp_path / "t1.wav"), _click_track(120.0))
    p2 = _write_test_wav(str(tmp_path / "t2.wav"), _click_track(122.0))
    return [
        MixFetch(query="track one", path=p1, title="Track One",
                 artist="A", source="test"),
        MixFetch(query="track two", path=p2, title="Track Two",
                 artist="B", source="test"),
    ]


def test_mixfetch_ok():
    assert MixFetch(query="x", path="/p").ok
    assert not MixFetch(query="x", error="nope").ok
    assert not MixFetch(query="x").ok


def test_wav_roundtrip(tmp_path):
    samples = _click_track(120.0, seconds=5.0)
    p = str(tmp_path / "rt.wav")
    assert _write_wav(p, samples)
    back = _read_segment(p, 0, 5.0)
    assert abs(len(back) - len(samples)) < SR  # within a second


def test_read_segment_skip(tmp_path):
    samples = _click_track(120.0, seconds=20.0)
    p = _write_test_wav(str(tmp_path / "s.wav"), samples)
    seg = _read_segment(p, 8.0, 10.0)
    assert abs(len(seg) - 10 * SR) < SR


def test_render_mixdrop_two_tracks(two_tracks, tmp_path):
    res = render_mixdrop(two_tracks, tmp_path, play_s=20.0,
                         mix_name="test_mix")
    assert res["ok"], res.get("reason")
    assert os.path.exists(res["path"])
    assert len(res["tracklist"]) == 2
    # both tracks at ~120 BPM -> beatmatched blend expected
    kinds = [t["kind"] for t in res["transitions"]]
    assert kinds == ["blend"], kinds
    # mix duration ≈ 2 tracks minus blend overlap
    assert 25.0 < res["duration_s"] < 40.0


def test_render_mixdrop_all_failed(tmp_path):
    bad = [MixFetch(query="nope", error="not found")]
    res = render_mixdrop(bad, tmp_path)
    assert not res["ok"]
    assert res["skipped"]


def test_render_mixdrop_partial(tmp_path):
    p1 = _write_test_wav(str(tmp_path / "t1.wav"), _click_track(120.0))
    fetches = [
        MixFetch(query="good", path=p1, title="Good", source="test"),
        MixFetch(query="bad", error="download failed"),
    ]
    res = render_mixdrop(fetches, tmp_path, play_s=15.0)
    assert res["ok"]
    assert len(res["tracklist"]) == 1
    assert len(res["skipped"]) == 1


def test_control_dj_mix_no_tracks(monkeypatch):
    # resolver finds nothing -> honest message, no crash
    monkeypatch.setattr(md, "fetch_track_audio",
                        lambda q, wd, ctx=None: MixFetch(query=q,
                                                        error="nope"))
    monkeypatch.setattr(md, "_trending_queries", lambda g, n, c=None: [])
    out = control_dj_mix("", context=None)
    assert "couldn't find" in out or "empty" in out


def test_control_dj_mix_explicit_songs(monkeypatch, tmp_path):
    p1 = _write_test_wav(str(tmp_path / "t1.wav"), _click_track(120.0))
    p2 = _write_test_wav(str(tmp_path / "t2.wav"), _click_track(124.0))

    def fake_fetch(q, wd, ctx=None):
        return {"song a": MixFetch(query=q, path=p1, title="Song A",
                                   source="test"),
                "song b": MixFetch(query=q, path=p2, title="Song B",
                                   source="test")}[q]

    monkeypatch.setattr(md, "fetch_track_audio", fake_fetch)
    delivered = {}

    def fake_audio(path, caption):
        delivered["path"] = path
        delivered["caption"] = caption

    out = control_dj_mix("song a, song b",
                         deliver_audio=fake_audio,
                         deliver_text=lambda t: None,
                         context=None)
    assert "mixdrop" in out.lower()
    assert delivered.get("path", "").endswith(".wav")
    assert os.path.exists(delivered["path"])


def test_control_dj_mix_genre_path(monkeypatch):
    seen = {}

    def fake_trending(genre, n, context=None):
        seen["genre"] = genre
        return ["artist x track y"]

    def fake_fetch(q, wd, ctx=None):
        return MixFetch(query=q, error="offline in test")

    monkeypatch.setattr(md, "_trending_queries", fake_trending)
    monkeypatch.setattr(md, "fetch_track_audio", fake_fetch)
    out = control_dj_mix("afrobeats", context=None)
    assert seen["genre"] == "afrobeats"
    assert "couldn't fetch" in out
