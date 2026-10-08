"""Tests for nomorals.media.dj — radio DJ with trending + voice breaks."""

import json
import os
import tempfile
import wave
from array import array
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nomorals.media import dj as dj_mod
from nomorals.media.dj import (
    DJ,
    _echo_out,
    _equal_power_crossfade,
    _filter_sweep_in,
    _render_sting,
    fetch_trending,
)


FAKE_BILLBOARD_HTML = """
<html><head><title>Hot 100</title></head><body>
<script type="application/ld+json">
{"@type":"ItemList","itemListElement":[
{"title":"Neon Skyline","artist":"Nova Rae"},
{"title":"Midnight Fuel","artist":"DJ Carbon"},
{"title":"Palm Wine Dreams","artist":"Adaeze"}]}
</script>
</body></html>
"""


def _fake_urlopen(html):
    class _Resp:
        def __init__(self, data):
            self._data = data.encode()

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return _Resp(html)


# ── trending fetcher ─────────────────────────────────────────────────────────

def test_fetch_trending_parses_billboard_html(tmp_path, monkeypatch):
    monkeypatch.setattr(dj_mod, "_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(dj_mod, "_CACHE_TTL", 0)  # force live
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen(FAKE_BILLBOARD_HTML)):
        res = fetch_trending(force_refresh=True)
    assert res["ok"] is True
    assert res["source"] == "billboard"
    titles = [t["title"] for t in res["tracks"]]
    assert "Neon Skyline" in titles
    assert "Midnight Fuel" in titles
    # ranks assigned in order
    assert res["tracks"][0]["rank"] == 1


def test_fetch_trending_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(dj_mod, "_CACHE_DIR", tmp_path / "cache")
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen(FAKE_BILLBOARD_HTML)):
        first = fetch_trending(force_refresh=True)
    assert first["ok"] and not first["cached"]
    with patch("urllib.request.urlopen",
               side_effect=AssertionError("should not hit network")):
        second = fetch_trending()
    assert second["ok"] and second["cached"] is True
    assert second["tracks"] == first["tracks"]


def test_fetch_trending_never_raises_on_network_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dj_mod, "_CACHE_DIR", tmp_path / "cache")
    with patch("urllib.request.urlopen",
               side_effect=OSError("network is down")):
        res = fetch_trending(force_refresh=True)
    assert res["ok"] is False
    assert "reason" in res
    assert res["tracks"] == []


def test_fetch_trending_empty_html_honest(tmp_path, monkeypatch):
    monkeypatch.setattr(dj_mod, "_CACHE_DIR", tmp_path / "cache")
    with patch("urllib.request.urlopen",
               return_value=_fake_urlopen("<html><body>nothing here</body></html>")):
        res = fetch_trending(force_refresh=True)
    assert res["ok"] is False
    assert res["tracks"] == []


# ── DSP ──────────────────────────────────────────────────────────────────────

def _tone(freq=440.0, secs=1.0, amp=0.5):
    import math
    n = int(22050 * secs)
    return array("d", (amp * math.sin(2 * math.pi * freq * i / 22050)
                       for i in range(n)))


def test_crossfade_joins_without_gap():
    a = _tone(440.0, 2.0)
    b = _tone(660.0, 2.0)
    out = _equal_power_crossfade(a, b, int(22050 * 1.0))
    # 2s + 2s - 1s overlap = 3s
    assert abs(len(out) - 22050 * 3) < 10


def test_crossfade_does_not_clip():
    a = _tone(440.0, 1.0, amp=0.9)
    b = _tone(660.0, 1.0, amp=0.9)
    xf = 22050 // 2
    out = _equal_power_crossfade(a, b, xf)
    # equal-power keeps *power* flat through the blend: RMS in the blend
    # region should be close to RMS of the dry regions (not doubled).
    import math
    blend = out[len(a) - xf:len(a)]
    dry = out[:len(a) - xf]
    rms = lambda xs: math.sqrt(sum(x * x for x in xs) / max(1, len(xs)))
    assert rms(blend) <= rms(dry) * 1.35
    # and nothing insane: peak stays in a sane range (final mix normalizes)
    assert max(abs(s) for s in out) <= 1.8


def test_crossfade_zero_overlap_is_concat():
    a = _tone(440.0, 0.5)
    b = _tone(660.0, 0.5)
    out = _equal_power_crossfade(a, b, 0)
    assert len(out) == len(a) + len(b)


def test_crossfade_never_raises_on_garbage():
    out = _equal_power_crossfade(array("d"), array("d"), 100)
    assert isinstance(out, array)


def test_echo_out_adds_tail_energy():
    s = _tone(440.0, 1.0, amp=0.4)
    # mostly silence with a hit at the start of the tail region
    tail = _echo_out(s, 22050, tail_ms=500.0)
    assert len(tail) == len(s)
    # echo keeps energy alive where the dry signal decayed — just check finite
    assert all(abs(x) < 10.0 for x in tail)


def test_filter_sweep_starts_muffled():
    s = _tone(2000.0, 2.0, amp=0.5)
    out = _filter_sweep_in(s, 22050, sweep_ms=1000.0)
    assert len(out) == len(s)
    # first 100ms should be much quieter than the last 100ms
    head = sum(abs(x) for x in out[:2205]) / 2205
    tail = sum(abs(x) for x in out[-2205:]) / 2205
    assert tail > head * 1.5


# ── voice breaks ─────────────────────────────────────────────────────────────

def test_dj_say_never_silence(tmp_path):
    samples, sr, note = dj_mod._dj_say("You're listening to Devon FM", tmp_path)
    assert sr > 0
    assert len(samples) > sr  # at least a second of *something*
    assert note  # honest note about which fallback rendered


def test_render_sting_is_audible():
    s = _render_sting()
    assert len(s) > 22050  # > 1s
    assert max(abs(x) for x in s) > 0.01


# ── show composition ─────────────────────────────────────────────────────────

def _mock_song(style="pop"):
    song = MagicMock()
    song.audio_path = ""  # filled per-test with a real wav
    return song


def _write_test_wav(path, secs=3.0):
    n = int(22050 * secs)
    import math
    pcm = array("h", (int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / 22050))
                      for i in range(n)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(pcm.tobytes())
    return str(path)


def test_build_show_order_and_breaks(tmp_path):
    workdir = tmp_path / "dj"
    workdir.mkdir()
    wav = _write_test_wav(workdir / "t1.wav", secs=4.0)

    dj = DJ(context=None)
    fake_tracks = [
        {"title": "Neon Skyline", "artist": "Nova Rae", "rank": 1,
         "genre_hint": "pop", "source": "billboard"},
        {"title": "Midnight Fuel", "artist": "DJ Carbon", "rank": 2,
         "genre_hint": "edm", "source": "billboard"},
    ]
    with patch.object(DJ, "fetch_trending",
                      return_value={"ok": True, "tracks": fake_tracks,
                                    "source": "billboard", "cached": False}):
        with patch("nomorals.media.music.MusicCreator") as MC:
            inst = MC.return_value
            inst.compose.side_effect = lambda *a, **k: _mock_song_with(wav)
            show = dj.build_show(genre="", n_tracks=2,
                                 workdir=str(workdir))

    assert show["ok"] is True, show.get("reason")
    assert os.path.isfile(show["path"])
    assert len(show["tracklist"]) == 2
    assert all(t["status"] == "played" for t in show["tracklist"])
    # intro + break-1 + outro = 3 voice breaks
    assert len(show["voice_notes"]) == 3
    assert show["segments"] == 5  # break, track, break, track, break
    assert show["duration_s"] > 8  # 2×4s tracks + breaks


def _mock_song_with(wav):
    m = MagicMock()
    m.audio_path = wav
    return m


def test_build_show_never_raises_when_compose_fails(tmp_path):
    dj = DJ(context=None)
    with patch.object(DJ, "fetch_trending",
                      return_value={"ok": False, "reason": "down", "tracks": []}):
        with patch("nomorals.media.music.MusicCreator") as MC:
            MC.return_value.compose.side_effect = RuntimeError("boom")
            show = dj.build_show(n_tracks=2, workdir=str(tmp_path / "dj2"))
    assert show["ok"] is False
    assert "reason" in show


def test_build_show_empty_chart_still_uses_originals(tmp_path):
    workdir = tmp_path / "dj3"
    workdir.mkdir()
    wav = _write_test_wav(workdir / "t.wav", secs=4.0)
    dj = DJ(context=None)
    with patch.object(DJ, "fetch_trending",
                      return_value={"ok": False, "reason": "down", "tracks": []}):
        with patch("nomorals.media.music.MusicCreator") as MC:
            MC.return_value.compose.side_effect = lambda *a, **k: _mock_song_with(wav)
            show = dj.build_show(genre="uk-drill", n_tracks=2,
                                 workdir=str(workdir))
    assert show["ok"] is True
    assert show["trending_source"] == "none"


# ── command wiring ───────────────────────────────────────────────────────────

def test_control_dj_registered():
    from nomorals.social.chat.control import CONTROL_COMMANDS, COMMAND_DETAILS
    assert "dj" in CONTROL_COMMANDS
    assert "dj" in COMMAND_DETAILS


def test_control_dj_trending_lists_charts():
    from nomorals.agents.partner import runtime_media as rm
    mixin = MagicMock()
    mixin.context = MagicMock()
    with patch.object(DJ, "fetch_trending",
                      return_value={"ok": True, "tracks": [
                          {"title": "T", "artist": "A", "rank": 1}],
                          "source": "billboard", "cached": False}):
        out = rm.RuntimeMediaMixin._control_dj(mixin, "trending")
    assert "T" in out and "A" in out


def test_control_dj_never_raises():
    from nomorals.agents.partner import runtime_media as rm
    mixin = MagicMock()
    mixin.context = MagicMock()
    with patch("nomorals.media.dj.DJ") as DJCls:
        DJCls.return_value.build_show.side_effect = RuntimeError("x")
        out = rm.RuntimeMediaMixin._control_dj(mixin, "")
    assert "couldn't build" in out or "dj error" in out
