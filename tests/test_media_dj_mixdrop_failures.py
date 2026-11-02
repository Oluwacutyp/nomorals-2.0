"""Tail-91 pins: nomorals/media/dj_mixdrop.py census sites.

Module posture: ``fetch_track_audio`` / ``render_mixdrop``
"Never raises" — every fetch or render failure lands in an
honest error string, a ``skipped`` entry, or an ``ok: False``
result; nothing is ever presented as a mix that didn't render.
Every site is held as an exclusion:

129 — cookie-opts crash -> pass inside ``_ytdlp_download``.
The yt-dlp cookie options are an optional enhancement rung:
when the cookies helper explodes, the download proceeds
cookieless with the identical success/failure surface (a
download that then fails returns "" and logs at the sibling
guard). Probe-proven: with cookie opts raising, a download
still completes.
172 — no-ffmpeg WAV probe crash -> pass in ``_to_wav``, then
``return ""``. The guarded ``wave.open`` is a validity probe
("already a usable 22050 Hz mono WAV?"); a file that can't be
opened is genuinely not usable as-is, and "" surfaces to the
caller as the honest "couldn't convert ... to WAV" error.
216 — ``_read_segment`` crash -> empty array. Empty is the
module's unreadable value, and the sole consumer
(render_mixdrop) treats it as such: the track lands in
``skipped`` with "unreadable audio segment" and a mix with no
readable tracks returns ok: False — never a silent hole in a
"successful" mix.
400 — first progress ``deliver_text`` crash -> pass. A
best-effort chat progress ping ("digging N tracks…"); the
operation is the fetch+render pipeline, whose real outcome is
the returned string. A dead delivery callback must not kill
the mix.
414 — second progress ``deliver_text`` crash -> pass. Same
shape ("got N/M — analyzing + beatmatching…").
"""
from __future__ import annotations

import logging
import math
import shutil
import struct
import sys
import types
import wave

from nomorals.media import dj_mixdrop as M
import nomorals.media.cookies as cookies_mod


# ── 129: cookie-opts crash — download proceeds cookieless ───────

def test_129_cookie_opts_crash_download_still_succeeds(
        tmp_path, monkeypatch, caplog):
    def _boom():
        raise RuntimeError("cookies boom")

    monkeypatch.setattr(cookies_mod, "ytdlp_cookie_opts", _boom)
    (tmp_path / "dl_vid1.webm").write_bytes(b"x")

    class FakeYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=True):
            return {"id": "vid1"}

    fake_ytdlp = types.ModuleType("yt_dlp")
    fake_ytdlp.YoutubeDL = FakeYDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_ytdlp)
    with caplog.at_level(logging.DEBUG,
                         logger="nomorals.media.dj_mixdrop"):
        got = M._ytdlp_download("http://example.invalid/v", tmp_path)
    assert got == str(tmp_path / "dl_vid1.webm")
    # the site records nothing — no warning is emitted for it
    assert [r for r in caplog.records
            if r.name == "nomorals.media.dj_mixdrop"] == []


# ── 172: no-ffmpeg WAV validity probe ───────────────────────────

def _write_wav(path, rate, channels):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(struct.pack("<h", 100) * 100 * channels)


def test_172_no_ffmpeg_wav_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    good = tmp_path / "good.wav"
    _write_wav(good, 22050, 1)
    # control: an already-usable WAV passes through untouched
    assert M._to_wav(str(good), tmp_path) == str(good)
    wrong = tmp_path / "wrong.wav"
    _write_wav(wrong, 44100, 2)
    assert M._to_wav(str(wrong), tmp_path) == ""
    junk = tmp_path / "junk.mp3"
    junk.write_bytes(b"not audio at all")
    # wave.open explodes on the junk file; the pass is the probe
    # answering "not a usable WAV" — the caller reports the
    # honest couldn't-convert error
    assert M._to_wav(str(junk), tmp_path) == ""


# ── 216: unreadable segment — empty, and the mix says so ────────

def test_216_read_segment_crash_returns_empty(tmp_path):
    assert len(M._read_segment("/nonexistent/x.wav", 0.0, 1.0)) == 0
    seg_path = tmp_path / "seg.wav"
    with wave.open(str(seg_path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(22050)
        f.writeframes(b"".join(
            struct.pack("<h", int(10000 * math.sin(i / 10)))
            for i in range(22050)))
    # control: a real WAV reads its samples
    assert len(M._read_segment(str(seg_path), 0.0, 1.0)) == 22050


def test_216_consumer_marks_unreadable_track_skipped(tmp_path):
    res = M.render_mixdrop(
        [M.MixFetch(query="ghost", path="/nonexistent/x.wav",
                    title="Ghost")],
        tmp_path / "mixout")
    assert res["ok"] is False
    assert res["skipped"][0]["error"] == "unreadable audio segment"


# ── 400 / 414: progress-ping delivery crashes swallowed ─────────

def test_400_414_deliver_text_crashes_swallowed(monkeypatch):
    monkeypatch.setattr(
        M, "fetch_track_audio",
        lambda q, wd, context=None: M.MixFetch(query=q, path="/x.wav",
                                               title=q))
    monkeypatch.setattr(
        M, "render_mixdrop",
        lambda fetches, wd, **k: {"ok": False, "reason": "probe"})

    def _boom_text(msg):
        raise RuntimeError("deliver boom")

    out = M.control_dj_mix("song a, song b", deliver_text=_boom_text)
    # both progress pings exploded; the pipeline's real outcome
    # is still the returned string
    assert out == "mix failed: probe"
