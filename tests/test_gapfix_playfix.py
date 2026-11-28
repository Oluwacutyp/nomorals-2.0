"""Failing-first tests for the /play rebuild (gapfix168).

1. A Spotify link in chat no longer dead-ends at the DRM refusal:
   the link is identified on Spotify and the FULL track is fetched
   through the /play ladder, labelled by where it came from.
2. /play <song> reverb|nightcore|slowed|sped up (and /play reverb
   <song>) returns that version: the version is pulled from the
   source first, else rendered onto the fetched full track with
   the DSP chain. A requested version is the answer, never a
   demoted fallback.
"""

import math
import os
import time
import wave
from array import array
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.resolver import ResolvedAudio

SPOTIFY_URL = ("https://open.spotify.com/track/55t0oslolWtghfZX5VAJ9X"
               "?si=ftHP2uitRAmGd5VKTcbfBw&utm_source=copy-link")
SPOTIFY_URI = "spotify:track:55t0oslolWtghfZX5VAJ9X"

_RT_LOGGER = "nomorals.agents.partner.runtime_media"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    from nomorals.media import download_failures as _df

    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    _df.clear_failures()
    yield
    _df.clear_failures()


class _FakeGateway:
    """Records sends/files/edits; plays the adapter for edit verbs."""

    def __init__(self):
        self.sent = []
        self.files = []
        self.edits = []
        self._n = 0

    def send(self, platform, chat, text, **kw):
        self._n += 1
        self.sent.append((platform, chat, text))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id=f"m{self._n}")

    def send_file(self, platform, target, path, caption="", kind=None):
        self.files.append((platform, target, path, caption))
        return SimpleNamespace(ok=True, platform=platform,
                               message_id="f1")

    def _adapter_for(self, platform):
        return self

    def edit_message(self, chat, message_id, text, *, buttons=None):
        self.edits.append((str(message_id), text))
        return {"ok": True}

    def edit_progress(self, chat, message_id, text):
        self.edits.append((str(message_id), text))
        return {"ok": True}


def _handler(chat=None):
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    rt = RuntimeMediaMixin.__new__(RuntimeMediaMixin)
    rt.context = SimpleNamespace()
    rt.gateway = _FakeGateway()
    if chat is not None:
        rt._ref_from_key = lambda key: chat
    return rt


def _chat(chat_id="12345"):
    chat = mock.Mock()
    chat.platform = "telegram"
    chat.chat_id = chat_id
    chat.key = f"telegram:{chat_id}"
    return chat


def _collect_play_job(chat_key, title, timeout=10.0):
    from nomorals.core.jobs import shared_registry

    deadline = time.time() + timeout
    snaps = []
    while time.time() < deadline:
        snaps = shared_registry().list(
            labels={"owner": "play", "chat": chat_key, "title": title})
        if snaps:
            break
        time.sleep(0.05)
    assert snaps, f"no play job for {chat_key}/{title}"
    job = shared_registry().collect(snaps[-1]["id"], timeout=timeout)
    assert job is not None and job.done(), \
        f"job never finished: {job.status if job else None}"
    return job


def _spotify_drm_res():
    return ResolvedAudio(
        ok=True, path_or_url=SPOTIFY_URI, title=SPOTIFY_URI,
        kind="spotify", source_name="Spotify link",
        downloadable=False, block_reason="drm",
        attempts=["spotify-link — ok"])


def _yt_res(title="Nightmares To Dreams", artist="Meek Mill"):
    return ResolvedAudio(
        ok=True, path_or_url="https://www.youtube.com/watch?v=abcdefghijk",
        title=title, artist=artist, kind="youtube",
        source_name="YouTube", downloadable=True,
        attempts=["youtube-search — ok (1)"])


def _yt_cand(title="Nightmares To Dreams", artist="Meek Mill",
             path="https://www.youtube.com/watch?v=abcdefghijk"):
    return ResolvedAudio(
        ok=True, path_or_url=path, title=title, artist=artist,
        kind="youtube", source_name="YouTube", downloadable=True)


def _meta():
    from nomorals.media.spotify_meta import SpotifyMeta, SpotifyTrack

    return SpotifyMeta(
        kind="track", spotify_id="55t0oslolWtghfZX5VAJ9X",
        name="Nightmares To Dreams", artist="Meek Mill",
        duration_s=200.0,
        tracks=[SpotifyTrack(title="Nightmares To Dreams",
                             artist="Meek Mill", duration_s=200.0)],
        source="oembed")


def _engine_with_download(download_fn):
    engine = mock.Mock()
    engine._enqueue.side_effect = lambda p, k, t, **kw: {
        "kind": k, "path": p, "title": t,
        "artist": kw.get("artist", ""), "duration": kw.get("duration", 0.0)}
    engine.queue.return_value = []
    engine.download.side_effect = download_fn
    return engine


def _wav_file(tmp_path, name="song.wav", seconds=2.0, sr=22050):
    """A real mono WAV (sine) the DSP chain can render."""
    from nomorals.audio.dsp import write_mono_wav

    n = int(sr * seconds)
    try:
        import numpy as np

        t = np.arange(n, dtype=float) / sr
        samples = array("d", (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).tolist())
    except ImportError:
        samples = array("d", [0.0]) * n
        for i in range(n):
            samples[i] = 0.5 * math.sin(2.0 * math.pi * 440.0 * i / sr)
    path = str(tmp_path / name)
    write_mono_wav(path, samples, sr)
    return path


def _wav_seconds(path):
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


# ── defect 1: Spotify link in chat ────────────────────────────────────


def test_spotify_url_in_chat_matches_full_track_via_ladder():
    rt = _handler(_chat())
    engine = _engine_with_download(
        lambda item: {"ok": True, "path": "/tmp/yt-song.mp3",
                      "title": item.get("title", "audio")})
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine), \
            mock.patch("nomorals.media.spotify_meta.fetch_spotify_meta",
                       return_value=_meta()):
        SR.return_value.resolve.side_effect = [
            _spotify_drm_res(), _yt_res()]
        SR.return_value.search_candidates.return_value = []
        out = rt._control_play(SPOTIFY_URL, "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", SPOTIFY_URI)
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    assert "/tmp/yt-song.mp3" in rt.gateway.files[0]
    caption = rt.gateway.files[0][3]
    assert "Nightmares To Dreams" in caption
    assert "matched via" in caption
    assert "DRM" not in caption
    assert "can't be downloaded" not in caption


def test_spotify_url_no_full_match_is_honest_not_a_preview():
    rt = _handler(_chat())
    engine = _engine_with_download(
        lambda item: {"ok": False, "reason": "never called"})
    no_match = ResolvedAudio(ok=False, attempts=["youtube — no results"])
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine), \
            mock.patch("nomorals.media.spotify_meta.fetch_spotify_meta",
                       return_value=_meta()):
        SR.return_value.resolve.side_effect = [_spotify_drm_res(), no_match]
        SR.return_value.search_candidates.return_value = []
        out = rt._control_play(SPOTIFY_URL, "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", SPOTIFY_URI)
    assert job.result["sent"] is False
    assert rt.gateway.files == []
    text = job.result["text"]
    assert "no full-track source has it" in text
    assert "30s preview" in text
    assert "DRM" not in text


def test_spotify_url_unidentifiable_is_honest():
    rt = _handler(_chat())
    engine = _engine_with_download(
        lambda item: {"ok": False, "reason": "never called"})
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine), \
            mock.patch("nomorals.media.spotify_meta.fetch_spotify_meta",
                       return_value=None):
        SR.return_value.resolve.side_effect = [_spotify_drm_res()]
        out = rt._control_play(SPOTIFY_URL, "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", SPOTIFY_URI)
    assert job.result["sent"] is False
    assert "couldn't identify that Spotify link" in job.result["text"]


# ── defect 2: version requests ────────────────────────────────────────


def test_version_parse_forms():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin

    parse = RuntimeMediaMixin._play_parse_version
    assert parse("nightmares to dreams reverb") == (
        "reverb", "nightmares to dreams")
    assert parse("reverb nightmares to dreams") == (
        "reverb", "nightmares to dreams")
    assert parse("nightmares to dreams nightcore") == (
        "nightcore", "nightmares to dreams")
    assert parse("nightmares to dreams slowed + reverb") == (
        "slowed + reverb", "nightmares to dreams")
    assert parse("nightmares to dreams slowed reverb version") == (
        "slowed + reverb", "nightmares to dreams")
    assert parse("nightmares to dreams sped up") == (
        "sped up", "nightmares to dreams")
    assert parse("nightmares to dreams spedup") == (
        "sped up", "nightmares to dreams")
    assert parse("nightmares to dreams reverb version") == (
        "reverb", "nightmares to dreams")
    assert parse("nightmares to dreams") == (None, "nightmares to dreams")
    assert parse("reverb") == (None, "reverb")


def test_play_reverb_trailing_pulls_sourced_version_first():
    rt = _handler(_chat())
    engine = _engine_with_download(
        lambda item: {"ok": True, "path": item.get("path", ""),
                      "title": item.get("title", "audio")})
    orig = _yt_res()
    reverb_cand = _yt_cand(
        title="Nightmares To Dreams (Reverb Version)",
        path="https://www.youtube.com/watch?v=reverbver1")
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = orig
        SR.return_value.search_candidates_detailed.return_value = (
            [reverb_cand], ["youtube-search — ok (1)"])
        out = rt._control_play("nightmares to dreams reverb", "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", "Nightmares To Dreams")
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    assert "reverbver1" in rt.gateway.files[0][2]
    caption = rt.gateway.files[0][3]
    assert "reverb" in caption.lower()
    assert "pulled from" in caption.lower()


def test_play_reverb_leading_form():
    rt = _handler(_chat())
    engine = _engine_with_download(
        lambda item: {"ok": True, "path": item.get("path", ""),
                      "title": item.get("title", "audio")})
    orig = _yt_res()
    reverb_cand = _yt_cand(
        title="Nightmares To Dreams (Reverb Version)",
        path="https://www.youtube.com/watch?v=reverbver1")
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = orig
        SR.return_value.search_candidates_detailed.return_value = (
            [reverb_cand], ["youtube-search — ok (1)"])
        out = rt._control_play("reverb nightmares to dreams", "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", "Nightmares To Dreams")
    assert job.result["sent"] is True
    assert "reverbver1" in rt.gateway.files[0][2]
    caption = rt.gateway.files[0][3]
    # A REQUESTED version is the answer, never a consolation: the
    # caption names it as pulled, never as "the original wasn't
    # obtainable" (the old code's edit label for this same delivery).
    assert "pulled from" in caption.lower()
    assert "wasn't obtainable" not in caption


def test_version_request_is_not_demoted():
    """A requested edit is the answer even when the original is
    obtainable — the edit-demotion sort must not bury it."""
    rt = _handler(_chat())
    engine = _engine_with_download(
        lambda item: {"ok": True, "path": item.get("path", ""),
                      "title": item.get("title", "audio")})
    orig = _yt_res()
    orig_cand = _yt_cand(title="Nightmares To Dreams")
    reverb_cand = _yt_cand(
        title="Nightmares To Dreams (Reverb Version)",
        path="https://www.youtube.com/watch?v=reverbver1")
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = orig
        SR.return_value.search_candidates_detailed.return_value = (
            [orig_cand, reverb_cand], ["youtube-search — ok (2)"])
        out = rt._control_play("nightmares to dreams reverb", "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", "Nightmares To Dreams")
    assert job.result["sent"] is True
    assert "reverbver1" in rt.gateway.files[0][2]


def test_play_nightcore_renders_with_dsp_when_no_sourced_version(
        tmp_path):
    rt = _handler(_chat())
    wav = _wav_file(tmp_path)

    def _dl(item):
        if item.get("kind") == "file":
            return {"ok": True, "path": item["path"],
                    "title": item.get("title", "audio")}
        return {"ok": True, "path": wav, "title": "Nightmares To Dreams"}

    engine = _engine_with_download(_dl)
    orig = _yt_res()
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = orig
        SR.return_value.search_candidates_detailed.return_value = (
            [], ["youtube-search — no results"])
        out = rt._control_play("nightmares to dreams nightcore",
                               "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", "Nightmares To Dreams")
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    sent_path = rt.gateway.files[0][2]
    assert "nightcore" in sent_path.lower()
    assert sent_path != wav
    assert os.path.isfile(sent_path)
    # Nightcore is tempo-up: the render is SHORTER than the original.
    assert _wav_seconds(sent_path) < _wav_seconds(wav) * 0.9
    caption = rt.gateway.files[0][3]
    assert "nightcore version" in caption.lower()
    assert "rendered" in caption.lower()


def test_play_slowed_renders_slower(tmp_path):
    rt = _handler(_chat())
    wav = _wav_file(tmp_path)

    def _dl(item):
        if item.get("kind") == "file":
            return {"ok": True, "path": item["path"],
                    "title": item.get("title", "audio")}
        return {"ok": True, "path": wav, "title": "Nightmares To Dreams"}

    engine = _engine_with_download(_dl)
    orig = _yt_res()
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = orig
        SR.return_value.search_candidates_detailed.return_value = (
            [], ["youtube-search — no results"])
        out = rt._control_play("nightmares to dreams slowed", "chat-key")
    assert out == ""
    job = _collect_play_job("chat-key", "Nightmares To Dreams")
    assert job.result["sent"] is True
    sent_path = rt.gateway.files[0][2]
    assert "slowed" in sent_path.lower()
    assert os.path.isfile(sent_path)
    # Slowed is tempo-down: the render is LONGER than the original.
    assert _wav_seconds(sent_path) > _wav_seconds(wav) * 1.1
    assert "slowed version" in rt.gateway.files[0][3].lower()


def test_spotify_url_with_version_renders(tmp_path):
    rt = _handler(_chat())
    # The matched item carries the Spotify duration hint (200s), so
    # the fixture file must BE a full-length track or the acceptance
    # contract rightly rejects it as a preview clip.
    wav = _wav_file(tmp_path, seconds=200.0)

    def _dl(item):
        if item.get("kind") == "file":
            return {"ok": True, "path": item["path"],
                    "title": item.get("title", "audio")}
        return {"ok": True, "path": wav, "title": "Nightmares To Dreams"}

    engine = _engine_with_download(_dl)
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine), \
            mock.patch("nomorals.media.spotify_meta.fetch_spotify_meta",
                       return_value=_meta()):
        SR.return_value.resolve.side_effect = [
            _spotify_drm_res(), _yt_res()]
        SR.return_value.search_candidates.return_value = []
        out = rt._control_play(f"{SPOTIFY_URL} reverb", "chat-key")
    assert out == ""
    # The 200s reverb render is CPU-bound (~18s here) — the
    # default 10s collect deadline is too short for it.
    job = _collect_play_job("chat-key", SPOTIFY_URI, timeout=60.0)
    assert job.result["sent"] is True
    assert len(rt.gateway.files) == 1
    sent_path = rt.gateway.files[0][2]
    assert "reverb" in sent_path.lower()
    assert os.path.isfile(sent_path)
    assert "reverb version" in rt.gateway.files[0][3].lower()


# ── CLI: a version request is never silently dropped either ──────────

def test_cli_version_renders_instead_of_playing_original(tmp_path):
    rt = _handler(_chat())
    wav = _wav_file(tmp_path)

    def _dl(item):
        return {"ok": True, "path": wav, "title": item.get("title", "audio")}

    engine = _engine_with_download(_dl)
    engine.queue.return_value = [{"title": "x"}]
    with mock.patch("nomorals.media.resolver.SourceResolver") as SR, \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       return_value=engine):
        SR.return_value.resolve.return_value = _yt_res()
        out = rt._control_play("nightmares to dreams reverb", None)
    assert "reverb version" in out.lower()
    assert "rendered" in out.lower()
    enqueued = engine._enqueue.call_args[0]
    assert "reverb" in enqueued[2].lower()
    assert enqueued[0].endswith(".wav")


# ── dsp: varispeed is the honest nightcore/slowed primitive ───────────


def test_varispeed_effect_registered_and_moves_tempo(tmp_path):
    from nomorals.audio.dsp import EffectChain, read_mono_wav

    wav = _wav_file(tmp_path)
    samples, sr = read_mono_wav(wav)
    fast = EffectChain([("varispeed", {"factor": 1.25})]).run(samples, sr)
    assert len(fast) < len(samples) * 0.85
    slow = EffectChain([("varispeed", {"factor": 0.8})]).run(samples, sr)
    assert len(slow) > len(samples) * 1.2


def test_apply_fx_varispeed_chain(tmp_path):
    from nomorals.audio.edit import apply_fx

    wav = _wav_file(tmp_path)
    res = apply_fx(wav, "varispeed factor=1.25, normalize",
                   suffix="nightcore")
    assert res["ok"], res
    assert res["output"].endswith("-nightcore.wav")
    assert _wav_seconds(res["output"]) < _wav_seconds(wav) * 0.9
