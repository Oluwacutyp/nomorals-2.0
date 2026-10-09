"""Tests for the SoundCloud DRM → YouTube download fallback.

- SoundCloud DRM/protection errors trigger a one-shot YouTube fallback
  (reusing the known artist/title) instead of failing outright.
- YouTube also failing → honest combined error.
- Non-DRM errors (network, missing deps) do NOT trigger the fallback.
- Never raises on garbage input.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from nomorals.media.playback import is_protection_error


# ── is_protection_error ──────────────────────────────────────────────


class TestIsProtectionError:
    def test_drm(self):
        assert is_protection_error("ERROR: [soundcloud] 123: This video is DRM protected")

    def test_protected(self):
        assert is_protection_error("track is protected content")

    def test_not_downloadable(self):
        assert is_protection_error("This track is not downloadable")

    def test_private(self):
        assert is_protection_error("private track, login required")

    def test_case_insensitive(self):
        assert is_protection_error("DRM PROTECTED")

    def test_network_error_not_protection(self):
        assert not is_protection_error("connection timed out")

    def test_missing_dep_not_protection(self):
        assert not is_protection_error("pip install yt-dlp")

    def test_none_and_empty(self):
        assert not is_protection_error(None)
        assert not is_protection_error("")


# ── download() fallback ──────────────────────────────────────────────


def _engine(download_result=None, download_exc=None,
            yt_search_id="dQw4w9WgXcQ", yt_audio_path="/tmp/song.mp3",
            yt_audio_exc=None):
    """PlaybackEngine with stubbed context/tools and YouTube methods."""
    ctx = MagicMock()
    tools = MagicMock()

    def fake_call(name, **kwargs):
        if download_exc is not None:
            raise download_exc
        return SimpleNamespace(ok=True, value={"path": "/tmp/sc.mp3"})

    tools.call = fake_call
    ctx.tools = tools

    from nomorals.media.playback import PlaybackEngine
    engine = PlaybackEngine.__new__(PlaybackEngine)
    engine.context = ctx
    engine._soundcloud_play_url = lambda item: "https://soundcloud.com/x/y"  # noqa: E731
    engine._youtube_search_id = lambda query: yt_search_id  # noqa: E731
    if yt_audio_exc is not None:
        def _raise(item):
            raise yt_audio_exc
        engine._youtube_audio = _raise
    else:
        engine._youtube_audio = lambda item: yt_audio_path  # noqa: E731
    return engine


def _sc_item():
    return {"kind": "soundcloud", "path": "https://soundcloud.com/a/b",
            "title": "Lifestyle (YA MAN)", "artist": "Ayo Maff"}


class TestDrmFallback:
    def test_drm_error_triggers_youtube_fallback(self):
        engine = _engine(download_exc=Exception(
            "[soundcloud] 2361175013: This video is DRM protected"))
        res = engine.download(_sc_item())
        assert res["ok"] is True
        assert res["path"] == "/tmp/song.mp3"
        assert "YouTube" in res.get("note", "")

    def test_drm_error_from_failed_result_triggers_fallback(self):
        ctx = MagicMock()
        tools = MagicMock()
        tools.call = lambda name, **kw: SimpleNamespace(
            ok=False, error="This track is DRM protected")
        ctx.tools = tools
        from nomorals.media.playback import PlaybackEngine
        engine = PlaybackEngine.__new__(PlaybackEngine)
        engine.context = ctx
        engine._soundcloud_play_url = lambda item: "https://soundcloud.com/x/y"  # noqa: E731
        engine._youtube_search_id = lambda query: "dQw4w9WgXcQ"  # noqa: E731
        engine._youtube_audio = lambda item: "/tmp/song.mp3"  # noqa: E731
        res = engine.download(_sc_item())
        assert res["ok"] is True

    def test_youtube_also_failing_gives_honest_error(self):
        engine = _engine(
            download_exc=Exception("DRM protected"),
            yt_audio_exc=Exception("video unavailable"))
        res = engine.download(_sc_item())
        assert res["ok"] is False
        assert "SoundCloud" in res["reason"]
        assert "YouTube" in res["reason"]

    def test_youtube_search_failing_gives_honest_error(self):
        from nomorals.media import playback as pb

        ctx = MagicMock()
        tools = MagicMock()

        def boom(name, **kw):
            raise Exception("DRM protected")
        tools.call = boom
        ctx.tools = tools
        engine = pb.PlaybackEngine.__new__(pb.PlaybackEngine)
        engine.context = ctx
        engine._soundcloud_play_url = lambda item: "https://soundcloud.com/x/y"  # noqa: E731

        def no_search(query):
            raise Exception("no results")
        engine._youtube_search_id = no_search
        res = engine.download(_sc_item())
        assert res["ok"] is False
        assert "YouTube search failed" in res["reason"]

    def test_non_drm_error_does_not_trigger_fallback(self):
        engine = _engine(download_exc=Exception("connection timed out"))
        res = engine.download(_sc_item())
        assert res["ok"] is False
        assert "download failed" in res["reason"]
        assert "YouTube" not in res["reason"]

    def test_non_soundcloud_kind_no_fallback(self):
        ctx = MagicMock()
        tools = MagicMock()

        def boom(name, **kw):
            raise Exception("DRM protected")
        tools.call = boom
        ctx.tools = tools
        from nomorals.media.playback import PlaybackEngine
        engine = PlaybackEngine.__new__(PlaybackEngine)
        engine.context = ctx
        engine._youtube_search_id = lambda query: "dQw4w9WgXcQ"  # noqa: E731
        res = engine.download({"kind": "url", "path": "https://example.com/x.mp3",
                               "title": "x"})
        assert res["ok"] is False
        assert "download failed" in res["reason"]

    def test_never_raises_on_garbage(self):
        from nomorals.media.playback import PlaybackEngine
        engine = PlaybackEngine.__new__(PlaybackEngine)
        engine.context = MagicMock()
        for bad in ({}, {"kind": None}, {"kind": "soundcloud"}):
            res = engine.download(bad)
            assert isinstance(res, dict)
            assert "ok" in res
