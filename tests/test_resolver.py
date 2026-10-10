"""Tests for nomorals.media.resolver — the dynamic source strategy chain.

- Strategy order: local file → yt-dlp (universal) → SoundCloud API
  (with yt-dlp fallback) → text search
- SoundCloud API failure falls back to yt-dlp instead of erroring
- All-strategies-fail returns an honest result (never raises)
- Missing yt-dlp → install hint, not a crash
- Never raises on garbage input
"""

from types import SimpleNamespace

import pytest

from nomorals.media import resolver
from nomorals.media.resolver import (
    ResolutionError,
    ResolvedAudio,
    SourceResolver,
    YT_DLP_HINT,
)


# ── doubles ───────────────────────────────────────────────────────────


class FakeSoundCloud:
    """SoundCloud adapter double: fails or returns canned tracks."""

    def __init__(self, fail_resolve=False, tracks=None):
        self.fail_resolve = fail_resolve
        self.tracks = tracks or []
        self.resolve_calls = []
        self.search_calls = []

    def resolve(self, url):
        self.resolve_calls.append(url)
        if self.fail_resolve:
            raise RuntimeError("api-v2 is down (simulated)")
        return {"kind": "track", "track": self.tracks[0]}

    def search_tracks(self, query, limit=1):
        self.search_calls.append(query)
        return self.tracks[:limit]


class FakeSpotify:
    def __init__(self, tracks=None):
        self.tracks = tracks or []

    def search(self, query, types=None, limit=1):
        return {"tracks": {"items": self.tracks[:limit]}}


def _track(permalink="https://soundcloud.com/artist/song",
           title="Song", artist="Artist", ms=180000):
    return {"permalink_url": permalink, "title": title,
            "artist": artist, "duration_ms": ms}


def _ctx(workspace_dir=""):
    return SimpleNamespace(
        settings=SimpleNamespace(workspace_dir=workspace_dir))


# ── never-raises ──────────────────────────────────────────────────────


def test_never_raises_on_garbage():
    r = SourceResolver(None)
    for bad in [None, "", "   ", 123, 4.5, object(), ["x"], {"q": 1}]:
        out = r.resolve(bad)
        assert isinstance(out, ResolvedAudio), f"no result for {bad!r}"
        assert not out.ok, f"garbage resolved ok: {bad!r}"
        assert out.attempts, "attempts must be recorded"


def test_empty_query_hint():
    out = SourceResolver(None).resolve("   ")
    assert not out.ok
    assert "usage" in out.hint.lower()


# ── strategy order: local file first ──────────────────────────────────


def test_local_file_wins_without_network(tmp_path):
    audio = tmp_path / "tune.mp3"
    audio.write_bytes(b"fake")
    sc = FakeSoundCloud(tracks=[_track()])
    out = SourceResolver(_ctx(str(tmp_path)),
                         soundcloud=sc).resolve(str(audio))
    assert out.ok
    assert out.kind == "file"
    assert out.path_or_url == str(audio)
    assert len(out.attempts) == 1
    assert "local-file" in out.attempts[0]
    assert sc.resolve_calls == [] and sc.search_calls == []


def test_path_like_but_missing_falls_through_to_search(tmp_path):
    sc = FakeSoundCloud(tracks=[_track(title="Found It")])
    out = SourceResolver(_ctx(str(tmp_path)),
                         soundcloud=sc).resolve("music/missing-song.mp3")
    # not a real file → treated as a title → SoundCloud search hits
    assert out.ok
    assert out.kind == "soundcloud"
    assert out.title == "Found It"
    assert any("local-file" in a and "failed" in a for a in out.attempts)


def test_workspace_scan_matches_title(tmp_path):
    (tmp_path / "midnight-drive.mp3").write_bytes(b"x")
    (tmp_path / "unrelated.wav").write_bytes(b"x")
    out = SourceResolver(_ctx(str(tmp_path))).resolve("midnight drive")
    assert out.ok
    assert out.kind == "file"
    assert out.path_or_url.endswith("midnight-drive.mp3")
    assert "workspace scan" in out.source_name


# ── SoundCloud API failure → yt-dlp fallback (the key behavior) ───────


def test_soundcloud_api_failure_falls_back_to_ytdlp(monkeypatch):
    sc = FakeSoundCloud(fail_resolve=True)

    def fake_probe(url):
        assert "soundcloud.com" in url
        return {"id": "12345", "title": "Cool Track",
                "uploader": "DJ X", "duration": 200.0,
                "extractor": "soundcloud",
                "webpage_url": url}

    monkeypatch.setattr(resolver, "probe", fake_probe)
    out = SourceResolver(None, soundcloud=sc).resolve(
        "https://soundcloud.com/artist/cool-track")
    assert out.ok, f"errors: {out.errors}"
    assert out.kind == "soundcloud"
    assert "yt-dlp" in out.source_name
    assert out.title == "Cool Track"
    # the API was tried and failed, yt-dlp picked it up
    assert any("soundcloud-api" in a and "failed" in a
               for a in out.attempts), out.attempts
    assert any(a.startswith("yt-dlp") and "ok" in a
               for a in out.attempts), out.attempts


def test_soundcloud_api_success_uses_api_not_ytdlp(monkeypatch):
    sc = FakeSoundCloud(tracks=[_track(title="API Track")])

    def fake_probe(url):  # pragma: no cover - must not be called
        raise AssertionError("yt-dlp should not run when API works")

    monkeypatch.setattr(resolver, "probe", fake_probe)
    out = SourceResolver(None, soundcloud=sc).resolve(
        "https://soundcloud.com/artist/api-track")
    assert out.ok
    assert out.source_name == "SoundCloud API"
    assert out.title == "API Track"


# ── yt-dlp universal routing (no per-domain branches) ──────────────────


def test_ytdlp_routes_unknown_domain_dynamically(monkeypatch):
    # Bandcamp has zero dedicated code — yt-dlp handles it.
    def fake_probe(url):
        return {"id": "bc1", "title": "Indie Gem", "duration": 240.0,
                "extractor": "bandcamp", "webpage_url": url}

    monkeypatch.setattr(resolver, "probe", fake_probe)
    out = SourceResolver(None).resolve(
        "https://artist.bandcamp.com/track/indie-gem")
    assert out.ok
    assert out.kind == "url"
    assert "yt-dlp (bandcamp)" in out.source_name


def test_ytdlp_youtube_url_becomes_youtube_kind(monkeypatch):
    def fake_probe(url):
        return {"id": "dQw4w9WgXcQ", "title": "Famous Video",
                "duration": 212.0, "extractor": "youtube",
                "webpage_url": url}

    monkeypatch.setattr(resolver, "probe", fake_probe)
    out = SourceResolver(None).resolve(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert out.ok
    assert out.kind == "youtube"
    assert out.path_or_url == \
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def test_youtube_search_via_ytdlp(monkeypatch):
    from nomorals.media.playback import PlaybackEngine

    monkeypatch.setattr(
        PlaybackEngine, "_youtube_search_id",
        classmethod(lambda cls, q: "dQw4w9WgXcQ"))
    out = SourceResolver(_ctx("/nonexistent-ws")).resolve(
        "never gonna give you up")
    assert out.ok
    assert out.kind == "youtube"
    assert "dQw4w9WgXcQ" in out.path_or_url


# ── missing yt-dlp → install hint ─────────────────────────────────────


def test_missing_ytdlp_gives_install_hint(monkeypatch, tmp_path):
    def fake_probe(url):
        raise RuntimeError(
            "could not probe: install yt-dlp for site extraction")

    monkeypatch.setattr(resolver, "probe", fake_probe)
    out = SourceResolver(_ctx(str(tmp_path))).resolve(
        "https://example.com/some-track")
    assert not out.ok
    assert "pip install yt-dlp" in out.hint
    assert out.hint == YT_DLP_HINT


def test_garbage_url_fails_all_strategies_honestly(monkeypatch, tmp_path):
    def fake_probe(url):
        raise RuntimeError("unsupported URL")

    monkeypatch.setattr(resolver, "probe", fake_probe)
    out = SourceResolver(_ctx(str(tmp_path))).resolve(
        "https://example.com/not-a-song-page")
    assert not out.ok
    # every URL strategy was tried and recorded
    assert len(out.attempts) >= 3, out.attempts
    assert len(out.errors) == len(out.attempts)
    assert out.hint  # honest guidance, not silence


def test_unknown_title_fails_all_strategies_honestly(monkeypatch, tmp_path):
    from nomorals.media.playback import PlaybackEngine

    def fake_probe(url):  # pragma: no cover - not a URL path
        raise AssertionError("probe must not run for titles")

    monkeypatch.setattr(resolver, "probe", fake_probe)
    monkeypatch.setattr(
        PlaybackEngine, "_youtube_search_id",
        classmethod(lambda cls, q: (_ for _ in ()).throw(
            RuntimeError("youtube search needs yt-dlp"))))
    sc = FakeSoundCloud(tracks=[])  # search finds nothing
    out = SourceResolver(_ctx(str(tmp_path)),
                         soundcloud=sc).resolve("xyqz totally unknown 999")
    assert not out.ok
    tried = " ".join(out.attempts)
    for name in ("workspace-scan", "soundcloud-search",
                 "youtube-search", "spotify-search"):
        assert name in tried, tried
    assert "pip install yt-dlp" in out.hint


# ── Spotify: honest, no extraction ────────────────────────────────────


def test_spotify_url_is_honest_not_downloadable():
    out = SourceResolver(None).resolve(
        "https://open.spotify.com/track/6r7b1UHvO3fBZe7wBXWTaZ")
    assert out.ok
    assert out.kind == "spotify"
    assert out.path_or_url == \
        "spotify:track:6r7b1UHvO3fBZe7wBXWTaZ"
    assert out.downloadable is False


def test_spotify_search_only_when_linked():
    out = SourceResolver(_ctx("/nope")).resolve("some obscure title xyz")
    # spotify adapter is None → strategy fails, recorded honestly
    assert any("spotify-search" in a for a in out.attempts)


def test_spotify_search_works_when_linked(tmp_path):
    sp = FakeSpotify(tracks=[{
        "uri": "spotify:track:abc",
        "name": "Linked Song",
        "artists": [{"name": "Some Artist"}],
        "id": "abc",
    }])
    out = SourceResolver(_ctx(str(tmp_path)), spotify=sp).resolve(
        "linked song")
    assert out.ok
    assert out.kind == "spotify"
    assert out.title == "Some Artist – Linked Song"
    assert out.downloadable is False


# ── download() SoundCloud → yt-dlp fallback ───────────────────────────


def _make_engine(tmpdir):
    from unittest.mock import MagicMock, patch
    from nomorals.media.playback import PlaybackEngine

    ctx = MagicMock()
    ctx.home = tmpdir
    ctx.db = MagicMock()
    with patch.object(PlaybackEngine, "__init__", lambda s, c, **k: None):
        engine = PlaybackEngine.__new__(PlaybackEngine)
        engine.context = ctx
    return engine, ctx


def test_download_soundcloud_api_failure_falls_back_to_ytdlp(tmp_path):
    import os
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    engine, ctx = _make_engine(tmp_path)
    audio = os.path.join(str(tmp_path), "sc_fallback.mp3")
    Path(audio).write_bytes(b"fallback-audio")
    mock_out = MagicMock()
    mock_out.ok = True
    mock_out.value = {"path": audio}
    ctx.tools.call.return_value = mock_out
    item = {"kind": "soundcloud",
            "path": "https://soundcloud.com/artist/down-track",
            "title": "Down Track"}
    with patch.object(engine, "_soundcloud_play_url",
                       side_effect=RuntimeError("api-v2 is down")):
        result = engine.download(item)
    assert result["ok"], f"should fall back to yt-dlp, got: {result}"
    assert result["path"] == audio
    # the fallback downloads the permalink itself via media_download
    called_url = ctx.tools.call.call_args[1]["url"]
    assert called_url == "https://soundcloud.com/artist/down-track"


def test_download_soundcloud_api_success_unchanged(tmp_path):
    import os
    from pathlib import Path
    from unittest.mock import MagicMock, patch

    engine, ctx = _make_engine(tmp_path)
    audio = os.path.join(str(tmp_path), "sc_ok.mp3")
    Path(audio).write_bytes(b"ok-audio")
    mock_out = MagicMock()
    mock_out.ok = True
    mock_out.value = {"path": audio}
    ctx.tools.call.return_value = mock_out
    item = {"kind": "soundcloud",
            "path": "https://soundcloud.com/artist/ok-track",
            "title": "OK Track"}
    with patch.object(engine, "_soundcloud_play_url",
                       return_value="https://stream.url/audio.mp3"):
        result = engine.download(item)
    assert result["ok"]
    called_url = ctx.tools.call.call_args[1]["url"]
    assert called_url == "https://stream.url/audio.mp3"


# ── unit: ResolutionError carries the strategy ────────────────────────


def test_resolution_error_carries_strategy():
    exc = ResolutionError("yt-dlp", "boom")
    assert exc.strategy == "yt-dlp"
    assert "boom" in str(exc)


def test_resolve_result_shape():
    out = SourceResolver(None).resolve("https://example.com/x")
    for attr in ("ok", "path_or_url", "title", "attempts",
                 "errors", "hint", "kind", "source_name",
                 "downloadable", "extra_tracks"):
        assert hasattr(out, attr), f"missing {attr}"


# ── preview-only detection + fall-through ───────────────────────────────


def test_track_result_rejects_preview_policies():
    for policy in ("PREVIEW", "SNIP", "BLOCK"):
        tr = _track(title="T", permalink="https://soundcloud.com/a/t")
        tr["policy"] = policy
        with pytest.raises(ResolutionError) as ei:
            SourceResolver._track_result(tr, "SoundCloud API")
        assert "preview" in str(ei.value).lower()
    # ALLOW and empty policy still pass
    for policy in ("ALLOW", ""):
        tr = _track(title="T", permalink="https://soundcloud.com/a/t")
        tr["policy"] = policy
        out = SourceResolver._track_result(tr, "SoundCloud API")
        assert out.ok and out.kind == "soundcloud"


def test_track_result_rejects_preview_flag():
    tr = _track(title="T", permalink="https://soundcloud.com/a/t")
    tr["preview"] = True
    with pytest.raises(ResolutionError):
        SourceResolver._track_result(tr, "SoundCloud API")


def test_track_list_skips_preview_only_first_track():
    sc_tracks = [
        {"permalink_url": "https://soundcloud.com/a/preview",
         "title": "Preview", "policy": "PREVIEW", "duration_ms": 30000},
        {"permalink_url": "https://soundcloud.com/a/full",
         "title": "Full", "policy": "ALLOW", "duration_ms": 180000},
    ]
    r = SourceResolver(None)
    out = r._track_list_result(
        sc_tracks, "https://soundcloud.com/a/playlist", "SoundCloud API")
    assert out.ok and out.title == "Full"
    assert out.extra_tracks == []


def test_track_list_all_preview_fails_honestly():
    sc_tracks = [
        {"permalink_url": "https://soundcloud.com/a/p1",
         "title": "P1", "policy": "PREVIEW"},
    ]
    r = SourceResolver(None)
    with pytest.raises(ResolutionError) as ei:
        r._track_list_result(sc_tracks, "https://soundcloud.com/a/pl",
                             "SoundCloud API")
    assert "preview-only" in str(ei.value)


def test_info_is_preview_only_signal():
    from nomorals.media.resolver import _info_is_preview_only
    assert not _info_is_preview_only(
        {"formats": [{"url": "https://cf-hls-media.sndcdn.com/abc.m3u8"}]})
    assert _info_is_preview_only(
        {"formats": [{"url": "https://preview.sndcdn.com/preview/abcd.mp3"}]})
    assert _info_is_preview_only(
        {"url": "https://api-v2.soundcloud.com/preview/xyz"})
    assert not _info_is_preview_only({})


def test_ytdlp_soundcloud_preview_falls_back_to_youtube(monkeypatch):
    from nomorals.media import playback as playback_mod
    sc = FakeSoundCloud(fail_resolve=True)

    def fake_probe(url):
        return {"id": "12345", "title": "Cool Track",
                "uploader": "DJ X", "duration": 30.0,
                "extractor": "soundcloud", "webpage_url": url,
                "formats": [
                    {"url": "https://preview.sndcdn.com/preview/x.mp3"}]}

    def fake_yt_search(cls, query):
        assert "Cool Track" in query
        return "dQw4w9WgXcQ"

    monkeypatch.setattr(resolver, "probe", fake_probe)
    monkeypatch.setattr(playback_mod.PlaybackEngine, "_youtube_search_id",
                        classmethod(fake_yt_search))
    out = SourceResolver(None, soundcloud=sc).resolve(
        "https://soundcloud.com/artist/cool-track")
    assert out.ok, f"errors: {out.errors}"
    assert out.kind == "youtube"
    assert "preview" in out.source_name.lower()
    assert "youtube.com/watch?v=dQw4w9WgXcQ" in out.path_or_url
    assert "preview" in out.hint.lower()


def test_ytdlp_soundcloud_preview_no_youtube_match_fails_honestly(
        monkeypatch):
    from nomorals.media import playback as playback_mod
    sc = FakeSoundCloud(fail_resolve=True)

    def fake_probe(url):
        return {"id": "12345", "title": "Obscure Track",
                "uploader": "Nobody", "duration": 30.0,
                "extractor": "soundcloud", "webpage_url": url,
                "formats": [
                    {"url": "https://preview.sndcdn.com/preview/x.mp3"}]}

    def fake_yt_search(cls, query):
        return ""  # no match

    monkeypatch.setattr(resolver, "probe", fake_probe)
    monkeypatch.setattr(playback_mod.PlaybackEngine, "_youtube_search_id",
                        classmethod(fake_yt_search))
    out = SourceResolver(None, soundcloud=sc).resolve(
        "https://soundcloud.com/artist/obscure-track")
    assert not out.ok
    assert any("preview" in e.lower() for e in out.errors), out.errors
