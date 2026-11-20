"""YouTube multi-search fix — regression tests (lane 23).

Owner live report 2026-11-17: both YouTube legs died on /play
"tell me one lie" — (1) yt-dlp's bot check on an unauthenticated
search, (2) ``ytmsearch8:`` rejected as an unknown scheme.

The defects, pinned here:
- ``ytmsearch`` is not a scheme current yt-dlp registers (no
  ``youtube:music:search`` extractor; the supported route is the
  music search URL). The Music leg must search
  music.youtube.com, not build ``ytmsearchN:``.
- The YouTube search legs used the user-file-only cookie helpers,
  so they ran cookieless even when Devon's own browser-profile
  estate holds cookies. They must use the per-URL helpers (the
  ones the download paths already use).
- A search fetched every result's video page (non-flat), which is
  the surface the bot check hit. Searches must be flat — ids and
  titles only, no per-video page fetches.
"""

import sys
import types

import pytest

from nomorals.media import download_failures as _df
from nomorals.media.playback import PlaybackEngine


@pytest.fixture(autouse=True)
def _isolated_download_failure_home(tmp_path, monkeypatch):
    # The YouTube search legs now record a bot check in the durable
    # search-failure memory — tests that simulate one must not share
    # a home with the CLI tests, or the recorded block gates every
    # later search in the run.
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    _df.clear_failures()
    yield
    _df.clear_failures()


class _FakeYDL:
    """Stand-in for yt_dlp.YoutubeDL: captures opts + target."""

    seen_opts: list = []
    seen_targets: list = []
    entries: list = []
    error: Exception | None = None

    def __init__(self, opts):
        self._opts = dict(opts or {})
        _FakeYDL.seen_opts.append(self._opts)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, target, download=False):
        _FakeYDL.seen_targets.append(target)
        if _FakeYDL.error is not None:
            raise _FakeYDL.error
        return {"entries": list(_FakeYDL.entries)}


def _install_fake_ydl(monkeypatch, entries=None, error=None):
    fake_module = types.ModuleType("yt_dlp")
    fake_module.YoutubeDL = _FakeYDL
    _FakeYDL.seen_opts.clear()
    _FakeYDL.seen_targets.clear()
    _FakeYDL.entries = list(entries or [])
    _FakeYDL.error = error
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_module)
    return fake_module


def _yt_entries():
    return [
        {"id": "oQQ84L-b0Ss", "title": "Tell Me One Lie",
         "duration": 161, "uploader": "Ronis Goliath"},
        {"id": "dQw4w9WgXcQ", "title": "Other Track",
         "duration": 200, "uploader": "Someone"},
    ]


def test_ytmsearch_builds_music_search_url(monkeypatch):
    """The Music leg searches music.youtube.com — never ytmsearchN:."""
    _install_fake_ydl(monkeypatch, _yt_entries())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    out = PlaybackEngine._youtube_search_many(
        "tell me one lie", limit=8, prefix="ytmsearch")
    assert _FakeYDL.seen_targets, "yt-dlp was never called"
    target = _FakeYDL.seen_targets[0]
    assert "ytmsearch" not in target
    assert target == (
        "https://music.youtube.com/search?q=tell+me+one+lie")
    assert out and out[0]["video_id"] == "oQQ84L-b0Ss"


def test_ytsearch_keeps_prefix_form(monkeypatch):
    """Plain YouTube search keeps the documented <prefix>N: form."""
    _install_fake_ydl(monkeypatch, _yt_entries())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    out = PlaybackEngine._youtube_search_many(
        "tell me one lie", limit=8, prefix="ytsearch")
    assert _FakeYDL.seen_targets[0] == "ytsearch8:tell me one lie"
    assert out and out[0]["video_id"] == "oQQ84L-b0Ss"


def test_search_uses_estate_cookies(monkeypatch):
    """Search hands yt-dlp the per-URL cookies (estate export included)."""
    _install_fake_ydl(monkeypatch, _yt_entries())
    seen_urls: list = []

    def _cookie_opts(url):
        seen_urls.append(url)
        return {"cookiefile": "/tmp/estate/browser-cookies.txt"}

    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for", _cookie_opts)
    PlaybackEngine._youtube_search_many("tell me one lie", limit=8)
    assert _FakeYDL.seen_opts, "YoutubeDL was never constructed"
    assert _FakeYDL.seen_opts[0].get("cookiefile") == (
        "/tmp/estate/browser-cookies.txt")
    assert seen_urls and "youtube.com" in seen_urls[0]


def test_search_is_flat(monkeypatch):
    """Search never fetches result video pages (bot-check surface)."""
    _install_fake_ydl(monkeypatch, _yt_entries())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    PlaybackEngine._youtube_search_many("tell me one lie", limit=8)
    assert _FakeYDL.seen_opts, "YoutubeDL was never constructed"
    assert _FakeYDL.seen_opts[0].get("extract_flat") is True


def test_search_id_ytmsearch_uses_music_url(monkeypatch):
    """Single-result Music search also takes the URL route, flat."""
    _install_fake_ydl(monkeypatch, _yt_entries())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    vid = PlaybackEngine._youtube_search_id(
        "tell me one lie", prefix="ytmsearch")
    assert vid == "oQQ84L-b0Ss"
    assert _FakeYDL.seen_targets[0] == (
        "https://music.youtube.com/search?q=tell+me+one+lie")
    assert _FakeYDL.seen_opts[0].get("extract_flat") is True


def test_search_id_ytsearch_unchanged(monkeypatch):
    """Single-result plain search keeps the ytsearch1: form."""
    _install_fake_ydl(monkeypatch, _yt_entries())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    vid = PlaybackEngine._youtube_search_id("tell me one lie")
    assert vid == "oQQ84L-b0Ss"
    assert _FakeYDL.seen_targets[0] == "ytsearch1:tell me one lie"


def test_music_flat_entries_without_duration_still_land(monkeypatch):
    """Music search flat entries carry id+title only — still rows."""
    _install_fake_ydl(monkeypatch, [
        {"id": "oQQ84L-b0Ss", "title": "Tell Me One Lie"},
        {"id": "dQw4w9WgXcQ", "title": "Other Track"},
    ])
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    out = PlaybackEngine._youtube_search_many(
        "tell me one lie", limit=8, prefix="ytmsearch")
    assert [r["video_id"] for r in out] == ["oQQ84L-b0Ss", "dQw4w9WgXcQ"]
    assert out[0]["duration"] == 0.0
    assert out[0]["uploader"] == ""


def test_bot_check_search_returns_empty_not_raise(monkeypatch):
    """A bot-checked search fails empty (never raises, never fakes)."""
    from nomorals.media.cookies import is_bot_detection_error

    _install_fake_ydl(
        monkeypatch, error=Exception(
            "ERROR: [youtube] oQQ84L-b0Ss: Sign in to confirm "
            "you're not a bot"))
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_opts_for",
        lambda url: {})
    out = PlaybackEngine._youtube_search_many("tell me one lie", limit=8)
    assert out == []
    assert is_bot_detection_error(
        "Sign in to confirm you're not a bot") is True


def test_cli_search_is_flat_and_cookied(monkeypatch, tmp_path):
    """CLI fallback: flat playlist, estate cookies, music URL route."""
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/yt-dlp")
    seen_argv: list = []

    class _Proc:
        returncode = 0
        stdout = ("oQQ84L-b0Ss\tTell Me One Lie\t161\tRonis Goliath\n")
        stderr = ""

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: seen_argv.append(a[0]) or _Proc())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_args_for",
        lambda url: ["--cookies", "/tmp/estate/browser-cookies.txt"])
    out = PlaybackEngine._youtube_search_many(
        "tell me one lie", limit=8, prefix="ytmsearch")
    assert seen_argv, "yt-dlp CLI was never invoked"
    argv = seen_argv[0]
    assert "--flat-playlist" in argv
    assert argv[argv.index("--cookies") + 1] == (
        "/tmp/estate/browser-cookies.txt")
    assert argv[-1] == (
        "https://music.youtube.com/search?q=tell+me+one+lie")
    assert out and out[0]["video_id"] == "oQQ84L-b0Ss"


def test_cli_search_id_music_url(monkeypatch):
    """CLI single-result Music search takes the URL route."""
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/yt-dlp")
    seen_argv: list = []

    class _Proc:
        returncode = 0
        stdout = "oQQ84L-b0Ss\n"
        stderr = ""

    monkeypatch.setattr(
        "subprocess.run",
        lambda *a, **k: seen_argv.append(a[0]) or _Proc())
    monkeypatch.setattr(
        "nomorals.media.cookies.ytdlp_cookie_args_for",
        lambda url: [])
    vid = PlaybackEngine._youtube_search_id(
        "tell me one lie", prefix="ytmsearch")
    assert vid == "oQQ84L-b0Ss"
    assert seen_argv[0][-1] == (
        "https://music.youtube.com/search?q=tell+me+one+lie")
    assert "--flat-playlist" in seen_argv[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
