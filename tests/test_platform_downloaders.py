"""FEI gap wave A — dedicated platform downloaders.

Pins:
- URL→platform detection incl. shortlinks (vm./vt.tiktok.com,
  instagr.am, spoti.fi, youtu.be) and lookalike-host rejection;
- TikTok: yt-dlp route with watermark truth from the probe (the
  extractor marks download renditions 'watermarked'; play
  renditions are clean), tikwm fallback, per-route honest failure;
- Instagram: public success + login wall named as a wall;
- Spotify: link → metadata (embed/oEmbed shapes) → full-track match
  through the /play ladder, labeled by its real source; playlist
  cap + matched/failed summary; a previews-only world yields the
  honest absence and NO delivery;
- /aio routing per platform + unsupported-link honesty;
- registry presence: the commands parse, dispatch, and appear in
  the WhatsApp-visible catalog.

All network is faked at the seams (probe/api/download callables and
module getters) with payloads shaped from the real tools' outputs:
yt-dlp's TikTok format dicts, the tikwm API JSON, Spotify's embed
__NEXT_DATA__ entity and oEmbed JSON.  Never raises, never network.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from nomorals.media.downloader import DownloadReport, StageLog

# ── shared fixtures ─────────────────────────────────────────────────

TT_URL = "https://www.tiktok.com/@dancer/video/7234567890123456789"
IG_URL = "https://www.instagram.com/reel/CabcDEF123/"
SP_URL = "https://open.spotify.com/track/0TnOYISbd1XYRBk9myaseg"

#: yt-dlp TikTok probe info, app-path shape: play_addr renditions
#: are the clean stream; download_addr is marked watermarked by the
#: extractor itself.
YTDLP_TT_CLEAN = {
    "id": "7234567890123456789",
    "title": "some dance #fyp",
    "duration": 21,
    "extractor": "TikTok",
    "formats": [
        {"format_id": "play_addr", "format_note": "Direct video",
         "url": "https://v16.tiktokcdn.com/play/aa.mp4", "ext": "mp4"},
        {"format_id": "h264_720p_1234567",
         "format_note": "Playback video",
         "url": "https://v16.tiktokcdn.com/br/bb.mp4", "ext": "mp4"},
        {"format_id": "download_addr",
         "format_note": "Download video, watermarked",
         "url": "https://v16.tiktokcdn.com/dl/cc.mp4", "ext": "mp4"},
    ],
}

#: Web-path shape: the only rendition is the 'download' format the
#: extractor unconditionally notes as watermarked.
YTDLP_TT_WM_ONLY = {
    "id": "7234567890123456789",
    "title": "some dance #fyp",
    "formats": [
        {"format_id": "download", "format_note": "watermarked",
         "url": "https://v16.tiktokcdn.com/dl/cc.mp4", "ext": "mp4"},
    ],
}

TIKWM_OK = {
    "code": 0, "msg": "success",
    "data": {
        "id": "7234567890123456789", "title": "some dance",
        "duration": 21, "author": {"nickname": "dancer"},
        "play": "https://v16.tiktokcdn.com/clean/play.mp4",
        "wmplay": "https://v16.tiktokcdn.com/wm/wmplay.mp4",
        "hdplay": "https://v16.tiktokcdn.com/clean/hdplay.mp4",
        "music": "https://v16.tiktokcdn.com/music.mp3",
    },
}

TIKWM_FAIL = {"code": -1, "msg": "Url parsing failed"}


def _media_file(tmp_path: Path, name: str = "video.mp4",
                size: int = 4096) -> Path:
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return p


def _ok_report(path: Path, stage: str = "direct") -> DownloadReport:
    r = DownloadReport(url=TT_URL)
    r.ok = True
    r.path = str(path)
    r.title = "some dance #fyp"
    r.size_bytes = path.stat().st_size
    r.stages = [StageLog(stage=stage, url=TT_URL, ok=True)]
    return r


def _fail_report(reason: str, stage: str = "direct") -> DownloadReport:
    r = DownloadReport(url=TT_URL)
    r.ok = False
    r.failure = "site_error"  # type: ignore[assignment]
    r.likely_cause = reason
    r.stages = [StageLog(stage=stage, url=TT_URL,
                         error_class="site_error", error=reason)]
    return r


class FakeDownloader:
    """MediaDownloader-shaped: records calls, returns a canned report."""

    def __init__(self, report=None, dest: Path | None = None):
        self.report = report
        self.calls: list[tuple[str, dict]] = []
        self._dest = dest

    def download(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.report

    def _dest_dir(self, dest_dir=None):
        return Path(dest_dir) if dest_dir else self._dest


# ── detection ───────────────────────────────────────────────────────


@pytest.mark.parametrize("url,expected", [
    ("https://www.tiktok.com/@dancer/video/7234567890123456789", "tiktok"),
    ("https://vm.tiktok.com/ZMhAbCdEf/", "tiktok"),
    ("https://vt.tiktok.com/ZSwAbCdEf/", "tiktok"),
    ("https://www.instagram.com/reel/CabcDEF123/", "instagram"),
    ("https://instagr.am/p/CabcDEF123/", "instagram"),
    ("https://open.spotify.com/track/0TnOYISbd1XYRBk9myaseg", "spotify"),
    ("https://spoti.fi/3AbCdEf", "spotify"),
    ("spotify:track:0TnOYISbd1XYRBk9myaseg", "spotify"),
    ("https://youtu.be/abcdefghijk", "youtube"),
    ("https://music.youtube.com/watch?v=abcdefghijk", "youtube"),
    ("https://www.youtube.com/watch?v=abcdefghijk", "youtube"),
    ("https://soundcloud.com/artist/track", "soundcloud"),
    ("https://x.com/user/status/123456", "twitter"),
    ("https://twitter.com/user/status/123456", "twitter"),
    ("https://www.facebook.com/watch/?v=123", "facebook"),
    ("https://vimeo.com/123456", "vimeo"),
    ("https://dai.ly/x8abcde", "dailymotion"),
    ("https://www.twitch.tv/videos/123456", "twitch"),
    ("https://www.reddit.com/r/videos/comments/abc/x/", "reddit"),
    ("https://audiomack.com/artist/song/slug", "audiomack"),
    ("https://artist.bandcamp.com/track/x", "bandcamp"),
    # lookalikes and unknowns are nothing
    ("https://faketiktok.com/@u/video/1", ""),
    ("https://tiktok.com.evil.example/@u/video/1", ""),
    ("https://example.com/some/article", ""),
    ("", ""),
    ("not a url", ""),
])
def test_detect_platform(url, expected):
    from nomorals.media.platforms import detect_platform
    assert detect_platform(url) == expected


def test_is_media_file_url():
    from nomorals.media.platforms import is_media_file_url
    assert is_media_file_url("https://cdn.example.net/movie.mp4")
    assert is_media_file_url("https://cdn.example.net/song.mp3?dl=1")
    assert not is_media_file_url("https://example.com/article.html")
    assert not is_media_file_url("https://example.com/watch?v=1")


# ── TikTok ──────────────────────────────────────────────────────────


def test_tiktok_format_facts():
    from nomorals.media.platforms import tiktok_format_facts
    assert tiktok_format_facts(YTDLP_TT_CLEAN) == (True, True)
    assert tiktok_format_facts(YTDLP_TT_WM_ONLY) == (False, True)
    assert tiktok_format_facts(None) == (False, False)
    assert tiktok_format_facts({}) == (False, False)
    assert tiktok_format_facts({"formats": []}) == (False, False)


def test_tiktok_clean_route_uses_clean_spec_and_labels_clean(tmp_path):
    from nomorals.media.platforms import TIKTOK_CLEAN_FORMAT_SPEC, fetch_tiktok
    landed = _media_file(tmp_path)
    dl = FakeDownloader(report=_ok_report(landed))
    res = fetch_tiktok(
        TT_URL, downloader=dl,
        probe_fn=lambda url: YTDLP_TT_CLEAN,
        api_get=lambda url: pytest.fail("tikwm must not run"),
        fetch_bytes=lambda u, d: pytest.fail("no byte fetch needed"))
    assert res.ok is True
    assert res.route == "yt-dlp"
    assert res.watermarked is False
    assert dl.calls[0][1].get("format_spec") == TIKTOK_CLEAN_FORMAT_SPEC


def test_tiktok_watermarked_only_labels_watermarked(tmp_path):
    from nomorals.media.platforms import fetch_tiktok
    landed = _media_file(tmp_path)
    dl = FakeDownloader(report=_ok_report(landed))
    res = fetch_tiktok(
        TT_URL, downloader=dl,
        probe_fn=lambda url: YTDLP_TT_WM_ONLY,
        api_get=lambda url: None,
        fetch_bytes=lambda u, d: False)
    assert res.ok is True
    assert res.watermarked is True
    # the clean spec was NOT used to fetch a watermarked file under
    # a clean pretense — the default spec ran
    assert "format_spec" not in dl.calls[0][1]


def test_tiktok_probe_failure_labels_unverified(tmp_path):
    from nomorals.media.platforms import fetch_tiktok
    landed = _media_file(tmp_path)
    dl = FakeDownloader(report=_ok_report(landed))
    res = fetch_tiktok(
        TT_URL, downloader=dl,
        probe_fn=lambda url: None,
        api_get=lambda url: None,
        fetch_bytes=lambda u, d: False)
    assert res.ok is True
    assert res.watermarked is None


def test_tiktok_falls_back_to_tikwm_clean_rendition(tmp_path):
    from nomorals.media.platforms import fetch_tiktok
    fetched: list[str] = []

    def fake_fetch(url, dest):
        fetched.append(url)
        Path(dest).write_bytes(b"\x00" * 4096)
        return True

    dl = FakeDownloader(report=_fail_report("yt-dlp exploded"),
                        dest=tmp_path)
    res = fetch_tiktok(
        TT_URL, downloader=dl,
        probe_fn=lambda url: YTDLP_TT_CLEAN,
        api_get=lambda url: TIKWM_OK,
        fetch_bytes=fake_fetch)
    assert res.ok is True
    assert res.route == "tikwm"
    assert res.watermarked is False
    # the HD clean rendition was fetched; the watermarked wmplay
    # rendition was never requested
    assert fetched == ["https://v16.tiktokcdn.com/clean/hdplay.mp4"]


def test_tiktok_all_routes_fail_honestly(tmp_path):
    from nomorals.media.platforms import fetch_tiktok
    dl = FakeDownloader(report=_fail_report("HTTP Error 403"),
                        dest=tmp_path)
    res = fetch_tiktok(
        TT_URL, downloader=dl,
        probe_fn=lambda url: None,
        api_get=lambda url: TIKWM_FAIL,
        fetch_bytes=lambda u, d: False)
    assert res.ok is False
    assert "yt-dlp" in res.failure
    assert "tikwm" in res.failure
    assert "Url parsing failed" in res.failure


def test_tiktok_unverified_landing_falls_through(tmp_path):
    """A 'success' report whose file is a stub is not a success."""
    from nomorals.media.platforms import fetch_tiktok
    stub = _media_file(tmp_path, name="stub.mp4", size=64)
    dl = FakeDownloader(report=_ok_report(stub), dest=tmp_path)
    res = fetch_tiktok(
        TT_URL, downloader=dl,
        probe_fn=lambda url: YTDLP_TT_CLEAN,
        api_get=lambda url: TIKWM_OK,
        fetch_bytes=lambda u, d: (Path(d).write_bytes(b"\x00" * 4096)
                                  or True))
    assert res.ok is True
    assert res.route == "tikwm"


# ── Instagram ───────────────────────────────────────────────────────


def test_instagram_public_success(tmp_path):
    from nomorals.media.platforms import fetch_instagram
    landed = _media_file(tmp_path, name="reel.mp4")
    dl = FakeDownloader(report=_ok_report(landed))
    res = fetch_instagram(IG_URL, downloader=dl)
    assert res.ok is True
    assert res.route == "yt-dlp"
    assert res.path == str(landed)


def test_instagram_login_wall_is_named(tmp_path):
    from nomorals.media.platforms import fetch_instagram
    report = _fail_report(
        "[instagram] CabcDEF123: Login required — sign in to view "
        "this post")
    dl = FakeDownloader(report=report)
    res = fetch_instagram(IG_URL, downloader=dl)
    assert res.ok is False
    assert res.wall == "login"
    assert "Instagram wants a login for this one" in res.failure
    # plain words, not the raw extractor line
    assert "Login required — sign in" not in res.failure


def test_instagram_other_failure_is_not_a_wall(tmp_path):
    from nomorals.media.platforms import fetch_instagram
    dl = FakeDownloader(report=_fail_report("HTTP Error 404: gone"))
    res = fetch_instagram(IG_URL, downloader=dl)
    assert res.ok is False
    assert res.wall == ""
    assert "404" in res.failure


# ── Spotify metadata ────────────────────────────────────────────────

SPOTIFY_TRACK_EMBED = (
    '<html><head></head><body>'
    '<script id="__NEXT_DATA__" type="application/json">'
    '{"props":{"pageProps":{"state":{"data":{"entity":'
    '{"title":"FEAR","subtitle":"NF","duration":213000,'
    '"uri":"spotify:track:0TnOYISbd1XYRBk9myaseg"}'
    "}}}}}</script></body></html>")

SPOTIFY_PLAYLIST_EMBED = (
    '<html><body>'
    '<script id="__NEXT_DATA__" type="application/json">'
    '{"props":{"pageProps":{"state":{"data":{"entity":'
    '{"name":"Chill Mix","type":"playlist","trackList":['
    '{"title":"FEAR","subtitle":"NF","duration":213000,'
    '"uri":"spotify:track:a1","isPlayable":true},'
    '{"title":"HOME","subtitle":"NF","duration":180000,'
    '"uri":"spotify:track:a2","isPlayable":true},'
    '{"title":"SORRY","subtitle":"NF, James Arthur",'
    '"duration":195000,"uri":"spotify:track:a3","isPlayable":true}'
    "]}}}}}}</script></body></html>")


@pytest.mark.parametrize("text,expected", [
    ("https://open.spotify.com/track/0TnOYISbd1XYRBk9myaseg",
     ("track", "0TnOYISbd1XYRBk9myaseg")),
    ("https://open.spotify.com/intl-de/album/4LH4d3cOWNNsVw41Gqt2kv?si=x",
     ("album", "4LH4d3cOWNNsVw41Gqt2kv")),
    ("check this https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M !",
     ("playlist", "37i9dQZF1DXcBWIGoYBM5M")),
    ("spotify:episode:512ojhOuo1ktJprKbVcKyQ",
     ("episode", "512ojhOuo1ktJprKbVcKyQ")),
    ("https://spoti.fi/3AbCdEf", None),   # short link: expand first
    ("https://example.com/track/abc", None),
    ("", None),
])
def test_parse_spotify_ref(text, expected):
    from nomorals.media.spotify_meta import parse_spotify_ref
    assert parse_spotify_ref(text) == expected


def test_spotify_meta_track_from_embed():
    from nomorals.media.spotify_meta import fetch_spotify_meta
    meta = fetch_spotify_meta(
        "track", "0TnOYISbd1XYRBk9myaseg",
        get_text=lambda url: SPOTIFY_TRACK_EMBED,
        get_json=lambda url: None)
    assert meta is not None
    assert meta.name == "FEAR"
    assert meta.artist == "NF"
    assert meta.duration_s == 213.0
    assert len(meta.tracks) == 1
    assert meta.tracks[0].query == "FEAR by NF"
    assert meta.source == "embed"


def test_spotify_meta_playlist_from_embed():
    from nomorals.media.spotify_meta import fetch_spotify_meta
    meta = fetch_spotify_meta(
        "playlist", "37i9dQZF1DXcBWIGoYBM5M",
        get_text=lambda url: SPOTIFY_PLAYLIST_EMBED,
        get_json=lambda url: None)
    assert meta is not None
    assert meta.name == "Chill Mix"
    assert [t.title for t in meta.tracks] == ["FEAR", "HOME", "SORRY"]
    assert meta.tracks[1].duration_s == 180.0
    assert meta.tracks[2].artist == "NF, James Arthur"


def test_spotify_meta_oembed_fallback():
    from nomorals.media.spotify_meta import fetch_spotify_meta
    meta = fetch_spotify_meta(
        "track", "0TnOYISbd1XYRBk9myaseg",
        get_text=lambda url: "<html>no payload</html>",
        get_json=lambda url: {"title": "FEAR", "author_name": "NF",
                              "provider_name": "Spotify"})
    assert meta is not None
    assert meta.name == "FEAR"
    assert meta.artist == "NF"
    assert meta.source == "oembed"
    assert len(meta.tracks) == 1  # synthesized from the single entity


def test_spotify_meta_dead_link_is_none():
    from nomorals.media.spotify_meta import fetch_spotify_meta
    assert fetch_spotify_meta(
        "track", "0TnOYISbd1XYRBk9myaseg",
        get_text=lambda url: "", get_json=lambda url: None) is None
    assert fetch_spotify_meta(
        "bogus", "x", get_text=lambda url: "",
        get_json=lambda url: None) is None


# ── runtime handlers ────────────────────────────────────────────────


def _mixin():
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    return RuntimeMediaMixin()


class FakeEngine:
    """PlaybackEngine-shaped for the Spotify delivery paths."""

    instances: list = []

    def __init__(self, context=None):
        self.q: list[dict] = []
        FakeEngine.instances.append(self)

    def _enqueue(self, path, kind, title, *, artist="", duration=0.0):
        item = {"path": path, "kind": kind, "title": title,
                "artist": artist, "duration": duration}
        self.q.append(item)
        return item

    def play(self):
        return {"status": "playing",
                "current": self.q[0]["title"] if self.q else "",
                "backend": "fake"}

    def queue(self):
        return list(self.q)


def test_control_tiktok_guards():
    m = _mixin()
    assert "usage: /tiktok" in m._control_tiktok("")
    out = m._control_tiktok("https://www.youtube.com/watch?v=abcdefghijk")
    assert "not a TikTok link" in out


def test_control_instagram_guards():
    m = _mixin()
    assert "usage: /instagram" in m._control_instagram("")
    out = m._control_instagram("https://www.tiktok.com/@u/video/1")
    assert "not an Instagram link" in out


def test_control_aio_unsupported_is_honest_upfront():
    m = _mixin()
    out = m._control_aio("https://example.com/some/article")
    assert "can't fetch from example.com" in out
    assert "TikTok" in out and "Spotify" in out  # names what IS supported
    assert "usage: /aio" in m._control_aio("")


def test_control_aio_delegates_supported_links():
    m = _mixin()
    m._control_download = lambda tail, chat_key="": "DELEGATED"
    assert m._control_aio(TT_URL) == "DELEGATED"
    # a direct media file link is supported even on an unknown host
    assert m._control_aio("https://cdn.example.net/movie.mp4") == "DELEGATED"


def test_control_download_routes_spotify_and_tiktok():
    m = _mixin()
    m._control_spotify = lambda tail, chat_key="": "SPOTIFY ROUTE"
    assert m._control_download(SP_URL) == "SPOTIFY ROUTE"
    m._tiktok_produce = lambda url, audio_only, chat: ("TIKTOK ROUTE", True)
    assert m._control_download(TT_URL) == "TIKTOK ROUTE"
    m._instagram_produce = lambda url, chat: ("IG ROUTE", True)
    assert m._control_download(IG_URL) == "IG ROUTE"


def _spotify_meta_track():
    from nomorals.media.spotify_meta import SpotifyMeta, SpotifyTrack
    return SpotifyMeta(
        kind="track", spotify_id="0TnOYISbd1XYRBk9myaseg",
        name="FEAR", artist="NF", duration_s=213.0,
        tracks=[SpotifyTrack(title="FEAR", artist="NF",
                             duration_s=213.0)],
        source="embed")


def test_spotify_track_matched_and_labeled_by_real_source():
    import nomorals.media.spotify_meta as sm
    m = _mixin()
    item = {"kind": "youtube", "path": "https://youtube.com/watch?v=x",
            "title": "FEAR", "artist": "NF", "duration": 213.0,
            "downloadable": True, "preview_only": False}
    m._spotify_match = lambda q, hint=0.0: (
        item, ["youtube-search — ok"], "YouTube search")
    m.context = SimpleNamespace()
    FakeEngine.instances = []
    with mock.patch.object(sm, "fetch_spotify_meta",
                           return_value=_spotify_meta_track()), \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       FakeEngine):
        out = m._control_spotify(SP_URL)
    assert "“FEAR” — NF" in out
    assert "full track was matched via YouTube search" in out
    assert "playing" in out  # CLI path starts playback


def test_spotify_previews_only_world_delivers_nothing():
    """The preview law on /spotify: only previews exist → honest
    absence, and the delivery machinery is never even built."""
    import nomorals.media.spotify_meta as sm
    from nomorals.media.resolver import ResolvedAudio

    class PreviewResolver:
        def resolve(self, q):
            return ResolvedAudio(
                ok=True, downloadable=False, preview_only=True,
                kind="itunes",
                path_or_url="https://music.apple.com/us/song/fear/1",
                title="FEAR", artist="NF",
                attempts=["itunes-search — ok (8), preview only"])

        def search_candidates(self, q):
            return [ResolvedAudio(
                ok=True, downloadable=False, preview_only=True,
                kind="deezer", path_or_url="https://deezer.com/track/1",
                title="FEAR", artist="NF")]

    def _boom(context=None):
        raise AssertionError("engine built for an undeliverable track")

    m = _mixin()
    m.context = SimpleNamespace()
    with mock.patch.object(sm, "fetch_spotify_meta",
                           return_value=_spotify_meta_track()), \
            mock.patch("nomorals.media.resolver.SourceResolver",
                       lambda ctx: PreviewResolver()), \
            mock.patch("nomorals.media.playback.PlaybackEngine", _boom):
        out = m._control_spotify(SP_URL)
    assert "won't pass a 30s preview off as the song" in out
    assert "no full-track source has it" in out


def test_spotify_unidentifiable_link_is_honest():
    import nomorals.media.spotify_meta as sm
    m = _mixin()
    with mock.patch.object(sm, "fetch_spotify_meta", return_value=None):
        out = m._control_spotify(SP_URL)
    assert "couldn't identify that Spotify link" in out


def test_spotify_playlist_cap_and_summary():
    import nomorals.media.spotify_meta as sm
    from nomorals.media.spotify_meta import SpotifyMeta, SpotifyTrack

    tracks = [SpotifyTrack(title=f"T{i}", artist="A", duration_s=100.0)
              for i in range(12)]
    meta = SpotifyMeta(kind="playlist", spotify_id="pl1",
                       name="Big Mix", tracks=tracks, source="embed")

    def fake_match(query, hint=0.0):
        n = int(query.split(" by ")[0][1:])
        if n % 2 == 0:
            return ({"kind": "youtube", "path": f"https://yt/{n}",
                     "title": f"T{n}", "artist": "A", "duration": 100.0,
                     "downloadable": True, "preview_only": False},
                    [], "YouTube search")
        return None, [], ""

    m = _mixin()
    m._spotify_match = fake_match
    m.context = SimpleNamespace()
    FakeEngine.instances = []
    with mock.patch.object(sm, "fetch_spotify_meta", return_value=meta), \
            mock.patch("nomorals.media.playback.PlaybackEngine",
                       FakeEngine):
        out = m._control_spotify(
            "https://open.spotify.com/playlist/pl1")
    assert "fetching the first 10" in out
    assert "matched 5 of 10 in full" in out
    assert "✓ “T0”" in out and "✗ “T1”" in out
    # tracks past the cap are not silently dropped from the story
    assert "12 tracks" in out


def test_spotify_send_many_counts_honestly():
    m = _mixin()
    items = [{"title": "A"}, {"title": "B"}]
    m._play_send_in_chat_result = lambda e, it, out, chat: (
        ("", True) if it["title"] == "A"
        else ("\ncouldn't download the audio: boom", False))
    text, sent = m._spotify_send_many(object(), items, "SUMMARY",
                                      object())
    assert sent is True
    assert "sent 1 of 2 files" in text
    assert "“B”" in text and "boom" in text


# ── registry ────────────────────────────────────────────────────────

NEW_KINDS = ("tiktok", "ttdl", "instagram", "spotify", "aio")


def test_new_commands_registered_everywhere():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        parse_control,
    )
    grouped = [k for _t, kinds in LIST_GROUPS for k in kinds]
    for kind in NEW_KINDS:
        assert kind in CONTROL_COMMANDS
        page = COMMAND_DETAILS[kind]
        assert page["what"] and page["usage"] and page["example"]
        assert LIST_ONELINERS[kind]
        assert grouped.count(kind) == 1
        cmd = parse_control(f"/{kind} https://example.com/x")
        assert cmd is not None and cmd.kind == kind


def test_new_commands_in_whatsapp_catalog():
    from nomorals.social.chat.control import whatsapp_menu
    menu = whatsapp_menu()
    for kind in NEW_KINDS:
        assert f"/{kind}" in menu


def test_new_commands_have_dispatch_branches():
    src = (Path(__file__).resolve().parent.parent
           / "nomorals" / "agents" / "partner" / "runtime.py"
           ).read_text()
    assert 'kind in {"tiktok", "ttdl"}' in src
    assert 'kind == "instagram"' in src
    assert 'kind == "spotify"' in src
    assert 'kind == "aio"' in src

