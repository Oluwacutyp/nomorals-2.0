"""Failing-first tests for the SoundCloud download path (lane 24).

Defect 4, traced from the repo + the owner's EC2 log: the song
fell through to SoundCloud and the download died there, but the
SoundCloud connector rung — built exactly for this — never ran.
``_stage_connector`` re-derived the site descriptor from the WORK
url, and the work url here is the CDN playlist link
(``playback.media-streaming.soundcloud.cloud``), which no
descriptor claims.  The playback path already passes ``page_url``
(the soundcloud.com track page); the connector stage must resolve
through it instead.  (The descriptor gate itself is untouched —
no per-host check; this is routing, not gate-weakening.)

Defect 3, traced live: the signed HLS playlist URL is valid
(``expires`` still ~1.5h out, CloudFront Policy carries only
``DateLessThan`` — no IP binding) and fetches HTTP 200 with a
plain bare GET, while yt-dlp's generic fetch 403'd on EC2 with
the estate cookies riding along.  A stale bot-management cookie
is what the CDN refuses while the signed URL itself is fine, so
the direct backend retries a 403 once with NO cookies before
giving up.

Defect 2 (browser stage finds nothing on a SoundCloud JS-player
page) is pinned here as honest-empty: the page exposes no static
media, so the stage reports no media — it must not invent a link.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nomorals.core.errors import MediaError  # noqa: E402
from nomorals.media.downloader import (  # noqa: E402
    DownloadReport,
    MediaDownloader,
)

SC_PAGE = "https://soundcloud.com/ronis-goliath/tell-me-one-lie"
CDN_URL = ("https://playback.media-streaming.soundcloud.cloud/"
           "1cw8Sdk8wH99/aac_160k/b2744c55-45f1-4760-94db-1b2eb9315553b/"
           "playlist.m3u8?expires=1794962792&Policy=eyJ&Signature=abc"
           "&Key-Pair-Id=K346O6QXLEIRF3")


def _ctx(tmpdir):
    ctx = MagicMock()
    ctx.home = Path(tmpdir)
    ctx.db = MagicMock()
    ctx.settings = None
    return ctx


def _budget():
    import time
    return {"deadline": time.perf_counter() + 600.0}


class _FakeConnector:
    """A SoundCloud connector recording what it was asked to resolve."""

    def __init__(self):
        self.seen_urls = []

    def stream_url(self, url):
        self.seen_urls.append(url)
        return {
            "url": CDN_URL,
            "protocol": "hls",
            "mime_type": "application/vnd.apple.mpegurl",
            "track": {"title": "Tell Me One Lie",
                      "permalink_url": SC_PAGE},
        }


class ConnectorRoutingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _dl(self, connector):
        dl = MediaDownloader(_ctx(self.tmp), direct_retries=1,
                             soundcloud_connector=connector)
        # The fetch half is another rung's job; routing is what's
        # under test — the HLS fetch is stubbed to succeed.
        dl._connector_hls = lambda *a, **k: True  # noqa: SLF001
        return dl

    def _attempt_connector(self, dl, work_url, page_url=""):
        report = DownloadReport(url=work_url)
        ok = dl._attempt_stage(
            "connector", report, work_url, Path(self.tmp),
            audio_only=True, format_spec="bestaudio/best",
            timeout=60.0, budget=_budget(), page_url=page_url)
        return ok, report

    def test_connector_resolves_through_the_source_page(self):
        """CDN work url + soundcloud page_url: the connector runs,
        against the PAGE — the url it can resolve."""
        conn = _FakeConnector()
        ok, _ = self._attempt_connector(self._dl(conn), CDN_URL,
                                        page_url=SC_PAGE)
        self.assertTrue(ok)
        self.assertEqual(conn.seen_urls, [SC_PAGE])

    def test_connector_still_dark_without_a_page(self):
        """CDN url and no page_url: no descriptor claims it, the
        rung stays off — the old behaviour, preserved."""
        conn = _FakeConnector()
        ok, _ = self._attempt_connector(self._dl(conn), CDN_URL)
        self.assertFalse(ok)
        self.assertEqual(conn.seen_urls, [])

    def test_connector_unchanged_when_work_url_is_the_page(self):
        """The normal case (work url IS the track page) still
        resolves against it."""
        conn = _FakeConnector()
        ok, _ = self._attempt_connector(self._dl(conn), SC_PAGE)
        self.assertTrue(ok)
        self.assertEqual(conn.seen_urls, [SC_PAGE])


class _FakeBrowserSession:
    """A SoundCloud track page as the browser stage sees it: a JS
    player shell with no static media anywhere in it."""

    url = SC_PAGE

    def extract(self, kind="meta"):
        return {"og": {"og:title": "Tell Me One Lie",
                       "og:type": "music.song"},
                "other": {}}

    def html(self, max_chars=2_000_000):
        return {"html": (
            "<html><head>"
            "<meta property=\"og:title\" content=\"Tell Me One Lie\">"
            "<meta property=\"og:type\" content=\"music.song\">"
            "</head><body>"
            "<div id=\"app\">loading…</div>"
            "<script src=\"https://a-v2.sndcdn.com/assets/55-5f1a3ddb.js\">"
            "</script>"
            "</body></html>")}

    def links(self, max_links=200):
        return {"links": []}


class BrowserStageHonestEmptyTests(unittest.TestCase):
    def test_js_player_page_with_no_static_media_is_empty(self):
        """A SoundCloud track page carries no og:audio/video, no
        <audio>/<video>/<source>, no contentUrl — the browser
        stage must find nothing rather than invent a link."""
        dl = MediaDownloader(_ctx(tempfile.mkdtemp()), direct_retries=1)
        self.assertEqual(dl._media_candidates(_FakeBrowserSession()), [])


class Direct403BareRetryTests(unittest.TestCase):
    """tools/media._download_python_api retries a cookie'd 403 bare."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.target = Path(self.tmp) / "out"
        self.target.mkdir()
        self.landed = self.target / "track.mp3"
        self.landed.write_bytes(b"ID3" + b"\x00" * 100)

    def _run(self, calls, cookies_file):
        from nomorals.tools import media as media_mod

        landed = self.landed

        class FakeYDL:
            def __init__(self, opts):
                self.opts = dict(opts)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=True):
                calls.append(dict(self.opts))
                if "cookiefile" in self.opts:
                    raise Exception("ERROR: [generic] playlist: Unable "
                                    "to download webpage: HTTP Error "
                                    "403: Forbidden")
                return {"id": "track", "title": "Tell Me One Lie",
                        "filepath": str(landed)}

            def prepare_filename(self, info):
                return str(landed)

        fake_module = MagicMock()
        fake_module.YoutubeDL = FakeYDL
        with patch.object(media_mod, "load_optional",
                          return_value=fake_module):
            return media_mod.download(
                CDN_URL, self.target, format_spec="bestaudio/best",
                audio_only=True, cookies_file=cookies_file)

    def test_403_with_cookies_retries_bare_and_lands(self):
        calls = []
        result = self._run(calls, cookies_file="/tmp/estate-cookies.txt")
        self.assertEqual(len(calls), 2)
        self.assertIn("cookiefile", calls[0])
        self.assertNotIn("cookiefile", calls[1])
        self.assertEqual(result["path"], str(self.landed))

    def test_403_without_cookies_is_not_retried_bare(self):
        from nomorals.tools import media as media_mod

        class FakeYDL:
            def __init__(self, opts):
                self.opts = dict(opts)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=True):
                raise Exception("ERROR: [generic] playlist: Unable to "
                                "download webpage: HTTP Error 403: "
                                "Forbidden")

            def prepare_filename(self, info):
                return ""

        fake_module = MagicMock()
        fake_module.YoutubeDL = FakeYDL
        with (patch.object(media_mod, "load_optional",
                           return_value=fake_module),
              self.assertRaises(MediaError) as cm):
            media_mod.download(CDN_URL, self.target,
                               format_spec="bestaudio/best",
                               audio_only=True, cookies_file="")
        self.assertIn("403", str(cm.exception))

    def test_non_403_error_is_not_retried(self):
        from nomorals.tools import media as media_mod

        seen = []

        class FakeYDL:
            def __init__(self, opts):
                self.opts = dict(opts)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=True):
                seen.append(dict(self.opts))
                raise Exception("ERROR: [generic] playlist: Unable to "
                                "download webpage: HTTP Error 404: "
                                "Not Found")

            def prepare_filename(self, info):
                return ""

        fake_module = MagicMock()
        fake_module.YoutubeDL = FakeYDL
        with (patch.object(media_mod, "load_optional",
                           return_value=fake_module),
              self.assertRaises(MediaError)):
            media_mod.download(CDN_URL, self.target,
                               format_spec="bestaudio/best",
                               audio_only=True,
                               cookies_file="/tmp/estate.txt")
        self.assertEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
