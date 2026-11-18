"""Cobalt (52-websites #1) — uniform paste-link media fetch.

Failing-first battery: the surface under test is
``fetch_media(url, want, quality)`` over the existing per-service
resolvers in :mod:`nomorals.media.downloader`, the ``fetch_media``
brain tool, the extended ``/download`` command, and the bare-URL
chat hook.  Every test names the real behaviour it proves — a file
that lands, a format and a duration that match the source's own
metadata, or a refusal that names the service and the reason.

Live-chain tests run against a loopback fixture server (ffmpeg-made
mp4 / mp3 / jpg with known durations), never the public internet:
yt-dlp is not installed on this box, so the chain's direct stage
does the fetch, end to end.
"""

import shutil
import subprocess
import tempfile
import threading
import unittest
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import MagicMock, patch

from nomorals.media.download_kit import ExtractorDescriptor, Licence
from nomorals.media.downloader import (
    FetchMediaResult,
    fetch_media,
    formats_for,
    is_bare_media_url,
)

YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
ARTICLE_URL = "https://example.com/some/news/article-about-cats"


def _ctx(tmpdir):
    ctx = MagicMock()
    ctx.home = Path(tmpdir)
    ctx.db = MagicMock()
    ctx.settings = None  # safe_path falls back to cwd/workspace
    return ctx


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class FixtureServer(unittest.TestCase):
    """Loopback HTTP server over ffmpeg-made fixtures."""

    server = None
    thread = None
    base_url = ""
    fixture_dir = None

    @classmethod
    def setUpClass(cls):
        if not _ffmpeg_available():
            raise unittest.SkipTest("ffmpeg not installed on this box")
        cls.fixture_dir = Path(tempfile.mkdtemp(prefix="cobalt-fixtures-"))
        clip = cls.fixture_dir / "clip.mp4"
        song = cls.fixture_dir / "song.mp3"
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=10",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-c:a", "aac", str(clip)],
            check=True, timeout=120)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
             "-c:a", "libmp3lame", str(song)],
            check=True, timeout=120)
        from PIL import Image
        Image.new("RGB", (64, 64), (200, 30, 30)).save(
            cls.fixture_dir / "photo.jpg", "JPEG")
        handler = partial(SimpleHTTPRequestHandler,
                          directory=str(cls.fixture_dir))
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.base_url = (
            f"http://127.0.0.1:{cls.server.server_address[1]}")
        cls.thread = threading.Thread(target=cls.server.serve_forever,
                                      daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        if cls.server is not None:
            cls.server.shutdown()
            cls.server.server_close()
        if cls.fixture_dir is not None:
            shutil.rmtree(cls.fixture_dir, ignore_errors=True)


class SurfaceTests(unittest.TestCase):
    def test_formats_for_youtube_names_service_and_labels(self):
        info = formats_for(YOUTUBE_URL)
        self.assertEqual(info["service"], "youtube")
        self.assertTrue(info["downloadable"])
        labels = {f["label"] for f in info["formats"]}
        self.assertIn("720p", labels)
        self.assertIn("best", labels)
        self.assertIn("audio", labels)
        self.assertIn("image", labels)
        self.assertIn("gif", labels)
        row_720 = next(f for f in info["formats"]
                       if f["label"] == "720p")
        self.assertEqual(row_720["format_spec"], "best[height<=720]")
        row_audio = next(f for f in info["formats"]
                         if f["label"] == "audio")
        self.assertEqual(row_audio["format_spec"], "bestaudio/best")

    def test_unknown_quality_refused_with_offered_labels(self):
        result = fetch_media(YOUTUBE_URL, quality="4k")
        self.assertIsInstance(result, FetchMediaResult)
        self.assertFalse(result.ok)
        self.assertIn("4k", result.likely_cause)
        self.assertIn("best", result.likely_cause)
        self.assertIn("720p", result.likely_cause)

    def test_unknown_want_refused_with_valid_wants(self):
        result = fetch_media(YOUTUBE_URL, want="pdf")
        self.assertFalse(result.ok)
        for want in ("video", "audio", "image", "gif"):
            self.assertIn(want, result.likely_cause)

    def test_not_a_url_refused(self):
        result = fetch_media("not a link at all")
        self.assertFalse(result.ok)

    def test_bare_url_predicate(self):
        self.assertEqual(is_bare_media_url(YOUTUBE_URL), YOUTUBE_URL)
        self.assertEqual(
            is_bare_media_url("https://example.com/photo.jpg"),
            "https://example.com/photo.jpg")
        self.assertEqual(is_bare_media_url(ARTICLE_URL), "")
        self.assertEqual(
            is_bare_media_url("have a look " + YOUTUBE_URL), "")
        self.assertEqual(is_bare_media_url(""), "")
        self.assertEqual(is_bare_media_url("https://example.com/x.mp4"),
                         "https://example.com/x.mp4")

    def test_not_downloadable_source_refused_by_name(self):
        tmp = tempfile.mkdtemp()
        try:
            from nomorals.media.downloader import MediaDownloader
            dl = MediaDownloader(_ctx(tmp), direct_retries=1,
                                 max_proxies=0)
            fake_registry = MagicMock()
            fake_registry.best_for.return_value = ExtractorDescriptor(
                name="spotify", extraction_rule="none",
                licence=Licence.UNKNOWN,
                notes="does not serve downloads",
                downloadable=False)
            with patch.object(
                    MediaDownloader, "_extractor_registry",
                    return_value=fake_registry), patch.object(
                        MediaDownloader, "_site_descriptor_for",
                        return_value=None):
                result = fetch_media(
                    "https://open.spotify.com/track/abc",
                    downloader=dl, dest_dir=Path(tmp) / "out")
            self.assertFalse(result.ok)
            self.assertIn("spotify", result.likely_cause)
            self.assertIn("does not serve downloads",
                          result.likely_cause)
            self.assertEqual(result.service, "spotify")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class ChainTests(FixtureServer):
    def _run(self, path, want="video", quality="best"):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        from nomorals.media.downloader import MediaDownloader
        dl = MediaDownloader(_ctx(tmp), direct_retries=1, max_proxies=0)
        return fetch_media(
            f"{self.base_url}/{path}", want=want, quality=quality,
            downloader=dl, dest_dir=Path(tmp) / "out"), tmp

    def test_direct_mp4_lands_with_format_and_duration(self):
        result, _tmp = self._run("clip.mp4", want="video")
        self.assertTrue(result.ok, result.likely_cause)
        self.assertEqual(result.kind, "video")
        self.assertEqual(result.format, "mp4")
        self.assertTrue(Path(result.path).is_file())
        self.assertGreater(result.size_bytes, 0)
        self.assertIsNotNone(result.duration_s)
        self.assertAlmostEqual(result.duration_s, 2.0, delta=0.6)

    def test_direct_mp3_audio_lands_with_duration(self):
        result, _tmp = self._run("song.mp3", want="audio")
        self.assertTrue(result.ok, result.likely_cause)
        self.assertEqual(result.kind, "audio")
        self.assertTrue(Path(result.path).is_file())
        self.assertIsNotNone(result.duration_s)
        self.assertAlmostEqual(result.duration_s, 2.0, delta=0.6)

    def test_direct_jpg_image_lands(self):
        result, _tmp = self._run("photo.jpg", want="image")
        self.assertTrue(result.ok, result.likely_cause)
        self.assertEqual(result.kind, "image")
        self.assertEqual(result.format, "jpg")
        self.assertTrue(Path(result.path).is_file())

    def test_image_want_on_video_refused_by_name(self):
        result, _tmp = self._run("clip.mp4", want="image")
        self.assertFalse(result.ok)
        self.assertIn("video", result.likely_cause)

    def test_gif_from_video_lands(self):
        result, _tmp = self._run("clip.mp4", want="gif")
        self.assertTrue(result.ok, result.likely_cause)
        self.assertEqual(result.kind, "image")
        self.assertEqual(result.format, "gif")
        self.assertTrue(Path(result.path).is_file())
        self.assertTrue(result.path.endswith(".gif"))
        self.assertTrue(Path(result.gif_source).is_file())

    def test_gif_without_ffmpeg_refused_before_downloading(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        from nomorals.media.downloader import MediaDownloader
        dl = MediaDownloader(_ctx(tmp), direct_retries=1, max_proxies=0)
        out = Path(tmp) / "out"
        with patch("shutil.which", return_value=None):
            result = fetch_media(
                f"{self.base_url}/clip.mp4", want="gif",
                downloader=dl, dest_dir=out)
        self.assertFalse(result.ok)
        self.assertIn("ffmpeg", result.likely_cause)
        self.assertFalse(out.exists() and any(out.iterdir()),
                         "refusal must happen before any download")


class _FakeRegistry:
    """Records the tool functions ``register()`` declares."""

    context = None

    def __init__(self):
        self.tools = {}

    def register(self, name, **kwargs):
        def deco(fn):
            self.tools[name] = fn
            return fn

        return deco


class ToolTests(unittest.TestCase):
    def test_fetch_media_tool_registers(self):
        from nomorals.tools import fetch_media as tool_mod
        from nomorals.tools.registry import ToolRegistry
        reg = ToolRegistry()
        tool_mod.register(reg)
        self.assertIn("fetch_media", reg._tools)

    def test_tool_fetch_formats_action(self):
        from nomorals.tools import fetch_media as tool_mod
        registry = _FakeRegistry()
        tool_mod.register(registry)
        fn = registry.tools["fetch_media"]
        out = fn(url=YOUTUBE_URL, action="formats")
        self.assertEqual(out["service"], "youtube")
        self.assertTrue(out["downloadable"])

    def test_tool_fetch_refuses_unknown_quality(self):
        from nomorals.tools import fetch_media as tool_mod
        registry = _FakeRegistry()
        tool_mod.register(registry)
        fn = registry.tools["fetch_media"]
        out = fn(url=YOUTUBE_URL, action="fetch", quality="4k")
        self.assertFalse(out["ok"])


class CommandTests(unittest.TestCase):
    def test_download_parses_want_and_quality(self):
        from nomorals.social.chat.control import parse_control
        cmd = parse_control("/download https://example.com/x.mp4 gif 720p")
        self.assertEqual(cmd.kind, "download")
        self.assertIn("gif", cmd.tail)
        self.assertIn("720p", cmd.tail)

    def test_download_usage_names_format_choices(self):
        from nomorals.social.chat.control import COMMAND_DETAILS
        usage = COMMAND_DETAILS["download"]["usage"]
        self.assertIn("gif", usage)
        self.assertIn("720p", usage)


if __name__ == "__main__":
    unittest.main()
