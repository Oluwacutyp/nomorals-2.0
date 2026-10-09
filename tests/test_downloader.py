"""Tests for the unified download fallback chain (nomorals/media/downloader.py).

- error taxonomy: bot_block vs site_error vs network_failure vs
  no_media_found vs dependency_missing
- stage ordering: browser stage only runs after direct and proxy fail
- proxy failures demote the proxy and continue the download
- failure reports name every stage tried — never a bare "download failed"
- download() never raises
- structured stage logging
"""

import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from nomorals.media.downloader import (
    DownloadReport,
    FailureKind,
    MediaDownloader,
    StageLog,
    classify_failure,
    classify_http_status,
    detect_profile,
)


def _ctx(tmpdir):
    ctx = MagicMock()
    ctx.home = Path(tmpdir)
    ctx.db = MagicMock()
    ctx.settings = None  # safe_path falls back to cwd/workspace
    return ctx


class TaxonomyTests(unittest.TestCase):
    def test_bot_block_markers(self):
        for msg in ("Sign in to confirm you're not a bot",
                    "HTTP 429 Too Many Requests",
                    "captcha challenge required",
                    "ERROR: unable to download: cloudflare"):
            self.assertEqual(classify_failure(msg), FailureKind.BOT_BLOCK,
                             msg)

    def test_network_failure_markers(self):
        for msg in ("download timed out after 60s",
                    "Connection reset by peer",
                    "Temporary failure in name resolution"):
            self.assertEqual(classify_failure(msg),
                             FailureKind.NETWORK_FAILURE, msg)
        self.assertEqual(classify_failure(TimeoutError("x")),
                         FailureKind.NETWORK_FAILURE)
        self.assertEqual(classify_failure(ConnectionError("x")),
                         FailureKind.NETWORK_FAILURE)

    def test_no_media_markers(self):
        self.assertEqual(
            classify_failure("yt-dlp produced no file"),
            FailureKind.NO_MEDIA_FOUND)
        self.assertEqual(
            classify_failure("direct download produced an empty file"),
            FailureKind.NO_MEDIA_FOUND)

    def test_dependency_markers(self):
        self.assertEqual(
            classify_failure("youtube search needs yt-dlp "
                             "(pip install yt-dlp) — not installed here"),
            FailureKind.DEPENDENCY_MISSING)

    def test_site_error_is_default(self):
        self.assertEqual(classify_failure("Video unavailable"),
                         FailureKind.SITE_ERROR)
        self.assertEqual(classify_failure("ERROR: [youtube] abc: Private "
                                          "video"),
                         FailureKind.SITE_ERROR)

    def test_http_status_mapping(self):
        self.assertEqual(classify_http_status(403), FailureKind.BOT_BLOCK)
        self.assertEqual(classify_http_status(429), FailureKind.BOT_BLOCK)
        self.assertEqual(classify_http_status(404), FailureKind.SITE_ERROR)
        self.assertEqual(classify_http_status(503),
                         FailureKind.NETWORK_FAILURE)


class ProfileTests(unittest.TestCase):
    def test_termux_budget_tighter(self):
        d = MediaDownloader(profile="termux")
        w = MediaDownloader(profile="workstation")
        self.assertLess(d.max_proxies, w.max_proxies)
        self.assertLess(d.browser_candidates, w.browser_candidates)
        self.assertLess(d.stage_timeout, w.stage_timeout)

    def test_detect_profile_termux(self):
        with patch.dict(os.environ, {"PREFIX": "/data/data/com.termux/files/usr"}):
            with patch("nomorals.media.downloader.sys") as mock_sys:
                mock_sys.platform = "android"
                self.assertEqual(detect_profile(), "termux")


class OrderingTests(unittest.TestCase):
    """The browser stage only runs after direct and proxy have failed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dl = MediaDownloader(_ctx(self.tmp), direct_retries=1,
                                  max_proxies=1)

    def _fail(self, msg):
        def _raise(*a, **k):
            raise RuntimeError(msg)
        return _raise

    def test_browser_runs_last(self):
        order = []

        def direct(*a, **k):
            order.append("direct")
            raise RuntimeError("connection reset by peer")

        def proxy(report, *a, **k):
            order.append("proxy")
            report.stages.append(StageLog(stage="proxy", note="mocked"))
            return False

        def browser(report, *a, **k):
            order.append("browser")
            report.stages.append(StageLog(stage="browser", note="mocked"))
            return False

        with patch("nomorals.tools.media.download", side_effect=direct):
            with patch.object(self.dl, "_stage_proxy", side_effect=proxy):
                with patch.object(self.dl, "_stage_browser",
                                  side_effect=browser):
                    with patch.object(self.dl, "_proxy_pool",
                                       return_value=None):
                        report = self.dl.download("https://x.test/a.mp3")
        self.assertEqual(order, ["direct", "proxy", "browser"])
        self.assertFalse(report.ok)
        self.assertEqual(report.tried(), ["direct", "proxy", "browser"])

    def test_proxy_skipped_when_pool_missing_chain_continues(self):
        def proxy(report, *a, **k):
            report.stages.append(StageLog(stage="proxy", note="mocked"))
            return False

        def browser(report, *a, **k):
            report.stages.append(StageLog(stage="browser", note="mocked"))
            return False

        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("timeout")):
            with patch.object(self.dl, "_proxy_pool", return_value=None):
                with patch.object(self.dl, "_stage_proxy",
                                  side_effect=proxy):
                    with patch.object(self.dl, "_stage_browser",
                                      side_effect=browser):
                        report = self.dl.download("https://x.test/a.mp3")
        proxy_logs = [s for s in report.stages if s.stage == "proxy"]
        self.assertTrue(proxy_logs)
        # browser stage still ran despite proxy being unavailable
        self.assertEqual(report.tried(), ["direct", "proxy", "browser"])

    def test_success_short_circuits(self):
        audio = os.path.join(self.tmp, "ok.mp3")
        Path(audio).write_bytes(b"data")

        def direct(url, dest, **k):
            return {"url": url, "path": audio, "bytes": 4, "title": "t",
                    "extractor": "yt_dlp", "seconds": 1.0}

        with patch("nomorals.tools.media.download", side_effect=direct):
            with patch.object(self.dl, "_stage_proxy") as mp:
                with patch.object(self.dl, "_stage_browser") as mb:
                    report = self.dl.download("https://x.test/a.mp3")
        self.assertTrue(report.ok)
        mp.assert_not_called()
        mb.assert_not_called()
        self.assertEqual(report.path, audio)


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dl = MediaDownloader(_ctx(self.tmp), direct_retries=3,
                                  max_proxies=0)

    def test_network_failure_retried_then_succeeds(self):
        calls = []

        def direct(url, dest, **k):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("connection reset by peer")
            audio = os.path.join(self.tmp, "r.mp3")
            Path(audio).write_bytes(b"data")
            return {"url": url, "path": audio, "bytes": 4, "title": "t",
                    "extractor": "direct", "seconds": 0.1}

        with patch("nomorals.tools.media.download", side_effect=direct):
            with patch.object(self.dl, "_proxy_pool", return_value=None):
                with patch.object(self.dl, "_stage_browser",
                                  return_value=False):
                    with patch("time.sleep"):  # no real backoff in tests
                        report = self.dl.download("https://x.test/a.mp3")
        self.assertTrue(report.ok)
        self.assertEqual(len(calls), 3)
        attempts = [s.attempt for s in report.stages
                    if s.stage == "direct"]
        self.assertEqual(attempts, [1, 2, 3])

    def test_bot_block_not_retried(self):
        calls = []

        def direct(url, dest, **k):
            calls.append(1)
            raise RuntimeError("Sign in to confirm you're not a bot")

        with patch("nomorals.tools.media.download", side_effect=direct):
            with patch.object(self.dl, "_proxy_pool", return_value=None):
                with patch.object(self.dl, "_stage_browser",
                                  return_value=False):
                    report = self.dl.download("https://x.test/a.mp3")
        self.assertEqual(len(calls), 1)  # no point retrying a bot check
        self.assertFalse(report.ok)
        self.assertEqual(report.failure, FailureKind.BOT_BLOCK)


class ProxyDemoteTests(unittest.TestCase):
    #: Neutralize ambient proxy env vars so the fake direct stage sees
    #: exactly what _proxy_env sets (same trick as the proxypool tests).
    _CLEAR_PROXY_ENV = {
        "http_proxy": "", "https_proxy": "", "HTTP_PROXY": "",
        "HTTPS_PROXY": "", "all_proxy": "", "ALL_PROXY": "",
    }

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dl = MediaDownloader(_ctx(self.tmp), direct_retries=1,
                                  max_proxies=3)
        self._env = patch.dict(os.environ, self._CLEAR_PROXY_ENV)
        self._env.start()

    def tearDown(self):
        self._env.stop()

    def _pool(self, proxies):
        pool = MagicMock()
        pool.healthy_proxies.return_value = proxies
        return pool

    def _proxy(self, pid, proto="https", port=8080):
        return {"id": pid, "protocol": proto, "host": "10.0.0.1",
                "port": port,
                "proxy_url": f"{proto}://user:pw@10.0.0.1:{port}",
                "url": f"{proto}://10.0.0.1:{port}"}

    def test_failing_proxy_demoted_next_tried(self):
        pool = self._pool([self._proxy("p1", port=8081),
                           self._proxy("p2", port=8082)])
        audio = os.path.join(self.tmp, "p.mp3")
        Path(audio).write_bytes(b"data")
        calls = []

        def direct(url, dest, **k):
            proxy = os.environ.get("HTTPS_PROXY", "")
            calls.append(proxy)
            if not proxy:
                raise RuntimeError("connection reset by peer")  # stage 1 dies
            if "8081" in proxy:
                raise RuntimeError("connection refused")  # p1 dies, demoted
            return {"url": url, "path": audio, "bytes": 4, "title": "t",
                    "extractor": "yt_dlp", "seconds": 0.1}

        with patch("nomorals.tools.media.download", side_effect=direct):
            with patch.object(self.dl, "_proxy_pool", return_value=pool):
                report = self.dl.download("https://x.test/a.mp3")
        self.assertTrue(report.ok)
        # p1 was demoted with its failure; p2 delivered
        pool.record_failure.assert_called_once()
        args, _ = pool.record_failure.call_args
        self.assertEqual(args[0], "p1")
        pool.record_success.assert_called_once_with("p2",
                                                    latency_ms=unittest.mock.ANY)
        # stage 1 (direct, no proxy) + p1 (demoted) + p2 (delivered)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0], "")
        self.assertIn("8081", calls[1])
        self.assertIn("8082", calls[2])
        # env is restored after the stage
        self.assertNotIn("10.0.0.1", os.environ.get("HTTPS_PROXY", ""))

    def test_socks_proxy_skipped_without_socks_support(self):
        pool = self._pool([self._proxy("s1", proto="socks5", port=1080)])
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("timeout")) as md:
            with patch.object(self.dl, "_proxy_pool", return_value=pool):
                with patch.object(MediaDownloader, "_socks_usable",
                                  return_value=False):
                    with patch.object(self.dl, "_stage_browser",
                                      return_value=False):
                        report = self.dl.download("https://x.test/a.mp3")
        # the only download attempt was stage 1 (direct) — the proxy
        # stage skipped socks without touching it
        self.assertEqual(md.call_count, 1)
        pool.record_failure.assert_not_called()  # skipped ≠ failed
        skip = [s for s in report.stages
                if s.stage == "proxy" and "skipped" in s.note]
        self.assertTrue(skip)


class BrowserStageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dl = MediaDownloader(_ctx(self.tmp), direct_retries=1,
                                  max_proxies=0)

    def _session(self, media_url, ctype="audio/mpeg"):
        session = MagicMock()
        session.url = "https://blog.test/post/123"
        session.open.return_value = {"ok": True, "status": 200,
                                     "url": session.url}
        session.extract.return_value = {
            "og": {"og:audio": media_url}, "other": {}}
        session.html.return_value = {"html": ""}
        session.links.return_value = {"links": []}

        def fetch_bytes(url, dest):
            Path(dest).write_bytes(b"audio-bytes")
            return {"url": url, "status": 200, "content_type": ctype,
                    "bytes": 11, "path": str(dest)}

        session.fetch_bytes.side_effect = fetch_bytes
        return session

    def test_browser_extracts_og_audio_and_downloads(self):
        media_url = "https://cdn.test/track.mp3"
        session = self._session(media_url)
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("connection reset")):
            with patch.object(self.dl, "_proxy_pool", return_value=None):
                with patch.object(self.dl, "_browser_session",
                                  return_value=session):
                    report = self.dl.download(
                        "https://blog.test/post/123", audio_only=True)
        self.assertTrue(report.ok)
        self.assertEqual(report.extractor, "browser-session")
        self.assertTrue(os.path.isfile(report.path))
        self.assertIn("browser", report.tried())
        # failure of direct did not stop the chain
        self.assertEqual(report.tried()[0], "direct")

    def test_browser_no_media_honest(self):
        session = self._session("")
        session.extract.return_value = {"og": {}, "other": {}}
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("timeout")):
            with patch.object(self.dl, "_proxy_pool", return_value=None):
                with patch.object(self.dl, "_browser_session",
                                  return_value=session):
                    report = self.dl.download("https://blog.test/empty")
        self.assertFalse(report.ok)
        self.assertEqual(report.failure, FailureKind.NO_MEDIA_FOUND)
        self.assertIn("no downloadable media", report.likely_cause)


class ReportTests(unittest.TestCase):
    def test_failure_report_names_every_stage(self):
        report = DownloadReport(
            url="https://x.test/a.mp3",
            stages=[
                StageLog(stage="direct", url="https://x.test/a.mp3",
                         error_class="network_failure", error="timeout",
                         latency_s=1.0),
                StageLog(stage="proxy", url="https://x.test/a.mp3",
                         proxy="https://1.2.3.4:8080",
                         error_class="site_error", error="HTTP 403",
                         latency_s=2.0),
                StageLog(stage="browser", url="https://x.test/a.mp3",
                         error_class="no_media_found",
                         error="no media tags", latency_s=3.0),
            ],
            failure=FailureKind.SITE_ERROR,
            likely_cause="mixed failures across stages",
            hint="try again later")
        text = report.summary()
        for token in ("direct", "proxy", "browser", "network_failure",
                      "site_error", "no_media_found", "likely cause",
                      "hint"):
            self.assertIn(token, text, token)
        self.assertNotEqual(text.strip(), "download failed")

    def test_download_never_raises(self):
        dl = MediaDownloader(_ctx(tempfile.mkdtemp()), direct_retries=1,
                             max_proxies=0)
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("boom")):
            with patch.object(dl, "_proxy_pool", return_value=None):
                with patch.object(dl, "_browser_session", return_value=None):
                    report = dl.download("https://x.test/a.mp3")
        self.assertFalse(report.ok)
        self.assertTrue(report.stages)
        self.assertTrue(report.likely_cause)

    def test_empty_url_honest(self):
        dl = MediaDownloader(_ctx(tempfile.mkdtemp()))
        report = dl.download("")
        self.assertFalse(report.ok)
        self.assertIn("no URL", report.likely_cause)


class LoggingTests(unittest.TestCase):
    def test_structured_stage_logs(self):
        dl = MediaDownloader(_ctx(tempfile.mkdtemp()), direct_retries=1,
                             max_proxies=0)
        with self.assertLogs("nomorals.media.downloader",
                             level="INFO") as cm:
            with patch("nomorals.tools.media.download",
                       side_effect=RuntimeError("connection reset")):
                with patch.object(dl, "_proxy_pool", return_value=None):
                    with patch.object(dl, "_browser_session",
                                      return_value=None):
                        dl.download("https://x.test/a.mp3")
        blob = "\n".join(cm.output)
        self.assertIn("stage=direct", blob)
        self.assertIn("error_class=network_failure", blob)
        self.assertIn("stage=proxy", blob)  # skipped stage is logged too


class ProxyPoolMethodTests(unittest.TestCase):
    """record_failure / record_success / healthy_proxies on the real
    connector with an in-memory vault."""

    def setUp(self):
        from nomorals.accounts.vault import CredentialVault
        from nomorals.connectors.proxypool import ProxyPoolConnector
        from nomorals.storage.db import Database
        vault = CredentialVault(Database(":memory:"),
                                master_passphrase="test")
        self.pool = ProxyPoolConnector(vault)
        self.pool.add_proxy("10.9.9.1", 8080, protocol="http")
        self.pool.add_proxy("10.9.9.2", 8443, protocol="https")

    def test_record_failure_demotes(self):
        healthy = self.pool.healthy_proxies()
        # fresh proxies have unknown health — force healthy first
        for p in self.pool._load_all():
            self.pool.record_success(p.username)
        self.assertEqual(len(self.pool.healthy_proxies()), 2)
        pid = self.pool.healthy_proxies()[0]["id"]
        self.pool.record_failure(pid, "connection refused")
        remaining = [p["id"] for p in self.pool.healthy_proxies()]
        self.assertNotIn(pid, remaining)
        self.assertEqual(len(remaining), 1)

    def test_record_success_restores(self):
        for p in self.pool._load_all():
            self.pool.record_success(p.username)
        pid = self.pool.healthy_proxies()[0]["id"]
        self.pool.record_failure(pid, "boom")
        self.assertEqual(len(self.pool.healthy_proxies()), 1)
        self.pool.record_success(pid, latency_ms=12.0)
        self.assertEqual(len(self.pool.healthy_proxies()), 2)

    def test_healthy_proxies_carry_proxy_url(self):
        for p in self.pool._load_all():
            self.pool.record_success(p.username)
        for p in self.pool.healthy_proxies():
            self.assertIn("proxy_url", p)
            self.assertTrue(p["proxy_url"].startswith(
                ("http://", "https://")))


class ToolWiringTests(unittest.TestCase):
    """media_download routes through the orchestrator."""

    def _registry(self):
        from types import SimpleNamespace
        ctx = _ctx(tempfile.mkdtemp())
        registry = SimpleNamespace(context=ctx, _tools={})

        def register(name, description="", capability=None):
            def deco(fn):
                registry._tools[name] = fn
                return fn
            return deco

        registry.register = register
        from nomorals.tools import media as media_mod
        media_mod.register(registry)
        return registry._tools["media_download"]

    def test_tool_success_shape(self):
        tool = self._registry()
        report = DownloadReport(ok=True, url="https://x.test/a.mp3",
                                path="/tmp/a.mp3", size_bytes=10,
                                title="t", extractor="yt_dlp", total_s=1.0,
                                stages=[StageLog(stage="direct", ok=True)])
        with patch("nomorals.media.downloader.MediaDownloader") as MD:
            MD.return_value.download.return_value = report
            out = tool("https://x.test/a.mp3", audio_only=True)
        self.assertEqual(out["path"], "/tmp/a.mp3")
        self.assertEqual(out["stages"], ["direct"])

    def test_tool_failure_raises_with_stage_summary(self):
        from nomorals.core.errors import MediaError
        tool = self._registry()
        report = DownloadReport(
            url="https://x.test/a.mp3",
            stages=[StageLog(stage="direct",
                             error_class="network_failure",
                             error="timeout"),
                    StageLog(stage="proxy", note="pool empty"),
                    StageLog(stage="browser",
                             error_class="no_media_found",
                             error="no media tags")],
            failure=FailureKind.NETWORK_FAILURE,
            likely_cause="network unreachable",
            hint="add proxies")
        with patch("nomorals.media.downloader.MediaDownloader") as MD:
            MD.return_value.download.return_value = report
            with self.assertRaises(MediaError) as cm:
                tool("https://x.test/a.mp3")
        text = str(cm.exception)
        self.assertIn("direct", text)
        self.assertIn("proxy", text)
        self.assertIn("browser", text)
        self.assertIn("likely cause", text)

    def test_tool_accepts_page_url(self):
        tool = self._registry()
        report = DownloadReport(ok=True, url="u", path="/tmp/a.mp3",
                                size_bytes=1, title="t", extractor="e",
                                total_s=0.1)
        with patch("nomorals.media.downloader.MediaDownloader") as MD:
            MD.return_value.download.return_value = report
            tool("https://cdn.test/a.mp3",
                 page_url="https://blog.test/post")
        _, kwargs = MD.return_value.download.call_args
        self.assertEqual(kwargs.get("page_url"), "https://blog.test/post")


class FetchBytesTests(unittest.TestCase):
    """BrowserSession.fetch_bytes streams bytes with the session's
    opener (cookies/proxy intact)."""

    def test_fetch_bytes_writes_file(self):
        import io
        import urllib.request
        from unittest.mock import MagicMock
        from nomorals.tools.browser import BrowserSession

        session = BrowserSession.__new__(BrowserSession)
        session.user_agent = "test-agent"
        session.url = "https://blog.test/post"
        session.respect_robots = False
        session.timeout = 10
        session.request_count = 0

        body = b"ID3" + b"\x00" * 100
        resp = MagicMock()
        resp.__enter__.return_value = resp
        resp.__exit__.return_value = False
        resp.status = 200
        resp.headers = {"Content-Type": "audio/mpeg"}
        resp.geturl.return_value = "https://cdn.test/a.mp3"
        resp.read.side_effect = [body, b""]
        opener = MagicMock()
        opener.open.return_value = resp
        session._opener = opener

        dest = os.path.join(tempfile.mkdtemp(), "a.mp3")
        got = session.fetch_bytes("https://cdn.test/a.mp3", dest)
        self.assertEqual(got["bytes"], len(body))
        self.assertEqual(got["content_type"], "audio/mpeg")
        self.assertEqual(Path(dest).read_bytes(), body)


if __name__ == "__main__":
    unittest.main()
