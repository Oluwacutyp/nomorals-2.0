"""Fetch resilience (Pass 2) — the two owner-reported failures, pinned.

FAILURE 1: a chat research mission died in 2s because one search hit
was a Bing click-tracking URL whose host robots.txt vetoed the fetch,
the veto was raised as a step-killing RuntimeError, and the step retry
fetched the identical URL again.  Pinned here:
- tracking URLs unwrap to their real destination (parser + search +
  fetch entry + pipeline intake);
- a disallowed/dead source is SKIPPED with a journaled reason — the
  research step completes on the sources that answer;
- a failed URL is never fetched identically twice in a run (ledger);
- the alternate route (browser) is tried once before skipping.

FAILURE 2: a TikTok short-link download failed on every leg with no
usable information.  Pinned here:
- short links expand to canonical URLs as step zero;
- yt-dlp receives Devon's own browser-profile cookies when she holds
  a login for the host;
- the browser leg mines embedded JSON state (playAddr/contentUrl),
  not just og tags;
- every leg reports a real reason (proxy legs name proxy + protocol),
  and the chat rendering never shows a bare "failed";
- ANSI escapes from subprocess tools never reach the report.
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from nomorals.core.http import unwrap_redirect_url
from nomorals.core.result import Err, Ok
from nomorals.core.text import strip_ansi

DEST = "https://example.com/story"
CK_A = ("https://www.bing.com/ck/a?!&&p=abc123"
        "&u=a1" + base64.urlsafe_b64encode(DEST.encode()).decode().rstrip("=")
        + "&ntb=1")


# ── unwrap ────────────────────────────────────────────────────────────


class UnwrapTests(unittest.TestCase):
    def test_bing_ck_a_base64(self):
        self.assertEqual(unwrap_redirect_url(CK_A), DEST)

    def test_bing_ck_a_plain_u(self):
        url = "https://www.bing.com/ck/a?u=https://example.org/plain&x=1"
        self.assertEqual(unwrap_redirect_url(url), "https://example.org/plain")

    def test_google_url_param(self):
        url = "https://www.google.com/url?q=https%3A%2F%2Fexample.net%2Fa&sa=U"
        self.assertEqual(unwrap_redirect_url(url), "https://example.net/a")

    def test_ddg_uddg(self):
        url = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Falpha&rut=x"
        self.assertEqual(unwrap_redirect_url(url), "https://example.com/alpha")

    def test_yahoo_ru(self):
        url = ("https://r.search.yahoo.com/_ylt=A/RV=2/RE=1/RO=10/"
               "RU=https%3A%2F%2Fexample.org%2Fy/RK=2/RS=z")
        self.assertEqual(unwrap_redirect_url(url), "https://example.org/y")

    def test_ordinary_url_unchanged(self):
        self.assertEqual(unwrap_redirect_url(DEST), DEST)
        self.assertEqual(unwrap_redirect_url("https://www.bing.com/search?q=x"),
                         "https://www.bing.com/search?q=x")

    def test_garbage_u_unchanged(self):
        url = "https://www.bing.com/ck/a?u=a1!!!not-base64!!!"
        self.assertEqual(unwrap_redirect_url(url), url)


class BingParseTests(unittest.TestCase):
    def test_parse_bing_unwraps_ck_a(self):
        from nomorals.tools import web
        markup = f"""
        <ol><li class="b_algo"><h2><a href="{CK_A}">Story title</a></h2>
        <div class="b_caption"><p>snippet</p></div></li></ol>"""
        results = web._parse_bing(markup)
        self.assertEqual([r["url"] for r in results], [DEST])


# ── research agent resilience ─────────────────────────────────────────


class _ResearchRegistry:
    """Registry fake: web_search returns canned hits; web_fetch fails
    for URLs in ``dead`` with the given message; browser optional."""

    def __init__(self, hits, dead=None, browser_text=""):
        self.hits = hits
        self.dead = dead or {}
        self.browser_text = browser_text
        self.fetch_calls: list[str] = []
        self.browser_calls: list[str] = []

    def call(self, name, *, actor="system", capabilities=None,
             idempotency_key=None, **kwargs):
        if name == "web_search":
            return Ok({"results": list(self.hits)})
        if name == "web_fetch":
            url = kwargs.get("url", "")
            self.fetch_calls.append(url)
            if url in self.dead:
                return Err(RuntimeError(self.dead[url]))
            return Ok({"url": url, "title": "T", "text": "body text"})
        if name == "browser":
            self.browser_calls.append(kwargs.get("action", ""))
            if kwargs.get("action") == "text":
                if self.browser_text:
                    return Ok({"text": self.browser_text})
                return Err(RuntimeError("no page"))
            return Ok({"ok": True})
        return Err(ValueError(f"unknown tool {name}"))


def _research_ctx(registry):
    return SimpleNamespace(
        tools=registry, memory=None, router=None,
        settings=SimpleNamespace(), extras={})


def _research_agent(registry):
    from nomorals.agents.roles import ResearchAgent
    return ResearchAgent(context=_research_ctx(registry))


class ResearchAgentResilienceTests(unittest.TestCase):
    def test_robots_dead_source_skipped_step_completes(self):
        reg = _ResearchRegistry(
            hits=[{"url": "https://blocked.test/a", "title": "A"},
                  {"url": "https://good.test/b", "title": "B"}],
            dead={"https://blocked.test/a":
                  "robots.txt disallows https://blocked.test/a"})
        out = _research_agent(reg).work("research money")
        self.assertNotIn("error", out)
        self.assertEqual([s["url"] for s in out["sources"]],
                         ["https://good.test/b"])
        self.assertEqual(len(out["skipped"]), 1)
        self.assertEqual(out["skipped"][0]["url"], "https://blocked.test/a")
        self.assertIn("robots.txt", out["skipped"][0]["reason"])

    def test_tracker_hit_is_fetched_at_destination(self):
        reg = _ResearchRegistry(hits=[{"url": CK_A, "title": "Story"}])
        out = _research_agent(reg).work("research money")
        self.assertEqual(reg.fetch_calls, [DEST])
        self.assertEqual(out["sources"][0]["url"], DEST)

    def test_browser_route_rescues_fetch_failure(self):
        reg = _ResearchRegistry(
            hits=[{"url": "https://js.test/page", "title": "JS"}],
            dead={"https://js.test/page": "403 bot wall"},
            browser_text="rendered article body")
        out = _research_agent(reg).work("research money")
        self.assertNotIn("error", out)
        self.assertEqual(out["sources"][0]["url"], "https://js.test/page")
        self.assertIn("open", reg.browser_calls)

    def test_all_sources_dead_reports_every_skip(self):
        reg = _ResearchRegistry(
            hits=[{"url": "https://a.test/1", "title": "1"},
                  {"url": "https://b.test/2", "title": "2"}],
            dead={"https://a.test/1": "robots.txt disallows x",
                  "https://b.test/2": "connection reset"})
        out = _research_agent(reg).work("research money")
        self.assertIn("no sources retrieved", out["error"])
        self.assertIn("robots.txt", out["error"])
        self.assertIn("connection reset", out["error"])
        self.assertEqual(len(out["skipped"]), 2)

    def test_failed_url_never_refetched_within_run(self):
        reg = _ResearchRegistry(
            hits=[{"url": "https://dead.test/x", "title": "X"},
                  {"url": "https://good.test/y", "title": "Y"}],
            dead={"https://dead.test/x": "robots.txt disallows x"})
        agent = _research_agent(reg)
        first = agent.work("research money")
        self.assertNotIn("error", first)
        self.assertEqual(reg.fetch_calls.count("https://dead.test/x"), 1)
        # a step retry (same context = same run) must not fetch the
        # dead URL identically again — it is skipped from the ledger
        second = agent.work("research money")
        self.assertNotIn("error", second)
        self.assertEqual(reg.fetch_calls.count("https://dead.test/x"), 1)
        self.assertTrue(any(s["url"] == "https://dead.test/x"
                            for s in second["skipped"]))

    def test_fresh_run_retries_url(self):
        reg = _ResearchRegistry(
            hits=[{"url": "https://dead.test/x", "title": "X"}],
            dead={"https://dead.test/x": "robots.txt disallows x"})
        _research_agent(reg).work("research money")
        _research_agent(reg).work("research money")  # new context = new run
        self.assertEqual(reg.fetch_calls.count("https://dead.test/x"), 2)


class PipelineIntakeTests(unittest.TestCase):
    def test_run_job_unwraps_tracker_results(self):
        from nomorals.research.pipeline import (
            ResearchContext, ResearchJob, run_job)

        class _Reg:
            def call(self, name, *, actor="system", **kwargs):
                if name == "web_search":
                    return Ok({"results": [
                        {"url": CK_A, "title": "Story", "snippet": "s"}]})
                if name == "web_fetch":
                    return Ok({"url": kwargs.get("url"), "text": "body"})
                return Err(ValueError("unknown"))

        db = SimpleNamespace(query_one=lambda *a, **k: None)
        rctx = ResearchContext(db=db, registry=_Reg())
        job = ResearchJob(id="j1", topic="money", queries=["money"],
                          fetch_top=1)
        findings = run_job(job, rctx)
        self.assertEqual([f.url for f in findings], [DEST])


# ── short links ───────────────────────────────────────────────────────


class ShortLinkTests(unittest.TestCase):
    def test_is_short_link(self):
        from nomorals.media.downloader import is_short_link
        self.assertTrue(is_short_link("https://vt.tiktok.com/ZSgUWn3os/"))
        self.assertTrue(is_short_link("https://vm.tiktok.com/abc/"))
        self.assertTrue(is_short_link("https://www.tiktok.com/t/ZTabc/"))
        self.assertTrue(is_short_link("https://bit.ly/3xyz"))
        self.assertFalse(
            is_short_link("https://www.tiktok.com/@user/video/729123"))
        self.assertFalse(is_short_link("https://example.com/a.mp4"))

    def _downloader(self, tmp):
        from nomorals.media.downloader import MediaDownloader
        ctx = MagicMock()
        ctx.settings = None
        ctx.db = MagicMock()
        return MediaDownloader(ctx, profile="laptop", direct_retries=1,
                               max_proxies=0)

    def test_short_link_expanded_before_stages(self):
        tmp = tempfile.mkdtemp()
        dl = self._downloader(tmp)
        seen = {}

        def direct(url, dest, **k):
            seen["url"] = url
            path = os.path.join(str(dest), "v.mp4")
            Path(path).write_bytes(b"video")
            return {"url": url, "path": path, "bytes": 5, "title": "t",
                    "extractor": "yt_dlp", "seconds": 0.1}

        canonical = "https://www.tiktok.com/@user/video/729123"
        with patch("nomorals.media.downloader.expand_short_url",
                   return_value=canonical) as expand:
            with patch("nomorals.tools.media.download",
                       side_effect=direct):
                report = dl.download("https://vt.tiktok.com/ZSgUWn3os/")
        expand.assert_called_once()
        self.assertTrue(report.ok)
        self.assertEqual(seen["url"], canonical)
        self.assertEqual(report.resolved_url, canonical)
        self.assertEqual(report.stages[0].stage, "resolve")
        self.assertTrue(report.stages[0].ok)

    def test_unexpandable_short_link_continues_with_original(self):
        tmp = tempfile.mkdtemp()
        dl = self._downloader(tmp)

        def direct(url, dest, **k):
            raise RuntimeError("extractor died")

        with patch("nomorals.media.downloader.expand_short_url",
                   side_effect=lambda u: u):
            with patch("nomorals.tools.media.download",
                       side_effect=direct):
                with patch.object(dl, "_proxy_pool", return_value=None):
                    with patch.object(dl, "_browser_session",
                                      return_value=None):
                        report = dl.download("https://vt.tiktok.com/ZZZ/")
        self.assertFalse(report.ok)
        self.assertEqual(report.resolved_url, "")
        resolve = [s for s in report.stages if s.stage == "resolve"]
        self.assertEqual(len(resolve), 1)
        self.assertIn("could not be expanded", resolve[0].note)

    def test_normal_url_never_touches_expander(self):
        tmp = tempfile.mkdtemp()
        dl = self._downloader(tmp)
        with patch("nomorals.media.downloader.expand_short_url") as expand:
            with patch("nomorals.tools.media.download",
                       side_effect=RuntimeError("nope")):
                with patch.object(dl, "_proxy_pool", return_value=None):
                    with patch.object(dl, "_browser_session",
                                      return_value=None):
                        dl.download("https://example.com/a.mp4")
        expand.assert_not_called()


# ── browser-profile cookies ───────────────────────────────────────────


def _cookie_state(domain=".tiktok.com", name="sessionid", value="abc"):
    return {"version": 0, "name": name, "value": value, "port": None,
            "port_specified": False, "domain": domain,
            "domain_specified": True, "domain_initial_dot": True,
            "path": "/", "path_specified": True, "secure": True,
            "expires": 1999999999, "discard": False, "comment": None,
            "comment_url": None, "rest": {}, "rfc2109": False}


class BrowserCookieExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.sessions = Path(self.tmp) / "sessions"
        self.sessions.mkdir(parents=True)
        (self.sessions / "default.json").write_text(json.dumps(
            [_cookie_state(), _cookie_state(".youtube.com", "SID", "y")]))

    def test_export_merges_sessions_to_netscape(self):
        from nomorals.media.cookies import export_browser_cookies
        path = export_browser_cookies(str(self.sessions))
        self.assertIsNotNone(path)
        text = Path(path).read_text()
        self.assertIn("# Netscape HTTP Cookie File", text)
        self.assertIn(".tiktok.com\tTRUE\t/\tTRUE\t1999999999\tsessionid\tabc",
                      text)
        self.assertIn(".youtube.com", text)

    def test_cookies_file_for_prefers_matching_export(self):
        from nomorals.media.cookies import cookies_file_for
        with patch.dict(os.environ, {"HOME": self.tmp}):
            with patch("nomorals.media.cookies.find_cookies_file",
                       return_value=None):
                got = cookies_file_for("https://www.tiktok.com/@u/video/1",
                                       str(self.sessions))
        self.assertIsNotNone(got)
        self.assertTrue(got.endswith("browser-cookies.txt"))

    def test_no_sessions_no_cookies(self):
        from nomorals.media.cookies import (
            cookies_file_for, export_browser_cookies)
        empty = Path(self.tmp) / "empty"
        empty.mkdir()
        self.assertIsNone(export_browser_cookies(str(empty)))
        with patch("nomorals.media.cookies.find_cookies_file",
                   return_value=None):
            self.assertIsNone(
                cookies_file_for("https://www.tiktok.com/x", str(empty)))

    def test_ytdlp_python_api_receives_browser_cookies(self):
        from nomorals.tools.media import _download_python_api
        captured = {}

        class FakeYDL:
            def __init__(self, opts):
                captured.update(opts)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=True):
                target = Path(self._target)
                target.write_bytes(b"vid")
                return {"id": "1", "title": "t", "filepath": str(target)}

            def prepare_filename(self, info):
                return ""

        module = SimpleNamespace(YoutubeDL=FakeYDL)
        target_dir = Path(self.tmp) / "dl"
        target_dir.mkdir()
        FakeYDL._target = str(target_dir / "v.mp4")
        with patch("nomorals.media.cookies.export_browser_cookies",
                   return_value="/profile/browser-cookies.txt"):
            with patch("nomorals.media.cookies.find_cookies_file",
                       return_value=None):
                out = _download_python_api(
                    module, "https://www.tiktok.com/@u/video/1",
                    target_dir, "best", False, None, 0.0)
        self.assertEqual(captured.get("cookiefile"),
                         "/profile/browser-cookies.txt")
        self.assertEqual(out["path"], str(target_dir / "v.mp4"))


# ── embedded JSON mining ──────────────────────────────────────────────


TIKTOK_HTML = """
<html><head><title>t</title></head><body>
<script id="__UNIVERSAL_DATA_FOR_REHYDRATION__" type="application/json">
{"__DEFAULT_SCOPE__":{"webapp.video-detail":{"itemInfo":{"itemStruct":{
"video":{"playAddr":"https:\\/\\/cdn.example.com\\/play\\/abc.mp4",
"downloadAddr":"https:\\/\\/cdn.example.com\\/dl\\/abc.mp4"}}}}}}
</script></body></html>
"""

LD_HTML = """
<html><head>
<script type="application/ld+json">
{"@type":"VideoObject","contentUrl":"https://cdn.example.com/ld.mp4"}
</script></head><body></body></html>
"""


class EmbeddedJsonTests(unittest.TestCase):
    def test_hydration_state_play_addr(self):
        from nomorals.media.downloader import _embedded_media_urls
        urls = _embedded_media_urls(TIKTOK_HTML)
        self.assertIn("https://cdn.example.com/play/abc.mp4", urls)
        self.assertIn("https://cdn.example.com/dl/abc.mp4", urls)

    def test_json_ld_content_url(self):
        from nomorals.media.downloader import _embedded_media_urls
        self.assertEqual(_embedded_media_urls(LD_HTML),
                         ["https://cdn.example.com/ld.mp4"])

    def test_browser_stage_uses_embedded_json(self):
        from nomorals.media.downloader import MediaDownloader
        tmp = tempfile.mkdtemp()
        ctx = MagicMock()
        ctx.settings = None
        ctx.db = MagicMock()
        dl = MediaDownloader(ctx, profile="laptop", direct_retries=1,
                             max_proxies=0)
        session = MagicMock()
        session.url = "https://www.tiktok.com/@u/video/729"
        session.open.return_value = {"ok": True, "status": 200,
                                     "url": session.url}
        session.extract.return_value = {"og": {}, "other": {}}
        session.html.return_value = {"html": TIKTOK_HTML}
        session.links.return_value = {"links": []}

        def fetch_bytes(url, dest):
            Path(dest).write_bytes(b"video-bytes")
            return {"url": url, "status": 200,
                    "content_type": "video/mp4", "bytes": 11,
                    "path": str(dest)}

        session.fetch_bytes.side_effect = fetch_bytes
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError("extractor rot")):
            with patch.object(dl, "_proxy_pool", return_value=None):
                with patch.object(dl, "_browser_session",
                                  return_value=session):
                    report = dl.download(
                        "https://www.tiktok.com/@u/video/729")
        self.assertTrue(report.ok)
        self.assertEqual(report.extractor, "browser-session")
        fetched_url = session.fetch_bytes.call_args[0][0]
        self.assertEqual(fetched_url,
                         "https://cdn.example.com/play/abc.mp4")


# ── per-leg reporting + ANSI ──────────────────────────────────────────


class LegReportingTests(unittest.TestCase):
    def test_strip_ansi(self):
        self.assertEqual(strip_ansi("\x1b[0;31mERROR:\x1b[0m boom"),
                         "ERROR: boom")
        self.assertEqual(strip_ansi("plain"), "plain")
        self.assertEqual(strip_ansi(""), "")

    def test_stage_errors_are_ansi_clean(self):
        from nomorals.media.downloader import MediaDownloader
        tmp = tempfile.mkdtemp()
        ctx = MagicMock()
        ctx.settings = None
        ctx.db = MagicMock()
        dl = MediaDownloader(ctx, profile="laptop", direct_retries=1,
                             max_proxies=0)
        with patch("nomorals.tools.media.download",
                   side_effect=RuntimeError(
                       "\x1b[0;31mERROR:\x1b[0m extractor died")):
            with patch.object(dl, "_proxy_pool", return_value=None):
                with patch.object(dl, "_browser_session",
                                  return_value=None):
                    report = dl.download("https://example.com/a.mp4")
        self.assertNotIn("\x1b", report.stages[0].error)
        self.assertNotIn("\x1b", report.summary())

    def test_proxy_leg_names_proxy_and_protocol(self):
        from nomorals.media.downloader import MediaDownloader
        tmp = tempfile.mkdtemp()
        ctx = MagicMock()
        ctx.settings = None
        ctx.db = MagicMock()
        dl = MediaDownloader(ctx, profile="laptop", direct_retries=1,
                             max_proxies=2)
        pool = MagicMock()
        pool.healthy_proxies.return_value = [
            {"id": "p9", "protocol": "http", "host": "10.0.0.9",
             "port": 8080, "proxy_url": "http://user:pw@10.0.0.9:8080",
             "url": "http://10.0.0.9:8080"}]
        with patch.dict(os.environ, {
                "http_proxy": "", "https_proxy": "", "HTTP_PROXY": "",
                "HTTPS_PROXY": "", "all_proxy": "", "ALL_PROXY": ""}):
            with patch("nomorals.tools.media.download",
                       side_effect=RuntimeError("connection refused")):
                with patch.object(dl, "_proxy_pool", return_value=pool):
                    with patch.object(dl, "_browser_session",
                                      return_value=None):
                        report = dl.download("https://example.com/a.mp4")
        legs = [s for s in report.stages
                if s.stage == "proxy" and s.error]
        self.assertEqual(len(legs), 1)
        self.assertEqual(legs[0].proxy, "http://10.0.0.9:8080")
        self.assertIn("p9", legs[0].note)
        self.assertIn("http", legs[0].note)
        self.assertIn("connection refused", legs[0].error)
        pool.record_failure.assert_called_once()

    def test_chat_rendering_shows_notes_not_bare_failed(self):
        from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
        from nomorals.media.downloader import (
            DownloadReport, FailureKind, StageLog)
        report = DownloadReport(
            url="https://vt.tiktok.com/ZZZ/",
            stages=[
                StageLog(stage="direct", url="https://vt.tiktok.com/ZZZ/",
                         error_class="site_error",
                         error="extractor died", latency_s=1.0),
                StageLog(stage="proxy", url="https://vt.tiktok.com/ZZZ/",
                         note="proxy pool unavailable (no "
                              "NM_VAULT_PASSPHRASE or no stored "
                              "proxies)"),
                StageLog(stage="browser",
                         url="https://vt.tiktok.com/ZZZ/",
                         note="page opened; 0 media candidate(s) "
                              "extracted"),
                StageLog(stage="browser",
                         url="https://vt.tiktok.com/ZZZ/",
                         error_class="no_media_found",
                         error="page has no og:video/og:audio, media "
                               "tags, embedded media JSON, or "
                               "download links", latency_s=2.0),
            ],
            failure=FailureKind.NO_MEDIA_FOUND,
            likely_cause="the source page loaded in the browser but "
                         "exposed no downloadable media link",
            hint="the browser stage already ran with your saved "
                 "browser-session cookies")
        mixin = object.__new__(RuntimeMediaMixin)
        mixin.context = SimpleNamespace()
        with patch("nomorals.media.downloader.MediaDownloader") as MD:
            MD.return_value.download.return_value = report
            text, sent = mixin._download_produce(
                "https://vt.tiktok.com/ZZZ/", False, None)
        self.assertFalse(sent)
        self.assertIn("proxy pool unavailable", text)
        self.assertIn("extractor died", text)
        self.assertIn("likely cause", text)
        self.assertNotIn(": failed", text)
        self.assertNotIn("\x1b", text)


if __name__ == "__main__":
    unittest.main()
