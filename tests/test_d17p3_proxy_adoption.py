"""Doc 17 Pass 3 — downloader callsite adoption + ``_proxy_env`` retirement.

Pins:
- every proxy-stage attempt mints a route receipt through the one
  authority (consumer ``downloader``), with the machinery's real
  reason and credential-free fields; outcomes fold back honestly
  (failed on fetch error, used on a landed file, NONE on a content
  failure the pool itself neither credits nor demotes);
- the fetch receives the decided proxy EXPLICITLY (tools.media's
  ``proxy=`` → yt-dlp's proxy option / ``--proxy`` flag / HttpClient's
  ``proxy_url``) — the stage-1 direct call passes no proxy at all, so
  the resolver → owner-default chain keeps its shipped precedence;
- ``os.environ`` is byte-identical before, during and after download
  cycles (success and failure): the retired process-wide ``*_PROXY``
  window stays retired, and a former env-window reader (pumpdesk's
  ``_proxy_for``) no longer inherits routing it never asked for;
- journaling is fail-open: a broken receipts journal never breaks a
  download.
"""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import nomorals.media.downloader as downloader_module
import nomorals.tools.media as media_module
from nomorals.core.proxy_routing import RouteReceiptJournal
from nomorals.media.downloader import MediaDownloader

PROXY_ENV_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                  "http_proxy", "https_proxy", "all_proxy")
# Fixture credential, assembled from parts at runtime (secaudit rule:
# no full-format secret-shaped literal in source).
PASSWORD = "S3CRET-" + "PIN-991"


def _ctx(tmpdir):
    ctx = MagicMock()
    ctx.home = Path(tmpdir)
    ctx.db = MagicMock()
    ctx.settings = None
    return ctx


def _proxy(pid, proto="https", port=8080, tier=None):
    row = {"id": pid, "protocol": proto, "host": "10.0.0.1",
           "port": port,
           "proxy_url": f"{proto}://user:{PASSWORD}@10.0.0.1:{port}",
           "url": f"{proto}://10.0.0.1:{port}"}
    if tier:
        row["tier"] = tier
    return row


def _pool(proxies):
    pool = MagicMock()
    pool.healthy_proxies.return_value = proxies
    return pool


def _env_snapshot():
    return {k: os.environ.get(k) for k in PROXY_ENV_KEYS}


class _ChainBase(unittest.TestCase):
    """A downloader over a fake pool + a fake tools.media.download."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dl = MediaDownloader(_ctx(self.tmp), direct_retries=1,
                                  max_proxies=3)
        self._env = patch.dict(os.environ,
                               {k: "" for k in PROXY_ENV_KEYS})
        self._env.start()
        self.audio = os.path.join(self.tmp, "p.mp3")
        Path(self.audio).write_bytes(b"data")
        self.calls = []        # (proxy kwarg value, "proxy" in kwargs)
        self.env_during = []   # env snapshots taken inside the fetch

    def tearDown(self):
        self._env.stop()

    def _fake_download(self, fail_ports=()):
        def direct(url, dest, **k):
            proxy = k.get("proxy", "")
            self.calls.append((proxy, "proxy" in k))
            self.env_during.append(_env_snapshot())
            if not proxy:
                raise RuntimeError("connection reset by peer")
            if any(str(port) in proxy for port in fail_ports):
                raise RuntimeError("connection refused")
            return {"url": url, "path": self.audio, "bytes": 4,
                    "title": "t", "extractor": "yt_dlp", "seconds": 0.1}
        return direct

    def _run(self, pool, fail_ports=()):
        with patch("nomorals.tools.media.download",
                   side_effect=self._fake_download(fail_ports)), \
                patch.object(self.dl, "_proxy_pool",
                             return_value=pool), \
                patch.object(self.dl, "_stage_browser",
                             return_value=False):
            return self.dl.download("https://x.test/a.mp3")

    def _journal(self):
        return RouteReceiptJournal(
            Path(self.tmp) / "proxies" / "route_receipts.jsonl")


class ReceiptMintingTests(_ChainBase):
    def test_attempts_mint_receipts_and_outcomes_fold(self):
        pool = _pool([_proxy("p1", port=8081), _proxy("p2", port=8082)])
        report = self._run(pool, fail_ports=(8081,))
        self.assertTrue(report.ok)
        rows = self._journal().recent(consumer="downloader")
        self.assertEqual([r["proxy_id"] for r in rows], ["p1", "p2"])
        first, second = rows
        for row in rows:
            self.assertEqual(row["consumer"], "downloader")
            self.assertEqual(row["source"], "proxypool")
            self.assertEqual(row["selected_by"], "healthy_order")
            self.assertEqual(row["reason_code"], "healthy_order")
            self.assertEqual(row["host"], "x.test")
            self.assertEqual(row["candidates"], 2)
            self.assertEqual(row["candidate_tiers"], {"free": 2})
            self.assertEqual(row["route"], "free")
            self.assertTrue(row["authenticated"])
        self.assertIn("candidate 1 of 2", first["reason"])
        self.assertIn("candidate 2 of 2", second["reason"])
        # the failed attempt's outcome is its failure; the landed
        # attempt's outcome is used — folded onto the decisions
        self.assertEqual(first["outcome"], "failed")
        self.assertIn("network_failure", first["outcome_detail"])
        self.assertEqual(second["outcome"], "used")

    def test_receipts_never_carry_credentials(self):
        pool = _pool([_proxy("p1", port=8081)])
        report = self._run(pool)
        self.assertTrue(report.ok)
        raw = (Path(self.tmp) / "proxies" / "route_receipts.jsonl") \
            .read_text(encoding="utf-8")
        self.assertNotIn(PASSWORD, raw)
        self.assertNotIn("user:", raw)
        for row in self._journal().recent():
            self.assertNotIn("proxy_url", row)
            self.assertNotIn("password", row)
            self.assertNotIn("url_with_auth", row)

    def test_paid_tier_row_receipts_a_paid_route(self):
        pool = _pool([_proxy("vip", port=8089, tier="paid")])
        report = self._run(pool)
        self.assertTrue(report.ok)
        rows = self._journal().recent(consumer="downloader")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["route"], "paid")
        self.assertEqual(rows[0]["tier"], "paid")
        self.assertEqual(rows[0]["candidate_tiers"], {"paid": 1})

    def test_broken_journal_never_breaks_the_download(self):
        # "proxies" as a regular file: every journal append fails.
        Path(self.tmp, "proxies").write_text("not a directory")
        pool = _pool([_proxy("p1", port=8081)])
        report = self._run(pool)
        self.assertTrue(report.ok)
        pool.record_success.assert_called_once()


class EnvInvarianceTests(_ChainBase):
    def test_env_identical_across_success_and_failure_cycles(self):
        ambient = {k: "" for k in PROXY_ENV_KEYS}
        ambient["HTTPS_PROXY"] = "http://ambient.test:9999"
        with patch.dict(os.environ, ambient):
            before = _env_snapshot()
            pool = _pool([_proxy("p1", port=8081),
                          _proxy("p2", port=8082)])
            report = self._run(pool, fail_ports=(8081,))
            self.assertTrue(report.ok)
            # every fetch — direct, failed attempt, landed attempt —
            # saw the ambient env exactly; the pool proxies travelled
            # only through the explicit argument
            for seen in self.env_during:
                self.assertEqual(seen, before)
            self.assertEqual(_env_snapshot(), before)
            self.assertEqual(self.calls[0], ("", False))
            self.assertIn("8081", self.calls[1][0])
            self.assertIn("8082", self.calls[2][0])
            self.assertTrue(all(flag for _, flag in self.calls[1:]))
            # a fully failing cycle leaves the env equally untouched
            self.calls.clear()
            self.env_during.clear()
            pool = _pool([_proxy("p1", port=8081)])
            report = self._run(pool, fail_ports=(8081,))
            self.assertFalse(report.ok)
            for seen in self.env_during:
                self.assertEqual(seen, before)
            self.assertEqual(_env_snapshot(), before)


class PumpdeskLeakTests(_ChainBase):
    def test_former_env_reader_no_longer_inherits_download_routing(self):
        from nomorals.agents.pumpdesk.wsclient import _proxy_for

        url = "wss://socket.test/ws"
        sampled = []

        def direct(url_, dest, **k):
            sampled.append(_proxy_for(url))
            return self._fake_download()(url_, dest, **k)

        pool = _pool([_proxy("p1", port=8081)])
        # the desk resolves from the ambient env: with the proxy vars
        # genuinely ABSENT (their normal state) it connects direct
        with patch.dict(os.environ, {}, clear=False):
            for key in PROXY_ENV_KEYS:
                os.environ.pop(key, None)
            self.assertIsNone(_proxy_for(url))
            with patch("nomorals.tools.media.download",
                       side_effect=direct), \
                    patch.object(self.dl, "_proxy_pool",
                                 return_value=pool), \
                    patch.object(self.dl, "_stage_browser",
                                 return_value=False):
                report = self.dl.download("https://x.test/a.mp3")
            self.assertTrue(report.ok)
            # DURING the proxied fetch the desk's env resolution still
            # saw no proxy — under the retired _proxy_env window it
            # returned this download's pool proxy URL.
            self.assertTrue(sampled)
            self.assertTrue(all(v is None for v in sampled))
            self.assertIsNone(_proxy_for(url))


class ExplicitChannelTests(unittest.TestCase):
    """tools.media.download threads proxy= to every backend."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_python_api_receives_the_proxy_option(self):
        captured = {}
        target = Path(self.tmp)

        class FakeYDL:
            def __init__(self, opts):
                captured.update(opts)

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def extract_info(self, url, download=True):
                landed = target / "landed.mp4"
                landed.write_bytes(b"\x00\x00\x00\x18ftypmp42")
                return {"id": "abc", "title": "t", "extractor": "fake",
                        "filepath": str(landed)}

            def prepare_filename(self, info):
                return info["filepath"]

        fake_module = SimpleNamespace(YoutubeDL=FakeYDL)
        with patch.object(media_module, "load_optional",
                          return_value=fake_module):
            out = media_module.download(
                "https://x.test/v", self.tmp,
                proxy="http://10.0.0.1:8080")
        self.assertEqual(captured.get("proxy"), "http://10.0.0.1:8080")
        self.assertEqual(out["extractor"], "yt_dlp")
        captured.clear()
        with patch.object(media_module, "load_optional",
                          return_value=fake_module):
            media_module.download("https://x.test/v", self.tmp)
        self.assertNotIn("proxy", captured)

    def test_cli_receives_the_proxy_flag(self):
        argv_seen = []

        class FakePopen:
            returncode = 0

            def __init__(self, argv, **kwargs):
                argv_seen.append(list(argv))
                out = Path(argv[argv.index("-o") + 1])
                (out.parent / "landed.mp4").write_bytes(b"data")

            def communicate(self, timeout=None):
                return "", ""

        def which(name):
            return "/usr/bin/yt-dlp" if name == "yt-dlp" else None

        with patch.object(media_module, "load_optional",
                          return_value=None), \
                patch.object(media_module, "which", side_effect=which), \
                patch.object(media_module.subprocess, "Popen",
                             FakePopen):
            media_module.download(
                "https://x.test/v", self.tmp,
                proxy="http://10.0.0.1:8080")
            media_module.download("https://x.test/v", self.tmp)
        proxied, plain = argv_seen
        self.assertIn("--proxy", proxied)
        self.assertEqual(proxied[proxied.index("--proxy") + 1],
                         "http://10.0.0.1:8080")
        self.assertNotIn("--proxy", plain)

    def test_direct_fallback_explicit_proxy_and_default_chain(self):
        captured = {}

        class FakeClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def request(self, method, url, stream_to=None, resume=True,
                        progress=None):
                Path(stream_to).write_bytes(b"\xff\xfb\x90\x00payload")
                return SimpleNamespace(
                    headers={"content-type": "audio/mpeg"})

        from nomorals.core.http import set_default_proxy
        set_default_proxy("http://owner.test:3128")
        try:
            with patch.object(media_module, "load_optional",
                              return_value=None), \
                    patch.object(media_module, "which",
                                 side_effect=lambda name: None), \
                    patch("nomorals.core.http.HttpClient", FakeClient):
                media_module.download("https://x.test/a.mp3",
                                      self.tmp)
                # no decision: the explicit slot stays empty, so
                # HttpClient's resolver → owner-default chain applies
                # exactly as before
                self.assertEqual(captured.get("proxy_url"), "")
                media_module.download(
                    "https://x.test/a.mp3", self.tmp,
                    proxy="http://10.0.0.1:8080")
                # a decided route is explicit — and explicit outranks
                # the owner default in HttpClient
                self.assertEqual(captured.get("proxy_url"),
                                 "http://10.0.0.1:8080")
        finally:
            set_default_proxy("")


class RetirementPinTests(unittest.TestCase):
    def test_proxy_env_is_gone(self):
        self.assertFalse(hasattr(downloader_module, "_proxy_env"))


if __name__ == "__main__":
    unittest.main()
