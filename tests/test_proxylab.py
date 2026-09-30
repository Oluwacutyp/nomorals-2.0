"""Tests for the restored original proxy lab (wave 83 architecture).

Covers the real classes recovered from git history (Sep 29 2026):
Proxy, ProxyScraper, SourceRegistry, ProxyLab, plus tool registration.
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

from nomorals.tools import proxylab as PL
from nomorals.tools import proxysources as PS
from nomorals.tools import proxy as PX
from nomorals.tools import ssh_socks as SS


def _ctx():
    home = tempfile.mkdtemp()
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(home=home))


class ProxyTests(unittest.TestCase):
    def test_url_key_roundtrip(self):
        p = PL.Proxy(host="1.2.3.4", port=8080, scheme="http")
        self.assertEqual(p.url, "http://1.2.3.4:8080")
        self.assertEqual(p.key, ("1.2.3.4", 8080, "http"))
        d = p.to_dict()
        p2 = PL.Proxy.from_dict(d)
        self.assertEqual(p2.url, p.url)
        self.assertEqual(p2.host, "1.2.3.4")


class ParseTests(unittest.TestCase):
    def test_parse_list_basic(self):
        text = "1.2.3.4:8080\n5.6.7.8:3128\n# comment\n\nnot a proxy\n"
        out = PL.ProxyScraper.parse_list(text, "http")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].host, "1.2.3.4")
        self.assertEqual(out[0].scheme, "http")

    def test_parse_list_annotated(self):
        text = "1.2.3.4:8080 US fast [elite]\n"
        out = PL.ProxyScraper.parse_list(text, "https")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].port, 8080)

    def test_parse_protocol(self):
        text = "socks5://9.9.9.9:1080\nhttp://1.1.1.1:80\n"
        out = PL.ProxyScraper.parse_protocol(text)
        schemes = {p.scheme for p in out}
        self.assertIn("socks5", schemes)
        self.assertIn("http", schemes)


class ScraperTests(unittest.TestCase):
    def test_scrape_dedupe_and_counts(self):
        def fake_fetch(url):
            if "a" in url:
                return b"1.2.3.4:8080\n1.2.3.4:8080\n5.6.7.8:3128\n"
            return b"9.9.9.9:1080\n"

        scraper = PL.ProxyScraper(
            fetcher=fake_fetch,
            sources=[("s-a", "http://a/list.txt", "list"),
                     ("s-b", "http://b/list.txt", "list")],
        )
        report = scraper.scrape("http")
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["by_source"]["s-a"], 2)
        self.assertEqual(report["by_source"]["s-b"], 1)
        self.assertEqual(report["errors"], {})

    def test_scrape_isolates_dead_source(self):
        def fake_fetch(url):
            if "dead" in url:
                raise ConnectionError("boom")
            return b"1.2.3.4:8080\n"

        scraper = PL.ProxyScraper(
            fetcher=fake_fetch,
            sources=[("s-dead", "http://dead/x.txt", "list"),
                     ("s-ok", "http://ok/x.txt", "list")],
        )
        report = scraper.scrape("http")
        self.assertEqual(report["total"], 1)
        self.assertIn("s-dead", report["errors"])
        self.assertNotIn("s-ok", report["errors"])


class SourceRegistryTests(unittest.TestCase):
    def _reg(self):
        tmp = tempfile.mkdtemp()
        return PS.SourceRegistry(Path(tmp) / "sources.json")

    def test_sources_lists_builtins(self):
        reg = self._reg()
        srcs = reg.sources()
        names = [n for n, _, _ in srcs]
        self.assertIn("thespeedx-http", names)
        self.assertIn("monosans-json", names)
        kinds = {k for _, _, k in srcs}
        self.assertTrue({"list", "html", "json", "protocol"} <= kinds)

    def test_record_success_clears_fails(self):
        reg = self._reg()
        reg.sources()
        reg.record("fpl-http", ok=False, error="404")
        reg.record("fpl-http", ok=False, error="404")
        reg.record("fpl-http", ok=True, found=100)
        self.assertEqual(reg.disabled(), [])
        st = reg.stats()
        self.assertEqual(st["disabled"], 0)

    def test_three_failures_disables_for_cooldown(self):
        reg = self._reg()
        reg.sources()
        for _ in range(3):
            reg.record("fpl-http", ok=False, error="timeout")
        dis = reg.disabled()
        self.assertEqual(len(dis), 1)
        self.assertEqual(dis[0]["name"], "fpl-http")
        self.assertGreater(dis[0]["disabled_for_hours"], 23.0)
        # disabled source excluded from active list
        names = [n for n, _, _ in reg.sources()]
        self.assertNotIn("fpl-http", names)

    def test_add_discovered_and_forget(self):
        reg = self._reg()
        reg.sources()
        reg.add_discovered("my-src", "https://example.com/p.txt", "list",
                           seed="https://seed", found=42)
        st = reg.stats()
        self.assertEqual(st["discovered"], 1)
        names = [n for n, _, _ in reg.sources()]
        self.assertIn("my-src", names)
        self.assertTrue(reg.forget("my-src"))
        self.assertFalse(reg.forget("my-src"))

    def test_health_shape(self):
        reg = self._reg()
        reg.sources()
        reg.record("thespeedx-http", ok=True, found=1500)
        h = {e["name"]: e for e in reg.health()}
        e = h["thespeedx-http"]
        self.assertEqual(e["tests"], 1)
        self.assertEqual(e["fails"], 0)
        self.assertEqual(e["last_found"], 1500)
        self.assertFalse(e["disabled"])


class ProxyLabTests(unittest.TestCase):
    def test_scrape_feeds_registry_health(self):
        lab = PL.ProxyLab(_ctx())
        lab._scraper = PL.ProxyScraper(
            fetcher=lambda url: b"1.2.3.4:8080\n",
            sources=[("s-ok", "http://ok/x.txt", "list")],
        )
        report = lab.scrape("http", sources=[("s-ok", "http://ok/x.txt", "list")])
        self.assertGreater(report["total"], 0)
        self.assertIn("source_stats", report)
        self.assertIn("disabled_sources", report)


class RegistrationTests(unittest.TestCase):
    def test_all_proxy_tools_register(self):
        from nomorals.tools.registry import ToolRegistry
        r = ToolRegistry()
        for mod in (PX, PL, SS):
            mod.register(r)
        names = set(r._tools)
        for expected in ["proxy_status", "proxy_list", "proxy_set",
                         "proxy_clear", "proxy_test", "proxy_scrape",
                         "proxy_discover", "proxy_sources", "proxy_refresh",
                         "proxy_pool", "proxy_file", "proxy_rotate",
                         "ssh_socks"]:
            self.assertIn(expected, names, expected)


if __name__ == "__main__":
    unittest.main()
