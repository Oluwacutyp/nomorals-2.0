"""gapfix147: fresh free proxy sources, live-verified 2026-11-25.

The owner reported the proxy Refresh reading 0 candidates across
most visible scrape steps while the lab still tested 150 and found
17 working: the sources were mostly alive, but the first-come scrape
cap let the first sources fill the whole pool, so every later source
reported 0 new. This lane fixes both halves — fresh sources found
LIVE on the internet (not from memory) plus a fair-share cap — and
retires the sources that are actually dead.

Sample bodies below are REAL excerpts fetched 2026-11-25 from the
sources' own list URLs (full 100-line samples live under
~/workspace/proxysources-verify/samples/); unit tests never
live-fetch. A test pin here may change only when factually wrong
about the code, with the reason named in the test.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path

from nomorals.tools import proxy_sources as TPS
from nomorals.tools import proxylab as PL
from nomorals.tools import proxysources as PS


def _ctx(test=None):
    home = tempfile.mkdtemp()
    if test is not None:
        test.addCleanup(shutil.rmtree, home, ignore_errors=True)
    return types.SimpleNamespace(
        settings=types.SimpleNamespace(home=home))


# -- the 13 new sources: (name, url, kind), as listed in the catalog --
NEW_SOURCES: list[tuple[str, str, str]] = [
    ('hproxy-http', 'https://raw.githubusercontent.com/hproxy-com/free-proxy-list/main/http.txt', 'list'),
    ('hproxy-https', 'https://raw.githubusercontent.com/hproxy-com/free-proxy-list/main/https.txt', 'list'),
    ('hproxy-socks4', 'https://raw.githubusercontent.com/hproxy-com/free-proxy-list/main/socks4.txt', 'list'),
    ('hproxy-socks5', 'https://raw.githubusercontent.com/hproxy-com/free-proxy-list/main/socks5.txt', 'list'),
    ('relayglass-http', 'https://raw.githubusercontent.com/relayglass/free-proxy-list/main/protocol/http/http.txt', 'list'),
    ('relayglass-https', 'https://raw.githubusercontent.com/relayglass/free-proxy-list/main/protocol/https/https.txt', 'list'),
    ('relayglass-socks4', 'https://raw.githubusercontent.com/relayglass/free-proxy-list/main/protocol/socks4/socks4.txt', 'list'),
    ('relayglass-socks5', 'https://raw.githubusercontent.com/relayglass/free-proxy-list/main/protocol/socks5/socks5.txt', 'list'),
    ('proxio-http', 'https://raw.githubusercontent.com/proxio-io/proxy-list/main/http.txt', 'list'),
    ('proxio-https', 'https://raw.githubusercontent.com/proxio-io/proxy-list/main/https.txt', 'list'),
    ('proxio-socks4', 'https://raw.githubusercontent.com/proxio-io/proxy-list/main/socks4.txt', 'list'),
    ('proxio-socks5', 'https://raw.githubusercontent.com/proxio-io/proxy-list/main/socks5.txt', 'list'),
    ('pubproxy-json', 'http://pubproxy.com/api/proxy?limit=20', 'json'),
]


# -- real fetched excerpts, 2026-11-25 (first 10 lines / full JSON) --
SAMPLES: dict[str, str] = {
    'hproxy-http': "159.203.61.169:3128\n137.66.1.45:80\n213.188.192.1:80\n159.203.61.169:8080\n198.199.86.11:8080\n198.199.86.11:80\n198.199.86.11:3128\n209.97.150.167:8080\n209.97.150.167:80\n209.97.150.167:3128\n",
    'hproxy-https': "3.21.156.147:20303\n167.99.8.113:8018\n47.89.184.18:3128\n18.222.28.236:25137\n3.231.160.150:9459\n191.101.239.34:9595\n165.154.162.73:8888\n45.79.62.91:3128\n40.177.104.199:12985\n98.94.14.234:23987\n",
    'hproxy-socks4': "172.104.210.252:1080\n64.49.67.164:5678\n70.60.132.130:5678\n107.181.161.81:4145\n199.187.210.54:4145\n206.220.175.2:4145\n107.152.98.5:4145\n67.201.58.190:4145\n104.37.135.145:4145\n192.111.134.10:4145\n",
    'hproxy-socks5': "172.104.210.252:1080\n159.203.61.169:1080\n107.174.30.93:1080\n209.97.150.167:1080\n198.199.86.11:1080\n167.99.3.26:6178\n167.99.3.26:5714\n138.68.60.8:1080\n47.251.249.48:7777\n167.99.3.26:9197\n",
    'relayglass-http': "152.53.183.107:8081\n213.157.6.50:80\n157.90.10.50:8080\n157.90.10.50:8888\n157.90.10.50:5955\n157.90.10.50:999\n103.237.102.191:11111\n109.199.119.160:80\n45.86.245.81:8080\n116.202.108.111:3128\n",
    'relayglass-https': "152.53.183.107:8081\n103.237.102.191:11111\n45.86.245.81:8080\n62.141.38.35:3128\n158.255.212.55:9090\n129.151.240.183:1080\n169.58.97.115:1080\n213.197.136.218:5678\n185.87.255.54:1080\n185.87.255.47:1080\n",
    'relayglass-socks4': "46.21.186.252:1080\n129.151.240.183:1080\n213.197.136.218:5678\n37.52.13.164:5678\n45.151.102.248:10808\n170.168.101.28:1080\n45.61.188.134:44499\n107.150.41.226:18080\n103.41.33.169:51951\n8.219.94.23:7890\n",
    'relayglass-socks5': "169.58.97.115:1080\n185.87.255.54:1080\n185.87.255.47:1080\n104.245.245.218:1080\n47.245.165.201:1080\n161.35.90.93:1083\n109.205.182.143:1088\n45.138.157.212:1080\n213.32.70.99:3128\n85.174.84.22:1080\n",
    'proxio-http': "203.175.103.67:3125\n85.112.65.130:8080\n103.105.76.10:3125\n77.52.187.196:18080\n192.81.249.2:999\n191.103.219.225:48612\n84.44.39.159:9090\n210.79.141.88:8080\n163.227.168.130:8080\n185.129.0.78:8080\n",
    'proxio-https': "103.144.54.73:8082\n200.59.191.27:999\n103.130.61.61:8081\n189.51.168.165:999\n160.22.198.17:8082\n185.142.156.203:80\n103.175.238.102:8082\n49.149.24.144:8082\n103.176.96.32:8082\n103.20.191.118:8080\n",
    'proxio-socks4': "116.97.117.27:4153\n103.157.116.179:10810\n154.88.189.130:5678\n95.46.105.163:4145\n177.91.35.202:4153\n105.27.180.194:1080\n181.129.182.138:5678\n194.50.50.100:1080\n177.85.65.177:4153\n179.1.134.73:1080\n",
    'proxio-socks5': "147.45.233.151:1080\n31.41.227.175:2080\n119.73.109.210:1080\n94.141.178.82:1088\n8.218.48.35:8099\n103.124.172.134:1080\n175.107.30.126:1080\n170.233.72.43:1080\n45.194.33.12:30001\n160.191.49.84:1080\n",
    'pubproxy-json': "{\"data\":[{\"ipPort\":\"74.82.50.155:3128\",\"ip\":\"74.82.50.155\",\"port\":\"3128\",\"country\":\"US\",\"last_checked\":\"2026-11-25 12:35:01\",\"proxy_level\":\"anonymous\",\"type\":\"http\",\"speed\":\"4\",\"support\":{\"https\":0,\"get\":1,\"post\":1,\"cookies\":1,\"referer\":1,\"user_agent\":1,\"google\":0}},{\"ipPort\":\"185.49.31.205:8080\",\"ip\":\"185.49.31.205\",\"port\":\"8080\",\"country\":\"PL\",\"last_checked\":\"2026-11-25 12:35:55\",\"proxy_level\":\"anonymous\",\"type\":\"http\",\"speed\":\"1\",\"support\":{\"https\":0,\"get\":1,\"post\":1,\"cookies\":1,\"referer\":1,\"user_agent\":1,\"google\":0}}],\"count\":2}",
}


def _scheme_of(name: str) -> str:
    for token in ("socks4", "socks5", "https", "http"):
        if name.endswith(token):
            return token
    return "http"


class CatalogTests(unittest.TestCase):
    def test_new_sources_present_with_real_urls(self):
        catalog = {n: (u, k) for n, u, k in PS.BUILT_IN_SOURCES}
        for name, url, kind in NEW_SOURCES:
            self.assertIn(name, catalog)
            self.assertEqual(catalog[name], (url, kind))
        # 101 built-ins - 5 retired + 13 fresh = 109. A future
        # catalog change moves this number and updates this pin
        # with its reason; it must never drift silently.
        self.assertEqual(len(PS.BUILT_IN_SOURCES), 109)
        names = [n for n, _u, _k in PS.BUILT_IN_SOURCES]
        self.assertEqual(len(names), len(set(names)))

    def test_new_sources_parse_real_samples(self):
        for name, _url, kind in NEW_SOURCES:
            text = SAMPLES[name]
            if kind == "list":
                out = PL.ProxyScraper.parse_list(text, _scheme_of(name))
            else:
                out = PL.ProxyScraper.parse_json(text)
            self.assertGreater(len(out), 0, name)
            for proxy in out:
                self.assertEqual(proxy.scheme, _scheme_of(name), name)

    def test_parse_json_reads_type_field(self):
        # pubproxy records carry "type", not "protocol": the scheme
        # must survive the parse instead of collapsing to http.
        text = json.dumps({"data": [
            {"ip": "203.0.113.1", "port": 1080, "type": "socks5",
             "country": "CN"},
            {"ip": "198.51.100.2", "port": 3128, "type": "http",
             "country": "RU"},
            {"ip": "192.0.2.3", "port": 8080, "type": "elite",
             "country": "DE"},
        ]})
        out = PL.ProxyScraper.parse_json(text)
        self.assertEqual([p.scheme for p in out],
                         ["socks5", "http", "http"])


class RetirementTests(unittest.TestCase):
    def test_dead_builtins_retired(self):
        catalog = {n for n, _u, _k in PS.BUILT_IN_SOURCES}
        for name in ("spys-http", "spys-mixed", "spys-socks",
                     "thespeedx-https", "proxyfree-https"):
            self.assertNotIn(name, catalog)
            self.assertIn(name, PS.RETIRED_SOURCES)

    def test_dead_tier_entries_unverified_but_listed(self):
        # Kept in SOURCE_BY_NAME with their URLs (the tier test pins
        # proxynova/openproxy.space URLs) so the gap stays visible —
        # but verified False, so refresh_sources never adds them.
        for name in ("proxynova", "proxy-list.download",
                     "proxyscrape-site", "spys-one", "openproxy.space"):
            self.assertIn(name, TPS.SOURCE_BY_NAME)
            self.assertFalse(TPS.SOURCE_BY_NAME[name]["verified"], name)
            self.assertIn(name, PS.RETIRED_SOURCES)

    def test_retired_sources_pruned_from_registry(self):
        # A registry built BEFORE the retirement still holds the
        # dead entries on disk; the next sources() call forgets
        # them, and live entries survive untouched.
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        registry = PS.SourceRegistry(f"{tmp}/sources.json")
        registry.add_discovered(
            "proxynova", "https://www.proxynova.com/proxy-server-list/",
            "html", seed="test")
        registry.add_discovered(
            "spys-http", "https://spys.one/en/http-proxy-list/",
            "html", seed="test")
        registry.add_discovered(
            "hproxy-http",
            "https://raw.githubusercontent.com/hproxy-com/free-proxy-list/main/http.txt",
            "list", seed="test")
        names = {n for n, _u, _k in registry.sources()}
        self.assertNotIn("proxynova", names)
        self.assertNotIn("spys-http", names)
        self.assertIn("hproxy-http", names)
        on_disk = json.loads(
            Path(f"{tmp}/sources.json").read_text(encoding="utf-8"))
        self.assertNotIn("proxynova", on_disk)
        self.assertNotIn("spys-http", on_disk)


class FairShareTests(unittest.TestCase):
    def _scraper(self, bodies, sources):
        def fetch(url):
            if url not in bodies:
                raise OSError("no fixture for " + url)
            return bodies[url]
        return PL.ProxyScraper(fetcher=fetch, sources=sources)

    def test_three_way_split_on_identical_pools(self):
        # Three sources returning the SAME 10 proxies: the old
        # first-come cap gave the first source all 6 and the rest
        # 0; fair share gives each source 2 of a limit-6 pool.
        rows = "".join(f"10.0.0.{i}:8080\n" for i in range(1, 11))
        bodies = {f"http://x/{n}.txt": rows.encode()
                  for n in ("a", "b", "c")}
        sources = [(n, f"http://x/{n}.txt", "list")
                   for n in ("a", "b", "c")]
        report = self._scraper(bodies, sources).scrape("http", limit=6)
        self.assertEqual(report["total"], 6)
        self.assertEqual(report["by_source"], {"a": 2, "b": 2, "c": 2})

    def test_top_up_fills_the_pool_in_catalog_order(self):
        rows = "".join(f"10.0.0.{i}:8080\n" for i in range(1, 11))
        bodies = {f"http://x/{n}.txt": rows.encode()
                  for n in ("a", "b", "c")}
        sources = [(n, f"http://x/{n}.txt", "list")
                   for n in ("a", "b", "c")]
        report = self._scraper(bodies, sources).scrape("http", limit=7)
        self.assertEqual(report["total"], 7)
        self.assertEqual(report["by_source"], {"a": 3, "b": 2, "c": 2})

    def test_tail_source_contributes_when_head_is_huge(self):
        # The owner's exact shape: the head source alone could fill
        # the pool 100x over. The tail source must still contribute.
        big = "".join(f"10.1.0.{i}:8080\n" for i in range(1, 1001))
        small = "".join(f"10.2.0.{i}:8080\n" for i in range(1, 6))
        bodies = {"http://x/big.txt": big.encode(),
                  "http://x/small.txt": small.encode()}
        sources = [("big", "http://x/big.txt", "list"),
                   ("small", "http://x/small.txt", "list")]
        report = self._scraper(bodies, sources).scrape("http", limit=10)
        self.assertEqual(report["total"], 10)
        self.assertEqual(report["by_source"]["small"], 5)

    def test_total_never_exceeds_limit(self):
        rows = "".join(f"10.0.0.{i}:8080\n" for i in range(1, 51))
        bodies = {f"http://x/{n}.txt": rows.encode()
                  for n in ("a", "b")}
        sources = [(n, f"http://x/{n}.txt", "list")
                   for n in ("a", "b")]
        for limit in (1, 2, 3, 5, 13):
            report = self._scraper(bodies, sources).scrape(
                "http", limit=limit)
            self.assertEqual(report["total"], limit)
            self.assertEqual(sum(report["by_source"].values()), limit)


class RefreshTests(unittest.TestCase):
    def test_refresh_with_mocked_fetches_yields_new_sources(self):
        # The default catalog (never explicit sources): every new
        # source contributes candidates from its mocked fetch, and
        # no retired source is scraped at all.
        bodies = {}
        for name, url, _kind in NEW_SOURCES:
            bodies[url] = SAMPLES[name].encode("utf-8")

        def fetch(url):
            if url not in bodies:
                raise OSError("no fixture for " + url)
            return bodies[url]

        lab = PL.ProxyLab(_ctx(self))
        lab._scraper = PL.ProxyScraper(fetcher=fetch)  # noqa: SLF001
        events: list[tuple] = []
        report = lab.scrape(
            "http,https,socks4,socks5", limit=2000,
            on_source=lambda n, a, e: events.append((n, a, e)))
        for name, _url, _kind in NEW_SOURCES:
            self.assertGreater(report["by_source"].get(name, 0), 0, name)
        event_names = {n for n, _a, _e in events}
        for name in ("spys-http", "spys-mixed", "spys-socks",
                     "thespeedx-https", "proxyfree-https"):
            self.assertNotIn(name, event_names)
        self.assertEqual(report["total"],
                         sum(report["by_source"].values()))


if __name__ == "__main__":
    unittest.main()
