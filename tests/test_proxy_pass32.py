"""Pass 32 — proxy subsystem audit fixes.

1. ``tools/proxy.py`` ProxyManager rendered owner proxy URLs verbatim
   in status output, probe results, error strings and log lines — an
   owner proxy URL normally embeds ``user:password@``, so the password
   leaked everywhere the URL was shown.  The pool connector and the
   downloader mask; the manager now does too.

2. ``ProxyPoolConnector.healthy_proxies()`` ignored quarantine: a
   proxy parked via ``quarantine()`` (health untouched, still HEALTHY)
   was skipped by ``select()``/``rotate()`` but kept being handed to
   the media downloader, the pool's production consumer of that method.

3. ``SourceRegistry.add_discovered(found=0)`` stamped ``last_ok=now``
   — a source that was registered but never scraped reported a success
   0 seconds ago and counted as "fresh" in staleness checks.

4. ``proxy_sources.SourceYieldTracker`` (per-source live-rate EWMA,
   auto-quarantine, probe budgets) had NO production caller: the lab
   never recorded a round, never enforced a budget, never skipped a
   yield-quarantined source.  It is now wired into ProxyLab.refresh /
   scrape / source_health.

Plus numeric pins: both EWMAs (pool latency, source yield) converge
to the scripted true value in closed form.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import types
import unittest
from unittest import mock

from nomorals.accounts.vault import CredentialVault
from nomorals.connectors.proxypool import ProxyPoolConnector
from nomorals.core.errors import ToolError
from nomorals.storage.db import Database
from nomorals.tools import proxy as PX
from nomorals.tools import proxylab as PL
from nomorals.tools.proxysources import SourceRegistry

_CRED_URL = "http://alice:secretpass@203.0.113.9:8080"
_PLAIN_URL = "http://198.51.100.7:3128"


def _manager(known: str = "", active: str = "") -> PX.ProxyManager:
    proxy = types.SimpleNamespace(
        probe_url="https://api.ipify.org", probe_timeout=1.0,
        known=known, active=active, url="")
    ctx = types.SimpleNamespace(
        settings=types.SimpleNamespace(proxy=proxy), db=None)
    return PX.ProxyManager(ctx)


# ── 1. ProxyManager must never render proxy credentials ────────────────


class TestProxyManagerMasking(unittest.TestCase):
    def test_status_masks_known_and_active(self):
        m = _manager(known=_CRED_URL, active=_CRED_URL)
        st = m.status()
        self.assertNotIn("secretpass", str(st))
        self.assertEqual(st["known"], ["http://alice:***@203.0.113.9:8080"])
        self.assertEqual(st["active"], "http://alice:***@203.0.113.9:8080")
        # the real URL is still what the manager holds/applies
        self.assertEqual(m._active, _CRED_URL)
        self.assertEqual(m.known(), [_CRED_URL])

    def test_plain_url_untouched(self):
        m = _manager(known=_PLAIN_URL)
        self.assertEqual(m.status()["known"], [_PLAIN_URL])

    def test_probe_result_masks_proxy(self):
        m = _manager()
        result = m.test(_CRED_URL)["results"][0]
        self.assertNotIn("secretpass", str(result))
        self.assertEqual(result["proxy"],
                         "http://alice:***@203.0.113.9:8080")

    def test_set_active_verify_error_masks_proxy(self):
        m = _manager()
        dead = "http://alice:secretpass@127.0.0.1:1"
        with self.assertRaises(ToolError) as cm:
            m.set_active(dead, verify=True)
        self.assertNotIn("secretpass", str(cm.exception))
        self.assertIn("***", str(cm.exception))

    def test_set_active_logs_masked(self):
        m = _manager()
        self.addCleanup(m.clear)  # restore process-wide direct routing
        with self.assertLogs("nomorals.tools.proxy", level="INFO") as logs:
            m.set_active(_PLAIN_URL)
            m.set_active(_CRED_URL)
        joined = "\n".join(logs.output)
        self.assertNotIn("secretpass", joined)
        self.assertIn("alice:***@203.0.113.9", joined)


# ── 2. healthy_proxies() honours quarantine ─────────────────────────────


class TestHealthyProxiesQuarantine(unittest.TestCase):
    def _pool(self) -> ProxyPoolConnector:
        return ProxyPoolConnector(
            CredentialVault(Database(":memory:"), master_passphrase="test"))

    def test_quarantined_proxy_not_listed_healthy(self):
        pool = self._pool()
        pool.add_proxy("198.51.100.23", 8080)
        pid = "http://198.51.100.23:8080"
        with mock.patch.object(pool, "_probe_via_proxy",
                               return_value=(200, 42.0)):
            pool.health_check(pid)
        self.assertEqual([p["id"] for p in pool.healthy_proxies()], [pid])
        pool.quarantine(pid, seconds=600, reason="target blocked us")
        # select()/rotate() already skip it; the downloader's surface
        # (healthy_proxies) must agree
        self.assertEqual(pool.healthy_proxies(), [])
        # release path: a later successful check re-admits it
        pool.record_success(pid, latency_ms=30.0)
        self.assertEqual([p["id"] for p in pool.healthy_proxies()], [pid])


# ── 3. add_discovered(found=0) must not fake a success ─────────────────


class TestAddDiscoveredHonesty(unittest.TestCase):
    def _reg(self) -> SourceRegistry:
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        return SourceRegistry(f"{tmp}/sources.json")

    def test_found_zero_records_no_success(self):
        reg = self._reg()
        entry = reg.add_discovered("never-scraped",
                                   "https://example.com/x.txt", "list",
                                   found=0)
        self.assertEqual(entry["last_ok"], 0.0)
        health = {r["name"]: r for r in reg.health()}
        self.assertIsNone(health["never-scraped"]["last_ok"])

    def test_found_positive_still_records_success(self):
        reg = self._reg()
        entry = reg.add_discovered("real", "https://example.com/y.txt",
                                   "list", found=7)
        self.assertGreater(entry["last_ok"], 0.0)
        self.assertEqual(entry["last_found"], 7)


# ── 4. yield tracker wired into the lab ────────────────────────────────


class _StubTester:
    """ProxyTester stand-in: marks a fixed fraction of each batch alive."""

    def __init__(self, alive_fraction: float = 0.0) -> None:
        self.detect_country = True
        self.alive_fraction = alive_fraction
        self.batches: list[list] = []

    def test_many(self, proxies, *, limit: int = 150):
        batch = list(proxies)[:limit]
        self.batches.append(batch)
        n_alive = round(len(batch) * self.alive_fraction)
        for i, p in enumerate(batch):
            p.alive = i < n_alive
        return batch


def _lab(test: unittest.TestCase) -> PL.ProxyLab:
    home = tempfile.mkdtemp()
    test.addCleanup(shutil.rmtree, home, ignore_errors=True)
    ctx = types.SimpleNamespace(settings=types.SimpleNamespace(home=home))
    return PL.ProxyLab(ctx)


def _seed_candidates(lab: PL.ProxyLab, source: str, n: int,
                     base: int = 0) -> None:
    proxies = [PL.Proxy(host=f"203.0.113.{base + i + 1}", port=8080,
                        scheme="http", source=source)
               for i in range(n)]
    lab.store.save([], candidates=proxies)


class TestYieldWiring(unittest.TestCase):
    def test_refresh_records_yield(self):
        lab = _lab(self)
        _seed_candidates(lab, "src-a", 10)
        lab._tester = _StubTester(alive_fraction=0.5)
        lab.refresh(detect_country=False)
        snap = lab.yield_tracker.snapshot()
        self.assertIn("src-a", snap)
        self.assertEqual(snap["src-a"]["scraped"], 10)
        self.assertEqual(snap["src-a"]["alive"], 5)
        self.assertAlmostEqual(snap["src-a"]["live_rate_ewma"], 0.5)

    def test_refresh_enforces_probe_budget(self):
        lab = _lab(self)
        # proxifly-post is a catalog source with probe budget 2
        _seed_candidates(lab, "proxifly-post", 5)
        tester = _StubTester(alive_fraction=1.0)
        lab._tester = tester
        lab.refresh(detect_country=False)
        tested = [p for batch in tester.batches for p in batch
                  if p.source == "proxifly-post"]
        self.assertEqual(len(tested), 2)
        snap = lab.yield_tracker.snapshot()
        self.assertEqual(snap["proxifly-post"]["probes_spent"], 2)

    def test_dead_source_quarantined_then_skipped_by_scrape(self):
        lab = _lab(self)
        _seed_candidates(lab, "src-dead", 4)
        lab._tester = _StubTester(alive_fraction=0.0)
        for _ in range(3):
            lab.refresh(detect_country=False)
            _seed_candidates(lab, "src-dead", 4)  # restock for next round
        self.assertTrue(lab.yield_tracker.quarantined("src-dead"))
        lab._scraper = PL.ProxyScraper(
            fetcher=lambda url: b"1.2.3.4:8080\n",
            sources=[("src-dead", "http://x/dead.txt", "list"),
                     ("src-ok", "http://x/ok.txt", "list")])
        report = lab.scrape(
            "http",
            sources=[("src-dead", "http://x/dead.txt", "list"),
                     ("src-ok", "http://x/ok.txt", "list")])
        self.assertEqual(report["yield_quarantined"], ["src-dead"])
        self.assertNotIn("src-dead", report["by_source"])
        self.assertIn("src-ok", report["by_source"])

    def test_source_health_includes_yield(self):
        lab = _lab(self)
        self.assertIn("yield", lab.source_health())


# ── EWMA numeric pins (proofs — pass before and after) ──────────────────


class TestEwmaProofs(unittest.TestCase):
    def test_yield_ewma_matches_closed_form(self):
        from nomorals.tools.proxy_sources import SourceYieldTracker

        tracker = SourceYieldTracker()
        for _ in range(5):
            rec = tracker.record_scrape("s", scraped=400, alive=100)
        self.assertAlmostEqual(rec["live_rate_ewma"], 0.25, places=4)
        prev = 0.25
        for k in range(1, 8):
            rec = tracker.record_scrape("s", scraped=400, alive=300)
            expect = 0.75 + (prev - 0.75) * (0.7 ** k)
            self.assertAlmostEqual(rec["live_rate_ewma"], expect,
                                   delta=1e-3)

    def test_pool_latency_ewma_matches_closed_form(self):
        pool = ProxyPoolConnector(
            CredentialVault(Database(":memory:"), master_passphrase="test"))
        pool.add_proxy("198.51.100.24", 8080)
        pid = "http://198.51.100.24:8080"
        for _ in range(3):
            view = pool.record_success(pid, latency_ms=200.0)
        self.assertAlmostEqual(view["health"]["latency_ewma_ms"], 200.0)
        view = pool.record_success(pid, latency_ms=500.0)
        # alpha=0.3: 0.3*500 + 0.7*200 = 290
        self.assertAlmostEqual(view["health"]["latency_ewma_ms"], 290.0)


if __name__ == "__main__":
    unittest.main()
