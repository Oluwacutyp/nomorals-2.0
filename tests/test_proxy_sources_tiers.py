"""PROXY_SOURCES: tier gate, 2026 source catalog, yield discipline.

Offline by design — no real proxies, no network: the Proxifly feed is
exercised through an injected ``post`` callable, the pool through an
in-memory vault, the browser service through injected fake pools.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest

from nomorals.accounts.browser_login import login_with_vault
from nomorals.accounts.vault import Credential, CredentialVault
from nomorals.browser import BrowserService
from nomorals.browser.botwall import ProxyRouter, RUNG_PROFILE_PROXY
from nomorals.connectors.proxypool import ProxyPoolConnector, ProxyPoolError
from nomorals.core.proxy_tiers import (
    TierDecision,
    TierGate,
    TierGateRefusal,
    normalize_tier,
    tier_satisfies,
)
from nomorals.storage.db import Database
from nomorals.tools.proxy_sources import (
    SOURCE_BY_NAME,
    SOURCES_2026,
    SourceYieldTracker,
    check_staleness,
    dedupe_proxies,
    fetch_proxifly,
    refresh_sources,
    verify_content_integrity,
)
from nomorals.tools.proxylab import Proxy
from nomorals.tools.proxysources import SourceRegistry


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test")


def _healthy_pool() -> ProxyPoolConnector:
    """Pool with one free + one paid proxy, both marked healthy (no network)."""
    pool = ProxyPoolConnector(_vault())
    pool.add_proxy("127.0.0.1", 3111, tier="free")
    pool.add_proxy("127.0.0.1", 3222, tier="paid")
    pool.record_success("http://127.0.0.1:3111")
    pool.record_success("http://127.0.0.1:3222")
    return pool


class TestTierGate(unittest.TestCase):
    def test_free_plus_credentials_refused_typed(self):
        decision = TierGate.check("free", carries_credentials=True)
        self.assertIsInstance(decision, TierDecision)
        self.assertFalse(decision.allowed)
        self.assertIn("free proxy", decision.reason)
        with self.assertRaises(TierGateRefusal) as ctx:
            TierGate.require("free", carries_credentials=True)
        self.assertEqual(ctx.exception.tier, "free")
        self.assertTrue(ctx.exception.carries_credentials)

    def test_free_plus_anonymous_allowed(self):
        self.assertTrue(TierGate.check("free").allowed)
        self.assertTrue(TierGate.check("free", carries_credentials=False).allowed)

    def test_paid_plus_credentials_allowed(self):
        self.assertTrue(
            TierGate.check("paid", carries_credentials=True).allowed)

    def test_direct_plus_credentials_allowed(self):
        self.assertTrue(
            TierGate.check("direct", carries_credentials=True).allowed)

    def test_require_tier(self):
        self.assertFalse(TierGate.check("free", require_tier="paid").allowed)
        self.assertTrue(TierGate.check("paid", require_tier="paid").allowed)
        # direct satisfies any tier demand — no intermediary at all
        self.assertTrue(TierGate.check("direct", require_tier="paid").allowed)
        self.assertTrue(TierGate.check("direct", require_tier="direct").allowed)
        self.assertFalse(TierGate.check("paid", require_tier="direct").allowed)

    def test_normalize(self):
        self.assertEqual(normalize_tier("  PAID "), "paid")
        self.assertEqual(normalize_tier(""), "free")  # unlabelled = untrusted
        self.assertEqual(normalize_tier(None), "free")
        with self.assertRaises(ValueError):
            normalize_tier("platinum")

    def test_tier_satisfies(self):
        self.assertTrue(tier_satisfies("paid", None))
        self.assertTrue(tier_satisfies("direct", "paid"))
        self.assertFalse(tier_satisfies("free", "paid"))


class TestCredentialTier(unittest.TestCase):
    def test_proxy_credential_defaults_free(self):
        cred = Credential(id=1, service="connector:proxypool",
                          username="http://127.0.0.1:3111",
                          password="{}", credential_type="proxy")
        self.assertEqual(cred.tier, "free")

    def test_proxy_credential_metadata_tier(self):
        cred = Credential(id=1, service="connector:proxypool",
                          username="http://127.0.0.1:3222",
                          password="{}", credential_type="proxy",
                          metadata={"tier": "paid"})
        self.assertEqual(cred.tier, "paid")

    def test_generic_credential_defaults_direct(self):
        cred = Credential(id=1, service="gmail", username="a@b.c",
                          password="x", credential_type="password")
        self.assertEqual(cred.tier, "direct")
        self.assertEqual(cred.to_dict()["tier"], "direct")

    def test_invalid_tier_fails_closed_to_free(self):
        cred = Credential(id=1, service="connector:proxypool",
                          username="http://127.0.0.1:3111",
                          password="{}", credential_type="proxy",
                          metadata={"tier": "platinum"})
        self.assertEqual(cred.tier, "free")


class TestPoolTierGate(unittest.TestCase):
    def test_add_proxy_stores_tier(self):
        pool = ProxyPoolConnector(_vault())
        view = pool.add_proxy("127.0.0.1", 3111, tier="paid")
        self.assertEqual(view["tier"], "paid")
        self.assertEqual(pool.list_proxies()[0]["tier"], "paid")

    def test_add_proxy_rejects_bad_tier(self):
        pool = ProxyPoolConnector(_vault())
        with self.assertRaises(ProxyPoolError):
            pool.add_proxy("127.0.0.1", 3111, tier="platinum")

    def test_select_credentials_picks_paid_only(self):
        pool = _healthy_pool()
        picked = pool.select(carries_credentials=True)
        self.assertEqual(picked["tier"], "paid")

    def test_select_anonymous_may_pick_free(self):
        pool = _healthy_pool()
        picked = pool.select(carries_credentials=False)
        self.assertIn(picked["tier"], ("free", "paid"))

    def test_select_credentials_free_only_refuses_typed(self):
        pool = ProxyPoolConnector(_vault())
        pool.add_proxy("127.0.0.1", 3111, tier="free")
        pool.record_success("http://127.0.0.1:3111")
        with self.assertRaises(TierGateRefusal):
            pool.select(carries_credentials=True)

    def test_select_require_tier_direct_refuses_paid_pool(self):
        pool = _healthy_pool()
        with self.assertRaises(TierGateRefusal):
            pool.select(require_tier="direct")

    def test_rotate_credentials_free_only_refuses(self):
        pool = ProxyPoolConnector(_vault())
        pool.add_proxy("127.0.0.1", 3111, tier="free")
        pool.record_success("http://127.0.0.1:3111")
        with self.assertRaises(TierGateRefusal):
            pool.rotate(carries_credentials=True)

    def test_sticky_pin_is_not_a_gate_bypass(self):
        pool = _healthy_pool()
        pool.sticky_acquire("sess-1")  # pins free or paid, anonymous
        # A later credential-carrying request on a free pin must NOT ride
        # the pin: it re-pins to a paid proxy (or refuses when none exists).
        pool._sticky = {"sess-1": ("http://127.0.0.1:3111", 9999999999.0)}
        details = pool.sticky_acquire("sess-1", carries_credentials=True)
        self.assertEqual(details["tier"], "paid")
        # With no paid proxy at all, the same call refuses.
        paid_only_free = ProxyPoolConnector(_vault())
        paid_only_free.add_proxy("127.0.0.1", 3111, tier="free")
        paid_only_free.record_success("http://127.0.0.1:3111")
        paid_only_free._sticky = {"sess-1": ("http://127.0.0.1:3111",
                                             9999999999.0)}
        with self.assertRaises(TierGateRefusal):
            paid_only_free.sticky_acquire("sess-1", carries_credentials=True)


class TestSourceCatalog(unittest.TestCase):
    def test_no_duplicate_names(self):
        names = [entry["name"] for entry in SOURCES_2026]
        self.assertEqual(len(names), len(set(names)))

    def test_proxifly_post_feed(self):
        entry = SOURCE_BY_NAME["proxifly-post"]
        self.assertEqual(entry["method"], "POST")
        self.assertEqual(entry["measured_live_rate"], 0.0)
        self.assertEqual(entry["probe_budget"], 2)  # near-zero probes
        self.assertEqual(entry["url"], "https://api.proxifly.dev/get-proxy")

    def test_proxmint_entries(self):
        api = SOURCE_BY_NAME["proxmint-api"]
        self.assertTrue(api["url"].startswith("https://proxmint.com/api/"))
        self.assertIn("30 min", api["notes"])

    def test_vakhov_deprioritised(self):
        entry = SOURCE_BY_NAME["vakhov-socks5"]
        self.assertEqual(entry["measured_live_rate"], 0.0)
        self.assertEqual(entry["probe_budget"], 2)

    def test_measured_rates_recorded(self):
        self.assertEqual(SOURCE_BY_NAME["iplocate-all"]["measured_live_rate"], 0.117)
        self.assertEqual(SOURCE_BY_NAME["thespeedx"]["measured_live_rate"], 0.083)
        self.assertEqual(SOURCE_BY_NAME["monosans"]["measured_live_rate"], 0.008)

    def test_unverified_never_fetched(self):
        for name in ("hidemy-name", "scrappey"):
            entry = SOURCE_BY_NAME[name]
            self.assertFalse(entry["verified"])
            self.assertEqual(entry["url"], "")
            self.assertEqual(entry["probe_budget"], 0)

    def test_html_sites_present(self):
        for name in ("proxynova", "proxy-list.download", "openproxy.space",
                     "geonode-site", "proxyscrape-site"):
            self.assertIn(name, SOURCE_BY_NAME)
        self.assertEqual(
            SOURCE_BY_NAME["proxynova"]["url"],
            "https://www.proxynova.com/proxy-server-list/")
        self.assertEqual(
            SOURCE_BY_NAME["openproxy.space"]["url"],
            "https://openproxy.space/list")

    def test_fetch_proxifly_post_injected(self):
        seen: dict[str, object] = {}

        def fake_post(url, payload, headers, timeout):
            seen["url"] = url
            seen["payload"] = dict(payload)
            return b"203.0.113.1:8080\n198.51.100.2:3128\n"

        proxies = fetch_proxifly(protocol="http", quantity=99, post=fake_post)
        self.assertEqual(seen["url"], "https://api.proxifly.dev/get-proxy")
        # quantity clamped to Proxifly's 1-20 range
        self.assertEqual(seen["payload"]["quantity"], 20)
        self.assertEqual(len(proxies), 2)
        self.assertTrue(all(p.source == "proxifly-post" for p in proxies))
        self.assertEqual(proxies[0].url, "http://203.0.113.1:8080")


class TestYieldTracker(unittest.TestCase):
    def test_ewma_and_live_rate(self):
        tracker = SourceYieldTracker()
        rec = tracker.record_scrape("s1", scraped=10, alive=5)
        self.assertEqual(rec["live_rate"], 0.5)
        self.assertEqual(rec["live_rate_ewma"], 0.5)
        rec = tracker.record_scrape("s1", scraped=10, alive=0)
        self.assertAlmostEqual(rec["live_rate_ewma"], 0.35)  # 0.3*0 + 0.7*0.5

    def test_zero_yield_quarantine(self):
        tracker = SourceYieldTracker()
        for _ in range(2):
            tracker.record_scrape("dead", scraped=10, alive=0)
            self.assertFalse(tracker.quarantined("dead"))
        tracker.record_scrape("dead", scraped=10, alive=0)
        self.assertTrue(tracker.quarantined("dead"))
        self.assertFalse(tracker.should_probe("dead"))

    def test_probe_budget_cap_zero_yield_source(self):
        tracker = SourceYieldTracker()
        self.assertEqual(tracker.probe_budget_for("proxifly-post"), 2)
        self.assertTrue(tracker.should_probe("proxifly-post"))
        tracker.record_probes("proxifly-post", 2)
        self.assertFalse(tracker.should_probe("proxifly-post"))
        tracker.reset_budget("proxifly-post")
        self.assertTrue(tracker.should_probe("proxifly-post"))

    def test_persistence(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        path = os.path.join(tmp, "yield.json")
        SourceYieldTracker(path=path).record_scrape("s1", scraped=4, alive=1)
        reloaded = SourceYieldTracker(path=path)
        self.assertEqual(reloaded.snapshot()["s1"]["scraped"], 4)


class TestDedupeAndIntegrity(unittest.TestCase):
    def test_dedupe_by_key_keeps_provenance(self):
        proxies = [
            Proxy(host="203.0.113.1", port=8080, scheme="http", source="a"),
            Proxy(host="203.0.113.1", port=8080, scheme="http", source="b"),
            Proxy(host="203.0.113.1", port=8080, scheme="socks5", source="b"),
            Proxy(host="198.51.100.2", port=3128, scheme="http", source="a"),
        ]
        unique, dups = dedupe_proxies(proxies)
        self.assertEqual(len(unique), 3)  # scheme differs → distinct key
        self.assertEqual(len(dups), 1)
        winner = [p for p in unique if p.scheme == "http"][0]
        self.assertEqual(winner.source, "a")
        self.assertEqual(winner.also_from, ["b"])

    def test_dedupe_dicts(self):
        unique, dups = dedupe_proxies([
            {"host": "203.0.113.1", "port": 8080, "scheme": "http", "source": "a"},
            {"host": "203.0.113.1", "port": 8080, "scheme": "http", "source": "b"},
        ])
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["also_from"], ["b"])
        self.assertEqual(len(dups), 1)

    def test_integrity_match(self):
        body = b"<html>stable</html>"
        expected = hashlib.sha256(body).hexdigest()
        result = verify_content_integrity(lambda: body, expected)
        self.assertTrue(result["ok"])
        self.assertFalse(result["tampered"])

    def test_integrity_rewriter_caught(self):
        body = b"<html>stable</html>"
        expected = hashlib.sha256(body).hexdigest()
        result = verify_content_integrity(
            lambda: b"<html>stable</html><script>evil()</script>", expected)
        self.assertFalse(result["ok"])
        self.assertTrue(result["tampered"])

    def test_integrity_fetch_failure_not_a_pass(self):
        def boom():
            raise ConnectionError("proxy down")

        result = verify_content_integrity(boom, "0" * 64)
        self.assertFalse(result["ok"])
        self.assertIn("proxy down", result["error"])


class TestStalenessAndRefresh(unittest.TestCase):
    def test_check_staleness_split(self):
        now = 1_800_000_000.0
        out = check_staleness([
            {"name": "fresh", "last_ok": now - 100},
            {"name": "stale", "last_ok": now - 40 * 86400},
            {"name": "never", "last_ok": 0},
        ], now=now)
        self.assertEqual(out["fresh"], ["fresh"])
        self.assertEqual(out["stale"], ["stale"])
        self.assertEqual(out["never_ok"], ["never"])
        self.assertEqual(out["stale_count"], 2)

    def test_refresh_adds_verified_skips_unverified(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        registry = SourceRegistry(os.path.join(tmp, "sources.json"))
        out = refresh_sources(registry)
        self.assertIn("proxifly-github-all", out["added"])
        self.assertIn("proxmint-api", out["added"])
        self.assertIn("proxynova", out["added"])
        self.assertIn("hidemy-name", out["skipped_unverified"])
        self.assertIn("scrappey", out["skipped_unverified"])
        # built-ins / references are already present, never re-added
        self.assertIn("thespeedx", out["already_present"])
        self.assertIn("monosans", out["already_present"])
        second = refresh_sources(registry)
        self.assertIn("proxifly-github-all", second["already_present"])
        self.assertEqual(second["added"], [])


class _TieredFakePool:
    """Duck-typed pool: select() does NOT enforce the gate (service must)."""

    def __init__(self, tier: str):
        self.tier = tier

    def rotate(self):
        return {"url_with_auth": "http://127.0.0.1:9999",
                "id": "fake-1", "tier": self.tier}

    def select(self, **kwargs):
        return self.rotate()


class TestBrowserWiring(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.svc = BrowserService(data_dir=os.path.join(self.tmp, "data"))

    def test_botwall_router_refuses_free_for_credentials(self):
        router = ProxyRouter(
            proxy_for=lambda site, locale: "http://127.0.0.1:3111",
            tier_for=lambda url: "free")
        with self.assertRaises(TierGateRefusal):
            router.route("example.com", rung=RUNG_PROFILE_PROXY,
                         carries_credentials=True)
        # anonymous route through the same free proxy is fine
        route = router.route("example.com", rung=RUNG_PROFILE_PROXY)
        self.assertEqual(route.proxy_url, "http://127.0.0.1:3111")

    def test_botwall_router_allows_paid_for_credentials(self):
        router = ProxyRouter(
            proxy_for=lambda site, locale: "http://127.0.0.1:3222",
            tier_for=lambda url: "paid")
        route = router.route("example.com", rung=RUNG_PROFILE_PROXY,
                             carries_credentials=True)
        self.assertEqual(route.tier, "paid")

    def test_botwall_router_fails_closed_without_tier_info(self):
        router = ProxyRouter(
            proxy_for=lambda site, locale: "http://127.0.0.1:3111")
        with self.assertRaises(TierGateRefusal):
            router.route("example.com", rung=RUNG_PROFILE_PROXY,
                         carries_credentials=True)

    def test_service_rotate_refuses_free_for_credentials(self):
        self.svc.attach_proxy_pool(_TieredFakePool("free"))
        self.svc.open_session("s")
        with self.assertRaises(TierGateRefusal):
            self.svc.rotate_proxy("s", carries_credentials=True)

    def test_service_rotate_paid_for_credentials(self):
        self.svc.attach_proxy_pool(_TieredFakePool("paid"))
        self.svc.open_session("s")
        out = self.svc.rotate_proxy("s", carries_credentials=True)
        self.assertEqual(out["tier"], "paid")
        self.assertEqual(self.svc.session_proxy_tier("s"), "paid")

    def test_service_set_session_proxy_gate(self):
        self.svc.open_session("s")
        with self.assertRaises(TierGateRefusal):
            self.svc.set_session_proxy("s", "http://127.0.0.1:3111",
                                       tier="free", carries_credentials=True)
        out = self.svc.set_session_proxy("s", "http://127.0.0.1:3111",
                                         tier="free")
        self.assertEqual(out["tier"], "free")
        with self.assertRaises(TierGateRefusal):
            self.svc.require_session_tier("s", carries_credentials=True)

    def test_login_with_vault_refuses_free_proxy(self):
        class _Mgr:
            def resolve_account(self, service, username):
                raise AssertionError("credential must not resolve on refusal")

        with self.assertRaises(TierGateRefusal):
            login_with_vault(_Mgr(), object(), lambda: None,
                             service="example", proxy_tier="free")


if __name__ == "__main__":
    unittest.main()
