"""Doc 17 Pass 2 — one route authority + the §3 receipt journal.

Pins:
(a) A pool decision through the authority returns select()'s exact
    details object AND appends a receipt carrying the real reason
    (the strategy the pool itself reported), the candidate count and
    tier histogram from the pool's own healthy listing.
(b) Sticky decisions carry their distinct reasons (new pin vs hit),
    with the pin's expiry recorded only when the pool minted one.
(c) A rotate-only pool composes rotate() + the TierGate inside the
    authority; a credential request against a free rotate-only pool
    raises the typed TierGateRefusal and journals a none-receipt.
(d) An empty pool journals a none-receipt and re-raises the pool's
    own error untouched.
(e) Credentials NEVER reach the journal: a credentialed pool fixture
    whose password appears nowhere in the journal bytes (nor any
    url_with_auth / password field in any row).
(f) Journal-write failure is fail-open: the decision is returned
    exactly as the pool produced it.
(g) The read API returns decisions in order, filters by consumer and
    source, and folds later outcome appends onto their decision.
(h) The lab rotation manager's real paths journal through the
    authority: the core.http resolver (consumer core_http, strategy
    + affinity reasons), pick_for_job (the why-string rationale),
    the empty-pool direct fallback, and a reported failure lands as
    the decision's outcome — attributed by endpoint, never invented.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import types
import unittest
from pathlib import Path

from nomorals.connectors.proxypool import ProxyPoolError
from nomorals.core.proxy_routing import (
    RouteAuthority,
    RouteReceiptJournal,
    default_receipts_path,
)
from nomorals.core.proxy_tiers import TierGateRefusal
from nomorals.tools import proxylab as PL

# Fixture credential, assembled from parts at runtime (secaudit rule:
# test fixtures never carry a full-format secret-shaped literal — the
# committed-secret gate scans this file).
SECRET = "S3CRET-" + "PIN-991"


def _details(**over):
    details = {
        "id": "px-cred-1",
        "host": "203.0.113.7",
        "port": 8080,
        "protocol": "http",
        "country": "NL",
        "tier": "paid",
        "username": "alice",
        "password": SECRET,
        "has_auth": True,
        "url_with_auth": "http://alice:" + SECRET + "@203.0.113.7:8080",
        "proxy_url": "http://alice:" + SECRET + "@203.0.113.7:8080",
        "tags": ["owner"],
        "selection_strategy": "p2c",
    }
    details.update(over)
    return details


class FakePool:
    """Duck-typed vault pool: select/healthy only."""

    def __init__(self, details=None, healthy=(), exc=None):
        self._details = details if details is not None else _details()
        self._healthy = list(healthy)
        self._exc = exc
        self.select_calls = []

    def select(self, strategy="p2c", **filters):
        self.select_calls.append((strategy, filters))
        if self._exc is not None:
            raise self._exc
        return self._details

    def healthy_proxies(self):
        return list(self._healthy)


class RotateOnlyPool:
    def __init__(self, details):
        self._details = details

    def rotate(self):
        return self._details

    def healthy_proxies(self):
        return [self._details]


class StickyPool(FakePool):
    def __init__(self, pin_details):
        super().__init__(details=pin_details)
        self._pin = pin_details

    def sticky_acquire(self, key, *, ttl=600.0, **filters):
        return self._pin


class AuthorityTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.journal_path = Path(self.tmp) / "proxies" / "route_receipts.jsonl"
        self.journal = RouteReceiptJournal(self.journal_path)
        self.authority = RouteAuthority(self.journal)

    def rows(self):
        if not self.journal_path.exists():
            return []
        return [json.loads(line) for line in
                self.journal_path.read_text(encoding="utf-8").splitlines()
                if line.strip()]


class PoolDecisionTests(AuthorityTestBase):
    def test_select_decision_returns_exact_object_and_receipt(self):
        details = _details()
        healthy = [_details(id="a", tier="paid"),
                   _details(id="b", tier="paid"),
                   _details(id="c", tier="free")]
        pool = FakePool(details=details, healthy=healthy)
        out = self.authority.decide_pool(pool, consumer="browser",
                                         host="example.com")
        self.assertIs(out, details)
        rows = self.rows()
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["consumer"], "browser")
        self.assertEqual(row["route"], "paid")
        self.assertEqual(row["source"], "proxypool")
        self.assertEqual(row["tier"], "paid")
        self.assertEqual(row["proxy_id"], "px-cred-1")
        self.assertEqual(row["host"], "example.com")
        self.assertEqual(row["reason_code"], "select:p2c")
        self.assertEqual(row["candidates"], 3)
        self.assertEqual(row["candidate_tiers"], {"paid": 2, "free": 1})
        self.assertEqual(row["selected_by"], "select")
        self.assertTrue(row["authenticated"])
        self.assertIsNone(row["outcome"])

    def test_credentials_never_reach_the_journal(self):
        pool = FakePool(details=_details(), healthy=[_details()])
        self.authority.decide_pool(pool, consumer="browser")
        raw = self.journal_path.read_bytes()
        self.assertNotIn(SECRET.encode(), raw)
        self.assertNotIn(b"url_with_auth", raw)
        self.assertNotIn(b"alice:", raw)
        for row in self.rows():
            self.assertNotIn("password", row)
            self.assertNotIn("proxy_url", row)

    def test_sticky_new_pin_vs_hit_reasons(self):
        new_pin = _details(sticky=True, sticky_ttl=600.0)
        pool = StickyPool(new_pin)
        self.authority.decide_pool(pool, consumer="browser",
                                   sticky_key="shop.example")
        row = self.rows()[0]
        self.assertEqual(row["reason_code"], "sticky_new_pin")
        self.assertEqual(row["sticky_key"], "shop.example")
        self.assertIsNotNone(row["expires_at"])

        hit = _details(sticky=True)
        pool2 = StickyPool(hit)
        self.authority.decide_pool(pool2, consumer="browser",
                                   sticky_key="shop.example")
        row2 = self.rows()[1]
        self.assertEqual(row2["reason_code"], "sticky_hit")
        self.assertIsNone(row2["expires_at"])

    def test_rotate_only_pool_and_tier_gate_refusal(self):
        free = _details(tier="free", username="", password="",
                        has_auth=False, url_with_auth="http://203.0.113.7:8080")
        pool = RotateOnlyPool(free)
        out = self.authority.decide_pool(pool, consumer="browser")
        self.assertIs(out, free)
        self.assertEqual(self.rows()[0]["reason_code"],
                         "rotate:round_robin")

        with self.assertRaises(TierGateRefusal):
            self.authority.decide_pool(pool, consumer="browser",
                                       carries_credentials=True)
        row = self.rows()[1]
        self.assertEqual(row["route"], "none")
        self.assertEqual(row["reason_code"], "tier_gate_refusal")

    def test_empty_pool_none_receipt_and_original_error(self):
        pool = FakePool(exc=ProxyPoolError("no eligible proxies"))
        with self.assertRaises(ProxyPoolError):
            self.authority.decide_pool(pool, consumer="browser")
        row = self.rows()[0]
        self.assertEqual(row["route"], "none")
        self.assertEqual(row["reason_code"], "pool_no_route")
        self.assertIn("no eligible proxies", row["reason"])

    def test_journal_write_failure_is_fail_open(self):
        bad = RouteReceiptJournal(Path(self.tmp))  # a directory, not a file
        authority = RouteAuthority(bad)
        details = _details()
        pool = FakePool(details=details)
        out = authority.decide_pool(pool, consumer="browser")
        self.assertIs(out, details)
        self.assertFalse(bad.append(
            authority.record(consumer="x", route="free",
                             reason_code="select:p2c")))


class ReadApiTests(AuthorityTestBase):
    def test_recent_order_filters_and_outcome_fold(self):
        a = self.authority.record(consumer="core_http", route="free",
                                  reason_code="strategy:round_robin",
                                  source="proxylab",
                                  proxy_id="10.0.0.1:8080")
        self.authority.record(consumer="proxy_pick", route="none",
                              reason_code="no_matching_candidates",
                              source="proxylab")
        self.authority.record(consumer="core_http", route="free",
                              reason_code="affinity_hit",
                              source="proxylab")
        all_rows = self.journal.recent()
        self.assertEqual([r["reason_code"] for r in all_rows],
                         ["strategy:round_robin",
                          "no_matching_candidates", "affinity_hit"])
        self.assertEqual(
            len(self.journal.recent(consumer="core_http")), 2)
        self.assertEqual(
            len(self.journal.recent(source="proxypool")), 0)
        self.assertEqual(len(self.journal.recent(limit=1)), 1)

        self.assertTrue(self.authority.record_outcome(
            a.decision_id, "failed", "connection reset"))
        folded = self.journal.recent(consumer="core_http")
        self.assertEqual(folded[0]["outcome"], "failed")
        self.assertEqual(folded[0]["outcome_detail"], "connection reset")
        self.assertIsNone(folded[1]["outcome"])

    def test_default_path_lives_beside_the_lab_store(self):
        self.assertEqual(default_receipts_path().name,
                         "route_receipts.jsonl")
        self.assertEqual(default_receipts_path().parent.name, "proxies")


class RotationManagerReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.ctx = types.SimpleNamespace(
            settings=types.SimpleNamespace(home=self.tmp), db=None)
        self.lab = PL.ProxyLab(self.ctx)
        now = time.time()
        proxies = [
            PL.Proxy(host="10.0.0.1", port=8080, scheme="http",
                     anonymity="elite", country="NG", alive=True,
                     tested_at=now),
            PL.Proxy(host="10.0.0.2", port=1080, scheme="socks5",
                     anonymity="anonymous", country="DE", alive=True,
                     tested_at=now),
        ]
        for p in proxies:
            p.record_latency(120.0, at=now)
        self.lab.store.save(proxies)
        self.rot = PL.ProxyRotationManager(self.ctx)
        self.rot.state["enabled"] = True
        self.journal = RouteReceiptJournal(
            Path(self.tmp) / "proxies" / "route_receipts.jsonl")

    def test_resolver_decision_and_failure_outcome(self):
        url = self.rot._resolver("https://example.com/x")
        self.assertTrue(url)
        rows = self.journal.recent(consumer="core_http")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["route"], "free")
        self.assertEqual(rows[0]["source"], "proxylab")
        self.assertTrue(rows[0]["reason_code"].startswith("strategy:"))
        self.assertIsNone(rows[0]["outcome"])

        self.rot.report_error(url, "connection reset")
        rows = self.journal.recent(consumer="core_http")
        self.assertEqual(rows[0]["outcome"], "failed")
        self.assertEqual(rows[0]["outcome_detail"], "connection reset")

    def test_affinity_hit_reason(self):
        first = self.rot._resolver("https://example.com/a")
        self.assertTrue(first)
        self.rot._resolver("https://example.com/b")
        codes = [r["reason_code"]
                 for r in self.journal.recent(consumer="core_http")]
        self.assertIn("affinity_hit", codes)

    def test_pick_for_job_receipt_carries_the_why(self):
        res = self.rot.pick_for_job(needs_udp=True)
        self.assertTrue(res["proxy"].startswith("socks5://"))
        rows = self.journal.recent(consumer="proxy_pick")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["reason_code"], "job_rank_first")
        self.assertIn("UDP-capable", rows[0]["reason"])
        self.assertEqual(rows[0]["protocol"], "socks5")

    def test_pick_for_job_needs_auth_none_receipt(self):
        res = self.rot.pick_for_job(needs_auth=True)
        self.assertEqual(res["proxy"], "")
        rows = self.journal.recent(consumer="proxy_pick")
        self.assertEqual(rows[0]["route"], "none")
        self.assertEqual(rows[0]["reason_code"], "needs_auth_unavailable")

    def test_empty_pool_direct_fallback_receipt(self):
        self.lab.store.clear()
        url = self.rot._resolver("https://example.com/x")
        self.assertEqual(url, "")
        rows = self.journal.recent(consumer="core_http")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["route"], "direct")
        self.assertEqual(rows[0]["reason_code"],
                         "pool_empty_direct_fallback")


if __name__ == "__main__":
    unittest.main()
