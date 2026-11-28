"""gapfix160: pumpdesk sources verified live, then wired in.

Probes (plain clients, this box, 2026-11-25 — full tables in
~/workspace/pumpdesksources-verify/SURVEY.md):
* DexScreener profiles + token snapshot, RugCheck report,
  Solana public RPC (getSlot/getHealth/getTokenSupply),
  PumpPortal WS (real create events), pump.fun frontend API
  (200, 3/3 — the 2026-11-14 Cloudflare block no longer holds):
  all WORKING.
* PublicNode free tier: basic reads answer (getAccountInfo /
  getBalance / getHealth / getSignaturesForAddress /
  getTransaction 200) but indexed methods are refused by policy
  (HTTP 403 "Indexed requests require a personal token" for
  getTokenLargestAccounts / getTokenSupply; empty 403 for
  getSlot).  RoutedRPC must never spend an indexed call on it.
* dRPC: still dead (HTTP 400 code 35, re-confirmed).  Ankr +
  solana.public-rpc.com: unreachable from this box (timeouts) —
  NOT wired.  Helius / Birdeye: live, key-required (401).

Tests use fakes for transport; the LIVE probe results live in
SURVEY.md, not in these tests.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.agents.pumpdesk import dossier as dossier_mod
from nomorals.agents.pumpdesk import firehose as firehose_mod
from nomorals.agents.pumpdesk import outcomes as outcomes_mod
from nomorals.agents.pumpdesk import sources as sources_mod
from nomorals.agents.pumpdesk.chain import RoutedRPC, RPCError, SolanaRPC


class _FakeClient:
    """A RoutedRPC route double: records calls, answers per script."""

    def __init__(self, endpoint, indexed=True, script=None):
        self.endpoint = endpoint
        self.indexed = indexed
        self.calls = []
        self.script = script or {}

    def _run(self, method, *args):
        self.calls.append((method, args))
        if method in self.script:
            out = self.script[method]
            if isinstance(out, Exception):
                raise out
            return out
        return f"{method}-ok"

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: self._run(name, *a)


class TestCanonicalEndpoints(unittest.TestCase):
    """One source of truth: every desk module reads its URLs from
    sources.py — the verified values, never a second literal."""

    def test_constants_are_the_verified_urls(self):
        self.assertEqual(sources_mod.FRONTEND_API,
                         "https://frontend-api-v3.pump.fun")
        self.assertEqual(sources_mod.DEX_PROFILES,
                         "https://api.dexscreener.com/token-profiles/latest/v1")
        self.assertEqual(sources_mod.DEX_API,
                         "https://api.dexscreener.com")
        self.assertEqual(sources_mod.RUGCHECK_API,
                         "https://api.rugcheck.xyz")
        self.assertEqual(sources_mod.PUMPPORTAL_WS,
                         "wss://pumpportal.fun/api/data")
        self.assertEqual(sources_mod.SOLANA_PUBLIC_RPC,
                         "https://api.mainnet-beta.solana.com")
        self.assertEqual(sources_mod.PUBLICNODE_RPC,
                         "https://solana-rpc.publicnode.com")

    def test_modules_import_the_canonical_endpoints(self):
        self.assertIs(firehose_mod.FRONTEND_API, sources_mod.FRONTEND_API)
        self.assertIs(firehose_mod.DEX_PROFILES, sources_mod.DEX_PROFILES)
        self.assertIs(firehose_mod.PUMPPORTAL_WS, sources_mod.PUMPPORTAL_WS)
        self.assertIs(dossier_mod.RUGCHECK_API, sources_mod.RUGCHECK_API)
        self.assertIs(outcomes_mod.DEX_API, sources_mod.DEX_API)

    def test_no_second_literal_of_any_endpoint(self):
        import pathlib
        desk = pathlib.Path(__file__).resolve().parent.parent \
            / "nomorals" / "agents" / "pumpdesk"
        literals = {
            "https://frontend-api-v3.pump.fun",
            "https://api.dexscreener.com/token-profiles/latest/v1",
            "https://api.dexscreener.com",
            "https://api.rugcheck.xyz",
            "wss://pumpportal.fun/api/data",
            "https://api.mainnet-beta.solana.com",
            "https://solana-rpc.publicnode.com",
        }
        for path in sorted(desk.glob("*.py")):
            if path.name == "sources.py":
                continue
            text = path.read_text()
            for lit in literals:
                self.assertNotIn(lit, text,
                                 f"{path.name} carries its own "
                                 f"literal of {lit}")

    def test_default_rpc_endpoint_is_the_verified_floor(self):
        self.assertEqual(SolanaRPC().endpoint, sources_mod.SOLANA_PUBLIC_RPC)


class TestIndexedPartition(unittest.TestCase):
    """PublicNode's free tier refuses indexed reads by policy, so
    the desk never spends one on it — but still uses it for the
    basic reads it does serve."""

    def test_routes_mark_publicnode_not_indexed(self):
        routes = sources_mod.rpc_routes(None)
        by_domain = {r["domain"]: r for r in routes}
        self.assertTrue(by_domain["api.mainnet-beta.solana.com"]["indexed"])
        self.assertFalse(by_domain["solana-rpc.publicnode.com"]["indexed"])
        self.assertEqual(
            by_domain["solana-rpc.publicnode.com"]["endpoint"],
            sources_mod.PUBLICNODE_RPC)

    def test_indexed_method_set_is_the_verified_refused_set(self):
        self.assertEqual(
            sources_mod.INDEXED_RPC_METHODS,
            {"get_token_largest", "get_token_supply", "get_slot"})

    def test_routed_rpc_skips_publicnode_for_indexed_methods(self):
        public = _FakeClient("https://api.mainnet-beta.solana.com",
                             indexed=True,
                             script={"get_token_largest": [{"amount": "5"}]})
        node = _FakeClient("https://solana-rpc.publicnode.com",
                           indexed=False)
        routed = RoutedRPC([public, node])
        self.assertEqual(routed.get_token_largest("MINT"),
                         [{"amount": "5"}])
        self.assertEqual(routed.get_token_supply("MINT"), "get_token_supply-ok")
        self.assertEqual(routed.get_slot(), "get_slot-ok")
        self.assertEqual([c for c, _ in node.calls], [])

    def test_routed_rpc_still_uses_publicnode_for_basic_reads(self):
        public = _FakeClient("https://api.mainnet-beta.solana.com",
                             indexed=True,
                             script={"get_account": Exception("boom")})
        node = _FakeClient("https://solana-rpc.publicnode.com",
                           indexed=False,
                           script={"get_account": b"data"})
        routed = RoutedRPC([public, node])
        # public fails the account read; failover reaches PublicNode
        self.assertEqual(routed.get_account("MINT"), b"data")
        self.assertEqual([c for c, _ in node.calls], ["get_account"])
        # a basic read served by the first route never touches node
        node2 = _FakeClient("https://solana-rpc.publicnode.com",
                            indexed=False)
        public2 = _FakeClient("https://api.mainnet-beta.solana.com",
                              indexed=True,
                              script={"get_transaction": {"tx": 1}})
        routed2 = RoutedRPC([public2, node2])
        self.assertEqual(routed2.get_transaction("SIG"), {"tx": 1})
        self.assertEqual(node2.calls, [])

    def test_no_indexed_capable_route_fails_honestly(self):
        node = _FakeClient("https://solana-rpc.publicnode.com",
                           indexed=False)
        routed = RoutedRPC([node])
        with self.assertRaises(RPCError) as ctx:
            routed.get_token_largest("MINT")
        self.assertIn("no indexed-capable route", str(ctx.exception))
        self.assertEqual(node.calls, [])

    def test_route_doubles_without_indexed_flag_count_as_indexed(self):
        class _OldStyle:
            endpoint = "https://old.example"
            calls = []

            def get_token_largest(self, mint):
                self.calls.append(("get_token_largest", mint))
                return [{"amount": "1"}]

        routed = RoutedRPC([_OldStyle()])
        self.assertEqual(routed.get_token_largest("MINT"),
                         [{"amount": "1"}])

    def test_all_routes_failing_still_raises_never_silent_zero(self):
        dead = _FakeClient("https://api.mainnet-beta.solana.com",
                           indexed=True,
                           script={"get_token_largest":
                                   RPCError("getTokenLargestAccounts: HTTP 429")})
        routed = RoutedRPC([dead])
        with self.assertRaises(RPCError):
            routed.get_token_largest("MINT")


class TestNoDeadRoutesWired(unittest.TestCase):
    """dRPC stays retired; unreachable candidates were never added."""

    def test_drpc_not_a_route(self):
        routes = sources_mod.rpc_routes(None)
        domains = [r["domain"] for r in routes]
        self.assertNotIn("solana.drpc.org", domains)
        self.assertNotIn("rpc.ankr.com", domains)
        self.assertNotIn("solana.public-rpc.com", domains)

    def test_public_rpc_endpoints_shape(self):
        for row in sources_mod.PUBLIC_RPC_ENDPOINTS:
            ep, indexed = row[0], row[3]
            self.assertTrue(ep.startswith("https://"))
            self.assertIsInstance(indexed, bool)


if __name__ == "__main__":
    unittest.main()
