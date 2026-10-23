"""Pump desk burn-down — the five P1/P2 weaknesses, fixed and
pinned:

1. holder data off the single 429-bound RPC: routed multi-route
   reads (Helius first when provisioned, registry-health order),
   plus a PARTIAL holder picture derived from the desk's own
   ledger when largest-accounts is unreachable everywhere;
2. attention beyond web search: platform comment velocity,
   DexScreener pair surface (socials/boosts), cross-launch social
   handle recurrence — all as named evidence, fail-closed;
3. FIFO grading on partial exits (>= 50% of basis closed), to
   the lot, fractions carried on the record;
4. cohort detection with typed evidence: windowed co-entry
   promotes only when it recurs across launches;
5. the Teachings -> KG mirror proven against a LIVE kg database
   (real Database + migrations + kg's own query API read-back),
   with the fail-soft contract intact.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from nomorals.agents.pumpdesk import attention, chain, graph, outcomes
from nomorals.agents.pumpdesk.chain import (
    BondingCurveState, MintInfo, RPCError, RoutedRPC, SolanaRPC)
from nomorals.agents.pumpdesk.dossier import build_dossier
from nomorals.agents.pumpdesk.shelf import shelve_dossier, shelve_notable
from nomorals.agents.pumpdesk.sources import rpc_routes
from nomorals.agents.pumpdesk.store import PumpStore

SOL = 1_000_000_000
BASE_TS = 1_700_000_000.0
CREATOR = "Creator66666666666666666666666666666666666"
W1 = "WalletA111111111111111111111111111111111111"
W2 = "WalletB222222222222222222222222222222222222"


def _mint(i: int) -> str:
    return f"Mint{i:039d}"


def _trade(user, mint, is_buy, sol, toks, ts, slot=5000):
    return {"user": user, "mint": mint, "is_buy": is_buy,
            "sol_amount": int(sol * SOL), "token_amount": int(toks),
            "timestamp": ts, "slot": slot}


# ── 1. holder routes ──────────────────────────────────────────────────

class _FakeRegistry:
    def __init__(self, inactive=()):
        self._inactive = set(inactive)

    def for_domain(self, domain):
        return SimpleNamespace(domain=domain)

    def is_active(self, desc):
        return desc.domain not in self._inactive


def test_rpc_routes_helius_first_when_keyed(monkeypatch):
    monkeypatch.setenv("HELIUS_API_KEY", "test-key")
    routes = rpc_routes(_FakeRegistry())
    assert routes[0]["source"] == "helius (provisioned)"
    assert "api-key=test-key" in routes[0]["endpoint"]
    domains = [r["domain"] for r in routes]
    assert domains[1:] == ["api.mainnet-beta.solana.com",
                           "solana-rpc.publicnode.com",
                           "solana.drpc.org"]


def test_rpc_routes_public_floor_and_health_order(monkeypatch):
    monkeypatch.delenv("HELIUS_API_KEY", raising=False)
    routes = rpc_routes(None)
    assert routes[0]["domain"] == "api.mainnet-beta.solana.com"
    assert len(routes) == 3
    # a stood-down route sinks below the still-active ones
    reg = _FakeRegistry(inactive={"solana-rpc.publicnode.com"})
    domains = [r["domain"] for r in rpc_routes(reg)]
    assert domains == ["api.mainnet-beta.solana.com",
                       "solana.drpc.org",
                       "solana-rpc.publicnode.com"]


class _Route:
    """One RoutedRPC leg with a scripted outcome."""

    def __init__(self, endpoint, largest=None, error=None):
        self.endpoint = endpoint
        self._largest = largest
        self._error = error
        self.largest_calls = 0
        self.account_calls = 0

    def get_token_largest(self, mint):
        self.largest_calls += 1
        if self._error is not None:
            raise self._error
        return self._largest

    def get_account(self, pubkey):
        self.account_calls += 1
        return None      # account does not exist — an ANSWER


def test_routed_rpc_fails_over_and_records_provenance():
    dead = _Route("https://dead.example", error=RPCError("HTTP 429"))
    live = _Route("https://live.example",
                  largest=[{"amount": "500"}, {"amount": "300"}])
    routed = RoutedRPC([dead, live])
    rows = routed.get_token_largest(_mint(1))
    assert [r["amount"] for r in rows] == ["500", "300"]
    assert routed.served_by["get_token_largest"] == "https://live.example"
    assert dead.largest_calls == 1 and live.largest_calls == 1


def test_routed_rpc_none_answer_stops_the_walk():
    first = _Route("https://a.example")
    second = _Route("https://b.example")
    routed = RoutedRPC([first, second])
    assert routed.get_account("Anything") is None
    assert first.account_calls == 1 and second.account_calls == 0


def test_routed_rpc_all_routes_dead_raises_last_error():
    routed = RoutedRPC([
        _Route("https://a.example", error=RPCError("boom-a")),
        _Route("https://b.example", error=RPCError("boom-b")),
    ])
    with pytest.raises(RPCError, match="boom-b"):
        routed.get_token_largest(_mint(1))


def test_by_owner_read_normalizes_json_parsed():
    rpc = SolanaRPC("https://rpc.example")

    def fake_call(method, params):
        assert method == "getTokenAccountsByOwner"
        return {"value": [{
            "pubkey": "Acct111",
            "account": {"data": {"parsed": {"info": {
                "owner": W1, "mint": _mint(3),
                "tokenAmount": {"amount": "42000"}}}}}},
            {"pubkey": "Acct222",
             "account": {"data": {"parsed": {"info": {
                 "owner": W1, "mint": _mint(3),
                 "tokenAmount": {"amount": "8000"}}}}}}]}

    rpc.call = fake_call
    rows = rpc.get_token_accounts_by_owner(W1, _mint(3))
    assert [r["amount"] for r in rows] == [42000, 8000]
    assert rows[0]["owner"] == W1 and rows[0]["mint"] == _mint(3)


class _DossierRPC:
    """Chain reads fine; largest-accounts throttled everywhere;
    by-owner balances scripted per wallet."""

    endpoint = "https://routed.example"

    def __init__(self, balances=None, largest=None, largest_error=None):
        self._balances = balances or {}
        self._largest = largest
        self._largest_error = largest_error
        self._curve = BondingCurveState(
            virtual_token_reserves=400_000_000_000_000,
            virtual_sol_reserves=100 * SOL,
            real_token_reserves=393_100_000_000_000,
            real_sol_reserves=20 * SOL,
            token_total_supply=1_000_000, creator=CREATOR)
        self._info = MintInfo(supply=1_000_000, decimals=6,
                              initialized=True)

    def get_curve(self, mint):
        return self._curve

    def get_mint_info(self, mint):
        return self._info

    def get_token_largest(self, mint):
        if self._largest_error is not None:
            raise self._largest_error
        return self._largest

    def get_token_accounts_by_owner(self, owner, mint=""):
        amount = self._balances.get(owner, 0)
        if not amount:
            return []
        return [{"address": f"acct-{owner[:6]}", "owner": owner,
                 "mint": mint, "amount": amount}]

    def get_account(self, pubkey):
        return None


def _holder_store(tmp_path, mint):
    store = PumpStore(str(tmp_path))
    store.record_launch({"mint": mint, "creator": CREATOR, "slot": 5,
                         "name": "H", "symbol": "HLD",
                         "ts": time.time(), "source": "test"})
    store.record_trade(_trade(W1, mint, True, 5, 1000, BASE_TS))
    store.record_trade(_trade(W2, mint, True, 3, 500, BASE_TS + 10))
    return store


def test_holders_full_picture_from_largest_accounts(tmp_path):
    mint = _mint(11)
    store = _holder_store(tmp_path, mint)
    rpc = _DossierRPC(largest=[{"amount": 400_000},
                               {"amount": 100_000},
                               {"amount": 50_000}])
    d = build_dossier(store, rpc, mint)
    h = d["holders"]
    assert h["partial"] is False and h["source"] == "largest-accounts"
    assert h["top1_share"] == pytest.approx(0.4)
    assert h["top10_share"] == pytest.approx(0.55)
    assert not any("holder" in g for g in d["gaps"])


def test_holders_partial_picture_derived_from_ledger(tmp_path):
    mint = _mint(12)
    store = _holder_store(tmp_path, mint)
    rpc = _DossierRPC(
        balances={W1: 300_000, W2: 100_000},
        largest_error=RPCError("getTokenLargestAccounts: HTTP 429"))
    d = build_dossier(store, rpc, mint)
    h = d["holders"]
    assert h is not None and h["partial"] is True
    assert h["source"] == "ledger-derived"
    # shares are against TOTAL supply: lower bounds, labeled so
    assert h["top1_share"] == pytest.approx(0.3)
    assert h["top10_share"] == pytest.approx(0.4)
    assert h["wallets_checked"] == 3          # W1, W2 + the creator
    assert h["wallets_holding"] == 2
    assert h["observed_supply_share"] == pytest.approx(0.4)
    assert not any("holder concentration unavailable" in g
                   for g in d["gaps"])
    from nomorals.agents.pumpdesk.dossier import render_dossier
    assert "partial" in render_dossier(d)


def test_holders_both_routes_dead_names_both(tmp_path):
    mint = _mint(13)
    store = _holder_store(tmp_path, mint)
    rpc = _DossierRPC(balances={},
                      largest_error=RPCError("HTTP 429"))
    d = build_dossier(store, rpc, mint)
    assert d["holders"] is None
    gap = next(g for g in d["gaps"] if "holder" in g)
    assert "largest-accounts unreachable" in gap
    assert "ledger-derived" in gap


# ── 2. attention legs ─────────────────────────────────────────────────

def _att_dossier(**over):
    d = {
        "mint": _mint(42), "symbol": "PHNT", "name": "Phantom",
        "uri": "", "created_at": BASE_TS,
        "trades": {"trades": 40, "unique_buyers": 20},
        "gaps": [],
    }
    d.update(over)
    return d


def _no_web(query, limit=10):
    return []


def test_attention_platform_comments_organic():
    now = BASE_TS + 3600

    def platform(mint):
        return {"reply_count": 30,
                "created_timestamp": int(BASE_TS * 1000)}

    att = attention.verify_attention(
        _att_dossier(), search_fn=_no_web, platform_fn=platform,
        now=now)
    assert att["verdict"] == attention.ORGANIC
    assert att["evidence"]["platform_comments"] == 30
    assert att["evidence"]["platform_comments_per_hour"] == \
        pytest.approx(30.0)
    assert att["evidence"]["organic_basis"].startswith("platform")
    assert any("commenter identities" in g for g in att["gaps"])


def test_attention_comment_flood_is_a_manufactured_point():
    now = BASE_TS + 3600

    def platform(mint):
        return {"reply_count": 400,
                "created_timestamp": int(BASE_TS * 1000)}

    d = _att_dossier()
    d["trades"] = {"trades": 40, "unique_buyers": 2}
    att = attention.verify_attention(
        d, search_fn=_no_web, platform_fn=platform, now=now)
    assert any("comment flood" in p
               for p in att["evidence"]["manufactured_points"])
    assert att["verdict"] == attention.MIXED   # one point, no more


def test_attention_flood_plus_boosts_is_manufactured():
    now = BASE_TS + 3600

    def platform(mint):
        return {"reply_count": 400,
                "created_timestamp": int(BASE_TS * 1000)}

    def dex(mint):
        return {"socials": [], "websites": 0, "boosts_active": 3}

    d = _att_dossier()
    d["trades"] = {"trades": 40, "unique_buyers": 2}
    att = attention.verify_attention(
        d, search_fn=_no_web, platform_fn=platform, dex_fn=dex,
        now=now)
    assert att["verdict"] == attention.MANUFACTURED
    assert any("boost" in p
               for p in att["evidence"]["manufactured_points"])


def test_attention_dex_surface_named_not_zeroed():
    def dex(mint):
        return {"socials": [{"type": "twitter", "url": "https://x.com/a"},
                            {"type": "telegram", "url": "https://t.me/a"}],
                "websites": 1, "boosts_active": 0}

    att = attention.verify_attention(
        _att_dossier(), search_fn=_no_web, dex_fn=dex, now=BASE_TS)
    assert att["evidence"]["dex_socials"] == ["twitter", "telegram"]
    assert att["evidence"]["dex_websites"] == 1
    assert att["evidence"]["dex_boosts_active"] == 0
    assert att["evidence"]["manufactured_points"] == []


def test_attention_dex_absent_fields_are_gaps_not_zeros():
    def dex(mint):
        return {"socials": None, "websites": None,
                "boosts_active": None}

    att = attention.verify_attention(
        _att_dossier(), search_fn=_no_web, dex_fn=dex, now=BASE_TS)
    assert att["evidence"]["dex_socials"] is None
    assert att["evidence"]["dex_boosts_active"] is None
    assert any("no socials block" in g for g in att["gaps"])
    assert att["verdict"] == attention.UNKNOWN


def test_attention_recycled_social_handle_flagged(tmp_path):
    store = PumpStore(str(tmp_path))
    mine, other = _mint(21), _mint(22)
    store.record_launch({"mint": mine, "creator": CREATOR,
                         "twitter": "@coolhandle", "ts": BASE_TS,
                         "source": "test"})
    store.record_launch({"mint": other, "creator": "SomeoneElse",
                         "twitter": "https://x.com/coolhandle",
                         "ts": BASE_TS, "source": "test"})
    rows = [{"title": "one ordinary mention", "snippet": "a look",
             "url": "https://example.com/p1", "source": "web",
             "timestamp": BASE_TS}]
    att = attention.verify_attention(
        _att_dossier(mint=mine),
        search_fn=lambda q, limit=10: list(rows), store=store,
        now=BASE_TS)
    assert att["evidence"]["recycled_handles"]["twitter"] == [other]
    assert att["evidence"]["recycled_socials"] is True
    assert att["verdict"] == attention.MANUFACTURED
    assert any("recycled twitter" in p
               for p in att["evidence"]["manufactured_points"])


def test_attention_all_legs_dead_stays_unknown():
    att = attention.verify_attention(
        _att_dossier(), search_fn=_no_web,
        platform_fn=lambda mint: None, dex_fn=lambda mint: None,
        now=BASE_TS)
    assert att["verdict"] == attention.UNKNOWN
    assert att["quality"] is None


def test_dex_token_attention_parses_pair_surface():
    payload = {"pairs": [{
        "chainId": "solana", "dexId": "pumpswap",
        "pairAddress": "Pair111",
        "liquidity": {"usd": 5000},
        "info": {"socials": [{"type": "twitter", "url": "https://x.com/a"}],
                 "websites": [{"url": "https://a.example"}]},
        "boosts": {"active": 2}}]}

    def http_get(url, headers):
        return 200, json.dumps(payload)

    out = outcomes.token_attention(_mint(1), http_get)
    assert out["socials"] == [{"type": "twitter",
                               "url": "https://x.com/a"}]
    assert out["websites"] == 1
    assert out["boosts_active"] == 2

    bare = {"pairs": [{"chainId": "solana", "dexId": "pumpswap",
                       "pairAddress": "Pair111",
                       "liquidity": {"usd": 1}, "info": None}]}
    out = outcomes.token_attention(
        _mint(1), lambda u, h: (200, json.dumps(bare)))
    assert out["socials"] is None and out["boosts_active"] is None


# ── 3. FIFO partial exits ─────────────────────────────────────────────

def _graph_for(store, wallet):
    g = graph.build_wallet_graph(store, now=BASE_TS + 10_000_000)
    return g["wallets"][wallet]


def test_fifo_partial_exit_math_to_the_lot(tmp_path):
    store = PumpStore(str(tmp_path))
    w, m = W1, _mint(31)
    # two lots: 600 @ 0.01 SOL, 400 @ 0.02 SOL (basis 14 SOL)
    store.record_trade(_trade(w, m, True, 6, 600, BASE_TS))
    store.record_trade(_trade(w, m, True, 8, 400, BASE_TS + 10))
    # sell 600 @ 0.02: FIFO consumes lot 1 whole -> +6.0 realized,
    # 60% of the position's basis closed, position still open
    store.record_trade(_trade(w, m, False, 12, 600, BASE_TS + 20))
    rec = _graph_for(store, w)
    assert rec["realized_sol"] == pytest.approx(6.0)
    assert rec["closed_trades"] == 1
    assert rec["full_exits"] == 0 and rec["partial_exits"] == 1
    pos = rec["positions"][0]
    assert pos["closed_fraction"] == pytest.approx(0.6)
    assert pos["realized_sol"] == pytest.approx(6.0)
    assert pos["full_exit"] is False
    # the open remainder still carries its basis
    assert rec["open_cost_sol"] == pytest.approx(8.0)


def test_fifo_gradeable_fraction_boundary_both_sides(tmp_path):
    store = PumpStore(str(tmp_path))
    m = _mint(32)
    store.record_trade(_trade(W1, m, True, 10, 1000, BASE_TS))
    store.record_trade(_trade(W1, m, False, 10, 500, BASE_TS + 10))
    store.record_trade(_trade(W2, m, True, 10, 1000, BASE_TS))
    store.record_trade(_trade(W2, m, False, 10, 499, BASE_TS + 10))
    g = graph.build_wallet_graph(store, now=BASE_TS + 10_000_000)
    assert g["wallets"][W1]["closed_trades"] == 1     # exactly 50%
    assert g["wallets"][W1]["positions"][0]["closed_fraction"] == \
        pytest.approx(0.5)
    assert g["wallets"][W2]["closed_trades"] == 0     # 49.9%: not yet


def test_fifo_partial_wins_grade_a_wallet(tmp_path):
    store = PumpStore(str(tmp_path))
    w = W1
    for i in range(3):
        m = _mint(40 + i)
        store.record_trade(_trade(w, m, True, 10, 1000,
                                  BASE_TS + i * 3600))
        # close 60% at a profit, ride the rest — gradeable now
        store.record_trade(_trade(w, m, False, 12, 600,
                                  BASE_TS + i * 3600 + 60))
    rec = _graph_for(store, w)
    assert rec["closed_trades"] == 3
    assert rec["closed_wins"] == 3
    assert rec["partial_exits"] == 3 and rec["full_exits"] == 0
    assert rec["verdict"] == "smart"
    assert rec["weight"] > 0.0


def test_fifo_mixed_basis_known_and_unknown_sells(tmp_path):
    store = PumpStore(str(tmp_path))
    w, m = W1, _mint(35)
    store.record_trade(_trade(w, m, True, 5, 500, BASE_TS))
    # sell 800: only 500 attributable; proceeds on the extra 300
    # are NOT booked; matched PnL = 500 * (0.02 - 0.01) = +5.0
    store.record_trade(_trade(w, m, False, 16, 800, BASE_TS + 10))
    rec = _graph_for(store, w)
    assert rec["basis_unknown_sells"] == 1
    assert rec["realized_sol"] == pytest.approx(5.0)
    assert rec["closed_trades"] == 1
    assert rec["positions"][0]["full_exit"] is True
    assert rec["positions"][0]["closed_fraction"] == \
        pytest.approx(1.0)


def test_fifo_episodes_do_not_merge_round_trips(tmp_path):
    store = PumpStore(str(tmp_path))
    w, m = W1, _mint(36)
    store.record_trade(_trade(w, m, True, 10, 1000, BASE_TS))
    store.record_trade(_trade(w, m, False, 20, 1000, BASE_TS + 10))
    store.record_trade(_trade(w, m, True, 10, 1000, BASE_TS + 20))
    store.record_trade(_trade(w, m, False, 12, 600, BASE_TS + 30))
    rec = _graph_for(store, w)
    assert rec["closed_trades"] == 2
    assert rec["full_exits"] == 1 and rec["partial_exits"] == 1
    # schema the other organs consume stays a superset of P2's
    for key in ("realized_sol", "realized_recent_sol", "closed_wins",
                "win_rate", "verdict", "weight", "open_cost_sol",
                "basis_unknown_sells", "buys", "sells"):
        assert key in rec


# ── 4. cohort evidence typing ─────────────────────────────────────────

def _windowed_pair(store, a, b, mint, slot_a, slot_b):
    store.record_trade(_trade(a, mint, True, 3, 300,
                              BASE_TS + slot_a, slot=slot_a))
    store.record_trade(_trade(b, mint, True, 3, 300,
                              BASE_TS + slot_b, slot=slot_b))


def test_cohort_windowed_recurrence_promotes(tmp_path):
    store = PumpStore(str(tmp_path))
    _windowed_pair(store, W1, W2, _mint(51), 1000, 1003)
    _windowed_pair(store, W1, W2, _mint(52), 2000, 2004)
    g = graph.build_wallet_graph(store)
    assert len(g["clusters"]) == 1
    ev = g["clusters"][0]["evidence"]
    assert set(g["clusters"][0]["members"]) == {W1, W2}
    assert ev["shared_launches"] == 2
    assert ev["windowed_launches"] == 2
    assert ev["co_entry_events"] == 0        # no strong events
    assert "windowed" in ev["evidence_kinds"]
    assert "unknown" in ev["funding"]


def test_cohort_windowed_single_launch_promotes_nothing(tmp_path):
    store = PumpStore(str(tmp_path))
    _windowed_pair(store, W1, W2, _mint(53), 1000, 1002)
    g = graph.build_wallet_graph(store)
    assert g["clusters"] == []
    # and confluence sees two independent wallets, not an entity
    conf = graph.graph_confluence(store, _mint(53), graph=g)
    assert all("cohort" not in (e.get("note") or "")
               for e in conf["entries"])


def test_cohort_beyond_window_promotes_nothing(tmp_path):
    store = PumpStore(str(tmp_path))
    _windowed_pair(store, W1, W2, _mint(54), 1000,
                   1000 + graph.COHORT_WINDOW_SLOTS + 1)
    _windowed_pair(store, W1, W2, _mint(55), 2000,
                   2000 + graph.COHORT_WINDOW_SLOTS + 1)
    g = graph.build_wallet_graph(store)
    assert g["clusters"] == []


def test_cohort_mixed_strong_and_windowed_promotes_and_counts_once(
        tmp_path):
    store = PumpStore(str(tmp_path))
    m1, m2 = _mint(56), _mint(57)
    # same-slot on m1 (strong), windowed on m2
    store.record_trade(_trade(W1, m1, True, 3, 300, BASE_TS, slot=700))
    store.record_trade(_trade(W2, m1, True, 3, 300, BASE_TS, slot=700))
    _windowed_pair(store, W1, W2, m2, 3000, 3002)
    # both are profitable closers elsewhere, on solo mints/slots
    ts = BASE_TS + 50000
    for w in (W1, W2):
        for i in range(3):
            m = _mint(60 + i) if w == W1 else _mint(70 + i)
            slot = 11000 if w == W1 else 12000
            store.record_trade(_trade(w, m, True, 10, 1000, ts,
                                      slot=slot))
            store.record_trade(_trade(w, m, False, 15, 1000,
                                      ts + 60, slot=slot))
            ts += 3600
    g = graph.build_wallet_graph(store, now=BASE_TS + 10_000_000)
    assert len(g["clusters"]) == 1
    ev = g["clusters"][0]["evidence"]
    assert ev["same_slot_launches"] == 1
    assert ev["windowed_launches"] == 1
    assert set(ev["evidence_kinds"]) == {"same_slot", "windowed"}
    conf = graph.graph_confluence(store, m1, graph=g)
    assert conf["hits"] == 1                # one entity, counted once
    weights = sorted(e["weight"] for e in conf["entries"])
    assert weights[0] == 0.0 and weights[1] > 0.0


# ── 5. KG mirror against a live database ──────────────────────────────

def _shelf_dossier(mint):
    return {
        "mint": mint, "symbol": "KGT", "name": "KG Token",
        "creator": CREATOR, "uri": "ipfs://kg",
        "curve": {"curve_pct": 0.9, "curve_pct_basis": "supply-split",
                  "price_sol": 1e-6, "market_cap_sol": 90.0,
                  "real_sol": 80.0, "virtual_sol": 110.0,
                  "reserve_divergence": 0.0, "complete_flag": False,
                  "quote_mint": "SOL", "graduated": True,
                  "state": {}},
        "authorities": {"mint_authority_live": False,
                        "freeze_authority_live": False},
        "holders": {"top1_share": 0.1, "top3_share": 0.2,
                    "top10_share": 0.3, "accounts_seen": 20},
        "dev": {"creator": CREATOR, "launches": 2,
                "known_outcomes": 2, "rug_ratio": 0.0,
                "graduation_rate": 0.5, "serial_rugger": False},
        "trades": {"trades": 30, "unique_buyers": 12,
                   "buy_sol": 40.0, "sell_sol": 5.0,
                   "block0_absorption_share": None},
        "verdict": {"verdict": "CALL_CANDIDATE", "composite": 0.7,
                    "evidence": "fixture"},
        "gaps": [],
    }


def _graded_wallet_rec(wallet):
    return {
        "wallet": wallet, "verdict": "smart", "weight": 0.8,
        "realized_sol": 24.0, "realized_recent_sol": 24.0,
        "closed_trades": 4, "closed_wins": 4, "win_rate": 1.0,
        "full_exits": 4, "partial_exits": 0, "positions": [],
        "basis_unknown_sells": 0, "open_cost_sol": 0.0,
        "buys": 4, "sells": 4, "tokens_traded": 4,
        "first_seen": BASE_TS, "last_seen": BASE_TS + 100,
        "momentum_leads": 1,
        "funding": "unknown — transfers are not observed",
        "cohorts": [],
    }


def test_kg_mirror_round_trips_through_live_db(tmp_path):
    from nomorals.agents.kg import KnowledgeGraph
    from nomorals.storage.db import Database

    db = Database(str(tmp_path / "kg-live.db"))
    db.migrate()
    ctx = SimpleNamespace(db=db)
    store = PumpStore(str(tmp_path / "desk"))
    mint = _mint(61)
    store.save_dossier(_shelf_dossier(mint))
    store.save_graph({"wallets": {W1: _graded_wallet_rec(W1)},
                      "clusters": [], "built_at": BASE_TS})
    out = shelve_notable(store, context=ctx, now=BASE_TS)
    assert out == {"dossiers": 1, "wallets": 1}
    items = {it["kind"]: it for it in store.knowledge()}
    assert items["pump-dossier"]["kg_mirrored"] is True
    assert items["pump-wallet"]["kg_mirrored"] is True

    kg = KnowledgeGraph(db)
    node = kg.find_node(mint)
    assert node is not None
    assert node.properties["kind"] == "pump-token"
    assert node.properties["symbol"] == "KGT"
    assert node.properties["graduated"] is True
    assert node.properties["verdict"] == "CALL_CANDIDATE"
    assert node.properties["creator"] == CREATOR
    creator = kg.find_node(CREATOR)
    assert creator is not None
    assert creator.properties["kind"] == "pump-wallet"
    edges = kg.query(subject=CREATOR, relation="launched", obj=mint)
    assert len(edges) == 1
    wallet_node = kg.find_node(W1)
    assert wallet_node is not None
    assert wallet_node.properties["kind"] == "pump-wallet"
    assert wallet_node.properties["verdict"] == "smart"
    assert wallet_node.properties["closed_trades"] == 4


def test_kg_mirror_fail_soft_on_dead_db(tmp_path):
    store = PumpStore(str(tmp_path))
    mint = _mint(62)
    ctx = SimpleNamespace(db=object())       # no query API at all
    item = shelve_dossier(store, _shelf_dossier(mint),
                          context=ctx, now=BASE_TS)
    assert item["kg_mirrored"] is False
    assert store.knowledge()[0]["id"] == item["id"]   # shelf intact
    item2 = shelve_dossier(store, _shelf_dossier(mint),
                           context=None, now=BASE_TS)
    assert item2["kg_mirrored"] is False
