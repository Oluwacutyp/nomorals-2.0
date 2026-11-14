"""Pump desk feed selection — the 2026-11-14 "right stuff" fix.

The desk kept failing on the owner's box because its feed
selection pointed at sources that deny this deployment while
working sources sat unused.  The trace (live probes from this
box, 2026-11-14):

* DexScreener token-profiles/latest: 200, 30 rows — WORKS, and
  pump mints are identifiable (address ends in "pump", the
  PumpVet filter).  The desk used DexScreener only as a grader.
* RugCheck report: 200 — WORKS, and it carries exactly the
  de-rug evidence the dossier reads from the throttled RPC
  (mint/freeze authority, top holders, insiders, risks, rugged).
  The desk never called it; ``build_dossier``'s ``http_get``
  parameter was dead plumbing.
* pump.fun frontend /coins: 403, empty body — Cloudflare-blocked
  from datacenter IPs.  The desk polled it as its primary
  discovery leg every cycle.
* dRPC: HTTP 400 code 35 "chain is not available on free plan"
  on getSlot AND getAccountInfo — deterministically dead, yet
  still in the route list.

The fix pins the desk to the right stuff: DexScreener profiles
are the primary discovery leg (frontend API is the fallback),
RugCheck is the dossier's de-rug fallback, dRPC leaves the route
list.  Signals only — no buying/selling/wallet code anywhere.
"""

from __future__ import annotations

import json
import time

import pytest

from nomorals.agents.pumpdesk import chain, firehose, score, sources
from nomorals.agents.pumpdesk.desk import PumpDesk
from nomorals.agents.pumpdesk.dossier import build_dossier, fetch_rugcheck_report, rugcheck_evidence
from nomorals.agents.pumpdesk.firehose import (
    dex_profile_candidate,
    fetch_dex_profiles,
)
from nomorals.agents.pumpdesk.store import PumpStore
from nomorals.media.source_registry import SourceRegistry

# A Solana pump mint (addresses end in "pump") and a creator.
PUMP_MINT = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpjM5xTApump"
OTHER_MINT = "9xQeWvG816bUx9EPjHmaT23yvVM2ZWbrrpjM5xTAother"
CREATOR = chain.b58encode(bytes(range(1, 33)))
INIT_VT = 1_073_000_000_000_000
INIT_VS = 30_000_000_000
INIT_RT = 793_100_000_000_000
SUPPLY = 1_000_000_000_000_000

# A RugCheck report in the shape the live API returns (verified
# 2026-11-14): the top holder IS the pump.fun pool (its owner is
# the market pubkey) — counting it as concentration is wrong.
_RUG_REPORT = {
    "mint": PUMP_MINT,
    "creator": CREATOR,
    "creatorBalance": 0,
    "rugged": False,
    "score": 1,
    "totalHolders": 271,
    "mintAuthority": None,
    "freezeAuthority": None,
    "token": {"supply": SUPPLY, "decimals": 6, "mintAuthority": None,
              "freezeAuthority": None, "isInitialized": True},
    "tokenMeta": {"name": "Test", "symbol": "TEST"},
    "topHolders": [
        {"address": "9yVUrYZKksx9pool",
         "owner": "Ct616JhqxevCntwodgXST1CjK3GPdiXY656QxL8u9sWg",
         "pct": 54.38, "insider": False},
        {"address": "6LvKJr6UipRE", "owner": "FLDpn97SuDz8",
         "pct": 3.55, "insider": False},
        {"address": "9fJvrMJ4nPaa", "owner": "VJSDW6S74YXR",
         "pct": 3.03, "insider": True},
        {"address": "8D3ZBVnuRBgk", "owner": "9Za4f5sTCpkV",
         "pct": 2.61, "insider": False},
    ],
    "markets": [{"pubkey": "Ct616JhqxevCntwodgXST1CjK3GPdiXY656QxL8u9sWg",
                 "mintLP": "11111111111111111111111111111111",
                 "liquidityA": "6ctoW8ER2spao9hgrywoAGY8ycgcVG9hYEvwPNgz",
                 "liquidityB": "So11111111111111111111111111111111111111112",
                 "marketType": "pump_fun"}],
    "risks": [{"name": "Low Liquidity", "level": "warn"}],
    "graphInsidersDetected": 1,
}


def _curve():
    return chain.BondingCurveState(
        virtual_token_reserves=INIT_VT, virtual_sol_reserves=INIT_VS,
        real_token_reserves=INIT_RT, real_sol_reserves=0,
        token_total_supply=SUPPLY, creator=CREATOR)


def _mint_info():
    return chain.MintInfo(mint_authority=None, freeze_authority=None,
                          supply=SUPPLY, decimals=6, initialized=True)


class _HealthyRPC:
    """Duck-typed SolanaRPC double: every chain read answers."""

    def __init__(self):
        self.rugcheck_calls = 0

    def get_curve(self, mint):
        return _curve()

    def get_mint_info(self, mint):
        return _mint_info()

    def get_token_largest(self, mint):
        return [{"amount": "50000000000000"},
                {"amount": "30000000000000"}]

    def get_account(self, pubkey):
        return None

    def get_signatures(self, address, limit=50, before=""):
        return []

    def get_transaction(self, signature):
        return None


class _DeadRPC(_HealthyRPC):
    """The RPC reads the dossier's de-rug evidence needs are dead;
    the curve still answers (so the dossier has a curve)."""

    def get_mint_info(self, mint):
        raise RuntimeError("rpc down")

    def get_token_largest(self, mint):
        raise RuntimeError("rpc down")


class _HttpGet:
    """An http_get double: scripted (status, body) per URL
    substring, with a call count."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append(url)
        for needle, (status, body) in self.mapping.items():
            if needle in url:
                return status, body
        return 404, ""


def _seeded_registry():
    reg = SourceRegistry()
    sources.ensure_pump_sources(reg)
    return reg


# ── DexScreener discovery ───────────────────────────────────────────

def test_dex_profile_candidate_accepts_solana_pump_mint():
    cand = dex_profile_candidate({
        "chainId": "solana", "tokenAddress": PUMP_MINT,
        "url": "https://dexscreener.com/solana/" + PUMP_MINT})
    assert cand is not None
    assert cand["mint"] == PUMP_MINT
    assert cand["source"] == "dexscreener-profiles"


def test_dex_profile_candidate_rejects_non_pump_and_non_solana():
    assert dex_profile_candidate({
        "chainId": "solana", "tokenAddress": OTHER_MINT}) is None
    assert dex_profile_candidate({
        "chainId": "ethereum", "tokenAddress": PUMP_MINT}) is None
    assert dex_profile_candidate({"chainId": "solana"}) is None
    assert dex_profile_candidate("not a dict") is None


def test_fetch_dex_profiles_returns_rows_and_sends_desk_ua():
    rows = [{"chainId": "solana", "tokenAddress": PUMP_MINT}]
    seen = {}

    def http_get(url, headers):
        seen.update(headers)
        assert url == ("https://api.dexscreener.com/"
                       "token-profiles/latest/v1")
        return 200, json.dumps(rows)

    assert fetch_dex_profiles(http_get) == rows
    # the fetch sends the desk's ONE resolved user-agent (the
    # configured tools.user_agent in this environment, the desk
    # default where none is configured) — never a per-call literal
    assert seen["User-Agent"] == sources.desk_user_agent()


def test_fetch_dex_profiles_raises_on_non_200_and_caps_limit():
    def http_get(url, headers):
        return 429, "slow down"

    with pytest.raises(OSError, match="429"):
        fetch_dex_profiles(http_get)
    rows = [{"chainId": "solana", "tokenAddress": f"mint{i}pump"}
            for i in range(5)]
    got = fetch_dex_profiles(
        lambda url, headers: (200, json.dumps(rows)), limit=2)
    assert len(got) == 2


# ── RugCheck report ─────────────────────────────────────────────────

def test_fetch_rugcheck_report_200_and_404():
    http_get = _HttpGet({"rugcheck.xyz": (200, json.dumps(_RUG_REPORT))})
    report = fetch_rugcheck_report(http_get, PUMP_MINT)
    assert report is not None and report["mint"] == PUMP_MINT
    assert http_get.calls and "rugcheck.xyz" in http_get.calls[0]
    assert fetch_rugcheck_report(
        _HttpGet({"rugcheck.xyz": (404, "")}), PUMP_MINT) is None


def test_rugcheck_evidence_excludes_pool_and_reads_flags():
    ev = rugcheck_evidence(_RUG_REPORT)
    auth = ev["authorities"]
    assert auth["mint_authority_live"] is False    # revoked (None)
    assert auth["freeze_authority_live"] is False
    assert auth["supply"] == SUPPLY and auth["source"] == "rugcheck"
    holders = ev["holders"]
    # the 54.38% pool holder is excluded: top-1 is the 3.55% wallet
    assert holders["top1_share"] == pytest.approx(0.0355)
    assert holders["top10_share"] == pytest.approx(0.0919)
    assert holders["accounts_seen"] == 3
    assert holders["partial"] is False
    assert holders["source"] == "rugcheck"
    assert holders["insider_pct"] == pytest.approx(0.0303)
    assert holders["creator_pct"] == pytest.approx(0.0)
    assert holders["total_holders"] == 271
    assert ev["creator"] == CREATOR
    assert ev["rugged"] is False
    assert ev["risks_warn"] == ["Low Liquidity"]
    assert ev["risks_danger"] == []


def test_rugcheck_evidence_live_mint_authority_and_rugged():
    report = dict(_RUG_REPORT, rugged=True,
                  mintAuthority="Auth11111111111111111111111111111111")
    ev = rugcheck_evidence(report)
    assert ev["authorities"]["mint_authority_live"] is True
    assert ev["rugged"] is True


# ── dossier fallback ────────────────────────────────────────────────

def test_build_dossier_falls_back_to_rugcheck_when_rpc_dead(tmp_path):
    store = PumpStore(str(tmp_path))
    store.record_launch({"mint": PUMP_MINT, "creator": CREATOR,
                         "name": "Test", "symbol": "TEST",
                         "ts": time.time(), "source": "test"})
    http_get = _HttpGet({"rugcheck.xyz": (200, json.dumps(_RUG_REPORT))})
    d = build_dossier(store, _DeadRPC(), PUMP_MINT, http_get=http_get)
    assert len(http_get.calls) == 1    # one report fetch, not one per gap
    assert d["authorities"]["source"] == "rugcheck"
    assert d["authorities"]["mint_authority_live"] is False
    assert d["authorities"]["freeze_authority_live"] is False
    assert d["holders"]["source"] == "rugcheck"
    assert d["holders"]["top1_share"] == pytest.approx(0.0355)
    assert d["holders"]["top10_share"] == pytest.approx(0.0919)
    assert d["creator"] == CREATOR
    assert d["rugged"] is False
    assert d["rugcheck"]["risks_warn"] == ["Low Liquidity"]
    assert "authority flags unknown" not in d["gaps"]
    assert not any("holder concentration unavailable" in g
                   for g in d["gaps"])


def test_build_dossier_prefers_chain_when_rpc_answers(tmp_path):
    store = PumpStore(str(tmp_path))
    store.record_launch({"mint": PUMP_MINT, "creator": CREATOR,
                         "name": "Test", "symbol": "TEST",
                         "ts": time.time(), "source": "test"})
    http_get = _HttpGet({"rugcheck.xyz": (200, json.dumps(_RUG_REPORT))})
    d = build_dossier(store, _HealthyRPC(), PUMP_MINT,
                      http_get=http_get)
    assert http_get.calls == []    # the chain answered: no fallback
    assert "source" not in (d["authorities"] or {})
    assert d["holders"]["source"] == "largest-accounts"
    assert "rugged" not in d    # no report consulted, no rugged claim


def test_build_dossier_rugcheck_gap_when_no_report(tmp_path):
    store = PumpStore(str(tmp_path))
    store.record_launch({"mint": PUMP_MINT, "creator": CREATOR,
                         "name": "Test", "symbol": "TEST",
                         "ts": time.time(), "source": "test"})
    http_get = _HttpGet({"rugcheck.xyz": (404, "")})
    d = build_dossier(store, _DeadRPC(), PUMP_MINT, http_get=http_get)
    assert d["authorities"] is None
    assert d["holders"] is None
    assert "authority flags unknown" in d["gaps"]
    assert any("holder concentration unavailable" in g
               for g in d["gaps"])


# ── score: the rugged gate ──────────────────────────────────────────

def _passing_dossier():
    return {
        "mint": PUMP_MINT, "symbol": "TEST", "name": "Test",
        "curve": {"curve_pct": 0.5, "market_cap_sol": 60.0,
                  "price_sol": 6e-8, "real_sol": 40.0,
                  "complete_flag": False, "quote_mint": chain.SOL_MINT},
        "authorities": {"mint_authority_live": False,
                        "freeze_authority_live": False},
        "holders": {"top1_share": 0.05, "top3_share": 0.12,
                    "top10_share": 0.3, "accounts_seen": 20},
        "dev": {"creator": CREATOR, "launches": 5, "known_outcomes": 5,
                "rug_ratio": 0.0, "graduation_rate": 0.2,
                "serial_rugger": False},
        "trades": {"trades": 40, "unique_buyers": 20,
                   "buy_sol": 30.0, "sell_sol": 10.0,
                   "buy_sell_ratio": 3.0, "velocity_sol_per_min": 1.5,
                   "sol_per_trade": 0.75,
                   "block0_absorption_share": None,
                   "block0_total_share": None,
                   "dev_sold": False, "dev_sold_sol": 0.0,
                   "bump_alpha_max": 0.0, "bump_wallets": 0},
        "gaps": [],
    }


def test_score_rugged_gate_blocks_call():
    d = _passing_dossier()
    assert score.evaluate(d).verdict == "CALL_CANDIDATE"
    d["rugged"] = True
    v = score.evaluate(d)
    assert v.verdict == "NO_CALL"
    assert any("rugcheck reports this token as rugged" in g
               for g in v.gates)
    assert v.rug_risk >= score.RUG_GATE


# ── routes: dRPC out, RugCheck in ───────────────────────────────────

def test_rpc_routes_have_no_drpc(monkeypatch):
    monkeypatch.delenv("HELIUS_API_KEY", raising=False)
    routes = sources.rpc_routes(None)
    assert len(routes) == 2
    assert [r["domain"] for r in routes] == [
        "api.mainnet-beta.solana.com", "solana-rpc.publicnode.com"]
    monkeypatch.setenv("HELIUS_API_KEY", "test-key")
    keyed = sources.rpc_routes(None)
    assert keyed[0]["source"] == "helius (provisioned)"
    assert "solana.drpc.org" not in [r["domain"] for r in keyed]


def test_rugcheck_seed_registered():
    reg = _seeded_registry()
    assert reg.for_domain("api.rugcheck.xyz") is not None
    rows = sources.pump_status(reg)["sources"]
    assert any(r["domain"] == "api.rugcheck.xyz" for r in rows)


# ── watch cycle: discovery order ────────────────────────────────────

def _profile_row(mint=PUMP_MINT):
    return {"chainId": "solana", "tokenAddress": mint,
            "url": "https://dexscreener.com/solana/" + mint}


def _desk(tmp_path, rpc):
    return PumpDesk(context=None, desk_dir=str(tmp_path),
                    registry=_seeded_registry(), rpc=rpc)


def test_watch_cycle_dexscreener_discovers_and_frontend_skipped(
        tmp_path, monkeypatch):
    desk = _desk(tmp_path, _HealthyRPC())
    monkeypatch.setattr(
        firehose, "fetch_dex_profiles",
        lambda http_get, limit=30: [_profile_row()])
    monkeypatch.setattr(
        PumpDesk, "_dex_snapshot",
        lambda self, mint: {"dex": "pumpswap", "pair": "pairaddr",
                            "name": "Test", "symbol": "TEST",
                            "price_native": 0.001,
                            "market_cap_usd": 50000.0,
                            "liquidity_usd": 20000.0,
                            "pair_created_at": 1234567890000,
                            "url": "https://dexscreener.com/solana/x"})
    called = []
    monkeypatch.setattr(
        firehose, "fetch_frontend",
        lambda http_get, limit=50: called.append(True) or [])
    out = desk.watch_cycle()
    assert out["dexscreener"]["promoted"] == 1
    assert out["dexscreener"]["profiles"] == 1
    assert called == []    # the blocked leg is never polled
    assert "skipped" in out["frontend"]
    assert desk.store.launch(PUMP_MINT) is not None
    assert desk.store.dossier(PUMP_MINT) is not None
    assert desk.store.dossier(PUMP_MINT)["symbol"] == "TEST"


def test_watch_cycle_dexscreener_pairless_ledgers_only(
        tmp_path, monkeypatch):
    desk = _desk(tmp_path, _HealthyRPC())
    monkeypatch.setattr(
        firehose, "fetch_dex_profiles",
        lambda http_get, limit=30: [_profile_row()])
    monkeypatch.setattr(PumpDesk, "_dex_snapshot",
                        lambda self, mint: None)
    called = []
    monkeypatch.setattr(
        firehose, "fetch_frontend",
        lambda http_get, limit=50: called.append(True) or [])
    out = desk.watch_cycle()
    assert out["dexscreener"]["promoted"] == 0
    assert out["dexscreener"]["new"] == 1
    assert desk.store.launch(PUMP_MINT) is not None
    assert desk.store.dossier(PUMP_MINT) is None    # no dossier spend
    assert called == []    # DexScreener still delivered something new


def test_watch_cycle_frontend_fallback_when_dexscreener_empty(
        tmp_path, monkeypatch):
    desk = _desk(tmp_path, _HealthyRPC())
    monkeypatch.setattr(firehose, "fetch_dex_profiles",
                        lambda http_get, limit=30: [])
    row = {"mint": OTHER_MINT, "creator": CREATOR, "name": "G",
           "symbol": "G", "virtual_token_reserves": INIT_VT,
           "virtual_sol_reserves": INIT_VS,
           "real_token_reserves": INIT_RT // 2,
           "real_sol_reserves": 40_000_000_000,
           "total_supply": SUPPLY, "complete": False,
           "bonding_curve": chain.bonding_curve_pda(OTHER_MINT)}
    monkeypatch.setattr(firehose, "fetch_frontend",
                        lambda http_get, limit=50: [row])
    out = desk.watch_cycle()
    assert out["dexscreener"]["profiles"] == 0
    assert out["frontend"]["promoted"] == 1
    assert desk.store.dossier(OTHER_MINT) is not None


def test_watch_cycle_frontend_fallback_when_dexscreener_errors(
        tmp_path, monkeypatch):
    desk = _desk(tmp_path, _HealthyRPC())
    monkeypatch.setattr(
        firehose, "fetch_dex_profiles",
        lambda http_get, limit=30: (_ for _ in ()).throw(
            OSError("dexscreener profiles answered HTTP 429")))
    monkeypatch.setattr(firehose, "fetch_frontend",
                        lambda http_get, limit=50: [])
    out = desk.watch_cycle()
    assert "error" in out["dexscreener"]
    assert out["frontend"]["rows"] == 0
