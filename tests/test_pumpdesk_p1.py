"""Pump desk P1 — intake, dossier, de-rug gate, calls, self-scoring.

Fixtures are built from the design's VERIFIED constants
(pump-mining-design-2026-10-22 §7): init reserves
virtual_token=1,073,000,000,000,000 / virtual_sol=30,000,000,000,
real_token=793,100,000,000,000, supply=1,000,000,000,000,000.
Event payloads are packed with struct here, so the decoders are
tested against the Anchor layout, not against themselves.
"""

from __future__ import annotations

import base64
import json
import os
import socket
import struct
import threading
import time

import pytest

from nomorals.agents.pumpdesk import chain, firehose, score
from nomorals.agents.pumpdesk.calls import (CallEngine, PumpCall,
                                            calibration)
from nomorals.agents.pumpdesk.desk import PumpDesk
from nomorals.agents.pumpdesk.dossier import (build_dev_profile,
                                              trade_facts)
from nomorals.agents.pumpdesk.firehose import (Firehose, Prefilter,
                                               base_backoff,
                                               frontend_candidate)
from nomorals.agents.pumpdesk.store import PumpStore
from nomorals.media.source_registry import SourceRegistry

INIT_VT = 1_073_000_000_000_000
INIT_VS = 30_000_000_000
INIT_RT = 793_100_000_000_000
SUPPLY = 1_000_000_000_000_000
CREATOR = chain.b58encode(bytes(range(1, 33)))
MINT = chain.b58encode(bytes(range(33, 65)))
CURVE = chain.bonding_curve_pda(MINT)


def _curve_bytes(vt=INIT_VT, vs=INIT_VS, rt=INIT_RT, rs=0, supply=SUPPLY,
                 complete=0, creator_b=None):
    return (b"\x00" * 8 + struct.pack("<QQQQQ", vt, vs, rt, rs, supply)
            + bytes([complete]) + (creator_b or bytes(range(1, 33))))


def _trade_event_bytes(mint_b, sol, toks, is_buy, user_b, ts, vs, vt,
                       rs=0, rt=0):
    payload = chain.event_discriminator("TradeEvent")
    payload += mint_b
    payload += struct.pack("<Q", sol) + struct.pack("<Q", toks)
    payload += bytes([1 if is_buy else 0])
    payload += user_b
    payload += struct.pack("<q", ts)
    payload += struct.pack("<Q", vs) + struct.pack("<Q", vt)
    payload += struct.pack("<Q", rs) + struct.pack("<Q", rt)
    return payload


# ── chain: base58 / PDA / account parsing ─────────────────────────────

def test_b58_known_vector():
    assert chain.b58encode(b"hello world") == "StV1DL6CwTryKyV"
    assert chain.b58decode("StV1DL6CwTryKyV") == b"hello world"
    assert len(chain.b58decode(chain.SOL_MINT)) == 32


def test_pda_deterministic_and_mint_specific():
    a = chain.bonding_curve_pda(MINT)
    assert a == chain.bonding_curve_pda(MINT)
    assert a != chain.bonding_curve_pda(CREATOR)
    # a PDA is NOT a valid ed25519 point (the defining property)
    assert not chain._is_on_curve(chain.b58decode(a))


def test_parse_bonding_curve_exact():
    st = chain.parse_bonding_curve(_curve_bytes())
    assert st.virtual_token_reserves == INIT_VT
    assert st.virtual_sol_reserves == INIT_VS
    assert st.real_token_reserves == INIT_RT
    assert st.token_total_supply == SUPPLY
    assert st.complete is False
    assert st.creator == CREATOR
    assert st.quote_mint == chain.SOL_MINT


def test_parse_mint_account_authorities():
    def mint_bytes(ma_tag, fa_tag):
        out = struct.pack("<I", ma_tag) + bytes(range(1, 33))
        out += struct.pack("<Q", SUPPLY) + bytes([6, 1])
        out += struct.pack("<I", fa_tag) + bytes(range(33, 65))
        return out
    info = chain.parse_mint_account(mint_bytes(0, 0))
    assert info.mint_authority is None and info.freeze_authority is None
    assert info.supply == SUPPLY and info.decimals == 6
    info2 = chain.parse_mint_account(mint_bytes(1, 1))
    assert info2.mint_authority == CREATOR
    assert info2.freeze_authority == MINT


# ── chain: curve math from a fixture TradeEvent ───────────────────────

def test_trade_event_decode_and_curve_math():
    payload = _trade_event_bytes(
        chain.b58decode(MINT), sol=5_000_000_000, toks=150_000_000_000_000,
        is_buy=True, user_b=bytes(range(1, 33)), ts=1_700_000_000,
        vs=115_000_000_000, vt=279_900_000_000_000,
        rs=85_000_000_000, rt=0)
    ev = chain.decode_program_data(payload)
    assert ev["event"] == "TradeEvent"
    assert ev["mint"] == MINT and ev["is_buy"] is True
    assert ev["sol_amount"] == 5_000_000_000
    assert ev["real_sol_reserves"] == 85_000_000_000
    # curve math on the event's own virtual reserves (sell-out state)
    st = chain.BondingCurveState(
        virtual_token_reserves=ev["virtual_token_reserves"],
        virtual_sol_reserves=ev["virtual_sol_reserves"],
        real_token_reserves=0, real_sol_reserves=85_000_000_000,
        token_total_supply=SUPPLY)
    mc_sol = chain.market_cap_quote(st) / 1e9
    expected = 115_000_000_000 * SUPPLY / 279_900_000_000_000 / 1e9
    assert mc_sol == pytest.approx(expected, rel=1e-9)
    assert mc_sol == pytest.approx(410.86, abs=0.05)   # §7.3: ~410.9 SOL
    pct, basis = chain.curve_pct(st)
    assert pct == 1.0 and basis == "supply-split"
    # at init: price 2.8e-8 SOL, MC ~27.96 SOL (design §7.1)
    st0 = chain.BondingCurveState(
        virtual_token_reserves=INIT_VT, virtual_sol_reserves=INIT_VS,
        real_token_reserves=INIT_RT, real_sol_reserves=0,
        token_total_supply=SUPPLY)
    assert chain.price_quote_per_token(st0) / 1e9 == pytest.approx(
        2.8e-8, rel=0.01)
    assert chain.market_cap_quote(st0) / 1e9 == pytest.approx(27.96,
                                                              abs=0.05)
    pct0, _ = chain.curve_pct(st0)
    assert pct0 == 0.0
    # observed-init basis: exact per-token sell-through
    half = chain.BondingCurveState(
        virtual_token_reserves=INIT_VT, virtual_sol_reserves=INIT_VS,
        real_token_reserves=INIT_RT // 2, real_sol_reserves=1,
        token_total_supply=SUPPLY)
    pct_h, basis_h = chain.curve_pct(half, initial_real_token_reserves=INIT_RT)
    assert pct_h == pytest.approx(0.5) and basis_h == "observed-init"


def test_create_event_decode():
    def _s(x):
        b = x.encode()
        return struct.pack("<I", len(b)) + b
    payload = chain.event_discriminator("CreateEvent")
    payload += _s("Gold Coin") + _s("GOLD") + _s("https://ipfs/x")
    payload += chain.b58decode(MINT) + chain.b58decode(CURVE)
    payload += bytes(range(1, 33))
    ev = chain.decode_program_data(payload)
    assert ev["event"] == "CreateEvent"
    assert ev["name"] == "Gold Coin" and ev["symbol"] == "GOLD"
    assert ev["mint"] == MINT and ev["creator"] == CREATOR
    lines = ["Program log: something",
             "Program data: " + base64.b64encode(payload).decode()]
    evs = chain.decode_log_lines(lines)
    assert len(evs) == 1 and evs[0]["mint"] == MINT


# ── graduation: migration event / pool, NOT CompleteEvent ─────────────

class FakeRPC:
    """Duck-typed SolanaRPC double with canned accounts."""

    def __init__(self, curve=None, mint_info=None, pool=False,
                 largest=None):
        self._curve = curve
        self._mint_info = mint_info
        self._pool = pool
        self._largest = largest or []
        self.curve_reads = 0

    def get_curve(self, mint):
        self.curve_reads += 1
        return self._curve

    def get_mint_info(self, mint):
        return self._mint_info

    def get_token_largest(self, mint):
        return self._largest

    def get_account(self, pubkey):
        return b"\x01" * 100 if self._pool else None

    def get_signatures(self, address, limit=50, before=""):
        return []

    def get_transaction(self, signature):
        return None


def test_graduation_keys_on_pool_not_complete_event(tmp_path):
    store = PumpStore(str(tmp_path))
    store.record_launch({"mint": MINT, "creator": CREATOR, "slot": 5,
                         "name": "G", "symbol": "G", "ts": time.time(),
                         "source": "test"})
    curve = chain.BondingCurveState(
        virtual_token_reserves=INIT_VT, virtual_sol_reserves=INIT_VS,
        real_token_reserves=INIT_RT, real_sol_reserves=0,
        token_total_supply=SUPPLY, creator=CREATOR)
    mint_info = chain.MintInfo(supply=SUPPLY, decimals=6, initialized=True)
    from nomorals.agents.pumpdesk.dossier import build_dossier
    # pool absent -> NOT graduated, even though curve exists
    d = build_dossier(store, FakeRPC(curve, mint_info, pool=False), MINT)
    assert d["curve"]["graduated"] is False
    # a CompleteEvent alone changes nothing (design §7.5)
    payload = chain.event_discriminator("CompleteEvent")
    payload += chain.b58decode(MINT) + bytes(range(1, 33))
    payload += struct.pack("<q", 1_700_000_000) + chain.b58decode(CURVE)
    ev = chain.decode_program_data(payload)
    assert ev["event"] == "CompleteEvent"   # decodes, but proves nothing
    # canonical pool exists -> graduated
    d2 = build_dossier(store, FakeRPC(curve, mint_info, pool=True), MINT)
    assert d2["curve"]["graduated"] is True


# ── block-0 absorption from fixture slot trades ───────────────────────

def test_block0_absorption_share():
    def trade(user, toks, slot, buy=True, sol=1_000_000_000, ts=100):
        return {"user": user, "token_amount": toks, "sol_amount": sol,
                "is_buy": buy, "slot": slot, "timestamp": ts}
    trades = [
        trade(CREATOR, 200_000_000_000_000, 1000),          # dev buy
        trade("WalletA", 100_000_000_000_000, 1000),        # block-0
        trade("WalletB", 50_000_000_000_000, 1000),         # block-0
        trade("WalletA", 700_000_000_000_000, 1001),        # later slot
    ]
    facts = trade_facts(trades, CREATOR, create_slot=1000,
                        sellable_raw=INIT_RT)
    assert facts["block0_absorption_share"] == pytest.approx(
        150_000_000_000_000 / INIT_RT, abs=1e-4)   # non-dev only: 18.9%
    assert facts["block0_total_share"] == pytest.approx(
        350_000_000_000_000 / INIT_RT, abs=1e-4)   # incl. dev: 44.1%
    assert facts["unique_buyers"] == 3
    assert facts["trades"] == 4


def test_bump_score_flags_wash_wallet():
    trades = []
    # WalletW flips buy/sell 6 times ending ~flat: the bump signature
    for i in range(6):
        trades.append({"user": "WalletW", "token_amount": 10_000_000,
                       "sol_amount": 1000, "is_buy": i % 2 == 0,
                       "slot": 2000 + i, "timestamp": 100 + i})
    # WalletH buys and holds
    trades.append({"user": "WalletH", "token_amount": 500_000_000,
                   "sol_amount": 9000, "is_buy": True, "slot": 2000,
                   "timestamp": 100})
    facts = trade_facts(trades, CREATOR, create_slot=0, sellable_raw=0)
    assert facts["bump_alpha_max"] >= 2.0
    assert facts["bump_wallets"] == 1


# ── dev profile: rug ratio + serial-rugger flag ───────────────────────

def test_dev_profile_serial_rugger(tmp_path):
    store = PumpStore(str(tmp_path))
    now = time.time()
    # 3 rugged launches: ran to 4x birth, collapsed to 0.2x peak
    for i in range(3):
        mint_i = chain.b58encode(bytes([100 + i]) * 32)
        store.record_launch({"mint": mint_i, "creator": CREATOR,
                             "ts": now - 90000, "birth_price": 100.0,
                             "source": "test"})
        store.record_observation(mint_i, now - 80000, 100.0, 1e10, 0.02,
                                 "test")
        store.record_observation(mint_i, now - 70000, 400.0, 4e10, 0.20,
                                 "test")
        store.record_observation(mint_i, now - 60000, 80.0, 8e9, 0.21,
                                 "test")
    # 1 graduated launch
    mint_g = chain.b58encode(bytes([200]) * 32)
    store.record_launch({"mint": mint_g, "creator": CREATOR,
                         "ts": now - 90000, "graduated": True,
                         "source": "test"})
    prof = build_dev_profile(store, CREATOR, now=now)
    assert prof["launches"] == 4
    assert prof["known_outcomes"] == 4
    assert prof["rug_ratio"] == pytest.approx(0.75)
    assert prof["graduation_rate"] == pytest.approx(0.25)
    assert prof["serial_rugger"] is True   # >=3 launches, >=67% rugged


# ── score: floors, gates, candidate ───────────────────────────────────

def _dossier_dict(trades_n=40, buyers=20, rug_ratio=0.0, known=5,
                  serial=False, mint_auth=False, curve_pct_v=0.5,
                  block0=None, dev_sold=False):
    return {
        "mint": MINT, "symbol": "GOLD", "name": "Gold",
        "curve": {"curve_pct": curve_pct_v, "market_cap_sol": 60.0,
                  "price_sol": 6e-8, "real_sol": 40.0,
                  "complete_flag": False, "quote_mint": chain.SOL_MINT},
        "authorities": {"mint_authority_live": mint_auth,
                        "freeze_authority_live": False},
        "holders": {"top1_share": 0.05, "top3_share": 0.12,
                    "top10_share": 0.3, "accounts_seen": 20},
        "dev": {"creator": CREATOR, "launches": 5, "known_outcomes": known,
                "rug_ratio": rug_ratio, "graduation_rate": 0.2,
                "serial_rugger": serial},
        "trades": {"trades": trades_n, "unique_buyers": buyers,
                   "buy_sol": 30.0, "sell_sol": 10.0, "buy_sell_ratio": 3.0,
                   "velocity_sol_per_min": 1.5, "sol_per_trade": 0.75,
                   "block0_absorption_share": block0,
                   "block0_total_share": None,
                   "dev_sold": dev_sold, "dev_sold_sol": 0.0,
                   "bump_alpha_max": 0.0, "bump_wallets": 0},
        "gaps": [],
    }


def test_verdict_floor_insufficient_data():
    v = score.evaluate(_dossier_dict(trades_n=3, buyers=2))
    assert v.verdict == "INSUFFICIENT_DATA"
    assert "sample floors unmet" in v.evidence


def test_verdict_serial_rugger_gate_beats_momentum():
    v = score.evaluate(_dossier_dict(serial=True, rug_ratio=1.0))
    assert v.verdict == "NO_CALL"
    assert any("serial-rugger" in g for g in v.gates)


def test_verdict_block0_bundle_gate():
    v = score.evaluate(_dossier_dict(block0=0.55))
    assert v.verdict == "NO_CALL"
    assert v.manipulation >= score.MANIP_GATE


def test_verdict_call_candidate():
    v = score.evaluate(_dossier_dict())
    assert v.verdict == "CALL_CANDIDATE", v.evidence
    assert v.composite >= score.CALL_THRESHOLD


# ── calls: publish format + ledger, grading ───────────────────────────

class FakeNotifier:
    def __init__(self):
        self.sent = []

    def publish(self, kind, title, body, **kw):
        self.sent.append({"kind": kind, "title": title, "body": body,
                          "kw": kw})
        return {"sent": True}


def test_call_publish_ledger_and_format(tmp_path):
    notifier = FakeNotifier()
    engine = CallEngine(str(tmp_path), notifier=notifier)
    d = _dossier_dict()
    v = score.evaluate(d)
    call = engine.build_call(d, v, now=1_700_000_000.0)
    assert call is not None
    assert call.target_mc_sol == pytest.approx(120.0)
    assert call.invalidation_mc_sol == pytest.approx(33.0)
    out = engine.publish(call)
    assert out["delivered"] is True
    rows = engine.log.calls()
    assert len(rows) == 1 and rows[0].call_id == call.call_id
    body = notifier.sent[0]["body"]
    assert "SOL mcap" in body and "invalidation" in body
    assert "expires" in body and MINT in body
    assert notifier.sent[0]["kind"] == "pumpdesk"
    # no hints -> calls are un-tiered standing content (Pass-7 rule)
    assert "alert_hints" not in notifier.sent[0]["kw"]


def _expired_call(tmp_path, engine=None):
    engine = engine or CallEngine(str(tmp_path))
    call = PumpCall(mint=MINT, symbol="GOLD", published_at=1000.0,
                    expires_at=2000.0, entry_price_sol=3e-8,
                    entry_mc_sol=30.0, target_mc_sol=60.0,
                    invalidation_mc_sol=16.5, call_id="c1")
    engine.log.add(call)
    return engine, call


def test_score_due_win_first_touch(tmp_path):
    store = PumpStore(str(tmp_path))
    engine, _ = _expired_call(tmp_path)
    for ts, mc_sol in [(1100, 30), (1200, 45), (1300, 65), (1400, 50)]:
        store.record_observation(MINT, ts, 3e-8, mc_sol * 1e9, 0.4, "t")
    rep = engine.score_due(store, now=3000.0)
    c = engine.log.calls()[0]
    assert c.scored is True and c.outcome == "win"
    assert c.graded_via == "observations"
    assert c.max_favorable == pytest.approx(65 / 30, abs=0.01)
    assert rep["hit_rate"] == 1.0 and rep["scored_now"] == 1


def test_score_due_loss_invalidation_first(tmp_path):
    store = PumpStore(str(tmp_path))
    engine, _ = _expired_call(tmp_path)
    for ts, mc_sol in [(1100, 30), (1200, 15), (1300, 70)]:
        store.record_observation(MINT, ts, 3e-8, mc_sol * 1e9, 0.4, "t")
    engine.score_due(store, now=3000.0)
    c = engine.log.calls()[0]
    assert c.scored is False and c.outcome == "loss"


def test_score_due_no_data_is_a_miss(tmp_path):
    store = PumpStore(str(tmp_path))
    engine, _ = _expired_call(tmp_path)
    rep = engine.score_due(store, snapshot_fn=lambda m: None, now=3000.0)
    c = engine.log.calls()[0]
    assert c.scored is False and c.outcome == "miss-no-data"
    assert c.graded_via == "none"
    assert rep["misses_no_data"] == 1


def test_score_due_dexscreener_fallback(tmp_path):
    store = PumpStore(str(tmp_path))
    engine, _ = _expired_call(tmp_path)
    snap = {"dex": "pumpswap", "price_native": 9e-8}   # 3x entry price
    engine.score_due(store, snapshot_fn=lambda m: snap, now=3000.0)
    c = engine.log.calls()[0]
    assert c.scored is True and c.graded_via == "dexscreener"
    assert c.graduated is True


def test_score_due_leaves_unexpired_alone(tmp_path):
    store = PumpStore(str(tmp_path))
    engine = CallEngine(str(tmp_path))
    call = PumpCall(mint=MINT, published_at=1000.0, expires_at=99999.0,
                    entry_mc_sol=30.0, call_id="c2")
    engine.log.add(call)
    rep = engine.score_due(store, now=3000.0)
    assert rep["scored_now"] == 0
    assert engine.log.calls()[0].scored is None


# ── registry seeds + feed failover ────────────────────────────────────

def test_registry_seeds_and_status():
    from nomorals.agents.pumpdesk.sources import (ensure_pump_sources,
                                                  pump_status)
    reg = SourceRegistry()
    ensure_pump_sources(reg)
    ensure_pump_sources(reg)   # idempotent
    st = pump_status(reg)
    assert st["count"] == 7   # 5 P1 seeds + 2 added public RPC routes
    assert all(s["health"] == "unknown" for s in st["sources"])


def test_firehose_failover_skips_stood_down_backend():
    from nomorals.agents.pumpdesk.sources import ensure_pump_sources
    reg = SourceRegistry()
    ensure_pump_sources(reg)
    desc = reg.for_domain("pumpportal.fun")
    for _ in range(3):
        reg.record_outcome(desc, False, error_class="WSClosed")
    assert reg.health_for(desc) == "stood-down"
    fh = Firehose(lambda k, p: None, registry=reg)
    backends = fh._backends()
    assert [b.name for b in backends] == ["rpc-logs"]
    # and with both healthy, pumpportal leads
    reg2 = SourceRegistry()
    ensure_pump_sources(reg2)
    fh2 = Firehose(lambda k, p: None, registry=reg2)
    assert [b.name for b in fh2._backends()] == ["pumpportal", "rpc-logs"]


def test_backoff_schedule():
    assert base_backoff(0) == 2.0
    assert base_backoff(1) == 4.0
    assert base_backoff(3) == 16.0
    assert base_backoff(20) == 300.0


# ── pre-filter: junk never reaches dossier work ───────────────────────

def test_prefilter_decisions():
    pf = Prefilter(watched_devs={CREATOR}, watched_mints=set())
    assert pf.decide({"mint": "X", "creator": CREATOR}) == "dossier"
    assert pf.decide({"mint": "Y", "creator": "Stranger"}) == "ledger"


def test_desk_ingest_junk_stays_ledger_only(tmp_path):
    reg = SourceRegistry()
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), registry=reg)
    for i in range(100):
        desk._ingest("launch", {
            "mint": chain.b58encode(bytes([i % 250 + 1]) * 32),
            "creator": "StrangerDev", "name": "junk", "symbol": "J",
            "source": "test", "ts": time.time()})
    assert len(desk.store.launches()) == 100
    assert desk.store.dossiers() == {}     # zero dossier work happened


def test_frontend_candidate_gate():
    row = {"mint": MINT, "creator": CREATOR, "name": "G", "symbol": "G",
           "virtual_token_reserves": INIT_VT,
           "virtual_sol_reserves": INIT_VS,
           "real_token_reserves": INIT_RT // 2,
           "real_sol_reserves": 40_000_000_000,
           "total_supply": SUPPLY, "complete": False,
           "bonding_curve": CURVE}
    cand = frontend_candidate(row)
    assert cand is not None and cand["curve_pct_seen"] == pytest.approx(0.5)
    row2 = dict(row, real_token_reserves=int(INIT_RT * 0.97))
    assert frontend_candidate(row2) is None    # 3% sold: no dossier


# ── jobs + tool registration ──────────────────────────────────────────

class FakeScheduler:
    def __init__(self):
        self.jobs = []

    def list_jobs(self):
        return list(self.jobs)

    def add(self, name, spec, kind, payload, routine=False):
        job = {"name": name, "id": f"j{len(self.jobs)}", "routine": routine}
        self.jobs.append(job)
        return job


def test_jobs_register_idempotent():
    from nomorals.agents.pumpdesk.jobs import ensure_pumpdesk_jobs
    sch = FakeScheduler()
    out = ensure_pumpdesk_jobs(sch)
    assert len(sch.jobs) == 2
    assert all(j["routine"] for j in sch.jobs)
    out2 = ensure_pumpdesk_jobs(sch)
    assert len(sch.jobs) == 2
    assert all(o.get("already_scheduled") for o in out2)


def test_tool_module_registered():
    from nomorals.tools.agents import AGENT_TOOL_MODULES
    assert "pumpdesk" in AGENT_TOOL_MODULES
    import nomorals.agents.pumpdesk as pkg
    captured = {}

    class FakeRegistry:
        context = None

        def register(self, name, **kw):
            def deco(fn):
                captured[name] = fn
                return fn
            return deco

    pkg.register(FakeRegistry())
    assert "pumpdesk" in captured
    res = captured["pumpdesk"](action="status")
    assert "launches_seen" in res and "execution" in res


# ── websocket client against a local server ───────────────────────────

def _ws_server_once(handler):
    """Run handler(conn) for one accepted connection in a thread."""
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        conn, _ = srv.accept()
        try:
            handler(conn)
        finally:
            conn.close()
            srv.close()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return port, t


def _server_handshake(conn):
    import hashlib
    req = b""
    while b"\r\n\r\n" not in req:
        req += conn.recv(4096)
    key = [ln.split(":", 1)[1].strip() for ln in req.decode().split("\r\n")
           if ln.lower().startswith("sec-websocket-key")][0]
    accept = base64.b64encode(
        hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-"
                             "C5AB0DC85B11").encode()).digest()).decode()
    conn.sendall(f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: "
                 f"websocket\r\nConnection: Upgrade\r\n"
                 f"Sec-WebSocket-Accept: {accept}\r\n\r\n".encode())


def _server_read_frame(conn):
    data = conn.recv(4096)
    b1 = data[1]
    ln = b1 & 0x7F
    off = 2
    if ln == 126:
        ln = struct.unpack("!H", data[2:4])[0]
        off = 4
    mask = data[off:off + 4]
    off += 4
    payload = bytes(b ^ mask[i % 4] for i, b in enumerate(data[off:off + ln]))
    return data[0] & 0x0F, payload


def _server_send_text(conn, text, fragment=False):
    data = text.encode()
    if fragment and len(data) > 4:
        mid = len(data) // 2
        conn.sendall(struct.pack("!BB", 0x01, mid) + data[:mid])
        conn.sendall(struct.pack("!BB", 0x80, len(data) - mid)
                     + data[mid:])
        return
    conn.sendall(struct.pack("!BB", 0x81, len(data)) + data)


def test_wsclient_roundtrip_fragment_and_pong(monkeypatch):
    monkeypatch.delenv("https_proxy", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.delenv("all_proxy", raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    seen = {}

    def handler(conn):
        conn.settimeout(5)
        _server_handshake(conn)
        op, payload = _server_read_frame(conn)     # client's "hello"
        seen["client_msg"] = payload.decode()
        _server_send_text(conn, "frag-mented reply", fragment=True)
        conn.sendall(struct.pack("!BB", 0x89, 0))  # server pings
        op2, _ = _server_read_frame(conn)          # client's auto-pong
        seen["pong_op"] = op2
        time.sleep(0.2)

    port, t = _ws_server_once(handler)
    from nomorals.agents.pumpdesk.wsclient import WebSocketClient
    ws = WebSocketClient(f"ws://127.0.0.1:{port}/data", timeout=5)
    ws.connect()
    ws.send_text("hello")
    msg = ws.recv(timeout=5)
    assert msg == "frag-mented reply"
    # the next read consumes the server's ping, answers with a pong,
    # then blocks for more data — pong is sent before the block ends
    from nomorals.agents.pumpdesk.wsclient import WSClosed
    with pytest.raises((TimeoutError, WSClosed)):
        ws.recv(timeout=2)
    ws.close()
    t.join(timeout=5)
    assert seen["client_msg"] == "hello"
    assert seen["pong_op"] == 0xA


# ── firehose slice over a fake PumpPortal transport ───────────────────

class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def connect(self):
        return self

    def send_text(self, text):
        self.sent.append(text)

    def ping(self):
        pass

    def recv(self, timeout=None):
        if self.messages:
            time.sleep(0.01)
            return self.messages.pop(0)
        time.sleep(0.05)
        raise TimeoutError("no more scripted messages")

    def close(self):
        pass


def test_firehose_slice_consumes_create_events():
    create = json.dumps({
        "txType": "create", "mint": MINT, "traderPublicKey": CREATOR,
        "name": "Gold", "symbol": "GOLD", "uri": "https://ipfs/x",
        "bondingCurveKey": CURVE, "signature": "sig1", "initialBuy": 0,
        "solAmount": 0})
    ack = json.dumps({"message": "Successfully subscribed"})
    fake = FakeWS([ack, create])
    events = []
    fh = Firehose(lambda k, p: events.append((k, p)),
                  registry=None,
                  client_factory=lambda url: fake)
    rep = fh.run_slice(seconds=1.0)
    assert rep["ok"] is True and rep["backend"] == "pumpportal"
    launches = [p for k, p in events if k == "launch"]
    assert len(launches) == 1
    assert launches[0]["mint"] == MINT
    assert launches[0]["creator"] == CREATOR
    subs = [json.loads(s)["method"] for s in fake.sent]
    assert subs == ["subscribeNewToken", "subscribeMigration"]


def test_rpc_backend_decodes_create_from_logs():
    def _s(x):
        b = x.encode()
        return struct.pack("<I", len(b)) + b
    payload = chain.event_discriminator("CreateEvent")
    payload += _s("Gold") + _s("GOLD") + _s("uri")
    payload += chain.b58decode(MINT) + chain.b58decode(CURVE)
    payload += bytes(range(1, 33))
    notif = json.dumps({
        "jsonrpc": "2.0", "method": "logsNotification",
        "params": {"subscription": 7, "result": {
            "context": {"slot": 4242},
            "value": {"signature": "sigX", "err": None,
                      "logs": ["Program data: "
                               + base64.b64encode(payload).decode()]}}}})
    suback = json.dumps({"jsonrpc": "2.0", "id": 1, "result": 7})
    fake = FakeWS([suback, notif])
    backend = firehose.RpcLogBackend("https://api.mainnet-beta.solana.com",
                                     client_factory=lambda url: fake)
    events = []
    ctl = firehose._Control()
    timer = threading.Timer(1.0, ctl.stop)
    timer.start()
    backend.serve(lambda k, p: events.append((k, p)), ctl)
    timer.cancel()
    launches = [p for k, p in events if k == "launch"]
    assert len(launches) == 1
    assert launches[0]["mint"] == MINT and launches[0]["slot"] == 4242
    assert launches[0]["creator"] == CREATOR


# ── execution boundary audit ──────────────────────────────────────────

def test_no_execution_machinery_in_package():
    import nomorals.agents.pumpdesk as pkg
    root = os.path.dirname(pkg.__file__)
    forbidden = ("Keypair", "sign_transaction", "send_transaction",
                 "sendTransaction", "from_private_key", "private_key_bytes")
    offenders = []
    for name in os.listdir(root):
        if name.endswith(".py"):
            text = open(os.path.join(root, name)).read()
            for token in forbidden:
                if token in text:
                    offenders.append((name, token))
    assert offenders == []
