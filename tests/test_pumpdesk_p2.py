"""Pump desk P2 — smart-money graph, attention verifier, learning
loop, Library shelving.

Fixtures are offline trade streams with KNOWN realized outcomes:
the smart wallet's wins and the loser wallet's losses are built
into the fixture arithmetic, so the graph's grades are checked
against ground truth, not against the code's own hopes.  The
reserve-divergence fixture uses the desk's live finding verbatim:
703 SOL of virtual reserves against 0 real SOL.
"""

from __future__ import annotations

import time

import pytest

from nomorals.agents.pumpdesk import attention, graph, learn, score
from nomorals.agents.pumpdesk.calls import CallEngine, PumpCall
from nomorals.agents.pumpdesk.chain import BondingCurveState, MintInfo
from nomorals.agents.pumpdesk.desk import PumpDesk
from nomorals.agents.pumpdesk.dossier import build_dossier
from nomorals.agents.pumpdesk.shelf import search_shelf, shelve_notable
from nomorals.agents.pumpdesk.store import PumpStore

SOL = 1_000_000_000
SMART = "SmartWallet1111111111111111111111111111111"
LOSER = "LoserWallet2222222222222222222222222222222"
GHOST = "GhostWallet3333333333333333333333333333333"
COHORT_A = "CohortA44444444444444444444444444444444444"
COHORT_B = "CohortB55555555555555555555555555555555555"
CREATOR = "Creator66666666666666666666666666666666666"
BASE_TS = 1_700_000_000.0


def _trade(user, mint, is_buy, sol, toks, ts, slot=5000):
    return {"user": user, "mint": mint, "is_buy": is_buy,
            "sol_amount": int(sol * SOL), "token_amount": int(toks),
            "timestamp": ts, "slot": slot}


def _mint(i: int) -> str:
    return f"Mint{i:039d}"


def _seed_wallet_history(store):
    """SMART closes 4 winners; LOSER closes 4 losers; both fully
    exit every position (FIFO ground truth)."""
    ts = BASE_TS
    for i in range(4):
        m = _mint(100 + i)
        # buy 10 SOL worth, sell for 16 SOL: +6 realized (distinct
        # slots per wallet — these two are independent actors, not
        # a co-entry cohort)
        store.record_trade(_trade(SMART, m, True, 10, 1_000_000, ts,
                                  slot=5000))
        store.record_trade(_trade(SMART, m, False, 16, 1_000_000,
                                  ts + 60, slot=5000))
        # buy 10 SOL worth, sell for 6 SOL: -4 realized
        store.record_trade(_trade(LOSER, m, True, 10, 1_000_000, ts,
                                  slot=5001))
        store.record_trade(_trade(LOSER, m, False, 6, 1_000_000,
                                  ts + 60, slot=5001))
        ts += 3600


# ── graph: realized PnL grading ───────────────────────────────────────

def test_graph_grades_wallets_by_realized_pnl(tmp_path):
    store = PumpStore(str(tmp_path))
    _seed_wallet_history(store)
    g = graph.build_wallet_graph(store, now=BASE_TS + 100000)
    smart = g["wallets"][SMART]
    assert smart["closed_trades"] == 4
    assert smart["realized_sol"] == pytest.approx(24.0)
    assert smart["realized_recent_sol"] == pytest.approx(24.0)
    assert smart["win_rate"] == 1.0
    assert smart["verdict"] == "smart"
    assert smart["weight"] > 0.5
    loser = g["wallets"][LOSER]
    assert loser["realized_sol"] == pytest.approx(-16.0)
    assert loser["verdict"] == "losing"
    assert loser["weight"] == 0.0


def test_graph_unknown_wallet_has_no_record(tmp_path):
    store = PumpStore(str(tmp_path))
    _seed_wallet_history(store)
    graph.build_wallet_graph(store)
    assert graph.wallet_record(store, GHOST) is None
    rendered = graph.render_wallet(None, GHOST)
    assert "no record" in rendered


def test_graph_thin_wallet_stays_unknown(tmp_path):
    store = PumpStore(str(tmp_path))
    m = _mint(7)
    store.record_trade(_trade(SMART, m, True, 5, 500_000, BASE_TS))
    store.record_trade(_trade(SMART, m, False, 9, 500_000, BASE_TS + 30))
    g = graph.build_wallet_graph(store)
    rec = g["wallets"][SMART]
    assert rec["closed_trades"] == 1
    assert rec["verdict"] == "unknown"     # below the floor: no assertion
    assert rec["weight"] == 0.0


def test_graph_basis_unknown_sells_excluded_from_pnl(tmp_path):
    store = PumpStore(str(tmp_path))
    m = _mint(9)
    # sells tokens never bought in the observed stream
    store.record_trade(_trade(SMART, m, False, 40, 2_000_000, BASE_TS))
    g = graph.build_wallet_graph(store)
    rec = g["wallets"][SMART]
    assert rec["basis_unknown_sells"] == 1
    assert rec["realized_sol"] == 0.0      # proceeds are NOT profit


# ── graph: cohorts + confluence ───────────────────────────────────────

def _seed_cohort(store):
    # A and B buy the same mints in the same slots, twice over
    for i, slot in ((201, 7000), (202, 9000)):
        m = _mint(i)
        store.record_trade(_trade(COHORT_A, m, True, 3, 300_000,
                                  BASE_TS + slot, slot=slot))
        store.record_trade(_trade(COHORT_B, m, True, 3, 300_000,
                                  BASE_TS + slot, slot=slot))
    # both are individually profitable closers elsewhere (distinct
    # slots per wallet — these are solo trades, not co-entries)
    ts = BASE_TS + 20000
    for w, slot in ((COHORT_A, 5000), (COHORT_B, 6000)):
        for i in range(3):
            m = _mint(300 + i)
            store.record_trade(_trade(w, m, True, 10, 1_000_000, ts,
                                      slot=slot))
            store.record_trade(_trade(w, m, False, 15, 1_000_000,
                                      ts + 60, slot=slot))
            ts += 3600


def test_graph_clusters_repeated_coentry(tmp_path):
    store = PumpStore(str(tmp_path))
    _seed_cohort(store)
    g = graph.build_wallet_graph(store)
    assert len(g["clusters"]) == 1
    members = set(g["clusters"][0]["members"])
    assert members == {COHORT_A, COHORT_B}
    ev = g["clusters"][0]["evidence"]
    assert ev["shared_launches"] == 2
    assert "unknown" in ev["funding"]      # no funding link claimed


def test_confluence_counts_cohort_once(tmp_path):
    store = PumpStore(str(tmp_path))
    _seed_cohort(store)
    g = graph.build_wallet_graph(store)
    target = _mint(201)
    conf = graph.graph_confluence(store, target, graph=g)
    # both cohort members bought, but linked wallets are ONE entity
    assert conf["hits"] == 1
    weights = sorted(e["weight"] for e in conf["entries"])
    assert weights[0] == 0.0 and weights[1] > 0.0


def test_confluence_ignores_losing_and_unknown_wallets(tmp_path):
    store = PumpStore(str(tmp_path))
    _seed_wallet_history(store)
    g = graph.build_wallet_graph(store)
    m = _mint(100)
    conf = graph.graph_confluence(store, m, graph=g)
    entries = {e["wallet"]: e for e in conf["entries"]}
    assert entries[SMART]["weight"] > 0.0
    assert entries[LOSER]["weight"] == 0.0
    assert conf["hits"] == 1


# ── reserve divergence: the live pattern as fixture ──────────────────

class FakeRPC:
    def __init__(self, curve=None, mint_info=None, pool=False):
        self._curve = curve
        self._mint_info = mint_info
        self._pool = pool

    def get_curve(self, mint):
        return self._curve

    def get_mint_info(self, mint):
        return self._mint_info

    def get_token_largest(self, mint):
        return []

    def get_account(self, pubkey):
        return b"\x01" * 100 if self._pool else None

    def get_signatures(self, address, limit=50, before=""):
        return []

    def get_transaction(self, signature):
        return None


def test_divergence_computed_from_live_pattern(tmp_path):
    store = PumpStore(str(tmp_path))
    mint = _mint(42)
    store.record_launch({"mint": mint, "creator": CREATOR, "slot": 5,
                         "name": "Phantom", "symbol": "PHNT",
                         "ts": time.time(), "source": "test"})
    # the live finding: 703 SOL virtual, 0 SOL real
    curve = BondingCurveState(
        virtual_token_reserves=400_000_000_000_000,
        virtual_sol_reserves=703 * SOL,
        real_token_reserves=393_100_000_000_000,
        real_sol_reserves=0,
        token_total_supply=1_000_000_000_000_000, creator=CREATOR)
    info = MintInfo(supply=1_000_000_000_000_000, decimals=6,
                    initialized=True)
    d = build_dossier(store, FakeRPC(curve, info), mint)
    assert d["curve"]["virtual_sol"] == pytest.approx(703.0)
    assert d["curve"]["real_sol"] == 0.0
    assert d["curve"]["reserve_divergence"] == pytest.approx(
        (703 - 30) / 703, abs=1e-3)


def _dossier_dict(**over):
    d = {
        "mint": _mint(42), "symbol": "PHNT", "name": "Phantom",
        "curve": {"curve_pct": 0.5, "curve_pct_basis": "supply-split",
                  "market_cap_sol": 200.0,
                  "price_sol": 2e-7, "real_sol": 0.0,
                  "virtual_sol": 703.0,
                  "reserve_divergence": (703 - 30) / 703,
                  "complete_flag": False, "quote_mint": "SOL",
                  "state": {"virtual_sol_reserves": 703 * SOL,
                            "real_sol_reserves": 0}},
        "authorities": {"mint_authority_live": False,
                        "freeze_authority_live": False},
        "holders": {"top1_share": 0.05, "top3_share": 0.12,
                    "top10_share": 0.3, "accounts_seen": 20},
        "dev": {"creator": CREATOR, "launches": 5, "known_outcomes": 5,
                "rug_ratio": 0.0, "graduation_rate": 0.2,
                "serial_rugger": False},
        "trades": {"trades": 40, "unique_buyers": 20, "buy_sol": 30.0,
                   "sell_sol": 10.0, "buy_sell_ratio": 3.0,
                   "velocity_sol_per_min": 1.5, "sol_per_trade": 0.75,
                   "block0_absorption_share": None,
                   "block0_total_share": None, "dev_sold": False,
                   "dev_sold_sol": 0.0, "bump_alpha_max": 0.0,
                   "bump_wallets": 0},
        "gaps": [],
    }
    d.update(over)
    return d


def test_divergence_scores_as_manipulation():
    v = score.evaluate(_dossier_dict())
    assert v.manipulation >= score.MANIP_GATE
    assert v.verdict == "NO_CALL"
    assert any("reserve divergence" in r for r in v.reasons)


def test_healthy_reserves_do_not_trip_divergence():
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], virtual_sol=100.0, real_sol=68.0,
                      reserve_divergence=(100 - 30 - 68) / 100,
                      state={"virtual_sol_reserves": 100 * SOL,
                             "real_sol_reserves": 68 * SOL})
    v = score.evaluate(d)
    assert not any("reserve divergence" in r for r in v.reasons)


# ── attention verifier ────────────────────────────────────────────────

def _search_factory(rows):
    def _search(query, limit=10):
        return list(rows)
    return _search


def _mention(i, snippet, domain="example.com", ts=None):
    return {"title": f"post {i}", "snippet": snippet,
            "url": f"https://{domain}/p{i}", "source": "web",
            "timestamp": ts}


def _botnet_dossier():
    """The manufactured-attention target: heavy templated noise
    against a 2-buyer on-chain base."""
    d = _dossier_dict()
    d["trades"] = dict(d["trades"], unique_buyers=2)
    return d


def test_attention_manufactured_templated_botnet():
    rows = [_mention(i, "PHNT to the moon!! buy now apes this is the one")
            for i in range(10)]
    att = attention.verify_attention(_botnet_dossier(),
                                     search_fn=_search_factory(rows))
    assert att["verdict"] == attention.MANUFACTURED
    assert att["quality"] == pytest.approx(0.1)
    assert att["evidence"]["templated_ratio"] >= 0.5
    assert att["evidence"]["manufactured_points"]


def test_attention_organic_varied_sources():
    rows = [
        _mention(1, "looked at the curve, dev has two prior grads",
                 "x.com", BASE_TS),
        _mention(2, "holder count climbing steadily since launch",
                 "telegram.me", BASE_TS + 600),
        _mention(3, "chart review: momentum building on real volume",
                 "dextools.io", BASE_TS + 1200),
        _mention(4, "community call notes and risk discussion thread",
                 "reddit.com", BASE_TS + 1800),
    ]
    att = attention.verify_attention(_dossier_dict(),
                                     search_fn=_search_factory(rows))
    assert att["verdict"] == attention.ORGANIC
    assert att["quality"] >= 0.8


def test_attention_fail_closed_on_no_evidence():
    att = attention.verify_attention(_dossier_dict(),
                                     search_fn=_search_factory([]))
    assert att["verdict"] == attention.UNKNOWN
    assert att["quality"] is None
    assert att["gaps"]
    # and UNKNOWN has no score effect
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], virtual_sol=100.0, real_sol=68.0,
                      reserve_divergence=0.02,
                      state={"virtual_sol_reserves": 100 * SOL,
                             "real_sol_reserves": 68 * SOL})
    base = score.evaluate(d)
    d2 = dict(d, attention=att)
    with_att = score.evaluate(d2)
    assert with_att.manipulation == base.manipulation


def test_attention_fail_closed_on_search_error():
    def boom(query, limit=10):
        raise RuntimeError("search backend down")
    att = attention.verify_attention(_dossier_dict(), search_fn=boom)
    assert att["verdict"] == attention.UNKNOWN
    assert any("failed" in g for g in att["gaps"])


def test_attention_manufactured_raises_manipulation_in_score():
    rows = [_mention(i, "PHNT to the moon!! buy now apes this is the one")
            for i in range(10)]
    att = attention.verify_attention(_botnet_dossier(),
                                     search_fn=_search_factory(rows))
    assert att["verdict"] == attention.MANUFACTURED
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], virtual_sol=100.0, real_sol=68.0,
                      reserve_divergence=0.02,
                      state={"virtual_sol_reserves": 100 * SOL,
                             "real_sol_reserves": 68 * SOL})
    d["attention"] = att
    v = score.evaluate(d)
    assert v.manipulation >= 0.55
    assert any("manufactured attention" in r for r in v.reasons)


def test_attention_recycled_uri_flagged(tmp_path):
    store = PumpStore(str(tmp_path))
    other = _mint(77)
    store.record_launch({"mint": other, "creator": "SomeoneElse",
                         "uri": "ipfs://same-metadata", "ts": BASE_TS,
                         "source": "test"})
    d = _dossier_dict(uri="ipfs://same-metadata")
    rows = [_mention(1, "a perfectly ordinary varied mention",
                     "x.com", BASE_TS)]
    att = attention.verify_attention(d, search_fn=_search_factory(rows),
                                     store=store)
    assert att["evidence"]["recycled_socials"] is True
    assert any("recycled" in p
               for p in att["evidence"]["manufactured_points"])


# ── score: graph confluence path ──────────────────────────────────────

def test_score_reads_graph_confluence_not_labels():
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], virtual_sol=100.0, real_sol=68.0,
                      reserve_divergence=0.02,
                      state={"virtual_sol_reserves": 100 * SOL,
                             "real_sol_reserves": 68 * SOL})
    d["smart"] = {"weighted": 0.9, "hits": 3, "buyers_seen": 12,
                  "entries": []}
    v = score.evaluate(d)
    assert v.smart_money == pytest.approx(0.9)
    assert any("graph-graded" in r for r in v.reasons)


# ── learning loop ─────────────────────────────────────────────────────

def _graded_call(i, win, momentum, smart_m, rug=0.1, manip=0.1):
    return PumpCall(
        mint=_mint(500 + i), published_at=BASE_TS,
        expires_at=BASE_TS + 100, entry_mc_sol=30.0,
        target_mc_sol=60.0, invalidation_mc_sol=16.5,
        scored=win, outcome="win" if win else "loss",
        components={"momentum": momentum, "smart_money": smart_m,
                    "rug_risk": rug, "manipulation": manip},
        call_id=f"c{i}")


def _journal_split():
    """15 winners (high momentum, low smart) + 9 losers (low
    momentum, high smart, riskier devs): a threshold step separates
    them cleanly."""
    calls = []
    for i in range(15):
        calls.append(_graded_call(i, True, 0.9, 0.2, rug=0.05))
    for i in range(15, 24):
        calls.append(_graded_call(i, False, 0.5, 0.9, rug=0.6))
    return calls


def test_learning_adjusts_with_journal_and_version(tmp_path):
    store = PumpStore(str(tmp_path))
    rec = learn.recalibrate(store, _journal_split(), now=BASE_TS)
    assert rec["status"] == "ADJUSTED", rec
    assert rec["from_version"] == 1 and rec["to_version"] == 2
    assert rec["backtest"]["hit_rate"] == 1.0
    assert rec["gain"] >= learn.MIN_GAIN
    saved = learn.current_params(store)
    assert saved["version"] == 2
    events = store.param_events()
    assert len(events) == 1 and events[0]["status"] == "ADJUSTED"
    assert events[0]["before"] and events[0]["after"]
    assert "in-sample" in events[0]["evidence"]["basis"]


def test_learning_refuses_below_sample_floor(tmp_path):
    store = PumpStore(str(tmp_path))
    calls = [_graded_call(i, True, 0.9, 0.2) for i in range(5)]
    rec = learn.recalibrate(store, calls, now=BASE_TS)
    assert rec["status"] == "INSUFFICIENT_DATA"
    assert learn.current_params(store)["version"] == 1
    assert store.param_events()[0]["status"] == "INSUFFICIENT_DATA"


def test_learning_excludes_calls_without_components(tmp_path):
    store = PumpStore(str(tmp_path))
    calls = []
    for i in range(24):
        c = _graded_call(i, True, 0.9, 0.2)
        c.components = {}
        calls.append(c)
    rec = learn.recalibrate(store, calls, now=BASE_TS)
    assert rec["status"] == "INSUFFICIENT_DATA"
    assert rec["evidence"]["graded_excluded_no_components"] == 24


def test_learning_holds_when_no_step_helps(tmp_path):
    store = PumpStore(str(tmp_path))
    calls = [_graded_call(i, True, 0.9, 0.2, rug=0.05)
             for i in range(24)]
    rec = learn.recalibrate(store, calls, now=BASE_TS)
    assert rec["status"] == "HOLD"
    assert learn.current_params(store)["version"] == 1


def test_score_honors_recalibrated_params(tmp_path):
    store = PumpStore(str(tmp_path))
    learn.recalibrate(store, _journal_split(), now=BASE_TS)
    params = learn.current_params(store)
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], virtual_sol=100.0, real_sol=68.0,
                      reserve_divergence=0.02,
                      state={"virtual_sol_reserves": 100 * SOL,
                             "real_sol_reserves": 68 * SOL})
    v_new = score.evaluate(d, params=params)
    v_old = score.evaluate(d)
    # the recalibrated version differs from v1 and is the one the
    # desk passes in — composite recomputed from the new weights
    assert params["version"] == 2
    assert v_new.composite == pytest.approx(
        params["weights"]["momentum"] * v_new.momentum
        + params["weights"]["smart"] * v_new.smart_money
        + params["weights"]["safety_rug"] * (1 - v_new.rug_risk)
        + params["weights"]["safety_manip"] * (1 - v_new.manipulation),
        abs=1e-3)
    assert v_old.composite >= 0  # v1 path still computable


# ── shelving ──────────────────────────────────────────────────────────

def test_shelving_graduated_dossier_and_retrieval(tmp_path):
    store = PumpStore(str(tmp_path))
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], graduated=True)
    store.save_dossier(d)
    out = shelve_notable(store, now=BASE_TS)
    assert out["dossiers"] == 1
    items = store.knowledge()
    assert len(items) == 1
    item = items[0]
    assert item["kind"] == "pump-dossier"
    assert item["source"]["type"] == "pumpdesk"
    assert item["source"]["ref"] == d["mint"]
    hits = search_shelf(store, "PHNT")
    assert len(hits) == 1
    assert search_shelf(store, "nonexistent-token") == []


def test_shelving_skips_ordinary_dossiers(tmp_path):
    store = PumpStore(str(tmp_path))
    d = _dossier_dict()          # not called, not graduated, no verdict
    store.save_dossier(d)
    out = shelve_notable(store, now=BASE_TS)
    assert out["dossiers"] == 0
    assert store.knowledge() == []


# ── desk integration: study cycle + views ─────────────────────────────

def test_study_cycle_builds_graph_learns_and_shelves(tmp_path):
    store = PumpStore(str(tmp_path))
    _seed_wallet_history(store)
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], graduated=True)
    store.save_dossier(d)
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), store=store,
                    registry=None)
    rep = desk.study_cycle()
    assert rep["graph"]["wallets"] >= 2
    assert rep["graph"]["graded"] == 2
    assert rep["learning"]["status"] == "INSUFFICIENT_DATA"
    assert rep["shelved"]["dossiers"] == 1
    wv = desk.wallet_view(SMART)
    assert wv["record"]["verdict"] == "smart"
    assert "realized PnL" in wv["rendered"]
    gv = desk.graph_view()
    assert "smart-money graph" in gv["rendered"]
    lv = desk.learn_view()
    assert lv["params"]["version"] == 1
    sv = desk.shelf_view("PHNT")
    assert len(sv["hits"]) == 1
    st = desk.status()
    assert st["graph"]["smart"] == 1
    assert st["shelf_items"] >= 1
    assert st["learning"]["params"]["version"] == 1


def test_desk_attention_view_fail_closed(tmp_path):
    store = PumpStore(str(tmp_path))
    d = _dossier_dict()
    store.save_dossier(d)
    desk = PumpDesk(context=None, desk_dir=str(tmp_path), store=store,
                    registry=None,
                    attention_search=lambda q, limit=10: [])
    av = desk.attention_view(d["mint"])
    assert av["attention"]["verdict"] == "UNKNOWN"
    assert "gap" in av["rendered"]


def test_call_carries_components(tmp_path):
    engine = CallEngine(str(tmp_path))
    d = _dossier_dict()
    d["curve"] = dict(d["curve"], virtual_sol=100.0, real_sol=68.0,
                      reserve_divergence=0.02,
                      state={"virtual_sol_reserves": 100 * SOL,
                             "real_sol_reserves": 68 * SOL})
    v = score.evaluate(d)
    call = engine.build_call(d, v, now=BASE_TS)
    assert call is not None
    assert call.components["momentum"] == pytest.approx(v.momentum)
    assert call.components["rug_risk"] == pytest.approx(v.rug_risk)


def test_tool_actions_p2(tmp_path, monkeypatch):
    monkeypatch.setenv("NOMORALS_PUMPDESK_DIR", str(tmp_path))
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
    fn = captured["pumpdesk"]
    assert "error" in fn(action="wallet")
    res = fn(action="wallet", wallet=GHOST)
    assert res["record"] is None and "no record" in res["rendered"]
    assert "smart-money graph" in fn(action="graph")["rendered"]
    assert fn(action="learn")["params"]["version"] == 1
    assert "error" in fn(action="shelf")
    assert fn(action="shelf", query="zzz")["hits"] == []
