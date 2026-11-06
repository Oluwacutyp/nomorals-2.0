"""Pass 4 of Devon's trading system: demo execution + management.

The chain under test (agents/execution.py over finance/trading_desk.py):
triggered setup + judgment ACT -> hard invariants (outside/above
judgment) -> doctrine sizing -> demo order (paper simulator, or the
Exness demo rung) -> per-playbook management -> everything journaled
to the Pass 1 trading journal.

Every test here is a NEW-BEHAVIOR test: on pristine 7c996e3 the
engine does not exist (the file fails at import), the desk's order
path has no idea gate / paper_modify / paper_close_partial, and the
MT5 adapter module is absent. Passes 1-3 behavior is carried by
their own suites + neighborhoods.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nomorals.agents import execution as ex
from nomorals.connectors import mt5_bridge
from nomorals.finance import trading_journal
from nomorals.finance.trading_desk import DeskError, TradingDesk
from nomorals.ta.analyst import TradeIdea

BASE = 1762362000.0  # 2025-11-05 17:00 UTC


# ── fakes ────────────────────────────────────────────────────────────

class FakeConnector:
    """The desk's connector protocol, in memory."""

    def __init__(self, *, equity: float = 10000.0, demo: bool = True,
                 positions: list | None = None,
                 deals: list | None = None) -> None:
        self.equity = equity
        self.demo = demo
        self.positions = positions or []
        self.deals = deals or []
        self.open_calls: list[dict[str, Any]] = []
        self.close_calls: list[dict[str, Any]] = []

    def get_snapshot(self) -> dict[str, Any]:
        return {"account_state": {"equity": self.equity,
                                  "balance": self.equity},
                "positions": list(self.positions)}

    def get_account_details(self) -> dict[str, Any]:
        return {"account": "demo trading account" if self.demo
                else "real trading account"}

    def get_candles(self, instrument: str, timeframe: str = "H1",
                    count: int = 200) -> list[dict[str, Any]]:
        return [{"o": 2000.0, "h": 2001.0, "l": 1999.0, "c": 2000.0}
                for _ in range(min(count, 5))]

    def get_instrument_condition(self, instrument: str) -> dict[str, Any]:
        return {"contract_size": 100, "quote_currency": "USD",
                "volume_min": 0.01, "volume_max": 100.0,
                "volume_step": 0.01}

    def open_position(self, instrument: str, side: str, volume: float,
                      **kw: Any) -> dict[str, Any]:
        self.open_calls.append({"instrument": instrument, "side": side,
                                "volume": volume, **kw})
        pid = f"ex_{len(self.open_calls)}"
        self.positions.append({"id": pid})
        return {"position_id": pid, "price": 2000.0}

    def close_position(self, position_id: str, **kw: Any) -> dict[str, Any]:
        self.close_calls.append({"position_id": position_id, **kw})
        self.positions = [p for p in self.positions
                          if p.get("id") != position_id]
        return {"closed": True}

    def deals_history(self, *, limit: int = 50, **filters: Any
                      ) -> dict[str, Any]:
        return {"deals": list(self.deals)}


def _quote(price: float = 2000.0, spread: float = 0.2):
    def fn(symbol: str, market: str) -> dict[str, Any]:
        return {"price": price, "bid": price - spread / 2,
                "ask": price + spread / 2, "source": "fake"}
    return fn


def _frame(closes: list[float], *, start: float = BASE - 25 * 300,
           step: int = 300, half_range: float = 0.1) -> pd.DataFrame:
    idx = [pd.Timestamp(start + i * step, unit="s", tz="UTC")
           for i in range(len(closes))]
    rows = [(c, c + half_range, c - half_range, c, 100.0)
            for c in closes]
    return pd.DataFrame(rows, columns=["open", "high", "low", "close",
                                       "volume"], index=idx)


def _engine(tmp_path: Path, connector: FakeConnector, *,
            mode: str = "paper", sizing: ex.SizingConfig | None = None,
            quote: Any = None, bars: Any = None,
            now: float = BASE) -> ex.ExecutionEngine:
    desk = TradingDesk(
        connector, policy=ex.system_policy(), mode=mode,
        journal_path=tmp_path / "desk.jsonl",
        state_path=tmp_path / "desk_state.json",
        now=lambda: now)
    return ex.ExecutionEngine(
        desk, journal_path=tmp_path / "sys.jsonl", sizing=sizing,
        quote_fn=quote or _quote(), bars_fn=bars,
        now=lambda: now)


def _setup(*, sid: str = "qm_m5_native:XAUUSD:k1",
           playbook: str = "qm_m5_native", family: str = "reversal",
           tf: str = "5m", direction: int = 1,
           entry: float = 2000.0, stop: float = 1990.0,
           targets: tuple = (2030.0,), zone: tuple = (2000.0, 1995.0),
           atr: float = 4.0) -> dict[str, Any]:
    return {
        "setup_id": sid, "symbol": "XAUUSD", "market": "forex",
        "playbook": playbook, "route": playbook, "family": family,
        "entry_timeframe": tf, "direction": direction,
        "pattern_key": "k1", "state": "triggered",
        "plan": {"entry_zone": {"top": zone[0], "bottom": zone[1]},
                 "entry_ref": entry, "invalidation": stop,
                 "targets": list(targets)},
        "evidence": {"atr": atr},
        "trigger": {"bar_ts": BASE, "price": entry},
        "read_taken_at": BASE}


def _act(*, confidence: float = 0.5) -> dict[str, Any]:
    return {"decision": "act", "reason_category": "quality",
            "chosen_scenario": "continuation", "expected_r": 0.4,
            "confidence": confidence,
            "why": "continuation wins (p=0.52)",
            "market_state": {"structure_class": "trend"},
            "subject": {"type": "setup", "state": "triggered"}}


def _events(engine: ex.ExecutionEngine, kind: str) -> list[dict[str, Any]]:
    return trading_journal.read_events(
        limit=0, event=kind, path=engine.journal_path)


# ── the full chain + doctrine sizing ─────────────────────────────────

def test_full_chain_opens_paper_position_with_doctrine_sizing(tmp_path):
    conn = FakeConnector(equity=10000.0)
    engine = _engine(tmp_path, conn)
    res = engine.execute_setup(_setup(), _act(confidence=0.5))
    assert res["ok"] is True
    pos = res["position"]
    # Variant A: 10000/1000 x 0.01 x (1 + 1x0.5) = 0.15 lots.
    assert pos["volume"] == 0.15
    assert pos["sizing"]["risk_amount"] == 150.0  # 10.0 stop x 100 x 0.15
    assert pos["sizing"]["floored_to_min"] is False
    assert pos["profile"] == "qm_fixed_bracket"
    assert pos["rung"] == "paper"
    assert pos["stop_loss"] == 1990.0 and pos["targets"] == [2030.0]
    # The desk really holds the simulated position, bracket and all.
    desk_pos = engine.desk.paper_positions()
    assert len(desk_pos) == 1 and desk_pos[0].volume == 0.15
    assert desk_pos[0].stop_loss == 1990.0
    assert desk_pos[0].take_profit == 2030.0
    # Journal order: intent lands BEFORE the fill record.
    raw = [e["event"] for e in reversed(
        trading_journal.read_events(limit=0, path=engine.journal_path))]
    assert raw.index("order_intent") < raw.index("position_opened")


def test_doctrine_size_floors_to_minimum_and_rounds_down():
    # Below the minimum: traded at the minimum, never abandoned.
    out = ex.doctrine_size(equity=400.0, entry=2000.0,
                           stop_loss=1900.0, contract_value=100.0,
                           confidence=0.0)
    assert out["volume"] == 0.01 and out["floored_to_min"] is True
    # Round DOWN to the broker step, never up into extra size.
    out = ex.doctrine_size(equity=10999.0, entry=2000.0,
                           stop_loss=1990.0, contract_value=100.0,
                           confidence=0.0)
    assert out["volume"] == 0.10
    # Variant B is expressible in the same one config place.
    out = ex.doctrine_size(equity=10000.0, entry=2000.0,
                           stop_loss=1990.0, contract_value=100.0,
                           confidence=1.0,
                           config=ex.SizingConfig(variant="B"))
    assert out["volume"] == 0.60  # 10000/500 x 0.01 x 3.0


# ── the invariants ───────────────────────────────────────────────────

def test_friction_gate_refuses_and_the_setup_stays_alive(tmp_path):
    conn = FakeConnector()
    engine = _engine(tmp_path, conn, quote=_quote(spread=4.0))
    res = engine.execute_setup(_setup(), _act())
    assert res["ok"] is False
    assert res["reason_category"] == "friction_gate"  # 4/10 = 0.4 > 0.10
    assert engine.open_positions() == []
    refused = _events(engine, "execution_refused")
    assert len(refused) == 1 and refused[0]["spread"] == 4.0
    # The lifecycle owns the setup's death: the engine wrote NO
    # setup transition for it — it stays alive for a later pass.
    kinds = {e["event"] for e in trading_journal.read_events(
        limit=0, path=engine.journal_path)}
    assert not any(k.startswith("setup_") for k in kinds)


def test_abstain_missing_and_error_judgments_all_refuse(tmp_path):
    engine = _engine(tmp_path, FakeConnector())
    abstain = {**_act(), "decision": "abstain",
               "reason_category": "regime_chop",
               "why": "chop wins (p=0.41)"}
    res = engine.execute_setup(_setup(), abstain)
    assert res["reason_category"] == "judgment_abstain"
    assert "regime_chop" in res["reason"]
    res = engine.execute_setup(_setup(), None)
    assert res["reason_category"] == "judgment_missing"
    res = engine.execute_setup(
        _setup(), {"decision": "error", "why": "boom"})
    assert res["reason_category"] == "judgment_error"
    assert engine.open_positions() == []
    assert len(_events(engine, "execution_refused")) == 3


def test_kill_switch_refuses(tmp_path):
    engine = _engine(tmp_path, FakeConnector())
    engine.desk.kill_switch_path().touch()
    res = engine.execute_setup(_setup(), _act())
    assert res["reason_category"] == "kill_switch"


def test_drawdown_halt_is_the_account_halt(tmp_path):
    conn = FakeConnector(equity=10000.0)
    engine = _engine(tmp_path, conn)
    assert engine.execute_setup(_setup(), _act())["ok"] is True
    conn.equity = 8400.0  # 16% below the observed peak
    res = engine.execute_setup(_setup(sid="qm_m5_native:XAUUSD:k2"),
                               _act())
    assert res["reason_category"] == "drawdown_halt"


def test_max_positions_refuses_the_fourth(tmp_path):
    engine = _engine(tmp_path, FakeConnector())
    for i in range(3):
        res = engine.execute_setup(_setup(sid=f"qm_m5_native:XAUUSD:k{i}"),
                                   _act())
        assert res["ok"] is True
    res = engine.execute_setup(_setup(sid="qm_m5_native:XAUUSD:k9"),
                               _act())
    assert res["reason_category"] == "max_positions"


def test_loss_streak_halt_after_six(tmp_path):
    engine = _engine(tmp_path, FakeConnector())
    for i in range(6):
        res = engine.execute_setup(_setup(sid=f"qm_m5_native:XAUUSD:k{i}"),
                                   _act())
        assert res["ok"] is True
        pid = res["position"]["position_id"]
        closed = engine.close_position(pid, price=1995.0)
        assert closed["ok"] is True and closed["pnl"] < 0
    res = engine.execute_setup(_setup(sid="qm_m5_native:XAUUSD:k9"),
                               _act())
    assert res["reason_category"] == "loss_streak_halt"


def test_non_demo_account_refused_at_the_boundary(tmp_path):
    conn = FakeConnector(demo=False)
    engine = _engine(tmp_path, conn, mode="live")
    res = engine.execute_setup(_setup(), _act())
    assert res["ok"] is False
    assert res["reason_category"] == "account_not_demo"
    assert conn.open_calls == []  # no order ever reached the broker


def test_demo_live_rung_executes_on_a_verified_demo_account(tmp_path):
    conn = FakeConnector(demo=True)
    engine = _engine(tmp_path, conn, mode="live")
    res = engine.execute_setup(_setup(), _act())
    assert res["ok"] is True
    assert res["position"]["rung"] == "exness_demo"
    assert res["position"]["position_id"] == "ex_1"
    assert len(conn.open_calls) == 1
    assert conn.open_calls[0]["confirmed"] is True


def test_mt5_rung_reports_unavailable_and_refuses(tmp_path):
    adapter = mt5_bridge.MT5BridgeConnector()
    assert adapter.status()["available"] is False
    with pytest.raises(mt5_bridge.MT5Unavailable):
        adapter.get_snapshot()
    engine = ex.ExecutionEngine(
        None, journal_path=tmp_path / "sys.jsonl", mt5_adapter=adapter,
        quote_fn=_quote(), now=lambda: BASE)
    res = engine.execute_setup(_setup(), _act())
    assert res["ok"] is False
    assert res["reason_category"] == "rung_unavailable"


# ── executor_check: genuinely in the order path ──────────────────────

def test_executor_check_blocks_the_order(tmp_path, monkeypatch):
    import nomorals.ta.lessons as lessons

    def _selective(symbol: str, timeframe: str = "") -> dict:
        if timeframe == "5m":
            return {"verdict": "DO NOT TRADE", "lesson_id": "L-test",
                    "evidence": "200 backtests, expectancy -0.3R",
                    "symbol": symbol, "timeframe": timeframe}
        return {"verdict": "RULE", "lesson_id": "L-ok",
                "symbol": symbol, "timeframe": timeframe}

    monkeypatch.setattr(lessons, "check_lesson", _selective)
    engine = _engine(tmp_path, FakeConnector())
    res = engine.execute_setup(_setup(), _act())  # 5m setup
    assert res["ok"] is False
    assert res["reason_category"] == "pre_order_check"
    assert "executor_check refused" in res["reason"]
    assert "L-test" in res["reason"]
    assert engine.desk.paper_positions() == []
    # And at the desk itself: an order carrying a malformed idea is
    # refused before any risk gate or fill.
    desk = engine.desk
    bad = TradeIdea(symbol="XAUUSD", side=0, confidence=0.0, entry=0.0,
                    invalidation=0.0, targets=[])
    with pytest.raises(DeskError, match="pre-order check refused"):
        desk.paper_open("XAUUSD", "buy", 0.1, entry=2000.0,
                        stop_loss=1990.0, idea=bad)
    good = TradeIdea(symbol="XAUUSD", side=1, confidence=50.0,
                     entry=2000.0, invalidation=1990.0,
                     targets=[(2030.0, 0.5)], timeframe="1h")
    pos = desk.paper_open("XAUUSD", "buy", 0.1, entry=2000.0,
                          stop_loss=1990.0, take_profit=2030.0, idea=good)
    assert pos.volume == 0.1


def test_the_real_lesson_book_refuses_xauusd_h1(tmp_path):
    """No patching: the shipped lesson database carries a DO NOT
    TRADE verdict for XAUUSD mean-reversion on 1h (xau-mr-no-edge).
    With executor_check genuinely wired into the order path, an H1
    setup is refused by it — the check is not decorative."""
    engine = _engine(tmp_path, FakeConnector())
    res = engine.execute_setup(_generic_setup(), _act())
    assert res["ok"] is False
    assert res["reason_category"] == "pre_order_check"
    assert "executor_check refused" in res["reason"]
    assert "xau-mr-no-edge" in res["reason"]
    assert engine.desk.paper_positions() == []


def _benign_lessons(monkeypatch) -> None:
    """The generic-profile management tests exercise MANAGEMENT,
    not the lesson gate: pin the lesson book to a benign RULE so
    the H1 setups can open (the real book's XAUUSD-1h refusal is
    proven separately above)."""
    import nomorals.ta.lessons as lessons

    def _rule(symbol: str, timeframe: str = "") -> dict:
        return {"verdict": "RULE", "lesson_id": "L-test",
                "symbol": symbol, "timeframe": timeframe}

    monkeypatch.setattr(lessons, "check_lesson", _rule)


# ── management: QM fixed bracket ─────────────────────────────────────

def test_qm_gets_no_management_beyond_its_bracket(tmp_path):
    engine = _engine(tmp_path, FakeConnector(),
                     quote=_quote(price=2015.0))
    res = engine.execute_setup(_setup(), _act(confidence=1.0))
    assert res["ok"] is True
    out = engine.manage_pass()
    # +1.5R of excursion: a generic profile would lock and take
    # partials here. The QM profile does NOTHING — ever.
    assert out["actions"] == []
    desk_pos = engine.desk.paper_positions()[0]
    assert desk_pos.stop_loss == 1990.0 and desk_pos.volume == 0.20
    # The bracket itself still fires: at the target the desk marks
    # it closed and the book reconciles from the simulator record.
    engine2 = _engine(tmp_path, engine.desk.connector,
                      quote=_quote(price=2030.0))
    engine2.desk = engine.desk
    out = engine2.manage_pass()
    assert [a["action"] for a in out["actions"]] == ["bracket_close"]
    pos = engine2.open_positions()
    assert pos == []
    closed = [p for p in engine2.load_positions()
              if p["state"] == "closed"]
    assert len(closed) == 1
    assert closed[0]["pnl"] == 600.0  # 30.0 x 100 x 0.20
    assert closed[0]["close_price_source"] == "simulator_record"


# ── management: the generic profile ──────────────────────────────────

def _generic_setup(**kw: Any) -> dict[str, Any]:
    base = _setup(sid="fvg_retest_h1:XAUUSD:k1",
                  playbook="fvg_retest_h1", family="trend", tf="1h",
                  targets=(2030.0, 2060.0), zone=(2000.0, 1996.0))
    base.update(kw)
    return base


def test_generic_partial_tiers_and_profit_lock(tmp_path, monkeypatch):
    _benign_lessons(monkeypatch)
    bars = lambda s, m, tf: _frame([2000.0] * 40)  # noqa: E731 - tiny ATR
    engine = _engine(tmp_path, FakeConnector(), bars=bars,
                     quote=_quote(price=2010.0))
    res = engine.execute_setup(_generic_setup(), _act(confidence=1.0))
    assert res["ok"] is True and res["position"]["volume"] == 0.20
    assert res["position"]["profile"] == "zone_retest"
    # At +1R: the 50% tier closes AND the profit lock arms.
    out = engine.manage_pass()
    kinds = [(a["action"], a.get("tier")) for a in out["actions"]]
    assert ("partial_close", "1R") in kinds
    assert ("profit_lock", None) in kinds
    lock = next(a for a in out["actions"] if a["action"] == "profit_lock")
    assert lock["applied"] is True
    # level = entry + max(0.5R=5.0, 0.75 x tiny ATR) = 2005.0
    assert lock["level"] == 2005.0
    desk_pos = engine.desk.paper_positions()[0]
    assert desk_pos.volume == 0.10 and desk_pos.stop_loss == 2005.0
    # At +2R: the 25% tier. At +3R: the last tier closes the trade.
    engine2 = _engine(tmp_path, engine.desk.connector, bars=bars,
                      quote=_quote(price=2020.0))
    engine2.desk = engine.desk
    out = engine2.manage_pass()
    assert [(a["action"], a.get("tier")) for a in out["actions"]] == \
        [("partial_close", "2R")]
    engine3 = _engine(tmp_path, engine.desk.connector, bars=bars,
                      quote=_quote(price=2030.0))
    engine3.desk = engine.desk
    out = engine3.manage_pass()
    last = out["actions"][-1]
    assert last["action"] == "partial_close" and last["tier"] == "3R"
    assert last.get("closed") is True
    closed = [p for p in engine3.load_positions()
              if p["state"] == "closed"]
    assert len(closed) == 1
    # 100 (1R x 0.10) + 100 (2R x 0.05) + 150 (3R x 0.05)
    assert closed[0]["pnl"] == 350.0
    assert closed[0]["close_reason"] == "partial_tier_3R_complete"


def test_generic_partials_need_003_lots_but_the_lock_does_not(
        tmp_path, monkeypatch):
    _benign_lessons(monkeypatch)
    bars = lambda s, m, tf: _frame([2000.0] * 40)  # noqa: E731
    conn = FakeConnector(equity=1400.0)
    engine = _engine(tmp_path, conn, bars=bars,
                     quote=_quote(price=2010.0))
    res = engine.execute_setup(_generic_setup(), _act(confidence=0.5))
    # 1400/1000 x 0.01 x 1.5 = 0.021 -> rounded down to 0.02 lots:
    # below the 0.03 partial-tier floor.
    assert res["position"]["volume"] == 0.02
    out = engine.manage_pass()
    kinds = [a["action"] for a in out["actions"]]
    assert "partial_close" not in kinds
    assert "profit_lock" in kinds
    assert engine.desk.paper_positions()[0].volume == 0.02


# ── management: the hot-scalp earned lock ────────────────────────────

def _scalp_setup(**kw: Any) -> dict[str, Any]:
    base = _setup(sid="hot_scalp_m5:XAUUSD:k1", playbook="hot_scalp_m5",
                  family="scalp", tf="5m", stop=1996.0,
                  targets=(2010.0, 2020.0), zone=(2000.0, 1996.0))
    base.update(kw)
    return base


def test_scalp_lock_does_not_arm_on_an_unproven_push(tmp_path):
    # Price pushes +1R but no M5 bar has CLOSED beyond T1: the
    # lock stays unarmed — an unproven push earns nothing.
    frame = _frame([2004.0] * 30)
    engine = _engine(tmp_path, FakeConnector(),
                     bars=lambda s, m, tf: frame,
                     quote=_quote(price=2004.0))
    res = engine.execute_setup(_scalp_setup(), _act(confidence=1.0))
    assert res["ok"] is True
    out = engine.manage_pass()
    assert [a for a in out["actions"]
            if a["action"] == "profit_lock"] == []
    assert engine.desk.paper_positions()[0].stop_loss == 1996.0


def test_scalp_lock_arms_on_m5_acceptance_capped_at_the_zone(tmp_path):
    closes = [2004.0] * 26 + [2012.0] * 4  # last bars close beyond T1
    frame = _frame(closes)
    engine = _engine(tmp_path, FakeConnector(),
                     bars=lambda s, m, tf: frame,
                     quote=_quote(price=2012.0))
    res = engine.execute_setup(_scalp_setup(), _act(confidence=1.0))
    assert res["ok"] is True
    out = engine.manage_pass()
    lock = next(a for a in out["actions"]
                if a["action"] == "profit_lock")
    assert lock["trigger"] == "m5_acceptance"
    assert lock["applied"] is True and lock["capped"] is True
    # Formula level exceeds the zone edge; the cap binds at the
    # entry zone's profit-side edge (2000.0 = breakeven here).
    assert lock["level"] == 2000.0
    assert engine.desk.paper_positions()[0].stop_loss == 2000.0


def test_scalp_lock_arms_on_the_2r_backstop(tmp_path):
    engine = _engine(tmp_path, FakeConnector(), bars=None,
                     quote=_quote(price=2008.0))  # exactly +2.0R
    res = engine.execute_setup(_scalp_setup(), _act(confidence=1.0))
    assert res["ok"] is True
    out = engine.manage_pass()
    lock = next(a for a in out["actions"]
                if a["action"] == "profit_lock")
    assert lock["trigger"] == "backstop_2r"


# ── the execution pass over the journal funnel ───────────────────────

def _journal_setup(journal: Path, *, state_events: list[str],
                   playbook: str = "qm_m5_native") -> None:
    base = {"setup_id": f"{playbook}:XAUUSD:k1", "symbol": "XAUUSD",
            "market": "forex", "playbook": playbook, "route": playbook,
            "family": "reversal", "entry_timeframe": "5m",
            "direction": 1, "pattern_key": "k1",
            "plan": {"entry_zone": {"top": 2000.0, "bottom": 1995.0},
                     "entry_ref": 2000.0, "invalidation": 1990.0,
                     "targets": [2030.0]},
            "evidence": {"atr": 4.0}, "read_taken_at": BASE}
    for event in state_events:
        fields = {**base, "reason_category": "pattern_formed",
                  "reason": "test"}
        if event == "setup_detected":
            fields.update({"to_state": "detected",
                           "expires_at": BASE + 3600,
                           "detected_bar_ts": BASE})
        elif event == "setup_armed":
            fields.update({"to_state": "armed", "armed_at": BASE,
                           "armed_bar_ts": BASE,
                           "expires_at": BASE + 3600})
        elif event == "setup_triggered":
            fields.update({"to_state": "triggered",
                           "trigger": {"bar_ts": BASE, "price": 2000.0}})
        trading_journal.append_event(event, path=journal, **fields)


def _journal_judgment(journal: Path, *, decision: str = "act") -> None:
    trading_journal.append_event(
        "judgment", path=journal, judged_at=BASE,
        kind="judgment", version=1, mode="shadow",
        subject={"type": "setup", "setup_id": "qm_m5_native:XAUUSD:k1",
                 "playbook": "qm_m5_native", "state": "triggered"},
        symbol="XAUUSD", market="forex", decision=decision,
        chosen_scenario="continuation", expected_r=0.4,
        confidence=0.5, why="continuation wins",
        reason_category="quality" if decision == "act" else "regime_chop")


def test_execution_pass_consumes_triggered_setups_once(tmp_path):
    journal = tmp_path / "sys.jsonl"
    _journal_setup(journal, state_events=[
        "setup_detected", "setup_armed", "setup_triggered"])
    _journal_judgment(journal)
    engine = _engine(tmp_path, FakeConnector())
    engine.journal_path = journal
    out = engine.execution_pass("XAUUSD", market="forex", read={})
    assert out["counts"] == {"executed": 1, "refused": 0, "skipped": 0}
    assert len(engine.open_positions()) == 1
    # A second pass skips it — a triggered setup executes once.
    out = engine.execution_pass("XAUUSD", market="forex", read={})
    assert out["counts"]["executed"] == 0
    assert out["counts"]["skipped"] == 1


def test_execution_pass_honors_a_journaled_abstention(tmp_path):
    journal = tmp_path / "sys.jsonl"
    _journal_setup(journal, state_events=[
        "setup_detected", "setup_armed", "setup_triggered"])
    _journal_judgment(journal, decision="abstain")
    engine = _engine(tmp_path, FakeConnector())
    engine.journal_path = journal
    out = engine.execution_pass("XAUUSD", market="forex", read={})
    assert out["counts"]["executed"] == 0
    assert out["counts"]["refused"] == 1
    assert out["refused"][0]["reason_category"] == "judgment_abstain"
    # And the refusal is terminal for that trigger: next pass skips.
    out = engine.execution_pass("XAUUSD", market="forex", read={})
    assert out["counts"] == {"executed": 0, "refused": 0, "skipped": 1}
