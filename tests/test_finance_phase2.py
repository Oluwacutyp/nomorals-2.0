"""Phase 2 §8 — commerce/finance deep improvement.

Covers the new native systems: alerts, unified balances, insights,
savings goals, the Exness trading desk, and the new chat commands.
All offline — price feeds, connectors, and the vault are faked.
Money movement is never exercised against real rails here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.finance.alerts import (
    AlertStore,
    add_alert,
    evaluate_alerts,
)
from nomorals.finance.goals import GoalStore, create_goal, goal_progress
from nomorals.finance.insights import compute_insights, detect_recurring
from nomorals.finance.ledger import Ledger, parse_amount
from nomorals.finance.overview import (
    MoneyOverview,
    RailBalance,
    collect_balances,
    render_overview,
)
from nomorals.finance.trading_desk import (
    DeskError,
    RiskPolicy,
    TradingDesk,
    size_position,
)

NOW = 1_790_000_000.0


# ── helpers ──────────────────────────────────────────────────────────

def _ledger(tmp_path: Path, txns: list[tuple] | None = None) -> Ledger:
    led = Ledger(tmp_path / "ledger.jsonl")
    for amount_kobo, category, note, kind, ts in (txns or []):
        led.log(amount_kobo, category=category, note=note, kind=kind, ts=ts)
    return led


class _FakeVault:
    """Behaves like a real vault with no credentials stored."""

    def list_all(self, service: str = "", **kw: object) -> list:
        return []

    def get(self, service: str, username: str, **kw: object) -> object:
        from nomorals.core.errors import NotFound
        raise NotFound(f"no credential for {service}")


class _FakeConnector:
    """Offline Exness stand-in for the desk."""

    def __init__(self, equity: float = 10_000.0):
        self._equity = equity
        self.opened: list[dict] = []

    def get_snapshot(self):
        return {"account_state": {"balance": self._equity,
                                  "equity": self._equity,
                                  "currency": "USD"},
                "positions": [], "orders": []}

    def get_account_details(self):
        return {"settings": {"currency": "USD", "trade_mode": "demo"}}

    def get_candles(self, instrument, timeframe="H1", **kw):
        count = kw.get("count", 200)
        return [{"close": 2650.0 + i * 0.1} for i in range(count)]

    def get_instrument_condition(self, instrument):
        return {"point_digits": 2, "contract_size": 100.0,
                "quote_currency": "USD", "volume_min": 0.01,
                "volume_max": 100.0, "volume_step": 0.01}

    def open_position(self, *a, **kw):
        raise AssertionError("paper tests must never call the connector")


def _desk(tmp_path: Path, **kw) -> TradingDesk:
    conn = kw.pop("connector", None) or _FakeConnector()
    return TradingDesk(
        conn, mode="paper",
        journal_path=tmp_path / "journal.jsonl",
        state_path=tmp_path / "desk.json",
        **kw)


# ── alerts ───────────────────────────────────────────────────────────

def test_alert_add_validates(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    with pytest.raises(ValueError):
        add_alert("bogus_kind", "BTC", 1.0, store=store)
    with pytest.raises(ValueError):
        add_alert("price_above", "BTC", -5.0, store=store)
    with pytest.raises(ValueError):
        add_alert("budget_pct", "food", 80.0, store=store)  # fraction, not %
    a = add_alert("price_above", "BTC", 90_000.0, store=store)
    assert a.id.startswith("alr_")
    assert store.get(a.id).threshold == 90_000.0


def test_alert_price_above_fires_once(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    add_alert("price_above", "BTC", 90_000.0, store=store)
    seen: list[tuple[str, str]] = []
    price = {"p": 95_000.0}

    def price_fn(symbol, market):
        return {"price": price["p"], "change_pct_24h": 1.0}

    out = evaluate_alerts(store=store, price_fn=price_fn,
                          notify=lambda t, b: seen.append((t, b)))
    assert out["fired"] == 1 and len(seen) == 1
    # One-shot: disabled after firing; a second round stays quiet.
    out2 = evaluate_alerts(store=store, price_fn=price_fn,
                           notify=lambda t, b: seen.append((t, b)))
    assert out2["fired"] == 0 and len(seen) == 1


def test_alert_repeating_rearms(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    add_alert("price_below", "BTC", 60_000.0, one_shot=False, store=store)
    seen: list[str] = []
    price = {"p": 59_000.0}

    def price_fn(symbol, market):
        return {"price": price["p"], "change_pct_24h": 0.0}

    notify = lambda t, b: seen.append(t)  # noqa: E731
    assert evaluate_alerts(store=store, price_fn=price_fn,
                           notify=notify)["fired"] == 1
    # Still below → no re-fire (no spam while the condition holds).
    assert evaluate_alerts(store=store, price_fn=price_fn,
                           notify=notify)["fired"] == 0
    # Condition clears, then re-triggers → fires again.
    price["p"] = 61_000.0
    assert evaluate_alerts(store=store, price_fn=price_fn,
                           notify=notify)["fired"] == 0
    price["p"] = 59_500.0
    assert evaluate_alerts(store=store, price_fn=price_fn,
                           notify=notify)["fired"] == 1
    assert len(seen) == 2


def test_alert_pct_change_and_fx(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    add_alert("pct_change", "ETH", 10.0, store=store)
    add_alert("fx_rate", "USD", 1600.0, store=store)

    def price_fn(symbol, market):
        return {"price": 3000.0, "change_pct_24h": -12.5}

    out = evaluate_alerts(store=store, price_fn=price_fn,
                          fx_fn=lambda base: 1650.0,
                          notify=lambda t, b: None)
    assert out["fired"] == 2


def test_alert_no_data_is_quiet(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    add_alert("price_above", "BTC", 90_000.0, store=store)
    out = evaluate_alerts(store=store,
                          price_fn=lambda s, m: None,
                          notify=lambda t, b: None)
    assert out["checked"] == 1 and out["fired"] == 0
    # Alert is still enabled — no data must not kill it.
    assert store.list()[0].enabled


def test_alert_remove(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    a = add_alert("price_above", "BTC", 1.0, store=store)
    assert store.remove(a.id)
    assert not store.remove(a.id)
    assert store.list() == []


# ── unified balances ─────────────────────────────────────────────────

def test_overview_all_rails_fail_soft():
    ov = collect_balances(_FakeVault())
    assert isinstance(ov, MoneyOverview)
    rails = {r.rail for r in ov.rails}
    assert {"mono", "binance", "coinbase", "exness", "wise"} <= rails
    # Nothing connected → no invented money.
    assert ov.total_ngn == 0.0
    text = render_overview(ov)
    assert "not connected" in text


def test_overview_rail_subset_and_errors():
    # A rail that explodes is reported, never fatal.
    import nomorals.finance.overview as ovm

    def boom(vault, fx_fn):
        raise RuntimeError("kaput")

    orig = ovm._RAILS
    ovm._RAILS = (("mono", boom),)
    try:
        ov = collect_balances(_FakeVault())
    finally:
        ovm._RAILS = orig
    assert ov.rails[0].status == "error"
    assert "kaput" in ov.rails[0].detail


def test_overview_ngn_conversion():
    fx = lambda base: {"USD": 1500.0, "EUR": 1600.0}[base]  # noqa: E731
    price = lambda s, m: {"price": 100_000.0}  # noqa: E731  # 1 BTC = $100k
    rb1 = RailBalance(rail="t", label="t", amount=100.0, currency="USD",
                      amount_ngn=100.0 * 1500.0)
    rb2 = RailBalance(rail="t", label="t2", amount=0.5, currency="BTC",
                      amount_ngn=0.5 * 100_000.0 * 1500.0)
    ov = MoneyOverview(rails=[rb1, rb2],
                       total_ngn=rb1.amount_ngn + rb2.amount_ngn,
                       collected_at=NOW)
    text = render_overview(ov)
    assert "₦150,000" in text
    assert "₦75,000,000" in text


# ── insights ─────────────────────────────────────────────────────────

def _insight_ledger(tmp_path: Path) -> Ledger:
    txns = []
    # 30 days of daily ₦2k food spends + one ₦50k salary/week income.
    for d in range(30):
        ts = NOW - d * 86400
        txns.append((200_000, "food", "mama put lunch", "spend", ts))
        if d % 7 == 0:
            txns.append((5_000_000, "income", "salary", "income", ts))
    # Monthly-ish recurring: Netflix ₦8.5k on the 1st-ish, 3 occurrences.
    for d in (29, 59, 89):
        txns.append((850_000, "entertainment", "Netflix subscription",
                     "spend", NOW - d * 86400))
    return _ledger(tmp_path, txns)


def test_insights_numbers(tmp_path):
    ins = compute_insights(_insight_ledger(tmp_path), window_days=30,
                           now=NOW)
    assert ins.total_spent_kobo > 0
    assert ins.total_income_kobo > 0
    # burn ≈ (30×200k + ~1 netflix in window) / 30 per day
    assert 180_000 <= ins.burn_rate_kobo_per_day <= 260_000
    assert 0.0 < ins.savings_rate < 1.0
    assert ins.runway_days is not None and ins.runway_days > 0
    cats = dict(ins.top_categories)
    assert cats["food"] > cats.get("entertainment", 0)


def test_insights_recurring_detection(tmp_path):
    txns = []
    for d in (29, 59, 89):
        txns.append((850_000, "entertainment", "Netflix subscription",
                     "spend", NOW - d * 86400))
    txns.append((200_000, "food", "random lunch", "spend", NOW - 1000))
    rec = detect_recurring(_ledger(tmp_path, txns).transactions())
    assert len(rec) == 1
    assert rec[0].occurrences == 3
    assert 25 <= rec[0].avg_interval_days <= 35
    assert rec[0].amount_kobo == 850_000


def test_insights_trend(tmp_path):
    # Spend doubles in the current window vs the previous one.
    txns = []
    for d in range(30):
        txns.append((100_000, "food", "x", "spend", NOW - d * 86400))
    for d in range(30, 60):
        txns.append((50_000, "food", "x", "spend", NOW - d * 86400))
    ins = compute_insights(_ledger(tmp_path, txns), window_days=30,
                           now=NOW)
    assert ins.trend_pct is not None
    assert 0.9 < ins.trend_pct < 1.1  # ~+100%


# ── goals ────────────────────────────────────────────────────────────

def test_goal_lifecycle(tmp_path):
    store = GoalStore(tmp_path / "goals.json")
    g = create_goal("emergency fund", 50_000_000, deadline="dec",
                    store=store)
    assert g.deadline_ts and g.deadline_ts > NOW
    led = _ledger(tmp_path, [
        (10_000_000, "savings", "ajo payout", "income", NOW - 100),
        (5_000_000, "savings", "set aside", "spend", NOW - 50),
    ])
    # Goal created "now" — backdate it so the txns count.
    g.created_at = NOW - 1000
    store.add(g)
    p = goal_progress(g, led, now=NOW)
    assert p["contributed_kobo"] == 15_000_000
    assert p["pct"] == pytest.approx(0.3)
    assert p["remaining_kobo"] == 35_000_000
    assert p["days_left"] and p["needed_per_day_kobo"] > 0
    assert store.mark_done(g.id)
    assert store.get(g.id).done


def test_goal_validation(tmp_path):
    store = GoalStore(tmp_path / "goals.json")
    with pytest.raises(ValueError):
        create_goal("", 1000, store=store)
    with pytest.raises(ValueError):
        create_goal("x", 0, store=store)
    with pytest.raises(ValueError):
        create_goal("x", 1000, deadline="someday", store=store)
    g = create_goal("x", 1000, deadline="in 90d", store=store)
    assert 89 * 86400 < g.deadline_ts - time.time() < 91 * 86400


# ── trading desk ─────────────────────────────────────────────────────

def test_size_position_math():
    # $10k equity, 1% risk = $100; SL 50 points away on XAUUSD
    # (contract value $100/point/lot) → 0.02 lots.
    out = size_position(equity=10_000.0, entry=2650.0, stop_loss=2600.0,
                        risk_pct=1.0, contract_value=100.0)
    assert out["volume"] == pytest.approx(0.02)
    assert out["actual_risk_pct"] == pytest.approx(1.0)
    assert out["risk_amount"] == pytest.approx(100.0)


def test_size_position_refusals():
    with pytest.raises(DeskError):  # stop == entry
        size_position(equity=1000, entry=100.0, stop_loss=100.0,
                      risk_pct=1.0, contract_value=100.0)
    with pytest.raises(DeskError):  # unknown contract value — no guessing
        size_position(equity=1000, entry=100.0, stop_loss=99.0,
                      risk_pct=1.0, contract_value=0.0)
    with pytest.raises(DeskError):  # min lot risks too much
        size_position(equity=100.0, entry=2650.0, stop_loss=2600.0,
                      risk_pct=1.0, contract_value=100.0,
                      volume_min=0.01)


def test_paper_open_close_cycle(tmp_path):
    desk = _desk(tmp_path)
    pos = desk.paper_open("XAUUSD", "buy", 0.02, entry=2650.0,
                          stop_loss=2600.0, take_profit=2700.0)
    assert pos.id.startswith("px_")
    assert len(desk.paper_positions()) == 1
    # Mark below the stop → auto-close at the stop, loss = 50pts × $100 × 0.02.
    closes = desk.paper_mark("XAUUSD", 2590.0)
    assert len(closes) == 1
    assert closes[0]["pnl"] == pytest.approx(-100.0)
    assert closes[0]["reason"] == "stop_loss"
    assert desk.paper_positions() == []
    stats = desk.stats()
    assert stats["total_trades"] == 1 and stats["losses"] == 1
    assert stats["realized_pnl"] == pytest.approx(-100.0)


def test_paper_take_profit(tmp_path):
    desk = _desk(tmp_path)
    pos = desk.paper_open("XAUUSD", "sell", 0.01, entry=2650.0,
                          stop_loss=2700.0, take_profit=2600.0)
    closes = desk.paper_mark("XAUUSD", 2595.0)
    assert closes[0]["reason"] == "take_profit"
    assert closes[0]["pnl"] == pytest.approx(50.0)  # 50pts × $100 × 0.01


def test_desk_requires_stop_loss(tmp_path):
    desk = _desk(tmp_path)
    with pytest.raises(DeskError, match="stop-loss is required"):
        desk.paper_open("XAUUSD", "buy", 0.01, entry=2650.0)
    # ...unless the policy says otherwise.
    desk2 = _desk(tmp_path, policy=RiskPolicy(require_stop_loss=False))
    pos = desk2.paper_open("XAUUSD", "buy", 0.01, entry=2650.0)
    assert pos.stop_loss is None


def test_desk_max_positions(tmp_path):
    desk = _desk(tmp_path, policy=RiskPolicy(max_open_positions=2))
    desk.paper_open("XAUUSD", "buy", 0.01, entry=2650.0, stop_loss=2600.0)
    desk.paper_open("EURUSD", "buy", 0.01, entry=1.10, stop_loss=1.09)
    with pytest.raises(DeskError, match="max open positions"):
        desk.paper_open("GBPUSD", "buy", 0.01, entry=1.30, stop_loss=1.29)


def test_desk_daily_loss_kill(tmp_path):
    desk = _desk(tmp_path, policy=RiskPolicy(max_daily_loss_pct=1.0))
    # Lose $150 on $10k equity (> 1%) then try to open → killed.
    desk.paper_open("XAUUSD", "buy", 0.03, entry=2650.0, stop_loss=2600.0)
    desk.paper_mark("XAUUSD", 2590.0)
    assert desk.daily_pnl() == pytest.approx(-150.0)
    with pytest.raises(DeskError, match="daily loss limit"):
        desk.paper_open("XAUUSD", "buy", 0.01, entry=2650.0,
                        stop_loss=2600.0)


def test_desk_live_gates(tmp_path):
    desk = _desk(tmp_path)  # paper mode
    with pytest.raises(DeskError, match="paper mode"):
        desk.live_open("XAUUSD", "buy", 0.01, stop_loss=2600.0)
    live = TradingDesk(_FakeConnector(), mode="live",
                       journal_path=tmp_path / "j2.jsonl",
                       state_path=tmp_path / "d2.json")
    # Demo account → no unlock needed, but confirmation still required:
    # the fake connector's open_position raises AssertionError only if
    # reached; the desk must pass risk gates first (SL required).
    with pytest.raises(DeskError, match="stop-loss is required"):
        live.live_open("XAUUSD", "buy", 0.01)
    # Non-demo account without unlock → refused before any API call.
    live.connector = _NonDemoConnector()
    with pytest.raises(DeskError, match="locked"):
        live.live_open("XAUUSD", "buy", 0.01, stop_loss=2600.0)


class _NonDemoConnector(_FakeConnector):
    def get_account_details(self):
        return {"settings": {"currency": "USD", "trade_mode": "real"}}


def test_desk_journal_and_stats(tmp_path):
    desk = _desk(tmp_path)
    desk.paper_open("XAUUSD", "buy", 0.01, entry=2650.0,
                    stop_loss=2600.0, take_profit=2700.0)
    desk.paper_mark("XAUUSD", 2705.0)
    entries = desk.journal()
    kinds = {e["event"] for e in entries}
    assert {"paper_open", "paper_close"} <= kinds
    stats = desk.stats()
    assert stats["wins"] == 1
    assert stats["per_instrument"]["XAUUSD"]["win_rate"] == 1.0


# ── chat commands ────────────────────────────────────────────────────

def _ctx(tmp_path: Path) -> Any:
    return SimpleNamespace(settings=SimpleNamespace(
        home_path=str(tmp_path)))


def test_control_mandate(tmp_path, monkeypatch):
    from nomorals.finance import commands as cmds
    from nomorals.finance import mandate as mand_mod

    monkeypatch.setattr(mand_mod.MandateStore, "__init__",
                        lambda self, path=None: setattr(
                            self, "path", tmp_path / "mandates.json") or
                        setattr(self, "_data", None))
    ctx = _ctx(tmp_path)
    out = cmds.control_mandate("issue transfer 50k 200k", ctx)
    assert "mandate issued" in out
    out = cmds.control_mandate("list", ctx)
    assert "ACTIVE" in out
    mid = [l for l in out.splitlines() if "mand_" in l][0].split()[1]
    out = cmds.control_mandate(f"revoke {mid}", ctx)
    assert "revoked" in out
    out = cmds.control_mandate("issue transfer 5k", ctx)
    assert "usage" in out


def test_control_alert(tmp_path, monkeypatch):
    from nomorals.finance import commands as cmds
    from nomorals.finance import alerts as alerts_mod

    monkeypatch.setattr(alerts_mod.AlertStore, "__init__",
                        lambda self, path=None: setattr(
                            self, "path", tmp_path / "alerts.json"))
    ctx = _ctx(tmp_path)
    out = cmds.control_alert("add price_above BTC 90000", ctx)
    assert "alert set" in out
    out = cmds.control_alert("list", ctx)
    assert "BTC above 90,000" in out
    out = cmds.control_alert("add bogus BTC 1", ctx)
    assert "couldn't add alert" in out


def test_control_goal_and_insights(tmp_path, monkeypatch):
    from nomorals.finance import commands as cmds
    from nomorals.finance import goals as goals_mod

    monkeypatch.setattr(goals_mod.GoalStore, "__init__",
                        lambda self, path=None: setattr(
                            self, "path", tmp_path / "goals.json"))
    ctx = _ctx(tmp_path)
    out = cmds.control_goal("add rainy-day 100k dec", ctx)
    assert "goal set" in out
    out = cmds.control_goal("list", ctx)
    assert "rainy-day" in out
    out = cmds.control_goal("add bad", ctx)
    assert "usage" in out
    out = cmds.control_insights("30", ctx)
    assert "money insights" in out
