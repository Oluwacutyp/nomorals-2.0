"""Sweep tests: every new/changed behavior in the finance module upgrade.

All offline. Money stays integer kobo throughout.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from nomorals.finance.alerts import (
    AlertStore,
    add_alert,
    evaluate_alerts,
    sync_bill_alerts,
    KIND_BILL_DUE,
    KIND_PRICE_ABOVE,
)
from nomorals.finance.budgets import (
    BudgetStatus,
    BudgetStore,
    budget_pace,
    budget_status,
    fifty_thirty_twenty,
    month_key,
    render_budget_grid,
    suggested_budgets,
)
from nomorals.finance.goals import (
    GOAL_BY_DATE,
    GOAL_MONTHLY,
    GOAL_TARGET,
    GoalStore,
    contribution_streak,
    create_goal,
    goal_progress,
    monthly_need,
    render_goals,
)
from nomorals.finance.guard import (
    check_outgoing,
    render_warning,
    risk_score,
)
from nomorals.finance.insights import (
    compute_insights,
    detect_recurring,
    forecast_month_end,
    safe_to_spend,
    spend_series,
    upcoming_bills,
)
from nomorals.finance.ledger import (
    Ledger,
    MerchantMemory,
    Transaction,
    categorize,
    extract_merchant,
    parse_amount,
)
from nomorals.finance.mandate import (
    MandateStore,
    check_mandate,
    issue_mandate,
    mandate_remaining,
)
from nomorals.finance.overview import (
    BalanceCache,
    MoneyOverview,
    RailBalance,
    asset_mix,
    net_worth_series,
    snapshot_net_worth,
)
from nomorals.finance.send import RecipientStore, masked_account
from nomorals.finance.style import (
    bar,
    current_theme,
    sparkline,
    table,
)
from nomorals.finance.trading_desk import (
    DeskError,
    RiskPolicy,
    TradingDesk,
    expected_value_r,
    size_position,
)


# ── helpers ──────────────────────────────────────────────────────────

def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


def _budgets(tmp_path: Path) -> BudgetStore:
    return BudgetStore(tmp_path / "budgets.json")


def _seed_spend(ledger: Ledger, now: float, amounts_notes: list,
                category: str = "food", kind: str = "spend") -> None:
    for i, (amt_kobo, note) in enumerate(amounts_notes):
        ledger.log(amt_kobo, category=category, note=note, kind=kind,
                   ts=now - i * 86400)


NOW = 1_790_000_000.0  # fixed "now" for deterministic tests


# ── style ────────────────────────────────────────────────────────────

def test_style_bar_clamps_and_plain_has_no_ansi():
    plain = current_theme("plain")
    assert "\x1b" not in bar(0.5, theme=plain)
    assert "\x1b" not in bar(1.5, theme=plain)  # clamped, no crash
    ninja = current_theme("ninja")
    assert "\x1b[" in bar(0.9, theme=ninja)
    assert "█" in bar(0.5) and "░" in bar(0.5)


def test_style_sparkline_and_table():
    assert sparkline([]) == "—"
    plain = current_theme("plain")
    assert len(sparkline([1, 2, 3, 4], theme=plain)) == 4
    out = table([["a", "1"], ["bb", "22"]], headers=["h1", "h2"])
    assert "h1" in out and "bb" in out


def test_style_env_override(monkeypatch):
    monkeypatch.setenv("FINANCE_THEME", "plain")
    assert current_theme().name == "plain"


# ── ledger ───────────────────────────────────────────────────────────

def test_parse_amount_word_suffixes():
    assert parse_amount("5 thousand") == 500_000
    assert parse_amount("2.5 million") == 250_000_000
    assert parse_amount("1 billion") == 100_000_000_000
    assert parse_amount("5k") == 500_000  # old shorthand still works
    assert parse_amount("₦1,500") == 150_000
    assert parse_amount("nope") is None


def test_transaction_id_and_merchant(tmp_path):
    ledger = _ledger(tmp_path)
    t = ledger.log(500_000, note="Netflix, monthly sub")
    assert t.id.startswith("txn_")
    assert t.merchant == "Netflix"
    # round-trip keeps the id
    back = ledger.transactions()[0]
    assert back.id == t.id
    # legacy lines without ids get one on read
    assert isinstance(back.id, str) and back.id


def test_merchant_memory_learning(tmp_path):
    mem = MerchantMemory(tmp_path / "mem.json")
    ledger = Ledger(tmp_path / "ledger.jsonl", memory=mem)
    assert ledger.learn("Starlink internet", "data")
    t = ledger.log(2_000_000, note="Starlink monthly")
    assert t.category == "data"  # learned, not keyword-matched
    assert categorize("Starlink again", mem) == "data"
    assert mem.forget("Starlink internet")
    assert mem.lookup("Starlink internet") is None


def test_ledger_search(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.log(500_000, note="mama put lunch")
    ledger.log(200_000, note="uber ride")
    hits = ledger.search("mama")
    assert len(hits) == 1 and "mama" in hits[0].note
    assert ledger.search("Uber")[0].merchant == "uber ride"


def test_ledger_duplicates(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.log(500_000, note="Netflix sub", ts=NOW)
    ledger.log(500_000, note="Netflix sub", ts=NOW + 3600)
    ledger.log(700_000, note="Netflix sub", ts=NOW + 7200)
    groups = ledger.duplicates()
    assert len(groups) == 1 and len(groups[0]) == 2


def test_ledger_import_csv(tmp_path):
    csv_path = tmp_path / "stmt.csv"
    csv_path.write_text(
        "Date,Narration,Amount\n"
        "2026-10-01,NETFLIX,-2500.00\n"
        "2026-10-02,SALARY,150000.00\n"
        "2026-10-03,NETFLIX,-2500.00\n"  # dup of line 1 (same merchant+amt)
        "2026-10-04,BROKEN,abc\n",
        encoding="utf-8")
    ledger = _ledger(tmp_path)
    report = ledger.import_csv(csv_path)
    assert report["imported"] == 2, report
    assert report["skipped_duplicates"] == 1, report
    assert len(report["errors"]) == 1, report
    kinds = {t.kind for t in ledger.transactions()}
    assert kinds == {"spend", "income"}  # sign auto-detect


def test_extract_merchant():
    assert extract_merchant("Uber - trip to Lekki") == "Uber"
    assert extract_merchant("Netflix, monthly sub") == "Netflix"
    assert extract_merchant("") == ""


# ── budgets ──────────────────────────────────────────────────────────

def test_budget_pace_verdicts():
    mk = month_key(NOW)
    s = BudgetStatus(category="food", month=mk,
                     budgeted_kobo=100_000_00, spent_kobo=90_000_00)
    pace = budget_pace(s, now=NOW)
    assert pace["verdict"] in ("behind pace", "on pace", "ahead of pace")
    assert 0 <= pace["month_elapsed_pct"] <= 1
    assert pace["projected_month_end_kobo"] >= 0
    # zero budget → explicit verdict, no crash
    s2 = BudgetStatus(category="x", month=mk, budgeted_kobo=0, spent_kobo=0)
    assert budget_pace(s2, now=NOW)["verdict"] == "no budget"


def test_budget_copy_forward(tmp_path):
    budgets = _budgets(tmp_path)
    budgets.set_budget("food", 5_000_000, month="2026-09")
    copied = budgets.copy_forward(from_month="2026-09", to_month="2026-10")
    assert copied == {"food": 5_000_000}
    assert budgets.get_budget("food", "2026-10") == 5_000_000
    assert budgets.copy_forward(from_month="1999-01",
                                to_month="1999-02") == {}
    assert budgets.delete_budget("food", "2026-10")
    assert budgets.get_budget("food", "2026-10") is None


def test_suggested_budgets_and_50_30_20(tmp_path):
    ledger = _ledger(tmp_path)
    # seed 3 months of food spend
    base = 1_790_000_000.0
    for m in range(3):
        ts = base - m * 30 * 86400
        ledger.log(3_000_000, category="food", note="groceries", ts=ts)
    sugg = suggested_budgets(ledger, months=3, now=base)
    assert sugg["food"] == 3_000_000  # exact average, rounded to ₦500
    split = fifty_thirty_twenty(100_000_00)
    assert split == {"needs": 50_000_00, "wants": 30_000_00,
                     "savings": 20_000_00}


def test_render_budget_grid(tmp_path):
    mk = month_key(NOW)
    s = BudgetStatus(category="food", month=mk,
                     budgeted_kobo=10_000_000, spent_kobo=9_000_000)
    out = render_budget_grid([s], mk, theme="plain", now=NOW)
    assert "food" in out and "█" in out and "pace" in out
    empty = render_budget_grid([], mk, theme="plain")
    assert "no budgets" in empty


# ── insights ─────────────────────────────────────────────────────────

def _seed_recurring(ledger: Ledger, now: float) -> None:
    # 4 monthly Netflix charges, last one hiked
    for i, amt in enumerate([250_000, 250_000, 250_000, 300_000]):
        ledger.log(amt, category="entertainment", note="Netflix monthly",
                   ts=now - (3 - i) * 30 * 86400)


def test_detect_recurring_price_hike_and_due(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_recurring(ledger, NOW)
    rec = detect_recurring(ledger.transactions(since=NOW - 120 * 86400))
    assert len(rec) == 1
    r = rec[0]
    assert r.price_changed
    assert r.price_change_kobo == 50_000
    assert r.next_due_ts > NOW
    assert r.yearly_cost_kobo == int(300_000 * 365.0 / 30)
    assert r.occurrences == 4


def test_upcoming_bills(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_recurring(ledger, NOW)
    bills = upcoming_bills(ledger, days=40, now=NOW)
    assert len(bills) == 1
    assert bills[0].note_pattern == "Netflix monthly"


def test_safe_to_spend(tmp_path):
    ledger = _ledger(tmp_path)
    # income: ₦150k/mo for 3 months
    for m in range(3):
        ledger.log(15_000_000, category="income", note="salary",
                   kind="income", ts=NOW - m * 30 * 86400)
    _seed_recurring(ledger, NOW)  # ~₦3k/mo committed
    safe = safe_to_spend(ledger, now=NOW)
    assert safe["safe_kobo"] >= 0
    assert safe["committed_recurring_kobo"] > 0
    assert safe["expected_income_kobo"] > 0


def test_forecast_and_series(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.log(1_000_000, category="food", note="lunch", ts=NOW - 86400)
    proj = forecast_month_end(ledger, now=NOW)
    assert proj >= 1_000_000
    series = spend_series(ledger, months=3, now=NOW)
    assert len(series) == 3
    assert all(isinstance(m, str) and isinstance(v, int)
               for m, v in series)


# ── guard ────────────────────────────────────────────────────────────

def _seed_baseline(ledger: Ledger, now: float) -> None:
    amounts = [420_000, 460_000, 500_000, 540_000, 580_000,
               440_000, 520_000, 480_000, 560_000, 500_000]
    for i, amt in enumerate(amounts):
        ledger.log(amt, category="transfer",
                   note="transfer to Mama", ts=now - (i + 1) * 86400)


def test_risk_score_weights(tmp_path):
    ledger = _ledger(tmp_path)
    # thin history: new recipient alone scores
    s = risk_score("Stranger", 500_000, ledger, now=NOW)
    assert s["signals"].get("new_recipient") == 25
    assert s["level"] == "normal"


def test_risk_score_high_on_scam_pattern(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_baseline(ledger, NOW)
    # new recipient + round ₦100k + unusual hour (02:00 UTC→local may vary;
    # force via explicit now at 2am local)
    import datetime as dt
    local_2am = dt.datetime(2026, 10, 10, 2, 0).astimezone().timestamp()
    s = risk_score("Stranger", 10_000_000, ledger, now=local_2am)
    assert s["score"] >= 60, s
    assert "new_recipient" in s["signals"]
    assert "unusual_hour" in s["signals"]
    assert "round_number_scam_pattern" in s["signals"]


def test_check_outgoing_returns_scored_warning(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_baseline(ledger, NOW)
    w = check_outgoing("Stranger", 10_000_000, ledger, now=NOW)
    assert w is not None
    assert w.score >= 60
    assert "risk" in w.message
    rendered = render_warning(w)
    assert "why this scored" in rendered
    # normal transfer to known recipient → no warning
    w2 = check_outgoing("Mama", 500_000, ledger, now=NOW)
    assert w2 is None


def test_check_outgoing_velocity(tmp_path):
    ledger = _ledger(tmp_path)
    for i in range(4):
        ledger.log(100_000, category="transfer",
                   note=f"transfer to P{i}", ts=NOW - i * 3600)
    w = check_outgoing("Mama", 100_000, ledger, now=NOW)
    assert w is not None
    assert any("velocity" in r for r in w.reasons)


# ── alerts ───────────────────────────────────────────────────────────

def test_bill_due_kind_validation(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    a = add_alert(KIND_BILL_DUE, "Netflix monthly", 3, store=store)
    assert a.kind == KIND_BILL_DUE and not a.one_shot
    with pytest.raises(ValueError):
        add_alert(KIND_BILL_DUE, "x", 365, store=store)
    with pytest.raises(ValueError):
        add_alert("bogus", "x", 1, store=store)


def test_sync_bill_alerts_idempotent(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_recurring(ledger, NOW)
    store = AlertStore(tmp_path / "alerts.json")
    r1 = sync_bill_alerts(store, ledger, days_ahead=40, now=NOW)
    assert r1["created"] == 1 and r1["total"] == 1
    r2 = sync_bill_alerts(store, ledger, days_ahead=40, now=NOW)
    assert r2["created"] == 0  # no duplicates
    assert len(store.list()) == 1


def test_bill_due_fires_once_per_cycle(tmp_path):
    ledger = _ledger(tmp_path)
    _seed_recurring(ledger, NOW)
    store = AlertStore(tmp_path / "alerts.json")
    a = add_alert(KIND_BILL_DUE, "Netflix monthly", 40, store=store)
    fired_msgs = []
    r1 = evaluate_alerts(store=store, ledger=ledger, now=NOW,
                         notify=lambda t, b: fired_msgs.append(t))
    assert r1["fired"] == 1, r1
    # second evaluation: same cycle → no re-fire (no spam)
    r2 = evaluate_alerts(store=store, ledger=ledger, now=NOW + 3600,
                         notify=lambda t, b: fired_msgs.append(t))
    assert r2["fired"] == 0, r2


def test_alert_snooze(tmp_path):
    store = AlertStore(tmp_path / "alerts.json")
    a = add_alert(KIND_PRICE_ABOVE, "BTC", 90000, store=store)
    assert store.snooze(a.id, days=7)
    result = evaluate_alerts(
        store=store, price_fn=lambda s, m: {"price": 100000.0},
        fx_fn=lambda b: None)
    assert result["skipped_snoozed"] == 1
    assert result["fired"] == 0


# ── goals ────────────────────────────────────────────────────────────

def test_goal_types_and_monthly_need(tmp_path):
    store = GoalStore(tmp_path / "goals.json")
    g = create_goal("rent", 36_000_000, deadline="in 90d",
                    goal_type=GOAL_BY_DATE, store=store)
    assert g.goal_type == GOAL_BY_DATE
    need = monthly_need(g, 0, now=NOW)
    assert need is not None and need > 0
    gm = create_goal("save-mo", 5_000_000, goal_type=GOAL_MONTHLY,
                     store=store)
    assert monthly_need(gm, 0, now=NOW) == 5_000_000
    with pytest.raises(ValueError):
        create_goal("x", 100, goal_type=GOAL_BY_DATE, store=store)  # no date
    with pytest.raises(ValueError):
        create_goal("x", 100, goal_type="bogus", store=store)


def test_goal_streak_and_progress(tmp_path):
    ledger = _ledger(tmp_path)
    store = GoalStore(tmp_path / "goals.json")
    g = create_goal("emergency", 50_000_000, store=store)
    # contributions count from goal creation — backdate it for the test
    g.created_at = NOW - 40 * 86400
    store.add(g)
    # contributions this month and last
    ledger.log(5_000_000, category="savings", note="save",
               kind="income", ts=NOW)
    ledger.log(5_000_000, category="savings", note="save",
               kind="income", ts=NOW - 35 * 86400)
    assert contribution_streak(g, ledger, now=NOW) >= 1
    p = goal_progress(g, ledger, now=NOW)
    assert p["contributed_kobo"] == 10_000_000
    assert p["streak_months"] >= 1
    assert p["goal_type"] == GOAL_TARGET


def test_goal_snooze(tmp_path):
    store = GoalStore(tmp_path / "goals.json")
    g = create_goal("trip", 10_000_000, store=store)
    assert store.snooze(g.id, days=30)
    assert store.get(g.id).snoozed
    assert store.unsnooze(g.id)
    assert not store.get(g.id).snoozed


def test_render_goals(tmp_path):
    ledger = _ledger(tmp_path)
    store = GoalStore(tmp_path / "goals.json")
    create_goal("emergency", 50_000_000, store=store)
    out = render_goals(store.list(), ledger, theme="plain", now=NOW)
    assert "emergency" in out and "█" not in out  # plain theme
    out2 = render_goals(store.list(), ledger, theme="ninja", now=NOW)
    assert "░" in out2


# ── mandate ──────────────────────────────────────────────────────────

def test_mandate_weekly_monthly_caps(tmp_path):
    store = MandateStore(tmp_path / "mandates.json")
    ledger = _ledger(tmp_path)
    m = issue_mandate(store, cap_per_txn=50_000_000, cap_per_day=200_000_000,
                      cap_per_week=300_000_000, cap_per_month=500_000_000)
    assert m.cap_per_week == 300_000_000
    # spend ₦290k two days ago: daily cap fine, weekly cap blown
    ledger.log(290_000_000, category="transfer", note="transfer to X",
               ts=time.time() - 2 * 86400)
    chk = check_mandate(store, "owner", "transfer", 20_000_000,
                        ledger=ledger)
    assert not chk.ok and "weekly" in chk.reason  # daily ok, weekly blown
    # without week/month caps → old behavior
    m2 = issue_mandate(store, cap_per_txn=50_000_000,
                       cap_per_day=200_000_000)
    chk2 = check_mandate(store, "owner", "transfer", 20_000_000,
                         ledger=ledger)
    assert chk2.ok  # latest mandate wins
    with pytest.raises(Exception):
        issue_mandate(store, cap_per_txn=1, cap_per_day=2,
                      cap_per_week=1)  # week < day


def test_mandate_describe_and_remaining(tmp_path):
    store = MandateStore(tmp_path / "mandates.json")
    ledger = _ledger(tmp_path)
    m = issue_mandate(store, cap_per_txn=50_000_000, cap_per_day=200_000_000,
                      ttl_days=30)
    desc = m.describe()
    assert "ACTIVE" in desc and "₦500,000" in desc and "Revocable" in desc
    rem = mandate_remaining(m, ledger)
    assert rem["daily"] == 200_000_000
    assert rem["per_txn"] == 50_000_000


# ── overview ─────────────────────────────────────────────────────────

def test_balance_cache_ttl():
    cache = BalanceCache(ttl_seconds=60)
    ov = MoneyOverview(
        rails=[RailBalance(rail="mono", label="Mono", amount=100.0,
                           currency="NGN", amount_ngn=100.0)],
        total_ngn=100.0, collected_at=time.time())
    calls = {"n": 0}

    def fake_collect(vault, **kw):
        calls["n"] += 1
        return ov

    import nomorals.finance.overview as ovm
    orig = ovm.collect_balances
    ovm.collect_balances = fake_collect
    try:
        cache.get(object())
        cache.get(object())
        assert calls["n"] == 1  # second hit served from cache
        cache.invalidate()
        cache.get(object())
        assert calls["n"] == 2
    finally:
        ovm.collect_balances = orig


def test_net_worth_snapshot_and_series(tmp_path):
    path = tmp_path / "nw.jsonl"
    ov = MoneyOverview(
        rails=[RailBalance(rail="binance", label="Binance · BTC",
                           amount=0.5, currency="BTC", amount_ngn=50_000_000.0)],
        total_ngn=50_000_000.0, collected_at=time.time())
    rec = snapshot_net_worth(ov, path=path)
    assert rec["total_ngn"] == 50_000_000.0
    assert rec["mix"]["crypto"] == 50_000_000.0
    series = net_worth_series(days=30, path=path)
    assert len(series) == 1
    mix = asset_mix(ov)
    assert mix["crypto"] == 50_000_000.0 and mix["fiat"] == 0.0


# ── send ─────────────────────────────────────────────────────────────

def test_masked_account():
    assert masked_account({"account_number": "0123456789",
                           "bank_name": "GTBank"}) == "GTBank ••6789"
    assert masked_account({}) == "••••"


def test_recipient_remove_rename(tmp_path):
    store = RecipientStore(tmp_path / "recipients.json")
    store.add("Mama", account_number="0123456789", bank_code="058",
              bank_name="GTBank")
    rec = store.rename("Mama", "Mama Bear")
    assert rec["name"] == "Mama Bear"
    assert store.get("mama bear")["account_number"] == "0123456789"
    assert store.get("mama") is None
    assert store.remove("Mama Bear")
    assert store.get("mama bear") is None
    assert not store.remove("nobody")


# ── trading desk ─────────────────────────────────────────────────────

class _FakeConnector:
    def __init__(self, equity: float = 10_000.0):
        self._equity = equity

    def get_snapshot(self):
        return {"account_state": {"equity": self._equity,
                                  "balance": self._equity},
                "positions": []}

    def get_candles(self, instrument, timeframe="H1", count=200):
        return [{"close": 2600.0, "c": 2600.0}]

    def get_instrument_condition(self, instrument):
        return {"contract_size": 100.0, "quote_currency": "USD",
                "volume_min": 0.01, "volume_max": 100.0,
                "volume_step": 0.01}


def _desk(tmp_path: Path, **kw) -> TradingDesk:
    return TradingDesk(_FakeConnector(), policy=RiskPolicy(**kw),
                       journal_path=tmp_path / "j.jsonl",
                       state_path=tmp_path / "s.json")


def test_desk_paper_round_trip_records_r(tmp_path):
    desk = _desk(tmp_path)
    pos = desk.paper_open("XAUUSD", "buy", 1.0, entry=2600.0,
                          stop_loss=2590.0, take_profit=2620.0)
    assert pos.risk_amount == pytest.approx(10 * 100 * 1.0)  # dist×cv×vol
    closed = desk.paper_close(pos.id, exit_price=2620.0)
    assert closed["pnl"] == pytest.approx(20 * 100 * 1.0)
    assert closed["r_multiple"] == pytest.approx(2.0)  # +2R
    s = desk.stats()
    assert s["total_trades"] == 1
    assert s["avg_r_multiple"] == pytest.approx(2.0)
    assert s["expectancy_r"] == pytest.approx(2.0)


def test_desk_stats_edge_metrics(tmp_path):
    desk = _desk(tmp_path)
    # win +2R, loss -1R, win +1R → win_rate 2/3, PF 3.0, avg R 2/3
    p1 = desk.paper_open("XAUUSD", "buy", 1.0, entry=100.0,
                         stop_loss=90.0, take_profit=120.0)
    desk.paper_close(p1.id, exit_price=120.0)
    p2 = desk.paper_open("XAUUSD", "buy", 1.0, entry=100.0,
                         stop_loss=90.0, take_profit=120.0)
    desk.paper_close(p2.id, exit_price=90.0)
    p3 = desk.paper_open("XAUUSD", "sell", 1.0, entry=100.0,
                         stop_loss=110.0, take_profit=90.0)
    desk.paper_close(p3.id, exit_price=90.0)
    s = desk.stats()
    assert s["total_trades"] == 3
    assert s["win_rate"] == pytest.approx(2 / 3, abs=0.01)
    assert s["profit_factor"] == pytest.approx(3.0, abs=0.05)
    assert s["avg_r_multiple"] == pytest.approx(2 / 3, abs=0.01)
    assert s["max_drawdown"] >= 0
    assert s["best_win_streak"] >= 1
    assert len(desk.equity_curve()) == 3
    assert s["longs"]["trades"] == 2 and s["shorts"]["trades"] == 1


def test_desk_heat_and_kelly(tmp_path):
    desk = _desk(tmp_path, max_risk_pct_per_trade=1.0)
    desk.paper_open("XAUUSD", "buy", 1.0, entry=2600.0, stop_loss=2590.0)
    heat = desk.portfolio_heat()
    assert heat["heat_pct"] == pytest.approx(10.0)  # 1000/10000
    assert not heat["within_limits"]  # 10% > 6%
    k = desk.kelly_fraction()
    assert not k["ok"]  # <10 trades


def test_desk_risk_gates_still_hold(tmp_path):
    desk = _desk(tmp_path, max_open_positions=1)
    desk.paper_open("XAUUSD", "buy", 1.0, entry=2600.0, stop_loss=2590.0)
    with pytest.raises(DeskError):
        desk.paper_open("XAUUSD", "buy", 1.0, entry=2600.0,
                        stop_loss=2590.0)  # max positions
    with pytest.raises(DeskError):
        desk.paper_open("XAUUSD", "buy", 1.0, entry=2600.0)  # no SL


def test_expected_value_r():
    assert expected_value_r(0.5, 2.0, 1.0) == pytest.approx(0.5)
    assert expected_value_r(0.4, 2.0, 1.0) == pytest.approx(0.2)


def test_size_position_guards():
    with pytest.raises(DeskError):
        size_position(equity=0, entry=100, stop_loss=90, risk_pct=1,
                      contract_value=1)
    sized = size_position(equity=10_000, entry=2600, stop_loss=2590,
                          risk_pct=1.0, contract_value=100.0)
    assert sized["volume"] == pytest.approx(0.1)
    assert sized["actual_risk_pct"] <= 1.0
