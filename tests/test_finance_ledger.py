"""Tests for Naira-first expense tracking + conversational budgeting.

All offline. Money is integer kobo throughout — several tests assert
exact integer arithmetic (no float drift by construction).
"""

from __future__ import annotations

import time
from decimal import Decimal
from pathlib import Path

import pytest

from nomorals.finance.budgets import (
    BudgetStatus,
    BudgetStore,
    budget_status,
    month_key,
    overspend_alerts,
    weekly_digest,
)
from nomorals.finance.commands import parse_spend_tail
from nomorals.finance.ledger import (
    Ledger,
    categorize,
    format_naira,
    naira_to_kobo,
    parse_amount,
)

NOW = 1_790_000_000.0  # fixed clock for determinism


def _ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


def _budgets(tmp_path: Path) -> BudgetStore:
    return BudgetStore(tmp_path / "budgets.json")


# ── parse_amount ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("5k", 500_000),
    ("5K", 500_000),
    ("2.5m", 250_000_000),
    ("2.5M", 250_000_000),
    ("₦1,500", 150_000),
    ("₦2000", 200_000),
    ("2000 naira", 200_000),
    ("2000", 200_000),
    ("1,500", 150_000),
    ("NGN 3.2m", 320_000_000),
    ("750", 75_000),
    ("0.5k", 50_000),
    ("₦99.99", 9_999),
])
def test_parse_amount_nigerian_shorthand(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize("text", [
    "", "abc", "time", "five k", "k", "₦", "5kk", "--5k", "5 k 5",
])
def test_parse_amount_garbage_returns_none(text):
    assert parse_amount(text) is None


def test_parse_amount_exact_kobo():
    # ₦99.99 is exactly 9999 kobo — no float involved.
    assert parse_amount("₦99.99") == 9_999
    assert naira_to_kobo(Decimal("0.1")) + naira_to_kobo(Decimal("0.2")) == \
        naira_to_kobo(Decimal("0.3"))


# ── categorize ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("note,expected", [
    ("uber to ikeja", "transport"),
    ("fuel for gen", "transport"),
    ("danfo fare", "transport"),
    ("mama put lunch", "food"),
    ("jollof rice", "food"),
    ("mtn data 10gb", "data"),
    ("airtime recharge", "data"),
    ("nepa bill", "housing"),
    ("rent", "housing"),
    ("pharmacy drugs", "health"),
    ("jumia order", "shopping"),
    ("dstv subscription", "data"),  # subscription keyword lives in data
    ("tithe", "giving"),
    ("random thing xyz", "other"),
    ("", "other"),
])
def test_categorize_keywords(note, expected):
    assert categorize(note) == expected


# ── ledger round-trip ────────────────────────────────────────────────────────

def test_ledger_log_and_query(tmp_path):
    ledger = _ledger(tmp_path)
    t1 = ledger.log(parse_amount("5k"), note="uber to ikeja")
    t2 = ledger.log(parse_amount("2k"), note="mama put", kind="spend")
    t3 = ledger.log(parse_amount("100k"), note="salary", kind="income")

    assert t1.category == "transport"
    assert t2.category == "food"
    assert t3.kind == "income"

    all_txns = ledger.transactions()
    assert len(all_txns) == 3
    assert ledger.total_spent() == 700_000  # exactly 7000.00 in kobo
    assert ledger.total_income() == 10_000_000

    assert len(ledger.transactions(category="food")) == 1
    assert len(ledger.transactions(kind="income")) == 1
    assert ledger.transactions(since=time.time() + 3600) == []


def test_ledger_rejects_bad_amounts(tmp_path):
    ledger = _ledger(tmp_path)
    with pytest.raises(ValueError):
        ledger.log(0, note="zero")
    with pytest.raises(ValueError):
        ledger.log(-100, note="negative")
    with pytest.raises(ValueError):
        ledger.log(100, note="x", kind="bogus")


def test_ledger_integer_arithmetic_no_drift(tmp_path):
    # The classic 0.1 + 0.2 float trap, in kobo: exact.
    ledger = _ledger(tmp_path)
    for _ in range(3):
        ledger.log(naira_to_kobo(Decimal("0.1")), note="kobo test")
    assert ledger.total_spent() == naira_to_kobo(Decimal("0.3")) == 30


def test_format_naira():
    assert format_naira(500_000) == "₦5,000"
    assert format_naira(9_999) == "₦99.99"
    assert format_naira(0) == "₦0"
    assert format_naira(-150_000) == "-₦1,500"


# ── budgets ──────────────────────────────────────────────────────────────────

def test_budget_set_get(tmp_path):
    budgets = _budgets(tmp_path)
    assert budgets.get_budget("food") is None
    budgets.set_budget("food", parse_amount("50k"))
    assert budgets.get_budget("food") == 5_000_000
    assert budgets.list_budgets() == {"food": 5_000_000}
    with pytest.raises(ValueError):
        budgets.set_budget("food", 0)


def test_budget_status_math(tmp_path):
    ledger, budgets = _ledger(tmp_path), _budgets(tmp_path)
    budgets.set_budget("food", parse_amount("50k"))
    budgets.set_budget("transport", parse_amount("20k"))
    ledger.log(parse_amount("10k"), category="food", note="groceries")
    ledger.log(parse_amount("5k"), category="transport", note="uber")

    statuses = budget_status(ledger, budgets)
    by_cat = {s.category: s for s in statuses}
    food = by_cat["food"]
    assert food.spent_kobo == 1_000_000
    assert food.remaining_kobo == 4_000_000
    assert food.pct_used == pytest.approx(0.2)
    assert food.state == "ok"
    # Sorted by pct_used desc: transport (25%) before food (20%).
    assert statuses[0].category == "transport"


def test_overspend_thresholds(tmp_path):
    ledger, budgets = _ledger(tmp_path), _budgets(tmp_path)
    budgets.set_budget("food", parse_amount("10k"))
    budgets.set_budget("data", parse_amount("10k"))
    ledger.log(parse_amount("8.5k"), category="food", note="x")   # 85% → warning
    ledger.log(parse_amount("12k"), category="data", note="y")   # 120% → over

    alerts = overspend_alerts(budget_status(ledger, budgets))
    by_cat = {a["category"]: a for a in alerts}
    assert by_cat["food"]["level"] == "warning"
    assert by_cat["data"]["level"] == "over"
    # Worst first.
    assert alerts[0]["category"] == "data"


def test_weekly_digest_text(tmp_path):
    ledger, budgets = _ledger(tmp_path), _budgets(tmp_path)
    budgets.set_budget("food", parse_amount("10k"), month=month_key(NOW))
    ledger.log(parse_amount("12k"), category="food", note="owambe",
               ts=NOW - 86400)
    ledger.log(parse_amount("5k"), category="transport", note="uber",
               ts=NOW - 2 * 86400)

    text = weekly_digest(ledger, budgets, now=NOW)
    assert "₦17,000" in text            # total spent
    assert "food" in text               # top category
    assert "over on food" in text       # 12k of 10k budget


def test_weekly_digest_quiet_week(tmp_path):
    ledger, budgets = _ledger(tmp_path), _budgets(tmp_path)
    budgets.set_budget("food", parse_amount("50k"), month=month_key(NOW))
    text = weekly_digest(ledger, budgets, now=NOW)
    assert "₦0" in text
    assert "all budgets on track" in text


# ── /spend command parsing ───────────────────────────────────────────────────

@pytest.mark.parametrize("tail,amount,category,note", [
    ("5k on transport", 500_000, "transport", ""),
    ("5k on transport for lunch", 500_000, "transport", "for lunch"),
    ("2000 lunch at mama put", 200_000, "", "lunch at mama put"),
    ("₦1,500", 150_000, "", ""),
    ("2.5m on rent january", 250_000_000, "rent", "january"),
])
def test_parse_spend_tail(tail, amount, category, note):
    parsed = parse_spend_tail(tail)
    assert parsed is not None
    assert parsed["amount_kobo"] == amount
    assert parsed["category"] == category
    assert parsed["note"] == note


@pytest.mark.parametrize("tail", ["", "abc on transport", "on transport"])
def test_parse_spend_tail_invalid(tail):
    assert parse_spend_tail(tail) is None


# ── NL intents (narrow regexes) ──────────────────────────────────────────────

def test_finance_intent_log_matches():
    from nomorals.agents.coremind import understand

    intents = [i for i in understand("I spent 5k on transport")
               if i.kind in ("finance_log", "finance_summary")]
    assert intents and intents[0].kind == "finance_log"
    assert intents[0].meta["amount_kobo"] == 500_000
    assert intents[0].meta["category"] == "transport"

    intents = [i for i in understand("spent 2000 on data")
               if i.kind in ("finance_log", "finance_summary")]
    assert intents and intents[0].kind == "finance_log"

    intents = [i for i in understand("I paid ₦1,500 for fuel")
               if i.kind in ("finance_log", "finance_summary")]
    assert intents and intents[0].kind == "finance_log"
    assert intents[0].meta["amount_kobo"] == 150_000


def test_finance_intent_summary_matches():
    from nomorals.agents.coremind import understand

    for text in ("how's my spending?", "how is my spending",
                 "how am i doing on food?", "how's my spending on data?"):
        intents = [i for i in understand(text)
                   if i.kind in ("finance_log", "finance_summary")]
        assert intents and intents[0].kind == "finance_summary", text


def test_finance_intent_no_misfire():
    from nomorals.agents.coremind import understand

    # "time" isn't an amount — never logs.
    for text in ("I spent time on transport",
                 "I spent five k on transport",
                 "how am i doing",
                 "spending time with family",
                 "I spent 5k"):
        intents = [i for i in understand(text)
                   if i.kind in ("finance_log", "finance_summary")]
        assert not intents, text
