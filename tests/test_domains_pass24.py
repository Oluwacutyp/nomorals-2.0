"""Pass #24 — health/travel/property/legal/finance second-wave surfaces.

Three verified bugs, each reproduced failing-first:

1. Legal portfolio: amending a contract double-counted it.
   ``portfolio_value()`` summed EVERY row in ``contracts`` — including
   superseded versions that ``amend()`` deliberately keeps for history —
   so one live ₦600,000 agreement reported ₦1,100,000. And the old
   version's open obligations stayed live, so attention/pipeline/alerts
   reported both versions' deadlines for the same real-world dates.

2. Finance: "next due" never rolled forward.
   ``detect_recurring`` set ``next_due = last + one interval`` and
   ``detect_subscriptions`` did the same for ``estimated_next_renewal``.
   Once the last charge was more than one period old, the estimate sat
   in the past forever: ``upcoming_bills`` returned nothing, bill-due
   alerts could never fire, renewal reminders never appeared, and the
   recurring-charges view printed a past date as "next ~".

3. Travel groups: percent splits could exceed the expense.
   ``compute_shares`` promises shares that "sum EXACTLY to
   amount_kobo", but percentages totalling slightly over 100 (within
   the 0.01 validation tolerance) produced shares summing to MORE than
   the bill — the dust-distribution loop only ever adds kobo, never
   removes an excess.
"""

from __future__ import annotations

import os
import tempfile
import time
from datetime import date, timedelta

from nomorals.finance.insights import detect_recurring, upcoming_bills
from nomorals.finance.ledger import Ledger
from nomorals.finance.subscriptions import (
    detect_subscriptions,
    renewal_reminders,
)
from nomorals.legal.portfolio import ContractPortfolio
from nomorals.travel.groups import compute_shares

DAY = 86400.0

DOC = (
    "SERVICE AGREEMENT between Ada Ltd (the Client) and Bola (the "
    "Contractor). Effective date: 1st January 2026 for a term of 1 year. "
    "The agreement shall automatically renew. "
    "Payment: ₦500,000 per month, payable within 30 days of invoice. "
    "Renewal notice 60 days before expiry. Expiry: 31st December 2026. "
    "Either party may terminate with 30 days written notice."
)


def _tmpdb() -> str:
    return os.path.join(tempfile.mkdtemp(), "portfolio.db")


def _ledger() -> Ledger:
    return Ledger(path=os.path.join(tempfile.mkdtemp(), "ledger.jsonl"))


# ── 1. legal portfolio: amendment must not double-count ─────────────────────


def test_portfolio_value_counts_amended_contract_once():
    p = ContractPortfolio(db_path=_tmpdb())
    c, _ = p.add_raw(DOC, name="Ada deal")
    assert p.portfolio_value() == 500000.0
    new_c, _ = p.amend(c.id, DOC.replace("₦500,000", "₦600,000"))
    assert new_c is not None
    # one live agreement, worth ₦600,000 — not 500k + 600k
    assert p.portfolio_value() == 600000.0
    # history still keeps both versions
    assert [x.id for x in p.history(new_c.id)] == [c.id, new_c.id]


def test_amend_retires_superseded_obligations():
    p = ContractPortfolio(db_path=_tmpdb())
    c, _ = p.add_raw(DOC, name="Ada deal")
    assert len(p.obligations(c.id)) > 0
    new_c, _ = p.amend(c.id, DOC.replace("₦500,000", "₦600,000"))
    # the superseded version's deadlines no longer alert: the amended
    # version re-extracted its own from the full amended text
    assert p.obligations(c.id) == []
    assert len(p.obligations(new_c.id)) > 0
    names = [cc.name for cc, _ in p.needs_attention(4000)]
    assert names == ["Ada deal (amended)"]


# ── 2. finance: next-due rolls forward to a future occurrence ───────────────


def _monthly_ledger(last_charge_days_ago: int, now: float) -> Ledger:
    led = _ledger()
    for back in (last_charge_days_ago + 60, last_charge_days_ago + 30,
                 last_charge_days_ago):
        led.log(1500000, category="data", note="Netflix monthly",
                kind="spend", ts=now - back * DAY)
    return led


def test_detect_recurring_next_due_is_in_the_future():
    now = time.time()
    led = _monthly_ledger(40, now)  # last charge 40d ago, monthly cadence
    rec = detect_recurring(led.transactions(), now=now)
    assert len(rec) == 1
    # last + 2 intervals = 20 days from now, not 10 days in the past
    assert rec[0].next_due_ts > now
    assert abs(rec[0].next_due_ts - (now + 20 * DAY)) < 3600


def test_upcoming_bills_sees_charge_older_than_one_period():
    now = time.time()
    led = _monthly_ledger(40, now)
    # real next due is 20 days out
    assert [r.note_pattern for r in upcoming_bills(led, days=25, now=now)] \
        == ["Netflix monthly"]
    # and 15 days later it is 5 days out, inside a 7-day window
    later = now + 15 * DAY
    assert [r.note_pattern for r in upcoming_bills(led, days=7, now=later)] \
        == ["Netflix monthly"]


def test_subscription_renewal_rolls_forward():
    txns = [
        {"merchant": "Spotify", "amount_minor": 90000,
         "date": (date.today() - timedelta(days=back)).isoformat()}
        for back in (100, 70, 40)
    ]
    subs = detect_subscriptions(txns, usage_records={"Spotify": date.today()})
    assert len(subs) == 1
    # last charge 40d ago on a 30d cadence → next renewal 20d from today,
    # not 10 days in the past
    assert subs[0].estimated_next_renewal == date.today() + timedelta(days=20)
    rems = renewal_reminders(subs, days_ahead=25)
    assert [r.merchant for r in rems] == ["Spotify"]
    assert rems[0].days_until == 20


# ── 3. travel groups: percent shares never exceed the expense ───────────────


def test_percent_split_summing_over_100_still_totals_exactly():
    shares = compute_shares(100_000_000, ["a", "b"], split="percent",
                            shares=[50.0045, 50.0045])  # totals 100.009
    assert sum(shares) == 100_000_000
    assert shares == [50_000_000, 50_000_000]


def test_percent_split_under_100_totals_exactly():
    shares = compute_shares(999_999, ["a", "b", "c"], split="percent",
                            shares=[33.33, 33.33, 33.33])  # totals 99.99
    assert sum(shares) == 999_999


def test_percent_split_exact_100_unchanged():
    shares = compute_shares(100_000, ["a", "b"], split="percent",
                            shares=[25.0, 75.0])
    assert shares == [25_000, 75_000]
