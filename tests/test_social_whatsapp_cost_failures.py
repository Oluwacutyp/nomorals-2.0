"""Tail-51 pins for nomorals/social/whatsapp_cost.py —
2 converts (222, 269) / 6 exclusions (152, 255, 304, 326,
345, 558). Amended per the parent ruling on tail-51.

8 tests for 8 sites (counts equal). Each test pins one site
with a control half proving the guarded path was actually
exercised, asserting via the double's own call log that the
failing operation itself was attempted. The file contains
no exception handlers, so it adds zero census sites.

whatsapp_cost.py calls no failures-ledger record function
anywhere (grep-proven), so NOT-recording is pinned against
the social package ledger (nomorals/social/failures.py,
pass #150) with the tail-8 snapshot method:
recent_failures() before/after each trigger is unchanged.

The module's documented contract is "never raises" end to
end (CostTracker, CostAwareSender, worth_it); every
fallback below is the documented genuine value for its
function. Verdict notes (full grounds in the tail-51
SURVEY.md):
- 152: __init__ os.makedirs failure -> pass; directory
  creation is preparatory only (it fires by design on every
  ":memory:" / existing-dir construction, where dirname is
  "" or the dir exists) and the sqlite connect that follows
  is the honest arbiter — it raises out of __init__ when
  the path is truly unusable.
- 222: CONVERTED (parent ruling) — spent_since db failure
  -> None (bound + debug-logged), propagating honestly:
  weekly_spend / spend_vs_revenue render a "spend data
  unavailable" line, check_budget blocks under a set hard
  budget, project_monthly returns its unavailable dict.
- 255: project_monthly inner failure -> the "projection
  unavailable" dict; the summary honestly names the
  failure (on_track stays True — observed, pinned as-is).
- 269: CONVERTED with 222 (parent ruling) — message_count
  db failure -> None (bound + debug-logged); it renders in
  the same weekly_spend sentence as 222's spend figure, so
  a count-only failure also renders the unavailable line.
- 304: get_budget db failure -> None, the documented
  no-budget-set value ("or None when no budget is set").
- 326: check_budget failure -> (True, 0, 0), the site's
  own documented policy: "fail open on accounting errors,
  never block".
- 345: budget_warning failure -> "", the documented
  fine/no-warning value; advisory only, no consumer
  decision rides on it.
- 558: worth_it unconvertible input -> False, the genuine
  not-worth-it decision value; fail-closed, and reachable
  only via exotic non-numeric inputs (the declared inputs
  are ints).
"""

import os
import sqlite3
from pathlib import Path

import nomorals.social.whatsapp_cost as WC
from nomorals.social import failures as soc_fail


def _snap():
    return list(soc_fail.recent_failures())


class _ExplodingDB:
    """DB double: execute() logs the SQL (attempt proof) and raises."""

    def __init__(self):
        self.sql = []

    def execute(self, sql, params=()):
        self.sql.append(sql)
        raise sqlite3.DatabaseError("db exploded")


class _SelectiveFailingDB:
    """DB wrapper: fails only queries containing ``needle``.

    Delegates every other query to the real connection, so
    e.g. a wa_costs failure still lets wa_budgets reads
    succeed (the check_budget hard-budget branch), and a
    COUNT-only failure still lets the SUM read succeed.
    execute() logs every SQL string (attempt proof).
    """

    def __init__(self, real, needle):
        self._real = real
        self._needle = needle
        self.sql = []

    def execute(self, sql, params=()):
        self.sql.append(sql)
        if self._needle in sql:
            raise sqlite3.DatabaseError("selective db failure")
        return self._real.execute(sql, params)


def _mem():
    return WC.CostTracker(":memory:")


def test_152_makedirs_failure_passes_and_connect_decides(monkeypatch):
    before = _snap()
    log = []

    def bad_makedirs(path, *a, **k):
        log.append(path)
        raise OSError("cannot create dir")

    monkeypatch.setattr(os, "makedirs", bad_makedirs)
    tracker = WC.CostTracker(":memory:")
    assert log == [""]  # dirname(":memory:") — the attempt happened
    assert tracker.track("+1", "service") == WC.RATE_SERVICE_KOBO
    assert tracker.spent_week() == WC.RATE_SERVICE_KOBO
    monkeypatch.undo()
    # control: the real makedirs creates the dir and the tracker works
    import tempfile

    path = str(Path(tempfile.mkdtemp()) / "sub" / "cost.db")
    control = WC.CostTracker(path)
    assert Path(path).parent.is_dir()
    assert control.track("+1", "service") == WC.RATE_SERVICE_KOBO
    assert _snap() == before


def test_222_spent_since_db_failure_returns_none_converted(monkeypatch):
    # CONVERTED site (parent ruling): failure -> None, honestly
    # propagated to every rendered / gated consumer.
    before = _snap()
    tracker = _mem()
    tracker.track("a", "marketing")  # ₦84 of real spend
    # control: healthy DB seeded with ₦84 reads back genuinely
    assert tracker.spent_since(0) == 8400
    assert tracker.weekly_spend() == (
        "💸 Devon spent ₦84 on WhatsApp this week (1 message).")
    boom = _ExplodingDB()
    monkeypatch.setattr(tracker, "_db", boom)
    assert tracker.spent_since(0) is None
    assert boom.sql and "SUM(cost_kobo)" in boom.sql[0]  # query attempted
    assert tracker.spent_week() is None
    assert tracker.weekly_spend() == (
        "💸 WhatsApp spend data unavailable — "
        "the cost ledger couldn't be read.")
    assert tracker.spend_vs_revenue(47_000_00) == (
        "WhatsApp spend data unavailable — the cost ledger "
        "couldn't be read, so there's no honest "
        "return-on-spend figure.")
    # budget-gated consumers: costs fail, budgets still read
    gated = _mem()
    gated.track("a", "marketing")
    gated.set_budget("default", 100_000, hard=True)
    selective = _SelectiveFailingDB(gated._db, "wa_costs")
    gated._db = selective
    assert gated.check_budget("default") == (False, 0, 100_000)
    assert any("SUM(cost_kobo)" in s for s in selective.sql)
    assert gated.project_monthly() == {
        "spent_7d_kobo": 0, "projected_30d_kobo": 0,
        "budget_monthly_kobo": 0, "on_track": True,
        "summary": "projection unavailable",
    }
    assert gated.budget_warning("default") == (
        "⚠️ WhatsApp spend data unavailable — "
        "the cost ledger couldn't be read.")
    # unknown-spend branches: soft budget warns-not-blocks,
    # no budget stays unlimited (spent wart pinned as 0)
    soft = _mem()
    soft.set_budget("default", 100_000, hard=False)
    soft._db = _SelectiveFailingDB(soft._db, "wa_costs")
    assert soft.check_budget("default") == (True, 0, 100_000)
    free = _mem()
    free._db = _SelectiveFailingDB(free._db, "wa_costs")
    assert free.check_budget("default") == (True, 0, 0)
    assert _snap() == before


def test_255_project_monthly_failure_returns_unavailable_dict(monkeypatch):
    before = _snap()
    tracker = _mem()
    tracker.set_budget("default", 100_000)
    tracker.track("a", "marketing")
    proj = tracker.project_monthly()
    assert proj["projected_30d_kobo"] > 0 and proj["on_track"] is True  # control
    calls = []

    def bad_spent_week(*a, **k):
        calls.append((a, k))
        raise RuntimeError("spent_week boom")

    monkeypatch.setattr(tracker, "spent_week", bad_spent_week)
    assert tracker.project_monthly() == {
        "spent_7d_kobo": 0, "projected_30d_kobo": 0,
        "budget_monthly_kobo": 0, "on_track": True,
        "summary": "projection unavailable",
    }
    assert calls  # spent_week was attempted inside project_monthly
    assert _snap() == before


def test_269_message_count_db_failure_returns_none_converted():
    # CONVERTED with 222 (parent ruling): a count-only failure
    # makes weekly_spend's shared sentence unavailable.
    before = _snap()
    tracker = _mem()
    tracker.track("a", "service")
    tracker.track("b", "service")
    assert tracker.message_count() == 2  # control
    assert tracker.weekly_spend() == (
        "💸 Devon spent ₦28 on WhatsApp this week (2 messages).")
    selective = _SelectiveFailingDB(tracker._db, "COUNT(*)")
    tracker._db = selective
    assert tracker.message_count() is None
    assert selective.sql and "COUNT(*)" in selective.sql[0]  # attempted
    # the spend read still succeeds, yet the line is unavailable
    assert tracker.spent_week() == 2 * WC.RATE_SERVICE_KOBO
    assert tracker.weekly_spend() == (
        "💸 WhatsApp spend data unavailable — "
        "the cost ledger couldn't be read.")
    assert _snap() == before


def test_304_get_budget_db_failure_returns_none(monkeypatch):
    before = _snap()
    tracker = _mem()
    tracker.set_budget("shop", 50_000, hard=True)
    assert tracker.get_budget("shop") == {
        "client": "shop", "amount_kobo": 50_000, "hard": True}  # control
    boom = _ExplodingDB()
    monkeypatch.setattr(tracker, "_db", boom)
    assert tracker.get_budget("shop") is None
    assert boom.sql and "wa_budgets" in boom.sql[0]  # SELECT attempted
    assert _snap() == before


def test_326_check_budget_failure_fails_open(monkeypatch):
    before = _snap()
    tracker = _mem()
    tracker.set_budget("shop", WC.RATE_SERVICE_KOBO)
    tracker.track("a", "service", client="shop")
    assert tracker.check_budget("shop") == (
        False, WC.RATE_SERVICE_KOBO, WC.RATE_SERVICE_KOBO)  # control: blocks
    calls = []

    def bad_get_budget(*a, **k):
        calls.append((a, k))
        raise RuntimeError("get_budget boom")

    monkeypatch.setattr(tracker, "get_budget", bad_get_budget)
    assert tracker.check_budget("shop") == (True, 0, 0)
    assert calls  # get_budget was attempted inside check_budget
    assert _snap() == before


def test_345_budget_warning_failure_returns_empty(monkeypatch):
    before = _snap()
    tracker = _mem()
    tracker.set_budget("shop", WC.RATE_MARKETING_KOBO * 10)
    for i in range(8):
        tracker.track(f"p{i}", "marketing", client="shop")
    warn = tracker.budget_warning("shop")
    assert "⚠️" in warn and "shop" in warn  # control: 80% consumed warns
    calls = []

    def bad_get_budget(*a, **k):
        calls.append((a, k))
        raise RuntimeError("get_budget boom")

    monkeypatch.setattr(tracker, "get_budget", bad_get_budget)
    assert tracker.budget_warning("shop") == ""
    assert calls  # get_budget was attempted inside budget_warning
    assert _snap() == before


def test_558_worth_it_unconvertible_returns_false():
    before = _snap()
    int_log = []

    class BadInt:
        def __int__(self):
            int_log.append("int")
            raise RuntimeError("unconvertible")

        __index__ = __int__

    assert WC.worth_it(BadInt(), WC.RATE_MARKETING_KOBO) is False
    assert int_log == ["int"]  # the int() conversion was attempted
    # controls: the same False is the genuine not-worth-it value,
    # and a genuinely worth-it input still returns True
    assert WC.worth_it(10_000, WC.RATE_MARKETING_KOBO) is False
    assert WC.worth_it(47_000_00 * 0.30, WC.RATE_MARKETING_KOBO) is True
    assert _snap() == before
