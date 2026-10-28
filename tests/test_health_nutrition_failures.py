"""Pins for nomorals/health/nutrition.py error sites (tail-13).

9 exclusions: failure degrades to a documented miss value, an honest
failure string, or a per-row empty value with the row preserved — and
nothing is recorded into the health failures ledger. 1 convert (599):
a failed water write used to return 0, which the module-level log_water
then confirmed as a successful "+Xml" log with a fabricated 0 total;
it now returns -1 and the module function reports the failure honestly.
"""
import os
import tempfile

from nomorals.health import nutrition as N
from nomorals.health.failures import recent_failures
from nomorals.health.nutrition import (
    DayCard,
    NutritionStore,
    daily_totals,
    log_water as module_log_water,
    repeat_meal,
)


def _ledger_snapshot():
    return [(f.site, f.exc_type, f.message) for f in recent_failures()]


def _store_with_rows(rows):
    d = tempfile.mkdtemp()
    s = NutritionStore(db_path=os.path.join(d, "n.db"))
    for rid, items_json in rows:
        s._db.execute(
            "INSERT INTO meals VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, "2026-10-28", items_json, 500, 700,
             20.0, 30.0, 50.0, 70.0, 15.0, 25.0, 1.0))
    s._db.commit()
    return s


class _BrokenDB:
    def execute(self, *args, **kwargs):
        raise RuntimeError("boom")

    def commit(self):
        raise RuntimeError("boom")


def _broken_store():
    s = NutritionStore(db_path=os.path.join(tempfile.mkdtemp(), "n.db"))
    s._db = _BrokenDB()
    return s


def _db_none_store():
    s = NutritionStore.__new__(NutritionStore)
    s._db = None
    return s


class _RaisingStore:
    def meals_on(self, day):
        raise RuntimeError("boom")

    def water_on(self, day):
        raise RuntimeError("boom")

    def log_water(self, ml, day=""):
        raise RuntimeError("boom")

    def recent_meals(self, limit=10):
        raise RuntimeError("boom")


def test_561_corrupt_items_row_keeps_row_and_empties_items():
    before = _ledger_snapshot()
    s = _store_with_rows([("meal_good", '[{"name": "rice"}]'),
                          ("meal_bad", "{not json")])
    rows = s.meals_on("2026-10-28")
    assert len(rows) == 2
    assert rows[0]["items"] == [{"name": "rice"}]
    assert rows[1]["items"] == []
    assert _ledger_snapshot() == before


def test_565_broken_db_meals_on_empty():
    before = _ledger_snapshot()
    assert _broken_store().meals_on("2026-10-28") == []
    assert _ledger_snapshot() == before


def test_581_corrupt_items_row_recent_empties_items():
    before = _ledger_snapshot()
    s = _store_with_rows([("meal_bad", "{not json")])
    rows = s.recent_meals()
    assert len(rows) == 1 and rows[0]["items"] == []
    assert _ledger_snapshot() == before


def test_585_broken_db_recent_meals_empty():
    before = _ledger_snapshot()
    assert _broken_store().recent_meals() == []
    assert _ledger_snapshot() == before


def test_599_failed_water_write_signals_failure_not_zero():
    # Convert pin: a failed write must not read back as a 0 total that
    # the module function then confirms as a successful log.
    before = _ledger_snapshot()
    assert _broken_store().log_water(500) == -1
    assert _db_none_store().log_water(500) == -1
    assert module_log_water(500, store=_broken_store()) == \
        "couldn't log water right now."
    assert module_log_water(500, store=_db_none_store()) == \
        "couldn't log water right now."
    assert _ledger_snapshot() == before


def test_599_success_path_unchanged():
    s = NutritionStore(db_path=os.path.join(tempfile.mkdtemp(), "n.db"))
    assert s.log_water(500, day="2026-10-28") == 500
    assert module_log_water(250, store=s, day="2026-10-28") == \
        "💧 +250ml water — 750/2500ml today."


def test_610_broken_db_water_on_zero():
    before = _ledger_snapshot()
    assert _broken_store().water_on("2026-10-28") == 0
    assert _ledger_snapshot() == before


def test_648_broken_targets_failure_string():
    before = _ledger_snapshot()
    old = N.MACRO_TARGETS
    N.MACRO_TARGETS = {}
    try:
        assert DayCard(day="2026-10-28").format() == \
            "couldn't build today's card."
    finally:
        N.MACRO_TARGETS = old
    assert _ledger_snapshot() == before


def test_672_raising_store_daily_totals_empty_card():
    before = _ledger_snapshot()
    card = daily_totals(store=_RaisingStore(), day="2026-10-28")
    assert card.meals == 0 and card.water_ml == 0
    assert _ledger_snapshot() == before


def test_684_raising_store_module_log_water_failure_string():
    before = _ledger_snapshot()
    assert module_log_water(500, store=_RaisingStore()) == \
        "couldn't log water right now."
    assert _ledger_snapshot() == before


def test_715_raising_store_repeat_meal_error():
    before = _ledger_snapshot()
    draft = repeat_meal(store=_RaisingStore())
    assert draft.error == "couldn't repeat that meal"
    assert _ledger_snapshot() == before
