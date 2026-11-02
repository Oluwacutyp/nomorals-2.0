"""Tail-101 pins: nomorals/commerce/cart_recovery.py census sites.

All four sites are held as exclusions:

235 — _init_schema's migration loop: ALTER TABLE ... ADD COLUMN
raises sqlite3.OperationalError when the column already exists ->
pass ("column already there"). This is the idempotent-migration
idiom; the except is narrow (OperationalError only) and a second
construction over an existing DB is the normal restart path.
630 — stats(): any failure inside the seven-day aggregate ->
return {}. Absence, not fabrication: no figure is invented, and
stats has no programmatic consumer in-tree (the one named
downstream, whatsapp_cost.spend_vs_revenue, receives
recovered_kobo_7d as an explicit argument). A broken DB fails
every other store method loudly on its own path.
644 — _cancel_step(): the cancel UPDATE crashes -> pass, no
recording. The step row honestly stays 'pending'; nothing reads
'pending' as 'cancelled'. The send decision is re-guarded on
every run_step by a fresh read of the carts table (status must
still be 'abandoned' and opted_in), so a failed cancel can never
let a message out — the next tick retries the cancel instead.
650 — _row_to_cart(): a corrupt items_json -> items=[]. The
rest of the row (phone, totals, status) survives intact; the
items list is rendered detail, and the module's own writer only
ever stores json.dumps output, so a parse failure means external
corruption, surfaced as an empty list rather than a crash.

NOT-recording is pinned with the snapshot method: none of these
sites call a failures-ledger record function, so both ledgers
must be unchanged across each trigger.
"""
from __future__ import annotations

import sqlite3

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.commerce.cart_recovery import CartRecovery


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _store(tmp_path, name="cart.db"):
    return CartRecovery(db_path=str(tmp_path / name),
                        sender=lambda phone, msg: True)


def _seed_cart(store, cart_id="c1"):
    return store.cart_abandoned(
        cart_id, "+2348000000000",
        [{"name": "Ankara fabric", "qty": 1}], 500_00, True)


# ── 235: re-opening an existing DB — migrations idempotent ─────

def test_235_second_construction_survives_existing_columns(tmp_path):
    before = _snap()
    first = _store(tmp_path)
    _seed_cart(first)
    first._db.close()
    # every ALTER in the migration loop now hits "duplicate column"
    second = _store(tmp_path)
    cart = second.cart_abandoned(
        "c2", "+2348000000001", [], 100_00, True)
    assert cart.cart_id == "c2"
    assert second.stats()["abandoned_7d"] >= 1
    assert _snap() == before


# ── 630: stats failure -> empty dict, no invented figures ──────

def test_630_stats_on_broken_db_returns_empty(tmp_path):
    before = _snap()
    store = _store(tmp_path)
    _seed_cart(store)
    store._db.close()  # every query now raises ProgrammingError
    assert store.stats() == {}
    assert _snap() == before


def test_630_stats_control_reports_real_counts(tmp_path):
    store = _store(tmp_path)
    _seed_cart(store)
    out = store.stats()
    assert out["abandoned_7d"] == 1
    assert out["recovered_7d"] == 0
    assert out["recovered_kobo_7d"] == 0


# ── 644: cancel write failure -> row honestly stays pending ────

class _BoomDB:
    def execute(self, *a, **k):
        raise sqlite3.OperationalError("disk gone")

    def commit(self):
        raise sqlite3.OperationalError("disk gone")


def _pending_step(store, cart_id="c1"):
    # cart_abandoned already schedules step rows; force step 1
    # back to pending (insert only if scheduling skipped it).
    cur = store._db.execute(
        "UPDATE cart_steps SET status='pending' "
        "WHERE cart_id=? AND step=1", (cart_id,))
    if cur.rowcount == 0:
        store._db.execute(
            "INSERT INTO cart_steps (cart_id, step, run_at, status) "
            "VALUES (?, 1, 0, 'pending')", (cart_id,))
    store._db.commit()


def _step_status(store, cart_id="c1"):
    return store._db.execute(
        "SELECT status FROM cart_steps WHERE cart_id=? AND step=1",
        (cart_id,)).fetchone()["status"]


def test_644_cancel_step_write_failure_keeps_pending(tmp_path):
    before = _snap()
    store = _store(tmp_path)
    _seed_cart(store)
    _pending_step(store)
    real_db = store._db
    store._db = _BoomDB()
    try:
        assert store._cancel_step("c1", 1) is None  # never raises
    finally:
        store._db = real_db
    assert _step_status(store) == "pending"  # nothing fabricated
    assert _snap() == before


def test_644_cancel_step_control_marks_cancelled(tmp_path):
    store = _store(tmp_path)
    _seed_cart(store)
    _pending_step(store)
    store._cancel_step("c1", 1)
    assert _step_status(store) == "cancelled"


# ── 650: corrupt items_json -> empty items, row survives ───────

def test_650_corrupt_items_json_yields_empty_items(tmp_path):
    before = _snap()
    store = _store(tmp_path)
    _seed_cart(store)
    store._db.execute(
        "UPDATE carts SET items_json='{broken' WHERE cart_id='c1'")
    store._db.commit()
    row = store._db.execute(
        "SELECT * FROM carts WHERE cart_id='c1'").fetchone()
    cart = store._row_to_cart(row)
    assert cart.items == []
    assert cart.cart_id == "c1"
    assert cart.total_kobo == 500_00
    assert _snap() == before


def test_650_row_to_cart_control_round_trips_items(tmp_path):
    store = _store(tmp_path)
    _seed_cart(store)
    row = store._db.execute(
        "SELECT * FROM carts WHERE cart_id='c1'").fetchone()
    cart = store._row_to_cart(row)
    assert cart.items == [{"name": "Ankara fabric", "qty": 1}]
