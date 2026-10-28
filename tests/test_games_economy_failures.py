"""Tail-28 pins: games/economy.py failure values + NOT-recording.

Verdict: pins-only, 0 converts / 9 exclusions. Every handler below is
documented never-raises (or best-effort); each pin drives the handler
into its except path and pins the exact fallback value plus the fact
that the agents failures ledger is unchanged (snapshot before/after,
tail-8 method). economy.py calls no record function anywhere
(grep-proven), so the ledger must not move.

Grounds in brief (full case in SURVEY.md): 340/374 are fire-and-forget
audit writes — a lost row narrows the faucet/drain diagnosis, never a
balance claim (auditable receipts live in game_wallet, written by the
player store, not game_ledger); 357/455 are schema setup whose callers
handle the downstream failure themselves; 396/466/548 return the
documented no-data values (zeros + "no data yet", 0, 0.0) and have NO
in-repo consumer today — the 396 caveat: if a consumer ever renders
economy_health, the verdict needs a distinct "unavailable" value, since
a failed query is indistinguishable from an empty ledger (latent, not
live); 484 returns the CURRENT balance (unchanged by the failed grant),
never a success flag; 501 fails closed (False, gems unspent, reason
wart only).

The pins file itself contains no except handlers, so it adds zero
census sites.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents import failures as ag_fail
from nomorals.games import economy as E
from nomorals.storage.db import Database


def _ledger():
    return list(ag_fail.recent_failures())


class _RaisingDB:
    def execute(self, *a, **k):
        raise RuntimeError("db down (execute)")

    def query(self, *a, **k):
        raise RuntimeError("db down (query)")

    def query_one(self, *a, **k):
        raise RuntimeError("db down (query_one)")


def _real_db():
    return Database(str(Path(tempfile.mkdtemp()) / "g.db"))


def test_ledger_sink_db_failure_is_silent():
    """340: failing sink record never raises; working db records the row."""
    before = _ledger()
    assert E._ledger_sink(SimpleNamespace(db=_RaisingDB()),
                          SimpleNamespace(key="p1"), 30, "shop:shield") is None
    db = _real_db()
    E._ledger_sink(SimpleNamespace(db=db), SimpleNamespace(key="p1"),
                   30, "shop:shield")
    rows = db.query("SELECT kind, amount FROM game_ledger")
    assert len(rows) == 1 and rows[0]["kind"] == "sink"
    assert _ledger() == before


def test_ensure_ledger_db_failure_is_silent():
    """357: failing schema setup never raises; working db creates it."""
    before = _ledger()
    assert E._ensure_ledger(_RaisingDB()) is None
    db = _real_db()
    E._ensure_ledger(db)
    assert len(db.query(
        "SELECT name FROM sqlite_master WHERE name='game_ledger'")) == 1
    assert _ledger() == before


def test_record_ledger_db_failure_is_silent():
    """374: failing INSERT never raises; working db records the row."""
    before = _ledger()
    assert E.record_ledger(_RaisingDB(), "p1", "source", 10, "win") is None
    db = _real_db()
    E.record_ledger(db, "p1", "source", 10, "win")
    rows = db.query("SELECT amount FROM game_ledger")
    assert len(rows) == 1 and rows[0]["amount"] == 10
    assert _ledger() == before


def test_economy_health_query_failure_returns_no_data():
    """396: query failure -> zeros + "no data yet" (control: real ratio)."""
    before = _ledger()
    h = E.economy_health(_RaisingDB())
    assert h["sources"] == 0 and h["sinks"] == 0
    assert h["verdict"] == "no data yet"
    db = _real_db()
    E.record_ledger(db, "p1", "source", 100, "win")
    E.record_ledger(db, "p1", "sink", 50, "shop")
    h = E.economy_health(db)
    assert h["sources"] == 100 and h["sinks"] == 50 and h["ratio"] == 2.0
    assert "INFLATION" in h["verdict"]
    assert _ledger() == before


def test_ensure_gems_db_failure_is_silent():
    """455: failing schema setup never raises; working db creates it."""
    before = _ledger()
    assert E._ensure_gems(_RaisingDB()) is None
    db = _real_db()
    E._ensure_gems(db)
    assert len(db.query(
        "SELECT name FROM sqlite_master WHERE name='game_gems'")) == 1
    assert _ledger() == before


def test_gem_balance_query_failure_returns_zero():
    """466: query failure -> 0 (control: seeded balance)."""
    before = _ledger()
    assert E.gem_balance(_RaisingDB(), "p1") == 0
    db = _real_db()
    E.grant_gems(db, "p1", 42, "test")
    assert E.gem_balance(db, "p1") == 42
    assert _ledger() == before


def test_grant_gems_insert_failure_returns_zero():
    """484: failed grant -> current balance (0 on dead db), no success flag."""
    before = _ledger()
    assert E.grant_gems(_RaisingDB(), "p1", 10, "test") == 0
    db = _real_db()
    assert E.grant_gems(db, "p1", 10, "test") == 10
    assert E.gem_balance(db, "p1") == 10
    assert _ledger() == before


def test_spend_gems_update_failure_returns_false():
    """501: failed UPDATE -> False, gems unspent (fail-closed).

    Selective stub: query_one works (balance 50) so the UPDATE except
    is genuinely exercised, not bypassed by the balance check.
    """
    before = _ledger()

    class _UpdateFailsDB:
        def execute(self, *a, **k):
            raise RuntimeError("db down (update)")

        def query(self, *a, **k):
            return []

        def query_one(self, *a, **k):
            return {"gems": 50}

    assert E.spend_gems(_UpdateFailsDB(), "p1", 20) is False
    db = _real_db()
    E.grant_gems(db, "p1", 50, "test")
    assert E.spend_gems(db, "p1", 20) is True
    assert E.gem_balance(db, "p1") == 30
    assert E.spend_gems(db, "p1", 999) is False
    assert _ledger() == before


def test_earn_velocity_query_failure_returns_zero():
    """548: query failure -> 0.0, so dynamic_price keeps the base cost."""
    before = _ledger()
    assert E.earn_velocity(_RaisingDB()) == 0.0
    db = _real_db()
    E.record_ledger(db, "p1", "source", 100, "win")
    assert E.earn_velocity(db) == 100.0 / 1 / 7
    assert _ledger() == before
