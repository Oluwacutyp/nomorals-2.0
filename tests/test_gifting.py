"""Tests for player-to-player gifting (/gift)."""

import sqlite3
import unittest

from nomorals.games.gear import GearStore
from nomorals.games.gifting import (
    GIFT_EXPIRY_S,
    GiftStore,
    resolve_recipient,
)
from nomorals.games.players import Player, PlayerStore


class _Q:
    """Minimal DB shim matching the Database surface the stores use."""

    def __init__(self, conn):
        self.conn = conn

    def execute(self, sql, params=()):
        self.conn.execute(sql, params)
        self.conn.commit()

    def query(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        cols = [d[0] for d in cur.description] if cur.description else []
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def query_one(self, sql, params=()):
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def transaction(self):
        from contextlib import contextmanager

        @contextmanager
        def tx():
            yield
            self.conn.commit()

        return tx()


def _db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    q = _Q(conn)
    # game_players schema (mirrors migrations.py; PlayerStore relies on it)
    q.execute(
        "CREATE TABLE IF NOT EXISTS game_players ("
        "player_key TEXT PRIMARY KEY, platform TEXT NOT NULL DEFAULT '', "
        "display TEXT NOT NULL DEFAULT '', coins INTEGER NOT NULL DEFAULT 0, "
        "points INTEGER NOT NULL DEFAULT 0, wins INTEGER NOT NULL DEFAULT 0, "
        "losses INTEGER NOT NULL DEFAULT 0, draws INTEGER NOT NULL DEFAULT 0, "
        "streak INTEGER NOT NULL DEFAULT 0, "
        "best_streak INTEGER NOT NULL DEFAULT 0, "
        "games_played INTEGER NOT NULL DEFAULT 0, xp INTEGER NOT NULL "
        "DEFAULT 0, per_game TEXT NOT NULL DEFAULT '{}', "
        "items TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL "
        "DEFAULT 0, updated_at REAL NOT NULL DEFAULT 0)")
    return q


def _players(db):
    store = PlayerStore(db)
    ada = Player(key="telegram:111", platform="telegram", name="Ada")
    bob = Player(key="telegram:222", platform="telegram", name="Bob")
    store.get(ada.key, name=ada.name, platform=ada.platform)
    store.get(bob.key, name=bob.name, platform=bob.platform)
    return store, ada, bob


class ResolveRecipientTests(unittest.TestCase):
    def test_exact_match(self):
        store, ada, bob = _players(_db())
        prof, err = resolve_recipient(store, "@bob")
        self.assertIsNotNone(prof)
        self.assertEqual(prof.key, bob.key)
        self.assertEqual(err, "")

    def test_case_insensitive(self):
        store, ada, bob = _players(_db())
        prof, _ = resolve_recipient(store, "@BOB")
        self.assertIsNotNone(prof)
        self.assertEqual(prof.key, bob.key)

    def test_no_at_sign_ok(self):
        store, ada, bob = _players(_db())
        prof, _ = resolve_recipient(store, "bob")
        self.assertIsNotNone(prof)

    def test_unknown(self):
        store, ada, bob = _players(_db())
        prof, err = resolve_recipient(store, "@nobody")
        self.assertIsNone(prof)
        self.assertIn("no player found", err)

    def test_empty(self):
        store, ada, bob = _players(_db())
        prof, err = resolve_recipient(store, "")
        self.assertIsNone(prof)
        self.assertIn("gift to whom", err)


class GiftStoreTests(unittest.TestCase):
    def test_create_and_pending(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        g = gs.create(ada.key, ada.name, bob.key, bob.name,
                      "coins", "100", amount=100)
        pend = gs.pending_for(ada.key)
        self.assertIsNotNone(pend)
        self.assertEqual(pend.amount, 100)
        self.assertEqual(pend.recipient_key, bob.key)

    def test_new_replaces_old(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        g1 = gs.create(ada.key, ada.name, bob.key, bob.name,
                       "coins", "100", amount=100)
        g2 = gs.create(ada.key, ada.name, bob.key, bob.name,
                       "coins", "50", amount=50)
        pend = gs.pending_for(ada.key)
        self.assertEqual(pend.id, g2.id)
        self.assertEqual(pend.amount, 50)

    def test_cancel(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        gs.create(ada.key, ada.name, bob.key, bob.name,
                  "coins", "100", amount=100)
        dropped = gs.cancel(ada.key)
        self.assertIsNotNone(dropped)
        self.assertIsNone(gs.pending_for(ada.key))

    def test_cancel_nothing(self):
        db = _db()
        gs = GiftStore(db)
        self.assertIsNone(gs.cancel("nobody"))

    def test_mark_done_clears_pending(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        g = gs.create(ada.key, ada.name, bob.key, bob.name,
                      "coins", "100", amount=100)
        gs.mark_done(g.id)
        self.assertIsNone(gs.pending_for(ada.key))

    def test_history(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        g = gs.create(ada.key, ada.name, bob.key, bob.name,
                      "coins", "100", amount=100)
        gs.mark_done(g.id)
        hist = gs.history(ada.key)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0].kind, "coins")
        # recipient sees it too
        hist_b = gs.history(bob.key)
        self.assertEqual(len(hist_b), 1)

    def test_expiry(self):
        import time
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        g = gs.create(ada.key, ada.name, bob.key, bob.name,
                      "coins", "100", amount=100)
        # fake age past expiry
        db.execute(
            "UPDATE game_gifts SET created_at = ? WHERE id = ?",
            (time.time() - GIFT_EXPIRY_S - 1, g.id))
        self.assertIsNone(gs.pending_for(ada.key))

    def test_describe(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GiftStore(db)
        g = gs.create(ada.key, ada.name, bob.key, bob.name,
                      "coins", "100", amount=100)
        self.assertIn("100 coins", g.describe())
        self.assertIn("Bob", g.describe())


class CoinTransferTests(unittest.TestCase):
    def test_spend_then_add(self):
        db = _db()
        store, ada, bob = _players(db)
        store.add_coins(ada, 500, "test")
        new_bal = store.spend_coins(ada, 100, "gift:out")
        self.assertEqual(new_bal, 400)
        store.add_coins(bob, 100, "gift:in")
        self.assertEqual(store.get(bob.key).coins, 100)
        self.assertEqual(store.get(ada.key).coins, 400)

    def test_insufficient(self):
        db = _db()
        store, ada, bob = _players(db)
        self.assertIsNone(store.spend_coins(ada, 100, "gift:out"))


class GearTransferTests(unittest.TestCase):
    def test_transfer_moves_ownership(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GearStore(db)
        inst = gs.grant(ada.key, "katana_common")
        ok, name = gs.transfer(inst.id, bob.key)
        self.assertTrue(ok)
        self.assertIn("Katana", name)
        self.assertTrue(any(i.id == inst.id for i in gs.list(bob.key)))
        self.assertFalse(any(i.id == inst.id for i in gs.list(ada.key)))

    def test_transfer_unequips(self):
        db = _db()
        store, ada, bob = _players(db)
        gs = GearStore(db)
        inst = gs.grant(ada.key, "katana_common")
        gs.equip(ada.key, "katana_common")
        self.assertTrue(gs.equipped(ada.key))
        ok, _ = gs.transfer(inst.id, bob.key)
        self.assertTrue(ok)
        # recipient gets it unequipped
        got = gs.get(inst.id)
        self.assertFalse(got.equipped)
        self.assertEqual(got.player_key, bob.key)

    def test_transfer_missing(self):
        db = _db()
        gs = GearStore(db)
        ok, msg = gs.transfer("nope", "telegram:222")
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
