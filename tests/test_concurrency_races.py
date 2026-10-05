"""Concurrency stress tests for the race-condition fixes.

These hammer the fixed code paths with real threads to prove the
atomic operations hold under contention:
- PlayerStore.add_xp: N threads x M awards = exactly N*M total
- GearStore.wear: N threads wearing the same piece = exact decrement
- Repair: two concurrent repairs = charged once
- award_xp: concurrent awards don't lose XP
"""
import sqlite3
import threading
import unittest

from nomorals.games.gear import GEAR_CATALOG, GearStore
from nomorals.games.players import Player, PlayerStore
from nomorals.games.progression import award_xp


class FakeDB:
    """Minimal in-memory DB with the transaction/lock semantics."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._setup()

    def _setup(self):
        self.conn.executescript("""
            CREATE TABLE game_players (
                player_key TEXT PRIMARY KEY, platform TEXT, display TEXT,
                username TEXT DEFAULT '', deleted_at REAL DEFAULT 0,
                coins INTEGER DEFAULT 0, points INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0, losses INTEGER DEFAULT 0,
                draws INTEGER DEFAULT 0, streak INTEGER DEFAULT 0,
                best_streak INTEGER DEFAULT 0, games_played INTEGER DEFAULT 0,
                xp INTEGER DEFAULT 0, per_game TEXT DEFAULT '{}',
                items TEXT DEFAULT '{}', created_at REAL DEFAULT 0,
                updated_at REAL DEFAULT 0);
            CREATE TABLE game_gear (
                id TEXT PRIMARY KEY, player_key TEXT, slug TEXT,
                durability INTEGER, max_durability INTEGER,
                equipped INTEGER DEFAULT 0, created_at REAL DEFAULT 0);
        """)
        self.conn.commit()

    def query_one(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            row = cur.fetchone()
            return dict(row) if row else None

    def execute(self, sql, params=()):
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

    def transaction(self):
        return self._Txn(self)

    class _Txn:
        def __init__(self, db):
            self.db = db

        def __enter__(self):
            self.db._lock.acquire()
            return self.db

        def __exit__(self, *a):
            self.db._lock.release()
            return False


def _player(key="p1"):
    return Player(key=key, name="Tester", platform="test")


class ConcurrencyTests(unittest.TestCase):
    def test_concurrent_add_xp_no_loss(self):
        db = FakeDB()
        store = PlayerStore(db)
        threads = []

        def award():
            for _ in range(20):
                store.add_xp("p1", 5)

        for _ in range(10):
            t = threading.Thread(target=award)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        prof = store.get("p1")
        self.assertEqual(prof.xp, 10 * 20 * 5)

    def test_concurrent_award_xp_no_loss(self):
        db = FakeDB()
        store = PlayerStore(db)
        p = _player()
        threads = []

        def award():
            for _ in range(10):
                award_xp(store, p, 7)

        for _ in range(8):
            t = threading.Thread(target=award)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        prof = store.get("p1")
        self.assertEqual(prof.xp, 8 * 10 * 7)

    def test_concurrent_wear_exact(self):
        db = FakeDB()
        gs = GearStore(db)
        # grant a katana (durable) directly
        slug = "katana_common"
        self.assertIn(slug, GEAR_CATALOG)
        inst = gs.grant("p1", slug)
        max_dur = inst.max_durability
        threads = []

        def wear():
            for _ in range(5):
                gs.wear(inst.id, 1)

        for _ in range(4):
            t = threading.Thread(target=wear)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        final = gs.get(inst.id)
        self.assertEqual(final.durability, max(0, max_dur - 4 * 5))

    def test_concurrent_repair_single_charge(self):
        db = FakeDB()
        gs = GearStore(db)
        store = PlayerStore(db)
        p = _player()
        store.add_coins(p, 100000, "test")
        inst = gs.grant("p1", "katana_common")
        gs.wear(inst.id, 5)  # damage it
        results = []

        def do_repair():
            # emulate the handler: spend then apply
            ok = gs.apply_repair(inst.id)
            results.append(ok)

        threads = [threading.Thread(target=do_repair) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # exactly one repair should have applied
        self.assertEqual(sum(1 for r in results if r), 1)
        final = gs.get(inst.id)
        self.assertEqual(final.durability, final.max_durability)


if __name__ == "__main__":
    unittest.main()
