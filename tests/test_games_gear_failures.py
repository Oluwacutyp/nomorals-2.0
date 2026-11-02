"""Tail-88 pins: nomorals/games/gear.py census sites.

GearStore's posture: gear is never silently lost — wear only
decreases durability, breakage keeps the row, and the store's
read/migration paths degrade to the module's own absent values
instead of raising into game flow. Every site is held as an
exclusion:

397 — ``_ensure`` ALTER-TABLE guard -> pass. Schema probe: the
affixes column is added post-launch via ALTER; on every table
that already has it the ALTER raises "duplicate column" and the
pass is the designed outcome (the sibling CREATE is IF NOT
EXISTS; the outer guard binds + debug-logs). A genuine ALTER
failure leaves the column missing, and downstream affixes reads
degrade to () via the row.get default while loot INSERTs fail
loudly — no success is claimed anywhere.
416 — ``_row_to_instance`` affixes JSON crash -> (). The
deserialization boundary: one corrupt affixes cell degrades
that piece to plain catalog stats; the piece itself
(durability, equipped, slug) survives. () is the genuine
plain-piece value, and raising here would break list() for
every piece over one damaged row.
465 — ``get`` query crash -> None. None is the function's own
not-found value; consumers use list/find/equipped (get has no
production caller), and no surface distinguishes "crashed"
from "absent" because the contract is "piece or None".
742 — ``_migrate_legacy`` per-grant crash -> break. Partial-
progress ladder: grant as many legacy pieces as possible, stop
this slug at the first failure; the returned moved count is
literally true and its only caller (list) discards it — no
"all migrated" claim exists. Tension, stated plainly and
shelved: the profile's legacy counts are popped regardless of
how many grants landed, so a grant failure followed by a
successful profile write loses the un-granted counts (probe-
proven). The pop is surrounding flow, not this site; a
pop-only-what-was-granted fix is a targeted-fix candidate.
755 — ``_migrate_legacy`` outer crash -> 0. 0 is the genuine
nothing-migrated value; the per-process _migrated mark is not
persisted, the profile is untouched on this path (the write
has its own bind+debug guard inside), so the next process
retries the migration whole.
"""
from __future__ import annotations

from nomorals.games import gear as G
from nomorals.games.players import PlayerStore
from nomorals.storage.db import Database


def _fresh_db():
    db = Database(":memory:")
    db.migrate()
    return db


# ── 397: ensure's ALTER probe never raises ──────────────────────

class _AlterBoomDb:
    def execute(self, sql, args=()):
        if "ALTER" in sql:
            raise RuntimeError("duplicate column")
        return None


def test_397_alter_crash_swallowed_by_ensure():
    # a db whose ALTER always fails: _ensure still returns
    G.GearStore(_AlterBoomDb())._ensure()
    # control: on a real db, ensure is idempotent and gear works
    store = G.GearStore(_fresh_db())
    store._ensure()
    store._ensure()
    inst = store.grant("p1", "katana_common")
    assert inst.slug == "katana_common"


# ── 416: corrupt affixes degrade to plain piece ─────────────────

_ROW = {"id": "g1", "player_key": "p1", "slug": "loot:katana_common:ab",
        "durability": 10, "max_durability": 40, "equipped": 0,
        "created_at": 0.0, "affixes": "{corrupt"}


def test_416_corrupt_affixes_become_empty():
    store = G.GearStore(_fresh_db())
    inst = store._row_to_instance(dict(_ROW))
    assert inst.affixes == ()
    assert inst.slug == "loot:katana_common:ab"
    assert inst.durability == 10
    # control: a well-formed affixes cell parses into tuples
    good = store._row_to_instance(
        dict(_ROW, affixes='[["Bronze", {"atk": 3}]]'))
    assert good.affixes == (("Bronze", {"atk": 3}),)


# ── 465: get crash reads as not-found ───────────────────────────

class _QueryOneBoomDb:
    def execute(self, sql, args=()):
        return None

    def query_one(self, sql, args=()):
        raise RuntimeError("db down")


def test_465_get_query_crash_returns_none():
    assert G.GearStore(_QueryOneBoomDb()).get("anything") is None
    # controls: a real piece is returned; a missing id is None
    store = G.GearStore(_fresh_db())
    inst = store.grant("p1", "katana_common")
    assert store.get(inst.id).id == inst.id
    assert store.get("no-such-id") is None


# ── 742: migration grant ladder breaks at first failure ─────────

def _legacy_profile(db, key, count):
    ps = PlayerStore(db)
    prof = ps.get(key)
    prof.items["gear_sword"] = count
    ps._write(prof)
    return ps


def test_742_grant_crash_breaks_ladder_partial_progress(monkeypatch):
    db = _fresh_db()
    _legacy_profile(db, "legacy1", 3)
    store = G.GearStore(db)
    orig_grant = G.GearStore.grant
    calls = {"n": 0}

    def flaky(self, pk, slug):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("insert boom")
        return orig_grant(self, pk, slug)

    monkeypatch.setattr(G.GearStore, "grant", flaky)
    # one piece lands, the ladder stops, the count is honest
    assert store._migrate_legacy("legacy1") == 1


def test_742_first_grant_crash_migrates_nothing(monkeypatch):
    db = _fresh_db()
    _legacy_profile(db, "legacy2", 2)
    store = G.GearStore(db)

    def dead(self, pk, slug):
        raise RuntimeError("insert boom")

    monkeypatch.setattr(G.GearStore, "grant", dead)
    assert store._migrate_legacy("legacy2") == 0


# ── 755: migration outer crash reads as nothing migrated ────────

class _StoreBoomDb:
    def execute(self, sql, args=()):
        return None

    def query(self, sql, args=()):
        raise RuntimeError("db down")

    def query_one(self, sql, args=()):
        raise RuntimeError("db down")


def test_755_migration_store_crash_returns_zero():
    assert G.GearStore(_StoreBoomDb())._migrate_legacy("x") == 0
    # control: a player with no legacy items genuinely migrates 0
    store = G.GearStore(_fresh_db())
    assert store._migrate_legacy("ghost") == 0
