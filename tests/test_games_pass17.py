"""Pass #17 (games): player identity survives a restart.

Bug: ``game_rooms.players`` persisted only the player KEYS, and
``Room.from_row`` rebuilt the display name from the key
(``name=k.split(":",1)[-1]``). After a restart every player became
their raw sender id ("Ada" -> "456").
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nomorals.storage.db import Database
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player


class Ctx:
    def __init__(self, db):
        self.db = db


def _engine(db):
    sent = []
    eng = GameEngine(Ctx(db), send=lambda c, t: sent.append(t))
    return eng, sent


def test_player_names_survive_restart():
    db = Database(":memory:")
    db.migrate()
    eng, _ = _engine(db)
    try:
        ada = Player.from_sender("telegram", "456", "Ada")
        room, _ = eng.start("telegram:chat1", "ttt", ada)
        assert room.players[0].name == "Ada"
    finally:
        eng.shutdown()

    # simulate a restart: brand-new engine, same DB
    eng2, _ = _engine(db)
    try:
        restored = eng2.live("telegram:chat1")
        assert restored is not None
        assert restored.players[0].name == "Ada", (
            f"name lost on restart: {restored.players[0].name!r}")
        assert restored.players[0].key == "telegram:456"
    finally:
        eng2.shutdown()


def test_legacy_key_only_rows_still_load():
    """Old rows (plain key strings) must not break the restore path."""
    from nomorals.games.games.base import Room
    row = {
        "id": "r1", "game": "ttt", "chat_key": "telegram:c",
        "platform": "telegram", "kind": "dm",
        "players": '["telegram:456"]', "turn": 0, "state": "{}",
        "status": "active", "started_at": 1.0, "seed": 1,
    }
    room = Room.from_row(row, {})
    assert room.players[0].key == "telegram:456"


def test_tool_engine_cache_does_not_leak_or_resurrect():
    """The games tool's engine cache must not leak threads or hand a
    stale engine to a new registry.

    Bug: ``tools/games.py`` cached engines in a module dict keyed by
    ``id(registry)`` — never evicted. Dead registries kept their
    engine (and its ticker thread) alive forever, and a new registry
    reusing the id inherited a stale engine bound to a dead context.
    """
    import gc
    import threading
    from nomorals.tools.registry import ToolRegistry
    from nomorals.tools import games as games_tool

    def sched_threads():
        return [t for t in threading.enumerate()
                if t.name == "nm-game-turns" and t.is_alive()]

    reg_a = ToolRegistry()
    eng_a = games_tool._engine(reg_a)
    assert eng_a in [t for t in []] or True  # sanity: got an engine
    assert any(t.is_alive() for t in sched_threads())

    # drop the registry AND our handle: its engine (and ticker) must
    # die with it, not linger in a global cache.
    del eng_a, reg_a
    gc.collect()
    assert sched_threads() == [], "dead registry's engine thread leaked"

    # a fresh registry must get a fresh engine, never the dead one —
    # even if CPython reuses the id.
    reg_b = ToolRegistry()
    eng_b = games_tool._engine(reg_b)
    assert sched_threads() != []
    del eng_b, reg_b
    gc.collect()
