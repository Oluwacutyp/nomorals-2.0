"""Pass #122 pins: the GameEngine failure ledger.

Every genuinely-silent GameEngine handler records into
``nomorals.games.engine_failures``; the 45 `_log.debug(..., exc_info=True)`
sites stay (honest logging with a traceback — the scanner's `debug`
blind spot), and the `__del__` teardown site stays (must never raise
during interpreter teardown). Return values, control flow, and fallback
defaults are unchanged: each pin asserts the fallback value is still
what the caller saw before, plus the record.
"""
import ast
import threading
from types import SimpleNamespace as NS

import pytest

from nomorals.games import engine as engine_mod
from nomorals.games.engine import GameEngine, render_game_over_card
from nomorals.games.engine_failures import (
    clear_failures,
    failure_count,
    format_failures,
    recent_failures,
    record_failure,
)
from nomorals.games.players import Player


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def sites():
    return [f.site for f in recent_failures(500)]


def player(key="k", name="Kay"):
    return Player(key=key, platform="local", name=name, is_ai=False)


def raising_store():
    class Store:
        def get_for(self, p):
            raise RuntimeError("store down")

    return Store()


def ok_profile(**kw):
    base = dict(games_played=0, streak=0, wins=0, xp=0, coins=0,
                level=1, items={})
    base.update(kw)
    return NS(**base)


def ok_store(profile=None):
    class Store:
        def get_for(self, p):
            return profile if profile is not None else ok_profile()

    return Store()


def bare_engine(**attrs):
    eng = object.__new__(GameEngine)
    eng.db = None
    eng._send = lambda *a: None
    eng._by_id = {}
    eng._last_game = {}
    for k, v in attrs.items():
        setattr(eng, k, v)
    return eng


def raising_fn(msg="boom"):
    def _raise(*a, **k):
        raise RuntimeError(msg)

    return _raise


# ---- ledger basics -------------------------------------------------------

def test_ledger_basics_and_format():
    assert recent_failures() == []
    assert failure_count() == 0
    assert format_failures() == "no recorded engine failures"
    record_failure("test.site", ValueError("bad value"))
    assert failure_count() == 1
    [entry] = recent_failures()
    assert entry.site == "test.site"
    assert entry.exc_type == "ValueError"
    assert entry.message == "bad value"
    assert "test.site: ValueError: bad value" in format_failures()


def test_ledger_bound_keeps_newest():
    for _ in range(250):
        record_failure("flood", RuntimeError("x"))
    assert failure_count() == 250
    assert len(recent_failures(500)) == 200


def test_ledger_unprintable_exc_safe():
    class BadStr(Exception):
        def __str__(self):
            raise RuntimeError("no str for you")

    record_failure("bad.str", BadStr())
    [entry] = recent_failures()
    assert entry.exc_type == "BadStr"
    assert "unprintable" in entry.message


def test_direct_import_surface_and_all_unchanged():
    import nomorals.games.engine_failures as ledger

    assert ledger.__all__ == ["Failure", "record_failure", "recent_failures",
                             "failure_count", "clear_failures",
                             "format_failures"]
    assert engine_mod.__all__ == ["GameEngine", "SendFn"]


# ---- render_game_over_card ----------------------------------------------

def test_store_coins_failure_omits_bits_but_records():
    card = render_game_over_card("g", [player()], {"k": 5},
                                 store=raising_store(), db=None)
    assert "5 pts" in card
    assert "render_game_over_card:store_coins" in sites()


def test_glicko_failure_omits_tier_but_records(monkeypatch):
    import nomorals.games.matchmaking as mm

    monkeypatch.setattr(mm, "get_glicko", raising_fn("glicko down"))
    card = render_game_over_card("g", [player()], {"k": 5},
                                 store=ok_store(), db=object())
    assert "5 pts" in card
    assert "render_game_over_card:glicko" in sites()


def test_outer_render_failure_returns_empty_and_records():
    assert render_game_over_card("g", None, {}, db=None) == ""
    assert "render_game_over_card:outer" in sites()


# ---- dm feed / live load --------------------------------------------------

def test_ensure_cast_failure_narrates_flat_and_records():
    eng = bare_engine(_dm_obj=NS(ensure_cast=raising_fn("cast down"),
                                  announce=lambda name, feed, cast=None: "scene"))
    room = NS(kind="group", state={"_dm_feed": [{"text": "x"}]})
    out = []
    eng._drain_dm_feed(room, NS(name="g"), out, force=True)
    assert out == ["scene"]
    assert "_drain_dm_feed:ensure_cast" in sites()


def test_load_live_db_failure_returns_none_and_records():
    class Db:
        def query_one(self, *a, **k):
            raise RuntimeError("db down")

    eng = bare_engine(db=Db(), _rooms={}, _by_id={})
    assert eng._load_live("chat:1") is None
    assert "_load_live:db_query" in sites()


def test_load_live_missing_row_returns_none_without_record():
    class Db:
        def query_one(self, *a, **k):
            return None

    eng = bare_engine(db=Db(), _rooms={}, _by_id={})
    assert eng._load_live("chat:1") is None
    assert sites() == []


# ---- start / mastery -------------------------------------------------------

def test_start_mastery_failure_starts_at_tier_zero_and_records(monkeypatch):
    import nomorals.games.mastery as mastery

    monkeypatch.setattr(mastery, "get_one_game_stats",
                        raising_fn("stats down"))
    captured = {}

    def new_state(rng, **kw):
        captured.update(kw)
        return {}

    fake_game = NS(name="fakegame", needs_group=False, min_players=1,
                   max_players=2, ai_seats=0, channel_mode="x",
                   rng=lambda room: None, new_state=new_state,
                   setup=lambda room, mind: "welcome", howto="",
                   fair=False)
    eng = bare_engine(_lock=threading.RLock(), _rooms={}, _by_id={},
                      games={"fakegame": fake_game}, store=raising_store(),
                      gear=NS(list=raising_fn("gear down")),
                      _mind=NS(), _forge_obj=NS())
    room, msgs = eng.start("local:1", "fakegame", player())
    assert captured["mastery"] == 0
    assert "start:mastery" in sites()


# ---- mirrors ---------------------------------------------------------------

def test_mirror_inventory_and_gear_fallbacks_record():
    eng = bare_engine(store=raising_store(),
                      gear=NS(list=raising_fn("gear down")))
    room = NS(state={})
    eng._mirror_player(room, player())
    assert room.state["inventory"] == {"k": {}}
    assert room.state["loadout"] == {"k": {}}
    assert room.state["gear_closet"] == {"k": []}
    assert "_mirror_player:inventory" in sites()
    assert "_mirror_player:gear" in sites()


def test_mirror_level_points_failure_records(monkeypatch):
    import nomorals.games.stats as stats_mod

    class FakeStats:
        def to_dict(self):
            return {}

    class FakeStatStore:
        def __init__(self, db):
            pass

        def get(self, key):
            return FakeStats()

        def grant_level_points(self, key, level):
            raise RuntimeError("grant down")

    monkeypatch.setattr(stats_mod, "StatStore", FakeStatStore)
    eng = bare_engine(store=raising_store(),
                      gear=NS(list=raising_fn("gear down")))
    room = NS(state={})
    eng._mirror_player(room, player())
    assert "_mirror_player:level_points" in sites()


# ---- move hook ---------------------------------------------------------------

def test_bad_move_hook_never_breaks_chat_but_records():
    eng = bare_engine()
    eng._is_turn = lambda room, sender: False
    room = NS(kind="group", current=player("c", "Cur"), state={})
    game = NS(channel_mode="x", is_move_text=raising_fn("hook down"))
    assert eng._move_inner(room, game, "chat", "hello", player()) == []
    assert "_move_inner:is_move_text" in sites()


# ---- finish -------------------------------------------------------------------

def finish_engine(**store_kw):
    return bare_engine(store=store_kw.get("store", raising_store()))


def finish_game(**kw):
    base = dict(final_message=lambda room, mind: "done",
                winner=lambda room: None,
                score=lambda room, p: 10,
                coin_payout=lambda room, p, **k: (5, "ok"),
                difficulty=raising_fn("difficulty down"),
                finish_won=raising_fn("hook down"),
                dm_finale=False)
    base.update(kw)
    return NS(**base)


def finish_room(players):
    return NS(status="active", state={}, players=players, game="g",
              chat_key="c", id="1", kind="group", messages=[])


def test_finish_final_message_failure_uses_generic_and_records():
    eng = finish_engine()
    eng.games = {"g": finish_game(final_message=raising_fn("final down"))}
    msgs, _ = eng._finish_inner(finish_room([]))
    assert "game over." in msgs
    assert "_finish_inner:final_message" in sites()


def test_finish_difficulty_and_streak_fallbacks_flow_into_settlement():
    captured = {}

    def coin_payout(room, p, **kw):
        captured.update(kw)
        return (5, "ok")

    eng = finish_engine()
    eng.games = {"g": finish_game(winner=lambda room: "all",
                                  coin_payout=coin_payout)}
    msgs, _ = eng._finish_inner(finish_room([player()]))
    assert "_finish_inner:difficulty" in sites()
    assert "_finish_inner:streak" in sites()
    # won=True (co-op "all") so the streak fallback is 1, not 0
    assert captured["difficulty"] == "normal"
    assert captured["streak_after"] == 1


# ---- achievements ---------------------------------------------------------------

def award_engine(**kw):
    eng = bare_engine(**kw)
    return eng


def test_award_profile_failure_skips_milestones_and_records():
    eng = award_engine(store=raising_store())
    room = NS(game="arena", state={})
    assert eng._award_achievements(room, None, player(), False, 0) == []
    assert "_award_achievements:profile" in sites()


def test_award_myth_set_failure_records():
    eng = award_engine(store=ok_store(),
                       gear=NS(equipped=raising_fn("equipped down")))
    room = NS(game="arena", state={})
    eng._award_achievements(room, None, player(), True, 0)
    assert "_award_achievements:myth_set" in sites()


def test_award_no_gear_failure_records():
    class RaisingDict(dict):
        def get(self, *a, **k):
            raise RuntimeError("loadout unreadable")

    eng = award_engine(store=ok_store(), gear=NS(equipped=lambda key: []))
    room = NS(game="arena", state={"loadout": RaisingDict()})
    eng._award_achievements(room, None, player(), True, 0)
    assert "_award_achievements:no_gear" in sites()


def test_award_loss_streak_reset_failure_records_without_wins_100():
    eng = award_engine(store=ok_store(ok_profile(streak=0, wins=50)),
                       gear=NS(equipped=lambda key: []))
    room = NS(game="arena", state={})
    eng._award_achievements(room, None, player(), True, 0)
    assert "_award_achievements:loss_streak_reset" in sites()
    assert "_award_achievements:wins_100" not in sites()


def test_award_wins_100_failure_records_inside_reset_handler():
    # wins_100 lives inside the loss-streak-reset handler by construction:
    # entering it always records the reset site first. wins is NaN so
    # the milestone comparisons (>= 25, >= 100) stay False while int()
    # raises inside the wins_100 try.
    eng = award_engine(store=ok_store(ok_profile(streak=30, wins=float("nan"))),
                       gear=NS(equipped=lambda key: []))
    room = NS(game="arena", state={})
    eng._award_achievements(room, None, player(), True, 0)
    assert "_award_achievements:loss_streak_reset" in sites()
    assert "_award_achievements:wins_100" in sites()


# ---- describe -------------------------------------------------------------------

def test_describe_state_failure_omits_extra_but_records():
    eng = bare_engine(games={"g": NS(describe_state=raising_fn("state down"))})
    room = NS(game="g", status="active", kind="group", state={},
              players=[], current=None)
    out = eng.describe(room)
    assert "🎮 g" in out
    assert "describe:state" in sites()


# ---- deliberate exclusion + remainder scan ---------------------------------------

def test_del_teardown_never_raises():
    # __del__ runs shutdown during interpreter teardown: it must swallow.
    eng = object.__new__(GameEngine)
    eng.__del__()  # shutdown raises AttributeError (no _wake) — swallowed


def test_scan_remainder_is_debug_honest_plus_del():
    import pathlib

    from nomorals.tools import error_scan as es

    engine_path = pathlib.Path(engine_mod.__file__)
    old = es._noqa_suppresses
    es._noqa_suppresses = lambda line, rule: False
    try:
        report = es.scan([str(engine_path)])
    finally:
        es._noqa_suppresses = old
    src = engine_path.read_text()
    tree = ast.parse(src)
    funcs = [(n.lineno, n.end_lineno or n.lineno, n.name)
             for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    handlers = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Try):
            for h in n.handlers:
                handlers[h.lineno] = h

    def enclosing(lineno):
        best = None
        for s, e, name in funcs:
            if s <= lineno <= e and (best is None or s > best[0]):
                best = (s, name)
        return best[1] if best else "?"

    expected = {
        "_scheduler_worker": 1, "_drain_dm_feed": 1, "_drain_dm_feed_safe": 1,
        "_sweep_timeouts": 2, "_maybe_prune_history": 1, "_emit": 1,
        "_persist_inner": 1, "_delete_room_row": 1, "start": 4,
        "_mirror_player": 4, "_move_inner": 1, "_finish_inner": 19,
        "_settle_won": 1, "_settle_coins": 1, "rematch": 1,
        "_grant_victory_loot": 2, "grant": 1, "_award_achievements": 2,
        "__del__": 1,
    }
    found = {}
    for f in report.findings:
        name = enclosing(f.line)
        found[name] = found.get(name, 0) + 1
        body = "\n".join(src.splitlines()[handlers[f.line].lineno - 1:
                                       handlers[f.line].end_lineno])
        if name == "__del__":
            assert body.strip().endswith("pass")
        else:
            assert "exc_info=True" in body, f"{name}:{f.line} not debug-honest"
    assert found == expected, found
    assert sum(found.values()) == 46
