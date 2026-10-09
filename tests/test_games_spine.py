"""Tests for the spine-facing games rebuild: tools, invention, tournaments."""
import json

import pytest

from nomorals.games.invent import (
    validate_design, InventedGame, save_design, load_designs,
)
from nomorals.games.tournaments import (
    new_tournament, load_tournament, list_tournaments, MatchResult,
)
from nomorals.tools.registry import ToolRegistry


def _good_design():
    return {
        "name": "test_quest",
        "title": "Test Quest",
        "blurb": "a test game",
        "min_players": 2,
        "max_players": 4,
        "setup": "you stand at a crossroads",
        "turn": "describe what you do",
        "win": "reach the far tower",
    }


class TestInvention:
    def test_validate_good(self):
        assert validate_design(_good_design()) == []

    def test_validate_bad(self):
        assert validate_design({})  # missing everything
        bad = _good_design(); bad["name"] = "Bad Name!"
        assert validate_design(bad)
        bad = _good_design(); bad["max_players"] = 1
        assert validate_design(bad)

    def test_invented_game_plays(self):
        g = InventedGame(_good_design())
        assert g.name == "test_quest"
        st = g.new_state(__import__("random").Random(1))
        assert st["round"] == 1

    def test_design_persists(self, tmp_path, monkeypatch):
        import nomorals.games.invent as inv
        monkeypatch.setattr(inv, "invention_dir", lambda: tmp_path)
        save_design(_good_design())
        assert any(d["name"] == "test_quest" for d in load_designs())


class TestTournaments:
    def test_full_arc(self, tmp_path, monkeypatch):
        import nomorals.games.tournaments as tmod
        monkeypatch.setattr(tmod, "tournament_dir", lambda: tmp_path)
        t = new_tournament("Cup", ["wordchain"], ["a", "b"], rounds=1)
        assert t.status == "open"
        msgs = t.record_match(MatchResult("wordchain", ["a", "b"], "a"))
        assert t.status == "finished"
        assert t.winner == "a"
        assert any("champion" in m for m in msgs)

    def test_reload(self, tmp_path, monkeypatch):
        import nomorals.games.tournaments as tmod
        monkeypatch.setattr(tmod, "tournament_dir", lambda: tmp_path)
        t = new_tournament("Cup", ["wordchain"], ["a", "b"], rounds=2)
        t2 = load_tournament(t.id)
        assert t2 is not None and t2.name == "Cup"

    def test_stake_math(self, tmp_path, monkeypatch):
        import nomorals.games.tournaments as tmod
        monkeypatch.setattr(tmod, "tournament_dir", lambda: tmp_path)
        t = new_tournament("Cup", ["wordchain"], ["a", "b", "c"],
                           rounds=1, stake=10)
        assert t.pot == 30


class TestGamesTool:
    @pytest.fixture
    def reg(self):
        r = ToolRegistry()
        r.register_builtins()
        return r

    def test_list(self, reg):
        out = reg.call("games", action="list", actor="owner")
        assert out.ok and out.value["count"] > 30

    def test_start_move_state(self, reg):
        chat = "pytest:games1"
        out = reg.call("games", action="start", game="numberguess",
                       players="pytest", chat=chat, actor="owner")
        assert out.ok, out.value
        assert out.value["room_id"]
        out = reg.call("games", action="state", chat=chat, actor="owner")
        assert out.ok and out.value["game"] == "numberguess"
        out = reg.call("games", action="move", move="50", chat=chat,
                       name="pytest", actor="owner")
        assert out.ok

    def test_unknown_game(self, reg):
        out = reg.call("games", action="start", game="nosuchgame",
                       chat="pytest:games2", actor="owner")
        assert out.ok and not out.value["ok"]

    def test_spectate(self, reg):
        chat = "pytest:games3"
        reg.call("games", action="start", game="wordchain",
                 players="pytest", chat=chat, actor="owner")
        out = reg.call("games", action="spectate", chat=chat, actor="owner")
        assert out.ok and "transcript" in out.value

    def test_outsider_can_play(self, reg):
        out = reg.call("games", action="list", actor="outsider")
        assert out.ok

    def test_tournament_tool(self, reg):
        out = reg.call("games", action="tournament", game="wordchain",
                       players="x,y", name="Pytest Cup", rounds=1,
                       actor="owner")
        assert out.ok, out.value
        assert out.value["tournament"]["players"] == ["x", "y"]
