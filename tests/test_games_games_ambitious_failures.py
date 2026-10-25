"""Pass #190 durability + behavior pins for nomorals/games/games/ambitious.py.

The games swallow ledger (nomorals/games/failures.py, namespace
``games``, cap 200) is REUSED here; the ten converted ambitious sites
record into it, and the five excluded sites must stay silent. The
durability half mirrors the #141+ shape: store-level round-trip, cap,
clear, namespace isolation, corrupt recovery, save-failure counting,
concurrent saves, plus ledger-level memory degradation, disk hits,
concurrent records, and subprocess restart persistence.

Docstrings in this file avoid spelling out the bare-swallow handler
form, because the error scanner counts test files too.
"""
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.core import failure_store as fs
from nomorals.games import failures as gf
from nomorals.games.games import ambitious as amb

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    gf.clear_failures()
    yield
    gf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in gf.recent_failures()]


class _Boom(Exception):
    pass


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("games")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("games").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("games")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("games").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("games")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("games").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("games").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("games").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("games")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("games").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("games").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("games").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(
        monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    import unittest.mock as mock
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("games").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("games")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("games").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(
        monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    import unittest.mock as mock
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        gf.record_failure("no-home", ValueError("boom"))
        assert gf.failure_count() == 1
        assert [f.site for f in gf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []
    assert gf.format_failures() == "no games failures recorded."
    gf.record_failure("ambitious.site-a", ValueError("boom"))
    gf.record_failure("ambitious.site-b", RuntimeError("bang"))
    assert gf.failure_count() == 2
    assert [f.site for f in gf.recent_failures()] == [
        "ambitious.site-b", "ambitious.site-a"]
    assert [f.site for f in gf.recent_failures(1)] == ["ambitious.site-b"]
    text = gf.format_failures()
    assert ("2 total" in text and "ambitious.site-a" in text
            and "ambitious.site-b" in text)
    gf.clear_failures()
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    gf.record_failure("ambitious.disk", ValueError("boom"))
    entries, total = fs.Store("games").load()
    assert total == 1 and entries[0]["site"] == "ambitious.disk"
    gf.clear_failures()
    assert fs.Store("games").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    gf.record_failure("ambitious.mem", ValueError("boom"))
    assert gf.failure_count() == 1
    assert [f.site for f in gf.recent_failures()] == ["ambitious.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                gf.record_failure(f"ambitious.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert gf.failure_count() == 400
    assert len(gf.recent_failures(1000)) == 200
    entries, total = gf._store.load()
    assert len(entries) == 200
    assert total == 400


# ---- Restart (subprocess) ------------------------------------------------

def _run_subprocess(home: Path, code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["NM_HOME"] = str(home)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT), env=env,
        capture_output=True, text=True, timeout=120)


def _record_code(n: int, clear_first: bool = True) -> str:
    clear = "f.clear_failures()\n" if clear_first else ""
    return (
        "from nomorals.games import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.games import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.games import failures as f\n"
               "f.clear_failures()\nprint('ok')\n")


def test_restart_record_survives(tmp_path):
    home = tmp_path / "home"
    proc = _run_subprocess(home, _record_code(3))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 3
    assert data["sites"] == ["site2", "site1", "site0"]


def test_restart_clear_is_gone(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    assert _run_subprocess(home, _CLEAR_CODE).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


def test_restart_total_preserved_and_cap_holds(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(250)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 250
    assert len(data["sites"]) == 200
    assert data["sites"][0] == "site249"


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("games").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Shared harnesses for site + NOT-recording pins ----------------------

class _Player:
    def __init__(self, key, name="Hero"):
        self.key = key
        self.name = name


class _Room:
    def __init__(self, state, humans):
        self.state = state
        self.humans = humans


def _stub(monkeypatch, dotted, drop):
    """Install a copy of the module minus `drop` attrs (auto-restored)."""
    import importlib
    orig = importlib.import_module(dotted)
    stub = types.ModuleType(dotted)
    for attr in dir(orig):
        if attr not in drop:
            setattr(stub, attr, getattr(orig, attr))
    monkeypatch.setitem(sys.modules, dotted, stub)
    return orig


def _gear_room(player_power=0, myth=False, rank=5, weapon_atk=10):
    state = {
        "house": {"max_hp": 100, "hp": 100, "atk": 10, "def": 5},
        "player_power": player_power,
        "house_skills": {},
    }
    enemy = {
        "gear": {"weapon": {"atk": weapon_atk}, "armor": {"def": 4}},
        "rank_idx": rank,
        "myth_foe": myth,
    }
    return state, enemy


def _apply_house(state, enemy):
    amb.BattleArenaGame()._apply_house_gear(_Room(state, []), enemy)
    return state["house"]


def _setup_state(difficulty="normal"):
    return {
        "you": {"max_hp": 100, "hp": 100, "atk": 10, "def": 5},
        "house": {"max_hp": 100, "hp": 100, "atk": 10, "def": 5},
        "gear_base": {"atk": 10, "def": 5},
        "skills": {"p1": []}, "skill_tiers": {},
        "house_skill": 0, "_difficulty": difficulty, "_arena_bouts": 0,
        "loadout": {"p1": {}}, "rpg_stats": {"p1": {}},
    }


# ---- Site pins (one per convert; 1307 has two) ---------------------------

def test_pin_ambitious_spawn_enemy(monkeypatch):
    _stub(monkeypatch, "nomorals.games.enemies", {"roll_enemy"})
    room = _Room({"house_skill": 0,
                  "house": {"max_hp": 100, "hp": 100, "atk": 10,
                            "def": 5}}, [])
    amb.BattleArenaGame()._spawn_enemy(room)
    assert _sites() == ["ambitious.spawn_enemy"]
    assert "player_power" not in room.state


def test_pin_ambitious_gear_effectiveness(monkeypatch):
    _stub(monkeypatch, "nomorals.games.enemies", {"GEAR_EFFECTIVENESS"})
    state, enemy = _gear_room()
    h = _apply_house(state, enemy)
    assert _sites() == ["ambitious.gear_effectiveness"]
    assert h["atk"] == 15  # rank 5 genuine +6; failure 0.5 gives +5


def test_pin_ambitious_house_set_bonus(monkeypatch):
    _stub(monkeypatch, "nomorals.games.gear", {"SET_BONUSES"})
    state, enemy = _gear_room()
    enemy["gear"] = {"weapon": {"atk": 10, "set": "storm"},
                     "armor": {"def": 4, "set": "storm"}}
    _apply_house(state, enemy)
    assert _sites() == ["ambitious.house_set_bonus"]
    assert state.get("house_set_bonus") is None


def test_pin_ambitious_house_set_bonus_nameerror(monkeypatch):
    _stub(monkeypatch, "nomorals.games.gear", {"SET_BONUSES"})
    state, enemy = _gear_room(player_power=100)
    with pytest.raises(NameError, match="fighter_power"):
        _apply_house(state, enemy)
    assert _sites() == ["ambitious.house_set_bonus"]


def test_pin_ambitious_myth_ascension(monkeypatch):
    _stub(monkeypatch, "nomorals.games.power", {"fighter_power"})
    state, enemy = _gear_room(player_power=100, myth=True)
    with pytest.raises(NameError, match="fighter_power"):
        _apply_house(state, enemy)
    # newest-first: ascension recorded after the house-set-bonus record
    assert _sites() == ["ambitious.myth_ascension",
                        "ambitious.house_set_bonus"]
    # myth effectiveness 1.0 -> gear +10 -> 20; ascension would swell on
    assert state["house"]["atk"] == 20


def test_pin_ambitious_forbidden_cap(monkeypatch):
    _stub(monkeypatch, "nomorals.games.skills", {"is_enemy_skill"})
    state, enemy = _gear_room(player_power=10)
    state["house"]["atk"] = 100
    h = _apply_house(state, enemy)
    assert _sites() == ["ambitious.forbidden_cap"]
    assert h["atk"] < 100


def test_pin_ambitious_loadout_set_bonus(monkeypatch):
    _stub(monkeypatch, "nomorals.games.gear", {"SET_BONUSES"})
    state = {"you": {}, "gear_base": {"atk": 10, "def": 5},
             "loadout": {"p1": {"weapon": {"atk": 3, "set": "storm"},
                                "armor": {"def": 2, "set": "storm"}}}}
    amb.BattleArenaGame()._apply_loadout(_Room(state, [_Player("p1")]))
    assert _sites() == ["ambitious.loadout_set_bonus"]
    assert state.get("set_bonus") is None
    assert state["you"]["atk"] == 13


def test_pin_ambitious_passive_skills(monkeypatch):
    _stub(monkeypatch, "nomorals.games.skills", {"passive_bonuses"})
    state = {"you": {}, "skills": {"p1": ["slug"]}}
    amb.BattleArenaGame()._apply_skills(_Room(state, [_Player("p1")]))
    assert _sites() == ["ambitious.passive_skills"]
    assert state["you"] == {}


def test_pin_ambitious_rpg_stats(monkeypatch):
    _stub(monkeypatch, "nomorals.games.stats", {"StatBlock"})
    state = {"you": {}, "rpg_stats": {"p1": {}}}
    amb.BattleArenaGame()._apply_stats(_Room(state, [_Player("p1")]))
    assert _sites() == ["ambitious.rpg_stats"]
    assert state["you"] == {}


def test_pin_ambitious_title_effects(monkeypatch):
    _stub(monkeypatch, "nomorals.games.titles", {"title_battle_effects"})
    state = {"you": {}, "rpg_stats": {"p1": {}},
             "titles": {"p1": "Champion"}}
    amb.BattleArenaGame()._apply_stats(_Room(state, [_Player("p1")]))
    assert _sites() == ["ambitious.title_effects"]
    assert "title_effects" not in state


def test_pin_ambitious_mana_regen(monkeypatch):
    _stub(monkeypatch, "nomorals.games.stats", {"MANA_REGEN_PER_TURN"})
    state = {"you": {"hp": 100, "max_hp": 100, "mana": 5, "max_mana": 20},
             "house": {"hp": 100}, "skill_cd": {}, "turns": 0}
    amb.BattleArenaGame().on_move(_Room(state, [_Player("p1")]),
                                  _Player("p1"), "boguscommand", None)
    assert _sites() == ["ambitious.mana_regen"]
    assert state["you"]["mana"] == 5


# ---- NOT-recording pins (one per exclusion) ------------------------------

def test_pin_no_record_power_line(monkeypatch):
    _stub(monkeypatch, "nomorals.games.power", {"power_bar"})
    intro = amb.BattleArenaGame().setup(
        _Room(_setup_state(), [_Player("p1")]), None)
    assert _sites() == []
    assert "⚡ power" not in intro


def test_pin_no_record_diff_note(monkeypatch):
    _stub(monkeypatch, "nomorals.games.enemies", {"DIFFICULTY_BLURB"})
    intro = amb.BattleArenaGame().setup(
        _Room(_setup_state("hard"), [_Player("p1")]), None)
    assert _sites() == []
    assert "🎯" not in intro


def test_pin_no_record_player_power_parse_ascension(monkeypatch):
    # 1327: garbage player_power with a myth foe -> ascension skipped
    state, enemy = _gear_room(player_power="bogus", myth=True)
    h = _apply_house(state, enemy)
    assert _sites() == []
    assert h["atk"] == 20  # myth effectiveness 1.0, no swelling


def test_pin_no_record_player_power_parse_cap(monkeypatch):
    # 1352: garbage player_power -> cap loop skipped, gear-only stats
    state, enemy = _gear_room(player_power="bogus")
    h = _apply_house(state, enemy)
    assert _sites() == []
    assert h["atk"] == 16  # rank 5 genuine 0.65 on weapon 10


def test_pin_no_record_skill_note(monkeypatch):
    from nomorals.games import skills as real_skills
    slug = next(iter(real_skills.SKILL_CATALOG))
    _stub(monkeypatch, "nomorals.games.skills", {"SKILL_CATALOG"})
    state = {"skills": {"p1": [slug]}, "skill_tiers": {}}
    note = amb.BattleArenaGame()._skill_note(_Room(state, [_Player("p1")]))
    assert _sites() == []
    assert note == ""
