"""Pass #167 pins: the games failure ledger (REUSED) + players conversions.

``nomorals/games/failures.py`` (namespace ``games``) is reused from #161:
players is a games-subsystem module, so reuse is correct (unlike #161's
engine/trivia split). The 17 durability tests below mirror the #141+
shape exactly; the 21 site pins prove each conversion records without
changing flow (fallback value unchanged AND a ledger entry under its
site name). There are 0 exclusions in this pass, so no NOT-recording
pins. No pin hardcodes scan totals or players counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the real
home, never the repo tree). Restart tests use real subprocesses. Every
trigger was probed live before its pin was written (probe.py).
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.games import failures as gf


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


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


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


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("games").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("games")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
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

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        gf.record_failure("no-home", ValueError("boom"))
        assert gf.failure_count() == 1
        assert [f.site for f in gf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []
    assert gf.format_failures() == "no games failures recorded."
    gf.record_failure("games.site-a", ValueError("boom"))
    gf.record_failure("games.site-b", RuntimeError("bang"))
    assert gf.failure_count() == 2
    assert [f.site for f in gf.recent_failures()] == ["games.site-b", "games.site-a"]
    assert [f.site for f in gf.recent_failures(1)] == ["games.site-b"]
    text = gf.format_failures()
    assert "2 total" in text and "games.site-a" in text and "games.site-b" in text
    gf.clear_failures()
    assert gf.failure_count() == 0
    assert gf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    gf.record_failure("games.disk", ValueError("boom"))
    entries, total = fs.Store("games").load()
    assert total == 1 and entries[0]["site"] == "games.disk"
    gf.clear_failures()
    assert fs.Store("games").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    gf.record_failure("games.mem", ValueError("boom"))
    assert gf.failure_count() == 1
    assert [f.site for f in gf.recent_failures()] == ["games.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                gf.record_failure(f"games.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
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


# ---- Site pins ------------------------------------------------------------

from nomorals.games.players import PlayerStore, Profile


class _Tx:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _Cur:
    rowcount = 0


class FakeDB:
    """Dispatch on SQL text; raise only where the pin asks."""

    def __init__(self, fail=None, rows=None, one=None, one_fn=None):
        self.fail = fail or {}
        self.rows = rows or {}
        self.one = one or {}
        self.one_fn = one_fn

    def _hit(self, table, sql):
        for sub in table:
            if sub in sql:
                return sub
        return None

    def query(self, sql, args=()):
        sub = self._hit(self.fail, sql)
        if sub:
            raise RuntimeError(f"db boom on {sub}")
        for key, val in self.rows.items():
            if key in sql:
                return val
        return []

    def query_one(self, sql, args=()):
        sub = self._hit(self.fail, sql)
        if sub:
            raise RuntimeError(f"db boom on {sub}")
        if self.one_fn is not None:
            return self.one_fn(sql, args)
        for key, val in self.one.items():
            if key in sql:
                return val
        return None

    def execute(self, sql, args=()):
        sub = self._hit(self.fail, sql)
        if sub:
            raise RuntimeError(f"db boom on {sub}")
        return _Cur()

    def transaction(self):
        return _Tx()


FULL_ROW = {
    "player_key": "telegram:Mary", "display": "Mary", "platform": "telegram",
    "username": "mary", "deleted_at": 0, "coins": 3, "points": 4, "wins": 1,
    "losses": 0, "draws": 0, "streak": 2, "best_streak": 2, "games_played": 1,
    "xp": 5, "per_game": "{}", "items": "{}",
    "created_at": 100.0, "updated_at": 200.0,
}


def test_pin_from_row_load():
    p = Profile.from_row({**FULL_ROW, "per_game": "corrupt{"})
    assert p.per_game == {}
    assert _sites() == ["players.from_row_load"]


def test_pin_get_reread():
    store = PlayerStore(FakeDB(fail={"SELECT * FROM game_players": True}))
    prof = store.get("telegram:Mary")
    assert prof.key == "telegram:Mary"
    assert _sites() == ["players.get_reread"]


def test_pin_find_by_username():
    store = PlayerStore(FakeDB(fail={"game_players": True}))
    assert store.find_by_username("mary") is None
    assert _sites() == ["players.find_by_username"]


def test_pin_purge_tables():
    store = PlayerStore(FakeDB(fail={"sqlite_master": True}))
    store._purge_profile("telegram:Mary")  # returns None either way
    assert _sites() == ["players.purge_tables"]


def test_pin_known_names():
    store = PlayerStore(FakeDB(fail={"SELECT": True}))
    assert store._known_names("telegram", "123") == []
    assert _sites() == ["players.known_names"]


def test_pin_merge_username_rows():
    store = PlayerStore(FakeDB(fail={"lower(username)": True}))
    store._merge_all_legacy_names("telegram:12345", "Mary", "telegram",
                                  username="mary")
    assert _sites() == ["players.merge_username_rows"]


def test_pin_merge_key_rows():
    # "WHERE COALESCE" hits only the bare key-sweep query; the username
    # query carries "AND COALESCE" instead.
    store = PlayerStore(FakeDB(fail={"WHERE COALESCE": True}))
    store._merge_all_legacy_names("telegram:12345", "Mary", "telegram",
                                  username="mary")
    assert _sites() == ["players.merge_key_rows"]


def test_pin_merge_seen_key():
    # A key-less username row fires merge_seen_key in the first loop AND
    # merge_legacy_key_cast in the final loop (same rows, same expression).
    store = PlayerStore(FakeDB(
        rows={"lower(username)": [{}], "WHERE COALESCE": []}))
    store._merge_all_legacy_names("telegram:12345", "Mary", "telegram",
                                  username="mary")
    assert "players.merge_seen_key" in _sites()


def test_pin_merge_key_cast():
    store = PlayerStore(FakeDB(
        rows={"lower(username)": [], "WHERE COALESCE": [{}]}))
    store._merge_all_legacy_names("telegram:12345", "Mary", "telegram",
                                  username="mary")
    assert _sites() == ["players.merge_key_cast"]


def test_pin_merge_legacy_key_cast():
    # Same coupling as merge_seen_key: the key-less row fires both sites.
    store = PlayerStore(FakeDB(
        rows={"lower(username)": [{}], "WHERE COALESCE": []}))
    store._merge_all_legacy_names("telegram:12345", "Mary", "telegram",
                                  username="mary")
    assert "players.merge_legacy_key_cast" in _sites()


def test_pin_merge_existence():
    store = PlayerStore(FakeDB(fail={"SELECT 1 FROM": True}))
    out = store.merge_legacy_names("telegram:12345", ["Mary"])
    assert out == {"id_key": "telegram:12345", "merged": []}
    # One record per alias candidate (telegram, telegram-bot,
    # telegram-personal) — the existence check fails for each.
    assert _sites() == ["players.merge_existence"] * 3


def test_pin_resolve_legacy_keys():
    store = PlayerStore(FakeDB(fail={"LIKE": True}))
    assert store._resolve_legacy_keys(["telegram"], "Mary") == ["telegram:Mary"]
    assert _sites() == ["players.resolve_legacy_keys"]


def test_pin_merge_tables():
    store = PlayerStore(FakeDB(fail={"sqlite_master": True}))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_tables"]


def test_pin_merge_q1():
    store = PlayerStore(FakeDB(
        rows={"sqlite_master": [{"name": "game_players"}]},
        fail={"WHERE player_key = ?": True}))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_q1"]


def test_pin_merge_cols():
    # PRAGMA fails but dict rows keep the merge alive; only merge_cols fires.
    store = PlayerStore(FakeDB(
        rows={"sqlite_master": [{"name": "game_players"}]},
        one={"WHERE player_key = ?": dict(FULL_ROW)},
        fail={"PRAGMA": True}))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_cols"]


def test_pin_merge_qall():
    # Both skill-slug reads fail (ID slugs, then legacy slugs).
    store = PlayerStore(FakeDB(
        rows={"sqlite_master": [{"name": "game_players"},
                                {"name": "game_skills"}]},
        one={"WHERE player_key = ?": dict(FULL_ROW)},
        fail={"FROM game_skills": True}))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_qall"] * 2


def test_pin_merge_num():
    leg = dict(FULL_ROW, player_key="telegram:Mary", streak="abc")
    store = PlayerStore(FakeDB(
        rows={"sqlite_master": [{"name": "game_players"}]},
        one_fn=lambda sql, args: leg if args[0] == "telegram:Mary"
        else dict(FULL_ROW)))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_num"]


def test_pin_merge_fnum():
    leg = dict(FULL_ROW, player_key="telegram:Mary", created_at="junk")
    store = PlayerStore(FakeDB(
        rows={"sqlite_master": [{"name": "game_players"}]},
        one_fn=lambda sql, args: leg if args[0] == "telegram:Mary"
        else dict(FULL_ROW)))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_fnum"]


def test_pin_merge_jload():
    leg = dict(FULL_ROW, player_key="telegram:Mary", per_game="not json{")
    store = PlayerStore(FakeDB(
        rows={"sqlite_master": [{"name": "game_players"}]},
        one_fn=lambda sql, args: leg if args[0] == "telegram:Mary"
        else dict(FULL_ROW)))
    store._merge_one_legacy_key("telegram:12345", "telegram:Mary")
    assert _sites() == ["players.merge_jload"]


def test_pin_all_rows():
    store = PlayerStore(FakeDB(fail={"SELECT * FROM": True}))
    assert store.all() == []
    assert _sites() == ["players.all_rows"]


def test_pin_add_xp():
    # get() succeeds (full row), then the read-back SELECT xp fails.
    store = PlayerStore(FakeDB(
        one={"SELECT * FROM": dict(FULL_ROW)},
        fail={"SELECT xp FROM": True}))
    assert store.add_xp("telegram:Mary", 10) == 0
    assert _sites() == ["players.add_xp"]
