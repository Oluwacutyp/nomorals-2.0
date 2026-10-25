"""Pass #172 pins: the games failure ledger (REUSED) + achievements conversions.

``nomorals/games/failures.py`` (namespace ``games``) is reused from #161:
achievements is a games-subsystem module, so reuse is correct (the #167
players precedent, unlike #161's engine/trivia split). The 17 durability
tests below mirror the #141+ shape exactly; the 18 site pins prove each
conversion records without changing flow (fallback value unchanged AND a
ledger entry under its site name); the 1 exclusion pin proves live_rarity
does NOT record. No pin hardcodes scan totals or achievements counts.
Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written (probe.py).
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

from nomorals.games import achievements as ach


class _Cur:
    def __init__(self, rowcount=0, rows=None):
        self.rowcount = rowcount
        self._rows = rows or []

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class FakeDB:
    """fail: (method, substring) pairs that raise. exec_map/query_map/
    one_map route SQL substrings to canned values."""

    def __init__(self, fail=(), exec_map=(), query_map=(), one_map=(),
                 rowcount=0):
        self.fail = set(fail)
        self.exec_map = list(exec_map)
        self.query_map = list(query_map)
        self.one_map = list(one_map)
        self.rowcount = rowcount

    def _boom(self, method, sql):
        for m, sub in self.fail:
            if m == method and sub in sql:
                raise RuntimeError(f"db boom {method} {sub}")

    def execute(self, sql, args=()):
        self._boom("execute", sql)
        for sub, cur in self.exec_map:
            if sub in sql:
                return cur
        return _Cur(rowcount=self.rowcount)

    def query(self, sql, args=()):
        self._boom("query", sql)
        for sub, val in self.query_map:
            if sub in sql:
                return val
        return []

    def query_one(self, sql, args=()):
        self._boom("query_one", sql)
        for sub, val in self.one_map:
            if sub in sql:
                return val
        return None


class _RaisingTitles:
    def __init__(self, db):
        pass

    def check_unlocks(self, player_key):
        raise RuntimeError("titles boom")


class _QuietTitles:
    def __init__(self, db):
        pass

    def check_unlocks(self, player_key):
        return []


def test_pin_ensure_unlocks_table():
    ach._ensure_unlocks_table(
        FakeDB(fail={("execute", "progression_unlocks")}))
    assert _sites() == ["achievements.ensure_unlocks_table"]


def test_pin_persist_unlocks():
    ach._persist_unlocks(
        FakeDB(fail={("execute",
                      "INSERT OR IGNORE INTO progression_unlocks")}),
        "p1", "craps_high_roller")
    assert _sites() == ["achievements.persist_unlocks"]


def test_pin_get_unlocks():
    db = FakeDB(fail={("execute", "SELECT unlock_id FROM progression_unlocks")})
    assert ach.get_unlocks(db, "p1") == []
    assert _sites() == ["achievements.get_unlocks"]


def test_pin_has_unlock():
    db = FakeDB(fail={("execute", "SELECT 1 FROM progression_unlocks")})
    assert ach.has_unlock(db, "p1", "unlock:command:predict") is False
    assert _sites() == ["achievements.has_unlock"]


def test_pin_progression_capabilities():
    # A None grant makes the loop body raise inside the capabilities
    # handler (get_unlocks itself never raises, so its rows must).
    db = FakeDB(exec_map=[("SELECT unlock_id", _Cur(rows=[(None,)]))])
    assert ach.progression_capabilities(db, "p1") == set()
    assert _sites() == ["achievements.progression_capabilities"]


def test_pin_unlock_achievement_insert():
    db = FakeDB(fail={("execute", "INSERT OR IGNORE INTO achievements")})
    assert ach.unlock_achievement(db, "p1", "games_10") is False
    assert _sites() == ["achievements.unlock_achievement_insert"]


def test_pin_unlock_titles_check():
    db = FakeDB(rowcount=1)
    with patch("nomorals.games.titles.TitleStore", _RaisingTitles):
        assert ach.unlock_achievement(db, "p1", "games_10") is True
    assert _sites() == ["achievements.unlock_titles_check"]


def test_pin_unlock_persist_call():
    # _persist_unlocks swallows its own INSERT failures, so the outer
    # handler fires only when the persist call itself raises — pinned
    # via monkeypatch (the #162 can_afford precedent).
    db = FakeDB(rowcount=1)
    with patch("nomorals.games.titles.TitleStore", _QuietTitles), \
            patch.object(ach, "_persist_unlocks", side_effect=_boom):
        assert ach.unlock_achievement(db, "p1", "craps_high_roller") is True
    assert _sites() == ["achievements.unlock_persist_call"]


def test_pin_ensure_rarity_tables():
    ach._ensure_rarity_tables(FakeDB(fail={("execute", "CREATE TABLE")}))
    assert _sites() == ["achievements.ensure_rarity_tables"]


def test_pin_ensure_progress_table():
    ach._ensure_progress_table(
        FakeDB(fail={("execute",
                      "CREATE TABLE IF NOT EXISTS achievement_progress")}))
    assert _sites() == ["achievements.ensure_progress_table"]


def test_pin_track_progress():
    db = FakeDB(fail={("query_one",
                       "SELECT progress FROM achievement_progress")})
    assert ach.track_progress(db, "p1", "games_10") == (0, 10, False)
    assert _sites() == ["achievements.track_progress"]


def test_pin_get_progress():
    db = FakeDB(fail={("query_one",
                       "SELECT progress FROM achievement_progress")})
    assert ach.get_progress(db, "p1", "games_10") == (0, 10)
    assert _sites() == ["achievements.get_progress"]


def test_pin_repeatable_check():
    db = FakeDB(fail={("query_one",
                       "SELECT progress FROM achievement_progress")})
    assert ach.repeatable_check(db, "p1", "wins_100", 250) == 0
    assert _sites() == ["achievements.repeatable_check"]


def test_pin_anniversaries_query():
    db = FakeDB(fail={("query", "SELECT achievement_id, unlocked_at")})
    assert ach.achievement_anniversaries(db, "p1") == []
    assert _sites() == ["achievements.anniversaries_query"]


def test_pin_anniversaries_row_parse():
    db = FakeDB(query_map=[("SELECT achievement_id, unlocked_at",
                            [{"achievement_id": "games_10",
                              "unlocked_at": "not-a-float"}])])
    assert ach.achievement_anniversaries(db, "p1") == []
    assert _sites() == ["achievements.anniversaries_row_parse"]


def test_pin_ensure_showcase_table():
    ach._ensure_showcase_table(
        FakeDB(fail={("execute",
                      "CREATE TABLE IF NOT EXISTS achievement_showcase")}))
    assert _sites() == ["achievements.ensure_showcase_table"]


def test_pin_set_showcase():
    db = FakeDB(
        exec_map=[("SELECT achievement_id, unlocked_at",
                   _Cur(rows=[("games_10", 1000.0)]))],
        fail={("execute", "DELETE FROM achievement_showcase")})
    assert ach.set_showcase(db, "p1", ["games_10"]) == (
        False, "couldn't save your showcase.")
    assert _sites() == ["achievements.set_showcase"]


def test_pin_get_showcase():
    db = FakeDB(fail={("query",
                       "SELECT achievement_id FROM achievement_showcase")})
    assert ach.get_showcase(db, "p1") == []
    assert _sites() == ["achievements.get_showcase"]


# ---- Exclusion pins -------------------------------------------------------

def test_pin_live_rarity_does_not_record():
    # E1 (outcome-equivalence): a query outage returns exactly the
    # documented small-sample fallback, and records nothing.
    db = FakeDB(fail={("query", "SELECT COUNT")})
    assert ach.live_rarity(db, "games_10") == ("common", 0.0)
    assert "achievements.live_rarity" not in _sites()
    assert _sites() == []
