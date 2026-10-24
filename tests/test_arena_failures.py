"""Pass #143 pins: the agents/arena failure ledger + durability wiring.

Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through the new ``nomorals.agents.arena.failures`` ledger; restart
tests use real subprocesses (a fresh interpreter is the honest
simulated restart — no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name.
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents.arena import failures as arena


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    arena.clear_failures()
    yield
    arena.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("agents.arena")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents.arena").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents.arena")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents.arena").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents.arena")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents.arena").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents.arena").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents.arena").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents.arena")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents.arena").load() == ([], 0)
    assert store.save(_entries(1, "after"), 1) is True
    entries, total = fs.Store("agents.arena").load()
    assert [e["site"] for e in entries] == ["after0"]
    assert total == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents.arena").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    # The #141 fix-up shape, pinned for the new ledger too: save()
    # must resolve its path INSIDE the try, so a home-resolution
    # failure degrades to a counted write failure, never raises.
    # NM_HOME must be unset: an absolute NM_HOME never calls
    # expanduser, which is how scratch-home batches mask this.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents.arena").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        arena.record_failure("no-home", ValueError("boom"))
        assert arena.failure_count() == 1
        assert [f.site for f in arena.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents.arena")
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
    entries, _ = fs.Store("agents.arena").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert arena.failure_count() == 0
    assert arena.recent_failures() == []
    assert arena.format_failures() == "no agents.arena failures recorded."
    arena.record_failure("arena.site-a", ValueError("boom"))
    arena.record_failure("arena.site-b", RuntimeError("bang"))
    assert arena.failure_count() == 2
    assert [f.site for f in arena.recent_failures()] == ["arena.site-b", "arena.site-a"]
    assert [f.site for f in arena.recent_failures(1)] == ["arena.site-b"]
    text = arena.format_failures()
    assert "2 total" in text and "arena.site-a" in text and "arena.site-b" in text
    arena.clear_failures()
    assert arena.failure_count() == 0
    assert arena.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    arena.record_failure("disk-a", ValueError("boom"))
    arena.record_failure("disk-b", RuntimeError("bang"))
    assert arena.failure_count() == 2
    assert arena._store.path().exists()
    assert arena._store.path().name == "agents.arena.json"
    arena.clear_failures()
    assert arena.failure_count() == 0
    assert not arena._store.path().exists()


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    arena.record_failure("blocked", ValueError("boom"))
    assert arena.failure_count() == 1
    assert [f.site for f in arena.recent_failures()] == ["blocked"]
    assert fs.write_failures() > before


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                arena.record_failure(f"w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert arena.failure_count() == 400
    assert len(arena.recent_failures(1000)) == 200
    entries, total = arena._store.load()
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
        "from nomorals.agents.arena import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents.arena import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents.arena import failures as f\n"
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
    store_path = home / "failures" / "agents.arena.json"
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

class _RaisingTx:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, *exc):
        return False


class _RaisingDb:
    """Every storage call raises: the fully-broken-db shape."""

    def transaction(self):
        return _RaisingTx(self)

    def execute(self, *a, **k):
        raise RuntimeError("db down")

    def executemany(self, *a, **k):
        raise RuntimeError("db down")

    def query(self, *a, **k):
        raise RuntimeError("db down")

    def query_one(self, *a, **k):
        raise RuntimeError("db down")


class _InsertBrokenDb(_RaisingDb):
    """DDL passes, INSERT fails: reaches record()'s insert handler
    past _ensure_table instead of tripping the table gate."""

    def execute(self, sql, *a, **k):
        if str(sql).lstrip().upper().startswith("INSERT"):
            raise RuntimeError("insert down")
        return None


def _sites():
    return [f.site for f in arena.recent_failures()]


def test_converted_ensure_table_records_and_blocks_record():
    from nomorals.agents.arena import activity
    # _ensure_table fails -> record() bails before its insert, exactly
    # as before; the table failure is now journaled, not silent.
    assert activity.record(_RaisingDb(), "command", "x") is None
    assert "activity._ensure_table" in _sites()


def test_converted_record_insert_records_and_falls_back():
    from nomorals.agents.arena import activity
    assert activity.record(_InsertBrokenDb(), "command", "x") is None
    assert "activity.record" in _sites()


def test_converted_recent_commands_records_and_falls_back():
    from nomorals.agents.arena import activity
    assert activity.recent_commands(_RaisingDb()) == []
    assert "activity.recent_commands" in _sites()


def test_converted_save_knowledge_records_and_still_returns_id():
    from nomorals.agents.arena.core import Arena
    # The sharpest pin in the pass: the old code returned the kid
    # while the row never persisted. That behavior is unchanged —
    # but now the lost write is journaled under its site.
    ctx = SimpleNamespace(db=_RaisingDb(), settings=None)
    kid = Arena(ctx)._save_knowledge("t", "c", "d", [])
    assert len(kid) == 12 and all(c in "0123456789abcdef" for c in kid)
    assert "core._save_knowledge" in _sites()


def test_converted_set_build_status_records_and_never_raises():
    from nomorals.agents.arena.core import Arena
    ctx = SimpleNamespace(db=_RaisingDb(), settings=None)
    Arena(ctx)._set_build_status("deadbeef0001", "approved")
    assert "core._set_build_status" in _sites()


def test_converted_stream_records_and_never_raises():
    from nomorals.agents.arena.core import Arena
    ctx = SimpleNamespace(db=_RaisingDb(), settings=None)
    Arena(ctx)._stream("topic", {"topic": "t"})
    assert "core._stream" in _sites()


def test_converted_persist_state_records_and_falls_back():
    from nomorals.agents.arena import sampling
    assert sampling._persist_state(_RaisingDb(), {"ts": 0}) is False
    assert "sampling._persist_state" in _sites()


def test_converted_category_scores_records_and_falls_back():
    from nomorals.agents.arena import scoring
    assert scoring.category_scores(_RaisingDb()) == {}
    assert "scoring.category_scores" in _sites()


def test_converted_clear_recent_topics_records_both_layers():
    from nomorals.agents.arena import topics
    # The inner recent_topics read fails first (recorded), then the
    # delete fails (recorded); the reported count stays 0, as before.
    assert topics.clear_recent_topics(_RaisingDb()) == 0
    sites = _sites()
    assert "topics.recent_topics" in sites
    assert "topics.clear_recent_topics" in sites
