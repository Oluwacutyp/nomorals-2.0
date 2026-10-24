"""Durability pins for the pass #141 failure-ledger persistence.

Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through one real wired ledger (``nomorals.voice.failures``); restart
tests use real subprocesses (a fresh interpreter is the honest
simulated restart — no module reload tricks inside a shared process).
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.voice import failures as voice


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    voice.clear_failures()
    yield
    voice.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


# ---- Store-level -------------------------------------------------------

def test_store_round_trip(tmp_path):
    store = fs.Store("voice")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("voice").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("voice")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("voice").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("voice")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("voice").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("voice").save(_entries(2, "v"), 2) is True
    assert fs.Store("storage").save(_entries(1, "s"), 1) is True
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0", "v1"]
    entries, _ = fs.Store("storage").load()
    assert [e["site"] for e in entries] == ["s0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("voice")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("voice").load() == ([], 0)
    assert store.save(_entries(1, "after"), 1) is True
    entries, total = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["after0"]
    assert total == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("voice").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    # Parent regression pin (Pass #141 fix-up): save() resolved
    # its path OUTSIDE the try, so a home-resolution failure
    # (no home, mocked expanduser — the exact shape of
    # test_send_module_helper_failures' default-DB test) raised
    # out of record_failure instead of degrading. Path
    # resolution is now inside the try: never raises, counts.
    # NM_HOME must be unset: an absolute NM_HOME never calls
    # expanduser, which is how the agent's scratch-home batches
    # masked this defect.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("voice").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    # Ledger level of the same pin: with home resolution broken,
    # the record still lands in memory and the count is right.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        voice.record_failure("no-home", ValueError("boom"))
        assert voice.failure_count() == 1
        assert [f.site for f in voice.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("voice")
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
    entries, _ = fs.Store("voice").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_and_clear_hit_disk():
    voice.record_failure("disk-a", ValueError("boom"))
    voice.record_failure("disk-b", RuntimeError("bang"))
    assert voice.failure_count() == 2
    assert voice._store.path().exists()
    voice.clear_failures()
    assert voice.failure_count() == 0
    assert not voice._store.path().exists()


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    voice.record_failure("blocked", ValueError("boom"))
    assert voice.failure_count() == 1
    assert [f.site for f in voice.recent_failures()] == ["blocked"]
    assert fs.write_failures() > before


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                voice.record_failure(f"w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert voice.failure_count() == 400
    assert len(voice.recent_failures(1000)) == 200
    entries, total = voice._store.load()
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
        "from nomorals.voice import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.voice import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = "from nomorals.voice import failures as f\nf.clear_failures()\nprint('ok')\n"


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
    store_path = home / "failures" / "voice.json"
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
