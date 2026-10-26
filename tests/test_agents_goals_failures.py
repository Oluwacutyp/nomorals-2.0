"""Pass #208 pins: goals converts record, goals exclusions stay silent.

``nomorals/agents/goals.py`` converts 2 of its 12 true sites
(`deadline_text`, `linked_room`) to the REUSED agents ledger, and
excludes 10 (parse-chain probes, documented defaults/failure values,
schema guards, enhancement fallbacks, genuine False/None identities,
optional rooms). The 17 durability tests below mirror the #141+ shape
exactly (same as #206's, against the REUSED agents ledger); the 2
behavior pins prove the converts record while keeping their fallback;
the 10 NOT-recording pins prove each exclusion stays silent with its
designed output. No pin hardcodes scan totals or goals counts. Every
test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree). Restart tests use real
subprocesses. Every trigger was probed live before its pin was written.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents import failures as af
from nomorals.agents import goals as G


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _assert_silent():
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def _goal_system():
    return G.GoalSystem.__new__(G.GoalSystem)


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("agents")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("agents").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents")
    errors = []

    def worker(k):
        try:
            store.save(_entries(50, f"w{k}-"), k)
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("agents").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no agents failures recorded."
    af.record_failure("agents.site-a", ValueError("boom"))
    af.record_failure("agents.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["agents.site-b", "agents.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["agents.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "agents.site-a" in text and "agents.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("agents.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "agents.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("agents.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["agents.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"agents.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert af.failure_count() == 400
    assert len(af.recent_failures(1000)) == 200
    entries, total = af._store.load()
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
        "from nomorals.agents import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents import failures as f\n"
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
    store_path = fs.Store("agents").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


# ---- Behavior pins (one per convert) --------------------------------------

def test_behavior_590_deadline_text_records_and_keeps_rendering():
    gs = _goal_system()
    assert gs.deadline_text(G.Goal(id="g", title="t", deadline=1e300)) == "deadline set"
    assert [f.site for f in af.recent_failures()] == ["goals.deadline_text"]
    assert af.recent_failures()[0].exc_type == "OverflowError"


def test_behavior_826_linked_room_records_and_keeps_none():
    class _RaisingMgr:
        def get_by_linked(self, *a, **k):
            raise RuntimeError("db down")

    gs = _goal_system()
    gs._rooms_cache = _RaisingMgr()
    assert gs._linked_room("g") is None
    assert [f.site for f in af.recent_failures()] == ["goals.linked_room"]


# ---- NOT-recording pins (one per exclusion) ------------------------------

def test_not_recording_157_parse_chain_probe_falls_through():
    assert G._parse_deadline("garbage") == 0.0
    _assert_silent()


def test_not_recording_172_outer_overflow_returns_documented_zero():
    assert G._parse_deadline(10**400) == 0.0
    _assert_silent()


class _TableFailingDB:
    """Fails one table's PRAGMA; everything else answers empty."""

    def __init__(self, table):
        self.table = table

    def query(self, sql, *a, **k):
        if self.table in sql:
            raise RuntimeError("table gone")
        return []

    def execute(self, *a, **k):
        return None


def test_not_recording_209_schema_goals_table_missing_passes():
    gs = _goal_system()
    gs.db = _TableFailingDB("agent_goals")
    gs._ensure_schema()
    _assert_silent()


def test_not_recording_218_schema_steps_table_missing_passes():
    gs = _goal_system()
    gs.db = _TableFailingDB("agent_goal_steps")
    gs._ensure_schema()
    _assert_silent()


def test_not_recording_379_structuring_failure_falls_back_to_empty():
    from nomorals.agents import structuring

    gs = _goal_system()
    with patch.object(structuring, "structure_text",
                      side_effect=RuntimeError("llm down")):
        assert gs._structured_plan("t", "a real description here") == []
    _assert_silent()


class _FakeEngine:
    def __init__(self, *a, **k):
        pass

    def reason(self, *a, **k):
        return SimpleNamespace(subgoals=["step one"])


class _RaisingSkills:
    def __init__(self, *a, **k):
        pass

    def context_block(self, *a, **k):
        raise RuntimeError("skills down")


class _RaisingEngine:
    def __init__(self, *a, **k):
        pass

    def reason(self, *a, **k):
        raise RuntimeError("model down")


def _decompose_system():
    gs = _goal_system()
    gs.context = SimpleNamespace(router=object(), db=SimpleNamespace())
    gs.db = SimpleNamespace()
    return gs


def test_not_recording_400_skills_failure_keeps_decompose():
    from nomorals.agents import reasoning as R
    from nomorals.agents import skills as S

    gs = _decompose_system()
    with patch.object(R, "ReasoningEngine", _FakeEngine), \
            patch.object(S, "SkillLibrary", _RaisingSkills):
        assert gs._decompose("t", "d") == ["step one"]
    _assert_silent()


def test_not_recording_405_engine_failure_falls_back_to_empty():
    from nomorals.agents import reasoning as R

    gs = _decompose_system()
    with patch.object(R, "ReasoningEngine", _RaisingEngine):
        assert gs._decompose("t", "d") == []
    _assert_silent()


def test_not_recording_673_estimates_store_failure_returns_false():
    from nomorals.planning import estimates as E

    step = G.GoalStep(id="s", goal_id="g", position=0,
                      description="do thing", est_minutes=10.0)
    gs = _goal_system()
    gs._steps = lambda gid: [step]
    with patch.object(E, "get_store", side_effect=RuntimeError("store down")):
        assert gs.record_step_outcome("g", "s", 30.0) is False
    _assert_silent()


def test_not_recording_815_room_init_failure_disables_rooms():
    gs = _goal_system()
    gs.context = SimpleNamespace()
    assert gs._room_manager() is None
    assert gs._rooms_cache is False
    assert gs._room_manager() is None
    _assert_silent()


def test_not_recording_833_rooms_config_failure_returns_default_on():
    gs = _goal_system()
    gs.context = SimpleNamespace()
    assert gs._rooms_auto_create() is True
    _assert_silent()
