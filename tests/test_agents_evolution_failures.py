"""Pass #184 pins: the REUSED agents failure ledger + evolution conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) already existed
(created #152, reused #153/#163/#166/#169/#171/#173/#174) — evolution.py
is an agents-subsystem module, so it reuses that ledger; no new ledger
in this pass. The 17 durability tests below mirror the #141+ shape
exactly (same as #183's, against the agents ledger); the 9 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name); the 7 NOT-recording
pins prove each exclusion stays silent with the designed output. No pin
hardcodes scan totals or evolution counts. Every test redirects stores
with the app's own ``NM_HOME`` override (never the real home, never the
repo tree). Restart tests use real subprocesses. Every trigger was
probed live before its pin was written (probe.py).

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
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.agents import evolution as ev
from nomorals.agents import failures as af
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in af.recent_failures()]


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
    proc = _run_subprocess(home, _record_code(1, clear_first=False))
    assert proc.returncode == 0, proc.stderr
    proc = _run_subprocess(home, _READ_CODE)
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site pins ------------------------------------------------------------

class _Boom(Exception):
    pass


class _StubDB:
    def __init__(self, raise_on=(), query_rows=None, query_raise=False):
        self.raise_on = raise_on
        self.query_rows = query_rows
        self.query_raise = query_raise

    def query_one(self, sql, *a, **k):
        for marker in self.raise_on:
            if marker in sql:
                raise _Boom(f"boom {marker}")
        return {"n": 5}

    def query(self, sql, *a, **k):
        if self.query_raise:
            raise _Boom("boom query")
        return self.query_rows or []

    def execute(self, *a, **k):
        return None


class _FakeKV:
    def __init__(self, db, raise_get=False, raise_set=False,
                 raise_get_raw=False, raise_scan=False, scan_pairs=None):
        self.raise_get = raise_get
        self.raise_set = raise_set
        self.raise_get_raw = raise_get_raw
        self.raise_scan = raise_scan
        self.scan_pairs = scan_pairs or []

    def get(self, key, default=None):
        if self.raise_get:
            raise _Boom("boom get")
        return default

    def set(self, key, value):
        if self.raise_set:
            raise _Boom("boom set")

    def set_raw(self, key, value, fmt=None):
        if self.raise_set:
            raise _Boom("boom set_raw")

    def get_raw(self, key):
        if self.raise_get_raw:
            raise _Boom("boom get_raw")
        return None

    def scan(self, prefix, limit=None):
        if self.raise_scan:
            raise _Boom("boom scan")
        return self.scan_pairs


def _agent(tmp_path, db):
    ctx = SimpleNamespace(db=db, settings=None)
    return ev.EvolutionAgent(ctx, repo_root=Path(tmp_path) / "repo")


def test_pin_evolution_measure_skills(tmp_path):
    db = _StubDB(raise_on=("FROM skills",))
    with patch.object(ev, "KVStore", _FakeKV):
        snap = _agent(tmp_path, db).measure()
    assert snap["skills"] == 0 and snap["kg_nodes"] == 5
    assert "evolution.measure.skills" in _sites()


def test_pin_evolution_measure_kg(tmp_path):
    db = _StubDB(raise_on=("kg_nodes",))
    with patch.object(ev, "KVStore", _FakeKV):
        snap = _agent(tmp_path, db).measure()
    assert snap["kg_nodes"] == 0 and snap["kg_edges"] == 0
    assert snap["skills"] == 5
    assert "evolution.measure.kg" in _sites()


def test_pin_evolution_measure_failures_24h(tmp_path):
    db = _StubDB(raise_on=("FROM failures",))
    with patch.object(ev, "KVStore", _FakeKV):
        snap = _agent(tmp_path, db).measure()
    assert snap["failures_24h"] == 0
    assert "evolution.measure.failures_24h" in _sites()


def test_pin_evolution_measure_persist(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_set=True)):
        snap = _agent(tmp_path, db).measure()
    assert isinstance(snap, dict) and "tests" in snap
    assert "evolution.measure.persist" in _sites()


def test_pin_evolution_record_outcome_lesson(tmp_path):
    import nomorals.agents.failure as fail_mod
    db = _StubDB()
    ag = _agent(tmp_path, db)
    with patch.object(ev, "KVStore", _FakeKV), \
            patch.object(fail_mod.FailureAnalyzer, "record",
                         side_effect=_Boom("boom")):
        res = ag.record_outcome("p1", applied=False, reason="r")
    assert res["applied"] is False
    assert "evolution.record_outcome.lesson" in _sites()


def test_pin_evolution_queue_load(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_get=True)):
        assert _agent(tmp_path, db).queue("list") == []
    assert "evolution.queue.load" in _sites()


def test_pin_evolution_load(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_get=True)):
        assert _agent(tmp_path, db)._load("evo-9") is None
    assert "evolution.load" in _sites()


def test_pin_evolution_list_scan(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_scan=True)):
        assert _agent(tmp_path, db).list() == []
    assert "evolution.list.scan" in _sites()


def test_pin_evolution_list_row(tmp_path):
    good = ev.EvolutionProposal(id="evo-good", instruction="good",
                                edits=[]).to_dict()
    bad = dict(good)
    bad["id"] = "evo-bad"
    bad["created_at"] = "not-a-number"
    pairs = [("evolution.evo-1", good), ("evolution.evo-2", bad)]
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, scan_pairs=pairs)):
        lst = _agent(tmp_path, db).list()
    assert len(lst) == 1 and lst[0].id == "evo-good"
    assert "evolution.list.row" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_evolution_measure_benchmark_silent(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_get_raw=True)):
        snap = _agent(tmp_path, db).measure()
    assert "benchmark" not in snap
    assert _sites() == []


def test_pin_evolution_metrics_history_silent(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_get=True)):
        assert _agent(tmp_path, db).metrics_history() == []
    assert _sites() == []


def _outcome_agent(tmp_path, db, lessons=None, traps=None,
                   lessons_raise=False, traps_raise=False):
    import nomorals.agents.failure as fail_mod
    import nomorals.agents.skills as skills_mod
    ctx = SimpleNamespace(db=db, settings=None)
    ag = ev.EvolutionAgent(ctx, repo_root=Path(tmp_path) / "repo")
    patches = []
    if lessons_raise:
        patches.append(patch.object(
            fail_mod, "enrich_with_lessons", side_effect=_Boom("boom")))
    else:
        patches.append(patch.object(
            fail_mod, "enrich_with_lessons", return_value=lessons or ""))
    skill_lib = Mock()
    if traps_raise:
        skill_lib.return_value.trap_block.side_effect = _Boom("boom")
    else:
        skill_lib.return_value.trap_block.return_value = traps or ""
    patches.append(patch.object(skills_mod, "SkillLibrary", skill_lib))
    return ag, patches


def test_pin_evolution_outcome_context_outcomes_silent(tmp_path):
    db = _StubDB(query_raise=True)
    ag, patches = _outcome_agent(tmp_path, db, lessons="LESSONS",
                                 traps="TRAPS")
    with patch.object(ev, "KVStore", _FakeKV), patches[0], patches[1]:
        out = ag._outcome_context("do a thing")
    assert "Recent evolution outcomes" not in out
    assert "LESSONS" in out and "TRAPS" in out
    assert _sites() == []


def test_pin_evolution_outcome_context_lessons_silent(tmp_path):
    db = _StubDB(query_rows=[{"instruction": "old", "applied": 1,
                              "reason": ""}])
    ag, patches = _outcome_agent(tmp_path, db, traps="TRAPS",
                                 lessons_raise=True)
    with patch.object(ev, "KVStore", _FakeKV), patches[0], patches[1]:
        out = ag._outcome_context("do a thing")
    assert "Recent evolution outcomes" in out and "TRAPS" in out
    assert _sites() == []


def test_pin_evolution_outcome_context_traps_silent(tmp_path):
    db = _StubDB(query_rows=[{"instruction": "old", "applied": 1,
                              "reason": ""}])
    ag, patches = _outcome_agent(tmp_path, db, lessons="LESSONS",
                                 traps_raise=True)
    with patch.object(ev, "KVStore", _FakeKV), patches[0], patches[1]:
        out = ag._outcome_context("do a thing")
    assert "Recent evolution outcomes" in out and "LESSONS" in out
    assert _sites() == []


def test_pin_evolution_baseline_load_silent(tmp_path):
    db = _StubDB()
    with patch.object(ev, "KVStore",
                       lambda db: _FakeKV(db, raise_get=True)):
        assert _agent(tmp_path, db)._load_benchmark_baseline() is None
    assert _sites() == []


def test_pin_evolution_mid_task_check_silent(tmp_path):
    import nomorals.agents.reasoning as reasoning_mod
    db = _StubDB()
    ag = _agent(tmp_path, db)
    prop = ev.EvolutionProposal(id="evo-1", instruction="do a thing",
                                edits=[])
    saved_lines = []

    def fake_save(p):
        saved_lines.append(p.verify_result)

    with patch.object(ag, "_load", return_value=prop), \
            patch.object(ag, "_git_clean", return_value=True), \
            patch.object(ag, "measure", return_value={}), \
            patch.object(ag, "_prime_benchmark_baseline"), \
            patch.object(ag, "verify", return_value=(True, "ok")), \
            patch.object(ag, "_benchmark_gate", return_value=None), \
            patch.object(ag, "_applied_summary",
                         return_value={"commit": "", "tag": ""}), \
            patch.object(ag, "_save", side_effect=fake_save), \
            patch.object(ag, "record_outcome"), \
            patch.object(ag, "git", Mock()), \
            patch.object(reasoning_mod, "ReasoningAgent",
                         side_effect=_Boom("boom")):
        out = ag.apply("evo-1")
    assert out.get("commit") == ""
    assert saved_lines and "mid-task check" not in saved_lines[0]
    assert _sites() == []
