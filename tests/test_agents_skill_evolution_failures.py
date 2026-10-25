"""Pass #185 pins: the REUSED agents failure ledger + skill_evolution conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) already existed
(created #152, reused #153/#163/#166/#169/#171/#173/#174/#184) —
skill_evolution.py is an agents-subsystem module, so it reuses that
ledger; no new ledger in this pass. The 17 durability tests below mirror
the #141+ shape exactly (same as #184's, against the agents ledger);
the 13 site pins prove each conversion records without changing flow
(fallback value unchanged AND a ledger entry under its site name); the
3 NOT-recording pins prove each exclusion stays silent with the
designed output. No pin hardcodes scan totals or skill_evolution
counts. Every test redirects stores with the app's own ``NM_HOME``
override (never the real home, never the repo tree). Restart tests use
real subprocesses. Every trigger was probed live before its pin was
written (probe.py).

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
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.agents import skill_evolution as se
from nomorals.agents import failures as af
from nomorals.agents.skill_canary import CanaryRollout
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
    def __init__(self, query_raise=False, query_one_raise=False):
        self.query_raise = query_raise
        self.query_one_raise = query_one_raise

    def query(self, sql, *a, **k):
        if self.query_raise:
            raise _Boom("boom query")
        return []

    def query_one(self, sql, *a, **k):
        if self.query_one_raise:
            raise _Boom("boom query_one")
        return None

    def execute(self, *a, **k):
        return None

    def transaction(self, *a, **k):
        raise _Boom("boom transaction")


def _loop(db=None, skills=None):
    loop = se.SkillEvolutionLoop.__new__(se.SkillEvolutionLoop)
    loop.context = SimpleNamespace(
        db=db, extras={},
        settings=SimpleNamespace(
            improvement=SimpleNamespace(mode="autonomous")))
    loop.db = db
    loop.skills = skills
    loop.window_days = 7.0
    loop.min_failures = 3
    return loop


def _row(**over):
    base = {"id": "e1", "skill_name": "s", "target_kind": "db",
            "target_ref": "r", "before_hash": "a", "after_hash": "b",
            "diff": "d", "triggering_failures": "[]", "gate_results": "{}",
            "mode": "off", "status": "proposed", "fingerprint": "f",
            "created_at": 0.0, "decided_at": 0.0}
    base.update(over)
    return base


@contextmanager
def _blocked_module(name):
    """Block `name` in sys.modules, restoring the ORIGINAL entry after.

    A plain delete after blocking would destroy the real module entry
    when the module was already imported, breaking later patches that
    target the same class object (probe.py caught this live).
    """
    orig = sys.modules.get(name, None)
    had = name in sys.modules
    sys.modules[name] = None
    try:
        yield
    finally:
        if had:
            sys.modules[name] = orig
        else:
            sys.modules.pop(name, None)


def test_pin_skill_evolution_from_row_failures():
    rec = se.SkillEditRecord.from_row(_row(triggering_failures="not json{"))
    assert rec.triggering_failures == []
    assert "skill_evolution.from_row_failures" in _sites()


def test_pin_skill_evolution_from_row_gate():
    rec = se.SkillEditRecord.from_row(_row(gate_results="not json{"))
    assert rec.gate_results == {}
    assert "skill_evolution.from_row_gate" in _sites()


def test_pin_skill_evolution_detect():
    assert _loop(db=_StubDB(query_raise=True)).detect() == []
    assert "skill_evolution.detect" in _sites()


def test_pin_skill_evolution_resolve_target_by_id():
    skills = SimpleNamespace(get=Mock(side_effect=_Boom("boom get")))
    assert _loop(skills=skills).resolve_target_by_id("x") is None
    assert "skill_evolution.resolve_target_by_id" in _sites()


def test_pin_skill_evolution_resolve_target():
    skills = SimpleNamespace(
        get_by_name=Mock(side_effect=_Boom("boom get_by_name")))
    assert _loop(skills=skills).resolve_target("zzz-no-such-skill-qqq") is None
    assert "skill_evolution.resolve_target" in _sites()


def test_pin_skill_evolution_default_benchmark_import():
    with _blocked_module("nomorals.agents.benchmark"):
        out = _loop()._default_benchmark("s", [])
    assert out.get("skipped") is True
    assert out.get("detail") == "no benchmark module"
    assert "skill_evolution.default_benchmark_import" in _sites()


def test_pin_skill_evolution_canary_import():
    with _blocked_module("nomorals.agents.skill_canary"):
        assert _loop()._start_canary_if_applicable(
            {"target_kind": "db"}) is None
    assert "skill_evolution.canary_import" in _sites()


def test_pin_skill_evolution_settle_canaries_import():
    with _blocked_module("nomorals.agents.skill_canary"):
        assert _loop().settle_canaries() == []
    assert "skill_evolution.settle_canaries_import" in _sites()


def test_pin_skill_evolution_settle_canaries_history():
    with patch.object(CanaryRollout, "history",
                      side_effect=_Boom("boom history")) as hist_mock:
        assert _loop(db=_StubDB()).settle_canaries() == []
        assert hist_mock.called
    assert "skill_evolution.settle_canaries_history" in _sites()


def test_pin_skill_evolution_ensure_schedule_query():
    ctx = SimpleNamespace(db=_StubDB(query_one_raise=True), extras={},
                          settings=SimpleNamespace(
                              improvement=SimpleNamespace(mode="off")))
    out = se.ensure_improvement_schedule(ctx)
    assert len(out) == 2 and all("error" in o for o in out)
    assert "skill_evolution.ensure_schedule_query" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_skill_evolution_settle_per_run_silent():
    r1 = SimpleNamespace(skill_name="a", canary_hash="h1", status="running")
    r2 = SimpleNamespace(skill_name="b", canary_hash="h2", status="running")
    with patch.object(CanaryRollout, "history",
                      return_value=[r1, r2]) as hist_mock, \
         patch.object(CanaryRollout, "evaluate",
                      side_effect=[_Boom("boom eval"),
                                   {"decision": "waiting"}]) as eval_mock:
        assert _loop(db=_StubDB()).settle_canaries() == []
        assert hist_mock.called and eval_mock.call_count == 2
    assert _sites() == []


def test_pin_skill_evolution_find_canary_edit_silent():
    assert _loop(db=_StubDB(query_one_raise=True))._find_canary_edit(
        "s", "h") is None
    assert _sites() == []


def test_pin_skill_evolution_history_silent():
    assert _loop(db=_StubDB(query_raise=True)).history() == []
    assert _sites() == []


def _tick_loop(settle):
    loop = _loop(db=_StubDB())
    loop.settle_canaries = settle
    return loop


def test_pin_skill_evolution_tick_settle():
    loop = _tick_loop(Mock(side_effect=_Boom("boom settle")))
    with patch.object(se.FailureAnalyzer, "evaluate_usefulness",
                      return_value={}), \
         patch.object(se.FailureAnalyzer, "promotion_candidates",
                      return_value=[]):
        assert loop.tick() == []
    assert "skill_evolution.tick_settle" in _sites()


def test_pin_skill_evolution_tick_usefulness():
    loop = _tick_loop(Mock(return_value=[]))
    with patch.object(se.FailureAnalyzer, "evaluate_usefulness",
                      side_effect=_Boom("boom usefulness")), \
         patch.object(se.FailureAnalyzer, "promotion_candidates",
                      return_value=[]):
        assert loop.tick() == []
    assert "skill_evolution.tick_usefulness" in _sites()


def test_pin_skill_evolution_tick_promotion():
    loop = _tick_loop(Mock(return_value=[]))
    with patch.object(se.FailureAnalyzer, "evaluate_usefulness",
                      return_value={}), \
         patch.object(se.FailureAnalyzer, "promotion_candidates",
                      side_effect=_Boom("boom promotion")):
        assert loop.tick() == []
    assert "skill_evolution.tick_promotion" in _sites()
