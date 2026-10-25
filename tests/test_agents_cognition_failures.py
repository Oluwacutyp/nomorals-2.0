"""Pass #189 durability + behavior pins for nomorals/agents/cognition.py.

The agents swallow ledger (nomorals/agents/failures.py, namespace
``agents``, cap 200) is REUSED here; the six converted cognition sites
record into it, and the nine excluded sites must stay silent. The
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
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents import cognition as cog
from nomorals.agents import failures as af
from nomorals.core import failure_store as fs

ROOT = Path(__file__).resolve().parents[1]


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


class _Boom(Exception):
    pass


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


def test_store_save_home_resolution_failure_never_raises_and_counts(
        monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    import unittest.mock as mock
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents")
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
    entries, _ = fs.Store("agents").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(
        monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    import unittest.mock as mock
    with mock.patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        af.record_failure("no-home", ValueError("boom"))
        assert af.failure_count() == 1
        assert [f.site for f in af.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no agents failures recorded."
    af.record_failure("cognition.site-a", ValueError("boom"))
    af.record_failure("cognition.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == [
        "cognition.site-b", "cognition.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["cognition.site-b"]
    text = af.format_failures()
    assert ("2 total" in text and "cognition.site-a" in text
            and "cognition.site-b" in text)
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("cognition.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "cognition.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("cognition.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["cognition.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"cognition.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
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


# ---- Shared harnesses for site + NOT-recording pins ----------------------

def _autonomy(**over):
    base = dict(daily_model_calls=0, interval_hours=6.0,
                adaptive_cadence=True, max_project_heals=3, enabled=False)
    base.update(over)
    return SimpleNamespace(**base)


def _ctx(db=None, settings=None, **extra):
    return SimpleNamespace(
        db=db,
        settings=settings or SimpleNamespace(autonomy=_autonomy()),
        **extra)


class _KV:
    """KVStore stub: mode 'raise' or 'healthy' (return the default)."""

    def __init__(self, mode):
        self.mode = mode

    def __call__(self, db):
        return self

    def get(self, key, default=None):
        if self.mode == "raise":
            raise _Boom("kv down")
        return default

    def set(self, key, value):
        if self.mode == "raise":
            raise _Boom("kv down")
        return None


class _DBRaise:
    def query(self, *a, **k):
        raise _Boom("db down")

    def query_one(self, *a, **k):
        raise _Boom("db down")

    def execute(self, *a, **k):
        raise _Boom("db down")


class _DBRows:
    def __init__(self, rows=None):
        self.rows = rows if rows is not None else []

    def query(self, *a, **k):
        return self.rows

    def query_one(self, *a, **k):
        return None

    def execute(self, *a, **k):
        return None


def _patch_status_modules(monkeypatch, raising_job=False):
    import types
    goals = types.ModuleType("nomorals.agents.goals")

    class _GS:
        def __init__(self, ctx):
            pass

        def status(self):
            return {"by_status": {}}

    goals.GoalSystem = _GS
    monkeypatch.setitem(sys.modules, "nomorals.agents.goals", goals)
    improv = types.ModuleType("nomorals.agents.improvement")

    class _IL:
        def __init__(self, ctx):
            self.settings = SimpleNamespace(mode="off")

        def pending_approvals(self):
            return []

        def history(self, limit=3):
            return []

    improv.ImprovementLoop = _IL
    monkeypatch.setitem(sys.modules, "nomorals.agents.improvement", improv)
    si = types.ModuleType("nomorals.self_improvement")

    class _Job:
        def __init__(self, ctx):
            pass

        def status(self):
            if raising_job:
                raise _Boom("training down")
            return {"decision": {}}

    si.SelfImprovementJob = _Job
    monkeypatch.setitem(sys.modules, "nomorals.self_improvement", si)


# ---- Site pins (converted sites must record) -----------------------------

def test_pin_cognition_kv_override(monkeypatch):
    monkeypatch.setattr(cog, "KVStore", _KV("raise"))
    assert cog._kv_override(_ctx(db=object())) is None
    assert "cognition.kv_override" in _sites()


def test_pin_cognition_budget_usage(monkeypatch):
    monkeypatch.setattr(cog, "KVStore", _KV("raise"))
    out = cog.ModelBudget(_ctx(db=object())).usage()
    assert out["calls"] == 0 and out["tokens"] == 0 and out["ticks"] == 0
    assert "cognition.budget_usage" in _sites()


def test_pin_cognition_reservations_raw(monkeypatch):
    monkeypatch.setattr(cog, "KVStore", _KV("raise"))
    assert cog.ModelBudget(_ctx(db=object()))._reservations_raw() == []
    assert "cognition.reservations_raw" in _sites()


def test_pin_cognition_usage_snapshot():
    router = SimpleNamespace(
        providers=lambda: (_ for _ in ()).throw(_Boom("router down")))
    assert cog.ModelBudget.usage_snapshot(
        SimpleNamespace(router=router)) is None
    assert "cognition.usage_snapshot" in _sites()


def test_pin_cognition_portfolio_pending():
    loop = cog.CognitiveLoop(_ctx(db=_DBRaise()))
    assert loop._portfolio_has_pending_work() is False
    assert "cognition.portfolio_pending" in _sites()


def test_pin_cognition_status_training(monkeypatch):
    _patch_status_modules(monkeypatch, raising_job=True)
    monkeypatch.setattr(cog, "KVStore", _KV("healthy"))
    st = cog.CognitiveLoop(_ctx(db=_DBRows())).status()
    assert st["training"] == {}
    assert "cognition.status_training" in _sites()


# ---- NOT-recording pins (excluded sites must stay silent) ----------------

def test_pin_no_record_flip(monkeypatch):
    monkeypatch.setattr(cog, "KVStore", _KV("healthy"))
    assert cog.set_autonomy_enabled(
        _ctx(db=object(), settings=SimpleNamespace()), True) is True
    assert _sites() == []


class _RaisingSettingsCtx:
    def __init__(self, db):
        self.db = db

    @property
    def settings(self):
        raise RuntimeError("settings down")


def test_pin_no_record_autonomy_outer(monkeypatch):
    monkeypatch.setattr(cog, "KVStore", _KV("healthy"))
    assert cog.autonomy_enabled(_RaisingSettingsCtx(object())) is False
    assert _sites() == []


def test_pin_no_record_effective_interval(monkeypatch):
    monkeypatch.setattr(cog, "KVStore", _KV("raise"))
    assert cog.CognitiveLoop(_ctx(db=object())).effective_interval() == 6.0
    assert _sites() == []


def test_pin_no_record_adaptive_interval():
    loop = cog.CognitiveLoop(_ctx(db=_DBRaise()))
    assert loop.adaptive_interval() == 6.0
    assert _sites() == []


def test_pin_no_record_throttle(monkeypatch):
    class _MB:
        def __init__(self, ctx):
            pass

        @property
        def cap(self):
            return 100

        @property
        def pressure(self):
            raise _Boom("pressure down")

    monkeypatch.setattr(cog, "ModelBudget", _MB)
    monkeypatch.setattr(cog, "KVStore", _KV("healthy"))
    loop = cog.CognitiveLoop(_ctx(db=_DBRows()))
    summary = {}
    loop._update_adaptive_cadence(summary)
    assert summary.get("next_interval_hours") == 6.0
    assert _sites() == []


def test_pin_no_record_set_interval(monkeypatch):
    class _Sched:
        def set_interval(self, *a, **k):
            raise _Boom("scheduler down")

    monkeypatch.setattr(cog, "KVStore", _KV("healthy"))
    loop = cog.CognitiveLoop(_ctx(db=_DBRows(), extras={"scheduler": _Sched()}))
    summary = {}
    loop._update_adaptive_cadence(summary)
    assert summary.get("next_interval_hours") == 6.0
    assert _sites() == []


def test_pin_no_record_recent():
    assert cog.CognitiveLoop(_ctx(db=_DBRaise())).recent() == []
    assert _sites() == []


def test_pin_no_record_report():
    rep = cog.CognitiveLoop(_ctx(db=_DBRaise())).report()
    assert rep["ticks"] == 0 and rep["total_seconds"] == 0.0
    assert rep["recent"] == []
    assert _sites() == []


def test_pin_no_record_adapt(monkeypatch):
    import types
    goal = SimpleNamespace(id="g1",
                           steps=[SimpleNamespace(status="blocked")],
                           priority=1, created_at=0, project_id="")

    class _GS:
        def __init__(self, ctx):
            pass

        def list(self, status=None, limit=None):
            return [goal]

        def adapt(self, gid):
            raise _Boom("adapt down")

        def status(self):
            return {"by_status": {}}

    goals = types.ModuleType("nomorals.agents.goals")
    goals.GoalSystem = _GS
    monkeypatch.setitem(sys.modules, "nomorals.agents.goals", goals)
    out = cog.CognitiveLoop(_ctx(db=_DBRaise()))._tick_goals()
    assert "g1" not in out["advanced"]
    assert _sites() == []
