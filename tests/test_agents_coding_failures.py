"""Pass #201 durability + NOT-recording pins for nomorals/agents/coding.py.

The agents swallow ledger (nomorals/agents/failures.py, namespace
``agents``, cap 200) is REUSED here. All 13 coding sites are exclusions,
so this file carries no converted-site pins: each excluded site must
stay silent when its failure path is driven. The durability half
mirrors the #141+ shape: store-level round-trip, cap, clear, namespace
isolation, corrupt recovery, save-failure counting, concurrent saves,
plus ledger-level memory degradation, disk hits, concurrent records,
and subprocess restart persistence.

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

from nomorals.agents import coding
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
    af.record_failure("coding.site-a", ValueError("boom"))
    af.record_failure("coding.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == [
        "coding.site-b", "coding.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["coding.site-b"]
    text = af.format_failures()
    assert ("2 total" in text and "coding.site-a" in text
            and "coding.site-b" in text)
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("coding.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "coding.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("coding.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["coding.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"coding.w{k}-{i}", ValueError("boom"))
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


# ---- Shared harnesses for NOT-recording pins -----------------------------

class _DBRaise:
    def query(self, *a, **k):
        raise _Boom("db down")

    def query_one(self, *a, **k):
        raise _Boom("db down")

    def execute(self, *a, **k):
        raise _Boom("db down")


def _raise(*a, **k):
    raise _Boom("down")


# ---- NOT-recording pins (excluded sites must stay silent) ----------------

def test_pin_no_record_workdir_glob(tmp_path, monkeypatch):
    def _glob(self, pat):
        raise _Boom("glob down")

    monkeypatch.setattr(Path, "glob", _glob)
    assert coding._workdir_has_tests(tmp_path) is False
    assert _sites() == []


def test_pin_no_record_substantive_lines():
    path = SimpleNamespace(read_text=_raise)
    assert coding._substantive_lines(path) == 0
    assert _sites() == []


def test_pin_no_record_response_usage():
    class _UsageBoom:
        @property
        def usage(self):
            raise RuntimeError("usage down")

    assert coding._response_usage(_UsageBoom()) is None
    assert _sites() == []


def test_pin_no_record_substantive_artifact(tmp_path, monkeypatch):
    target = tmp_path / "x.py"
    target.write_text("print('real code line one')\n" * 6, encoding="utf-8")
    monkeypatch.setattr(coding, "_substantive_lines", _raise)
    assert coding.CodingAgent._substantive_artifact(
        None, ["x.py"], tmp_path) is False
    assert _sites() == []


def test_pin_no_record_brain(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.llm.brain", None)
    fake = SimpleNamespace(router="ROUTER")
    assert coding.CodingAgent._brain(fake) == "ROUTER"
    assert _sites() == []


def test_pin_no_record_brief(monkeypatch):
    monkeypatch.setitem(sys.modules, "nomorals.agents.brief", None)
    ctx = SimpleNamespace(db=object(), router=object())
    agent = coding.CodingAgent(ctx)
    agent._resolve = _raise
    with pytest.raises(_Boom):
        agent.run("do a thing")
    assert _sites() == []


def test_pin_no_record_spec_promotion(monkeypatch):
    import types
    plan_mode = types.ModuleType("nomorals.agents.plan_mode")

    class _PlanStore:
        @classmethod
        def get(cls, plan_id):
            return SimpleNamespace(approved=True, task="t", files=[])

        @classmethod
        def approve(cls, plan_id):
            return cls.get(plan_id)

    plan_mode.PlanStore = _PlanStore
    monkeypatch.setitem(sys.modules, "nomorals.agents.plan_mode", plan_mode)
    plan_spec = types.ModuleType("nomorals.agents.plan_spec")
    plan_spec.plan_to_spec = _raise
    monkeypatch.setitem(sys.modules, "nomorals.agents.plan_spec", plan_spec)
    fake = SimpleNamespace(
        checkpoint=lambda **k: None,
        run=lambda *a, **k: "RUN-OK",
        _plan_scope=None)
    assert coding.CodingAgent.execute_plan(fake, "p1") == "RUN-OK"
    assert _sites() == []


def test_pin_no_record_plan_to_spec(monkeypatch):
    import types
    plan_mode = types.ModuleType("nomorals.agents.plan_mode")

    class _PlanStore:
        @classmethod
        def get(cls, plan_id):
            raise _Boom("plan store down")

    plan_mode.PlanStore = _PlanStore
    monkeypatch.setitem(sys.modules, "nomorals.agents.plan_mode", plan_mode)
    assert coding.CodingAgent.plan_to_spec(SimpleNamespace(), "x") is None
    assert _sites() == []


def test_pin_no_record_regenerate_read(monkeypatch):
    import types
    plan_spec = types.ModuleType("nomorals.agents.plan_spec")

    class _SpecStore:
        @classmethod
        def get(cls, spec_key):
            return SimpleNamespace(
                key="s1", goals="g", architecture="a", data_model="d",
                api="api", version="1", tasks=[{"path": "f.py"}])

    plan_spec.SpecStore = _SpecStore
    monkeypatch.setitem(sys.modules, "nomorals.agents.plan_spec", plan_spec)
    fake = SimpleNamespace(_resolve=_raise,
                           _draft=lambda *a, **k: "new code")
    out = coding.CodingAgent.regenerate_from_spec(fake, "s1")
    assert out["files"]["f.py"]["ok"] is True
    assert _sites() == []


def test_pin_no_record_regenerate_draft(monkeypatch):
    import types
    plan_spec = types.ModuleType("nomorals.agents.plan_spec")

    class _SpecStore:
        @classmethod
        def get(cls, spec_key):
            return SimpleNamespace(
                key="s1", goals="g", architecture="a", data_model="d",
                api="api", version="1", tasks=[{"path": "f.py"}])

    plan_spec.SpecStore = _SpecStore
    monkeypatch.setitem(sys.modules, "nomorals.agents.plan_spec", plan_spec)
    fake = SimpleNamespace(
        _resolve=lambda rel: SimpleNamespace(is_file=lambda: False),
        _draft=_raise)
    out = coding.CodingAgent.regenerate_from_spec(fake, "s1")
    assert out["files"]["f.py"] == {"ok": False,
                                    "reason": "generation failed"}
    assert _sites() == []


class _RaisingSettingsCtx:
    @property
    def settings(self):
        raise RuntimeError("settings down")


def test_pin_no_record_review_floor_config(monkeypatch):
    import types
    power = types.ModuleType("nomorals.agents.power")
    power.power_mode_for = lambda ctx: SimpleNamespace(active=False)
    monkeypatch.setitem(sys.modules, "nomorals.agents.power", power)
    fake = SimpleNamespace(context=_RaisingSettingsCtx())
    assert coding.CodingAgent._review_floor(fake) is False
    assert _sites() == []


def test_pin_no_record_review_floor_power(monkeypatch):
    import types
    power = types.ModuleType("nomorals.agents.power")
    power.power_mode_for = _raise
    monkeypatch.setitem(sys.modules, "nomorals.agents.power", power)
    ctx = SimpleNamespace(
        settings=SimpleNamespace(reasoning_mode="auto"))
    fake = SimpleNamespace(context=ctx)
    assert coding.CodingAgent._review_floor(fake) is False
    assert _sites() == []


def test_pin_no_record_sessions():
    fake = SimpleNamespace(db=_DBRaise())
    assert coding.CodingAgent.sessions(fake) == []
    assert _sites() == []
