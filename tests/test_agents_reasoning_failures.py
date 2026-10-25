"""Pass #163 pins: the agents failure ledger (REUSED from #152) + reasoning conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) already exists
from #152 and was reused by #153; this pass reuses it again with site
names ``reasoning.*`` — no new ledger file. The 17 durability tests
mirror the #141+ shape exactly; the 18 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger
entry under its site name); the 4 exclusion pins prove the designed
probe/routine-signal outcome AND no ledger entry. No pin hardcodes scan
totals or reasoning counts. Every test redirects stores with the app's
own ``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written.
"""
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.agents import failures as af
from nomorals.agents import reasoning as rn
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


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
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

class _BadSettingsCtx:
    @property
    def settings(self):
        raise RuntimeError("boom")


class _BadUsage:
    @property
    def prompt_tokens(self):
        raise ValueError("boom")


class _RaisingQueryDB:
    def query(self, *args, **kwargs):
        raise sqlite3.OperationalError("boom")


class _EmptyQueryDB:
    def query(self, *args, **kwargs):
        return []


class _SkillLibEmpty:
    def __init__(self, db):
        pass

    def recall(self, *args, **kwargs):
        return []

    def match_errors(self, *args, **kwargs):
        return []

    def match_preventions(self, *args, **kwargs):
        return []


class _SkillLibRaisingRecall(_SkillLibEmpty):
    def recall(self, *args, **kwargs):
        raise RuntimeError("boom")


class _SkillLibRaisingMatch(_SkillLibEmpty):
    def match_errors(self, *args, **kwargs):
        raise RuntimeError("boom")


class _RaisingKV:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


class _RaisingFailureAnalyzer:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("boom")


def _agent(db=None):
    """ReasoningAgent with a realistic settings-bearing context, and the
    ledger cleared AFTER construction (the __init__ hook-budget probe
    may itself record on exotic contexts)."""
    ctx = SimpleNamespace(db=db, settings=SimpleNamespace(reasoning_mode="auto"))
    agent = rn.ReasoningAgent(ctx)
    af.clear_failures()
    return agent


def _stub_engine_quiet(agent):
    agent.engine._llm = lambda *args, **kwargs: ""
    agent._journal = lambda *args, **kwargs: None


def test_pin_response_usage():
    assert rn._response_usage(SimpleNamespace(usage=_BadUsage())) is None
    assert "reasoning.response_usage" in _sites()


def test_pin_enabled_mode():
    assert rn.reasoning_enabled(_BadSettingsCtx()) is False
    assert "reasoning.enabled_mode" in _sites()


def test_pin_enabled_power(monkeypatch):
    monkeypatch.setattr("nomorals.agents.power.power_mode_for", _boom)
    ctx = SimpleNamespace(settings=SimpleNamespace(reasoning_mode="auto"))
    assert rn.reasoning_enabled(ctx, complex_ok=True) is True
    assert "reasoning.enabled_power" in _sites()


def test_pin_permitted_mode():
    assert rn.reasoning_permitted(_BadSettingsCtx()) is True
    assert "reasoning.permitted_mode" in _sites()
    assert af.failure_count() == 1


def test_pin_hook_budget(monkeypatch):
    monkeypatch.setattr("nomorals.agents.power.power_mode_for", _boom)
    ctx = SimpleNamespace(settings=SimpleNamespace(reasoning_mode="auto"))
    assert rn._hook_budget(ctx) == (3, 45.0)
    assert "reasoning.hook_budget" in _sites()


def test_pin_precheck_family(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibEmpty)
    agent = _agent(db=_RaisingQueryDB())
    out = agent.mid_task_check("review plan", log=False)
    assert not any("same action family" in r for r in out["risks"])
    assert "reasoning.precheck_family" in _sites()


def test_pin_precheck_traps(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibRaisingMatch)
    agent = _agent(db=None)
    out = agent.mid_task_check("review plan", log=False)
    assert not any("known trap" in r for r in out["risks"])
    assert "reasoning.precheck_traps" in _sites()


def test_pin_precheck_journal(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibEmpty)
    monkeypatch.setattr("nomorals.agents.reasoning.KVStore", _RaisingKV)
    agent = _agent(db=_EmptyQueryDB())
    out = agent.mid_task_check("review plan", log=True)
    assert out["action"] == "review plan"
    assert "reasoning.precheck_journal" in _sites()


def test_pin_precheck_mine(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibEmpty)
    agent = _agent(db=None)
    monkeypatch.setattr(agent, "mine_preventions", _boom)
    out = agent.mid_task_check("review plan", hard_risks=["hard risk: test"], log=False)
    assert out["proceed"] is False
    assert "reasoning.precheck_mine" in _sites()


def test_pin_midtask_journal(monkeypatch):
    monkeypatch.setattr("nomorals.agents.reasoning.KVStore", _RaisingKV)
    agent = _agent(db=object())
    assert agent.mid_task_journal() == []
    assert "reasoning.midtask_journal" in _sites()


def test_pin_plan_tools_permitted(monkeypatch):
    monkeypatch.setattr("nomorals.agents.reasoning.reasoning_permitted", _boom)
    agent = _agent(db=None)
    assert agent.plan_tools("do x", [("tool1", "desc")]) == []
    assert "reasoning.plan_tools_permitted" in _sites()


def test_pin_journal_write(monkeypatch):
    monkeypatch.setattr("nomorals.agents.reasoning.KVStore", _RaisingKV)
    agent = _agent(db=object())
    assert agent._journal("k", {"ts": 1.0}, bound=10) is None
    assert "reasoning.journal_write" in _sites()


def test_pin_journal_read(monkeypatch):
    monkeypatch.setattr("nomorals.agents.reasoning.KVStore", _RaisingKV)
    ctx = SimpleNamespace(db=object())
    assert rn.ReasoningAgent._read_journal(ctx, "k") == []
    assert "reasoning.journal_read" in _sites()


def test_pin_record_outcome(monkeypatch):
    monkeypatch.setattr("nomorals.agents.failure.FailureAnalyzer", _RaisingFailureAnalyzer)
    agent = _agent(db=None)
    assert agent.record_outcome("scope", "task", ok=False) is None
    assert "reasoning.record_outcome" in _sites()


def test_pin_course_family(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibEmpty)
    agent = _agent(db=_RaisingQueryDB())
    _stub_engine_quiet(agent)
    out = agent.course_correct("deploy failed", attempts=1)
    assert out["signals"] == []
    assert "reasoning.course_family" in _sites()


def test_pin_course_skills(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibRaisingRecall)
    agent = _agent(db=None)
    _stub_engine_quiet(agent)
    out = agent.course_correct("deploy failed", attempts=1)
    assert not any("historically weak" in s for s in out["signals"])
    assert "reasoning.course_skills" in _sites()


def test_pin_course_traps(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibRaisingMatch)
    agent = _agent(db=None)
    _stub_engine_quiet(agent)
    out = agent.course_correct("deploy failed", attempts=1)
    assert not any("known trap" in s for s in out["signals"])
    assert "reasoning.course_traps" in _sites()


def test_pin_prevention_hits():
    assert rn.ReasoningAgent._prevention_hits(SimpleNamespace(body="{invalid")) == 0
    assert "reasoning.prevention_hits" in _sites()


# ---- Exclusion pins (NOT-recording) ---------------------------------------

def test_excl_knowledge_block(monkeypatch):
    monkeypatch.setattr("nomorals.agents.kg.knowledge_context", _boom)
    assert rn._knowledge_block(SimpleNamespace(), "q") == ""
    assert af.failure_count() == 0


def test_excl_precheck_advisory(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibEmpty)
    monkeypatch.setattr("nomorals.agents.reasoning.reasoning_permitted", _boom)
    agent = _agent(db=None)
    out = agent.mid_task_check("review plan", log=False)
    assert out["advice"] == ""
    assert af.failure_count() == 0


def test_excl_course_budget_exhausted(monkeypatch):
    monkeypatch.setattr("nomorals.agents.skills.SkillLibrary", _SkillLibEmpty)
    agent = _agent(db=None)
    agent._journal = lambda *args, **kwargs: None

    def _exhausted(*args, **kwargs):
        raise rn._BudgetExhausted()

    agent.engine._llm = _exhausted
    out = agent.course_correct("deploy failed", attempts=1)
    assert out["pivot"] == "" and out["should_pivot"] is False
    assert af.failure_count() == 0


def test_excl_challenge_budget_exhausted(monkeypatch):
    agent = _agent(db=None)
    agent._journal = lambda *args, **kwargs: None

    def _exhausted(*args, **kwargs):
        raise rn._BudgetExhausted()

    agent.engine._llm = _exhausted
    out = agent.self_challenge("the cache TTL should be 60s", evidence="cited")
    assert out["verdict"] == "no attack landed"
    assert af.failure_count() == 0
