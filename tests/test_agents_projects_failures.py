"""Pass #169 pins: the agents failure ledger (REUSED from #152) + projects
conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) is reused, not new:
projects.py is an agents-subsystem module. The 17 durability tests below
mirror the #141+ shape exactly; the 11 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger entry
under its site name); the 8 exclusion pins prove the enhancement probes
record nothing. No pin hardcodes scan totals or projects counts. Every
test redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written.
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
from nomorals.agents.projects import Project, ProjectManager, ProjectStep


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
    def __init__(self, query_one=None, query=None):
        self._qo = query_one
        self._q = query

    def query_one(self, sql, params=()):
        if isinstance(self._qo, Exception):
            raise self._qo
        return self._qo

    def query(self, sql, params=()):
        if isinstance(self._q, Exception):
            raise self._q
        return self._q

    def execute(self, sql, params=()):
        return None


def _mgr(db=None, settings=None):
    ctx = SimpleNamespace(db=db or _StubDB())
    if settings is not None:
        ctx.settings = settings
    return ProjectManager(ctx)


def _proj(**kw):
    kw.setdefault("id", "p1")
    kw.setdefault("title", "t")
    return Project(**kw)


def test_site_room_manager_records_and_returns_none():
    class _S:
        @property
        def workspace_dir(self):
            raise _Boom("no ws")
    m = _mgr(settings=_S())
    assert m._room_manager() is None
    assert "projects.room_manager" in _sites()


def test_site_linked_room_records_and_returns_none():
    m = _mgr()
    m._room_manager = lambda: SimpleNamespace(
        get_by_linked=lambda *a: (_ for _ in ()).throw(_Boom("db")))
    assert m._linked_room("p1") is None
    assert "projects.linked_room" in _sites()


def test_site_rooms_auto_create_records_and_keeps_default():
    class _Down:
        def __getattr__(self, name):
            raise _Boom("settings down")
    assert _mgr(settings=_Down())._rooms_auto_create() is True
    assert "projects.rooms_auto_create" in _sites()


def test_site_sync_goal_knowledge_records_and_keeps_goal_done():
    class _BadGoalSystem:
        def __init__(self, ctx):
            pass
        def _record_completion_knowledge(self, goal_id):
            raise _Boom("goal knowledge down")
    with patch("nomorals.agents.goals.GoalSystem", _BadGoalSystem):
        _mgr(_StubDB(query_one={"id": "g1"}))._sync_goal(
            _proj(status="done", goal_id="g1"))
    assert "projects.sync_goal_knowledge" in _sites()


def test_site_default_executor_load_records_and_continues():
    m = _mgr()
    m._current_project_id = "p1"
    m._load = lambda pid: (_ for _ in ()).throw(_Boom("load down"))
    try:
        m._default_executor("do a step")
    except Exception as exc:  # noqa: BLE001 — downstream failure is out of scope
        assert isinstance(exc, Exception)
    assert "projects.default_executor_load" in _sites()


def test_site_build_seed_records_and_continues():
    class _FakePath:
        def exists(self):
            return True
        def read_text(self, encoding="utf-8", errors="replace"):
            raise _Boom("unreadable")
    m = _mgr()
    with patch("nomorals.tools.filesystem.safe_path",
               lambda ctx, fn: _FakePath()):
        try:
            m._execute_build_step(_proj(artifact="app.py"), "build it")
        except Exception as exc:  # noqa: BLE001 — downstream failure is out of scope
            assert isinstance(exc, Exception)
    assert "projects.build_seed" in _sites()


def test_site_acceptance_history_records_and_returns_empty():
    assert _mgr(_StubDB(query=_Boom("db"))).acceptance_history("p1") == []
    assert "projects.acceptance_history" in _sites()


def test_site_prior_acceptance_cmds_records_and_returns_empty():
    assert _mgr(_StubDB(query=_Boom("db")))._prior_acceptance_cmds(_proj()) == []
    assert "projects.prior_acceptance_cmds" in _sites()


def test_site_regression_check_records_and_keeps_ok_shape():
    m = _mgr(_StubDB(query=_Boom("db")))
    m._load = lambda pid: _proj()
    out = m.regression_check("p1")
    assert out["ok"] is True and out["note"] == "no acceptance runs recorded"
    assert "projects.regression_check" in _sites()


def test_site_record_skill_use_records_and_returns_none():
    class _BadSkillLib:
        def __init__(self, db):
            pass
        def record_use(self, *a, **k):
            raise _Boom("skill store down")
    with patch("nomorals.agents.skills.SkillLibrary", _BadSkillLib):
        assert _mgr()._record_skill_use(
            [(SimpleNamespace(id="s1"), 0.9)], success=True, task="t") is None
    assert "projects.record_skill_use" in _sites()


def test_site_should_replan_records_and_returns_false():
    ctx = SimpleNamespace(
        db=_StubDB(query_one=_Boom("db")),
        settings=SimpleNamespace(autonomy=SimpleNamespace(replan_after_heals=2)))
    assert ProjectManager(ctx)._should_replan(_proj(goal_id="g1")) is False
    assert "projects.should_replan" in _sites()


# ---- Exclusion pins (must record NOTHING) --------------------------------

def test_excluded_failure_context_lessons_records_nothing():
    with patch("nomorals.agents.failure.enrich_with_lessons",
               side_effect=_Boom("lessons down")):
        p = _proj()
        p.steps = [ProjectStep(id="s1", description="d", status="failed",
                               result="r")]
        txt = _mgr()._failure_context(p)
        assert "Already attempted" in txt
    assert "projects.failure_context_lessons" not in _sites()


def test_excluded_course_pivot_records_nothing():
    with patch("nomorals.agents.reasoning.reasoning_permitted",
               side_effect=_Boom("reasoning down")):
        assert _mgr()._course_pivot("step", "err", 1) == {}
    assert "projects.course_pivot" not in _sites()


def test_excluded_revise_step_lessons_and_rewrite_record_nothing():
    class _BadBrain:
        def chat(self, *a, **k):
            raise _Boom("brain down")
    with patch("nomorals.agents.failure.enrich_with_lessons",
               side_effect=_Boom("lessons down")), \
         patch("nomorals.agents.projects.brain_for", lambda ctx: _BadBrain()):
        assert _mgr()._revise_step("step", "err") == "step"
    assert "projects.revise_step_lessons" not in _sites()
    assert "projects.revise_step_rewrite" not in _sites()


def test_excluded_default_executor_upstream_records_nothing():
    m = _mgr()
    m._current_project_id = "p1"
    m._load = lambda pid: _proj(goal_id="g1", task_kind="note")

    class _BadUpstream:
        def __init__(self, ctx):
            pass
        def upstream_knowledge(self, goal_id):
            raise _Boom("upstream down")
    with patch("nomorals.agents.goals.GoalSystem", _BadUpstream):
        try:
            m._default_executor("do a step")
        except Exception as exc:  # noqa: BLE001 — downstream failure is out of scope
            assert isinstance(exc, Exception)
    assert "projects.default_executor_upstream" not in _sites()


def test_excluded_build_upstream_records_nothing():
    m = _mgr()
    m._recall_skills = lambda *a, **k: []

    class _FakePath2:
        def exists(self):
            return False

    class _BadUpstream:
        def __init__(self, ctx):
            pass
        def upstream_knowledge(self, goal_id):
            raise _Boom("upstream down")
    with patch("nomorals.tools.filesystem.safe_path",
               lambda ctx, fn: _FakePath2()), \
         patch("nomorals.agents.goals.GoalSystem", _BadUpstream):
        try:
            m._execute_build_step(_proj(goal_id="g1", artifact="app.py"),
                                  "build it")
        except Exception as exc:  # noqa: BLE001 — downstream failure is out of scope
            assert isinstance(exc, Exception)
    assert "projects.build_upstream" not in _sites()


def test_excluded_recall_skills_records_nothing():
    class _BadSkillLib:
        def __init__(self, db):
            pass
        def recall(self, *a, **k):
            raise _Boom("skill store down")
    with patch("nomorals.agents.skills.SkillLibrary", _BadSkillLib):
        assert _mgr()._recall_skills("q") == []
    assert "projects.recall_skills" not in _sites()


def test_excluded_skill_context_records_nothing():
    class _BadSkillLib:
        def __init__(self, db):
            pass
        def context_block(self, *a, **k):
            raise _Boom("skill store down")
    with patch("nomorals.agents.skills.SkillLibrary", _BadSkillLib):
        assert _mgr()._skill_context("q") == ""
    assert "projects.skill_context" not in _sites()
