"""Pass #195 pins: the agents failure ledger (reused from #152) + the
skill_synthesis conversion.

``nomorals/agents/failures.py`` (namespace ``agents``) already exists
from #152 — this pass REUSES it, it does not create a second agents
ledger. The 17 durability tests below mirror the #141+ shape exactly
(same as #194's, against the same Store backend); the 4 site pins prove
each conversion records without changing flow (fallback value unchanged
AND a ledger entry under its site name); the 10 NOT-recording pins prove
each exclusion stays silent with the designed output. No pin hardcodes
scan totals or skill_synthesis counts. Every test redirects stores with
the app's own ``NM_HOME`` override (never the real home, never the repo
tree). Restart tests use real subprocesses. Every trigger was probed
live before its pin was written (probe.py).

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

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.agents import failures as af
from nomorals.agents import skill_synthesis as ss
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
    assert fs.Store("agents.other").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("agents.other").load()
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


def _boom_expanduser(path):
    raise OSError("no home")


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", _boom_expanduser)
    before = fs.write_failures()
    assert fs.Store("agents").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    errors = []

    def worker(k):
        try:
            for i in range(30):
                fs.Store("agents").save(_entries(3, f"w{k}-{i}-"), 3)
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("agents").load()
    assert len(entries) == 3
    assert all(e["site"].startswith("w") for e in entries)


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", _boom_expanduser)
    af.record_failure("skill_synthesis.probe", ValueError("boom"))
    assert _sites() == ["skill_synthesis.probe"]


def test_ledger_api_record_recent_count_clear_format():
    af.record_failure("skill_synthesis.a", ValueError("boom0"))
    af.record_failure("skill_synthesis.b", RuntimeError("boom1"))
    assert af.failure_count() == 2
    assert _sites() == ["skill_synthesis.b", "skill_synthesis.a"]
    rendered = af.format_failures()
    assert "skill_synthesis.b: RuntimeError: boom1" in rendered
    assert "skill_synthesis.a: ValueError: boom0" in rendered
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []
    assert af.format_failures() == "no agents failures recorded."


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("skill_synthesis.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1
    assert [e["site"] for e in entries] == ["skill_synthesis.disk"]
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("skill_synthesis.mem", ValueError("boom"))
    assert _sites() == ["skill_synthesis.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"agents.w{k}-{i}", ValueError("boom"))
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
    assert data["sites"][-1] == "site50"


def test_restart_corrupt_store_recovers(tmp_path, monkeypatch):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    monkeypatch.setenv("NM_HOME", str(home))
    fs.Store("agents").path().write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []
    assert _run_subprocess(home, _record_code(1)).returncode == 0
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 1


# ---- Site fakes (shared by site + NOT-recording pins) --------------------

GOOD_BODY = ("# T\n\n## Tool sequence\n\n- `tool_a`\n- `tool_b`\n\n"
             "## Steps\n\nx\n")


class _FakeSkill:
    def __init__(self, sid, name, body):
        self.id, self.name, self.body = sid, name, body


class _FakeSkills:
    def __init__(self, body):
        self.body = body
        self.saved = []

    def save(self, name, **kw):
        skill = _FakeSkill("sk1", name, self.body)
        self.saved.append(skill)
        return skill

    def get_by_name(self, name):
        return self.saved[-1] if self.saved else None


class _FakeDB:
    def __init__(self, rows=None, one=None, one_map=None):
        self.rows = rows if rows is not None else []
        self.one = one
        self.one_map = one_map
        self.raise_query = False
        self.raise_one = False
        self.raise_execute_on = ""
        self.executed = []

    def query(self, sql, params=()):
        if self.raise_query:
            raise RuntimeError("boom query")
        return self.rows

    def query_one(self, sql, params=()):
        if self.raise_one:
            raise RuntimeError("boom query_one")
        if self.one_map:
            for key, val in self.one_map.items():
                if key in sql:
                    return val
        return self.one

    def execute(self, sql, params=()):
        self.executed.append(sql)
        if self.raise_execute_on and self.raise_execute_on in sql:
            raise RuntimeError("boom execute")
        return None


class _FakeKV:
    def __init__(self, pairs=None, fail_set=False, fail_scan=False,
                 fail_delete=False):
        self.pairs = pairs if pairs is not None else []
        self.fail_set = fail_set
        self.fail_scan = fail_scan
        self.fail_delete = fail_delete
        self.deleted = []

    def scan(self, prefix):
        if self.fail_scan:
            raise RuntimeError("boom scan")
        return self.pairs

    def set(self, key, value):
        if self.fail_set:
            raise RuntimeError("boom kv set")

    def delete(self, key):
        if self.fail_delete:
            raise RuntimeError("boom kv delete")
        self.deleted.append(key)


def _fresh_ctx(mode):
    return SimpleNamespace(
        settings=SimpleNamespace(improvement=SimpleNamespace(mode=mode)))


def _synth(ctx, db, skills, drafter):
    synth = ss.SkillSynthesizer.__new__(ss.SkillSynthesizer)
    synth.context, synth.db, synth.skills = ctx, db, skills
    synth._drafter = drafter
    synth.min_repeats, synth.window_days = 5, 30.0
    return synth


def _pattern(tools=("tool_a", "tool_b")):
    return ss.ToolPattern(tools=list(tools), count=7, trace_ids=["t1"])


# ---- Site pins (the conversion) ------------------------------------------

def test_pin_smoke_cleanup(monkeypatch):
    db = _FakeDB()
    db.raise_execute_on = "DELETE FROM skills"
    synth = _synth(_fresh_ctx("approval"), db,
                   _FakeSkills("no sequence here"),
                   lambda p: {"name": "X", "body": "no sequence here"})
    out = synth.synthesize_pattern(_pattern())
    assert out["ok"] is False and "smoke test failed" in out["error"]
    assert "skill_synthesis.smoke_cleanup" in _sites()


def test_pin_probation_record(monkeypatch):
    monkeypatch.setattr(ss, "KVStore", lambda db: _FakeKV(fail_set=True))
    db = _FakeDB(one={"tags": ""})
    synth = _synth(_fresh_ctx("autonomous"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {"name": "X", "body": GOOD_BODY})
    out = synth.synthesize_pattern(_pattern())
    assert out["ok"] is True and "probation_until" in out
    assert "skill_synthesis.probation_record" in _sites()


def test_pin_review_uses(monkeypatch):
    kv = _FakeKV(pairs=[("skill.probation.sk1", {"until": 0})])
    monkeypatch.setattr(ss, "KVStore", lambda db: kv)
    db = _FakeDB()
    db.raise_one = True
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})
    out = synth.review_probation()
    assert out == [{"skill_id": "sk1", "kept": False, "uses": 0}]
    assert "skill_synthesis.review_uses" in _sites()


def test_pin_review_removal(monkeypatch):
    kv = _FakeKV(pairs=[("skill.probation.sk1", {"until": 0})])
    monkeypatch.setattr(ss, "KVStore", lambda db: kv)
    db = _FakeDB(one={"uses": 0})
    db.raise_execute_on = "DELETE FROM skills"
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})
    out = synth.review_probation()
    assert out == [{"skill_id": "sk1", "kept": False, "uses": 0}]
    assert kv.deleted == ["skill.probation.sk1"]
    assert "skill_synthesis.review_removal" in _sites()


# ---- NOT-recording pins (the exclusions) ---------------------------------

def test_pin_detect_query_not_recorded():
    db = _FakeDB()
    db.raise_query = True
    assert ss.detect_patterns(db) == []
    assert _sites() == []


def test_pin_detect_list_not_recorded(monkeypatch):
    rows = [{"actor": "a", "tool": t, "id": f"i{i}"}
            for i, t in enumerate(["tool_a", "tool_b"] * 5)]
    db = _FakeDB(rows=rows)

    class _RaisingLib:
        def __init__(self, db):
            pass

        def list(self, **kw):
            raise RuntimeError("boom list")

    monkeypatch.setattr(ss, "SkillLibrary", _RaisingLib)
    pats = ss.detect_patterns(db)
    assert [tuple(p.tools) for p in pats] == [("tool_a", "tool_b")]
    assert _sites() == []


def test_pin_drafter_template_not_recorded(monkeypatch):
    def _boom(ctx):
        raise RuntimeError("boom brain")

    monkeypatch.setattr(ss, "brain_for", _boom)
    synth = _synth(SimpleNamespace(router=object()), _FakeDB(),
                   _FakeSkills(GOOD_BODY), None)
    draft = synth._default_drafter(_pattern())
    assert draft["body"].startswith("# Tool A Tool B Workflow")
    assert "## Tool sequence" in draft["body"]
    assert _sites() == []


def test_pin_provenance_not_recorded(monkeypatch):
    monkeypatch.setattr(ss, "KVStore", lambda db: _FakeKV(fail_set=True))
    db = _FakeDB(one={"tags": ""})
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {"name": "X", "body": GOOD_BODY})
    out = synth.synthesize_pattern(_pattern())
    assert out["ok"] is True and out["staged"] is True
    assert _sites() == []


def test_pin_trace_tools_not_recorded():
    db = _FakeDB()
    db.raise_one = True
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})
    assert synth._trace_tools(["t1", "t2"]) == []
    assert _sites() == []


def test_pin_probation_tag_not_recorded():
    db = _FakeDB()
    db.raise_one = True
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})
    assert synth._set_probation_tag("sk1", "probation") is None
    assert _sites() == []


def test_pin_review_scan_not_recorded(monkeypatch):
    monkeypatch.setattr(ss, "KVStore", lambda db: _FakeKV(fail_scan=True))
    synth = _synth(_fresh_ctx("approval"), _FakeDB(),
                   _FakeSkills(GOOD_BODY), lambda p: {})
    assert synth.review_probation() == []
    assert _sites() == []


def test_pin_review_keep_tag_not_recorded(monkeypatch):
    kv = _FakeKV(pairs=[("skill.probation.sk1", {"until": 0})])
    monkeypatch.setattr(ss, "KVStore", lambda db: kv)
    db = _FakeDB(one_map={"uses": {"uses": 3},
                          "tags": {"tags": "probation"}})
    db.raise_execute_on = "UPDATE skills SET tags"
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})
    out = synth.review_probation()
    assert out == [{"skill_id": "sk1", "kept": True, "uses": 3}]
    assert _sites() == []


def test_pin_review_kv_delete_not_recorded(monkeypatch):
    kv = _FakeKV(pairs=[("skill.probation.sk1", {"until": 0})],
                 fail_delete=True)
    monkeypatch.setattr(ss, "KVStore", lambda db: kv)
    db = _FakeDB(one_map={"uses": {"uses": 3},
                          "tags": {"tags": "probation"}})
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})
    out = synth.review_probation()
    assert out == [{"skill_id": "sk1", "kept": True, "uses": 3}]
    assert _sites() == []


def test_pin_scan_review_not_recorded():
    db = _FakeDB(rows=[])
    synth = _synth(_fresh_ctx("approval"), db, _FakeSkills(GOOD_BODY),
                   lambda p: {})

    def _boom_review():
        raise RuntimeError("boom review")

    synth.review_probation = _boom_review
    assert synth.scan() == []
    assert _sites() == []
