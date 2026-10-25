"""Pass #202 durability + NOT-recording pins for nomorals/agents/failure.py.

The agents swallow ledger (nomorals/agents/failures.py, namespace
``agents``, cap 200) is REUSED here. All 13 failure sites are
exclusions, so this file carries no converted-site pins: each excluded
site must stay silent when its failure path is driven. The durability
half mirrors the #141+ shape: store-level round-trip, cap, clear,
namespace isolation, corrupt recovery, save-failure counting,
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

from nomorals.agents import failure
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


def _analyzer(db=None, skills=None, router=None):
    analyzer = object.__new__(failure.FailureAnalyzer)
    analyzer.context = SimpleNamespace(router=router)
    analyzer.db = db
    analyzer.skills = skills
    return analyzer


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
    af.record_failure("failure.site-a", ValueError("boom"))
    af.record_failure("failure.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == [
        "failure.site-b", "failure.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["failure.site-b"]
    text = af.format_failures()
    assert ("2 total" in text and "failure.site-a" in text
            and "failure.site-b" in text)
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("failure.disk", ValueError("boom"))
    entries, total = fs.Store("agents").load()
    assert total == 1 and entries[0]["site"] == "failure.disk"
    af.clear_failures()
    assert fs.Store("agents").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("failure.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["failure.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"failure.w{k}-{i}", ValueError("boom"))
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


# ---- Shared harnesses for NOT-recording pins ----------------------------

class _DBRaise:
    def query(self, *a, **k):
        raise _Boom("db down")

    def query_one(self, *a, **k):
        raise _Boom("db down")

    def execute(self, *a, **k):
        raise _Boom("db down")


def _raise(*a, **k):
    raise _Boom("down")


_CODING_ROW = {"task": "t", "stderr": "e", "stdout": "", "attempt": 1,
               "created_at": 1.0}
_DEVON_ROW = {"task": "t", "digest": "d", "ts": 1.0}
_LEDGER_ROW = {"source": "tool", "summary": "s", "error": "e", "ts": 1.0,
               "skill": ""}


class _CollectDB:
    """Raises only for the named source; other sources return rows."""

    def __init__(self, down):
        self.down = down

    def query(self, sql, *a, **k):
        if self.down == "coding" and "coding_log" in sql:
            raise _Boom("coding down")
        if self.down == "devon" and "devon_memory" in sql:
            raise _Boom("devon down")
        if self.down == "ledger" and "FROM failures" in sql:
            raise _Boom("ledger down")
        if "coding_log" in sql:
            return [_CODING_ROW]
        if "devon_memory" in sql:
            return [_DEVON_ROW]
        if "FROM failures" in sql:
            return [_LEDGER_ROW]
        return []

    def query_one(self, *a, **k):
        return None

    def execute(self, *a, **k):
        return None


# ---- NOT-recording pins (excluded sites must stay silent) ----------------

def test_pin_no_record_enrich_with_lessons(monkeypatch):
    class _CtxRaise:
        def __init__(self, context):
            pass

        def prevention_context(self, *a, **k):
            raise _Boom("context down")

    monkeypatch.setattr(failure, "FailureAnalyzer", _CtxRaise)
    assert failure.enrich_with_lessons(SimpleNamespace(), "q") == ""
    assert _sites() == []


def test_pin_no_record_collect_coding():
    analyzer = _analyzer(db=_CollectDB("coding"))
    cases = analyzer.collect(limit=5)
    assert [c.source for c in cases] == ["devon", "tool"]
    assert _sites() == []


def test_pin_no_record_collect_devon():
    analyzer = _analyzer(db=_CollectDB("devon"))
    cases = analyzer.collect(limit=5)
    assert [c.source for c in cases] == ["coding", "tool"]
    assert _sites() == []


def test_pin_no_record_collect_ledger():
    analyzer = _analyzer(db=_CollectDB("ledger"))
    cases = analyzer.collect(limit=5)
    assert [c.source for c in cases] == ["coding", "devon"]
    assert _sites() == []


def test_pin_no_record_model_analyze(monkeypatch):
    monkeypatch.setattr(failure, "brain_for", _raise)
    analyzer = _analyzer(router=object())
    out = analyzer._model_analyze(
        failure.FailureCase(source="tool", summary="s", error="e"))
    assert out is None
    assert _sites() == []


def test_pin_no_record_learn_from_all_seen():
    class _SeenDownDB(_CollectDB):
        def query(self, sql, *a, **k):
            if "fingerprint FROM lessons" in sql:
                raise _Boom("seen down")
            if "FROM failures" in sql:
                return []
            return []

    skills = SimpleNamespace(save=lambda *a, **k: SimpleNamespace(id="s1"))
    analyzer = _analyzer(db=_SeenDownDB("none"), skills=skills)
    # Seen query fails -> re-analysis runs; the window holds one ledger
    # case only if the ledger query returns it: drive that separately.
    assert analyzer.learn_from_all(limit=5) == 0
    assert _sites() == []


def test_pin_no_record_learn_from_all_seen_reanalyzes():
    class _SeenDownDB(_CollectDB):
        def query(self, sql, *a, **k):
            if "fingerprint FROM lessons" in sql:
                raise _Boom("seen down")
            return super().query(sql, *a, **k)

    skills = SimpleNamespace(save=lambda *a, **k: SimpleNamespace(id="s1"))
    analyzer = _analyzer(db=_SeenDownDB("none"), skills=skills)
    assert analyzer.learn_from_all(limit=5) == 3
    assert _sites() == []


def test_pin_no_record_rank():
    analyzer = _analyzer(db=_DBRaise())
    assert analyzer.rank("q", limit=4) == []
    assert _sites() == []


def test_pin_no_record_evaluate_pending():
    class _PendingDownDB:
        def query(self, sql, *a, **k):
            if "lesson_surfacings" in sql:
                raise _Boom("pending down")
            return []

        def query_one(self, *a, **k):
            return {"n": 0}

        def execute(self, *a, **k):
            return None

    analyzer = _analyzer(db=_PendingDownDB())
    assert analyzer.evaluate_usefulness() == {
        "evaluated": 0, "demoted": 0, "archived": 0}
    assert _sites() == []


def test_pin_no_record_evaluate_per_item():
    class _ItemDownDB:
        def query(self, sql, *a, **k):
            if "lesson_surfacings" in sql:
                return [{"fingerprint": "fp1", "surfaced_at": 1.0,
                         "lesson_id": "L1"}]
            return []

        def query_one(self, *a, **k):
            raise _Boom("item down")

        def execute(self, *a, **k):
            return None

    analyzer = _analyzer(db=_ItemDownDB())
    assert analyzer.evaluate_usefulness() == {
        "evaluated": 0, "demoted": 0, "archived": 0}
    assert _sites() == []


def test_pin_no_record_evaluate_demote_query():
    class _DemoteDownDB:
        def query(self, sql, *a, **k):
            if "lesson_surfacings" in sql:
                return []
            raise _Boom("demote down")

        def query_one(self, *a, **k):
            return {"n": 0}

        def execute(self, *a, **k):
            return None

    analyzer = _analyzer(db=_DemoteDownDB())
    assert analyzer.evaluate_usefulness() == {
        "evaluated": 0, "demoted": 0, "archived": 0}
    assert _sites() == []


def test_pin_no_record_promotion_candidates():
    analyzer = _analyzer(db=_DBRaise())
    assert analyzer.promotion_candidates() == []
    assert _sites() == []


def test_pin_no_record_stats_usefulness():
    class _UsefulnessDownDB:
        def query(self, sql, *a, **k):
            return []

        def query_one(self, sql, *a, **k):
            if "SUM(times_surfaced)" in sql:
                raise _Boom("v2 down")
            return {"n": 0}

        def execute(self, *a, **k):
            return None

    analyzer = _analyzer(db=_UsefulnessDownDB())
    out = analyzer.stats()
    assert "usefulness" not in out
    assert out["failures"] == 0
    assert _sites() == []


def test_pin_no_record_stats_failures():
    class _FailuresDownDB:
        def query(self, sql, *a, **k):
            return []

        def query_one(self, sql, *a, **k):
            if "SUM(times_surfaced)" in sql:
                return {"n": 0, "s": 0, "p": 0, "d": 0}
            if "FROM failures" in sql:
                raise _Boom("failures down")
            return {"n": 0}

        def execute(self, *a, **k):
            return None

    analyzer = _analyzer(db=_FailuresDownDB())
    out = analyzer.stats()
    assert "usefulness" in out
    assert "failures" not in out
    assert _sites() == []
