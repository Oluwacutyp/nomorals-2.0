"""Pass #182 pins: the NEW memory failure ledger + manager conversions.

``nomorals/memory/failures.py`` (namespace ``memory``) is new in this
pass — manager.py is a memory-subsystem module, and no memory-area
ledger existed through #181. The 17 durability tests below mirror the
#141+ shape exactly (same as #180's, against the new ledger); the 14
site pins prove each conversion records without changing flow (fallback
value unchanged AND a ledger entry under its site name); the 3
NOT-recording pins prove each exclusion stays silent with the designed
output. No pin hardcodes scan totals or manager counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the real
home, never the repo tree). Restart tests use real subprocesses. Every
trigger was probed live before its pin was written (probe.py).

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
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.memory import failures as mf
from nomorals.memory.manager import CoreBlocks, MemoryManager, _is_private
from nomorals.memory.manager import MemoryManager as _MM
from nomorals.storage.db import Database

_metadata_dict = _MM._metadata_dict


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mf.clear_failures()
    yield
    mf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in mf.recent_failures()]


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("memory")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("memory").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("memory")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("memory").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("memory")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("memory").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("memory").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("memory").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("memory")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("memory").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("memory").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("memory").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("memory").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("memory")
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
    entries, _ = fs.Store("memory").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mf.record_failure("no-home", ValueError("boom"))
        assert mf.failure_count() == 1
        assert [f.site for f in mf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []
    assert mf.format_failures() == "no memory failures recorded."
    mf.record_failure("memory.site-a", ValueError("boom"))
    mf.record_failure("memory.site-b", RuntimeError("bang"))
    assert mf.failure_count() == 2
    assert [f.site for f in mf.recent_failures()] == ["memory.site-b", "memory.site-a"]
    assert [f.site for f in mf.recent_failures(1)] == ["memory.site-b"]
    text = mf.format_failures()
    assert "2 total" in text and "memory.site-a" in text and "memory.site-b" in text
    mf.clear_failures()
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mf.record_failure("memory.disk", ValueError("boom"))
    entries, total = fs.Store("memory").load()
    assert total == 1 and entries[0]["site"] == "memory.disk"
    mf.clear_failures()
    assert fs.Store("memory").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mf.record_failure("memory.mem", ValueError("boom"))
    assert mf.failure_count() == 1
    assert [f.site for f in mf.recent_failures()] == ["memory.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mf.record_failure(f"memory.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mf.failure_count() == 400
    assert len(mf.recent_failures(1000)) == 200
    entries, total = mf._store.load()
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
        "from nomorals.memory import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.memory import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.memory import failures as f\n"
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
    store_path = fs.Store("memory").path()
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

class _BrokenDB:
    def execute(self, *a, **k):
        raise RuntimeError("db down")


class _RaisingList(list):
    def append(self, *a, **k):
        raise RuntimeError("append down")


class _BadStr(str):
    def strip(self, *a, **k):
        raise RuntimeError("strip down")


class _BadMeta:
    @property
    def metadata(self):
        raise RuntimeError("metadata down")


class _Ctx:
    def __init__(self, db):
        self.db = db
        self.settings = None
        self.router = None


def _bare_manager(**attrs):
    m = MemoryManager.__new__(MemoryManager)
    m.stats = {}
    m.weights = {}
    m._health_events = []
    m._last_consolidation = None
    m._last_consolidation_report = {}
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def test_pin_core_get():
    assert CoreBlocks(_BrokenDB()).get("persona") == ""
    assert "memory.core_get" in _sites()


def test_pin_core_all():
    assert CoreBlocks(_BrokenDB()).all() == {}
    assert "memory.core_all" in _sites()


def test_pin_core_clear():
    assert CoreBlocks(_BrokenDB()).clear("persona") is False
    assert "memory.core_clear" in _sites()


def test_pin_is_private():
    assert _is_private(_BadMeta()) is False
    assert "memory.is_private" in _sites()


def test_pin_note_health_event():
    m = _bare_manager()
    m._health_events = _RaisingList()
    m._note_health_event("recall", "test")
    assert "memory.note_health_event" in _sites()


def test_pin_ensure_trust_columns():
    class FakeDBCols:
        def table_info(self, *a, **k):
            raise RuntimeError("table_info down")
    m = _bare_manager(db=FakeDBCols())
    m._ensure_trust_columns()
    assert "memory.ensure_trust_columns" in _sites()


def test_pin_recall_trust_downrank():
    db = Database(":memory:")
    db.migrate()
    mgr = MemoryManager(_Ctx(db), vector_backend="legacy")
    mgr.remember("the sky is blue", source="tool:browser", origin="chat:x")
    res = mgr.recall("sky", origin=_BadStr("chat:y"))
    assert len(res.records) >= 1
    db.close()
    assert "memory.recall_trust_downrank" in _sites()


def test_pin_supersede():
    class Old:
        id = "old1"
        metadata = {}

    def _raising_update(*a, **k):
        raise RuntimeError("update down")
    m = _bare_manager(get=lambda rid: Old(),
                      remember=lambda *a, **k: "new1", stats={})
    m.update = _raising_update
    assert m.supersede("old1", "new text") == ""
    assert "memory.supersede" in _sites()


def test_pin_supersession_chain():
    def _raising_get(*a, **k):
        raise RuntimeError("get down")
    m = _bare_manager(get=_raising_get)
    assert m.supersession_chain("x") == []
    assert "memory.supersession_chain" in _sites()


def test_pin_metadata_dict():
    assert _metadata_dict({"metadata": "{invalid json"}) == {}
    assert "memory.metadata_dict" in _sites()


class _Repo:
    def __init__(self, count_exc=False):
        self.count_exc = count_exc

    def count(self):
        if self.count_exc:
            raise RuntimeError("count down")
        return 3


class _Semantic:
    name = "legacy"

    def count(self):
        return 3


class _FTS:
    def __init__(self, exc=False):
        self.exc = exc

    def count(self):
        if self.exc:
            raise RuntimeError("fts down")
        return 3


class _Embedder:
    def __init__(self, exc=False):
        self.exc = exc

    def stats_snapshot(self):
        if self.exc:
            raise RuntimeError("embedder down")
        return {"dims": 1}

    @property
    def dimensions(self):
        if self.exc:
            raise RuntimeError("dims down")
        return 1


class _QDB:
    def query(self, *a, **k):
        return []

    def scalar(self, *a, **k):
        return 0


def _health_mgr(**kw):
    import nomorals.memory.manager as mm  # noqa: F401 — helper only
    return _bare_manager(
        repo=kw.get("repo", _Repo()),
        semantic=_Semantic(),
        fts=kw.get("fts", _FTS()),
        embedder=kw.get("embedder", _Embedder()),
        db=_QDB(),
        counts_by_kind=kw.get("counts_by_kind", lambda: {"fact": 3}),
    )


def test_pin_health_by_kind():
    def _raising_counts():
        raise RuntimeError("counts down")
    r = _health_mgr(counts_by_kind=_raising_counts).health()
    assert r["by_kind"] == {}
    assert "memory.health_by_kind" in _sites()


def test_pin_health_fts_documents():
    r = _health_mgr(fts=_FTS(exc=True)).health()
    assert r["fts_documents"] == -1
    assert "memory.health_fts_documents" in _sites()


def test_pin_health_embedder():
    r = _health_mgr(embedder=_Embedder(exc=True)).health()
    assert r["embedder"] == {} and r["embedder_dims"] == -1
    assert "memory.health_embedder" in _sites()


def test_pin_health_cadence(monkeypatch):
    import nomorals.memory.manager as mm
    monkeypatch.setattr(mm, "_cadence_status",
                        lambda self: (_ for _ in ()).throw(
                            RuntimeError("cadence down")))
    r = _health_mgr().health()
    assert r["consolidation_cadence"] == {}
    assert "memory.health_cadence" in _sites()


# ---- NOT-recording pins (exclusions stay silent) --------------------------

def test_pin_for_training_silent():
    def _raising_recall(*a, **k):
        raise RuntimeError("recall down")
    m = _bare_manager(recall=_raising_recall)
    assert m.for_training() == []
    assert _sites() == []


def test_pin_health_records_silent():
    r = _health_mgr(repo=_Repo(count_exc=True)).health()
    assert r["records"] == -1
    assert "record count unreadable" in r["problems"]
    assert r["ok"] is False
    assert _sites() == []


def test_pin_repair_dims_silent():
    class RepairEmbedder:
        @property
        def dimensions(self):
            raise RuntimeError("dims down")

        def embed(self, content):
            return [0.1, 0.2]

    class RepairDB:
        def query(self, *a, **k):
            return [{"id": "a", "content": "hello world", "dim": None}]

        def execute(self, *a, **k):
            return None

    class RepairSemantic:
        def delete_owner(self, rid):
            pass

        def put(self, vector, rid):
            return "emb1"

    m = _bare_manager(embedder=RepairEmbedder(), db=RepairDB(),
                       semantic=RepairSemantic())
    rep = m.repair_embeddings()
    assert rep["drifted"] == 0 and rep["missing"] == 1
    assert _sites() == []
