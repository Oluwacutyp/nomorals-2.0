"""Pass #194 pins: the planning failure ledger (reused from #156) + the
commitment_sync conversion.

``nomorals/planning/failures.py`` (namespace ``planning``) already
exists from #156 — this pass REUSES it, it does not create a second
planning ledger. The 17 durability tests below mirror the #141+ shape
exactly (same as #193's, against the same Store backend); the 1 site
pin proves the conversion records without changing flow (fallback
value unchanged AND a ledger entry under its site name); the 14
NOT-recording pins prove each exclusion stays silent with the
designed output. No pin hardcodes scan totals or commitment_sync
counts. Every test redirects stores with the app's own ``NM_HOME``
override (never the real home, never the repo tree). Restart tests
use real subprocesses. Every trigger was probed live before its pin
was written (probe.py).

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

from nomorals.core import failure_store as fs
from nomorals.planning import commitment_sync as cs
from nomorals.planning import failures as pf


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    pf.clear_failures()
    yield
    pf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


def _sites():
    return [f.site for f in pf.recent_failures()]


def _commitment(cid="c1", status="open", deadline=None, promisee=""):
    return SimpleNamespace(id=cid, what="pay rent", status=status,
                          deadline=deadline, promisee=promisee,
                          promisor="", language="", source_text="",
                          source_message_id=None, evidence="",
                          created_at=None, closed_at=None)


class _Node:
    def __init__(self, node_id, type_, label, attrs, external_ref=None):
        self.node_id = node_id
        self.type = type_
        self.label = label
        self.attrs = dict(attrs)
        self.external_ref = external_ref


class _StubGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self._db = None
        self._n = 0

    def add_node(self, type_, label, attrs, external_ref=None):
        self._n += 1
        node = _Node(f"n{self._n}", type_, label, attrs, external_ref)
        self.nodes.append(node)
        return node

    def list_nodes(self, type_=None):
        if type_ is None:
            return list(self.nodes)
        return [n for n in self.nodes if n.type == type_]

    def get(self, node_id):
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def add_edge(self, from_id, to_id, type_):
        self.edges.append((from_id, to_id, type_))
        return True


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("planning")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("planning").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("planning")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("planning").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("planning")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("planning").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("planning").save(_entries(2, "c"), 2) is True
    assert fs.Store("planning.other").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("planning").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("planning.other").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("planning")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("planning").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("planning").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("planning").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def _boom_expanduser(path):
    raise OSError("no home")


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", _boom_expanduser)
    before = fs.write_failures()
    assert fs.Store("planning").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    errors = []

    def worker(k):
        try:
            for i in range(30):
                fs.Store("planning").save(_entries(3, f"w{k}-{i}-"), 3)
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    entries, _ = fs.Store("planning").load()
    assert len(entries) == 3
    assert all(e["site"].startswith("w") for e in entries)


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    monkeypatch.setattr("os.path.expanduser", _boom_expanduser)
    pf.record_failure("commitment_sync.probe", ValueError("boom"))
    assert _sites() == ["commitment_sync.probe"]


def test_ledger_api_record_recent_count_clear_format():
    pf.record_failure("commitment_sync.a", ValueError("boom0"))
    pf.record_failure("commitment_sync.b", RuntimeError("boom1"))
    assert pf.failure_count() == 2
    assert _sites() == ["commitment_sync.b", "commitment_sync.a"]
    rendered = pf.format_failures()
    assert "commitment_sync.b: RuntimeError: boom1" in rendered
    assert "commitment_sync.a: ValueError: boom0" in rendered
    pf.clear_failures()
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []
    assert pf.format_failures() == "no planning failures recorded."


def test_ledger_record_and_clear_hit_disk():
    pf.record_failure("commitment_sync.disk", ValueError("boom"))
    entries, total = fs.Store("planning").load()
    assert total == 1
    assert [e["site"] for e in entries] == ["commitment_sync.disk"]
    pf.clear_failures()
    assert fs.Store("planning").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    pf.record_failure("commitment_sync.mem", ValueError("boom"))
    assert _sites() == ["commitment_sync.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                pf.record_failure(f"planning.w{k}-{i}", ValueError("boom"))
        except Exception as exc:
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert pf.failure_count() == 400
    assert len(pf.recent_failures(1000)) == 200
    entries, total = pf._store.load()
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
        "from nomorals.planning import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.planning import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.planning import failures as f\n"
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
    fs.Store("planning").path().write_text("{not json", encoding="utf-8")
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


# ---- Site pin (the conversion) -------------------------------------------

def test_pin_persist_attrs_primary():
    class FailingDB:
        def execute(self, *a, **k):
            raise RuntimeError("primary down")

        def commit(self):
            raise RuntimeError("primary down")

    graph = _StubGraph()
    graph._db = FailingDB()
    node = graph.add_node("commitment", "pay rent", {"status": "open"},
                          "commitment:c1")
    out = cs._persist_attrs(graph, node, {"status": "kept"})
    assert out is node
    assert node.attrs.get("status") == "kept"
    assert _sites() == ["commitment_sync.persist_attrs_primary"]


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_pin_find_by_external_ref_not_recorded():
    class BrokenList:
        def list_nodes(self, *a, **k):
            raise RuntimeError("list down")

    assert cs._find_by_external_ref(BrokenList(), "commitment:c1") is None
    assert _sites() == []


def test_pin_persist_attrs_fallback_not_recorded():
    class RaisingDict(dict):
        def update(self, *a, **k):
            raise RuntimeError("attrs frozen")

    graph = _StubGraph()
    node = _Node("n1", "commitment", "x", {})
    node.attrs = RaisingDict({"status": "open"})
    assert cs._persist_attrs(graph, node, {"status": "kept"}) is node
    assert _sites() == []


def test_pin_upsert_no_identity_not_recorded():
    class RaisingAdd:
        def add_node(self, *a, **k):
            raise RuntimeError("add down")

    assert cs._upsert_node(RaisingAdd(), "commitment", "x", {}, "") is None
    assert _sites() == []


def test_pin_upsert_typeerror_fallback_not_recorded():
    class OldThenBroken:
        def add_node(self, *a, **k):
            if "external_ref" in k:
                raise TypeError("no external_ref kwarg")
            raise RuntimeError("plain add down")

    assert cs._upsert_node(OldThenBroken(), "commitment", "x", {},
                           "commitment:c1") is None
    assert _sites() == []


def test_pin_upsert_primary_insert_not_recorded():
    class RaisingAdd:
        def add_node(self, *a, **k):
            raise RuntimeError("add down")

    assert cs._upsert_node(RaisingAdd(), "commitment", "x", {},
                           "commitment:c1") is None
    assert _sites() == []


def test_pin_add_edge_not_recorded():
    class BrokenEdge:
        def add_edge(self, *a, **k):
            raise RuntimeError("edge down")

    assert cs._add_edge(BrokenEdge(), "a", "b", "owned_by") is False
    assert _sites() == []


def test_pin_iter_snapshot_not_recorded():
    class NoSnapshot:
        @property
        def commitments(self):
            raise RuntimeError("snapshot down")

    assert cs._iter_ledger_commitments(NoSnapshot()) == []
    assert _sites() == []


def test_pin_iter_method_loop_not_recorded():
    class OneMethodDown:
        def open_commitments(self):
            raise RuntimeError("open down")

        def all_commitments(self):
            return [_commitment("c9")]

    assert [c.id for c in cs._iter_ledger_commitments(OneMethodDown())] == ["c9"]
    assert _sites() == []


def test_pin_iter_last_resort_not_recorded():
    class LastResortDown:
        @property
        def all(self):
            raise RuntimeError("all down")

        @property
        def rows(self):
            raise RuntimeError("rows down")

    assert cs._iter_ledger_commitments(LastResortDown()) == []
    assert _sites() == []


def test_pin_sync_per_row_not_recorded(monkeypatch):
    class OneCommitLedger:
        def open_commitments(self):
            return [_commitment("c1")]

        def all_commitments(self):
            return [_commitment("c1")]

    def _boom(*a, **k):
        raise RuntimeError("contract break")

    monkeypatch.setattr(cs, "sync_commitment_to_graph", _boom)
    assert cs.sync_ledger_to_graph(OneCommitLedger(), _StubGraph()) == 0
    assert _sites() == []


def test_pin_sync_outer_not_recorded(monkeypatch):
    class OneCommitLedger:
        def open_commitments(self):
            return [_commitment("c1")]

    def _boom(*a, **k):
        raise RuntimeError("contract break")

    monkeypatch.setattr(cs, "_iter_ledger_commitments", _boom)
    assert cs.sync_ledger_to_graph(OneCommitLedger(), _StubGraph()) == 0
    assert _sites() == []


def test_pin_sync_extracted_per_commitment_not_recorded(monkeypatch):
    class FakeLedger:
        def extract_and_record(self, *a, **k):
            return [_commitment("c1")]

    def _boom(*a, **k):
        raise RuntimeError("contract break")

    monkeypatch.setattr(cs, "sync_commitment_to_graph", _boom)
    res = cs.sync_extracted_to_graph({"text": "x"}, FakeLedger(), _StubGraph())
    assert res.get("added") == 0
    assert _sites() == []


def test_pin_overdue_per_node_not_recorded():
    class BadAttrsNode:
        node_id = "bad1"
        type = "commitment"

        @property
        def attrs(self):
            raise RuntimeError("attrs corrupt")

    class OneBadNode:
        def list_nodes(self, *a, **k):
            return [BadAttrsNode()]

    assert cs.overdue_commitments_in_graph(OneBadNode()) == []
    assert _sites() == []


def test_pin_overdue_outer_not_recorded():
    class ListDown:
        def list_nodes(self, *a, **k):
            raise RuntimeError("store down")

    assert cs.overdue_commitments_in_graph(ListDown()) == []
    assert _sites() == []
