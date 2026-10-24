"""Pass #156 pins: the planning failure ledger (NEW) + graph conversions.

No planning ledger existed before this pass: ``nomorals/planning/failures.py``
(namespace ``planning``) is created here, durability-wired through the shared
``nomorals/core/failure_store.py`` exactly like the #141–#155 ledgers. The
17 durability tests below mirror #154's shape exactly; the 25 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name). There is 1 exclusion
(the ``critical_path_pert.te_sigma`` inner handler — see SURVEY.md E1),
proven NOT recording. No pin hardcodes scan totals or graph counts.
Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree). Restart tests use real subprocesses.
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
from nomorals.planning import failures as pf
from nomorals.planning import graph as G


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


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


def _graph():
    return G.WorldGraph(":memory:")


class _BrokenDB:
    def execute(self, *args, **kwargs):
        raise RuntimeError("boom")

    def commit(self):
        raise RuntimeError("boom")


def _broken_graph():
    g = _graph()
    g._db = _BrokenDB()
    return g


# ---- Store-level (durability mirror of #154) -----------------------------

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
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("planning").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
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


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("planning").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("planning")
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
    entries, _ = fs.Store("planning").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        pf.record_failure("no-home", ValueError("boom"))
        assert pf.failure_count() == 1
        assert [f.site for f in pf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []
    assert pf.format_failures() == "no planning failures recorded."
    pf.record_failure("planning.site-a", ValueError("boom"))
    pf.record_failure("planning.site-b", RuntimeError("bang"))
    assert pf.failure_count() == 2
    assert [f.site for f in pf.recent_failures()] == ["planning.site-b", "planning.site-a"]
    assert [f.site for f in pf.recent_failures(1)] == ["planning.site-b"]
    text = pf.format_failures()
    assert "2 total" in text and "planning.site-a" in text and "planning.site-b" in text
    pf.clear_failures()
    assert pf.failure_count() == 0
    assert pf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    pf.record_failure("planning.disk", ValueError("boom"))
    entries, total = fs.Store("planning").load()
    assert total == 1 and entries[0]["site"] == "planning.disk"
    pf.clear_failures()
    assert fs.Store("planning").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    pf.record_failure("planning.mem", ValueError("boom"))
    assert pf.failure_count() == 1
    assert [f.site for f in pf.recent_failures()] == ["planning.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                pf.record_failure(f"planning.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
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


def test_restart_corrupt_store_recovers(tmp_path):
    home = tmp_path / "home"
    assert _run_subprocess(home, _record_code(2)).returncode == 0
    store_path = fs.Store("planning").path()
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def test_converted_is_termux_records_and_returns_false(monkeypatch):
    monkeypatch.setattr(sys, "prefix", 123)
    assert G._is_termux() is False
    assert _sites() == ["graph._is_termux"]


def test_converted_get_records_and_returns_none():
    assert _broken_graph().get("x") is None
    assert _sites() == ["graph.get"]


def test_converted_list_nodes_records_and_returns_empty():
    assert _broken_graph().list_nodes() == []
    assert _sites() == ["graph.list_nodes"]


def test_converted_remove_node_records_and_returns_false():
    assert _broken_graph().remove_node("x") is False
    assert _sites() == ["graph.remove_node"]


def test_converted_add_edge_records_and_returns_false():
    # add_edge resolves both endpoints via get() first, so a fully broken
    # DB trips the get site before reaching the insert. Break only the
    # INSERT to probe the add_edge site itself.
    g = _graph()
    na = g.add_node("person", "a")
    nb = g.add_node("person", "b")
    assert na is not None and nb is not None
    real = g._db

    class _InsertBrokenDB:
        def execute(self, sql, params=()):
            if sql.lstrip().upper().startswith("INSERT"):
                raise RuntimeError("boom")
            return real.execute(sql, params)

        def commit(self):
            return real.commit()

    g._db = _InsertBrokenDB()
    assert g.add_edge(na.node_id, nb.node_id, "depends_on") is False
    assert _sites() == ["graph.add_edge"]


def test_converted_edges_records_and_returns_empty():
    assert _broken_graph().edges() == []
    assert _sites() == ["graph.edges"]


def test_converted_dependents_records_and_returns_empty(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "_propagate", _boom)
    assert g.dependents("x") == []
    assert _sites() == ["graph.dependents"]


def test_converted_dependencies_records_and_returns_empty(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "edges", _boom)
    assert g.dependencies("x") == []
    assert _sites() == ["graph.dependencies"]


def test_converted_what_breaks_if_records_and_returns_query_failed(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "get", _boom)
    assert g.what_breaks_if("x") == {"ok": False, "reason": "query failed"}
    assert _sites() == ["graph.what_breaks_if"]


def test_converted_critical_path_records_and_returns_empty(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "list_nodes", _boom)
    assert g.critical_path() == []
    assert _sites() == ["graph.critical_path"]


def test_converted_propagate_records_and_returns_seeds(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "edges", _boom)
    assert g._propagate({"seed"}) == {"seed"}
    assert _sites() == ["graph._propagate"]


def test_converted_undirected_adj_records_and_returns_partial(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "edges", _boom)
    assert g._undirected_adj() == {}
    assert _sites() == ["graph._undirected_adj"]


def test_converted_directed_adj_records_and_returns_partial(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "edges", _boom)
    assert g._directed_adj() == {}
    assert _sites() == ["graph._directed_adj"]


def test_converted_path_between_records_and_returns_empty(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "_undirected_adj", _boom)
    assert g.path_between("a", "b") == []
    assert _sites() == ["graph.path_between"]


def test_converted_neighborhood_records_and_returns_empty(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "_undirected_adj", _boom)
    assert g.neighborhood("a") == []
    assert _sites() == ["graph.neighborhood"]


def test_converted_to_mermaid_records_and_returns_unavailable(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "list_nodes", _boom)
    assert g.to_mermaid() == "graph LR\n    %% unavailable"
    assert _sites() == ["graph.to_mermaid"]


def test_converted_clear_disruption_records_and_returns_false():
    assert _broken_graph().clear_disruption("x") is False
    assert _sites() == ["graph.clear_disruption"]


def test_converted_disrupted_records_and_returns_empty(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "list_nodes", _boom)
    assert g.disrupted() == []
    assert _sites() == ["graph.disrupted"]


def test_converted_sync_from_memory_import_records_and_returns_zero():
    g = _graph()
    with patch("nomorals.memory.manager.MemoryManager",
               side_effect=RuntimeError("boom")):
        assert g.sync_from_memory(None) == 0
    assert _sites() == ["graph.sync_from_memory.import"]


def test_converted_sync_from_memory_record_records_and_continues(monkeypatch):
    g = _graph()
    good = SimpleNamespace(tag="good")
    bad = SimpleNamespace(tag="bad")

    def _to_node(rec):
        if rec.tag == "bad":
            raise RuntimeError("boom")
        return object()

    monkeypatch.setattr(g, "_record_to_node", _to_node)
    mem = SimpleNamespace(
        recall=lambda q, limit: SimpleNamespace(records=[good, bad]))
    assert g.sync_from_memory(mem) == 1
    assert _sites() == ["graph.sync_from_memory.record"]


def test_converted_record_to_node_memory_created_records_and_keeps_node():
    g = _graph()
    rec = SimpleNamespace(text="pay rent deadline", kind="deadline", tags="",
                          record_id="r1", created_at="not-a-number")
    node = g._record_to_node(rec)
    assert node is not None
    assert "memory_created" not in node.attrs
    assert _sites() == ["graph._record_to_node.memory_created"]


def test_converted_record_to_node_outer_records_and_returns_none(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "add_node", _boom)
    rec = SimpleNamespace(text="pay rent deadline", kind="deadline", tags="",
                          record_id="r1", created_at=None)
    assert g._record_to_node(rec) is None
    assert _sites() == ["graph._record_to_node.outer"]


def test_converted_sync_from_tasks_records_and_returns_zero(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "add_node", _boom)
    assert g.sync_from_tasks([{"label": "x"}]) == 0
    assert _sites() == ["graph.sync_from_tasks"]


def test_converted_summary_records_and_returns_unavailable(monkeypatch):
    g = _graph()
    monkeypatch.setattr(g, "list_nodes", _boom)
    assert g.summary() == "🌐 world graph: unavailable"
    assert _sites() == ["graph.summary"]


def test_converted_disruption_alerts_records_and_returns_empty():
    assert G.disruption_alerts(SimpleNamespace(disrupted=_boom)) == []
    assert _sites() == ["graph.disruption_alerts"]


# ---- Excluded-handler behavior: proven NOT recording ---------------------

def test_excluded_pert_inner_does_not_record(monkeypatch):
    # E1 (SURVEY.md): pert() is total — it returns (0.0, 0.0) on bad input
    # itself — so the te_sigma inner handler is defensive dead code. Patch
    # pert to raise to fire the handler anyway. A zero-te node only lands
    # on the best path as a middle link (a lone zero-te path ties and
    # loses), so chain A depends_on M depends_on B: the query must still
    # return ok: True, M must count as unestimated, and nothing records.
    g = _graph()
    a = g.add_node("deadline", "alpha",
                   {"optimistic": 10, "most_likely": 20, "pessimistic": 30})
    m = g.add_node("deadline", "middle",
                   {"optimistic": "bad", "most_likely": "bad",
                    "pessimistic": "bad"})
    b = g.add_node("deadline", "beta",
                   {"optimistic": 10, "most_likely": 20, "pessimistic": 30})
    assert a is not None and m is not None and b is not None
    assert g.add_edge(a.node_id, m.node_id, "depends_on") is True
    assert g.add_edge(m.node_id, b.node_id, "depends_on") is True

    def _fake_pert(o, m_, p):
        if isinstance(o, str):
            raise RuntimeError("boom")
        return (5.0, 1.0)

    with patch("nomorals.planning.estimates.pert", side_effect=_fake_pert):
        res = g.critical_path_pert()
    assert res["ok"] is True
    assert res["node_ids"] == [a.node_id, m.node_id, b.node_id]
    assert res["unestimated"] == 1
    assert pf.recent_failures() == []
