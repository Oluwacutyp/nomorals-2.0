"""Pass #159 pins: route conversions on the REUSED planning ledger.

``nomorals/planning/failures.py`` (namespace ``planning``) already
existed from #156 — this pass reuses it, no second planning ledger.
The 17 durability tests below mirror #156's shape exactly (they pin
the shared store contract, not a new file); the 22 site pins prove
each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name). There are 3
exclusions (E1 h3 import probe, E2 ortools-availability probe, E3
control add-parse user-input probe — see SURVEY.md), each proven NOT
recording. No pin hardcodes scan totals or route counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written.
"""
import importlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.planning import failures as pf
from nomorals.planning import route


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


def _cost():
    return route.CostModel(":memory:")


def _planner():
    return route.RoutePlanner(db_path=":memory:")


def _stop(sid, label, **kw):
    return route.Stop(sid, label, **kw)


# ---- Store-level (durability mirror of #156) -----------------------------

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

def test_converted_geo_add_records_and_returns_empty(monkeypatch):
    monkeypatch.setattr(route, "cell_for", _boom)
    assert route.GeoIndex().add("s", 1.0, 2.0) == ""
    assert _sites() == ["route.geo_add"]


def test_converted_nearby_kring_records_and_falls_back_to_full_scan(monkeypatch):
    class _H3Stub:
        def geo_to_h3(self, lat, lng, res):
            return "8a123456789abcdef"

        def k_ring(self, cell, rings):
            raise RuntimeError("boom")

    monkeypatch.setattr(route, "_h3", _H3Stub())
    geo = route.GeoIndex()
    assert geo.add("s1", 6.5, 3.3) == "8a123456789abcdef"
    assert geo.nearby(6.5, 3.3) == ["s1"]
    assert _sites() == ["route.nearby_kring"]


def test_converted_nearby_records_and_returns_empty(monkeypatch):
    monkeypatch.setattr(route, "cell_for", _boom)
    assert route.GeoIndex().nearby(1.0, 2.0) == []
    assert _sites() == ["route.nearby"]


def test_converted_same_area_records_and_returns_false():
    assert route.GeoIndex._same_area("g:x:1", "g:1:1") is False
    assert _sites() == ["route.same_area"]


class _AlterFailConn:
    """Delegating connection that fails ALTERs — simulates a genuine
    migration failure while letting everything else through."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def execute(self, sql, *a, **k):
        if isinstance(sql, str) and sql.strip().upper().startswith("ALTER"):
            raise RuntimeError("boom")
        return self._real.execute(sql, *a, **k)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        if name == "_real":
            object.__setattr__(self, name, value)
        else:
            setattr(self._real, name, value)


def test_converted_cost_init_migration_records(tmp_path, monkeypatch):
    db_file = tmp_path / "cost.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE route_actuals (cell_a TEXT, cell_b TEXT, minutes REAL,"
        " cost_kobo REAL, n INTEGER, PRIMARY KEY (cell_a, cell_b))")
    conn.commit()
    conn.close()
    real_connect = sqlite3.connect
    monkeypatch.setattr(route.sqlite3, "connect",
                        lambda path: _AlterFailConn(real_connect(path)))
    route.CostModel(str(db_file))
    assert _sites() == ["route.cost_init_migration"]


def test_converted_estimate_records_and_returns_unknown(monkeypatch):
    model = _cost()
    monkeypatch.setattr(model, "estimate_detail", _boom)
    assert model.estimate(_stop("a", "A"), _stop("b", "B")) == (30.0, 0)
    assert _sites() == ["route.estimate"]


def test_converted_estimate_detail_records_and_returns_heuristic(monkeypatch):
    model = _cost()
    monkeypatch.setattr(model, "_learned", _boom)
    assert model.estimate_detail(_stop("a", "A"), _stop("b", "B")) == (30.0, 0, "heuristic")
    assert _sites() == ["route.estimate_detail"]


def test_converted_get_global_records_and_returns_default(tmp_path):
    model = route.CostModel(str(tmp_path / "g.db"))
    model._db.close()
    assert model._get_global("speed_kmh", 32.0) == 32.0
    assert _sites() == ["route.get_global"]


def test_converted_set_global_records_and_drops_write(tmp_path):
    model = route.CostModel(str(tmp_path / "g.db"))
    model._db.close()
    model._set_global("speed_kmh", 99.0)
    assert _sites() == ["route.set_global"]


def test_converted_stats_recent_records_and_counts_all_recent():
    class _RowShim:
        def __init__(self, row):
            self._row = row

        def fetchone(self):
            return self._row

    class _StatsDB:
        def __init__(self):
            self.calls = 0

        def execute(self, sql, params=()):
            self.calls += 1
            if self.calls == 1:
                return _RowShim({"s": 5})
            raise RuntimeError("boom")

    model = _cost()
    model._db = _StatsDB()
    s = model.stats()
    assert s["recorded_trips"] == 5 and s["recent_trips_30d"] == 5
    assert _sites() == ["route.stats_recent"]


def test_converted_stats_records_and_returns_empty(tmp_path):
    model = route.CostModel(str(tmp_path / "s.db"))
    model._db.close()
    assert model.stats() == {"recorded_trips": 0, "source": "heuristic-defaults"}
    assert _sites() == ["route.stats"]


def test_converted_greedy_score_window_records_and_scores_ontime():
    cost = _cost()
    solver = route.RouteSolver("greedy")
    origin = _stop("o", "origin")
    bad = _stop("b", "badwin", window=("x", "y"))
    res = solver.solve([origin, bad], cost)
    assert [s.stop_id for s in res.order] == ["o", "b"]
    assert "route.greedy_score_window" in _sites()
    # n=2 < 4: _two_opt returns early, so the 2-opt window site must NOT fire.
    assert "route.two_opt_window" not in _sites()


def test_converted_two_opt_window_records():
    cost = _cost()
    solver = route.RouteSolver("greedy")
    stops = [_stop("o", "origin"),
             _stop("s1", "one"),
             _stop("s2", "two"),
             _stop("b", "badwin", window=("x", "y"))]
    res = solver.solve(stops, cost)
    assert len(res.order) == 4
    assert "route.two_opt_window" in _sites()


def test_converted_window_violations_records_and_returns_partial():
    class _BadWindow:
        label = "x"

        @property
        def window(self):
            raise RuntimeError("boom")

    assert route.RouteSolver._window_violations([_BadWindow()], [1000.0]) == []
    assert _sites() == ["route.window_violations"]


def test_converted_planner_init_migration_records(tmp_path, monkeypatch):
    # NOTE: planner __init__ row loading is itself broken upstream (its
    # ad-hoc connection sets no row_factory, so row["stop_id"] raises
    # TypeError and the outer debug-logged handler swallows it — stops
    # never restore from disk). That pre-existing bug is OUT of scope
    # here; the pin supplies the row_factory the code assumes so the
    # migration handler itself can be probed. Reported to parent.
    db_file = tmp_path / "stops.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE route_stops (stop_id TEXT PRIMARY KEY, label TEXT,"
        " lat REAL, lng REAL, dwell_minutes REAL)")
    conn.execute("INSERT INTO route_stops VALUES ('s1', 'home', 6.5, 3.3, 15.0)")
    conn.commit()
    conn.close()
    real_connect = sqlite3.connect

    def _row_alter_fail_connect(path):
        c = real_connect(path)
        c.row_factory = sqlite3.Row
        return _AlterFailConn(c)

    monkeypatch.setattr(route.sqlite3, "connect", _row_alter_fail_connect)
    planner = route.RoutePlanner(db_path=str(db_file))
    assert planner._stops["s1"].label == "home"
    assert "route.planner_init_migration" in _sites()


def test_converted_planner_row_window_records_and_drops_window(tmp_path, monkeypatch):
    # Same row_factory caveat as the migration pin above: the pin
    # supplies the row_factory the loader assumes, so the window-parse
    # handler itself can be probed.
    db_file = tmp_path / "stops2.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        "CREATE TABLE route_stops (stop_id TEXT PRIMARY KEY, label TEXT,"
        " lat REAL, lng REAL, dwell_minutes REAL,"
        " window_start REAL DEFAULT 0, window_end REAL DEFAULT 0)")
    conn.execute(
        "INSERT INTO route_stops VALUES ('s1', 'badwin', 6.5, 3.3, 15.0,"
        " 'bad', 'bad')")
    conn.commit()
    conn.close()
    real_connect = sqlite3.connect

    def _row_connect(path):
        c = real_connect(path)
        c.row_factory = sqlite3.Row
        return c

    monkeypatch.setattr(route.sqlite3, "connect", _row_connect)
    planner = route.RoutePlanner(db_path=str(db_file))
    assert planner._stops["s1"].window is None
    assert "route.planner_row_window" in _sites()


def test_converted_add_stop_persist_records_and_still_returns_stop(tmp_path):
    # The DROP TABLE trigger below is the pin's probe — but note the
    # sharper live finding: add_stop's INSERT supplies 5 values for the
    # 7-column table (window_start/window_end added by migration), so
    # persist ALSO fails on every live table (verified: OperationalError
    # "7 columns but 5 values"). Pre-existing, out of scope; the ledger
    # now surfaces it instead of the silent pass.
    db_file = tmp_path / "p.db"
    planner = route.RoutePlanner(db_path=str(db_file))
    conn = sqlite3.connect(str(db_file))
    conn.execute("DROP TABLE route_stops")
    conn.commit()
    conn.close()
    stop = planner.add_stop("corner shop", lat=6.5, lng=3.3)
    assert stop is not None
    assert _sites() == ["route.add_stop_persist"]


def test_converted_add_stop_records_and_returns_none():
    assert _planner().add_stop("x", lat="bad") is None
    assert _sites() == ["route.add_stop"]


def test_converted_set_window_records_and_returns_false(tmp_path):
    # NOTE: the stop is placed directly in _stops, not via add_stop —
    # add_stop's own persist is arity-broken upstream (5-value INSERT
    # into the 7-column table, see the add_stop_persist pin), so routing
    # through it would co-fire that site. The UPDATE failure below is
    # the set_window site's own trigger.
    db_file = tmp_path / "w.db"
    planner = route.RoutePlanner(db_path=str(db_file))
    planner._stops["bank"] = route.Stop("bank", "bank", 6.5, 3.3)
    conn = sqlite3.connect(str(db_file))
    conn.execute("DROP TABLE route_stops")
    conn.commit()
    conn.close()
    assert planner.set_window("bank", "16:00", "17:30") is False
    assert _sites() == ["route.set_window"]


def test_converted_find_stop_records_and_returns_none():
    class _BadLabel:
        @property
        def label(self):
            raise RuntimeError("boom")

    planner = _planner()
    planner._stops["bad"] = _BadLabel()
    assert planner.find_stop("anything") is None
    assert _sites() == ["route.find_stop"]


def test_converted_plan_records_and_returns_unrouted_stops(monkeypatch):
    planner = _planner()
    monkeypatch.setattr(planner.solver, "solve", _boom)
    stop = _stop("a", "A")
    res = planner.plan([stop])
    assert [s.stop_id for s in res.order] == ["a"]
    assert res.legs == [] and res.total_minutes == 0.0
    assert _sites() == ["route.plan"]


def test_converted_format_window_records_and_hides_window():
    stop = _stop("s1", "win", window=("bad", "bad"))
    res = route.RouteResult([stop], [], 0.0, 0, "greedy",
                            arrivals=[1700000000.0])
    text = route.format_route(res)
    assert "window" not in text
    assert _sites() == ["route.format_window"]


# ---- Excluded-handler behavior: proven NOT recording ---------------------

def test_excluded_h3_import_does_not_record():
    # E1 (SURVEY.md): h3 is an optional dependency — absence is the
    # designed grid-fallback configuration, queryable via h3_available().
    # Reloading re-runs the import handler; nothing may record.
    importlib.reload(route)
    assert route.h3_available() is False
    assert "route.h3_import" not in _sites()


def test_excluded_ortools_available_does_not_record():
    # E2 (SURVEY.md): ortools is a profile-gated optional backend —
    # absence is the designed greedy configuration, visible via backend.
    assert route._ortools_available() is False
    assert "route.ortools_available" not in _sites()


def test_excluded_control_add_parse_does_not_record():
    # E3 (SURVEY.md): the trailing token has a comma but is not
    # lat,lng — the whole text stays the label, and the reply shows
    # exactly that (no coords suffix). User-input probe, visible default.
    planner = _planner()
    reply = route.control_route("add cafe central west,", planner=planner)
    assert reply == "✅ saved stop: cafe central west,"
    assert "route.control_add_coords" not in _sites()
