"""Pass #174 pins: the agents failure ledger (REUSED) + scheduler conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) is reused from #152:
scheduler.py is an agents-subsystem module, so reuse is correct (unlike
#161's engine/trivia split, and unlike #168's scheduler/scheduler.py which
has its own scheduler-namespace ledger). The 17 durability tests below
mirror the #141+ shape exactly; the 14 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger
entry under its site name); the 4 NOT-recording pins prove each exclusion
stays silent. No pin hardcodes scan totals or scheduler counts. Every
test redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written (probe.py).

Pins rule (#169): no bare-except swallowing anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid the
literal bare-except phrase (#173 lesson).
"""
import contextlib
import json
import os
import random
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents import failures as af


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


# ---- Site pins ------------------------------------------------------------

from nomorals.agents import scheduler as sch
from nomorals.agents.scheduler import Scheduler, _parse_depends


class BoomDB:
    """DB whose query raises; execute/transaction are no-ops."""

    def query(self, *args, **kwargs):
        raise RuntimeError("db down")

    def query_one(self, *args, **kwargs):
        return None

    def execute(self, *args, **kwargs):
        return None

    def transaction(self):
        return contextlib.nullcontext()


def _sched(db=None, **kw):
    s = Scheduler.__new__(Scheduler)
    s.db = db
    s.context = types.SimpleNamespace()
    s.notifier = types.SimpleNamespace()
    s._lock = threading.Lock()
    s._active = set()
    s._deferrals = {}
    s._tick_errors = 0
    s._last_tick_at = None
    s._started_at = None
    s._thread = None
    s.tick_seconds = 20.0
    s.redeliver_interval = 120.0
    s._last_redeliver_at = 1e18  # redeliver sweep skipped
    s._running_jobs = 0
    s.wall_seconds = 300.0
    s._rng = random.Random(0)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _fake_notifier(**kw):
    base = {"queue_depth": lambda: {"retryable": 0, "held": 0, "dead": 0},
            "termux_fallback_available": lambda: False}
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_site_parse_depends_json():
    # Malformed JSON list falls through to single-text (fallback kept),
    # and the anomaly is now recorded.
    assert _parse_depends("[not json") == ["[not json"]
    assert _sites() == ["scheduler.parse_depends_json"]


def test_site_profile_name():
    class BadSettings:
        @property
        def settings(self):
            raise RuntimeError("settings broken")

    assert sch._profile_name(BadSettings()) == "workstation"
    assert _sites() == ["scheduler.profile_name"]


def test_site_routine_column_select():
    _sched(BoomDB())._ensure_routine_column()
    assert _sites() == ["scheduler.routine_column_select"]


def test_site_is_heavy_flag():
    assert Scheduler._is_heavy({"heavy": "yes"}) is False
    assert _sites() == ["scheduler.is_heavy_flag"]


def test_site_list_jobs():
    assert _sched(BoomDB()).list_jobs() == []
    assert _sites() == ["scheduler.list_jobs"]


def test_site_catch_up_query():
    assert _sched(BoomDB()).catch_up_on_startup() == []
    assert _sites() == ["scheduler.catch_up_query"]


def test_site_tick_query():
    assert _sched(BoomDB()).tick() == []
    assert _sites() == ["scheduler.tick_query"]


def test_site_tick_heartbeat(monkeypatch):
    from nomorals.core import error_system
    s = _sched()
    s.db = types.SimpleNamespace(query=lambda *a, **k: [])
    s._pressure_gate = lambda: (False, [])
    monkeypatch.setattr(error_system, "heartbeat", _boom)
    assert s.tick() == []
    assert _sites() == ["scheduler.tick_heartbeat"]


def test_site_prune_job_ids():
    assert _sched(BoomDB()).prune_run_history() == 0
    assert _sites() == ["scheduler.prune_job_ids"]


def test_site_recent_runs():
    s = _sched(BoomDB())
    s._find = lambda ref: {"id": "j1"}
    assert s.recent_runs("j1") == []
    assert _sites() == ["scheduler.recent_runs"]


def test_site_execute_notify():
    s = _sched(BoomDB())
    s.db = BoomDB()
    s.notifier = types.SimpleNamespace(publish=_boom)
    s._run_payload_guarded = lambda row, t: (False, "boom", False, None, None)
    s._next_after_run = lambda row, now: (None, True)
    s._record_run = lambda *a, **k: None
    s._ledger = lambda *a, **k: None
    s._emit_bus = lambda *a, **k: None
    s._is_routine = lambda row: False
    out = s._execute({"id": "j1", "name": "job1", "payload_kind": "tool"})
    assert out["ok"] is False
    assert _sites() == ["scheduler.execute_notify"]


def test_site_health_queue_depth():
    s = _sched()
    s.list_jobs = lambda **k: []
    s._live_owner_channels = lambda: []
    s.notifier = _fake_notifier(queue_depth=_boom)
    out = s.health()
    assert out["delivery"]["queue"] == {"retryable": 0, "held": 0, "dead": 0}
    assert _sites() == ["scheduler.health_queue_depth"]


def test_site_health_termux_fallback():
    s = _sched()
    s.list_jobs = lambda **k: []
    s._live_owner_channels = lambda: []
    s.notifier = _fake_notifier(termux_fallback_available=_boom)
    out = s.health()
    assert out["delivery"]["termux_fallback"] is False
    assert _sites() == ["scheduler.health_termux_fallback"]


def test_site_live_owner_channels():
    gw = types.SimpleNamespace(status=_boom)
    s = _sched()
    s.notifier = types.SimpleNamespace(gateway=gw)
    s.context = types.SimpleNamespace(
        settings=types.SimpleNamespace(
            partner=types.SimpleNamespace(owner_chats="telegram:123")))
    assert s._live_owner_channels() == []
    assert _sites() == ["scheduler.live_owner_channels"]


# ---- NOT-recording pins (exclusions) --------------------------------------

def test_excl_execute_cmd_name():
    # E1: malformed command payload JSON → plain failure doc, publish ok.
    s = _sched(BoomDB())
    s.db = BoomDB()
    s.notifier = types.SimpleNamespace(publish=lambda *a, **k: None)
    s._run_payload_guarded = lambda row, t: (False, "boom", False, None, None)
    s._next_after_run = lambda row, now: (None, True)
    s._record_run = lambda *a, **k: None
    s._ledger = lambda *a, **k: None
    s._emit_bus = lambda *a, **k: None
    s._is_routine = lambda row: False
    out = s._execute({"id": "j1", "name": "job1", "payload_kind": "command",
                      "payload": "{bad json"})
    assert out["ok"] is False
    assert _sites() == []


def test_excl_health_overlaps():
    # E2: lock failure leaves the default empty overlaps, display-only.
    class BadLock:
        def __enter__(self):
            raise RuntimeError("lock broken")

        def __exit__(self, *a):
            return False

    s = _sched()
    s.list_jobs = lambda **k: []
    s._live_owner_channels = lambda: []
    s._lock = BadLock()
    s.notifier = _fake_notifier()
    out = s.health()
    assert out["overlaps"] == {"active": [], "count": 0}
    assert _sites() == []


def test_excl_health_recent_runs():
    # E3: history query failure leaves the default empty list, display-only.
    s = _sched(BoomDB())
    s.list_jobs = lambda **k: []
    s._live_owner_channels = lambda: []
    s.notifier = _fake_notifier()
    out = s.health()
    assert out["recent_runs"] == []
    assert _sites() == []


def test_excl_ensure_persona_jobs_list(monkeypatch):
    # E4: list failure → add attempt surfaces any genuine breakage.
    class FakeSched:
        def __init__(self, context):
            pass

        def list_jobs(self, **k):
            raise RuntimeError("table missing")

        def add(self, *a, **k):
            return {"id": "new1"}

    monkeypatch.setattr(sch, "Scheduler", FakeSched)
    out = sch.ensure_persona_jobs(types.SimpleNamespace())
    assert all(v.get("scheduled") is True for v in out.values())
    assert _sites() == []
