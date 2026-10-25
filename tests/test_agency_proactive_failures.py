"""Pass #178 pins: the agency failure ledger (REUSED from #154) + proactive conversions.

``nomorals/agency/failures.py`` (namespace ``agency``) is reused from
#154: proactive.py is an agency-subsystem module, so reuse is the
expected call (unlike #161's engine/trivia split). The 17 durability
tests below mirror the #141+ shape exactly (same ledger as #154);
the 17 site pins prove each conversion records without changing flow
(fallback value unchanged AND a ledger entry under its site name).
There are no exclusions in this pass, so there are no NOT-recording
pins. No pin hardcodes scan totals or proactive counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree). Restart tests use real subprocesses.
Every trigger was probed live before its pin was written (probe.py).

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import contextlib
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agency import failures as af
from nomorals.agency import proactive as P


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


class FakeDB:
    def __init__(self, rows=None, fail=False):
        self.rows = rows
        self.fail = fail

    def execute(self, *a, **k):
        if self.fail:
            raise RuntimeError("db down")

    def query(self, *a, **k):
        if self.fail:
            raise RuntimeError("db down")
        return self.rows


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("agency")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agency").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agency")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agency").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agency")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agency").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agency").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agency").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agency")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agency").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("agency").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agency").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agency").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agency")
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
    entries, _ = fs.Store("agency").load()
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
    assert af.format_failures() == "no agency failures recorded."
    af.record_failure("agency.site-a", ValueError("boom"))
    af.record_failure("agency.site-b", RuntimeError("bang"))
    assert af.failure_count() == 2
    assert [f.site for f in af.recent_failures()] == ["agency.site-b", "agency.site-a"]
    assert [f.site for f in af.recent_failures(1)] == ["agency.site-b"]
    text = af.format_failures()
    assert "2 total" in text and "agency.site-a" in text and "agency.site-b" in text
    af.clear_failures()
    assert af.failure_count() == 0
    assert af.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    af.record_failure("agency.disk", ValueError("boom"))
    entries, total = fs.Store("agency").load()
    assert total == 1 and entries[0]["site"] == "agency.disk"
    af.clear_failures()
    assert fs.Store("agency").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    af.record_failure("agency.mem", ValueError("boom"))
    assert af.failure_count() == 1
    assert [f.site for f in af.recent_failures()] == ["agency.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                af.record_failure(f"agency.w{k}-{i}", ValueError("boom"))
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
        "from nomorals.agency import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agency import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agency import failures as f\n"
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
    store_path = fs.Store("agency").path()
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


# ---- Site pins (fallback unchanged AND ledger entry) ----------------------

def test_pin_load_row_decode():
    st = P.load_proactive_state(FakeDB(rows=[{"key": "deals",
                                               "value": "{not json"}]))
    assert st["deals"] == {"seen_alert_ids": set()}
    assert _sites() == ["proactive.load_row_decode"]


def test_pin_load_state():
    assert P.load_proactive_state(FakeDB(fail=True)) == P._fresh_state()
    assert _sites() == ["proactive.load_state"]


def test_pin_save_state():
    assert P.save_proactive_state(FakeDB(fail=True), {}) is None
    assert _sites() == ["proactive.save_state"]


def test_pin_detect_dispatch():
    src = SimpleNamespace(check_alerts=lambda: [123])
    assert P.detect_triggers("deals", src, {}, now=0.0) == []
    assert _sites() == ["proactive.detect_dispatch"]


def test_pin_deals_alerts():
    src = SimpleNamespace(check_alerts=_boom)
    assert P.detect_triggers("deals", src, {}, now=0.0) == []
    assert _sites() == ["proactive.deals_alerts"]


def test_pin_weather_snapshot():
    def snap(place):
        if place == "a":
            raise RuntimeError("boom")
        return {"days": [], "alerts": []}

    state = {}
    assert P.detect_triggers("weather", SimpleNamespace(snapshot=snap),
                             state, now=0.0, places=["a", "b"]) == []
    assert state["weather"]["baselines"].get("b") is not None
    assert _sites() == ["proactive.weather_snapshot"]


def test_pin_news_clusters():
    src = SimpleNamespace(clusters=_boom)
    assert P.detect_triggers("news", src, {}, now=0.0) == []
    assert _sites() == ["proactive.news_clusters"]


def test_pin_social_backlog():
    src = SimpleNamespace(backlog=_boom)
    assert P.detect_triggers("social", src, {}, now=0.0) == []
    assert _sites() == ["proactive.social_backlog"]


def test_pin_leads_live():
    src = SimpleNamespace(live_leads=_boom)
    assert P.detect_triggers("leads", src, {}, now=0.0) == []
    assert _sites() == ["proactive.leads_live"]


def test_pin_gigs_scan():
    src = SimpleNamespace(scan=_boom, follow_ups_due=lambda: [])
    assert P.detect_triggers("gigs", src, {}, now=0.0) == []
    assert _sites() == ["proactive.gigs_scan"]


def test_pin_gigs_followups():
    src = SimpleNamespace(scan=lambda: [], follow_ups_due=_boom)
    assert P.detect_triggers("gigs", src, {}, now=0.0) == []
    assert _sites() == ["proactive.gigs_followups"]


def test_pin_research_gaps():
    src = SimpleNamespace(open_gaps=_boom, due_watches=lambda now=None: [])
    assert P.detect_triggers("research", src, {}, now=0.0) == []
    assert _sites() == ["proactive.research_gaps"]


def test_pin_research_watches():
    src = SimpleNamespace(open_gaps=lambda limit=50: [],
                          due_watches=_boom)
    assert P.detect_triggers("research", src, {}, now=0.0) == []
    assert _sites() == ["proactive.research_watches"]


def test_pin_wisdom_queue():
    src = SimpleNamespace(_pending_manifest=_boom)
    assert P.detect_triggers("wisdom", src, {}, now=0.0) == []
    assert _sites() == ["proactive.wisdom_queue"]


def test_pin_impossible_queue():
    src = SimpleNamespace(queue_list=_boom)
    assert P.detect_triggers("impossible", src, {}, now=0.0) == []
    assert _sites() == ["proactive.impossible_queue"]


def test_pin_digestion_count():
    src = SimpleNamespace(count=_boom)
    assert P.detect_triggers("digestion", src, {}, now=0.0) == []
    assert _sites() == ["proactive.digestion_count"]


def test_pin_planner_list():
    src = SimpleNamespace(list_active=_boom)
    assert P.detect_triggers("planner", src, {}, now=0.0) == []
    assert _sites() == ["proactive.planner_list"]
