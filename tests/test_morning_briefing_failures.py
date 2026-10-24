"""Pass #146 pins: the agents/morning_briefing failure ledger + durability.

Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through the new ``nomorals.agents.morning_briefing_failures`` ledger;
restart tests use real subprocesses (a fresh interpreter is the honest
simulated restart — no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents import morning_briefing_failures as mb


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    mb.clear_failures()
    yield
    mb.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("agents.morning_briefing")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("agents.morning_briefing").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("agents.morning_briefing")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("agents.morning_briefing").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("agents.morning_briefing")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("agents.morning_briefing").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("agents.morning_briefing").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("agents.morning_briefing").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("agents.morning_briefing")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("agents.morning_briefing").load() == ([], 0)
    assert store.save(_entries(1, "after"), 1) is True
    entries, total = fs.Store("agents.morning_briefing").load()
    assert [e["site"] for e in entries] == ["after0"]
    assert total == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("agents.morning_briefing").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    # The #141 fix-up shape, pinned for the new ledger too: save()
    # must resolve its path INSIDE the try, so a home-resolution
    # failure degrades to a counted write failure, never raises.
    # NM_HOME must be unset: an absolute NM_HOME never calls
    # expanduser, which is how scratch-home batches mask this.
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("agents.morning_briefing").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        mb.record_failure("no-home", ValueError("boom"))
        assert mb.failure_count() == 1
        assert [f.site for f in mb.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("agents.morning_briefing")
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
    entries, _ = fs.Store("agents.morning_briefing").load()
    # Exactly one worker's complete snapshot (unique tmp names per save
    # make a torn document impossible); a torn file would parse to [].
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_api_record_recent_count_clear_format():
    assert mb.failure_count() == 0
    assert mb.recent_failures() == []
    assert mb.format_failures() == "no agents.morning_briefing failures recorded."
    mb.record_failure("mb.site-a", ValueError("boom"))
    mb.record_failure("mb.site-b", RuntimeError("bang"))
    assert mb.failure_count() == 2
    assert [f.site for f in mb.recent_failures()] == ["mb.site-b", "mb.site-a"]
    assert [f.site for f in mb.recent_failures(1)] == ["mb.site-b"]
    text = mb.format_failures()
    assert "2 total" in text and "mb.site-a" in text and "mb.site-b" in text
    mb.clear_failures()
    assert mb.failure_count() == 0
    assert mb.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mb.record_failure("disk-a", ValueError("boom"))
    mb.record_failure("disk-b", RuntimeError("bang"))
    assert mb.failure_count() == 2
    assert mb._store.path().exists()
    assert mb._store.path().name == "agents.morning_briefing.json"
    mb.clear_failures()
    assert mb.failure_count() == 0
    assert not mb._store.path().exists()


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    mb.record_failure("blocked", ValueError("boom"))
    assert mb.failure_count() == 1
    assert [f.site for f in mb.recent_failures()] == ["blocked"]
    assert fs.write_failures() > before


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mb.record_failure(f"w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert mb.failure_count() == 400
    assert len(mb.recent_failures(1000)) == 200
    entries, total = mb._store.load()
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
        "from nomorals.agents import morning_briefing_failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.agents import morning_briefing_failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.agents import morning_briefing_failures as f\n"
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
    store_path = home / "failures" / "agents.morning_briefing.json"
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

def _sites():
    return [f.site for f in mb.recent_failures()]


def _ws_ctx(tmp_path):
    return SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path)))


def test_converted_prefs_sidecar_records_and_returns_defaults(tmp_path):
    from nomorals.agents import morning_briefing as mod
    (tmp_path / "briefing_prefs.json").write_text("{not json",
                                                  encoding="utf-8")
    out = mod._prefs(_ws_ctx(tmp_path))
    assert out == {"topics": [], "symbols": [], "pinned_sections": [],
                   "time": "07:00", "max_items_per_section": None}
    assert "morning_briefing._prefs.sidecar" in _sites()


def test_converted_save_prefs_records_and_never_raises(tmp_path):
    from nomorals.agents import morning_briefing as mod
    blocker = tmp_path / "afile"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    ctx = SimpleNamespace(settings=SimpleNamespace(workspace_dir=str(blocker)))
    mod._save_prefs(ctx, {"topics": ["x"]})  # must not raise
    assert "morning_briefing._save_prefs.write" in _sites()


def test_converted_markets_quote_records_and_omits_symbol(tmp_path):
    from nomorals.agents import morning_briefing as mod

    class _RaisingProvider:
        def quote(self, sym, market="crypto"):
            raise RuntimeError("quote down")

    ctx = SimpleNamespace(settings=SimpleNamespace(
        briefing=SimpleNamespace(symbols=["BTC"], topics=[],
                                 pinned_sections=[], time="07:00",
                                 max_items_per_section=None)))
    assert mod.MarketsProvider(provider=_RaisingProvider()).collect(ctx, 0) is None
    assert "morning_briefing.MarketsProvider.collect.quote" in _sites()


def test_converted_alerts_held_records_and_returns_none(monkeypatch):
    from nomorals.agents import morning_briefing as mod

    class _RaisingStore:
        def held_alerts(self):
            raise RuntimeError("store down")

    monkeypatch.setattr(mod, "_watcher_store", lambda ctx: _RaisingStore())
    assert mod.OvernightAlertsProvider().collect(None, 0) is None
    assert "morning_briefing.OvernightAlertsProvider.collect.held_alerts" in _sites()


def test_converted_alerts_repo_ids_records_and_keeps_repo_alert(monkeypatch):
    from nomorals.agents import morning_briefing as mod

    class _Store:
        def held_alerts(self):
            return [{"id": "a1", "title": "t", "body": "b",
                     "watcher_id": "w1", "created_at": time.time(),
                     "severity": "info"}]

        def list(self):
            raise RuntimeError("list down")

        def mark_digested(self, ids):
            pass

    monkeypatch.setattr(mod, "_watcher_store", lambda ctx: _Store())
    sec = mod.OvernightAlertsProvider().collect(None, 0)
    assert sec is not None  # fallback value unchanged: repo alert kept
    assert [i["id"] for i in sec.items] == ["a1"]
    assert "morning_briefing.OvernightAlertsProvider.collect.repo_ids" in _sites()


def test_converted_alerts_mark_digested_records_and_returns_section(monkeypatch):
    from nomorals.agents import morning_briefing as mod

    class _Store:
        def held_alerts(self):
            return [{"id": "a1", "title": "t", "body": "b",
                     "watcher_id": "w1", "created_at": time.time(),
                     "severity": "info"}]

        def list(self):
            return [SimpleNamespace(id="w1", kind="other")]

        def mark_digested(self, ids):
            raise RuntimeError("digest down")

    monkeypatch.setattr(mod, "_watcher_store", lambda ctx: _Store())
    sec = mod.OvernightAlertsProvider().collect(None, 0)
    assert sec is not None
    assert "morning_briefing.OvernightAlertsProvider.collect.mark_digested" in _sites()


def test_converted_calendar_list_events_records_and_returns_empty():
    from nomorals.agents import morning_briefing as mod

    class _RaisingCal:
        def list_events(self):
            raise RuntimeError("cal down")

    ctx = SimpleNamespace(tools=None, calendar=_RaisingCal())
    assert mod.CalendarProvider()._today(ctx) == []
    assert "morning_briefing.CalendarProvider._today.list_events" in _sites()


def test_converted_engagement_set_pinned_records_and_never_raises():
    from nomorals.agents import morning_briefing as mod

    class _RaisingDb:
        def execute(self, *a, **k):
            raise RuntimeError("db down")

    mod._EngagementStore(_RaisingDb()).set_pinned("news", True)
    assert "morning_briefing._EngagementStore.set_pinned.execute" in _sites()


def test_converted_engagement_bump_records_and_never_raises():
    from nomorals.agents import morning_briefing as mod

    class _RaisingDb:
        def execute(self, *a, **k):
            raise RuntimeError("db down")

    mod._EngagementStore(_RaisingDb()).record_view("news")
    assert "morning_briefing._EngagementStore._bump.execute" in _sites()


def test_converted_latest_briefing_records_and_returns_none():
    from nomorals.agents import morning_briefing as mod

    class _RaisingDb:
        def query_one(self, *a, **k):
            raise RuntimeError("db down")

    ctx = SimpleNamespace(db=_RaisingDb())
    assert mod.latest_briefing(ctx) is None
    assert "morning_briefing.latest_briefing.query" in _sites()


def test_converted_ensure_job_remove_records_and_schedules(tmp_path, monkeypatch):
    from nomorals.agents import morning_briefing as mod
    import nomorals.agents.scheduler as sched_mod

    class _Sched:
        def __init__(self, ctx):
            pass

        def list_jobs(self):
            return [{"name": "briefing", "spec": "daily 08:00", "id": "j1"}]

        def remove(self, jid):
            raise RuntimeError("remove down")

        def add(self, *a, **k):
            return {"id": "j2"}

    monkeypatch.setattr(sched_mod, "Scheduler", _Sched)
    out = mod.ensure_briefing_job(_ws_ctx(tmp_path))
    assert out["scheduled"] is True and out["job_id"] == "j2"
    assert "morning_briefing.ensure_briefing_job.remove_stale" in _sites()


def test_converted_fallback_publish_records_and_reports_delivered(monkeypatch):
    from nomorals.agents import morning_briefing as mod
    import nomorals.agents.notifier as notif_mod

    class _RaisingNotifier:
        def __init__(self, ctx):
            pass

        def publish(self, *a, **k):
            raise RuntimeError("notify down")

    monkeypatch.setattr(notif_mod, "Notifier", _RaisingNotifier)
    out = mod._deliver_fallback(None, "2026-10-24", "boom")
    assert out["delivered"] is True  # fallback value unchanged
    assert "morning_briefing._deliver_fallback.publish" in _sites()


def test_converted_status_summary_records_and_reports_empty(tmp_path, monkeypatch):
    from nomorals.agents import morning_briefing as mod
    import nomorals.agents.notifier as notif_mod

    class _Notifier:
        def __init__(self, ctx):
            pass

        def delivery_summary(self, *a, **k):
            raise RuntimeError("summary down")

        def delivery_counts(self, *a, **k):
            return {}

    monkeypatch.setattr(notif_mod, "Notifier", _Notifier)
    monkeypatch.setattr(notif_mod, "resolve_gateway", lambda ctx: None)
    monkeypatch.setattr(notif_mod, "_in_quiet_hours_now", lambda ctx: False)
    out = mod.proactive_status(_ws_ctx(tmp_path))
    assert out["recent"] == []
    assert "morning_briefing.proactive_status.delivery_summary" in _sites()
