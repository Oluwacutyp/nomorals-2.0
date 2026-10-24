"""Pass #152 pins: the agents failure ledger (new) + morning_pulse conversions.

No agents ledger existed (``nomorals/agents/failures.py`` absent; the
singular ``nomorals/agents/failure.py`` is an unrelated failure-analysis
agent, not a ledger), so this pass creates one: namespace ``agents``,
durability-wired through the shared failure_store helper, mirroring the
legal/social/marketing ledgers exactly.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.agents.failures``; restart tests use real
subprocesses (a fresh interpreter is the honest simulated restart — no
module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow: each
converted fallback returns exactly its old value AND leaves a ledger
entry under its site name. The one excluded handler is pinned as NOT
recording: the ``_migrate_seen_store`` rename (expected-failure cleanup
that fires on every poll after the first migration — recording it would
flood the ledger with non-failures).
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
from nomorals.agents import failures as af
from nomorals.agents import morning_pulse as mp


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


def _ctx(tmp_path, workspace=None):
    ws = str(workspace if workspace is not None else tmp_path / "ws")
    Path(ws).mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(settings=SimpleNamespace(workspace_dir=ws))


def _ctx_pulse(**pulse_attrs):
    return SimpleNamespace(
        settings=SimpleNamespace(pulse=SimpleNamespace(**pulse_attrs)))


class _RaisingSettings:
    """workspace_dir access itself raises — for the *_path guard pins."""

    @property
    def workspace_dir(self):
        raise RuntimeError("settings gone")


class _RaisingCtx:
    def __init__(self):
        self.settings = _RaisingSettings()


class _SchedStub:
    def __init__(self, jobs=None, list_raises=False, remove_raises=False):
        self._jobs = jobs if jobs is not None else []
        self._list_raises = list_raises
        self._remove_raises = remove_raises
        self.removed = []
        self.added = []

    def list_jobs(self):
        if self._list_raises:
            raise RuntimeError("scheduler table gone")
        return list(self._jobs)

    def remove(self, job_id):
        if self._remove_raises:
            raise RuntimeError("remove refused")
        self.removed.append(job_id)

    def add(self, *args, **kwargs):
        self.added.append(args)
        return {"id": "new-job"}


class _Registry:
    def __init__(self, context):
        self.context = context
        self.fn = None

    def register(self, *args, **kwargs):
        def deco(fn):
            self.fn = fn
            return fn
        return deco


# ---- Store-level -------------------------------------------------------

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
    from unittest.mock import patch
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
    from unittest.mock import patch
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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def test_converted_prefs_sidecar_records_and_returns_defaults(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pulse_prefs.json").write_text("{not json", encoding="utf-8")
    out = mp._prefs(_ctx(tmp_path, ws))
    assert out["time"] == mp.DEFAULT_PULSE_TIME
    assert out["timezone"] == mp.PULSE_TIMEZONE
    assert "morning_pulse._prefs.sidecar" in _sites()


def test_converted_marker_path_records_and_returns_none():
    assert mp._marker_path(_RaisingCtx()) is None
    assert "morning_pulse._marker_path" in _sites()


def test_converted_read_marker_records_and_returns_empty(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pulse_last_run.json").write_text("{not json", encoding="utf-8")
    assert mp._read_marker(_ctx(tmp_path, ws)) == {}
    assert "morning_pulse._read_marker" in _sites()


def test_converted_pulse_due_today_records_and_returns_zero():
    assert mp._pulse_due_today(_ctx_pulse(time="bogus")) == 0.0
    assert "morning_pulse._pulse_due_today" in _sites()


def test_converted_ensure_pulse_list_jobs_records_and_still_schedules(tmp_path, monkeypatch):
    stub = _SchedStub(list_raises=True)
    monkeypatch.setattr("nomorals.agents.scheduler.Scheduler", lambda ctx: stub)
    out = mp.ensure_pulse_job(_ctx(tmp_path))
    assert out["scheduled"] is True
    assert "morning_pulse.ensure_pulse_job.list_jobs" in _sites()


def test_converted_ensure_pulse_remove_records_and_still_schedules(tmp_path, monkeypatch):
    stub = _SchedStub(
        jobs=[{"name": "morning-pulse", "id": "j1", "spec": "daily 00:00"}],
        remove_raises=True)
    monkeypatch.setattr("nomorals.agents.scheduler.Scheduler", lambda ctx: stub)
    out = mp.ensure_pulse_job(_ctx(tmp_path))
    assert out["scheduled"] is True
    assert "morning_pulse.ensure_pulse_job.remove" in _sites()


def test_converted_deliver_audio_status_records_and_returns_false(tmp_path, monkeypatch):
    class _Gw:
        def status(self):
            raise RuntimeError("gateway down")

    monkeypatch.setattr("nomorals.agents.notifier.resolve_gateway", lambda ctx: _Gw())
    monkeypatch.setattr("nomorals.social.chat.gateway.resolve_owner_targets", lambda *a, **k: [])
    audio = tmp_path / "pulse.wav"
    audio.write_bytes(b"RIFF")
    assert mp._deliver_audio(_ctx(tmp_path), str(audio)) is False
    assert "morning_pulse._deliver_audio.status" in _sites()


def test_converted_deliver_audio_channel_records_and_returns_false(tmp_path, monkeypatch):
    class _Gw:
        def status(self):
            return {"telegram": {"running_in_session": True}}

        def _adapter_for(self, plat):
            return SimpleNamespace()

    monkeypatch.setattr("nomorals.agents.notifier.resolve_gateway", lambda ctx: _Gw())
    monkeypatch.setattr(
        "nomorals.social.chat.gateway.resolve_owner_targets",
        lambda *a, **k: [SimpleNamespace(platform="telegram", chat_id="1")])
    monkeypatch.setattr("nomorals.partner.chat_modes.proactive_allowed", lambda k: True)

    def _boom_send(*args, **kwargs):
        raise RuntimeError("transport down")

    monkeypatch.setattr("nomorals.social.chat.voice_transport.send_voice_note", _boom_send)
    audio = tmp_path / "pulse.wav"
    audio.write_bytes(b"RIFF")
    assert mp._deliver_audio(_ctx(tmp_path), str(audio)) is False
    assert "morning_pulse._deliver_audio.channel" in _sites()


def test_converted_register_status_jobs_records_and_lists_none(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler",
        lambda ctx: _SchedStub(list_raises=True))
    reg = _Registry(_ctx(tmp_path))
    mp.register(reg)
    out = reg.fn(action="status", slot="morning")
    assert out["job"] is None
    assert "morning_pulse.register.list_jobs" in _sites()


def test_converted_slot_prefs_records_and_returns_defaults(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pulse_prefs.json").write_text("{not json", encoding="utf-8")
    out = mp._slot_prefs(_ctx(tmp_path, ws), "afternoon")
    assert out["time"] == "13:00"
    assert "morning_pulse._slot_prefs" in _sites()


def test_converted_slot_marker_path_records_and_returns_none():
    assert mp._slot_marker_path(_RaisingCtx(), "afternoon") is None
    assert "morning_pulse._slot_marker_path" in _sites()


def test_converted_read_slot_marker_records_and_returns_empty(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pulse_last_run_afternoon.json").write_text("{not json", encoding="utf-8")
    assert mp._read_slot_marker(_ctx(tmp_path, ws), "afternoon") == {}
    assert "morning_pulse._read_slot_marker" in _sites()


def test_converted_seen_path_records_and_returns_none():
    assert mp._seen_path(_RaisingCtx()) is None
    assert "morning_pulse._seen_path" in _sites()


def test_converted_deliver_stories_notifier_records_and_returns_empty(tmp_path, monkeypatch):
    class _BoomNotifier:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("notifier down")

    monkeypatch.setattr("nomorals.agents.notifier.Notifier", _BoomNotifier)
    assert mp._deliver_stories(_ctx(tmp_path), "afternoon", [{"title": "x"}]) == []
    assert "morning_pulse._deliver_stories.notifier" in _sites()


def test_converted_afternoon_quote_records_and_omits_symbol(tmp_path, monkeypatch):
    def _quote(sym):
        if sym == "XAUUSD":
            raise RuntimeError("feed down")
        return {"price": 100.0, "change_pct_24h": 1.0}

    monkeypatch.setattr("nomorals.integrations.market_data.quote", _quote)
    text, keys = mp._afternoon_markets(_ctx(tmp_path))
    assert "XAUUSD" not in text and "mkt-XAUUSD" not in keys
    assert "BTC" in text
    assert "morning_pulse._afternoon_markets.quote" in _sites()


def test_converted_afternoon_outer_records_and_returns_empty(tmp_path, monkeypatch):
    class _RaisingIter:
        def __iter__(self):
            raise RuntimeError("symbols gone")

    monkeypatch.setattr(mp, "AFTERNOON_SYMBOLS", _RaisingIter())
    assert mp._afternoon_markets(_ctx(tmp_path)) == ("", [])
    assert "morning_pulse._afternoon_markets.outer" in _sites()


def test_converted_fresh_news_poll_records_and_returns_empty(tmp_path, monkeypatch):
    class _BoomAgent:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("news down")

    monkeypatch.setattr("nomorals.agents.news.NewsAgent", _BoomAgent)
    assert mp._fresh_news_items(_ctx(tmp_path)) == []
    assert "morning_pulse._fresh_news_items.poll" in _sites()


def test_converted_today_spec_records_and_skips_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler",
        lambda ctx: _SchedStub(jobs=[{"name": "backup", "spec": "daily "}]))
    assert mp._today_remainder(_ctx(tmp_path)) == ""
    assert "morning_pulse._today_remainder.spec" in _sites()


def test_converted_today_outer_records_and_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler",
        lambda ctx: _SchedStub(list_raises=True))
    assert mp._today_remainder(_ctx(tmp_path)) == ""
    assert "morning_pulse._today_remainder.outer" in _sites()


def test_converted_day_wrap_records_and_returns_empty(tmp_path, monkeypatch):
    class _BoomLedger:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("ledger down")

    monkeypatch.setattr("nomorals.agents.autonomy_ledger.AutonomyLedger", _BoomLedger)
    assert mp._day_wrap(_ctx(tmp_path)) == ""
    assert "morning_pulse._day_wrap" in _sites()


def test_converted_tomorrow_spec_records_and_skips_job(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler",
        lambda ctx: _SchedStub(jobs=[{"name": "backup", "spec": "daily "}]))
    assert mp._tomorrow_preview(_ctx(tmp_path)) == ""
    assert "morning_pulse._tomorrow_preview.spec" in _sites()


def test_converted_tomorrow_outer_records_and_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.scheduler.Scheduler",
        lambda ctx: _SchedStub(list_raises=True))
    assert mp._tomorrow_preview(_ctx(tmp_path)) == ""
    assert "morning_pulse._tomorrow_preview.outer" in _sites()


def test_converted_pending_proposals_records_and_omits_section(tmp_path, monkeypatch):
    class _Db:
        def query(self, sql, *args, **kwargs):
            if "proactive_log" in sql:
                raise RuntimeError("db gone")
            return []

    class _Store:
        def __init__(self, *args, **kwargs):
            pass

        def list_enabled(self):
            return []

    monkeypatch.setattr("nomorals.finance.alerts.AlertStore", _Store)
    ctx = _ctx(tmp_path)
    ctx.db = _Db()
    assert "proposal" not in mp._pending_items(ctx)
    assert "morning_pulse._pending_items.proposals" in _sites()
    assert "morning_pulse._pending_items.alerts" not in _sites()


def test_converted_pending_alerts_records_and_omits_section(tmp_path, monkeypatch):
    class _Db:
        def query(self, sql, *args, **kwargs):
            return []

    class _BoomStore:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("alerts down")

    monkeypatch.setattr("nomorals.finance.alerts.AlertStore", _BoomStore)
    ctx = _ctx(tmp_path)
    ctx.db = _Db()
    assert "alert" not in mp._pending_items(ctx)
    assert "morning_pulse._pending_items.alerts" in _sites()
    assert "morning_pulse._pending_items.proposals" not in _sites()


def test_converted_slot_due_today_records_and_returns_zero():
    ctx = _ctx_pulse(afternoon_time="bogus")
    assert mp._slot_due_today(ctx, "afternoon") == 0.0
    assert "morning_pulse._slot_due_today" in _sites()


def test_converted_slot_disable_remove_records_and_returns_disabled(tmp_path, monkeypatch):
    stub = _SchedStub(
        jobs=[{"name": "afternoon-pulse", "id": "j1"}],
        remove_raises=True)
    monkeypatch.setattr("nomorals.agents.scheduler.Scheduler", lambda ctx: stub)
    out = mp.ensure_slot_job(_ctx_pulse(afternoon_enabled=False), "afternoon")
    assert out == {"name": "afternoon-pulse", "disabled": True}
    assert "morning_pulse.ensure_slot_job.disable_remove" in _sites()


def test_converted_slot_list_jobs_records_and_still_schedules(tmp_path, monkeypatch):
    stub = _SchedStub(list_raises=True)
    monkeypatch.setattr("nomorals.agents.scheduler.Scheduler", lambda ctx: stub)
    out = mp.ensure_slot_job(_ctx(tmp_path), "afternoon")
    assert out["scheduled"] is True
    assert "morning_pulse.ensure_slot_job.list_jobs" in _sites()


def test_converted_slot_remove_records_and_still_schedules(tmp_path, monkeypatch):
    stub = _SchedStub(
        jobs=[{"name": "afternoon-pulse", "id": "j1", "spec": "daily 00:00"}],
        remove_raises=True)
    monkeypatch.setattr("nomorals.agents.scheduler.Scheduler", lambda ctx: stub)
    out = mp.ensure_slot_job(_ctx(tmp_path), "afternoon")
    assert out["scheduled"] is True
    assert "morning_pulse.ensure_slot_job.remove" in _sites()


# ---- Excluded handler: proven NOT recording ------------------------------

def test_excluded_migrate_rename_records_nothing(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "pulse_seen.json").write_text("{}", encoding="utf-8")
    retired = ws / "pulse_seen.json.retired"
    retired.mkdir()
    (retired / "keep.txt").write_text("x", encoding="utf-8")
    assert mp._migrate_seen_store(_ctx(tmp_path, ws)) == 0
    assert _sites() == []
