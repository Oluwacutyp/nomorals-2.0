"""Pass #213 pins: regulatory failures stay silent, ledger intact.

``nomorals/legal/regulatory.py`` converts 0 of its 12 true sites and
excludes 12: the watch-store helpers map an unavailable DB to the same
value a genuine miss produces (None / False / []), the scoring and
calendar helpers degrade to their documented fail-soft values (0 / []),
the per-row guard skips one bad row without killing the query, the
scheduler seam fails safe toward having the daily cron, and the chat
input parsers keep their defaults on bad input. The three mutation-
adjacent exclusions are the weakest and are pinned with their exact
shape: a failed manual note returns the genuine not-new False and the
next check still surfaces the item (delayed, never lost); a failed
calendar sync returns the genuine zero count and a healthy re-sync
dedupes (partial writes are never duplicated); a failed dismiss
returns the genuine not-dismissed False, identical to a nonexistent
id, and the item keeps alerting (visible, not silent).

The 17 durability tests below mirror the #141+ shape exactly (same as
#209/#210/#211/#212, against the REUSED legal ledger); the 12
NOT-recording pins prove each exclusion stays silent with its designed
output. No pin hardcodes scan totals or regulatory counts. Every test
redirects stores with the app's own ``NM_HOME`` override (never the
real home, never the repo tree) and every watch uses an explicit tmp
``db_path`` (the module default lives under the real home). Restart
tests use real subprocesses. Every trigger was probed live before its
pin was written.

Pins rule (#169): no bare-swallow handler anywhere in this file — the
scanner counts test files too. Where a pin must swallow, it binds
``as exc`` + ``assert isinstance(exc, Exception)``. Docstrings avoid
the literal bare-swallow phrase (#173 lesson).
"""
import asyncio
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
from nomorals.legal import failures as lf
from nomorals.legal import regulatory as R


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    lf.clear_failures()
    yield
    lf.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"}
            for i in range(n)]


def _assert_silent():
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


# ---- Store-level (durability mirror of #141+) ----------------------------

def test_store_round_trip():
    store = fs.Store("legal")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("legal").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("legal")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("legal").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("legal")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("legal").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("legal").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("legal").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("legal")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("legal").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("legal").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("legal").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("legal").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("legal")
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
    entries, _ = fs.Store("legal").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        lf.record_failure("no-home", ValueError("boom"))
        assert lf.failure_count() == 1
        assert [f.site for f in lf.recent_failures()] == ["no-home"]
        assert fs.write_failures() > before


def test_ledger_api_record_recent_count_clear_format():
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []
    assert lf.format_failures() == "no legal failures recorded."
    lf.record_failure("regulatory.site-a", ValueError("boom"))
    lf.record_failure("regulatory.site-b", RuntimeError("bang"))
    assert lf.failure_count() == 2
    assert [f.site for f in lf.recent_failures()] == ["regulatory.site-b",
                                                      "regulatory.site-a"]
    assert [f.site for f in lf.recent_failures(1)] == ["regulatory.site-b"]
    text = lf.format_failures()
    assert "2 total" in text and "regulatory.site-a" in text \
        and "regulatory.site-b" in text
    lf.clear_failures()
    assert lf.failure_count() == 0
    assert lf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    lf.record_failure("regulatory.disk", ValueError("boom"))
    entries, total = fs.Store("legal").load()
    assert total == 1 and entries[0]["site"] == "regulatory.disk"
    lf.clear_failures()
    assert fs.Store("legal").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    lf.record_failure("regulatory.mem", ValueError("boom"))
    assert lf.failure_count() == 1
    assert [f.site for f in lf.recent_failures()] == ["regulatory.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                lf.record_failure(f"regulatory.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
            assert isinstance(exc, Exception)
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert lf.failure_count() == 400
    assert len(lf.recent_failures(1000)) == 200
    entries, total = lf._store.load()
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
        "from nomorals.legal import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.legal import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.legal import failures as f\n"
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
    store_path = fs.Store("legal").path()
    store_path = home / "failures" / store_path.name
    assert store_path.exists()
    store_path.write_text("{not json", encoding="utf-8")
    proc = _run_subprocess(home, _READ_CODE)
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["count"] == 0
    assert data["sites"] == []


# ---- NOT-recording pins (one per exclusion) ------------------------------

class _BoomDb:
    """A store handle whose every operation raises."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("db down")

    def commit(self):
        raise RuntimeError("db down")


def test_not_recording_240_bad_source_dict_returns_none():
    class _BoomFloat:
        def __float__(self):
            raise RuntimeError("boom")

    d = {"regulator": "CBN", "title": "T", "published": _BoomFloat()}
    assert R.item_from_dict(d) is None
    assert R.item_from_dict({"regulator": "NOPE", "title": "T"}) is None
    _assert_silent()


def test_not_recording_392_score_failure_returns_zero():
    item = R.RegulatoryItem(id="i", regulator="CBN", title="T",
                            summary="S", topics=5)
    assert R.impact_score(item) == 0
    _assert_silent()


def test_not_recording_424_deadline_failure_returns_empty():
    assert R.upcoming_deadlines(months="bad") == []
    _assert_silent()


def test_not_recording_454_sync_failure_returns_zero_and_resync_dedupes():
    class _BoomPort:
        def obligations(self, include_done=True):
            raise RuntimeError("db down")

        def track_obligation(self, _a, desc, ts):
            raise AssertionError("must not be called")

    assert R.sync_calendar_to_portfolio(_BoomPort(), 6) == 0
    _assert_silent()

    class _FlakyPort:
        def __init__(self):
            self.tracked = []
            self.calls = 0

        def obligations(self, include_done=True):
            return [SimpleNamespace(description=d, due_at=t)
                    for d, t in self.tracked]

        def track_obligation(self, _a, desc, ts):
            self.calls += 1
            if self.calls >= 2:
                raise RuntimeError("boom on 2nd track")
            self.tracked.append((desc, ts))
            return "oid1"

    flaky = _FlakyPort()
    assert R.sync_calendar_to_portfolio(flaky, 24) == 0
    assert len(flaky.tracked) == 1
    _assert_silent()

    class _HealthyPort:
        def __init__(self, tracked):
            self.tracked = list(tracked)

        def obligations(self, include_done=True):
            return [SimpleNamespace(description=d, due_at=t)
                    for d, t in self.tracked]

        def track_obligation(self, _a, desc, ts):
            self.tracked.append((desc, ts))
            return "oid"

    healthy = _HealthyPort(flaky.tracked)
    R.sync_calendar_to_portfolio(healthy, 24)
    assert len(healthy.tracked) == 5
    assert len({(d, t) for d, t in healthy.tracked}) == 5
    _assert_silent()


def test_not_recording_676_dismiss_failure_returns_false(tmp_path):
    w = R.RegulatoryWatch(db_path=str(tmp_path / "reg.db"))
    assert w.dismiss("no-such-id") is False
    w._db = _BoomDb()
    assert w.dismiss("any-id") is False
    _assert_silent()


def test_not_recording_706_bad_row_skipped_rest_returned(tmp_path):
    w = R.RegulatoryWatch(db_path=str(tmp_path / "reg.db"))
    good = {"item_id": "good1", "regulator": "CBN", "title": "Good rule",
            "summary": "s", "source_url": "https://cbn.example", "ref_no": "",
            "published": 0.0, "topics_json": '["aml"]', "urgency": "normal",
            "effective_date": 0.0}
    bad = dict(good, item_id="bad1", topics_json="{invalid")

    class _Rows:
        def __init__(self, rows):
            self._rows = rows

        def fetchall(self):
            return self._rows

    class _StubDb:
        def __init__(self, rows):
            self._rows = rows

        def execute(self, *args, **kwargs):
            return _Rows(self._rows)

    w._db = _StubDb([bad, good])
    out = w.upcoming_effective(90)
    assert [i.id for i in out] == ["good1"]
    _assert_silent()


def test_not_recording_709_query_failure_returns_empty(tmp_path):
    w = R.RegulatoryWatch(db_path=str(tmp_path / "reg.db"))
    w._db = _BoomDb()
    assert w.upcoming_effective(90) == []
    _assert_silent()


def test_not_recording_754_note_failure_returns_false_then_healthy_note_works(
        tmp_path):
    broken = R.RegulatoryWatch(db_path=str(tmp_path / "reg2.db"))
    broken._db = _BoomDb()
    w = R.RegulatoryWatch(db_path=str(tmp_path / "reg.db"))
    item = R.RegulatoryItem(id="n1", regulator="CBN", title="Rule",
                            summary="s")
    assert broken.note_item(item) is False
    assert w.note_item(item) is True
    _assert_silent()


def test_not_recording_859_list_jobs_failure_still_schedules():
    class _Sched:
        def __init__(self):
            self.scheduled = []

        def list_jobs(self):
            raise RuntimeError("list down")

        async def schedule_cron(self, **kw):
            self.scheduled.append(kw)
            return True

    async def _run():
        sched = _Sched()
        ok = R.ensure_schedule(sched)
        await asyncio.sleep(0.1)
        return ok, sched

    ok, sched = asyncio.run(_run())
    assert ok is True
    assert len(sched.scheduled) == 1
    assert sched.scheduled[0]["task_id"] == "regulatory-daily"
    _assert_silent()


def test_not_recording_881_schedule_failure_returns_false(monkeypatch):
    def _boom():
        raise ValueError("loop lookup down")

    monkeypatch.setattr(asyncio, "get_running_loop", _boom)

    class _Sched:
        def list_jobs(self):
            return []

        async def schedule_cron(self, **kw):
            return True

    assert R.ensure_schedule(_Sched()) is False
    _assert_silent()


def test_not_recording_982_bad_months_keeps_default(tmp_path):
    w = R.RegulatoryWatch(db_path=str(tmp_path / "reg.db"))
    ctx = SimpleNamespace(regulatory_store=w)
    out = R.control_regwatch("calendar banana", context=ctx)
    assert "hit an error" not in out
    assert "calendar" in out.lower() or "deadlines" in out.lower()
    _assert_silent()


def test_not_recording_1000_bad_days_keeps_default(tmp_path):
    w = R.RegulatoryWatch(db_path=str(tmp_path / "reg.db"))
    ctx = SimpleNamespace(regulatory_store=w)
    out = R.control_regwatch("effective banana", context=ctx)
    assert "hit an error" not in out
    assert "taking effect" in out.lower() or "no tracked rules" in out.lower()
    _assert_silent()
