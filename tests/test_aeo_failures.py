"""Pass #151 pins: the marketing failure ledger (reused from #100) + aeo conversions.

The marketing ledger already existed (``nomorals/marketing/failures.py``,
namespace ``marketing``, durability-wired through the shared failure_store
helper); this pass reuses it, it does not create a second one.
Every test redirects stores with the app's own ``NM_HOME`` override
(never the real home, never the repo tree). Store-level tests exercise
:class:`nomorals.core.failure_store.Store` directly; ledger-level tests
go through ``nomorals.marketing.failures``; restart tests use real
subprocesses (a fresh interpreter is the honest simulated restart —
no module reload tricks inside a shared process).
Behavior pins prove the conversions record without changing flow:
each converted fallback returns exactly its old value AND leaves a
ledger entry under its site name. The excluded handlers are pinned as
NOT recording: the ``__init__`` ALTER duplicate-column answer (migration
idempotency, fires on every restart of a migrated DB) and
``expand_prompts`` (failure yields exactly the no-expansion outcome).
"""
import json
import os
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.marketing import aeo as aeo_mod
from nomorals.marketing import failures as mf
from nomorals.marketing.aeo import (
    AEOTracker, Mention, VisibilityReport, _gauge, citation_quality,
    classify_positioning, cross_check, ensure_weekly, estimate_cost,
    expand_prompts, find_mentions, parse_citations, sentiment_score,
    site_readiness_check, sparkline,
)


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


def _tracker(tmp_path, name="aeo.db"):
    return AEOTracker(db_path=str(tmp_path / name))


class _BrokenDb:
    """execute() always raises; everything else is inert."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("db gone")

    def commit(self):
        return None


class _BadInt:
    def __int__(self):
        raise RuntimeError("boom")


class _BadLower:
    def lower(self):
        raise RuntimeError("boom")


# ---- Store-level -------------------------------------------------------

def test_store_round_trip():
    store = fs.Store("marketing")
    assert store.save(_entries(3), 3) is True
    entries, total = fs.Store("marketing").load()
    assert total == 3
    assert [e["site"] for e in entries] == ["site0", "site1", "site2"]


def test_store_cap_enforced_on_disk():
    store = fs.Store("marketing")
    assert store.save(_entries(250), 250) is True
    entries, total = fs.Store("marketing").load()
    assert len(entries) == 200
    assert entries[0]["site"] == "site50"
    assert entries[-1]["site"] == "site249"
    assert total == 250


def test_store_clear_removes_file():
    store = fs.Store("marketing")
    assert store.save(_entries(2), 2) is True
    assert store.path().exists()
    store.clear()
    assert not store.path().exists()
    assert fs.Store("marketing").load() == ([], 0)


def test_store_namespaces_do_not_clobber():
    assert fs.Store("marketing").save(_entries(2, "c"), 2) is True
    assert fs.Store("voice").save(_entries(1, "v"), 1) is True
    entries, _ = fs.Store("marketing").load()
    assert [e["site"] for e in entries] == ["c0", "c1"]
    entries, _ = fs.Store("voice").load()
    assert [e["site"] for e in entries] == ["v0"]


def test_store_corrupt_file_falls_back_and_recovers():
    store = fs.Store("marketing")
    assert store.save(_entries(2), 2) is True
    store.path().write_text("{not json", encoding="utf-8")
    assert fs.Store("marketing").load() == ([], 0)
    assert store.save(_entries(1), 1) is True
    entries, total = fs.Store("marketing").load()
    assert total == 1 and len(entries) == 1


def test_store_save_failure_never_raises_and_counts(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    before = fs.write_failures()
    assert fs.Store("marketing").save(_entries(1), 1) is False
    assert fs.write_failures() == before + 1


def test_store_save_home_resolution_failure_never_raises_and_counts(monkeypatch):
    from unittest.mock import patch
    monkeypatch.delenv("NM_HOME", raising=False)
    with patch("os.path.expanduser", side_effect=OSError("no home")):
        before = fs.write_failures()
        assert fs.Store("marketing").save(_entries(1), 1) is False
        assert fs.write_failures() == before + 1


def test_store_concurrent_saves_never_raise_and_never_tear():
    store = fs.Store("marketing")
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
    entries, _ = fs.Store("marketing").load()
    assert len(entries) == 50
    assert len({e["site"] for e in entries}) == len(entries)


# ---- Ledger-level (in-process) ------------------------------------------

def test_ledger_record_home_resolution_failure_degrades_to_memory(monkeypatch):
    from unittest.mock import patch
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
    assert mf.format_failures() == "no marketing failures recorded."
    mf.record_failure("marketing.site-a", ValueError("boom"))
    mf.record_failure("marketing.site-b", RuntimeError("bang"))
    assert mf.failure_count() == 2
    assert [f.site for f in mf.recent_failures()] == ["marketing.site-b", "marketing.site-a"]
    assert [f.site for f in mf.recent_failures(1)] == ["marketing.site-b"]
    text = mf.format_failures()
    assert "2 total" in text and "marketing.site-a" in text and "marketing.site-b" in text
    mf.clear_failures()
    assert mf.failure_count() == 0
    assert mf.recent_failures() == []


def test_ledger_record_and_clear_hit_disk():
    mf.record_failure("marketing.disk", ValueError("boom"))
    entries, total = fs.Store("marketing").load()
    assert total == 1 and entries[0]["site"] == "marketing.disk"
    mf.clear_failures()
    assert fs.Store("marketing").load() == ([], 0)


def test_ledger_write_failure_degrades_to_memory(tmp_path, monkeypatch):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a dir", encoding="utf-8")
    monkeypatch.setenv("NM_HOME", str(blocker))
    mf.record_failure("marketing.mem", ValueError("boom"))
    assert mf.failure_count() == 1
    assert [f.site for f in mf.recent_failures()] == ["marketing.mem"]


def test_ledger_concurrent_records_no_loss_beyond_cap():
    errors = []

    def worker(k):
        try:
            for i in range(50):
                mf.record_failure(f"marketing.w{k}-{i}", ValueError("boom"))
        except Exception as exc:  # noqa: BLE001 — test must surface it
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
        "from nomorals.marketing import failures as f\n" + clear +
        "".join(f"f.record_failure('site{i}', ValueError('boom{i}'))\n"
                for i in range(n)) +
        "print('ok')\n")


_READ_CODE = (
    "import json\n"
    "from nomorals.marketing import failures as f\n"
    "print(json.dumps({'count': f.failure_count(), "
    "'sites': [x.site for x in f.recent_failures(1000)]}))\n")

_CLEAR_CODE = ("from nomorals.marketing import failures as f\n"
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
    store_path = fs.Store("marketing").path()
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

def _report(**kwargs):
    base = dict(report_id="aeo_x", brand="Acme", prompts=["p"], engines=["e"])
    base.update(kwargs)
    return VisibilityReport(**base)


def test_converted_share_of_voice_records_and_returns_zero():
    rep = _report(competitor_hits={"rival": _BadInt()})
    assert rep.share_of_voice == 0.0
    assert _sites() == ["aeo.VisibilityReport.share_of_voice"]


def test_converted_consideration_records_and_returns_empty():
    rep = _report(mentions=[Mention(engine="e", prompt="p", confirmed=True,
                                     positioning=_BadLower())])
    assert rep.consideration == {}
    assert _sites() == ["aeo.VisibilityReport.consideration"]


def test_converted_sentiment_breakdown_records_and_returns_empty():
    rep = _report(mentions=[Mention(engine="e", prompt="p", confirmed=True,
                                     sentiment=object())])
    assert rep.sentiment_breakdown == {}
    assert _sites() == ["aeo.VisibilityReport.sentiment_breakdown"]


def test_converted_format_records_and_returns_stub():
    rep = _report(mentions=[Mention(engine="e", prompt="p", confirmed=True,
                                     context=123)])
    assert rep.format() == "AEO report for Acme (format failed)."
    assert _sites() == ["aeo.VisibilityReport.format"]


def test_converted_parse_citations_records_and_returns_empty():
    assert parse_citations(object()) == []
    assert _sites() == ["aeo.parse_citations"]


def test_converted_find_mentions_records_and_returns_empty():
    assert find_mentions(object(), "Acme", "e", "p") == []
    assert _sites() == ["aeo.find_mentions"]


def test_converted_classify_positioning_records_and_returns_mentioned():
    assert classify_positioning(object()) == "mentioned"
    assert _sites() == ["aeo.classify_positioning"]


def test_converted_sentiment_score_records_and_returns_zero():
    assert sentiment_score(123) == 0.0
    assert _sites() == ["aeo.sentiment_score"]


def test_converted_citation_quality_records_and_returns_partial():
    out = citation_quality([Mention(engine="e", prompt="p", citations=123)])
    assert out == {"own": 0, "third_party": 0, "review_sites": 0, "total": 0}
    assert _sites() == ["aeo.citation_quality"]


def test_converted_sparkline_records_and_returns_empty():
    assert sparkline(["x"]) == ""
    assert _sites() == ["aeo.sparkline"]


def test_converted_gauge_records_and_returns_empty():
    assert _gauge("x") == ""
    assert _sites() == ["aeo._gauge"]


def test_converted_fetch_records_and_returns_unreachable(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    res = site_readiness_check("example.com")
    assert res["domain"] == "example.com"
    assert _sites() == ["aeo.site_readiness_check._fetch",
                        "aeo.site_readiness_check._fetch"]


def test_converted_site_readiness_outer_records_and_returns_partial():
    res = site_readiness_check(object())
    assert res == {"domain": "", "score": 0, "grade": "F",
                   "checks": [], "notes": []}
    assert _sites() == ["aeo.site_readiness_check"]


def test_converted_cross_check_records_and_returns_false():
    m = Mention(engine="e", prompt="p", brand="Acme", context="Acme is great")
    assert cross_check(m, "claude", _boom) is False
    assert m.confirmed is False
    assert _sites() == ["aeo.cross_check"]


def test_converted_default_db_records_and_returns_memory(monkeypatch):
    monkeypatch.setattr("os.path.expanduser", _boom)
    assert aeo_mod._default_db() == ":memory:"
    assert _sites() == ["aeo._default_db"]


def test_converted_tracker_init_records_and_dead_store(tmp_path):
    tracker = AEOTracker(db_path=str(tmp_path))  # a directory, not a file
    assert tracker._db is None
    assert _sites() == ["aeo.AEOTracker.__init__"]


def test_converted_track_caller_records_and_empty_response(tmp_path):
    tracker = _tracker(tmp_path)
    rep = tracker.track_visibility("Acme", ["Is Acme good?"],
                                   engines=["chatgpt"], engine_caller=_boom)
    assert len(rep.responses) == 1 and rep.responses[0].ok is False
    assert rep.gaps == ["Is Acme good?"]
    assert _sites() == ["aeo.AEOTracker.track_visibility.caller"]


def test_converted_track_cross_check_records_and_unconfirmed(tmp_path, monkeypatch):
    # NOTE: a raising *verifier caller* is caught inside cross_check itself
    # (pinned at aeo.cross_check). This inner handler guards the cross_check
    # *function* failing, so the pin patches the function to raise.
    monkeypatch.setattr(aeo_mod, "cross_check", _boom)

    def caller(engine, prompt):
        return "Acme is the best tool around."

    tracker = _tracker(tmp_path)
    rep = tracker.track_visibility("Acme", ["Tell me about Acme."],
                                   engines=["chatgpt"], engine_caller=caller)
    assert len(rep.mentions) == 1 and rep.mentions[0].confirmed is False
    assert _sites() == ["aeo.AEOTracker.track_visibility.cross_check"]


def test_converted_record_records_and_returns_none(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._db = _BrokenDb()
    tracker._record(_report())
    assert _sites() == ["aeo.AEOTracker._record"]


def test_converted_get_gaps_records_and_returns_empty_gaps(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._db.execute(
        "INSERT OR REPLACE INTO aeo_reports "
        "(report_id, brand, share, pairs, confirmed, gaps_json, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        ("aeo_x", "Acme", 0.0, 1, 0, "{bad json", 0.0))
    tracker._db.commit()
    rep = tracker.get("aeo_x")
    assert rep is not None and rep.gaps == []
    assert _sites() == ["aeo.AEOTracker.get.gaps"]


def test_converted_get_outer_records_and_returns_none(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._db = _BrokenDb()
    assert tracker.get("aeo_x") is None
    assert _sites() == ["aeo.AEOTracker.get"]


def test_converted_latest_records_and_returns_none(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._db = _BrokenDb()
    assert tracker.latest("Acme") is None
    assert _sites() == ["aeo.AEOTracker.latest"]


def test_converted_history_records_and_returns_empty(tmp_path):
    tracker = _tracker(tmp_path)
    tracker._db = _BrokenDb()
    assert tracker.history("Acme") == []
    assert _sites() == ["aeo.AEOTracker.history"]


def test_converted_trend_report_records_and_returns_stub(tmp_path, monkeypatch):
    tracker = _tracker(tmp_path)
    monkeypatch.setattr(AEOTracker, "history", _boom)
    assert tracker.trend_report("Acme") == "couldn't build that trend report."
    assert _sites() == ["aeo.AEOTracker.trend_report"]


def test_converted_content_briefs_records_and_returns_empty():
    rep = _report(gaps=123)
    assert AEOTracker(db_path=":memory:").content_briefs(rep) == []
    assert _sites() == ["aeo.AEOTracker.content_briefs"]


def test_converted_estimate_cost_records_and_returns_zero():
    assert estimate_cost(object(), 2) == (0.0, 0.0)
    assert _sites() == ["aeo.estimate_cost"]


class _FakeScheduler:
    def __init__(self, jobs=None, schedule_exc=None):
        self.jobs = jobs if jobs is not None else []
        self.schedule_exc = schedule_exc
        self.scheduled = []

    def list_jobs(self):
        return self.jobs

    async def schedule_cron(self, **kwargs):
        if self.schedule_exc is not None:
            raise self.schedule_exc
        self.scheduled.append(kwargs)
        return None


class _RaisingListJobs(_FakeScheduler):
    def list_jobs(self):
        raise RuntimeError("list gone")


def test_converted_ensure_list_jobs_records_and_still_schedules():
    sched = _RaisingListJobs()
    assert ensure_weekly(sched) is True
    assert len(sched.scheduled) == 1
    assert _sites() == ["aeo.ensure_weekly._ensure.list_jobs"]


def test_converted_ensure_outer_records_and_returns_false():
    sched = _FakeScheduler(schedule_exc=RuntimeError("sched down"))
    assert ensure_weekly(sched) is False
    assert _sites() == ["aeo.ensure_weekly._ensure"]


# ---- Excluded handlers: proven NOT recording -----------------------------

def test_excluded_alter_duplicate_column_records_nothing(tmp_path):
    db = str(tmp_path / "aeo.db")
    AEOTracker(db_path=db)  # creates the table
    AEOTracker(db_path=db)  # ALTER fires duplicate-column, stays quiet
    assert mf.recent_failures() == []


def test_excluded_expand_prompts_records_nothing():
    probes = [object()]
    assert expand_prompts(probes) == probes
    assert mf.recent_failures() == []
