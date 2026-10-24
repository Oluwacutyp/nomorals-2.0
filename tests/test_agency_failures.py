"""Pass #154 pins: the agency failure ledger (NEW) + adapters conversions.

No agency ledger existed before this pass: ``nomorals/agency/failures.py``
(namespace ``agency``) is created here, durability-wired through the shared
``nomorals/core/failure_store.py`` exactly like the #141–#153 ledgers. The
17 durability tests below mirror #153's shape exactly; the 27 site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name). There are 0 exclusions
(adapters.py holds no store, no migration ladder, no routine probes), so
there are 0 exclusion pins — that is the surveyed result, not an omission.
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
from nomorals.agency import failures as af
from nomorals.agency import adapters as A


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


# ---- Store-level (durability mirror of #153) -----------------------------

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


# ---- Converted-site behavior: fallback unchanged + recorded -------------

def test_converted_deals_check_alerts_records_and_returns_empty():
    d = A.DealsAdapter(hunter=SimpleNamespace(check_alerts=_boom))
    assert d.check_alerts() == []
    assert "DealsAdapter.check_alerts" in _sites()


def test_converted_deals_search_records_and_returns_empty():
    d = A.DealsAdapter(hunter=SimpleNamespace(search=_boom))
    assert d.search("q") == []
    assert "DealsAdapter.search" in _sites()


def test_converted_deals_price_detail_records_and_returns_empty():
    d = A.DealsAdapter(hunter=SimpleNamespace(price_detail=_boom))
    assert d.price_detail("p") == {}
    assert "DealsAdapter.price_detail" in _sites()


def test_converted_deals_price_history_records_and_returns_empty():
    d = A.DealsAdapter(hunter=SimpleNamespace(price_history=_boom))
    assert d.price_history("p") == {}
    assert "DealsAdapter.price_history" in _sites()


def test_converted_check_need_detail_records_and_evaluates(monkeypatch):
    d = A.DealsAdapter(hunter=SimpleNamespace())
    monkeypatch.setattr(A.DealsAdapter, "price_detail", _boom)
    out = d.check_need({"product": "p"})
    assert out["observed"]["price"] == 0
    assert "DealsAdapter.check_need.detail" in _sites()


def test_converted_check_need_history_records_and_evaluates(monkeypatch):
    d = A.DealsAdapter(hunter=SimpleNamespace())
    monkeypatch.setattr(A.DealsAdapter, "price_history", _boom)
    out = d.check_need({"product": "p"})
    assert out["observed"]["historical_low"] == float("inf")
    assert "DealsAdapter.check_need.history" in _sites()


def test_converted_check_need_price_check_records_and_evaluates():
    d = A.DealsAdapter(price_check=_boom)
    out = d.check_need({"product": "p"})
    assert out["observed"]["price"] == 0
    assert "DealsAdapter.check_need.price_check" in _sites()


def test_converted_weather_places_provider_records_and_falls_through():
    w = A.WeatherAdapter(weather=None, places_provider=_boom)
    assert w.places() == []
    assert "WeatherAdapter.places.provider" in _sites()


def test_converted_weather_snapshot_records_and_returns_empty():
    w = A.WeatherAdapter(weather=SimpleNamespace(forecast=_boom, now=_boom))
    assert w.snapshot("Lagos") == {"place": "Lagos", "days": [], "alerts": []}
    assert "WeatherAdapter.snapshot" in _sites()


def test_converted_weather_forecast_records_and_returns_not_ok():
    w = A.WeatherAdapter(weather=SimpleNamespace(forecast=_boom))
    assert w.forecast("Lagos") == {"ok": False, "place": "Lagos"}
    assert "WeatherAdapter.forecast" in _sites()


def test_converted_news_recent_records_and_returns_empty():
    n = A.NewsAdapter(agent=SimpleNamespace(recent=_boom))
    assert n.recent() == []
    assert "NewsAdapter.recent" in _sites()


def test_converted_news_fetch_digest_records_and_returns_empty():
    n = A.NewsAdapter(agent=SimpleNamespace(run=_boom))
    assert n.fetch_digest() == {}
    assert "NewsAdapter.fetch_digest" in _sites()


def test_converted_iter_relationships_records_and_returns_empty():
    assert A._iter_relationships(SimpleNamespace(all=_boom)) == []
    assert "_iter_relationships" in _sites()


def test_converted_inbox_unanswered_records_and_counts_zero():
    bad = SimpleNamespace(last_inbound="not-a-number", last_outbound=0)
    inbox = A.default_social_inbox(relationships=[bad])
    out = inbox()
    assert out["unanswered"] == 0
    assert "default_social_inbox.unanswered" in _sites()


def test_converted_inbox_milestones_records_and_counts_zero():
    inbox = A.default_social_inbox(
        relationships=SimpleNamespace(due_for_reconnect=_boom))
    assert inbox()["milestones"] == 0
    assert "default_social_inbox.milestones" in _sites()


def test_converted_inbox_promised_records_and_counts_zero():
    inbox = A.default_social_inbox(
        draft_queue=SimpleNamespace(pending=_boom))
    assert inbox()["promised_replies"] == 0
    assert "default_social_inbox.promised" in _sites()


def test_converted_inbox_triage_records_and_counts_zero():
    manager = SimpleNamespace(
        triage_log=SimpleNamespace(digest=_boom))
    inbox = A.default_social_inbox(manager=manager)
    assert inbox()["unanswered"] == 0
    assert "default_social_inbox.triage" in _sites()


def test_converted_inbox_gateway_records_and_keeps_defaults():
    class _BoomGateway:
        @property
        def stop_signal(self):
            raise RuntimeError("gateway down")

    inbox = A.default_social_inbox(gateway=_BoomGateway())
    out = inbox()
    assert out["stop_signal"] is False
    assert out["cadence_ok"] is True
    assert "default_social_inbox.gateway" in _sites()


def test_converted_social_backlog_records_and_returns_zeroed():
    s = A.SocialAdapter(inbox=_boom)
    assert s.backlog() == {"unanswered": 0, "promised_replies": 0,
                           "milestones": 0, "stop_signal": False,
                           "cadence_ok": True}
    assert "SocialAdapter.backlog" in _sites()


def test_converted_social_due_posts_records_and_returns_empty():
    s = A.SocialAdapter(manager=SimpleNamespace(due=_boom))
    assert s.due_posts() == []
    assert "SocialAdapter.due_posts" in _sites()


def test_converted_social_publish_due_records_and_returns_empty():
    s = A.SocialAdapter(manager=SimpleNamespace(run_due=_boom))
    assert s.publish_due() == []
    assert "SocialAdapter.publish_due" in _sites()


def test_converted_leads_live_leads_records_and_returns_empty():
    la = A.LeadsAdapter(store=SimpleNamespace(list_leads=_boom))
    assert la.live_leads() == []
    assert "LeadsAdapter.live_leads" in _sites()


def test_converted_leads_set_store_status_records_and_returns_false():
    la = A.LeadsAdapter(store=SimpleNamespace(set_lead_status=_boom))
    assert la.set_store_status("l1", "dmed") is False
    assert "LeadsAdapter.set_store_status" in _sites()


def test_converted_gigs_scan_records_and_returns_empty():
    g = A.GigsAdapter(hunter=SimpleNamespace(scan=_boom))
    assert g.scan() == []
    assert "GigsAdapter.scan" in _sites()


def test_converted_gigs_score_payout_records_and_matches_no_extractor():
    g = A.GigsAdapter(payout_extractor=_boom)
    plain = A.GigsAdapter()
    assert g.score({"payout_text": "x"}) == plain.score({"payout_text": "x"})
    assert "GigsAdapter.score.payout" in _sites()


def test_converted_gigs_follow_ups_due_records_and_returns_empty():
    g = A.GigsAdapter(applier=SimpleNamespace(follow_ups_due=_boom))
    assert g.follow_ups_due() == []
    assert "GigsAdapter.follow_ups_due" in _sites()


def test_converted_planner_consume_records_and_keeps_graph():
    class _OnceEq:
        """Equal to anything exactly once, then never again."""

        def __init__(self):
            self._n = 0

        def __eq__(self, other):
            self._n += 1
            return self._n == 1

        def __hash__(self):
            return object.__hash__(self)

    graph = _OnceEq()
    p = A.PlannerAdapter(
        graphs=[object()],
        execute_fn=lambda real, actions: SimpleNamespace(
            completed=["a"], failed=[], attempts=1))
    out = p.execute_graph(graph)
    assert out["completed"] == ["a"]
    assert len(p._graphs) == 1
    assert "PlannerAdapter.execute_graph.consume" in _sites()
