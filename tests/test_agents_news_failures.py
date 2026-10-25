"""Pass #166 pins: the agents failure ledger (REUSED from #152) + news conversions.

``nomorals/agents/failures.py`` (namespace ``agents``) already exists
from #152 and was reused by #153 and #163; this pass reuses it again with
site names ``news.*`` — no new ledger file. The 17 durability tests
mirror the #141+ shape exactly; the 17 site pins prove each conversion
records without changing flow (fallback value unchanged AND a ledger
entry under its site name); the 4 exclusion pins prove the designed
probe/routine-signal outcome AND no ledger entry. No pin hardcodes scan
totals or news counts. Every test redirects stores with the app's own
``NM_HOME`` override (never the real home, never the repo tree).
Restart tests use real subprocesses. Every trigger was probed live
before its pin was written.
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

from nomorals.agents import failures as af
from nomorals.agents import news as nw
from nomorals.core import failure_store as fs


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("NM_HOME", str(tmp_path / "home"))
    af.clear_failures()
    yield
    af.clear_failures()


def _entries(n, prefix="site"):
    return [{"ts": float(i), "site": f"{prefix}{i}",
             "exc_type": "ValueError", "message": f"boom{i}"} for i in range(n)]


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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
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
    monkeypatch.delenv("NM_HOME", raising=False)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("os.path.expanduser", lambda p: (_ for _ in ()).throw(OSError("no home")))
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


# ---- Site pins ------------------------------------------------------------

RSS_ONE = ("<rss version=\"2.0\"><channel>"
            "<item><title>Gold rallies</title><link>http://example.com/gold</link>"
            "<pubDate>Sat Oct 24 10:00:00 2026</pubDate></item>"
            "</channel></rss>")


class _Tx:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        return self.db

    def __exit__(self, *exc):
        return False


class _RaisingDB:
    """Raises for news_* statements; tolerates the agency_source_health
    CREATE TABLE that SourceHealth runs in poll()'s construction."""

    def query(self, sql, params=None):
        if "news_items" in sql or "news_interest" in sql:
            raise RuntimeError("boom")
        return []

    def query_one(self, sql, params=None):
        if "news_items" in sql or "news_interest" in sql:
            raise RuntimeError("boom")
        return None

    def execute(self, sql, params=None):
        if "agency_source_health" in sql:
            return None
        raise RuntimeError("boom")

    def transaction(self):
        return _Tx(self)


class _SelectiveDB(_RaisingDB):
    """SELECT succeeds (one matching row); UPDATE raises."""

    def query(self, sql, params=None):
        return [{"id": "x", "url": "http://u"}]


def _agent(db=None):
    """NewsAgent with the ledger cleared AFTER construction (the ensure
    helpers run at __init__ and must not pollute the pin's count)."""
    ctx = SimpleNamespace(db=db, settings=None)
    agent = nw.NewsAgent(ctx)
    af.clear_failures()
    return agent


def _count(site):
    return sum(1 for f in af.recent_failures(1000) if f.site == site)


def _item():
    return {"source": "Fake", "source_name": "", "source_url": "",
            "title": "T", "url": "http://u", "summary": "s", "source_ts": 0.0}


def test_pin_known_titles():
    agent = _agent(_RaisingDB())
    agent._feeds = lambda: []
    agent.poll()
    assert _count("news.known_titles") == 1


def test_pin_url_known():
    agent = _agent(_RaisingDB())
    agent._feeds = lambda: [("Fake", "http://fake/rss")]
    agent._client = SimpleNamespace(
        get=lambda url, headers=None: SimpleNamespace(text=RSS_ONE))
    agent.poll()
    assert _count("news.url_known") == 1
    # fallback unchanged: the item was treated as fresh, not dropped
    assert len(agent._pending) == 1


def test_pin_held_count():
    agent = _agent(_RaisingDB())
    agent._feeds = lambda: []
    report = agent.poll()
    assert _count("news.held_count") == 1
    assert report["held"] == 0


def test_pin_store_fresh_legacy():
    agent = _agent(_RaisingDB())
    rows = agent._store_fresh([_item()])
    assert _count("news.store_fresh_legacy") == 1
    # the excluded first attempt recorded nothing: exactly one entry total
    assert af.failure_count() == 1
    assert len(rows) == 1
    assert len(agent._pending) == 1


def test_pin_load_profile():
    agent = _agent(_RaisingDB())
    profile = agent._load_profile()
    assert _count("news.load_profile") == 1
    # fallback unchanged: seeds stand in for the unreadable profile
    assert [b["topic"] for b in profile["beats"]] == [
        t for t, _ in nw.INTEREST_SEED_BEATS]


def test_pin_save_profile():
    agent = _agent(_RaisingDB())
    agent._save_profile({"beats": [], "pins": []})
    assert _count("news.save_profile") == 1


def test_pin_set_state():
    agent = _agent(_RaisingDB())
    row = {"id": "x", "delivery_state": "undelivered"}
    agent._set_state(row, "pushed")
    assert _count("news.set_state") == 1
    # in-memory state still advances; only persistence was lost
    assert row["delivery_state"] == "pushed"


def test_pin_pushes_today():
    agent = _agent(_RaisingDB())
    assert agent._pushes_today() == 0
    assert _count("news.pushes_today") == 1


def test_pin_quiet_now(monkeypatch):
    agent = _agent(_RaisingDB())
    monkeypatch.setattr(
        "nomorals.agents.notifier._in_quiet_hours_now", _boom)
    assert agent._quiet_now() is False
    assert _count("news.quiet_now") == 1


def test_pin_score_tier_backlog():
    agent = _agent(_RaisingDB())
    out = agent._score_and_tier([])
    assert _count("news.score_tier_backlog") == 1
    assert out["scored"] == 0


def test_pin_score_tier_update():
    agent = _agent(_RaisingDB())
    agent._score_interest = lambda rows: ({"x": 0.9}, "ok")
    row = {"id": "x", "title": "T", "url": "http://u", "summary": "s",
           "delivery_state": "undelivered", "interest": None, "created_at": 0}
    agent._score_and_tier([row])
    assert _count("news.score_tier_update") == 1
    # in-memory score still lands; only persistence was lost
    assert row["interest"] == 0.9


def test_pin_score_tier_publish():
    agent = _agent(None)
    agent._score_interest = lambda rows: ({"x": 0.9}, "ok")
    agent._quiet_now = lambda: False
    agent._push_cap = lambda: 6
    agent._thresholds = lambda: (0.75, 0.35)
    agent.notifier = SimpleNamespace(publish=_boom)
    row = {"id": "x", "title": "T", "url": "http://u", "summary": "s",
           "delivery_state": "undelivered", "interest": None, "created_at": 0}
    out = agent._score_and_tier([row])
    assert _count("news.score_tier_publish") == 1
    assert out["pushed"] == 0
    assert row["delivery_state"] == "undelivered"


def test_pin_undelivered():
    agent = _agent(_RaisingDB())
    assert agent.undelivered() == []
    assert _count("news.undelivered") == 1


def test_pin_mark_delivered_select():
    agent = _agent(_RaisingDB())
    assert agent.mark_delivered(["http://u"]) == 0
    assert _count("news.mark_delivered_select") == 1


def test_pin_mark_delivered_update():
    agent = _agent(_SelectiveDB())
    assert agent.mark_delivered(["http://u"]) == 0
    assert _count("news.mark_delivered_update") == 1


def test_pin_status():
    agent = _agent(_RaisingDB())
    result = agent.status()
    assert _count("news.status") == 1
    assert result["by_state"] == {}


def test_pin_recent():
    agent = _agent(_RaisingDB())
    assert agent.recent() == []
    assert _count("news.recent") == 1


# ---- Exclusion pins (prove NO recording + designed outcome) ---------------

def test_excl_parse_feed_date():
    bad_date = ("<rss version=\"2.0\"><channel>"
                "<item><title>T</title><link>http://u</link>"
                "<pubDate>not a date at all</pubDate></item>"
                "</channel></rss>")
    items = nw.parse_feed(bad_date)
    assert len(items) == 1
    assert items[0]["source_ts"] == 0.0
    assert af.failure_count() == 0


def test_excl_ensure_delivery_columns():
    class _EnsureDB:
        def execute(self, sql, params=None):
            if sql.strip().upper().startswith("ALTER"):
                raise Exception("duplicate column name: delivery_state")
            raise RuntimeError("boom")

    assert nw._ensure_delivery_columns(_EnsureDB()) is None
    assert af.failure_count() == 0


def test_excl_model_summaries(monkeypatch):
    ctx = SimpleNamespace(db=None, settings=None, router=object())
    agent = nw.NewsAgent(ctx)
    af.clear_failures()
    monkeypatch.setattr(nw, "brain_for", _boom)
    assert agent._model_summaries(
        [{"title": "T", "summary": "s", "url": "http://u"}]) == {}
    assert af.failure_count() == 0


def test_excl_store_fresh_ladder():
    class _LadderDB:
        def execute(self, sql, params=None):
            if "delivery_state" in sql:
                raise RuntimeError("no such column")
            return None

        def transaction(self):
            return _Tx(self)

    agent = _agent(_LadderDB())
    rows = agent._store_fresh([_item()])
    assert len(rows) == 1
    # the expected ladder rung recorded nothing
    assert af.failure_count() == 0
