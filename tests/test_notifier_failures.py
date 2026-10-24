"""Pass #153 pins: the agents failure ledger (REUSED from #152) + notifier conversions.

The agents ledger already exists (``nomorals/agents/failures.py``,
namespace ``agents``, created by #152 for morning_pulse) — this pass reuses
it, it does not create a second one. The 17 durability tests below mirror
#152's shape exactly, pinning the reused ledger's contract; the site pins
prove each conversion records without changing flow (fallback value
unchanged AND a ledger entry under its site name), and each excluded
migration-ladder rung is pinned as NOT recording (the rung fires on every
call on an old DB, but its fallback succeeds — genuine errors surface via
the converted terminal rung).
Every test redirects stores with the app's own ``NM_HOME`` override (never
the real home, never the repo tree). Restart tests use real subprocesses.
"""
import json
import os
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]

from nomorals.core import failure_store as fs
from nomorals.agents import failures as af
from nomorals.agents import notifier as N
from nomorals.agents.notifier import Notifier


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


def _ctx(db=None, partner=None):
    return SimpleNamespace(
        db=db, settings=SimpleNamespace(partner=partner))


class _Txn:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _FakeDB:
    """query/execute/query_one keyed by caller-supplied predicates."""

    def __init__(self, query=None, execute=None, query_one=None):
        self.query_fn = query
        self.execute_fn = execute
        self.query_one_fn = query_one
        self.executed = []

    def transaction(self):
        return _Txn()

    def query(self, sql, params=()):
        if self.query_fn is not None:
            return self.query_fn(sql, params)
        return []

    def query_one(self, sql, params=()):
        if self.query_one_fn is not None:
            return self.query_one_fn(sql, params)
        return None

    def execute(self, sql, params=()):
        self.executed.append(sql)
        if self.execute_fn is not None:
            return self.execute_fn(sql, params)
        return None


def _boom(*args, **kwargs):
    raise RuntimeError("boom")


# ---- Store-level (durability mirror of #152) -----------------------------

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

class _BoomExtras:
    @property
    def extras(self):
        raise RuntimeError("extras gone")


def test_converted_resolve_gateway_records_and_returns_none():
    assert N.resolve_gateway(_BoomExtras()) is None
    assert "notifier.resolve_gateway" in _sites()


def test_converted_quiet_hours_tz_records_and_returns_bool():
    partner = SimpleNamespace(quiet_start=22, quiet_end=8,
                              timezone="Bogus/Zone")
    ctx = SimpleNamespace(settings=SimpleNamespace(partner=partner))
    assert isinstance(N._in_quiet_hours_now(ctx), bool)
    assert "notifier._in_quiet_hours_now.tz" in _sites()


def test_converted_is_duplicate_records_and_returns_false():
    db = _FakeDB(query_one=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n._is_duplicate("news", "t") is False
    assert "notifier._is_duplicate" in _sites()


def test_converted_store_terminal_records_and_keeps_id():
    db = _FakeDB(execute=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    out = n._store("news", "t", "b", 0, nid="n1")
    assert out["id"] == "n1"
    assert "notifier._store.terminal" in _sites()


def test_converted_deliver_buttons_records_and_sends_none(monkeypatch):
    monkeypatch.setattr(
        "nomorals.social.chat.adaptive_buttons.adaptive_buttons_for", _boom)
    n = Notifier(_ctx(), gateway=None)
    out = n._deliver("x", "t", "b", output_kind="scheduled",
                     output_meta={"job": "j"})
    assert out.attempts == []
    assert "notifier._deliver.buttons" in _sites()


def test_converted_deliver_doc_buttons_records(monkeypatch):
    from nomorals.social.chat.present import Doc
    monkeypatch.setattr(
        "nomorals.social.chat.present.actions_rows", _boom)
    n = Notifier(_ctx(), gateway=None)
    out = n._deliver("x", "t", "b", doc=Doc(title="t", summary="s"))
    assert out.attempts == []
    assert "notifier._deliver.doc_buttons" in _sites()


def test_converted_deliver_overflow_records_and_stays_inline(monkeypatch):
    from nomorals.social.chat.present import Doc

    class _RouteBoom:
        def route(self, text):
            raise RuntimeError("router down")

    monkeypatch.setattr(
        "nomorals.social.chat.progress_throttle.LongOutput", _RouteBoom)
    gw = SimpleNamespace(status=lambda: {},
                         send_file=lambda *a, **k: None)
    n = Notifier(_ctx(), gateway=gw)
    out = n._deliver("x", "t", "b",
                     doc=Doc(title="t", summary="s",
                             overflow_text="y" * 100000))
    assert out.attempts == []
    assert _sites() == ["notifier._deliver.overflow"]


def test_converted_deliver_unlink_records_and_still_reports_sent(monkeypatch):
    from nomorals.social.chat.present import Doc

    leaked = []

    def _unlink_boom(path):
        leaked.append(path)
        raise OSError("unlink refused")

    gw = SimpleNamespace(
        status=lambda: {"telegram": {"running_in_session": True}},
        send_file=lambda *a, **k: SimpleNamespace(ok=False),
        send=lambda *a, **k: SimpleNamespace(ok=True, message_id="m1"))
    partner = SimpleNamespace(owner_chats="telegram:123")
    n = Notifier(_ctx(partner=partner), gateway=gw)
    doc = Doc(title="t", summary="s", overflow_text="z" * 100000)
    with patch("os.unlink", side_effect=_unlink_boom):
        out = n._deliver("x", "t", "b", doc=doc)
    assert len(leaked) == 1
    os.unlink(leaked[0])  # the pin proves the leak is recorded, not left
    assert out.attempts[0]["state"] == N.ATTEMPT_SENT
    assert "notifier._deliver.unlink" in _sites()


def test_converted_delivery_attempts_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.delivery_attempts("n1") == []
    assert "notifier.delivery_attempts" in _sites()


def test_converted_termux_probe_records_and_returns_false(monkeypatch):
    monkeypatch.setattr("nomorals.core.profile.is_termux", _boom)
    n = Notifier(_ctx(), gateway=None)
    assert n.termux_fallback_available() is False
    assert "notifier.termux_fallback_available" in _sites()


def test_converted_recent_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.recent() == []
    assert "notifier.recent" in _sites()


def test_converted_pending_fallback_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.pending() == []
    assert _sites() == ["notifier.pending.fallback"]


def test_converted_digest_pending_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.digest_pending() == []
    assert "notifier.digest_pending" in _sites()


def test_converted_digest_unfolded_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.digest_unfolded() == []
    assert "notifier.digest_unfolded" in _sites()


def test_converted_mark_digest_folded_records_and_returns_zero():
    db = _FakeDB(execute=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.mark_digest_folded(["a"]) == 0
    assert "notifier.mark_digest_folded" in _sites()


def test_converted_alert_journal_records_and_returns_empty(monkeypatch):
    monkeypatch.setattr(
        "nomorals.agents.alert_policy.AlertPolicy.journal", _boom)
    n = Notifier(_ctx(db=_FakeDB()), gateway=None)
    assert n.alert_journal() == []
    assert "notifier.alert_journal" in _sites()


def test_converted_dead_letters_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.dead_letters() == []
    assert "notifier.dead_letters" in _sites()


def test_converted_queue_depth_records_and_returns_zeros():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.queue_depth() == {"retryable": 0, "held": 0, "dead": 0}
    assert "notifier.queue_depth" in _sites()


def test_converted_due_redeliveries_fallback_records_and_returns_empty():
    def query(sql, params):
        if "UNION ALL" in sql:
            raise RuntimeError("no backoff cols")
        return []

    db = _FakeDB(query=query)
    n = Notifier(_ctx(db=db), gateway=None)
    # Bad limit: the primary args evaluation raises inside the outer try,
    # then the fallback's own int(limit) raises — the terminal rung fires.
    assert n._due_redeliveries("xx") == []
    assert _sites() == ["notifier._due_redeliveries.fallback"]


def test_converted_redeliver_row_doc_records_and_falls_back_to_string(monkeypatch):
    monkeypatch.setattr(
        "nomorals.social.chat.present.doc_from_json", _boom)
    gw = SimpleNamespace(status=lambda: {})
    n = Notifier(_ctx(db=_FakeDB(), partner=None), gateway=gw)
    row = {"id": "r1", "kind": "news", "title": "t", "body": "b",
           "retry_count": 0, "doc_json": "garbage"}
    assert n._redeliver_row(row, 0.0, consume_attempt=False) is False
    assert _sites() == ["notifier._redeliver_row.doc"]


def test_converted_redeliver_row_state_records_and_returns_true():
    db = _FakeDB(execute=_boom)
    gw = SimpleNamespace(
        status=lambda: {"telegram": {"running_in_session": True}},
        send=lambda *a, **k: SimpleNamespace(ok=True, message_id="m1"))
    partner = SimpleNamespace(owner_chats="telegram:123")
    n = Notifier(_ctx(db=db, partner=partner), gateway=gw)
    row = {"id": "r1", "kind": "news", "title": "t", "body": "b",
           "retry_count": 0}
    # reached=True (live send lands) so both the primary state write and
    # the fallback's delivered=1 write run — and both raise.
    assert n._redeliver_row(row, 0.0, consume_attempt=False) is True
    assert _sites() == ["notifier._redeliver_row.state"]


def test_converted_delivery_counts_fallback_records_and_returns_empty():
    db = _FakeDB(query=_boom)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.delivery_counts() == {}
    assert _sites() == ["notifier.delivery_counts.fallback"]


# ---- Excluded handlers: proven NOT recording -----------------------------

def test_excluded_store_doc_json_rung_does_not_record():
    def execute(sql, params):
        if "doc_json" in sql:
            raise RuntimeError("no doc_json column")

    db = _FakeDB(execute=execute)
    n = Notifier(_ctx(db=db), gateway=None)
    out = n._store("news", "t", "b", 0, nid="n1")
    assert out["id"] == "n1"
    assert _sites() == []


def test_excluded_store_retry_cols_rung_does_not_record():
    def execute(sql, params):
        if "doc_json" in sql or "retry_count" in sql:
            raise RuntimeError("no such column")

    db = _FakeDB(execute=execute)
    n = Notifier(_ctx(db=db), gateway=None)
    out = n._store("news", "t", "b", 0, nid="n1")
    assert out["id"] == "n1"
    assert _sites() == []


def test_excluded_store_delivery_state_rung_does_not_record():
    def execute(sql, params):
        if ("doc_json" in sql or "retry_count" in sql
                or "delivery_state" in sql):
            raise RuntimeError("no such column")

    db = _FakeDB(execute=execute)
    n = Notifier(_ctx(db=db), gateway=None)
    out = n._store("news", "t", "b", 0, nid="n1")
    assert out["id"] == "n1"
    assert _sites() == []


def test_excluded_pending_outer_does_not_record():
    def query(sql, params):
        if "delivery_state" in sql:
            raise RuntimeError("no delivery_state column")
        return [{"id": "p"}]

    db = _FakeDB(query=query)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.pending() == [{"id": "p"}]
    assert _sites() == []


def test_excluded_due_redeliveries_outer_does_not_record():
    def query(sql, params):
        if "UNION ALL" in sql:
            raise RuntimeError("no backoff cols")
        return [{"delivery_state": N.STATE_PENDING}]

    db = _FakeDB(query=query)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n._due_redeliveries(5) == [{"delivery_state": N.STATE_PENDING}]
    assert _sites() == []


def test_excluded_redeliver_row_outer_does_not_record():
    # Primary reached-branch SQL carries "channel"; the fallback's
    # delivered=1 SQL does not — so the primary raises, the fallback
    # succeeds, and nothing records. (With reached=False the fallback
    # executes nothing at all, so the pin drives reached=True via a
    # live send to genuinely exercise outer-then-fallback.)

    def execute(sql, params):
        if "channel" in sql:
            raise RuntimeError("no retry cols")

    db = _FakeDB(execute=execute)
    gw = SimpleNamespace(
        status=lambda: {"telegram": {"running_in_session": True}},
        send=lambda *a, **k: SimpleNamespace(ok=True, message_id="m1"))
    partner = SimpleNamespace(owner_chats="telegram:123")
    n = Notifier(_ctx(db=db, partner=partner), gateway=gw)
    row = {"id": "r1", "kind": "news", "title": "t", "body": "b",
           "retry_count": 0}
    assert n._redeliver_row(row, 0.0, consume_attempt=False) is True
    assert _sites() == []


def test_excluded_delivery_counts_outer_does_not_record():
    def query(sql, params):
        if "GROUP BY state" in sql:
            raise RuntimeError("no delivery_state column")
        return [{"delivered": 1, "n": 1}]

    db = _FakeDB(query=query)
    n = Notifier(_ctx(db=db), gateway=None)
    assert n.delivery_counts() == {"sent": 1}
    assert _sites() == []
