"""Pass #132: send queue/delivery failures stop vanishing.

``nomorals/marketing/send.py`` is a never-raises engine, and its
queue/delivery pipeline returned safe defaults on DB/gate failures with
no trace: a failed enqueue lost the send ("" looks like "no send"),
a dead rate gate returned True (fail-open: sends flow unthrottled),
a dead cost gate returned True (fail-open: spend flows unbudgeted),
a failed status/log/cost write erased the delivery's trace, and a
corrupt vars payload rendered with {} as if no vars were ever given.
Ten sites in the delivery region now record into the shared marketing
ledger (``nomorals/marketing/failures.py``, reused from #100 — no new
ledger); returns and control flow are unchanged.

Deliberately NOT in scope here: ``process`` (debug-honest,
``_log.debug(..., exc_info=True)``), the template/provider methods,
and the campaign/variant/warmup/suppression/blocklist methods — the
remainder the scan pin below counts (36 sites, after #133 converted
the 11 campaigns/variants sites).
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

import nomorals.tools.error_scan as es
from nomorals.marketing.failures import clear_failures, recent_failures
from nomorals.marketing.send import SendEngine
from nomorals.tools.error_scan import scan


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(1000)]


class _BoomDB:
    """DB handle whose execute/commit always explode."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("boom-db")

    def commit(self, *args, **kwargs):
        raise RuntimeError("boom-commit")


def _boom_engine():
    e = SendEngine.__new__(SendEngine)
    e._db = _BoomDB()
    e._tracker = None
    e._callables = {}
    return e


def _eng():
    e = SendEngine(db_path=tempfile.mktemp(suffix=".db"))
    e.register_provider("mock", kind="mock", rate_per_minute=1000,
                        sender=lambda to, subj, body: True)
    return e


# ── enqueue: a send that never queued must leave a trace ──────────────

def test_enqueue_db_failure_records_but_still_returns_empty():
    e = _boom_engine()
    assert e.enqueue("a@example.com", "hi", {}, "mock") == ""
    sites = _sites()
    assert sites == ["send.SendEngine.enqueue"]
    assert recent_failures()[0].exc_type == "RuntimeError"
    assert "boom-db" in recent_failures()[0].message


# ── queue_depth: 0 on a dead DB falsely claims an empty queue ─────────

def test_queue_depth_db_failure_records_but_still_returns_zero():
    e = _boom_engine()
    assert e.queue_depth() == 0
    assert _sites() == ["send.SendEngine.queue_depth"]


# ── _rate_ok: a crashed gate returning True is fail-open ──────────────

def test_rate_ok_db_failure_records_but_still_returns_true():
    e = _boom_engine()
    assert e._rate_ok("mock", 10) is True
    assert _sites() == ["send.SendEngine._rate_ok"]


# ── _attempt vars: corrupt vars are not "no vars" ─────────────────────

def test_attempt_malformed_vars_records_but_still_sends_with_empty_vars():
    e = _eng()
    e.save_template("hi", "Hello {{ name }}", "Hey")
    row = {"send_id": "snd_test1", "provider": "mock",
           "recipient": "a@example.com", "template": "hi",
           "vars": "{bad json", "subject": "", "variant": ""}
    assert e._attempt(row) == "sent"
    assert _sites() == ["send.SendEngine._attempt.vars"]


def test_attempt_valid_vars_records_nothing():
    e = _eng()
    e.save_template("hi", "Hello {{ name }}", "Hey")
    row = {"send_id": "snd_test2", "provider": "mock",
           "recipient": "a@example.com", "template": "hi",
           "vars": '{"name": "Ada"}', "subject": "", "variant": ""}
    assert e._attempt(row) == "sent"
    assert _sites() == []


# ── _retry_or_dead: an unscheduled retry must not look scheduled ──────

def test_retry_or_dead_db_failure_records_but_still_returns_deferred():
    e = _boom_engine()
    row = {"send_id": "snd_x", "provider": "mock", "attempts": 0}
    assert e._retry_or_dead(row, "provider down") == "deferred"
    assert _sites() == ["send.SendEngine._retry_or_dead"]


# ── _set_status / _log: dropped delivery traces ───────────────────────

def test_set_status_db_failure_records_but_never_raises():
    e = _boom_engine()
    assert e._set_status("snd_x", "sent") is None
    assert _sites() == ["send.SendEngine._set_status"]


def test_log_db_failure_records_but_never_raises():
    e = _boom_engine()
    assert e._log("snd_x", "mock", "sent") is None
    assert _sites() == ["send.SendEngine._log"]


# ── _cost_ok: a crashed budget gate returning True is fail-open ───────

def test_cost_ok_tracker_failure_records_but_still_returns_true():
    e = _boom_engine()
    tracker = Mock()
    tracker.check_budget.side_effect = RuntimeError("tracker-boom")
    e._tracker = tracker
    assert e._cost_ok({"cost_kobo": 100}) is True
    assert _sites() == ["send.SendEngine._cost_ok"]


def test_cost_ok_no_tracker_records_nothing():
    e = _boom_engine()
    assert e._cost_ok({"cost_kobo": 100}) is True
    assert _sites() == []


# ── _record_cost: spend invisible to the budget ───────────────────────

def test_record_cost_tracker_failure_records_but_cost_row_still_lands():
    e = _eng()
    tracker = Mock()
    tracker.track.side_effect = RuntimeError("tracker-boom")
    e._tracker = tracker
    e._record_cost({"name": "mock", "kind": "mock", "cost_kobo": 100})
    assert _sites() == ["send.SendEngine._record_cost.tracker"]
    row = e._db.execute(
        "SELECT COUNT(*) c FROM send_costs WHERE provider = 'mock'"
    ).fetchone()
    assert int(row["c"]) == 1


def test_record_cost_db_failure_records_outer_site_only():
    e = _boom_engine()
    e._record_cost({"name": "mock", "kind": "mock", "cost_kobo": 100})
    assert _sites() == ["send.SendEngine._record_cost"]


# ── no-record controls on the happy path ──────────────────────────────

def test_happy_path_enqueue_and_process_records_nothing():
    e = _eng()
    e.save_template("hi", "Hello {{ name }}", "Hey")
    sid = e.enqueue("a@example.com", "hi", {"name": "Ada"}, "mock")
    assert sid.startswith("snd_")
    assert e.queue_depth() == 1
    stats = e.process()
    assert stats["sent"] == 1
    assert e.queue_depth() == 0
    assert _sites() == []


# ── structural + scanner pins ─────────────────────────────────────────

EXPECTED_SITES = [
    "send.SendEngine.enqueue",
    "send.SendEngine.queue_depth",
    "send.SendEngine._rate_ok",
    "send.SendEngine._attempt.vars",
    "send.SendEngine._retry_or_dead",
    "send.SendEngine._set_status",
    "send.SendEngine._log",
    "send.SendEngine._cost_ok",
    "send.SendEngine._record_cost",
    "send.SendEngine._record_cost.tracker",
]


def test_all_ten_delivery_sites_record_in_source():
    src = Path("nomorals/marketing/send.py").read_text()
    tree = ast.parse(src)
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "record_failure"
                and node.args
                and isinstance(node.args[0], ast.Constant)):
            found.add(node.args[0].value)
    for site in EXPECTED_SITES:
        assert site in found, f"missing record site {site}"


def test_scan_send_remainder_is_exactly_36(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 47 true sites before #133 - 11 campaigns/variants converted = 36
    # remainder: 3 debug-honest (render_template, __init__, process) +
    # 33 untouched (module helpers, templates/providers, warmup,
    # suppression/blocklist, status, get_engine).
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 36


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
