"""Pass #134: send suppression/warmup/blocklist failures stop vanishing.

``nomorals/marketing/send.py`` is a never-raises engine, and its
suppression/warmup/blocklist region returned fail-open shapes on DB
failures with no trace — the sharpest shape in the file, because the
queue/delivery gates consult these very methods before sending:

* ``enqueue``/``process`` refuse blocklisted/suppressed recipients and
  enforce the warmup cap — but a dead DB made ``is_suppressed`` /
  ``is_blocklisted`` answer ``False`` ("send away") and
  ``warmup_status`` answer ``{}`` ("no warmup", so ``_warmup_ok``
  answers ``True`` and the cap never binds). A suppressed address, a
  hard-bounced address, or a mid-ramp provider all looked healthy.
* The writes failed just as quietly: a dead ``unsubscribe`` /
  ``set_warmup`` / ``blocklist`` / ``unblocklist`` returned ``False``,
  indistinguishable from the explicit ``False`` for an invalid
  address or a missing DB — the owner clicking "unsubscribe" or
  reporting a hard bounce got no trace that nothing was recorded.
* ``suppression_count`` answered ``0`` ("nobody suppressed") and
  ``process_bounce`` answered ``kind: unknown`` with nothing
  blocklisted, so a bounce storm looked like noise.

Ten sites in the region now record into the shared marketing ledger
(``nomorals/marketing/failures.py``, reused from #100 — no new
ledger); returns and control flow are unchanged. The record is what
makes the fail-open shape honest: the gate still answers, but the
failure is visible instead of vanishing.

Deliberately NOT in scope here: the template/provider methods, the
module helpers, ``spam_check_template``, ``status``, and
``get_engine`` — the remainder the scan pin below counts (13 sites,
after #135 converted the 5 templates/providers sites
and #137 the 8 module-helper sites).
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import nomorals.marketing.send as send_mod
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
    return SendEngine(db_path=tempfile.mktemp(suffix=".db"))


# ── unsubscribe: a dead opt-out write is not a declined opt-out ───────

def test_unsubscribe_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.unsubscribe("a@example.com") is False
    assert _sites() == ["send.SendEngine.unsubscribe"]


# ── is_suppressed: False on a dead DB is "send away", not "clean" ────

def test_is_suppressed_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.is_suppressed("a@example.com") is False
    assert _sites() == ["send.SendEngine.is_suppressed"]


def test_is_suppressed_benign_path_records_nothing():
    e = _eng()
    assert e.is_suppressed("a@example.com") is False
    assert e.unsubscribe("a@example.com") is True
    assert e.is_suppressed("a@example.com") is True
    assert _sites() == []


# ── suppression_count: 0 on a dead DB is not "nobody suppressed" ─────

def test_suppression_count_db_failure_records_but_still_returns_zero():
    e = _boom_engine()
    assert e.suppression_count() == 0
    assert _sites() == ["send.SendEngine.suppression_count"]


def test_suppression_count_benign_path_records_nothing():
    e = _eng()
    assert e.suppression_count() == 0
    e.unsubscribe("a@example.com")
    assert e.suppression_count() == 1
    assert _sites() == []


# ── set_warmup: a dead ramp write is not a declined ramp ─────────────

def test_set_warmup_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.set_warmup("mock", 100) is False
    assert _sites() == ["send.SendEngine.set_warmup"]


# ── warmup_status: {} on a dead DB is not "no warmup" ────────────────

def test_warmup_status_db_failure_records_but_still_returns_empty():
    e = _boom_engine()
    assert e.warmup_status("mock") == {}
    assert _sites() == ["send.SendEngine.warmup_status"]


def test_warmup_status_benign_path_records_nothing():
    e = _eng()
    assert e.set_warmup("mock", 100) is True
    st = e.warmup_status("mock")
    assert st["provider"] == "mock" and st["cap"] > 0
    assert _sites() == []


# ── _warmup_ok: the cap gate that fails open ──────────────────────────

def test_warmup_ok_db_failure_records_at_source_not_at_gate():
    # warmup_status records at its own site and returns {}; the gate
    # then answers True from that {} — one record, at the source,
    # never a second record at the gate for the same failure.
    e = _boom_engine()
    assert e._warmup_ok("mock") is True
    assert _sites() == ["send.SendEngine.warmup_status"]


def test_warmup_ok_residual_raise_records_but_still_returns_true():
    # warmup_status never raises by contract; if it ever did, the
    # gate's own guard is the record, and the answer stays True.
    e = _eng()
    with patch.object(SendEngine, "warmup_status",
                      side_effect=RuntimeError("warmup-boom")):
        assert e._warmup_ok("mock") is True
    assert _sites() == ["send.SendEngine._warmup_ok"]


# ── process_bounce: an unparsed bounce is not "nothing to block" ─────

def test_process_bounce_parse_failure_records_but_still_returns_unknown():
    e = _eng()
    with patch.object(send_mod, "parse_dsn",
                      side_effect=RuntimeError("dsn-boom")):
        assert e.process_bounce("garbage") == {
            "addresses": [], "kind": "unknown", "blocklisted": []}
    assert _sites() == ["send.SendEngine.process_bounce"]


def test_process_bounce_hard_bounce_benign_path_records_nothing():
    e = _eng()
    out = e.process_bounce(
        "Action: failed\nStatus: 5.1.1\n"
        "Final-Recipient: rfc822; hard@example.com\n")
    assert out["kind"] == "hard"
    assert out["blocklisted"] == ["hard@example.com"]
    assert e.is_blocklisted("hard@example.com") is True
    assert _sites() == []


# ── blocklist: a dead block write is not a declined block ────────────

def test_blocklist_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.blocklist("a@example.com", "hard bounce") is False
    assert _sites() == ["send.SendEngine.blocklist"]


def test_blocklist_benign_path_records_nothing():
    e = _eng()
    assert e.blocklist("a@example.com", "manual") is True
    assert e.is_blocklisted("a@example.com") is True
    assert _sites() == []


# ── unblocklist: a dead unblock is not a declined unblock ────────────

def test_unblocklist_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.unblocklist("a@example.com") is False
    assert _sites() == ["send.SendEngine.unblocklist"]


def test_unblocklist_benign_path_records_nothing():
    e = _eng()
    e.blocklist("a@example.com", "manual")
    assert e.unblocklist("a@example.com") is True
    assert e.is_blocklisted("a@example.com") is False
    assert _sites() == []


# ── is_blocklisted: False on a dead DB is "send away", not "clean" ───

def test_is_blocklisted_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.is_blocklisted("a@example.com") is False
    assert _sites() == ["send.SendEngine.is_blocklisted"]


# ── structural + scanner pins ─────────────────────────────────────────

EXPECTED_SITES = [
    "send.SendEngine.unsubscribe",
    "send.SendEngine.is_suppressed",
    "send.SendEngine.suppression_count",
    "send.SendEngine.set_warmup",
    "send.SendEngine.warmup_status",
    "send.SendEngine._warmup_ok",
    "send.SendEngine.process_bounce",
    "send.SendEngine.blocklist",
    "send.SendEngine.unblocklist",
    "send.SendEngine.is_blocklisted",
]


def test_all_ten_hygiene_sites_record_in_source():
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


def test_scan_send_remainder_is_exactly_13(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 21 true sites after #135 - 8 module-helper converts in #137
    # = 13 remainder: 3 debug-honest (render_template, __init__,
    # process) + 4 pinned exclusions (parse_dsn, _dns_txt
    # socket-continue, _dns_txt parse-continue, deliverability_check
    # inner) + 6 untouched (3 __init__ silent, 1 spam_check_template,
    # 1 status, 1 get_engine).
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 13


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
