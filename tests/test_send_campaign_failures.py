"""Pass #133: send campaigns/variants failures stop vanishing.

``nomorals/marketing/send.py`` is a never-raises engine, and its
campaigns/variants region returned empty shapes on DB failures with
no trace: a dead estimate priced a campaign at ₦0, a failed variant
write looked like an invalid variant, an empty variant list looked
like "no variants yet", a dead A/B funnel looked like "no data", a
failed engagement query silently picked the winner by fail rate
instead of engagement, and a crashed campaign enqueue lost the
campaign_id that would have tracked the sends that did land.
Eleven sites in the campaigns/variants region now record into the
shared marketing ledger (``nomorals/marketing/failures.py``, reused
from #100 — no new ledger); returns and control flow are unchanged.

Deliberately NOT in scope here: the template/provider methods, the
warmup/suppression/blocklist methods, and the module helpers — the
remainder the scan pin below counts (4 sites after #134 converted
the 10 warmup/suppression/blocklist sites, #135 the 5
templates/providers sites, #137 the 8 module-helper sites,
#138 the 3 tail sites, #139 the 3 __init__ sites, and #140
recognizes the 3 debug-honest sites via exc_info=True).
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

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
    e.save_template("hi", "Hello {{ name }}", "Hey")
    return e


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _SelectiveDB:
    """A/B funnel rows land; the engagement query explodes."""

    def execute(self, sql, *args, **kwargs):
        if "send_log" in sql:
            raise RuntimeError("eng-boom")
        return _Rows([
            {"variant": "a", "status": "sent", "c": 10},
            {"variant": "b", "status": "sent", "c": 5},
            {"variant": "b", "status": "failed", "c": 5},
        ])

    def commit(self, *args, **kwargs):
        pass


def _selective_engine():
    e = SendEngine.__new__(SendEngine)
    e._db = _SelectiveDB()
    e._tracker = None
    e._callables = {}
    return e


# ── estimate_campaign: a dead price lookup is not a ₦0 campaign ───────

def test_estimate_campaign_db_failure_records_but_still_returns_zeros():
    e = _boom_engine()
    assert e.estimate_campaign(100, "mock") == {
        "count": 0, "per_kobo": 0, "total_kobo": 0, "text": ""}
    assert _sites() == ["send.SendEngine.estimate_campaign"]


# ── save_variant: a dead DB is not an invalid variant ────────────────

def test_save_variant_db_failure_records_but_still_returns_false():
    e = _boom_engine()
    assert e.save_variant("hi", "a", "body") is False
    assert _sites() == ["send.SendEngine.save_variant"]


# ── list_variants: an empty list is not "no variants yet" ────────────

def test_list_variants_db_failure_records_but_still_returns_empty():
    e = _boom_engine()
    assert e.list_variants("hi") == []
    assert _sites() == ["send.SendEngine.list_variants"]


# ── get_variant: None is not "no such variant" ───────────────────────

def test_get_variant_db_failure_records_but_still_returns_none():
    e = _boom_engine()
    assert e.get_variant("hi", "a") is None
    assert _sites() == ["send.SendEngine.get_variant"]


# ── create_ab_campaign: a crash mid-split must not lose the trace ────

def test_create_ab_campaign_vars_failure_records_but_still_returns_empty():
    e = _boom_engine()

    def _boom_vars(recip, variant):
        raise RuntimeError("vars-boom")

    assert e.create_ab_campaign("c", "hi", ["a", "b"],
                                ["x@example.com"], "mock",
                                vars_fn=_boom_vars) == {
        "campaign_id": "", "per_variant": {}}
    assert _sites() == ["send.SendEngine.create_ab_campaign"]


# ── ab_report: an empty funnel is not "no data" ──────────────────────

def test_ab_report_db_failure_records_but_still_returns_empty_funnel():
    e = _boom_engine()
    assert e.ab_report("cmp_x") == {"campaign_id": "cmp_x", "variants": {}}
    assert _sites() == ["send.SendEngine.ab_report"]


# ── ab_winner.engagement: a lost signal must not silently re-pick ────

def test_ab_winner_engagement_failure_records_but_still_picks_by_fail_rate():
    e = _selective_engine()
    out = e.ab_winner("cmp_x")
    # Engagement lost, so the fail-rate fallback picks "a" (0% vs 50%) —
    # but the loss itself is recorded, not silently absorbed.
    assert out["winner"] == "a"
    assert out["by"] == "fail_rate"
    assert _sites() == ["send.SendEngine.ab_winner.engagement"]


# ── ab_winner outer: "unavailable" must carry its cause ──────────────

def test_ab_winner_db_failure_records_but_still_returns_unavailable():
    # ab_report itself never raises (it records its own failure and
    # returns an empty funnel, which ab_winner reads as "no data") —
    # so the outer handler fires only when the report step itself
    # breaks, e.g. a patched/broken ab_report.
    e = _eng()
    with patch.object(e, "ab_report", side_effect=RuntimeError("report-boom")):
        assert e.ab_winner("cmp_x") == {"winner": "", "reason": "unavailable"}
    assert _sites() == ["send.SendEngine.ab_winner"]


# ── schedule_campaign: a crash must not lose the campaign trace ──────

def test_schedule_campaign_vars_failure_records_but_still_returns_empty():
    e = _boom_engine()

    def _boom_vars(recip):
        raise RuntimeError("vars-boom")

    assert e.schedule_campaign("hi", ["x@example.com"], "mock", 9999.0,
                               vars_fn=_boom_vars) == {
        "campaign_id": "", "queued": 0, "send_at": 0.0}
    assert _sites() == ["send.SendEngine.schedule_campaign"]


# ── send_seed: a failed seed run is not "0 of 0" ─────────────────────

def test_send_seed_enqueue_failure_records_but_still_returns_zeros():
    e = _eng()
    with patch.object(e, "enqueue", side_effect=RuntimeError("seed-boom")):
        assert e.send_seed("hi", ["s@example.com"], "mock") == {
            "seeded": 0, "of": 0}
    assert _sites() == ["send.SendEngine.send_seed"]


# ── campaign_report: a dead funnel is not an empty campaign ──────────

def test_campaign_report_db_failure_records_but_still_returns_empty():
    e = _boom_engine()
    assert e.campaign_report("cmp_x") == {
        "campaign_id": "cmp_x", "total": 0,
        "by_status": {}, "by_provider": {}, "spent_kobo": 0}
    assert _sites() == ["send.SendEngine.campaign_report"]


# ── no-record controls on the happy path ──────────────────────────────

def test_happy_path_campaigns_record_nothing():
    e = _eng()
    assert e.estimate_campaign(10, "mock")["count"] == 10
    assert e.save_variant("hi", "a", "Hello A") is True
    assert [v["variant_name"] for v in e.list_variants("hi")] == ["a"]
    assert e.get_variant("hi", "a")["body"] == "Hello A"
    camp = e.create_ab_campaign("c", "hi", ["a"], ["x@example.com"],
                                "mock")
    assert camp["per_variant"] == {"a": 1}
    assert e.ab_report(camp["campaign_id"])["variants"] != {}
    assert e.ab_winner(camp["campaign_id"])["winner"] == "a"
    sched = e.schedule_campaign("hi", ["y@example.com"], "mock", 9999.0)
    assert sched["queued"] == 1
    assert e.send_seed("hi", ["s@example.com"], "mock") == {
        "seeded": 1, "of": 1}
    assert e.campaign_report(camp["campaign_id"])["total"] == 1
    assert _sites() == []


# ── structural + scanner pins ─────────────────────────────────────────

EXPECTED_SITES = [
    "send.SendEngine.estimate_campaign",
    "send.SendEngine.save_variant",
    "send.SendEngine.list_variants",
    "send.SendEngine.get_variant",
    "send.SendEngine.create_ab_campaign",
    "send.SendEngine.ab_report",
    "send.SendEngine.ab_winner",
    "send.SendEngine.ab_winner.engagement",
    "send.SendEngine.schedule_campaign",
    "send.SendEngine.send_seed",
    "send.SendEngine.campaign_report",
]


def test_all_eleven_campaign_sites_record_in_source():
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


def test_scan_send_remainder_is_exactly_4(monkeypatch):
    # Suppression-off: noqa comments ignored, so converted sites (which
    # now bind and pass `e` to record_failure) drop out of the findings.
    # 10 remainder after #138 - 3 __init__ converts in #139
    # = 7 remainder: 3 debug-honest (render_template, __init__ DB
    # handler, process) + 4 pinned exclusions (parse_dsn, _dns_txt
    # socket-continue, _dns_txt parse-continue, deliverability_check
    # inner) + 0 untouched sites: send.py is closed except for
    # pinned judgments.
    # #140: the scanner now recognizes exc_info=True logging, so the
    # 3 debug-honest sites no longer count: 7 - 3 = 4 remainder =
    # 4 pinned exclusions + 0 untouched + 0 debug-honest-counted.
    monkeypatch.setattr(es, "_noqa_suppresses", lambda line, rule: False)
    rep = scan(["nomorals/marketing/send.py"])
    assert len(rep.findings) == 4


def test_scan_send_flagged_is_clean():
    # Default scan respects noqa; converted sites also use `e` now.
    rep = scan(["nomorals/marketing/send.py"])
    assert rep.findings == []
