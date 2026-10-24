"""Pass #100: marketing failures stop vanishing.

guardrails + competitor are never-raises modules by design — but until
this pass every caught failure vanished without a trace. The failure
ledger (nomorals/marketing/failures.py) records each one; these tests
pin the recording at the sites that matter, the never-raises contract,
the ledger bound, the chat surface, and the scanner verdict.
"""

import tempfile
from unittest.mock import Mock

import pytest

import nomorals.marketing.competitor as competitor_mod
import nomorals.marketing.guardrails as guardrails_mod
from nomorals.marketing.competitor import CompetitorStore, control_competitor
from nomorals.marketing.failures import (
    clear_failures, failure_count, recent_failures, record_failure,
)
from nomorals.marketing.guardrails import (
    GuardrailStore, Rule, control_guardrails, evaluate, preview,
)
from nomorals.tools.error_scan import scan


@pytest.fixture(autouse=True)
def _clean_ledger():
    clear_failures()
    yield
    clear_failures()


def _sites():
    return [f.site for f in recent_failures(1000)]


class _BoomDB:
    """DB handle whose execute always explodes."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("boom-db")

    def commit(self):
        raise RuntimeError("boom-commit")


def test_store_db_failure_recorded_and_safe_default_kept():
    store = GuardrailStore.__new__(GuardrailStore)
    store._db = _BoomDB()
    assert store.add_rule(Rule(adset_id="a1")) is None
    entries = recent_failures()
    assert len(entries) == 1
    assert entries[0].site == "guardrails.GuardrailStore.add_rule"
    assert entries[0].exc_type == "RuntimeError"
    assert "boom-db" in entries[0].message


def test_evaluate_bad_rule_recorded_and_pass_continues():
    store = GuardrailStore(db_path=tempfile.mktemp(suffix=".db"))
    good = Rule(adset_id="good", action="alert", metric="cpa", op=">",
                threshold=1.0, days=1)
    store.add_rule(good)

    def _raising_rule(*args, **kwargs):
        raise ValueError("rule exploded")

    monkey_rules = [good, Mock()]
    orig_list = store.list_rules
    store.list_rules = lambda active_only=False: monkey_rules  # noqa: ARG005
    orig_eval_rule = guardrails_mod._evaluate_rule
    guardrails_mod._evaluate_rule = (
        lambda s, r, m, e: (_raising_rule() if isinstance(r, Mock) else orig_eval_rule(s, r, m, e))
    )
    try:
        fired = evaluate(store, metrics_fn=lambda adset: {"cpa": 999999.0})
    finally:
        guardrails_mod._evaluate_rule = orig_eval_rule
        store.list_rules = orig_list
    # The good rule still fired; the bad rule was recorded, not fatal.
    assert any(fa.adset_id == "good" for fa in fired)
    assert any(s.startswith("guardrails.evaluate:") for s in _sites())


def test_preview_bad_rule_recorded_and_skipped():
    store = GuardrailStore(db_path=tempfile.mktemp(suffix=".db"))
    good = Rule(adset_id="good", action="alert", metric="cpa", op=">",
                threshold=1.0, days=1)
    store.add_rule(good)
    # Real conditions (so _conditions_of returns them) + a Mock streak
    # (so the streak math raises inside preview's per-rule handler).
    bad = Mock()
    bad.rule_id = "bad-rule"
    bad.conditions = [{"metric": "cpa", "op": ">", "threshold": 1.0}]
    store.list_rules = lambda active_only=False: [good, bad]  # noqa: ARG005
    out = preview(store, metrics_fn=lambda adset: {"cpa": 999999.0})
    assert any(info.get("rule") is good or "good" in str(info) for info in out)
    assert "guardrails.preview:bad-rule" in _sites()


def test_competitor_scrape_failure_recorded_and_empty_report_kept():
    store = CompetitorStore(db_path=tempfile.mktemp(suffix=".db"))

    def _boom(*args, **kwargs):
        raise ConnectionError("scrape down")

    report = store.analyze("rival", scrape_fn=_boom)
    assert report.post_count == 0
    assert "competitor.CompetitorStore.analyze" in _sites()


def test_outer_chat_catch_recorded():
    def _boom_store(**kwargs):
        raise OSError("store exploded")

    orig_g = guardrails_mod._get_store
    orig_c = competitor_mod._get_store
    guardrails_mod._get_store = _boom_store
    competitor_mod._get_store = _boom_store
    try:
        assert control_guardrails("list") == "guardrails hit a snag — try again."
        assert control_competitor("list") == "competitor intel hit a snag — try again."
    finally:
        guardrails_mod._get_store = orig_g
        competitor_mod._get_store = orig_c
    assert "guardrails.control_guardrails" in _sites()
    assert "competitor.control_competitor" in _sites()


def test_ledger_bounded_and_counted():
    for i in range(250):
        record_failure(f"test.site{i}", ValueError(f"v{i}"))
    assert len(recent_failures(1000)) == 200
    assert failure_count() == 250
    newest = recent_failures(1)[0]
    assert newest.site == "test.site249"


def test_chat_failures_renders_and_empty_state():
    store = GuardrailStore(db_path=tempfile.mktemp(suffix=".db"))
    assert control_guardrails("failures", store=store) == "no marketing failures recorded."
    record_failure("guardrails.preview", RuntimeError("metrics exploded"))
    out = control_guardrails("failures", store=store)
    assert "guardrails.preview" in out and "RuntimeError" in out
    assert "metrics exploded" in out
    out_c = control_competitor("failures", store=CompetitorStore(db_path=tempfile.mktemp(suffix=".db")))
    assert "guardrails.preview" in out_c  # shared ledger, both surfaces read it


def test_error_scan_marketing_clean():
    rep = scan(["nomorals/marketing/guardrails.py",
                "nomorals/marketing/competitor.py",
                "nomorals/marketing/failures.py"])
    assert rep.findings == []
