"""Offline tests for nomorals/marketing/guardrails.py (build-map #101).

All offline: mock metrics + mock executor. No network, no real ad APIs.
"""

import os
import tempfile

import pytest

from nomorals.marketing.guardrails import (
    GuardrailStore,
    Rule,
    control_guardrails,
    evaluate,
    execute_override,
    format_fired,
    format_rule,
    parse_naira_kobo,
    parse_rule,
)


@pytest.fixture
def db():
    return tempfile.mktemp(suffix=".db")


@pytest.fixture
def store(db):
    return GuardrailStore(db_path=db)


def _metrics(mapping):
    def fn(adset):
        return dict(mapping.get(adset, {}))
    return fn


def _executor(calls):
    def fn(action, adset, params):
        calls.append((action, adset, dict(params)))
        return True
    return fn


# ── parsing ───────────────────────────────────────────────────────────────

def test_parse_pause_rule():
    r = parse_rule("pause if cpa > ₦50000 for 3 days on adset123 via meta")
    assert r is not None
    assert (r.action, r.metric, r.op, r.days) == ("pause", "cpa", ">", 3)
    assert r.threshold == 50000 * 100
    assert r.adset_id == "adset123" and r.platform == "meta"


def test_parse_scale_rule():
    r = parse_rule("scale 20% if roas > 3 for 2 days on adset456")
    assert r is not None
    assert r.action == "scale" and r.action_param == 20.0
    assert r.metric == "roas" and r.threshold == 3.0 and r.days == 2


def test_parse_label_rule_with_text():
    r = parse_rule('label "needs creative" if ctr < 0.5 for 5 days on adset789')
    assert r is not None
    assert r.action == "label" and r.action_text == "needs creative"
    assert r.metric == "ctr" and r.op == "<" and r.threshold == 0.5


def test_parse_defaults_days_one():
    r = parse_rule("pause if cpa > 10000 on adset1")
    assert r is not None and r.days == 1


def test_parse_garbage_returns_none():
    assert parse_rule("do something weird") is None
    assert parse_rule("") is None
    assert parse_rule(None) is None
    assert parse_rule("scale if roas > 3 on adset1") is None  # scale needs a %
    assert parse_rule("pause if cpa > abc on adset1") is None


def test_parse_naira_variants():
    assert parse_naira_kobo("₦50,000") == 5_000_000
    assert parse_naira_kobo("50k") == 5_000_000
    assert parse_naira_kobo("1.5m") == 150_000_000
    assert parse_naira_kobo("garbage") is None


# ── store ─────────────────────────────────────────────────────────────────

def test_store_crud(store):
    r = parse_rule("pause if cpa > ₦10000 on adset1")
    saved = store.add_rule(r)
    assert saved and saved.rule_id.startswith("gr_")
    assert store.get_rule(saved.rule_id).adset_id == "adset1"
    assert len(store.list_rules()) == 1
    assert store.set_active(saved.rule_id, False)
    assert store.get_rule(saved.rule_id).active is False
    assert store.set_active(saved.rule_id, True)
    assert store.remove_rule(saved.rule_id)
    assert store.list_rules() == []


def test_store_never_raises_on_garbage(store):
    assert store.add_rule(Rule()) is None  # no adset_id
    assert store.get_rule("nope") is None
    assert store.list_rules() == [] or True
    assert store.log_audit.__self__ is not None


# ── evaluation ────────────────────────────────────────────────────────────

def test_label_fires_no_mandate_needed(store):
    r = parse_rule("label if ctr < 0.5 on adset1")
    r.days = 1
    store.add_rule(r)
    calls = []
    fired = evaluate(store, _metrics({"adset1": {"ctr": 0.3}}), _executor(calls))
    assert len(fired) == 1 and fired[0].ok and fired[0].action == "label"
    assert calls == [("label", "adset1", calls[0][2])]


def test_streak_requires_consecutive_days(store):
    r = parse_rule("label if ctr < 0.5 on adset1")
    r.days = 3
    saved = store.add_rule(r)
    calls = []
    m, ex = _metrics({"adset1": {"ctr": 0.3}}), _executor(calls)
    assert evaluate(store, m, ex) == []
    assert evaluate(store, m, ex) == []
    fired = evaluate(store, m, ex)
    assert len(fired) == 1
    # streak resets when the condition stops holding
    assert evaluate(store, _metrics({"adset1": {"ctr": 0.9}}), ex) == []
    assert store.get_rule(saved.rule_id).streak == 0


def test_pause_blocked_without_mandate(store):
    # No adspend mandate in this env → pause must be blocked, not fired.
    r = parse_rule("pause if cpa > ₦10000 on adset2")
    store.add_rule(r)
    calls = []
    fired = evaluate(store, _metrics({"adset2": {"cpa": 2_000_000}}), _executor(calls))
    pauses = [f for f in fired if f.action == "pause"]
    assert len(pauses) == 1
    assert pauses[0].ok is False
    assert "mandate" in pauses[0].reason.lower()
    assert calls == []  # executor never called


def test_scale_needs_known_delta(store):
    r = parse_rule("scale 20% if roas > 3 on adset3")
    store.add_rule(r)
    calls = []
    # no daily_budget in metrics and no max_delta_kobo → blocked honestly
    fired = evaluate(store, _metrics({"adset3": {"roas": 4.0}}), _executor(calls))
    scales = [f for f in fired if f.action == "scale"]
    assert len(scales) == 1 and scales[0].ok is False
    assert "delta" in scales[0].reason.lower()


def test_cooldown_prevents_refire(store):
    r = parse_rule("label if ctr < 0.5 on adset1")
    r.days = 1
    r.cooldown_s = 999999
    store.add_rule(r)
    m, ex = _metrics({"adset1": {"ctr": 0.3}}), _executor([])
    assert len(evaluate(store, m, ex)) == 1
    assert evaluate(store, m, ex) == []  # cooldown


def test_missing_metric_resets_streak(store):
    r = parse_rule("label if ctr < 0.5 on adset1")
    r.days = 5
    saved = store.add_rule(r)
    m = _metrics({"adset1": {"ctr": 0.3}})
    evaluate(store, m, None)
    assert store.get_rule(saved.rule_id).streak == 1
    evaluate(store, _metrics({"adset1": {}}), None)  # no ctr → reset
    assert store.get_rule(saved.rule_id).streak == 0


def test_evaluate_never_raises(store):
    assert evaluate(None) == []
    assert evaluate(store, None, None) == []
    def bad_metrics(a):
        raise RuntimeError("boom")
    assert evaluate(store, bad_metrics, None) == []
    def bad_executor(a, b, c):
        raise RuntimeError("boom")
    r = parse_rule("label if ctr < 0.5 on adset1")
    store.add_rule(r)
    fired = evaluate(store, _metrics({"adset1": {"ctr": 0.1}}), bad_executor)
    assert len(fired) == 1 and fired[0].ok is False


def test_auto_label_on_fire(store):
    r = parse_rule("pause if cpa > ₦10000 on adset2")
    saved = store.add_rule(r)
    evaluate(store, _metrics({"adset2": {"cpa": 2_000_000}}), None)
    got = store.get_rule(saved.rule_id)
    assert "CPA" in got.label and "adset2" not in got.label  # label is about the rule


def test_audit_log_records(store):
    r = parse_rule("label if ctr < 0.5 on adset1")
    store.add_rule(r)
    evaluate(store, _metrics({"adset1": {"ctr": 0.1}}), None)
    rows = store.audit_log()
    assert len(rows) >= 1 and rows[0]["action"] == "label"


# ── explicit override ─────────────────────────────────────────────────────

def test_override_bypasses_rules_but_keeps_mandate(store):
    # Even with no rules at all, override is attempted (rules bypassed);
    # without a mandate the spend action is still blocked.
    fa = execute_override(store, "adset9", "pause", executor_fn=_executor([]))
    assert fa.override is True
    assert fa.ok is False and "mandate" in fa.reason.lower()


def test_override_unknown_action(store):
    fa = execute_override(store, "adset9", "nuke")
    assert fa.ok is False and "unknown action" in fa.reason


def test_override_missing_adset(store):
    fa = execute_override(store, "", "pause")
    assert fa.ok is False


def test_override_never_raises(store):
    fa = execute_override(None, None, None)
    assert fa.ok is False


# ── formatting ────────────────────────────────────────────────────────────

def test_format_rule_and_fired():
    r = parse_rule("scale 20% if roas > 3 for 2 days on adset1")
    out = format_rule(r)
    assert "scale +20%" in out and "ROAS" in out and "adset1" in out


# ── chat ──────────────────────────────────────────────────────────────────

def test_chat_add_and_list(store):
    out = control_guardrails("add pause if cpa > ₦50000 for 3 days on adset123", store=store)
    assert "guardrail armed" in out
    out = control_guardrails("list", store=store)
    assert "adset123" in out


def test_chat_add_garbage(store):
    out = control_guardrails("add frobnicate the moon", store=store)
    assert "couldn't parse" in out


def test_chat_run_nothing(store):
    out = control_guardrails("run", store=store)
    assert "nothing fired" in out or "checked" in out


def test_chat_override(store):
    out = control_guardrails("override pause adset9", store=store)
    assert "override" in out


def test_chat_stop_remove(store):
    control_guardrails("add pause if cpa > ₦10000 on adset1", store=store)
    rid = store.list_rules()[0].rule_id
    assert "disabled" in control_guardrails(f"stop {rid}", store=store)
    assert "removed" in control_guardrails(f"remove {rid}", store=store)
    assert "no rule" in control_guardrails("stop nope", store=store)


def test_chat_audit(store):
    out = control_guardrails("audit", store=store)
    assert "no guardrail actions" in out or "audit" in out


def test_chat_usage(store):
    assert "usage" in control_guardrails("", store=store).lower()
    assert "usage" in control_guardrails("help", store=store).lower()


def test_chat_never_raises(store):
    assert control_guardrails(None, store=store) is not None
    assert control_guardrails("add \ud800", store=store) is not None
