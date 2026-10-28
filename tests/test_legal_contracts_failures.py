"""Tail-30 pins: legal/contracts.py failure values + NOT-recording.

Verdict: pins-only, 0 converts / 9 exclusions, with ONE site flagged for
parent adjudication (945 — see SURVEY.md). Every handler below is the
module's documented never-raises contract in action; each pin drives the
handler into its failure path and pins the exact fallback value plus the
fact that the agents failures ledger is unchanged (snapshot before/after,
tail-8 method). contracts.py calls no record function anywhere
(grep-proven: no logger, no record_* import, no failures import), so the
ledger must not move.

Grounds in brief (full case in SURVEY.md): 175 returns exactly one EMPTY
clause for non-str input — no content fabricated, and no rule can fire on
empty text; 964 fail-CLOSES (grade F, score 0, no findings) — the honest
direction; 1168/1218/1313 return partial results whose present entries are
all genuine matches, with missing entries rendering as absence (the meta
renderer uses an em-dash for missing fields); 1266 returns the input
wording unchanged — the clause itself, never invented text; 1359 returns
visibly-unknown "?" grades with zero scores and no findings; 1385 returns
the disclaimer alone, claiming no comparison at all. 945 pins the
OBSERVED behavior (a raising rule is skipped, the rest of the review
still runs) but is FLAGGED, not endorsed: the dropped finding is silent
(no log, no record, no Review field) and the rendered grade plus "No
clauses flagged against the playbook." is an owner-visible verdict — the
parent decides whether that warrants a convert (e.g. a skipped-rule note)
or stands as the documented "one bad rule never kills a review" tradeoff.

The pins file itself contains no exception handlers, so it adds zero
census sites.
"""

import sys
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.legal import contracts as C

TENANCY = (
    "This agreement is between Alice Ltd and Bob Ltd, "
    "governed by the laws of Nigeria. "
    "The landlord may increase the rent at any time during the tenancy. "
    "The tenant must pay a service charge every month."
)


def _ledger():
    return list(ag_fail.recent_failures())


def test_175_non_str_input_returns_one_empty_clause():
    before = _ledger()
    clauses = C.extract_clauses(12345)
    assert clauses == [C.Clause(heading="", body="")]
    assert _ledger() == before


def test_945_raising_rule_skipped_rest_of_review_runs():
    rule_t02 = next(r for r in C._PLAYBOOKS["tenancy"] if r.rule_id == "T-02")
    before = _ledger()
    with mock.patch.object(rule_t02, "check",
                           side_effect=RuntimeError("boom")) as m:
        rev = C.review_contract(TENANCY, "tenancy")
    assert m.called
    assert "T-01" in [f.rule_id for f in rev.findings]
    assert "T-02" not in [f.rule_id for f in rev.findings]
    assert _ledger() == before


def test_964_outer_failure_fail_closes():
    before = _ledger()
    rev = C.review_contract(12345)
    assert rev.grade == "F" and rev.score == 0
    assert rev.findings == [] and rev.clauses_seen == 0
    assert _ledger() == before


def test_1168_metadata_failure_keeps_filled_fields_defaults_rest():
    class _RaisingSearch:
        def search(self, *a, **k):
            raise RuntimeError("boom")

    before = _ledger()
    with mock.patch.object(C, "_GOVLAW_RE", _RaisingSearch()):
        meta = C.extract_metadata(TENANCY)
    assert len(meta.parties) > 0
    assert meta.governing_law == ""
    assert _ledger() == before


def test_1218_categorize_failure_keeps_good_clause_drops_failed():
    orig_has = C._has

    def _has_marker(text, *patterns):
        if "MARKER" in text:
            raise RuntimeError("boom")
        return orig_has(text, *patterns)

    clauses = [C.Clause("Rent", "The rent is NGN 500000 per year."),
               C.Clause("MARKER", "The tenant must give notice in writing.")]
    before = _ledger()
    with mock.patch.object(C, "_has", _has_marker):
        part = C.categorize_clauses(clauses)
    assert "payment" in part
    assert "notice" not in part
    assert _ledger() == before


def test_1266_plain_english_failure_returns_input_unchanged():
    src = "The party hereinafter referred to as the tenant hereby agrees."
    before = _ledger()
    with mock.patch("re.sub", side_effect=RuntimeError("boom")) as m:
        out = C.plain_english(src)
    assert m.called
    assert out == src
    assert _ledger() == before


def test_1313_rights_failure_keeps_collected_rights():
    text = ("The tenant shall be entitled to a refund of the deposit. "
            "The tenant has the right to renewal of the tenancy.")
    orig_search = C.re.search

    def _search_marker(pattern, string, *a, **k):
        if isinstance(string, str) and "renewal" in string:
            raise RuntimeError("boom")
        return orig_search(pattern, string, *a, **k)

    before = _ledger()
    with mock.patch("re.search", _search_marker):
        rights = C.extract_rights(text)
    assert rights == ["The tenant shall be entitled to a refund of the deposit."]
    assert _ledger() == before


def test_1359_compare_failure_returns_unknown_grades():
    old = C.Review(contract_type="tenancy", score=50, grade="C",
                   findings=None)
    new = C.Review(contract_type="tenancy", score=80, grade="B",
                   findings=[])
    before = _ledger()
    diff = C.compare_reviews(old, new)
    assert diff.old_grade == "?" and diff.new_grade == "?"
    assert diff.old_score == 0 and diff.new_score == 0
    assert diff.new_findings == [] and diff.resolved_findings == []
    assert diff.carried_findings == []
    assert _ledger() == before


def test_1385_format_failure_returns_disclaimer_only():
    diff = C.ReviewDiff(old_grade="C", new_grade="B",
                        old_score=75, new_score=80)
    diff.summary = lambda: 1 / 0
    before = _ledger()
    assert C.format_diff(diff) == C.DISCLAIMER
    assert _ledger() == before
