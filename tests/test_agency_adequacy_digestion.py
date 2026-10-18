"""Failing-first: gap-resolution adequacy + digestion three-deltas.

A nonempty synthesis must NOT close a gap (audit §3 concrete weakness).
Digestion must emit knowledge / decision / action deltas (audit §4);
action delta NONE is valid and must not manufacture work.
"""
from types import SimpleNamespace

from nomorals.agency.adequacy import assess_gap_adequacy
from nomorals.agency.digestion import Claim, digest


def _finding(url, snippet="support"):
    return SimpleNamespace(title="t", url=url, snippet=snippet)


def _deep(synthesis, findings, subquestions=None):
    return SimpleNamespace(
        synthesis=synthesis, findings=findings,
        conflicts=[], subquestions=subquestions or [])


def test_paragraph_alone_does_not_close_gap():
    gap = {"question": "What is the MT5 tick API?", "context": "{}"}
    deep = _deep("A fluent paragraph with no citations.", [_finding("https://a.example/x")])
    verdict = assess_gap_adequacy(deep, gap)
    assert not verdict.adequate
    assert "sources" in " ".join(verdict.reasons).lower() or verdict.reasons


def test_adequate_when_sources_cover_subquestions_and_task_can_proceed():
    gap = {"question": "MT5 tick API?", "context": '{"subquestions": ["auth", "subscribe"]}'}
    findings = [_finding("https://a.example/auth", "auth token flow"),
                _finding("https://b.example/sub", "subscribe ticks")]
    deep = _deep("auth via token. subscribe ticks. [a] [b]", findings)
    verdict = assess_gap_adequacy(
        deep, gap, proceed_test=lambda d: "subscribe" in d.synthesis.lower())
    assert verdict.adequate, verdict.reasons
    assert verdict.unresolved_subquestions == []


def test_unresolved_subquestions_preserved_and_contradictions_kept():
    gap = {"question": "Q?", "context": '{"subquestions": ["auth", "pricing"]}'}
    findings = [_finding("https://a.example/auth", "auth")]
    deep = _deep("auth only", findings)
    deep.conflicts = [{"claim": "pricing is free vs paid"}]
    verdict = assess_gap_adequacy(deep, gap)
    assert not verdict.adequate
    assert "pricing" in verdict.unresolved_subquestions
    assert verdict.contradictions


def test_digestion_emits_three_deltas_and_none_action_is_valid():
    known = [Claim(text="XAUUSD spread 0.2", source="feed-a",
                   observed_at=1.0, scope="global")]
    deltas = digest("feed-a", [Claim(text="XAUUSD spread 0.2", source="feed-a",
                                       observed_at=2.0, scope="global")],
                    known_claims=known)
    assert deltas.knowledge_delta == []  # duplicate: nothing new
    assert deltas.action_delta is None  # NONE is valid, no manufactured work


def test_digestion_conflict_produces_decision_delta():
    known = [Claim(text="Vendor ships in 2 days", source="vendor",
                   observed_at=1.0, scope="NG")]
    deltas = digest("review-site", [Claim(
        text="Vendor ships in 2 days is false; takes 14 days",
        source="review-site", observed_at=2.0, scope="NG",
        contradicts="Vendor ships in 2 days")], known_claims=known)
    assert deltas.decision_delta, "conflict must change a decision/assumption"
    assert deltas.knowledge_delta
