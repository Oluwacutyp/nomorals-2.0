"""Pass #14: digestion freshness enforcement.

Each Claim carries a freshness rule (``freshness_s``) — but ``digest()``
accepted ``now`` and never used it, so a stale known claim blocked fresh
re-observations forever. A fact re-observed after its freshness expired
is confirmation, not a duplicate.
"""
from nomorals.agency.digestion import Claim, digest


def _claim(text, observed_at, freshness_s=86400.0, source="feed-a"):
    return Claim(text=text, source=source, observed_at=observed_at,
                 freshness_s=freshness_s)


def test_stale_known_claim_does_not_block_fresh_reobservation():
    # Known claim observed 30 days ago, 1-day freshness -> stale.
    known = [_claim("XAUUSD spread 0.2", observed_at=0.0)]
    now = 30 * 86400.0
    deltas = digest("feed-a", [_claim("XAUUSD spread 0.2", observed_at=now)],
                    known_claims=known, now=now)
    assert len(deltas.knowledge_delta) == 1, (
        "a stale known claim must not block a fresh re-observation")


def test_fresh_known_claim_still_blocks_duplicate():
    # Known claim observed 1 hour ago, 1-day freshness -> fresh: duplicate.
    known = [_claim("XAUUSD spread 0.2", observed_at=0.0)]
    now = 3600.0
    deltas = digest("feed-a", [_claim("XAUUSD spread 0.2", observed_at=now)],
                    known_claims=known, now=now)
    assert deltas.knowledge_delta == [], (
        "a fresh known claim must still suppress the duplicate")


def test_default_now_zero_keeps_old_behavior():
    # No ``now`` passed (now=0.0): age is not positive -> not stale ->
    # the old skip-everything behavior holds for callers without a clock.
    known = [_claim("XAUUSD spread 0.2", observed_at=1.0)]
    deltas = digest("feed-a", [_claim("XAUUSD spread 0.2", observed_at=2.0)],
                    known_claims=known)
    assert deltas.knowledge_delta == []


def test_stale_check_ignores_malformed_freshness():
    # A claim with no usable freshness rule is never treated as stale.
    known = [_claim("XAUUSD spread 0.2", observed_at=0.0, freshness_s=0.0)]
    deltas = digest("feed-a", [_claim("XAUUSD spread 0.2",
                                       observed_at=10 * 86400.0)],
                    known_claims=known, now=10 * 86400.0)
    assert deltas.knowledge_delta == []
