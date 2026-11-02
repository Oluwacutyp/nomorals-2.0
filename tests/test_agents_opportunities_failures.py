"""Tail-97 pins: nomorals/agents/opportunities.py census sites.

All four sites are held as exclusions — every failure value points
the safe direction for a money-opportunity hunter whose outputs feed
dedupe, ranking, and a preference filter, never a verdict:

190 — ``normalize_url``: urlsplit crash -> the raw stripped/lowered
URL as the dedupe key. A malformed URL can at worst MISS a dedupe
match (a duplicate survives); the fallback can never merge two
different opportunities into one identity.
283 — ``score_opportunity``: host-parse crash -> the TRUSTED_DOMAINS
+8 bonus is simply not awarded. The bonus can be lost, never falsely
gained; a malformed URL under-scores, which is the safe direction
for a ranking the owner reads.
667 — ``load_seen``: a corrupt JSONL line is skipped, valid lines
still load. Worst case a previously-seen opportunity resurfaces as
new (a duplicate in the list); the file itself is never rewritten
by the reader.
680 — ``load_profile``: corrupt profile.json -> factory-default
OpportunityProfile for the session. The stored file is NOT
overwritten (``save_profile`` has no automatic caller — the fallback
persists nothing), so the user can repair the file and the next load
honors it. WEAKEST EXCLUSION of this batch: defaults widen
``accepts()`` (all kinds, global region, no payout floor) versus
whatever the user had configured, so a corrupt file silently relaxes
the filter for the session. Held because the corrupt contents are
unrecoverable by definition, the defaults are the module's own
documented starting state, and raising would kill every scan until
the JSON is hand-fixed — but a parent may reasonably want a visible
warning surfaced on this path (shelved sketch territory, not a
batch convert).

NOT-recording is pinned with the snapshot method: opportunities.py
calls no failures-ledger record function anywhere, so both the
agency and agents ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

import json
import time

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.agents import opportunities as O


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _hunter(tmp_path):
    h = O.OpportunityHunter()
    h._data_dir = tmp_path
    return h


# ── 190: malformed URL -> raw lowered fallback key ──────────────

def test_190_malformed_url_falls_back_to_raw_lowered():
    before = _snap()
    # urlsplit raises ValueError on the unmatched IPv6 bracket; the
    # fallback is the raw URL, stripped and lowered — never a
    # canonical form borrowed from some other URL.
    assert O.normalize_url("http://[::1") == "http://[::1"
    # control: a well-formed URL canonicalizes (https forced, www
    # stripped, tracking params removed).
    out = O.normalize_url("HTTPS://www.Upwork.COM/jobs/?utm_source=x&b=1")
    assert out == "https://upwork.com/jobs?b=1"
    assert _snap() == before


# ── 283: unparseable host -> trust bonus withheld, never granted ─

def _opp(url: str) -> O.Opportunity:
    return O.Opportunity(
        kind="gig", title="t", source="s", url=url, payout_text="",
        effort="medium", fetched_at=time.time() - 25 * 3600,
        is_new=False)


def test_283_unparseable_url_gets_no_trust_bonus():
    before = _snap()
    # gig base 55.0 x medium 1.0, no payout, freshness exhausted:
    # the only variable left is the +8 trust bonus.
    assert O.score_opportunity(_opp("http://[::1")) == 55.0
    # control halves: an ordinary untrusted host also scores 55.0,
    # and a trusted host earns exactly +8.
    assert O.score_opportunity(_opp("https://example.com/x")) == 55.0
    assert O.score_opportunity(_opp("https://www.upwork.com/x")) == 63.0
    assert _snap() == before


# ── 667: corrupt seen line skipped, valid lines still load ──────

def test_667_corrupt_seen_line_is_skipped(tmp_path):
    before = _snap()
    h = _hunter(tmp_path)
    a = O.Opportunity(kind="gig", title="A", source="s",
                      url="https://example.com/a")
    b = O.Opportunity(kind="gig", title="B", source="s",
                      url="https://example.com/b")
    (tmp_path / "seen.jsonl").write_text(
        json.dumps(a.to_dict()) + "\n{corrupt line\n"
        + json.dumps(b.to_dict()) + "\n", encoding="utf-8")
    seen = h.load_seen()
    assert set(seen) == {a.id, b.id}
    assert _snap() == before


# ── 680: corrupt profile -> session defaults, file untouched ────

def test_680_corrupt_profile_falls_back_to_defaults(tmp_path):
    before = _snap()
    h = _hunter(tmp_path)
    (tmp_path / "profile.json").write_text("{not json", encoding="utf-8")
    prof = h.load_profile()
    assert prof.to_dict() == O.OpportunityProfile().to_dict()
    # the fallback persists nothing: the corrupt file survives
    # byte-for-byte, so repairing it restores the user's profile.
    assert (tmp_path / "profile.json").read_text(
        encoding="utf-8") == "{not json"
    # control: a valid stored profile loads as stored.
    (tmp_path / "profile.json").write_text(
        json.dumps({"min_payout_usd": 25.0}), encoding="utf-8")
    assert h.load_profile().min_payout_usd == 25.0
    assert _snap() == before
