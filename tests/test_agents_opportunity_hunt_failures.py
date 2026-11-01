"""Tail-84 pins for nomorals/agents/opportunity_hunt.py — 5 exclusions.

Each test pins one census site's fallback behavior plus a control half
proving the except path was taken. The file itself contains no
exception handlers, so it adds zero census sites.

opportunity_hunt.py calls no failures-ledger record function anywhere
(grep-proven: only _log), so NOT-recording is pinned with the snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

The module's vetting contract is documented fail-closed (the
verify_candidate docstring): a candidate passes only on evidence found
in fetched page text; every failure kind — unverifiable or excluded —
drops the candidate. None of the five sites can turn a failure into a
"verified" verdict.

Verdict notes (full grounds in SURVEY.md):
- 152: platform_of failure -> "" identity. urlsplit raises only on
  malformed input (e.g. an unmatched IPv6 bracket). _same_domain
  fails closed on "" (`if not a or not b: return False`), so a
  malformed candidate can never borrow another domain's payout docs.
- 269: HTMLParser.feed failure -> keep the links collected before the
  failure ("malformed HTML: what parsed"). Partial evidence only ever
  shrinks the doc set the vetting can find — fail-closed direction.
- 328: candidate fetch failure -> text "" -> verdict stays
  ok=False / kind="unverifiable", reason "source page unreadable or
  too thin to vet", fetches still counted against the budget.
- 373: payout_doc_links failure -> doc_urls [] -> no docs checked;
  verdict unverifiable with the "no payout doc found" reason — the
  same words as the genuine no-docs case; the decision (drop) is the
  honest one either way.
- 380: payout doc fetch failure -> doc_text "" -> that doc yields no
  rail evidence; verdict unverifiable, reason distinguishes "payout
  doc found, no qualifying rail" because the doc WAS checked.
"""

from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
from nomorals.agents import opportunity_hunt as oh


def _snap():
    return list(ag_fail.recent_failures(limit=200))


_PAGE = ("This programme is open to applicants worldwide and pays "
         "contributors monthly for editorial work. " * 6)


def _opp(url="https://platform.example.com/job"):
    return SimpleNamespace(url=url, id="x1", title="Gig", payout_text="")


def test_152_platform_of_failure_is_empty_identity():
    before = _snap()
    assert oh.platform_of("http://[::1") == ""
    # the empty identity can never satisfy a same-domain check
    assert oh._same_domain("http://[::1", "https://example.com") is False
    # control: a normal URL resolves its bare host
    assert oh.platform_of("https://www.Example.com/x") == "example.com"
    assert _snap() == before


def test_269_anchor_parse_keeps_links_before_failure():
    before = _snap()
    parser = oh._AnchorParser()
    orig_feed = parser._parser.feed

    def flaky_feed(markup):
        orig_feed('<a href="/a">Alpha</a>')
        raise RuntimeError("parser boom")

    parser._parser.feed = flaky_feed
    got = parser.parse("ignored", "https://example.com/base/")
    assert got == [("https://example.com/a", "Alpha")]
    assert _snap() == before


def test_328_verify_fetch_failure_is_unverifiable():
    before = _snap()

    def boom_fetch(url):
        raise RuntimeError("fetch boom")

    v = oh.verify_candidate(_opp(), boom_fetch)
    assert v["ok"] is False and v["kind"] == "unverifiable"
    assert v["reason"] == "source page unreadable or too thin to vet"
    assert v["fetches"] == 1
    assert _snap() == before


def test_373_doc_links_failure_checks_no_docs(monkeypatch):
    before = _snap()

    def boom_links(url, links):
        raise RuntimeError("links boom")

    monkeypatch.setattr(oh, "payout_doc_links", boom_links)
    v = oh.verify_candidate(
        _opp(),
        lambda url: _PAGE,
        lambda url: [("https://platform.example.com/payouts", "payout")],
    )
    assert v["ok"] is False and v["kind"] == "unverifiable"
    assert v["payout_docs_checked"] == []
    assert "no payout doc found" in v["reason"]
    assert _snap() == before


def test_380_doc_fetch_failure_is_no_evidence():
    before = _snap()
    doc = "https://platform.example.com/payouts"

    def fetch_text(url):
        if url == doc:
            raise RuntimeError("doc boom")
        return _PAGE

    v = oh.verify_candidate(
        _opp(), fetch_text, lambda url: [(doc, "payout terms")])
    assert v["ok"] is False and v["kind"] == "unverifiable"
    assert v["payout_docs_checked"] == [doc]
    assert "payout doc found, no qualifying rail" in v["reason"]
    assert _snap() == before
