"""Failing-first tests: source health — every source verified-live or stood down.

Owner rule (2026-10-18): a source that doesn't work is WORSE than no
source. Lists must contain only verified-live sources; anything else is
removed, fixed, or stood down by the circuit breaker and re-checked.
These tests pin the mechanism with an injected fake fetcher (no network).
"""

import time

import pytest

from nomorals.agency.sourcehealth import (
    FetchResult,
    SourceHealth,
    SourceSpec,
    check_one,
    RSS,
    JSON,
    HTML_LINKS,
)
from nomorals.storage.db import Database


FEED_XML = (
    b'<?xml version="1.0"?><rss version="2.0"><channel>'
    b'<item><title>A</title><link>https://x.example/a</link></item>'
    b'<item><title>B</title><link>https://x.example/b</link></item>'
    b'</channel></rss>'
)
EMPTY_FEED = (
    b'<?xml version="1.0"?><rss version="2.0"><channel></channel></rss>'
)
JSON_BODY = b'{"results": [{"id": 1}, {"id": 2}]}'
CHALLENGE = b'<html><head><title>Just a moment...</title></head>' \
            b'<body>Please verify you are human. Cloudflare.</body></html>'


def _health():
    return SourceHealth(Database(":memory:"))


def _spec(**kw):
    base = dict(key="news:world:bbc", category="news", name="BBC",
                url="https://feeds.example/rss.xml", kind=RSS)
    base.update(kw)
    return SourceSpec(**base)


def _fetch_ok(url, headers=None):
    return FetchResult(200, FEED_XML, {})


def _fetch_404(url, headers=None):
    return FetchResult(404, b"not found", {})


def test_verify_live_rss_counts_items_and_resets_failures():
    h = _health()
    rec = h.verify(_spec(), fetcher=_fetch_ok, now=1000.0)
    assert rec.status == "verified-live"
    assert rec.items == 2
    assert rec.http == 200
    assert rec.consecutive_failures == 0
    assert rec.last_ok == 1000.0


def test_verify_404_is_dead_not_error():
    h = _health()
    rec = h.verify(_spec(), fetcher=_fetch_404, now=1000.0)
    assert rec.status == "dead"
    assert rec.consecutive_failures == 1
    assert rec.last_error != ""


def test_verify_empty_feed_is_dead_zero_usable_items():
    h = _health()
    rec = h.verify(_spec(), fetcher=lambda u, headers=None: FetchResult(200, EMPTY_FEED, {}))
    assert rec.status == "dead"
    assert rec.items == 0


def test_verify_challenge_page_is_bot_blocked_not_live():
    h = _health()
    rec = h.verify(_spec(), fetcher=lambda u, headers=None: FetchResult(403, CHALLENGE, {}))
    assert rec.status == "bot-blocked"
    assert "bot" in rec.last_error.lower() or "challenge" in rec.last_error.lower()


def test_verify_json_counts_results():
    h = _health()
    rec = h.verify(_spec(kind=JSON, url="https://api.example/search"),
                   fetcher=lambda u, headers=None: FetchResult(200, JSON_BODY, {}))
    assert rec.status == "verified-live"
    assert rec.items == 2


def test_circuit_breaker_stands_down_after_three_consecutive_failures():
    h = _health()
    spec = _spec()
    for i in range(2):
        rec = h.verify(spec, fetcher=_fetch_404, now=1000.0 + i)
        assert rec.status == "dead"
    rec = h.verify(spec, fetcher=_fetch_404, now=1002.0)
    assert rec.status == "stood-down"
    assert rec.consecutive_failures == 3
    # Stood-down sources are NOT in the active set.
    assert h.active_sources() == []


def test_circuit_breaker_disables_after_ten_consecutive_failures():
    h = _health()
    spec = _spec()
    for i in range(10):
        rec = h.verify(spec, fetcher=_fetch_404, now=1000.0 + i)
    assert rec.status == "disabled"
    assert rec.consecutive_failures == 10


def test_success_after_failures_resets_streak_and_reactivates():
    h = _health()
    spec = _spec()
    for i in range(3):
        h.verify(spec, fetcher=_fetch_404, now=1000.0 + i)
    assert h.record("news:world:bbc").status == "stood-down"
    rec = h.verify(spec, fetcher=_fetch_ok, now=2000.0)
    assert rec.status == "verified-live"
    assert rec.consecutive_failures == 0
    assert [s.key for s in h.active_sources()] == ["news:world:bbc"]


def test_conditional_request_uses_etag_and_304_counts_as_live():
    seen: dict = {}

    def fetching(url, headers=None):
        seen.update(headers or {})
        if headers and headers.get("If-None-Match") == '"abc"':
            return FetchResult(304, b"", {"etag": '"abc"'})
        return FetchResult(200, FEED_XML, {"etag": '"abc"'})

    h = _health()
    spec = _spec()
    first = h.verify(spec, fetcher=fetching, now=1000.0)
    assert first.status == "verified-live"
    assert first.etag == '"abc"'
    second = h.verify(spec, fetcher=fetching, now=2000.0)
    assert second.status == "verified-live"
    assert seen.get("If-None-Match") == '"abc"'
    # 304 means "unchanged": the previous item count stands.
    assert second.items == 2


def test_network_error_is_unverifiable_not_dead():
    h = _health()
    rec = h.verify(_spec(), fetcher=lambda u, headers=None: FetchResult(
        0, b"", {}, error="Remote end closed connection"))
    assert rec.status == "unverifiable"
    assert rec.consecutive_failures == 1


def test_records_persist_across_instances_on_same_db():
    db = Database(":memory:")
    h1 = SourceHealth(db)
    h1.verify(_spec(), fetcher=_fetch_ok, now=1000.0)
    h2 = SourceHealth(db)
    rec = h2.record("news:world:bbc")
    assert rec is not None
    assert rec.status == "verified-live"
    assert rec.items == 2


def test_verify_all_returns_summary_counts():
    h = _health()
    specs = [
        _spec(key="a", name="A", url="https://ok.example/a"),
        _spec(key="b", name="B", url="https://bad.example/b"),
        _spec(key="c", name="C", url="https://ok.example/c"),
    ]

    def fetching2(url, headers=None):
        if "bad.example" in url:
            return FetchResult(404, b"nf", {})
        return FetchResult(200, FEED_XML, {})

    summary = h.verify_all(specs, fetcher=fetching2, now=1000.0)
    assert summary["checked"] == 3
    assert summary["verified_live"] == 2
    assert summary["dead"] == 1
    assert len(h.all_records()) == 3


def test_check_one_pure_function_never_raises():
    def boom(url, headers=None):
        raise RuntimeError("network exploded")

    rec = check_one(_spec(), fetcher=boom, now=1000.0)
    assert rec.status == "unverifiable"


def test_html_links_kind_requires_real_links():
    h = _health()
    html = (b'<html><body><a href="https://a.example/1">one</a>'
            b'<a href="https://b.example/2">two</a></body></html>')
    rec = h.verify(_spec(kind=HTML_LINKS, url="https://search.example/?q=x"),
                   fetcher=lambda u, headers=None: FetchResult(200, html, {}))
    assert rec.status == "verified-live"
    assert rec.items == 2
    shell = h.verify(_spec(key="shell", kind=HTML_LINKS, url="https://spa.example/"),
                     fetcher=lambda u, headers=None: FetchResult(
                         200, b"<html><body>loading...</body></html>", {}))
    assert shell.status == "dead"


def test_451_is_geo_blocked():
    h = _health()
    rec = h.verify(_spec(), fetcher=lambda u, headers=None: FetchResult(
        451, b"Unavailable For Legal Reasons", {}))
    assert rec.status == "geo-blocked"
