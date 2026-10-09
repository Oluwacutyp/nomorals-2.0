"""Tests for the retired-endpoint registry in nomorals.core.http.

A dead API endpoint must fail fast with redirect guidance — never burn
a DNS lookup / connection on it, and never surface a cryptic error.
"""

import pytest

from nomorals.core.http import (
    DEAD_ENDPOINTS,
    HttpClient,
    RequestError,
    _check_dead_endpoint,
)


def test_registry_has_entries():
    assert len(DEAD_ENDPOINTS) >= 1
    for host, prefix, guidance in DEAD_ENDPOINTS:
        assert host and prefix.startswith("/") and guidance


def test_dead_endpoint_raises_with_guidance():
    with pytest.raises(RequestError) as ei:
        _check_dead_endpoint(
            "https://api.coindesk.com/v1/bpi/currentprice/BTC.json")
    assert "Retired endpoint" in str(ei.value)
    assert "finance_price" in str(ei.value)


def test_dead_endpoint_prefix_match():
    # Any path under the retired prefix is blocked, query strings included.
    with pytest.raises(RequestError):
        _check_dead_endpoint(
            "https://api.coindesk.com/v1/bpi/currentprice.json?x=1")


def test_live_endpoints_unaffected():
    assert _check_dead_endpoint(
        "https://api.binance.com/api/v3/ticker/24hr") is None
    assert _check_dead_endpoint(
        "https://api.coingecko.com/api/v3/simple/price") is None
    # Same host, different (live) path prefix — not blocked.
    assert _check_dead_endpoint(
        "https://www.coindesk.com/arc/outboundfeeds/rss/") is None


def test_client_request_rejects_before_network():
    # Must raise before any DNS/connection attempt.
    client = HttpClient()
    with pytest.raises(RequestError) as ei:
        client.get("https://api.coindesk.com/v1/bpi/currentprice/BTC.json")
    assert "Retired endpoint" in str(ei.value)
