"""Pins for Pass tail-111: nomorals/browser/profiles.py — 3 exclusions.

160 E103 masked_proxy: urlparse/.port raise -> the RAW proxy (password
    included) is returned and lands in masked_summary, the module's
    log-safe surface. Redaction fail-open; NOT success-shaped, and the
    shape is the standing redaction posture family (cf. tail-87) — held
    as an exclusion, flagged for the owner posture list. Pinned as-is.
186 E104 cookie_header_for: unparseable URL -> "" (no cookies sent).
202 E103 cookie_header_for: unparseable expires -> cookie is INCLUDED
    (treated as a session cookie); past expiry still excluded, future
    still included. Flagged-accepted remainder (no noqa on the guard).
"""

from __future__ import annotations

import logging
import time

from nomorals.browser.profiles import BrowserProfile


# ---------------------------------------------------------------- 160

def test_masked_proxy_healthy_hides_password():
    p = BrowserProfile(name="p", proxy="http://user:secretpw@proxy.example:8080")
    assert p.masked_proxy() == "http://user:***@proxy.example:8080"
    assert "secretpw" not in str(p.masked_summary())


def test_masked_proxy_crash_returns_raw_proxy(caplog):
    # Port 99999 is out of range: urlparse succeeds, .port raises
    # ValueError, the guard swallows, the raw URL is returned.
    p = BrowserProfile(name="p", proxy="http://user:secretpw@proxy.example:99999")
    with caplog.at_level(logging.DEBUG, logger="nomorals.browser.profiles"):
        assert p.masked_proxy() == "http://user:secretpw@proxy.example:99999"
    # The failure is not recorded anywhere either (observability gap).
    assert caplog.records == []
    # ...and the raw value flows into the "log-safe" summary verbatim.
    assert "secretpw" in str(p.masked_summary())


def test_masked_proxy_direct_and_passwordless():
    assert BrowserProfile(name="p").masked_proxy() == "direct"
    p = BrowserProfile(name="p", proxy="http://proxy.example:8080")
    assert p.masked_proxy() == "http://proxy.example:8080"


# ---------------------------------------------------------------- 186

def test_cookie_header_for_bad_url_yields_no_cookies():
    jar = [{"name": "session", "value": "v1", "domain": "example.com",
            "path": "/"}]
    p = BrowserProfile(name="p", cookies=jar)
    assert p.cookie_header_for("https://example.com/x") == "session=v1"
    assert p.cookie_header_for("https://other.test/") == ""
    # Invalid IPv6 literal: urlparse raises ValueError -> "".
    assert p.cookie_header_for("http://[::1") == ""


# ---------------------------------------------------------------- 202

def test_cookie_header_for_expiry_guard():
    base = {"name": "s", "value": "v", "domain": "example.com", "path": "/"}
    url = "https://example.com/"
    past = BrowserProfile(name="p", cookies=[dict(base, expires=1)])
    assert past.cookie_header_for(url) == ""
    future = BrowserProfile(
        name="p", cookies=[dict(base, expires=time.time() + 10000)])
    assert future.cookie_header_for(url) == "s=v"
    # Unparseable expiry: float() raises, guard passes, cookie included.
    garbage = BrowserProfile(
        name="p", cookies=[dict(base, expires="not-a-date")])
    assert garbage.cookie_header_for(url) == "s=v"
