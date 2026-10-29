"""Tail-47 pins for nomorals/core/http.py — 0 converts /
8 exclusions, 0 flagged.

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the guarded path
was actually exercised, asserting via the double's/wrapper's
own call log that the failing operation itself was attempted.
The file contains no exception handlers, so it adds zero
census sites.

http.py calls no failures-ledger record function anywhere
(grep-proven: no record_failure / recent_failures reference),
so NOT-recording is pinned with the tail-8 snapshot method:
agents recent_failures() before/after each trigger must be
unchanged. The module's own observability surface for the
request-path sites is HttpClient.stats, pinned where relevant.

Verdict notes (full grounds in the tail-47 SURVEY.md):
- 68: _check_dead_endpoint urlparse failure -> return; the
  request path's SSRF guard rejects the same unparseable URL
  loudly, so no dead-endpoint claim is ever rendered.
- 145: _error_snippet bad error-JSON -> raw body fallthrough;
  the server's own text is preserved, not replaced by a claim.
- 213: _parse_retry_after unparsable Retry-After date -> pass
  and fall through to the reset headers, else the documented
  1.0s fallback; the RateLimited error itself still raises.
- 297: _certifi_cafile certifi.where() failure -> None, the
  documented platform-default-store fallback; TLS verification
  stays on either way.
- 527: HTTPError body read failure -> body stays b""; the
  mapped status error still raises, honestly reporting
  "<no readable error detail>", and stats count the error.
- 543: HTTPError headers items failure -> headers None; the
  mapped status error still raises, with the documented
  retry_after default of 1.0 for a 429.
- 689: _b64url_text non-base64/non-UTF8 payload -> "", the
  function's documented genuine value for "isn't one".
- 737: unwrap_redirect_url internal failure -> the original
  URL unchanged; the function is documented "Never raises"
  and the unchanged URL is its genuine no-unwrap value.
"""
import base64
import email.utils
import io
import json
import sys
import time
import types
import urllib.error
import urllib.parse

import pytest

sys.path.insert(0, "/home/hatch/workspace/nm-tail47-tree")

from nomorals.agents import failures as ag_fail
from nomorals.core import http as http_mod
from nomorals.core.errors import NoMoralsError
from nomorals.core.http import HttpClient, RequestError


def _snap():
    return list(ag_fail.recent_failures())


def test_68_dead_endpoint_unparseable_url_returns_none():
    before = _snap()
    calls = []
    orig = urllib.parse.urlparse

    def counting(url, *a, **k):
        calls.append(url)
        return orig(url, *a, **k)

    urllib.parse.urlparse = counting
    try:
        out = http_mod._check_dead_endpoint("http://[::1")
        assert "http://[::1" in calls  # the parse was attempted
        assert out is None  # no dead-endpoint check, no claim
        with pytest.raises(RequestError):
            http_mod._check_dead_endpoint(
                "https://api.coindesk.com/v1/bpi/currentprice/BTC.json")
        assert http_mod._check_dead_endpoint(
            "https://example.com/x") is None  # control: live URL
    finally:
        urllib.parse.urlparse = orig
    # the request path rejects the same malformed URL loudly
    with pytest.raises(NoMoralsError):
        HttpClient(allow_private_ips=False)._check_ssrf("http://[::1")
    assert _snap() == before


def test_145_error_snippet_bad_json_falls_through_to_raw():
    before = _snap()
    calls = []
    orig = json.loads

    def counting(s, *a, **k):
        calls.append(s)
        return orig(s, *a, **k)

    json.loads = counting
    try:
        bad_body = '{"error": {broken json'
        out = http_mod._error_snippet(bad_body)
        assert bad_body in calls  # the JSON parse was attempted
        assert out == bad_body  # raw body preserved verbatim
        good = http_mod._error_snippet(
            '{"error": {"message": "model gone"}}')
        assert good == "model gone"  # control: valid JSON unwrapped
        assert http_mod._error_snippet("plain reason") == "plain reason"
    finally:
        json.loads = orig
    assert _snap() == before


def test_213_retry_after_bad_date_falls_through():
    before = _snap()
    calls = []
    orig = email.utils.parsedate_to_datetime

    def counting(s, *a, **k):
        calls.append(s)
        return orig(s, *a, **k)

    email.utils.parsedate_to_datetime = counting
    try:
        reset = str(time.time() + 120)
        out = http_mod._parse_retry_after(
            {"Retry-After": "garbage!!", "X-RateLimit-Reset": reset})
        assert "garbage!!" in calls  # the date parse was attempted
        assert 100.0 < out <= 120.0  # fell through to the reset header
        out2 = http_mod._parse_retry_after({"Retry-After": "garbage!!"})
        assert out2 == 1.0  # documented fallback when nothing usable
        assert http_mod._parse_retry_after({"Retry-After": "5"}) == 5.0
    finally:
        email.utils.parsedate_to_datetime = orig
    assert _snap() == before


def test_297_certifi_where_failure_returns_none():
    before = _snap()
    saved = sys.modules.get("certifi", "ABSENT")
    fake = types.ModuleType("certifi")
    calls = []

    def raising_where():
        calls.append(1)
        raise RuntimeError("certifi exploded")

    fake.where = raising_where
    sys.modules["certifi"] = fake
    try:
        assert http_mod._certifi_cafile() is None
        assert len(calls) == 1  # certifi.where() was attempted
    finally:
        if saved == "ABSENT":
            sys.modules.pop("certifi", None)
        else:
            sys.modules["certifi"] = saved
    fake2 = types.ModuleType("certifi")
    fake2.where = lambda: http_mod.__file__
    sys.modules["certifi"] = fake2
    try:
        assert http_mod._certifi_cafile() == http_mod.__file__  # control
    finally:
        if saved == "ABSENT":
            sys.modules.pop("certifi", None)
        else:
            sys.modules["certifi"] = saved
    assert _snap() == before


class _RaisingFP:
    def __init__(self):
        self.calls = 0

    def read(self, *a, **k):
        self.calls += 1
        raise RuntimeError("read exploded")

    def readline(self, *a, **k):
        return b""

    def close(self):
        pass

    def info(self):
        return {}


class _BadHeaders:
    def __init__(self):
        self.calls = 0

    def items(self):
        self.calls += 1
        raise RuntimeError("items exploded")


class _FakeOpener:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def open(self, request, timeout=None):
        self.calls += 1
        raise self.exc


def _request_with_opener(http_error_exc, holder):
    """Run one GET whose opener raises http_error_exc; records the
    opener and client in holder. The mapped error propagates."""
    opener = _FakeOpener(http_error_exc)
    holder["opener"] = opener
    orig = http_mod._no_proxy_opener
    http_mod._no_proxy_opener = lambda kwargs: opener
    client = HttpClient(allow_private_ips=True)
    holder["client"] = client
    try:
        client.get("http://example.test/x")
    finally:
        http_mod._no_proxy_opener = orig


def test_527_http_error_body_read_failure_keeps_status_error():
    before = _snap()
    fp = _RaisingFP()
    err = urllib.error.HTTPError("http://example.test/x", 500, "err",
                                 {"Content-Type": "text/plain"}, fp)
    holder = {}
    with pytest.raises(Exception) as ei:
        _request_with_opener(err, holder)
    assert holder["opener"].calls == 1  # the open was attempted
    assert fp.calls == 1  # the body read was attempted
    assert holder["client"].stats["errors"] == 1  # module's own surface
    raised = ei.value
    assert raised.details.get("http_status") == 500
    assert "<no readable error detail>" in str(raised)
    err2 = urllib.error.HTTPError(
        "http://example.test/x", 500, "err",
        {"Content-Type": "text/plain"}, io.BytesIO(b"server boom"))
    with pytest.raises(Exception) as ei2:
        _request_with_opener(err2, {})
    assert "server boom" in str(ei2.value)  # control: body surfaces
    assert _snap() == before


def test_543_http_error_headers_failure_keeps_status_error():
    before = _snap()
    bh = _BadHeaders()
    err = urllib.error.HTTPError("http://example.test/x", 429, "err", bh,
                                 io.BytesIO(b"slow down"))
    holder = {}
    with pytest.raises(Exception) as ei:
        _request_with_opener(err, holder)
    assert holder["opener"].calls == 1  # the open was attempted
    assert bh.calls == 1  # headers.items() was attempted
    raised = ei.value
    assert raised.details.get("http_status") == 429
    assert getattr(raised, "retry_after", None) == 1.0  # documented default
    assert "slow down" in str(raised)  # body detail still surfaces
    err2 = urllib.error.HTTPError("http://example.test/x", 429, "err",
                                  {"Retry-After": "7"},
                                  io.BytesIO(b"slow down"))
    with pytest.raises(Exception) as ei2:
        _request_with_opener(err2, {})
    assert getattr(ei2.value, "retry_after", None) == 7.0  # control
    assert _snap() == before


def test_689_b64url_text_non_utf8_returns_empty():
    before = _snap()
    calls = []
    orig = base64.urlsafe_b64decode

    def counting(s, *a, **k):
        calls.append(s)
        return orig(s, *a, **k)

    base64.urlsafe_b64decode = counting
    try:
        raw = base64.urlsafe_b64encode(b"\xff\xfe").decode().rstrip("=")
        assert http_mod._b64url_text(raw) == ""
        assert len(calls) == 1  # the decode was attempted
        good = base64.urlsafe_b64encode(
            b"https://example.org/").decode().rstrip("=")
        assert http_mod._b64url_text(good) == "https://example.org/"
    finally:
        base64.urlsafe_b64decode = orig
    assert _snap() == before


def test_737_unwrap_failure_returns_original_url():
    before = _snap()
    calls = []
    orig = urllib.parse.urlparse

    def raising(url, *a, **k):
        calls.append(url)
        raise RuntimeError("parse exploded")

    url = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fa"
    urllib.parse.urlparse = raising
    try:
        out = http_mod.unwrap_redirect_url(url)
        assert out == url  # documented never-raises: original returned
        assert len(calls) >= 1  # the parse was attempted
    finally:
        urllib.parse.urlparse = orig
    assert http_mod.unwrap_redirect_url(url) == "https://example.org/a"
    plain = "https://example.com/plain"
    assert http_mod.unwrap_redirect_url(plain) == plain  # control
    assert _snap() == before
