"""Tail-92 pins: nomorals/mesh/http_transport.py census sites.

All five sites are best-effort rungs inside a transport whose
failures DO surface: _request logs every failed attempt, counts
it against the circuit breaker, and raises HubUnreachable /
TransportError / the translated wire exception when attempts run
out. The swallows here only guard secondary work — parsing a
timing hint, closing a dead connection, extracting error detail —
and each degrades to the honest absence value (None / "") while
the primary failure travels on. All five held as exclusions:

79 — ``_retry_after_seconds``: a headers object whose .get
raises -> None (no server hint). The retry loop still applies
full-jitter backoff and the circuit still counts the failure;
the hint is advisory timing only.
220 — ``_KeepAlivePool.request``: closing the half-dead
connection after a request failure raises -> swallowed, and
the ORIGINAL request exception is re-raised untouched (``raise``
on the next line). Cleanup noise must not mask the real error.
238 — ``_KeepAlivePool.close``: one connection's close()
raises -> swallowed so the loop still closes every other pooled
connection; close() is teardown, its result is consumed nowhere.
385 — ``_error_detail``: an error body that is not JSON -> "".
The caller still logs the status and raises TransportError with
the status in the message; only the server's detail text is lost.
394 — ``_translate_http_error``: an unparseable body -> {} and
the mapping falls back to status alone (401 AuthError, 404
TransportError without the node_unknown code, 400 ValueError).
The exception still raises — degraded detail, never success.

NOT-recording is pinned with the snapshot method: this module
calls no failures-ledger record function anywhere, so both
ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

from unittest import mock

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.mesh import http_transport as H
from nomorals.mesh.errors import AuthError, NodeUnknown, TransportError


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


class _ExplodingHeaders:
    def get(self, name):
        raise RuntimeError("headers boom")


class _DeadConn:
    """A pooled connection whose request and close both fail."""

    def __init__(self):
        self.closed = False

    def request(self, method, path, body=None, headers=None):
        raise ConnectionError("connection dead")

    def close(self):
        raise RuntimeError("close boom")


class _GoodConn:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


# ── 79: exploding headers -> no Retry-After hint ────────────────

def test_79_exploding_headers_yield_no_retry_after():
    before = _snap()
    assert H._retry_after_seconds(_ExplodingHeaders()) is None
    # controls: a real hint parses, a garbage hint degrades to None
    assert H._retry_after_seconds({"Retry-After": "5"}) == 5.0
    assert H._retry_after_seconds({"Retry-After": "soon"}) is None
    assert _snap() == before


# ── 220: dead conn close failure never masks the real error ─────

def test_220_close_failure_reraises_original_request_error():
    before = _snap()
    pool = H._KeepAlivePool()
    dead = _DeadConn()
    with mock.patch.object(
            H._KeepAlivePool, "_new_connection",
            staticmethod(lambda scheme, host, port, timeout: dead)):
        with pytest.raises(ConnectionError):
            pool.request("POST", "http://hub.local:9999/mesh/poll",
                         b"{}", {}, 1.0)
    # the broken connection was dropped from the pool, not kept
    assert pool._conns == {}
    assert _snap() == before


# ── 238: pool close survives one bad connection ─────────────────

def test_238_pool_close_closes_the_rest_despite_one_bad_conn():
    before = _snap()
    pool = H._KeepAlivePool()
    good = _GoodConn()
    pool._conns[("http", "a.local", 80)] = _DeadConn()
    pool._conns[("http", "b.local", 80)] = good
    pool.close()  # must not raise
    assert good.closed is True
    assert pool._conns == {}
    assert _snap() == before


# ── 385: unparseable error body -> empty detail ─────────────────

def test_385_error_detail_degrades_to_empty_string():
    before = _snap()
    assert H.HttpTransport._error_detail(b"this is not json") == ""
    ok = H.HttpTransport._error_detail(b'{"detail": "hub says no"}')
    assert ok == "hub says no"
    assert _snap() == before


# ── 394: unparseable body -> status-only error translation ──────

def test_394_translate_falls_back_to_status_mapping():
    before = _snap()
    exc = H.HttpTransport._translate_http_error(404, b"garbage")
    assert isinstance(exc, TransportError)
    assert not isinstance(exc, NodeUnknown)  # no code -> no node mapping
    t = H.HttpTransport._translate_http_error
    assert isinstance(t(401, b"garbage"), AuthError)
    assert isinstance(t(400, b"garbage"), ValueError)
    # control: a parseable node_unknown body still maps precisely
    ok = t(404, b'{"code": "node_unknown"}')
    assert isinstance(ok, NodeUnknown)
    assert _snap() == before
