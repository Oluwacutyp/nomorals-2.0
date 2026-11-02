"""Tail-95 pins: tests/test_connectors_l2_proxypool.py census sites.

All five sites live inside the file's TEST DOUBLES (a fake HTTP
forward proxy, two fake SOCKS servers and their byte relay) —
not in product code. Their swallows are the doubles' designed
connection semantics, and the outward signal in each case is
honest: the client of the double sees a real 502, a closed
relay direction, or a dropped dead connection — exactly what
the proxypool suite then asserts on through real loopback
sockets. All exclusions.

119 — _ForwardProxyHandler.do_GET retry loop: 5 attempts with
backoff, then the pre-set 502 ("bad gateway") is sent. The
failure value IS the surface the test asserts on; a crashed
upstream and an unreachable one both honestly read as 502 to
the proxied client, which is what a forward proxy owes.

217 / 222 — _socks_relay.fwd: an OSError on recv ends that
pump direction (finally still half-closes the peer), and an
OSError from the shutdown itself — the socket is already
dead — has no further action available. Canonical relay
teardown.

282 / 316 — the SOCKS5 / SOCKS4a handlers' outer
``except (ConnectionError, OSError): pass``: a client that
dies mid-handshake ends its own handler; a test double has
no error surface to report to, and the server keeps serving.

NOT-recording is pinned with the snapshot method: these
doubles call no failures-ledger record function, so both the
agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import http.server
import importlib.util
import io
import socket
import threading
from pathlib import Path
from types import SimpleNamespace

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail

_spec = importlib.util.spec_from_file_location(
    "proxypool_doubles",
    Path(__file__).with_name("test_connectors_l2_proxypool.py"))
doubles = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(doubles)


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


# ── 119: dead upstream -> retries exhausted -> honest 502 ───────

def _bare_handler(path):
    handler = doubles._ForwardProxyHandler.__new__(
        doubles._ForwardProxyHandler)
    handler.server = SimpleNamespace(require_auth=False)
    handler.path = path
    handler.wfile = io.BytesIO()
    codes = []
    handler.send_response = lambda code: codes.append(code)
    handler.send_header = lambda *a: None
    handler.end_headers = lambda: None
    return handler, codes


def test_119_forward_proxy_dead_upstream_answers_502():
    before = _snap()
    handler, codes = _bare_handler("http://127.0.0.1:9/never")
    handler.do_GET()
    assert codes == [502]
    assert handler.wfile.getvalue() == b"bad gateway"

    # control: a live target is forwarded to, untouched semantics
    target = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), doubles._TargetHandler)
    thread = threading.Thread(target=target.serve_forever, daemon=True)
    thread.start()
    try:
        port = target.server_address[1]
        handler2, codes2 = _bare_handler(f"http://127.0.0.1:{port}/x")
        handler2.do_GET()
    finally:
        target.shutdown()
        target.server_close()
    assert codes2 == [200]
    assert handler2.wfile.getvalue() == b"target-ok"
    assert _snap() == before


# ── 217 / 222: relay teardown on dead sockets ───────────────────

class _DeadSock:
    def __init__(self, *, fail_shutdown=False):
        self.shutdown_calls = []
        self._fail_shutdown = fail_shutdown

    def recv(self, n):
        raise OSError("connection dead")

    def sendall(self, data):
        raise OSError("connection dead")

    def shutdown(self, how):
        self.shutdown_calls.append(how)
        if self._fail_shutdown:
            raise OSError("already dead")


def test_217_relay_recv_oserror_ends_direction_and_half_closes():
    before = _snap()
    a, b = _DeadSock(), _DeadSock()
    assert doubles._socks_relay(a, b) is None
    # each direction's finally still attempted the half-close
    assert a.shutdown_calls == [socket.SHUT_WR]
    assert b.shutdown_calls == [socket.SHUT_WR]
    assert _snap() == before


def test_222_relay_shutdown_oserror_is_terminal_quiet():
    before = _snap()
    a = _DeadSock(fail_shutdown=True)
    b = _DeadSock(fail_shutdown=True)
    assert doubles._socks_relay(a, b) is None
    assert a.shutdown_calls == [socket.SHUT_WR]
    assert b.shutdown_calls == [socket.SHUT_WR]
    assert _snap() == before


# ── 282 / 316: dead client mid-handshake ends its handler ───────

class _EofConn:
    def recv(self, n):
        return b""

    def sendall(self, data):
        pass


def test_282_socks5_handler_survives_dead_client():
    before = _snap()
    handler = doubles._Socks5Handler.__new__(doubles._Socks5Handler)
    handler.request = _EofConn()
    handler.server = SimpleNamespace(require_auth=False)
    assert handler.handle() is None
    assert _snap() == before


def test_316_socks4a_handler_survives_dead_client():
    before = _snap()
    handler = doubles._Socks4aHandler.__new__(doubles._Socks4aHandler)
    handler.request = _EofConn()
    assert handler.handle() is None
    assert _snap() == before
