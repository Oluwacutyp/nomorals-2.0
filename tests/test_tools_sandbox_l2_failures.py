"""Tail-34 pins for nomorals/tools/sandbox_l2.py — 0 converts / 9 exclusions.

Each test pins one site's fallback behavior plus a control half proving the
except path was taken (trigger == fallback AND control != fallback). The file
itself contains no exception handlers, so it adds zero census sites.

sandbox_l2.py calls no record function anywhere (grep-proven), so
NOT-recording is pinned with the tail-8 snapshot method: recent_failures()
before/after each trigger must be unchanged — a proxy failure leaves no
trace anywhere. (The proxy's own debug log is not a failure record.)

Common shape of all 9 sites: every pass fires only when the peer is already
gone (BrokenPipe/ConnectionReset on a client socket) or during best-effort
teardown (stop/finally). There is no consumer left to mislead: the client
that would receive an error text has disconnected, and teardown failures
leave residue only (daemon thread / bound socket / fd until GC), never a
claim. Site 237's boundary is proven by code-read, not assumed: egress.py
_one_hop wraps every upstream OSError into EgressDenied, so fetch can never
raise BrokenPipe/ConnectionReset into the handler — the pass fires only on
client-side socket failures.
"""

import socket
from types import SimpleNamespace

from nomorals.agents import failures as ag_fail
from nomorals.core import egress as E
from nomorals.tools import sandbox_l2 as S


def _snap():
    return list(ag_fail.recent_failures())


def _fake_handler():
    h = S._ProxyHandler.__new__(S._ProxyHandler)
    h.server = SimpleNamespace(proxy=None)
    h.path = "http://example.com/"
    h.headers = {}
    h.close_connection = False
    return h


def _proxy_for(handler, client):
    p = SimpleNamespace(
        token=SimpleNamespace(token_id="t", task_id="k"),
        client=client,
        controller=SimpleNamespace(check=lambda t: None,
                                   track_socket=lambda s: None,
                                   untrack_socket=lambda s: None,
                                   killed=False))
    handler.server.proxy = p
    return p


class FakeSocket:
    def __init__(self, log, name, fail_setblocking=False, fail_close=False):
        self._log = log
        self._name = name
        self._fail_sb = fail_setblocking
        self._fail_close = fail_close

    def setblocking(self, flag):
        self._log.append((f"{self._name}.setblocking", (flag,)))
        if self._fail_sb:
            raise OSError("dead socket")

    def close(self):
        self._log.append((f"{self._name}.close", ()))
        if self._fail_close:
            raise OSError("dead socket")


def _connect_handler(log, token_check=None, url_check=None, send_ok=None,
                     send_text=None):
    h = S._ProxyHandler.__new__(S._ProxyHandler)
    h.path = "example.com:443"
    h.connection = FakeSocket(log, "client")
    h.close_connection = False
    ctl = SimpleNamespace(check=token_check or (lambda t: None),
                          track_socket=lambda s: log.append(("track", ())),
                          untrack_socket=lambda s: log.append(("untrack", ())),
                          killed=False)
    tok = SimpleNamespace(check_url=url_check or (lambda u: None),
                          allow_private_ips=False, timeout=5,
                          revoked=False, expired=False)
    h.server = SimpleNamespace(proxy=SimpleNamespace(token=tok, controller=ctl))
    h._send_connect_ok = send_ok or (lambda: log.append(("connect_ok", ())))
    h._tunnel = lambda *a: log.append(("tunnel", ()))
    h._send_text = send_text or (lambda status, text: log.append(
        ("send_text", (status, text))))
    return h


def _run_connect(h):
    orig_check = S.check_host_allowed
    orig_create = socket.create_connection
    S.check_host_allowed = lambda host, allow_private_ips=False: ["127.0.0.1"]
    socket.create_connection = lambda addr, timeout=None: FakeSocket([], "up")
    try:
        h.do_CONNECT()
    finally:
        S.check_host_allowed = orig_check
        socket.create_connection = orig_create


def test_196_stop_shutdown_failure_is_best_effort():
    before = _snap()
    log = []

    class Server:
        def shutdown(self):
            log.append("shutdown")
            raise RuntimeError("shutdown blew up")

        def server_close(self):
            log.append("server_close")

    proxy = S.EgressProxy.__new__(S.EgressProxy)
    proxy._server = Server()
    assert proxy.stop() is None
    assert log == ["shutdown", "server_close"]
    assert _snap() == before

    log.clear()

    class GoodServer:
        def shutdown(self):
            log.append("shutdown")

        def server_close(self):
            log.append("server_close")

    proxy._server = GoodServer()
    assert proxy.stop() is None
    assert log == ["shutdown", "server_close"]


def test_200_stop_close_failure_is_best_effort():
    before = _snap()
    log = []

    class Server:
        def shutdown(self):
            log.append("shutdown")

        def server_close(self):
            log.append("server_close")
            raise RuntimeError("close blew up")

    proxy = S.EgressProxy.__new__(S.EgressProxy)
    proxy._server = Server()
    assert proxy.stop() is None
    assert log == ["shutdown", "server_close"]
    assert _snap() == before

    log.clear()

    class GoodServer:
        def shutdown(self):
            log.append("shutdown")

        def server_close(self):
            log.append("server_close")

    proxy._server = GoodServer()
    assert proxy.stop() is None
    assert log == ["shutdown", "server_close"]


def test_237_relay_broken_pipe_passes_silently():
    before = _snap()
    sent = []
    h = _fake_handler()
    h._read_body = lambda: b""

    def raising_relay(status, headers, body):
        raise BrokenPipeError("client gone")

    h._send_relay = raising_relay
    h._send_text = lambda status, text: sent.append((status, text))
    _proxy_for(h, SimpleNamespace(fetch=lambda *a, **k: SimpleNamespace(
        status=200, headers={}, body=b"hi")))
    assert h._handle_proxy_request("GET") is None
    assert sent == []
    assert _snap() == before

    relayed = []
    h2 = _fake_handler()
    h2._read_body = lambda: b""
    h2._send_relay = lambda status, headers, body: relayed.append((status, body))
    h2._send_text = lambda status, text: sent.append((status, text))
    _proxy_for(h2, SimpleNamespace(fetch=lambda *a, **k: SimpleNamespace(
        status=200, headers={}, body=b"ok")))
    h2._handle_proxy_request("GET")
    assert relayed == [(200, b"ok")]


def test_243_generic_502_send_broken_pipe():
    before = _snap()
    sent = []

    def raising_text(status, text):
        sent.append((status, text))
        raise BrokenPipeError("client gone during 502")

    h = _fake_handler()
    h._read_body = lambda: b""
    h._send_text = raising_text
    _proxy_for(h, SimpleNamespace(fetch=lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom"))))
    assert h._handle_proxy_request("GET") is None
    assert sent == [(502, "egress proxy error: boom\n")]
    assert _snap() == before

    sent.clear()
    h2 = _fake_handler()
    h2._read_body = lambda: b""
    h2._send_text = lambda status, text: sent.append((status, text))
    _proxy_for(h2, SimpleNamespace(fetch=lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("boom"))))
    h2._handle_proxy_request("GET")
    assert sent == [(502, "egress proxy error: boom\n")]


def test_359_finally_setblocking_failure_passes():
    before = _snap()
    log = []
    h = _connect_handler(log)
    client_sock = h.connection

    def failing_sb(flag):
        log.append(("client.setblocking", (flag,)))
        raise OSError("dead socket")

    client_sock.setblocking = failing_sb
    _run_connect(h)
    names = [c[0] for c in log]
    assert "tunnel" in names and "untrack" in names
    assert _snap() == before

    log.clear()
    h2 = _connect_handler(log)
    _run_connect(h2)
    names = [c[0] for c in log]
    assert "tunnel" in names and "untrack" in names


def test_363_finally_close_failure_passes():
    before = _snap()
    log = []
    h = _connect_handler(log)
    orig_check = S.check_host_allowed
    orig_create = socket.create_connection
    S.check_host_allowed = lambda host, allow_private_ips=False: ["127.0.0.1"]
    up = FakeSocket(log, "upstream", fail_close=True)
    socket.create_connection = lambda addr, timeout=None: up
    try:
        h.do_CONNECT()
    finally:
        S.check_host_allowed = orig_check
        socket.create_connection = orig_create
    assert ("upstream.close", ()) in log
    assert "tunnel" in [c[0] for c in log]
    assert _snap() == before


def test_368_denial_403_send_broken_pipe():
    before = _snap()
    sent = []

    def raising_text(status, text):
        sent.append((status, text))
        raise BrokenPipeError("client gone during 403")

    h = _connect_handler(
        log=[],
        url_check=lambda u: (_ for _ in ()).throw(E.EgressDenied("not allowed")),
        send_text=raising_text)
    _run_connect(h)
    assert sent == [(403, "egress denied: not allowed\n")]
    assert _snap() == before

    sent.clear()
    h2 = _connect_handler(
        log=[],
        url_check=lambda u: (_ for _ in ()).throw(E.EgressDenied("not allowed")),
        send_text=lambda status, text: sent.append((status, text)))
    _run_connect(h2)
    assert sent == [(403, "egress denied: not allowed\n")]


def test_370_connect_ack_broken_pipe_passes():
    before = _snap()
    log = []
    sent = []
    h = _connect_handler(
        log,
        send_ok=lambda: (_ for _ in ()).throw(
            BrokenPipeError("client gone at ack")),
        send_text=lambda status, text: sent.append((status, text)))
    _run_connect(h)
    assert sent == []
    assert "untrack" in [c[0] for c in log]  # finally ran, not a bypass
    assert _snap() == before

    log.clear()
    sent.clear()
    h2 = _connect_handler(
        log, send_text=lambda status, text: sent.append((status, text)))
    _run_connect(h2)
    assert sent == []
    assert "connect_ok" in [c[0] for c in log]
    assert "tunnel" in [c[0] for c in log]


def test_375_generic_502_send_broken_pipe():
    before = _snap()
    sent = []

    def raising_text(status, text):
        sent.append((status, text))
        raise ConnectionResetError("client gone during 502")

    h = _connect_handler(
        log=[],
        token_check=lambda t: (_ for _ in ()).throw(RuntimeError("ctl boom")),
        send_text=raising_text)
    _run_connect(h)
    assert sent == [(502, "egress proxy error: ctl boom\n")]
    assert _snap() == before

    sent.clear()
    h2 = _connect_handler(
        log=[],
        token_check=lambda t: (_ for _ in ()).throw(RuntimeError("ctl boom")),
        send_text=lambda status, text: sent.append((status, text)))
    _run_connect(h2)
    assert sent == [(502, "egress proxy error: ctl boom\n")]
