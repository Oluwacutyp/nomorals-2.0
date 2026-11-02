"""Tail-107 pins: nomorals/tools/network.py census sites
(250, 304, 354, 451).

All four sites are held as EXCLUSIONS. The module is an operator
network toolkit (own infrastructure only); every guard degrades
fail-closed, to an already-proven result, or to a truthful partial
answer — none fabricates a success.

250 — ``is_scannable`` scope guard: ``ipaddress`` raising
ValueError is the branch DISCRIMINATOR, not a swallow — the target
is a hostname, and control falls through to the allowlist-only
check beneath (exact literal match, case-insensitive). A hostname
gets NO range-based pass: the failure direction is fail-CLOSED on
an authority-adjacent gate, and the fence is the allowlist itself.
Pinned both ways (hostname denied/allowed, loopback pass, public
IP denied/allowed). ``_assert_scannable`` consumes this verdict to
enforce port-scan scope.

304 — ``tcp_probe`` banner grab: the port is already proven open
(``connect_ex`` returned 0) before the banner recv runs; a banner
timeout/OSError degrades the answer to plain "open" — the proven
fact, minus the optional decoration. The noqa prints the contract
("banner grab is best-effort; port already known open"). Pinned
with a silent local listener (banner times out -> "open"), a
banner-speaking control ("open:<banner>"), and a closed control.

354 — ``local_listeners`` ss fallback: the primary source is
/proc/net/tcp{,6}; the ``ss`` subprocess runs ONLY when /proc
yielded nothing (non-Linux). Its failure leaves the port list
partial (here: empty) — the docstring discloses both sources, and
the consumer is an informational "what is listening" tool surface;
no gate consumes the list. Pinned by blocking /proc AND raising
the ss run (returns [], never raises), plus a control proving a
real listener is listed, plus non-recording.

451 — ``http_craft``: an HTTPError (4xx/5xx) is an ANSWER for a
request-crafting tool, reported with its real status and
``ok: False``. Only the error BODY read degrades: if ``exc.read``
raises, ``body`` becomes "" while ``status``/``ok`` stay truthful.
No success is fabricated — the failure surface is the status line,
which is preserved verbatim. Pinned (500 + unreadable body ->
status 500, ok False, body "") with a 200 control.
"""
from __future__ import annotations

import socket
import subprocess
import threading
import urllib.error
import urllib.request
from unittest import mock

from nomorals.tools import network as netmod


# ── 250: scope guard — hostname falls to the allowlist-only check ─

def test_250_hostname_not_allowlisted_denied():
    assert netmod.is_scannable("example.com", "") is False


def test_250_hostname_allowlisted_case_insensitive():
    assert netmod.is_scannable("example.com", "EXAMPLE.com") is True


def test_250_loopback_ip_passes():
    assert netmod.is_scannable("127.0.0.1", "") is True


def test_250_public_ip_not_allowlisted_denied():
    assert netmod.is_scannable("8.8.8.8", "") is False


def test_250_public_ip_allowlisted_passes():
    assert netmod.is_scannable("8.8.8.8", "8.8.8.8") is True


# ── 304: banner grab degrades to the proven "open" ────────────────

def _serve_once(payload, hold=0.0):
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def run():
        try:
            conn, _ = srv.accept()
            if payload:
                conn.sendall(payload)
            if hold:
                threading.Event().wait(hold)
            conn.close()
        except Exception as exc:
            assert isinstance(exc, Exception)
        finally:
            srv.close()

    threading.Thread(target=run, daemon=True).start()
    return port


def test_304_silent_banner_degrades_to_open():
    port = _serve_once(None, hold=2.0)
    assert netmod.tcp_probe("127.0.0.1", port, timeout=0.5,
                            banner=True) == "open"


def test_304_banner_control():
    port = _serve_once(b"SSH-2.0-probe\r\n")
    assert netmod.tcp_probe("127.0.0.1", port, timeout=1.5,
                            banner=True) == "open:SSH-2.0-probe"


def test_304_closed_control():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    dead = sock.getsockname()[1]
    sock.close()
    assert netmod.tcp_probe("127.0.0.1", dead, timeout=1.0) == "closed"


# ── 354: ss fallback failure leaves a partial list, never raises ──

def test_354_proc_and_ss_both_fail_returns_empty():
    real_open = open

    def blocking_open(path, *args, **kwargs):
        if str(path).startswith("/proc/net/tcp"):
            raise OSError("no proc")
        return real_open(path, *args, **kwargs)

    log = mock.Mock()
    with mock.patch.object(netmod, "_log", log), \
            mock.patch("builtins.open", blocking_open), \
            mock.patch.object(subprocess, "run",
                              side_effect=OSError("no ss")):
        assert netmod.local_listeners() == []
    log.error.assert_not_called()
    log.warning.assert_not_called()


def test_354_control_real_listener_listed():
    port = _serve_once(None, hold=2.0)
    listeners = netmod.local_listeners()
    assert any(d["port"] == port for d in listeners)


# ── 451: HTTPError body degrades; the status stays truthful ───────

class _ReadBoomError(urllib.error.HTTPError):
    def read(self, *args, **kwargs):
        raise OSError("body gone")


class _FakeOpener:
    def __init__(self, mode):
        self.mode = mode

    def open(self, request, timeout=None):
        if self.mode == "err":
            raise _ReadBoomError(request.full_url, 500, "Internal",
                                 None, None)

        class Resp:
            status = 200
            headers = {}

            def read(self, n=-1):
                return b"hello"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        return Resp()


def test_451_error_body_unreadable_status_truthful():
    with mock.patch.object(urllib.request, "build_opener",
                           lambda *h: _FakeOpener("err")):
        res = netmod.http_craft("GET", "http://127.0.0.1/x")
    assert res["status"] == 500
    assert res["ok"] is False
    assert res["body"] == ""


def test_451_control_ok_response_body_returned():
    with mock.patch.object(urllib.request, "build_opener",
                           lambda *h: _FakeOpener("ok")):
        res = netmod.http_craft("GET", "http://127.0.0.1/x")
    assert res["status"] == 200
    assert res["ok"] is True
    assert res["body"] == "hello"
