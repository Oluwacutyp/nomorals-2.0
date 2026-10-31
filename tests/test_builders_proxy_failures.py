"""Tail-70 pins: nomorals/builders_proxy.py silent-swallow sites.

All five sites are adjudicated EXCLUSIONS (see
~/workspace/nm-tail70-verify/SURVEY.md): the cryptography->openssl
ladder rungs (106, 372 — the fallback IS the contract and total
failure is loud), the _do last-resort error-page guard (221), the
finally connection-close teardown (227), and run()'s deliberate
top-level KeyboardInterrupt shutdown hook (294, E103+E106).

Each pin proves the guarded operation was ATTEMPTED (spy) and that
the documented fallback behavior is what production does.
"""

from __future__ import annotations

import http.client
import io
import subprocess
import sys

import pytest

import nomorals.builders_proxy as bp


def _block_crypto(monkeypatch):
    monkeypatch.setitem(sys.modules, "cryptography", None)


def _openssl_spy(monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy(*a, **kw):
        calls.append(a[0] if a else [])
        return real_run(*a, **kw)

    monkeypatch.setattr(subprocess, "run", spy)
    return calls


# ── 106: ensure_self_signed_cert ImportError rung ───────────────────────────

def test_106_missing_cryptography_falls_back_to_openssl(tmp_path, monkeypatch):
    cert, key = str(tmp_path / "c.crt"), str(tmp_path / "c.key")
    calls = _openssl_spy(monkeypatch)
    _block_crypto(monkeypatch)
    out = bp.ensure_self_signed_cert(cert, key, common_name="probe.test")
    assert out == (cert, key)
    assert any(c and c[0] == "openssl" for c in calls), calls
    with open(cert, "rb") as fh:
        assert fh.read(27) == b"-----BEGIN CERTIFICATE-----"


def test_106_both_generators_dead_raises_loudly(tmp_path, monkeypatch):
    cert, key = str(tmp_path / "c.crt"), str(tmp_path / "c.key")

    def dead_run(*a, **kw):
        raise FileNotFoundError("no openssl")

    monkeypatch.setattr(subprocess, "run", dead_run)
    _block_crypto(monkeypatch)
    with pytest.raises(RuntimeError, match="could not generate"):
        bp.ensure_self_signed_cert(cert, key)
    assert not __import__("os").path.exists(cert)


# ── 372: verify_tls_cert ImportError rung ───────────────────────────────────

def test_372_missing_cryptography_verifies_via_openssl(tmp_path, monkeypatch):
    cert, key = str(tmp_path / "c.crt"), str(tmp_path / "c.key")
    bp.ensure_self_signed_cert(cert, key, common_name="probe.test")
    calls = _openssl_spy(monkeypatch)
    _block_crypto(monkeypatch)
    out = bp.verify_tls_cert(cert, key_path=key)
    assert out["ok"] is True
    assert out["subject"]
    assert any(c and c[0] == "openssl" and "x509" in c for c in calls), calls


# ── 221 / 227: ReverseProxyHandler._do guards ───────────────────────────────

def _handler(monkeypatch, conn_cls, send_error):
    import email.message

    monkeypatch.setattr(http.client, "HTTPConnection", conn_cls)
    h = bp.ReverseProxyHandler.__new__(bp.ReverseProxyHandler)
    h.command = "GET"
    h.path = "/x"
    h.headers = email.message.Message()
    h.rfile = io.BytesIO(b"")
    h.send_error = send_error.__get__(h, type(h))
    return h


def test_221_failed_error_page_write_never_kills_handler(monkeypatch):
    attempts = []

    class FakeConn:
        def __init__(self, *a, **kw):
            pass

        def request(self, *a, **kw):
            raise ValueError("garbage from backend")

        def close(self):
            pass

    def bad_send_error(self, code, message=None, explain=None):
        attempts.append(code)
        raise RuntimeError("client already gone")

    h = _handler(monkeypatch, FakeConn, bad_send_error)
    assert h._do() is None  # the swallow is the behavior: thread survives
    assert attempts == [502]


def test_227_finally_close_failure_is_teardown_only(monkeypatch):
    events = []

    class FakeConn:
        def __init__(self, *a, **kw):
            pass

        def request(self, *a, **kw):
            raise ConnectionRefusedError("backend down")

        def close(self):
            events.append("close")
            raise OSError("close failed")

    def ok_send_error(self, code, message=None, explain=None):
        events.append(("send_error", code, message))

    h = _handler(monkeypatch, FakeConn, ok_send_error)
    assert h._do() is None
    assert events[0][0] == "send_error" and events[0][1] == 502
    assert "backend unavailable" in events[0][2]
    assert "close" in events  # close was attempted; its failure is teardown


# ── 294: run() KeyboardInterrupt shutdown hook ──────────────────────────────

def test_294_keyboard_interrupt_is_the_shutdown_hook(monkeypatch, capsys):
    closed = []

    class FakeServer:
        daemon_threads = False

        def __init__(self, addr, handler):
            self.addr = addr

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            closed.append(True)

    monkeypatch.setattr(bp, "ThreadingHTTPServer", FakeServer)
    assert bp.run("127.0.0.1", 0, "127.0.0.1", 8000) is None
    assert closed == [True]  # finally still closes the server
    assert '"proxy": "ready"' in capsys.readouterr().out
