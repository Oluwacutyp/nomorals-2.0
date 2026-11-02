"""Pass tail-118 pins — nomorals/security/netleak.py (3 exclusions).

Sites pinned (all E103 enumeration/probe degradations):
- 177: ``local_interface_ips`` hostname-resolution source (getaddrinfo)
  raises OSError -> that source is skipped; the independent default-route
  source still enumerates. The module's verdict path (STUN vs HTTP) never
  consumes this list, and its printed doctrine refuses to call absence
  "clean" — degradation, not a fabricated verdict.
- 189: per-remote default-route probe raises OSError -> that family's
  source address is skipped; the socket is still closed in ``finally``.
- 203: ``ipv6_egress`` — ``socket.has_ipv6`` AttributeError -> ``supported``
  keeps its initial documented value False (the key the verdict path
  never reads; egress/local are gathered by separate rungs).

Each pin proves the behavior AND the non-recording: the two enumeration
handlers bind nothing and emit no log record (their STUN/HTTP siblings in
the same module do ``_log.debug`` — these do not).
"""

from __future__ import annotations

import logging
import socket

import pytest

from nomorals.security import netleak


class _FakeSock:
    def __init__(self, fam):
        self.fam = fam

    def settimeout(self, t):
        pass

    def connect(self, remote):
        if self.fam == socket.AF_INET6:
            raise OSError("no v6 route")

    def getsockname(self):
        return ("192.168.1.5", 0)

    def close(self):
        pass


class _DeadSock:
    def __init__(self, fam):
        self.fam = fam

    def settimeout(self, t):
        pass

    def connect(self, remote):
        raise OSError("unreachable")

    def getsockname(self):
        return ("0.0.0.0", 0)

    def close(self):
        pass


def test_177_getaddrinfo_crash_step2_still_enumerates(monkeypatch, caplog):
    monkeypatch.setattr(socket, "socket", lambda fam, kind: _FakeSock(fam))

    def gai_boom(host, port, *a, **k):
        raise OSError("dns boom")

    monkeypatch.setattr(socket, "getaddrinfo", gai_boom)
    with caplog.at_level(logging.DEBUG, logger="nomorals.security.netleak"):
        res = netleak.local_interface_ips()
    assert res == [{"ip": "192.168.1.5", "class": "rfc1918-private"}]
    # non-recording: the 177 handler emits nothing
    assert [r for r in caplog.records
            if r.name == "nomorals.security.netleak"] == []


def test_177_healthy_control_both_sources_land(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda fam, kind: _FakeSock(fam))
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port, *a, **k: [
            (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("10.0.0.9", 0))])
    res = netleak.local_interface_ips()
    assert {"ip": "10.0.0.9", "class": "rfc1918-private"} in res
    assert {"ip": "192.168.1.5", "class": "rfc1918-private"} in res


def test_189_all_remotes_fail_returns_empty(monkeypatch):
    monkeypatch.setattr(socket, "socket", lambda fam, kind: _DeadSock(fam))
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    assert netleak.local_interface_ips() == []


def test_203_missing_has_ipv6_supported_stays_false(monkeypatch):
    monkeypatch.setattr(netleak, "_http_text", lambda url, timeout=0: "")
    monkeypatch.setattr(netleak, "local_interface_ips", lambda: [])
    monkeypatch.delattr(socket, "has_ipv6")
    out = netleak.ipv6_egress()
    assert out["supported"] is False
    assert out["egress"] == "" and out["local"] == []


def test_203_healthy_control_supported_true(monkeypatch):
    monkeypatch.setattr(netleak, "_http_text", lambda url, timeout=0: "")
    monkeypatch.setattr(netleak, "local_interface_ips", lambda: [])
    monkeypatch.setattr(socket, "has_ipv6", True)
    assert netleak.ipv6_egress()["supported"] is True
