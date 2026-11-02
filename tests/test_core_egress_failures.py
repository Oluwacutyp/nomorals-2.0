"""Tail-104 pins for nomorals/core/egress.py — 0 converts / 4 exclusions.

Each test pins one census site's fallback behavior plus a control half.
Non-recording is structural: the module's only reporting channel is
``_log`` (grep-proven — no failure-ledger import); pins attach a capture
handler to ``_log`` and assert no ERROR record is emitted by the
exercised swallow paths.

Sites:
110 — ``is_ip_literal``: ``ipaddress.ip_address`` ValueError -> ``pass``,
the predicate falls through to the ``inet_aton`` check for disguised
literals (``127.1``, ``0x7f000001``). Try-parse control flow of a
boolean classifier; both parsers are real checks, nothing is claimed.
459 — ``kill_all``: ``sock.shutdown`` OSError -> ``pass`` during kill-
switch teardown.
463 — ``kill_all``: ``sock.close`` OSError -> ``pass``, same teardown.
(459+463 pinned together: two findings, one teardown duty — the kill
report is computed and logged BEFORE the closes and returned regardless;
a dying socket that refuses to close changes no outcome.)
780 — ``_one_hop`` finally: ``conn.close()`` Exception -> ``pass``
(``# noqa`` in source: "close must not mask the result"). Cleanup in a
finally block; the hop's real result/exception is already decided.
"""

from __future__ import annotations

import http.client
import logging
import urllib.parse
from unittest import mock

from nomorals.core import egress
from nomorals.core.egress import EgressClient, EgressController, is_ip_literal


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=1)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def errors(self) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno >= logging.ERROR]


def _capture_log():
    cap = _Capture()
    old_level = egress._log.level
    egress._log.addHandler(cap)
    egress._log.setLevel(1)
    return cap, old_level


def _release_log(cap: _Capture, old_level: int) -> None:
    egress._log.removeHandler(cap)
    egress._log.setLevel(old_level)


# ── 110: ip parse failure falls through to inet_aton ─────────────

def test_110_ip_literal_fallthrough():
    # dotted quad via ipaddress (first parser)
    assert is_ip_literal("127.0.0.1") is True
    # disguised forms: ipaddress raises ValueError (the swallowed site),
    # inet_aton still catches them — the fall-through IS the behavior
    assert is_ip_literal("127.1") is True
    assert is_ip_literal("0x7f000001") is True
    assert is_ip_literal("2130706433") is True
    # real hostnames fail both parsers
    assert is_ip_literal("example.com") is False
    assert is_ip_literal("999.999.999.999") is False
    assert is_ip_literal("") is False


# ── 459 + 463: kill-switch socket teardown swallows OSError ──────

class _BadSock:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self, how) -> None:
        self.shutdown_calls += 1
        raise OSError("socket already dead")

    def close(self) -> None:
        self.close_calls += 1
        raise OSError("socket already dead")


class _GoodSock:
    def __init__(self) -> None:
        self.shutdown_calls = 0
        self.close_calls = 0

    def shutdown(self, how) -> None:
        self.shutdown_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_459_463_kill_all_teardown_swallows_socket_errors():
    cap, old_level = _capture_log()
    try:
        controller = EgressController()
        bad, good = _BadSock(), _GoodSock()
        controller.track_socket(bad)  # type: ignore[arg-type]
        controller.track_socket(good)  # type: ignore[arg-type]
        report = controller.kill_all("pin reason")
    finally:
        _release_log(cap, old_level)
    # both teardowns were attempted on both sockets
    assert bad.shutdown_calls == 1 and bad.close_calls == 1
    assert good.shutdown_calls == 1 and good.close_calls == 1
    # the kill report is intact despite the bad socket's refusals
    assert report["scope"] == "global"
    assert report["reason"] == "pin reason"
    assert controller.killed is True
    # the kill was reported as a warning, never recorded as an error
    assert cap.errors() == []
    assert any(r.levelno == logging.WARNING for r in cap.records)


# ── 780: finally conn.close() must not mask the hop result ───────

class _FakeResp:
    status = 200

    def getheaders(self):
        return []

    def read(self, n: int = -1) -> bytes:
        return b""


class _FakeConn:
    sock = None

    def __init__(self, *a, **k) -> None:
        pass

    def connect(self) -> None:
        pass

    def putrequest(self, *a, **k) -> None:
        pass

    def putheader(self, *a, **k) -> None:
        pass

    def endheaders(self, *a, **k) -> None:
        pass

    def getresponse(self) -> _FakeResp:
        return _FakeResp()

    def close(self) -> None:
        raise RuntimeError("close boom")


def test_780_close_failure_does_not_mask_result():
    cap, old_level = _capture_log()
    try:
        client = EgressClient(controller=EgressController(),
                              cassette_mode="off")
        parsed = urllib.parse.urlparse("http://example.com/path")
        with mock.patch.object(http.client, "HTTPConnection", _FakeConn):
            status, headers, body, location = client._one_hop(
                object(), "GET", parsed, ["93.184.216.34"], {}, None, 5.0)
    finally:
        _release_log(cap, old_level)
    # the successful hop result survives the exploding close()
    assert status == 200
    assert headers == {}
    assert body == b""
    assert location == ""
    assert cap.errors() == []
