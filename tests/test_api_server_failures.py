"""Tail-99 pins: nomorals/api/server.py census sites.

All three sites are the API surface's degrade-don't-die law,
printed at each site: a health probe must answer, and a
supervisor's stop signal must drain the server — neither may
become a 500 or a hang. All held as exclusions:

605 — ``GET /health``: ``router.providers()`` crash ->
``providers: []``. The payload's ``ok`` claims the API server
itself (which demonstrably answered); the providers list is a
best-effort enrichment degrading to the same [] a router-less
context honestly reports. The comment at the site states the
contract: "health must degrade, not 500".
631 — ``GET /health``: the partner-runtime beacon read crash
-> ``{"state": "unknown"}``. The failure value itself declares
ignorance — the opposite of a fabricated state; a working read
still reports stopped/alive/stale with the beacon age.
1495 — ``serve()``: ``except KeyboardInterrupt: pass`` with
teardown in ``finally`` (E103 + E106 on one statement). The
E106 rule carves out exactly this: a deliberate top-level
shutdown hook. SIGTERM is mapped onto KeyboardInterrupt by
``install_sigterm_as_interrupt`` so supervisors drain the same
way Ctrl-C does; the finally block calls shutdown() +
server_close() before serve returns 0.

NOT-recording is pinned with the snapshot method: server.py
calls no failures-ledger record function anywhere, so both the
agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.beacon as beacon
import nomorals.agents.failures as agents_fail
import nomorals.api.server as S
from nomorals.api.server import APIServer


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


def _context(router=None, home=""):
    return SimpleNamespace(
        db=None, router=router, tools=None, bus=None, memory=None,
        settings=SimpleNamespace(
            api=SimpleNamespace(token="", max_body_mb=0), home=home))


# ── 605: providers() crash -> providers [] ──────────────────────

def test_605_providers_crash_degrades_to_empty():
    before = _snap()
    srv = APIServer(_context(router=SimpleNamespace(providers=_raise)),
                    token="")
    status, payload = srv.dispatch("GET", "/health", {}, {})
    assert status == 200
    assert payload["ok"] is True
    assert payload["providers"] == []
    # control: a healthy router's providers are reported verbatim,
    # and a router-less context reports the same [] honestly
    srv = APIServer(_context(router=SimpleNamespace(
        providers=lambda: ["p1", "p2"])), token="")
    status, payload = srv.dispatch("GET", "/health", {}, {})
    assert status == 200 and payload["providers"] == ["p1", "p2"]
    srv = APIServer(_context(router=None), token="")
    status, payload = srv.dispatch("GET", "/health", {}, {})
    assert status == 200 and payload["providers"] == []
    assert _snap() == before


# ── 631: beacon read crash -> runtime state "unknown" ───────────

def test_631_beacon_crash_reports_unknown_not_a_state():
    before = _snap()
    with mock.patch.object(beacon, "read_status", _raise):
        srv = APIServer(_context(home="/anywhere"), token="")
        status, payload = srv.dispatch("GET", "/health", {}, {})
    assert status == 200
    assert payload["runtime"] == {"state": "unknown"}
    # control: a working beacon read is reported faithfully
    with mock.patch.object(beacon, "read_status",
                           lambda home: ({"stopped": True}, 2.5)):
        srv = APIServer(_context(home="/anywhere"), token="")
        status, payload = srv.dispatch("GET", "/health", {}, {})
    assert status == 200
    assert payload["runtime"] == {"state": "stopped", "beacon_age_s": 2.5}
    assert _snap() == before


# ── 1495: KeyboardInterrupt is the deliberate shutdown hook ─────

class _FakeHTTPD:
    instances: list = []

    def __init__(self, addr, handler):
        self.shutdown_called = False
        self.close_called = False
        _FakeHTTPD.instances.append(self)

    def serve_forever(self):
        raise KeyboardInterrupt

    def shutdown(self):
        self.shutdown_called = True

    def server_close(self):
        self.close_called = True


class _RuntimeHTTPD(_FakeHTTPD):
    def serve_forever(self):
        raise RuntimeError("serve boom")


def _serve_ctx():
    return SimpleNamespace(
        db=None, tools=None,
        settings=SimpleNamespace(
            api=SimpleNamespace(token="t", max_body_mb=0)))


def test_1495_keyboard_interrupt_drains_and_returns_zero():
    before = _snap()
    _FakeHTTPD.instances = []
    with mock.patch.object(S, "ThreadingHTTPServer", _FakeHTTPD), \
            mock.patch(
                "nomorals.core.shutdown.install_sigterm_as_interrupt",
                lambda: True):
        rc = S.serve(_serve_ctx(), token="tok", protocols=False)
    assert rc == 0
    httpd = _FakeHTTPD.instances[-1]
    assert httpd.shutdown_called is True
    assert httpd.close_called is True
    # control: the hook swallows ONLY KeyboardInterrupt — a real
    # serve failure still propagates (teardown still runs)
    _FakeHTTPD.instances = []
    with mock.patch.object(S, "ThreadingHTTPServer", _RuntimeHTTPD), \
            mock.patch(
                "nomorals.core.shutdown.install_sigterm_as_interrupt",
                lambda: True):
        try:
            S.serve(_serve_ctx(), token="tok", protocols=False)
        except RuntimeError as exc:
            assert isinstance(exc, Exception)
        else:
            raise AssertionError("RuntimeError should propagate")
    assert _FakeHTTPD.instances[-1].shutdown_called is True
    assert _snap() == before
