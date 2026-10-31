"""Tail-75 pins for nomorals/llm/local_server.py —
0 converts / 6 exclusions (181, 184, 813, 856, 864, 875).

6 tests for 6 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED (healthy fake returns a distinctive value and the
surface shows it), then the failure half asserting the pinned
degradation. The file contains no exception handlers, so it
adds zero census sites.

Verdict notes (full grounds in the tail-75 SURVEY.md):
- 181: a 200 with a non-JSON body reads "ok" — the module's
  documented contract (non-JSON 200 = a live test double);
  an EMPTY body still reads "dead", so the fallback needs
  positive evidence of life. NOT-recording pinned.
- 184: any transport failure reads "dead" — the stated
  contract ("dead when nothing answers at all"); consumers
  (lifecycle readiness) treat it as not-ready, the honest
  direction. NOT-recording pinned.
- 813: stop() teardown — terminate/wait/kill ladder, generic
  failure swallowed AFTER the handle is detached; teardown
  family (drift 348 precedent). The manager keeps no claim
  over the process afterwards; status() re-derives reality
  from the port scan.
- 856: core-count probe failure -> 4, the documented sizing
  default (a thread-count hint, nothing decides on it).
- 864: machine probe failure -> False (not-ARM), the
  conservative default for perf-flag selection.
- 875: stdout-tail read failure -> "" — byte-identical to a
  silent process; the tail is diagnostics only.
"""

import logging
from unittest import mock

from nomorals.llm import local_server as ls

LS_LOG = "nomorals.llm.local_server"


class _Resp:
    def __init__(self, ok, text):
        self.ok = ok
        self.text = text


class _Client:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc

    def get(self, url, timeout=None):
        if self._exc is not None:
            raise self._exc
        return self._resp


def _client_factory(client):
    return lambda timeout=None: client


def test_health_status_non_json_200_with_body_is_ok(caplog):
    # Control: a JSON health body is read verbatim.
    with mock.patch.object(
            ls, "HttpClient",
            _client_factory(_Client(_Resp(True, '{"status": "loading"}')))):
        assert ls.health_status("h", 1) == "loading"
    # Failure half (site 181): non-JSON 200 WITH a body -> "ok";
    # with an EMPTY body -> "dead". Neither is recorded.
    with mock.patch.object(
            ls, "HttpClient",
            _client_factory(_Client(_Resp(True, "garbage")))):
        with caplog.at_level(logging.DEBUG, logger=LS_LOG):
            caplog.clear()
            assert ls.health_status("h", 1) == "ok"
    with mock.patch.object(
            ls, "HttpClient",
            _client_factory(_Client(_Resp(True, "")))):
        assert ls.health_status("h", 1) == "dead"
    assert [r for r in caplog.records if r.name == LS_LOG] == []


def test_health_status_transport_failure_is_dead(caplog):
    # Control: a "busy" body passes through untouched.
    with mock.patch.object(
            ls, "HttpClient",
            _client_factory(_Client(_Resp(True, '{"status": "busy"}')))):
        assert ls.health_status("h", 1) == "busy"
    # Failure half (site 184): no answer of any kind -> "dead",
    # and the failure is not recorded on the module logger.
    with mock.patch.object(
            ls, "HttpClient",
            _client_factory(_Client(exc=RuntimeError("refused")))):
        with caplog.at_level(logging.DEBUG, logger=LS_LOG):
            caplog.clear()
            assert ls.health_status("h", 1) == "dead"
    assert [r for r in caplog.records if r.name == LS_LOG] == []


class _BadProc:
    pid = 999

    def terminate(self):
        raise RuntimeError("already gone weirdly")

    def wait(self, timeout=None):
        return None


def test_stop_swallows_terminate_failure_and_detaches():
    mgr = ls.GGUFServerManager.__new__(ls.GGUFServerManager)
    mgr._process = _BadProc()
    # Site 813: stop() does not raise and the handle is detached.
    assert mgr.stop() is None
    assert mgr._process is None


def test_physical_cores_failure_defaults_to_4():
    # Control: the real probe answers a positive count.
    assert ls._physical_cores() >= 1
    # Failure half (site 856): both count sources raise -> 4.
    with mock.patch.object(ls.os, "process_cpu_count",
                           side_effect=RuntimeError("nope"),
                           create=True), \
         mock.patch.object(ls.os, "cpu_count",
                           side_effect=RuntimeError("nope")):
        assert ls._physical_cores() == 4


def test_is_arm_failure_defaults_false():
    # Control: the real probe answers a bool without raising.
    assert isinstance(ls._is_arm(), bool)
    # Failure half (site 864): machine probe raises -> False.
    with mock.patch.object(ls.platform, "machine",
                           side_effect=RuntimeError("nope")):
        assert ls._is_arm() is False


class _BadStdout:
    def read(self):
        raise RuntimeError("pipe broke")


class _GoodStdout:
    def read(self):
        return b"some server log tail"


class _Proc:
    def __init__(self, stdout):
        self.stdout = stdout


def test_drain_tail_failure_returns_empty():
    # Control: a readable pipe yields its decoded tail.
    assert ls._drain_tail(_Proc(_GoodStdout())) == "some server log tail"
    # Failure half (site 875): a broken pipe reads as "".
    assert ls._drain_tail(_Proc(_BadStdout())) == ""
