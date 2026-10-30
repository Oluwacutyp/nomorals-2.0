"""Tail-66 pins for nomorals/api/acp.py —
0 converts / 6 exclusions, 1 flagged-and-resolved-by-precedent
(259). Each test pins one site with a control half proving
the guarded operation was actually ATTEMPTED via the
double's own call log. The file contains no exception
handlers, so it adds zero census sites.

acp.py binds + logs its dispatch errors but calls no
failures-ledger record function at any census site, so
NOT-recording is pinned against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-66 SURVEY.md):
- 147 / 154: _safe_json's docstring IS the contract —
  "Never raises"; the truncated-str fallback is the
  designed degraded payload for the notification surface
  (rawInput/rawOutput are display data; nothing decides
  on them).
- 259: a raising principal resolver falls back to the
  server's CONFIGURED principal — the identical guard in
  the sibling api/mcp_server.py was adjudicated in
  tail-43 and pinned as designed behaviour
  (test_api_mcp_server_failures.py::test_323). The
  fallback target is bounded by configuration
  (DEFAULT_PRINCIPAL is a read-mostly grant, "never a
  silent CapabilitySet.all()"), never an escalation
  beyond what the operator configured.
- 412: outcome inspection for the tool_call_update
  notification is tracing only — the real outcome object
  reaches the loop unmodified whatever inspection does.
- 645: thread-local cleanup in dispatch's finally; the
  AttributeError only fires when the override is already
  gone, which is the state the cleanup exists to reach.
- 870: a disconnected SSE client is the end of the
  stream by definition; there is no second channel to
  report the failure on.
"""

import json
from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as agent_fail
from nomorals.agents import orchestration as orch_mod
from nomorals.api import acp as acp_mod
from nomorals.api.acp import ACPServer, ACPSession, _safe_json


def _snap():
    return list(agent_fail.recent_failures(200))


def test_147_safe_json_dumps_failure_returns_str():
    before = _snap()
    circ = []
    circ.append(circ)
    assert _safe_json(circ) == str(circ)[:4000] == "[[...]]"
    assert _safe_json({"a": 1}) == {"a": 1}
    assert _snap() == before


def test_154_safe_json_loads_failure_returns_str():
    before = _snap()
    loads_calls = []

    def boom_loads(*a, **k):
        loads_calls.append(1)
        raise ValueError("loads dead")

    with mock.patch.object(acp_mod.json, "loads", boom_loads):
        out = _safe_json({"a": 1})
    assert out == str({"a": 1})[:4000]
    assert loads_calls == [1]
    assert _safe_json({"a": 1}) == {"a": 1}
    assert _snap() == before


def test_259_principal_resolver_failure_falls_back():
    before = _snap()
    configured = SimpleNamespace(name="configured", grant="G")
    calls = []

    def bad_resolver():
        calls.append("resolve")
        raise RuntimeError("wiring boom")

    srv = ACPServer(principal=configured, principal_resolver=bad_resolver)
    assert srv.principal is configured
    assert calls == ["resolve"]
    other = SimpleNamespace(name="other", grant="G2")
    srv2 = ACPServer(principal=configured,
                     principal_resolver=lambda: other)
    assert srv2.principal is other
    srv3 = ACPServer(principal=configured,
                     principal_resolver=lambda: None)
    assert srv3.principal is configured
    assert _snap() == before


class _BadVal:
    def __str__(self):
        raise RuntimeError("no str")


class _FakeRegistry:
    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = []

    def call(self, name, *a, **k):
        self.calls.append(name)
        return self.outcome

    def schemas(self, capabilities=None):
        return []


def _fake_run_factory(box):
    def fake_run(prompt, *, llm, registry, history, actor, capabilities,
                 resume_from):
        box["got"] = registry.call("dummy_tool")
        return SimpleNamespace(response="done",
                               memory_snapshot={"m": 1},
                               asked_user=False, success=True, question="")
    return fake_run


def _updates(emitted):
    return [n["params"]["update"] for n in emitted
            if n["params"]["update"].get("sessionUpdate")
            == "tool_call_update"]


def test_412_traced_call_inspection_failure_keeps_turn():
    before = _snap()
    box = {}
    bad_outcome = SimpleNamespace(ok=True, value=_BadVal())
    server = ACPServer(
        context=SimpleNamespace(llm=object(), tools=None),
        registry=_FakeRegistry(bad_outcome),
        principal=SimpleNamespace(name="t", grant={"x"}))
    emitted = []
    with mock.patch.object(orch_mod, "run_agentic", _fake_run_factory(box)):
        ret = server._default_turn(ACPSession(id="s1", cwd="/tmp"), "hi",
                                   emitted.append)
    assert ret == "done"
    assert box["got"] is bad_outcome
    ups = _updates(emitted)
    assert len(ups) == 1
    assert ups[0]["status"] == "completed" and ups[0]["content"] == []
    assert "rawOutput" not in ups[0]
    box2 = {}
    good_outcome = SimpleNamespace(ok=True, value={"r": 1})
    server2 = ACPServer(
        context=SimpleNamespace(llm=object(), tools=None),
        registry=_FakeRegistry(good_outcome),
        principal=SimpleNamespace(name="t", grant={"x"}))
    emitted2 = []
    with mock.patch.object(orch_mod, "run_agentic",
                           _fake_run_factory(box2)):
        ret2 = server2._default_turn(ACPSession(id="s2", cwd="/tmp"), "hi",
                                     emitted2.append)
    assert ret2 == "done"
    assert box2["got"] is good_outcome
    ups2 = _updates(emitted2)
    assert ups2[0]["status"] == "completed"
    assert ups2[0]["rawOutput"] == {"r": 1}
    assert ups2[0]["content"][0]["content"]["text"] == str({"r": 1})
    assert _snap() == before


def test_645_dispatch_tls_cleanup_tolerates_missing_principal():
    before = _snap()
    server = ACPServer(turn_fn=lambda s, t, e: "warmup")
    sid = server.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new",
         "params": {"cwd": "/tmp"}}, lambda n: None)["result"]["sessionId"]

    def deleting_turn(session, text, emit):
        del server._principal_tls.principal
        return "ok-text"

    server._turn_fn = deleting_turn
    resp = server.dispatch(
        {"jsonrpc": "2.0", "id": 2, "method": "session/prompt",
         "params": {"sessionId": sid,
                    "prompt": [{"type": "text", "text": "hi"}]}},
        lambda n: None, principal=SimpleNamespace(name="p", grant=None))
    assert resp["result"] == {"stopReason": "end_turn"}
    assert getattr(server._principal_tls, "principal", None) is None
    assert _snap() == before


class _FakeWFile:
    def __init__(self, boom):
        self.boom = boom
        self.writes = []

    def write(self, b):
        if self.boom:
            raise BrokenPipeError("client gone")
        self.writes.append(b)
        return len(b)

    def flush(self):
        pass


class _FakeHandler:
    def __init__(self, boom):
        self.wfile = _FakeWFile(boom)
        self.codes = []
        self._request_id = ""

    def send_response(self, code):
        self.codes.append(code)

    def send_header(self, k, v):
        pass

    def end_headers(self):
        pass


def test_870_sse_broken_pipe_ends_stream_quietly():
    before = _snap()
    server = ACPServer(turn_fn=lambda s, t, e: "x")
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "session/new",
                       "params": {"cwd": "/tmp"}}).encode()
    api_server = SimpleNamespace(_inflight_principal=None)
    h1 = _FakeHandler(boom=True)
    acp_mod.serve_acp_sse(h1, api_server, server, body)
    assert h1.codes == [200]
    h2 = _FakeHandler(boom=False)
    acp_mod.serve_acp_sse(h2, api_server, server, body)
    assert h2.codes == [200]
    assert any(b"sessionId" in w for w in h2.wfile.writes)
    assert _snap() == before
