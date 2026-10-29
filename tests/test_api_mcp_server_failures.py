"""Tail-43 pins for nomorals/api/mcp_server.py — 0 converts /
8 exclusions, none flagged.

8 tests for 8 sites (counts equal). Each test pins one site's
fallback behavior plus a control half proving the except path was
taken (trigger == fallback AND control != fallback), asserting via
the double's own call log that the failing operation itself was
attempted. The file contains no exception handlers, so it adds
zero census sites.

mcp_server.py calls no failures-ledger record function anywhere
(grep-proven: no record_failure import or call), so NOT-recording
is pinned with the tail-8 snapshot method: agents recent_failures()
before/after each trigger must be unchanged.

Verdict notes (full grounds in the tail-43 SURVEY.md):
- 323: principal resolver failure -> the configured principal,
  the documented no-resolution fallback (register_mcp's mount
  comment); no consumer renders the principal as a claim.
- 363/386: notification/progress emit failure -> dropped;
  server->client notifications are fire-and-forget by spec and
  never gate the JSON-RPC response.
- 426/433: dispatch finally-block thread-local cleanup; the del
  IS the operation and AttributeError means the desired end
  state already holds (control-flow idiom).
- 771/783: completion fact-listing failure -> [], byte-identical
  to a genuinely empty memory; the rendered total counts the
  returned list itself. Near-unreachable: _list_active_facts is
  itself fail-soft.
- 887: one corrupt entity file is skipped; the resource payload
  is the parsed entities, all literally true, no count asserted.
"""
import json
import sys
from types import SimpleNamespace

sys.path.insert(0, "/home/hatch/workspace/nm-tail43-tree")

from nomorals.agents import failures as ag_fail
from nomorals.api.mcp_server import MCPServer
from nomorals.api.server import DEFAULT_PRINCIPAL


def _snap():
    return list(ag_fail.recent_failures())


def _srv(**kw):
    kw.setdefault("principal", DEFAULT_PRINCIPAL)
    return MCPServer(memory=None, **kw)


def test_323_principal_resolver_failure_falls_back():
    before = _snap()
    calls = []

    def bad_resolver():
        calls.append("resolve")
        raise RuntimeError("wiring boom")

    configured = SimpleNamespace(name="configured", grant=None)
    srv = _srv(principal=configured, principal_resolver=bad_resolver)
    assert srv.principal is configured
    assert calls == ["resolve"]  # the resolver itself was attempted
    other = SimpleNamespace(name="other", grant=None)
    srv2 = _srv(principal=configured, principal_resolver=lambda: other)
    assert srv2.principal is other  # control: a resolved value wins
    srv3 = _srv(principal=configured, principal_resolver=lambda: None)
    assert srv3.principal is configured  # control: None falls back too
    assert _snap() == before


def test_363_notify_emit_failure_is_dropped():
    before = _snap()
    srv = _srv()
    srv._log_level = "info"
    calls = []

    def bad_emit(note):
        calls.append(note)
        raise OSError("broken pipe")

    srv._emit_tls.emit = bad_emit
    srv._notify("info", "hello")  # must not raise
    assert len(calls) == 1
    assert calls[0]["method"] == "notifications/message"
    got = []
    srv._emit_tls.emit = got.append
    srv._notify("info", "hello")
    assert len(got) == 1 and got[0]["params"]["data"] == "hello"
    got.clear()
    srv._notify("debug", "quiet")  # below the set threshold
    assert got == []  # control: the gate, not the swallow, blocks this
    assert _snap() == before


def test_386_progress_emit_failure_is_dropped():
    before = _snap()
    srv = _srv()
    calls = []

    def bad_emit(note):
        calls.append(note)
        raise OSError("broken pipe")

    srv._emit_tls.emit = bad_emit
    srv._report_progress("tok", 1, message="running")  # must not raise
    assert len(calls) == 1
    assert calls[0]["method"] == "notifications/progress"
    assert calls[0]["params"]["progressToken"] == "tok"
    got = []
    srv._emit_tls.emit = got.append
    srv._report_progress("tok", 2)
    assert len(got) == 1  # control: a working emit receives it
    got.clear()
    srv._report_progress(None, 1)
    assert got == []  # control: no token -> no emit at all
    assert _snap() == before


def test_426_dispatch_principal_cleanup_survives_midflight_delete():
    before = _snap()
    srv = _srv()
    srv._log_level = "info"  # initialize emits via _notify
    override = SimpleNamespace(name="override", grant=None)
    seen = {}

    def deleting_emit(note):
        seen["during"] = srv.principal
        del srv._principal_tls.principal

    resp = srv.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25"}},
        deleting_emit, principal=override)
    # Without site 426's guard, the finally-block del would raise
    # AttributeError out of dispatch (it has no outer try).
    assert isinstance(resp, dict) and "result" in resp
    assert seen["during"] is override  # the override really was live
    assert getattr(srv._principal_tls, "principal", None) is None
    resp2 = srv.dispatch({"jsonrpc": "2.0", "id": 2, "method": "ping"},
                         lambda n: None, principal=override)
    assert "result" in resp2  # control: normal del path, same end state
    assert getattr(srv._principal_tls, "principal", None) is None
    assert _snap() == before


def test_433_dispatch_emit_cleanup_survives_midflight_delete():
    before = _snap()
    srv = _srv()
    srv._log_level = "info"

    def deleting_emit(note):
        del srv._emit_tls.emit

    resp = srv.dispatch(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25"}}, deleting_emit)
    assert isinstance(resp, dict) and "result" in resp
    assert getattr(srv._emit_tls, "emit", None) is None
    resp2 = srv.dispatch({"jsonrpc": "2.0", "id": 2, "method": "ping"},
                         lambda n: None)
    assert "result" in resp2  # control: normal del path, same end state
    assert getattr(srv._emit_tls, "emit", None) is None
    assert _snap() == before


def test_771_prompt_completion_failure_returns_empty():
    before = _snap()
    srv = _srv()
    calls = []

    def bad_facts(limit=100):
        calls.append(limit)
        raise RuntimeError("db down")

    srv._list_active_facts = bad_facts
    assert srv._complete_prompt_argument("devon-brief", "topic", "") == []
    assert calls == [50]  # the listing itself was attempted, as coded
    resp = srv.dispatch(
        {"jsonrpc": "2.0", "id": 9, "method": "completion/complete",
         "params": {"ref": {"type": "ref/prompt", "name": "devon-brief"},
                    "argument": {"name": "topic", "value": ""}}},
        lambda n: None)
    assert resp["result"]["completion"] == {
        "values": [], "total": 0, "hasMore": False}
    srv._list_active_facts = lambda limit=100: [
        {"text": "alpha"}, {"text": "beta"}, {"text": "alpha"}]
    out = srv._complete_prompt_argument("devon-brief", "topic", "")
    assert out == ["alpha", "beta"]  # control: deduped suggestions
    assert _snap() == before


def test_783_resource_completion_failure_returns_empty():
    before = _snap()
    srv = _srv()
    calls = []

    def bad_facts(limit=100):
        calls.append(limit)
        raise RuntimeError("db down")

    srv._list_active_facts = bad_facts
    out = srv._complete_resource_argument(
        "devon://memory/facts/{id}", "id", "")
    assert out == []
    assert calls == [100]  # the listing itself was attempted, as coded
    srv._list_active_facts = lambda limit=100: [
        {"id": "f1"}, {"id": "f2"}, {"id": ""}]
    out2 = srv._complete_resource_argument(
        "devon://memory/facts/{id}", "id", "")
    assert out2 == ["f1", "f2"]  # control: ids, empty id filtered
    assert _snap() == before


def test_887_entities_read_skips_corrupt_file(tmp_path, monkeypatch):
    before = _snap()
    edir = tmp_path / ".nomorals" / "memory" / "entities" / "person"
    edir.mkdir(parents=True)
    (edir / "bad.json").write_text("{not json")
    (edir / "good.json").write_text(json.dumps({"name": "Ada"}))
    monkeypatch.setenv("HOME", str(tmp_path))
    srv = _srv()
    # bad.json sorts before good.json: returning Ada proves the loop
    # continued past the failure instead of truncating.
    assert srv._read_entities() == [{"name": "Ada"}]
    (edir / "bad.json").write_text(json.dumps({"name": "Grace"}))
    out = srv._read_entities()  # control: both valid -> both returned
    assert out == [{"name": "Grace"}, {"name": "Ada"}]
    assert _snap() == before
