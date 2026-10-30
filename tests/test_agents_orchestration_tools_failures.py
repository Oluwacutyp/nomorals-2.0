"""Tail-58 pins for nomorals/agents/orchestration/tools.py —
0 converts / 6 exclusions / 1 flagged (47, 50, 58, 101, 199,
337, 367). The 199 pin asserts OBSERVED behaviour only (the
site is flagged in the tail-58 SURVEY, not blessed).

7 tests for 7 sites (counts equal). Each test pins one site
with a control half proving the guarded operation was actually
ATTEMPTED, asserting via the operation's own call log / double.
The file contains no exception handlers (raising doubles use
explicit `raise` inside functions, never except blocks), so it
adds zero census sites.

tools.py calls no failures-ledger record function anywhere
(grep-proven) and has no logger, so NOT-recording is pinned
against the agents package ledger
(nomorals/agents/failures.py): recent_failures() snapshot
before/after every trigger.

Verdict notes (full grounds in the tail-58 SURVEY.md):
- 47: _scrub_audit_text scrub pipeline failure -> str(text);
  the object's own string, literally true, rendered as a name
  or error message whose failure is carried by the outcome /
  failed flag, never as an absence or success claim.
- 50: inner rung: str(text) raises -> ""; no renderable text
  exists, "" is the genuine no-text value.
- 58: _scrub_audit_name name conversion raises -> ""; mirror
  of 50 for tool names; stored names and the guard's want
  name degrade through the same function, consistently.
- 101: _scrub_audit_args scrub failure -> {}; the function's
  documented no-scrubbed-args value (same as None / non-dict
  input). The live call already received the real args.
- 199: FLAGGED — _schemas registry.schemas() failure -> []
  and describe() renders "(no tools available)", the genuine
  empty-registry rendering. Pinned as observed only.
- 337: call_many registry.call_many failure -> serial
  fallback through the audited single-call path; every
  result is the tool's real outcome, only parallelism lost.
- 367: _format_result json.dumps failure (e.g. circular
  result) -> str(value), the value's own true rendering.
"""

from unittest import mock

import nomorals.agents.orchestration.tools as tools
from nomorals.agents.failures import recent_failures
from nomorals.core.result import Ok


def _snap():
    return list(recent_failures(200))


class _BadStr:
    def __init__(self):
        self.calls = 0

    def __str__(self):
        self.calls += 1
        raise RuntimeError("str boom")


# ── 47: scrub pipeline failure -> str(text) ────────────────────

def test_excluded_47_scrub_failure_returns_str():
    before = _snap()
    calls = []

    def bad_scrub(text):
        calls.append(text)
        raise RuntimeError("scrub boom")

    with mock.patch.object(tools, "scrub_secrets", bad_scrub):
        out = tools._scrub_audit_text("hello world")
    assert out == "hello world"
    assert calls == ["hello world"]  # scrub was attempted
    assert tools._scrub_audit_text("hello world") == "hello world"
    assert _snap() == before


# ── 50: str(text) raises -> "" ─────────────────────────────────

def test_excluded_50_unstringable_returns_empty():
    before = _snap()
    bad = _BadStr()
    assert tools._scrub_audit_text(bad) == ""
    # outer str() attempt + inner str() attempt both happened
    assert bad.calls == 2
    assert _snap() == before


# ── 58: name conversion raises -> "" ───────────────────────────

def test_excluded_58_unstringable_name_returns_empty():
    before = _snap()
    bad = _BadStr()
    assert tools._scrub_audit_name(bad) == ""
    assert bad.calls == 1  # conversion was attempted
    assert tools._scrub_audit_name("do_thing") == "do_thing"
    assert _snap() == before


# ── 101: args scrub failure -> {} ──────────────────────────────

def test_excluded_101_args_scrub_failure_returns_empty_dict():
    before = _snap()
    key = _BadStr()
    assert tools._scrub_audit_args({key: "v"}) == {}
    assert key.calls >= 1  # the key's str() was attempted
    assert tools._scrub_audit_args({"q": "hello"}) == {"q": "hello"}
    assert tools._scrub_audit_args({"api_key": "x"}) == {
        "api_key": "[REDACTED]"}
    assert _snap() == before


# ── 199: FLAGGED, observed only — schemas failure -> [] ────────

def test_flagged_199_schemas_failure_renders_no_tools():
    before = _snap()

    class RaisingReg:
        def __init__(self):
            self.calls = 0

        def schemas(self, capabilities=None):
            self.calls += 1
            raise RuntimeError("schemas boom")

        def get(self, name):
            return None

    reg = RaisingReg()
    adapter = tools.ToolAdapter(reg)
    assert adapter._schemas() == []
    assert reg.calls == 1  # schemas() was attempted
    # observed wrapper claim on the failure path
    assert adapter.describe() == "(no tools available)"

    class GoodReg:
        def schemas(self, capabilities=None):
            return [{"name": "do_thing", "description": "does",
                     "parameters": {}}]

        def get(self, name):
            return {"name": name}

    assert "do_thing" in tools.ToolAdapter(GoodReg()).describe()
    assert _snap() == before


# ── 337: call_many failure -> serial fallback ──────────────────

def test_excluded_337_call_many_failure_falls_back_serial():
    before = _snap()

    class CMReg:
        def __init__(self):
            self.cm_calls = 0
            self.call_calls = []

        def get(self, name):
            return {"name": name}

        def schemas(self, capabilities=None):
            return []

        def call_many(self, calls, max_workers=None, actor=None,
                      capabilities=None):
            self.cm_calls += 1
            raise RuntimeError("cm boom")

        def call(self, name, actor=None, capabilities=None, **kw):
            self.call_calls.append(name)
            return Ok(f"did {name}")

    reg = CMReg()
    adapter = tools.ToolAdapter(reg)
    res = adapter.call_many([("alpha", {}), ("beta", {})])
    assert reg.cm_calls == 1  # parallel path was attempted
    assert reg.call_calls == ["alpha", "beta"]  # serial re-attempt
    assert res == [(True, "did alpha"), (True, "did beta")]

    class CMOkReg(CMReg):
        def call_many(self, calls, max_workers=None, actor=None,
                      capabilities=None):
            self.cm_calls += 1
            return [Ok(f"par {n}") for n, _a in calls]

    reg2 = CMOkReg()
    res2 = tools.ToolAdapter(reg2).call_many([("alpha", {})])
    assert res2 == [(True, "par alpha")]
    assert reg2.call_calls == []  # control: no serial fallback
    assert _snap() == before


# ── 367: json.dumps failure -> str(value) ──────────────────────

def test_excluded_367_format_result_dumps_failure_uses_str():
    before = _snap()
    real_dumps = tools.json.dumps
    dump_calls = []

    def counting_dumps(obj, *a, **k):
        dump_calls.append(obj)
        return real_dumps(obj, *a, **k)

    circ = {}
    circ["self"] = circ
    with mock.patch.object(tools.json, "dumps", counting_dumps):
        out = tools.ToolAdapter._format_result(circ)
    assert len(dump_calls) == 1  # dumps was attempted
    assert out == str(circ)
    assert tools.ToolAdapter._format_result({"a": 1}) == '{"a": 1}'
    assert _snap() == before
