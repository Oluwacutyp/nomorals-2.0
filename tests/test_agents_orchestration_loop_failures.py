"""Tail-84 pins for nomorals/agents/orchestration/loop.py — 5 exclusions.

Each test pins one census site's fallback behavior plus a control half
proving the except path was taken. The file itself contains no
exception handlers, so it adds zero census sites.

loop.py calls no failures-ledger record function anywhere
(grep-proven: only _log), so NOT-recording is pinned with the snapshot
method: agency recent_failures() before/after each trigger must be
unchanged.

Verdict notes (full grounds in SURVEY.md):
- 136: get_profile_kind failure -> code_mode False, the documented
  safe default (code mode is a termux round-trip optimization; an
  explicit code_mode arg bypasses the consult entirely). The guard is
  reachable only when the direct consult fails after profile_value's
  own consult succeeded — __init__ consults get_profile_kind exactly
  twice, and the pin fails the second call only.
- 197: idempotency_scope assignment failure on a foreign adapter
  shape -> skip ("foreign adapter shape: skip"). ToolAdapter's own
  default scope is "" and derive is inert without a scope
  (tools.py:188), so the run proceeds with the unscoped default
  instead of crashing on an adapter it cannot stamp.
- 258: maybe_distill failure -> pass; the completed run is finalized
  and returned untouched ("distillation never breaks a run").
  Distillation is a side-benefit skill-draft hook; its failure
  falsifies nothing in the LoopResult.
- 479: extract_json failure -> the manual {...} extraction rung of
  the same ladder; a decision is returned only if a real dict parses.
- 502: json.loads failure inside the manual rung -> None, the
  documented "no decision" value; _think's caller records the step
  as failed ("model output was not valid JSON") and retries with a
  nudge — no fabricated decision ever reaches the run.
"""

import json

import nomorals.agency.failures as ag_fail
import nomorals.agents.orchestration.loop as loop_mod
import nomorals.agents.skill_distillation as sd
import nomorals.core.profiles as profiles
from nomorals.agents.orchestration.context import LoopMemory
from nomorals.agents.orchestration.loop import AgenticLoop
from nomorals.agents.orchestration.tools import ToolAdapter
from nomorals.core.policy import CapabilitySet
from nomorals.core.result import Err, Ok


def _snap():
    return list(ag_fail.recent_failures(limit=200))


class _FakeResponse:
    def __init__(self, text, ok=True):
        self.text = text
        self.ok = ok


class _TextLLM:
    def __init__(self, text):
        self._text = text

    def chat(self, messages, params=None, **kw):
        return _FakeResponse(self._text)


class _FakeRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, fn, description="", parameters=None):
        self._tools[name] = {"name": name, "fn": fn,
                             "description": description,
                             "capability": "",
                             "parameters": parameters or {}}

    def get(self, name):
        return self._tools.get(name)

    def schemas(self, capabilities=None):
        return list(self._tools.values())

    def call(self, name, actor="system", capabilities=None, **kwargs):
        spec = self._tools.get(name)
        if spec is None:
            return Err(Exception(f"unknown tool {name!r}"))
        return Ok(spec["fn"](**kwargs))


def _adapter():
    return ToolAdapter(_FakeRegistry(), capabilities=CapabilitySet.all())


def test_136_code_mode_profile_failure_defaults_off(monkeypatch):
    before = _snap()
    orig = profiles.get_profile_kind
    calls = {"n": 0}

    def flaky_kind():
        calls["n"] += 1
        if calls["n"] == 1:
            return orig()
        raise RuntimeError("profile boom")

    adapter = _adapter()
    adapter2 = _adapter()  # built pre-arm: ToolAdapter.__init__ itself
    # consults profile_value -> get_profile_kind (tools.py:166)
    monkeypatch.setattr(profiles, "get_profile_kind", flaky_kind)
    loop = AgenticLoop(_TextLLM("{}"), adapter, step_budget=3)
    assert calls["n"] == 2  # the guarded consult was the one that failed
    assert loop.code_mode is False
    # disarm before the control: monkeypatch reverts only at teardown,
    # and even an explicit-arg construction consults profile_value
    monkeypatch.setattr(profiles, "get_profile_kind", orig)
    # control: an explicit arg wins without any consult
    loop2 = AgenticLoop(_TextLLM("{}"), adapter2, step_budget=3,
                        code_mode=True)
    assert loop2.code_mode is True
    assert _snap() == before


class _FrozenAdapter(ToolAdapter):
    """ToolAdapter whose idempotency_scope assignment fails once armed."""

    def __setattr__(self, name, value):
        if name == "idempotency_scope" and getattr(self, "_armed", False):
            raise AttributeError("frozen adapter")
        super().__setattr__(name, value)


def test_197_foreign_adapter_scope_failure_skips():
    before = _snap()
    adapter = _FrozenAdapter(_FakeRegistry(),
                             capabilities=CapabilitySet.all())
    object.__setattr__(adapter, "_armed", True)
    llm = _TextLLM(json.dumps({"action": "respond", "response": "done"}))
    loop = AgenticLoop(llm, adapter, step_budget=3, code_mode=False,
                       confidence_ux=False)
    result = loop.run("hello")
    assert result.response == "done"
    assert _snap() == before


def test_258_distill_failure_never_breaks_run(monkeypatch):
    before = _snap()

    def boom_distill(*a, **k):
        raise RuntimeError("distill boom")

    monkeypatch.setattr(sd, "maybe_distill", boom_distill)
    llm = _TextLLM(json.dumps({"action": "respond",
                               "response": "final answer"}))
    loop = AgenticLoop(llm, _adapter(), step_budget=3, code_mode=False,
                       confidence_ux=False)
    result = loop.run("do a thing")
    assert result.response == "final answer"
    assert _snap() == before


def test_479_extract_failure_falls_to_manual_rung(monkeypatch):
    before = _snap()

    def boom_extract(text):
        raise RuntimeError("extract boom")

    monkeypatch.setattr(loop_mod, "extract_json", boom_extract)
    llm = _TextLLM('prose {"action": "respond", "response": "hi"} tail')
    loop = AgenticLoop(llm, _adapter(), step_budget=3, code_mode=False,
                       confidence_ux=False)
    decision = loop._think(LoopMemory(user_message="x"), 1)
    assert decision == {"action": "respond", "response": "hi"}
    assert _snap() == before


def test_502_manual_extract_bad_json_is_none():
    before = _snap()
    assert AgenticLoop._manual_json_extract("{not json}") is None
    assert AgenticLoop._manual_json_extract("no braces at all") is None
    # control: a balanced, valid block parses
    assert AgenticLoop._manual_json_extract('x {"a": 1} y') == {"a": 1}
    assert _snap() == before
