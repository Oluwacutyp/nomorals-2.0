"""Calibrated confidence UX (build-map #34) — all offline."""
import json

import pytest

from nomorals.agents.confidence import (
    Confidence,
    assess_confidence,
    format_with_confidence,
    HIGH_THRESHOLD,
    MEDIUM_THRESHOLD,
    VERY_LOW_THRESHOLD,
)
from nomorals.agents.orchestration.loop import AgenticLoop, LoopResult
from nomorals.agents.orchestration.tools import ToolAdapter


# ── assess_confidence ────────────────────────────────────────────────

def test_high_confidence_cited_tool_backed():
    resp = ("The repo has 936 capabilities [1]. Verified against the build "
            "map [2].")
    conf = assess_confidence(resp, tools_called=["read_file", "grep"])
    assert conf.level == "high"
    assert conf.score >= HIGH_THRESHOLD
    # high ships unchanged
    assert format_with_confidence(resp, conf) == resp


def test_citation_boost_capped():
    # 10 citations must not fake certainty on a hedged, tool-less answer
    resp = "Maybe it's " + " ".join(f"[{i}]" for i in range(1, 11))
    conf = assess_confidence(resp, tools_called=[])
    assert conf.score < HIGH_THRESHOLD


def test_verified_evidence_boosts():
    resp = "Funding hit $2B in 2024 [1]."
    plain = assess_confidence(resp, tools_called=[])
    verified = assess_confidence(
        resp, tools_called=[],
        evidence=[{"verified": True}, {"verified": False}])
    assert verified.score > plain.score


def test_hedged_response_medium_with_touch():
    resp = "The meeting is tomorrow afternoon at the usual place downtown."
    conf = assess_confidence(resp, tools_called=[])
    assert conf.level == "medium"
    out = format_with_confidence(resp, conf)
    assert out.startswith(resp)
    assert "double-check" in out


def test_short_reply_gets_no_touch():
    # chit-chat passes through clean — the touch is for answers with
    # enough content to be wrong about
    for short in ["HI", "done", "recovered", "Got it."]:
        conf = assess_confidence(short, tools_called=[])
        assert format_with_confidence(short, conf) == short


def test_uncited_numbers_low_with_prefix():
    resp = "Revenue was ₦2.5m in 2024, up 40%."
    conf = assess_confidence(resp, tools_called=[])
    assert conf.level == "low"
    out = format_with_confidence(resp, conf)
    assert out.startswith("I'm not sure about this — ")
    assert "what I'd check" in out


def test_very_low_idk_framing_drops_shaky_answer():
    resp = ("I don't know, maybe it's 42, or possibly 43, probably. "
            "I'm not sure.")
    conf = assess_confidence(resp, tools_called=[])
    assert conf.score < VERY_LOW_THRESHOLD
    out = format_with_confidence(resp, conf)
    assert out.startswith("I don't know — and I'd rather say so than guess.")
    assert "42" not in out  # shaky answer not shipped as fact
    assert "Here's what I'd check" in out


def test_threshold_boundaries():
    substantive = ("The quarterly report shows steady growth across all "
                   "three regional divisions this period.")
    assert format_with_confidence(
        substantive, Confidence(0.75, "high", [])).startswith("The quarterly")
    assert "double-check" in format_with_confidence(
        substantive, Confidence(0.74, "medium", []))
    assert "double-check" in format_with_confidence(
        substantive, Confidence(0.45, "medium", []))
    assert format_with_confidence(
        "x", Confidence(0.44, "low", [])).startswith("I'm not sure")
    assert format_with_confidence(
        "x", Confidence(0.24, "low", [])).startswith("I don't know")


def test_never_raises_on_weird_input():
    for bad in ["", "   ", "(no response produced)", None, 123, ["x"]]:
        conf = assess_confidence(bad)
        assert 0.0 <= conf.score <= 1.0
        out = format_with_confidence(str(bad or ""), conf)
        assert isinstance(out, str)


def test_check_suggestions_come_from_reasons():
    resp = "Revenue was ₦2.5m in 2024."
    conf = assess_confidence(resp, tools_called=[])
    out = format_with_confidence(resp, conf)
    assert "confirming the numbers" in out


# ── loop integration ─────────────────────────────────────────────────


class _Resp:
    def __init__(self, text):
        self.text = text


class _FakeLLM:
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0

    def chat(self, messages, params=None, **kw):
        idx = min(self.calls, len(self.scripts) - 1)
        self.calls += 1
        return _Resp(json.dumps(self.scripts[idx]))


class _FakeRegistry:
    def __init__(self):
        self._tools = {}

    def schemas(self, capabilities=None):
        return list(self._tools.values())

    def call(self, name, actor="system", capabilities=None, **kwargs):
        raise AssertionError("no tools registered")


def _run_loop(response_text, **loop_kw):
    from nomorals.agents.orchestration.tools import ToolAdapter
    reg = _FakeRegistry()
    llm = _FakeLLM([{"action": "respond", "response": response_text,
                     "thought": "done"}])
    adapter = ToolAdapter(reg)
    loop = AgenticLoop(llm, adapter, step_budget=3, **loop_kw)
    return loop.run("test question")


def test_loop_result_carries_confidence():
    result = _run_loop("The meeting is tomorrow afternoon at the usual "
                       "place downtown.")
    assert isinstance(result, LoopResult)
    assert result.confidence["level"] == "medium"
    assert 0.0 <= result.confidence["score"] <= 1.0
    assert "double-check" in result.response


class _FakeRegistryWithTool(_FakeRegistry):
    def get(self, name):
        return {"name": name} if name == "lookup" else None

    def schemas(self, capabilities=None):
        return [{"name": "lookup", "description": "look things up",
                 "capability": "", "parameters": {}}]

    def call(self, name, actor="system", capabilities=None, **kwargs):
        class _Ok:
            ok = True
            value = "936 capabilities"
        return _Ok()


def test_loop_high_confidence_unchanged():
    # tool-backed + cited: the real high-confidence path through the loop
    reg = _FakeRegistryWithTool()
    llm = _FakeLLM([
        {"action": "tool", "tool": "lookup", "args": {},
         "thought": "checking"},
        {"action": "respond",
         "response": "The repo has 936 capabilities [1] [2].",
         "thought": "done"},
    ])
    adapter = ToolAdapter(reg)
    loop = AgenticLoop(llm, adapter, step_budget=3)
    result = loop.run("how many capabilities?")
    assert result.confidence["level"] == "high"
    assert result.response == "The repo has 936 capabilities [1] [2]."


def test_loop_opt_out_returns_raw_response():
    resp = "I think maybe the meeting is tomorrow, probably."
    result = _run_loop(resp, confidence_ux=False)
    assert result.response == resp
    # default field, untouched by assessment
    assert result.confidence == {"score": 0.5, "level": "medium",
                                 "reasons": []}
