"""Multi-model adjudication: judge parsing, fan-out, one-shot. Offline."""
from __future__ import annotations

import pytest

from nomorals.llm.adjudicate import Judge, adjudicate, fan_out
from nomorals.llm.base import LLMResponse


def _ask_factory(text: str, model: str = "judge-7b"):
    def ask(messages, params, **kw):
        assert kw.get("task_kind") == "judge"
        return LLMResponse(text=text, model=model)
    return ask


def test_judge_picks_winner():
    ask = _ask_factory('{"winner": 2, "ranking": [2, 1], "rationale": "b is correct"}')
    j = Judge(ask).adjudicate("what is 2+2?", ["5", "4"])
    assert j.ok and j.winner == 1 and j.ranking == [1, 0]
    assert j.rationale == "b is correct" and j.judge_model == "judge-7b"


def test_judge_clamps_bad_winner():
    ask = _ask_factory('{"winner": 9, "ranking": [1], "rationale": "x"}')
    j = Judge(ask).adjudicate("q?", ["a", "b", "c"])
    assert j.winner == 0  # out of range → first candidate
    assert sorted(j.ranking) == [0, 1, 2]  # ranking completed


def test_judge_non_json_still_ranks():
    ask = _ask_factory("candidate one is clearly better, trust me")
    j = Judge(ask).adjudicate("q?", ["a", "b"])
    assert j.ok and j.winner == 0 and j.ranking == [0, 1]


def test_judge_single_candidate_short_circuits():
    calls = []
    j = Judge(lambda *a, **k: calls.append(1) or LLMResponse(text="x"))
    out = j.adjudicate("q?", ["only"])
    assert out.winner == 0 and not calls


def test_judge_no_candidates_is_error():
    out = Judge(lambda *a, **k: LLMResponse(text="x")).adjudicate("q?", [])
    assert not out.ok and out.winner == -1


def test_judge_ask_failure_never_raises():
    def boom(messages, params, **kw):
        raise RuntimeError("down")
    out = Judge(boom).adjudicate("q?", ["a", "b"])
    assert not out.ok and out.winner == 0


def test_judge_error_response_never_raises():
    ask = lambda *a, **k: LLMResponse(text="", error="all dead")
    out = Judge(ask).adjudicate("q?", ["a", "b"])
    assert not out.ok


def test_fan_out_collects_ok_only():
    def ask_one(i):
        if i == 1:
            return LLMResponse(text="", error="down")
        return LLMResponse(text=f"answer {i}")
    out = fan_out(ask_one, "q?", 3)
    assert sorted(out) == ["answer 0", "answer 2"]


def test_fan_out_survives_exceptions():
    def ask_one(i):
        raise RuntimeError("boom")
    assert fan_out(ask_one, "q?", 2) == []


def test_one_shot_adjudicate():
    out = adjudicate("q?", ["a", "b"], _ask_factory('{"winner": 1}'))
    assert out.winner == 0
