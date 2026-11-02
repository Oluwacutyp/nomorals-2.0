"""Tail-114 pins: nomorals/llm/adjudicate.py census sites (352, 386, 444).

All three are held as exclusions. ``Judge`` total failures (ask
raising, response not ok, empty text) already return ``ok=False``
fallback judgments — the two ``_parse`` guards are per-field leniency
inside a dict-shaped, successful judge response:

352 — winner field unparseable -> winner stays 0, the parser's
     initialized default. Fenced at the aggregation surfaces:
     ``pairwise`` runs both orderings, so a positional default maps
     to opposite candidates and comes back as an inconsistency tie
     (winner=-1, confidence 0.35) rather than a win; ``panel``
     recomputes its winner/confidence from vote tallies.
386 — confidence field unparseable -> 0.8, the module's standing
     default (initialized before parsing; a missing field yields
     the identical value on the legitimate path).
444 — ``fan_out`` worker: a provider raising contributes nothing.
     The docstring prints the contract: "Never raises — failed
     providers contribute nothing." Absence-shaped, not success-
     shaped; consumers receive a shorter list, never a fabricated
     text.
"""

from __future__ import annotations

import json

from nomorals.llm.adjudicate import Judge, fan_out
from nomorals.llm.base import LLMResponse


def _ask_returning(payload):
    def _ask(messages, params, **kw):
        return LLMResponse(text=json.dumps(payload), model="judge-x")
    return _ask


# ── 352: unparseable winner defaults to candidate 0 ──────────────────

def test_352_unparseable_winner_defaults_to_first_candidate():
    judge = Judge(_ask_returning({"winner": "candidate B",
                                  "confidence": 0.3}))
    out = judge.adjudicate("q?", ["a", "b"])
    assert out.ok
    assert out.winner == 0


def test_352_control_parseable_winner_is_honored():
    judge = Judge(_ask_returning({"winner": 2, "confidence": 0.3}))
    out = judge.adjudicate("q?", ["a", "b"])
    assert out.winner == 1


def test_352_pairwise_default_becomes_a_tie_not_a_win():
    judge = Judge(_ask_returning({"winner": "candidate B"}))
    out = judge.pairwise("q?", "a", "b")
    assert out.ok
    assert out.consistent is False
    assert out.winner == -1


# ── 386: unparseable confidence defaults to 0.8 ──────────────────────

def test_386_unparseable_confidence_defaults_to_point_eight():
    judge = Judge(_ask_returning({"winner": 1, "confidence": "high"}))
    out = judge.adjudicate("q?", ["a", "b"])
    assert out.ok
    assert out.confidence == 0.8


def test_386_control_parseable_confidence_is_kept():
    judge = Judge(_ask_returning({"winner": 1, "confidence": 0.42}))
    out = judge.adjudicate("q?", ["a", "b"])
    assert out.confidence == 0.42


def test_386_missing_confidence_yields_the_same_default():
    judge = Judge(_ask_returning({"winner": 1}))
    out = judge.adjudicate("q?", ["a", "b"])
    assert out.confidence == 0.8


# ── 444: fan_out provider failures contribute nothing ────────────────

def _bad_ask(i):
    raise RuntimeError("provider down")


def _good_ask(i):
    return LLMResponse(text=f"text-{i}", model="m")


def test_444_all_providers_fail_returns_empty():
    assert fan_out(_bad_ask, "p", 3) == []


def test_444_failed_provider_contributes_nothing_others_stand():
    out = fan_out(lambda i: _bad_ask(i) if i == 1 else _good_ask(i), "p", 3)
    assert sorted(out) == ["text-0", "text-2"]


def test_444_control_all_providers_answer():
    out = fan_out(_good_ask, "p", 3)
    assert sorted(out) == ["text-0", "text-1", "text-2"]
