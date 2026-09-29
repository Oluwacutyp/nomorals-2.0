"""Wave 48 — the reasoning engine (fully hermetic).

- agents/reasoning.py: explicit, auditable, multi-strategy reasoning
  (cot, decompose, hypothesize, critique, tree, auto) with a typed
  trace, budgets, defensive JSON parsing, and a built-in eval
- wiring: the shared tool registry (``reason`` / ``reasoning_eval``),
  devon's tool catalog + handlers, the ``/think`` control command,
  ``nm reason`` CLI, the partner's automatic private reasoning pass,
  and the ``NM_PARTNER_REASONING`` config knob

Every model call is served by a scripted in-process router that replies
based on markers in the prompt — no network, no real provider, so the
tests prove the control flow (decomposition, backtracking, critique
rewrites, budget caps, trace integrity) of each strategy, not any model.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest

from nomorals.agents.reasoning import (
    ReasoningEngine,
    _extract_json,
    looks_complex,
    reasoning_eval,
    trace_text,
)
from nomorals.llm.base import LLMResponse


# ── scripted model ───────────────────────────────────────────────────────────


class FakeRouter:
    """Replies by matching a marker in the user prompt, in order.

    ``script`` maps a marker substring → reply.  The FIRST matching
    marker wins.  A queue of one-shot replies (``queue``) is drained
    before markers are consulted, which lets a test force a specific
    answer on the Nth call.
    """

    def __init__(self, script: dict[str, str] | None = None,
                 queue: list[str] | None = None) -> None:
        self.script = dict(script or {})
        self.queue = list(queue or [])
        self.calls = 0
        self.prompts: list[str] = []

    def chat(self, messages, params=None, **kw):
        prompt = messages[-1].content
        self.calls += 1
        self.prompts.append(prompt)
        if self.queue:
            return LLMResponse(text=self.queue.pop(0), model="fake")
        for marker, reply in self.script.items():
            if marker in prompt:
                return LLMResponse(text=reply, model="fake")
        return LLMResponse(text="ANSWER: default answer\nCONFIDENCE: 0.5",
                           model="fake")


class Ctx:
    """Minimal context: just a router (+ optional tool registry)."""

    def __init__(self, router, tools=None) -> None:
        self.router = router
        self.tools = tools


def _engine(router, **kw):
    kw.setdefault("max_llm_calls", 40)
    kw.setdefault("max_seconds", 60.0)
    return ReasoningEngine(Ctx(router), **kw)


# canonical structured replies the strategies ask for
_SUBGOALS = '{"subgoals": ["goal A", "goal B", "goal C"]}'
_HYPOTHESES = ('{"hypotheses": [{"claim": "the proxy was down", '
               '"why": "egress was direct"}, {"claim": "a filter fired", '
               '"why": "the target matched a block rule"}]}')
_SCORE = '{"supports": ["egress was direct"], "refutes": [], "score": 0.8}'
_SCORE_LOW = '{"supports": [], "refutes": ["the filter was disabled"], ' \
             '"score": 0.2}'
_CRIT_PASS = '{"passed": true, "flaws": []}'
_CRIT_FAIL = '{"passed": false, "flaws": ["skipped the error case", ' \
              '"assumed the disk was free"]}'
_BRANCHES = ('{"plan": "check the logs first", "branches": '
             '[{"action": "read the egress log", "expected": "the route"}, '
             '{"action": "test the proxy pool", "expected": "liveness"}]}')
_BRANCH_JUDGE = '{"viable": true, "value": 0.7, "reason": "gives evidence"}'
_BRANCH_JUDGE_DEAD = '{"viable": false, "value": 0.05, "reason": "a guess"}'
_TOOLCALL = '{"tool": "logs_tail", "args": {"n": "50"}}'
_COT = ("REASONING:\n1. compute 17*24 = 408\n2. compute 8*6 = 48\n"
        "3. add 408 + 48\nANSWER: 456\nCONFIDENCE: 0.92")


# ── JSON extraction ──────────────────────────────────────────────────────────


class ExtractJsonTest(unittest.TestCase):
    def test_raw_object(self):
        self.assertEqual(_extract_json('here {"a": 1} ok'), {"a": 1})

    def test_fenced(self):
        self.assertEqual(_extract_json("```json\n[1, 2]\n```"), [1, 2])

    def test_embedded_in_prose(self):
        text = "Sure! Here is the JSON:\n{\"subgoals\": [\"a\"]}\nHope that helps."
        self.assertEqual(_extract_json(text), {"subgoals": ["a"]})

    def test_braces_inside_strings(self):
        self.assertEqual(_extract_json('{"s": "a}b"}'), {"s": "a}b"})

    def test_nested(self):
        self.assertEqual(_extract_json('x {"a": {"b": [1, 2]}} y'),
                         {"a": {"b": [1, 2]}})

    def test_no_json(self):
        self.assertIsNone(_extract_json("no json here at all"))
        self.assertIsNone(_extract_json(""))
        self.assertIsNone(_extract_json(None))

    def test_malformed_then_valid(self):
        self.assertEqual(_extract_json("{bad json} then {\"ok\": 1}"),
                         {"ok": 1})


# ── the looks_complex heuristic ──────────────────────────────────────────────


class LooksComplexTest(unittest.TestCase):
    def test_short_greetings_are_not_complex(self):
        for t in ("hey", "ok", "thanks", "what did you do?"):
            self.assertFalse(looks_complex(t), t)

    def test_long_messages_are_complex(self):
        self.assertTrue(looks_complex("x" * 150))
        self.assertTrue(looks_complex("a" * 119))

    def test_multi_question_is_complex(self):
        self.assertTrue(looks_complex("why did it fail? also is it safe?"))

    def test_analytical_markers(self):
        self.assertTrue(looks_complex(
            "why did the egress check report direct when the proxy was up?"))
        self.assertTrue(looks_complex(
            "how should I architect this with the current limits?"))
        self.assertFalse(looks_complex("how are you?"))


# ── cot ──────────────────────────────────────────────────────────────────────


class CotTest(unittest.TestCase):
    def test_parses_steps_answer_confidence(self):
        e = _engine(FakeRouter(script={
            "QUESTION:": _COT,
        }))
        res = e.reason("what is 17*24 + 8*6?", strategy="cot")
        self.assertEqual(res.strategy, "cot")
        self.assertEqual(res.stopped, "complete")
        self.assertEqual(res.answer, "456")
        self.assertAlmostEqual(res.confidence, 0.92, places=2)
        # the numbered steps became trace notes
        notes = [s for s in res.trace if s.kind == "note"]
        self.assertTrue(any("408" in s.text for s in notes))
        # trace is ordered and has a verdict at the end
        self.assertEqual(res.trace[-1].kind, "verdict")
        self.assertEqual(res.trace[0].kind, "plan")

    def test_degrades_on_unformatted_reply(self):
        e = _engine(FakeRouter(script={
            "QUESTION:": "just a plain sentence with no format",
        }))
        res = e.reason("what is 2+2?", strategy="cot")
        self.assertEqual(res.stopped, "complete")
        self.assertEqual(res.answer, "just a plain sentence with no format")


# ── decompose ────────────────────────────────────────────────────────────────


class DecomposeTest(unittest.TestCase):
    def test_subgoals_and_synthesis(self):
        router = FakeRouter(script={
            "Decompose this": _SUBGOALS,
            "Synthesize the FINAL answer":
                "ANSWER: the combined plan\nCONFIDENCE: 0.88",
            "QUESTION:": _COT,
        })
        e = _engine(router)
        res = e.reason("design a backup system and then verify it",
                       strategy="decompose", depth=1)
        self.assertEqual(res.strategy, "decompose")
        self.assertEqual(res.subgoals, ["goal A", "goal B", "goal C"])
        self.assertEqual(res.answer, "the combined plan")
        # each subgoal produced an observation
        obs = [s for s in res.trace if s.kind == "observation"]
        self.assertEqual(len(obs), 3)
        # the subgoal work is preserved in the main trace (auditable)
        self.assertTrue(any(s.kind == "subgoal" for s in res.trace))

    def test_recursive_depth(self):
        router = FakeRouter(script={
            "Decompose this": _SUBGOALS,
            "Synthesize the FINAL answer":
                "ANSWER: done\nCONFIDENCE: 0.8",
            "QUESTION:": _COT,
        })
        e = _engine(router)
        res = e.reason("design a system and verify it",
                       strategy="decompose", depth=2)
        self.assertEqual(res.stopped, "complete")
        # deeper recursion makes strictly more model calls than depth 1
        router2 = FakeRouter(script={
            "Decompose this": _SUBGOALS,
            "Synthesize the FINAL answer":
                "ANSWER: done\nCONFIDENCE: 0.8",
            "QUESTION:": _COT,
        })
        _engine(router2).reason("design a system and verify it",
                                strategy="decompose", depth=1)
        self.assertGreater(router.calls, router2.calls)

    def test_degrades_when_decomposition_unparsable(self):
        e = _engine(FakeRouter(script={
            "Decompose this": "I cannot give you JSON",
            "QUESTION:": _COT,
        }))
        res = e.reason("plan a backup and verify it", strategy="decompose")
        self.assertEqual(res.subgoals, [])
        self.assertEqual(res.stopped, "complete")
        self.assertEqual(res.answer, "456")  # the cot fallback answer
        self.assertTrue(any(s.kind == "note" and "unparsable" in s.text
                            for s in res.trace))


# ── hypothesize ──────────────────────────────────────────────────────────────


class HypothesizeTest(unittest.TestCase):
    def test_scores_and_picks_best(self):
        router = FakeRouter(script={
            "Generate the most plausible": _HYPOTHESES,
            'Respond with JSON ONLY: {"supports"': _SCORE,
            "Given this evidence":
                "ANSWER: the proxy was down\nCONFIDENCE: 0.9",
        })
        e = _engine(router)
        res = e.reason("why did the egress check report direct?",
                       strategy="hypothesize")
        self.assertEqual(res.strategy, "hypothesize")
        self.assertEqual(len(res.hypotheses), 2)
        # both scored the same canned 0.8; the first is chosen (stable max)
        self.assertEqual(res.hypotheses[0]["score"], 0.8)
        self.assertEqual(res.hypotheses[1]["score"], 0.8)
        self.assertEqual(res.answer, "the proxy was down")
        self.assertTrue(any(s.kind == "observation" and "score" in s.text
                            for s in res.trace))

    def test_confidence_capped_when_hypotheses_real(self):
        e = _engine(FakeRouter(script={
            "Generate the most plausible": _HYPOTHESES,
            'Respond with JSON ONLY: {"supports"': _SCORE,
            "Given this evidence":
                "ANSWER: it failed\nCONFIDENCE: 1.0",
        }))
        res = e.reason("why did it fail?", strategy="hypothesize")
        self.assertLessEqual(res.confidence, 0.95)

    def test_degrades_when_hypotheses_unparsable(self):
        e = _engine(FakeRouter(script={
            "Generate the most plausible": "no structured output",
            "QUESTION:": _COT,
        }))
        res = e.reason("why did it fail?", strategy="hypothesize")
        self.assertEqual(res.hypotheses, [])
        self.assertEqual(res.answer, "456")  # fell back to a chain


# ── critique ────────────────────────────────────────────────────────────────


class CritiqueTest(unittest.TestCase):
    # NOTE on marker order: the critic and revise prompts also contain the
    # string "QUESTION:", so the strategy-specific markers must be listed
    # FIRST — the scripted router returns the first marker that matches.
    def test_passes_when_critic_agrees(self):
        e = _engine(FakeRouter(script={
            "harsh reviewer": _CRIT_PASS,
            "QUESTION:": _COT,
        }))
        res = e.reason("review this claim", strategy="critique")
        self.assertEqual(res.stopped, "complete")
        self.assertAlmostEqual(res.confidence, 0.92, places=2)
        self.assertTrue(any(s.kind == "critique" and "passed" in s.text
                            for s in res.trace))

    def test_rewrites_on_flaws(self):
        # critic finds flaws → a revision is requested → the critic is asked
        # again (it still disagrees) → the loop ends at the round cap with
        # the latest revision
        e = _engine(FakeRouter(script={
            "harsh reviewer": _CRIT_FAIL,
            "Revise the answer":
                "ANSWER: the fixed answer that handles the error case and "
                "checks disk first\nCONFIDENCE: 0.9",
            "QUESTION:": _COT,
        }))
        res = e.reason("review this design", strategy="critique")
        self.assertEqual(res.stopped, "complete")
        # a rewrite happened (the revision prompt was sent)
        self.assertTrue(any("Revise the answer" in p
                            for p in router_prompts(e)))
        # the flaws were recorded in the trace, and the revised answer is
        # what came out
        self.assertTrue(any(s.kind == "critique" and "flaws" in s.text
                            for s in res.trace))
        self.assertIn("fixed answer", res.answer)

    def test_stops_after_max_rounds(self):
        # critic never passes: the loop must terminate after the cap
        e = _engine(FakeRouter(script={
            "harsh reviewer": _CRIT_FAIL,
            "Revise the answer":
                "ANSWER: another try\nCONFIDENCE: 0.6",
            "QUESTION:": _COT,
        }))
        res = e.reason("review this", strategy="critique")
        self.assertEqual(res.stopped, "complete")
        # at most _MAX_CRITIQUE_ROUNDS critic prompts were sent
        critic_prompts = [p for p in e.context.router.prompts
                          if "harsh reviewer" in p]
        self.assertLessEqual(len(critic_prompts), 2)


def router_prompts(engine) -> list[str]:
    return engine.context.router.prompts


# ── tree (plan-branch-evaluate, incl. backtracking) ──────────────────────────


class TreeTest(unittest.TestCase):
    def test_picks_viable_branch_and_finishes(self):
        e = _engine(FakeRouter(script={
            "Plan the first move": _BRANCHES,
            "Judge this branch": _BRANCH_JUDGE,
            "Complete the answer":
                "ANSWER: read the log, the route was direct\nCONFIDENCE: 0.85",
        }))
        res = e.reason("which option should i pick to debug egress?",
                       strategy="tree")
        self.assertEqual(res.strategy, "tree")
        self.assertEqual(res.stopped, "complete")
        self.assertEqual(res.answer, "read the log, the route was direct")
        actions = [s for s in res.trace if s.kind == "action"]
        self.assertEqual(len(actions), 2)

    def test_backtracks_when_branch_dead(self):
        # branch 1 judged dead, branch 2 viable — the engine must choose 2.
        # The judge prompt echoes the branch's action text, so branch-
        # specific markers select each verdict (router = first match wins).
        e = _engine(FakeRouter(script={
            "Plan the first move": _BRANCHES,
            "BRANCH: read the egress log": _BRANCH_JUDGE_DEAD,  # dead end
            "BRANCH: test the proxy pool": _BRANCH_JUDGE,       # viable
            "Complete the answer":
                "ANSWER: tested the pool, one proxy was alive\n"
                "CONFIDENCE: 0.8",
        }))
        res = e.reason("debug the egress", strategy="tree")
        self.assertEqual(res.stopped, "complete")
        # the chosen branch was the viable one, not the dead first one
        chosen = [s for s in res.trace if s.kind == "observation"
                  and "chosen branch" in s.text]
        self.assertTrue(chosen)
        self.assertIn("test the proxy pool", chosen[0].text)

    def test_tool_mode_executes_action(self):
        # the engine's tools callable must return a STRING digest of the
        # tool result (that is the contract the registry/CLI callers use),
        # so the fake does the same json.dumps(outcome.value) conversion.
        class FakeTools:
            def __init__(self) -> None:
                self.calls = []

            def call(self, tool, actor="", **kw):
                self.calls.append((tool, kw))
                return json.dumps({"lines": ["route: direct"]})

        tools = FakeTools()
        e = _engine(FakeRouter(script={
            "Plan the first move": _BRANCHES,
            'Respond with JSON ONLY: {"tool"': _TOOLCALL,
            "Complete the answer":
                "ANSWER: done with evidence\nCONFIDENCE: 0.8",
        }), tools=tools)
        e.tools = tools.call  # action steps execute real tools
        res = e.reason("debug egress with tools", strategy="tree")
        self.assertEqual(res.stopped, "complete")
        self.assertTrue(tools.calls)
        self.assertEqual(tools.calls[0][0], "logs_tail")
        self.assertTrue(any(s.kind == "observation" and "route: direct"
                            in s.text for s in res.trace))

    def test_degrades_when_plan_unparsable(self):
        e = _engine(FakeRouter(script={
            "Plan the first move": "I cannot plan in JSON",
            "QUESTION:": _COT,
        }))
        res = e.reason("debug it", strategy="tree")
        self.assertEqual(res.answer, "456")  # fell back to a chain


# ── auto dispatch ────────────────────────────────────────────────────────────


class AutoTest(unittest.TestCase):
    def test_classifier(self):
        e = _engine(FakeRouter())
        self.assertEqual(e._classify("why did the proxy drop packets?"),
                         "hypothesize")
        self.assertEqual(e._classify("find the flaw in this argument"),
                         "critique")
        self.assertEqual(e._classify("which of these options is best"),
                         "tree")
        self.assertEqual(e._classify("design a backup system and verify it"),
                         "decompose")
        self.assertEqual(e._classify("what is 2+2?"), "cot")

    def test_auto_runs_a_full_trace(self):
        e = _engine(FakeRouter(script={
            "QUESTION:": _COT,
        }))
        res = e.reason("what is 17*24+8*6?")
        self.assertEqual(res.strategy, "cot")
        self.assertEqual(res.stopped, "complete")
        self.assertTrue(res.trace)


# ── budgets & trace integrity ────────────────────────────────────────────────


class BudgetTest(unittest.TestCase):
    def test_call_budget_exhausts(self):
        e = _engine(FakeRouter(script={
            "Generate the most plausible": _HYPOTHESES,
            'Respond with JSON ONLY: {"supports"': _SCORE,
        }), max_llm_calls=2, max_seconds=60)
        res = e.reason("why did it fail?", strategy="hypothesize")
        self.assertEqual(res.stopped, "budget")
        self.assertEqual(res.llm_calls, 2)
        self.assertTrue(any(s.kind == "note" and "budget" in s.text
                            for s in res.trace))

    def test_every_trace_is_a_plan_then_verdict(self):
        router = FakeRouter(script={
            "Generate the most plausible": _HYPOTHESES,
            'Respond with JSON ONLY: {"supports"': _SCORE,
            "Given this evidence": "ANSWER: x\nCONFIDENCE: 0.7",
        })
        e = _engine(router)
        res = e.reason("why?", strategy="hypothesize")
        # indices are contiguous and start at 0
        self.assertEqual([s.index for s in res.trace],
                         list(range(len(res.trace))))
        self.assertEqual(res.trace[0].kind, "plan")
        self.assertEqual(res.trace[-1].kind, "verdict")
        # every step has a confidence in range
        for s in res.trace:
            self.assertGreaterEqual(s.confidence, 0.0)
            self.assertLessEqual(s.confidence, 1.0)

    def test_trace_renders_for_a_human(self):
        e = _engine(FakeRouter(script={"QUESTION:": _COT}))
        res = e.reason("what is 17*24+8*6?", strategy="cot")
        text = trace_text(res)
        self.assertIn("reasoning [cot]", text)
        self.assertIn("ANSWER: 456", text)


# ── the built-in eval ────────────────────────────────────────────────────────


class EvalTest(unittest.TestCase):
    def test_eval_scores_tasks(self):
        # every task answered by a correct-enough canned reply
        router = FakeRouter(script={
            "harsh reviewer": _CRIT_PASS,
            "Synthesize the FINAL answer":
                ("ANSWER: compress the database dump in chunks, transfer "
                 "the chunks to a remote volume with enough space, then "
                 "restore it to a scratch instance to verify the backup "
                 "is usable\nCONFIDENCE: 0.9"),
            "Decompose this": _SUBGOALS,
            "Generate the most plausible": _HYPOTHESES,
            'Respond with JSON ONLY: {"supports"': _SCORE,
            "Given this evidence":
                ("ANSWER: most likely the proxy was down, because the "
                 "egress check reported a direct route when the pool "
                 "should carry all traffic\nCONFIDENCE: 0.9"),
            "QUESTION:": ("REASONING:\n1. work it\n"
                          "ANSWER: no — no bloops are lazzies because "
                          "every bloop is a razzie and no razzie is a "
                          "lazzie; compress then transfer to remote; the "
                          "statement is a liar paradox (self-"
                          "contradiction); deploy to production is last; "
                          "the proxy was down; 432 and 48 total\n"
                          "CONFIDENCE: 0.9"),
        })
        report = reasoning_eval(Ctx(router), limit=0)
        self.assertEqual(report["total"], 6)
        self.assertGreaterEqual(report["passed"], 4)
        self.assertGreaterEqual(report["score"], 0.6)
        # each task reports a strategy and a pass flag
        for t in report["tasks"]:
            self.assertIn("pass", t)
            self.assertIn("strategy", t)

    def test_eval_limit(self):
        router = FakeRouter(script={"QUESTION:": _COT,
                                    "Decompose this": _SUBGOALS,
                                    "Synthesize the FINAL answer":
                                        "ANSWER: compress, transfer, "
                                        "verify\nCONFIDENCE: 0.9"})
        report = reasoning_eval(Ctx(router), limit=2)
        self.assertEqual(report["total"], 2)


# ── wiring: registry, devon, control, config ─────────────────────────────────


class WiringTest(unittest.TestCase):
    def test_registry_registers_reason_tools(self):
        from nomorals.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register_builtins()
        names = set(registry._tools.keys())
        self.assertIn("reason", names)
        self.assertIn("reasoning_eval", names)

    def test_devon_catalog_and_handlers(self):
        from nomorals.agents.devon import TOOL_CATALOG, DevonAgent

        names = {n for n, _ in TOOL_CATALOG}
        self.assertIn("reason", names)
        self.assertIn("reasoning_eval", names)
        # the handler methods devon dispatches to actually exist
        self.assertTrue(hasattr(DevonAgent, "_tool_reason"))
        self.assertTrue(hasattr(DevonAgent, "_tool_reasoning_eval"))

    def test_control_command_think_registered(self):
        from nomorals.social.chat.control import CONTROL_COMMANDS

        self.assertIn("think", CONTROL_COMMANDS)
        min_args, max_args = CONTROL_COMMANDS["think"]
        self.assertEqual(min_args, 1)

    def test_control_parses_think(self):
        from nomorals.social.chat.control import parse_control

        c = parse_control("/think why did the proxy drop?")
        self.assertIsNotNone(c)
        self.assertEqual(c.kind, "think")
        c2 = parse_control("/think design a backup system critique")
        self.assertEqual(c2.kind, "think")
        self.assertIn("critique", c2.tail)

    def test_config_reasoning_knob(self):
        from nomorals.core.config import PartnerSettings

        self.assertEqual(PartnerSettings().reasoning, "auto")
        p = PartnerSettings(reasoning="always")
        self.assertEqual(p.reasoning, "always")

    def test_env_maps_reasoning(self):
        from nomorals.core.config import env_var_path

        self.assertEqual(env_var_path("NM_PARTNER_REASONING"),
                         "partner.reasoning")


if __name__ == "__main__":
    unittest.main()
