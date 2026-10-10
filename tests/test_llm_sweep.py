"""Sweep tests for the nomorals.llm system-wide upgrade.

Covers the new capability added in the llm sweep: native chat templates,
full-jitter backoff, model-card tags/pricing, cost-per-task, case-facts
context strategies, versioned prompt library, debiased adjudication,
download validation, router SLO shedding / key pools / budgets / route
knobs, broker escalation & quality-tier, brain escalation & best-of-N,
provider usage details, registry pricing selectors, and cost display.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("NM_COST_LOG", "off")

from nomorals.llm.adjudicate import adjudicate as adjudicate_fn
from nomorals.llm.adjudicate import Judge, format_judgment, pairwise
from nomorals.llm.base import (
    CHAT_TEMPLATES,
    LLMResponse,
    Message,
    Usage,
    detect_template,
    messages_to_text,
)
from nomorals.llm.benchmarks import BenchmarkDB
from nomorals.llm.brain import Brain, _with_quality, estimate_complexity
from nomorals.llm.broker import BrokerConstraints, ModelBroker
from nomorals.llm.capabilities import Capability, ModelCard
from nomorals.llm.context_fit import (
    CaseFacts,
    PrioritizeEnds,
    fit_messages,
    strategy_for,
    trim_tool_results,
    with_cache_breakpoints,
)
from nomorals.llm.cost_display import budget_alert_line, format_cost_table
from nomorals.llm.defaults import key_pool_from_env
from nomorals.llm.download import format_progress, validate_gguf
from nomorals.llm.failures import (
    RECOVERY,
    FailureClass,
    RetryBudget,
    backoff_delay,
    is_retryable,
    retry_after_s,
    should_failover,
)
from nomorals.llm.prompts import PromptLibrary, rubric_prompt
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.providers.openai_compat import (
    OpenAICompatProvider,
    _parse_usage,
)
from nomorals.llm.router import CostLog, LLMRouter, estimate_cost


def _msgs(*texts: str) -> list[Message]:
    return [Message.user(t) for t in texts]


def _judge_json(winner_1based: int, ranking: list[int] | None = None,
                scores: dict | None = None) -> str:
    payload = {
        "winner": winner_1based,
        "ranking": ranking or [winner_1based],
        "rationale": "test verdict",
        "confidence": 0.9,
    }
    if scores:
        payload["scores"] = scores
    return json.dumps(payload)


def _fake_ask_factory(text: str, model: str = "judge-m"):
    def _ask(messages, params, **kw):
        return LLMResponse(text=text, model=model)
    return _ask


class ChatTemplateTests(unittest.TestCase):
    def test_detect_template_families(self):
        self.assertEqual(detect_template("meta-llama/Llama-3.1-8B-Instruct"), "llama3")
        self.assertEqual(detect_template("mistralai/Mistral-7B-Instruct-v0.3"), "mistral")
        self.assertEqual(detect_template("Qwen/Qwen2.5-7B-Instruct"), "qwen")
        self.assertEqual(detect_template("google/gemma-2-9b-it"), "gemma")
        self.assertEqual(detect_template("deepseek-ai/DeepSeek-V3"), "deepseek")
        self.assertEqual(detect_template("Cutyp/codebeast-7b-vl"), "chatml")
        self.assertEqual(detect_template("unknown-model-xyz"), "chatml")

    def test_auto_template_picks_native(self):
        msgs = [Message.system("sys"), Message.user("hi")]
        mistral = messages_to_text(msgs, template="auto",
                                   model="mistralai/Mistral-7B")
        self.assertIn("[INST]", mistral)
        self.assertNotIn("<|im_start|>", mistral)
        llama = messages_to_text(msgs, template="auto",
                                 model="meta-llama/Llama-3.1-8B")
        self.assertIn("<|start_header_id|>", llama)
        gemma = messages_to_text(msgs, template="auto",
                                 model="google/gemma-2-9b-it")
        self.assertIn("<start_of_turn>", gemma)

    def test_default_chatml_unchanged(self):
        out = messages_to_text(_msgs("hi"))
        self.assertIn("<|im_start|>user", out)
        self.assertTrue(out.rstrip().endswith("<|im_start|>assistant"))

    def test_unknown_template_plain_fallback(self):
        out = messages_to_text(_msgs("hi"), template="nope-not-real")
        self.assertIn("user: hi", out)

    def test_template_registry_covers_families(self):
        for name in ("chatml", "llama3", "llama2", "mistral", "gemma",
                     "qwen", "deepseek", "phi3", "auto"):
            self.assertIn(name, CHAT_TEMPLATES)

    def test_response_carries_trace_and_cost(self):
        resp = LLMResponse(text="hi")
        resp.route_trace = [{"provider": "a", "ok": True}]
        resp.cost_usd = 0.001
        d = resp.to_dict()
        self.assertEqual(d["route_trace"][0]["provider"], "a")
        self.assertAlmostEqual(d["cost_usd"], 0.001)

    def test_usage_cached_reasoning_fields(self):
        u = Usage(prompt_tokens=100, completion_tokens=50,
                  cached_tokens=80, reasoning_tokens=200)
        self.assertEqual(u.as_dict["cached_tokens"], 80)
        self.assertEqual(u.as_dict["reasoning_tokens"], 200)


class FailureResilienceTests(unittest.TestCase):
    def test_backoff_full_jitter_bounds(self):
        for attempt in range(6):
            exp = min(1.0 * (2.0 ** attempt), 60.0)
            for _ in range(25):
                d = backoff_delay(attempt, base=1.0, cap=60.0, jitter="full")
                self.assertGreaterEqual(d, 0.0)
                self.assertLessEqual(d, exp)

    def test_backoff_none_deterministic(self):
        self.assertEqual(backoff_delay(2, base=1.0, cap=60.0, jitter="none"),
                         backoff_delay(2, base=1.0, cap=60.0, jitter="none"))
        self.assertEqual(backoff_delay(2, jitter="none"), 4.0)

    def test_backoff_equal_half_fixed(self):
        for _ in range(25):
            d = backoff_delay(3, base=1.0, cap=60.0, jitter="equal")
            self.assertGreaterEqual(d, 4.0)
            self.assertLessEqual(d, 8.0)

    def test_retryable_terminal_classes(self):
        self.assertFalse(is_retryable(FailureClass.AUTH))
        self.assertFalse(is_retryable(FailureClass.CONFIG))
        self.assertFalse(is_retryable("auth"))
        self.assertTrue(is_retryable(FailureClass.RATE_LIMITED))
        self.assertTrue(is_retryable(FailureClass.TIMEOUT))

    def test_should_failover_table(self):
        self.assertTrue(should_failover("rate_limited"))
        self.assertFalse(should_failover("budget"))
        self.assertEqual(RECOVERY[FailureClass.BUDGET].failover, False)

    def test_retry_after_parsed(self):
        self.assertEqual(retry_after_s("429 slow down Retry-After: 30"), 30.0)
        self.assertIsNone(retry_after_s("plain 500 error"))

    def test_retry_budget(self):
        budget = RetryBudget(deadline_s=0.05)
        self.assertFalse(budget.exhausted)
        self.assertGreater(budget.remaining, 0)
        time.sleep(0.06)
        self.assertTrue(budget.exhausted)
        self.assertEqual(budget.next_delay(), 0.0)

    def test_retry_budget_delay_clamped(self):
        budget = RetryBudget(deadline_s=100.0)
        d = budget.next_delay(base=1.0, cap=60.0, jitter="none")
        self.assertEqual(d, 1.0)
        self.assertEqual(budget.attempts, 1)


class ModelCardTests(unittest.TestCase):
    def _provider(self, name="groq", model_id="llama-3.1-8b", caps=None):
        class P:
            pass
        p = P()
        p.name = name
        p.model_id = model_id
        p.capabilities = caps or {"chat"}
        return p

    def test_tags_auto_derived(self):
        card = ModelCard.from_provider(
            self._provider(model_id="Qwen2.5-Coder-32B"),
            card_id="coder", context_len=32768, cost_per_1k=0.0)
        self.assertIn("code", card.tags)
        self.assertIn("cheap", card.tags)
        self.assertIn("long-context", card.tags)
        local = ModelCard.from_provider(
            self._provider(name="llama_cpp"), card_id="local",
            local=True)
        self.assertIn("local", local.tags)

    def test_has_tags(self):
        card = ModelCard(id="m", capabilities={"chat"}, tags={"code", "fast"})
        self.assertTrue(card.has_tags({"code"}))
        self.assertTrue(card.has_tags("fast"))
        self.assertFalse(card.has_tags({"vision"}))

    def test_price_fallback_and_estimate(self):
        card = ModelCard(id="m", capabilities={"chat"}, cost_per_1k=0.002)
        self.assertAlmostEqual(card.price_per_1m_in, 2.0)
        card2 = ModelCard(id="m2", capabilities={"chat"},
                          price_in=1.5, price_out=3.0)
        self.assertAlmostEqual(card2.price_per_1m_in, 1.5)
        self.assertAlmostEqual(
            card2.estimated_call_cost(1_000_000, 1_000_000), 4.5)

    def test_quality_clamped(self):
        self.assertEqual(ModelCard(id="m", quality=9.0).quality, 1.0)
        self.assertEqual(ModelCard(id="m", quality=-2.0).quality, 0.0)

    def test_to_dict_has_new_fields(self):
        d = ModelCard(id="m", capabilities={"chat"}).to_dict()
        self.assertIn("tags", d)
        self.assertIn("price_in_per_1m", d)
        self.assertIn("quality", d)


class BenchmarkSweepTests(unittest.TestCase):
    def test_cost_per_task_excludes_failures(self):
        db = BenchmarkDB()
        for _ in range(4):
            db.record("m1", "chat", 0.5, True)
        for _ in range(4):
            db.record("m1", "chat", 2.0, False)
        cpt = db.cost_per_task("m1", "chat", cost_per_1k=2.0)
        self.assertEqual(cpt["successful_tasks"], 4)
        self.assertEqual(cpt["failed_tasks"], 4)
        self.assertAlmostEqual(cpt["failure_rate"], 0.5)
        # 4 successful tasks at $2/1k per task
        self.assertAlmostEqual(cpt["cost_per_task_usd"], 0.002)

    def test_leaderboard_orders_by_success(self):
        db = BenchmarkDB()
        for _ in range(5):
            db.record("good", "chat", 0.2, True)
        db.record("bad", "chat", 0.2, True)
        db.record("bad", "chat", 0.2, False)
        board = db.leaderboard("chat")
        self.assertEqual(board[0]["model_id"], "good")
        self.assertGreater(board[0]["success_rate"], board[1]["success_rate"])


class ContextFitSweepTests(unittest.TestCase):
    def test_case_facts_pinned_verbatim(self):
        msgs = [Message.system("sys"), Message.user("what is 2+2?")]
        fitted = fit_messages(msgs, 100, pinned_facts=["the launch code is 48151623"])
        self.assertEqual(fitted.facts_kept, 1)
        joined = "\n".join(m.content for m in fitted.messages)
        self.assertIn("the launch code is 48151623", joined)
        # facts block sits right after the system prompt
        self.assertEqual(fitted.messages[0].role, "system")
        self.assertEqual(fitted.messages[1].role, "system")
        self.assertIn("verbatim", fitted.messages[1].content)

    def test_case_facts_survive_summarization(self):
        turns = [Message.system("sys")]
        for i in range(10):
            turns.append(Message.user(f"question {i} " + "x" * 300))
            turns.append(Message.assistant(f"answer {i} " + "y" * 300))
        fitted = fit_messages(
            turns, 200, task_kind="judge",
            summarizer=lambda t: "SUMMARY",
            pinned_facts=["SSN 123-45-6789"])
        joined = "\n".join(m.content for m in fitted.messages)
        self.assertIn("SSN 123-45-6789", joined)
        self.assertIn("SUMMARY", joined)

    def test_prioritize_ends_reorders(self):
        strat = PrioritizeEnds()
        long_ref = "R" * 2000
        msgs = [Message.system("sys"),
                Message.user("q1"),
                Message.assistant(long_ref),
                Message.user("final question?")]
        out = strat.apply(msgs, 100000, {})
        self.assertEqual(out[1].content, long_ref)
        self.assertEqual(out[-1].content, "final question?")

    def test_trim_tool_results(self):
        big = "L\n" * 5000
        msgs = [Message.user("run it"), Message.tool(big, tool_call_id="1"),
                Message.user("thanks")]
        out = trim_tool_results(msgs, max_chars=2000)
        self.assertLess(len(out[1].content), len(big))
        self.assertIn("trimmed", out[1].content)
        self.assertEqual(out[2].content, "thanks")

    def test_cache_breakpoints(self):
        msgs = [Message.system("sys"), Message.user("a"),
                Message.user("b"), Message.user("c"),
                Message.user("d"), Message.user("e")]
        rows = with_cache_breakpoints(msgs, window=3)
        marked = [r for r in rows if r["cache_breakpoint"]]
        # system + last 3 non-system
        self.assertEqual(len(marked), 4)
        self.assertEqual(marked[0]["role"], "system")
        self.assertEqual(marked[-1]["content"], "e")

    def test_strategy_registered(self):
        self.assertIsInstance(strategy_for("case_facts"), CaseFacts)
        self.assertIsInstance(strategy_for("prioritize_ends"), PrioritizeEnds)


class PromptLibraryTests(unittest.TestCase):
    def test_builtin_seed(self):
        lib = PromptLibrary()
        self.assertIn("judge", lib.get("judge"))
        self.assertIn("judge", lib.names())

    def test_versioning(self):
        lib = PromptLibrary()
        v1 = lib.get("chat")
        lib.register("chat", "v2 template {x}", note="test")
        self.assertEqual(lib.get("chat", 1), v1)
        self.assertEqual(lib.get("chat"), "v2 template {x}")
        self.assertEqual(lib.get("chat", "latest"), "v2 template {x}")
        history = lib.history("chat")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[1]["note"], "test")

    def test_render_slots(self):
        lib = PromptLibrary()
        lib.register("greet", "hello {name}, welcome to {place}")
        self.assertEqual(lib.render("greet", name="Ada", place="Lagos"),
                         "hello Ada, welcome to Lagos")
        # partial rendering keeps unknown slots
        self.assertIn("{place}", lib.render("greet", name="Ada"))

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.json"
            lib = PromptLibrary(path=path)
            lib.register("custom", "custom template")
            lib2 = PromptLibrary(path=path)
            self.assertEqual(lib2.get("custom"), "custom template")

    def test_rubric_prompt(self):
        rp = rubric_prompt({"correctness": "must be right"},
                           reference="the gold answer")
        self.assertIn("the gold answer", rp)
        self.assertIn("correctness", rp)
        self.assertIn('"winner"', rp)
        self.assertIn("step by step", rp.lower())


class AdjudicateSweepTests(unittest.TestCase):
    def test_pairwise_consistent(self):
        # judge prefers answer A in both orderings (winner=1 when A is
        # first, winner=2 when A is second) → consistent win for A
        answers = [_judge_json(1, [1, 2]), _judge_json(2, [2, 1])]
        calls = {"n": 0}

        def steady_ask(messages, params, **kw):
            text = answers[calls["n"]]
            calls["n"] += 1
            return LLMResponse(text=text, model="steady")

        verdict = Judge(steady_ask).pairwise("q?", "answer A", "answer B")
        self.assertTrue(verdict.ok)
        self.assertTrue(verdict.consistent)
        self.assertEqual(verdict.winner, 0)
        self.assertGreaterEqual(verdict.confidence, 0.9)

    def test_pairwise_position_bias_tie(self):
        # judge always picks the FIRST presented candidate (position bias)
        calls = []

        def biased_ask(messages, params, **kw):
            calls.append(messages)
            return LLMResponse(text=_judge_json(1, [1, 2]), model="biased")

        verdict = Judge(biased_ask).pairwise("q?", "answer A", "answer B")
        self.assertTrue(verdict.ok)
        # first call: A presented first → picks A (idx 0); second call:
        # B presented first → picks B (idx 1 in texts) → split = tie
        self.assertFalse(verdict.consistent)
        self.assertEqual(verdict.winner, -1)
        self.assertLess(verdict.confidence, 0.5)

    def test_pairwise_module_fn(self):
        answers = [_judge_json(2, [2, 1]), _judge_json(1, [1, 2])]
        calls = {"n": 0}

        def steady_ask(messages, params, **kw):
            text = answers[calls["n"]]
            calls["n"] += 1
            return LLMResponse(text=text, model="steady")

        verdict = pairwise("q?", "A", "B", steady_ask)
        self.assertTrue(verdict.consistent)
        self.assertEqual(verdict.winner, 1)

    def test_panel_majority(self):
        asks = [
            _fake_ask_factory(_judge_json(1, [1, 2]), model="m1"),
            _fake_ask_factory(_judge_json(1, [1, 2]), model="m2"),
            _fake_ask_factory(_judge_json(2, [2, 1]), model="m3"),
        ]
        judgment = Judge(asks[0]).panel("q?", ["A", "B"], asks)
        self.assertTrue(judgment.ok)
        self.assertEqual(judgment.winner, 0)
        self.assertEqual(judgment.method, "panel-majority")
        self.assertIn("m1", judgment.judge_model)

    def test_panel_split_plurality(self):
        asks = [
            _fake_ask_factory(_judge_json(1, [1, 2]), model="m1"),
            _fake_ask_factory(_judge_json(2, [2, 1]), model="m2"),
        ]
        judgment = Judge(asks[0]).panel("q?", ["A", "B"], asks)
        self.assertTrue(judgment.ok)
        self.assertEqual(judgment.method, "panel-plurality")
        self.assertLess(judgment.confidence, 0.6)

    def test_adjudicate_scores_parsed(self):
        scores = {"1": {"correctness": 5, "clarity": 4},
                  "2": {"correctness": 2, "clarity": 3}}
        j = adjudicate_fn("q?", ["A", "B"],
                               _fake_ask_factory(_judge_json(1, [1, 2], scores)))
        self.assertTrue(j.ok)
        self.assertEqual(j.scores[0]["correctness"], 5)
        self.assertEqual(j.candidate_lengths, [1, 1])

    def test_length_controlled_note(self):
        seen = []

        def ask(messages, params, **kw):
            seen.extend(messages)
            return LLMResponse(text=_judge_json(1, [1]), model="m")

        Judge(ask).adjudicate("q?", ["A", "B"], length_controlled=True)
        joined = " ".join(m.content for m in seen)
        self.assertIn("ignore answer length", joined)

    def test_format_judgment_card(self):
        j = adjudicate_fn("q?", ["short", "longer answer here"],
                               _fake_ask_factory(_judge_json(2, [2, 1])))
        card = format_judgment(j, ["short", "longer answer here"])
        self.assertIn("🏆 candidate 2", card)
        self.assertIn("confidence", card)


class DownloadSweepTests(unittest.TestCase):
    def test_validate_gguf_magic(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "m.gguf"
            good.write_bytes(b"GGUF" + b"\x00" * 4096)
            ok, why = validate_gguf(good)
            self.assertTrue(ok, why)
            bad = Path(tmp) / "b.gguf"
            bad.write_bytes(b"NOPE" + b"\x00" * 4096)
            ok, why = validate_gguf(bad)
            self.assertFalse(ok)
            self.assertIn("magic", why)
            tiny = Path(tmp) / "t.gguf"
            tiny.write_bytes(b"GGUF")
            ok, _ = validate_gguf(tiny)
            self.assertFalse(ok)

    def test_format_progress(self):
        line = format_progress("model.gguf", 512 * 1024**2, 1024 * 1024**2,
                               started=time.perf_counter() - 2.0)
        self.assertIn("50.0%", line)
        self.assertIn("ETA", line)
        self.assertIn("model.gguf", line)


class RouterSweepTests(unittest.TestCase):
    def _router(self, **kw):
        kw.setdefault("cost_log_path", str(
            Path(tempfile.mkdtemp()) / "cost.jsonl"))
        return LLMRouter(**kw)

    def test_route_trace_on_success(self):
        r = self._router()
        r.add(MockProvider(scripted={"hi": "hello"}), name="a")
        resp = r.chat(_msgs("hi"))
        self.assertTrue(resp.ok)
        self.assertEqual(len(resp.route_trace), 1)
        self.assertEqual(resp.route_trace[0]["provider"], "a")
        self.assertTrue(resp.route_trace[0]["ok"])

    def test_route_trace_on_failure(self):
        r = self._router()
        p = MockProvider()
        p.chat = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("boom"))
        r.add(p, name="a")
        resp = r.chat(_msgs("hi"))
        self.assertFalse(resp.ok)
        self.assertEqual(len(resp.route_trace), 1)
        self.assertFalse(resp.route_trace[0]["ok"])
        self.assertIn("boom", resp.route_trace[0]["error"])

    def test_route_only_and_ignore(self):
        r = self._router()
        r.add(MockProvider(scripted={"hi": "from-a"}), name="a")
        r.add(MockProvider(scripted={"hi": "from-b"}), name="b")
        resp = r.chat(_msgs("hi"), route={"only": ["b"]})
        self.assertEqual(resp.text, "from-b")
        resp = r.chat(_msgs("hi"), route={"ignore": ["a"]})
        self.assertEqual(resp.text, "from-b")

    def test_allow_fallbacks_false(self):
        r = self._router()
        bad = MockProvider()
        bad.chat = lambda *a, **k: LLMResponse(text="", error="dead")
        good = MockProvider(scripted={"hi": "recovered"})
        r.add(bad, name="a", primary=True)
        r.add(good, name="b")
        resp = r.chat(_msgs("hi"), route={"allow_fallbacks": False})
        self.assertFalse(resp.ok)
        self.assertEqual(resp.failed_providers, ["a"])
        # and with fallbacks allowed it recovers
        resp = r.chat(_msgs("hi"))
        self.assertTrue(resp.ok)

    def test_latency_slo_shedding(self):
        r = self._router(latency_slo_ms=100.0)
        fast = MockProvider(scripted={"hi": "fast"})
        slow = MockProvider(scripted={"hi": "slow"})
        r.add(slow, name="slow", primary=True)
        r.add(fast, name="fast")
        # seed rolling latencies: slow p95 = 500ms, fast p95 = 20ms
        for _ in range(10):
            r._health["slow"].record_success(500.0)
            r._health["fast"].record_success(20.0)
        resp = r.chat(_msgs("hi"))
        self.assertEqual(resp.text, "fast")
        self.assertGreater(r.stats["slo_sheds"], 0)

    def test_key_rotation_on_429(self):
        from nomorals.core.errors import RateLimited

        calls = {"n": 0}

        class Flaky(OpenAICompatProvider):
            name = "flaky"

            def chat(self, messages, params=None, **kw):
                calls["n"] += 1
                seen_keys.append(self.api_key)
                if calls["n"] == 1:
                    raise RateLimited("429 rate limit")
                return LLMResponse(text="after-rotation")

        seen_keys: list[str] = []
        p = Flaky(base_url="http://localhost:9", api_keys=["k1", "k2"])
        r = self._router()
        r.add(p, name="flaky")
        resp = r.chat(_msgs("hi"))
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.text, "after-rotation")
        self.assertEqual(seen_keys, ["k1", "k2"])
        self.assertEqual(r.stats["key_rotations"], 1)

    def test_budget_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cost.jsonl")
            log = CostLog(path)
            log.record(provider="x", model="m", cost_usd=5.0)
            r = LLMRouter(daily_budget_usd=1.0, cost_log_path=path)
            r.add(MockProvider(), name="a")
            resp = r.chat(_msgs("hi"))
            self.assertFalse(resp.ok)
            self.assertEqual(resp.failure_class, "budget")
            self.assertIn("budget", resp.error)
            self.assertEqual(r.stats["budget_blocks"], 1)

    def test_budget_allows_under_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "cost.jsonl")
            r = LLMRouter(daily_budget_usd=100.0, cost_log_path=path)
            r.add(MockProvider(scripted={"hi": "ok"}), name="a")
            resp = r.chat(_msgs("hi"))
            self.assertTrue(resp.ok)

    def test_sort_price_uses_broker_cards(self):
        from nomorals.llm.broker import ModelBroker
        r = self._router()
        r.add(MockProvider(scripted={"hi": "cheap"}), name="cheap")
        r.add(MockProvider(scripted={"hi": "pricey"}), name="pricey")
        broker = ModelBroker()
        broker.register(ModelCard(id="cheap", provider="cheap",
                                  capabilities={"chat"}, price_in=0.5))
        broker.register(ModelCard(id="pricey", provider="pricey",
                                  capabilities={"chat"}, price_in=50.0))
        r.set_broker(broker)
        # pricey is primary; sort=price should try cheap first
        r.set_active("pricey")
        resp = r.chat(_msgs("hi"), route={"sort": "price"})
        self.assertEqual(resp.text, "cheap")

    def test_health_p50_p95(self):
        r = self._router()
        r.add(MockProvider(), name="a")
        h = r._health["a"]
        for v in (10, 20, 30, 40, 50):
            h.record_success(float(v))
        self.assertEqual(h.p50_latency_ms, 30.0)
        self.assertGreaterEqual(h.p95_latency_ms, 40.0)

    def test_half_open_probe_budget(self):
        from nomorals.core.retry import CircuitBreaker
        r = self._router(half_open_probe_budget=1)
        r.add(MockProvider(), name="a")
        health = r._health["a"]
        # force half-open state
        health.breaker._state = CircuitBreaker.HALF_OPEN
        r._half_open_probes["a"] = 1  # budget already consumed
        resp = r.chat(_msgs("hi"))
        self.assertFalse(resp.ok)
        self.assertIn("cooling down", resp.error)


class CostLogSweepTests(unittest.TestCase):
    def test_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = CostLog(Path(tmp) / "c.jsonl")
            log.record(provider="groq", operation="chat", cost_usd=0.01,
                       prompt_tokens=100, completion_tokens=50,
                       latency_ms=200.0, cached_tokens=80,
                       reasoning_tokens=10)
            log.record(provider="groq", operation="judge", cost_usd=0.02,
                       prompt_tokens=200, completion_tokens=20,
                       latency_ms=400.0)
            bd = log.breakdown()
            self.assertEqual(bd["total"]["calls"], 2)
            self.assertAlmostEqual(bd["total"]["cost_usd"], 0.03)
            self.assertEqual(bd["total"]["cached_tokens"], 80)
            self.assertEqual(bd["total"]["reasoning_tokens"], 10)
            self.assertIn("chat", bd["by_operation"])
            self.assertIn("groq", bd["by_provider"])
            self.assertIn("avg_latency_ms", bd["by_provider"]["groq"])

    def test_budget_report_alerts(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = CostLog(Path(tmp) / "c.jsonl")
            log.record(provider="x", cost_usd=0.9)
            rep = log.budget_report(1.0)
            self.assertEqual(rep["alert"], "warning")
            self.assertAlmostEqual(rep["pct"], 90.0)
            rep = log.budget_report(0.5)
            self.assertEqual(rep["alert"], "exceeded")
            rep = log.budget_report(0.0)
            self.assertTrue(rep["unlimited"])

    def test_estimate_cost_cached_discount(self):
        full = estimate_cost("groq", "llama", 1000, 100)
        cached = estimate_cost("groq", "llama", 1000, 100,
                               cached_tokens=1000)
        self.assertLess(cached, full)
        # reasoning tokens ride the output price
        with_reasoning = estimate_cost("groq", "llama", 1000, 100,
                                       reasoning_tokens=1000)
        self.assertGreater(with_reasoning, full)


class BrokerSweepTests(unittest.TestCase):
    def _broker(self) -> ModelBroker:
        b = ModelBroker()
        b.register(ModelCard(id="cheap-chat", provider="cheap",
                             capabilities={"chat"}, price_in=0.5,
                             quality=0.4, tags={"cheap"}))
        b.register(ModelCard(id="coder", provider="coder",
                             capabilities={"chat", "code"}, price_in=5.0,
                             quality=0.8, tags={"code"}))
        b.register(ModelCard(id="best", provider="best",
                             capabilities={"chat"}, price_in=20.0,
                             quality=0.95))
        return b

    def test_select_by_tags(self):
        b = self._broker()
        card = b.select_by_tags({"code"})
        self.assertIsNotNone(card)
        self.assertEqual(card.id, "coder")

    def test_escalation_chain_cheapest_first(self):
        b = self._broker()
        chain = b.escalation_chain("chat")
        prices = [c.price_per_1m_in for c in chain]
        self.assertEqual(prices, sorted(prices))
        self.assertEqual(chain[0].id, "cheap-chat")

    def test_select_for_quality(self):
        b = self._broker()
        # cheapest meeting 0.7 → coder (0.8); cheap-chat (0.4) excluded
        card = b.select_for_quality(0.7, "chat")
        self.assertEqual(card.id, "coder")
        # unreachable target → top-quality card
        card = b.select_for_quality(0.99, "chat")
        self.assertEqual(card.id, "best")

    def test_min_quality_constraint(self):
        b = self._broker()
        card = b.select("chat", constraints={"min_quality": 0.9})
        self.assertEqual(card.id, "best")

    def test_explain_and_format(self):
        b = self._broker()
        info = b.explain("chat")
        self.assertEqual(info["winner"], b.select("chat").id)
        self.assertEqual(len(info["candidates"]), 3)
        table = b.format_ranking("chat")
        self.assertIn("🏆", table)
        self.assertIn("cheap-chat", table)


class BrainSweepTests(unittest.TestCase):
    def test_estimate_complexity(self):
        self.assertLess(estimate_complexity("hi"), 0.5)
        hard = ("Debug this race condition step by step:\n```python\n"
                + "x = 1  # line\n" * 120)
        self.assertGreater(estimate_complexity(hard), 0.5)
        for text in ("", "a" * 20000, "why does the sky blue? " * 10):
            c = estimate_complexity(text)
            self.assertGreaterEqual(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_with_quality(self):
        self.assertIsNone(_with_quality(None, None))
        self.assertEqual(_with_quality(None, 0.8), {"min_quality": 0.8})
        self.assertEqual(
            _with_quality({"a": 1}, 0.3), {"a": 1, "min_quality": 0.3})
        cons = BrokerConstraints()
        out = _with_quality(cons, 0.9)
        self.assertIsInstance(out, BrokerConstraints)
        self.assertEqual(out.min_quality, 0.9)

    def test_escalate_cheapest_first(self):
        r = LLMRouter(cost_log_path=str(
            Path(tempfile.mkdtemp()) / "c.jsonl"))
        bad = MockProvider()
        bad.chat = lambda *a, **k: LLMResponse(text="", error="dead")
        good = MockProvider(scripted={"hi": "served"})
        r.add(bad, name="cheap")
        r.add(good, name="pricey")
        broker = ModelBroker()
        broker.register(ModelCard(id="cheap", provider="cheap",
                                  capabilities={"chat"}, price_in=0.5))
        broker.register(ModelCard(id="pricey", provider="pricey",
                                  capabilities={"chat"}, price_in=50.0))
        r.set_broker(broker)
        brain = Brain(router=r)
        resp = brain.escalate("hi")
        self.assertTrue(resp.ok, resp.error)
        self.assertEqual(resp.text, "served")
        self.assertIn("escalation", resp.fallback_note)
        self.assertEqual(resp.failed_providers, ["cheap"])

    def test_best_of_single_provider_fallback(self):
        r = LLMRouter(cost_log_path=str(
            Path(tempfile.mkdtemp()) / "c.jsonl"))
        r.add(MockProvider(scripted={"hi": "solo"}), name="only")
        brain = Brain(router=r)
        text, judgment = brain.best_of("hi", n=3)
        self.assertEqual(text, "solo")
        self.assertEqual(judgment.rationale, "single provider")


class ProviderSweepTests(unittest.TestCase):
    def test_key_pool_rotation(self):
        p = OpenAICompatProvider(base_url="http://localhost:9",
                                 api_keys=["k1", "k2", "k3"])
        self.assertEqual(p.key_pool_size, 3)
        self.assertEqual(p.api_key, "k1")
        self.assertTrue(p.rotate_key())
        self.assertEqual(p.api_key, "k2")
        p.rotate_key()
        p.rotate_key()
        self.assertEqual(p.api_key, "k1")  # wraps around

    def test_single_key_no_rotation(self):
        p = OpenAICompatProvider(base_url="http://localhost:9",
                                 api_key="solo")
        self.assertFalse(p.rotate_key())
        self.assertEqual(p.api_key, "solo")

    def test_key_pool_dedup(self):
        p = OpenAICompatProvider(base_url="http://localhost:9",
                                 api_key="k1", api_keys=["k1", "k2"])
        self.assertEqual(p.key_pool_size, 2)

    def test_parse_usage_details(self):
        u = _parse_usage({
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 80},
            "completion_tokens_details": {"reasoning_tokens": 30},
        })
        self.assertEqual(u.cached_tokens, 80)
        self.assertEqual(u.reasoning_tokens, 30)
        self.assertEqual(u.prompt_tokens, 100)
        # missing details → zeros, never raises
        u2 = _parse_usage(None)
        self.assertEqual(u2.cached_tokens, 0)

    def test_key_pool_from_env(self):
        os.environ["SWEEPTEST_API_KEY"] = "primary"
        os.environ["SWEEPTEST_API_KEYS"] = "primary, extra1, extra2"
        try:
            pool = key_pool_from_env("SWEEPTEST_API_KEY")
            self.assertEqual(pool, ["extra1", "extra2"])
        finally:
            del os.environ["SWEEPTEST_API_KEY"]
            del os.environ["SWEEPTEST_API_KEYS"]


class RegistrySweepTests(unittest.TestCase):
    def _registry(self):
        from nomorals.llm.registry import ModelRegistry
        from nomorals.storage.db import Database
        db = Database(":memory:")
        db.execute(
            "CREATE TABLE models (id TEXT PRIMARY KEY, name TEXT NOT NULL,"
            " family TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT"
            " 'foundation', source TEXT NOT NULL DEFAULT '', revision TEXT"
            " NOT NULL DEFAULT '', params INTEGER NOT NULL DEFAULT 0,"
            " context_length INTEGER NOT NULL DEFAULT 0, quantization TEXT"
            " NOT NULL DEFAULT '', license TEXT NOT NULL DEFAULT '',"
            " sha256 TEXT NOT NULL DEFAULT '', path TEXT NOT NULL DEFAULT '',"
            " size_bytes INTEGER NOT NULL DEFAULT 0, active INTEGER NOT NULL"
            " DEFAULT 0, base_model TEXT NOT NULL DEFAULT '',"
            " eval_scores TEXT NOT NULL DEFAULT '{}', created_at REAL NOT NULL,"
            " metadata TEXT NOT NULL DEFAULT '{}', UNIQUE(name, revision))")
        return ModelRegistry(db)

    def test_set_pricing_and_cheapest(self):
        reg = self._registry()
        reg.register("m1", kind="foundation", path="/models/m1.gguf")
        reg.register("m2", kind="foundation", path="/models/m2.gguf")
        reg.set_pricing("m1", price_in=2.0, price_out=4.0, quality=0.7)
        rec = reg.by_name("m1")
        self.assertAlmostEqual(rec.price_in_per_1m, 2.0)
        self.assertAlmostEqual(rec.quality, 0.7)
        cheapest = reg.cheapest_local()
        self.assertEqual(cheapest.name, "m2")  # free beats priced

    def test_best_eval(self):
        reg = self._registry()
        reg.register("m1", kind="foundation")
        reg.register("m2", kind="foundation")
        reg.record_eval("m1", {"score": 0.6})
        reg.record_eval("m2", {"score": 0.9})
        best = reg.best_eval()
        self.assertEqual(best.name, "m2")
        self.assertIsNone(reg.best_eval("nonexistent-metric"))


class CostDisplaySweepTests(unittest.TestCase):
    def test_format_cost_table(self):
        bd = {
            "total": {"calls": 3, "prompt_tokens": 300,
                      "completion_tokens": 150, "cached_tokens": 100,
                      "reasoning_tokens": 20, "cost_usd": 0.05,
                      "avg_latency_ms": 250.0},
            "by_operation": {
                "chat": {"calls": 2, "cost_usd": 0.04,
                         "avg_latency_ms": 200.0},
                "judge": {"calls": 1, "cost_usd": 0.01,
                          "avg_latency_ms": 350.0},
            },
            "by_provider": {
                "groq": {"calls": 3, "cost_usd": 0.05,
                         "avg_latency_ms": 250.0},
            },
        }
        table = format_cost_table(bd)
        self.assertIn("💰 LLM spend", table)
        self.assertIn("by operation:", table)
        self.assertIn("groq", table)
        self.assertIn("reasoning 20", table)

    def test_budget_alert_line(self):
        self.assertIn("EXCEEDED", budget_alert_line(1.5, 1.0))
        self.assertIn("warning", budget_alert_line(0.85, 1.0))
        self.assertIn("watch", budget_alert_line(0.6, 1.0))
        self.assertIn("ok", budget_alert_line(0.1, 1.0))
        self.assertIn("no budget", budget_alert_line(0.1, 0.0))


class ThreadingSmokeTests(unittest.TestCase):
    def test_concurrent_dispatch_threadsafe(self):
        r = LLMRouter(cost_log_path=str(
            Path(tempfile.mkdtemp()) / "c.jsonl"))
        r.add(MockProvider(scripted={"hi": "ok"}), name="a")
        results: list[bool] = []

        def _call():
            try:
                results.append(r.chat(_msgs("hi")).ok)
            except Exception:  # noqa: BLE001
                results.append(False)

        threads = [threading.Thread(target=_call) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
        self.assertEqual(len(results), 10)
        self.assertTrue(all(results))


if __name__ == "__main__":
    unittest.main()
