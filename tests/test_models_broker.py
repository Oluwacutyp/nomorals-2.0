"""Tests for the capability-based model broker (nomorals.llm.broker).

All offline.  Providers are MockProviders; trajectory stores are fakes.
"""

from __future__ import annotations

import argparse
import io
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

from nomorals.core.config import Settings
from nomorals.llm.base import LLMResponse, Message
from nomorals.llm.benchmarks import BenchmarkDB
from nomorals.llm.broker import BrokerConstraints, ModelBroker, NoCandidate
from nomorals.llm.capabilities import Capability, ModelCard
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.router import LLMRouter


def _card(card_id: str, provider: str, caps, **kw) -> ModelCard:
    return ModelCard(
        id=card_id, provider=provider, model_id=card_id,
        capabilities=set(caps), context_len=kw.get("context_len", 8192),
        local=kw.get("local", False), quant=kw.get("quant", ""),
        cost_per_1k=kw.get("cost_per_1k", 0.0),
    )


class FakeTrajectories:
    """Duck-typed trajectory store: success_rate(task_kind, capability, model_id)."""

    def __init__(self, rates: dict[str, float]) -> None:
        self.rates = rates
        self.calls: list[tuple] = []

    def success_rate(self, task_kind: str, capability: str, model_id: str) -> float:
        self.calls.append((task_kind, capability, model_id))
        return self.rates.get(model_id, 0.5)


class BrokerSelectionTests(unittest.TestCase):
    def setUp(self):
        self.broker = ModelBroker(benchmarks=BenchmarkDB())
        self.fast = _card("fast", "fast", {Capability.CHAT, Capability.CODE})
        self.slow = _card("slow", "slow", {Capability.CHAT})
        self.vision = _card("seer", "seer", {Capability.CHAT, Capability.VISION})
        for c in (self.fast, self.slow, self.vision):
            self.broker.register(c)

    def test_capability_selection_beats_name_order(self):
        # Even though "fast" sorts first and would win a name-based pick, a
        # vision capability must route to the vision-capable card.
        card = self.broker.select(Capability.VISION)
        self.assertIsNotNone(card)
        self.assertEqual(card.id, "seer")

    def test_capability_hard_filter(self):
        broker = ModelBroker(benchmarks=BenchmarkDB())
        broker.register(_card("chatonly", "p", {Capability.CHAT}))
        self.assertIsNone(broker.select(Capability.VISION))
        with self.assertRaises(NoCandidate):
            broker.require(Capability.VISION)

    def test_code_prefers_code_card(self):
        card = self.broker.select(Capability.CHAT, task_kind="code")
        self.assertEqual(card.id, "fast")

    def test_operator_override_always_wins(self):
        self.broker.promote("slow")
        self.assertEqual(self.broker.select(Capability.CHAT).id, "slow")
        self.assertEqual(self.broker.select(Capability.CHAT, task_kind="code").id, "slow")
        # …but only for capabilities it can serve:
        self.assertEqual(self.broker.select(Capability.VISION).id, "seer")
        self.broker.demote()
        self.assertEqual(self.broker.select(Capability.CHAT, task_kind="code").id, "fast")

    def test_promote_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.broker.promote("nope")

    def test_trajectory_store_influences_selection(self):
        broker = ModelBroker(
            benchmarks=BenchmarkDB(),
            trajectories=FakeTrajectories({"fast": 0.1, "slow": 0.95}),
        )
        broker.register(_card("fast", "fast", {Capability.CHAT}))
        broker.register(_card("slow", "slow", {Capability.CHAT}))
        # No benchmark rows → both 0.5; the trajectory record decides.
        self.assertEqual(broker.select(Capability.CHAT).id, "slow")

    def test_no_trajectory_store_still_works(self):
        broker = ModelBroker(benchmarks=BenchmarkDB())  # trajectories=None
        broker.register(_card("a", "a", {Capability.CHAT}))
        card = broker.select(Capability.CHAT)
        self.assertEqual(card.id, "a")

    def test_broken_trajectory_store_is_advisory(self):
        class Broken:
            def success_rate(self, *a):
                raise RuntimeError("boom")
        broker = ModelBroker(benchmarks=BenchmarkDB(), trajectories=Broken())
        broker.register(_card("a", "a", {Capability.CHAT}))
        self.assertEqual(broker.select(Capability.CHAT).id, "a")

    def test_benchmark_score_breaks_ties(self):
        db = BenchmarkDB()
        broker = ModelBroker(benchmarks=db)
        broker.register(_card("fast", "fast", {Capability.CHAT}))
        broker.register(_card("slow", "slow", {Capability.CHAT}))
        for _ in range(5):
            db.record("fast", "chat", 0.05, True)
            db.record("slow", "chat", 3.0, True)
        self.assertEqual(broker.select(Capability.CHAT).id, "fast")

    def test_constraints(self):
        broker = ModelBroker(benchmarks=BenchmarkDB())
        broker.register(_card("cloudy", "c", {Capability.CHAT}, local=False))
        broker.register(_card("localbox", "l", {Capability.CHAT}, local=True))
        card = broker.select(Capability.CHAT, constraints={"local_only": True})
        self.assertEqual(card.id, "localbox")
        card = broker.select(Capability.CHAT,
                             constraints=BrokerConstraints(cloud_only=True))
        self.assertEqual(card.id, "cloudy")
        self.assertIsNone(broker.select(
            Capability.CHAT, constraints={"providers": ("nope",)}))

    def test_ranked_lists_all_candidates(self):
        ranked = self.broker.ranked(Capability.CHAT)
        self.assertEqual(len(ranked), 3)  # fast, slow, seer (chat fallback)
        ids = [c.id for c, _ in ranked]
        self.assertIn("fast", ids)

    def test_build_from_router(self):
        router = LLMRouter()
        router.add(MockProvider(model="m-fast"), name="m-fast")
        broker = ModelBroker(benchmarks=BenchmarkDB())
        cards = broker.build_from_router(router)
        self.assertEqual(len(cards), 1)
        self.assertIn(Capability.CHAT, cards[0].capabilities)


class BrokerRouterWiringTests(unittest.TestCase):
    def _router(self) -> LLMRouter:
        router = LLMRouter()
        router.add(MockProvider(model="alpha"), name="alpha", primary=True)
        router.add(MockProvider(model="beta"), name="beta")
        return router

    def test_no_broker_old_path_unchanged(self):
        router = self._router()
        resp = router.chat([Message.user("hi")])
        self.assertTrue(resp.ok)
        self.assertEqual(router.active, "alpha")

    def test_consult_moves_active_to_selected_provider(self):
        router = self._router()
        broker = ModelBroker(benchmarks=BenchmarkDB())
        broker.register(_card("alpha", "alpha", {Capability.CHAT}))
        broker.register(_card("beta", "beta", {Capability.CHAT}))
        db = broker.benchmarks
        for _ in range(3):
            db.record("beta", "chat", 0.01, True)
            db.record("alpha", "chat", 2.0, True)
        router.set_broker(broker)
        resp = router.chat([Message.user("hi")])
        self.assertTrue(resp.ok)
        self.assertEqual(router.active, "beta")
        self.assertIn("beta", resp.provider)

    def test_broker_error_falls_back_to_name_chain(self):
        router = self._router()

        class ExplodingBroker:
            def consult(self, router, operation, task_kind=""):
                raise RuntimeError("broker exploded")

        router.set_broker(ExplodingBroker())
        resp = router.chat([Message.user("hi")])
        self.assertTrue(resp.ok)
        self.assertEqual(router.active, "alpha")  # untouched

    def test_broker_with_no_candidate_keeps_chain(self):
        router = self._router()
        broker = ModelBroker(benchmarks=BenchmarkDB())  # no cards at all
        router.set_broker(broker)
        resp = router.chat([Message.user("hi")])
        self.assertTrue(resp.ok)
        self.assertEqual(router.active, "alpha")

    def test_detach_restores_plain_routing(self):
        router = self._router()
        broker = ModelBroker(benchmarks=BenchmarkDB())
        broker.register(_card("beta", "beta", {Capability.CHAT}))
        broker.promote("beta")
        router.set_broker(broker)
        router.chat([Message.user("hi")])
        self.assertEqual(router.active, "beta")
        router.set_broker(None)
        router.set_active("alpha")
        router.chat([Message.user("hi")])
        self.assertEqual(router.active, "alpha")


class CLISmokeTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.context import build_context
        from nomorals.cmdline.commands.models import _cmd_model_broker
        from nomorals.llm.lifecycle import ModelLifecycle
        self._cmd = _cmd_model_broker
        self._Lifecycle = ModelLifecycle
        self.home = tempfile.mkdtemp(prefix="nm-modelcli-")
        self._ctx_mgr = build_context(Settings(home=self.home))
        self.ctx = self._ctx_mgr.__enter__()
        self.addCleanup(self._ctx_mgr.__exit__, None, None, None)
        self.addCleanup(shutil.rmtree, self.home, True)
        # A fake GGUF the lifecycle can walk offline.
        gguf = tempfile.NamedTemporaryFile(suffix=".gguf", delete=False)
        gguf.write(b"GGUF" + b"\x00" * 4096)
        gguf.close()
        self.gguf_path = gguf.name
        self.addCleanup(__import__("os").unlink, self.gguf_path)

    def _run(self, **kwargs):
        defaults = dict(
            json=False, model_action="", model_target="", capability="",
            quant="Q4_K_M", context_len=0, task_kind="", rounds=3)
        defaults.update(kwargs)
        args = argparse.Namespace(**defaults)
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = self._cmd(args, self.ctx)
        return rc, buf.getvalue()

    def _only_id(self) -> str:
        return self._Lifecycle(self.ctx.db).list()[0].id

    def test_list_empty_then_add_then_list(self):
        rc, out = self._run(model_action="list")
        self.assertEqual(rc, 0)
        self.assertIn("no models", out)
        rc, out = self._run(model_action="add", model_target=self.gguf_path,
                            context_len=8192)
        self.assertEqual(rc, 0)
        self.assertIn("registered", out)
        rc, out = self._run(model_action="list")
        self.assertEqual(rc, 0)
        self.assertIn("stage=registered", out)

    def test_use_promotes_and_writes_boot_contract(self):
        from pathlib import Path
        rc, _ = self._run(model_action="add", model_target=self.gguf_path)
        self.assertEqual(rc, 0)
        model_id = self._only_id()
        rc, out = self._run(model_action="use", model_target=model_id)
        self.assertEqual(rc, 0, out)
        self.assertIn("primary mind", out)
        lc = self._Lifecycle(self.ctx.db)
        self.assertEqual(lc.primary, model_id)
        # boot contract written for a local GGUF
        env = (Path(self.home) / ".env").read_text()
        self.assertIn("NM_LLM_PROVIDER=llama_cpp", env)
        self.assertIn(self.gguf_path, env)

    def test_use_refuses_unverified_when_unwalkable(self):
        # A GGUF path that does not exist cannot be walked to verified →
        # use fails cleanly instead of promoting a model with no bytes.
        # (No network touched: the missing local file fails before any
        # download attempt.)
        rc, _ = self._run(model_action="add",
                          model_target="/nonexistent/missing.gguf")
        self.assertEqual(rc, 0)
        model_id = self._only_id()
        rc, out = self._run(model_action="use", model_target=model_id)
        self.assertEqual(rc, 1)
        self.assertNotEqual(self._Lifecycle(self.ctx.db).primary, model_id)

    def test_select_dry_run(self):
        rc, _ = self._run(model_action="add", model_target=self.gguf_path)
        self.assertEqual(rc, 0)
        rc, out = self._run(model_action="select", model_target="chat")
        self.assertEqual(rc, 0)
        self.assertIn("chat ->", out)

    def test_benchmark_records_synthetic_rows(self):
        from nomorals.llm.benchmarks import BenchmarkDB
        rc, _ = self._run(model_action="add", model_target=self.gguf_path)
        self.assertEqual(rc, 0)
        model_id = self._only_id()
        rc, out = self._run(model_action="benchmark", model_target=model_id)
        self.assertEqual(rc, 0, out)
        self.assertIn("source='synthetic'", out)
        db = BenchmarkDB(self.ctx.db)
        self.assertEqual(db.summary(model_id)["sources"], ["synthetic"])
        self.assertEqual(db.summary(model_id)["samples"], 3)

    def test_remove(self):
        rc, _ = self._run(model_action="add", model_target=self.gguf_path)
        self.assertEqual(rc, 0)
        model_id = self._only_id()
        rc, out = self._run(model_action="remove", model_target=model_id)
        self.assertEqual(rc, 0, out)
        from nomorals.core.errors import NotFound
        with self.assertRaises(NotFound):
            self._Lifecycle(self.ctx.db).get(model_id)

    def test_remove_primary_refused(self):
        rc, _ = self._run(model_action="add", model_target=self.gguf_path)
        self.assertEqual(rc, 0)
        model_id = self._only_id()
        rc, _ = self._run(model_action="use", model_target=model_id)
        self.assertEqual(rc, 0)
        rc, out = self._run(model_action="remove", model_target=model_id)
        self.assertEqual(rc, 1)
        self.assertIn("primary", out)


class BrokerPrefetchTests(unittest.TestCase):
    """R8 perf: select() must consult the benchmark DB exactly once.

    Before the prefetch, every select issued 3N identical samples queries
    for N candidates (summary, summary's score, selection score) — on the
    hot path of every model call when a broker is wired to the router.
    """

    def _broker_with_data(self, n_cards=8):
        bench = BenchmarkDB()
        for i in range(n_cards):
            for j in range(10):
                bench.record(f"c{i}", Capability.CHAT,
                             0.1 + 0.05 * j + 0.01 * i,
                             success=(j % 4 != 0))
        broker = ModelBroker(benchmarks=bench)
        for i in range(n_cards):
            broker.register(_card(f"c{i}", f"p{i}", {Capability.CHAT}))
        return broker, bench

    def _count_queries(self, bench):
        calls = []
        inner = bench.db.query

        def counting(sql, params=()):
            calls.append(sql)
            return inner(sql, params)

        bench.db.query = counting
        return calls, inner

    def test_select_single_db_query(self):
        from nomorals.llm.broker import BrokerConstraints
        broker, bench = self._broker_with_data()
        calls, inner = self._count_queries(bench)
        try:
            winner = broker.select(Capability.CHAT)
        finally:
            bench.db.query = inner
        self.assertIsNotNone(winner)
        self.assertEqual(len(calls), 1)

    def test_ranked_single_db_query(self):
        broker, bench = self._broker_with_data()
        calls, inner = self._count_queries(bench)
        try:
            ranked = broker.ranked(Capability.CHAT)
        finally:
            bench.db.query = inner
        self.assertEqual(len(ranked), 8)
        self.assertEqual(len(calls), 1)

    def test_select_matches_naive_per_card_path(self):
        # The prefetch path must pick the same winner as the old naive
        # per-card queries (summary/score per model) would.
        import statistics
        from nomorals.llm.benchmarks import SUMMARY_WINDOW
        from nomorals.llm.broker import BrokerConstraints
        broker, bench = self._broker_with_data()
        winner = broker.select(Capability.CHAT)
        rows = {c.id: bench.samples(c.id, Capability.CHAT, limit=SUMMARY_WINDOW)
                for c in broker.cards()}
        medians = {}
        for mid, rws in rows.items():
            lats = [r["latency_s"] for r in rws if r["success"]]
            if lats:
                medians[mid] = max(round(statistics.median(lats), 4), 1e-6)
        best = min(medians.values())
        rank = {m: best / v for m, v in medians.items()}
        naive = sorted(
            ((broker._score(c, Capability.CHAT, None, BrokerConstraints(),
                            rank.get(c.id), rows[c.id]), c.id)
             for c in broker.cards()),
            key=lambda t: (-t[0], t[1]),
        )
        self.assertEqual(winner.id, naive[0][1])
        # fastest card wins on the latency rank tie-break
        self.assertEqual(winner.id, "c0")


if __name__ == "__main__":
    unittest.main()
