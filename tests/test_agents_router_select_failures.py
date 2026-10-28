"""Tail-24 pins: router_select.py failure values + NOT-recording.

Verdict: pins-only, 0 converts / 9 exclusions. Every handler below is
documented never-raises or best-effort, and every fallback value is
either the module's documented default, the value of a supported
no-evidence configuration, or a visibly-empty value — never a
fabricated success. Each pin drives the handler into its except path
and pins the exact fallback value plus the fact that the agents
failures ledger is unchanged (snapshot before/after, tail-8 method).
router_select.py calls no record function anywhere (grep-proven), so
the ledger must not move.

The pins file itself contains no except handlers, so it adds zero
census sites.
"""

from types import SimpleNamespace
from unittest import mock

from nomorals.agents import failures as ag_fail
from nomorals.agents.router_select import (
    ChatBaseline,
    ModelProfile,
    ScoreStrategy,
    TaskRouter,
)


def _ledger():
    return list(ag_fail.recent_failures())


class _Ctx:
    def __init__(self, router, intelligent="on"):
        self.router = router
        self.settings = SimpleNamespace(router_intelligent=intelligent)


class _Provider:
    def __init__(self, name, caps=("chat",), model_id=""):
        self.name = name
        self.capabilities = set(caps)
        self.model_id = model_id or name

    def stats_snapshot(self):
        return {"calls": 10, "errors": 0, "avg_latency_ms": 100.0}


class _Router:
    def __init__(self, providers, broker=None):
        self._ps = providers
        self._broker = broker

    def providers(self):
        return list(self._ps)

    def get(self, name):
        return self._ps.get(name)

    @property
    def broker(self):
        return self._broker


def test_providers_failure_falls_back_to_by_name():
    """378: providers() failure -> the router's internal registry, a
    genuine alternate source, not a fabrication. Control: providers()
    works -> profiles from providers()."""
    before = _ledger()

    class _ByNameRouter(_Router):
        def providers(self):
            raise RuntimeError("provider list down")

    router = _ByNameRouter({"a": _Provider("a")})
    agent = TaskRouter(_Ctx(router))
    router._by_name = {"a": _Provider("a")}
    assert [p.name for p in agent.profiles()] == ["a"]
    agent2 = TaskRouter(_Ctx(_Router({"a": _Provider("a")})))
    assert [p.name for p in agent2.profiles()] == ["a"]
    assert _ledger() == before


def test_broker_failure_uses_no_evidence_defaults():
    """391: broker.cards() failure -> cost 0.0 + quality 0.5, the
    no-evidence values (identical to a provider with no broker card).
    Control: a working card keeps its cost, quality stays unknown."""
    before = _ledger()

    class _RaisingBroker:
        def cards(self):
            raise RuntimeError("broker down")

    agent = TaskRouter(_Ctx(_Router({"a": _Provider("a")},
                                   broker=_RaisingBroker())))
    prof = agent.profiles()[0]
    assert prof.cost_per_1k == 0.0 and prof.quality == 0.5

    class _Card:
        provider = "a"
        cost_per_1k = 1.5
        id = "c1"

    class _Broker:
        def cards(self):
            return [_Card()]

    agent = TaskRouter(_Ctx(_Router({"a": _Provider("a")},
                                   broker=_Broker())))
    prof = agent.profiles()[0]
    assert prof.cost_per_1k == 1.5 and prof.quality == 0.5
    assert _ledger() == before


def test_benchmark_failure_keeps_card_quality_unknown():
    """407: benchmarks.score() failure -> quality 0.5 (the documented
    unknown), card cost preserved: only the score fell back, not the
    whole card. Control: a working score is kept."""
    before = _ledger()

    class _Card:
        provider = "a"
        cost_per_1k = 1.5
        id = "c1"

    class _RaisingBench:
        def score(self, *args):
            raise RuntimeError("bench down")

    class _Broker:
        def cards(self):
            return [_Card()]

        benchmarks = _RaisingBench()

    agent = TaskRouter(_Ctx(_Router({"a": _Provider("a")},
                                   broker=_Broker())))
    prof = agent.profiles()[0]
    assert prof.quality == 0.5 and prof.cost_per_1k == 1.5

    class _Bench:
        def score(self, *args):
            return 0.9

    class _Broker2:
        def cards(self):
            return [_Card()]

        benchmarks = _Bench()

    agent = TaskRouter(_Ctx(_Router({"a": _Provider("a")},
                                   broker=_Broker2())))
    assert agent.profiles()[0].quality == 0.9
    assert _ledger() == before


def test_breakdown_strategy_failure_is_zero():
    """467: a raising strategy contributes 0.0 to the breakdown — the
    same treatment score() gives it (skip = +0), so the two methods
    agree. Control: a working strategy keeps its contribution."""
    before = _ledger()

    class _Boom(ScoreStrategy):
        name = "boom"

        def score(self, profile, task_type, objective, complexity):
            raise RuntimeError("strategy down")

    prof = ModelProfile(name="m", capabilities=frozenset({"chat"}))
    agent = TaskRouter(_Ctx(_Router({})),
                       strategies=[ChatBaseline(), _Boom()])
    bd = agent.score_breakdown(prof, "chat", "balanced")
    assert bd["boom"] == 0.0 and bd["chat_baseline"] == 1.0
    assert agent.score(prof, "chat", "balanced") == 1.0
    assert _ledger() == before


def test_unparseable_latency_keeps_full_reward():
    """516: the latency discount applies only to KNOWN slowness (>30s);
    an unparseable latency is not known-slow, so the documented base
    reward (1.0 for success) stands. Controls: slow -> 0.5, fail -> 0.0."""
    before = _ledger()
    agent = TaskRouter(_Ctx(_Router({})))
    agent.record_outcome("m", ok=True, latency_ms="garbage")
    agent.record_outcome("m", ok=True, latency_ms=60_000)
    agent.record_outcome("m", ok=True, latency_ms=1000)
    agent.record_outcome("n", ok=False)
    assert agent._feedback["m"] == [1.0, 0.5, 1.0]
    assert agent._feedback["n"] == [0.0]
    assert _ledger() == before


def test_feedback_score_failure_is_no_data_value():
    """531: feedback_score failure -> 0.5, byte-identical to the
    documented no-data value (empty ledger). Control: real data means
    the real mean."""
    before = _ledger()
    agent = TaskRouter(_Ctx(_Router({})))
    agent._feedback = None
    assert agent.feedback_score("m") == 0.5
    agent._feedback = {}
    assert agent.feedback_score("m") == 0.5
    agent._feedback = {"m": [1.0, 1.0]}
    assert agent.feedback_score("m") == 1.0
    assert _ledger() == before


def test_explain_failure_names_pick_without_drivers():
    """558: explain() on a disqualified profile cannot compute drivers
    (the breakdown is a disqualified_by marker, not numbers), so the
    fallback names the real pick and omits drivers — never invents
    them. Control: a normal profile gets the full sentence."""
    before = _ledger()
    agent = TaskRouter(_Ctx(_Router({})))
    text_only = ModelProfile(name="txt", capabilities=frozenset({"chat"}))
    assert agent.score_breakdown(text_only, "vision", "balanced") == \
        {"disqualified_by": "capability_gate"}
    assert agent.explain(text_only, "vision", "balanced") == "picked txt"
    ok_prof = ModelProfile(name="vl",
                           capabilities=frozenset({"chat", "vision"}))
    assert agent.explain(ok_prof, "vision", "balanced").startswith(
        "picked **vl**")
    assert _ledger() == before


def test_classifier_failure_uses_chat_default():
    """603: task-kind classification failure -> "chat", the module-wide
    default task type (every select/route/decision signature defaults
    to it). The kind reaching select() is asserted directly, so the
    control (classifier returns "coding") is not vacuous."""
    before = _ledger()
    import nomorals.agents.task_type as task_type_mod

    seen = {}

    def _fake_select(self, task_type="chat", objective="balanced",
                     complexity=None):
        seen["task_type"] = task_type
        return SimpleNamespace(name="picked")

    agent = TaskRouter(_Ctx(_Router({"c": _Provider("c", ("chat",))})))
    with mock.patch.object(TaskRouter, "select", _fake_select), \
            mock.patch.object(task_type_mod, "classify_task",
                              side_effect=RuntimeError("classifier down")):
        agent.route_by_complexity("do a thing")
    assert seen["task_type"] == "chat"
    with mock.patch.object(TaskRouter, "select", _fake_select), \
            mock.patch.object(task_type_mod, "classify_task",
                              return_value=SimpleNamespace(kind="coding")):
        agent.route_by_complexity("do a thing")
    assert seen["task_type"] == "coding"
    assert _ledger() == before


def test_active_provider_failure_is_empty():
    """612: the chain's active provider failing -> "" (visibly empty,
    no fabricated name) with a real complexity. Control: a working
    active provider is kept."""
    before = _ledger()

    class _ActiveBoom:
        @property
        def active(self):
            raise RuntimeError("chain down")

    agent = TaskRouter(_Ctx(_ActiveBoom(), intelligent="off"))
    name, level = agent.route_by_complexity("do a thing", task_type="chat")
    assert name == "" and level in ("easy", "medium", "hard")

    class _Active:
        active = "prov"

    agent = TaskRouter(_Ctx(_Active(), intelligent="off"))
    name, level = agent.route_by_complexity("do a thing", task_type="chat")
    assert name == "prov"
    assert _ledger() == before
