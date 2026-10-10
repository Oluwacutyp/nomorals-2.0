"""TaskRouter strategy chain: gates, objectives, complexity, reliability. Offline."""
from __future__ import annotations

import pytest

from nomorals.agents.router_select import (
    CapabilityGate,
    ComplexityBias,
    CostObjective,
    ModelProfile,
    QualityObjective,
    ReliabilityPenalty,
    ScoreStrategy,
    SpeedObjective,
    TaskAffinity,
    TaskRouter,
    default_strategies,
)


def _local():
    return ModelProfile(name="llama_cpp", capabilities=frozenset({"chat"}),
                        local=True, free=True, avg_latency_ms=120.0)


def _cloud():
    return ModelProfile(name="hf_serverless", capabilities=frozenset({"chat"}),
                        local=False, free=False, avg_latency_ms=2000.0)


def _vision():
    return ModelProfile(name="vl", capabilities=frozenset({"chat", "vision"}),
                        local=False, avg_latency_ms=3000.0)


class _Ctx:
    def __init__(self, router, intelligent="on"):
        self.router = router
        self.settings = type("S", (), {"router_intelligent": intelligent})()


class _Router:
    def __init__(self, providers):
        self._ps = providers

    def providers(self):
        return list(self._ps)

    def get(self, name):
        return self._ps.get(name)

    @property
    def broker(self):
        return None


class _Provider:
    def __init__(self, name, caps=("chat",), model_id=""):
        self.name = name
        self.capabilities = set(caps)
        self.model_id = model_id or name

    def stats_snapshot(self):
        return {"calls": 10, "errors": 0, "avg_latency_ms": 100.0}


def _router(names):
    return TaskRouter(_Ctx(_Router({n: _Provider(n) for n in names})))


# ── gates ──────────────────────────────────────────────────────────────────

def test_capability_gate_disqualifies():
    gate = CapabilityGate()
    assert gate.disqualifies(_local(), "vision", "balanced", None)
    assert not gate.disqualifies(_vision(), "vision", "balanced", None)
    assert not gate.disqualifies(_local(), "chat", "balanced", None)


def test_score_returns_neg_inf_when_gated():
    r = _router(["llama_cpp"])
    assert r.score(_local(), "vision", "balanced") == float("-inf")


def test_select_skips_disqualified():
    r = TaskRouter(_Ctx(_Router({
        "llama_cpp": _Provider("llama_cpp", ("chat",)),
        "vl": _Provider("vl", ("chat", "vision")),
    })))
    pick = r.select("vision", "balanced")
    assert pick is not None and pick.name == "vl"


# ── objectives ─────────────────────────────────────────────────────────────

def test_speed_prefers_local():
    r = _router(["x"])
    assert r.score(_local(), "chat", "speed") > r.score(_cloud(), "chat", "speed")


def test_cost_prefers_free():
    r = _router(["x"])
    assert r.score(_local(), "chat", "cost") > r.score(_cloud(), "chat", "cost")


def test_quality_prefers_evidence_over_local_bias():
    strong_local = ModelProfile(name="llama_cpp",
                                capabilities=frozenset({"chat"}),
                                local=True, quality=0.9)
    weak_cloud = ModelProfile(name="hf_serverless",
                              capabilities=frozenset({"chat"}),
                              local=False, quality=0.2)
    r = _router(["x"])
    # evidence (0.9 vs 0.2) beats the non-local bias: the strong local wins
    assert r.score(strong_local, "chat", "quality") > \
        r.score(weak_cloud, "chat", "quality")


def test_quality_defaults_still_prefer_cloud():
    r = _router(["x"])
    assert r.score(_cloud(), "chat", "quality") > \
        r.score(_local(), "chat", "quality")


# ── complexity ─────────────────────────────────────────────────────────────

def test_complexity_easy_prefers_cheap():
    r = _router(["x"])
    assert r.score(_local(), "chat", "balanced", "easy") > \
        r.score(_cloud(), "chat", "balanced", "easy")


def test_complexity_hard_prefers_strong():
    r = _router(["x"])
    assert r.score(_cloud(), "chat", "balanced", "hard") > \
        r.score(_local(), "chat", "balanced", "hard")


# ── reliability ────────────────────────────────────────────────────────────

def test_reliability_penalty_configurable():
    flaky = ModelProfile(name="x", capabilities=frozenset({"chat"}),
                         calls=10, errors=9)
    r = _router(["x"])
    assert r.score(flaky, "chat", "balanced") < 0
    strict = TaskRouter(_Ctx(_Router({})), strategies=[ReliabilityPenalty(max_error_rate=0.95)])
    assert strict.score(flaky, "chat", "balanced") == 0.0


# ── chain mechanics ────────────────────────────────────────────────────────

def test_default_chain_has_nine_strategies():
    assert len(default_strategies()) == 9


def test_weights_reweight():
    r = _router(["x", "y"])
    r2 = TaskRouter(_Ctx(_Router({"x": _Provider("x"), "y": _Provider("y")})),
                    weights={"cost_objective": 100.0})
    # with cost blown up, the free local must win a cost race by a mile
    assert r2.score(_local(), "chat", "cost") > r.score(_local(), "chat", "cost")


def test_custom_strategy_extends_chain():
    class BanGroq(ScoreStrategy):
        name = "ban_groq"

        def disqualifies(self, profile, task_type, objective, complexity):
            return profile.name == "groq"

    groq = ModelProfile(name="groq", capabilities=frozenset({"chat"}))
    r = TaskRouter(_Ctx(_Router({})), strategies=default_strategies() + [BanGroq()])
    assert r.score(groq, "chat", "balanced") == float("-inf")


def test_score_breakdown_introspection():
    r = _router(["x"])
    bd = r.score_breakdown(_local(), "chat", "speed")
    assert bd["speed_objective"] > 0
    assert "chat_baseline" in bd
    gated = r.score_breakdown(_local(), "vision", "balanced")
    assert gated == {"disqualified_by": "capability_gate"}


def test_off_by_default():
    r = TaskRouter(_Ctx(_Router({"x": _Provider("x")}), intelligent="off"))
    assert not r.enabled()
    assert r.select("chat", "quality") is None
    d = r.decision("chat", "quality")
    assert d["enabled"] is False and d["choice"] is None


def test_select_picks_best_when_on():
    r = TaskRouter(_Ctx(_Router({
        "llama_cpp": _Provider("llama_cpp", ("chat",)),
        "hf_serverless": _Provider("hf_serverless", ("chat",)),
    })))
    pick = r.select("chat", "speed")
    assert pick is not None and pick.name == "llama_cpp"
