"""Failing-first tests for cost-adaptive token budgeting.

Distilled spec (Research-loop proposal #3 Adaptive Agents, cost/token
model §4.4 of arXiv 2508.11126v1 — "Token Economics for LLM Agents"):
§4.4 optimises on the *intensive margin*: each message is made more
informative per token via (a) format redesign / compact protocols,
(b) cost-aware model cascading, (c) runtime supervision.  Distilled
into Devon behaviour:

1. TRACK per operation: prompt + completion tokens and USD cost, per
   operation and in total, against configured daily + per-operation
   budgets (tokens AND cost).  Tracking is the accountant; it never
   silently changes behaviour on its own.
2. ADAPT as utilisation rises (thresholds from config, defaults below):
   - >= 50% (downgrade): cascade to the cheapest tier — route
     ``sort=price`` + ``max_price`` cap.
   - >= 80% (truncate): shrink context to a capped budget (compact +
     truncate-oldest; system prompt + latest turn are sacred) and cap
     the completion length.  Critical work truncates only at >= 95%.
   - >= 80% (defer): background / non-urgent work is deferred, never
     executed — with an honest "deferred" response, not a silent drop.
3. BLOCK at >= 100%: honest "budget exceeded" error naming the budget
   and the spend, ``failure_class == "budget"``, no call made.
4. HONEST, NEVER SILENT: every adaptation lands on the response as
   ``adaptation_note`` + ``degraded=True`` (downgrade / truncation) or
   ``deferred=True`` (deferral), so the caller can see exactly what was
   changed and why.  Unlimited (no budgets configured) == pure allow.
5. CONFIG: ``AdaptiveBudgetSettings`` (daily token/cost budgets,
   thresholds, per-operation budgets, background operations) drives
   everything; env ``NM_ADAPTIVE_BUDGET_*`` overrides it.

Pre-existing pieces this does NOT rebuild: ``CostLog`` (metered
per-call spend), ``ContextBudget.fit`` (static per-section fit),
``BudgetGovernor`` (per-organ deferral, in-memory), the router's
``daily_budget_usd`` hard block.  None of them adapts: no tier cascade,
no budget-driven context shrink, no per-operation token budget, no
deferral of non-urgent LLM work.  This module + the router wiring close
exactly those gaps.
"""

import pytest

from nomorals.llm.adaptive_budget import (
    Adaptation,
    AdaptiveBudget,
    AdaptiveBudgetConfig,
)
from nomorals.llm.base import Message, SamplingParams
from nomorals.llm.router import LLMRouter


def _cfg(**kw):
    base = dict(daily_token_budget=1000, daily_cost_usd=1.0)
    base.update(kw)
    return AdaptiveBudgetConfig(**base)


def _msgs(n=12):
    out = [Message(role="system", content="You are helpful.")]
    for i in range(n):
        out.append(Message(role="user", content=f"question {i} " + ("word " * 40)))
        out.append(Message(role="assistant", content=f"answer {i} " + ("word " * 40)))
    out.append(Message(role="user", content="final question"))
    return out


# ── 1. per-operation tracking ──────────────────────────────────────────────

def test_per_operation_tracking():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=100, completion_tokens=50, cost_usd=0.10)
    b.record("chat", prompt_tokens=100, completion_tokens=50, cost_usd=0.10)
    b.record("research", prompt_tokens=400, completion_tokens=0, cost_usd=0.40)
    assert b.spent("chat")["tokens"] == 300
    assert b.spent("chat")["cost_usd"] == pytest.approx(0.20)
    assert b.spent("research")["tokens"] == 400
    assert b.spent()["tokens"] == 700
    assert b.spent()["calls"] == 3


def test_remaining_respects_budgets():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=600, completion_tokens=0, cost_usd=0.0)
    rem = b.remaining()
    assert rem["tokens"] == 400
    assert rem["cost_usd"] == pytest.approx(1.0)


# ── 2. adaptation triggers at thresholds ───────────────────────────────────

def test_downgrade_at_50_percent():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=500, completion_tokens=0, cost_usd=0.0)
    ad = b.decide("chat")
    assert ad.action == "downgrade"
    assert ad.downgrade is True
    assert ad.max_price_per_1m is not None and ad.max_price_per_1m > 0
    assert "50%" in ad.note or "downgrad" in ad.note.lower()


def test_truncate_at_80_percent():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    ad = b.decide("chat")
    assert ad.action == "truncate"
    assert ad.context_budget_tokens is not None
    assert ad.context_budget_tokens < 8192
    assert ad.context_budget_tokens >= 1024  # min-context floor
    assert "truncat" in ad.note.lower()


def test_under_threshold_allows_clean():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=100, completion_tokens=0, cost_usd=0.0)
    ad = b.decide("chat")
    assert ad.action == "allow"
    assert ad.degraded is False
    assert ad.note == ""


def test_cost_utilisation_also_adapts():
    # Tokens barely used, but cost at 85% -> truncation must still fire.
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=10, completion_tokens=0, cost_usd=0.85)
    assert b.decide("chat").action == "truncate"


# ── 3. deferral ────────────────────────────────────────────────────────────

def test_background_deferred_at_80_percent():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    ad = b.decide("research", urgency="background")
    assert ad.action == "defer"
    assert ad.deferred is True
    assert "defer" in ad.note.lower()


def test_background_operation_name_implies_deferral():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    assert b.decide("research").action == "defer"  # default background op


def test_critical_never_deferred_below_hard_cap():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    ad = b.decide("chat", urgency="critical")
    assert ad.action in ("downgrade", "truncate")
    assert ad.deferred is False


def test_critical_truncates_only_at_95_percent():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    assert b.decide("chat", urgency="critical").action == "downgrade"
    b.record("chat", prompt_tokens=150, completion_tokens=0, cost_usd=0.0)
    assert b.decide("chat", urgency="critical").action == "truncate"


# ── 4. honest budget-exceeded ──────────────────────────────────────────────

def test_block_at_100_percent_names_budget_and_spend():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=1000, completion_tokens=0, cost_usd=0.0)
    ad = b.decide("chat")
    assert ad.action == "block"
    assert "budget" in ad.note.lower()
    assert "1000" in ad.note  # the budget, named


def test_block_beats_defer_for_background():
    b = AdaptiveBudget(_cfg())
    b.record("chat", prompt_tokens=1000, completion_tokens=0, cost_usd=0.0)
    assert b.decide("research", urgency="background").action == "block"


# ── 5. config respected ────────────────────────────────────────────────────

def test_per_operation_budget_blocks_only_that_operation():
    b = AdaptiveBudget(_cfg(per_operation={"research": {"tokens": 100}}))
    b.record("research", prompt_tokens=100, completion_tokens=0, cost_usd=0.0)
    assert b.decide("research").action == "block"
    assert b.decide("chat").action == "allow"


def test_custom_thresholds_respected():
    b = AdaptiveBudget(_cfg(downgrade_threshold=0.9, truncate_threshold=0.95,
                            defer_threshold=0.95))
    b.record("chat", prompt_tokens=500, completion_tokens=0, cost_usd=0.0)
    assert b.decide("chat").action == "allow"


def test_unlimited_is_pure_allow():
    b = AdaptiveBudget(AdaptiveBudgetConfig())
    b.record("chat", prompt_tokens=10_000_000, completion_tokens=0, cost_usd=999.0)
    assert b.decide("chat").action == "allow"


def test_day_rollover_resets_spend():
    clock = [1_700_000_000.0]
    b = AdaptiveBudget(_cfg(), clock=lambda: clock[0])
    b.record("chat", prompt_tokens=900, completion_tokens=0, cost_usd=0.0)
    assert b.decide("chat").action == "truncate"
    clock[0] += 86400  # next UTC day
    assert b.decide("chat").action == "allow"
    assert b.spent()["tokens"] == 0


# ── 6. router wiring ───────────────────────────────────────────────────────

class _MockProvider:
    name = "mock"
    model_id = "mock-model"
    capabilities = {"chat", "complete"}

    def __init__(self):
        self.seen_messages = None
        self.seen_params = None

    def chat(self, messages, params=None, **kw):
        from nomorals.llm.base import LLMResponse, Usage
        self.seen_messages = list(messages)
        self.seen_params = params
        return LLMResponse(text="ok", model="mock-model", provider="mock",
                           usage=Usage(prompt_tokens=10, completion_tokens=5))


def _router_with_budget(**cfg_kw):
    budget = AdaptiveBudget(_cfg(**cfg_kw))
    r = LLMRouter(adaptive_budget=budget)
    p = _MockProvider()
    r.add(p, name="mock")
    return r, p, budget


def test_router_truncates_context_and_marks_degraded():
    r, p, budget = _router_with_budget(full_context_tokens=1024,
                                      min_context_tokens=256)
    budget.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    resp = r.chat(_msgs())
    assert resp.ok
    assert resp.degraded is True
    assert resp.adaptation_note != ""
    assert "truncat" in resp.adaptation_note.lower()
    assert p.seen_messages is not None
    assert len(p.seen_messages) < len(_msgs())
    # Sacred turns survive: system first, latest user question last.
    assert p.seen_messages[0].role == "system"
    assert p.seen_messages[-1].content == "final question"


def test_router_defers_background_honestly():
    r, p, budget = _router_with_budget()
    budget.record("chat", prompt_tokens=800, completion_tokens=0, cost_usd=0.0)
    resp = r.chat(_msgs(), task_kind="research", urgency="background")
    assert not resp.ok
    assert resp.failure_class == "budget"
    assert resp.deferred is True
    assert "defer" in resp.error.lower()
    assert p.seen_messages is None  # no call was made


def test_router_blocks_at_100_percent_without_calling():
    r, p, budget = _router_with_budget()
    budget.record("chat", prompt_tokens=1000, completion_tokens=0, cost_usd=0.0)
    resp = r.chat(_msgs())
    assert not resp.ok
    assert resp.failure_class == "budget"
    assert "budget" in resp.error.lower()
    assert p.seen_messages is None


def test_router_records_actuals_and_adapts_next_call():
    r, p, budget = _router_with_budget()
    assert budget.spent()["calls"] == 0
    resp = r.chat(_msgs())
    assert resp.ok
    assert budget.spent()["calls"] == 1
    assert budget.spent()["tokens"] == 15  # mock usage 10 + 5


def test_router_adaptation_visible_in_stats():
    r, p, budget = _router_with_budget()
    budget.record("chat", prompt_tokens=500, completion_tokens=0, cost_usd=0.0)
    snap = r.stats_snapshot()
    assert snap["adaptive_budget"]["utilisation"] == pytest.approx(0.5)
    assert snap["adaptive_budget"]["action"] == "downgrade"


def test_router_explicit_route_wins_over_adaptation():
    r, p, budget = _router_with_budget()
    budget.record("chat", prompt_tokens=500, completion_tokens=0, cost_usd=0.0)
    # Caller explicitly asked for latency sort: adaptation must not override.
    resp = r.chat(_msgs(), route={"sort": "latency"})
    assert resp.ok


def test_config_round_trip():
    from nomorals.core.config import AdaptiveBudgetSettings, Settings
    s = Settings()
    assert hasattr(s, "adaptive_budget")
    assert isinstance(s.adaptive_budget, AdaptiveBudgetSettings)
    cfg = AdaptiveBudgetConfig.from_settings(s)
    assert cfg.daily_token_budget == 0  # unlimited by default
