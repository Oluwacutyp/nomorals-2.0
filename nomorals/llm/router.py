"""Provider routing with fallback and hot-swap.

The router is what makes "which model am I talking to?" a runtime decision instead
of a config constant. It maintains an ordered chain, tracks per-provider health and
latency, fails over automatically, and can swap the active model while calls are in
flight — new calls use the new model, running ones finish on the old one.

Failover is bounded: after the chain is exhausted the caller gets a typed error
rather than an exception from whichever provider happened to be last.
"""

from __future__ import annotations

import json
import random
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.errors import ModelError, ProviderUnavailable, classify
from ..core.events import EventBus
from ..core.logging_setup import get_logger
from ..core.retry import CircuitBreaker, CircuitOpen
from ..research.costs import COST_TABLE
from .base import LLMProvider, LLMResponse, Message, SamplingParams

__all__ = [
    "LLMRouter",
    "ProviderHealth",
    "CostLog",
    "estimate_cost",
    "log_llm_call",
    "get_cost_log",
    "total_spend",
]

_log = get_logger(__name__)


# ── per-call cost logging (build-map #18) ────────────────────────────────────
#
# Every successful _dispatch records provider, model, tier, token usage and
# an estimated USD cost to ~/.nomorals/llm/cost.jsonl.  This is the metered
# counterpart to the research budget's planning estimates (#7): the budget
# caps a run *before* it spends; the cost log records what was *actually*
# spent, per call.  Costs are estimates, not invoices — providers change
# prices, and free tiers cost nothing.

#: Flat per-call planning estimate reused from the research budget (#7)
#: when a model has no token price below.
_FLAT_LLM_CALL_USD: float = float(COST_TABLE.get("llm_call", 0.0008))

#: USD per 1M tokens (prompt, completion) — rough public pricing for the
#: providers Devon actually uses.  Matched by provider-name substring,
#: then model-name substring.  Local/free providers are (0.0, 0.0).
_TOKEN_PRICES: dict[str, tuple[float, float]] = {
    "groq": (0.35, 0.40),
    "hf_serverless": (0.50, 0.50),
    "openrouter": (1.50, 3.00),
    "openai_compat": (2.50, 10.00),
    "llama_cpp": (0.0, 0.0),
    "mock": (0.0, 0.0),
    "ocr": (0.0, 0.0),
}


def _default_cost_path() -> Path:
    return Path.home() / ".nomorals" / "llm" / "cost.jsonl"


def estimate_cost(
    provider_name: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> float:
    """Estimated USD for one LLM call.  Never raises.

    Token-priced when the provider/model is in the table, otherwise the
    flat research planning estimate (an overestimate for free tiers —
    deliberately conservative).  Cached input tokens are priced at 0.1x
    (the Anthropic/OpenAI convention); reasoning tokens ride the output
    price — they are billed even though the user never reads them, which
    is exactly why they must be counted.
    """
    try:
        price: tuple[float, float] | None = None
        for needle in ((provider_name or "").lower(), (model or "").lower()):
            if not needle:
                continue
            for name, p in _TOKEN_PRICES.items():
                if name in needle:
                    price = p
                    break
            if price is not None:
                break
        if price is None:
            return _FLAT_LLM_CALL_USD
        per_m_prompt, per_m_completion = price
        fresh_prompt = max(0, prompt_tokens - max(0, cached_tokens))
        return round(
            (fresh_prompt / 1_000_000) * per_m_prompt
            + (max(0, cached_tokens) / 1_000_000) * per_m_prompt * 0.1
            + (max(0, completion_tokens + reasoning_tokens) / 1_000_000)
            * per_m_completion,
            9,
        )
    except Exception:  # noqa: BLE001 — cost math never breaks the caller
        return _FLAT_LLM_CALL_USD


class CostLog:
    """Append-only JSONL log of metered LLM spend.  Thread-safe."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else _default_cost_path()
        self._lock = threading.Lock()

    def record(
        self,
        *,
        provider: str,
        model: str = "",
        tier: str = "",
        operation: str = "chat",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_usd: float = 0.0,
        latency_ms: float = 0.0,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        feature: str = "",
        request_id: str = "",
    ) -> dict[str, Any]:
        """Append one call record.  Never raises; returns the entry."""
        entry: dict[str, Any] = {
            "ts": time.time(),
            "provider": provider,
            "model": model,
            "tier": tier,
            "operation": operation,
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "cost_usd": round(float(cost_usd), 9),
            "latency_ms": round(float(latency_ms), 2),
            "cached_tokens": int(cached_tokens),
            "reasoning_tokens": int(reasoning_tokens),
            "feature": feature or "",
            "request_id": request_id or "",
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(entry) + "\n")
        except Exception:  # noqa: BLE001 — logging never breaks routing
            _log.debug("cost log write failed", exc_info=True)
        return entry

    def entries(self, since: float = 0.0) -> list[dict[str, Any]]:
        """Read back entries with ts >= since.  Never raises."""
        out: list[dict[str, Any]] = []
        try:
            with open(self.path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and float(row.get("ts", 0.0)) >= since:
                        out.append(row)
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            _log.debug("cost log read failed", exc_info=True)
        return out

    def total_spend(self, since: float = 0.0) -> float:
        """Sum of cost_usd for entries with ts >= since.  Never raises."""
        try:
            return round(
                sum(float(r.get("cost_usd", 0.0)) for r in self.entries(since)), 9
            )
        except Exception:  # noqa: BLE001
            return 0.0

    def breakdown(self, since: float = 0.0) -> dict[str, Any]:
        """Per-operation/per-provider aggregates (Langfuse/Helicone shape).

        Answers "which feature is burning the budget" in one call: totals
        plus per-operation and per-provider slices with calls, tokens,
        cost and average latency.  Never raises.
        """
        rows = self.entries(since)
        total = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                 "cached_tokens": 0, "reasoning_tokens": 0,
                 "cost_usd": 0.0, "latency_ms": 0.0}
        by_operation: dict[str, dict[str, Any]] = {}
        by_provider: dict[str, dict[str, Any]] = {}
        for r in rows:
            for key, val in (
                ("calls", 1),
                ("prompt_tokens", r.get("prompt_tokens", 0)),
                ("completion_tokens", r.get("completion_tokens", 0)),
                ("cached_tokens", r.get("cached_tokens", 0)),
                ("reasoning_tokens", r.get("reasoning_tokens", 0)),
                ("cost_usd", r.get("cost_usd", 0.0)),
                ("latency_ms", r.get("latency_ms", 0.0)),
            ):
                try:
                    total[key] += float(val)
                except (TypeError, ValueError):
                    pass
            for bucket, name in (
                (by_operation, str(r.get("operation") or "chat")),
                (by_provider, str(r.get("provider") or "unknown")),
            ):
                slot = bucket.setdefault(name, {"calls": 0, "prompt_tokens": 0,
                                                "completion_tokens": 0,
                                                "cost_usd": 0.0,
                                                "latency_ms": 0.0})
                slot["calls"] += 1
                for key in ("prompt_tokens", "completion_tokens", "cost_usd",
                            "latency_ms"):
                    try:
                        slot[key] += float(r.get(key, 0.0))
                    except (TypeError, ValueError):
                        pass
        for bucket in (by_operation, by_provider):
            for slot in bucket.values():
                calls = slot["calls"] or 1
                slot["avg_latency_ms"] = round(slot.pop("latency_ms") / calls, 1)
                slot["cost_usd"] = round(slot["cost_usd"], 9)
        total["calls"] = len(rows)
        total["cost_usd"] = round(total["cost_usd"], 9)
        total["avg_latency_ms"] = round(
            total.pop("latency_ms") / (len(rows) or 1), 1)
        return {"total": total, "by_operation": by_operation,
                "by_provider": by_provider}

    def budget_report(self, daily_budget: float,
                      since: float = 0.0) -> dict[str, Any]:
        """Spend vs budget with alert levels (50/80/100%).

        Never raises.  ``alert`` is "ok" | "watch" | "warning" | "exceeded".
        """
        try:
            spent = self.total_spend(since)
            budget = max(0.0, float(daily_budget))
        except Exception:  # noqa: BLE001
            return {"spent_usd": 0.0, "budget_usd": 0.0, "alert": "ok",
                    "pct": 0.0}
        if budget <= 0:
            return {"spent_usd": spent, "budget_usd": 0.0, "alert": "ok",
                    "pct": 0.0, "unlimited": True}
        pct = spent / budget * 100.0
        alert = "ok"
        if pct >= 100:
            alert = "exceeded"
        elif pct >= 80:
            alert = "warning"
        elif pct >= 50:
            alert = "watch"
        return {
            "spent_usd": spent,
            "budget_usd": budget,
            "remaining_usd": round(max(0.0, budget - spent), 9),
            "pct": round(pct, 1),
            "alert": alert,
        }


_cost_log: CostLog | None = None
_cost_log_lock = threading.Lock()


def _shared_log(path: str | Path | None = None) -> CostLog:
    global _cost_log
    if path is not None:
        return CostLog(path)
    with _cost_log_lock:
        if _cost_log is None:
            _cost_log = CostLog()
        return _cost_log


def log_llm_call(
    provider_name: str,
    model: str,
    response: Any,
    *,
    tier: str = "",
    operation: str = "chat",
    path: str | Path | None = None,
) -> float:
    """Record one successful LLM call in the cost log.

    Returns the estimated USD.  Never raises — safe to call on the hot path.
    Set ``NM_COST_LOG=off`` to skip recording (handy for test harnesses);
    the estimate is still returned.
    """
    import os as _os

    usage = getattr(response, "usage", None)
    prompt_t = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_t = int(getattr(usage, "completion_tokens", 0) or 0)
    cached_t = int(getattr(usage, "cached_tokens", 0) or 0)
    reasoning_t = int(getattr(usage, "reasoning_tokens", 0) or 0)
    feature = str(getattr(response, "feature", "") or "")
    request_id = str(getattr(response, "request_id", "") or "")
    if path is None and _os.environ.get("NM_COST_LOG", "").strip().lower() in (
        "off", "0", "no",
    ):
        # Opt-out of the default home-dir log (test harnesses); an explicit
        # path always records.  The estimate is still returned.
        return estimate_cost(provider_name, model, prompt_t, completion_t,
                             cached_tokens=cached_t,
                             reasoning_tokens=reasoning_t)
    try:
        cost = estimate_cost(provider_name, model, prompt_t, completion_t,
                             cached_tokens=cached_t,
                             reasoning_tokens=reasoning_t)
        try:
            response.cost_usd = cost
        except Exception:  # noqa: BLE001 — duck-typed responses
            pass
        _shared_log(path).record(
            provider=provider_name,
            model=model or "",
            tier=tier or "",
            operation=operation,
            prompt_tokens=prompt_t,
            completion_tokens=completion_t,
            cost_usd=cost,
            latency_ms=float(getattr(response, "latency_ms", 0.0) or 0.0),
            cached_tokens=cached_t,
            reasoning_tokens=reasoning_t,
            feature=feature,
            request_id=request_id,
        )
        return cost
    except Exception:  # noqa: BLE001
        _log.debug("log_llm_call failed", exc_info=True)
        return 0.0


def get_cost_log(since: float = 0.0, path: str | Path | None = None) -> list[dict[str, Any]]:
    """Cost-log entries with ts >= since.  Never raises."""
    return _shared_log(path).entries(since)


def total_spend(since: float = 0.0, path: str | Path | None = None) -> float:
    """Total metered USD spend since ts.  Never raises."""
    return _shared_log(path).total_spend(since)


@dataclass
class ProviderHealth:
    """Rolling health for one provider.

    The skip/retry decision is driven by :attr:`breaker` — the shared
    three-state :class:`~nomorals.core.retry.CircuitBreaker` (closed →
    open → half-open probe).  ``cooldown_until`` is kept in sync as the
    dashboard-facing surface (the console reads it directly).
    """

    name: str
    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    last_error: str = ""
    total_latency_ms: float = 0.0
    cooldown_until: float = 0.0
    breaker: CircuitBreaker | None = field(default=None, repr=False)
    #: Typed failure class of the most recent failure
    #: (``nomorals.llm.failures.FailureClass`` value, "" when none).
    failure_class: str = ""
    #: Rolling successful-attempt latencies (bounded).  Drives the p50/p95
    #: used by latency-SLO shedding and adaptive timeouts — the *average*
    #: hides the tail that actually hurts interactive calls.
    latencies: deque = field(default_factory=lambda: deque(maxlen=50),
                             repr=False)

    @property
    def error_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    @property
    def p50_latency_ms(self) -> float:
        try:
            return round(statistics.median(self.latencies), 2) \
                if self.latencies else 0.0
        except statistics.StatisticsError:
            return 0.0

    @property
    def p95_latency_ms(self) -> float:
        try:
            if not self.latencies:
                return 0.0
            ordered = sorted(self.latencies)
            idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
            return round(ordered[idx], 2)
        except Exception:  # noqa: BLE001
            return 0.0

    def available(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) >= self.cooldown_until

    def record_success(self, latency_ms: float = 0.0) -> None:
        self.calls += 1
        self.consecutive_failures = 0
        self.last_success = time.time()
        if latency_ms > 0:
            self.total_latency_ms += latency_ms
            self.latencies.append(float(latency_ms))

    def record_failure(self, error: str, cooldown: float,
                       now: float | None = None) -> None:
        self.calls += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure = time.time()
        self.last_error = error
        if cooldown > 0:
            # `now` lets the router pass its injected clock (tests); the
            # default keeps the old time.monotonic() behaviour.
            self.cooldown_until = (time.monotonic() if now is None else now) + cooldown

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "failures": self.failures,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p50_latency_ms": self.p50_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "cooling_down": not self.available(),
            "breaker_state": self.breaker.state if self.breaker is not None else "closed",
            "failure_class": self.failure_class,
        }


class LLMRouter:
    """Routes calls across a chain of providers.

        router = LLMRouter()
        router.add(hf_provider, primary=True)
        router.add(mock_provider)
        reply = router.chat([Message.user("hi")])
    """

    def __init__(
        self,
        *,
        cooldown_seconds: float = 30.0,
        rate_limit_cooldown_seconds: float = 120.0,
        failure_threshold: int = 3,
        max_cooldown_seconds: float = 900.0,
        auth_cooldown_seconds: float = 600.0,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        repair_hooks: dict[str, Callable] | list[Callable] | None = None,
        latency_slo_ms: float = 0.0,
        half_open_probe_budget: int = 2,
        daily_budget_usd: float = 0.0,
        cost_log_path: str | Path | None = None,
    ) -> None:
        self._providers: list[LLMProvider] = []
        self._by_name: dict[str, LLMProvider] = {}
        self._health: dict[str, ProviderHealth] = {}
        self._active: str = ""
        self._lock = threading.RLock()
        self.cooldown_seconds = cooldown_seconds
        # Rate limits (429) get a longer cooldown — the quota window is
        # typically 60s, so hammering the provider every 30s just burns
        # more quota. Back off for 2 minutes on rate limits.
        self.rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self.failure_threshold = failure_threshold
        # Auth/config failures (bad key, retired model name) are not
        # transient: park the provider long so we stop burning quota on a
        # credential that will never work until the operator fixes it.
        self.auth_cooldown_seconds = auth_cooldown_seconds
        # Circuit breaker: the cooldown grows exponentially with consecutive
        # failures (LiteLLM's cooldown_time pattern taken one step further —
        # a provider that fails 20 times in a row is not "transient", it is
        # down, and should not be re-probed every 30s).  Capped so a long
        # outage does not park a provider forever.
        self.max_cooldown_seconds = max_cooldown_seconds
        self.bus = bus
        self._clock = clock
        self.repair_hooks = repair_hooks if repair_hooks is not None else {}
        self.repair_cooldown_seconds = cooldown_seconds
        self._last_repair_time: dict[str, float] = {}
        self.stats = {"calls": 0, "failovers": 0, "failures": 0, "repairs": 0,
                      "slo_sheds": 0, "budget_blocks": 0, "key_rotations": 0}
        # Latency SLO shedding (nexus-llm-router `latency-slo-shed`): when
        # set, a provider whose rolling p95 exceeds the SLO is skipped while
        # faster alternatives exist.  0 = disabled.
        self.latency_slo_ms = max(0.0, float(latency_slo_ms))
        # Half-open probe budget (nexus `circuit-breaker-half-open-probe`):
        # at most this many concurrent probes into recovering providers —
        # a recovering provider must not absorb the whole fleet's traffic.
        self.half_open_probe_budget = max(1, int(half_open_probe_budget))
        self._half_open_probes: dict[str, int] = {}
        # Daily spend cap (gateway-style budget).  0 = unlimited.  When the
        # metered spend for the current UTC day reaches the cap, calls come
        # back as a typed budget error instead of spending more.
        self.daily_budget_usd = max(0.0, float(daily_budget_usd))
        self._budget_warned: bool = False
        self._budget_checked_at: float = 0.0
        self._budget_cached_spent: float = 0.0
        #: Override for the metered cost log (tests); None → the shared
        #: ~/.nomorals/llm/cost.jsonl.
        self._cost_log_path = Path(cost_log_path) if cost_log_path else None
        # Optional capability broker (nomorals.llm.broker.ModelBroker).  When
        # unset the router keeps its original name-based behaviour exactly.
        self._broker: Any = None
        # Optional learning hook (nomorals.llm.learning.attach_learning).
        # Duck-typed on purpose: the router never imports the learning
        # module, so a broken or missing learning stack cannot break
        # routing.  While unset the dispatch path below is exactly the
        # pre-learning code.
        self._learning: Any = None

    def _cost_log(self) -> CostLog:
        if self._cost_log_path is not None:
            return CostLog(self._cost_log_path)
        return _shared_log()

    # ── registration ─────────────────────────────────────────────────────────
    def add(self, provider: LLMProvider, *, primary: bool = False, name: str = "") -> LLMRouter:
        key = name or provider.name
        with self._lock:
            if key in self._by_name:
                raise ValueError(f"provider {key!r} already registered")
            # Wrap in a lightweight shim so the same class can appear twice under
            # different names (e.g. two different HF models).
            provider.name = key
            self._by_name[key] = provider
            health = ProviderHealth(name=key)
            # One circuit breaker per provider, on the router's clock so
            # fake-clock tests and the cooldown math agree.  The breaker
            # drives the skip decision; ProviderHealth stays the stats
            # surface the console reads.
            health.breaker = CircuitBreaker(
                name=f"llm:{key}",
                failure_threshold=self.failure_threshold,
                reset_timeout=self.cooldown_seconds,
                clock=self._clock,
            )
            self._health[key] = health
            if primary or not self._active:
                self._providers.insert(0, provider)
                self._active = key
            else:
                self._providers.append(provider)
        return self

    def remove(self, name: str) -> bool:
        with self._lock:
            provider = self._by_name.pop(name, None)
            if provider is None:
                return False
            self._providers = [p for p in self._providers if p.name != name]
            self._health.pop(name, None)
            if self._active == name:
                self._active = self._providers[0].name if self._providers else ""
        return True

    def providers(self) -> list[str]:
        with self._lock:
            return [p.name for p in self._providers]

    def get(self, name: str) -> LLMProvider | None:
        with self._lock:
            return self._by_name.get(name)

    @property
    def active(self) -> str:
        with self._lock:
            return self._active

    @property
    def active_model(self) -> str:
        """Alias for ``active`` — used by the CLI."""
        return self.active

    # ── hot-swap ─────────────────────────────────────────────────────────────
    def set_active(self, name: str) -> str:
        """Switch the model used by *new* calls. In-flight calls are unaffected."""
        with self._lock:
            if name not in self._by_name:
                raise KeyError(f"unknown provider {name!r}; known: {sorted(self._by_name)}")
            previous = self._active
            self._active = name
            provider = self._by_name[name]
            self._providers = [provider] + [p for p in self._providers if p.name != name]
        if previous != name:
            _log.info("active model swapped: %s -> %s", previous or "(none)", name)
            if self.bus is not None:
                self.bus.emit("llm.swapped", previous=previous, active=name)
        return name

    def reorder(self, order: Sequence[str]) -> None:
        """Set the explicit fallback order."""
        with self._lock:
            missing = [n for n in order if n not in self._by_name]
            if missing:
                raise KeyError(f"unknown providers in order: {missing}")
            ordered = [self._by_name[n] for n in order]
            rest = [p for p in self._providers if p.name not in set(order)]
            self._providers = ordered + rest
            self._active = self._providers[0].name

    # ── capability broker ────────────────────────────────────────────────────
    def set_broker(self, broker: Any | None) -> LLMRouter:
        """Attach a :class:`nomorals.llm.broker.ModelBroker`.

        While attached, every :meth:`chat`/:meth:`complete`/:meth:`describe_image`
        first asks the broker which provider should serve the operation and
        moves the active provider there.  The broker is best-effort: if it
        has no candidate (or errors), the router falls back to its existing
        name-based chain unchanged.  Pass ``None`` to detach.
        """
        with self._lock:
            self._broker = broker
        return self

    @property
    def broker(self) -> Any | None:
        with self._lock:
            return self._broker

    # ── startup verification ───────────────────────────────────────────────
    def verify(self, *, timeout_seconds: float = 10.0) -> dict[str, Any]:
        """Probe each provider at startup to catch config errors early.

        Makes a cheap probe call against every registered provider. Providers
        that fail with a 4xx (bad model name, bad key, not hosted) are marked
        unhealthy immediately with a clear message naming the problem —
        instead of discovering it at chat time when the user is waiting.

        Returns a report dict:
            {"ok": [...], "failed": [...], "details": {name: ...}}
        where each detail has "ok" (bool) and "error" (str, if failed).

        This would have prevented the "no working brain" outage: invalid
        model names (e.g. a stale Groq model, an unhosted HF model) are
        caught here, at boot, with the bad model named in the log.
        """
        from .base import Message, SamplingParams

        report: dict[str, Any] = {"ok": [], "failed": [], "details": {}}
        probe_messages = [Message.user("ping")]
        probe_params = SamplingParams(max_tokens=4, temperature=0.0)

        for provider in list(self._providers):
            name = provider.name
            model = getattr(provider, "model", "")
            detail: dict[str, Any] = {"model": model}
            try:
                provider.chat(
                    probe_messages, probe_params, timeout=timeout_seconds,
                )
                detail["ok"] = True
                report["ok"].append(name)
                _log.info("llm verify: %s ok (model=%s)", name, model or "?")
            except Exception as exc:  # noqa: BLE001 - probe must not raise
                err = classify(exc)
                msg = err.message
                detail["ok"] = False
                detail["error"] = msg
                report["failed"].append(name)
                # 4xx = config error (bad model, bad key). Mark unhealthy
                # with a long cooldown so we don't hammer a broken config.
                # Name the model so the user knows what to fix.
                status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
                is_config_error = (
                    (isinstance(status, int) and 400 <= status < 500)
                    or "404" in msg or "not found" in msg.lower()
                    or "not hosted" in msg.lower() or "invalid" in msg.lower()
                )
                if is_config_error:
                    self._note_failure(name, f"config error: {msg} (model={model})")
                    # Extend the cooldown: this isn't transient.
                    health = self._health.get(name)
                    if health is not None:
                        health.cooldown_until = self._clock() + 3600.0
                    _log.warning(
                        "llm verify: %s FAILED — config error (model=%r): %s. "
                        "Marked unhealthy for 1h. Fix the model name or key.",
                        name, model, msg,
                    )
                else:
                    _log.warning("llm verify: %s probe failed (transient?): %s", name, msg)
            report["details"][name] = detail

        if not report["ok"]:
            _log.error(
                "llm verify: NO working providers! Devon has no brain. "
                "Check model names and API keys. Failed: %s",
                ", ".join(report["failed"]) or "none registered",
            )
        return report

    # ── learning hook ────────────────────────────────────────────────────
    def set_learning(self, hook: Any | None) -> LLMRouter:
        """Attach/detach the learning hook.

        ``hook`` is called once per provider *attempt* as
        ``hook(operation=..., provider_name=..., success=..., latency_s=...,
        error=...)``.  It must never block the caller — the learning module
        enqueues and returns.  Any exception it raises is swallowed here:
        learning is advisory and must never break routing.  Pass ``None``
        to detach.
        """
        with self._lock:
            self._learning = hook
        return self

    @property
    def learning(self) -> Any | None:
        with self._lock:
            return self._learning

    def _note_learning(
        self,
        operation: str,
        provider_name: str,
        *,
        success: bool,
        latency_s: float,
        error: str = "",
    ) -> None:
        hook = self._learning
        if hook is None:
            return
        try:
            hook(
                operation=operation,
                provider_name=provider_name,
                success=success,
                latency_s=latency_s,
                error=error,
            )
        except Exception:  # noqa: BLE001 — learning must never break routing
            _log.debug("learning hook raised; ignoring", exc_info=True)

    def is_cooling_down(self, name: str) -> bool:
        """True when the router would currently skip ``name``.

        Used by the broker to avoid promoting a dead provider to active.
        A half-open breaker (probe admitted) does NOT count as cooling —
        the probe must be allowed through.
        """
        with self._lock:
            health = self._health.get(name)
            if health is None:
                return False
            if not health.available(self._clock()):
                return True
            breaker = health.breaker
            return breaker is not None and breaker.state == CircuitBreaker.OPEN

    def cooldown_remaining(self, name: str) -> float:
        """Seconds until ``name`` may be retried (0 when not cooling)."""
        with self._lock:
            health = self._health.get(name)
            if health is None:
                return 0.0
            return max(0.0, health.cooldown_until - self._clock())

    # ── calling ──────────────────────────────────────────────────────────────
    def _chain(self) -> list[LLMProvider]:
        with self._lock:
            if not self._providers:
                return []
            active = self._by_name.get(self._active)
            others = [p for p in self._providers if p.name != self._active]
            return ([active] if active else []) + others

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None,
        tier: str | None = None, task_kind: str = "",
        constraints: Any = None,
        route: dict[str, Any] | None = None, **kw: Any
    ) -> LLMResponse:
        return self._dispatch("chat", lambda p: p.chat(messages, params, **kw),
                              tier=tier, task_kind=task_kind,
                              constraints=constraints, route=route)

    def complete(self, prompt: str, params: SamplingParams | None = None,
                 tier: str | None = None, task_kind: str = "",
                 constraints: Any = None,
                 route: dict[str, Any] | None = None, **kw: Any) -> LLMResponse:
        return self._dispatch("complete", lambda p: p.complete(prompt, params, **kw),
                              tier=tier, task_kind=task_kind,
                              constraints=constraints, route=route)

    def embed(self, texts: Sequence[str], **kw: Any) -> list[list[float]]:
        chain = [p for p in self._chain() if "embed" in p.capabilities]
        if not chain:
            raise ModelError("no registered provider supports embeddings")
        failed: list[str] = []
        last_error = ""
        for provider in chain:
            health = self._health.get(provider.name)
            if health and not health.available(self._clock()):
                continue
            if health and health.breaker is not None:
                try:
                    health.breaker.before_call()
                except CircuitOpen:
                    continue
            try:
                vectors = provider.embed(texts, **kw)
            except Exception as exc:  # noqa: BLE001
                last_error = classify(exc).message
                self._note_failure(provider.name, last_error)
                failed.append(provider.name)
                continue
            self._note_success(provider.name)
            if failed:
                _log.warning("embedding failover: %s failed; served by %s",
                             ", ".join(failed), provider.name)
            return vectors
        raise ProviderUnavailable(f"all embedding providers failed; last error: {last_error}")

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None,
        tier: str | None = None, task_kind: str = "vision",
        constraints: Any = None,
        route: dict[str, Any] | None = None, **kw: Any
    ) -> LLMResponse:
        return self._dispatch(
            "vision",
            lambda p: p.describe_image(image, prompt, params, **kw),
            require="vision",
            tier=tier,
            task_kind=task_kind,
            constraints=constraints,
            route=route,
        )

    # ── routing policy (OpenRouter-style per-request knobs) ──────────────────
    def _route_price(self, provider_name: str) -> float:
        """USD/1M input tokens for a provider (broker card when attached)."""
        try:
            broker = self._broker
            if broker is not None:
                for card in broker.cards():
                    if card.provider == provider_name:
                        return float(card.price_per_1m_in)
        except Exception:  # noqa: BLE001
            pass
        return 0.0

    def _apply_route(
        self,
        chain: list[LLMProvider],
        route: dict[str, Any] | None,
    ) -> list[LLMProvider]:
        """Apply per-request routing knobs (OpenRouter provider-routing style).

        ``route`` keys: ``sort`` ("latency"|"price"|"throughput"),
        ``only`` / ``ignore`` (provider names), ``max_price`` (USD/1M
        input tokens).  Then latency-SLO shedding drops slow providers
        while faster alternatives exist.
        """
        route = route or {}
        only = route.get("only")
        ignore = route.get("ignore")
        names_only = {only} if isinstance(only, str) else set(only or ())
        names_ignore = {ignore} if isinstance(ignore, str) else set(ignore or ())
        out = [p for p in chain
               if (not names_only or p.name in names_only)
               and p.name not in names_ignore]
        max_price = route.get("max_price")
        if max_price is not None:
            try:
                cap = float(max_price)
                out = [p for p in out if self._route_price(p.name) <= cap]
            except (TypeError, ValueError):
                pass
        if self.latency_slo_ms > 0 and len(out) > 1:
            def _under_slo(p: LLMProvider) -> bool:
                health = self._health.get(p.name)
                if health is None:
                    return True
                p95 = health.p95_latency_ms
                return p95 <= 0.0 or p95 <= self.latency_slo_ms

            fast = [p for p in out if _under_slo(p)]
            if fast and len(fast) < len(out):
                shed = [p.name for p in out if p not in fast]
                self.stats["slo_sheds"] += len(shed)
                _log.info("slo shed (p95 > %.0fms): %s",
                          self.latency_slo_ms, ", ".join(shed))
                out = fast
        sort = str(route.get("sort") or "").lower()
        if sort in ("latency", "throughput"):
            out.sort(key=lambda p: (
                self._health[p.name].p50_latency_ms or float("inf"), p.name))
        elif sort == "price":
            out.sort(key=lambda p: (self._route_price(p.name), p.name))
        return out

    def _rotate_provider_key(self, provider: LLMProvider) -> bool:
        """Bifrost-style key pooling: rotate to the next API key on 429.

        A different key from the SAME provider is tried before failing over
        to another provider — the quota that is exhausted is usually the
        key's, not the provider's.  True when a rotation happened.
        """
        rotate = getattr(provider, "rotate_key", None)
        if not callable(rotate):
            return False
        try:
            if rotate():
                self.stats["key_rotations"] += 1
                _log.info("key pool rotation on %s (rate limited)",
                          provider.name)
                return True
        except Exception:  # noqa: BLE001
            _log.debug("key rotation failed on %s", provider.name,
                       exc_info=True)
        return False

    def _attempt_with_rotation(
        self,
        provider: LLMProvider,
        call: Callable[[LLMProvider], LLMResponse],
    ) -> tuple[LLMResponse, str]:
        """One provider attempt, rotating the key pool once on 429.

        Returns (response, failure_class).  ``response.ok`` is False on
        failure; the caller records it exactly once.
        """
        from .failures import FailureClass, classify_failure

        for key_try in range(2):
            try:
                response = call(provider)
            except Exception as exc:  # noqa: BLE001 - never let a provider crash the router
                error = classify(exc)
                note = f"{error.code}: {error.message}"
                info = classify_failure(note, exc)
                if (info.failure_class is FailureClass.RATE_LIMITED
                        and key_try == 0
                        and self._rotate_provider_key(provider)):
                    continue
                return (LLMResponse(
                    text="",
                    model=provider.model_id,
                    provider=provider.name,
                    error=note,
                ), info.failure_class.value)
            if response.ok:
                return response, ""
            note = response.error or "unknown error"
            info = classify_failure(note)
            if (info.failure_class is FailureClass.RATE_LIMITED
                    and key_try == 0
                    and self._rotate_provider_key(provider)):
                continue
            return response, info.failure_class.value
        return (LLMResponse(text="", model=provider.model_id,
                            provider=provider.name,
                            error="key rotation exhausted"),
                FailureClass.RATE_LIMITED.value)

    def _budget_block(self, operation: str) -> LLMResponse | None:
        """Typed budget error when the daily cap is reached, else None.

        The spend check is cached for 30s so the metered log is not read on
        every call.  Never raises — a broken budget check must not break
        routing.
        """
        if self.daily_budget_usd <= 0:
            return None
        try:
            from .failures import FailureClass

            now = time.time()
            if now - self._budget_checked_at < 30.0:
                spent = self._budget_cached_spent
            else:
                day_start = now - (now % 86400)  # UTC day boundary
                spent = self._cost_log().total_spend(day_start)
                self._budget_checked_at = now
                self._budget_cached_spent = spent
            if spent >= self.daily_budget_usd:
                self.stats["budget_blocks"] += 1
                return LLMResponse(
                    text="",
                    error=(f"daily LLM budget ${self.daily_budget_usd:.2f} "
                           f"reached (spent ${spent:.4f}); no further calls "
                           f"until the next UTC day"),
                    failure_class=FailureClass.BUDGET.value,
                )
            if (not self._budget_warned
                    and spent >= 0.8 * self.daily_budget_usd
                    and self.bus is not None):
                self._budget_warned = True
                try:
                    self.bus.emit("llm.budget_warning", spent_usd=spent,
                                  budget_usd=self.daily_budget_usd,
                                  operation=operation)
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001 — budget must never break routing
            _log.debug("budget check failed", exc_info=True)
        return None

    def _dispatch(
        self,
        operation: str,
        call: Callable[[LLMProvider], LLMResponse],
        *,
        require: str | None = None,
        tier: str | None = None,
        task_kind: str = "",
        constraints: Any = None,
        route: dict[str, Any] | None = None,
    ) -> LLMResponse:
        # Budget gate: a capped router returns a typed error instead of
        # spending.
        blocked = self._budget_block(operation)
        if blocked is not None:
            return blocked
        # Broker consult (opt-in): let the capability broker pick the starting
        # provider for this operation.  The task_kind is threaded through so
        # the broker's task specialisation (code → code-tuned, vision → VLM)
        # actually fires — previously consult() dropped it and every call
        # looked like generic chat.  Best-effort — on any failure the
        # original name-based chain below is used untouched.
        if self._broker is not None:
            try:
                self._broker.consult(self, operation, task_kind, constraints)
            except Exception:  # noqa: BLE001 — broker must never break routing
                _log.debug("broker consult raised; using name-based chain",
                           exc_info=True)
        chain = self._chain()
        if require is not None:
            capable = [p for p in chain if require in p.capabilities]
            if not capable:
                # mirror embed(): never silently fall back to providers that
                # can't do the job — the owner gets one clean error instead
                # of a chain of doomed attempts.
                raise ModelError(f"no registered provider supports {require}")
            chain = capable
        if not chain:
            return LLMResponse(text="", error="no providers registered")
        chain = self._apply_route(chain, route)
        if not chain:
            return LLMResponse(
                text="", error="routing rules excluded every provider",
                failure_class="config")
        allow_fallbacks = True if route is None \
            else bool(route.get("allow_fallbacks", True))

        self.stats["calls"] += 1
        attempted = 0
        failed: list[tuple[str, str]] = []  # (provider name, error) in attempt order
        trace: list[dict[str, Any]] = []  # per-attempt routing trace
        last = LLMResponse(text="", error="no attempt made")
        learning = self._learning  # read once; a detach mid-dispatch is harmless
        for provider in chain:
            health = self._health[provider.name]
            if not health.available(self._clock()):
                continue
            # Circuit breaker gate: an open breaker rejects without touching
            # the dependency; a half-open breaker admits probes within the
            # probe budget — a recovering provider must not absorb the whole
            # fleet's traffic at once.
            probing = False
            if health.breaker is not None:
                try:
                    health.breaker.before_call()
                except CircuitOpen:
                    continue
                if health.breaker.state == CircuitBreaker.HALF_OPEN:
                    with self._lock:
                        in_flight = self._half_open_probes.get(provider.name, 0)
                        if in_flight >= self.half_open_probe_budget:
                            continue
                        self._half_open_probes[provider.name] = in_flight + 1
                    probing = True
            attempted += 1
            attempt_start = self._clock()
            try:
                response, failure_class = self._attempt_with_rotation(
                    provider, call)
            finally:
                if probing:
                    with self._lock:
                        self._half_open_probes[provider.name] = max(
                            0, self._half_open_probes.get(provider.name, 1) - 1)
            latency_ms = (self._clock() - attempt_start) * 1000.0
            if response.ok:
                self._note_success(provider.name, latency_ms)
                trace.append({"provider": provider.name, "ok": True,
                              "latency_ms": round(latency_ms, 1), "error": ""})
                response.route_trace = list(trace)
                if learning is not None:
                    self._note_learning(
                        operation, provider.name, success=True,
                        latency_s=latency_ms / 1000.0,
                    )
                # Per-call cost logging (build-map #18): metered spend for
                # the call that just succeeded.  Best-effort and silent —
                # it must never add latency or break the hot path.
                log_llm_call(
                    provider.name,
                    getattr(provider, "model_id", "") or "",
                    response,
                    tier=tier or "",
                    operation=operation,
                    path=self._cost_log_path,
                )
                if failed:
                    # A failover happened: the response must say WHICH
                    # provider failed and WHAT fallback served it — the
                    # final answer alone would hide the degradation.
                    self.stats["failovers"] += 1
                    response.degraded = True
                    response.failed_providers = [name for name, _ in failed]
                    response.fallback_note = (
                        "; ".join(f"{name} failed ({err})" for name, err in failed)
                        + f"; served by {provider.name}"
                    )
                    _log.warning(
                        "provider failover for %s: %s",
                        operation,
                        response.fallback_note,
                    )
                return response
            self._note_failure(provider.name, response.error)
            failed.append((provider.name, response.error or "unknown error"))
            trace.append({"provider": provider.name, "ok": False,
                          "latency_ms": round(latency_ms, 1),
                          "error": response.error or "unknown error"})
            last = response
            if learning is not None:
                self._note_learning(
                    operation, provider.name, success=False,
                    latency_s=latency_ms / 1000.0,
                    error=response.error or "unknown error",
                )
            if not allow_fallbacks:
                break

        self.stats["failures"] += 1
        if attempted == 0:
            last.error = "all providers are cooling down"
        # Even a total failure reports the whole attempt chain.
        last.failed_providers = [name for name, _ in failed]
        last.route_trace = list(trace)
        if failed:
            chain_note = "; ".join(f"{name} failed ({err})" for name, err in failed)
            last.fallback_note = (f"{chain_note}; no provider served this call"
                                  if not last.fallback_note else last.fallback_note)
        # Typed failure class on the terminal response, so the brain and
        # callers recover the RIGHT way (shrink context vs back off vs
        # report a bad key) instead of guessing from the message text.
        if last.error and not last.failure_class:
            from .failures import classify_failure
            try:
                last.failure_class = classify_failure(last.error).failure_class.value
            except Exception:  # noqa: BLE001 — tagging never breaks the call
                pass
        # Total brain failure is operator-visible: emit a bus event so
        # monitoring/alerting sees it, not just the caller's error text.
        try:
            bus = getattr(self, "bus", None)
            if bus is not None:
                bus.emit(
                    "llm.brain_down",
                    operation=operation,
                    attempted=attempted,
                    failed_providers=[name for name, _ in failed],
                    error=last.error or "",
                )
        except Exception:  # noqa: BLE001 — telemetry never breaks dispatch
            pass
        return last

    def _note_success(self, name: str, latency_ms: float = 0.0) -> None:
        health = self._health.get(name)
        if health is not None:
            health.record_success(latency_ms)
            if health.breaker is not None:
                # A success closes the breaker — including after a half-open
                # probe, which is how a recovered provider rejoins rotation.
                health.breaker.record_success()
        # Error-budget heartbeat: feeds real success ratios (not failure-only).
        try:
            from ..core.error_system import heartbeat
            heartbeat(f"llm/{name}", True)
        except Exception:  # noqa: BLE001 - telemetry never breaks routing
            pass

    def _note_failure(self, name: str, error: str) -> None:
        from .failures import FailureClass, classify_failure

        health = self._health.get(name)
        if health is None:
            return
        # Typed failure, typed recovery.  The old code treated every
        # failure the same (substring "429" check + exponential cooldown);
        # the taxonomy distinguishes rate-limit vs auth vs context-overflow
        # vs network, each with the RIGHT recovery:
        #
        # * context_overflow is a CALLER-side bug (prompt too big) — never
        #   park the provider for it, and never feed the breaker.  The
        #   brain shrinks the context and retries instead.
        # * auth/config are not transient — park long, stop burning quota.
        # * rate limits keep the long backoff; everything else the classic
        #   threshold → exponential path.
        info = classify_failure(error)
        fc = info.failure_class
        health.failure_class = fc.value
        consecutive = health.consecutive_failures + 1  # this failure included
        feed_breaker = True
        if fc is FailureClass.CONTEXT_OVERFLOW:
            cooldown = 0.0
            feed_breaker = False
        elif fc in (FailureClass.AUTH, FailureClass.CONFIG):
            base = max(self.auth_cooldown_seconds, self.cooldown_seconds)
            steps = max(0, consecutive - 1)
            cooldown = min(self.max_cooldown_seconds, base * (2.0 ** steps))
            cooldown = max(1.0, cooldown * random.uniform(0.9, 1.1))
        elif fc is FailureClass.RATE_LIMITED:
            base, steps = self.rate_limit_cooldown_seconds, max(0, consecutive - 1)
            cooldown = min(self.max_cooldown_seconds, base * (2.0 ** steps))
            cooldown = max(1.0, cooldown * random.uniform(0.9, 1.1))
        elif consecutive >= self.failure_threshold:
            base, steps = self.cooldown_seconds, consecutive - self.failure_threshold
            cooldown = min(self.max_cooldown_seconds, base * (2.0 ** steps))
            cooldown = max(1.0, cooldown * random.uniform(0.9, 1.1))
        else:
            cooldown = 0.0
        health.record_failure(error, cooldown, now=self._clock())
        # Error-budget heartbeat: feeds real failure ratios.
        try:
            from ..core.error_system import heartbeat
            heartbeat(f"llm/{name}", False)
        except Exception:  # noqa: BLE001 - telemetry never breaks routing
            pass
        if health.breaker is not None:
            if cooldown > 0:
                health.breaker.reset_timeout = cooldown
            if feed_breaker:
                health.breaker.record_failure()

        # Call repair hooks with cooldown
        now = self._clock()
        if isinstance(self.repair_hooks, dict):
            for hook_name, hook in self.repair_hooks.items():
                last_time = self._last_repair_time.get(hook_name, 0.0)
                if now - last_time >= self.repair_cooldown_seconds:
                    try:
                        hook()
                        self._last_repair_time[hook_name] = now
                        self.stats["repairs"] += 1
                    except Exception:  # noqa: BLE001
                        pass
        else:
            for hook in self.repair_hooks:
                try:
                    hook(name, error)
                    self.stats["repairs"] += 1
                except Exception:  # noqa: BLE001
                    pass

    # ── introspection ────────────────────────────────────────────────────────
    def probe(self, timeout: float = 10.0) -> dict[str, bool]:
        """Cheap liveness check across the chain."""
        results: dict[str, bool] = {}
        for provider in self._chain():
            try:
                results[provider.name] = bool(provider.health())
            except Exception:  # noqa: BLE001
                results[provider.name] = False
        return results

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            snapshot = {
                **self.stats,
                "active": self._active,
                "chain": [p.name for p in self._providers],
                "health": {n: h.to_dict() for n, h in self._health.items()},
            }
        if self.daily_budget_usd > 0:
            try:
                now = time.time()
                day_start = now - (now % 86400)
                snapshot["budget"] = self._cost_log().budget_report(
                    self.daily_budget_usd, day_start)
            except Exception:  # noqa: BLE001
                pass
        return snapshot

    def reset_cooldowns(self) -> None:
        with self._lock:
            for health in self._health.values():
                health.cooldown_until = 0.0
                health.consecutive_failures = 0
                if health.breaker is not None:
                    health.breaker.reset()
