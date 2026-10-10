"""Multi-model task router (wave 50) — route each task to the best model.

Given a *task type* (coding, reasoning, vision, embedding, chat, …) and an
*objective* (quality, speed, cost), it scores every registered LLM provider
and picks the one most likely to do the job well. This is the intelligence
layer on top of the existing :class:`~nomorals.llm.router.LLMRouter` provider
chain: the chain guarantees availability; the router picks the *right* model
for the *kind* of task.

Design notes
------------
* **Off by default.** Per the standing constraint, intelligent routing stays
  ``off`` until the user turns it on (e.g. after downloading a local model).
  When off, :meth:`TaskRouter.route` simply delegates to the default provider
  chain — no behaviour change.
* **Local models are first-class.** ``llama_cpp`` (a user-downloaded GGUF)
  is cheap and low-latency, so it wins *speed* and *cost* objectives when
  available. Strong cloud models win *quality*.
* **Capability gating.** A vision task only routes to a vision-capable
  provider; an embedding task only to an embed-capable one.
* **No egress in tests.** Selection is pure; only ``route`` calls a provider,
  and it falls back to the chain on any failure.

Everything is modular and callable by the main AI and sub-agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..core.logging_setup import get_logger

__all__ = [
    "CapabilityGate",
    "ChatBaseline",
    "ComplexityBias",
    "CostObjective",
    "ModelProfile",
    "QualityObjective",
    "ReliabilityPenalty",
    "RequiredCapabilityBonus",
    "ScoreStrategy",
    "SpeedObjective",
    "TaskAffinity",
    "TaskRouter",
    "TASK_TYPES",
    "OBJECTIVES",
    "default_strategies",
    "register",
]

_log = get_logger(__name__)


#: The task kinds the router understands.
TASK_TYPES = (
    "chat", "coding", "reasoning", "planning", "vision",
    "embedding", "summarize", "creative", "extraction",
)

#: The optimisation objectives.
OBJECTIVES = ("quality", "speed", "cost", "balanced")

#: Which providers are "local" (low latency, near-zero cost) vs cloud.
_LOCAL_PROVIDERS = {"llama_cpp", "mock"}
#: Providers that are inherently cheap/free.
_FREE_PROVIDERS = {"llama_cpp", "mock", "ocr"}


@dataclass
class ModelProfile:
    """Static + measured profile of a single LLM provider."""

    name: str
    model: str = ""
    capabilities: frozenset[str] = field(default_factory=frozenset)
    local: bool = False
    free: bool = False
    # measured
    calls: int = 0
    errors: int = 0
    avg_latency_ms: float = 0.0
    #: USD per 1k tokens (broker card, 0.0 = free/local).
    cost_per_1k: float = 0.0
    #: Benchmark evidence score 0..1 (0.5 = unknown), from the broker.
    quality: float = 0.5

    @property
    def error_rate(self) -> float:
        return self.errors / self.calls if self.calls else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "capabilities": sorted(self.capabilities),
            "local": self.local,
            "free": self.free,
            "calls": self.calls,
            "avg_latency_ms": self.avg_latency_ms,
            "error_rate": round(self.error_rate, 3),
            "cost_per_1k": self.cost_per_1k,
            "quality": round(self.quality, 3),
        }


def _required_capability(task_type: str) -> str | None:
    if task_type == "vision":
        return "vision"
    if task_type == "embedding":
        return "embed"
    return None

# ── scoring strategies ───────────────────────────────────────────────────
# The old score() was a pile of hardcoded constants (s += 2.0, local -= 1.0).
# Scoring is now a chain of named strategies, each with a weight and
# parameters — inspectable, testable, and reweightable without editing
# code.  Default weights reproduce the historical ranking exactly.


class ScoreStrategy:
    """One named scoring rule in the chain."""

    name: str = "base"
    weight: float = 1.0

    def applies(self, profile: "ModelProfile", task_type: str,
                objective: str, complexity: str | None) -> bool:
        return True

    def disqualifies(self, profile: "ModelProfile", task_type: str,
                     objective: str, complexity: str | None) -> bool:
        """True → the profile is out entirely (score -inf)."""
        return False

    def score(self, profile: "ModelProfile", task_type: str,
              objective: str, complexity: str | None) -> float:
        return 0.0


class CapabilityGate(ScoreStrategy):
    """Hard gate: a vision task never routes to a text-only provider."""

    name = "capability_gate"

    def disqualifies(self, profile, task_type, objective, complexity) -> bool:
        required = _required_capability(task_type)
        return bool(required and required not in profile.capabilities)


class ChatBaseline(ScoreStrategy):
    name = "chat_baseline"

    def score(self, profile, task_type, objective, complexity) -> float:
        return 1.0 if "chat" in profile.capabilities else 0.0


class RequiredCapabilityBonus(ScoreStrategy):
    name = "required_capability"

    def score(self, profile, task_type, objective, complexity) -> float:
        required = _required_capability(task_type)
        return 2.0 if required and required in profile.capabilities else 0.0


class TaskAffinity(ScoreStrategy):
    """Coding/reasoning/planning prefer code-strong providers.

    The affinity set is a strategy *parameter*, not a hardcoded constant —
    reweight or replace it without touching the scorer.
    """

    name = "task_affinity"

    def __init__(self, code_strong: frozenset[str] | None = None,
                 bonus: float = 1.0) -> None:
        self.code_strong = code_strong if code_strong is not None else frozenset(
            {"hf_serverless", "openai_compat"})
        self.bonus = bonus

    def score(self, profile, task_type, objective, complexity) -> float:
        if task_type in {"coding", "reasoning", "planning"}:
            if profile.name in self.code_strong or "code" in profile.capabilities:
                return self.bonus
        return 0.0


class SpeedObjective(ScoreStrategy):
    """Speed objective: local first, then measured-fast."""

    name = "speed_objective"

    def __init__(self, local_bonus: float = 2.0, fast_bonus: float = 1.0,
                 fast_ms: float = 500.0) -> None:
        self.local_bonus = local_bonus
        self.fast_bonus = fast_bonus
        self.fast_ms = fast_ms

    def applies(self, profile, task_type, objective, complexity) -> bool:
        return objective == "speed"

    def score(self, profile, task_type, objective, complexity) -> float:
        s = 0.0
        if profile.local:
            s += self.local_bonus
        if profile.avg_latency_ms and profile.avg_latency_ms < self.fast_ms:
            s += self.fast_bonus
        return s


class CostObjective(ScoreStrategy):
    """Cost objective: free tiers win; zero-cost cards next."""

    name = "cost_objective"

    def __init__(self, free_bonus: float = 3.0, zero_cost_bonus: float = 1.0) -> None:
        self.free_bonus = free_bonus
        self.zero_cost_bonus = zero_cost_bonus

    def applies(self, profile, task_type, objective, complexity) -> bool:
        return objective == "cost"

    def score(self, profile, task_type, objective, complexity) -> float:
        if profile.free:
            return self.free_bonus
        if profile.cost_per_1k <= 0:
            return self.zero_cost_bonus
        return 0.0


class QualityObjective(ScoreStrategy):
    """Quality objective: measured benchmark evidence first, strong (non-
    local) models next.  Evidence beats the old hardcoded local penalty."""

    name = "quality_objective"

    def __init__(self, evidence_weight: float = 2.0,
                 strong_bonus: float = 1.0) -> None:
        self.evidence_weight = evidence_weight
        self.strong_bonus = strong_bonus

    def applies(self, profile, task_type, objective, complexity) -> bool:
        return objective == "quality"

    def score(self, profile, task_type, objective, complexity) -> float:
        s = (profile.quality - 0.5) * self.evidence_weight
        if not profile.local:
            s += self.strong_bonus
        return s


class ComplexityBias(ScoreStrategy):
    """Easy work belongs on cheap/fast models; hard work on the strongest
    available.  "Strong" is evidence-based (quality score), not a name."""

    name = "complexity_bias"

    def __init__(self, bonus: float = 2.0) -> None:
        self.bonus = bonus

    def score(self, profile, task_type, objective, complexity) -> float:
        cx = (complexity or "").strip().lower()
        if cx == "easy":
            if profile.local or profile.free:
                return self.bonus
        elif cx == "hard":
            if not profile.local and profile.quality >= 0.5:
                return self.bonus
        return 0.0


class ReliabilityPenalty(ScoreStrategy):
    """Flaky providers lose points.  The error-rate ceiling is a parameter."""

    name = "reliability"

    def __init__(self, max_error_rate: float = 0.3,
                 penalty: float = 1.5) -> None:
        self.max_error_rate = max_error_rate
        self.penalty = penalty

    def score(self, profile, task_type, objective, complexity) -> float:
        return -self.penalty if profile.error_rate > self.max_error_rate else 0.0


def default_strategies() -> list[ScoreStrategy]:
    """The standard chain.  Weights default to 1.0 (historical ranking)."""
    return [
        CapabilityGate(),
        ChatBaseline(),
        RequiredCapabilityBonus(),
        TaskAffinity(),
        SpeedObjective(),
        CostObjective(),
        QualityObjective(),
        ComplexityBias(),
        ReliabilityPenalty(),
    ]


class LearnedReliability(ScoreStrategy):
    """NotDiamond-style feedback loop: real outcomes adjust routing.

    ``ledger`` maps model name → rolling list of rewards in [0, 1].
    Recorded via :meth:`TaskRouter.record_outcome`; models with no data
    score 0 (neutral) so the chain's historical ranking is unchanged
    until evidence arrives.
    """

    name = "learned_reliability"

    def __init__(self, ledger: dict[str, list[float]] | None = None,
                 weight: float = 1.0, window: int = 50) -> None:
        self.ledger = ledger if ledger is not None else {}
        self.weight = float(weight)
        self.window = max(1, int(window))

    def score(self, profile, task_type, objective, complexity) -> float:
        rewards = self.ledger.get(profile.name)
        if not rewards:
            return 0.0
        recent = rewards[-self.window:]
        mean = sum(recent) / len(recent)
        # Center on 0.5 (neutral): good models gain, bad models lose.
        return mean - 0.5



class TaskRouter:
    """Pick the best provider for a task type + objective."""

    def __init__(self, context: Any,
                 strategies: list[ScoreStrategy] | None = None,
                 weights: dict[str, float] | None = None) -> None:
        self.context = context
        self.settings = context.settings
        self._profiles: dict[str, ModelProfile] = {}
        #: NotDiamond-style outcome ledger: model -> rolling rewards [0,1].
        #: Fed by record_outcome(); read by the LearnedReliability strategy.
        self._feedback: dict[str, list[float]] = {}
        #: Epsilon-greedy exploration (0 = pure argmax). Set
        #: ``router_exploration`` in settings to try non-best models.
        try:
            self.exploration = float(
                getattr(self.settings, "router_exploration", 0.0) or 0.0)
        except (TypeError, ValueError):
            self.exploration = 0.0
        #: The scoring chain.  Pass custom strategies to extend/replace the
        #: defaults; ``weights`` reweights by strategy name.
        self.strategies = strategies if strategies is not None else default_strategies()
        weights = weights or {}
        for strategy in self.strategies:
            if strategy.name in weights:
                strategy.weight = float(weights[strategy.name])
        # The feedback strategy must read THIS router's ledger, so it is
        # wired here rather than in default_strategies().
        if not any(isinstance(s, LearnedReliability) for s in self.strategies):
            self.strategies.append(LearnedReliability(ledger=self._feedback))

    # ── profiles ─────────────────────────────────────────────────────────────
    def _router(self):
        return self.context.router

    def enabled(self) -> bool:
        return str(getattr(self.settings, "router_intelligent", "off")) == "on"

    def profiles(self, refresh: bool = True) -> list[ModelProfile]:
        if refresh or not self._profiles:
            self._refresh()
        return list(self._profiles.values())

    def _refresh(self) -> None:
        profiles: dict[str, ModelProfile] = {}
        router = self._router()
        try:
            names = list(router.providers())
        except Exception:  # noqa: BLE001
            names = list(getattr(router, "_by_name", {}) or {})
        # Broker evidence (cost + benchmark quality) when a broker is
        # attached; the strategies use it, and fall back to the flags
        # below when it is absent.
        cards: dict[str, Any] = {}
        benchmarks: Any = None
        try:
            broker = getattr(router, "broker", None)
            if broker is not None:
                for card in broker.cards():
                    cards.setdefault(card.provider, card)
                benchmarks = getattr(broker, "benchmarks", None)
        except Exception:  # noqa: BLE001 — evidence is a bonus
            cards, benchmarks = {}, None
        for name in names:
            provider = getattr(router, "get", lambda n: None)(name)
            if provider is None:
                continue
            caps = frozenset(getattr(provider, "capabilities", {"chat"}) or {"chat"})
            stats = getattr(provider, "stats_snapshot", lambda: {})()
            card = cards.get(name)
            cost_per_1k = float(getattr(card, "cost_per_1k", 0.0) or 0.0)
            quality = 0.5
            if benchmarks is not None and card is not None:
                try:
                    from ..llm.capabilities import Capability

                    quality = float(benchmarks.score(card.id, Capability.CHAT))
                except Exception:  # noqa: BLE001
                    quality = 0.5
            profiles[name] = ModelProfile(
                name=name,
                model=str(getattr(provider, "model_id", name) or name),
                capabilities=caps,
                local=name in _LOCAL_PROVIDERS,
                free=name in _FREE_PROVIDERS,
                calls=int(stats.get("calls", 0) or 0),
                errors=int(stats.get("errors", 0) or 0),
                avg_latency_ms=float(stats.get("avg_latency_ms", 0.0) or 0.0),
                cost_per_1k=cost_per_1k,
                quality=max(0.0, min(1.0, quality)),
            )
        self._profiles = profiles

    # ── scoring ──────────────────────────────────────────────────────────────
    def score(self, profile: ModelProfile, task_type: str,
              objective: str, complexity: str | None = None) -> float:
        """Higher = better for this task+objective.

        Runs the strategy chain: gates disqualify first (score -inf),
        then each applicable strategy contributes ``weight * score``.
        ``complexity`` (``"easy"``/``"medium"``/``"hard"``) biases the
        pick: easy work belongs on cheap/fast models, hard work on the
        strongest available one.  ``None`` keeps the pre-complexity
        scoring exactly as it was.
        """
        task_type = (task_type or "chat").strip().lower()
        objective = (objective or "balanced").strip().lower()
        total = 0.0
        for strategy in self.strategies:
            try:
                if not strategy.applies(profile, task_type, objective, complexity):
                    continue
                if strategy.disqualifies(profile, task_type, objective, complexity):
                    return float("-inf")
                total += strategy.weight * strategy.score(
                    profile, task_type, objective, complexity)
            except Exception:  # noqa: BLE001 — one bad strategy never kills scoring
                _log.debug("strategy %s failed; skipping", strategy.name,
                           exc_info=True)
        return total

    def score_breakdown(self, profile: ModelProfile, task_type: str,
                        objective: str,
                        complexity: str | None = None) -> dict[str, float]:
        """Per-strategy contributions (introspection/UI).  Never raises."""
        task_type = (task_type or "chat").strip().lower()
        objective = (objective or "balanced").strip().lower()
        out: dict[str, float] = {}
        for strategy in self.strategies:
            try:
                if not strategy.applies(profile, task_type, objective, complexity):
                    continue
                if strategy.disqualifies(profile, task_type, objective, complexity):
                    return {"disqualified_by": strategy.name}  # type: ignore[dict-item]
                out[strategy.name] = round(
                    strategy.weight * strategy.score(
                        profile, task_type, objective, complexity), 3)
            except Exception:  # noqa: BLE001
                out[strategy.name] = 0.0
        return out

    def select(self, task_type: str = "chat",
               objective: str = "balanced",
               complexity: str | None = None) -> ModelProfile | None:
        """Return the best profile, or None when routing is off/unavailable.

        With ``router_exploration`` > 0, epsilon-greedy: with probability
        epsilon a random non-best candidate is picked instead of the
        argmax, so the feedback ledger keeps learning about alternatives.
        """
        task_type = (task_type or "chat").strip().lower()
        objective = (objective or "balanced").strip().lower()
        if not self.enabled():
            return None
        ranked: list[tuple[ModelProfile, float]] = []
        for p in self.profiles():
            ranked.append((p, self.score(p, task_type, objective,
                                         complexity=complexity)))
        ranked.sort(key=lambda t: t[1], reverse=True)
        if not ranked:
            return None
        if self.exploration > 0 and len(ranked) > 1:
            import random as _random
            if _random.random() < self.exploration:
                choice = _random.choice(ranked[1:])[0]
                _log.info("router exploring: %s (epsilon=%.2f)",
                          choice.name, self.exploration)
                return choice
        return ranked[0][0]

    # ── feedback learning ────────────────────────────────────────────
    def record_outcome(self, model: str, task_kind: str = "chat", *,
                       ok: bool, latency_ms: float | None = None,
                       note: str = "") -> None:
        """Feed a real outcome back into routing (NotDiamond-style).

        Reward: 1.0 for success, 0.0 for failure; slow successes are
        discounted so latency creeps into the learned score. Never raises.
        """
        try:
            reward = 1.0 if ok else 0.0
            if ok and latency_ms:
                # Halve the reward past 30s — slow is a soft failure.
                try:
                    if float(latency_ms) > 30_000:
                        reward = 0.5
                except (TypeError, ValueError):
                    pass
            ledger = self._feedback.setdefault(str(model), [])
            ledger.append(reward)
            del ledger[:-50]  # rolling window
        except Exception:  # noqa: BLE001 — learning never breaks routing
            _log.debug("router record_outcome failed", exc_info=True)

    def feedback_score(self, model: str) -> float:
        """Mean learned reward for a model, 0.5 when no data. Never raises."""
        try:
            rewards = self._feedback.get(str(model))
            if not rewards:
                return 0.5
            return sum(rewards) / len(rewards)
        except Exception:  # noqa: BLE001
            return 0.5

    def explain(self, profile: ModelProfile | None, task_type: str = "chat",
                objective: str = "balanced",
                complexity: str | None = None) -> str:
        """One plain-language sentence: why this model was picked."""
        from .render import truncate

        if profile is None:
            return ("intelligent routing is off — using the standard "
                    "provider chain")
        try:
            breakdown = self.score_breakdown(profile, task_type, objective,
                                             complexity=complexity)
            top = sorted(
                ((k, v) for k, v in breakdown.items() if v > 0),
                key=lambda kv: kv[1], reverse=True)[:2]
            drivers = ", ".join(
                f"{k.replace('_', ' ')} (+{v})" for k, v in top)
            fb = self.feedback_score(profile.name)
            learned = (f"; learned reliability {fb:.2f} from "
                       f"{len(self._feedback.get(profile.name, []))} outcomes"
                       if profile.name in self._feedback else "")
            bits = (f"picked **{profile.name}** for {task_type}/{objective}"
                    + (f" (drivers: {drivers})" if drivers else "")
            return truncate(bits + learned, 400)
        except Exception:  # noqa: BLE001
            return f"picked {getattr(profile, 'name', '?')}"

    def decision(self, task_type: str = "chat",
                 objective: str = "balanced",
                 complexity: str | None = None) -> dict[str, Any]:
        task_type = (task_type or "chat").strip().lower()
        objective = (objective or "balanced").strip().lower()
        cx = (complexity or "").strip().lower() or None
        scored = []
        if self.enabled():
            for p in self.profiles():
                scored.append({"provider": p.name, "score": round(self.score(p, task_type, objective, complexity=cx), 3)})
            scored.sort(key=lambda d: d["score"], reverse=True)
        choice = self.select(task_type, objective, complexity=cx)
        return {
            "enabled": self.enabled(),
            "task_type": task_type,
            "objective": objective,
            "complexity": cx,
            "choice": choice.name if choice else None,
            "reason": (
                "intelligent routing off (default); using the standard chain"
                if not self.enabled()
                else "highest-scoring provider for this task+objective"
            ),
            "scored": scored,
        }

    def route_by_complexity(
        self, prompt: str, *, task_type: str | None = None
    ) -> tuple[str, str]:
        """Classify ``prompt`` complexity, then pick a provider.

        Returns ``(provider_name, complexity)``.  When intelligent routing
        is off, the provider is the chain's active provider (today's
        behavior) and the complexity is still reported.
        """
        from .complexity import classify_complexity
        kind = (task_type or "").strip().lower()
        if not kind:
            try:
                from .task_type import classify_task
                kind = classify_task(
                    self.context, prompt, use_model=False).kind or "chat"
            except Exception:  # noqa: BLE001 — keyword layer is best-effort
                kind = "chat"
        complexity, _confidence = classify_complexity(prompt, task_type=kind)
        choice = self.select(task_type=kind or "chat", objective="balanced",
                             complexity=complexity)
        if choice is not None:
            return choice.name, complexity
        try:
            return self._router().active or "", complexity
        except Exception:  # noqa: BLE001
            return "", complexity

    # ── Jev: the cheap decision classifier (build-map #18 extension) ──
    def jev_decide(self, prompt: str) -> dict[str, Any]:
        """Deliberate Jev consult: classify/route/moderate before any brain spend.

        Hercules "Jev" pattern — the cheap decision classifier is a *named
        primitive*, explicitly separate from the main brain (zero LLM calls
        inside).  Callers reach for it deliberately instead of asking the
        big model "what kind of request is this?".  Never raises.
        """
        try:
            from .jev import jev
            return jev.decide(prompt).to_dict()
        except Exception:  # noqa: BLE001 — Jev must never break routing
            _log.debug("jev_decide failed; defaulting", exc_info=True)
            return {
                "classification": {"kind": "chat", "complexity": "medium"},
                "route": {"task_type": "chat", "handler": "chat"},
                "moderation": {"level": "safe", "reasons": []},
                "elapsed_ms": 0.0,
            }

    # ── routing ──────────────────────────────────────────────────────────────
    @staticmethod
    def _to_messages(messages: list[Any]) -> list[Any]:
        """Accept dicts or Message objects; return Message objects."""
        from ..llm.base import Message
        out = []
        for m in messages:
            if isinstance(m, Message):
                out.append(m)
            elif isinstance(m, dict):
                role = str(m.get("role", "user"))
                content = str(m.get("content", ""))
                if role == "system":
                    out.append(Message.system(content))
                elif role == "assistant":
                    out.append(Message.assistant(content))
                else:
                    out.append(Message.user(content))
            else:
                out.append(Message.user(str(m)))
        return out

    def route(self, messages: list[Any], *,
              task_type: str = "chat", objective: str = "balanced",
              complexity: str | None = None,
              system: str = "", temperature: float = 0.7,
              max_tokens: int = 2048, **kwargs: Any):
        """Chat via the chosen provider, falling back to the default chain.

        ``complexity`` (``"easy"``/``"medium"``/``"hard"``, build-map #18)
        biases the provider pick and is recorded as the call's tier in the
        cost log.  Returns an :class:`~nomorals.llm.base.LLMResponse`.
        """
        from ..llm.base import Message, SamplingParams
        from ..llm.router import log_llm_call
        msgs = self._to_messages(messages)
        if system:
            msgs = [Message.system(system)] + msgs
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        # A caller-supplied tier wins over the complexity argument.
        tier = kwargs.pop("tier", None) or (complexity or "")
        router = self._router()
        choice = self.select(task_type, objective, complexity=complexity)
        if choice is not None:
            provider = getattr(router, "get", lambda n: None)(choice.name)
            if provider is not None:
                try:
                    resp = provider.chat(msgs, params, **kwargs)
                    if resp.ok:
                        # Direct-provider path bypasses LLMRouter._dispatch,
                        # so log the metered cost here.
                        try:
                            log_llm_call(
                                choice.name,
                                getattr(provider, "model_id", "") or "",
                                resp, tier=tier, operation="chat")
                        except Exception:  # noqa: BLE001
                            _log.debug("cost log failed", exc_info=True)
                        return resp
                    choice_error = resp.error or "unknown error"
                    _log.warning(
                        "task router: chosen provider %s failed (%s); "
                        "falling back to the default chain",
                        choice.name, choice_error)
                except Exception as exc:  # noqa: BLE001 — fall through to the chain
                    choice_error = f"{type(exc).__name__}: {exc}"
                    _log.warning(
                        "task router: chosen provider %s raised (%s); "
                        "falling back to the default chain",
                        choice.name, exc)
            else:
                choice_error = "provider not registered"
                _log.warning(
                    "task router: chosen provider %s not registered; "
                    "falling back to the default chain", choice.name)
        else:
            choice_error = ""
        chain_resp = router.chat(msgs, params, tier=tier, **kwargs)
        # The chosen provider failed and the chain answered: attribute the
        # full degradation chain on the response instead of swallowing it.
        if choice_error and chain_resp is not None:
            chain_resp.degraded = True
            chain_resp.failed_providers = (
                [choice.name] + list(chain_resp.failed_providers))
            prefix = f"{choice.name} failed ({choice_error})"
            chain_resp.fallback_note = (
                prefix + ("; " + chain_resp.fallback_note
                          if chain_resp.fallback_note else ""))
        return chain_resp

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled(),
            "note": "task-based model routing (speed/quality/cost)",
            "profiles": [p.to_dict() for p in self.profiles(refresh=True)],
        }


# ── tool registration ──────────────────────────────────────────────────────────

def register(registry: Any) -> None:
    context = registry.context

    @registry.register(
        "model_route",
        description=("Route a task to the best model for its type and objective "
                     "(off by default). action=decision (show the pick + scores), "
                     "status (profiles), route (chat via the chosen model), "
                     "by_complexity (classify a prompt, name the provider)."),
        capability="model.call",
        parameters={
            "action": "str — decision | status | route | by_complexity",
            "task_type": "str — chat|coding|reasoning|vision|… (decision/route)",
            "objective": "str — quality|speed|cost|balanced (decision/route)",
            "complexity": "str — easy|medium|hard bias (decision/route, optional)",
            "messages_json": "json str [{role, content}] (route) or raw prompt (by_complexity)",
        },
    )
    def model_route(
        action: str = "decision", *, task_type: str = "chat",
        objective: str = "balanced", complexity: str = "",
        messages_json: str = "",
        temperature: str = "0.7", max_tokens: str = "2048",
    ) -> dict[str, Any]:
        router = TaskRouter(context)
        action = (action or "decision").strip().lower()
        cx = (complexity or "").strip().lower() or None
        if action == "status":
            return router.status()
        if action == "decision":
            return router.decision(task_type, objective, complexity=cx)
        if action == "by_complexity":
            name, level = router.route_by_complexity(
                messages_json, task_type=task_type or None)
            return {"ok": True, "provider": name, "complexity": level}
        if action == "route":
            import json
            try:
                messages = json.loads(messages_json) if messages_json else []
            except (ValueError, TypeError):
                return {"ok": False, "error": "messages_json must be a JSON array"}
            if not messages:
                return {"ok": False, "error": "messages_json is empty"}
            resp = router.route(
                messages, task_type=task_type, objective=objective,
                complexity=cx,
                temperature=float(temperature) if temperature else 0.7,
                max_tokens=int(max_tokens) if max_tokens else 2048,
            )
            return {
                "ok": bool(resp.ok),
                "model": resp.model,
                "text": (resp.text or "")[:2000],
                "routed": router.select(task_type, objective, complexity=cx),
            }
        return {"ok": False, "error": f"unknown action: {action}"}
