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

__all__ = ["ModelProfile", "TaskRouter", "TASK_TYPES", "OBJECTIVES", "register"]

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
        }


def _required_capability(task_type: str) -> str | None:
    if task_type == "vision":
        return "vision"
    if task_type == "embedding":
        return "embed"
    return None


class TaskRouter:
    """Pick the best provider for a task type + objective."""

    def __init__(self, context: Any) -> None:
        self.context = context
        self.settings = context.settings
        self._profiles: dict[str, ModelProfile] = {}

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
        for name in names:
            provider = getattr(router, "get", lambda n: None)(name)
            if provider is None:
                continue
            caps = frozenset(getattr(provider, "capabilities", {"chat"}) or {"chat"})
            stats = getattr(provider, "stats_snapshot", lambda: {})()
            profiles[name] = ModelProfile(
                name=name,
                model=str(getattr(provider, "model_id", name) or name),
                capabilities=caps,
                local=name in _LOCAL_PROVIDERS,
                free=name in _FREE_PROVIDERS,
                calls=int(stats.get("calls", 0) or 0),
                errors=int(stats.get("errors", 0) or 0),
                avg_latency_ms=float(stats.get("avg_latency_ms", 0.0) or 0.0),
            )
        self._profiles = profiles

    # ── scoring ──────────────────────────────────────────────────────────────
    def score(self, profile: ModelProfile, task_type: str,
              objective: str, complexity: str | None = None) -> float:
        """Higher = better for this task+objective.

        ``complexity`` (``"easy"``/``"medium"``/``"hard"``, build-map #18)
        biases the pick: easy work belongs on cheap/fast models, hard work
        on the strongest available one.  ``None`` keeps the pre-complexity
        scoring exactly as it was.
        """
        s = 0.0
        required = _required_capability(task_type)
        if required and required not in profile.capabilities:
            return -1.0  # hard disqualify
        if "chat" in profile.capabilities:
            s += 1.0
        if required and required in profile.capabilities:
            s += 2.0  # it can actually do this kind of task
        # task affinity
        if task_type in {"coding", "reasoning", "planning"}:
            if profile.name in {"hf_serverless", "openai_compat"}:
                s += 1.0
        # objective weighting
        if objective == "speed":
            if profile.local:
                s += 2.0
            if profile.avg_latency_ms and profile.avg_latency_ms < 500:
                s += 1.0
        elif objective == "cost":
            if profile.free:
                s += 3.0
        elif objective == "quality":
            if profile.local:
                s -= 1.0  # cloud/8B+ usually beat a tiny local for quality
        # complexity bias (build-map #18): orthogonal to the objective.
        # "Strong" here means non-local — consistent with the quality
        # objective above, which already penalizes local models.
        cx = (complexity or "").strip().lower()
        if cx == "easy":
            if profile.local or profile.free:
                s += 2.0
        elif cx == "hard":
            if not profile.local:
                s += 2.0
        # reliability
        if profile.error_rate > 0.3:
            s -= 1.5
        return s

    def select(self, task_type: str = "chat",
               objective: str = "balanced",
               complexity: str | None = None) -> ModelProfile | None:
        """Return the best profile, or None when routing is off/unavailable."""
        task_type = (task_type or "chat").strip().lower()
        objective = (objective or "balanced").strip().lower()
        if not self.enabled():
            return None
        best: ModelProfile | None = None
        best_score = float("-inf")
        for p in self.profiles():
            sc = self.score(p, task_type, objective, complexity=complexity)
            if sc > best_score:
                best_score = sc
                best = p
        return best

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
