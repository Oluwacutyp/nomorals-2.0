"""Provider routing with fallback and hot-swap.

The router is what makes "which model am I talking to?" a runtime decision instead
of a config constant. It maintains an ordered chain, tracks per-provider health and
latency, fails over automatically, and can swap the active model while calls are in
flight — new calls use the new model, running ones finish on the old one.

Failover is bounded: after the chain is exhausted the caller gets a typed error
rather than an exception from whichever provider happened to be last.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..core.errors import ModelError, ProviderUnavailable, classify
from ..core.events import EventBus
from ..core.logging_setup import get_logger
from .base import LLMProvider, LLMResponse, Message, SamplingParams

__all__ = ["LLMRouter", "ProviderHealth"]

_log = get_logger(__name__)


@dataclass
class ProviderHealth:
    """Rolling health for one provider."""

    name: str
    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    last_success: float = 0.0
    last_failure: float = 0.0
    last_error: str = ""
    total_latency_ms: float = 0.0
    cooldown_until: float = 0.0

    @property
    def error_rate(self) -> float:
        return self.failures / self.calls if self.calls else 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.calls if self.calls else 0.0

    def available(self, now: float | None = None) -> bool:
        return (now or time.monotonic()) >= self.cooldown_until

    def record_success(self) -> None:
        self.calls += 1
        self.consecutive_failures = 0
        self.last_success = time.time()

    def record_failure(self, error: str, cooldown: float) -> None:
        self.calls += 1
        self.failures += 1
        self.consecutive_failures += 1
        self.last_failure = time.time()
        self.last_error = error
        if cooldown > 0:
            self.cooldown_until = time.monotonic() + cooldown

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "calls": self.calls,
            "failures": self.failures,
            "error_rate": round(self.error_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "consecutive_failures": self.consecutive_failures,
            "last_error": self.last_error,
            "cooling_down": not self.available(),
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
        failure_threshold: int = 3,
        bus: EventBus | None = None,
        clock: Callable[[], float] = time.monotonic,
        repair_hooks: dict[str, Callable] | list[Callable] | None = None,
    ) -> None:
        self._providers: list[LLMProvider] = []
        self._by_name: dict[str, LLMProvider] = {}
        self._health: dict[str, ProviderHealth] = {}
        self._active: str = ""
        self._lock = threading.RLock()
        self.cooldown_seconds = cooldown_seconds
        self.failure_threshold = failure_threshold
        self.bus = bus
        self._clock = clock
        self.repair_hooks = repair_hooks if repair_hooks is not None else {}
        self.repair_cooldown_seconds = cooldown_seconds
        self._last_repair_time: dict[str, float] = {}
        self.stats = {"calls": 0, "failovers": 0, "failures": 0, "repairs": 0}

    # ── registration ─────────────────────────────────────────────────────────
    def add(self, provider: LLMProvider, *, primary: bool = False, name: str = "") -> "LLMRouter":
        key = name or provider.name
        with self._lock:
            if key in self._by_name:
                raise ValueError(f"provider {key!r} already registered")
            # Wrap in a lightweight shim so the same class can appear twice under
            # different names (e.g. two different HF models).
            provider.name = key
            self._by_name[key] = provider
            self._health[key] = ProviderHealth(name=key)
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

    # ── calling ──────────────────────────────────────────────────────────────
    def _chain(self) -> list[LLMProvider]:
        with self._lock:
            if not self._providers:
                return []
            active = self._by_name.get(self._active)
            others = [p for p in self._providers if p.name != self._active]
            return ([active] if active else []) + others

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        return self._dispatch("chat", lambda p: p.chat(messages, params, **kw))

    def complete(self, prompt: str, params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        return self._dispatch("complete", lambda p: p.complete(prompt, params, **kw))

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
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        return self._dispatch(
            "vision",
            lambda p: p.describe_image(image, prompt, params, **kw),
            require="vision",
        )

    def _dispatch(
        self,
        operation: str,
        call: Callable[[LLMProvider], LLMResponse],
        *,
        require: str | None = None,
    ) -> LLMResponse:
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

        self.stats["calls"] += 1
        attempted = 0
        failed: list[tuple[str, str]] = []  # (provider name, error) in attempt order
        last = LLMResponse(text="", error="no attempt made")
        for index, provider in enumerate(chain):
            health = self._health[provider.name]
            if not health.available(self._clock()):
                continue
            attempted += 1
            try:
                response = call(provider)
            except Exception as exc:  # noqa: BLE001 - never let a provider crash the router
                error = classify(exc)
                note = f"{error.code}: {error.message}"
                self._note_failure(provider.name, note)
                failed.append((provider.name, note))
                last = LLMResponse(
                    text="",
                    model=provider.model_id,
                    provider=provider.name,
                    error=note,
                )
                continue
            if response.ok:
                self._note_success(provider.name)
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
            last = response

        self.stats["failures"] += 1
        if attempted == 0:
            last.error = "all providers are cooling down"
        # Even a total failure reports the whole attempt chain.
        last.failed_providers = [name for name, _ in failed]
        if failed:
            chain_note = "; ".join(f"{name} failed ({err})" for name, err in failed)
            last.fallback_note = (f"{chain_note}; no provider served this call"
                                  if not last.fallback_note else last.fallback_note)
        return last

    def _note_success(self, name: str) -> None:
        health = self._health.get(name)
        if health is not None:
            health.record_success()

    def _note_failure(self, name: str, error: str) -> None:
        health = self._health.get(name)
        if health is None:
            return
        cooldown = self.cooldown_seconds if health.consecutive_failures + 1 >= self.failure_threshold else 0.0
        health.record_failure(error, cooldown)
        
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
            return {
                **self.stats,
                "active": self._active,
                "chain": [p.name for p in self._providers],
                "health": {n: h.to_dict() for n, h in self._health.items()},
            }

    def reset_cooldowns(self) -> None:
        with self._lock:
            for health in self._health.values():
                health.cooldown_until = 0.0
                health.consecutive_failures = 0
