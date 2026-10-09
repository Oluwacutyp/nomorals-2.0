"""The brain: one front door for "give me a completion".

Every subsystem (chat, the LLM-first music composer, briefings, …) talks to
the model plane through :class:`Brain` instead of reaching past it into
provider internals or reimplementing "is there a model?" checks.  The brain
owns the wiring:

* an :class:`~nomorals.llm.router.LLMRouter` with failover and a per-provider
  circuit breaker (exponential-backoff cooldowns, half-open probing);
* a :class:`~nomorals.llm.broker.ModelBroker` for task-aware selection
  (vision work → VL models, code work → code-tuned models) with the
  operator's own models preferred whenever they can serve;
* honest failure reporting — a dead brain comes back as an
  :class:`~nomorals.llm.base.LLMResponse` with ``error`` set, plus a
  plain-language diagnosis of what was tried.

The brain never raises: the worst case is an ``LLMResponse`` whose
``error`` names every provider that was attempted and why each failed.

    from nomorals.llm.brain import get_brain
    brain = get_brain()
    resp = brain.complete("write a haiku about Lagos traffic", task_kind="creative")
    if not resp.ok:
        print(brain.explain_failure(resp))
"""

from __future__ import annotations

import threading
from typing import Any, Sequence

from ..core.logging_setup import get_logger
from .base import LLMResponse, Message, SamplingParams

__all__ = [
    "Brain",
    "brain_for",
    "explain_failure",
    "get_brain",
    "reset_brain",
]

_log = get_logger(__name__)

#: Router names that are not a real brain — the scripted test double and
#: friends.  Centralised here so subsystems stop reimplementing the check.
_NON_BRAIN_NAMES = frozenset({"mock", "offline", "test"})

#: error-text → likely fix, for the honest-failure message.  Substring
#: match, first hit wins.  These are hints, not diagnoses.
_FIX_HINTS: tuple[tuple[str, str], ...] = (
    ("401", "the provider rejected the API key — check it is set and valid"),
    ("403", "the provider refused the request — check the key's permissions"),
    ("429", "rate/quota limit — the router backs off automatically; check plan quota"),
    ("connection refused", "local model server not reachable — is llama.cpp / ollama running?"),
    ("timed out", "provider timed out — it may be overloaded; the next call fails over"),
    ("404", "the model id may be retired — update the configured model name"),
    ("not hosted", "the model is not served on that endpoint — pick a hosted id"),
    ("cooling down", "temporary — the router retries automatically after the cooldown"),
)


def _fix_hint(text: str) -> str:
    low = (text or "").lower()
    for needle, hint in _FIX_HINTS:
        if needle in low:
            return hint
    return ""


def explain_failure(response: Any) -> str:
    """Plain-language account of a failed brain call.  Never raises.

    Names every provider that was attempted and why each failed, so the
    operator sees "groq → 429, hf → 503" instead of a bare empty string.
    """
    try:
        err = str(getattr(response, "error", "") or "").strip()
        note = str(getattr(response, "fallback_note", "") or "").strip()
        tried = [str(n) for n in (getattr(response, "failed_providers", None) or [])]
        lines = ["Brain unavailable — no provider served this call."]
        if tried:
            lines.append("Attempted: " + ", ".join(tried) + ".")
        if note:
            lines.append("Detail: " + note)
        elif err:
            lines.append("Error: " + err)
        hint = _fix_hint(err + " " + note)
        if hint:
            lines.append("Likely fix: " + hint + ".")
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 — the explainer itself never fails
        return "Brain unavailable — no provider served this call."


def _build_default_router() -> Any:
    """Env-based chain (free/local/owner first) + capability broker."""
    from .broker import ModelBroker
    from .defaults import build_chain, sync_broker_cards

    router = build_chain()
    try:
        broker = ModelBroker()
        sync_broker_cards(broker, router)
        router.set_broker(broker)
    except Exception:  # noqa: BLE001 — broker is best-effort
        _log.debug("brain: broker attach failed; name-based routing only",
                   exc_info=True)
    return router


class Brain:
    """One object for every "ask the model" call in the system.

    ``router`` may be an existing, fully-wired :class:`LLMRouter` (e.g. the
    agent context's own — preferred, because it carries the operator's
    settings-driven chain, broker and learning hook).  When omitted the
    brain builds the env-based default chain lazily on first use.
    """

    def __init__(self, router: Any = None) -> None:
        self._external_router = router
        self._router: Any = None
        self._lock = threading.Lock()

    # ── wiring ───────────────────────────────────────────────────────────
    def _get_router(self) -> Any:
        router = self._external_router
        if router is not None:
            return router
        if self._router is None:
            with self._lock:
                if self._router is None:
                    self._router = _build_default_router()
        return self._router

    @property
    def router(self) -> Any:
        """The underlying router (builds the default chain if needed)."""
        return self._get_router()

    # ── generation ───────────────────────────────────────────────────────
    @staticmethod
    def _accepts(fn: Any, name: str) -> bool:
        """True when ``fn`` accepts keyword ``name`` (or any **kw)."""
        try:
            import inspect as _inspect
            params = _inspect.signature(fn).parameters.values()
        except (TypeError, ValueError):
            return True  # uninspectable — assume modern, the call will tell
        return any(p.kind == p.VAR_KEYWORD for p in params) \
            or name in (p.name for p in params)

    def _call(self, op: str, fn: Any, *args: Any,
              task_kind: str = "", **kw: Any) -> LLMResponse:
        """Invoke a router method, degrading gracefully for legacy routers.

        The router may be any duck-typed object with ``chat``/``complete``/
        ``describe_image``.  Modern routers accept ``task_kind`` (task-aware
        selection), ``constraints`` and ``tier``; legacy ones do not — kwargs
        the signature cannot take are dropped up front instead of failing
        the call.  Never raises: the worst case is an error response.
        """
        call_kw = dict(kw)
        if task_kind and not self._accepts(fn, "task_kind"):
            _log.debug("brain: %s() has no task_kind support; calling plain",
                       op)
        else:
            call_kw["task_kind"] = task_kind
        for name in ("constraints", "tier"):
            if name in call_kw and not self._accepts(fn, name):
                del call_kw[name]
        try:
            return fn(*args, **call_kw)
        except Exception as exc:  # noqa: BLE001 — honest failure, not a raise
            _log.debug("brain.%s failed: %s", op, exc, exc_info=True)
            return LLMResponse(
                text="", error=f"brain.{op} failed before dispatch: {exc}")

    def chat(
        self,
        messages: Sequence[Message],
        *,
        task_kind: str = "",
        params: SamplingParams | None = None,
        constraints: Any = None,
        tier: str | None = None,
    ) -> LLMResponse:
        """Chat completion.  Never raises — failure is ``resp.error``."""
        return self._call(
            "chat", self._get_router().chat, messages, params,
            task_kind=task_kind, tier=tier, constraints=constraints)

    def complete(
        self,
        prompt: str,
        *,
        task_kind: str = "",
        params: SamplingParams | None = None,
        constraints: Any = None,
        tier: str | None = None,
    ) -> LLMResponse:
        """Single-prompt completion.  Never raises."""
        return self._call(
            "complete", self._get_router().complete, prompt, params,
            task_kind=task_kind, tier=tier, constraints=constraints)

    def describe_image(
        self,
        image: bytes,
        prompt: str = "",
        *,
        params: SamplingParams | None = None,
        constraints: Any = None,
        tier: str | None = None,
    ) -> LLMResponse:
        """Vision.  Never raises; routes to a VL-capable model."""
        return self._call(
            "describe_image", self._get_router().describe_image,
            image, prompt, params,
            task_kind="vision", tier=tier, constraints=constraints)

    # ── introspection ────────────────────────────────────────────────────
    def available(self) -> bool:
        """True when a real (non-mock) provider exists and at least one is
        not cooling down.  Never raises."""
        try:
            router = self._get_router()
            names = [n for n in router.providers()
                     if n not in _NON_BRAIN_NAMES]
            if not names:
                return False
            return any(not router.is_cooling_down(n) for n in names)
        except Exception:  # noqa: BLE001
            _log.debug("brain.available failed", exc_info=True)
            return False

    def active_model(self) -> str:
        """Name of the provider new calls start on ('' when none)."""
        try:
            return str(self._get_router().active or "")
        except Exception:  # noqa: BLE001
            return ""

    def status(self) -> dict[str, Any]:
        """Machine-readable brain health.  Never raises."""
        try:
            router = self._get_router()
            snap = router.stats_snapshot()
            health = snap.get("health", {})
            providers = []
            for name in snap.get("chain", []):
                h = health.get(name, {})
                providers.append({
                    "name": name,
                    "model": getattr(router.get(name), "model_id", ""),
                    "cooling_down": bool(h.get("cooling_down", False)),
                    "breaker": str(h.get("breaker_state", "closed")),
                    "error_rate": h.get("error_rate", 0.0),
                    "last_error": str(h.get("last_error", ""))[:160],
                    "owner": self._is_owner_card(router, name),
                })
            return {
                "available": self.available(),
                "active": snap.get("active", ""),
                "providers": providers,
                "broker_picks": self._broker_picks(router),
            }
        except Exception as exc:  # noqa: BLE001
            _log.debug("brain.status failed", exc_info=True)
            return {"available": False, "active": "",
                    "providers": [], "broker_picks": {},
                    "error": str(exc)[:200]}

    def diagnose(self) -> str:
        """Human-readable brain report.  Never raises."""
        try:
            st = self.status()
            state = "UP" if st["available"] else "DOWN"
            lines = [f"brain: {state} (active={st['active'] or 'none'})"]
            for p in st["providers"]:
                if p["cooling_down"]:
                    flag = "COOLING"
                elif p["breaker"] not in ("closed",):
                    flag = p["breaker"].upper()
                else:
                    flag = "ok"
                owner = " [owner]" if p["owner"] else ""
                err = f" — {p['last_error']}" if p["last_error"] else ""
                lines.append(
                    f"  {p['name']:<16} {flag:<9} {p['model']}{owner}{err}")
            picks = st.get("broker_picks") or {}
            if picks:
                lines.append("broker picks: " + ", ".join(
                    f"{op}→{pick}" for op, pick in picks.items()))
            if not st["available"]:
                lines.append(
                    "No usable brain: every provider is cooling down, "
                    "missing, or mock. Check API keys and local servers, "
                    "then the router retries automatically.")
            return "\n".join(lines)
        except Exception:  # noqa: BLE001
            return "brain: status unavailable (diagnose failed)"

    # ── internals ────────────────────────────────────────────────────────
    @staticmethod
    def _is_owner_card(router: Any, provider_name: str) -> bool:
        try:
            broker = router.broker
            if broker is None:
                return False
            for card in broker.cards():
                if card.provider == provider_name and card.owner:
                    return True
        except Exception:  # noqa: BLE001
            pass
        return False

    @staticmethod
    def _broker_picks(router: Any) -> dict[str, str]:
        picks: dict[str, str] = {}
        try:
            broker = router.broker
            if broker is None:
                return picks
            from .capabilities import Capability
            for op, cap in (("chat", Capability.CHAT),
                            ("vision", Capability.VISION),
                            ("code", Capability.CODE)):
                try:
                    card = broker.select(cap, op)
                    if card is not None:
                        picks[op] = card.id
                except Exception:  # noqa: BLE001 — one pick never kills status
                    pass
        except Exception:  # noqa: BLE001
            pass
        return picks


_brain: Brain | None = None
_brain_lock = threading.Lock()


def get_brain() -> Brain:
    """Process-wide shared brain (env-based default chain).  Thread-safe.

    Subsystems that already have a wired agent context should prefer
    ``Brain(router=context.router)`` — that router carries the operator's
    settings-driven chain.  ``get_brain()`` is for code with no context.
    """
    global _brain
    if _brain is None:
        with _brain_lock:
            if _brain is None:
                _brain = Brain()
    return _brain


def reset_brain() -> None:
    """Drop the shared brain (tests).  The next :func:`get_brain` rebuilds."""
    global _brain
    with _brain_lock:
        _brain = None


def brain_for(context: Any) -> Brain:
    """The brain for an agent context — one Brain per context, cached.

    Wraps ``context.router`` (the operator's settings-driven chain) so
    every call site gets the task-kind threading, timeouts, failure
    taxonomy, and context fitting without reimplementing the wiring.
    When the context has no router, falls back to the shared env-based
    brain.  Never raises.
    """
    try:
        router = getattr(context, "router", None)
        if router is None:
            return get_brain()
        cached = getattr(context, "_brain_for_ctx", None)
        if isinstance(cached, Brain) and cached._external_router is router:
            return cached
        brain = Brain(router=router)
        try:
            context._brain_for_ctx = brain  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — caching is a bonus
            pass
        return brain
    except Exception:  # noqa: BLE001
        _log.debug("brain_for failed; using shared brain", exc_info=True)
        return get_brain()
