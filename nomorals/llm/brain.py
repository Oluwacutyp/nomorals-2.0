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
import time
from typing import Any, Sequence

from ..core.logging_setup import get_logger
from .base import LLMResponse, Message, SamplingParams
from .context_fit import DEFAULT_CONTEXT_TOKENS, fit_messages, fit_prompt
from .failures import FailureClass, classify_failure
from .prompts import render_prompt, system_prompt_for

__all__ = [
    "Brain",
    "brain_for",
    "estimate_complexity",
    "explain_failure",
    "get_brain",
    "reset_brain",
]

_log = get_logger(__name__)

#: Router names that are not a real brain — the scripted test double and
#: friends.  Centralised here so subsystems stop reimplementing the check.
_NON_BRAIN_NAMES = frozenset({"mock", "offline", "test"})

#: error-text → likely fix, for the honest-failure message.  Substring
#: match, first hit wins.  These are hints, not diagnoses.  The typed
#: taxonomy (:mod:`nomorals.llm.failures`) is consulted first; this table
#: is the fallback for classes it does not cover.
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


def _fix_hint(text: str) -> str:    # Typed first: the failure taxonomy knows the right recovery for
    # each class (shrink context, back off, fix the key…).
    try:
        info = classify_failure(text)
        if info.failure_class is not FailureClass.UNKNOWN:
            return info.hint
    except Exception:  # noqa: BLE001 — hinting never breaks the caller
        pass
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


def _with_quality(constraints: Any, quality: float | None) -> Any:
    """Merge the complexity-tier ``quality`` knob into constraints.

    Accepts a mapping, a BrokerConstraints, or None — returns the same
    shape it was given, with ``min_quality`` set.
    """
    if quality is None:
        return constraints
    q = max(0.0, min(1.0, float(quality)))
    if constraints is None:
        return {"min_quality": q}
    if isinstance(constraints, dict):
        merged = dict(constraints)
        merged["min_quality"] = q
        return merged
    try:
        from .broker import BrokerConstraints

        if isinstance(constraints, BrokerConstraints):
            import dataclasses

            return dataclasses.replace(constraints, min_quality=q)
    except Exception:  # noqa: BLE001
        pass
    return constraints


def estimate_complexity(text: str) -> float:
    """Heuristic prompt complexity 0..1 (the query-classifier's cheap half).

    A small rule-based classifier decides the quality tier: short lookups
    → cheap models, long/code/multi-part prompts → stronger ones.  This is
    the *suggestion*, not the decision — callers pass it as
    ``quality=`` or let the broker rank on evidence.
    """
    text = text or ""
    score = 0.15
    length = len(text)
    if length > 500:
        score += 0.15
    if length > 2000:
        score += 0.15
    if length > 8000:
        score += 0.15
    # code fences / structured asks cost reasoning
    if "```" in text or text.count("\n") > 30:
        score += 0.15
    low = text.lower()
    hard_markers = ("prove", "derive", "debug", "refactor", "compare",
                    "trade-off", "tradeoff", "why does", "explain why",
                    "step by step", "architecture", "algorithm")
    if any(m in low for m in hard_markers):
        score += 0.2
    if low.count("?") >= 3 or len(text.split()) > 400:
        score += 0.1
    return round(min(1.0, score), 2)


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
        the call.  ``timeout_s`` (when given) bounds the call on a daemon
        thread — a stalled provider chain comes back as a typed timeout
        error, never a hung caller.  Never raises: the worst case is an
        error response.
        """
        timeout_s = kw.pop("timeout_s", None)
        call_kw = dict(kw)
        if task_kind and not self._accepts(fn, "task_kind"):
            _log.debug("brain: %s() has no task_kind support; calling plain",
                       op)
        else:
            call_kw["task_kind"] = task_kind
        for name in ("constraints", "tier", "route"):
            if name in call_kw and not self._accepts(fn, name):
                del call_kw[name]
        try:
            if timeout_s is not None and timeout_s > 0:
                return self._coerce_response(
                    op, self._call_bounded(op, fn, timeout_s, *args, **call_kw))
            return self._coerce_response(op, fn(*args, **call_kw))
        except Exception as exc:  # noqa: BLE001 — honest failure, not a raise
            _log.debug("brain.%s failed: %s", op, exc, exc_info=True)
            return LLMResponse(
                text="", error=f"brain.{op} failed before dispatch: {exc}")

    @staticmethod
    def _call_bounded(op: str, fn: Any, timeout_s: float, *args: Any,
                      **kw: Any) -> LLMResponse:
        """Run ``fn`` on a daemon thread; give up after ``timeout_s``.

        The provider chain behind a router can stall for minutes
        (per-provider timeout × retries × failover); interactive callers
        must never inherit that.  On expiry the in-flight attempt is
        abandoned and a typed ``timeout`` failure comes back — the brain
        contract (never raises) holds.
        """
        box: dict[str, Any] = {}

        def _run() -> None:
            try:
                box["resp"] = fn(*args, **kw)
            except Exception as exc:  # noqa: BLE001
                box["exc"] = exc

        worker = threading.Thread(target=_run, name=f"brain-{op}",
                                  daemon=True)
        worker.start()
        worker.join(timeout_s)
        if worker.is_alive():
            _log.warning("brain.%s timed out after %.1fs — abandoning",
                         op, timeout_s)
            return LLMResponse(
                text="",
                error=f"brain.{op} timed out after {timeout_s:.0f}s",
                failure_class=FailureClass.TIMEOUT.value,
            )
        if "exc" in box:
            exc = box["exc"]
            return LLMResponse(
                text="", error=f"brain.{op} failed before dispatch: {exc}")
        resp = box.get("resp")
        # The caller coerces this into an LLMResponse (duck-typed router
        # responses included) — returned raw here.
        return resp

    def _coerce_response(self, op: str, resp: Any) -> LLMResponse:
        """Coerce a duck-typed router response into an LLMResponse.

        Legacy routers and test doubles return ``SimpleNamespace(ok=..,
        text=..)`` instead of LLMResponse; the brain contract is
        LLMResponse, so coerce instead of failing the call.  Never raises.
        """
        if isinstance(resp, LLMResponse):
            return resp
        if resp is None:
            return LLMResponse(text="", error=f"brain.{op} returned no response")
        try:
            text = getattr(resp, "text", "") or ""
            error = getattr(resp, "error", "") or ""
            if not error and not getattr(resp, "ok", True):
                error = f"brain.{op} reported failure without detail"
            return LLMResponse(
                text=str(text),
                error=str(error),
                model=str(getattr(resp, "model", "") or ""),
                provider=str(getattr(resp, "provider", "") or ""),
                fallback_note=str(getattr(resp, "fallback_note", "") or ""),
            )
        except Exception:  # noqa: BLE001 — coercion never breaks the call
            return LLMResponse(text="", error=f"brain.{op} returned no response")

    def _classify(self, resp: LLMResponse) -> LLMResponse:
        """Attach the typed failure class to a failed response (in place).

        The modern router tags it; legacy routers do not — classify from
        the error text so every caller sees the same contract.
        """
        try:
            if not resp.ok and not resp.failure_class and resp.error:
                resp.failure_class = classify_failure(
                    resp.error).failure_class.value
        except Exception:  # noqa: BLE001 — tagging never breaks the call
            pass
        return resp

    def _context_budget(self) -> int:
        """Token budget for the active serving provider's window.

        Prefers the broker card's advertised context length for the active
        provider; falls back to a provider attribute; else the default.
        Never raises.
        """
        try:
            router = self._get_router()
            active = getattr(router, "active", "") or ""
            broker = getattr(router, "broker", None)
            if broker is not None and active:
                for card in broker.cards():
                    if card.provider == active and card.context_len:
                        return int(card.context_len)
            provider = router.get(active) if hasattr(router, "get") else None
            ctx_len = int(getattr(provider, "context_len", 0) or 0)
            if ctx_len > 0:
                return ctx_len
        except Exception:  # noqa: BLE001
            _log.debug("brain: budget lookup failed", exc_info=True)
        return DEFAULT_CONTEXT_TOKENS

    def _fit_for_send(self, messages: Sequence[Message],
                      task_kind: str) -> list[Message]:
        """Proactively fit messages to the serving window (cheap, local)."""
        from .base import estimate_messages

        msgs = list(messages)
        try:
            budget = self._context_budget()
            if estimate_messages(msgs) > budget:
                _log.info("brain: %d est. tokens exceed %d window — fitting "
                          "(%s)", estimate_messages(msgs), budget, task_kind)
                fitted = fit_messages(
                    msgs, budget, task_kind,
                    summarizer=self._summarize_for_fit)
                if fitted.ok:
                    return fitted.messages
                _log.warning("brain: context fit overflowed; sending anyway")
        except Exception:  # noqa: BLE001 — fitting is best-effort
            _log.debug("brain: pre-fit failed", exc_info=True)
        return msgs

    def _summarize_for_fit(self, text: str) -> str:
        """Summarizer backing the SummarizeMiddle strategy."""
        try:
            resp = self.complete(
                f"{system_prompt_for('summarize')}\n\n{text[:12000]}",
                task_kind="summarize",
                params=SamplingParams(temperature=0.1, max_tokens=400),
                timeout_s=30.0,
            )
            return resp.text if resp.ok else ""
        except Exception:  # noqa: BLE001
            return ""

    def summarize(self, text: str, *, max_tokens: int = 400,
                  timeout_s: float | None = 60.0) -> LLMResponse:
        """Summarize text (task_kind="summarize").  Never raises."""
        return self.complete(
            f"{system_prompt_for('summarize')}\n\n{(text or '')[:24000]}",
            task_kind="summarize",
            params=SamplingParams(temperature=0.1, max_tokens=max_tokens),
            timeout_s=timeout_s,
        )

    def chat(
        self,
        messages: Sequence[Message],
        params: SamplingParams | None = None,
        *,
        task_kind: str = "",
        constraints: Any = None,
        tier: str | None = None,
        timeout_s: float | None = None,
        quality: float | None = None,
        route: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Chat completion.  Never raises.

        ``params`` is positional to match the router's calling convention —
        the Brain is a drop-in for ``router.chat`` at every call site.
        """
        """Chat completion.  Never raises - failure is ``resp.error``.

        ``timeout_s`` bounds the whole provider chain.  Messages are
        proactively fitted to the serving window, and a
        ``context_overflow`` failure triggers one shrink-and-retry with
        the middle summarized - the caller never sees the raw 400.
        """
        msgs = self._fit_for_send(messages, task_kind or "chat")
        resp = self._classify(self._call(
            "chat", self._get_router().chat, msgs, params,
            task_kind=task_kind, tier=tier,
            constraints=_with_quality(constraints, quality),
            timeout_s=timeout_s, route=route))
        if (not resp.ok
                and resp.failure_class == FailureClass.CONTEXT_OVERFLOW.value):
            # The chain failed over and every provider choked on size: the
            # serving window is smaller than our budget estimate (or every
            # provider is small).  Shrink harder — summarize the middle —
            # at half the current estimate and retry once.
            from .base import estimate_messages as _est

            retry_budget = max(64, _est(list(messages)) * 2 // 3)
            fitted = fit_messages(
                list(messages), retry_budget, task_kind or "chat",
                summarizer=self._summarize_for_fit,
                chain=("compact", "summarize_middle", "truncate_oldest"),
                floor=64)
            if fitted.ok and fitted.messages != list(messages):
                _log.info("brain: context overflow - retrying with fitted "
                          "context (%d est. tokens)", fitted.tokens)
                resp = self._classify(self._call(
                    "chat", self._get_router().chat, fitted.messages, params,
                    task_kind=task_kind, tier=tier,
                    constraints=constraints, timeout_s=timeout_s))
        return resp

    def complete(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        *,
        task_kind: str = "",
        constraints: Any = None,
        tier: str | None = None,
        timeout_s: float | None = None,
        quality: float | None = None,
        route: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Single-prompt completion.  Never raises.

        ``params`` is positional to match the router's calling convention.
        ``quality`` (0..1) is the complexity-tier knob: the broker picks the
        cheapest model whose quality meets the bar instead of the default
        pick.  ``route`` carries per-request routing knobs
        (sort/only/ignore/max_price/allow_fallbacks).
        """
        text = prompt or ""
        try:
            from .base import estimate_tokens

            budget = self._context_budget()
            if estimate_tokens(text) > budget:
                text = fit_prompt(text, budget, task_kind or "chat")
        except Exception:  # noqa: BLE001 — fitting is best-effort
            _log.debug("brain: complete pre-fit failed", exc_info=True)
        return self._classify(self._call(
            "complete", self._get_router().complete, text, params,
            task_kind=task_kind, tier=tier,
            constraints=_with_quality(constraints, quality),
            timeout_s=timeout_s, route=route))

    def describe_image(
        self,
        image: bytes,
        prompt: str = "",
        *,
        params: SamplingParams | None = None,
        constraints: Any = None,
        tier: str | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Vision.  Never raises; routes to a VL-capable model."""
        return self._classify(self._call(
            "describe_image", self._get_router().describe_image,
            image, prompt, params,
            task_kind="vision", tier=tier, constraints=constraints,
            timeout_s=timeout_s))

    def embed(self, texts: Sequence[str], **kw: Any) -> tuple[list[list[float]], str]:
        """Embeddings.  Never raises — returns ``(vectors, error)``.

        ``vectors`` is empty and ``error`` names the problem on failure.
        """
        try:
            vectors = self._get_router().embed(list(texts), **kw)
            return list(vectors), ""
        except Exception as exc:  # noqa: BLE001 — honest failure, not a raise
            _log.debug("brain.embed failed: %s", exc, exc_info=True)
            return [], f"brain.embed failed: {exc}"

    # ── structured output (native response validation/repair) ─────────────
    def chat_json(
        self,
        messages: Sequence[Message],
        params: SamplingParams | None = None,
        *,
        task_kind: str = "",
        attempts: int = 3,
        timeout_s: float | None = None,
    ) -> tuple[Any | None, LLMResponse]:
        """Ask for JSON and get parsed data back — with repair retries.

        The extract system prompt is prepended; when the reply is not
        valid JSON the failed draft is shown back to the model with a
        repair nudge, up to ``attempts`` times.  Returns
        ``(data, last_response)`` — ``data`` is ``None`` when every
        attempt failed to produce parseable JSON.  Never raises.
        """
        from ..core.jsonutil import extract_json

        base_params = params or SamplingParams(temperature=0.2, max_tokens=1024)
        extract_system = system_prompt_for("extract")
        msgs: list[Message] = list(messages)
        if extract_system and not any(
                m.role == "system" and "ONLY" in m.content for m in msgs):
            msgs = [Message.system(extract_system)] + msgs
        last = LLMResponse(text="", error="no attempts made")
        data: Any | None = None
        for attempt in range(max(1, attempts)):
            last = self.chat(msgs, task_kind=task_kind or "judge",
                             params=base_params, timeout_s=timeout_s)
            if not last.ok or not (last.text or "").strip():
                continue
            data = extract_json(last.text)
            if data is not None:
                return data, last
            _log.debug("brain.chat_json: attempt %d/%d not JSON — repairing",
                       attempt + 1, attempts)
            msgs = msgs + [
                Message.assistant(last.text.strip()[:4000]),
                Message.user(
                    "That was not valid JSON. Reply with ONLY the JSON "
                    "object — no prose, no code fences."),
            ]
        return None, last

    # ── multi-model adjudication ──────────────────────────────────────────
    def best_of(
        self,
        prompt: str,
        *,
        n: int = 3,
        task_kind: str = "",
        params: SamplingParams | None = None,
        timeout_s: float | None = 90.0,
        judge_panel: bool = False,
        reference: str = "",
    ) -> tuple[str, Any]:
        """Fan out to ``n`` healthy providers, judge the answers, serve the winner.

        Returns ``(winning_text, Judgment)``.  When fan-out or judging
        fails, falls back to a single brain call.  ``judge_panel=True``
        runs one judge per provider and takes the majority vote across
        model families (offsets single-judge bias); otherwise a single
        debiased judge decides.  ``reference`` is the gold answer the
        judge grades closeness to.  Never raises.
        """
        from .adjudicate import Judge, Judgment, fan_out

        router = self._get_router()
        names: list[str] = []
        try:
            is_cooling = getattr(router, "is_cooling_down", None)
            for name in router.providers():
                if callable(is_cooling) and is_cooling(name):
                    continue
                names.append(name)
                if len(names) >= max(1, n):
                    break
        except Exception:  # noqa: BLE001
            names = []
        if len(names) < 2:
            # Not enough healthy providers for a real fan-out.
            resp = self.complete(prompt, task_kind=task_kind, params=params,
                                 timeout_s=timeout_s)
            return (resp.text if resp.ok else "",
                    Judgment(winner=0, ranking=[0],
                             rationale="single provider",
                             ok=resp.ok, error=resp.error))

        ask_params = params or SamplingParams(temperature=0.7)

        def _ask_one(index: int) -> LLMResponse:
            provider = router.get(names[index])
            return provider.chat([Message.user(prompt)], ask_params)

        candidates = fan_out(_ask_one, prompt, len(names),
                             timeout_s=timeout_s)
        if not candidates:
            resp = self.complete(prompt, task_kind=task_kind, params=params,
                                 timeout_s=timeout_s)
            return (resp.text if resp.ok else "",
                    Judgment(winner=0, ranking=[0],
                             rationale="fan-out produced nothing",
                             ok=resp.ok, error=resp.error or "fan-out empty"))
        if len(candidates) == 1:
            return candidates[0], Judgment(
                winner=0, ranking=[0], rationale="only one answer")
        # Brain.chat takes params keyword-only; the Judge's ask contract is
        # positional (messages, params) — adapt once here.
        ask = lambda messages, params, **kw: self.chat(  # noqa: E731
            messages, params=params, **kw)
        if judge_panel and len(names) >= 2:
            # One judge per healthy provider (different families vote) —
            # majority across families offsets any single model's bias.
            asks = []
            for name in names:
                provider = router.get(name)
                if provider is None:
                    continue

                def _judge_ask(messages: Any, p: Any,
                               _provider: Any = provider, **kw: Any) -> LLMResponse:
                    return _provider.chat(messages, p)

                asks.append(_judge_ask)
            judgment = Judge(ask).panel(
                prompt, candidates, asks, timeout_s=timeout_s,
                reference=reference)
        else:
            judgment = Judge(ask).adjudicate(
                prompt, candidates, timeout_s=timeout_s,
                reference=reference)
        winner = (candidates[judgment.winner]
                  if 0 <= judgment.winner < len(candidates)
                  else candidates[0])
        return winner, judgment

    def best_of_n(
        self,
        prompt: str,
        n: int = 5,
        *,
        task_kind: str = "",
        params: SamplingParams | None = None,
        timeout_s: float | None = 120.0,
        reference: str = "",
    ) -> tuple[str, Any]:
        """Best-of-N sampling (OpenRouter-maximizer style): generate ``n``
        answers across the chain, return the panel-judged best.

        ``n``x the cost of one call for a meaningfully better answer on
        hard prompts — the maximizer's headline feature, native.
        """
        return self.best_of(prompt, n=n, task_kind=task_kind,
                            params=params, timeout_s=timeout_s,
                            judge_panel=True, reference=reference)

    def escalate(
        self,
        prompt: str,
        *,
        task_kind: str = "chat",
        params: SamplingParams | None = None,
        timeout_s: float | None = None,
    ) -> LLMResponse:
        """Walk the broker's escalation chain cheapest-first (nexus cascade).

        The first success wins; every attempt is annotated on the
        response's ``route_trace``/``fallback_note`` so the operator sees
        which rung served.  Minimizes expected spend: the common case
        succeeds on the cheapest rung.  Never raises.
        """
        router = self._get_router()
        broker = getattr(router, "broker", None)
        chain: list[Any] = []
        if broker is not None:
            try:
                chain = broker.escalation_chain("chat", task_kind)
            except Exception:  # noqa: BLE001
                chain = []
        if not chain:
            return self.complete(prompt, params=params, task_kind=task_kind,
                                 timeout_s=timeout_s)
        attempts: list[str] = []
        last = LLMResponse(text="", error="no attempt made")
        for card in chain:
            provider_name = card.provider
            try:
                set_active = getattr(router, "set_active", None)
                if callable(set_active):
                    set_active(provider_name)
                resp = self.complete(prompt, params=params,
                                     task_kind=task_kind,
                                     timeout_s=timeout_s,
                                     route={"only": [provider_name],
                                            "allow_fallbacks": False})
            except Exception as exc:  # noqa: BLE001
                resp = LLMResponse(text="", error=str(exc))
            attempts.append(provider_name)
            if resp.ok:
                resp.fallback_note = (
                    f"escalation: served by {provider_name} "
                    f"(rung {len(attempts)}/{len(chain)}"
                    + (f"; cheaper rungs failed: {', '.join(attempts[:-1])}"
                       if len(attempts) > 1 else "") + ")")
                resp.failed_providers = attempts[:-1]
                return resp
            last = resp
        last.fallback_note = ("escalation exhausted: "
                              + ", ".join(attempts))
        last.failed_providers = attempts
        return last

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
                    "failure_class": str(h.get("failure_class", "") or ""),
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
                fclass = f" [{p['failure_class']}]" if p.get("failure_class") else ""
                err = f" — {p['last_error']}" if p["last_error"] else ""
                lines.append(
                    f"  {p['name']:<16} {flag:<9} {p['model']}{owner}{fclass}{err}")
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
