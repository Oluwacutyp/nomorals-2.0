"""OpenAI-compatible provider.

One implementation covers a lot of ground because the ecosystem converged on this
wire format: vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter, Together,
Groq, and any OpenAI-compatible gateway in front of a Dolphin checkpoint.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Callable, Collection, Sequence

from ...core.errors import (
    ModelError,
    ProviderError,
    classify,
    http_status_of,
    is_auth_error,
    is_not_found_error,
)
from ...core.http import HttpClient
from ...core.logging_setup import get_logger
from ...core.retry import BackoffPolicy, default_retryable, retry_call
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    SamplingParams,
    Usage,
    short_error,
    split_reasoning,
    validate_messages,
)
from ..failures import FailureClass, classify_failure


_log = get_logger(__name__)


#: A rate-limited model whose Retry-After window is at most this long
#: gets ONE brief in-leg retry before the model walk moves on — the
#: window is usually a per-minute reset a few seconds out. Longer
#: windows walk immediately: a turn never stalls on a dead model.
_IN_LEG_RETRY_AFTER_MAX_S = 5.0


def _same_model_retryable(exc: BaseException) -> bool:
    """May the transport retry re-send this request to the SAME model?

    Only transport transients (server / network / timeout /
    unclassified): re-sending can heal those. Model-scoped failures
    (rate limit, dead id, parse rejection) are the model walk's
    decision, and provider-scoped ones (auth) must abort the leg —
    the retry loop must not sit on either, sleeping out Retry-After
    windows against a model that cannot answer.
    """
    if not default_retryable(exc):
        return False
    status = http_status_of(exc)
    if status is not None and 400 <= status < 500:
        # A definitive client error (bad key, dead id, parse rejection,
        # rate limit) cannot heal by resending the identical payload —
        # it is the model walk's decision, or the leg's abort.
        return False
    info = classify_failure(str(exc), exc)
    return info.failure_class in (
        FailureClass.SERVER,
        FailureClass.NETWORK,
        FailureClass.TIMEOUT,
        FailureClass.UNKNOWN,
    )


def _model_scoped_failure(exc: BaseException) -> bool:
    """Is this failure about THIS model id rather than the provider?

    Model-scoped: the model is rate-limited (quotas are metered per
    model — Groq's free tier included), the id is dead/unknown (404,
    or a model_not_found body under another status), or the model
    rejected the request with a 400 (gpt-oss's harmony "Parsing
    failed" is exactly this: another model family accepts the same
    payload). Provider-scoped failures — auth, unreachable endpoint,
    context overflow — return False: a different id cannot fix them.
    """
    if is_auth_error(exc):
        return False
    if is_not_found_error(exc):
        return True
    info = classify_failure(str(exc), exc)
    if info.failure_class in (FailureClass.RATE_LIMITED,
                              FailureClass.CONFIG):
        return True
    return http_status_of(exc) == 400


def _retry_after_hint(exc: BaseException, info: Any) -> float | None:
    """The server's Retry-After for this failure, when it gave one."""
    hint = getattr(exc, "retry_after", None)
    if isinstance(hint, (int, float)) and not isinstance(hint, bool) \
            and hint > 0:
        return float(hint)
    return info.retry_after_s


def heal_model_id(configured: str, known: Collection[str]) -> str:
    """Heal a near-miss model id to a known-good one.

    A configured id that differs from a known id only by a missing
    trailing ``"b"`` (``openai/gpt-oss-120``), a missing ``":free"``
    suffix (``qwen/qwen3-8b``), or both (``qwen/qwen3-8``) is almost
    certainly a truncated copy of the known id — a hand-typed env value,
    a chat-pasted name, a log line re-typed from memory.  Sending the
    truncated id to the API 404s/400s every call, and the failover chain
    only helps if it recognises the error.  Heal *before* the first
    attempt so the name actually sent is valid.

    Only exact near-misses heal: the configured id plus ``"b"``,
    ``":free"``, or ``"b:free"`` must equal a known id.  Any other id
    passes through untouched — a deliberately different model is never
    second-guessed.
    """
    cfg = (configured or "").strip()
    if not cfg or cfg in known:
        return cfg
    for candidate in (cfg + "b", cfg + ":free", cfg + "b:free"):
        if candidate in known:
            return candidate
    # Missing "b" before a trailing ":free" ("qwen/qwen3-8:free").
    if cfg.endswith(":free"):
        stem = cfg[: -len(":free")]
        if stem + "b:free" in known:
            return stem + "b:free"
    return cfg


def _parse_usage(raw: Any) -> "Usage":
    """Usage from an OpenAI-style ``usage`` block, incl. cache/reasoning.

    Providers report ``prompt_tokens_details.cached_tokens`` and
    ``completion_tokens_details.reasoning_tokens`` — billed tokens the
    user never sees.  Counting only the visible text under-bills by up
    to 50x on reasoning-heavy calls.
    """
    usage = raw if isinstance(raw, dict) else {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    try:
        cached = int(prompt_details.get("cached_tokens") or 0)
    except (TypeError, ValueError):
        cached = 0
    try:
        reasoning = int(completion_details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        reasoning = 0
    return Usage(
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        total_tokens=int(usage.get("total_tokens") or 0),
        cached_tokens=cached,
        reasoning_tokens=reasoning,
    )


class OpenAICompatProvider(LLMProvider):
    """Talks to any ``/v1/chat/completions`` endpoint."""

    name = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "",
        model: str = "",
        timeout: float = 120.0,
        max_retries: int = 3,
        template: str = "chatml",
        organization: str = "",
        extra_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
        vision_model: str = "",
        vision_base_url: str = "",
        vision_api_key: str = "",
        api_keys: Sequence[str] | None = None,
        fallback_models: Sequence[str] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")
        # Key pool (Bifrost pattern): several keys for one provider act as
        # one pool — a 429 rotates to the next key before the router fails
        # over to another provider.  The exhausted quota is usually the
        # key's, not the provider's.
        pool = [k for k in [api_key, *(api_keys or [])] if k]
        seen: list[str] = []
        for k in pool:
            if k not in seen:
                seen.append(k)
        self.api_keys: list[str] = seen
        self._key_index = 0
        self.api_key = self.api_keys[0] if self.api_keys else ""
        self.model = model
        # Explicit ordered fallback models (operator config — NM_*_MODELS).
        # None = use the subclass roster (_fallback_models); a list, even
        # empty, is the operator's word and replaces the roster.
        self._explicit_fallbacks: list[str] | None = (
            [str(m).strip() for m in fallback_models if str(m).strip()]
            if fallback_models is not None else None
        )
        self.template = template
        self.extra_body = dict(extra_body or {})
        # A dedicated vision model/endpoint: text chat and image understanding
        # are often served by different models (a 7B chat model cannot see).
        self.vision_model = vision_model
        self.vision_base_url = (vision_base_url or "").rstrip("/")
        self.vision_api_key = vision_api_key
        headers = dict(extra_headers or {})
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if organization:
            headers["OpenAI-Organization"] = organization
        self.http = HttpClient(timeout=timeout, headers=headers)
        self._policy = BackoffPolicy(max_attempts=max(1, max_retries), cap=30.0)

    def _apply_key(self, key: str) -> None:
        """Swap the Authorization header to ``key`` (key-pool rotation)."""
        self.api_key = key
        try:
            self.http._headers["Authorization"] = f"Bearer {key}"
        except Exception:  # noqa: BLE001 — header store is an impl detail
            try:
                self.http.headers["Authorization"] = f"Bearer {key}"
            except Exception as exc:  # noqa: BLE001
                # Both header stores failed: the key pool rotated in
                # memory (self.api_key) but the HTTP client still sends
                # the old key. Log it — silently sending the stale key
                # on every call used to be invisible.
                _log.warning(
                    "openai_compat: key rotation could not update the "
                    "Authorization header: %s", exc
                )

    def rotate_key(self) -> bool:
        """Move to the next key in the pool.  True when one was available.

        Called by the router on a 429 before failing over — the quota that
        is exhausted is usually the key's, not the provider's.
        """
        if len(self.api_keys) < 2:
            return False
        self._key_index = (self._key_index + 1) % len(self.api_keys)
        self._apply_key(self.api_keys[self._key_index])
        return True

    @property
    def key_pool_size(self) -> int:
        return len(self.api_keys)

    @property
    def model_id(self) -> str:
        return self.model or self.base_url

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "complete", "embed", "vision"}

    def health(self) -> bool:
        return self.health_detail()[0]

    def health_detail(self) -> tuple[bool, str]:
        """Liveness probe plus the failure reason.

        ``GET {base_url}/models`` is the cheap check; on a non-OK status
        the reason carries the HTTP status + a short body snippet
        (e.g. ``401: Invalid API Key``) so ``/providers`` can tell the
        owner *why* the provider is dead instead of a bare \"failed\".
        """
        try:
            response = self.http.get(f"{self.base_url}/models", timeout=min(10.0, self.timeout))
        except Exception as exc:  # noqa: BLE001 - liveness probe
            return False, str(exc) or exc.__class__.__name__
        if getattr(response, "ok", False):
            return True, ""
        status = getattr(response, "status_code", None)
        body = getattr(response, "text", "") or getattr(response, "body", "") or ""
        snippet = " ".join(str(body).split())[:120]
        if status is not None:
            return False, f"{status}: {snippet}" if snippet else f"{status}"
        return False, snippet or "probe request failed"

    def _fallback_models(self) -> list[str]:
        """Model ids to try when the configured model 404s.

        A dead model id (retired name, typo in env config) is the most
        common cloud-provider outage this layer sees, and retrying the
        same dead id is pure waste.  Subclasses with a known-good roster
        (Groq, OpenRouter) override this; the generic base returns none —
        for an arbitrary OpenAI-compatible server there is no safe guess.
        """
        return []

    def _model_chain(self) -> list[str]:
        """Ordered model ids for one chat call: configured first.

        The fallback list is the explicit ``fallback_models`` when the
        operator configured one (config wins), else the subclass roster
        (:meth:`_fallback_models`). Deduped, order preserved.
        """
        fallbacks = (self._explicit_fallbacks
                     if self._explicit_fallbacks is not None
                     else self._fallback_models())
        chain: list[str] = []
        for model_id in [self.model, *fallbacks]:
            model_id = (model_id or "").strip()
            if model_id and model_id not in chain:
                chain.append(model_id)
        return chain or [self.model]

    def _post_with_model_walk(
        self,
        chain: list[str],
        post: Callable[[str], Any],
        *,
        soft_failure: Callable[[Any], str | None] | None = None,
    ) -> tuple[str, Any, list[tuple[str, str]]]:
        """Run ``post(model_id)`` across ``chain`` under chat's failover policy.

        This is the model walk :meth:`embed` and :meth:`describe_image`
        share with :meth:`chat` (chat's loop is its own because of its
        response shaping and dead-id heal, but the classification is
        the same module-level policy): the transport leg (``post``)
        retries only what :func:`_same_model_retryable` blesses; what
        reaches this loop is the per-model outcome — a model-scoped
        failure (this model's rate limit, dead id, parse rejection)
        walks to the next id, a provider-scoped failure (auth,
        unreachable endpoint) aborts immediately, because a different
        id cannot fix it. A 429 whose Retry-After window is short earns
        ONE brief in-leg retry of the same model before the walk moves
        on, exactly as in :meth:`chat`.

        ``soft_failure(raw)`` marks a 200 response that is really a
        per-model failure (the embeddings form of chat's empty
        choices): it walks like a model-scoped failure, and on the last
        model the reason raises as a :class:`ModelError`.

        Returns ``(model_id, raw, attempts)`` for the serving model.
        When every model fails, raises the last transport exception —
        the same terminal shape a single-model caller has always seen.
        """
        attempts: list[tuple[str, str]] = []
        for attempt_no, model_id in enumerate(chain):
            raw: Any = None
            exc: Exception | None = None
            try:
                raw = post(model_id)
            except Exception as first_exc:  # noqa: BLE001
                exc = first_exc
                attempts.append((model_id, short_error(first_exc)))
                info = classify_failure(str(first_exc), first_exc)
                hint = _retry_after_hint(first_exc, info)
                if (info.failure_class is FailureClass.RATE_LIMITED
                        and hint is not None
                        and 0.0 < hint <= _IN_LEG_RETRY_AFTER_MAX_S):
                    time.sleep(hint)
                    try:
                        raw = post(model_id)
                        exc = None
                    except Exception as second_exc:  # noqa: BLE001
                        exc = second_exc
                        attempts.append((model_id, short_error(second_exc)))
            if exc is not None:
                if _model_scoped_failure(exc) \
                        and attempt_no < len(chain) - 1:
                    continue
                raise exc
            if soft_failure is not None:
                reason = soft_failure(raw)
                if reason is not None:
                    attempts.append((model_id, reason))
                    if attempt_no < len(chain) - 1:
                        continue
                    raise ModelError(reason)
            return model_id, raw, attempts
        # Unreachable by construction: callers build the chain from
        # _model_chain() (never empty) or a pinned single model.
        raise ModelError("no model attempted")  # pragma: no cover

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        messages = validate_messages(messages, who="openai_compat.chat")
        sampling = (params or SamplingParams()).clamped()
        # A per-call timeout (the router's startup verify passes one)
        # bounds the HTTP call — it must never leak into the wire
        # payload, where strict APIs reject unknown fields.
        req_timeout = kw.pop("timeout", None)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            **sampling.to_openai(),
            **self.extra_body,
            **kw,
        }
        original_model = self.model
        chain = self._model_chain()
        attempts: list[tuple[str, str]] = []
        last_exc: Exception | None = None
        # Set when the walk leaves the configured model because its id
        # is dead (404 / model_not_found): the fallback id is then KEPT
        # as the active model (a dead id fails every call). Transient
        # walks (rate limit, parse rejection) restore the configured
        # model afterwards — its quota refills; its rank does not drop.
        dead_id_heal = False

        def _post(model_id: str) -> Any:
            payload["model"] = model_id
            return retry_call(
                lambda: self.http.post_json(
                    f"{self.base_url}/chat/completions", payload,
                    **({"timeout": req_timeout}
                       if req_timeout is not None else {})),
                policy=self._policy,
                retryable=_same_model_retryable,
            )

        for attempt_no, model_id in enumerate(chain):
            self.model = model_id
            raw: Any = None
            exc: Exception | None = None
            try:
                raw = _post(model_id)
            except Exception as first_exc:  # noqa: BLE001
                exc = first_exc
                attempts.append((model_id, short_error(first_exc)))
                # A 429 whose Retry-After window is short gets ONE
                # brief in-leg retry of the SAME model before the walk
                # moves on. Long windows walk immediately.
                info = classify_failure(str(first_exc), first_exc)
                hint = _retry_after_hint(first_exc, info)
                if (info.failure_class is FailureClass.RATE_LIMITED
                        and hint is not None
                        and 0.0 < hint <= _IN_LEG_RETRY_AFTER_MAX_S):
                    time.sleep(hint)
                    try:
                        raw = _post(model_id)
                        exc = None
                    except Exception as second_exc:  # noqa: BLE001
                        exc = second_exc
                        attempts.append((model_id, short_error(second_exc)))
            if exc is not None:
                last_exc = exc
                # Model-scoped failure (this model's rate limit, dead
                # id, parse rejection) → the next id may live. Anything
                # else — auth, unreachable endpoint, context overflow,
                # provider-side 5xx — is provider-scoped: the transport
                # policy already retried what retrying can heal, and a
                # different model id cannot fix the rest.
                if _model_scoped_failure(exc) \
                        and attempt_no < len(chain) - 1:
                    if attempt_no == 0 and is_not_found_error(exc):
                        # The CONFIGURED id is dead — the fallback that
                        # eventually serves is kept as the active model.
                        dead_id_heal = True
                    continue
                break

            data = raw.json()
            choices = data.get("choices") or []
            if not choices:
                # A soft per-model failure (the free-alias gateway does
                # this when its pick has no headroom): walk like any
                # other model-scoped failure; only the last model's
                # empty answer is reported as-is.
                attempts.append((model_id, "provider returned no choices"))
                if attempt_no < len(chain) - 1:
                    continue
                self.model = original_model
                return self._record(
                    LLMResponse(text="", model=model_id, error="provider returned no choices", raw=data),
                    started,
                )
            choice = choices[0]
            message = choice.get("message") or {}
            content = message.get("content")
            if content is None and message.get("tool_calls"):
                import json

                content = json.dumps(message["tool_calls"])
            # Channel separation: the message's dedicated reasoning
            # fields (reasoning_content / reasoning) are never read
            # into the text, and reasoning inlined into the content
            # channel (think-tag spans) is split out here — the
            # response text is the final-answer channel only.
            if content:
                content = split_reasoning(str(content))[0]
            usage = _parse_usage(data.get("usage"))
            response = LLMResponse(
                text=content or "",
                model=data.get("model", model_id),
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                raw=data,
            )
            if model_id != original_model:
                if isinstance(response.raw, dict):
                    response.raw["model_failover_attempts"] = [
                        {"model": m, "error": e} for m, e in attempts]
                    if dead_id_heal:
                        # Healed onto a fallback id: keep it (a dead id
                        # 404s every call), and record where we came from.
                        response.raw["healed_from"] = original_model
                    else:
                        response.raw["failover_from"] = original_model
                if not dead_id_heal:
                    self.model = original_model
            return self._record(response, started)

        # Every model id failed — restore the configured one and report the
        # whole attempt chain, not just the last error.
        self.model = original_model
        error = classify(last_exc) if last_exc is not None else ModelError("unknown error")
        chain_desc = "; ".join(f"{m} ({e})" for m, e in attempts) or "no attempt made"
        return self._record(
            LLMResponse(
                text="",
                model=original_model,
                error=f"{error.code}: all models failed on {self.name} — {chain_desc}",
                raw={"model_failover_attempts": [{"model": m, "error": e} for m, e in attempts]},
            ),
            started,
        )

    def embed(self, texts: Sequence[str], *, model: str = "", **kw: Any) -> list[list[float]]:
        inputs = list(texts)
        # An explicit ``model`` is the caller's word for this call: a
        # one-model chain. Otherwise embeddings walk the same roster
        # chat walks (_model_chain) under the same failover policy —
        # a 429 or dead id on one model must not fail the call while a
        # sibling id is healthy (the pre-ladderfix shape retried the
        # SAME model in place and then gave up). The walk is per-call
        # only: unlike chat it never heals ``self.model``, because an
        # embeddings 404 proves the id cannot embed, not that it
        # cannot chat.
        chain = [model] if model else self._model_chain()

        def _post(model_id: str) -> Any:
            payload = {"model": model_id, "input": inputs}
            return retry_call(
                lambda: self.http.post_json(
                    f"{self.base_url}/embeddings", payload),
                policy=self._policy,
                retryable=_same_model_retryable,
            )

        def _empty_data(raw: Any) -> str | None:
            # A 200 with no data items for real input is the embeddings
            # form of chat's empty choices: a soft per-model failure.
            if inputs and not (raw.json().get("data") or []):
                return (f"embedding provider returned 0 vectors "
                        f"for {len(inputs)} inputs")
            return None

        _served, raw, _attempts = self._post_with_model_walk(
            chain, _post, soft_failure=_empty_data)
        data = raw.json()
        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        vectors = [item.get("embedding") or [] for item in items]
        if len(vectors) != len(inputs):
            raise ModelError(
                f"embedding provider returned {len(vectors)} vectors for {len(inputs)} inputs"
            )
        return vectors

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        """Send an image through the chat endpoint using the data-URL convention."""
        encoded = base64.b64encode(image).decode("ascii")
        mime = kw.pop("mime", "image/png")
        req_timeout = kw.pop("timeout", None)
        messages = [
            Message(
                role="user",
                content=prompt or "Describe this image in detail.",
            )
        ]
        # OpenAI vision shape; servers that do not support it will reject it, which
        # is surfaced as an ordinary provider error rather than a crash.
        payload_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": messages[0].content},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
                ],
            }
        ]
        started = time.perf_counter()
        sampling = (params or SamplingParams()).clamped()
        # Resolve the vision target at call time: a dedicated vision endpoint
        # gets its own client (and its own key), never the text chat one.
        if self.vision_base_url:
            base = self.vision_base_url
            vheaders: dict[str, str] = {}
            if self.vision_api_key:
                vheaders["Authorization"] = f"Bearer {self.vision_api_key}"
            http = HttpClient(timeout=self.timeout, headers=vheaders)
        else:
            base, http = self.base_url, self.http
        # Model chain, mirroring chat's roster policy: a dedicated
        # vision model — or a dedicated vision endpoint, whose roster
        # is not the main endpoint's — is operator-pinned, so the chain
        # is that single target (the transport still retries only
        # transport transients, never a 429 in place). With neither,
        # vision rides the chat endpoint and walks the same roster
        # chat walks (_model_chain). The walk never rewrites
        # ``self.model`` and never heals: a vision-leg 404 is ambiguous
        # (dead id, or a server that cannot take image payloads), so
        # chat's dead-id heal stays chat's own decision.
        if self.vision_model:
            chain = [self.vision_model]
        elif self.vision_base_url:
            chain = [self.model]
        else:
            chain = self._model_chain()

        def _post(model_id: str) -> Any:
            payload = {"model": model_id, "messages": payload_messages,
                       **sampling.to_openai()}
            return retry_call(
                lambda: http.post_json(
                    f"{base}/chat/completions", payload,
                    **({"timeout": req_timeout}
                       if req_timeout is not None else {})),
                policy=self._policy,
                retryable=_same_model_retryable,
            )

        try:
            served_model, raw, _attempts = self._post_with_model_walk(
                chain, _post)
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            raise ProviderError(f"vision request failed: {error.message}", retryable=error.retryable) from exc
        data = raw.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return self._record(
            LLMResponse(text=split_reasoning(str(text))[0],
                        model=data.get("model", served_model), raw=data), started
        )
