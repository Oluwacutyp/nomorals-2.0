"""OpenAI-compatible provider.

One implementation covers a lot of ground because the ecosystem converged on this
wire format: vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter, Together,
Groq, and any OpenAI-compatible gateway in front of a Dolphin checkpoint.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Sequence

from ...core.errors import ModelError, ProviderError, classify, is_not_found_error
from ...core.http import HttpClient
from ...core.retry import BackoffPolicy, retry_call
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    SamplingParams,
    Usage,
    short_error,
    validate_messages,
)


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
            except Exception:  # noqa: BLE001
                pass

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
        try:
            response = self.http.get(f"{self.base_url}/models", timeout=min(10.0, self.timeout))
            return response.ok
        except Exception:  # noqa: BLE001 - liveness probe
            return False

    def _fallback_models(self) -> list[str]:
        """Model ids to try when the configured model 404s.

        A dead model id (retired name, typo in env config) is the most
        common cloud-provider outage this layer sees, and retrying the
        same dead id is pure waste.  Subclasses with a known-good roster
        (Groq, OpenRouter) override this; the generic base returns none —
        for an arbitrary OpenAI-compatible server there is no safe guess.
        """
        return []

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        messages = validate_messages(messages, who="openai_compat.chat")
        sampling = (params or SamplingParams()).clamped()
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            **sampling.to_openai(),
            **self.extra_body,
            **kw,
        }
        original_model = self.model
        fallbacks = [m for m in self._fallback_models() if m and m != original_model]
        attempts: list[tuple[str, str]] = []
        last_exc: Exception | None = None
        for attempt_no, model_id in enumerate([original_model, *fallbacks]):
            self.model = model_id
            payload["model"] = model_id
            try:
                raw = retry_call(
                    lambda: self.http.post_json(f"{self.base_url}/chat/completions", payload),
                    policy=self._policy,
                )
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                attempts.append((model_id, short_error(exc)))
                # 404 = this model id is dead; a different id may live.
                # Anything else (auth, rate limit, server error) is already
                # retried by the policy above — swapping the model cannot
                # fix a bad key or an overloaded server.
                if is_not_found_error(exc) and attempt_no < len(fallbacks):
                    continue
                break

            data = raw.json()
            choices = data.get("choices") or []
            if not choices:
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
            usage = _parse_usage(data.get("usage"))
            response = LLMResponse(
                text=content or "",
                model=data.get("model", model_id),
                usage=usage,
                finish_reason=choice.get("finish_reason", "stop"),
                raw=data,
            )
            # Healed onto a fallback id: keep it (a dead id 404s every call),
            # and record where we came from.
            if model_id != original_model and isinstance(response.raw, dict):
                response.raw["healed_from"] = original_model
            return self._record(response, started)

        # Every model id failed — restore the configured one and report the
        # whole attempt chain, not just the last error.
        self.model = original_model
        error = classify(last_exc) if last_exc is not None else ModelError("unknown error")
        chain = "; ".join(f"{m} ({e})" for m, e in attempts) or "no attempt made"
        return self._record(
            LLMResponse(
                text="",
                model=original_model,
                error=f"{error.code}: all models failed on {self.name} — {chain}",
                raw={"model_failover_attempts": [{"model": m, "error": e} for m, e in attempts]},
            ),
            started,
        )

    def embed(self, texts: Sequence[str], *, model: str = "", **kw: Any) -> list[list[float]]:
        payload = {"model": model or self.model, "input": list(texts)}
        raw = retry_call(
            lambda: self.http.post_json(f"{self.base_url}/embeddings", payload),
            policy=self._policy,
        )
        data = raw.json()
        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        vectors = [item.get("embedding") or [] for item in items]
        if len(vectors) != len(texts):
            raise ModelError(
                f"embedding provider returned {len(vectors)} vectors for {len(texts)} inputs"
            )
        return vectors

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        """Send an image through the chat endpoint using the data-URL convention."""
        encoded = base64.b64encode(image).decode("ascii")
        mime = kw.pop("mime", "image/png")
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
        model = self.vision_model or self.model
        payload = {"model": model, "messages": payload_messages, **sampling.to_openai()}
        try:
            raw = retry_call(
                lambda: http.post_json(f"{base}/chat/completions", payload),
                policy=self._policy,
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            raise ProviderError(f"vision request failed: {error.message}", retryable=error.retryable) from exc
        data = raw.json()
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        return self._record(
            LLMResponse(text=text, model=data.get("model", model), raw=data), started
        )
