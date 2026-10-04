"""OpenAI-compatible provider.

One implementation covers a lot of ground because the ecosystem converged on this
wire format: vLLM, Ollama, LM Studio, llama.cpp server, OpenRouter, Together,
Groq, and any OpenAI-compatible gateway in front of a Dolphin checkpoint.
"""

from __future__ import annotations

import base64
import time
from typing import Any, Sequence

from ...core.errors import ModelError, ProviderError, classify
from ...core.http import HttpClient
from ...core.retry import BackoffPolicy, retry_call
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    SamplingParams,
    Usage,
    validate_messages,
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
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.template = template
        self.extra_body = dict(extra_body or {})
        # A dedicated vision model/endpoint: text chat and image understanding
        # are often served by different models (a 7B chat model cannot see).
        self.vision_model = vision_model
        self.vision_base_url = (vision_base_url or "").rstrip("/")
        self.vision_api_key = vision_api_key
        headers = dict(extra_headers or {})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if organization:
            headers["OpenAI-Organization"] = organization
        self.http = HttpClient(timeout=timeout, headers=headers)
        self._policy = BackoffPolicy(max_attempts=max(1, max_retries), cap=30.0)

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
        try:
            raw = retry_call(
                lambda: self.http.post_json(f"{self.base_url}/chat/completions", payload),
                policy=self._policy,
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            return self._record(
                LLMResponse(text="", model=self.model, error=f"{error.code}: {error.message}"),
                started,
            )

        data = raw.json()
        choices = data.get("choices") or []
        if not choices:
            return self._record(
                LLMResponse(text="", model=self.model, error="provider returned no choices", raw=data),
                started,
            )
        choice = choices[0]
        message = choice.get("message") or {}
        content = message.get("content")
        if content is None and message.get("tool_calls"):
            import json

            content = json.dumps(message["tool_calls"])
        usage = data.get("usage") or {}
        return self._record(
            LLMResponse(
                text=content or "",
                model=data.get("model", self.model),
                usage=Usage(
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                ),
                finish_reason=choice.get("finish_reason", "stop"),
                raw=data,
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
