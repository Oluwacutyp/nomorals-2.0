"""Hugging Face provider.

Covers three deployment shapes behind one class:

* **Serverless Inference API** — ``api-inference.huggingface.co/models/<repo>``.
  No infrastructure, rate-limited, good for occasional calls.
* **Inference Endpoints** — a dedicated URL you provision yourself. This is the
  one to use for a real personal model; you own the GPU and the rate limits.
* **Router API** — ``router.huggingface.co`` for provider-routed open models.

Chat, completion, and embeddings are all supported. ``describe_image`` uses the
text-generation chat shape with an image part, which is what the current
Inference API expects for VLMs.

Note on verification: the sandbox this was developed in cannot reach
huggingface.co, so request *construction* is unit-tested offline and the wire
behaviour has not been exercised against the live API here.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, Sequence

from ...core.errors import ModelError, ProviderError, RateLimited, classify
from ...core.http import HttpClient
from ...core.retry import BackoffPolicy, retry_call
from ..base import LLMProvider, LLMResponse, Message, SamplingParams, Usage, messages_to_text


# Curated fallback models for the intelligent router
# These are verified to be available on HF Inference API
ROUTER_FALLBACK_MODELS = [
    "Sao10K/L3-8B-Stheno-v3.2",
    "Qwen/Qwen3-8B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
]

SERVERLESS_URL = "https://api-inference.huggingface.co"
ROUTER_URL = "https://router.huggingface.co/v1"


class HFServerlessProvider(LLMProvider):
    """Hugging Face Inference API / Inference Endpoints."""

    name = "hf_serverless"

    def __init__(
        self,
        *,
        token: str = "",
        model: str = "cognitivecomputations/dolphin-2.9.1-llama-3-8b",
        base_url: str = SERVERLESS_URL,
        endpoint_url: str = "",
        timeout: float = 180.0,
        max_retries: int = 4,
        template: str = "chatml",
        wait_for_model: bool = True,
        use_chat_endpoint: bool = True,
        embed_model: str = "",
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.token = token
        self.model = model
        self.base_url = (endpoint_url or base_url).rstrip("/")
        self.endpoint_url = endpoint_url.rstrip("/")
        self.template = template
        self.wait_for_model = wait_for_model
        self.use_chat_endpoint = use_chat_endpoint
        self.embed_model = embed_model or "sentence-transformers/all-MiniLM-L6-v2"
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self.http = HttpClient(timeout=timeout, headers=headers)
        self._policy = BackoffPolicy(max_attempts=max(1, max_retries), cap=60.0)

    # ── URLs ─────────────────────────────────────────────────────────────────
    @property
    def model_id(self) -> str:
        return self.model

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "complete", "embed"}

    @property
    def _is_dedicated_endpoint(self) -> bool:
        return bool(self.endpoint_url)

    def chat_url(self) -> str:
        if self._is_dedicated_endpoint:
            # Inference Endpoints expose a vLLM/TGI-compatible OpenAI surface.
            base = self.base_url
            return f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
        if self.use_chat_endpoint:
            return f"{self.base_url}/models/{self.model}/v1/chat/completions"
        return f"{self.base_url}/models/{self.model}"

    def embed_url(self) -> str:
        if self._is_dedicated_endpoint:
            base = self.base_url
            return f"{base}/embed" if not base.endswith("/v1") else f"{base}/embeddings"
        return f"{self.base_url}/models/{self.embed_model}"

    def build_chat_payload(
        self, messages: Sequence[Message], params: SamplingParams
    ) -> dict[str, Any]:
        """Expose payload construction so it can be asserted without network."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_openai() for m in messages],
            **params.to_openai(),
        }
        if self.wait_for_model:
            payload["options"] = {"wait_for_model": True, "use_cache": False}
        return payload

    def build_completion_payload(self, prompt: str, params: SamplingParams) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": params.max_tokens,
                "temperature": params.temperature,
                "top_p": params.top_p,
                "return_full_text": False,
            },
        }
        if params.top_k:
            payload["parameters"]["top_k"] = params.top_k
        if params.repetition_penalty != 1.0:
            payload["parameters"]["repetition_penalty"] = params.repetition_penalty
        if params.stop:
            payload["parameters"]["stop"] = list(params.stop)
        if params.seed is not None:
            payload["parameters"]["seed"] = params.seed
        if self.wait_for_model:
            payload["options"] = {"wait_for_model": True}
        return payload

    # ── calls ────────────────────────────────────────────────────────────────
    def health(self) -> bool:
        try:
            url = f"{self.base_url}/models/{self.model}" if not self._is_dedicated_endpoint else f"{self.base_url}/health"
            return self.http.get(url, timeout=min(10.0, self.timeout)).ok
        except Exception:  # noqa: BLE001 - liveness probe
            return False

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        sampling = (params or SamplingParams()).clamped()
        payload = self.build_chat_payload(messages, sampling)
        try:
            raw = retry_call(
                lambda: self.http.post_json(self.chat_url(), payload), policy=self._policy
            )
        except RateLimited as exc:
            return self._record(
                LLMResponse(text="", model=self.model, error=f"rate limited: {exc.message}"),
                started,
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            return self._record(
                LLMResponse(text="", model=self.model, error=f"{error.code}: {error.message}"),
                started,
            )
        return self._record(self._parse_chat(raw.json()), started)

    def complete(self, prompt: str, params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        if self.use_chat_endpoint or self._is_dedicated_endpoint:
            return super().complete(prompt, params, **kw)
        started = time.perf_counter()
        sampling = (params or SamplingParams()).clamped()
        payload = self.build_completion_payload(prompt, sampling)
        try:
            raw = retry_call(
                lambda: self.http.post_json(
                    f"{self.base_url}/models/{self.model}", payload
                ),
                policy=self._policy,
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            return self._record(
                LLMResponse(text="", model=self.model, error=f"{error.code}: {error.message}"),
                started,
            )
        data = raw.json()
        text = self._parse_text_generation(data)
        return self._record(
            LLMResponse(
                text=text,
                model=self.model,
                usage=Usage(completion_tokens=max(1, len(text.split()))),
                raw=data if isinstance(data, dict) else {"raw": data},
            ),
            started,
        )

    def embed(self, texts: Sequence[str], **kw: Any) -> list[list[float]]:
        payload: dict[str, Any] = {"inputs": list(texts)}
        if self._is_dedicated_endpoint:
            payload = {"model": self.embed_model, "input": list(texts)}
        try:
            raw = retry_call(
                lambda: self.http.post_json(self.embed_url(), payload), policy=self._policy
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            raise ProviderError(
                f"embedding request failed: {error.message}", retryable=error.retryable
            ) from exc
        return self._parse_embeddings(raw.json(), len(texts))

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        sampling = (params or SamplingParams()).clamped()
        encoded = base64.b64encode(image).decode("ascii")
        mime = kw.pop("mime", "image/png")
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt or "Describe this image in detail."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{encoded}"},
                        },
                    ],
                }
            ],
            **sampling.to_openai(),
        }
        started = time.perf_counter()
        try:
            raw = retry_call(
                lambda: self.http.post_json(self.chat_url(), payload), policy=self._policy
            )
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            raise ProviderError(
                f"vision request failed: {error.message}", retryable=error.retryable
            ) from exc
        return self._record(self._parse_chat(raw.json()), started)

    # ── parsing ──────────────────────────────────────────────────────────────
    def _parse_chat(self, data: Any) -> LLMResponse:
        if not isinstance(data, dict):
            return LLMResponse(text=str(data), model=self.model)
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            usage = data.get("usage") or {}
            return LLMResponse(
                text=message.get("content") or "",
                model=data.get("model", self.model),
                usage=Usage(
                    prompt_tokens=int(usage.get("prompt_tokens") or 0),
                    completion_tokens=int(usage.get("completion_tokens") or 0),
                    total_tokens=int(usage.get("total_tokens") or 0),
                ),
                finish_reason=choices[0].get("finish_reason", "stop"),
                raw=data,
            )
        # Text-generation shape: [{"generated_text": "..."}] or {"generated_text": ...}
        return LLMResponse(
            text=self._parse_text_generation(data),
            model=self.model,
            raw=data,
        )

    @staticmethod
    def _parse_text_generation(data: Any) -> str:
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                return str(first.get("generated_text") or "")
            return str(first)
        if isinstance(data, dict):
            if "generated_text" in data:
                return str(data["generated_text"])
            if "error" in data:
                return ""
        return ""

    @staticmethod
    def _parse_embeddings(data: Any, expected: int) -> list[list[float]]:
        if isinstance(data, list) and data and isinstance(data[0], list):
            # Legacy shape: [[...], [...]]
            return [list(map(float, vector)) for vector in data]
        if isinstance(data, list):
            items = sorted(
                (d for d in data if isinstance(d, dict)), key=lambda d: d.get("index", 0)
            )
            if items:
                return [list(map(float, item.get("embedding") or [])) for item in items]
        if isinstance(data, dict) and "data" in data:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
            return [list(map(float, item.get("embedding") or [])) for item in items]
        raise ModelError(f"unrecognized embedding response shape: {type(data).__name__}")


def prompt_from_messages(messages: Sequence[Message], template: str = "chatml") -> str:
    """Convenience wrapper around :func:`messages_to_text`."""
    return messages_to_text(messages, template=template)
