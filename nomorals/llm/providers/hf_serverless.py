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

from ...core.errors import ModelError, ProviderError, RateLimited, classify, is_auth_error
from ...core.http import HttpClient
from ...core.retry import BackoffPolicy, retry_call
from ..base import (
    LLMProvider,
    LLMResponse,
    Message,
    SamplingParams,
    Usage,
    messages_to_text,
    short_error,
    validate_messages,
)


# Curated fallback models for the intelligent router
# Verified Oct 2026 against the live HF router catalog
# (https://router.huggingface.co/v1/models).  These are all present in
# the catalog; listed models can still 400 if the provider deploys them,
# so the router tries each in order.
# The live catalog (fetch_catalog) is the primary source; this list
# is only used if the catalog API is unreachable.
ROUTER_FALLBACK_MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-3-4b-it",
    "Qwen/Qwen3-4B-Instruct-2507",
    "openai/gpt-oss-20b",
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
        # Default is verified live on the HF router catalog (routers serve
        # it today); the heal path below recovers from a stale override,
        # but starting on a model that 404s wastes a round trip every boot.
        model: str = "meta-llama/Llama-3.1-8B-Instruct",
        base_url: str = SERVERLESS_URL,
        endpoint_url: str = "",
        # Bounded by the interactive reply budget (25s): a hung network
        # call must fail over to the next model/provider long before the
        # responder abandons the turn.  Warm router models answer in
        # seconds; only cold dedicated endpoints want longer — pass a
        # bigger timeout explicitly for those.
        timeout: float = 90.0,
        max_retries: int = 4,
        template: str = "chatml",
        wait_for_model: bool = True,
        use_chat_endpoint: bool = True,
        embed_model: str = "",
        vision_model: str = "",
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.token = token
        self.model = model
        # A dedicated vision repo (e.g. Qwen/Qwen2.5-VL-7B-Instruct): on the
        # legacy serverless API each model owns its /models/<repo>/ URL, so a
        # vision model means a different URL, not just a different payload.
        self.vision_model = vision_model
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
        # vision: serverless VLMs (Qwen-VL, Phi-vision) ride the same chat shape;
        # describe_image routes pixels to the configured vision model.
        return {"chat", "complete", "embed", "vision"}

    def fetch_catalog(self) -> list[dict[str, Any]]:
        """Fetch the live catalog of available models from the HF router API."""
        try:
            # Query the live models endpoint — this is the source of truth
            # for which models are actually hosted on hf-inference.
            url = "https://router.huggingface.co/v1/models"
            resp = self.http.get(url, timeout=15.0)
            if not resp.ok:
                # Fall back to curated list if API fails
                return [
                    {"id": m, "provider": "hf_serverless", "capabilities": ["chat", "complete"]}
                    for m in ROUTER_FALLBACK_MODELS
                ]
            data = resp.json()
            models = data.get("data", []) if isinstance(data, dict) else []
            # Normalize to the format _discover_catalog_model expects
            result = []
            for m in models:
                if isinstance(m, dict) and m.get("id"):
                    result.append({
                        "id": m["id"],
                        "providers": m.get("providers", []),
                    })
            if result:
                return result
            # Empty catalog, fall back to curated list
            return [
                {"id": m, "provider": "hf_serverless", "capabilities": ["chat", "complete"]}
                for m in ROUTER_FALLBACK_MODELS
            ]
        except Exception:
            # Network failure, fall back to curated list
            return [
                {"id": m, "provider": "hf_serverless", "capabilities": ["chat", "complete"]}
                for m in ROUTER_FALLBACK_MODELS
            ]

    @property
    def _is_dedicated_endpoint(self) -> bool:
        return bool(self.endpoint_url)

    def chat_url(self, model: str = "") -> str:
        if self._is_dedicated_endpoint:
            # Inference Endpoints expose a vLLM/TGI-compatible OpenAI surface.
            base = self.base_url
            return f"{base}/v1/chat/completions" if not base.endswith("/v1") else f"{base}/chat/completions"
        repo = model or self.model
        if self.use_chat_endpoint:
            return f"{self.base_url}/models/{repo}/v1/chat/completions"
        return f"{self.base_url}/models/{repo}"

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
        messages = validate_messages(messages, who="hf_serverless.chat")
        sampling = (params or SamplingParams()).clamped()
        payload = self.build_chat_payload(messages, sampling)

        # Try with current model first
        attempts: list[tuple[str, str]] = []  # (model, error) for the final report
        try:
            raw = self._post_checked(self.chat_url(), payload)
            return self._record(self._parse_chat(raw), started)
        except Exception as exc:  # noqa: BLE001
            # Don't heal on authentication errors (401/403) — swapping models
            # won't fix a bad or unauthorized key.  Status comes from the
            # HTTP layer's details (machine-readable), not message sniffing.
            if is_auth_error(exc):
                error = classify(exc)
                return self._record(
                    LLMResponse(text="", model=self.model, error=f"{error.code}: {error.message}"),
                    started,
                )
            attempts.append((self.model, short_error(exc)))

            # Model failed (not hosted, 404, 400, etc.) — discover candidates
            # and try each in turn, not just one.
            original_model = self.model
            candidates = self._candidate_models(exclude=original_model)
            last_error = exc
            for new_model in candidates[:3]:  # cap at 3 swaps per call
                self.model = new_model
                try:
                    raw = self._post_checked(self.chat_url(), payload)
                    result = self._parse_chat(raw)
                    if result.error:
                        last_error = Exception(result.error)
                        attempts.append((new_model, short_error(last_error)))
                        continue
                    # Success with healed model — record where we came from.
                    if result.raw is None:
                        result.raw = {}
                    if isinstance(result.raw, dict):
                        result.raw["healed_from"] = original_model
                    return self._record(result, started)
                except Exception as retry_exc:  # noqa: BLE001
                    last_error = retry_exc
                    attempts.append((new_model, short_error(retry_exc)))
                    continue

            # All candidates failed — restore original and return the whole
            # attempt chain, not just the last error.  The owner needs to see
            # that model A 404'd AND model B 429'd, not one opaque line.
            self.model = original_model
            error = classify(last_error)
            chain = "; ".join(f"{m} ({e})" for m, e in attempts)
            return self._record(
                LLMResponse(
                    text="",
                    model=self.model,
                    error=f"{error.code}: all HF models failed — {chain}",
                    raw={"heal_attempts": [{"model": m, "error": e} for m, e in attempts]},
                ),
                started,
            )

    def _candidate_models(self, exclude: str = "") -> list[str]:
        """Ordered list of alternative models to try, excluding the given one."""
        seen: set[str] = set()
        ordered: list[str] = []
        # 1. Live catalog, best-first
        try:
            catalog = self.fetch_catalog()
            # _discover_catalog_model returns a single best; we want several.
            # Collect scored candidates directly.
            scored = self._score_catalog_models(catalog, exclude=exclude)
            for model_id in scored:
                if model_id not in seen:
                    seen.add(model_id)
                    ordered.append(model_id)
        except Exception:  # noqa: BLE001 - fall through to curated list
            pass
        # 2. Curated fallback list
        for model_id in ROUTER_FALLBACK_MODELS:
            if model_id != exclude and model_id not in seen:
                seen.add(model_id)
                ordered.append(model_id)
        return ordered

    @staticmethod
    def _score_catalog_models(catalog: list[dict], exclude: str = "") -> list[str]:
        """Score and order catalog models best-first (shared with _discover_catalog_model)."""
        if not catalog:
            return []
        candidates: list[str] = []
        for model in catalog:
            model_id = model.get("id", "")
            if not model_id or model_id == exclude:
                continue
            providers = model.get("providers")
            if providers is not None:
                if not any(p.get("status") == "live" for p in providers):
                    continue
            candidates.append(model_id)
        if not candidates:
            return []
        uncensored_keywords = ["abliterated", "uncensored", "dolphin"]
        small_keywords = ["7B", "8B", "3B", "1.5B"]

        def _score(model_id: str) -> tuple[int, str]:
            lower = model_id.lower()
            score = 0
            for keyword in uncensored_keywords:
                if keyword in lower:
                    score += 10
                    break
            for keyword in small_keywords:
                if keyword in lower:
                    score += 5
                    break
            return (-score, model_id)

        return [m for _, m in sorted(_score(m) for m in candidates)]

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
        # on serverless, a configured vision model means its own /models/<repo>/
        # URL — the chat model cannot see, and sending pixels to it is a 404.
        model = self.vision_model or self.model
        payload = {
            "model": model,
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
                lambda: self.http.post_json(self.chat_url(model), payload), policy=self._policy
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


    def _post_checked(self, url: str, payload: dict, **kwargs) -> dict:
        """Post a request and check the response.
        
        Args:
            url: The URL to post to
            payload: The payload to send
            **kwargs: Additional arguments to pass to post_json
        
        Returns:
            The response data
        """
        response = self.http.post_json(url, payload, **kwargs)
        return response

    @staticmethod
    def _discover_catalog_model(catalog: list[dict], exclude: str = "") -> str:
        """Discover the best model from a catalog.

        Args:
            catalog: List of model dicts with 'id' and 'providers' keys
            exclude: Model ID to exclude from selection

        Returns:
            The selected model ID, or empty string if none found
        """
        scored = HFServerlessProvider._score_catalog_models(catalog, exclude=exclude)
        return scored[0] if scored else None



def prompt_from_messages(messages: Sequence[Message], template: str = "chatml") -> str:
    """Convenience wrapper around :func:`messages_to_text`."""
    return messages_to_text(messages, template=template)


def hf_doctor(provider: HFServerlessProvider) -> list[str]:
    """Diagnose HuggingFace provider issues."""
    report = []
    
    # Check token
    token = getattr(provider, "token", "") or ""
    if not token:
        report.append("1. token: MISSING (set NM_HF_TOKEN environment variable)")
    else:
        report.append(f"1. token: present ({len(token)} chars)")
    
    # Check URL
    url = getattr(provider, "url", "") or SERVERLESS_URL
    report.append(f"2. URL: {url}")
    
    # Check model
    model = getattr(provider, "model", "")
    report.append(f"3. model: {model or 'not set'}")
    
    # Try to fetch catalog
    try:
        catalog = provider.fetch_catalog()
        report.append(f"4. catalog: {len(catalog)} models available")
    except Exception as e:
        report.append(f"4. catalog: FAILED ({type(e).__name__}: {e})")
    
    # Try a simple request
    try:
        http = getattr(provider, "http", None)
        if http:
            response = http.post_json(f"{url}/models/{model or 'test'}", {})
            if response:
                report.append("5. API: responding")
            else:
                report.append("5. API: no response")
        else:
            report.append("5. API: no HTTP client")
    except Exception as e:
        report.append(f"5. API: FAILED ({type(e).__name__}: {e})")
    
    # Verdict
    if not token:
        report.append("\nVERDICT: NOT working (missing token)")
    else:
        report.append("\nVERDICT: appears functional")
    
    return report

