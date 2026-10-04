"""Native Ollama provider.

Ollama exposes two wire formats: a native REST API under ``/api`` and an
OpenAI-compatible shim under ``/v1``.  The generic
:class:`~nomorals.llm.providers.openai_compat.OpenAICompatProvider` already
covers the shim; this module talks to the native API because it exposes
capabilities the shim hides:

* model management — ``/api/tags`` (list), ``/api/pull`` (download),
  ``/api/delete`` (remove), ``/api/show`` (capabilities/details),
  ``/api/ps`` (what is loaded right now);
* per-request ``options`` (``num_predict``, ``keep_alive``, …) and exact
  token counters in every non-streaming response;
* vision through the native ``images`` message field (base64, no data-URL
  wrapper needed).

Fully local and free: no account, no key, no network beyond the machine.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any, Callable, Sequence

from ...core.errors import ModelError, ProviderError, classify
from ...core.http import HttpClient
from ...core.retry import BackoffPolicy, retry_call
from ..base import LLMProvider, LLMResponse, Message, SamplingParams, Usage, validate_messages

__all__ = ["OllamaProvider", "OLLAMA_DEFAULT_HOST"]


OLLAMA_DEFAULT_HOST = "http://localhost:11434"

#: SamplingParams → Ollama ``options`` key mapping.
_OPTIONS_MAP = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "num_predict",
    "min_p": "min_p",
    "repetition_penalty": "repeat_penalty",
    "frequency_penalty": "frequency_penalty",
    "presence_penalty": "presence_penalty",
    "seed": "seed",
}


def _options(params: SamplingParams) -> dict[str, Any]:
    """Translate clamped sampling params to Ollama's ``options`` object."""
    sampling = params.clamped()
    out: dict[str, Any] = {}
    for attr, key in _OPTIONS_MAP.items():
        value = getattr(sampling, attr)
        out[key] = value
    if sampling.stop:
        out["stop"] = list(sampling.stop)
    return out


class OllamaProvider(LLMProvider):
    """Talks to Ollama's native API (``/api/*``), plus model management."""

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str = OLLAMA_DEFAULT_HOST,
        model: str = "",
        keep_alive: str = "5m",
        options: dict[str, Any] | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        api_key: str = "",  # accepted for shared-wiring compat; Ollama needs no key
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        # Tolerate a shim URL (http://host:11434/v1) handed over by the shared
        # provider wiring: the native API lives one level up.
        base = (base_url or OLLAMA_DEFAULT_HOST).rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self.base_url = base or OLLAMA_DEFAULT_HOST
        self.model = model or os.environ.get("OLLAMA_MODEL", "")
        self.keep_alive = keep_alive
        self.options = dict(options or {})
        self.http = HttpClient(timeout=timeout, headers=dict(headers or {}))
        self._policy = BackoffPolicy(max_attempts=max(1, max_retries), cap=30.0)

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def model_id(self) -> str:
        return self.model or "ollama"

    @property
    def capabilities(self) -> set[str]:
        # Vision is model-dependent (llava / llama3.2-vision serve it, a pure
        # chat model rejects images) — same "advertise, surface real errors"
        # convention as OpenAICompatProvider.
        return {"chat", "complete", "embed", "vision"}

    def health(self) -> bool:
        try:
            response = self.http.get(f"{self.base_url}/api/version",
                                     timeout=min(10.0, self.timeout))
            return response.ok
        except Exception:  # noqa: BLE001 - liveness probe
            return False

    # ── inference ────────────────────────────────────────────────────────────
    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        messages = validate_messages(messages, who="ollama.chat")
        payload = self._chat_payload(messages, params, **kw)
        data = self._post("/api/chat", payload, operation="chat")
        message = data.get("message") or {}
        return self._record(
            LLMResponse(
                text=message.get("content") or "",
                model=data.get("model", self.model),
                usage=Usage(
                    prompt_tokens=int(data.get("prompt_eval_count") or 0),
                    completion_tokens=int(data.get("eval_count") or 0),
                ),
                finish_reason="stop" if data.get("done") else "length",
                raw=data,
            ),
            started,
        )

    def complete(self, prompt: str, params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {**_options(params or SamplingParams()), **kw.pop("options", {})},
            **kw,
        }
        data = self._post("/api/generate", payload, operation="complete")
        return self._record(
            LLMResponse(
                text=data.get("response") or "",
                model=data.get("model", self.model),
                usage=Usage(
                    prompt_tokens=int(data.get("prompt_eval_count") or 0),
                    completion_tokens=int(data.get("eval_count") or 0),
                ),
                finish_reason="stop" if data.get("done") else "length",
                raw=data,
            ),
            started,
        )

    def embed(self, texts: Sequence[str], *, model: str = "", **kw: Any) -> list[list[float]]:
        payload = {"model": model or self.model, "input": list(texts)}
        data = self._post("/api/embed", payload, operation="embed")
        vectors = data.get("embeddings") or []
        if len(vectors) != len(texts):
            raise ModelError(
                f"ollama returned {len(vectors)} embeddings for {len(texts)} inputs"
            )
        return [list(v) for v in vectors]

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        """Vision through the native ``images`` message field (raw base64)."""
        started = time.perf_counter()
        encoded = base64.b64encode(image).decode("ascii")
        message = Message(
            role="user",
            content=prompt or "Describe this image in detail.",
            images=[encoded],
        )
        payload = self._chat_payload([message], params, **kw)
        data = self._post("/api/chat", payload, operation="vision")
        content = (data.get("message") or {}).get("content") or ""
        return self._record(
            LLMResponse(text=content, model=data.get("model", self.model), raw=data),
            started,
        )

    # ── model management ─────────────────────────────────────────────────────
    def list_models(self) -> list[dict[str, Any]]:
        """Every model installed locally (``ollama list`` over the API)."""
        data = self._get("/api/tags")
        return list(data.get("models") or [])

    def model_names(self) -> list[str]:
        return [m.get("name", "") for m in self.list_models() if m.get("name")]

    def has_model(self, name: str) -> bool:
        wanted = name.split(":")[0]
        return any(n == name or n.split(":")[0] == wanted for n in self.model_names())

    def show(self, name: str = "") -> dict[str, Any]:
        """Modelfile, parameters, template, and capabilities of one model."""
        return self._post("/api/show", {"name": name or self.model}, operation="show")

    def model_capabilities(self, name: str = "") -> list[str]:
        """What the model itself reports (e.g. ``["completion"]`` or
        ``["vision", "completion"]``).  Empty when the model is unknown."""
        try:
            return list(self.show(name).get("capabilities") or [])
        except ProviderError:
            return []

    def running_models(self) -> list[dict[str, Any]]:
        """Models currently loaded in memory (``ollama ps`` over the API)."""
        data = self._get("/api/ps")
        return list(data.get("models") or [])

    def pull(
        self,
        name: str,
        *,
        insecure: bool = False,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Download a model from the Ollama library (``ollama pull``).

        Uses a non-streaming request, so very large models take a while but the
        call either returns ``{"status": "success"}`` or raises.  ``progress``
        is accepted for API symmetry and currently unused with stream=False.
        """
        payload: dict[str, Any] = {"name": name, "stream": False}
        if insecure:
            payload["insecure"] = True
        return self._post("/api/pull", payload, operation="pull", timeout=self.timeout)

    def delete_model(self, name: str) -> bool:
        """Remove a local model (``ollama rm``). Returns True when gone."""
        try:
            raw = self.http.request("DELETE", f"{self.base_url}/api/delete",
                                    json={"name": name}, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001 - surface as a clean provider error
            error = classify(exc)
            raise ProviderError(f"ollama delete failed: {error.message}",
                                retryable=error.retryable) from exc
        return raw.ok

    def unload(self, name: str = "") -> None:
        """Unload a model from memory (``ollama stop``): a generate call with
        ``keep_alive: 0`` and an empty prompt unloads it."""
        self._post("/api/generate",
                   {"model": name or self.model, "prompt": "", "keep_alive": 0,
                    "stream": False},
                   operation="unload")

    # ── internals ────────────────────────────────────────────────────────────
    def _chat_payload(
        self, messages: Sequence[Message], params: SamplingParams | None, **kw: Any
    ) -> dict[str, Any]:
        sampling = (params or SamplingParams()).clamped()
        wire: list[dict[str, Any]] = []
        for message in messages:
            entry: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.images:
                entry["images"] = list(message.images)
            wire.append(entry)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": wire,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {**_options(sampling), **kw.pop("options", {})},
        }
        if sampling.json_mode:
            payload["format"] = "json"
        payload.update(kw)
        return payload

    def _post(self, path: str, payload: dict[str, Any], *,
              operation: str, timeout: float | None = None) -> dict[str, Any]:
        try:
            raw = retry_call(
                lambda: self.http.post_json(f"{self.base_url}{path}", payload,
                                           timeout=timeout or self.timeout),
                policy=self._policy,
            )
        except Exception as exc:  # noqa: BLE001 - normalise to provider errors
            error = classify(exc)
            raise ProviderError(f"ollama {operation} failed: {error.message}",
                                retryable=error.retryable) from exc
        data = raw.json()
        if isinstance(data, dict) and data.get("error"):
            raise ProviderError(f"ollama {operation} failed: {data['error']}")
        if not isinstance(data, dict):
            raise ProviderError(f"ollama {operation} returned a non-JSON body")
        return data

    def _get(self, path: str) -> dict[str, Any]:
        try:
            raw = retry_call(
                lambda: self.http.get(f"{self.base_url}{path}", timeout=self.timeout),
                policy=self._policy,
            )
        except Exception as exc:  # noqa: BLE001 - normalise to provider errors
            error = classify(exc)
            raise ProviderError(f"ollama request failed: {error.message}",
                                retryable=error.retryable) from exc
        data = raw.json()
        if not isinstance(data, dict):
            raise ProviderError("ollama returned a non-JSON body")
        return data
