"""Local llama.cpp server provider.

``llama-server`` speaks the OpenAI protocol, so this is a thin specialization of
:class:`OpenAICompatProvider` that adds the llama.cpp-specific knobs (grammar
constraints, slot control) and a health check against ``/health``.

This is the path for running a quantized personal fine-tune on your own hardware
with no network involvement at all.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from ...core.errors import ModelError, classify
from ...core.retry import retry_call
from ..base import LLMResponse, Message, SamplingParams
from .openai_compat import OpenAICompatProvider


class LlamaCppProvider(OpenAICompatProvider):
    """llama.cpp ``llama-server`` backend."""

    name = "llama_cpp"

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        api_key: str = "",
        model: str = "local",
        timeout: float = 300.0,
        max_retries: int = 2,
        grammar: str = "",
        n_slots: int = 1,
        **kw: Any,
    ) -> None:
        # llama-server mounts the OpenAI surface under /v1.
        url = base_url.rstrip("/")
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        super().__init__(
            base_url=url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            **kw,
        )
        self.grammar = grammar
        self.n_slots = n_slots
        self.raw_base = base_url.rstrip("/")
        # llama-server always runs on the user's own machine (loopback).
        # The SSRF guard would block 127.0.0.1 — allow it here only.
        # This is the user's own box talking to itself, not an SSRF attack.
        self.http.allow_private_ips = True

    @property
    def capabilities(self) -> set[str]:
        return {"chat", "complete", "embed", "grammar"}

    def health(self) -> bool:
        try:
            return self.http.get(f"{self.raw_base}/health", timeout=5.0).ok
        except Exception:  # noqa: BLE001 - liveness probe
            return False

    def slots(self) -> list[dict[str, Any]]:
        """Inspect llama.cpp slot state — useful for capacity planning."""
        try:
            return self.http.get(f"{self.raw_base}/slots", timeout=10.0).json()
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"could not read slot state: {classify(exc).message}") from exc

    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        sampling = params or SamplingParams()
        if self.grammar:
            # GBNF grammars constrain the output to a language — this is how you
            # force structurally valid JSON out of a small local model.
            kw.setdefault("grammar", self.grammar)
        if sampling.json_mode and not self.grammar:
            kw.setdefault("response_format", {"type": "json_object"})
        return super().chat(messages, sampling, **kw)

    def tokenize(self, text: str) -> list[int]:
        """Use the server's own tokenizer — exact counts, no approximation."""
        try:
            raw = retry_call(
                lambda: self.http.post_json(f"{self.raw_base}/tokenize", {"content": text}),
                policy=self._policy,
            )
            return list(raw.json().get("tokens") or [])
        except Exception as exc:  # noqa: BLE001
            raise ModelError(f"tokenization failed: {classify(exc).message}") from exc

    def context_size(self) -> int:
        try:
            props = self.http.get(f"{self.raw_base}/props", timeout=10.0).json()
            return int(props.get("default_generation_settings", {}).get("n_ctx", 0))
        except Exception:  # noqa: BLE001
            return 0


def json_grammar(schema_hint: str = "object") -> str:
    """A small GBNF grammar that constrains output to a JSON object.

    Handy with 7B-class local models that otherwise drift out of JSON mid-stream.
    """
    return r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array  ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" ws
number ::= "-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws
ws     ::= [ \t\n]?
"""
