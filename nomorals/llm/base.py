"""Provider protocol and shared message types.

Every backend — Hugging Face, an OpenAI-compatible server, llama.cpp, or the
offline mock — implements the same four methods. That uniformity is what makes
hot-swapping a model a one-line operation and what lets the whole test suite run
with no network at all.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

from ..core.errors import ModelError

__all__ = [
    "CHAT_TEMPLATES",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "Role",
    "SamplingParams",
    "Usage",
    "detect_template",
    "messages_to_text",
    "short_error",
    "validate_messages",
]

Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class Message:
    """One turn in a conversation."""

    role: str
    content: str
    name: str = ""
    tool_call_id: str = ""
    images: list[str] = field(default_factory=list)

    def to_openai(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name:
            payload["name"] = self.name
        if self.tool_call_id:
            payload["tool_call_id"] = self.tool_call_id
        return payload

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str, **kw: Any) -> "Message":
        return cls(role="user", content=content, **kw)

    @classmethod
    def assistant(cls, content: str, **kw: Any) -> "Message":
        return cls(role="assistant", content=content, **kw)

    @classmethod
    def tool(cls, content: str, tool_call_id: str = "", **kw: Any) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, **kw)


def messages_to_text(
    messages: Sequence[Message],
    *,
    template: str = "chatml",
    model: str = "",
) -> str:
    """Render messages to a single prompt string.

    Needed for completion-only endpoints and for any model served without a chat
    template.  ``template="auto"`` (or ``template=""``) detects the model's
    native family from ``model`` via :func:`detect_template` — hand-rolling one
    format for every family costs 10–30% benchmark accuracy on the wrong
    family (ChatML tokens on a Mistral model, …), so the native template is
    the right default when the model id is known.  ChatML stays the default
    because it is what Dolphin-family models are trained on.
    """
    name = (template or "").strip().lower()
    if name in ("auto", ""):
        name = detect_template(model) if model else "chatml"
    if name == "chatml":
        parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>" for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "\n".join(parts)
    if name == "llama3":
        parts = []
        for m in messages:
            tag = {"system": "system", "user": "user", "assistant": "assistant"}.get(m.role, m.role)
            parts.append(f"<|start_header_id|>{tag}<|end_header_id|>\n\n{m.content}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)
    if name == "llama2":
        # [INST] <<SYS>>…<</SYS>> … [/INST]
        out: list[str] = []
        system = next((m.content for m in messages if m.role == "system"), "")
        turns = [m for m in messages if m.role != "system"]
        sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
        i = 0
        while i < len(turns):
            user_turns = []
            while i < len(turns) and turns[i].role == "user":
                user_turns.append(turns[i].content)
                i += 1
            prompt_part = " ".join(user_turns)
            assistant_part = ""
            if i < len(turns) and turns[i].role == "assistant":
                assistant_part = " " + turns[i].content
                i += 1
            prefix = "<s>[INST] " if not out else "[INST] "
            out.append(f"{prefix}{sys_block if not out else ''}{prompt_part} [/INST]{assistant_part} </s>")
            sys_block = ""
        if not out:
            return f"<s>[INST] {sys_block} [/INST] "
        # generation prompt when the last turn was a user turn
        if turns and turns[-1].role != "assistant":
            out.append("[INST] ")
        return "".join(out)
    if name == "mistral":
        out = []
        for m in messages:
            if m.role == "system":
                out.append(f"<s>[INST] {m.content} [/INST]")
            elif m.role == "user":
                out.append(f"[INST] {m.content} [/INST]")
            elif m.role == "assistant":
                out.append(f" {m.content}</s>")
            else:
                out.append(f"[INST] {m.content} [/INST]")
        text = "".join(out)
        if messages and messages[-1].role != "assistant":
            text += " "
        return text
    if name == "gemma":
        parts = []
        for m in messages:
            tag = "model" if m.role == "assistant" else m.role
            if tag not in ("system", "user", "model"):
                tag = "user"
            parts.append(f"<start_of_turn>{tag}\n{m.content}<end_of_turn>\n")
        parts.append("<start_of_turn>model\n")
        return "".join(parts)
    if name == "qwen":
        # Qwen2/2.5/3 ChatML-family with generation prompt
        parts = [f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n" for m in messages]
        parts.append("<|im_start|>assistant\n")
        return "".join(parts)
    if name == "deepseek":
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(m.content)
            elif m.role == "user":
                parts.append(f"### Instruction:\n{m.content}\n")
            elif m.role == "assistant":
                parts.append(f"### Response:\n{m.content}\n")
            else:
                parts.append(f"### {m.role}:\n{m.content}\n")
        parts.append("### Response:\n")
        return "\n".join(parts)
    if name == "phi3":
        parts = []
        for m in messages:
            parts.append(f"<|{m.role}|>\n{m.content}<|end|>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)
    if name == "vicuna":
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(m.content)
            elif m.role == "user":
                parts.append(f"USER: {m.content}")
            elif m.role == "assistant":
                parts.append(f"ASSISTANT: {m.content}</s>")
            else:
                parts.append(f"{m.role.upper()}: {m.content}")
        parts.append("ASSISTANT:")
        return "\n".join(parts)
    if name == "zephyr":
        parts = []
        for m in messages:
            parts.append(f"<|{m.role}|>\n{m.content}</s>\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)
    if name == "openchat":
        parts = []
        for m in messages:
            if m.role == "system":
                parts.append(f"{m.content}<|end_of_turn|>")
            else:
                parts.append(f"GPT4 Correct {m.role.capitalize()}: {m.content}<|end_of_turn|>")
        parts.append("GPT4 Correct Assistant:")
        return "".join(parts)
    if name == "alpaca":
        system = next((m.content for m in messages if m.role == "system"), "")
        turns = [m for m in messages if m.role != "system"]
        body = "\n".join(f"### {m.role.capitalize()}:\n{m.content}" for m in turns)
        header = f"{system}\n\n" if system else ""
        return f"{header}{body}\n### Assistant:\n"
    # unknown template name → plain fallback (previous default branch)
    return "\n".join(f"{m.role}: {m.content}" for m in messages) + "\nassistant:"


#: Model-id fragments → native chat template family.  Checked in order;
#: first hit wins.  (HF convention: never hand-roll — the template must
#: match what the model was trained on.)
_TEMPLATE_FAMILY_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("llama3", ("llama-3", "llama3", "llama-3.1", "llama-3.2", "llama-3.3")),
    ("qwen", ("qwen",)),
    ("mistral", ("mistral", "mixtral")),
    ("gemma", ("gemma",)),
    ("deepseek", ("deepseek",)),
    ("phi3", ("phi-3", "phi3", "phi-4", "phi4")),
    ("vicuna", ("vicuna",)),
    ("zephyr", ("zephyr",)),
    ("openchat", ("openchat",)),
    ("llama2", ("llama-2", "llama2", "codellama")),
    ("chatml", ("dolphin", "chatml", "codebeast", "hermes")),
)


def detect_template(model_id: str) -> str:
    """Native chat-template family for a model id (``"chatml"`` fallback).

    Fragment match, first hit wins — mirrors how ``apply_chat_template``
    picks the model's own Jinja template from its tokenizer config.
    """
    lowered = (model_id or "").lower()
    for family, hints in _TEMPLATE_FAMILY_HINTS:
        if any(h in lowered for h in hints):
            return family
    return "chatml"


#: All template names :func:`messages_to_text` understands.
CHAT_TEMPLATES: tuple[str, ...] = (
    "chatml", "llama3", "llama2", "mistral", "gemma", "qwen",
    "deepseek", "phi3", "vicuna", "zephyr", "openchat", "alpaca", "auto",
)


def validate_messages(messages: Sequence[Message], *, who: str = "chat") -> list[Message]:
    """Fail-fast validation for provider ``chat()`` inputs.

    Every provider's ``chat()`` must call this first so callers get a clear,
    immediate error instead of the cryptic ``AttributeError`` from
    ``m.to_openai()`` or the confusing upstream rejection a
    ``"messages": []`` payload produces.

    Returns the messages as a list. Raises:

    * :class:`TypeError` when ``messages`` is not a sequence of
      :class:`Message` (a bare string, ``None``, a dict list, …);
    * :class:`ValueError` when the sequence is empty — an empty
      conversation is meaningless to send to a model.
    """
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise TypeError(
            f"{who}: messages must be a sequence of Message, "
            f"got {type(messages).__name__}"
        )
    items = list(messages)
    if not items:
        raise ValueError(f"{who}: messages must not be empty")
    bad = next((m for m in items if not isinstance(m, Message)), None)
    if bad is not None:
        raise TypeError(
            f"{who}: every message must be a Message, "
            f"got {type(bad).__name__}"
        )
    return items


@dataclass
class SamplingParams:
    """Decoding configuration."""

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    max_tokens: int = 1024
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    stop: tuple[str, ...] = ()
    seed: int | None = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    json_mode: bool = False

    def clamped(self) -> "SamplingParams":
        """Return a copy with values inside the range every backend accepts."""
        return SamplingParams(
            temperature=max(0.0, min(2.0, self.temperature)),
            top_p=max(0.01, min(1.0, self.top_p)),
            top_k=max(0, self.top_k),
            max_tokens=max(1, min(128_000, self.max_tokens)),
            min_p=max(0.0, min(1.0, self.min_p)),
            repetition_penalty=max(0.5, min(2.0, self.repetition_penalty)),
            stop=self.stop,
            seed=self.seed,
            frequency_penalty=max(-2.0, min(2.0, self.frequency_penalty)),
            presence_penalty=max(-2.0, min(2.0, self.presence_penalty)),
            json_mode=self.json_mode,
        )

    def to_openai(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.frequency_penalty:
            params["frequency_penalty"] = self.frequency_penalty
        if self.presence_penalty:
            params["presence_penalty"] = self.presence_penalty
        if self.stop:
            params["stop"] = list(self.stop)
        if self.seed is not None:
            params["seed"] = self.seed
        if self.json_mode:
            params["response_format"] = {"type": "json_object"}
        return params


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    #: Provider-reported cache-read input tokens (billed ~0.1x).
    cached_tokens: int = 0
    #: Provider-reported reasoning tokens (billed, never shown to the user).
    reasoning_tokens: int = 0

    @property
    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class LLMResponse:
    """A completed generation."""

    text: str
    model: str = ""
    provider: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = "stop"
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    #: True when a provider failed and the router served this response from
    #: a fallback instead — the text alone never hides a degradation.
    degraded: bool = False
    #: Names of the providers that failed before the serving one, in order.
    failed_providers: list[str] = field(default_factory=list)
    #: Human-readable degradation line, e.g.
    #: ``"hf_serverless failed (ModelError: 503); served by groq"``.
    fallback_note: str = ""
    #: Typed failure class for a failed response
    #: (``nomorals.llm.failures.FailureClass`` value, e.g. "rate_limited").
    #: Empty on success.  Lets callers recover the RIGHT way — shrink the
    #: context on "context_overflow" instead of blindly retrying.
    failure_class: str = ""
    #: Per-attempt routing trace, filled by the router: one dict per
    #: provider attempt — {"provider", "ok", "latency_ms", "error"} — in
    #: attempt order.  Langfuse-style observability without a service.
    route_trace: list[dict[str, Any]] = field(default_factory=list)
    #: Metered USD cost of this response (estimate_cost).  0.0 = unknown/free.
    cost_usd: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage.as_dict,
            "finish_reason": self.finish_reason,
            "latency_ms": round(self.latency_ms, 2),
            "error": self.error,
            "degraded": self.degraded,
            "failed_providers": list(self.failed_providers),
            "fallback_note": self.fallback_note,
            "failure_class": self.failure_class,
            "route_trace": [dict(t) for t in self.route_trace],
            "cost_usd": round(self.cost_usd, 9),
        }


class LLMProvider(abc.ABC):
    """The interface every backend implements."""

    name: str = "provider"

    def __init__(self, *, timeout: float = 120.0, max_retries: int = 3) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.stats = {
            "calls": 0,
            "errors": 0,
            "latency_ms": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }

    # ── core API ─────────────────────────────────────────────────────────────
    @abc.abstractmethod
    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        """Generate a reply to a conversation.

        ``messages`` must be a non-empty sequence of :class:`Message` —
        concrete providers validate via :func:`validate_messages` and raise
        :class:`TypeError` / :class:`ValueError` otherwise.
        """

    def complete(self, prompt: str, params: SamplingParams | None = None, **kw: Any) -> LLMResponse:
        """Raw completion. Defaults to a single-user-turn chat."""
        return self.chat([Message.user(prompt)], params, **kw)

    def embed(self, texts: Sequence[str], **kw: Any) -> list[list[float]]:
        """Embed texts. Not every provider supports this."""
        raise ModelError(f"provider {self.name!r} does not support embeddings")

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        """Vision. Not every provider supports this."""
        raise ModelError(f"provider {self.name!r} does not support vision")

    # ── introspection ────────────────────────────────────────────────────────
    def health(self) -> bool:
        """Cheap liveness probe. Defaults to optimistic."""
        return True

    @property
    def capabilities(self) -> set[str]:
        return {"chat"}

    @property
    def model_id(self) -> str:
        return self.name

    def stats_snapshot(self) -> dict[str, Any]:
        calls = self.stats["calls"] or 1
        return {
            **self.stats,
            "avg_latency_ms": round(self.stats["latency_ms"] / calls, 2),
            "error_rate": round(self.stats["errors"] / calls, 4),
        }

    def _record(self, response: LLMResponse, started: float) -> LLMResponse:
        elapsed = (time.perf_counter() - started) * 1000
        self.stats["calls"] += 1
        self.stats["latency_ms"] += elapsed
        self.stats["prompt_tokens"] += response.usage.prompt_tokens
        self.stats["completion_tokens"] += response.usage.completion_tokens
        if response.error:
            self.stats["errors"] += 1
        response.latency_ms = elapsed
        response.provider = self.name
        return response


def short_error(exc: BaseException, limit: int = 120) -> str:
    """One-line error for attempt chains (truncated, no newlines)."""
    text = str(exc).replace("\n", " ").strip() or type(exc).__name__
    return text[:limit] + "…" if len(text) > limit else text


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for budget accounting before a call."""
    from ..core.text import approx_token_count

    return approx_token_count(text)


def estimate_messages(messages: Iterable[Message]) -> int:
    return sum(estimate_tokens(m.content) + 4 for m in messages)
