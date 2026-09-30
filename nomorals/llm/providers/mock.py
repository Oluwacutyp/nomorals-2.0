"""Deterministic offline provider.

This is not a stub — it is the reason the entire system is testable and the
reason it still works on a phone with no network. It:

* is deterministic for a given input and seed, so tests are reproducible;
* emits well-formed tool-call JSON when the prompt asks for a tool, which lets the
  agent loop, the DAG scheduler, and the tool registry be exercised end to end;
* simulates latency and failure rates, so retry and circuit-breaker behaviour can
  be tested without flaky real endpoints;
* produces real embeddings (hashing-based), so memory retrieval is exercised for
  real rather than mocked away.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from typing import Any, Sequence

from ...core.text import approx_token_count, word_frequencies
from ..base import LLMProvider, LLMResponse, Message, SamplingParams, Usage


class MockProvider(LLMProvider):
    """A predictable model for offline operation and testing."""

    name = "mock"

    def __init__(
        self,
        *,
        model: str = "mock-7b",
        latency_ms: float = 0.0,
        failure_rate: float = 0.0,
        seed: int | None = None,
        dimensions: int = 256,
        timeout: float = 120.0,
        max_retries: int = 3,
        scripted: dict[str, str] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, max_retries=max_retries)
        self.model = model
        self.latency_ms = latency_ms
        self.failure_rate = failure_rate
        self.seed = seed
        self.dimensions = dimensions
        self.scripted = dict(scripted or {})
        self.calls: list[list[Message]] = []

    @property
    def model_id(self) -> str:
        return self.model

    @property
    def capabilities(self) -> set[str]:
        # "vision" is deliberately absent: the mock's "[mock vision]" blurb is a
        # test fixture, not a real VLM description. Router vision failover must
        # never settle for it (tests/test_vision2.py).
        return {"chat", "complete", "embed"}

    def health(self) -> bool:
        return True

    # ── generation ───────────────────────────────────────────────────────────
    def chat(
        self, messages: Sequence[Message], params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        self.calls.append(list(messages))
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000.0)
        if self.failure_rate and self._rng(messages).random() < self.failure_rate:
            return self._record(
                LLMResponse(text="", model=self.model, error="simulated provider failure"),
                started,
            )

        prompt = messages[-1].content if messages else ""
        params = (params or SamplingParams()).clamped()
        text = self._respond(prompt, messages, params)
        text = self._apply_limits(text, params)

        response = LLMResponse(
            text=text,
            model=self.model,
            usage=Usage(
                prompt_tokens=sum(approx_token_count(m.content) for m in messages),
                completion_tokens=approx_token_count(text),
            ),
            finish_reason="stop",
        )
        response.usage.total_tokens = response.usage.prompt_tokens + response.usage.completion_tokens
        return self._record(response, started)

    def embed(self, texts: Sequence[str], **kw: Any) -> list[list[float]]:
        return [self._hash_embedding(text) for text in texts]

    def describe_image(
        self, image: bytes, prompt: str = "", params: SamplingParams | None = None, **kw: Any
    ) -> LLMResponse:
        started = time.perf_counter()
        digest = hashlib.sha256(image).hexdigest()
        width, height = _sniff_dimensions(image)
        text = (
            f"[mock vision] image sha256={digest[:16]} size={width}x{height} "
            f"bytes={len(image)}; "
            + (
                "the image contains structured visual content."
                if len(image) > 1024
                else "the image is a very small placeholder."
            )
        )
        if prompt:
            text = f"{text} (in response to: {prompt[:120]})"
        return self._record(
            LLMResponse(
                text=text,
                model=self.model,
                usage=Usage(prompt_tokens=len(image) // 1000, completion_tokens=approx_token_count(text)),
            ),
            started,
        )

    # ── internals ────────────────────────────────────────────────────────────
    def _rng(self, messages: Sequence[Message]) -> random.Random:
        material = "|".join(m.content for m in messages)
        digest = int(hashlib.sha256(material.encode()).hexdigest()[:8], 16)
        return random.Random(self.seed if self.seed is not None else digest)

    def _respond(self, prompt: str, messages: Sequence[Message], params: SamplingParams) -> str:
        for needle, reply in self.scripted.items():
            if needle in prompt:
                return reply

        lowered = prompt.lower()

        # Tool-call requests: return well-formed JSON so the agent loop can be
        # exercised without a real model.
        tool_match = re.search(r"available tools?:\s*([a-z0-9_,\s\-]+)", lowered)
        if "use the tool" in lowered or "call the tool" in lowered or "tool call" in lowered:
            tool_name = "web_search"
            if tool_match:
                candidates = [c.strip() for c in tool_match.group(1).split(",") if c.strip()]
                if candidates:
                    tool_name = candidates[0]
            query = _extract_topic(prompt)
            return json.dumps(
                {
                    "thought": "I need external information to answer this.",
                    "tool": tool_name,
                    "arguments": {"query": query},
                }
            )

        if "json" in lowered and ("schema" in lowered or "object" in lowered):
            return json.dumps(
                {"answer": _extract_topic(prompt), "confidence": 0.72, "needs_more": False}
            )

        if lowered.startswith(("plan", "decompose", "break down", "steps")):
            topic = _extract_topic(prompt)
            return "\n".join(
                f"{i}. {verb} {topic}"
                for i, verb in enumerate(
                    ["Research", "Gather sources for", "Analyse", "Draft output on", "Verify"], 1
                )
            )

        if lowered.startswith(("summar", "summaris", "tl;dr")):
            prior = next(
                (m.content for m in reversed(messages[:-1]) if m.role == "user"), prompt
            )
            counts = word_frequencies(prior)
            keywords = ", ".join(word for word, _ in counts.most_common(5)) or "the topic"
            return f"Summary: the discussion covers {keywords}."

        if lowered.startswith(("reflect", "evaluate", "critique", "score")):
            return (
                "Evaluation: the approach covered the request but skipped verification. "
                "Score: 0.7. Lesson: add an explicit check step before reporting completion."
            )

        if "?" in prompt or lowered.startswith(("what", "why", "how", "who", "when", "explain")):
            topic = _extract_topic(prompt)
            return (
                f"{topic.capitalize() if topic else 'That'} can be approached in three parts: "
                "define the constraint, gather the relevant evidence, then choose the option "
                "that survives the evidence. I can work through each part if useful."
            )

        topic = _extract_topic(prompt) or "the request"
        return f"Acknowledged. Working on {topic}."

    def _apply_limits(self, text: str, params: SamplingParams) -> str:
        words = text.split()
        if len(words) > params.max_tokens:
            return " ".join(words[: params.max_tokens])
        for stop in params.stop:
            if stop and stop in text:
                text = text.split(stop, 1)[0]
        return text

    def _hash_embedding(self, text: str) -> list[float]:
        """Deterministic feature-hashing embedding.

        Not semantic — but stable, dependency-free, and enough to exercise the
        retrieval path end to end. Overlapping vocabulary produces genuinely
        higher similarity, which is all the vector store needs to be testable.
        """
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            vector[0] = 1.0
            return vector
        for token in tokens:
            digest = int(hashlib.blake2b(token.encode(), digest_size=8).digest().hex(), 16)
            index = digest % self.dimensions
            sign = 1.0 if (digest >> 63) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


def _extract_topic(prompt: str) -> str:
    """Pull the salient phrase out of a prompt, for building a plausible reply."""
    cleaned = re.sub(r"^\s*(?:please\s+)?(?:can you\s+)?(?:tell me\s+)?", "", prompt.strip(), flags=re.I)
    cleaned = re.sub(r"[^\w\s,.:-]", " ", cleaned).strip()
    words = [w for w in cleaned.split() if len(w) > 1]
    stop = {"the", "and", "for", "with", "what", "how", "why", "you", "are", "can", "please", "about"}
    kept = [w for w in words if w.lower() not in stop][:8]
    return " ".join(kept) or " ".join(words[:8])


def _sniff_dimensions(data: bytes) -> tuple[int, int]:
    """Best-effort width/height from image headers, no PIL required."""
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")
    if data.startswith(b"GIF8") and len(data) >= 10:
        return int.from_bytes(data[6:8], "little"), int.from_bytes(data[8:10], "little")
    if data.startswith(b"BM") and len(data) >= 26:
        return int.from_bytes(data[18:22], "little"), abs(int.from_bytes(data[22:26], "little", signed=True))
    if data.startswith(b"\xff\xd8"):
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in {0xC0, 0xC1, 0xC2, 0xC3}:
                return int.from_bytes(data[i + 7 : i + 9], "big"), int.from_bytes(data[i + 5 : i + 7], "big")
            length = int.from_bytes(data[i + 2 : i + 4], "big")
            i += 2 + max(2, length)
    return (0, 0)
