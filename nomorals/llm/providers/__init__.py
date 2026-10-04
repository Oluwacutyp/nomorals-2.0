"""Provider backends."""

from __future__ import annotations

from typing import Any

from ..base import LLMProvider


def build_provider(kind: str, **kwargs: Any) -> LLMProvider:
    """Instantiate a provider by name.

    Imports are deferred so an unused backend never costs an import, and a missing
    optional dependency in one backend cannot break the others.

    Raises :class:`ValueError` for a missing/blank kind or an unknown one —
    never an ``AttributeError`` off ``None``.
    """
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError(
            f"provider kind must be a non-empty string, got {kind!r}"
        )
    kind = kind.strip().lower()
    if kind in {"mock", "offline", "test"}:
        from .mock import MockProvider

        return MockProvider(**kwargs)
    if kind in {"ollama", "ollama_native"}:
        from .ollama import OllamaProvider

        return OllamaProvider(**kwargs)
    if kind == "groq":
        from .groq import GroqProvider

        return GroqProvider(**kwargs)
    if kind == "openrouter":
        from .openrouter import OpenRouterProvider

        return OpenRouterProvider(**kwargs)
    if kind in {"openai", "openai_compat", "vllm", "lmstudio", "ollama_compat"}:
        from .openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(**kwargs)
    if kind in {"hf", "hf_serverless", "huggingface", "hf_endpoint"}:
        from .hf_serverless import HFServerlessProvider

        return HFServerlessProvider(**kwargs)
    if kind in {"llama_cpp", "llamacpp", "gguf"}:
        from .llama_cpp import LlamaCppProvider

        return LlamaCppProvider(**kwargs)
    if kind in {"ocr", "tesseract"}:
        from .ocr import OCRProvider

        return OCRProvider(**kwargs)
    raise ValueError(
        "unknown provider "
        f"{kind!r}; expected mock | ollama | groq | openrouter | openai_compat | "
        "hf_serverless | llama_cpp | ocr"
    )


__all__ = ["LLMProvider", "build_provider"]
