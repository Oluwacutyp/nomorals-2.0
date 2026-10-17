"""Model-id healing: a truncated configured id must resolve to a valid id.

Regression: the owner saw Groq fail on ``openai/gpt-oss-120`` (missing
trailing ``"b"``) and OpenRouter fail on ``qwen/qwen3-8`` (missing ``"b"``
and ``":free"``) — "no model is answering".  No code stripped the ``"b"``;
the truncated names arrived via configuration, and two gaps turned a
one-character typo into a total outage:

1. nothing healed a near-miss id onto the known-good roster id, so the
   truncated id was sent to the API on every call;
2. ``is_not_found_error`` only recognised HTTP 404, but Groq answers a
   dead model id with 400 + ``model_not_found``, so the
   ``_fallback_models`` healing never triggered either.

These tests pin the fix: the resolved model sent to Groq is
``openai/gpt-oss-120b`` and to OpenRouter ``qwen/qwen3-8b:free``.
"""

from __future__ import annotations

import pytest

from nomorals.core.errors import ProviderError, is_not_found_error
from nomorals.llm.providers.groq import GroqProvider
from nomorals.llm.providers.openai_compat import (
    OpenAICompatProvider,
    heal_model_id,
)
from nomorals.llm.providers.openrouter import (
    OPENROUTER_FREE_ALIAS,
    OpenRouterProvider,
)


class TestHealModelId:
    def test_missing_trailing_b(self):
        assert (
            heal_model_id("openai/gpt-oss-120", {"openai/gpt-oss-120b"})
            == "openai/gpt-oss-120b"
        )

    def test_missing_free_suffix(self):
        assert (
            heal_model_id("qwen/qwen3-8b", {"qwen/qwen3-8b:free"})
            == "qwen/qwen3-8b:free"
        )

    def test_missing_both(self):
        assert (
            heal_model_id("qwen/qwen3-8", {"qwen/qwen3-8b:free"})
            == "qwen/qwen3-8b:free"
        )

    def test_exact_id_untouched(self):
        assert (
            heal_model_id("openai/gpt-oss-120b", {"openai/gpt-oss-120b"})
            == "openai/gpt-oss-120b"
        )

    def test_unrelated_id_untouched(self):
        # A deliberately different model is never second-guessed.
        assert (
            heal_model_id("someone/else-7b", {"openai/gpt-oss-120b"})
            == "someone/else-7b"
        )

    def test_ambiguous_prefix_not_healed(self):
        # "openai/gpt-oss-1" + "b" is not a known id — no guess.
        assert (
            heal_model_id("openai/gpt-oss-1", {"openai/gpt-oss-120b"})
            == "openai/gpt-oss-1"
        )


class TestGroqProviderHealing:
    def test_truncated_id_resolves_to_120b(self):
        assert GroqProvider(model="openai/gpt-oss-120").model == "openai/gpt-oss-120b"

    def test_truncated_20b_resolves(self):
        assert GroqProvider(model="openai/gpt-oss-20").model == "openai/gpt-oss-20b"

    def test_correct_id_untouched(self):
        assert (
            GroqProvider(model="openai/gpt-oss-120b").model == "openai/gpt-oss-120b"
        )

    def test_empty_model_falls_back_to_default(self):
        assert GroqProvider(model="").model == "openai/gpt-oss-120b"


class TestOpenRouterProviderHealing:
    def test_truncated_id_resolves_to_qwen_free(self):
        assert (
            OpenRouterProvider(model="qwen/qwen3-8").model == "qwen/qwen3-8b:free"
        )

    def test_missing_b_before_free_resolves(self):
        assert (
            OpenRouterProvider(model="qwen/qwen3-8:free").model
            == "qwen/qwen3-8b:free"
        )

    def test_missing_free_suffix_resolves(self):
        assert (
            OpenRouterProvider(model="qwen/qwen3-8b").model == "qwen/qwen3-8b:free"
        )

    def test_alias_untouched(self):
        assert OpenRouterProvider(model="openrouter/free").model == "openrouter/free"

    def test_empty_model_falls_back_to_alias(self):
        assert OpenRouterProvider(model="").model == OPENROUTER_FREE_ALIAS


class TestIsNotFoundError:
    def _err(self, status: int, message: str) -> ProviderError:
        return ProviderError(message, details={"http_status": status},
                             retryable=False)

    def test_404_still_not_found(self):
        assert is_not_found_error(self._err(404, "404 not found")) is True

    def test_groq_400_model_not_found(self):
        # Groq answers a dead model id with 400 + model_not_found.
        assert is_not_found_error(
            self._err(400, "400 The model `openai/gpt-oss-120` does not exist "
                           "(code: model_not_found)")
        ) is True

    def test_model_does_not_exist_message(self):
        assert is_not_found_error(
            self._err(400, "The model `foo` does not exist")
        ) is True

    def test_openrouter_no_endpoints(self):
        assert is_not_found_error(
            self._err(404, "No endpoints found for qwen/qwen3-8")
        ) is True

    def test_generic_400_is_not_not_found(self):
        assert is_not_found_error(
            self._err(400, "Bad request: unsupported sampling parameter")
        ) is False

    def test_auth_error_never_not_found(self):
        assert is_not_found_error(
            self._err(401, "401 unauthorized: model not found in key scope")
        ) is False


class TestFailoverOnGroqStyle400:
    """A 400 model_not_found must trigger the _fallback_models chain."""

    def test_chat_heals_onto_fallback_after_400_model_not_found(self):
        class FakeResp:
            def json(self):
                return {
                    "choices": [
                        {"message": {"content": "hi"}, "finish_reason": "stop"}
                    ],
                    "model": "good-model",
                }

        calls: list[str] = []

        class FakeHttp:
            def post_json(self, url, payload):
                calls.append(payload["model"])
                if payload["model"] == "dead-model":
                    raise ProviderError(
                        "400 The model `dead-model` does not exist "
                        "(code: model_not_found)",
                        details={"http_status": 400},
                        retryable=False,
                    )
                return FakeResp()

        class HealingProvider(OpenAICompatProvider):
            name = "healing"

            def _fallback_models(self):
                return ["good-model"]

        provider = HealingProvider(model="dead-model", max_retries=1)
        provider.http = FakeHttp()  # type: ignore[assignment]
        from nomorals.llm.base import Message

        resp = provider.chat([Message(role="user", content="hi")])
        assert resp.text == "hi"
        assert calls == ["dead-model", "good-model"]
