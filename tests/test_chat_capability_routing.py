"""Chat must never fall back to the OCR provider (nomorals.llm.router).

Regression: with Groq + OpenRouter down, a chat request fell through to
the OCR provider and failed with "the ocr provider reads images; it does
not chat".  The router filtered embed() and describe_image() by
capability but chat()/complete() took the unfiltered chain, and OCR
advertised "vision" — so it looked like a candidate for everything.

These tests pin: chat/complete only select providers with that
capability; vision reads accept vision OR ocr (the OCR floor still
serves describe_image); an all-down chat fleet fails honestly naming
the down chat providers, never OCR.
"""

from __future__ import annotations

import unittest

from nomorals.core.errors import ModelError, ProviderError
from nomorals.llm.base import LLMProvider, LLMResponse, Message
from nomorals.llm.capabilities import Capability, ModelCard
from nomorals.llm.providers.mock import MockProvider
from nomorals.llm.providers.ocr import OCRProvider
from nomorals.llm.router import LLMRouter


class _BoomChat(MockProvider):
    """A chat-capable provider whose chat() always blows up (API down)."""

    def chat(self, messages, params=None, **kw):
        raise RuntimeError(f"{self.name} probe failed: connection refused")


class _StubOCR(LLMProvider):
    """Stand-in for the real OCR provider (no tesseract binary needed)."""

    name = "ocr"

    def __init__(self) -> None:
        super().__init__()
        self.chat_calls = 0
        self.complete_calls = 0
        self.image_calls = 0

    @property
    def capabilities(self) -> set[str]:
        return {"ocr"}

    def chat(self, messages, params=None, **kw):
        self.chat_calls += 1
        raise ProviderError("the ocr provider reads images; it does not chat")

    def complete(self, prompt, params=None, **kw):
        self.complete_calls += 1
        raise ProviderError("the ocr provider reads images; it does not chat")

    def describe_image(self, image, prompt="", params=None, **kw):
        self.image_calls += 1
        return LLMResponse(text="HELLO", model="tesseract", provider="ocr")


def _router(**kw):
    return LLMRouter(cooldown_seconds=60.0, failure_threshold=2, **kw)


class ChatCapabilityRoutingTests(unittest.TestCase):
    def test_chat_never_selects_ocr_when_chat_providers_down(self):
        r = _router()
        r.add(_BoomChat(model="gpt-oss-120b"), name="groq", primary=True)
        r.add(_BoomChat(model="qwen3-8b"), name="openrouter")
        ocr = _StubOCR()
        r.add(ocr, name="ocr")
        resp = r.chat([Message.user("hi")])
        self.assertFalse(resp.ok)
        # OCR was never attempted: the error must not blame it.
        self.assertEqual(ocr.chat_calls, 0)
        self.assertEqual(resp.failed_providers, ["groq", "openrouter"])
        self.assertNotIn("ocr", (resp.error or "").lower())
        self.assertNotIn("ocr", (resp.fallback_note or "").lower())
        self.assertNotIn("does not chat", resp.error or "")

    def test_chat_with_only_ocr_raises_no_chat_provider(self):
        r = _router()
        r.add(_StubOCR(), name="ocr")
        with self.assertRaises(ModelError) as ctx:
            r.chat([Message.user("hi")])
        self.assertIn("no registered provider supports chat", str(ctx.exception))
        self.assertIn("ocr(ocr)", str(ctx.exception))

    def test_complete_never_selects_ocr(self):
        r = _router()
        r.add(_StubOCR(), name="ocr")
        with self.assertRaises(ModelError) as ctx:
            r.complete("hi")
        self.assertIn("no registered provider supports complete", str(ctx.exception))
        self.assertEqual(r.get("ocr").complete_calls, 0)

    def test_cooling_down_chat_error_names_down_providers(self):
        r = _router()
        r.add(_BoomChat(model="gpt-oss-120b"), name="groq", primary=True)
        r.add(_BoomChat(model="qwen3-8b"), name="openrouter")
        r.add(_StubOCR(), name="ocr")
        r.chat([Message.user("hi")])  # failure 1 each
        r.chat([Message.user("hi")])  # failure 2 each -> cooling down
        resp = r.chat([Message.user("hi")])  # nothing attempted
        self.assertFalse(resp.ok)
        self.assertIn("no chat provider available", resp.error or "")
        self.assertIn("groq (cooling down)", resp.error or "")
        self.assertIn("openrouter (cooling down)", resp.error or "")
        self.assertIn("cooling down", resp.error or "")
        self.assertNotIn("ocr", (resp.error or "").lower())
        self.assertEqual(r.get("ocr").chat_calls, 0)

    def test_vision_still_routes_to_ocr(self):
        r = _router()
        ocr = _StubOCR()
        r.add(ocr, name="ocr")
        resp = r.describe_image(b"\x89PNG fake bytes", "what does it say?")
        self.assertTrue(resp.ok)
        self.assertEqual(resp.text, "HELLO")
        self.assertEqual(ocr.image_calls, 1)

    def test_real_ocr_provider_advertises_ocr_only(self):
        self.assertEqual(OCRProvider().capabilities, {"ocr"})

    def test_broker_card_for_ocr_is_ocr_not_vision(self):
        card = ModelCard.from_provider(OCRProvider())
        self.assertEqual(card.capabilities, {Capability.OCR})


if __name__ == "__main__":
    unittest.main()
