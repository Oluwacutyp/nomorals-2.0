"""New LLM provider tests: anthropic (native), deepseek, gemini.

HTTP is fully mocked — no network. Covers: factory registration,
Anthropic's wire format (system as top-level param, role alternation,
vision blocks, usage mapping, error surfacing), the OpenAI-compat
subclasses' pinned base URLs / env keys, and the defaults chain order
(groq stays last).
"""

from __future__ import annotations

import os
import unittest
from typing import Any
from unittest import mock

from nomorals.llm.base import Message, SamplingParams
from nomorals.llm.providers import build_provider
from nomorals.llm.providers.anthropic import (
    ANTHROPIC_API_URL,
    AnthropicProvider,
)
from nomorals.llm.providers.deepseek import DEEPSEEK_BASE_URL, DeepSeekProvider
from nomorals.llm.providers.gemini import GEMINI_BASE_URL, GeminiProvider


class FakeResponse:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


class FakeHttp:
    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.calls: list[tuple[str, Any]] = []

    def post_json(self, url: str, payload: Any, **kw: Any) -> FakeResponse:
        self.calls.append((url, payload))
        return FakeResponse(self.payload)


_ANTHROPIC_OK = {
    "id": "msg_1",
    "type": "message",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "hello there"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 5},
}


def _anthropic(payload: Any = None) -> AnthropicProvider:
    p = AnthropicProvider(api_key="sk-ant-test", model="claude-sonnet-4-6")
    p.http = FakeHttp(payload if payload is not None else _ANTHROPIC_OK)
    return p


class FactoryTest(unittest.TestCase):
    def test_kinds(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "x"}):
            self.assertIsInstance(build_provider("anthropic"),
                                  AnthropicProvider)
            self.assertIsInstance(build_provider("claude"), AnthropicProvider)
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "x"}):
            self.assertIsInstance(build_provider("deepseek"),
                                  DeepSeekProvider)
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "x"}):
            self.assertIsInstance(build_provider("gemini"), GeminiProvider)

    def test_unknown_still_rejected(self):
        with self.assertRaises(ValueError):
            build_provider("definitely_not_a_provider")


class DeepSeekGeminiTest(unittest.TestCase):
    def test_pinned_endpoints(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "d"}):
            p = DeepSeekProvider()
            self.assertEqual(p.base_url, DEEPSEEK_BASE_URL)
            self.assertEqual(p.name, "deepseek")
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "g"}):
            p = GeminiProvider()
            self.assertEqual(p.base_url, GEMINI_BASE_URL.rstrip("/"))
            self.assertEqual(p.name, "gemini")

    def test_no_embed_on_deepseek(self):
        with mock.patch.dict(os.environ, {"DEEPSEEK_API_KEY": "d"}):
            self.assertNotIn("embed", DeepSeekProvider().capabilities)


class AnthropicWireTest(unittest.TestCase):
    def test_system_is_top_level(self):
        p = _anthropic()
        resp = p.chat([Message.system("you are devon"),
                       Message.user("hi")],
                      SamplingParams(max_tokens=64))
        self.assertEqual(resp.text, "hello there")
        url, payload = p.http.calls[0]
        self.assertEqual(url, ANTHROPIC_API_URL)
        self.assertEqual(payload["system"], "you are devon")
        self.assertEqual(payload["max_tokens"], 64)
        self.assertNotIn("system", [m["role"] for m in payload["messages"]])

    def test_role_alternation_merges_runs(self):
        p = _anthropic()
        p.chat([Message.user("a"), Message.user("b"),
                Message.assistant("c"), Message.user("d")])
        _url, payload = p.http.calls[0]
        roles = [m["role"] for m in payload["messages"]]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertEqual(len(payload["messages"][0]["content"]), 2)

    def test_usage_mapped(self):
        p = _anthropic()
        resp = p.chat([Message.user("hi")])
        self.assertEqual(resp.usage.prompt_tokens, 12)
        self.assertEqual(resp.usage.completion_tokens, 5)
        self.assertEqual(resp.usage.total_tokens, 17)
        self.assertEqual(resp.finish_reason, "end_turn")

    def test_api_error_surfaced(self):
        p = _anthropic({"type": "error",
                        "error": {"type": "authentication_error",
                                  "message": "bad key"}})
        resp = p.chat([Message.user("hi")])
        self.assertIn("authentication_error", resp.error)
        self.assertEqual(resp.text, "")

    def test_vision_block(self):
        p = _anthropic()
        uri = ("data:image/png;base64,"
               "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
               "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
        p.chat([Message.user("what is this?", images=[uri])])
        _url, payload = p.http.calls[0]
        blocks = payload["messages"][0]["content"]
        images = [b for b in blocks if b["type"] == "image"]
        self.assertEqual(len(images), 1)
        self.assertEqual(images[0]["source"]["media_type"], "image/png")
        self.assertEqual(images[0]["source"]["type"], "base64")

    def test_headers(self):
        p = AnthropicProvider(api_key="sk-ant-test")
        self.assertEqual(p.http.headers["x-api-key"], "sk-ant-test")
        self.assertEqual(p.http.headers["anthropic-version"], "2023-06-01")

    def test_no_key_fails_fast(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(Exception):
                AnthropicProvider()


class DefaultsChainTest(unittest.TestCase):
    def test_new_providers_in_chain_before_groq(self):
        from nomorals.llm.defaults import specs_from_env
        env = {"GROQ_API_KEY": "g", "HF_TOKEN": "h",
               "OPENROUTER_API_KEY": "o", "ANTHROPIC_API_KEY": "a",
               "DEEPSEEK_API_KEY": "d", "GEMINI_API_KEY": "gm"}
        with mock.patch.dict(os.environ, env, clear=True):
            names = [s.name for s in specs_from_env()]
        for new in ("anthropic", "deepseek", "gemini"):
            self.assertIn(new, names)
            self.assertLess(names.index(new), names.index("groq"))
        self.assertEqual(names[-1], "groq")

    def test_absent_keys_absent_specs(self):
        from nomorals.llm.defaults import specs_from_env
        with mock.patch.dict(os.environ, {}, clear=True):
            names = [s.name for s in specs_from_env()]
        for new in ("anthropic", "deepseek", "gemini"):
            self.assertNotIn(new, names)


if __name__ == "__main__":
    unittest.main()
