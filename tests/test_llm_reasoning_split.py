"""Reasoning/answer channel separation at extraction (2026-11-06).

The incident (owner's personal WhatsApp, 2026-11-05 23:32): a
reasoning rung's raw deliberation was sent AS THE OWNER to a real
contact. One half of the seam was here, at generation: every
provider extraction took the completion's content channel verbatim
— reasoning a provider inlined into that channel (think-tag spans)
became the response text whole, and nothing downstream ever asked
the draft to be the final answer rather than the scratch work.

The discipline pinned here: a completion's outbound text is its
final-answer channel ONLY, separated at extraction by the one
helper (``llm.base.split_reasoning``) every provider funnels
through. Dedicated reasoning fields (``reasoning_content``,
``thinking``) are never read into the text; a generation with no
answer channel yields empty text — no generation — never the
deliberation as a fallback body.
"""

from __future__ import annotations

import unittest
from typing import Any

from nomorals.llm.base import Message, SamplingParams
from nomorals.llm.providers.anthropic import AnthropicProvider
from nomorals.llm.providers.hf_serverless import HFServerlessProvider
from nomorals.llm.providers.ollama import OllamaProvider
from nomorals.llm.providers.openai_compat import OpenAICompatProvider


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


def _openai_payload(content: Any, **message_extra: Any) -> dict[str, Any]:
    message = {"role": "assistant", "content": content}
    message.update(message_extra)
    return {
        "id": "chatcmpl-1",
        "model": "reasoner-1",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 40,
                  "total_tokens": 50},
    }


def _compat(payload: Any) -> OpenAICompatProvider:
    provider = OpenAICompatProvider(
        base_url="http://localhost:9999/v1", api_key="x",
        model="reasoner-1")
    provider.http = FakeHttp(payload)
    return provider


def _split(text: str) -> tuple[str, str]:
    # Lazy: on the pristine tree the helper does not exist yet, and
    # the provider-extraction teeth below must fail on their
    # assertions (deliberation in the text), not on this import.
    from nomorals.llm.base import split_reasoning

    return split_reasoning(text)


class SplitReasoningTests(unittest.TestCase):
    def test_plain_text_is_unchanged(self) -> None:
        answer, reasoning = _split("just chilling")
        self.assertEqual(answer, "just chilling")
        self.assertEqual(reasoning, "")

    def test_think_span_is_separated(self) -> None:
        answer, reasoning = _split(
            "<think>the user said good morning, i should keep it "
            "short</think>morning")
        self.assertEqual(answer, "morning")
        self.assertIn("keep it short", reasoning)

    def test_think_only_yields_no_answer(self) -> None:
        answer, _ = _split(
            "<think>let me work through the turn exchange first")
        self.assertEqual(answer, "")

    def test_unclosed_think_is_all_reasoning(self) -> None:
        # Cut off mid-deliberation: nothing after the tag is answer.
        answer, _ = _split(
            "fine<think>wait, actually the user sent two messages")
        self.assertEqual(answer, "fine")
        answer, _ = _split(
            "<think>the user seems to be continuing the conversation")
        self.assertEqual(answer, "")

    def test_thinking_tag_variant_and_case(self) -> None:
        answer, _ = _split(
            "<THINKING>deliberation</THINKING>k")
        self.assertEqual(answer, "k")

    def test_answer_around_spans_is_kept(self) -> None:
        answer, _ = _split(
            "ok <think>meta</think> i dey come")
        self.assertEqual(answer, "ok  i dey come")


class OpenAICompatExtractionTests(unittest.TestCase):
    def test_inline_think_is_split_at_extraction(self) -> None:
        provider = _compat(_openai_payload(
            "<think>the user sent a greeting; a short reply fits"
            "</think>morning"))
        resp = provider.chat([Message.user("good morning")],
                             SamplingParams(max_tokens=64))
        self.assertEqual(resp.text, "morning")

    def test_reasoning_field_is_never_the_text(self) -> None:
        provider = _compat(_openai_payload(
            "fine",
            reasoning_content="the user seems to be continuing "
                              "the conversation at length"))
        resp = provider.chat([Message.user("ok")],
                             SamplingParams(max_tokens=64))
        self.assertEqual(resp.text, "fine")

    def test_reasoning_only_completion_is_no_generation(self) -> None:
        provider = _compat(_openai_payload(
            "<think>the user seems to be continuing the conversation, "
            "but the message is just a greeting followed by"))
        resp = provider.chat([Message.user("ok")],
                             SamplingParams(max_tokens=64))
        self.assertEqual(resp.text, "")


class OllamaExtractionTests(unittest.TestCase):
    def _provider(self, payload: dict[str, Any]) -> OllamaProvider:
        provider = OllamaProvider(model="reasoner-1")
        provider._post = lambda *a, **k: payload  # type: ignore[method-assign]
        return provider

    def test_thinking_field_ignored_inline_think_split(self) -> None:
        provider = self._provider({
            "model": "reasoner-1", "done": True,
            "message": {
                "role": "assistant",
                "thinking": "the user is continuing the conversation",
                "content": "<think>draft the reply</think>just chilling",
            },
            "prompt_eval_count": 5, "eval_count": 9,
        })
        resp = provider.chat([Message.user("wetin u dey do")])
        self.assertEqual(resp.text, "just chilling")


class HFExtractionTests(unittest.TestCase):
    def _provider(self) -> HFServerlessProvider:
        return HFServerlessProvider(token="x", model="reasoner-1")

    def test_chat_shape_think_split(self) -> None:
        resp = self._provider()._parse_chat({
            "model": "reasoner-1",
            "choices": [{
                "message": {"role": "assistant",
                            "content": "<think>deliberate</think>k"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4,
                      "total_tokens": 7},
        })
        self.assertEqual(resp.text, "k")

    def test_text_generation_shape_think_split(self) -> None:
        # The list shape reaches extraction via complete()'s
        # _parse_text_generation, not _parse_chat.
        text = HFServerlessProvider._parse_text_generation(
            [{"generated_text": "<think>deliberate</think>no wahala"}])
        self.assertEqual(text, "no wahala")


class AnthropicExtractionPinTests(unittest.TestCase):
    def test_thinking_blocks_never_reach_text(self) -> None:
        # Pin (passes before and after): Anthropic extraction reads
        # text blocks only, so the thinking channel is structurally
        # excluded on this rung — the discipline the other rungs
        # now match via split_reasoning.
        provider = AnthropicProvider(api_key="x", model="claude-x")
        provider.http = FakeHttp({
            "id": "msg_1", "type": "message", "model": "claude-x",
            "content": [
                {"type": "thinking",
                 "thinking": "the user seems to be continuing"},
                {"type": "text", "text": "morning"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 9, "output_tokens": 12},
        })
        resp = provider.chat([Message.user("good morning")],
                             SamplingParams(max_tokens=64))
        self.assertEqual(resp.text, "morning")


if __name__ == "__main__":
    unittest.main()
