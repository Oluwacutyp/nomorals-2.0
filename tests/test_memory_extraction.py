"""Memory v2: extraction pipeline, tags/origin, dedupe, /remember /recall /forget.

Covers the contract the pipeline promises:

* heuristic pass classifies facts/preferences/decisions/relationship and
  ignores questions, small talk, and opinions
* a near-duplicate turn does NOT create a second memory (reinforces instead)
* non-owner speakers are attributed in group chats
* the opt-in LLM pass parses a JSON array and fails closed on garbage
* remember() stores tags + origin; recall() filters by tag; find_one/forget
  resolve free text; the /remember /recall /forget commands work end-to-end
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.core.config import Settings
from nomorals.llm.base import LLMResponse, Message
from nomorals.memory.base import MemoryKind
from nomorals.memory.extract import (
    ExtractedMemory,
    MemoryExtractor,
    _heuristic_pass,
    extract_sentences,
    is_duplicate,
)
from nomorals.memory.manager import join_tags
from nomorals.social.chat.control import parse_control


def _make_context(test=None) -> Any:
    tmp = tempfile.mkdtemp(prefix="nm-memx-")
    if test is not None:
        test.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
    settings = Settings(home=tmp)
    ctx = build_context(settings)
    ctx.__enter__()
    return ctx


def _stub_runtime(memory: Any) -> Any:
    """A PartnerRuntime with just enough surface for the memory commands."""
    from types import SimpleNamespace

    from nomorals.agents.partner_runtime import PartnerRuntime

    class ChatRef:
        platform = "telegram"
        key = "tg:console"
        name = "test"

    rt = PartnerRuntime.__new__(PartnerRuntime)
    sent: list[str] = []
    rt._ref_from_key = lambda k: ChatRef()
    rt._send_long = lambda p, c, text: sent.append(text)
    rt._sent = sent
    rt.context = SimpleNamespace(memory=memory)
    return rt


# ── heuristics ───────────────────────────────────────────────────────────────


class HeuristicTests(unittest.TestCase):
    def test_facts_about_himself(self) -> None:
        for text in [
            "hey, i live in lagos and i work as a backend dev",
            "i am from nigeria originally",
            "i have two dogs, one is called rex",
            "i bought a new phone last week",
            "i just got the new job as a dev at a fintech in lagos",
            "your name is oluwaseun right",
        ]:
            got = _heuristic_pass(text)
            self.assertTrue(got, f"expected a fact in {text!r}")
            self.assertEqual(got[0].kind, MemoryKind.FACT)

    def test_preferences(self) -> None:
        got = _heuristic_pass("i'd rather you keep it short, dont call me after 11pm")
        self.assertTrue(got)
        self.assertEqual(got[0].kind, MemoryKind.PREFERENCE)

    def test_decisions(self) -> None:
        got = _heuristic_pass("let's use the q8 gguf for the phone build")
        self.assertTrue(got)
        self.assertEqual(got[0].kind, MemoryKind.DECISION)

    def test_relationship(self) -> None:
        got = _heuristic_pass("you're my best friend, i'm glad to have you here")
        self.assertTrue(got)
        self.assertEqual(got[0].kind, MemoryKind.RELATIONSHIP)

    def test_noise_is_ignored(self) -> None:
        for text in ["haha nice", "do i live in lagos?", "i think you should get some sleep"]:
            self.assertEqual(_heuristic_pass(text), [], text)

    def test_sentence_splitting(self) -> None:
        parts = extract_sentences("one thing. two things! third?\nnewline four")
        self.assertEqual(len(parts), 4)


class DedupeTests(unittest.TestCase):
    def test_identical_normalized(self) -> None:
        self.assertTrue(is_duplicate("I prefer short messages!", "i prefer short messages.", 0.8))

    def test_containment(self) -> None:
        self.assertTrue(
            is_duplicate("i prefer short messages", "i prefer short messages, dont call me at night", 0.8))

    def test_different_things(self) -> None:
        self.assertFalse(is_duplicate("i live in lagos", "i work as a backend dev", 0.8))


# ── end-to-end with a real context ───────────────────────────────────────────


class ExtractionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _make_context(self)
        self.memory = self.context.memory

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def test_turn_stores_a_fact(self) -> None:
        actions = MemoryExtractor(self.context).extract_turn(
            "hey, i live in lagos and i work as a backend dev",
            chat_key="tg:123",
        )
        stored = [a for a in actions if a["action"] == "stored"]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["kind"], MemoryKind.FACT)
        record = self.memory.get(stored[0]["id"])
        self.assertIsNotNone(record)
        self.assertIn("lagos", record.tags)
        self.assertEqual(record.origin, "chat:tg:123")

    def test_repeated_turn_does_not_duplicate(self) -> None:
        ex = MemoryExtractor(self.context)
        text = "i prefer short messages, dont call me after 11pm"
        first = ex.extract_turn(text, chat_key="tg:1")
        second = ex.extract_turn(text, chat_key="tg:1")
        self.assertEqual([a["action"] for a in first], ["stored"])
        self.assertEqual([a["action"] for a in second], ["duplicate"])
        self.assertEqual(self.memory.repo.count(), 1)

    def test_small_talk_stores_nothing(self) -> None:
        actions = MemoryExtractor(self.context).extract_turn("haha nice, how are you")
        self.assertTrue(all(a["action"] != "stored" for a in actions))
        self.assertEqual(self.memory.repo.count(), 0)

    def test_group_speaker_is_attributed(self) -> None:
        actions = MemoryExtractor(self.context).extract_turn(
            "i moved to abuja in june",
            chat_key="tg:group42",
            speaker="seun",
            is_owner=False,
        )
        stored = [a for a in actions if a["action"] == "stored"]
        self.assertEqual(len(stored), 1)
        self.assertTrue(stored[0]["content"].startswith("seun: "))
        record = self.memory.get(stored[0]["id"])
        self.assertEqual(record.origin, "chat:tg:group42")

    def test_disabled_by_settings_stores_nothing(self) -> None:
        self.context.settings.memory.extract_enabled = False
        actions = MemoryExtractor(self.context).extract_turn("i live in lagos, ok?")
        self.assertEqual(actions, [])

    def test_llm_pass_parses_json_array(self) -> None:
        self.context.settings.memory.extract_llm = True
        llm_json = (
            '[{"kind": "fact", "content": "his gym is open on sundays", '
            '"importance": 0.7}, {"kind": "preference", '
            '"content": "he wants the reports at 9am", "importance": 0.8}]'
        )
        self.context.router = _FakeRouter(llm_json)
        actions = MemoryExtractor(self.context).extract_turn(
            "also, his gym is open on sundays btw")
        self.assertEqual(sum(1 for a in actions if a["action"] == "stored"), 2)

    def test_llm_pass_fails_closed_on_garbage(self) -> None:
        self.context.settings.memory.extract_llm = True
        self.context.router = _FakeRouter("no json here, just words")
        actions = MemoryExtractor(self.context).extract_turn(
            "i have a dog called rex, and we should pick friday for the call")
        # heuristic still stores its match (one sentence → one memory);
        # the LLM garbage adds nothing
        stored = [a for a in actions if a["action"] == "stored"]
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["kind"], MemoryKind.DECISION)


class _FakeRouter:
    def __init__(self, text: str) -> None:
        self._text = text

    def chat(self, messages, params=None, **kw):
        self.last_messages = list(messages)
        return LLMResponse(text=self._text)


# ── manager v2: tags / origin / find_one / forget ────────────────────────────


class ManagerV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = _make_context(self)
        self.memory = self.context.memory

    def tearDown(self) -> None:
        self.context.__exit__(None, None, None)

    def test_join_tags_normalizes(self) -> None:
        self.assertEqual(join_tags(["Fitness", "coffee", "COFFEE", " fitness"]), "fitness,coffee")
        self.assertEqual(join_tags(""), "")
        self.assertEqual(join_tags(None), "")

    def test_remember_with_tags_and_origin(self) -> None:
        rid = self.memory.remember(
            "he likes his coffee black", kind="preference",
            tags=["coffee"], origin="chat:tg:7",
        )
        record = self.memory.get(rid)
        self.assertEqual(record.tags, "coffee")
        self.assertEqual(record.origin, "chat:tg:7")

    def test_recall_tag_filter(self) -> None:
        self.memory.remember("he takes his coffee black", kind="preference", tags="coffee")
        self.memory.remember("he runs 5km in the morning", kind="preference", tags="fitness")
        coffee = self.memory.recall("coffee", limit=5, tags="coffee")
        self.assertTrue(coffee.records)
        for r in coffee.records:
            self.assertIn("coffee", r.tags.split(","))

    def test_find_one_and_forget(self) -> None:
        rid = self.memory.remember("the server password is in the vault", kind="fact")
        found = self.memory.find_one("server password vault")
        self.assertIsNotNone(found)
        self.assertEqual(found.id, rid)
        self.assertEqual(self.memory.forget(rid), 1)
        self.assertIsNone(self.memory.get(rid))

    def test_forget_by_text_through_control(self) -> None:
        rt = _stub_runtime(self.memory)
        out = rt._control_remember("my sister's name is funmi", chat_key="tg:1")
        self.assertIn("remembered", out)
        self.assertIn("funmi", out)
        out = rt._control_recall("funmi sister")
        self.assertIn("funmi", out)
        out = rt._control_forget("funmi sister name")
        self.assertIn("forgotten", out)
        self.assertEqual(self.memory.repo.count(), 0)


# ── control commands ─────────────────────────────────────────────────────────


class ControlCommandTests(unittest.TestCase):
    def test_parse_remember_recall_forget(self) -> None:
        c = parse_control("/remember i live in lagos fact")
        self.assertEqual(c.kind, "remember")
        self.assertEqual(c.arg, "i")
        self.assertIn("lagos", c.tail)

        r = parse_control("/recall coffee")
        self.assertEqual(r.kind, "recall")

        f = parse_control("/forget my sister's name")
        self.assertEqual(f.kind, "forget")

    def test_remember_with_kind_and_tags(self) -> None:
        self.context = _make_context(self)
        self.memory = self.context.memory
        rt = _stub_runtime(self.memory)
        out = rt._control_remember("he likes espresso, not coffee latte preference tags:coffee,espresso",
                                   chat_key="tg:9")
        self.assertIn("[preference]", out)
        # the kind token is stripped from the content
        self.assertNotIn(" preference", out.split("]")[1])
        record = next(r for r in self.memory._recent(3) if r.content.startswith("he likes espresso"))
        self.assertEqual(record.kind, "preference")
        self.assertIn("coffee", record.tags)
        self.assertIn("espresso", record.tags)

    def test_remember_usage(self) -> None:
        self.context = _make_context(self)
        rt = _stub_runtime(self.context.memory)
        self.assertIn("usage", rt._control_remember(""))

    def test_recall_empty_memory(self) -> None:
        self.context = _make_context(self)
        rt = _stub_runtime(self.context.memory)
        out = rt._control_recall("anything at all")
        self.assertIn("nothing in memory", out)

    def test_forget_unknown_id(self) -> None:
        self.context = _make_context(self)
        rt = _stub_runtime(self.context.memory)
        out = rt._control_forget("deadbeef0000")
        self.assertIn("no memory with id", out)


if __name__ == "__main__":
    unittest.main()
