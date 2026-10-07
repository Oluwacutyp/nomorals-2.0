"""Tests for source-grounded research mode (no network)."""

import unittest

from nomorals.research.grounded import (
    GroundedSession, GroundedAnswer, GroundedError, _chunk,
)


def _llm_citing(prompt):
    # model cites with [S1] label as instructed
    assert "[S1]" in prompt
    return "The capital is Lagos [S1]. It is the largest city [S1]."


def _llm_cannot(prompt):
    return "CANNOT_ANSWER"


def _llm_invents(prompt):
    return "The capital is Abuja [S99]. Totally real [MadeUp]."


def _llm_no_citations(prompt):
    return ("Lagos is the capital and it has millions of people living "
            "there with a vibrant economy and culture.")


class ChunkTests(unittest.TestCase):
    def test_basic(self):
        chunks = _chunk("Hello world. " * 200, size=200, overlap=20)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(c for c in chunks))

    def test_empty(self):
        self.assertEqual([], _chunk(""))
        self.assertEqual([], _chunk("   "))


class SessionTests(unittest.TestCase):
    def _session(self):
        s = GroundedSession()
        s.add_text(
            "Lagos is the capital of Nigeria. Lagos is the largest city "
            "in Africa with over 15 million people. The city is known for "
            "its vibrant markets and tech startups.",
            title="nigeria doc",
        )
        return s

    def test_ingest(self):
        s = self._session()
        self.assertEqual(1, len(s.doc_ids))

    def test_ingest_empty_raises(self):
        s = GroundedSession()
        with self.assertRaises(GroundedError):
            s.add_text("   ")

    def test_ask_with_citations(self):
        s = self._session()
        ans = s.ask("What is the capital?", llm_fn=_llm_citing)
        self.assertFalse(ans.refused)
        # model emitted [S1] -> code numbered it [1]
        self.assertIn("[1]", ans.text)
        self.assertNotIn("[S1]", ans.text)
        self.assertEqual(1, len(ans.sources))
        rendered = ans.render()
        self.assertIn("Sources:", rendered)
        self.assertIn("[1]", rendered)

    def test_ask_refuses_when_model_cannot(self):
        s = self._session()
        ans = s.ask("What is the GDP of Mars?", llm_fn=_llm_cannot)
        self.assertTrue(ans.refused)
        self.assertIn("can't answer", ans.text)

    def test_ask_refuses_on_no_hits(self):
        s = self._session()
        ans = s.ask("zxqwkj asdfzxcv qwerty", llm_fn=_llm_citing)
        self.assertTrue(ans.refused)

    def test_invented_citations_stripped(self):
        s = self._session()
        ans = s.ask("capital?", llm_fn=_llm_invents)
        self.assertNotIn("[S99]", ans.text)
        self.assertNotIn("[MadeUp]", ans.text)

    def test_uncited_claims_refused(self):
        s = self._session()
        ans = s.ask("tell me everything", llm_fn=_llm_no_citations)
        self.assertTrue(ans.refused)

    def test_empty_question_raises(self):
        s = self._session()
        with self.assertRaises(GroundedError):
            s.ask("")

    def test_no_docs_raises(self):
        s = GroundedSession()
        with self.assertRaises(GroundedError):
            s.ask("anything?", llm_fn=_llm_citing)


if __name__ == "__main__":
    unittest.main()


class NonStrictTests(unittest.TestCase):
    def _session(self):
        s = GroundedSession(strict=False)
        s.add_text("Lagos is a city in Nigeria.", title="nigeria doc")
        return s

    def test_ungrounded_fallback_labeled(self):
        s = self._session()
        def llm(prompt):
            if "do not cover" in prompt:
                return "Paris is the capital of France."
            return "CANNOT_ANSWER"
        ans = s.ask("capital of France?", llm_fn=llm)
        self.assertFalse(ans.refused)
        self.assertIn("Not in your documents", ans.text)
        self.assertIn("Paris", ans.text)

    def test_marker_added_when_model_forgets(self):
        s = self._session()
        def llm(prompt):
            if "do not cover" in prompt:
                return "just the answer, no marker"
            return "CANNOT_ANSWER"
        ans = s.ask("capital of France?", llm_fn=llm)
        self.assertTrue(ans.text.startswith("⚠️ Not in your documents"))

    def test_strict_still_refuses(self):
        s = GroundedSession(strict=True)
        s.add_text("Lagos is a city in Nigeria.", title="nigeria doc")
        ans = s.ask("capital of France?", llm_fn=lambda p: "CANNOT_ANSWER")
        self.assertTrue(ans.refused)
