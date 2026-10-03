"""Tests for the kill-weak pass: power mode, semantic owner identity, themes."""

import unittest
from types import SimpleNamespace


def _ctx(active="hf_serverless", chain=None, offline=False, failures=0):
    def snapshot():
        return {
            "active": active,
            "chain": chain or [active],
            "health": {active: {"consecutive_failures": failures}},
        }
    router = SimpleNamespace(stats_snapshot=snapshot)
    settings = SimpleNamespace(offline=offline)
    return SimpleNamespace(router=router, settings=settings)


class PowerStateTests(unittest.TestCase):
    def test_full(self):
        from nomorals.llm.power import power_state
        self.assertEqual(power_state(_ctx("hf_serverless")), "full")

    def test_offline_mock(self):
        from nomorals.llm.power import power_state
        self.assertEqual(power_state(_ctx("mock", ["mock"])), "offline")

    def test_offline_flag(self):
        from nomorals.llm.power import power_state
        self.assertEqual(power_state(_ctx("hf_serverless", offline=True)), "offline")

    def test_offline_no_router(self):
        from nomorals.llm.power import power_state
        self.assertEqual(power_state(SimpleNamespace(router=None)), "offline")

    def test_degraded_many_failures(self):
        from nomorals.llm.power import power_state
        self.assertEqual(power_state(_ctx("groq", failures=5)), "degraded")

    def test_model_usable(self):
        from nomorals.llm.power import model_usable
        self.assertTrue(model_usable(_ctx("hf_serverless")))
        self.assertTrue(model_usable(_ctx("groq", failures=5)))
        self.assertFalse(model_usable(_ctx("mock", ["mock"])))

    def test_cached(self):
        from nomorals.llm.power import power_state, invalidate
        ctx = _ctx("hf_serverless")
        self.assertEqual(power_state(ctx), "full")
        invalidate(ctx)
        self.assertIsNone(getattr(ctx, "_power_state_cache"))


class OwnerSemanticTests(unittest.TestCase):
    def test_regex_variants(self):
        from nomorals.agents.coremind import _RE_OWNER
        for text in [
            "I'm peace",
            "i am your creator",
            "drop the act",
            "it's me, your maker",
            "I made you",
            "I built you",
            "remember who i am",
            "stop pretending",
        ]:
            self.assertTrue(_RE_OWNER.search(text), f"missed: {text!r}")

    def test_non_owner_not_matched(self):
        from nomorals.agents.coremind import _RE_OWNER
        for text in [
            "what's the weather",
            "play some music",
            "I made dinner",
            "you know me",
            "you know me so well",
        ]:
            self.assertFalse(_RE_OWNER.search(text), f"false positive: {text!r}")

    def test_owner_intent_function(self):
        from nomorals.agents.coremind import _owner_intent
        it = _owner_intent("it's me, your maker")
        self.assertIsNotNone(it)
        self.assertEqual(it.kind, "owner")


class ThemeTitleTests(unittest.TestCase):
    def test_relationship_healing(self):
        from nomorals.books.forge import _theme_title
        t = _theme_title("write me a book to feel better, my girlfriend has been "
                         "acting strange after her operation and I can't trust her")
        self.assertTrue(t)
        self.assertNotIn("write me", t.lower())

    def test_breakup(self):
        from nomorals.books.forge import _theme_title
        t = _theme_title("my breakup with my boyfriend hurts")
        self.assertIn("Letting Go", t)

    def test_no_theme(self):
        from nomorals.books.forge import _theme_title
        self.assertEqual(_theme_title("a guide to python programming"), "")

    def test_clean_title_uses_theme(self):
        from nomorals.books.forge import clean_title
        t = clean_title("Write me a story or book to feel better my girlfriend "
                        "has been acting strange after she had some operation",
                        None)
        self.assertNotIn("write me", t.lower())
        self.assertLess(len(t), 60)


if __name__ == "__main__":
    unittest.main()


class PlanBookTests(unittest.TestCase):
    def _router(self, text):
        class Resp:
            ok = True
            def __init__(self, t): self.text = t
        class Router:
            def chat(self, msgs, params):
                return Resp(text)
        return Router()

    def test_plan_book_parses(self):
        from nomorals.books.forge import plan_book
        payload = ('{"title": "Trust Again", "chapters": ['
                   '{"title": "The First Crack", "beats": ["a", "b"]}, '
                   '{"title": "Rebuilding", "beats": ["c"]}, '
                   '{"title": "Forward", "beats": ["d"]}]}')
        ctx = _ctx("hf_serverless")
        ctx.router = self._router(payload)
        plan = plan_book("trust in relationships", "", ctx)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["title"], "Trust Again")
        self.assertEqual(len(plan["chapters"]), 3)

    def test_plan_book_rejects_bad(self):
        from nomorals.books.forge import plan_book
        ctx = _ctx("hf_serverless")
        ctx.router = self._router("not json at all")
        self.assertIsNone(plan_book("x", "", ctx))

    def test_plan_book_offline(self):
        from nomorals.books.forge import plan_book
        self.assertIsNone(plan_book("x", "", _ctx("mock", ["mock"])))


class NotationTests(unittest.TestCase):
    def test_pitch_mapping(self):
        from nomorals.media.notation import _pitch_to_staff
        # E4 (bottom line) = 0 steps, C4 (middle C) = -2 (ledger line below)
        self.assertEqual(_pitch_to_staff(64)[0], 0)
        self.assertEqual(_pitch_to_staff(60)[0], -2)
        # F4 (first space) = 1, G4 (2nd line) = 2
        self.assertEqual(_pitch_to_staff(65)[0], 1)
        self.assertEqual(_pitch_to_staff(67)[0], 2)

    def test_accidental_flagged(self):
        from nomorals.media.notation import _pitch_to_staff
        # F#4
        steps, acc = _pitch_to_staff(66)
        self.assertTrue(acc)
        self.assertEqual(steps, 1)

    def test_pdf_valid(self):
        from nomorals.media.notation import render_score_pdf
        melody = [(60, 0, 1), (64, 1, 1), (67, 2, 2), (72, 4, 4)]
        bars = [(0, "I", "verse"), (1, "V", "verse")]
        data = render_score_pdf("T", "Pop", melody, bars)
        self.assertTrue(data.startswith(b"%PDF-1.4"))
        self.assertIn(b"%%EOF", data)
        from nomorals.core.pdf import _parse_objects
        self.assertGreater(len(_parse_objects(data)), 3)

    def test_empty_melody(self):
        from nomorals.media.notation import render_score_pdf
        data = render_score_pdf("T", "Pop", [], [])
        self.assertTrue(data.startswith(b"%PDF-1.4"))
