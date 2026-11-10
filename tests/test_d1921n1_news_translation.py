"""Docs 19+21 Pass N1 — the news leg.

Pins: translate-before-summarize (the summarizer reads the owner's-
language text when the setting is ON; the source text when OFF, with
the OFF digest byte-identical to the pre-N1 render), the durable
news_translations record (written once, read on re-digest, keyed by
item identity + target and guarded by a content hash), translation
badges (verified vs unverified visibly distinct; identity and
same-language items carry none), undetected-language honesty, the
/news target verbs, and a real Yoruba item end to end.

Offline: the translation ladder runs on a fake brain rung (the rung
is faked; the language data is not) and the summarizer is a fake at
the brain_for seam that records what it was fed.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nomorals.agents.news import NewsAgent
from nomorals.core.config import NewsSettings
from nomorals.llm.translate import Translator
from nomorals.storage.db import Database

# ── fixture language data ─────────────────────────────────────────────

FR_T1 = "les marches ouvrent en hausse ce matin"
EN_T1 = "markets open higher this morning"
FR_S1 = "la banque centrale releve ses taux dans la matinee"
EN_S1 = "the central bank raises its rates in the morning"
FR_T2 = "le gouvernement annonce un plan pour les retraites"
EN_T2 = "the government announces a plan for pensions"
FR_GARBAGE_BACK = "quelque chose sans aucun rapport vraiment"
FR_T3 = "les agriculteurs recoltent le ble cette semaine"
EN_T3 = "farmers harvest the wheat this week"
FR_S3 = "les prix du pain restent stables en ville"
EN_S3 = "bread prices stay stable in town"
FR_T4 = "une equipe de chercheurs publie ses resultats"
EN_T4 = "a team of researchers publishes its results"

YO_T1 = ("Ọjà Ìláwé ti kún fun ẹ̀wà ati ìrẹsì lónìí, "
         "awọn ènìyàn si n ra wọn ni owó rere.")
EN_T1 = ("Ilawe market is full of beans and rice today, "
         "and people are buying at good prices.")
YO_S1 = ("Awọn agbẹ ni o mu iṣu ati ata wa si ọja, wọn si ta "
         "wọn ni owó ti o dara fun awọn ènìyàn.")
EN_S1 = ("The farmers brought yams and peppers to the market, "
         "and sold them at a good price for the people.")

#: (chunk, target language name) -> translation for the fake brain.
#: FR_T2's reverse pair is deliberately garbage (gate fails ->
#: unverified); FR_T4 has NO reverse pair (the round-trip cannot
#: confirm it -> the gate scores 0 -> also unverified, never a
#: check it did not earn).
_BRAIN_PAIRS: dict[tuple[str, str], str] = {
    (FR_T1, "english"): EN_T1,
    (EN_T1, "french"): FR_T1,
    (FR_S1, "english"): EN_S1,
    (EN_S1, "french"): FR_S1,
    (FR_T2, "english"): EN_T2,
    (EN_T2, "french"): FR_GARBAGE_BACK,
    (FR_T3, "english"): EN_T3,
    (EN_T3, "french"): FR_T3,
    (FR_S3, "english"): EN_S3,
    (EN_S3, "french"): FR_S3,
    (FR_T4, "english"): EN_T4,
    (YO_T1, "english"): EN_T1,
    (EN_T1, "yoruba"): YO_T1,
    (YO_S1, "english"): EN_S1,
    (EN_S1, "yoruba"): YO_S1,
}


class RoundTripBrain:
    """Brain fake answering both directions from a pair table."""

    def __init__(self, pairs=None):
        self.calls: list[str] = []
        self.chunks: list[str] = []
        self.pairs = dict(_BRAIN_PAIRS if pairs is None else pairs)

    def chat(self, messages, task_kind: str = ""):
        prompt = messages[0].content
        self.calls.append(task_kind)
        match = re.search(r"to (\w+)\.", prompt)
        target = match.group(1).lower() if match else ""
        chunk = prompt.split("\n\n", 1)[1]
        self.chunks.append(chunk)
        out = self.pairs.get((chunk, target))
        if out is None:
            raise RuntimeError("brain has no answer for this chunk")
        return SimpleNamespace(text=out)


class DeadHttp:
    """Every provider call fails: the brain rung or identity decides."""

    def __init__(self):
        self.calls: list[str] = []

    def get(self, url: str, **kw):
        self.calls.append(url)
        raise ConnectionError("network down")

    def post_json(self, url: str, payload, **kw):
        self.calls.append(url)
        raise ConnectionError("network down")


class FakeSummarizer:
    """The digest's brain at the brain_for seam: records prompts,
    answers one fixed numbered sentence per item."""

    def __init__(self):
        self.prompts: list[str] = []

    def chat(self, messages, task_kind: str = ""):
        prompt = messages[0].content
        self.prompts.append(prompt)
        n = len([ln for ln in prompt.splitlines()
                 if re.match(r"\d+\. ", ln)])
        text = "\n".join(f"{i}. Model summary {i}."
                         for i in range(1, n + 1))
        return SimpleNamespace(text=text)


def _row(rid, title, summary, source="Example Wire"):
    return {"id": rid, "url": f"https://example.test/{rid}",
            "title": title, "summary": summary, "source": source,
            "published": 1700000000.0,
            "delivery_state": "undelivered"}


class _NewsCase(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp(
            prefix="n1-",
            dir=os.environ.get("NM_TEST_TMPDIR") or None)
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        self.db = Database(os.path.join(tmp, "test.db"))
        self.db.migrate()
        self.brain = RoundTripBrain()
        self.summarizer = FakeSummarizer()
        settings = SimpleNamespace(
            workspace_dir=tmp, news=NewsSettings(),
            partner=SimpleNamespace(timezone="UTC"))
        self.ctx = SimpleNamespace(
            db=self.db, settings=settings, extras={},
            router=object())
        self.ctx._translator_for_ctx = Translator(
            None, db=self.db, brain=self.brain, client=DeadHttp())
        self.agent = NewsAgent(self.ctx)
        patcher = patch("nomorals.agents.news.brain_for",
                        return_value=self.summarizer)
        patcher.start()
        self.addCleanup(patcher.stop)

    def enable(self, target="en"):
        self.agent.set_news_translation(enabled=True, target=target)

    def record(self, item_ref, target="en"):
        return self.db.query_one(
            "SELECT * FROM news_translations "
            "WHERE item_ref = ? AND target_lang = ?",
            (str(item_ref), target))


class ReorderTests(_NewsCase):
    def test_summarizer_reads_translated_text_when_on(self):
        self.enable("en")
        self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        prompt = self.summarizer.prompts[-1]
        self.assertIn(EN_T1, prompt)
        self.assertIn(EN_S1, prompt)
        self.assertNotIn(FR_T1, prompt)
        self.assertNotIn(FR_S1, prompt)

    def test_summarizer_reads_source_text_when_off(self):
        self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        prompt = self.summarizer.prompts[-1]
        self.assertIn(FR_T1, prompt)
        self.assertIn(FR_S1, prompt)

    def test_off_digest_is_byte_identical_to_legacy_render(self):
        digest = self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        self.assertEqual(
            digest,
            "■ Example Wire\n"
            f"  • {EN_T1} — Model summary 1.\n"
            f"    ◦ original: {FR_T1}")
        self.assertNotIn("🌐", digest)

    def test_translation_span_is_what_the_summarizer_reads(self):
        """The span pin: a 500-char summary is translated as exactly
        summary[:160] — the slice _model_summaries consumes — never
        the whole body."""
        self.enable("en")
        long_summary = FR_S1 + " " + ("filler words here " * 40)
        self.assertGreater(len(long_summary), 500)
        # the fake brain knows only the [:160] span: any longer chunk
        # would miss the pair table and fall to identity.
        span = long_summary[:160]
        self.brain.pairs[(span, "english")] = EN_S1
        self.brain.pairs[(EN_S1, "french")] = span
        digest = self.agent._compose_digest(
            [_row(1, FR_T1, long_summary)])
        self.assertIn(span, self.brain.chunks)
        self.assertTrue(all(len(c) <= 160 or c == FR_T1
                            for c in self.brain.chunks))
        self.assertIn(EN_T1, digest)


class RecordTests(_NewsCase):
    def test_record_written_with_receipt_and_mark(self):
        self.enable("en")
        self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        rec = self.record(1)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["source_lang"], "fr")
        self.assertGreater(rec["source_confidence"], 0.0)
        self.assertEqual(rec["target_lang"], "en")
        self.assertEqual(rec["translated"], 1)
        self.assertEqual(rec["translated_title"], EN_T1)
        self.assertEqual(rec["translated_summary"], EN_S1)
        self.assertEqual(rec["quality_mark"], "verified")
        self.assertTrue(rec["receipt_id"].startswith("tr-"))
        self.assertTrue(rec["title_receipt_id"].startswith("tr-"))
        self.assertEqual(rec["rung"], "brain")

    def test_record_reused_on_redigest(self):
        """One translation, two renders: the second compose runs
        with a dead translator and still renders the recorded
        translation — the record, not a re-translation, serves it."""
        self.enable("en")
        first = self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        calls = len(self.brain.calls)
        self.ctx._translator_for_ctx = Translator(
            None, db=self.db, brain=RoundTripBrain(pairs={}),
            client=DeadHttp())
        second = self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        self.assertEqual(first, second)
        self.assertIn(EN_T1, second)
        rows = self.db.query(
            "SELECT COUNT(*) AS n FROM news_translations")
        self.assertEqual(rows[0]["n"], 1)
        self.assertGreater(calls, 0)  # the first compose did translate

    def test_changed_content_retranslates(self):
        """Same item identity, new text: the content hash refuses the
        stale record and the item is translated again."""
        self.enable("en")
        self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        digest = self.agent._compose_digest([_row(1, FR_T3, FR_S3)])
        self.assertIn(EN_T3, digest)
        rec = self.record(1)
        self.assertEqual(rec["translated_title"], EN_T3)

    def test_news_items_never_written_by_translation(self):
        """news_items stays canonical: composing with translation ON
        leaves the stored row byte-identical."""
        self.db.execute(
            "INSERT INTO news_items (url, title, summary, source, "
            "published, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            ("https://example.test/1", FR_T1, FR_S1, "Example Wire",
             1700000000.0, 1700000000.0))
        before = self.db.query("SELECT * FROM news_items")
        self.enable("en")
        self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        after = self.db.query("SELECT * FROM news_items")
        self.assertEqual(before, after)


class BadgeTests(_NewsCase):
    def test_verified_badge(self):
        self.enable("en")
        digest = self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        self.assertIn(f"  • {EN_T1} 🌐 fr→en ✓ — Model summary 1.",
                      digest)
        self.assertIn(f"    ◦ original: {FR_T1}", digest)

    def test_unverified_badge_is_visibly_distinct(self):
        self.enable("en")
        digest = self.agent._compose_digest([_row(2, FR_T2, FR_S1)])
        self.assertIn(f"  • {EN_T2} 🌐 fr→en ⚠ unverified",
                      digest)
        self.assertNotIn("🌐 fr→en ✓", digest)
        rec = self.record(2)
        self.assertEqual(rec["quality_mark"], "unverified")

    def test_unconfirmable_translation_never_wears_the_check(self):
        """FR_T4 has no reverse pair: the round-trip cannot confirm
        it, the gate scores the empty back-translation 0 (failed),
        and the item wears the unverified badge — never a check it
        did not earn."""
        self.enable("en")
        digest = self.agent._compose_digest([_row(4, FR_T4, "")])
        self.assertIn(f"  • {EN_T4} 🌐 fr→en ⚠ unverified", digest)
        self.assertNotIn("🌐 fr→en ✓", digest)
        rec = self.record(4)
        self.assertEqual(rec["quality_mark"], "unverified")

    def test_same_language_item_carries_no_badge(self):
        self.enable("en")
        digest = self.agent._compose_digest(
            [_row(5, EN_T1, EN_S1)])
        self.assertNotIn("🌐", digest)
        self.assertIn(f"  • {EN_T1}", digest)
        rec = self.record(5)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["translated"], 0)
        self.assertEqual(rec["quality_mark"], "same-language")

    def test_no_badge_when_off(self):
        digest = self.agent._compose_digest([_row(1, FR_T1, FR_S1)])
        self.assertNotIn("🌐", digest)
        self.assertIsNone(self.record(1))


class UndetectedTests(_NewsCase):
    def test_undetected_language_row_says_so(self):
        """Gibberish: the record's source_lang stays empty (never a
        guessed language), nothing is dressed as translated, and the
        digest renders the source text plainly."""
        self.enable("en")
        digest = self.agent._compose_digest(
            [_row(9, "xkcd zqwv kjpl", "qqq zzz www xxx")])
        self.assertIn("  • xkcd zqwv kjpl", digest)
        self.assertNotIn("🌐", digest)
        rec = self.record(9)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["source_lang"], "")
        self.assertEqual(rec["source_confidence"], 0.0)
        self.assertEqual(rec["translated"], 0)


class SettingsAndVerbTests(_NewsCase):
    def test_default_posture_is_off(self):
        posture = self.agent.news_translation()
        self.assertFalse(posture["enabled"])
        self.assertEqual(posture["target"], "")

    def test_settings_seed_is_honoured(self):
        self.ctx.settings.news = NewsSettings(
            translation_enabled=True, translation_target="yo")
        posture = NewsAgent(self.ctx).news_translation()
        self.assertTrue(posture["enabled"])
        self.assertEqual(posture["effective_target"], "yo")
        self.assertEqual(posture["source"], "settings")

    def test_stored_pref_wins_over_settings_seed(self):
        self.ctx.settings.news = NewsSettings(
            translation_enabled=True, translation_target="yo")
        agent = NewsAgent(self.ctx)
        agent.set_news_translation(enabled=False)
        posture = NewsAgent(self.ctx).news_translation()
        self.assertFalse(posture["enabled"])
        self.assertEqual(posture["source"], "stored")

    def test_config_defaults_never_silently_translate(self):
        cfg = NewsSettings()
        self.assertFalse(cfg.translation_enabled)
        self.assertEqual(cfg.translation_target, "")

    def _mixin(self):
        from nomorals.agents.partner.runtime_live import RuntimeLiveMixin

        mixin = RuntimeLiveMixin()
        mixin.context = self.ctx
        return mixin

    def test_news_target_verb_round_trip(self):
        mixin = self._mixin()
        out = mixin._control_news("target")
        self.assertIn("off", out)
        out = mixin._control_news("target yo")
        self.assertIn("Yoruba (yo)", out)
        # a fresh agent on the same db reads the stored posture back
        posture = NewsAgent(self.ctx).news_translation()
        self.assertTrue(posture["enabled"])
        self.assertEqual(posture["effective_target"], "yo")
        out = mixin._control_news("target")
        self.assertIn("on → Yoruba (yo)", out)
        out = mixin._control_news("target off")
        self.assertIn("off", out)
        self.assertFalse(
            NewsAgent(self.ctx).news_translation()["enabled"])
        out = mixin._control_news("target on")
        self.assertIn("on → Yoruba (yo)", out)

    def test_news_target_rejects_unknown_language(self):
        mixin = self._mixin()
        out = mixin._control_news("target klingon")
        self.assertIn("unknown language", out)
        self.assertFalse(self.agent.news_translation()["enabled"])

    def test_usage_string_names_target(self):
        mixin = self._mixin()
        out = mixin._control_news("bogus")
        self.assertIn("target [<code>|on|off]", out)


class YorubaNewsTests(_NewsCase):
    def test_yoruba_item_translated_recorded_and_badged(self):
        """The owner's flagship pair, end to end: a Yoruba market
        story is detected as yo, translated before summarizing, and
        lands in the record with its receipt and verified mark."""
        self.enable("en")
        digest = self.agent._compose_digest([_row(7, YO_T1, YO_S1)])
        prompt = self.summarizer.prompts[-1]
        self.assertIn(EN_T1, prompt)
        self.assertNotIn(YO_T1, prompt)
        self.assertIn(f"  • {EN_T1} 🌐 yo→en ✓", digest)
        self.assertIn(f"    ◦ original: {YO_T1}", digest)
        rec = self.record(7)
        self.assertEqual(rec["source_lang"], "yo")
        self.assertEqual(rec["translated_title"], EN_T1)
        self.assertEqual(rec["quality_mark"], "verified")


if __name__ == "__main__":
    unittest.main()
