"""BOOK_FORGE_HUMANE: voiceprint, AI-tells, craft passes, humane
structure, mechanical tests, collab-as-default.

Every check is advisory (blocked == False) — the tests prove the
reports exist, measure real properties, and never gate.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nomorals.books.ai_tells import AI_TELLS, scan_ai_tells
from nomorals.books.bible import build_bible
from nomorals.books.collab import (
    DEFAULT_MODE, CollaborativeSession,
)
from nomorals.books.craft import CRAFT_PASSES, run_craft_passes
from nomorals.books.fiction import FictionWriter, StoryState, engine_for
from nomorals.books.humane import (
    DedicatedTo, character_pushback, narrator_knows_less,
    register_guidance, silence_analysis,
)
from nomorals.books.mechanical import (
    delete_first_paragraph_per_scene, delete_first_paragraph_test,
    delete_last_paragraph_test, read_aloud_check,
)
from nomorals.books.model import Book
from nomorals.books.styles import prose_vs_presentation
from nomorals.books.voiceprint import (
    build_owner_voiceprint, compare, drift_score, fingerprint,
    fingerprint_for_character,
)

TELL_TEXT = (
    "It was a testament to her courage, and little did he know what waited. "
    "In that moment, she felt a mix of fear and hope, not just fear but dread. "
    "It wasn't fear, it was hunger. Something shifted. The air thick with smoke. "
    "The silence was deafening. Time seemed to slow. She felt a wave of grief. "
    "She couldn't help but smile. A chill ran down her spine. She took a breath "
    "she didn't know she was holding. She was devastated. Her heart raced. "
    "She saw the door, which meant that the way was open. "
    "She packed the bread, the cheese, and the wine. "
    '"I hate this," she said angrily. '
    '"The key is under the mat," Mara said. '
    "The key was under the mat, hidden from sight. "
    "She paused — and waited — and watched — and listened — and breathed — and stood. "
    "The road was long and the night was cold and the wind was high and the "
    "stars were out and the river ran dark beside the stones and the trees "
    "moved in the wind over the hills beyond the town where she had grown. "
    "\n\nWould she ever find hope again?"
)

PLAIN_TEXT = (
    "Mara counted 3 coins on the oak table, 12 inches from the knife. "
    "The air smelled of smoke. The floor felt rough under her hands. "
    "A bell rang somewhere beyond the wall. She saw the door stand open. "
    "\n\n\"Run,\" Mara said. \"Now.\"\n\n"
    "\"Why?\" Jide asked.\n\n"
    "She did not answer. The coins were warm from her pocket."
)

LONG_SENTENCE = (
    "The committee, having considered the matter at length and after "
    "consulting with every department that could conceivably have an "
    "opinion on the subject, the budget, the timeline, the staffing, "
    "and the political consequences of each possible course of action, "
    "finally, reluctantly, and with considerable misgiving, decided "
    "to postpone the decision until the following quarter had ended."
)


def _ctx():
    tmp = tempfile.mkdtemp()
    settings = SimpleNamespace(workspace_dir=tmp)
    return SimpleNamespace(db=None, settings=settings), tmp


class TestVoiceprint(unittest.TestCase):
    def test_fingerprint_measures(self) -> None:
        vp = fingerprint(PLAIN_TEXT, ["Mara", "Jide"])
        self.assertGreater(vp.words, 0)
        self.assertGreater(vp.sentence_len_mean, 0)
        self.assertGreaterEqual(vp.pct_under_8_words, 0)
        self.assertGreater(vp.specificity_per_1k, 0)
        self.assertGreater(vp.sensory.get("smell", 0), 0)
        self.assertGreater(vp.sensory.get("touch", 0), 0)
        self.assertGreater(vp.sensory.get("sound", 0), 0)
        self.assertGreater(vp.dialogue_ratio, 0)
        self.assertFalse(vp.blocked)
        # idiolect per character from attributed lines
        self.assertIn("Mara", vp.idiolect)
        self.assertIn("Jide", vp.idiolect)
        self.assertGreater(vp.idiolect["Mara"]["lines"], 0)

    def test_drift_self_zero_and_other_positive(self) -> None:
        vp = fingerprint(PLAIN_TEXT)
        self.assertEqual(drift_score(vp, vp), 0.0)
        other = fingerprint(
            "The consideration of the situation required extensive "
            "deliberation, and the committee's communication emphasized "
            "the necessity of a comprehensive evaluation of every "
            "consequence before any determination could be established. "
            "Furthermore, the implementation of the decision demanded "
            "patience, precision, and an abundance of caution.")
        self.assertGreater(drift_score(vp, other), 0.05)

    def test_compare_reports_not_blocks(self) -> None:
        vp = fingerprint(PLAIN_TEXT)
        other = fingerprint("Short. Words. Here. Now. Go. Run. Fast. Quick.")
        rep = compare(other, vp)
        self.assertFalse(rep["blocked"])
        self.assertTrue(rep["owner_may_overrule"])
        self.assertIn("drift", rep)
        self.assertIn("largest_deltas", rep)

    def test_owner_voiceprint_and_roundtrip(self) -> None:
        vp = build_owner_voiceprint(
            ["I write short notes. Quick. Direct.",
             "Another note, same clipped rhythm. Short sentences."])
        self.assertGreater(vp.words, 0)
        self.assertEqual(
            fingerprint("x").__class__.from_dict(vp.to_dict()).words, vp.words)
        empty = build_owner_voiceprint([])
        self.assertEqual(empty.words, 0)

    def test_character_fingerprint_empty_when_absent(self) -> None:
        vp = fingerprint_for_character(PLAIN_TEXT, "Nobody Here")
        self.assertEqual(vp.words, 0)


class TestAITells(unittest.TestCase):
    def test_starting_list_detected(self) -> None:
        rep = scan_ai_tells(TELL_TEXT)
        self.assertFalse(rep["blocked"])
        self.assertTrue(rep["owner_may_overrule"])
        found = {t["id"] for t in rep["tells"]}
        for want in ("testament_to", "little_did_know", "in_that_moment",
                     "mix_of", "not_just_but", "wasnt_x_was_y",
                     "something_shifted", "air_thick_with",
                     "silence_deafening", "time_slow", "felt_wave",
                     "couldnt_help_but", "chill_spine", "breath_holding",
                     "was_devastated", "heart_raced", "explanatory_clause",
                     "rule_of_three", "dialogue_tag_emotion",
                     "restating_sentence", "chapter_close",
                     "filtering_verbs"):
            self.assertIn(want, found, f"missing tell {want}")
        # counts + locations shape
        for tell in rep["tells"]:
            self.assertGreater(tell["count"], 0)
            self.assertTrue(tell["locations"])
            self.assertIn("line", tell["locations"][0])
            self.assertIn("excerpt", tell["locations"][0])

    def test_clean_text_low(self) -> None:
        rep = scan_ai_tells("The kettle whistled. Mara poured the water. "
                            "Outside, rain.")
        self.assertEqual(rep["total"], 0)

    def test_library_nonempty(self) -> None:
        self.assertGreaterEqual(len(AI_TELLS), 15)


class TestCraftPasses(unittest.TestCase):
    def test_all_eight_present_and_reporting(self) -> None:
        rep = run_craft_passes(TELL_TEXT + "\n\n" + PLAIN_TEXT)
        self.assertFalse(rep["blocked"])
        self.assertEqual(set(rep["passes"]), set(CRAFT_PASSES))
        for name, p in rep["passes"].items():
            self.assertFalse(p["blocked"], name)
            self.assertIn("metrics", p)
            self.assertIn("findings", p)
            self.assertIn("suggestions", p)
            self.assertGreaterEqual(p["score"], 0.0)
            self.assertLessEqual(p["score"], 1.0)

    def test_restraint_measures_cut_and_readability(self) -> None:
        rep = run_craft_passes(
            "She very really just quite slowly walked in order to see the "
            "room due to the fact that it was large. " * 8)
        restraint = rep["passes"]["restraint"]
        self.assertIn("readability_before", restraint["metrics"])
        self.assertIn("readability_after_trim", restraint["metrics"])
        self.assertIn("darlings", restraint["metrics"])
        self.assertEqual(restraint["metrics"]["cut_target_pct"], "10-15%")

    def test_ending_flags_moralising(self) -> None:
        rep = run_craft_passes(
            "She walked home through the rain.\n\n"
            "In the end, what mattered was that she had learned that "
            "hope is the lesson.")
        ending = rep["passes"]["ending"]
        self.assertTrue(ending["metrics"]["delete_last_line_better"])
        self.assertTrue(ending["metrics"]["moralising_phrases"])

    def test_de_filter_targets_zero(self) -> None:
        rep = run_craft_passes(
            "She saw the door. She felt the cold. She heard the bell. "
            "She noticed the crack. She realized the truth.")
        self.assertGreater(
            rep["passes"]["de_filter"]["metrics"]["filtering_per_1k"], 0)


class TestHumane(unittest.TestCase):
    def test_for_field_intimate_vs_formal(self) -> None:
        intimate = register_guidance(
            DedicatedTo(name="Ada", relationship="partner",
                        occasion="birthday"))
        self.assertEqual(intimate["register"], "intimate")
        self.assertEqual(intimate["explained_vs_assumed"], "assumed")
        self.assertEqual(intimate["dedication"], "For Ada, partner — birthday.")
        formal = register_guidance(
            DedicatedTo(name="Readers", relationship="readers"))
        self.assertEqual(formal["register"], "formal")
        self.assertEqual(formal["explained_vs_assumed"], "explained")
        book = Book(topic="t", slug="t",
                    for_whom={"name": "Ada", "relationship": "partner",
                              "occasion": "birthday"})
        self.assertEqual(book.dedication_line(), "For Ada, partner — birthday.")
        self.assertEqual(book.to_dict()["for"]["name"], "Ada")
        self.assertEqual(Book.from_dict(book.to_dict()).for_whom["name"], "Ada")

    def test_character_pushback(self) -> None:
        chars = [{"name": "Mara", "speech_tics": ["listen"]},
                 {"name": "Ghost"}]
        notes = character_pushback(
            "Mara suddenly knew the answer. She just happened to find the "
            "key, conveniently placed. \"We go now,\" Mara said.",
            chars)
        kinds = {n["kind"] for n in notes}
        self.assertIn("plot_convenience", kinds)
        self.assertIn("nothing_to_say", kinds)  # Ghost never appears
        self.assertTrue(all(n["blocked"] is False for n in notes))
        silent = character_pushback(
            "Mara stood in the doorway watching the rain.", chars)
        self.assertTrue(any(n["kind"] == "silent" for n in silent))

    def test_narrator_and_silence(self) -> None:
        rep = narrator_knows_less(
            "Little did Mara know, the letter would be her undoing. "
            "Unknown to her, the house was already sold.")
        self.assertGreater(rep["count"], 0)
        self.assertFalse(rep["blocked"])
        sil = silence_analysis(
            "She started to say it, then stopped… Never mind. "
            "They never spoke of it again.")
        self.assertTrue(sil["silence_present"])
        self.assertGreater(sil["evasion_count"], 0)
        self.assertGreater(sil["off_page_count"], 0)


class TestMechanical(unittest.TestCase):
    def test_read_aloud_stumbles(self) -> None:
        rep = read_aloud_check(
            LONG_SENTENCE + " The strengthless throngths clambered. "
            "Short now.")
        self.assertFalse(rep["blocked"])
        self.assertEqual(len(rep["sentences_over_40_words"]), 1)
        self.assertGreater(len(rep["consonant_pileups"]), 0)
        self.assertGreater(len(rep["clauses_before_verb_3plus"]), 0)
        self.assertGreater(rep["stumbles"], 0)
        clean = read_aloud_check("Mara ran. The door stood open. She went in.")
        self.assertEqual(clean["stumbles"], 0)

    def test_delete_first_paragraph(self) -> None:
        throat = ("For as long as Mara could remember, the town had been "
                  "quiet and grey and still.\n\n"
                  "\"Run!\" Mara shouted, grabbing the key.")
        rep = delete_first_paragraph_test(throat)
        self.assertEqual(rep["verdict"], "likely_better_without")
        keep = delete_first_paragraph_test(
            "\"Run!\" Mara shouted, grabbing the key.\n\n"
            "The door slammed behind her.")
        self.assertEqual(keep["verdict"], "keep")
        per_scene = delete_first_paragraph_per_scene(
            throat + "\n\n---\n\n" + throat)
        self.assertEqual(len(per_scene), 2)
        self.assertTrue(all(r["verdict"] == "likely_better_without"
                            for r in per_scene))
        last = delete_last_paragraph_test(
            "She walked home through the rain.\n\n"
            "In the end, what mattered was hope.")
        self.assertEqual(last["verdict"], "likely_better_without")

    def test_synthesize_never_raises_with_fake_tts(self) -> None:
        from nomorals.books.mechanical import synthesize_for_read_aloud

        class _Fake:
            def speak(self, text, out_path=""):
                return {"backend": "fake", "path": "/tmp/fake.wav",
                        "segments": 1}

        result = synthesize_for_read_aloud("Hello.", tts=_Fake())
        self.assertIsInstance(result, dict)
        self.assertIn("ok", result)
        if result["ok"]:
            self.assertEqual(result["backend"], "fake")


class TestFictionValidateWiring(unittest.TestCase):
    def test_validate_advisory(self) -> None:
        ctx, _ = _ctx()
        writer = FictionWriter(ctx)
        self.assertTrue(writer.use_collab)  # collab is default, not an option
        engine = engine_for("mystery", seed=42)
        state = StoryState(slug="test-mystery", title="T", premise="test",
                           genre=engine.name, mode="novel", total_planned=10,
                           for_whom={"name": "Ada", "relationship": "partner"})
        arcs = engine.plan_arcs("a detective hunts a killer", state)
        state.arcs = [a.to_dict() for a in arcs]
        state.arcs[0]["status"] = "active"
        brief = engine.chapter_brief(state, arcs[0], 2)
        culprit = state.ledger["culprit"]
        method = state.ledger["method"]
        bad = (f"It was {culprit} all along. The detective explained how "
               f"{method} had been done. " * 5)
        rep = writer.validate(bad, state, brief)
        self.assertFalse(rep["blocked"])
        self.assertTrue(any("fair-play" in p for p in rep["genre_problems"]))
        self.assertEqual(set(rep["craft"]["passes"]), set(CRAFT_PASSES))
        self.assertIn("total", rep["ai_tells"])
        self.assertIn("character_pushback", rep["humane"])
        self.assertIn("read_aloud", rep["mechanical"])
        self.assertIn("delete_first_paragraph", rep["mechanical"])
        # character pushback rides alongside genre validation
        self.assertIsInstance(
            engine.character_pushback(bad, state, brief), list)

    def test_start_stores_for_and_owner_voiceprint(self) -> None:
        ctx, _ = _ctx()
        writer = FictionWriter(ctx)
        out = writer.start("a detective hunts a killer", genre="mystery",
                           mode="novel", title="Humane Test",
                           for_whom={"name": "Ada", "relationship": "partner"},
                           owner_texts=["I write short notes. Quick. Direct."])
        self.assertIn("advisory", out["chapter_1"])
        self.assertIn("collab_critique", out["chapter_1"])
        state = writer.load(out["slug"])
        self.assertEqual(state.for_whom["name"], "Ada")
        self.assertIn("owner_voiceprint", state.style)

    def test_bible_voice_target_and_continuation_drift(self) -> None:
        ctx, _ = _ctx()
        bible = build_bible("humane-story", "Humane Story", [
            (1, "One", PLAIN_TEXT),
            (2, "Two", PLAIN_TEXT + " Mara walked to the river. The water "
             "was cold and clear and fast."),
        ])
        self.assertTrue(bible.voiceprint)
        self.assertGreater(bible.voiceprint["words"], 0)
        self.assertIn("VOICE TARGET", bible.brief())
        from nomorals.books.continuation import StoryContinuer
        cont = StoryContinuer(ctx)
        drift = cont.validate_voice(PLAIN_TEXT, bible)
        self.assertIsNotNone(drift)
        assert drift is not None
        self.assertFalse(drift["blocked"])
        self.assertLess(drift["drift"], 0.35)

    def test_collab_default_session_owner_voice(self) -> None:
        self.assertEqual(DEFAULT_MODE, "collab")
        sess = CollaborativeSession(
            "s1", owner_texts=["I write short notes. Quick. Direct."])
        self.assertIsNotNone(sess.owner_voiceprint)
        check = sess.voice_check(PLAIN_TEXT)
        self.assertIsNotNone(check)
        assert check is not None
        self.assertFalse(check["blocked"])

    def test_styles_presentation_vs_prose(self) -> None:
        text = prose_vs_presentation()
        self.assertIn("PRESENTATION", text)
        self.assertIn("PROSE STYLE", text)
        self.assertIn("voiceprint", text)


if __name__ == "__main__":
    unittest.main()
