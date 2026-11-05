"""Tests for the English G2P rung (Wave 3c) — media/g2p_en.py.

CMUdict-pattern: lexicon first (fixtures are real CMUdict entries),
letter-to-sound rules for out-of-lexicon words with the source
recorded honestly on every result, and the rung consumable from the
TTS front-end (UniversalTTS.pronounce) and the singing path.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nomorals.media.g2p_en import (
    ARPABET_CONSONANTS,
    ARPABET_VOWELS,
    base_phoneme,
    g2p_en,
    lexicon_stats,
    load_cmudict,
    phonemize,
    phonemize_flat,
)

# Real CMUdict (0.7b) pronunciations — the acceptance fixtures.
CMUDICT_FIXTURES = {
    "hello": ["HH", "AH0", "L", "OW1"],
    "world": ["W", "ER1", "L", "D"],
    "voice": ["V", "OY1", "S"],
    "computer": ["K", "AH0", "M", "P", "Y", "UW1", "T", "ER0"],
    "through": ["TH", "R", "UW1"],
    "thought": ["TH", "AO1", "T"],
    "english": ["IH1", "NG", "G", "L", "IH0", "SH"],
    "question": ["K", "W", "EH1", "S", "CH", "AH0", "N"],
    "beautiful": ["B", "Y", "UW1", "T", "IH0", "F", "AH0", "L"],
    "family": ["F", "AE1", "M", "AH0", "L", "IY0"],
    "water": ["W", "AO1", "T", "ER0"],
    "people": ["P", "IY1", "P", "AH0", "L"],
}


class TestLexiconFixtures(unittest.TestCase):
    def test_cmudict_fixture_words(self):
        for word, expected in CMUDICT_FIXTURES.items():
            with self.subTest(word=word):
                result = g2p_en(word)
                self.assertEqual(result.phonemes, expected)
                self.assertEqual(result.source, "lexicon")
                self.assertEqual(result.rung, "g2p_en")

    def test_case_and_punctuation_normalized(self):
        self.assertEqual(g2p_en("Hello!").phonemes,
                         CMUDICT_FIXTURES["hello"])
        self.assertEqual(g2p_en("WORLD").phonemes,
                         CMUDICT_FIXTURES["world"])

    def test_every_phoneme_is_arpabet(self):
        for word in CMUDICT_FIXTURES:
            for phoneme in g2p_en(word).phonemes:
                self.assertIn(base_phoneme(phoneme),
                              ARPABET_VOWELS | ARPABET_CONSONANTS)


class TestRulesFallback(unittest.TestCase):
    def test_oov_word_uses_rules_and_says_so(self):
        result = g2p_en("blorp")
        self.assertEqual(result.source, "rules")
        self.assertEqual(result.phonemes,
                         ["B", "L", "AO1", "R", "P"])

    def test_oov_digraphs(self):
        result = g2p_en("shplank")
        self.assertEqual(result.source, "rules")
        self.assertEqual(result.phonemes[0], "SH")
        self.assertIn("NG", result.phonemes)

    def test_unpronounceable_token_is_empty_not_invented(self):
        result = g2p_en("12345")
        self.assertEqual(result.phonemes, [])
        self.assertEqual(result.source, "rules")

    def test_sentence_sources_per_word(self):
        words = phonemize("hello blorp world")
        self.assertEqual([w.source for w in words],
                         ["lexicon", "rules", "lexicon"])
        flat = phonemize_flat("hello world")
        self.assertEqual(flat, CMUDICT_FIXTURES["hello"]
                         + CMUDICT_FIXTURES["world"])
        self.assertEqual(phonemize_flat("hello", with_stress=False),
                         ["HH", "AH", "L", "OW"])


class TestUserLexicon(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)
        import nomorals.media.g2p_en as g
        g._user_lexicon.clear()
        g._user_loaded_from.clear()

    def test_load_cmudict_layers_and_wins(self):
        path = os.path.join(self.tmp, "mini.dict")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(";;; mini fixture dictionary\n")
            fh.write("ZQXWIBBLE  Z AH0 K S W IH1 B AH0 L\n")
            fh.write("ZQXWIBBLE(1)  Z AH0 K S W IH1 B AH0 L IY0\n")
            fh.write("HELLO  HH AH0 L OW1\n")
        added = load_cmudict(path)
        self.assertEqual(added, 1)      # only zqxwibble is new coverage
        result = g2p_en("zqxwibble")
        self.assertEqual(result.source, "lexicon_user")
        # first entry wins; the (1) alternate is not kept
        self.assertEqual(result.phonemes[-1], "L")
        stats = lexicon_stats()
        self.assertEqual(stats["user"], 2)   # zqxwibble + layered hello
        self.assertGreaterEqual(stats["bundled"], 100)

    def test_empty_dictionary_file_refused_loudly(self):
        path = os.path.join(self.tmp, "empty.dict")
        open(path, "w").close()
        with self.assertRaises(ValueError):
            load_cmudict(path)


class TestFrontEndConsumption(unittest.TestCase):
    def test_universal_tts_pronounce(self):
        from nomorals.voice.tts import UniversalTTS

        engine = UniversalTTS()
        out = engine.pronounce("hello world")
        self.assertEqual([w["word"] for w in out], ["hello", "world"])
        self.assertEqual(out[0]["phonemes"],
                         CMUDICT_FIXTURES["hello"])
        self.assertEqual(out[0]["source"], "lexicon")
        self.assertEqual(out[0]["rung"], "g2p_en")

    def test_singing_path_prefers_rung(self):
        from nomorals.voice.singing import _english_phonemes

        # lexicon word → stress-stripped arpabet, not letter chunks
        self.assertEqual(_english_phonemes("hello"),
                         ["HH", "AH", "L", "OW"])
        # rules word still yields arpabet-shaped tokens
        self.assertEqual(_english_phonemes("blorp"),
                         ["B", "L", "AO", "R", "P"])


if __name__ == "__main__":
    unittest.main()
