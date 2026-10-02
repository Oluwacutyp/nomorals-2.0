"""Tests for the dynamic vocabulary-expansion lexicon
(nomorals/agents/research_lexicon.py) and the style.py consumer hook."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from nomorals.agents.research_lexicon import (
    LexiconStore,
    acquire_from_findings,
    dynamic_terms,
    mine_candidates,
    score_term,
)
from nomorals.partner.style import identity_leak_check, strip_robotic
from nomorals.storage.db import Database


def _finding(claim: str, angle: str = "a") -> SimpleNamespace:
    return SimpleNamespace(angle=angle, claim=claim, sources=[], confidence=0.5)


class ScoreTermTest(unittest.TestCase):
    def test_relevant_novel_term_scores_high(self):
        s = score_term(
            "abliterated weight merge technique",
            ["abliterated", "weight", "merge", "technique"],
            [],
        )
        self.assertGreaterEqual(s, 0.85)

    def test_no_keywords_is_neutral(self):
        # relevance 0.5, novelty 1.0, quality 1.0 -> 0.25 + 0.3 + 0.2
        self.assertEqual(score_term("some decent phrase", None, []), 0.75)

    def test_duplicate_scores_low_on_novelty(self):
        novel = score_term("alpha beta gamma", ["alpha", "beta"], [])
        dup = score_term("alpha beta gamma", ["alpha", "beta"], ["alpha beta gamma"])
        self.assertLess(dup, novel)
        self.assertEqual(dup, 0.533)

    def test_junk_scores_zero(self):
        self.assertEqual(score_term("https://example.com/foo"), 0.0)
        self.assertEqual(score_term("a"), 0.0)
        self.assertEqual(score_term("aaaaaa"), 0.0)
        self.assertEqual(score_term("   "), 0.0)

    def test_score_in_range(self):
        for t in ("quantum flux", "x9!!", "a reasonably long phrase here"):
            s = score_term(t, ["quantum"], ["other thing"])
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(s, 1.0)


class AcquireTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = LexiconStore(SimpleNamespace(db=self.db))

    def tearDown(self):
        self.db.close()

    def test_acquire_inserts_above_threshold(self):
        res = self.store.acquire(
            ["quantum flux capacitor", "x"],
            module="partner",
            category="robotic",
            source="test",
            threshold=0.55,
            category_keywords=["quantum", "flux", "capacitor"],
        )
        self.assertEqual(res["added"], ["quantum flux capacitor"])
        reasons = {s["term"]: s["reason"] for s in res["skipped"]}
        self.assertEqual(reasons.get("x"), "low_score")
        self.assertEqual(res["version"], 1)

    def test_acquire_invalid(self):
        res = self.store.acquire(
            ["", "   ", "y" * 81],
            module="partner",
            category="robotic",
            source="test",
        )
        self.assertEqual(res["added"], [])
        self.assertEqual(
            [s["reason"] for s in res["skipped"]], ["invalid"] * 3
        )

    def test_exact_and_fuzzy_dedup(self):
        self.store.acquire(["color"], module="partner", category="robotic",
                           source="t")
        res = self.store.acquire(["color", "colors"], module="partner",
                                 category="robotic", source="t")
        self.assertEqual(res["added"], [])
        self.assertEqual(
            {s["term"]: s["reason"] for s in res["skipped"]},
            {"color": "duplicate", "colors": "duplicate"},
        )

    def test_version_bumps(self):
        self.assertEqual(self.store.version("partner"), 0)
        self.store.acquire(["color"], module="partner", category="robotic",
                           source="t")
        self.assertEqual(self.store.version("partner"), 1)
        res = self.store.acquire(
            ["colors", "tangerine dream"], module="partner",
            category="robotic", source="t",
        )
        self.assertEqual(res["added"], ["tangerine dream"])
        self.assertEqual(res["version"], 2)
        self.assertEqual(self.store.version("partner"), 2)

    def test_no_add_no_bump(self):
        self.store.acquire(["color"], module="partner", category="robotic",
                           source="t")
        res = self.store.acquire(["color"], module="partner",
                                 category="robotic", source="t")
        self.assertEqual(res["version"], 1)
        self.assertEqual(self.store.version("partner"), 1)

    def test_terms_for_ordered_by_score_desc(self):
        self.store.acquire(
            ["zebra crossing", "quantum flux capacitor"],
            module="partner",
            category="robotic",
            source="t",
            category_keywords=["quantum", "zebra"],
        )
        terms = self.store.terms_for("partner", "robotic")
        self.assertEqual(len(terms), 2)
        scores = [t["score"] for t in terms]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(all(set(t) == {"term", "score", "version"} for t in terms))

    def test_retire(self):
        self.store.acquire(["color", "tangerine dream"], module="partner",
                           category="robotic", source="t")
        self.assertTrue(self.store.retire("color", "partner", reason="stale"))
        terms = [t["term"] for t in self.store.terms_for("partner", "robotic")]
        self.assertNotIn("color", terms)
        self.assertIn("tangerine dream", terms)
        # retiring again (or unknown term) changes nothing
        self.assertFalse(self.store.retire("color", "partner"))
        self.assertFalse(self.store.retire("nope", "partner"))

    def test_stats(self):
        self.store.acquire(["color", "tangerine dream"], module="partner",
                           category="robotic", source="t")
        self.store.acquire(["zebra crossing"], module="partner",
                           category="domain", source="t")
        self.store.retire("color", "partner")
        stats = self.store.stats("partner")
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["by_status"], {"active": 2, "retired": 1})
        self.assertEqual(stats["by_category"], {"robotic": 2, "domain": 1})

    def test_threshold_respected(self):
        res = self.store.acquire(
            ["mediocre-ish phrase"],
            module="partner",
            category="robotic",
            source="t",
            threshold=0.99,
        )
        self.assertEqual(res["added"], [])


class MineCandidatesTest(unittest.TestCase):
    def test_quoted_spans_extracted(self):
        findings = [
            _finding('The paper calls the method "quantum flux" and it works.'),
            _finding("Another angle mentions 'silent running mode' briefly."),
        ]
        cands = mine_candidates(findings)
        self.assertIn("quantum flux", cands)
        self.assertIn("silent running mode", cands)

    def test_repeated_bigrams_extracted(self):
        findings = [
            _finding("The quantum flux capacitor enables time travel.", "a1"),
            _finding("Engineers rebuilt the quantum flux capacitor twice.", "a2"),
            _finding("Unrelated claim about dinner plans.", "a3"),
        ]
        cands = mine_candidates(findings)
        self.assertIn("quantum flux capacitor", cands)
        self.assertIn("quantum flux", cands)

    def test_title_case_phrases_extracted(self):
        cands = mine_candidates([_finding("Neural Daredevil beats the baseline.")])
        self.assertIn("neural daredevil", cands)

    def test_junk_not_extracted(self):
        findings = [
            _finding('See "https://example.com/stuff" for details.'),
            _finding("the and for with that this from have"),
        ]
        cands = mine_candidates(findings)
        self.assertEqual(cands, [])

    def test_dedup_and_cap(self):
        findings = [_finding('It says "echo phrase" loudly.', f"a{i}") for i in range(5)]
        cands = mine_candidates(findings, max_terms=1)
        self.assertEqual(cands, ["echo phrase"])

    def test_dict_findings_supported(self):
        cands = mine_candidates([{"angle": "a", "claim": 'It says "dict works".'}])
        self.assertIn("dict works", cands)


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        self.db.migrate()
        self.store = LexiconStore(SimpleNamespace(db=self.db))

    def tearDown(self):
        self.db.close()

    def test_acquire_from_findings(self):
        findings = [
            _finding(
                'The swarm agrees on "warm brevity" as the house style.',
                "style",
            ),
            _finding(
                "A second angle also praises warm brevity in replies.",
                "style2",
            ),
        ]
        res = acquire_from_findings(
            findings,
            self.store,
            module="partner",
            category="style",
            source="swarm",
            category_keywords=["warm", "brevity", "style"],
        )
        self.assertIn("warm brevity", res["added"])
        self.assertEqual(res["version"], 1)

    def test_dynamic_terms(self):
        self.store.acquire(["warm brevity", "color"], module="partner",
                           category="style", source="t")
        terms = dynamic_terms(self.db, "partner", "style")
        self.assertIsInstance(terms, tuple)
        self.assertIn("warm brevity", terms)
        self.assertEqual(dynamic_terms(self.db, "partner", "missing"), ())

    def test_dynamic_terms_never_raises(self):
        self.assertEqual(dynamic_terms(None, "partner", "style"), ())

        class Broken:
            def query(self, *a: object) -> list:
                raise RuntimeError("boom")

        self.assertEqual(dynamic_terms(Broken(), "partner", "style"), ())


class StyleHookTest(unittest.TestCase):
    TEXT = "Hello there. Please forward this to the team. Thank you."

    def test_extra_phrase_stripped_when_passed(self):
        out = strip_robotic(self.TEXT, extra_phrases=("please forward",))
        self.assertNotIn("forward", out.lower())
        self.assertIn("hello there", out.lower())

    def test_default_behavior_unchanged(self):
        self.assertEqual(strip_robotic(self.TEXT), self.TEXT)
        self.assertEqual(
            strip_robotic(self.TEXT, extra_phrases=None), self.TEXT
        )

    def test_extra_phrase_ignored_when_absent(self):
        out = strip_robotic("Nothing robotic here at all.", extra_phrases=("zzzqqq",))
        self.assertEqual(out, "Nothing robotic here at all.")

    def test_identity_leak_extra_phrases(self):
        verdict = identity_leak_check(
            "I am a machine, trust me.", extra_phrases=("i am a machine",)
        )
        self.assertFalse(verdict.ok)
        self.assertTrue(identity_leak_check("I am a machine, trust me.").ok)
        self.assertTrue(
            identity_leak_check("Hello friend.", extra_phrases=("zzzqqq",)).ok
        )


if __name__ == "__main__":
    unittest.main()
