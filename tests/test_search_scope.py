"""Dual-scope awareness: Nigeria + US served together, or not at all.

Covers detect_scope / scope_relevance / regional_variants — the pure,
deterministic layer, no I/O.
"""

from __future__ import annotations

import unittest

from nomorals.agents.search.scope import (
    detect_scope,
    regional_variants,
    scope_label_for_variant,
    scope_relevance,
)


class DetectScopeTests(unittest.TestCase):
    def test_nigeria_signals(self) -> None:
        for q in ("bitcoin price in Nigeria", "best banks in Lagos",
                  "naira exchange rate today", "Yoruba movies 2026"):
            self.assertEqual(detect_scope(q), "ng", q)

    def test_us_signals(self) -> None:
        for q in ("mortgage rates in the United States", "best credit cards USA",
                  "dollar inflation 2026"):
            self.assertEqual(detect_scope(q), "us", q)

    def test_both_signals(self) -> None:
        self.assertEqual(detect_scope("Nigeria vs USA fintech regulation"), "both")

    def test_auto_for_plain_questions(self) -> None:
        for q in ("what is photosynthesis", "how do black holes form",
                  "best budget laptops", "bitcoin price"):
            self.assertEqual(detect_scope(q), "auto", q)


class ScopeRelevanceTests(unittest.TestCase):
    def test_factoid_scores_zero(self) -> None:
        self.assertEqual(scope_relevance("what is photosynthesis"), 0.0)
        self.assertEqual(scope_relevance(""), 0.0)

    def test_price_question_is_relevant(self) -> None:
        self.assertGreaterEqual(scope_relevance("bitcoin price today"), 0.40)

    def test_news_question_is_relevant(self) -> None:
        self.assertGreaterEqual(scope_relevance("latest election news"), 0.40)

    def test_jobs_question_is_relevant(self) -> None:
        self.assertGreaterEqual(scope_relevance("remote jobs for designers"), 0.40)

    def test_local_intent_is_strongly_relevant(self) -> None:
        self.assertGreaterEqual(scope_relevance("plumbers near me"), 0.60)

    def test_categories_stack_dynamically(self) -> None:
        # finance (0.50) + news (0.60) → capped at 1.0, and strictly more
        # than finance alone — scoring is additive, not a boolean switch
        self.assertGreater(
            scope_relevance("latest bitcoin price news"),
            scope_relevance("bitcoin price"),
        )
        self.assertLessEqual(scope_relevance("latest bitcoin price news"), 1.0)


class RegionalVariantsTests(unittest.TestCase):
    def test_factoid_stays_single_shot(self) -> None:
        variants = regional_variants("what is photosynthesis")
        self.assertEqual(len(variants), 1)
        self.assertEqual(variants[0], ("what is photosynthesis", "global"))

    def test_scope_sensitive_fans_out_to_both_regions(self) -> None:
        variants = regional_variants("best budget smartphones")
        by_label = {}
        for v, label in variants:
            by_label.setdefault(label, []).append(v)
        self.assertIn("global", by_label)
        self.assertTrue(any("Nigeria" in v or "Lagos" in v for v in by_label.get("ng", [])),
                        variants)
        self.assertTrue(any("United States" in v or " US" in v for v in by_label.get("us", [])),
                        variants)

    def test_explicit_nigeria_query_gets_no_us_noise(self) -> None:
        variants = regional_variants("best banks in Nigeria")
        labels = {label for _v, label in variants}
        self.assertNotIn("us", labels)
        self.assertTrue(any("nigeria" in v.lower() for v, _l in variants))

    def test_explicit_us_query_gets_no_ng_noise(self) -> None:
        variants = regional_variants("mortgage rates United States")
        labels = {label for _v, label in variants}
        self.assertNotIn("ng", labels)

    def test_scope_override_global(self) -> None:
        variants = regional_variants("bitcoin price", scope="global")
        self.assertEqual(len(variants), 1)

    def test_scope_override_ng(self) -> None:
        variants = regional_variants("bitcoin price", scope="ng")
        labels = {label for _v, label in variants}
        self.assertNotIn("us", labels)

    def test_max_variants_cap(self) -> None:
        variants = regional_variants("best budget smartphones", max_variants=3)
        self.assertLessEqual(len(variants), 3)

    def test_empty_query(self) -> None:
        self.assertEqual(regional_variants(""), [])

    def test_no_duplicate_variants(self) -> None:
        variants = regional_variants("cheapest flights")
        lowered = [v.lower() for v, _l in variants]
        self.assertEqual(len(lowered), len(set(lowered)))


class ScopeLabelTests(unittest.TestCase):
    def test_labels(self) -> None:
        self.assertEqual(scope_label_for_variant("q", "bitcoin price Nigeria"), "ng")
        self.assertEqual(scope_label_for_variant("q", "bitcoin price Lagos"), "ng")
        self.assertEqual(scope_label_for_variant("q", "bitcoin price United States"), "us")
        self.assertEqual(scope_label_for_variant("q", "bitcoin price"), "global")


if __name__ == "__main__":
    unittest.main()
