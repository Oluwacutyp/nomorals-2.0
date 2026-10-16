"""Tests for nomorals.scam.signals (general scam signals)."""

import unittest

from nomorals.property.scam import check_listing
from nomorals.scam.signals import SIGNAL_DESCRIPTORS, assess, assess_listing

SCAMMY = (
    "Unbelievably cheap flat, far below market price! Act now, limited time only! "
    "Send money upfront and pay before viewing to secure it."
)
BENIGN = "Hello, how are you? Let's meet for coffee tomorrow and catch up."


class TestScamSignals(unittest.TestCase):
    def test_descriptors_are_data(self):
        kinds = {d["kind"] for d in SIGNAL_DESCRIPTORS}
        for expected in ("listing", "message", "payment", "impersonation",
                         "urgency", "too_good", "advance_fee", "off_platform"):
            self.assertIn(expected, kinds)
        for d in SIGNAL_DESCRIPTORS:
            self.assertIn("id", d)
            self.assertIn("kind", d)
            self.assertIn("description", d)
            self.assertIn("weight", d)
            self.assertIn("examples", d)
            self.assertGreater(len(d["examples"]), 0)

    def test_too_good_plus_urgency_plus_advance_fee_scores_high(self):
        result = assess(SCAMMY, kind="message")
        self.assertEqual(result["verdict"], "suspicious")
        self.assertGreaterEqual(result["score"], 0.6)
        self.assertIn("too_good_price", result["flags"])
        self.assertIn("urgency_pressure", result["flags"])
        self.assertIn("advance_fee", result["flags"])
        self.assertTrue(result["reasons"])

    def test_single_strong_signal_is_review_not_suspicious(self):
        result = assess("Please send money upfront to secure your slot.", kind="message")
        self.assertEqual(result["verdict"], "review")
        self.assertIn("advance_fee", result["flags"])

    def test_benign_scores_low(self):
        result = assess(BENIGN, kind="message")
        self.assertEqual(result["verdict"], "ok")
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["flags"], [])

    def test_evidence_counts(self):
        result = assess(BENIGN, kind="message", evidence=["advance_fee", "urgency_pressure"])
        self.assertEqual(result["verdict"], "review")
        self.assertIn("advance_fee", result["flags"])
        self.assertIn("urgency_pressure", result["flags"])
        # Unknown evidence is ignored, not fatal.
        result2 = assess(BENIGN, evidence=["not_a_descriptor"])
        self.assertEqual(result2["verdict"], "ok")

    def test_score_bounds(self):
        for text in (SCAMMY, BENIGN, "", "pay with gift card, you won a prize, act now"):
            result = assess(text)
            self.assertGreaterEqual(result["score"], 0.0)
            self.assertLessEqual(result["score"], 1.0)
            self.assertIn(result["verdict"], ("suspicious", "review", "ok"))

    def test_listing_delegation_merges_property_flags(self):
        listing = (
            "2br flat in Lekki, 500000 naira per year, unbelievably cheap. "
            "Pay before viewing. Pay with gift card, urgent, hurry!"
        )
        result = assess_listing(listing, check_listing_fn=check_listing)
        # Property module flags the price anomaly / payment / no-viewing.
        self.assertEqual(result["verdict"], "suspicious")
        self.assertIn("price_anomaly", result["flags"])
        self.assertIn("listing_verdict", result)
        self.assertIn("listing_score", result)
        self.assertTrue(result["reasons"])

    def test_listing_delegation_benign(self):
        listing = (
            "2br flat in Lekki, 4500000 naira per year. "
            "Viewing available on Saturday. Contact the agent to arrange."
        )
        result = assess_listing(listing, check_listing_fn=check_listing)
        self.assertEqual(result["verdict"], "ok")
        self.assertEqual(result["flags"], [])

    def test_listing_delegation_empty(self):
        result = assess_listing("", check_listing_fn=check_listing)
        self.assertIn(result["verdict"], ("ok", "review", "suspicious"))
        self.assertIn("listing_verdict", result)


if __name__ == "__main__":
    unittest.main()
