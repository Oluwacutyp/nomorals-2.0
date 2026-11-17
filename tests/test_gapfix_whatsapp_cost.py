"""Gapfix R3-51 failing-first: surface the WhatsApp transport cost
comparison on the live /cost seam.

The comparison (social/chat/whatsapp_transport_cost.compare) has zero
production callers; the only live cost surface is /cost (runtime.py:9321
-> llm/cost_display.control_cost). The whatsapp subverb is routed in the
runtime's cost branch (L5 -> L4, downward) because llm/ (L3) may not
import social/ (L4).
"""

import tempfile
import unittest

from nomorals.social.chat.whatsapp_transport_cost import (
    CLOUD_CATEGORIES,
    whatsapp_cost_reply,
)
from nomorals.social.leads import _whatsapp_cost_kobo
from nomorals.social.whatsapp_cost import (
    RATE_MARKETING_KOBO,
    RATE_SERVICE_KOBO,
    CostTracker,
)


class WhatsappReplyTests(unittest.TestCase):
    def test_bare_whatsapp_names_both_estimates_and_recommendation(self):
        out = whatsapp_cost_reply("whatsapp")
        self.assertIn("cloud", out.lower())
        self.assertIn("₦14", out)          # 1 service conversation
        self.assertIn("baileys", out.lower())
        self.assertIn("₦0", out)
        self.assertIn("use baileys", out.lower())
        self.assertIn("warm", out.lower())

    def test_marketing_ten_conversations_recommends_cloud_with_price(self):
        out = whatsapp_cost_reply("whatsapp marketing 10")
        self.assertIn("cloud-api", out)
        self.assertIn("₦870", out)         # 10 x $0.058 x ₦1500
        self.assertIn("₦0", out)           # baileys leg still shown
        self.assertIn("template", out.lower())

    def test_cold_outreach_recommends_cloud_for_safety(self):
        out = whatsapp_cost_reply("whatsapp service 3 cold")
        self.assertIn("cloud-api", out)
        self.assertIn("ban", out.lower())
        self.assertIn("₦42", out)          # 3 x ₦14 service conversations

    def test_unknown_category_refuses_honestly_naming_valid_categories(self):
        out = whatsapp_cost_reply("whatsapp bogus")
        self.assertIn("unknown", out.lower())
        for cat in CLOUD_CATEGORIES:
            self.assertIn(cat, out)
        self.assertNotIn("₦", out)         # no invented number anywhere

    def test_week_branch_reads_the_real_ledger(self):
        tracker = CostTracker(db_path=tempfile.mktemp(suffix=".db"))
        tracker.track("+2348000000001", "service", client="default")
        tracker.track("+2348000000002", "service", client="default")
        out = whatsapp_cost_reply("whatsapp week", tracker=tracker)
        self.assertIn("₦28", out)
        self.assertIn("2 messages", out)

    def test_week_branch_empty_ledger_is_zero_not_unknown(self):
        tracker = CostTracker(db_path=tempfile.mktemp(suffix=".db"))
        out = whatsapp_cost_reply("whatsapp week", tracker=tracker)
        self.assertIn("₦0", out)
        self.assertIn("0 messages", out)

    def test_card_branch_shows_the_rate_card(self):
        out = whatsapp_cost_reply("whatsapp card")
        for cat in CLOUD_CATEGORIES:
            self.assertIn(cat, out)
        self.assertIn("$0.058", out)       # marketing USD rate

    def test_reply_names_the_card_vintage(self):
        out = whatsapp_cost_reply("whatsapp")
        self.assertIn("Oct 2026", out)


class LeadsSingleRateCardTests(unittest.TestCase):
    def test_leads_prices_from_the_single_rate_card(self):
        self.assertEqual(_whatsapp_cost_kobo(is_marketing=True),
                         RATE_MARKETING_KOBO)
        self.assertEqual(_whatsapp_cost_kobo(is_marketing=False),
                         RATE_SERVICE_KOBO)


class UsageTests(unittest.TestCase):
    def test_cost_usage_mentions_whatsapp(self):
        from nomorals.llm.cost_display import _usage
        self.assertIn("whatsapp", _usage().lower())

    def test_cost_catalog_allows_the_whatsapp_tail(self):
        from nomorals.social.chat.control import CONTROL_COMMANDS
        min_args, max_args = CONTROL_COMMANDS["cost"]
        self.assertEqual(min_args, 0)
        self.assertTrue(max_args is None or max_args >= 4)


if __name__ == "__main__":
    unittest.main()
