"""PlatformLimits descriptor tests (Step 0.5)."""

import unittest

from nomorals.social.chat.platform_limits import (
    PlatformLimits,
    all_limits,
    for_platform,
    register_limits,
)


class TestPlatformLimits(unittest.TestCase):
    def test_known_platform_values(self) -> None:
        tg = for_platform("telegram")
        self.assertEqual(tg.max_text, 4096)
        self.assertEqual(tg.max_caption, 1024)
        self.assertEqual(tg.max_file_mb, 50)
        self.assertTrue(tg.supports_buttons)
        self.assertTrue(tg.supports_html)
        self.assertEqual(tg.cost_model, "free")

        self.assertEqual(for_platform("telegram-bot").supports_media_edit, True)
        self.assertEqual(for_platform("telegram").supports_media_edit, False)
        self.assertEqual(for_platform("discord").max_text, 2000)
        self.assertEqual(for_platform("whatsapp").max_text, 4000)
        sms = for_platform("sms")
        self.assertEqual(sms.max_text, 160)
        self.assertEqual(sms.cost_model, "per_segment")
        self.assertFalse(sms.gateway_chunks)

    def test_aliases_resolve(self) -> None:
        self.assertEqual(for_platform("tgbot").platform, "telegram-bot")
        self.assertEqual(for_platform("tg").platform, "telegram")
        self.assertEqual(for_platform("wa").platform, "whatsapp")
        self.assertEqual(for_platform("twilio").platform, "sms")
        self.assertEqual(for_platform("console").platform, "local")
        self.assertEqual(for_platform("TELEGRAM").platform, "telegram")

    def test_unknown_platform_gets_conservative_default(self) -> None:
        row = for_platform("signal")
        self.assertEqual(row.platform, "signal")
        self.assertEqual(row.max_text, 2000)
        self.assertFalse(row.supports_buttons)

    def test_register_limits_adds_a_platform(self) -> None:
        try:
            register_limits(PlatformLimits(
                platform="test-platform", max_text=1234, max_caption=100,
                max_file_mb=5, supports_buttons=False, supports_html=False,
                supports_media_edit=False, rate_per_hour=0,
                cost_model="free",
            ))
            self.assertEqual(for_platform("test-platform").max_text, 1234)
        finally:
            # Test-only cleanup: drop the row added above.
            from nomorals.social.chat.platform_limits import _LIMITS

            _LIMITS.pop("test-platform", None)

    def test_rows_are_frozen(self) -> None:
        row = for_platform("telegram")
        with self.assertRaises(Exception):  # FrozenInstanceError
            row.max_text = 1  # type: ignore[misc]

    def test_table_covers_the_known_adapters(self) -> None:
        names = set(all_limits())
        for expected in ("telegram", "telegram-bot", "whatsapp", "discord",
                         "sms", "web", "local", "webhook"):
            self.assertIn(expected, names)


if __name__ == "__main__":
    unittest.main()
