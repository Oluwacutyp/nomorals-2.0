"""Tests for the user-supplied integrations: proxydb proxy source,
mail.tm + 1secmail temp-mail providers, and the simcodes temp-SMS
provider.  Network-touching tests are skipped unless NM_RUN_INTEGRATION
is set (offline suite stays offline).
"""
import os
import unittest

INTEGRATION = bool(os.environ.get("NM_RUN_INTEGRATION"))


class TestProxydbSource(unittest.TestCase):
    def test_proxydb_in_builtin_sources(self):
        from nomorals.tools.proxysources import BUILT_IN_SOURCES
        names = [s[0] for s in BUILT_IN_SOURCES]
        self.assertIn("proxydb-socks5", names)
        entry = next(s for s in BUILT_IN_SOURCES if s[0] == "proxydb-socks5")
        self.assertEqual(entry[2], "html")
        self.assertIn("proxydb.net", entry[1])
        self.assertIn("socks5", entry[1])

    def test_source_registry_loads(self):
        import tempfile
        from nomorals.tools.proxysources import SourceRegistry
        with tempfile.TemporaryDirectory() as d:
            reg = SourceRegistry(path=f"{d}/sources.json")
            names = [s[0] for s in reg.sources()]
        self.assertIn("proxydb-socks5", names)


class TestTempMailProviders(unittest.TestCase):
    def test_providers_registered(self):
        from nomorals.accounts.creator import AccountCreator
        self.assertIn("mailtm", AccountCreator.DISPOSABLE_EMAIL_SERVICES)
        self.assertIn("1secmail", AccountCreator.DISPOSABLE_EMAIL_SERVICES)

    def test_dispatcher_knows_providers(self):
        import asyncio
        import inspect
        from nomorals.accounts.creator import AccountCreator
        src = inspect.getsource(AccountCreator.create_email_account)
        self.assertIn('"mailtm"', src)
        self.assertIn('"1secmail"', src)

    def test_inbox_helpers_exist(self):
        from nomorals.accounts.creator import AccountCreator
        for m in ("mailtm_inbox", "mailtm_read", "onec_inbox", "onec_read",
                  "check_disposable_inbox", "_guerrilla_inbox",
                  "get_temp_number", "poll_sms_code"):
            self.assertTrue(hasattr(AccountCreator, m), m)

    @unittest.skipUnless(INTEGRATION, "needs network")
    def test_mailtm_full_cycle(self):
        import asyncio
        from unittest.mock import MagicMock
        from nomorals.accounts.creator import AccountCreator
        vault = MagicMock()
        vault.store.side_effect = lambda **kw: MagicMock(**kw)
        creator = AccountCreator(vault=vault)

        async def go():
            acc = await creator.create_email_account(provider="mailtm")
            return acc
        acc = asyncio.run(go())
        self.assertEqual(acc.status, "created")
        self.assertIn("@", acc.email)
        # inbox should be readable (empty is fine)
        msgs = creator.mailtm_inbox(acc.email, acc.password)
        self.assertIsInstance(msgs, list)


class TestTempSms(unittest.TestCase):
    def test_provider_registry(self):
        from nomorals.accounts import temp_sms
        self.assertIn("simcodes", temp_sms.PROVIDERS)
        p = temp_sms.get_provider("simcodes")
        self.assertEqual(p.name, "simcodes")

    def test_unknown_provider(self):
        from nomorals.accounts import temp_sms
        with self.assertRaises(ValueError):
            temp_sms.get_provider("nonexistent")

    def test_code_extraction(self):
        from nomorals.accounts.temp_sms import SmsMessage
        m = SmsMessage(sender="123", body="Your code is 847392. Don't share.")
        self.assertEqual(m.code, "847392")
        m2 = SmsMessage(sender="123", body="244 340 is your Instagram code")
        # spaced codes: first try strips spaces -> "244340" (6 digits)
        self.assertTrue(m2.code in ("244340", "244", "340"))

    def test_message_dataclass(self):
        from nomorals.accounts.temp_sms import TempNumber
        n = TempNumber(masked="+1508893****", inbox_id="23655",
                       country="us", provider="simcodes")
        self.assertEqual(n.inbox_id, "23655")

    @unittest.skipUnless(INTEGRATION, "needs network")
    def test_simcodes_live(self):
        from nomorals.accounts.temp_sms import get_provider
        p = get_provider("simcodes")
        nums = p.list_numbers("us", limit=3)
        self.assertTrue(len(nums) > 0)
        self.assertTrue(nums[0].inbox_id)
        msgs = p.get_messages(nums[0], limit=5)
        self.assertIsInstance(msgs, list)


if __name__ == "__main__":
    unittest.main()
