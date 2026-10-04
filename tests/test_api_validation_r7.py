"""Round 7 / Area 1: API surface validation.

Fail-fast checks on the public API surface audited in R7A1:
provider ``chat()`` message validation, ``build_provider()`` kind
validation, control-parser input validation, and
``AccountCreator.create_account()`` input validation.
"""

from __future__ import annotations

import asyncio
import unittest

from nomorals.accounts import AccountCreator, CredentialVault
from nomorals.llm.base import Message, validate_messages
from nomorals.llm.providers import build_provider
from nomorals.llm.providers.mock import MockProvider
from nomorals.social.chat.control import (
    ControlCommand,
    detailed_help,
    list_catalog,
    parse_control,
)


class ValidateMessagesTests(unittest.TestCase):
    def test_accepts_nonempty_message_list(self):
        msgs = [Message.user("hi")]
        self.assertEqual(validate_messages(msgs), msgs)

    def test_accepts_tuple(self):
        msgs = (Message.user("hi"),)
        out = validate_messages(msgs)
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 1)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            validate_messages([])

    def test_rejects_none(self):
        with self.assertRaises(TypeError):
            validate_messages(None)

    def test_rejects_string(self):
        # A bare string is iterable but not a message sequence.
        with self.assertRaises(TypeError):
            validate_messages("hello")

    def test_rejects_non_message_elements(self):
        with self.assertRaises(TypeError):
            validate_messages([{"role": "user", "content": "hi"}])

    def test_mock_chat_validates(self):
        provider = MockProvider()
        with self.assertRaises(ValueError):
            provider.chat([])
        with self.assertRaises(TypeError):
            provider.chat("not a list")
        resp = provider.chat([Message.user("hello")])
        self.assertTrue(resp.ok)


class BuildProviderTests(unittest.TestCase):
    def test_rejects_none(self):
        with self.assertRaises(ValueError):
            build_provider(None)

    def test_rejects_blank(self):
        for bad in ("", "   "):
            with self.assertRaises(ValueError):
                build_provider(bad)

    def test_rejects_non_string(self):
        with self.assertRaises(ValueError):
            build_provider(123)

    def test_unknown_still_value_error(self):
        with self.assertRaises(ValueError):
            build_provider("definitely-not-a-provider")

    def test_whitespace_tolerant(self):
        provider = build_provider("  mock ")
        self.assertIsInstance(provider, MockProvider)


class ControlParseTests(unittest.TestCase):
    def test_none_is_not_a_command(self):
        self.assertIsNone(parse_control(None))

    def test_non_string_raises_type_error(self):
        with self.assertRaises(TypeError):
            parse_control(123)

    def test_empty_slash_is_help(self):
        self.assertEqual(parse_control("/").kind, "help")

    def test_valid_command(self):
        cmd = parse_control("/status")
        self.assertIsInstance(cmd, ControlCommand)
        self.assertEqual(cmd.kind, "status")

    def test_detailed_help_rejects_non_string(self):
        with self.assertRaises(TypeError):
            detailed_help(123)
        self.assertIn("/status", detailed_help(None))  # catalog tolerates None

    def test_list_catalog_rejects_non_string(self):
        with self.assertRaises(TypeError):
            list_catalog(123)


def _creator() -> AccountCreator:
    from nomorals.storage.db import Database

    return AccountCreator(
        CredentialVault(Database(":memory:"), master_passphrase="test-pass")
    )


class CreateAccountValidationTests(unittest.TestCase):
    def test_rejects_bad_service(self):
        for bad in (None, "", "   ", 123):
            with self.assertRaises(ValueError, msg=f"service={bad!r}"):
                asyncio.run(_creator().create_account(bad))

    def test_rejects_non_string_username(self):
        with self.assertRaises(TypeError):
            asyncio.run(_creator().create_account("github", username=123))

    def test_rejects_non_string_email(self):
        with self.assertRaises(TypeError):
            asyncio.run(_creator().create_account("github", email=5))

    def test_rejects_non_string_password(self):
        with self.assertRaises(TypeError):
            asyncio.run(_creator().create_account("github", password=b"bytes"))


if __name__ == "__main__":
    unittest.main()
