"""Owner seal: ingrained identities + passphrase proof baked into code."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.power import PowerMode
from nomorals.core import owner
from nomorals.core.config import load_settings
from nomorals.core.owner import (
    MIN_PASSPHRASE_LEN,
    OWNER_IDENTITIES,
    bake_seal,
    is_owner_identity,
    make_seal,
    seal_configured,
    verify_owner,
    verify_passphrase,
)

PASSPHRASE = "correct horse battery staple moon river 99"


def _settings(tmp: str) -> Any:
    return load_settings(overrides={"home": tmp, "partner.platforms": "local",
                                    "chat.local_enabled": "true"})


def _make_context(tmp: str) -> Any:
    return build_context(_settings(tmp), with_executor=False, with_tools=False,
                         with_router=False, with_memory=False)


class IdentityTests(unittest.TestCase):
    def test_ingrained_names_present(self) -> None:
        self.assertEqual(OWNER_IDENTITIES,
                         ("oluwacutyp", "peace", "peacethefirt1"))

    def test_exact_and_case_insensitive(self) -> None:
        self.assertTrue(is_owner_identity("Oluwacutyp"))
        self.assertTrue(is_owner_identity("PEACE"))
        self.assertTrue(is_owner_identity("  peacethefirt1  "))

    def test_unknown_and_empty_rejected(self) -> None:
        self.assertFalse(is_owner_identity("stranger"))
        self.assertFalse(is_owner_identity(""))
        self.assertFalse(is_owner_identity("oluwacutypx"))


class SealTests(unittest.TestCase):
    def setUp(self) -> None:
        self._old = owner.PASSPHRASE_SEAL

    def tearDown(self) -> None:
        owner.PASSPHRASE_SEAL = self._old

    def test_make_seal_rejects_weak(self) -> None:
        with self.assertRaises(ValueError):
            make_seal("")
        with self.assertRaises(ValueError):
            make_seal("x" * (MIN_PASSPHRASE_LEN - 1))

    def test_round_trip(self) -> None:
        owner.PASSPHRASE_SEAL = make_seal(PASSPHRASE)
        self.assertTrue(seal_configured())
        self.assertTrue(verify_passphrase(PASSPHRASE))
        self.assertFalse(verify_passphrase(PASSPHRASE + "wrong"))
        self.assertFalse(verify_passphrase(""))

    def test_no_seal_configured(self) -> None:
        owner.PASSPHRASE_SEAL = ""
        self.assertFalse(seal_configured())
        self.assertFalse(verify_passphrase(PASSPHRASE))

    def test_verify_owner_needs_both(self) -> None:
        owner.PASSPHRASE_SEAL = make_seal(PASSPHRASE)
        self.assertTrue(verify_owner("peace", PASSPHRASE))
        self.assertFalse(verify_owner("peace", "wrong passphrase here 12345"))
        self.assertFalse(verify_owner("stranger", PASSPHRASE))
        self.assertFalse(verify_owner("", PASSPHRASE))

    def test_seals_differ_per_salt(self) -> None:
        self.assertNotEqual(make_seal(PASSPHRASE), make_seal(PASSPHRASE))


class BakeTests(unittest.TestCase):
    def test_bake_rewrites_constant_in_copy(self) -> None:
        tmp = tempfile.mkdtemp(prefix="nm-owner-")
        try:
            src = shutil.copy(owner.__file__, tmp + "/owner_copy.py")
            seal = make_seal(PASSPHRASE)
            written = bake_seal(seal, src)
            self.assertEqual(written, src)
            with open(src, encoding="utf-8") as fh:
                text = fh.read()
            self.assertIn(f'PASSPHRASE_SEAL = "{seal}"', text)
            self.assertNotIn(PASSPHRASE, text)  # secret never lands in code
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_bake_rejects_garbage(self) -> None:
        with self.assertRaises(ValueError):
            bake_seal("not-a-seal")


class PowerUnlockViaSealTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="nm-owner-")
        self.context = _make_context(self.tmp.name)
        self.context.settings.partner.owner_key = ""
        self._old = owner.PASSPHRASE_SEAL
        owner.PASSPHRASE_SEAL = make_seal(PASSPHRASE)

    def tearDown(self) -> None:
        owner.PASSPHRASE_SEAL = self._old
        self.context.close()
        self.tmp.cleanup()

    def test_unlock_with_identity_and_passphrase(self) -> None:
        power = PowerMode(self.context)
        result = power.unlock(PASSPHRASE, actor="test", identity="peace")
        self.assertTrue(result["ok"], result)
        self.assertTrue(power.active)
        self.assertEqual(power.unlocked_by, "test")
        power.lock(actor="test")
        self.assertFalse(power.active)

    def test_unlock_rejects_wrong_identity(self) -> None:
        power = PowerMode(self.context)
        result = power.unlock(PASSPHRASE, actor="test", identity="stranger")
        self.assertFalse(result["ok"])
        self.assertFalse(power.active)

    def test_unlock_rejects_wrong_passphrase(self) -> None:
        power = PowerMode(self.context)
        result = power.unlock("wrong passphrase entirely 12345",
                              actor="test", identity="oluwacutyp")
        self.assertFalse(result["ok"])
        self.assertFalse(power.active)

    def test_env_key_still_works_without_identity(self) -> None:
        self.context.settings.partner.owner_key = "env-key"
        power = PowerMode(self.context)
        result = power.unlock("env-key", actor="test")
        self.assertTrue(result["ok"], result)

    def test_nothing_configured_denies(self) -> None:
        owner.PASSPHRASE_SEAL = ""
        power = PowerMode(self.context)
        result = power.unlock("anything", actor="test", identity="peace")
        self.assertFalse(result["ok"])
        self.assertIn("no unlock is configured", result["message"])


if __name__ == "__main__":
    unittest.main()
