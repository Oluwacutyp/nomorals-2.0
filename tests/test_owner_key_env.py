"""NM_PARTNER_OWNER_KEY must reach power.unlock; `nm owner seal` must refuse
unknown identities.

Regression tests for the "wrong key for both" trap: the .env owner key had
no config mapping/field (so power.unlock's env-key check could never match),
and `nm owner seal` accepted any identity (baking a seal that could never
verify, since verify_owner requires an ingrained identity).
"""

from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.power import power_mode_for
from nomorals.cmdline.commands.owner import _cmd_owner
from nomorals.core.config import load_settings

OWNER_KEY = "test-owner-key-12345"
PASSPHRASE = "correct horse battery staple moon river 99"


def _settings_with_env_key(tmp: str):
    return load_settings(
        env={"NM_PARTNER_OWNER_KEY": OWNER_KEY},
        overrides={"home": tmp, "partner.platforms": "local",
                   "partner.power_default_on": False},
        use_env_file=False,
    )


class OwnerKeyEnvTests(unittest.TestCase):
    def test_env_key_reaches_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings_with_env_key(tmp)
        self.assertEqual(settings.partner.owner_key, OWNER_KEY)

    def test_power_unlock_with_env_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = _settings_with_env_key(tmp)
            ctx = build_context(settings, with_executor=False, with_tools=False,
                                with_router=False, with_memory=False)
            power = power_mode_for(ctx)
            self.assertFalse(power.active)
            result = power.unlock(OWNER_KEY, actor="test")
            self.assertTrue(result["ok"], result)
            self.assertTrue(power.active)
            # A wrong key still fails.
            power.lock(actor="test")
            bad = power.unlock("not-the-key", actor="test")
            self.assertFalse(bad["ok"])
            self.assertIn("wrong key", bad["message"])


class SealIdentityTests(unittest.TestCase):
    def _run_seal(self, identity: str):
        args = argparse.Namespace(owner_action="seal")
        with mock.patch("builtins.input", return_value=identity), \
                mock.patch("getpass.getpass", return_value=PASSPHRASE), \
                mock.patch("nomorals.core.owner.bake_seal",
                           return_value="/fake/owner.py") as bake:
            return _cmd_owner(args, context=None), bake

    def test_seal_rejects_unknown_identity(self) -> None:
        rc, bake = self._run_seal("stranger")
        self.assertEqual(rc, 1)
        bake.assert_not_called()

    def test_seal_accepts_ingrained_identity(self) -> None:
        rc, bake = self._run_seal("peace")
        self.assertEqual(rc, 0)
        bake.assert_called_once()


if __name__ == "__main__":
    unittest.main()
