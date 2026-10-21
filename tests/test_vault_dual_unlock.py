"""Pass 4: the credential vault opens with ANY owner credential.

The owner's ruling: the vault and provisioning authenticate via the
existing owner seal OR the power-mode key (both owner-set) — the legacy
NM_VAULT_PASSPHRASE becomes one accepted credential among them, not the
gatekeeper.  Acceptance is real cryptography (master key wrapped per
credential, HMAC-authenticated unwrap), wrong credentials fail closed,
and locked errors name every source that was checked.
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from nomorals.accounts.vault import (
    CredentialVault,
    VaultLockedError,
    open_vault,
)
from nomorals.core import owner as owner_mod
from nomorals.storage.db import Database

SEAL_SECRET = "seal-passphrase-for-tests-0123456789"
WRONG_SEAL = "wrong-seal-passphrase-9876543210"
POWER_KEY = "power-key-for-tests"
LEGACY_PASS = "legacy-vault-passphrase"

_AUDIT_DDL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    ts REAL NOT NULL,
    actor TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    capability TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL DEFAULT '',
    detail TEXT NOT NULL DEFAULT '{}'
)
"""


def _db() -> Database:
    return Database(":memory:")


def _seal_patched():
    return mock.patch.object(
        owner_mod, "PASSPHRASE_SEAL", owner_mod.make_seal(SEAL_SECRET))


class _EnvGuard(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.get("NM_VAULT_PASSPHRASE")
        os.environ.pop("NM_VAULT_PASSPHRASE", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("NM_VAULT_PASSPHRASE", None)
        else:
            os.environ["NM_VAULT_PASSPHRASE"] = self._saved


class FreshVaultDualUnlockTests(_EnvGuard):
    def test_seal_only_unlocks_fresh_vault(self):
        db = _db()
        with _seal_patched():
            vault = CredentialVault.unlock(db, seal_secret=SEAL_SECRET)
            vault.store("huggingface", "Cutyp", "hf-token-1")
            self.assertEqual(vault.wrapped_credential_types(),
                             ["owner_seal"])
            again = CredentialVault.unlock(db, seal_secret=SEAL_SECRET)
            self.assertEqual(
                again.get("huggingface", "Cutyp").password, "hf-token-1")

    def test_power_key_only_unlocks_fresh_vault(self):
        db = _db()
        vault = CredentialVault.unlock(db, power_key=POWER_KEY)
        vault.store("github", "Cutyp", "gh-token-1")
        self.assertEqual(vault.wrapped_credential_types(), ["power_key"])
        again = CredentialVault.unlock(db, power_key=POWER_KEY)
        self.assertEqual(again.get("github", "Cutyp").password, "gh-token-1")

    def test_open_vault_gathers_power_key_from_context(self):
        db = _db()
        context = SimpleNamespace(
            db=db, extras={},
            settings=SimpleNamespace(
                partner=SimpleNamespace(owner_key=POWER_KEY)))
        vault = open_vault(db, context=context)
        vault.store("svc", "user", "secret-1")
        again = open_vault(db, context=context)
        self.assertEqual(again.get("svc", "user").password, "secret-1")


class LegacyMigrationTests(_EnvGuard):
    def _legacy_vault(self, db: Database) -> CredentialVault:
        vault = CredentialVault(db, master_passphrase=LEGACY_PASS)
        vault.store("huggingface", "Cutyp", "hf-token-legacy")
        return vault

    def test_passphrase_only_still_unlocks_legacy_vault(self):
        db = _db()
        self._legacy_vault(db)
        vault = CredentialVault.unlock(db, passphrase=LEGACY_PASS)
        self.assertEqual(
            vault.get("huggingface", "Cutyp").password, "hf-token-legacy")

    def test_migration_wraps_every_available_credential(self):
        db = _db()
        self._legacy_vault(db)
        with _seal_patched():
            vault = CredentialVault.unlock(
                db, passphrase=LEGACY_PASS, power_key=POWER_KEY,
                seal_secret=SEAL_SECRET)
            self.assertEqual(
                vault.wrapped_credential_types(),
                ["owner_seal", "passphrase", "power_key"])
            # Each owner credential now opens the vault on its own.
            by_power = CredentialVault.unlock(db, power_key=POWER_KEY)
            self.assertEqual(
                by_power.get("huggingface", "Cutyp").password,
                "hf-token-legacy")
            by_seal = CredentialVault.unlock(db, seal_secret=SEAL_SECRET)
            self.assertEqual(
                by_seal.get("huggingface", "Cutyp").password,
                "hf-token-legacy")

    def test_vault_created_under_passphrase_accepts_seal_after_wrap(self):
        """The named migration case: passphrase-created vault, one
        wrap-on-unlock, then the seal alone opens it."""
        db = _db()
        self._legacy_vault(db)
        with _seal_patched():
            CredentialVault.unlock(db, passphrase=LEGACY_PASS,
                                   seal_secret=SEAL_SECRET)
            by_seal = CredentialVault.unlock(db, seal_secret=SEAL_SECRET)
            self.assertEqual(
                by_seal.get("huggingface", "Cutyp").password,
                "hf-token-legacy")

    def test_legacy_vault_without_wraps_cannot_open_via_power_key_yet(self):
        """Honest limit, named in the error: a pre-wrap vault's master
        is derived from its passphrase — an owner credential cannot
        derive it until one passphrase unlock creates the wraps."""
        db = _db()
        self._legacy_vault(db)
        with self.assertRaises(VaultLockedError) as ctx:
            CredentialVault.unlock(db, power_key=POWER_KEY)
        self.assertIn("predates multi-credential unlock",
                      str(ctx.exception))


class WrongCredentialTests(_EnvGuard):
    def _wrapped_db(self) -> Database:
        db = _db()
        vault = CredentialVault(db, master_passphrase=LEGACY_PASS)
        vault.store("svc", "user", "secret-1")
        with _seal_patched():
            CredentialVault.unlock(
                db, passphrase=LEGACY_PASS, power_key=POWER_KEY,
                seal_secret=SEAL_SECRET)
        return db

    def test_wrong_passphrase_fails_closed(self):
        db = self._wrapped_db()
        with self.assertRaises(VaultLockedError):
            CredentialVault.unlock(db, passphrase="not-the-passphrase")
        # …and the vault is unharmed for the right credential.
        vault = CredentialVault.unlock(db, passphrase=LEGACY_PASS)
        self.assertEqual(vault.get("svc", "user").password, "secret-1")

    def test_wrong_power_key_fails_closed(self):
        db = self._wrapped_db()
        with self.assertRaises(VaultLockedError):
            CredentialVault.unlock(db, power_key="not-the-power-key")

    def test_wrong_seal_secret_fails_closed(self):
        db = self._wrapped_db()
        with _seal_patched():
            with self.assertRaises(VaultLockedError) as ctx:
                CredentialVault.unlock(db, seal_secret=WRONG_SEAL)
        self.assertIn("failed verification", str(ctx.exception))

    def test_wrong_passphrase_on_legacy_vault_fails_closed(self):
        db = _db()
        vault = CredentialVault(db, master_passphrase=LEGACY_PASS)
        vault.store("svc", "user", "secret-1")
        with self.assertRaises(VaultLockedError):
            CredentialVault.unlock(db, passphrase="not-the-passphrase")


class LockedMessageTests(_EnvGuard):
    def test_locked_error_names_every_source_checked(self):
        db = _db()
        with self.assertRaises(VaultLockedError) as ctx:
            CredentialVault.unlock(db)
        message = str(ctx.exception)
        self.assertIn("owner seal", message)
        self.assertIn("power key", message)
        self.assertIn("NM_PARTNER_OWNER_KEY", message)
        self.assertIn("NM_VAULT_PASSPHRASE", message)
        self.assertGreaterEqual(len(ctx.exception.checks), 3)

    def test_seal_secret_presented_but_no_seal_baked(self):
        # The repo checkout carries no baked seal: presenting one is
        # reported as "not configured", never silently tried.
        db = _db()
        with self.assertRaises(VaultLockedError) as ctx:
            CredentialVault.unlock(db, seal_secret=SEAL_SECRET)
        self.assertIn("not configured", str(ctx.exception))


class AuditTests(_EnvGuard):
    def test_audit_records_credential_type_never_value(self):
        db = _db()
        db.execute(_AUDIT_DDL)
        with _seal_patched():
            CredentialVault.unlock(db, seal_secret=SEAL_SECRET,
                                   actor="test")
            with self.assertRaises(VaultLockedError):
                CredentialVault.unlock(db, power_key="wrong-key",
                                       actor="test")
        rows = db.query("SELECT action, detail FROM audit_log")
        self.assertTrue(rows)
        joined = " ".join(f"{r['action']} {r['detail']}" for r in rows)
        self.assertIn("vault.unlock", joined)
        self.assertIn("owner_seal", joined)
        self.assertNotIn(SEAL_SECRET, joined)
        self.assertNotIn("wrong-key", joined)


class RotationTests(_EnvGuard):
    def test_change_passphrase_refreshes_wraps(self):
        db = _db()
        vault = CredentialVault.unlock(db, passphrase=LEGACY_PASS)
        vault.store("svc", "user", "secret-1")
        vault.change_passphrase("a-brand-new-passphrase")
        reopened = CredentialVault.unlock(
            db, passphrase="a-brand-new-passphrase")
        self.assertEqual(reopened.get("svc", "user").password, "secret-1")
        with self.assertRaises(VaultLockedError):
            CredentialVault.unlock(db, passphrase=LEGACY_PASS)


class ProvisionGateTests(_EnvGuard):
    """The reported failure: /provision died at the vault gate when
    NM_VAULT_PASSPHRASE was unset.  With a power key configured (or a
    seal presented), the gate opens."""

    def _runtime(self, db: Database, context: SimpleNamespace):
        from nomorals.agents.partner.runtime_system import (
            RuntimeSystemMixin,
        )

        runtime = RuntimeSystemMixin.__new__(RuntimeSystemMixin)
        runtime.context = context
        return runtime

    def test_connectors_vault_opens_with_power_key_no_env_passphrase(self):
        db = _db()
        context = SimpleNamespace(
            db=db, extras={},
            settings=SimpleNamespace(
                partner=SimpleNamespace(owner_key=POWER_KEY)))
        runtime = self._runtime(db, context)
        vault = runtime._connectors_vault()
        vault.store("huggingface", "Cutyp", "hf-token-1",
                    credential_type="api_key")
        # Provisioning's own read path runs on the opened vault.
        from nomorals.accounts.provisioning import provisioned_status

        rows = provisioned_status(vault)
        self.assertTrue(any(r["service"] == "huggingface" for r in rows))
        row = next(r for r in rows if r["service"] == "huggingface")
        self.assertTrue(row["has_key"])

    def test_connectors_vault_opens_with_presented_seal(self):
        db = _db()
        context = SimpleNamespace(
            db=db, extras={"vault_seal_secret": SEAL_SECRET},
            settings=SimpleNamespace(
                partner=SimpleNamespace(owner_key="")))
        runtime = self._runtime(db, context)
        with _seal_patched():
            vault = runtime._connectors_vault()
            vault.store("huggingface", "Cutyp", "hf-token-1")
            self.assertEqual(
                vault.get("huggingface", "Cutyp").password, "hf-token-1")

    def test_connectors_vault_locked_message_names_sources(self):
        db = _db()
        context = SimpleNamespace(
            db=db, extras={},
            settings=SimpleNamespace(
                partner=SimpleNamespace(owner_key="")))
        runtime = self._runtime(db, context)
        from nomorals.agents.partner.runtime_system import (
            _ConnectorsChatError,
        )

        with self.assertRaises(_ConnectorsChatError) as ctx:
            runtime._connectors_vault()
        message = str(ctx.exception)
        self.assertIn("owner seal", message)
        self.assertIn("power key", message)
        self.assertIn("NM_VAULT_PASSPHRASE", message)


if __name__ == "__main__":
    unittest.main()
