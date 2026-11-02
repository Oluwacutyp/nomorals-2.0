"""Pins for nomorals/accounts/vault.py (tail-116): ONE CONVERT.

Site 1194 — ``CredentialVault._rekey_blobs`` (CONVERTED in this
pass): the SELECT over the blob table was guarded by a blanket
``except Exception: return 0`` whose printed contract was only
"table may not exist". ``_rekey_blobs`` runs INSIDE
``change_passphrase``'s transaction, after the credentials have
been re-encrypted and before the key wraps are dropped and the
verifier rewritten — so any other query failure (lock, I/O, schema
drift) silently reported "0 blobs to rekey", the rotation
committed, and every session blob stayed encrypted under a retired
master key: permanent, silent data loss reported as a successful
rotation (probe-proven pre-fix: healthy rekey = 1, missing table =
0, I/O error = a silent 0).

The fix narrows the guard to its printed contract using the DB
layer's own existence check (``Database.table_exists``): a missing
table still returns an honest 0, and any real query failure now
propagates — the surrounding transaction rolls the whole rotation
back (credentials included) and ``change_passphrase`` fails loudly
instead of stranding blobs. The return contract is unchanged for
every documented case; the sibling credential loop in the same
operation already failed loud on CipherError, so this matches the
module's own rotation style.

Pins below: missing-table 0 (preserved), healthy rekey (preserved),
query failure raises (new), and the end-to-end rollback: a crashing
sessions read aborts ``change_passphrase`` with the vault fully
intact under the old passphrase, while a healthy rotation moves
credentials AND session blobs to the new passphrase.

This file contains NO except handlers: crash triggers are instance
patches raised inside the target's own code paths, and expected
raises are asserted with ``assertRaises``.
"""
import tempfile
import unittest
from pathlib import Path

from nomorals.accounts.vault import CredentialVault
from nomorals.core.cipher import derive_key
from nomorals.storage.db import Database
import nomorals.accounts.vault as vault_mod

_OLD = "tail116-old-passphrase"
_NEW = "tail116-new-passphrase"


def _new_master(passphrase):
    return derive_key(passphrase, salt=b"nomorals-vault-master",
                      iterations=vault_mod._KDF_ITERATIONS).hex()


class _VaultCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Database(str(Path(self._tmp.name) / "vault.db"))
        self.vault = CredentialVault(self.db, _OLD)
        self.vault.store("svc", "user", "secret-value")
        self.db.execute(
            "CREATE TABLE sessions (service TEXT, username TEXT, "
            "session_data TEXT)")
        blob = self.vault.encrypt_blob("session-payload",
                                       purpose="sessions")
        self.db.execute(
            "INSERT INTO sessions (service, username, session_data) "
            "VALUES (?, ?, ?)", ("svc", "user", blob))

    def tearDown(self):
        self._tmp.cleanup()

    def _rekey(self, vault, table):
        return vault._rekey_blobs(
            vault._master_key, _new_master(_NEW), table=table,
            column="session_data", key_column=("service", "username"),
            purpose="sessions")

    def _session_blob(self):
        row = self.db.query_one(
            "SELECT session_data FROM sessions WHERE service = 'svc'")
        return row["session_data"]

    def _crash_sessions_query(self):
        real_query = self.db.query

        def crash_query(sql, params=()):
            if "FROM sessions" in sql:
                raise RuntimeError("disk I/O error")
            return real_query(sql, params)

        self.db.query = crash_query
        return real_query


class TestRekeyBlobs1194(_VaultCase):
    def test_missing_table_still_returns_zero(self):
        self.assertEqual(self._rekey(self.vault, "missing_table"), 0)

    def test_healthy_rekey_control(self):
        self.assertEqual(self._rekey(self.vault, "sessions"), 1)
        new_vault = CredentialVault(self.db, _NEW)
        self.assertEqual(
            new_vault.decrypt_blob(self._session_blob(),
                                   purpose="sessions"),
            "session-payload")

    def test_query_failure_now_raises(self):
        real_query = self._crash_sessions_query()
        try:
            with self.assertRaises(RuntimeError):
                self._rekey(self.vault, "sessions")
        finally:
            self.db.query = real_query


class TestChangePassphraseRollback(_VaultCase):
    def test_crashing_sessions_read_aborts_rotation_intact(self):
        old_master = self.vault._master_key
        real_query = self._crash_sessions_query()
        try:
            with self.assertRaises(RuntimeError):
                self.vault.change_passphrase(_NEW)
        finally:
            self.db.query = real_query
        # the whole rotation rolled back: master unchanged, the
        # credential still reads, the session blob still decrypts
        self.assertEqual(self.vault._master_key, old_master)
        self.assertEqual(self.vault.get("svc", "user").password,
                         "secret-value")
        self.assertEqual(
            self.vault.decrypt_blob(self._session_blob(),
                                    purpose="sessions"),
            "session-payload")

    def test_healthy_rotation_moves_blobs_control(self):
        self.vault.change_passphrase(_NEW)
        new_vault = CredentialVault(self.db, _NEW)
        self.assertEqual(new_vault.get("svc", "user").password,
                         "secret-value")
        self.assertEqual(
            new_vault.decrypt_blob(self._session_blob(),
                                   purpose="sessions"),
            "session-payload")


if __name__ == "__main__":
    unittest.main()
