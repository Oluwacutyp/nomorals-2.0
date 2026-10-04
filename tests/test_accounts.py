"""Tests for nomorals.accounts: vault, manager, creator, sessions.

Offline by design — all HTTP is mocked. No network, no real secrets.
"""

from __future__ import annotations

import asyncio
import json
import time
import unittest
from unittest import mock

from nomorals.accounts import (
    AccountCheckpointPending,
    AccountCreator,
    AccountExistsError,
    AccountManager,
    CheckpointKind,
    CheckpointState,
    CredentialVault,
    OAuthToken,
    Session,
    SessionInvalid,
    SessionManager,
)
from nomorals.core.errors import NotFound, StorageError
from nomorals.storage.db import Database


def _vault() -> CredentialVault:
    return CredentialVault(Database(":memory:"), master_passphrase="test-pass")


# ── vault ────────────────────────────────────────────────────────────────


class VaultStoreGetTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()

    def test_store_get_roundtrip(self):
        cred = self.vault.store("gmail", "bot@example.com", "s3cret",
                                tags=["email"], metadata={"k": "v"})
        self.assertEqual(cred.service, "gmail")
        fetched = self.vault.get("gmail", "bot@example.com")
        self.assertEqual(fetched.password, "s3cret")
        self.assertEqual(fetched.tags, ["email"])
        self.assertEqual(fetched.metadata, {"k": "v"})
        self.assertEqual(fetched.use_count, 1)
        self.assertIsNotNone(fetched.last_used)

    def test_store_twice_updates_not_duplicates(self):
        first = self.vault.store("github", "bot", "one")
        second = self.vault.store("github", "bot", "two")
        self.assertEqual(first.id, second.id)
        self.assertEqual(self.vault.get("github", "bot").password, "two")
        self.assertEqual(len(self.vault.list_all(service="github")), 1)

    def test_get_missing_raises_notfound(self):
        with self.assertRaises(NotFound):
            self.vault.get("nope", "nobody")

    def test_get_no_mark_used(self):
        self.vault.store("s", "u", "p")
        cred = self.vault.get("s", "u", mark_used=False)
        self.assertEqual(cred.use_count, 0)
        self.assertIsNone(cred.last_used)

    def test_list_masks_passwords(self):
        self.vault.store("s", "u", "p")
        listed = self.vault.list_all()
        self.assertEqual(listed[0].password, "***")

    def test_list_tag_filter(self):
        self.vault.store("a", "u1", "p", tags=["email", "primary"])
        self.vault.store("b", "u2", "p", tags=["social"])
        self.assertEqual(len(self.vault.list_all(tag="email")), 1)
        self.assertEqual(len(self.vault.list_all(tag="nomatch")), 0)

    def test_delete_missing_raises(self):
        with self.assertRaises(NotFound):
            self.vault.delete("nope", "nobody")

    def test_delete_removes(self):
        self.vault.store("s", "u", "p")
        self.vault.delete("s", "u")
        with self.assertRaises(NotFound):
            self.vault.get("s", "u")

    def test_deactivate_missing_raises(self):
        with self.assertRaises(NotFound):
            self.vault.deactivate("nope", "nobody")

    def test_deactivate_hides_from_active_list(self):
        self.vault.store("s", "u", "p")
        self.vault.deactivate("s", "u")
        self.assertEqual(self.vault.list_all(), [])
        self.assertEqual(len(self.vault.list_all(active_only=False)), 1)
        self.assertFalse(self.vault.get("s", "u").is_active)

    def test_rotate_preserves_fields(self):
        expires = time.time() + 3600
        before = self.vault.store(
            "s", "u", "old",
            credential_type="api_key",
            tags=["t1"],
            metadata={"m": 1},
            expires_at=expires,
        )
        time.sleep(0.01)
        rotated = self.vault.rotate("s", "u", "new")
        self.assertEqual(rotated.password, "new")
        self.assertEqual(rotated.credential_type, "api_key")
        self.assertEqual(rotated.tags, ["t1"])
        self.assertEqual(rotated.metadata, {"m": 1})
        self.assertEqual(rotated.expires_at, expires)
        self.assertGreater(rotated.updated_at, before.updated_at)
        self.assertEqual(rotated.id, before.id)

    def test_rotate_missing_raises(self):
        with self.assertRaises(NotFound):
            self.vault.rotate("nope", "nobody", "x")

    def test_blob_roundtrip(self):
        blob = self.vault.encrypt_blob('{"token": "abc"}', purpose="sessions")
        self.assertNotIn("abc", blob)
        self.assertEqual(
            self.vault.decrypt_blob(blob, purpose="sessions"),
            '{"token": "abc"}',
        )

    def test_blob_wrong_purpose_fails(self):
        blob = self.vault.encrypt_blob("data", purpose="sessions")
        with self.assertRaises(Exception):
            self.vault.decrypt_blob(blob, purpose="other")

    def test_blob_empty_purpose_rejected(self):
        with self.assertRaises(ValueError):
            self.vault.encrypt_blob("data", purpose="")


class VaultLegacyRowTests(unittest.TestCase):
    """Rows encrypted before JSON-wrapping must still decrypt."""

    def setUp(self):
        self.vault = _vault()

    def test_legacy_raw_password_row(self):
        import time as _time
        from nomorals.accounts.vault import _KDF_ITERATIONS
        from nomorals.core.cipher import aes_encrypt, derive_key

        # insert a row the old way (raw plaintext, no JSON wrap)
        with self.vault.db.transaction():
            cur = self.vault.db.execute(
                """INSERT INTO credentials
                   (service, username, password_encrypted, credential_type,
                    tags, metadata, created_at, updated_at, expires_at,
                    use_count, is_active)
                   VALUES (?,?,?,?,?,?,?,?,?,0,1)""",
                ("legacy", "u", "placeholder", "password",
                 "[]", "{}", _time.time(), _time.time(), None),
            )
            cred_id = cur.lastrowid
            key = derive_key(self.vault._master_key,
                             salt=f"credential-{cred_id}".encode(),
                             iterations=_KDF_ITERATIONS)
            blob = aes_encrypt(b"legacy-pw", key=key)
            self.vault.db.execute(
                "UPDATE credentials SET password_encrypted = ? WHERE id = ?",
                (blob, cred_id),
            )
        self.assertEqual(self.vault.get("legacy", "u").password, "legacy-pw")


class VaultProfileTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()

    def test_create_and_get_profile(self):
        profile = self.vault.create_profile("owner", description="the owner")
        self.assertEqual(profile.name, "owner")
        self.assertEqual(profile.description, "the owner")
        self.assertEqual(profile.credentials, [])

    def test_create_duplicate_profile_raises(self):
        self.vault.create_profile("owner")
        with self.assertRaises(StorageError):
            self.vault.create_profile("owner")

    def test_get_missing_profile_raises(self):
        with self.assertRaises(NotFound):
            self.vault.get_profile("ghost")

    def test_delete_missing_profile_raises(self):
        with self.assertRaises(NotFound):
            self.vault.delete_profile("ghost")

    def test_add_get_remove(self):
        self.vault.create_profile("owner")
        self.vault.store("gmail", "a@x.com", "p1", tags=["email"])
        self.vault.store("github", "bot", "p2")
        self.vault.add_to_profile("owner", "gmail", "a@x.com")

        profile = self.vault.get_profile("owner")
        self.assertEqual(len(profile.credentials), 1)
        self.assertEqual(profile.credentials[0].service, "gmail")
        self.assertEqual(profile.credentials[0].password, "***")  # masked

        # get_credential helper on the profile
        self.assertIsNotNone(profile.get_credential("gmail"))
        self.assertIsNone(profile.get_credential("github"))

        self.vault.remove_from_profile("owner", "gmail", "a@x.com")
        self.assertEqual(self.vault.get_profile("owner").credentials, [])

    def test_add_missing_credential_raises(self):
        self.vault.create_profile("owner")
        with self.assertRaises(NotFound):
            self.vault.add_to_profile("owner", "gmail", "ghost")

    def test_add_missing_profile_raises(self):
        self.vault.store("gmail", "a@x.com", "p1")
        with self.assertRaises(NotFound):
            self.vault.add_to_profile("ghost", "gmail", "a@x.com")

    def test_remove_not_member_raises(self):
        self.vault.create_profile("owner")
        self.vault.store("gmail", "a@x.com", "p1")
        with self.assertRaises(NotFound):
            self.vault.remove_from_profile("owner", "gmail", "a@x.com")

    def test_list_profiles(self):
        self.vault.create_profile("b-profile")
        self.vault.create_profile("a-profile")
        names = [p.name for p in self.vault.list_profiles()]
        self.assertEqual(names, ["a-profile", "b-profile"])

    def test_delete_profile_keeps_credentials(self):
        self.vault.create_profile("owner")
        self.vault.store("gmail", "a@x.com", "p1")
        self.vault.add_to_profile("owner", "gmail", "a@x.com")
        self.vault.delete_profile("owner")
        with self.assertRaises(NotFound):
            self.vault.get_profile("owner")
        # credential itself survives
        self.assertEqual(self.vault.get("gmail", "a@x.com").password, "p1")

    def test_delete_credential_cascades_membership(self):
        self.vault.create_profile("owner")
        self.vault.store("gmail", "a@x.com", "p1")
        self.vault.add_to_profile("owner", "gmail", "a@x.com")
        self.vault.delete("gmail", "a@x.com")
        self.assertEqual(self.vault.get_profile("owner").credentials, [])

    def test_profiles_of(self):
        self.vault.create_profile("p1")
        self.vault.create_profile("p2")
        self.vault.store("gmail", "a@x.com", "pw")
        self.vault.add_to_profile("p1", "gmail", "a@x.com")
        self.vault.add_to_profile("p2", "gmail", "a@x.com")
        self.assertEqual(self.vault.profiles_of("gmail", "a@x.com"),
                         ["p1", "p2"])

    def test_profiles_of_missing_credential_raises(self):
        with self.assertRaises(NotFound):
            self.vault.profiles_of("gmail", "ghost")


# ── manager ──────────────────────────────────────────────────────────────


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()
        self.mgr = AccountManager(self.vault)

    def test_get_credential(self):
        self.vault.store("gmail", "a@x.com", "pw")
        self.assertEqual(
            self.mgr.get_credential("gmail", "a@x.com").password, "pw")

    def test_is_expired(self):
        self.vault.store("s", "u", "p", expires_at=time.time() - 10)
        self.assertTrue(self.mgr.is_expired("s", "u"))
        self.vault.store("s2", "u", "p")
        self.assertFalse(self.mgr.is_expired("s2", "u"))
        self.assertFalse(self.mgr.is_expired("missing", "u"))

    def test_refresh_credential(self):
        self.vault.store("s", "u", "old", expires_at=time.time() + 10)
        new_exp = time.time() + 9999
        cred = self.mgr.refresh_credential("s", "u", "new", expires_at=new_exp)
        self.assertEqual(cred.password, "new")
        self.assertEqual(cred.expires_at, new_exp)

    def test_list_accounts_filters(self):
        self.vault.store("gmail", "a@x.com", "p", tags=["email"])
        self.vault.store("github", "bot", "p", tags=["code"])
        self.assertEqual(len(self.mgr.list_accounts()), 2)
        self.assertEqual(len(self.mgr.list_accounts(service="gmail")), 1)
        self.assertEqual(len(self.mgr.list_accounts(tag="code")), 1)
        info = self.mgr.list_accounts(service="gmail")[0]
        self.assertEqual(info.username, "a@x.com")
        self.assertFalse(info.is_expired)

    def test_deactivate_and_delete(self):
        self.vault.store("s", "u", "p")
        self.mgr.deactivate_account("s", "u")
        self.assertEqual(self.mgr.list_accounts(), [])
        self.mgr.delete_account("s", "u")
        self.assertEqual(self.mgr.list_accounts(active_only=False), [])

    def test_get_stats(self):
        self.vault.store("gmail", "a@x.com", "p", expires_at=time.time() - 1)
        self.vault.store("github", "bot", "p")
        stats = self.mgr.get_stats()
        self.assertEqual(stats["total_credentials"], 2)
        self.assertEqual(stats["active_credentials"], 2)
        self.assertEqual(stats["expired_credentials"], 1)
        self.assertEqual(stats["services"], {"gmail": 1, "github": 1})

    def test_health_check_flags(self):
        self.vault.store("old", "u", "p", expires_at=time.time() - 100)
        self.vault.store("soon", "u", "p",
                         expires_at=time.time() + 3 * 86400)
        self.vault.store("stale", "u", "p")
        stale = self.vault.get("stale", "u")
        # fake last_used 40 days ago
        with self.vault.db.transaction():
            self.vault.db.execute(
                "UPDATE credentials SET last_used = ? WHERE id = ?",
                (time.time() - 40 * 86400, stale.id),
            )
        issues = self.mgr.health_check()
        kinds = {(i["type"], i["service"]) for i in issues}
        self.assertIn(("expired", "old"), kinds)
        self.assertIn(("expiring_soon", "soon"), kinds)
        self.assertIn(("unused", "stale"), kinds)

    def test_get_profile_delegates(self):
        self.vault.create_profile("owner")
        self.vault.store("gmail", "a@x.com", "p")
        self.vault.add_to_profile("owner", "gmail", "a@x.com")
        profile = self.mgr.get_profile("owner")
        self.assertEqual(len(profile.credentials), 1)

    # connector-vault integration

    def test_register_and_fetch_connector_credential(self):
        cred = self.mgr.register_connector_credential(
            "github", "bot-login", "ghp_secret",
            credential_type="oauth_token",
            scopes=["repo"],
            metadata={"note": "x"},
        )
        self.assertEqual(cred.service, "connector:github")
        self.assertEqual(cred.metadata["scopes"], ["repo"])
        fetched = self.mgr.connector_credential("github")
        self.assertIsNotNone(fetched)
        assert fetched is not None
        self.assertEqual(fetched.password, "ghp_secret")

    def test_connector_credential_none_when_absent(self):
        self.assertIsNone(self.mgr.connector_credential("github"))

    def test_connector_credential_skips_expired(self):
        self.mgr.register_connector_credential(
            "mono", "old", "s1", expires_at=time.time() - 5)
        self.assertIsNone(self.mgr.connector_credential("mono"))

    def test_connector_credential_username_pin(self):
        self.mgr.register_connector_credential("x", "u1", "s1")
        self.mgr.register_connector_credential("x", "u2", "s2")
        cred = self.mgr.connector_credential("x", username="u2")
        assert cred is not None
        self.assertEqual(cred.username, "u2")

    def test_list_connector_credentials(self):
        self.mgr.register_connector_credential("github", "u", "s")
        self.vault.store("gmail", "a@x.com", "p")  # not a connector
        infos = self.mgr.list_connector_credentials()
        self.assertEqual(len(infos), 1)
        self.assertEqual(infos[0].service, "connector:github")
        scoped = self.mgr.list_connector_credentials(connector_id="mono")
        self.assertEqual(scoped, [])

    def test_revoke_connector_credentials(self):
        self.mgr.register_connector_credential("github", "u1", "s1")
        self.mgr.register_connector_credential("github", "u2", "s2")
        removed = self.mgr.revoke_connector_credentials("github")
        self.assertEqual(removed, 2)
        self.assertIsNone(self.mgr.connector_credential("github"))

    def test_connector_summary(self):
        self.mgr.register_connector_credential(
            "github", "u", "s", credential_type="oauth_token")
        self.mgr.register_connector_credential(
            "mono", "u", "s", expires_at=time.time() - 5)
        summary = self.mgr.connector_summary()
        self.assertEqual(summary["github"]["active"], 1)
        self.assertEqual(summary["mono"]["expired"], 1)
        self.assertEqual(summary["github"]["credential_types"],
                         ["oauth_token"])


# ── creator ──────────────────────────────────────────────────────────────


class CreatorIdentityTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()
        self.creator = AccountCreator(self.vault)

    def test_set_owner_identity(self):
        ident = self.creator.set_owner_identity("Death", "owner@example.com")
        self.assertEqual(ident["email"], "owner@example.com")
        self.assertEqual(self.creator.get_owner_identity()["name"], "Death")

    def test_set_owner_identity_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.creator.set_owner_identity("", "x@y.z")
        with self.assertRaises(ValueError):
            self.creator.set_owner_identity("Name", "")

    def test_create_account_needs_identity_checkpoint(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("github"))
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.kind, CheckpointKind.IDENTITY)
        self.assertEqual(cp.state, CheckpointState.PENDING)
        self.assertEqual(len(self.creator.get_pending_checkpoints()), 1)

    def test_identity_checkpoint_resume(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("github"))
        cp_id = ctx.exception.checkpoint.id
        resolved = self.creator.resume_checkpoint(
            cp_id, owner_name="Death", owner_email="owner@example.com")
        self.assertEqual(resolved.state, CheckpointState.RESOLVED)
        self.assertEqual(
            self.creator.get_owner_identity()["email"], "owner@example.com")
        # now the flow can proceed to the service checkpoint
        with self.assertRaises(AccountCheckpointPending) as ctx2:
            asyncio.run(self.creator.create_account(
                "github", username="my-bot"))
        self.assertEqual(ctx2.exception.checkpoint.kind,
                         CheckpointKind.CAPTCHA)

    def test_resume_identity_without_email_raises(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("github"))
        with self.assertRaises(Exception):
            self.creator.resume_checkpoint(ctx.exception.checkpoint.id)


class CreatorFlowTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()
        self.notifies: list[tuple[str, str, str]] = []
        self.creator = AccountCreator(
            self.vault,
            notify=lambda t, i, cid: self.notifies.append((t, i, cid)),
        )
        self.creator.set_owner_identity("Death", "owner@example.com")

    def test_github_flow_pauses_and_resumes(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account(
                "github", username="my-bot"))
        cp = ctx.exception.checkpoint
        self.assertEqual(cp.service, "github")
        self.assertIn("my-bot", cp.instructions)
        # owner was pinged
        self.assertEqual(len(self.notifies), 1)
        self.assertEqual(self.notifies[0][2], cp.id)
        # nothing stored yet
        self.assertEqual(self.vault.list_all(service="github"), [])

        account = self.creator.resume_checkpoint(cp.id, note="done")
        self.assertEqual(account.status, "created")
        self.assertEqual(account.service, "github")
        self.assertEqual(account.username, "my-bot")
        cred = self.vault.get("github", "my-bot")
        self.assertEqual(cred.credential_type, "account")
        self.assertEqual(len(self.creator.get_pending_checkpoints()), 0)

    def test_one_account_per_service(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("github"))
        self.creator.resume_checkpoint(ctx.exception.checkpoint.id)
        with self.assertRaises(AccountExistsError):
            asyncio.run(self.creator.create_account("github"))

    def test_one_per_service_allows_after_deactivate(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("github"))
        self.creator.resume_checkpoint(ctx.exception.checkpoint.id)
        self.vault.deactivate("github",
                              self.vault.list_all(service="github")[0].username)
        with self.assertRaises(AccountCheckpointPending):
            asyncio.run(self.creator.create_account("github"))

    def test_gmail_flow_kind(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("gmail"))
        self.assertEqual(ctx.exception.checkpoint.kind,
                         CheckpointKind.PHONE_2FA)

    def test_generic_service_flow(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("konga"))
        self.assertEqual(ctx.exception.checkpoint.kind,
                         CheckpointKind.MANUAL_STEP)

    def test_explicit_email_used(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account(
                "github", email="alt@example.com"))
        self.assertIn("alt@example.com",
                      ctx.exception.checkpoint.resume_state["email"])

    def test_finalize_account_direct(self):
        account = self.creator.finalize_account(
            "github", "bot", "pw", email="owner@example.com")
        self.assertEqual(account.status, "created")
        self.assertEqual(self.vault.get("github", "bot").password, "pw")
        self.assertEqual(len(self.creator.get_creation_history()), 1)

    def test_cancel_checkpoint(self):
        with self.assertRaises(AccountCheckpointPending) as ctx:
            asyncio.run(self.creator.create_account("github"))
        cp = self.creator.cancel_checkpoint(ctx.exception.checkpoint.id,
                                            "changed mind")
        self.assertEqual(cp.state, CheckpointState.CANCELLED)
        self.assertEqual(self.creator.get_pending_checkpoints(), [])

    def test_resolve_unknown_checkpoint_raises(self):
        with self.assertRaises(NotFound):
            self.creator.resume_checkpoint("achk_nope")

    def test_checkpoint_expiry(self):
        cp = self.creator.checkpoints.create(
            CheckpointKind.CAPTCHA, "t", "i", service="github",
            ttl_seconds=-1,
        )
        self.assertEqual(self.creator.checkpoints.get(cp.id).state,
                         CheckpointState.EXPIRED)

    def test_email_account_fallback_offline(self):
        # guerrillamail API unreachable -> fallback address, still stored
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("no network")):
            account = asyncio.run(
                self.creator.create_email_account(provider="guerrilla"))
        self.assertEqual(account.status, "created")
        self.assertTrue(account.email.endswith("@guerrillamail.com"))
        self.assertEqual(
            self.vault.get("email_guerrilla", account.email).username,
            account.email)

    def test_email_account_unknown_provider(self):
        account = asyncio.run(
            self.creator.create_email_account(provider="bogus"))
        self.assertEqual(account.status, "failed")

    def test_disposable_exempt_from_one_per_service(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("no network")):
            a1 = asyncio.run(self.creator.create_email_account())
            a2 = asyncio.run(self.creator.create_email_account())
        self.assertNotEqual(a1.email, a2.email)

    def test_notify_hook_failure_does_not_break_flow(self):
        def bad_notify(t, i, cid):
            raise RuntimeError("hook down")
        creator = AccountCreator(self.vault, notify=bad_notify)
        creator.set_owner_identity("Death", "owner@example.com")
        with self.assertRaises(AccountCheckpointPending):
            asyncio.run(creator.create_account("github"))


# ── sessions ─────────────────────────────────────────────────────────────


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.vault = _vault()
        self.sm = SessionManager(self.vault)

    def test_get_session_creates_and_caches(self):
        s1 = self.sm.get_session("github", "bot")
        s2 = self.sm.get_session("github", "bot")
        self.assertEqual(s1.service, "github")
        self.assertIs(s1, s2)  # same cached object

    def test_touch_persisted(self):
        s = self.sm.get_session("github", "bot")
        first_used = s.last_used
        time.sleep(0.01)
        self.sm.get_session("github", "bot")  # cache hit -> touch + save
        row = self.vault.db.query_one(
            "SELECT last_used FROM sessions WHERE service='github'")
        self.assertGreater(row["last_used"], first_used)

    def test_session_data_encrypted_at_rest(self):
        token = OAuthToken(access_token="super-secret-token",
                           expires_at=time.time() + 3600)
        self.sm.set_oauth_token("github", "bot", token)
        # drop cache so we read raw DB
        self.sm._sessions.clear()
        row = self.vault.db.query_one("SELECT session_data FROM sessions")
        self.assertNotIn("super-secret-token", row["session_data"])
        # ...but the session still loads fine
        s = self.sm.get_session("github", "bot")
        assert s.oauth_token is not None
        self.assertEqual(s.oauth_token.access_token, "super-secret-token")

    def test_legacy_plaintext_row_still_loads(self):
        payload = json.dumps(Session(service="x", username="y").to_dict())
        with self.vault.db.transaction():
            self.vault.db.execute(
                "INSERT INTO sessions (service, username, session_data,"
                " created_at, last_used) VALUES (?,?,?,?,?)",
                ("x", "y", payload, time.time(), time.time()),
            )
        s = self.sm.get_session("x", "y")
        self.assertEqual(s.service, "x")
        # next save re-encrypts it
        row = self.vault.db.query_one(
            "SELECT session_data FROM sessions WHERE service='x'")
        with self.assertRaises(Exception):
            json.loads(row["session_data"])

    def test_cookies_headers_merge(self):
        self.sm.set_cookies("s", "u", {"a": "1"})
        self.sm.set_cookies("s", "u", {"b": "2"})
        self.sm.set_headers("s", "u", {"X-A": "v"})
        s = self.sm.get_session("s", "u")
        self.assertEqual(s.cookies, {"a": "1", "b": "2"})
        self.assertEqual(s.headers, {"X-A": "v"})

    def test_auth_headers_with_bearer(self):
        self.sm.set_headers("s", "u", {"X-A": "v"})
        self.sm.set_oauth_token(
            "s", "u",
            OAuthToken(access_token="tok123", token_type="Bearer",
                       expires_at=time.time() + 3600))
        headers = self.sm.auth_headers("s", "u")
        self.assertEqual(headers["Authorization"], "Bearer tok123")
        self.assertEqual(headers["X-A"], "v")

    def test_auth_headers_expired_token_omitted(self):
        self.sm.set_oauth_token(
            "s", "u", OAuthToken(access_token="old",
                                 expires_at=time.time() - 10))
        self.assertNotIn("Authorization", self.sm.auth_headers("s", "u"))

    def test_set_oauth_token_vault_copy(self):
        token = OAuthToken(access_token="abc",
                           expires_at=time.time() + 3600)
        self.sm.set_oauth_token("gmail", "bot", token)
        cred = self.vault.get("gmail_oauth", "bot")
        self.assertEqual(cred.credential_type, "oauth_token")
        self.assertEqual(json.loads(cred.password)["access_token"], "abc")

    def test_get_valid_session_ok(self):
        self.sm.get_session("s", "u")
        self.assertTrue(self.sm.get_valid_session("s", "u").is_valid())

    def test_get_valid_session_expired_token(self):
        self.sm.set_oauth_token(
            "s", "u", OAuthToken(access_token="old",
                                 expires_at=time.time() - 10))
        with self.assertRaises(SessionInvalid):
            self.sm.get_valid_session("s", "u")

    def test_get_valid_session_inactive(self):
        s = self.sm.get_session("s", "u")
        # age the session without touching it (_save_session skips touch;
        # the encrypted blob is authoritative for last_used)
        s.last_used = time.time() - 25 * 3600
        self.sm._save_session(s)
        self.sm._sessions.clear()
        with self.assertRaises(SessionInvalid):
            self.sm.get_valid_session("s", "u")

    def test_ensure_oauth_token_none(self):
        self.sm.get_session("s", "u")
        self.assertIsNone(self.sm.ensure_oauth_token("s", "u"))

    def test_ensure_oauth_token_valid_passthrough(self):
        token = OAuthToken(access_token="good",
                           expires_at=time.time() + 3600)
        self.sm.set_oauth_token("s", "u", token)
        self.assertEqual(self.sm.ensure_oauth_token("s", "u").access_token,
                         "good")

    def test_ensure_oauth_token_expired_no_config_raises(self):
        self.sm.set_oauth_token(
            "s", "u", OAuthToken(access_token="old",
                                 expires_at=time.time() - 10))
        with self.assertRaises(SessionInvalid):
            self.sm.ensure_oauth_token("s", "u")

    def test_ensure_oauth_token_auto_refresh(self):
        old = OAuthToken(access_token="old", refresh_token="rt",
                         expires_at=time.time() - 10)
        self.sm.set_oauth_token("s", "u", old, auto_refresh={
            "token_url": "https://example.com/token",
            "client_id": "cid",
            "client_secret": "csec",
        })

        body = json.dumps({
            "access_token": "fresh", "token_type": "Bearer",
            "expires_in": 3600, "refresh_token": "rt2",
            "scope": "read",
        }).encode()

        class FakeResp:
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def read(self): return body

        with mock.patch("urllib.request.urlopen",
                        return_value=FakeResp()) as m:
            new = self.sm.ensure_oauth_token("s", "u")
        self.assertEqual(new.access_token, "fresh")
        self.assertEqual(new.refresh_token, "rt2")
        self.assertFalse(new.is_expired())
        # session now carries the fresh token
        s = self.sm.get_session("s", "u")
        assert s.oauth_token is not None
        self.assertEqual(s.oauth_token.access_token, "fresh")
        m.assert_called_once()

    def test_refresh_oauth_token_http_error_raises(self):
        with mock.patch("urllib.request.urlopen",
                        side_effect=OSError("down")):
            with self.assertRaises(OSError):
                self.sm.refresh_oauth_token("s", "u", "rt", "cid",
                                            "csec", "https://x/token")

    def test_clear_session(self):
        self.sm.set_oauth_token(
            "s", "u", OAuthToken(access_token="t",
                                 expires_at=time.time() + 100))
        self.sm.clear_session("s", "u")
        self.assertEqual(self.sm.list_sessions(), [])
        with self.assertRaises(NotFound):
            self.vault.get("s_oauth", "u")

    def test_list_sessions_valid_only(self):
        self.sm.get_session("fresh", "u")
        stale = self.sm.get_session("stale", "u")
        stale.last_used = time.time() - 25 * 3600
        self.sm._save_session(stale)
        self.sm._sessions.clear()
        self.assertEqual(len(self.sm.list_sessions()), 2)
        valid = self.sm.list_sessions(valid_only=True)
        self.assertEqual([s.service for s in valid], ["fresh"])

    def test_cleanup_expired(self):
        self.sm.get_session("fresh", "u")
        stale = self.sm.get_session("stale", "u")
        stale.last_used = 1.0
        stale.created_at = 1.0
        self.sm._save_session(stale)
        self.sm._sessions.clear()
        removed = self.sm.cleanup_expired()
        self.assertEqual(removed, 1)
        self.assertEqual(
            [s.service for s in self.sm.list_sessions()], ["fresh"])

    def test_oauth_token_serialization(self):
        token = OAuthToken(access_token="a", expires_at=123.0,
                           refresh_token="r", scope="s")
        self.assertEqual(OAuthToken.from_dict(token.to_dict()).access_token,
                         "a")
        self.assertTrue(OAuthToken(access_token="a").is_expired() is False)
        self.assertTrue(
            OAuthToken(access_token="a",
                       expires_at=time.time() - 1).is_expired())


# ── tempmail.plus (no-key REST API) ─────────────────────────────────────


class TempmailTests(unittest.TestCase):
    """tempmail provider: real @mailto.plus addresses + inbox polling.

    Offline by design — the HTTP helper is mocked; one test file does a
    live smoke check separately (not here).
    """

    def setUp(self):
        self.vault = _vault()
        self.creator = AccountCreator(self.vault)
        self.ok_payload = {
            "result": True,
            "mail_list": [
                {"mail_id": 101, "from_mail": "noreply@example.com",
                 "subject": "Your code is 482910",
                 "time": "2026-10-04 06:10:00"},
                {"mail_id": 102, "from_mail": "x@y.com",
                 "subject": "Welcome", "time": "2026-10-04 06:11:00"},
            ],
        }

    def test_create_mints_real_mailto_plus_address(self):
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                return_value={"result": True, "mail_list": []}):
            account = asyncio.run(
                self.creator.create_account("tempmail",
                                            username="Test User!"))
        self.assertEqual(account.status, "created")
        self.assertEqual(account.email, "testuser@mailto.plus")
        cred = self.vault.get("email_tempmail", "testuser@mailto.plus")
        self.assertEqual(cred.credential_type, "disposable_email")
        self.assertEqual(cred.metadata.get("provider"), "tempmail")

    def test_create_randomizes_short_username(self):
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                return_value={"result": True, "mail_list": []}):
            account = asyncio.run(
                self.creator.create_account("tempmail", username="ab"))
        self.assertEqual(account.status, "created")
        self.assertTrue(account.email.endswith("@mailto.plus"))
        self.assertGreater(len(account.email.split("@")[0]), 3)

    def test_create_api_down_reports_failure(self):
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                side_effect=RuntimeError("boom")):
            account = asyncio.run(
                self.creator.create_account("tempmail",
                                            username="someone"))
        self.assertEqual(account.status, "failed")
        self.assertIn("boom", account.notes)

    def test_inbox_parses_mail_list(self):
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                return_value=self.ok_payload):
            msgs = self.creator.tempmail_inbox("someone@mailto.plus")
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[0]["id"], 101)
        self.assertEqual(msgs[0]["from"], "noreply@example.com")
        self.assertEqual(msgs[0]["subject"], "Your code is 482910")
        self.assertEqual(msgs[0]["date"], "2026-10-04 06:10:00")

    def test_inbox_respects_limit(self):
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                return_value=self.ok_payload):
            msgs = self.creator.tempmail_inbox("someone@mailto.plus",
                                               limit=1)
        self.assertEqual(len(msgs), 1)

    def test_inbox_never_crashes(self):
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                side_effect=RuntimeError("net down")):
            self.assertEqual(
                self.creator.tempmail_inbox("someone@mailto.plus"), [])

    def test_read_returns_full_message(self):
        detail = {"result": True, "mail_id": 101,
                  "from_mail": "noreply@example.com", "subject": "code",
                  "date": "2026-10-04", "text": "Your code is 482910",
                  "html": "<b>482910</b>"}
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                return_value=detail):
            msg = self.creator.tempmail_read("someone@mailto.plus", 101)
        self.assertEqual(msg["text"], "Your code is 482910")
        self.assertEqual(msg["html"], "<b>482910</b>")
        self.assertEqual(msg["from"], "noreply@example.com")

    def test_check_disposable_inbox_dispatches_tempmail(self):
        self.vault.store("email_tempmail", "someone@mailto.plus", "",
                         credential_type="disposable_email",
                         metadata={"provider": "tempmail"})
        cred = self.vault.get("email_tempmail", "someone@mailto.plus")
        with mock.patch.object(
                AccountCreator, "_tempmail_request",
                return_value=self.ok_payload) as req:
            msgs = self.creator.check_disposable_inbox(cred, limit=5)
        self.assertEqual(len(msgs), 2)
        called_url = req.call_args[0][0]
        self.assertIn("someone%40mailto.plus", called_url)

    def test_check_disposable_inbox_unknown_provider_empty(self):
        self.vault.store("email_tempmail", "ghost@mailto.plus", "",
                         credential_type="disposable_email",
                         metadata={"provider": "nope"})
        cred = self.vault.get("email_tempmail", "ghost@mailto.plus")
        self.assertEqual(self.creator.check_disposable_inbox(cred), [])


if __name__ == "__main__":
    unittest.main()
