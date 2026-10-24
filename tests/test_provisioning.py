"""Self-provisioning credentials (accounts layer).

The loop the owner approved (2026-10-20): a feature dies for a missing
key → the failure is classified → Devon provisions the key herself on
the owner's own account → vault → env bridge → retry. These tests
drive the REAL registry, driver, login organ and page flow against
fake tabs/HTTP at the documented seams — no network, no real HF.

Fake-token hygiene rule under test everywhere: the created token and
any handed-over password must never appear in a result message, an
audit record, or a log line.
"""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from typing import Any
from unittest import mock

from nomorals.accounts import provisioning
from nomorals.accounts.creator import AccountCreator, CreatedAccount
from nomorals.accounts.provisioning import (
    ProvisionDriver,
    ProvisionOutcome,
    ProvisionService,
    bridge_vault_credentials,
    classify_missing_credential,
    get_provision_service,
    register_provision_service,
)
from nomorals.accounts.vault import CredentialVault
from nomorals.core.errors import MissingCredentialError
from nomorals.storage.db import Database

FAKE_TOKEN = "hf_" + "T" * 30
OWNER_PW = "owner-login-password-123"
GEN_EMAIL = "gen.persona@mailo.plus"


def _fake_email_mint(calls: list):
    """Stand-in for AccountCreator.create_email_account — the ONE
    network seam in the generated-identity path (tempmail.plus). Same
    contract as the real mint: a CreatedAccount carrying a real,
    provider-issued address, or status "failed" with no address."""

    async def _mint(self, *, provider="guerrilla", username=None):
        calls.append(provider)
        return CreatedAccount(
            service="email_tempmail", username=GEN_EMAIL, password="",
            email=GEN_EMAIL, status="created",
            notes="fake mint (test seam)")

    return _mint


def _vault(tmp: str) -> CredentialVault:
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()  # kv_store + friends: the creator/provisioning organs need them
    return CredentialVault(db, master_passphrase="test-pass")


class FakeTab:
    """Scripted stand-in for a driven browser tab (the same duck type
    browser_login's tests use), modelling the HF pages the flow walks:
    login form → tokens page → create dialog → created token; and the
    FLUX license page."""

    def __init__(self, *, create_button: bool = True,
                 captcha: list | None = None) -> None:
        self.url = ""
        self.fills: dict[str, str] = {}
        self.clicks: list[str] = []
        self.closed = False
        self._dialog = False
        self._created = False
        self._logged_in = False
        self._create_button = create_button
        self._captcha = captcha or []
        self._license_accepted = False

    # -- tab protocol ---------------------------------------------------
    def navigate(self, url: str):
        self.url = url
        return {"ok": True}

    def fill(self, name: str, value: str):
        if "tokens" in self.url:
            if name != "name" or not self._dialog:
                raise Exception(f"no field {name!r}")
        self.fills[name] = value
        return {"ok": True}

    def click(self, target: str):
        label = (target or "").strip().lower()
        self.clicks.append(target)
        if "tokens" in self.url:
            if label.startswith("create new token"):
                if not self._create_button:
                    raise Exception("no such button")
                self._dialog = True
                return {"ok": True}
            if label.startswith("create token") or label == "create":
                if not self._dialog:
                    raise Exception("dialog not open")
                self._created = True
                return {"ok": True}
            if label in ("read", "write"):
                return {"ok": True}
            raise Exception(f"no clickable {target!r}")
        if "FLUX.1-Kontext-dev" in self.url:
            if "agree" in label:
                self._license_accepted = True
                return {"ok": True}
            raise Exception(f"no clickable {target!r}")
        raise Exception(f"no clickable {target!r} on {self.url}")

    def submit(self, target: str = ""):
        self._logged_in = True
        return {"ok": True}

    def text(self, max_chars: int = 20000):
        if "/login" in self.url and not self._logged_in:
            return {"text": "Log In to Hugging Face — email, password"}
        if "tokens" in self.url:
            body = "Access Tokens — Create new token"
            if self._dialog:
                body += " — name your token, role Read, Create token"
            if self._created:
                body += f" — your new token: {FAKE_TOKEN} (shown once)"
            return {"text": body}
        if "FLUX.1-Kontext-dev" in self.url:
            if self._license_accepted:
                return {"text": "license accepted — you can use this model"}
            return {"text": "gated model — Agree and access repository "
                            "to accept the license"}
        return {"text": "Hugging Face — you are logged in"}

    def evaluate(self, js: str):
        # The token-extraction script: answer only once a token exists.
        if "hf_" in js and self._created:
            return FAKE_TOKEN
        return None

    def cookies(self):
        return [{"name": "session", "value": "abc"}]

    def check_captcha(self):
        if "tokens" in self.url:
            return {"challenges": list(self._captcha)}
        return {"challenges": []}

    def close(self):
        self.closed = True


def _http_ok(url: str, headers: dict) -> tuple[int, str]:
    assert headers.get("Authorization") == f"Bearer {FAKE_TOKEN}"
    return 200, '{"name": "Cutyp"}'


def _driver(vault: CredentialVault, *, environ=None, http_get=_http_ok,
            tab_factory=None, **kw) -> ProvisionDriver:
    tabs: list[FakeTab] = []

    def _open(service: str) -> FakeTab:
        tab = (tab_factory or FakeTab)()
        tabs.append(tab)
        return tab

    driver = ProvisionDriver(
        vault, open_tab=_open, http_get=http_get,
        environ={} if environ is None else environ, **kw)
    driver._test_tabs = tabs  # type: ignore[attr-defined]
    return driver


class _NoSleep:
    """login_with_vault waits after submit; tests don't."""

    def __enter__(self):
        self._patch = mock.patch(
            "nomorals.accounts.browser_login.time.sleep", lambda s: None)
        self._patch.start()
        return self

    def __exit__(self, *exc):
        self._patch.stop()
        return False


class RegistryAndClassificationTests(unittest.TestCase):
    def test_hf_entry_resolves_by_id_and_alias(self):
        svc = get_provision_service("huggingface")
        self.assertIsNotNone(svc)
        self.assertIs(get_provision_service("hf"), svc)
        self.assertIs(get_provision_service("HF"), svc)
        self.assertEqual(svc.env_keys, ("HF_TOKEN",))
        self.assertIn(svc.flow, provisioning._PAGE_FLOWS)
        self.assertIn(svc.validator, provisioning._VALIDATORS)
        self.assertTrue(svc.licenses)

    def test_classify_finds_typed_error_in_chain(self):
        inner = MissingCredentialError("needs a token", service="huggingface",
                                       env_key="HF_TOKEN")
        try:
            try:
                raise inner
            except MissingCredentialError as exc:
                raise RuntimeError("wrapped") from exc
        except RuntimeError as outer:
            self.assertIs(classify_missing_credential(outer), inner)
        self.assertIsNone(classify_missing_credential(ValueError("nope")))
        self.assertIsNone(classify_missing_credential(None))

    def test_hf_backend_raises_typed_error_without_token(self):
        from nomorals.media_edit.generate import (
            GenerativeEditError,
            HFInferenceBackend,
        )

        class _BoomClient:
            def text_to_image(self, prompt, model=None, **kw):
                raise Exception(
                    "You must provide an api_key to work with fal-ai "
                    "API or log in with `hf auth login`.")

        backend = HFInferenceBackend(token=None)
        with mock.patch.object(backend, "_client", return_value=_BoomClient()):
            with self.assertRaises(MissingCredentialError) as ctx:
                backend.generate("a cat")
        self.assertEqual(ctx.exception.service, "huggingface")
        self.assertEqual(ctx.exception.env_key, "HF_TOKEN")

        # With a token configured the same provider failure is NOT a
        # missing credential — it stays a plain backend failure.
        backend2 = HFInferenceBackend(token="hf_configured")
        with mock.patch.object(backend2, "_client", return_value=_BoomClient()):
            with self.assertRaises(GenerativeEditError):
                backend2.generate("a cat")


class DriveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-provision-")
        self.vault = _vault(self.tmp)

    def _seed_login(self, username: str = "Cutyp") -> None:
        self.vault.store("huggingface", username, OWNER_PW,
                         credential_type="password",
                         metadata={"login_url": "https://huggingface.co/login"})

    def test_full_loop_login_to_vaulted_key(self):
        self._seed_login()
        environ: dict[str, str] = {}
        driver = _driver(self.vault, environ=environ)
        with _NoSleep():
            result = driver.provision("hf")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertFalse(result.account_created)
        self.assertEqual(result.username, "Cutyp")
        # the key lives in the vault, as an api_key credential
        creds = self.vault.list_all(service="huggingface")
        key = next(c for c in creds if c.credential_type == "api_key")
        self.assertEqual(
            self.vault.get("huggingface", key.username).password, FAKE_TOKEN)
        # and the bridge wired it where the feature reads it
        self.assertEqual(environ.get("HF_TOKEN"), FAKE_TOKEN)
        self.assertEqual(result.env_keys_applied, ["HF_TOKEN"])
        # license step accepted, noted as a warning
        self.assertTrue(any("license" in w for w in result.warnings),
                        result.warnings)
        # hygiene: the token is in NO message, warning, or audit record
        blob = result.message + " ".join(result.warnings) + str(driver.audit_trail)
        self.assertNotIn(FAKE_TOKEN, blob)
        self.assertNotIn(OWNER_PW, blob)

    def test_already_provisioned_from_env(self):
        driver = _driver(self.vault, environ={"HF_TOKEN": "hf_existing"})
        result = driver.provision("huggingface")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.ALREADY_PROVISIONED)

    def test_already_provisioned_from_vault_bridges(self):
        self.vault.store("huggingface", "Cutyp", FAKE_TOKEN,
                         credential_type="api_key")
        environ: dict[str, str] = {}
        driver = _driver(self.vault, environ=environ)
        result = driver.provision("hf")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.ALREADY_PROVISIONED)
        self.assertEqual(environ.get("HF_TOKEN"), FAKE_TOKEN)

    def test_known_account_gets_handoff_not_creation(self):
        driver = _driver(self.vault, environ={"NM_HF_ACCOUNT": "Cutyp"})
        result = driver.provision("hf")
        self.assertEqual(result.outcome, ProvisionOutcome.NEEDS_OWNER_LOGIN)
        self.assertIn("Cutyp", result.message)
        self.assertIn("store-login", result.message)
        # creation was NOT attempted: no checkpoint, no credentials
        self.assertEqual(self.vault.list_all(service="huggingface"), [])

    def test_creation_without_owner_identity_uses_generated(self):
        """Owner's rule (2026-10-20, re-confirmed 2026-10-23): no
        recorded identity is NOT a dead-end — a recorded owner
        identity stays preferred, but creation proceeds on the
        service's self-generated IdentityBank persona when none is
        recorded, and the result says so plainly.

        PREMISE CORRECTION (2026-10-23 fix): this test used to end at
        the creator's CREATION_CHECKPOINT — the pause-as-first-
        response the owner flagged. HF registers a signup flow, so
        the driver now DRIVES the signup itself and completes
        end-to-end; the creator-checkpoint behaviour is pinned on
        flow-less services instead (see
        test_generated_creation_resume_provisions_and_discloses)."""
        from nomorals.accounts.identity_bank import IdentityBank

        mints: list = []
        driver = _driver(self.vault)
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            result = driver.provision("hf")
        # the signup was DRIVEN, not paused: provisioned, no checkpoint
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertTrue(result.account_created, result.message)
        self.assertIsNone(result.checkpoint)
        # the mint went through the trial flow's provider seam, once
        self.assertEqual(mints, ["tempmail"])
        # exactly one persona exists for the service, carrying the
        # minted address (persisted, so retries reuse it)
        personas = IdentityBank(db=self.vault.db, vault=self.vault).list(
            "huggingface")
        self.assertEqual(len(personas), 1)
        persona = personas[0]
        self.assertEqual(persona.email, GEN_EMAIL)
        # the message names the account as a plain fact — name and
        # address, no provenance lecture (PREMISE CORRECTION
        # 2026-10-23 purge: asserting the "self-generated" narration
        # pinned the boilerplate the owner ordered off the system)
        self.assertNotIn("self-generated", result.message)
        self.assertIn(f"Account: {persona.name} <{GEN_EMAIL}>.",
                      result.message)
        self.assertIn(persona.name, result.message)
        self.assertIn(GEN_EMAIL, result.message)
        # ...and the audit trail records the generated identity
        self.assertTrue(any(
            r.get("step") == "create" and r.get("identity") == "generated"
            for r in driver.audit_trail), driver.audit_trail)

    def _register_flowless_service(self, svc_id="flowless"):
        """A registered service with NO signup flow — the creator's
        human-checkpoint path is its creation route (the fallback the
        2026-10-23 fix keeps for services whose signup Devon cannot
        drive). Registered per-test and cleaned up after."""
        svc = register_provision_service(ProvisionService(
            id=svc_id, display="Flowless", vault_service=svc_id,
            env_keys=("FLOWLESS_KEY",),
            login_url="https://flowless.example/login",
            token_url="https://flowless.example/keys",
            validator="flowless_ok", flow="flowless_flow"))
        provisioning._PAGE_FLOWS["flowless_flow"] = (
            lambda tab, s, **kw: "flowless-secret-value")
        provisioning._VALIDATORS["flowless_ok"] = (
            lambda token, http_get: provisioning.ValidationResult(
                "ok", "fine"))
        self.addCleanup(PROVISION_REGISTRY.pop, svc_id, None)
        self.addCleanup(provisioning._PAGE_FLOWS.pop,
                        "flowless_flow", None)
        self.addCleanup(provisioning._VALIDATORS.pop,
                        "flowless_ok", None)
        return svc

    def test_generated_creation_resume_provisions_and_discloses(self):
        """Checkpoint → owner finishes the human step → the drive
        provisions, and the final summary STILL discloses that the
        account was created with a generated identity.

        PREMISE CORRECTION (2026-10-23 fix): this ran on HF, whose
        signup the driver now drives itself (no checkpoint at all —
        see test_creation_without_owner_identity_uses_generated).
        The checkpoint→resume path still exists for services with NO
        registered signup flow, so it is pinned here on one."""
        self._register_flowless_service()
        mints: list = []
        driver = _driver(self.vault)
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            first = driver.provision("flowless")
        self.assertEqual(first.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        # while paused, a re-run reports the same checkpoint WITH
        # the plain-fact account line (from the creation marker) —
        # no provenance lecture (2026-10-23 purge)
        second = driver.provision("flowless")
        self.assertEqual(second.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        self.assertEqual(second.checkpoint["id"], first.checkpoint["id"])
        self.assertNotIn("self-generated", second.message)
        self.assertIn(GEN_EMAIL, second.message)
        # owner completes the human step → credentials land in the vault
        creator = AccountCreator(self.vault, db=self.vault.db)
        finalized = creator.resume_checkpoint(first.checkpoint["id"])
        # the account was finalized under the generated address
        self.assertEqual(finalized.email, GEN_EMAIL)
        with _NoSleep():
            third = driver.provision("flowless")
        self.assertEqual(third.outcome, ProvisionOutcome.PROVISIONED,
                         third.message)
        self.assertTrue(third.account_created, third.message)
        self.assertIn("created a new account", third.message)
        self.assertNotIn("self-generated", third.message)
        self.assertIn(GEN_EMAIL, third.message)

    def test_generated_persona_reused_across_attempts(self):
        """A cancelled creation retried later reuses the SAME bank
        persona and the SAME minted address — no re-mint, ever.

        PREMISE CORRECTION (2026-10-23 fix): pinned on a flow-less
        service — on a signup-flow service (HF) the first attempt
        now completes instead of checkpointing."""
        from nomorals.accounts.identity_bank import IdentityBank

        self._register_flowless_service()
        mints: list = []
        driver = _driver(self.vault)
        with mock.patch.object(AccountCreator, "create_email_account",
                               _fake_email_mint(mints)):
            first = driver.provision("flowless")
            self.assertEqual(first.outcome,
                             ProvisionOutcome.CREATION_CHECKPOINT)
            # owner abandons the human step; creation starts over
            creator = AccountCreator(self.vault, db=self.vault.db)
            creator.cancel_checkpoint(first.checkpoint["id"])
            second = driver.provision("flowless")
        self.assertEqual(second.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT,
                         second.message)
        self.assertNotEqual(second.checkpoint["id"],
                            first.checkpoint["id"])
        self.assertIn(GEN_EMAIL, second.checkpoint["instructions"])
        # still exactly one persona, and the address was minted once
        personas = IdentityBank(db=self.vault.db, vault=self.vault).list(
            "flowless")
        self.assertEqual(len(personas), 1)
        self.assertEqual(mints, ["tempmail"])

    def test_generated_mint_failure_is_honest_not_a_dead_end(self):
        """If the temp-mail mint itself fails, the result says so —
        it does NOT fall back to demanding the owner's identity and
        does NOT invent an address."""
        async def _failed_mint(self, *, provider="guerrilla",
                               username=None):
            return CreatedAccount(
                service="email_tempmail", username="", password="",
                email="", status="failed",
                notes="tempmail.plus error: provider down")

        driver = _driver(self.vault)
        with mock.patch.object(AccountCreator, "create_email_account",
                               _failed_mint):
            result = driver.provision("hf")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.IDENTITY_MINT_FAILED,
                         result.message)
        self.assertNotIn("record your", result.message.lower())
        # nothing was created: no credentials, no checkpoint
        self.assertEqual(self.vault.list_all(service="huggingface"), [])
        from nomorals.accounts.creator import CheckpointStore
        self.assertEqual(
            CheckpointStore(self.vault.db).list_pending(
                service="huggingface"), [])

    def test_generated_identity_sync_completion_full_drive(self):
        """A creator that completes synchronously drives straight to
        a provisioned key, with the generated identity disclosed.

        PREMISE CORRECTION (2026-10-23 fix): the sync creator is the
        FALLBACK for services with no registered signup flow (on HF
        the driver now attempts the browser signup first), so this
        runs on a flow-less service."""
        self._register_flowless_service()
        captured: dict = {}

        class _SyncCreator:
            def __init__(self, vault, db=None, captcha_solver=None):
                self.vault = vault

            def get_owner_identity(self):
                return None

            async def create_email_account(self, *, provider="guerrilla",
                                           username=None):
                captured["provider"] = provider
                captured["mint_username"] = username
                return CreatedAccount(
                    service="email_tempmail", username=GEN_EMAIL,
                    password="", email=GEN_EMAIL, status="created")

            async def create_account(self, service, **kwargs):
                captured["create_kwargs"] = dict(kwargs)
                username = kwargs.get("username") or "genuser"
                self.vault.store(
                    service, username, "gen-password-123",
                    credential_type="password",
                    metadata={
                        "login_url": "https://flowless.example/login"})
                return CreatedAccount(
                    service=service, username=username,
                    password="gen-password-123",
                    email=kwargs.get("email", ""), status="created")

        fake = _SyncCreator(self.vault)
        driver = _driver(
            self.vault,
            creator_factory=lambda vault, db=None, captcha_solver=None: fake)
        with _NoSleep():
            result = driver.provision("flowless")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertTrue(result.account_created)
        # creation ran on the persona: minted address via the trial
        # provider seam, persona username + password passed through
        self.assertEqual(captured["provider"], "tempmail")
        self.assertEqual(captured["create_kwargs"]["email"], GEN_EMAIL)
        self.assertEqual(captured["create_kwargs"]["username"],
                         captured["mint_username"])
        self.assertTrue(captured["create_kwargs"]["password"])
        # plain-fact account line in the final summary + audit
        # (2026-10-23 purge: no "self-generated" narration)
        self.assertNotIn("self-generated", result.message)
        self.assertIn(GEN_EMAIL, result.message)
        self.assertTrue(any(r.get("identity") == "generated"
                            for r in driver.audit_trail))

    def test_creation_with_owner_identity_drives_signup(self):
        """Recorded owner identity + a signup-flow service: the
        driver drives the signup itself and completes in ONE call —
        no human-step checkpoint — and the reply states the identity
        used (owner identity on record, name only)."""
        from nomorals.accounts.identity_bank import IdentityBank

        creator = AccountCreator(self.vault, db=self.vault.db)
        creator.set_owner_identity("Owner", "owner@example.com")
        driver = _driver(self.vault)
        with _NoSleep():
            result = driver.provision("hf")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertTrue(result.account_created, result.message)
        self.assertIsNone(result.checkpoint)
        self.assertIn("created a new account", result.message)
        self.assertIn("Account: Owner.", result.message)
        self.assertNotIn("self-generated", result.message)
        # no persona was minted: the recorded identity was preferred
        self.assertEqual(
            IdentityBank(db=self.vault.db, vault=self.vault).list(
                "huggingface"), [])
        self.assertTrue(any(
            r.get("step") == "create" and r.get("identity") == "owner"
            for r in driver.audit_trail), driver.audit_trail)
        self.assertFalse(any(
            r.get("identity") == "generated" for r in driver.audit_trail))

    def test_creation_checkpoint_then_resume_provisions(self):
        """The creator-fallback path (service with NO signup flow):
        checkpoint on the recorded owner identity → owner finishes
        the human step → the drive provisions.

        PREMISE CORRECTION (2026-10-23 fix): this ran on HF, whose
        signup the driver now drives itself (pinned end-to-end by
        test_creation_with_owner_identity_drives_signup)."""
        self._register_flowless_service()
        creator = AccountCreator(self.vault, db=self.vault.db)
        creator.set_owner_identity("Owner", "owner@example.com")
        driver = _driver(self.vault)
        first = driver.provision("flowless")
        self.assertEqual(first.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT, first.message)
        self.assertIsNotNone(first.checkpoint)
        # the RECORDED owner identity was used, exactly as before:
        # its address is on the creation, and nothing claims a
        # generated identity anywhere
        self.assertIn("owner@example.com", first.checkpoint["instructions"])
        self.assertNotIn("self-generated", first.message)
        self.assertTrue(any(
            r.get("step") == "create" and r.get("identity") == "owner"
            for r in driver.audit_trail), driver.audit_trail)
        self.assertFalse(any(
            r.get("identity") == "generated" for r in driver.audit_trail))
        # still pending: a second call reports the SAME checkpoint,
        # never a duplicate creation
        second = driver.provision("flowless")
        self.assertEqual(second.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        self.assertEqual(second.checkpoint["id"], first.checkpoint["id"])
        # owner completes the human step → credentials land in the vault
        creator.resume_checkpoint(first.checkpoint["id"])
        with _NoSleep():
            third = driver.provision("flowless")
        self.assertEqual(third.outcome, ProvisionOutcome.PROVISIONED,
                         third.message)
        self.assertTrue(third.account_created, third.message)
        self.assertIn("created a new account", third.message)
        self.assertNotIn("self-generated", third.message)
        # ...and the final summary states the account as a plain
        # fact (2026-10-23 purge: no identity narration)
        self.assertIn("Account: Owner.", third.message)

    def test_validation_rejected_keeps_nothing(self):
        self._seed_login()
        driver = _driver(self.vault,
                          http_get=lambda u, h: (401, "unauthorized"))
        with _NoSleep():
            result = driver.provision("hf")
        self.assertEqual(result.outcome, ProvisionOutcome.KEY_REJECTED)
        creds = self.vault.list_all(service="huggingface")
        self.assertFalse(any(c.credential_type == "api_key" for c in creds))

    def test_validation_inconclusive_keeps_nothing(self):
        self._seed_login()

        def _boom(url, headers):
            raise OSError("network down")

        driver = _driver(self.vault, http_get=_boom)
        with _NoSleep():
            result = driver.provision("hf")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.VALIDATION_INCONCLUSIVE)
        creds = self.vault.list_all(service="huggingface")
        self.assertFalse(any(c.credential_type == "api_key" for c in creds))

    def test_page_changed_fails_loudly_at_the_step(self):
        self._seed_login()
        driver = _driver(self.vault,
                         tab_factory=lambda: FakeTab(create_button=False))
        with _NoSleep():
            result = driver.provision("hf")
        self.assertEqual(result.outcome, ProvisionOutcome.PAGE_CHANGED)
        self.assertEqual(result.step, "open_create_dialog")
        creds = self.vault.list_all(service="huggingface")
        self.assertFalse(any(c.credential_type == "api_key" for c in creds))

    def test_unknown_service(self):
        driver = _driver(self.vault)
        result = driver.provision("nosuchservice")
        self.assertEqual(result.outcome, ProvisionOutcome.SERVICE_UNKNOWN)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-bridge-")
        self.vault = _vault(self.tmp)

    def test_bridge_populates_and_never_overrides(self):
        self.vault.store("huggingface", "Cutyp", FAKE_TOKEN,
                         credential_type="api_key")
        environ: dict[str, str] = {}
        applied = bridge_vault_credentials(self.vault, environ=environ)
        self.assertEqual(applied, ["HF_TOKEN"])
        self.assertEqual(environ["HF_TOKEN"], FAKE_TOKEN)
        environ2 = {"HF_TOKEN": "hf_owner_set"}
        applied2 = bridge_vault_credentials(self.vault, environ=environ2)
        self.assertEqual(applied2, [])
        self.assertEqual(environ2["HF_TOKEN"], "hf_owner_set")


class SecondServiceTests(unittest.TestCase):
    """A new service is a registry entry + a flow + a validator —
    the driver plumbing does not change."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="nm-test-provision2-")
        self.vault = _vault(self.tmp)
        self.svc = register_provision_service(ProvisionService(
            id="dummysvc", display="Dummy", vault_service="dummysvc",
            env_keys=("DUMMY_KEY",), login_url="https://dummy.example/login",
            token_url="https://dummy.example/keys", validator="dummy_ok",
            flow="dummy_flow"))
        provisioning._PAGE_FLOWS["dummy_flow"] = (
            lambda tab, svc, **kw: "dummy-secret-value")
        provisioning._VALIDATORS["dummy_ok"] = (
            lambda token, http_get: provisioning.ValidationResult("ok", "fine"))

    def tearDown(self):
        PROVISION_REGISTRY.pop("dummysvc", None)
        provisioning._PAGE_FLOWS.pop("dummy_flow", None)
        provisioning._VALIDATORS.pop("dummy_ok", None)

    def test_dummy_service_provisions_through_same_plumbing(self):
        self.vault.store("dummysvc", "owner", OWNER_PW,
                         credential_type="password",
                         metadata={"login_url": "https://dummy.example/login"})
        environ: dict[str, str] = {}
        driver = _driver(self.vault, environ=environ)
        with _NoSleep():
            result = driver.provision("dummysvc")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED,
                         result.message)
        self.assertEqual(environ.get("DUMMY_KEY"), "dummy-secret-value")


from nomorals.accounts.provisioning import PROVISION_REGISTRY  # noqa: E402


class LogHygieneTests(unittest.TestCase):
    def test_token_and_password_never_in_logs(self):
        tmp = tempfile.mkdtemp(prefix="nm-test-provision-log-")
        vault = _vault(tmp)
        vault.store("huggingface", "Cutyp", OWNER_PW,
                    credential_type="password",
                    metadata={"login_url": "https://huggingface.co/login"})
        driver = _driver(vault, environ={})
        with self.assertLogs(level="DEBUG") as captured:
            with _NoSleep():
                result = driver.provision("hf")
        self.assertEqual(result.outcome, ProvisionOutcome.PROVISIONED)
        blob = "\n".join(captured.output)
        self.assertNotIn(FAKE_TOKEN, blob)
        self.assertNotIn(OWNER_PW, blob)


if __name__ == "__main__":
    unittest.main()
