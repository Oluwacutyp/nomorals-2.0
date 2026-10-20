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
from nomorals.accounts.creator import AccountCreator
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

    def test_creation_without_owner_identity(self):
        driver = _driver(self.vault)
        result = driver.provision("hf")
        self.assertEqual(result.outcome,
                         ProvisionOutcome.NEEDS_OWNER_IDENTITY)

    def test_creation_checkpoint_then_resume_provisions(self):
        creator = AccountCreator(self.vault, db=self.vault.db)
        creator.set_owner_identity("Owner", "owner@example.com")
        driver = _driver(self.vault)
        first = driver.provision("hf")
        self.assertEqual(first.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT, first.message)
        self.assertIsNotNone(first.checkpoint)
        # still pending: a second call reports the SAME checkpoint,
        # never a duplicate creation
        second = driver.provision("hf")
        self.assertEqual(second.outcome,
                         ProvisionOutcome.CREATION_CHECKPOINT)
        self.assertEqual(second.checkpoint["id"], first.checkpoint["id"])
        # owner completes the human step → credentials land in the vault
        creator.resume_checkpoint(first.checkpoint["id"])
        with _NoSleep():
            third = driver.provision("hf")
        self.assertEqual(third.outcome, ProvisionOutcome.PROVISIONED,
                         third.message)
        self.assertTrue(third.account_created, third.message)
        self.assertIn("created a new account", third.message)

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
