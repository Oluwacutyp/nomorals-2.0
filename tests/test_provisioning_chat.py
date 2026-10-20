"""Self-provisioning credentials (chat layer).

End-to-end through the REAL inbound pipeline (``PartnerRuntime._process``
with a fake adapter), the real grant/gating path, the real driver and
vault — with the browser tab factory and the validation HTTP getter
faked at the runtime's documented seams. No network, no real HF.

What is pinned here:
* a missing key on an owner surface produces the offer (+ armed
  retry); on the gated public bot it produces NEITHER (pass #55);
* a successful /provision drive retries the original /image request;
* the secure login handoff: the armed intake consumes the password
  message BEFORE the brain/autonomy — the secret lands in the vault
  and nowhere else (replies, brain calls and logs are scanned);
* a redelivered intake message is dropped, never reprocessed;
* /browser watch attaches a LiveView on demand, and says so honestly
  when no session is live.
"""

from __future__ import annotations

import logging
import os
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from nomorals.agents.context import build_context
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.core.errors import MissingCredentialError
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

OWNER_ID = "5478650254"
FAKE_TOKEN = "hf_" + "T" * 30
SECRET_PW = "handed-over-password-xyz"


# ── fakes ────────────────────────────────────────────────────────────


class FakeTab:
    """Compact script of the HF pages (same model as the accounts
    suite's FakeTab): login → tokens → dialog → created; license page."""

    def __init__(self) -> None:
        self.url = ""
        self.fills: dict[str, str] = {}
        self.closed = False
        self._dialog = False
        self._created = False
        self._logged_in = False
        self.on_action = None

    def navigate(self, url):
        self.url = url
        return {"ok": True}

    def fill(self, name, value):
        if "tokens" in self.url and (name != "name" or not self._dialog):
            raise Exception(f"no field {name!r}")
        self.fills[name] = value
        return {"ok": True}

    def click(self, target):
        label = (target or "").strip().lower()
        if "tokens" in self.url:
            if label.startswith("create new token"):
                self._dialog = True
                return {"ok": True}
            if label.startswith("create token") or label == "create":
                self._created = True
                return {"ok": True}
            if label in ("read", "write"):
                return {"ok": True}
        if "FLUX.1-Kontext-dev" in self.url and "agree" in label:
            return {"ok": True}
        raise Exception(f"no clickable {target!r}")

    def submit(self, target=""):
        self._logged_in = True
        return {"ok": True}

    def text(self, max_chars=20000):
        if "/login" in self.url and not self._logged_in:
            return {"text": "Log In — email, password"}
        if "tokens" in self.url:
            body = "Access Tokens — Create new token"
            if self._created:
                body += f" — your new token: {FAKE_TOKEN}"
            return {"text": body}
        if "FLUX.1-Kontext-dev" in self.url:
            return {"text": "gated — Agree and access repository"}
        return {"text": "you are logged in"}

    def evaluate(self, js):
        if "hf_" in js and self._created:
            return FAKE_TOKEN
        return None

    def cookies(self):
        return [{"name": "session", "value": "abc"}]

    def check_captcha(self):
        return {"challenges": []}

    def screenshot(self):
        fd, path = tempfile.mkstemp(suffix=".png")
        os.write(fd, b"\x89PNG")
        os.close(fd)
        return {"path": path}

    def close(self):
        self.closed = True


class _FakeAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[str] = []
        self.media_sent: list[str] = []

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def send(self, chat, text, *, reply_to="", buttons=None, **kw):
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")

    def send_media(self, chat, media, *, caption="", **kw):
        self.media_sent.append(caption)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"mm{len(self.media_sent)}")

    def typing(self, chat, seconds=3.0, action="typing"):
        return True


class _FakeImage:
    def save(self, path):
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG-fake")


def _missing_backend():
    backend = SimpleNamespace(name="hf")

    def _gen(prompt):
        raise MissingCredentialError(
            "HF text-to-image needs a Hugging Face token — none is "
            "configured", service="huggingface", env_key="HF_TOKEN",
            feature="media_edit.hf")

    backend.generate = _gen
    backend.describe = lambda: "hf (fake)"
    return backend


def _ok_backend():
    backend = SimpleNamespace(name="hf")
    backend.generate = lambda prompt: [_FakeImage()]
    backend.describe = lambda: "hf (fake)"
    return backend


def _make_runtime(platform: str):
    tmp = tempfile.TemporaryDirectory(prefix="nm-test-provision-chat-")
    settings = load_settings(
        overrides={
            "home": tmp.name,
            "partner.platforms": platform,
            "partner.owner_chats": OWNER_ID,
            "partner.owner_sender_ids": OWNER_ID,
        }
    )
    context = build_context(settings, with_executor=False, with_tools=False)
    adapter = _FakeAdapter(platform)
    gateway = ChatGateway({platform: adapter}, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    return runtime, adapter, context, tmp


def _inbound(platform: str, text: str, message_id: str = "") -> ChatMessage:
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=OWNER_ID,
                     kind=ChatKind.DM, peer="owner"),
        incoming=True, text=text, sender="owner", sender_id=OWNER_ID,
        meta={}, message_id=message_id)


class _LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.records.append(record.getMessage())
        except Exception:  # noqa: BLE001
            pass


class ProvisionChatTests(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k)
                     for k in ("NM_VAULT_PASSPHRASE", "HF_TOKEN",
                               "NM_PROVISION_LIVE_VIEW")}
        os.environ["NM_VAULT_PASSPHRASE"] = "test-pass"
        os.environ.pop("HF_TOKEN", None)
        self._patches = [
            mock.patch("nomorals.accounts.browser_login.time.sleep",
                       lambda s: None),
            mock.patch("nomorals.accounts.provisioning._urllib_get",
                       lambda url, headers: (200, '{"name": "Cutyp"}')),
        ]
        for patcher in self._patches:
            patcher.start()

    def tearDown(self):
        for patcher in self._patches:
            patcher.stop()
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            self.runtime.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            self.context.close()
        except Exception:  # noqa: BLE001
            pass
        self.tmp.cleanup()

    def _setup(self, platform: str = "telegram-personal") -> None:
        self.runtime, self.adapter, self.context, self.tmp = \
            _make_runtime(platform)
        # the runtime's browser seam, faked: fresh scripted tab per call
        tabs: list[FakeTab] = []

        def _open_tab(service, chat=None, label=""):
            tab = FakeTab()
            tabs.append(tab)
            self.runtime._provision_state()["watch_tabs"][service] = tab
            return tab

        self.tabs = tabs
        self._tab_patch = mock.patch.object(
            self.runtime, "_provision_open_tab", _open_tab)
        self._tab_patch.start()
        self.addCleanup(self._tab_patch.stop)

    def _run(self, platform: str, text: str, message_id: str = "") -> None:
        self.runtime._process(_inbound(platform, text, message_id))

    # ── the offer ────────────────────────────────────────────────────

    def test_image_missing_key_offers_on_owner_surface(self):
        self._setup("telegram-personal")
        with mock.patch("nomorals.media_edit.generate.get_backend",
                        return_value=_missing_backend()):
            self._run("telegram-personal", "/image a cat at sunset")
        self.assertEqual(len(self.adapter.sent), 1)
        reply = self.adapter.sent[0]
        self.assertIn("image generation failed", reply)
        self.assertIn("/provision huggingface", reply)
        state = self.runtime._provision_state()
        self.assertIn("telegram-personal:" + OWNER_ID, state["retries"])

    def test_image_missing_key_no_offer_on_gated_surface(self):
        self._setup("telegram-bot")
        with mock.patch("nomorals.media_edit.generate.get_backend",
                        return_value=_missing_backend()):
            self._run("telegram-bot", "/image a cat at sunset")
        self.assertEqual(len(self.adapter.sent), 1)
        reply = self.adapter.sent[0]
        self.assertIn("image generation failed", reply)
        self.assertNotIn("/provision", reply)

    # ── /provision command ───────────────────────────────────────────

    def test_provision_status_board(self):
        self._setup("telegram-personal")
        out = self.runtime.handle_control(
            "/provision", f"telegram-personal:{OWNER_ID}",
            message=_inbound("telegram-personal", "/provision"))
        self.assertIn("huggingface", out)
        self.assertIn("not provisioned", out)

    def test_provision_never_runs_on_gated_surface(self):
        self._setup("telegram-bot")
        self._run("telegram-bot", "/provision hf")
        self.assertEqual(len(self.adapter.sent), 1)
        self.assertNotIn("provisioned Hugging Face", self.adapter.sent[0])
        vault = self.runtime._provision_vault()
        self.assertEqual(vault.list_all(service="huggingface"), [])

    def test_drive_then_pending_image_retries(self):
        self._setup("telegram-personal")
        backends = [_missing_backend(), _ok_backend()]
        with mock.patch(
                "nomorals.media_edit.generate.get_backend",
                side_effect=lambda: backends.pop(0)):
            self._run("telegram-personal", "/image a cat at sunset")
            self.assertIn("/provision huggingface", self.adapter.sent[-1])
            # the owner hands Devon a login path: seed the vault login
            vault = self.runtime._provision_vault()
            vault.store("huggingface", "Cutyp", "pw-for-tests",
                        credential_type="password",
                        metadata={"login_url": "https://huggingface.co/login"})
            self._run("telegram-personal", "/provision hf")
        reply = self.adapter.sent[-1]
        self.assertIn("provisioned Hugging Face", reply)
        self.assertIn("retrying what you asked for:", reply)
        self.assertIn("🎨 generated", reply)
        # the key the drive created is the one the retry used
        self.assertEqual(os.environ.get("HF_TOKEN"), FAKE_TOKEN)

    # ── secure login handoff ─────────────────────────────────────────

    def _brain_recorder(self) -> list[str]:
        calls: list[str] = []

        def _record(message, **kw):
            calls.append(message.text)
            return SimpleNamespace(
                parts=[], presence=SimpleNamespace(delay_seconds=0))

        self.runtime.brain.handle_message = _record
        return calls

    def test_login_handoff_consumes_secret_before_brain(self):
        self._setup("telegram-personal")
        brain_calls = self._brain_recorder()
        capture = _LogCapture()
        logging.getLogger().addHandler(capture)
        try:
            self._run("telegram-personal", "/provision login hf Cutyp")
            self.assertIn("armed", self.adapter.sent[-1])
            sends_before = len(self.adapter.sent)
            self._run("telegram-personal", SECRET_PW, message_id="sec-1")
        finally:
            logging.getLogger().removeHandler(capture)
        # consumed: the owner got the handoff + drive replies
        new_sends = self.adapter.sent[sends_before:]
        self.assertTrue(any("login saved" in s for s in new_sends),
                        new_sends)
        self.assertTrue(any("provisioned Hugging Face" in s
                            for s in new_sends), new_sends)
        # the vault holds the login AND the fetched key
        vault = self.runtime._provision_vault()
        login = vault.get("huggingface", "Cutyp")
        self.assertEqual(login.password, SECRET_PW)
        self.assertTrue(any(c.credential_type == "api_key"
                            for c in vault.list_all(service="huggingface")))
        # hygiene: the secret reached NO reply, NO brain call, NO log
        self.assertFalse(any(SECRET_PW in s for s in self.adapter.sent))
        self.assertFalse(any(SECRET_PW in t for t in brain_calls))
        self.assertFalse(any(SECRET_PW in r for r in capture.records))
        self.assertFalse(any(FAKE_TOKEN in s for s in self.adapter.sent))
        # a redelivery of the same message id is dropped, not re-run
        count = len(self.adapter.sent)
        self._run("telegram-personal", SECRET_PW, message_id="sec-1")
        self.assertEqual(len(self.adapter.sent), count)
        self.assertFalse(any(SECRET_PW in t for t in brain_calls))

    def test_intake_not_armed_means_normal_chat(self):
        self._setup("telegram-personal")
        brain_calls = self._brain_recorder()
        self._run("telegram-personal", "just a normal message")
        self.assertEqual(brain_calls, ["just a normal message"])

    # ── /browser watch ───────────────────────────────────────────────

    def test_browser_watch_honest_when_nothing_live(self):
        self._setup("telegram-personal")
        out = self.runtime.handle_control(
            "/browser watch", f"telegram-personal:{OWNER_ID}",
            message=_inbound("telegram-personal", "/browser watch"))
        self.assertIn("no live browser session", out)

    def test_browser_watch_attaches_to_live_session(self):
        self._setup("telegram-personal")
        tab = FakeTab()
        self.runtime._provision_state()["watch_tabs"]["huggingface"] = tab
        out = self.runtime.handle_control(
            "/browser watch", f"telegram-personal:{OWNER_ID}",
            message=_inbound("telegram-personal", "/browser watch"))
        self.assertIn("live view attached", out)
        self.assertIn("huggingface", out)
        self.assertTrue(self.adapter.media_sent or self.adapter.sent)
        # the tab now reports actions through the view
        self.assertIsNotNone(tab.on_action)

    # ── startup bridge ───────────────────────────────────────────────

    def test_startup_bridge_populates_env(self):
        self._setup("telegram-personal")
        vault = self.runtime._provision_vault()
        vault.store("huggingface", "Cutyp", FAKE_TOKEN,
                    credential_type="api_key")
        applied = self.runtime._bridge_provisioned_credentials()
        self.assertEqual(applied, ["HF_TOKEN"])
        self.assertEqual(os.environ.get("HF_TOKEN"), FAKE_TOKEN)


class StoreLoginCliTests(unittest.TestCase):
    def test_store_login_writes_vault_via_hidden_prompt_env(self):
        from nomorals.cmdline.commands.account import _cmd_account, _vault
        from nomorals.cmdline.parser import _parser
        from nomorals.storage.db import Database

        tmp = tempfile.mkdtemp(prefix="nm-test-storelogin-")
        db = Database(os.path.join(tmp, "cli.db"))
        db.migrate()
        old = {k: os.environ.get(k)
               for k in ("NM_VAULT_PASSPHRASE", "NM_ACCOUNT_PASSWORD")}
        os.environ["NM_VAULT_PASSPHRASE"] = "test-pass"
        os.environ["NM_ACCOUNT_PASSWORD"] = SECRET_PW
        try:
            args = _parser().parse_args(
                ["account", "store-login", "--service", "hf",
                 "--username", "Cutyp"])
            context = SimpleNamespace(db=db, settings=None)
            rc = _cmd_account(args, context)
            self.assertEqual(rc, 0)
            vault = _vault(context)
            cred = vault.get("huggingface", "Cutyp")
            self.assertEqual(cred.password, SECRET_PW)
            self.assertEqual(cred.credential_type, "password")
            self.assertEqual(cred.metadata.get("login_url"),
                             "https://huggingface.co/login")
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
