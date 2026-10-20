"""Pass #43 — startup + doctor + first-run health.

Defects pinned here (all reproduced on pristine HEAD c99f41b0):

1. ``config_env_status`` (and its helpers ``dotted_path_exists``,
   ``is_direct_env_key``, ``direct_env_consumer``) did not exist, so
   ``nm doctor``'s config-keys section never printed and chat
   ``/doctor`` reported a spurious "config: check failed" forever.
2. ``NM_VOICE_TTS_PROVIDER`` — documented in the env template and the
   Windows installer — mapped to nothing; the owner's setting was
   silently ignored (only ``NM_VOICE_TTS_ENGINE`` worked).
3. ``NM_HOME_PLACE`` / ``NM_WORK_PLACE`` / ``NM_PLACES`` / ``NM_LOCALE``
   — documented in ``.env.example`` — were read by nothing; the weather
   adapter duck-types a ``settings.location`` section that did not
   exist, so configured places never reached it.
4. ``ping_owner_dm`` read the bot token ONLY from ``os.environ`` while
   ``load_settings`` reads the ``.env`` FILE without exporting it — on
   the file-based install the startup/ready/failure DMs never fired.
5. Chat ``/doctor``'s providers check read ``os.environ`` only — a
   different source than the runtime's effective settings — and used
   the bare ``OPENAI_API_KEY`` name the settings layer never reads.
6. ``nm setup`` wrote ``~/.nomorals/.env`` (not the file boot reads
   first), dropped every comment from the existing file, and wrote
   keys the router never reads for OpenRouter (``NM_OPENAI_API_KEY``),
   GGUF (``NM_LLAMA_CPP_MODEL``) and OpenAI (bare ``OPENAI_API_KEY``) —
   then printed a vacuous "✅ Model responded" off the mock fallback.
7. ``nm doctor`` exited 0 unconditionally, so the deploy script's
   "VERIFY FAILED" gate on a nonzero doctor exit could never fire.
"""
from __future__ import annotations

import io
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest import mock

from nomorals.agents.startup_progress import ping_owner_dm
from nomorals.core.config import (
    _ENV_MAP,
    config_env_status,
    direct_env_consumer,
    dotted_path_exists,
    env_var_path,
    is_direct_env_key,
    load_settings,
)


class TestConfigEnvStatus(unittest.TestCase):
    def test_classifies_set_keys(self):
        status = config_env_status(env={
            "NM_FOO_BAR": "1",
            "NM_VOICE_STT_PROVIDER": "api",
            "NM_CAPTCHA_SOLVER": "1",
            "NM_HOME": "/tmp/nm-home",
        })
        self.assertIn("NM_FOO_BAR", status["ignored"])
        self.assertIn("NM_VOICE_STT_PROVIDER", status["set_mapped"])
        self.assertNotIn("NM_VOICE_STT_PROVIDER", status["ignored"])
        self.assertIn("NM_CAPTCHA_SOLVER", status["set_direct"])
        self.assertIn("NM_HOME", status["set_mapped"])
        self.assertIn("NM_LLM_TIMEOUT", status["unset_mapped"])
        for value in status.values():
            self.assertIsInstance(value, list)

    def test_default_env_includes_boot_env_file(self):
        # The runtime's effective env is os.environ merged OVER the boot
        # .env file; the doctor must audit the same thing.
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            with open(os.path.join(tmp, ".env"), "w", encoding="utf-8") as fh:
                fh.write("NM_FOO_BAR=1\nNM_GROQ_API_KEY=gsk-file\n")
            old_cwd = os.getcwd()
            os.chdir(tmp)
            try:
                with mock.patch.dict(os.environ, {}, clear=True):
                    status = config_env_status()
            finally:
                os.chdir(old_cwd)
        self.assertIn("NM_FOO_BAR", status["ignored"])
        self.assertIn("NM_GROQ_API_KEY", status["set_mapped"])

    def test_direct_helpers(self):
        self.assertTrue(is_direct_env_key("NM_CAPTCHA_SOLVER"))
        self.assertEqual(direct_env_consumer("NM_CAPTCHA_SOLVER"),
                         "nomorals/tools/captcha.py")
        self.assertFalse(is_direct_env_key("NM_VOICE_STT_PROVIDER"))
        self.assertIsNone(direct_env_consumer("NM_VOICE_STT_PROVIDER"))
        self.assertIsNone(direct_env_consumer("NM_FOO_BAR"))
        self.assertIsNone(env_var_path("NM_FOO_BAR"))
        self.assertEqual(env_var_path("NM_VOICE_STT_PROVIDER"),
                         "audio.stt_provider")
        self.assertEqual(env_var_path("NM_CAPTCHA_SOLVER"),
                         "direct:nomorals/tools/captcha.py")

    def test_every_mapped_path_exists_and_direct_modules_read_their_key(self):
        from pathlib import Path
        root = Path(__file__).resolve().parent.parent
        for key, target in _ENV_MAP.items():
            with self.subTest(key=key):
                if target.startswith("direct:"):
                    module = target[len("direct:"):]
                    src = (root / module).read_text(encoding="utf-8")
                    self.assertIn(key, src)
                else:
                    self.assertTrue(dotted_path_exists(target),
                                    f"{key} -> {target}")
        self.assertFalse(dotted_path_exists("nope.nope"))

    def test_direct_key_does_not_break_settings_load(self):
        # A direct: entry is consumed by its module at runtime, NOT by
        # the settings tree: loading with it set must not raise, and no
        # bogus settings field may appear.
        settings = load_settings(env={"NM_CAPTCHA_SOLVER": "0"})
        self.assertFalse(hasattr(settings, "direct"))


class TestDocumentedKeysTakeEffect(unittest.TestCase):
    def test_tts_provider_alias(self):
        settings = load_settings(env={"NM_VOICE_TTS_PROVIDER": "espeak"})
        self.assertEqual(settings.audio.tts_engine, "espeak")

    def test_llama_cpp_model_legacy_alias(self):
        # Installs configured by the old wizard carry NM_LLAMA_CPP_MODEL;
        # it must drive llm.local_model like the canonical key.
        settings = load_settings(env={"NM_LLAMA_CPP_MODEL": "/m/x.gguf"})
        self.assertEqual(settings.llm.local_model, "/m/x.gguf")

    def test_location_keys(self):
        settings = load_settings(env={
            "NM_HOME_PLACE": "Gunnison",
            "NM_WORK_PLACE": "Crested Butte",
            "NM_PLACES": "Gunnison, Denver",
            "NM_LOCALE": "en-US",
        })
        self.assertEqual(settings.location.home_place, "Gunnison")
        self.assertEqual(settings.location.work_place, "Crested Butte")
        self.assertEqual(settings.location.places, ["Gunnison", "Denver"])
        self.assertEqual(settings.location.locale, "en-US")
        self.assertEqual(env_var_path("NM_HOME_PLACE"), "location.home_place")
        self.assertEqual(env_var_path("NM_PLACES"), "location.places")

    def test_location_reaches_weather_places(self):
        from nomorals.agency.adapters import places_from_settings
        settings = load_settings(env={"NM_HOME_PLACE": "Gunnison",
                                      "NM_PLACES": "Denver"})
        places = places_from_settings(settings)
        self.assertIn("Gunnison", places)
        self.assertIn("Denver", places)


class TestPingTokenSource(unittest.TestCase):
    def _ok_body(self):
        body = mock.Mock()
        body.read.return_value = b'{"ok": true}'
        body.__enter__ = lambda self: self
        body.__exit__ = lambda *args: False
        return body

    def test_token_from_settings_chat_section(self):
        # File-based install: the token lives in settings (loaded from
        # .env); os.environ never sees it. The ping must still fire.
        from nomorals.core.config import Settings
        settings = Settings()
        settings.chat.telegram_bot_token = "tok-from-settings"
        settings.partner.owner_chats = "5478650254"
        captured = {}

        def fake_urlopen(request, timeout=None):
            captured["url"] = request.full_url
            return self._ok_body()

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch("nomorals.agents.startup_progress."
                           "urllib.request.urlopen", side_effect=fake_urlopen):
            self.assertTrue(ping_owner_dm(settings, "Devon is starting…"))
        self.assertIn("bottok-from-settings", captured["url"])

    def test_env_token_fallback_preserved(self):
        settings = SimpleNamespace(
            partner=SimpleNamespace(owner_chats="telegram:1"))
        with mock.patch.dict(
                os.environ, {"NM_CHAT_TELEGRAM_BOT_TOKEN": "x"}, clear=False), \
                mock.patch("nomorals.agents.startup_progress."
                           "urllib.request.urlopen",
                           return_value=self._ok_body()):
            self.assertTrue(ping_owner_dm(settings, "hello"))


class TestChatDoctor(unittest.TestCase):
    def _runtime(self, tmp_home: str, **llm_overrides):
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings
        from nomorals.social.chat.base import ChatAdapter, SendResult
        from nomorals.social.chat.gateway import ChatGateway

        class FakeAdapter(ChatAdapter):
            def __init__(self):
                super().__init__(media_dir="/tmp/nm-test-media")
            def run(self, handler):
                pass
            def send(self, chat, text, *, reply_to="", buttons=None, **kwargs):
                return SendResult(ok=True, platform="telegram",
                                  message_id="m1")

        overrides = {
            "home": tmp_home,
            "partner.platforms": "telegram",
            "partner.owner_chats": "5478650254",
            "partner.owner_sender_ids": "5478650254",
        }
        overrides.update(llm_overrides)
        settings = load_settings(overrides=overrides)
        context = build_context(settings, with_executor=False,
                                with_tools=False)
        gateway = ChatGateway({"telegram": FakeAdapter()}, db=context.db)
        return PartnerRuntime(context, gateway=gateway)

    def test_config_line_reports_counts(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            rt = self._runtime(tmp)
            with mock.patch.dict(os.environ, {}, clear=True):
                out = rt.handle_control("/doctor", "telegram:5478650254")
        self.assertRegex(out, r"OK\s+config: \d+ set / \d+ unset, 0 ignored")

    def test_providers_check_reads_effective_settings(self):
        # Provider + key live in settings (from the .env file at boot);
        # the process env is empty. Pre-fix this reported a PASS for
        # "no provider chain configured" while groq was actually live.
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            rt = self._runtime(tmp, **{
                "llm.provider": "groq",
                "llm.groq_api_key": "gsk-from-settings",
            })
            with mock.patch.dict(os.environ, {}, clear=True):
                out = rt.handle_control("/doctor", "telegram:5478650254")
        self.assertIn("providers: chain groq", out)
        self.assertNotIn("FAIL providers:", out)

    def test_providers_check_missing_settings_key_fails(self):
        # The env var alone must NOT satisfy the check: the runtime
        # reads the key from settings, so an env-only key with an empty
        # settings field is a real failure the doctor must name.
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            rt = self._runtime(tmp, **{"llm.provider": "groq"})
            with mock.patch.dict(os.environ,
                                 {"NM_GROQ_API_KEY": "gsk-env-only"},
                                 clear=True):
                out = rt.handle_control("/doctor", "telegram:5478650254")
        self.assertIn("FAIL providers:", out)
        self.assertIn("groq", out)


class TestCliDoctorExitCode(unittest.TestCase):
    def _run_doctor(self, tmp: str, extra_env: dict,
                    *argv: str) -> tuple[int, str]:
        from nomorals.cli import main
        old_cwd = os.getcwd()
        os.chdir(tmp)
        try:
            with mock.patch.dict(os.environ, extra_env, clear=True):
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = main([*argv, "doctor"])
            return rc, buf.getvalue()
        finally:
            os.chdir(old_cwd)

    def test_ignored_key_exits_nonzero_and_is_named(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            rc, out = self._run_doctor(
                tmp, {"NM_HOME": tmp, "NM_FOO_BAR": "1"})
        self.assertEqual(rc, 1)
        self.assertIn("NM_FOO_BAR", out)
        self.assertIn("config keys:", out)

    def test_clean_environment_exits_zero(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            rc, out = self._run_doctor(tmp, {"NM_HOME": tmp})
        self.assertEqual(rc, 0, out)
        self.assertIn("config keys:", out)

    def test_json_mode_reports_config_and_exit_code(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-") as tmp:
            rc, out = self._run_doctor(
                tmp, {"NM_HOME": tmp, "NM_FOO_BAR": "1"}, "--json")
        self.assertEqual(rc, 1)
        payload = json.loads(out)
        self.assertIn("NM_FOO_BAR", payload["config_keys"]["ignored"])


class TestSetupWizard(unittest.TestCase):
    def _run_wizard(self, tmp_cwd: str, tmp_home: str, answers: list[str]):
        from nomorals.cmdline.commands.doctor import _cmd_setup
        env_file = os.path.join(tmp_cwd, ".env")
        with open(env_file, "w", encoding="utf-8") as fh:
            fh.write("# my hand-kept comment\nNM_UNRELATED_KEY=keepme\n")
        context = SimpleNamespace(
            router=SimpleNamespace(active_model="mock"),
            settings=SimpleNamespace(home=tmp_home),
            db=None, bus=None)
        old_cwd = os.getcwd()
        os.chdir(tmp_cwd)
        try:
            with mock.patch.dict(os.environ, {"NM_HOME": tmp_home},
                                 clear=True), \
                    mock.patch("builtins.input",
                               side_effect=list(answers)) as _in:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = _cmd_setup(SimpleNamespace(), context)
            with open(env_file, encoding="utf-8") as fh:
                return rc, fh.read(), buf.getvalue()
        finally:
            os.chdir(old_cwd)

    def test_openrouter_writes_keys_the_router_reads(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-wiz-") as cwd, \
                tempfile.TemporaryDirectory(prefix="nm-p43-home-") as home:
            rc, content, _out = self._run_wizard(
                cwd, home, ["1", "sk-or-testkey", "2", "n"])
        self.assertEqual(rc, 0)
        self.assertIn("NM_LLM_PROVIDER=openrouter", content)
        self.assertIn("NM_OPENROUTER_API_KEY=sk-or-testkey", content)
        self.assertIn("NM_OPENROUTER_MODEL=openai/gpt-oss-120b:free", content)
        self.assertNotIn("NM_OPENAI_API_KEY", content)
        # The operator's existing file survives: comments + other keys.
        self.assertIn("# my hand-kept comment", content)
        self.assertIn("NM_UNRELATED_KEY=keepme", content)

    def test_openai_writes_nm_prefixed_key(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-wiz-") as cwd, \
                tempfile.TemporaryDirectory(prefix="nm-p43-home-") as home:
            rc, content, _out = self._run_wizard(
                cwd, home, ["4", "sk-testkey", "n"])
        self.assertEqual(rc, 0)
        self.assertIn("NM_OPENAI_API_KEY=sk-testkey", content)
        self.assertNotRegex(content, r"(?m)^OPENAI_API_KEY=")

    def test_gguf_writes_local_model_key(self):
        with tempfile.TemporaryDirectory(prefix="nm-p43-wiz-") as cwd, \
                tempfile.TemporaryDirectory(prefix="nm-p43-home-") as home:
            gguf = os.path.join(cwd, "model.gguf")
            with open(gguf, "wb") as fh:
                fh.write(b"GGUF")
            rc, content, _out = self._run_wizard(
                cwd, home, ["3", "3", gguf, "n"])
        self.assertEqual(rc, 0)
        self.assertIn(f"NM_LLM_LOCAL_MODEL={gguf}", content)
        self.assertNotIn("NM_LLAMA_CPP_MODEL", content)

    def test_env_update_home_targets_the_boot_file(self):
        # cwd/.env is what load_settings reads FIRST; a promote/setup
        # write that lands in ~/.nomorals/.env is shadowed forever.
        from nomorals.cmdline.commands.doctor import _env_update_home
        with tempfile.TemporaryDirectory(prefix="nm-p43-wiz-") as cwd, \
                tempfile.TemporaryDirectory(prefix="nm-p43-home-") as home:
            with open(os.path.join(cwd, ".env"), "w",
                      encoding="utf-8") as fh:
                fh.write("NM_EXISTING=1\n")
            context = SimpleNamespace(
                settings=SimpleNamespace(home=home))
            old_cwd = os.getcwd()
            os.chdir(cwd)
            try:
                with mock.patch.dict(os.environ, {"NM_HOME": home},
                                     clear=True):
                    path = _env_update_home(
                        context, {"NM_LLM_PROVIDER": "llama_cpp"})
            finally:
                os.chdir(old_cwd)
            self.assertEqual(str(path), os.path.join(cwd, ".env"))
            with open(os.path.join(cwd, ".env"), encoding="utf-8") as fh:
                content = fh.read()
            self.assertIn("NM_LLM_PROVIDER=llama_cpp", content)
            self.assertIn("NM_EXISTING=1", content)
            self.assertFalse(os.path.exists(os.path.join(home, ".env")))


if __name__ == "__main__":
    unittest.main()
