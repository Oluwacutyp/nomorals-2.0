"""Env-variable → config resolution.

Regression: the generic NM_<SECTION>_<KEY> form used to search for a field
literally named e.g. ``partner_platforms`` — which never exists — so every
NM_CHAT_* / NM_PARTNER_PLATFORMS / NM_SOCIAL_* variable in .env was silently
ignored and the companion booted on "local" only with no warning.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from nomorals.core.config import env_var_path, load_settings


class EnvVarPathTest(unittest.TestCase):
    def test_explicit_map_wins(self) -> None:
        self.assertEqual(env_var_path("NM_HOME"), "home")
        self.assertEqual(env_var_path("HF_TOKEN"), "llm.hf_token")

    def test_bare_field_name(self) -> None:
        self.assertEqual(env_var_path("NM_MAX_SUBAGENTS"), "concurrency.max_subagents")

    def test_documented_section_key_form(self) -> None:
        self.assertEqual(env_var_path("NM_PARTNER_PLATFORMS"), "partner.platforms")
        self.assertEqual(env_var_path("NM_PARTNER_OWNER_CHATS"), "partner.owner_chats")
        self.assertEqual(env_var_path("NM_PARTNER_US_CHATS"), "partner.us_chats")
        self.assertEqual(env_var_path("NM_PARTNER_AUTONOMY_MODE"), "partner.autonomy_mode")
        self.assertEqual(env_var_path("NM_CHAT_TELEGRAM_ENABLED"), "chat.telegram_enabled")
        self.assertEqual(env_var_path("NM_CHAT_TELEGRAM_API_ID"), "chat.telegram_api_id")
        self.assertEqual(env_var_path("NM_CHAT_DISCORD_TOKEN"), "chat.discord_token")
        self.assertEqual(env_var_path("NM_CHAT_WHATSAPP_PORT"), "chat.whatsapp_port")
        self.assertEqual(env_var_path("NM_CHAT_LOCAL_ENABLED"), "chat.local_enabled")
        self.assertEqual(env_var_path("NM_SOCIAL_X_API_KEY"), "social.x_api_key")
        self.assertEqual(env_var_path("NM_SOCIAL_REDDIT_CLIENT_ID"), "social.reddit_client_id")
        self.assertEqual(env_var_path("NM_TRAINING_BACKEND"), "training.backend")
        self.assertEqual(env_var_path("NM_TRAINING_BASE_MODEL"), "training.base_model")
        self.assertEqual(env_var_path("NM_TRAINING_LORA_R"), "training.lora_r")
        self.assertEqual(env_var_path("NM_TRAINING_REGRESSION_TOLERANCE"),
                         "training.regression_tolerance")

    def test_unknown_variable_resolves_to_none(self) -> None:
        self.assertIsNone(env_var_path("NM_NO_SUCH_SECTION_NO_SUCH_FIELD"))
        self.assertIsNone(env_var_path("UNRELATED_VARIABLE"))


class LoadSettingsChannelConfigTest(unittest.TestCase):
    """The exact variables a user puts in ~/.nomorals/.env must land in
    settings — this is what was silently dropped before the fix."""

    def test_chat_and_partner_env_reach_settings(self) -> None:
        settings = load_settings(
            env={
                "NM_PROFILE": "termux",
                "NM_PARTNER_PLATFORMS": "telegram,discord,local",
                "NM_PARTNER_OWNER_CHATS": "telegram:12345",
                "NM_CHAT_TELEGRAM_ENABLED": "true",
                "NM_CHAT_TELEGRAM_API_ID": "12345678",
                "NM_CHAT_TELEGRAM_API_HASH": "0123456789abcdef",
                "NM_CHAT_DISCORD_ENABLED": "true",
                "NM_CHAT_DISCORD_TOKEN": "dctoken.value.here",
                "NM_CHAT_WHATSAPP_PORT": "9999",
                "NM_CHAT_LOCAL_ENABLED": "true",
            },
            use_env_file=False,
        )
        self.assertEqual(settings.partner.platforms, "telegram,discord,local")
        self.assertEqual(settings.partner.owner_chats, "telegram:12345")
        self.assertTrue(settings.chat.telegram_enabled)
        self.assertEqual(settings.chat.telegram_api_id, "12345678")
        self.assertEqual(settings.chat.telegram_api_hash, "0123456789abcdef")
        self.assertTrue(settings.chat.discord_enabled)
        self.assertEqual(settings.chat.discord_token, "dctoken.value.here")
        self.assertEqual(settings.chat.whatsapp_port, 9999)  # int coercion
        self.assertTrue(settings.chat.local_enabled)

    def test_defaults_stay_off(self) -> None:
        settings = load_settings(env={}, use_env_file=False)
        self.assertEqual(settings.partner.platforms, "local")
        self.assertFalse(settings.chat.telegram_enabled)
        self.assertFalse(settings.chat.discord_enabled)
        self.assertFalse(settings.chat.whatsapp_enabled)


class EnvExampleDocumentationTest(unittest.TestCase):
    """Meta-guard: every variable documented in .env.example (commented or
    not) must resolve to a real config field. This is the test that would
    have caught the silently-ignored NM_CHAT_* / NM_PARTNER_PLATFORMS bug."""

    def test_every_documented_variable_resolves(self) -> None:
        example = Path(__file__).resolve().parent.parent / ".env.example"
        self.assertTrue(example.exists(), ".env.example missing from repo root")
        unresolved = []
        for line in example.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                continue
            key = stripped.lstrip("# ").partition("=")[0].strip()
            # keys like "e.g. git@github.com:..." never appear before '=' in
            # a real variable line; guard just in case
            if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
                continue
            if env_var_path(key) is None:
                unresolved.append(key)
        self.assertEqual(unresolved, [],
                         f".env.example documents variables that never reach config: {unresolved}")


if __name__ == "__main__":
    unittest.main()


class LoadSettingsTrainingEnvTest(unittest.TestCase):
    """Training env vars must reach TrainingSettings, not vanish silently."""

    def test_backend_and_base_model_env_reach_settings(self) -> None:
        s = load_settings(
            env={"NM_TRAINING_BACKEND": "unsloth", "NM_TRAINING_BASE_MODEL": "org/model"},
            use_env_file=False,
        )
        self.assertEqual(s.training.backend, "unsloth")
        self.assertEqual(s.training.base_model, "org/model")

    def test_training_defaults_are_native_and_empty_base(self) -> None:
        s = load_settings(env={}, use_env_file=False)
        self.assertEqual(s.training.backend, "native")
        self.assertEqual(s.training.base_model, "")


class AudioSchedulerApiEnvTest(unittest.TestCase):
    """Voice, scheduler, and API-connector env vars must reach settings."""

    def test_audio_env_reaches_settings(self) -> None:
        s = load_settings(
            env={
                "NM_AUDIO_TTS_ENGINE": "espeak_ng",
                "NM_AUDIO_TTS_VOICE": "en-GB",
                "NM_AUDIO_STT_PROVIDER": "openai_compat",
                "NM_AUDIO_STT_BASE_URL": "https://api.example.com/v1",
                "NM_AUDIO_STT_MODEL": "whisper-1",
            },
            use_env_file=False,
        )
        self.assertEqual(s.audio.tts_engine, "espeak_ng")
        self.assertEqual(s.audio.tts_voice, "en-GB")
        self.assertEqual(s.audio.stt_provider, "openai_compat")
        self.assertEqual(s.audio.stt_base_url, "https://api.example.com/v1")
        self.assertEqual(s.audio.stt_model, "whisper-1")

    def test_audio_defaults(self) -> None:
        s = load_settings(env={}, use_env_file=False)
        self.assertEqual(s.audio.tts_engine, "auto")
        self.assertEqual(s.audio.stt_provider, "auto")
        self.assertEqual(s.audio.audio_dir, "audio")

    def test_scheduler_env_reaches_settings(self) -> None:
        s = load_settings(
            env={"NM_SCHEDULER_ENABLED": "false", "NM_SCHEDULER_TICK_SECONDS": "45"},
            use_env_file=False,
        )
        self.assertFalse(s.scheduler.enabled)
        self.assertEqual(s.scheduler.tick_seconds, 45.0)

    def test_scheduler_defaults(self) -> None:
        s = load_settings(env={}, use_env_file=False)
        self.assertTrue(s.scheduler.enabled)
        self.assertGreater(s.scheduler.tick_seconds, 0)

    def test_api_connector_env_reaches_settings(self) -> None:
        s = load_settings(
            env={
                "NM_API_GITHUB_TOKEN": "gh-token-123",
                "NM_API_WEATHER_LATITUDE": "1.25",
                "NM_API_WEATHER_LONGITUDE": "2.5",
                "NM_API_REQUEST_TIMEOUT": "30",
            },
            use_env_file=False,
        )
        self.assertEqual(s.api.github_token, "gh-token-123")
        self.assertAlmostEqual(s.api.weather_latitude, 1.25)
        self.assertAlmostEqual(s.api.weather_longitude, 2.5)
        self.assertEqual(s.api.request_timeout, 30.0)

    def test_new_secrets_are_redacted(self) -> None:
        s = load_settings(
            env={"NM_API_GITHUB_TOKEN": "gh-secret", "NM_AUDIO_STT_API_KEY": "k"},
            use_env_file=False,
        )
        d = s.to_dict()
        self.assertNotIn("gh-secret", str(d["api"]["github_token"]))
        self.assertNotIn("k", str(d["audio"]["stt_api_key"]))


class ListFieldEnvTest(unittest.TestCase):
    """Env values for list fields must parse into lists.

    Regression: a bare ``NM_LLM_FALLBACK_CHAIN=hf_serverless`` stayed a string
    and the router iterated it character by character — "registering" the
    providers 'h', 'f', '_', 's', … at boot.
    """

    def test_single_value_becomes_one_element_list(self) -> None:
        s = load_settings(env={"NM_LLM_FALLBACK_CHAIN": "hf_serverless"})
        self.assertEqual(s.llm.fallback_chain, ["hf_serverless"])

    def test_comma_separated_values_become_list(self) -> None:
        s = load_settings(env={"NM_LLM_FALLBACK_CHAIN": "groq, hf_serverless"})
        self.assertEqual(s.llm.fallback_chain, ["groq", "hf_serverless"])

    def test_hf_base_url_default_is_the_live_router(self) -> None:
        # api-inference.huggingface.co is retired (410 Gone, 2025-09). The
        # default is the free shared "HF Inference" provider under the new
        # router: the only surface that serves the uncensored fine-tunes
        # (dolphin) that the partner /v1 catalog does not host.
        s = load_settings()
        self.assertEqual(s.llm.hf_base_url, "https://router.huggingface.co/hf-inference")

    def test_retired_hf_serverless_url_is_self_healed(self) -> None:
        # Old .env files set NM_HF_BASE_URL to the decommissioned domain.
        # It must be mapped onto the live successor, not trusted as-is.
        s = load_settings(env={"NM_HF_BASE_URL": "https://api-inference.huggingface.co"})
        self.assertEqual(s.llm.hf_base_url, "https://router.huggingface.co/hf-inference")

    def test_live_hf_base_url_is_not_touched(self) -> None:
        s = load_settings(env={"NM_HF_BASE_URL": "https://router.huggingface.co/v1"})
        self.assertEqual(s.llm.hf_base_url, "https://router.huggingface.co/v1")
