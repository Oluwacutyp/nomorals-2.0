"""Infra spec P0 (§8.1 + §4.3): config mapping + never-empty errors.

Failing-first for G2 (documented .env keys silently ignored) and G1/G8
(generic / empty error reasons).
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

from nomorals.core.config import (
    _ENV_MAP,
    config_env_status,
    direct_env_consumer,
    dotted_path_exists,
    env_var_path,
    is_direct_env_key,
    load_settings,
)
from nomorals.core.errors import (
    classify_user_error,
    exc_str,
    format_error,
    redact_secrets,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "scripts" / "devon_ec2_env.template"

# Template lines that are NOT settable keys (same exclusions as the
# pre-existing test_devon_ec2_deploy.py): the documented unknown-key
# fixture, and prefix fragments (NM_HUB_ / NM_RUNTIME_ are prefixes,
# not keys you can set).
EXCLUDED = {"NM_FOO_BAR"}


def _template_keys() -> list[str]:
    keys = []
    for line in TEMPLATE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"#?\s*(NM_[A-Z0-9_]+)\s*=", line)
        if m:
            keys.append(m.group(1))
    return sorted(set(keys))


class TestConfigMapping(unittest.TestCase):
    def test_template_keys_all_mapped(self):
        """Every documented template key has an _ENV_MAP entry (spec §8.1)."""
        missing = [k for k in _template_keys()
                   if k not in _ENV_MAP and k not in EXCLUDED and not k.endswith("_")]
        self.assertEqual(missing, [], f"template keys with no _ENV_MAP entry: {missing}")

    def test_direct_entries_have_real_consumers(self):
        """Every direct: entry names a module that actually references the key."""
        bad = []
        for key, value in _ENV_MAP.items():
            if not value.startswith("direct:"):
                continue
            module = value[len("direct:"):]
            path = REPO_ROOT / module
            if not path.is_file() or key not in path.read_text(encoding="utf-8", errors="replace"):
                bad.append(f"{key} -> {module}")
        self.assertEqual(bad, [], f"direct entries with no real consumer: {bad}")

    def test_dotted_entries_resolve_to_real_fields(self):
        """Every dotted _ENV_MAP value names a real Settings field path."""
        bad = [f"{k} -> {v}" for k, v in _ENV_MAP.items()
               if not v.startswith("direct:") and not dotted_path_exists(v)]
        self.assertEqual(bad, [], f"_ENV_MAP values with no real field: {bad}")

    def test_env_var_path_helpers(self):
        self.assertEqual(env_var_path("NM_VOICE_STT_PROVIDER"), "audio.stt_provider")
        self.assertTrue(is_direct_env_key("NM_CAPTCHA_SOLVER"))
        self.assertEqual(direct_env_consumer("NM_CAPTCHA_SOLVER"),
                         "nomorals/tools/captcha.py")
        self.assertIsNone(direct_env_consumer("NM_VOICE_STT_PROVIDER"))
        self.assertIsNone(env_var_path("NM_FOO_BAR"))

    def test_ignored_keys_detected(self):
        status = config_env_status({"NM_FOO_BAR": "1", "NM_VOICE_STT_PROVIDER": "api"})
        self.assertIn("NM_FOO_BAR", status["ignored"])
        self.assertNotIn("NM_VOICE_STT_PROVIDER", status["ignored"])

    def test_alias_key_takes_effect(self):
        """NM_VOICE_TTS_PROVIDER (previously ignored) now drives audio.tts_engine."""
        settings = load_settings(env={"NM_VOICE_TTS_PROVIDER": "espeak"}, use_env_file=False)
        self.assertEqual(settings.audio.tts_engine, "espeak")

    def test_trading_keys_take_effect(self):
        settings = load_settings(env={"NM_TRADING_ALLOW_LIVE": "1"}, use_env_file=False)
        self.assertTrue(settings.trading.live_enabled)
        settings = load_settings(env={"NM_TRADING_ENABLED": "0"}, use_env_file=False)
        self.assertFalse(settings.trading.enabled)
        settings = load_settings(env={"NM_TRADING_MODE": "paper"}, use_env_file=False)
        self.assertEqual(settings.trading.mode, "paper")

    def test_direct_keys_not_injected_into_settings(self):
        """A direct: key must not create a phantom Settings field."""
        settings = load_settings(env={"NM_CAPTCHA_SOLVER": "2captcha"}, use_env_file=False)
        self.assertFalse(hasattr(settings, "captcha_solver"))


class TestNeverEmptyErrors(unittest.TestCase):
    def test_exc_str_never_empty(self):
        self.assertNotEqual(exc_str(EOFError()), "")
        self.assertIn("EOFError", exc_str(EOFError()))
        self.assertIn("wave", exc_str(EOFError(), context="wave.open"))
        self.assertNotEqual(exc_str(None), "")
        self.assertNotEqual(exc_str(ValueError("boom")), "")

    def test_format_error_shape(self):
        out = format_error("transcribe that voice note", EOFError())
        self.assertIn("transcribe", out)
        self.assertIn("EOFError", out)
        self.assertNotIn("()", out)
        out2 = format_error("transcribe that voice note",
                            Exception("STT API 401: bad key"))
        self.assertIn("auth", out2)
        self.assertNotIn("bad key", out2.replace("bad key", ""))  # cause present
        self.assertIn("STT API 401", out2)

    def test_redact_secrets(self):
        redacted = redact_secrets("failed: Bearer abcDEF123._-~/= and api_key=sekret123")
        self.assertNotIn("abcDEF123", redacted)
        self.assertNotIn("sekret123", redacted)
        self.assertIn("Bearer ***", redacted)

    def test_classify_user_error(self):
        class E(Exception):
            def __init__(self, msg, status=None):
                super().__init__(msg)
                self.status = status
        self.assertEqual(classify_user_error(E("401 Unauthorized", 401)), "auth")
        self.assertEqual(classify_user_error(
            E("unsupported_audio_format", 400)), "format")
        self.assertEqual(classify_user_error(E("connection refused")), "network")
        self.assertEqual(classify_user_error(E("no STT backend")), "config")
        self.assertEqual(classify_user_error(E("weird boom")), "bug")

    def test_short_stt_reason_never_bare(self):
        from nomorals.agents.partner.brain import _short_stt_reason as brain_reason
        from nomorals.voice.pingpong import _short_stt_reason as pingpong_reason
        for fn in (brain_reason, pingpong_reason):
            self.assertNotEqual(fn(EOFError()), "STT API error")
            self.assertIn("EOFError", fn(EOFError()))
            self.assertIn("401", fn(Exception("STT API 401: bad key")))
            self.assertNotIn("sekret", fn(Exception("api_key=sekret")))

    def test_vocal_lite_exc_str_delegates(self):
        from nomorals.media.vocal_lite import _exc_str as vocal_exc_str
        self.assertIn("EOFError", vocal_exc_str(EOFError()))
        self.assertEqual(vocal_exc_str(ValueError("boom")), "boom")


class TestTradingEnabledGate(unittest.TestCase):
    def test_disabled_trading_refuses(self):
        import types
        from nomorals.core.config import TradingSettings

        class FakeSettings:
            trading = TradingSettings(enabled=False)

        class FakeContext:
            settings = FakeSettings()
            db = None

        from nomorals.tools import trading as trading_mod

        # register() builds the closure over registry.context; capture it
        # via a fake registry whose .register is a pass-through decorator.
        captured: dict = {}

        class FakeRegistry:
            context = FakeContext()

            def register(self, name, **kw):
                def deco(fn):
                    captured[name] = fn
                    return fn
                return deco

        trading_mod.register(FakeRegistry())
        fn = captured["trading"]
        out = fn("analyze", symbol="XAUUSD")
        self.assertFalse(out.get("ok", True))
        self.assertIn("disabled", str(out.get("error", "")))
        # doctor + kill stay available even when disabled.
        self.assertNotIn("disabled", str(fn("doctor")))


if __name__ == "__main__":
    unittest.main()
