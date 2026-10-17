"""Telegram bot tokens must never appear in logs / errors / messages.

Regression: on Windows EC2 the bot logged
``chat preflight failed for telegram-bot: request to
https://api.telegram.org/bot<full token>/getMe failed: [SSL: ...]`` —
the FULL bot token in a log line. The token lives in the request URL
(HttpClient embeds it in error messages), so every layer that builds a
message from a URL must redact it.

The token below is FAKE (same shape as a real one, invented for tests).
Never paste a real token into a test.
"""

from __future__ import annotations

import importlib.util
import logging
import pathlib
import re

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

FAKE_BOT_ID = "123456789"
FAKE_SECRET = "AAFakeFakeFakeFakeFakeFakeFakeFake12"
FAKE_TOKEN = f"{FAKE_BOT_ID}:{FAKE_SECRET}"
FAKE_URL = f"https://api.telegram.org/bot{FAKE_TOKEN}/getMe"
FULL_TOKEN_RE = re.compile(r"bot\d+:[A-Za-z0-9_\-]{20,}")


def _assert_redacted(text: str) -> None:
    assert FAKE_SECRET not in text, "full bot token leaked"
    assert FULL_TOKEN_RE.search(text) is None, f"token pattern present: {text!r}"
    assert f"bot{FAKE_BOT_ID}:***" in text


def test_redact_scrubs_bot_token_url():
    from nomorals.core.logging_setup import redact

    _assert_redacted(redact(f"request to {FAKE_URL} failed: [SSL: CERTIFICATE_VERIFY_FAILED]"))


def test_redaction_filter_scrubs_log_record():
    from nomorals.core.logging_setup import RedactionFilter

    record = logging.LogRecord(
        name="test", level=logging.WARNING, pathname=__file__,
        lineno=1, msg="chat preflight failed for telegram-bot: %s",
        args=(f"request to {FAKE_URL} failed",), exc_info=None,
    )
    assert RedactionFilter().filter(record) is True
    _assert_redacted(record.getMessage())


def test_http_redact_url_and_http_error():
    from nomorals.core import http as http_mod

    _assert_redacted(http_mod._redact_url(FAKE_URL))
    err = http_mod.http_error(401, "{}", FAKE_URL)
    _assert_redacted(str(err))


def test_http_request_error_redacts_target(monkeypatch):
    """A URLError from urllib must not carry the token in the raised error."""
    from nomorals.core.http import HttpClient

    import urllib.error

    def _boom(*_args, **_kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    # Bypass the SSRF DNS check (sandbox DNS resolves the host to a
    # private address) so the request actually reaches urlopen.
    monkeypatch.setattr(
        "nomorals.core.http.HttpClient._check_ssrf", lambda self, target: None
    )
    client = HttpClient()
    with pytest.raises(Exception) as exc_info:
        client.request("POST", FAKE_URL)
    _assert_redacted(str(exc_info.value))


def test_gateway_preflight_errors_are_redacted():
    from nomorals.social.chat.gateway import ChatGateway

    class _FailingAdapter:
        def preflight(self):
            raise RuntimeError(f"request to {FAKE_URL} failed: [SSL: boom]")

        def start(self, _handler):
            return True

        def stop(self):
            pass

    gateway = ChatGateway({"telegram-bot": _FailingAdapter()})
    started = gateway.start(lambda _msg: None)
    assert started == []
    reason = gateway.preflight_errors.get("telegram-bot", "")
    assert reason, "preflight failure must be recorded"
    _assert_redacted(reason)


def _load_run_chat_bot():
    path = REPO_ROOT / "scripts" / "run_chat_bot.py"
    spec = importlib.util.spec_from_file_location("run_chat_bot_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bot_start_message_includes_real_reason():
    module = _load_run_chat_bot()

    class _Gateway:
        preflight_errors = {
            "telegram-bot": "request to https://api.telegram.org/bot123456789:***"
            "/getMe failed: [SSL: CERTIFICATE_VERIFY_FAILED]"
        }

    msg = module._telegram_bot_start_message(["local"], _Gateway())
    assert msg is not None
    assert "CERTIFICATE_VERIFY_FAILED" in msg
    assert "NM_CHAT_TELEGRAM_BOT_TOKEN" not in msg


def test_bot_start_message_falls_back_when_no_reason():
    module = _load_run_chat_bot()

    class _Gateway:
        preflight_errors = {}

    msg = module._telegram_bot_start_message(["local"], _Gateway())
    assert msg is not None
    assert "NM_CHAT_TELEGRAM_BOT_TOKEN" in msg
    assert module._telegram_bot_start_message(["telegram-bot"], _Gateway()) is None


def test_template_client_redacts_url_in_errors(monkeypatch):
    path = REPO_ROOT / "nomorals" / "builders" / "templates" / "telegram_bot" / "run.py"
    spec = importlib.util.spec_from_file_location("telegram_bot_template_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._redact_url(FAKE_URL) == (
        f"https://api.telegram.org/bot{FAKE_BOT_ID}:***/getMe"
    )

    import urllib.error

    def _boom(*_args, **_kwargs):
        raise urllib.error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", _boom)
    client = module.TelegramClient(FAKE_TOKEN)
    with pytest.raises(module.TelegramError) as exc_info:
        client._http_post(FAKE_URL, {})
    _assert_redacted(str(exc_info.value))
