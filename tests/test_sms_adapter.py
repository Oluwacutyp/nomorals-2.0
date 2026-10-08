"""Offline tests for the SMS fallback surface (build-map #19)."""

from __future__ import annotations

import pytest

from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.sms import (
    SMS_BLOCKED_REPLY,
    SMS_COMMANDS,
    SMS_MEDIA_REPLY,
    SMSAdapter,
    sms_allows,
    sms_enabled,
    split_sms,
)


class FakeTwilio:
    """Duck-typed stand-in for TwilioConnector.send_sms."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send_sms(self, to: str, body: str, *, from_number: str = "",
                 confirmed: bool = False, **kwargs) -> dict:
        self.calls.append({"to": to, "body": body,
                           "from_number": from_number, "confirmed": confirmed})
        return {"sid": f"SM{len(self.calls):06d}"}


class FakeChatSettings:
    def __init__(self, **kw):
        self.sms_enabled = kw.get("sms_enabled", False)
        self.sms_from_number = kw.get("sms_from_number", "")


class FakeSettings:
    def __init__(self, **kw):
        self.chat = FakeChatSettings(**kw)


# ── webhook parsing ──────────────────────────────────────────────────────

def test_webhook_valid_payload():
    adapter = SMSAdapter(FakeTwilio(), from_number="+10000000001")
    msg = adapter.handle_webhook({
        "From": "+15551234567",
        "Body": "what's my status",
        "MessageSid": "SMabc123",
        "NumMedia": "0",
    })
    assert msg is not None
    assert msg.text == "what's my status"
    assert msg.chat.platform == "sms"
    assert msg.chat.chat_id == "+15551234567"
    assert msg.chat.kind == ChatKind.DM
    assert msg.chat.key == "sms:+15551234567"
    assert msg.meta["sms_has_media"] is False
    assert msg.message_id == "SMabc123"


def test_webhook_missing_from_dropped():
    adapter = SMSAdapter(FakeTwilio())
    assert adapter.handle_webhook({"Body": "hi"}) is None
    assert adapter.handle_webhook({}) is None


def test_webhook_media_flagged_not_crashed():
    adapter = SMSAdapter(FakeTwilio())
    msg = adapter.handle_webhook({
        "From": "+15551234567", "Body": "", "NumMedia": "2",
    })
    assert msg is not None
    assert msg.meta["sms_has_media"] is True


def test_webhook_malformed_never_raises():
    adapter = SMSAdapter(FakeTwilio())
    assert adapter.handle_webhook({"From": None, "Body": None}) is None
    assert adapter.handle_webhook({"From": 12345, "Body": "hi"}) is not None


# ── segmentation ─────────────────────────────────────────────────────────

def test_split_exact_160_unnumbered():
    text = "x" * 160
    segs = split_sms(text)
    assert segs == [text]


def test_split_161_two_numbered_segments():
    segs = split_sms("w " * 80 + "x")  # 161 chars
    assert len(segs) == 2
    assert segs[0].startswith("(1/2) ")
    assert segs[1].startswith("(2/2) ")
    assert all(len(s) <= 160 for s in segs)
    # Reassembly recovers the original text (modulo whitespace collapse).
    joined = " ".join(s.split(" ", 1)[1] for s in segs)
    assert joined.replace(" ", "") == ("w " * 80 + "x").replace(" ", "")


def test_split_400_chars():
    segs = split_sms("word " * 80)  # 400 chars
    assert len(segs) == 3
    assert segs[0].startswith("(1/3) ")
    assert all(len(s) <= 160 for s in segs)


def test_split_empty():
    assert split_sms("") == [""]


# ── send ─────────────────────────────────────────────────────────────────

def test_send_without_twilio_fails_closed():
    adapter = SMSAdapter(None)
    chat = ChatRef.parse("sms:+15551234567", kind=ChatKind.DM)
    result = adapter.send(chat, "hello")
    assert result.ok is False
    assert "twilio" in result.error.lower()


def test_send_splits_and_confirms():
    twilio = FakeTwilio()
    adapter = SMSAdapter(twilio, from_number="+10000000001")
    chat = ChatRef.parse("sms:+15551234567", kind=ChatKind.DM)
    result = adapter.send(chat, "y " * 100)  # 200 chars → 2 segments
    assert result.ok is True
    assert len(twilio.calls) == 2
    assert all(c["to"] == "+15551234567" for c in twilio.calls)
    assert all(c["confirmed"] is True for c in twilio.calls)
    assert all(c["from_number"] == "+10000000001" for c in twilio.calls)
    assert twilio.calls[0]["body"].startswith("(1/2) ")


def test_send_twilio_error_never_raises():
    class Boom:
        def send_sms(self, *a, **k):
            raise RuntimeError("twilio down")

    adapter = SMSAdapter(Boom())
    chat = ChatRef.parse("sms:+15551234567", kind=ChatKind.DM)
    result = adapter.send(chat, "hello")
    assert result.ok is False
    assert "twilio down" in result.error


# ── command gating ───────────────────────────────────────────────────────

def test_sms_allows_lightweight_commands():
    for kind in ("status", "help", "remember", "recall", "spending",
                 "finance", "platforms", "list"):
        assert sms_allows(kind), kind


def test_sms_blocks_heavy_commands():
    for kind in ("poker", "ttt", "exec", "shell", "media", "game",
                 "arena", "voice", "say", "approve"):
        assert not sms_allows(kind), kind


def test_blocked_reply_points_at_real_commands():
    assert "/status" in SMS_BLOCKED_REPLY
    assert "/remember" in SMS_BLOCKED_REPLY
    # Every command named in the reply must actually be allowed.
    for token in SMS_BLOCKED_REPLY.split():
        if token.startswith("/"):
            assert sms_allows(token[1:].rstrip(",.")), token


def test_media_reply_is_honest():
    assert "SMS" in SMS_MEDIA_REPLY
    assert "pictures" in SMS_MEDIA_REPLY


# ── runtime dispatch gate ────────────────────────────────────────────────

class TestSmsCommandGate:
    """The handle_control gate: blocked kinds get the honest reply on SMS."""

    @pytest.fixture()
    def runtime(self, tmp_path):
        import tempfile
        from nomorals.agents.context import build_context
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.core.config import load_settings

        tmp = tempfile.TemporaryDirectory(prefix="nm-sms-")
        settings = load_settings(overrides={
            "home": tmp.name,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        context = build_context(settings, with_executor=False, with_tools=False)

        class _StubGateway:
            def send(self, *a, **k):
                from nomorals.social.chat.base import SendResult
                return SendResult(ok=True, platform="stub")

        rt = PartnerRuntime(context, gateway=_StubGateway())
        yield rt
        try:
            rt.stop()
        except Exception:
            pass
        context.close()
        tmp.cleanup()

    def test_blocked_command_gets_sms_reply(self, runtime):
        out = runtime.handle_control("/poker", "sms:+15551234567")
        assert out == SMS_BLOCKED_REPLY

    def test_allowed_command_not_gated(self, runtime):
        out = runtime.handle_control("/help", "sms:+15551234567")
        assert out != SMS_BLOCKED_REPLY
        assert out.strip()

    def test_gate_only_applies_to_sms(self, runtime):
        # Same command on another platform is not SMS-gated.
        out = runtime.handle_control("/poker", "local:console")
        assert out != SMS_BLOCKED_REPLY


# ── opt-in ───────────────────────────────────────────────────────────────

def test_sms_enabled_opt_in_logic():
    assert sms_enabled(FakeSettings()) is False  # off by default
    assert sms_enabled(FakeSettings(sms_enabled=True)) is False  # no number
    assert sms_enabled(FakeSettings(sms_from_number="+1000")) is False  # no opt-in
    assert sms_enabled(FakeSettings(sms_enabled=True,
                                    sms_from_number="+10000000001")) is True


def test_adapter_metadata():
    adapter = SMSAdapter(FakeTwilio())
    assert adapter.name == "sms"
    assert adapter.supported_kinds == (ChatKind.DM,)
    health = adapter.health()
    assert health["twilio_configured"] is True
    assert SMSAdapter(None).health()["twilio_configured"] is False
