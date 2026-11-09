"""Parity bundle #18 — Telegram bot self-profile + payments.

The census's last bundle, two ABSENT rows on the Telegram
BOT surface: (1) set the bot's own name / description /
short description — BotFather territory, no
getMyName/setMyName family anywhere in the adapter;
(2) Telegram Stars / invoice payments — no sendInvoice
family, and the payment handshake itself (pre-checkout)
was never even requested in allowed_updates, so no
payment could ever complete.

Everything here is offline: a fake Bot API session records
the exact wire calls, the same house harness as the other
bot-adapter suites.
"""

from __future__ import annotations

from nomorals.social.chat.base import (
    ChatKind,
    ChatMessage,
    ChatRef,
    bot_profile_shape_error,
    invoice_shape_error,
)
from nomorals.social.chat.chatops import (
    CHATOPS_COMMANDS,
    invoice_payload_for,
    parse_invoice_args,
)
from nomorals.social.chat.telegram import TelegramBotAdapter

# ── fake Bot API session ───────────────────────────────

class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class _FakeBotSession:
    """Records Bot API calls by method name; serves the
    configured result per method, or the configured error."""

    def __init__(self, results=None, errors=None):
        self.calls: list[tuple[str, dict]] = []
        self.results = dict(results or {})
        self.errors = dict(errors or {})

    def post_json(self, url, params, timeout=None):
        method = str(url).rsplit("/", 1)[-1]
        self.calls.append((method, dict(params)))
        if method in self.errors:
            return _FakeResp({"ok": False, "error_code": 400,
                              "description": self.errors[method]})
        return _FakeResp({"ok": True,
                          "result": self.results.get(method, True)})

    def post_multipart(self, url, fields, files=None, timeout=None):
        method = str(url).rsplit("/", 1)[-1]
        self.calls.append((method, dict(fields)))
        return _FakeResp({"ok": True, "result": {}})

    def get(self, url, timeout=None):
        self.calls.append((str(url), {}))
        return _FakeResp({"ok": True, "result": {}})

    def methods(self):
        return [m for m, _ in self.calls]


def _bot(tmp_path, session=None, **kw):
    kw.setdefault("token", "123456:TEST")
    kw.setdefault("media_dir", str(tmp_path / "media"))
    kw.setdefault("lock_path", str(tmp_path / "bot.lock"))
    kw["session"] = session or _FakeBotSession()
    return TelegramBotAdapter(**kw)


def _chat(chat_id="-1001234567890", platform="telegram-bot"):
    return ChatRef(platform=platform, chat_id=chat_id,
                   kind=ChatKind.GROUP, peer=chat_id)


_PROFILE_RESULTS = {
    "getMe": {"id": 777, "is_bot": True, "first_name": "Devon",
              "username": "Nomorals_bot"},
    "getMyName": {"name": "Devon"},
    "getMyDescription": {"description": "Your personal AI."},
    "getMyShortDescription": {"short_description": "On duty."},
}


# ── catalog pins ───────────────────────────────────────

def test_catalog_registrations():
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    assert {"botprofile", "invoice"} <= CHATOPS_COMMANDS
    for kind in ("botprofile", "invoice"):
        assert kind in CONTROL_COMMANDS
        assert kind in COMMAND_DETAILS
        assert kind in LIST_ONELINERS
        # owner-only bot configuration/payment acts — the
        # public bot surface must never grow them
        assert kind not in PUBLIC_COMMANDS
    grouped = [k for _title, kinds in LIST_GROUPS for k in kinds]
    assert "botprofile" in grouped and "invoice" in grouped


# ── shape rules (before traffic) ───────────────────────

def test_bot_profile_shape_rules():
    assert bot_profile_shape_error("name", "Devon") == ""
    assert bot_profile_shape_error("name", "") == ""  # clear
    assert "at most 64" in bot_profile_shape_error(
        "name", "x" * 65)
    assert "at most 512" in bot_profile_shape_error(
        "description", "x" * 513)
    assert "at most 120" in bot_profile_shape_error(
        "short_description", "x" * 121)
    assert "not a bot profile field" in \
        bot_profile_shape_error("photo", "x")


def test_invoice_shape_rules():
    assert invoice_shape_error("Pack", "One month", 50) == ""
    assert "title" in invoice_shape_error("", "One month", 50)
    assert "at most 32" in invoice_shape_error(
        "x" * 33, "One month", 50)
    assert "description" in invoice_shape_error("Pack", "", 50)
    assert "at most 255" in invoice_shape_error(
        "Pack", "x" * 256, 50)
    assert "at least 1" in invoice_shape_error("Pack", "x", 0)
    assert "whole number" in invoice_shape_error(
        "Pack", "x", 2.5)
    assert "whole number" in invoice_shape_error(
        "Pack", "x", True)


def test_parse_invoice_args():
    out = parse_invoice_args("Signal pack | One month | 50")
    assert out == {"ok": True, "title": "Signal pack",
                   "description": "One month", "amount": 50}
    out = parse_invoice_args("only | two")
    assert out["ok"] is False and "usage:" in out["error"]
    out = parse_invoice_args("T | D | fifty")
    assert out["ok"] is False
    assert "isn't a whole number of Stars" in out["error"]
    out = parse_invoice_args("T | D | 0")
    assert out["ok"] is False
    assert "at least 1" in out["error"]
    assert invoice_payload_for("Signal Pack — Deluxe!") \
        == "devon:signal-pack-deluxe"
    assert invoice_payload_for("") == "devon-invoice"
    assert invoice_payload_for("⭐⭐") == "devon-invoice"


# ── bot self-profile: adapter wire shapes ──────────────

def test_get_bot_profile_reads_all_fields(tmp_path):
    session = _FakeBotSession(results=_PROFILE_RESULTS)
    bot = _bot(tmp_path, session)
    verdict = bot.get_bot_profile()
    assert verdict["ok"] is True
    assert verdict["username"] == "Nomorals_bot"
    assert verdict["name"] == "Devon"
    assert verdict["description"] == "Your personal AI."
    assert verdict["short_description"] == "On duty."
    assert session.methods() == [
        "getMe", "getMyName", "getMyDescription",
        "getMyShortDescription"]


def test_get_bot_profile_field_error_is_named(tmp_path):
    session = _FakeBotSession(
        results=_PROFILE_RESULTS,
        errors={"getMyDescription": "temporarily unavailable"})
    bot = _bot(tmp_path, session)
    verdict = bot.get_bot_profile()
    assert verdict["ok"] is True
    assert verdict["name"] == "Devon"
    assert verdict["description"] == ""
    assert verdict["description_error"].endswith(
        "temporarily unavailable")


def test_get_bot_profile_getme_failure_is_the_verdict(tmp_path):
    session = _FakeBotSession(errors={"getMe": "unauthorized"})
    bot = _bot(tmp_path, session)
    verdict = bot.get_bot_profile()
    assert verdict["ok"] is False
    assert verdict["error"].endswith("unauthorized")
    assert session.methods() == ["getMe"]


def test_set_bot_fields_wire_shapes(tmp_path):
    session = _FakeBotSession()
    bot = _bot(tmp_path, session)
    assert bot.set_bot_name("Devon Two")["ok"] is True
    assert bot.set_bot_description("Long text")["ok"] is True
    assert bot.set_bot_short_description("Short")["ok"] is True
    assert session.calls == [
        ("setMyName", {"name": "Devon Two"}),
        ("setMyDescription", {"description": "Long text"}),
        ("setMyShortDescription",
         {"short_description": "Short"})]


def test_set_bot_field_shape_refuses_before_traffic(tmp_path):
    session = _FakeBotSession()
    bot = _bot(tmp_path, session)
    verdict = bot.set_bot_name("x" * 65)
    assert verdict["ok"] is False
    assert "at most 64" in verdict["error"]
    assert session.calls == []


def test_set_bot_field_api_refusal_carries_telegram(tmp_path):
    session = _FakeBotSession(
        errors={"setMyName": "Too many requests"})
    bot = _bot(tmp_path, session)
    verdict = bot.set_bot_name("Devon")
    assert verdict["ok"] is False
    assert "Too many requests" in verdict["error"]


# ── invoices: adapter wire shapes ──────────────────────

def test_send_invoice_wire_shape(tmp_path):
    session = _FakeBotSession(
        results={"sendInvoice": {"message_id": 777}})
    bot = _bot(tmp_path, session)
    verdict = bot.send_invoice(
        _chat(), "Signal pack", "One month of signals", 50,
        payload="devon:signal-pack")
    assert verdict == {"ok": True, "message_id": "777",
                       "amount": 50, "currency": "XTR"}
    assert session.calls == [(
        "sendInvoice",
        {"chat_id": -1001234567890, "title": "Signal pack",
         "description": "One month of signals",
         "payload": "devon:signal-pack", "currency": "XTR",
         "prices": [{"label": "Signal pack", "amount": 50}]})]


def test_send_invoice_shape_refuses_before_traffic(tmp_path):
    session = _FakeBotSession()
    bot = _bot(tmp_path, session)
    for bad in (0, -5, 2.5):
        verdict = bot.send_invoice(_chat(), "T", "D", bad)
        assert verdict["ok"] is False
    assert session.calls == []


def test_create_invoice_link(tmp_path):
    session = _FakeBotSession(
        results={"createInvoiceLink": "https://t.me/invoice/x"})
    bot = _bot(tmp_path, session)
    verdict = bot.create_invoice_link("Pack", "One month", 25)
    assert verdict == {"ok": True,
                       "link": "https://t.me/invoice/x",
                       "amount": 25, "currency": "XTR"}
    method, params = session.calls[0]
    assert method == "createInvoiceLink"
    assert params["currency"] == "XTR"
    assert params["payload"] == "devon-invoice"
    assert params["prices"] == [{"label": "Pack", "amount": 25}]


# ── the payment handshake ──────────────────────────────

def test_pre_checkout_is_answered(tmp_path):
    session = _FakeBotSession()
    bot = _bot(tmp_path, session)
    bot._handle_pre_checkout_query({
        "id": "pcq-1", "from": {"id": 42},
        "currency": "XTR", "total_amount": 50,
        "invoice_payload": "devon:signal-pack"})
    assert session.calls == [(
        "answerPreCheckoutQuery",
        {"pre_checkout_query_id": "pcq-1", "ok": True})]


def test_pre_checkout_without_id_is_dropped_quietly(tmp_path):
    session = _FakeBotSession()
    bot = _bot(tmp_path, session)
    bot._handle_pre_checkout_query({"currency": "XTR"})
    assert session.calls == []


def test_pre_checkout_answer_failure_never_raises(tmp_path):
    session = _FakeBotSession(
        errors={"answerPreCheckoutQuery": "query too old"})
    bot = _bot(tmp_path, session)
    bot._handle_pre_checkout_query({"id": "pcq-2"})  # no raise


def test_dispatch_routes_pre_checkout_not_conversation(tmp_path):
    session = _FakeBotSession()
    bot = _bot(tmp_path, session)
    delivered = []
    bot._dispatch_update(
        {"update_id": 5,
         "pre_checkout_query": {"id": "pcq-9", "currency": "XTR",
                                "total_amount": 50,
                                "invoice_payload": "devon:x"}},
        delivered.append)
    assert delivered == []
    assert session.methods() == ["answerPreCheckoutQuery"]


def test_pre_checkout_is_in_allowed_updates():
    import inspect

    from nomorals.social.chat import telegram as tg

    source = inspect.getsource(tg.TelegramBotAdapter._run_locked)
    assert '"pre_checkout_query"' in source


def test_successful_payment_becomes_a_service_event(tmp_path):
    bot = _bot(tmp_path)
    update = {
        "update_id": 6,
        "message": {
            "message_id": 99, "date": 1_700_000_000,
            "from": {"id": 42, "is_bot": False,
                     "first_name": "Ada"},
            "chat": {"id": 42, "type": "private"},
            "successful_payment": {
                "currency": "XTR", "total_amount": 50,
                "invoice_payload": "devon:signal-pack",
                "telegram_payment_charge_id": "tg-charge-1",
                "provider_payment_charge_id": "prov-1"}}}
    msg = bot._convert(update)
    assert msg is not None
    assert msg.text == ""
    assert msg.meta["service"] == "successful_payment"
    assert msg.meta["payment"] == {
        "currency": "XTR", "total_amount": 50,
        "invoice_payload": "devon:signal-pack",
        "telegram_payment_charge_id": "tg-charge-1",
        "provider_payment_charge_id": "prov-1"}


# ── base-adapter honesty on the other surfaces ─────────

def test_userbot_reports_the_honest_absence(tmp_path):
    from nomorals.social.chat.telegram import TelegramAdapter

    ad = TelegramAdapter(
        api_id=12345, api_hash="testhash",
        session_path=str(tmp_path / "tg.session"),
        media_dir=str(tmp_path / "media"))
    for verdict in (ad.get_bot_profile(),
                    ad.set_bot_name("X"),
                    ad.send_invoice(_chat(platform="telegram"),
                                    "T", "D", 5),
                    ad.create_invoice_link("T", "D", 5)):
        assert verdict["ok"] is False
        assert "not supported" in verdict["error"]


# ── chatops: the gate + the handlers ───────────────────

class _StubBotAdapter:
    """The bot adapter's #18 surface, recording calls."""

    def __init__(self):
        self.calls = []

    def get_bot_profile(self):
        self.calls.append(("get_bot_profile",))
        return {"ok": True, "username": "Nomorals_bot",
                "name": "Devon", "description": "Long.",
                "short_description": "Short."}

    def set_bot_name(self, value):
        self.calls.append(("set_bot_name", value))
        return {"ok": True}

    def set_bot_description(self, value):
        self.calls.append(("set_bot_description", value))
        return {"ok": True}

    def set_bot_short_description(self, value):
        self.calls.append(("set_bot_short_description", value))
        return {"ok": True}

    def send_invoice(self, chat, title, description, amount,
                     payload=""):
        self.calls.append(("send_invoice", title, amount,
                           payload))
        return {"ok": True, "message_id": "5"}

    def create_invoice_link(self, title, description, amount,
                            payload=""):
        self.calls.append(("create_invoice_link", title,
                           amount, payload))
        return {"ok": True, "link": "https://t.me/invoice/x"}


def _mixin(adapter=None, consulted=None):
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )

    mixin = RuntimeChatOpsMixin()

    def _for(platform):
        if consulted is not None:
            consulted.append(platform)
        return adapter

    mixin._adapter_for = _for  # type: ignore[method-assign]
    return mixin


def _msg(platform, chat_id="42"):
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=ChatKind.DM, peer=chat_id)
    return ChatMessage(chat=chat, incoming=True, text="x",
                       sender="Owner", sender_id="1")


def test_chatop_personal_surfaces_get_the_truth():
    consulted = []
    mixin = _mixin(adapter=_StubBotAdapter(),
                   consulted=consulted)
    out = mixin._chatops_botprofile("", _msg("whatsapp"))
    assert "/myprofile" in out
    out = mixin._chatops_invoice("T | D | 5", _msg("telegram"))
    assert "bot" in out and "Stars" in out
    assert consulted == []  # answered before any adapter work


def test_chatop_botprofile_read_and_set():
    stub = _StubBotAdapter()
    mixin = _mixin(adapter=stub)
    out = mixin._chatops_botprofile("", _msg("telegram-bot"))
    assert "@Nomorals_bot" in out
    assert "• name: Devon" in out
    assert "• description: Long." in out
    assert "• short: Short." in out
    out = mixin._chatops_botprofile(
        "name Devon Two", _msg("telegram-bot"))
    assert "done" in out and "Devon Two" in out
    out = mixin._chatops_botprofile(
        "description clear", _msg("telegram-personal"))
    assert "cleared" in out
    out = mixin._chatops_botprofile(
        "name clear", _msg("telegram-bot"))
    assert "back to its default" in out
    assert stub.calls == [
        ("get_bot_profile",),
        ("set_bot_name", "Devon Two"),
        ("set_bot_description", ""),
        ("set_bot_name", "")]


def test_chatop_botprofile_no_live_bot():
    mixin = _mixin(adapter=None)
    out = mixin._chatops_botprofile("", _msg("telegram-bot"))
    assert "no live bot surface" in out


def test_chatop_invoice_send_and_link():
    stub = _StubBotAdapter()
    mixin = _mixin(adapter=stub)
    out = mixin._chatops_invoice(
        "Signal pack | One month | 50", _msg("telegram-bot"))
    assert "invoice posted" in out and "50 XTR" in out
    out = mixin._chatops_invoice(
        "link Signal pack | One month | 50",
        _msg("telegram-bot"))
    assert "https://t.me/invoice/x" in out
    assert stub.calls == [
        ("send_invoice", "Signal pack", 50,
         "devon:signal-pack"),
        ("create_invoice_link", "Signal pack", 50,
         "devon:signal-pack")]


def test_chatop_invoice_bad_args_never_reach_the_adapter():
    stub = _StubBotAdapter()
    mixin = _mixin(adapter=stub)
    out = mixin._chatops_invoice("nonsense",
                                 _msg("telegram-bot"))
    assert "usage:" in out
    assert stub.calls == []
