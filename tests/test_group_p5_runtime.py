"""The §16 P5 layer at the runtime seam: WhatsApp dispatch
truth, menu absence, and ephemeral delivery end-to-end.

* Typed on WhatsApp, every P5 kind gets the one-line platform
  truth at dispatch — it never reaches a handler, never
  attempts an execution.
* WhatsApp's own menu text carries none of the P5 commands;
  the global catalog documents them all.
* /balance in a telegram-bot group with the p5 ephemeral
  switch on is delivered to the asker only: the adapter's
  ephemeral verb first; on its refusal, the same text by DM
  and nothing in the group; with the switch off, the reply
  posts in the room as it always has.
"""
from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

BOTKEY = "telegram-bot:-1002"


class _BotFake:
    """The telegram-bot adapter the runtime sees: just the
    ephemeral verb for this seam."""

    name = "telegram-bot"

    def __init__(self, *, ephemeral_ok=True):
        self.ephemeral_calls: list = []
        self.ephemeral_ok = ephemeral_ok

    def send_ephemeral(self, chat, user_id, text, **kw):
        self.ephemeral_calls.append((str(user_id), text))
        return {"ok": self.ephemeral_ok,
                "error": "" if self.ephemeral_ok else "too late"}


@pytest.fixture()
def rig_factory():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    made = []

    def make(bot):
        tmp = tempfile.mkdtemp(prefix="nm-p5rt-", dir=_TEST_TMPDIR)
        settings = load_settings(overrides={
            "home": tmp,
            "partner.platforms": "local",
            "chat.local_enabled": "true",
        })
        context = build_context(settings, with_executor=False,
                                with_tools=False)

        class _Router:
            def chat(self, *a, **k):
                raise AssertionError(
                    "the brain must not be called")

        context.router = _Router()
        sends: list = []

        class _Gateway:
            adapters = {"telegram-bot": bot}
            dry_run = False

            def status(self):
                return {"local": {"running_in_session": True}}

            def set_rate_limit(self, *a, **k):
                return None

            def typing(self, *a, **k):
                return True

            def send(self, platform, chat, text, **kw):
                sends.append((platform, chat, text))
                return SimpleNamespace(ok=True)

        runtime = PartnerRuntime(context, gateway=_Gateway())
        made.append((runtime, context))
        return runtime, sends

    yield make
    for runtime, context in made:
        try:
            runtime.stop()
        except Exception:  # noqa: BLE001 - teardown
            pass
        try:
            context.close()
        except Exception:  # noqa: BLE001 - teardown
            pass


def _msg(text, platform, kind, *, sender_id="42",
         chat_id="-1002"):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id,
                     kind=kind),
        incoming=True, text=text, sender="member",
        sender_id=sender_id, meta={})


# ── WhatsApp: dispatch truth + menu absence ──────────────────

def test_whatsapp_p5_kinds_get_platform_truth(rig_factory):
    runtime, _sends = rig_factory(_BotFake())
    for text in ("/tag @ada owl", "/bansender", "/unbansender",
                 "/reactwipe @ada", "/boosters"):
        out = runtime.handle_control(
            text, "whatsapp:777",
            message=_msg(text, "whatsapp", ChatKind.GROUP,
                         chat_id="777"))
        assert "Telegram tool" in out, text
        assert "WhatsApp" in out


def test_whatsapp_menu_has_no_p5_kinds():
    from nomorals.social.chat.control import (
        detailed_help, whatsapp_menu)

    menu = whatsapp_menu()
    for kind in ("tag", "bansender", "unbansender",
                 "reactwipe", "boosters"):
        assert f"/{kind}" not in menu
        # …while the global catalog documents each one.
        assert detailed_help(kind)


def test_telegram_typed_tag_reaches_the_handler(rig_factory):
    # On Telegram the same text dispatches (the handler's own
    # gates answer — here: the room isn't a claimed group the
    # sender admins, so a gate line comes back, NOT the
    # WhatsApp truth).
    runtime, _sends = rig_factory(_BotFake())
    out = runtime.handle_control(
        "/tag @ada owl", BOTKEY,
        message=_msg("/tag @ada owl", "telegram-bot",
                     ChatKind.GROUP))
    assert "Telegram tool" not in out


# ── ephemeral /balance end to end ────────────────────────────

def _armed(runtime, *, ephemeral=True):
    surface = runtime._group_surface()
    surface.set_designated(BOTKEY, True, by="owner")
    if ephemeral:
        surface.update_settings(BOTKEY, p5={"ephemeral": True})
    return surface


def test_balance_ephemeral_to_the_asker_only(rig_factory):
    bot = _BotFake()
    runtime, sends = rig_factory(bot)
    _armed(runtime)
    runtime._process_inner(_msg("/balance", "telegram-bot",
                                ChatKind.GROUP))
    # Delivered through the ephemeral verb, to the asker…
    assert len(bot.ephemeral_calls) == 1
    assert bot.ephemeral_calls[0][0] == "42"
    # …and NOTHING was posted to the group through the gateway.
    assert sends == []


def test_balance_ephemeral_refusal_falls_back_to_dm(rig_factory):
    bot = _BotFake(ephemeral_ok=False)
    runtime, sends = rig_factory(bot)
    _armed(runtime)
    runtime._process_inner(_msg("/balance", "telegram-bot",
                                ChatKind.GROUP))
    assert len(bot.ephemeral_calls) == 1  # attempted first
    # The same text went by DM; the group saw nothing.
    assert len(sends) == 1
    _platform, chat, _text = sends[0]
    assert chat.kind == ChatKind.DM
    assert chat.chat_id == "42"


def test_balance_public_when_ephemeral_off(rig_factory):
    bot = _BotFake()
    runtime, sends = rig_factory(bot)
    _armed(runtime, ephemeral=False)
    runtime._process_inner(_msg("/balance", "telegram-bot",
                                ChatKind.GROUP))
    assert bot.ephemeral_calls == []
    assert len(sends) == 1
    assert sends[0][1].kind == ChatKind.GROUP
