"""Channel posture (group build P0): a channel is a broadcast surface,
not a conversation. Nothing posted in a channel gets a reply — no
command output, no catalog, no game redirect, no persona chat — from
anyone, the owner included. Channel control lives in DM and the
linked discussion group; channel posts don't reach the brain.

Before the fix, a channel /help on the public bot dumped the whole
public catalog into the channel and a channel /game posted a redirect
notice; both are channel posts by Devon, which the posture forbids.
"""
from __future__ import annotations

import tempfile
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef


@pytest.fixture()
def rig():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="nm-channel-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError("the brain must not be called for a "
                                 "channel post")

    context.router = _Router()

    sends: list[str] = []

    class _Gateway:
        adapters: dict = {}
        dry_run = False

        def status(self):
            return {"local": {"running_in_session": True}}

        def set_rate_limit(self, *a, **k):
            return None

        def typing(self, *a, **k):
            return True

        def send(self, platform, chat, text, **kw):
            sends.append(text)
            return SimpleNamespace(ok=True)

    runtime = PartnerRuntime(context, gateway=_Gateway())
    return runtime, sends


def _msg(text, platform="telegram-bot", owner=False):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id="-100999",
                     kind=ChatKind.CHANNEL),
        incoming=True, text=text,
        sender="channel" if not owner else "owner",
        sender_id="42" if not owner else "1",
        meta={"is_owner": True} if owner else {})


def test_channel_help_gets_no_catalog(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/help"))
    assert sends == []


def test_channel_plain_post_reaches_nobody(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("big announcement everyone"))
    assert sends == []


def test_channel_game_gets_no_redirect_post(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/game chess"))
    assert sends == []


def test_channel_is_silent_even_for_the_owner(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/status", owner=True))
    assert sends == []
