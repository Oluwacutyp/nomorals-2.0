"""Claimed once (group build P0): a recognized command in a group is
claimed by the command path exactly once — it never also, or instead,
gets a conversational/persona reply.

Before the fix, a non-operator in a group on any non-telegram-bot
surface (userbot groups, WhatsApp, Discord) sending a recognized
public command fell through every claim branch to the brain. The
public-bot surface claimed the same command correctly; groups on the
other surfaces now claim it the same way. Owner-only kinds get one
honest redirect. Operators keep their own branch; DMs are untouched.
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

    tmp = tempfile.mkdtemp(prefix="nm-claim-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError("the brain must not be called: "
                                 "the command path owns this message")

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


def _msg(text, platform, kind, owner=False):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id="777", kind=kind),
        incoming=True, text=text, sender="member" if not owner else "owner",
        sender_id="42" if not owner else "1",
        meta={"is_owner": True} if owner else {})


def test_group_public_command_runs_once_no_persona(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/ping", "telegram", ChatKind.GROUP))
    assert len(sends) == 1
    assert "pong" in sends[0]


def test_group_public_command_claimed_on_whatsapp_too(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/ping", "whatsapp", ChatKind.GROUP))
    assert len(sends) == 1
    assert "pong" in sends[0]


def test_group_owner_only_command_gets_one_redirect(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/status", "telegram", ChatKind.GROUP))
    assert len(sends) == 1
    assert "owner-only" in sends[0]
    # the real /status output never ran, and no persona line followed
    assert "mood:" not in sends[0]


def test_group_operator_command_still_runs_via_operator_branch(rig):
    runtime, sends = rig
    runtime._process_inner(
        _msg("/ping", "telegram", ChatKind.GROUP, owner=True))
    assert len(sends) == 1
    assert "pong" in sends[0]
    assert "owner-only" not in sends[0]


def test_dm_stranger_command_is_not_group_claimed(rig):
    runtime, sends = rig
    runtime._process_inner(_msg("/status", "telegram", ChatKind.DM))
    # the group claim must not leak into DMs: neither its redirect nor
    # the command output may appear (DM behavior is the standing one).
    assert all("owner-only" not in s for s in sends)
    assert all("mood:" not in s for s in sends)
