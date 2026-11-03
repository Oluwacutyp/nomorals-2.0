"""/ping: the message's own journey, stage by stage (owner spec).

Registered as a public control command, stamped at real seams
(platform ts = arrival, on_message = receipt, _process_inner =
processing start) and answered from those stamps only.
"""
from __future__ import annotations

import tempfile
import time

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.control import (
    CONTROL_COMMANDS, PUBLIC_COMMANDS, caller_may_run, parse_control)


def test_ping_registered_and_public():
    assert CONTROL_COMMANDS["ping"] == (0, 0)
    assert "ping" in PUBLIC_COMMANDS
    cmd = parse_control("/ping")
    assert cmd is not None and cmd.kind == "ping"
    # anyone may run it — it is a liveness probe, in any chat
    assert caller_may_run("ping", is_owner=False)
    # and it has catalog presence (help groups + details)
    from nomorals.social.chat.control import COMMAND_DETAILS, _HELP_GROUPS
    assert "ping" in COMMAND_DETAILS
    assert any("ping" in kinds for _name, kinds in _HELP_GROUPS)


@pytest.fixture()
def runtime():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="nm-ping-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError("test must not call the model")

    context.router = _Router()

    class _Gateway:
        adapters: dict = {}

        def status(self):
            return {"local": {"running_in_session": True}}

        def set_rate_limit(self, *a, **k):
            return None

        def typing(self, *a, **k):
            return None

    return PartnerRuntime(context, gateway=_Gateway())


def _msg(text, chat_key, ts=None):
    platform, _, chat_id = chat_key.partition(":")
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM),
        incoming=True, text=text, sender="owner", sender_id="42",
        ts=ts if ts is not None else time.time(), meta={})


def test_ping_reports_every_stage_from_real_stamps(runtime):
    key = "telegram:999"
    msg = _msg("/ping", key, ts=time.time() - 2.0)
    runtime.on_message(msg)  # stamps receipt
    assert "received_at" in msg.meta
    # the processing stamp _process_inner writes at pickup
    msg.meta["process_at"] = msg.meta["received_at"] + 0.25
    out = runtime.handle_control("/ping", key, message=msg)
    assert "pong" in out
    assert "arrival:" in out
    assert "receipt:" in out and "in transit" in out
    assert "inbound:" in out and "in the queue" in out
    assert "processing:" in out and "reply composed in" in out
    assert "sent: this reply" in out and "end to end" in out
    assert "condition:" in out and "local" in out and "db ok" in out


def test_ping_without_stamps_says_so_and_still_answers(runtime):
    key = "telegram:999"
    msg = _msg("/ping", key)
    out = runtime.handle_control("/ping", key, message=msg)
    assert "pong" in out
    assert "receipt: not stamped" in out
    assert "condition:" in out


def test_ping_without_message_object(runtime):
    out = runtime.handle_control("/ping", "telegram:999")
    assert "pong" in out
    assert "arrival: not stamped" in out
