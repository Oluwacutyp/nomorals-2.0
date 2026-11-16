"""CROSS-PLATFORM SEARCH — one query over every surface
(parity census §6 superhuman candidate #1, 2026-11-07).

The census's search-depth bundle (#11) gave every surface
its own search verb — WhatsApp search_global fans out over
the bridge's chats, Telegram-personal search_global rides
messages.SearchGlobal — but /find and /findall each resolve
exactly ONE adapter from the message's own platform, so a
query never crosses surfaces.  /searchall fans out to both
personal surfaces at once and merges newest-first under
one cap, every hit naming its platform and its chat.

  * the renderer: format_search_results gains an opt-in
    show_platform flag — /find and /findall render exactly
    as before (no platform label), /searchall labels every
    row.  A hit whose chat carries no platform prints no
    label, never an id.
  * the handler: _chatops_searchall resolves each personal
    surface independently (paired account + live adapter);
    a surface with no paired account or no live adapter is
    NAMED in the reply and the other surface's hits still
    show.  Both down is a named failure, never silence.
    The origin surface does not limit the search — run
    from the bot DM it still searches the personal
    accounts, because the search targets are the personal
    accounts, not the room the command arrived in.  A read,
    not an act as you: the standing stop does not halt it,
    like /find and /findall.
  * registration: CONTROL_COMMANDS + COMMAND_DETAILS +
    LIST_GROUPS (exactly one slot) + LIST_ONELINERS, owner
    only — absent from PUBLIC_COMMANDS, like the find
    family.
"""
from __future__ import annotations

import itertools

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from tests.frozen_clock import MIDDAY_TS

OWNER = "2348012345678"
PN_POSTER = "2347099999999@s.whatsapp.net"
GROUP = "120363001234567890@g.us"
_MSG_SEQ = itertools.count(1)


# ── registry pins ────────────────────────────────────────

def test_catalog_registers_searchall():
    from nomorals.social.chat.chatops import CHATOPS_COMMANDS
    from nomorals.social.chat.control import (
        COMMAND_DETAILS,
        CONTROL_COMMANDS,
        LIST_GROUPS,
        LIST_ONELINERS,
        PUBLIC_COMMANDS,
    )

    assert "searchall" in CHATOPS_COMMANDS
    assert "searchall" in CONTROL_COMMANDS
    assert "searchall" in COMMAND_DETAILS
    assert "searchall" in LIST_ONELINERS
    slots = [names for _t, names in LIST_GROUPS
             if "searchall" in names]
    assert len(slots) == 1
    # owner-only account read — never a public command
    assert "searchall" not in PUBLIC_COMMANDS


# ── the renderer (pure) ──────────────────────────────────

def _hit(platform="whatsapp", chat_id="999888",
         title="Some Chat", sender="Ada",
         text="the invoice is ready", ts=MIDDAY_TS,
         incoming=True, mid="WAMID-SECRET-1"):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id,
                     kind=ChatKind.GROUP, title=title),
        incoming=incoming, text=text, sender=sender,
        message_id=mid, ts=ts)


def test_format_search_results_labels_platforms_opt_in():
    from nomorals.social.chat.chatops import format_search_results

    verdict = {"ok": True, "query": "invoice", "total": 2,
               "coverage": "a test reach",
               "messages": [
                   _hit(platform="whatsapp"),
                   _hit(platform="telegram", title="Signal Room",
                        sender="Bo", mid="TG-2")]}
    out = format_search_results(
        verdict, scope="every chat on every surface",
        show_chat=True, show_platform=True)
    assert "2 messages match “invoice”" in out
    assert "every chat on every surface" in out
    assert "WhatsApp · Some Chat ·" in out
    assert "Telegram · Signal Room ·" in out
    # ids never print: not the message id, not the chat id
    assert "WAMID-SECRET-1" not in out
    assert "999888" not in out


def test_format_search_results_no_platform_by_default():
    from nomorals.social.chat.chatops import format_search_results

    verdict = {"ok": True, "query": "invoice", "total": 1,
               "coverage": "a test reach",
               "messages": [_hit()]}
    out = format_search_results(verdict, scope="this chat")
    assert "WhatsApp" not in out
    assert "Telegram" not in out
    assert "Some Chat" not in out  # show_chat defaults off too
    out = format_search_results(verdict, scope="this chat",
                                show_chat=True)
    assert "Some Chat" in out
    assert "WhatsApp" not in out  # chat shown, platform not


def test_format_search_results_platformless_hit_no_label():
    from nomorals.social.chat.chatops import format_search_results

    hit = ChatMessage(
        chat=ChatRef(platform="", chat_id="1",
                     kind=ChatKind.DM, title="No Platform"),
        incoming=True, text="the invoice is ready",
        sender="Ada", message_id="M1", ts=MIDDAY_TS)
    out = format_search_results(
        {"ok": True, "query": "invoice", "total": 1,
         "messages": [hit]},
        show_chat=True, show_platform=True)
    assert "No Platform" in out
    assert "·  ·" not in out  # no empty label, no double dot


# ── handler pins: fake personal adapters ─────────────────

class _FakeAdapter:
    """One personal surface's search_global, canned."""

    def __init__(self, platform, hits):
        self.name = platform
        self.hits = hits
        self.queries: list[str] = []

    def search_global(self, query, limit=20):
        self.queries.append(query)
        return {"ok": True, "query": query,
                "messages": list(self.hits),
                "total": len(self.hits),
                "coverage": f"{self.name} test reach"}


class _RefusingAdapter:
    def __init__(self, error):
        self.error = error

    def search_global(self, query, limit=20):
        return {"ok": False, "error": self.error}


def _mixin(adapters, paired=("whatsapp", "telegram")):
    from nomorals.agents.partner.runtime_chatops import (
        RuntimeChatOpsMixin,
    )

    mixin = RuntimeChatOpsMixin()
    mixin._chatops_personal_account = (  # type: ignore[method-assign]
        lambda platform: (None, object())
        if platform in paired else (None, None))
    mixin._adapter_for = (  # type: ignore[method-assign]
        lambda platform: adapters.get(platform))
    return mixin


def _msg(platform="whatsapp", chat_id=PN_POSTER):
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=ChatKind.DM, title="", peer=chat_id)
    return ChatMessage(chat=chat, incoming=True, text="x",
                       sender="Owner", sender_id="1",
                       message_id=f"m-{next(_MSG_SEQ)}",
                       ts=MIDDAY_TS)


def test_searchall_merges_newest_first_with_provenance():
    wa_hit = _hit(platform="whatsapp", title="Poster Pal",
                  text="old invoice note",
                  ts=MIDDAY_TS - 100, mid="WA-1")
    tg_hit = _hit(platform="telegram", title="Signal Room",
                  text="new invoice totals",
                  ts=MIDDAY_TS, sender="Bo", mid="TG-1")
    mixin = _mixin({
        "whatsapp": _FakeAdapter("whatsapp", [wa_hit]),
        "telegram": _FakeAdapter("telegram", [tg_hit]),
    })
    out = mixin._chatops_searchall("invoice", _msg())
    assert "2 messages match “invoice”" in out
    assert "every chat on every surface" in out
    # newest first across platforms: the Telegram hit leads
    assert out.index("new invoice totals") < out.index(
        "old invoice note")
    # every row names its platform and its chat
    assert "Telegram · Signal Room ·" in out
    assert "WhatsApp · Poster Pal ·" in out
    assert "WhatsApp: whatsapp test reach" in out
    assert "Telegram: telegram test reach" in out


def test_searchall_caps_the_merge():
    hits = [_hit(platform="whatsapp", mid=f"WA-{i}",
                 ts=MIDDAY_TS - i) for i in range(25)]
    mixin = _mixin({"whatsapp": _FakeAdapter("whatsapp", hits),
                    "telegram": _FakeAdapter("telegram", [])})
    out = mixin._chatops_searchall("invoice", _msg())
    assert "25 messages match “invoice”" in out
    assert "showing the 20 most recent" in out


def test_searchall_one_surface_down_still_shows_the_other():
    wa_hit = _hit(platform="whatsapp", title="Poster Pal",
                  text="the invoice is ready", mid="WA-1")
    mixin = _mixin(
        {"whatsapp": _FakeAdapter("whatsapp", [wa_hit])},
        paired=("whatsapp",))  # Telegram has no paired account
    out = mixin._chatops_searchall("invoice", _msg())
    assert "1 message match “invoice”" in out
    assert "the invoice is ready" in out
    assert "Telegram" in out  # the missing surface is named
    assert "no paired" in out


def test_searchall_live_surface_refusal_is_named():
    mixin = _mixin(
        {"whatsapp": _RefusingAdapter("bridge is not logged in")},
        paired=("whatsapp",))
    out = mixin._chatops_searchall("invoice", _msg())
    assert "didn't run" in out
    assert "bridge is not logged in" in out
    assert "Telegram" in out


def test_searchall_both_down_names_both():
    mixin = _mixin({}, paired=())
    out = mixin._chatops_searchall("invoice", _msg())
    assert "didn't run" in out
    assert "WhatsApp" in out and "Telegram" in out


def test_searchall_from_bot_surface_still_searches_personal():
    # The origin surface does not limit the search: run from
    # the bot DM, it searches the personal accounts — the
    # targets are the personal accounts, not the room the
    # command arrived in.
    tg_hit = _hit(platform="telegram", title="Signal Room",
                  text="invoice totals are up",
                  ts=MIDDAY_TS, sender="Bo", mid="TG-1")
    mixin = _mixin({
        "whatsapp": _FakeAdapter("whatsapp", []),
        "telegram": _FakeAdapter("telegram", [tg_hit]),
    })
    out = mixin._chatops_searchall(
        "invoice", _msg(platform="telegram-bot",
                        chat_id="-1001234567890"))
    assert "invoice totals are up" in out
    assert "Telegram · Signal Room ·" in out


def test_searchall_empty_query_is_usage():
    mixin = _mixin({})
    out = mixin._chatops_searchall("   ", _msg())
    assert "usage: /searchall words to look for" in out


# ── composed chain: dispatch + registration + handler ────

@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    from pathlib import Path

    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings
    from nomorals.social.chat.gateway import ChatGateway
    from nomorals.social.chat.local import LocalAdapter
    from nomorals.social.pairing import AccountRecord

    tmp = tmp_path_factory.mktemp("xsearch")
    settings = load_settings(overrides={
        "home": str(tmp),
        "partner.platforms": "local",
        "partner.owner_sender_ids": OWNER,
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    runtime = PartnerRuntime(context, gateway=gateway)
    sup = runtime._pairing_supervisor()
    ws = Path(settings.workspace_dir)
    sup.registry.upsert(AccountRecord(
        id=f"whatsapp:{OWNER}", platform="whatsapp",
        identity=OWNER,
        creds_dir=str(ws / "data" / "whatsapp_sessions"
                      / f"wa-{OWNER}"),
        port=19999, status="linked", created_at=1.0))
    wa_hit = _hit(platform="whatsapp", title="Poster Pal",
                  text="the invoice is ready",
                  ts=MIDDAY_TS, mid="WA-1")
    wa = _FakeAdapter("whatsapp", [wa_hit])
    # The adapter resolution is stubbed to the fake; the
    # paired-account half stays REAL (registry above), so
    # the owner-stop pin below drives the true path.
    runtime._adapter_for = (  # type: ignore[method-assign]
        lambda platform: wa
        if platform == "whatsapp" else None)
    env = {"runtime": runtime, "wa": wa, "sup": sup, "tmp": tmp}
    yield env
    runtime.stop()
    runtime.gateway.stop()


def _op(env, text, *, platform="whatsapp",
        chat_id=PN_POSTER, kind=ChatKind.DM):
    chat = ChatRef(platform=platform, chat_id=chat_id,
                   kind=kind, title="", peer=chat_id)
    msg = ChatMessage(
        chat=chat, incoming=True, text=text, sender="Owner",
        sender_id=OWNER,
        message_id=f"op-{next(_MSG_SEQ)}",
        ts=MIDDAY_TS, meta={"owner_command": True})
    return env["runtime"].handle_control(
        text, chat.key, message=msg)


def test_searchall_composed_lists_whatsapp_hit(composed):
    out = _op(composed, "/searchall invoice")
    assert "1 message match “invoice”" in out
    assert "the invoice is ready" in out
    assert "WhatsApp · Poster Pal ·" in out
    assert "Telegram" in out  # unpaired surface named
    assert "no paired" in out
    assert composed["wa"].queries == ["invoice"]


def test_searchall_composed_empty_args_refused_by_gate(composed):
    before = len(composed["wa"].queries)
    out = _op(composed, "/searchall")
    assert "needs 1 argument(s)" in out
    assert len(composed["wa"].queries) == before


def test_searchall_is_a_read_not_an_act_under_owner_stop(
        composed):
    # The chatops family's standing doctrine: the owner's
    # standing stop gates ACTS as him, not reads of his own
    # messages.  /searchall is a read, so it answers under
    # a stop — driven here through the REAL paired-account
    # path, in deliberate contrast to /sendloc.
    composed["sup"].registry.set_owner_stopped(
        "whatsapp", True)
    try:
        out = _op(composed, "/searchall invoice")
        assert "the invoice is ready" in out
    finally:
        composed["sup"].registry.set_owner_stopped(
            "whatsapp", False)


def test_findall_rendering_unchanged(composed):
    # /findall still renders per-surface with no platform
    # label — the opt-in flag changes nothing for it.
    out = _op(composed, "/findall invoice")
    assert "1 message match “invoice”" in out
    assert "every chat on this surface" in out
    assert "Poster Pal" in out
    assert "WhatsApp ·" not in out
