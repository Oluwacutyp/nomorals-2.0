"""Member intelligence: account lookups + username/name-change
tracking (goal goal_d5240db4f87e).

The dossier must match and beat the Telegram userbot info bots
— ID, creation-date ESTIMATE (labelled, from the ID range),
phone only when legitimately visible, groups the account is
in — plus username/display-name history with dates, economy
standing and moderation record.  Every field is a verified
fact with its source, an honestly-labelled estimate, or
"not visible".  Never a confident guess.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-100123"
KEY_B = "telegram-bot:-100456"
CHANNEL = "telegram-bot:-100789"
NOW = 1_790_000_000.0


def _surface(tmp_path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "groups.json").load())


# ── durable per-member identity history ─────────────────────────


def test_identity_history_is_durable_with_dates(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    surface.handle_message(KEY, "7", username="ada_ng",
                           name="Ada N.", text="yo", ts=1001.0)
    surface.handle_message(KEY, "7", username="ada_ng",
                           name="Ada Lovelace", text="again", ts=1002.0)
    member = surface.member(KEY, "7")
    history = member.get("identity_history") or []
    assert len(history) == 3
    kinds = [(entry["kind"], entry["old"], entry["new"])
             for entry in history]
    assert ("username_changed", "ada", "ada_ng") in kinds
    assert ("name_changed", "Ada", "Ada N.") in kinds
    assert ("name_changed", "Ada N.", "Ada Lovelace") in kinds
    assert all(entry.get("ts") for entry in history)
    # survives a reload: it is stored on the member, not the ring
    reloaded = GroupSurface(
        GroupStore(tmp_path / "groups.json").load())
    assert len((reloaded.member(KEY, "7") or {})
               .get("identity_history") or []) == 3


def test_identity_history_caps_at_fifty_newest_kept(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="u0", name="N0",
                           text="hi", ts=1000.0)
    for i in range(1, 60):
        surface.handle_message(KEY, "7", username=f"u{i}",
                               name=f"N{i}", text="x", ts=1000.0 + i)
    history = (surface.member(KEY, "7") or {}).get(
        "identity_history") or []
    assert len(history) == 50
    # each message changed username AND name (two events), so
    # the newest entry is the last name change, the oldest kept
    # is far past the first iterations
    assert history[-1]["new"] == "N59"
    assert history[-2]["new"] == "u59"
    assert all(entry["new"] not in ("u0", "N0", "u1", "N1")
               for entry in history)


def test_identity_history_seeded_from_existing_ring(tmp_path):
    # A member whose changes predate the durable list (recorded
    # only in the per-group changes ring) gets them backfilled
    # once, the first time the member is observed afterwards.
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    surface.handle_message(KEY, "7", username="ada_ng", name="Ada",
                           text="yo", ts=1001.0)
    member = surface.member(KEY, "7")
    member.pop("identity_history", None)
    surface.store.save()
    surface.handle_message(KEY, "7", username="ada_ng", name="Ada",
                           text="still here", ts=1002.0)
    history = (surface.member(KEY, "7") or {}).get(
        "identity_history") or []
    assert any(entry["new"] == "ada_ng" for entry in history)


# ── groups_for_member ───────────────────────────────────────────


def test_groups_for_member_lists_groups_and_channels(tmp_path):
    surface = _surface(tmp_path)
    surface.record_join(KEY, "7", username="ada", name="Ada", ts=NOW)
    surface.record_join(KEY_B, "7", username="ada", name="Ada", ts=NOW)
    centry = surface.store.group(CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    centry.setdefault("members", {})["7"] = {
        "user_id": "7", "username": "ada", "name": "Ada"}
    surface.store.save()
    found = surface.groups_for_member("7")
    keys = {entry["key"] for entry in found}
    assert keys == {KEY, KEY_B, CHANNEL}
    by_key = {entry["key"]: entry for entry in found}
    assert by_key[CHANNEL]["kind"] == "channel"
    assert by_key[KEY]["kind"] == "group"


def test_groups_for_member_empty_for_stranger(tmp_path):
    surface = _surface(tmp_path)
    assert surface.groups_for_member("999") == []


# ── note_user_update fan-out ────────────────────────────────────


def test_note_user_update_fans_out_to_known_groups_only(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    surface.handle_message(KEY_B, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    surface.handle_message(KEY, "8", username="bo", name="Bo",
                           text="hi", ts=1000.0)
    touched = surface.note_user_update("telegram-bot", "7",
                                       username="ada_ng", name="Ada")
    assert set(touched) == {KEY, KEY_B}
    for key in (KEY, KEY_B):
        history = (surface.member(key, "7") or {}).get(
            "identity_history") or []
        assert any(entry["new"] == "ada_ng" for entry in history)
    # member 8 untouched; a stranger creates no phantom member
    assert (surface.member(KEY, "8") or {}).get("username") == "bo"
    assert surface.note_user_update("telegram-bot", "999",
                                    username="ghost",
                                    name="Ghost") == []
    assert surface.member(KEY, "999") is None


def test_note_user_update_ignores_other_platforms(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    assert surface.note_user_update("whatsapp", "7",
                                    username="ada_ng",
                                    name="Ada") == []


# ── enriched /member card ───────────────────────────────────────


def _card_surface(tmp_path):
    from nomorals.agents.partner.runtime_group import _member_card

    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    surface.handle_message(KEY, "7", username="ada_ng",
                           name="Ada N.", text="yo", ts=1001.0)
    surface.economy.award(KEY, "7", 100, "seed", now=NOW)
    member = surface.member(KEY, "7")
    member["warns"] = [{"ts": NOW, "reason": "spam",
                        "by": "admin1"}]
    surface.store.save()
    return surface, _member_card


def test_member_card_carries_economy_moderation_history(tmp_path):
    surface, card_fn = _card_surface(tmp_path)
    member = surface.member(KEY, "7")
    card = card_fn(surface, KEY, member)
    assert "Economy" in card
    # wallet is the 100 award plus the activity bonus the
    # economy grants on messages — read the real balance,
    # never hardcode the bonus
    assert str(surface.economy.balance(KEY, "7")) in card
    assert "Moderation" in card
    assert "warn" in card.lower()
    assert "Identity history" in card
    assert "ada → ada_ng" in card
    assert "Ada → Ada N." in card


# ── /lookup registration + dossier ──────────────────────────────


def test_lookup_is_a_public_command():
    from nomorals.social.chat.control import (
        CONTROL_COMMANDS,
        PUBLIC_COMMANDS,
        caller_may_run,
        parse_control,
    )

    assert "lookup" in CONTROL_COMMANDS
    assert "lookup" in PUBLIC_COMMANDS
    assert caller_may_run("lookup", is_owner=False)
    assert parse_control("/lookup 123").kind == "lookup"


class _LookupRuntimeTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.db import Database

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-lookup-")
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = SimpleNamespace(
            db=Database(":memory:"),
            settings=SimpleNamespace(
                community_dir=self._tmp.name),
        )
        rt._owner_chats = {"5478650254"}
        rt._owner_sender_ids = {"5478650254"}
        self.rt = rt
        surface = GroupSurface(
            GroupStore(Path(self._tmp.name) / "g.json").load())
        surface.record_join(KEY, "7", username="ada", name="Ada",
                            ts=NOW)
        surface.handle_message(KEY, "7", username="ada_ng",
                               name="Ada N.", text="hi", ts=NOW + 1)
        surface.economy.award(KEY, "7", 250, "seed", now=NOW)
        surface.record_join(KEY_B, "7", username="ada_ng",
                            name="Ada N.", ts=NOW)
        surface.store.save()
        self.surface = surface
        self.rt._group_surface_obj = surface

    def tearDown(self):
        self._tmp.cleanup()

    def _dm(self, sender_id, text):
        chat = SimpleNamespace(key=f"telegram:{sender_id}",
                               platform="telegram", kind=ChatKind.DM,
                               chat_id=sender_id)
        return SimpleNamespace(chat=chat, text=text, meta={},
                               sender="someone", sender_id=sender_id)

    def _lookup(self, sender_id, tail):
        out = self.rt.handle_control(
            f"/lookup {tail}", f"telegram:{sender_id}",
            self._dm(sender_id, f"/lookup {tail}"))
        return getattr(out, "text", out)

    def test_dossier_by_id_for_operator(self):
        out = self._lookup("5478650254", "7")
        assert "7" in out
        assert "ada_ng" in out
        assert "estimate" in out.lower()
        assert "not visible" in out.lower()  # phone: no mutual contact
        assert str(self.surface.economy.balance(KEY, "7")) in out
        assert KEY in out  # groups the account is in
        assert "ada → ada_ng" in out  # username history
        assert "Ada → Ada N." in out  # display-name history

    def test_dossier_by_username(self):
        out = self._lookup("5478650254", "@ada_ng")
        assert "ada_ng" in out
        assert "7" in out

    def test_dossier_unknown_account(self):
        out = self._lookup("5478650254", "999999")
        assert "no record" in out.lower() or "not found" in out.lower()

    def test_dossier_refused_for_stranger_in_dm(self):
        out = self._lookup("123456", "7")
        assert "admin" in out.lower() or "refus" in out.lower()

    def test_dossier_allowed_for_self(self):
        out = self._lookup("7", "7")
        assert "ada_ng" in out


# ── user_update seam: silent fan-out, no reply ──────────────────


class _UserUpdateTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.db import Database

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-uupd-")
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = SimpleNamespace(
            db=Database(":memory:"),
            settings=SimpleNamespace(
                community_dir=self._tmp.name),
        )
        rt._owner_chats = {"5478650254"}
        rt._owner_sender_ids = {"5478650254"}
        self.rt = rt
        surface = GroupSurface(
            GroupStore(Path(self._tmp.name) / "g.json").load())
        surface.handle_message(KEY, "7", username="ada", name="Ada",
                               text="hi", ts=1000.0)
        surface.store.save()
        self.surface = surface
        self.rt._group_surface_obj = surface

    def tearDown(self):
        self._tmp.cleanup()

    def test_process_inner_routes_user_update_silently(self):
        chat = SimpleNamespace(key="telegram:7", platform="telegram",
                               kind=ChatKind.DM, chat_id="7")
        msg = SimpleNamespace(
            chat=chat, text="", meta={
                "service": "user_update", "user_id": "7",
                "username": "ada_ng", "name": "Ada N."},
            sender="Ada", sender_id="7", message_id="upd-1")
        self.rt._process_inner(msg)
        history = (self.surface.member(KEY, "7") or {}).get(
            "identity_history") or []
        assert any(entry["new"] == "ada_ng" for entry in history)


# ── adapter: user_full + common_chats (fake telethon) ───────────


def _install_fake_telethon():
    telethon = types.ModuleType("telethon")
    tl = types.ModuleType("telethon.tl")
    functions = types.ModuleType("telethon.tl.functions")
    users = types.ModuleType("telethon.tl.functions.users")
    messages = types.ModuleType("telethon.tl.functions.messages")
    events_mod = types.ModuleType("telethon.events")

    class GetFullUserRequest:
        def __init__(self, user_id):
            self.user_id = user_id

    class GetCommonChatsRequest:
        def __init__(self, user_id, max_id=0, limit=100):
            self.user_id = user_id
            self.max_id = max_id
            self.limit = limit

    users.GetFullUserRequest = GetFullUserRequest
    messages.GetCommonChatsRequest = GetCommonChatsRequest
    telethon.tl = tl
    telethon.events = events_mod
    sys.modules["telethon"] = telethon
    sys.modules["telethon.tl"] = tl
    sys.modules["telethon.tl.functions"] = functions
    sys.modules["telethon.tl.functions.users"] = users
    sys.modules["telethon.tl.functions.messages"] = messages
    sys.modules["telethon.events"] = events_mod
    return GetFullUserRequest, GetCommonChatsRequest


class _FakeClient:
    def __init__(self, *, full=None, chats=None):
        self._full = full
        self._chats = chats
        self.calls = []

    async def __call__(self, request):
        self.calls.append(type(request).__name__)
        if type(request).__name__ == "GetFullUserRequest":
            return self._full
        if type(request).__name__ == "GetCommonChatsRequest":
            return self._chats
        raise AssertionError(f"unexpected request {request!r}")


def _adapter_with_fake_client(fake_client):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._client = fake_client
    adapter._loop = None
    adapter._run_on_loop = lambda coro, timeout=30.0: asyncio.run(coro)
    return adapter


def test_adapter_user_full_returns_phone_when_mutual():
    _install_fake_telethon()
    user = SimpleNamespace(id=7, username="ada_ng",
                           first_name="Ada", last_name="N.",
                           phone="+2348012345678", bot=False,
                           premium=True)
    fake = _FakeClient(full=SimpleNamespace(full_user=SimpleNamespace(
        user=user)))
    adapter = _adapter_with_fake_client(fake)
    out = adapter.user_full("7")
    assert out is not None
    assert out["phone"] == "+2348012345678"
    assert out["premium"] is True
    assert out["name"] == "Ada N."
    assert "GetFullUserRequest" in fake.calls


def test_adapter_user_full_phone_absent_when_not_mutual():
    _install_fake_telethon()
    user = SimpleNamespace(id=7, username="ada_ng",
                           first_name="Ada", last_name="",
                           phone=None, bot=False, premium=False)
    fake = _FakeClient(full=SimpleNamespace(full_user=SimpleNamespace(
        user=user)))
    adapter = _adapter_with_fake_client(fake)
    out = adapter.user_full("7")
    assert out is not None
    assert out["phone"] in (None, "")


def test_adapter_user_full_none_when_no_client():
    adapter = _adapter_with_fake_client(None)
    assert adapter.user_full("7") is None


def test_adapter_common_chats_lists_shared_groups():
    _install_fake_telethon()
    fake = _FakeClient(chats=SimpleNamespace(chats=[
        SimpleNamespace(id=-100123, title="Builders"),
        SimpleNamespace(id=-100456, title="Signals"),
    ]))
    adapter = _adapter_with_fake_client(fake)
    out = adapter.common_chats("7")
    assert out is not None
    assert len(out) == 2
    assert out[0]["title"] == "Builders"
    assert "GetCommonChatsRequest" in fake.calls


def test_adapter_user_update_facts_from_event():
    from nomorals.social.chat.telegram import TelegramAdapter

    event = SimpleNamespace(user=SimpleNamespace(
        id=7, username="ada_ng", first_name="Ada", last_name="N."))
    facts = TelegramAdapter._user_update_facts(event)
    assert facts == {"user_id": "7", "username": "ada_ng",
                     "name": "Ada N."}


def test_adapter_user_update_facts_none_when_no_user():
    from nomorals.social.chat.telegram import TelegramAdapter

    assert TelegramAdapter._user_update_facts(
        SimpleNamespace()) is None
