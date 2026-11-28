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

    def test_dossier_enriched_rows_from_live_read(self):
        # A live read that lands fills every new row; the
        # phone names its source, the photo its count, the
        # channel its title, the history its last change date.
        live = {
            "username": "ada_ng", "name": "Ada N.",
            "phone": "+2348012345678",
            "bot": False, "premium": True, "verified": True,
            "scam": False, "fake": False, "restricted": False,
            "support": False, "mutual_contact": True,
            "deleted": False, "lang_code": "en",
            "about": "Builder of things.",
            "status_kind": "offline",
            "last_seen_ts": NOW - 3600,
            "usernames": ["ada_old"],
            "personal_channel_id": "123",
            "personal_channel_title": "Ada's channel",
            "has_photo": True, "photo_id": "999", "photo_dc": 4,
            "photo_count": 3, "common_chats_count": 2,
        }
        self.rt._lookup_user_full = lambda uid: dict(live)
        out = self._lookup("5478650254", "7")
        # render_card HTML-escapes the rows: assert on the
        # human-readable text, not the escaped form.
        import html

        out = html.unescape(out)
        assert "Builder of things." in out  # Bio
        assert "last seen" in out.lower()  # Presence
        assert "+2348012345678 (mutual contact" in out
        assert "Telegram Premium" in out
        assert "verified" in out
        assert "Ada's channel" in out
        assert "@ada_old" in out  # Also known as
        assert "3 on file" in out  # Profile photo count
        assert "Last username change" in out
        assert "Last name change" in out

    def test_dossier_not_visible_when_live_read_fails(self):
        # The userbot could not read the account at all: every
        # live field says not visible — never a dressed-up
        # empty, never a guessed value.
        self.rt._lookup_user_full = lambda uid: {}
        out = self._lookup("5478650254", "7")
        assert "not visible" in out.lower()
        assert "no bio set" not in out.lower()
        assert "Telegram only shows a number" in out
        assert "Flags" in out and "none" in out.lower()

    def test_phone_from_userbot_contacts(self):
        adapter = SimpleNamespace(
            user_contact_phone=lambda uid: "+2348011112222"
            if str(uid) == "7" else None)
        self.rt._group_adapter = lambda platform: (
            adapter if platform == "telegram" else None)
        self.rt._lookup_user_full = lambda uid: {
            "username": "ada_ng", "name": "Ada N."}
        import html

        out = html.unescape(self._lookup("5478650254", "7"))
        assert "+2348011112222 (in the userbot's contacts)" in out

    def test_phone_from_devon_addcontact_record(self):
        vault = SimpleNamespace(journal=lambda kind="": [
            {"kind": "contact_added", "counterpart_id": "7",
             "text": "Ada N. · +2348099988877",
             "ts": NOW - 86400}] if kind == "contact_added"
            else [])
        self.rt._pairing_personal = lambda: {
            "personal": SimpleNamespace(vault=vault)}
        self.rt._lookup_user_full = lambda uid: {
            "username": "ada_ng", "name": "Ada N."}
        out = self._lookup("5478650254", "7")
        assert "+2348099988877 (added via /addcontact" in out

    def test_phone_journal_row_for_another_account_ignored(self):
        # A contact_added row naming a DIFFERENT account never
        # lends its number to this dossier.
        vault = SimpleNamespace(journal=lambda kind="": [
            {"kind": "contact_added", "counterpart_id": "8",
             "text": "Bo · +2348000000001",
             "ts": NOW - 86400}] if kind == "contact_added"
            else [])
        self.rt._pairing_personal = lambda: {
            "personal": SimpleNamespace(vault=vault)}
        self.rt._lookup_user_full = lambda uid: {
            "username": "ada_ng", "name": "Ada N."}
        out = self._lookup("5478650254", "7")
        assert "+2348000000001" not in out
        assert "not visible" in out.lower()

    def test_graph_links_empty_for_unknown_handle(self):
        assert self.rt._lookup_graph_links("ada_ng") == []
        assert self.rt._lookup_graph_links("") == []

    def test_graph_links_returns_public_links_on_file(self):
        from nomorals.agents.osint_graph import IdentityGraph

        # the bare test DB has no schema: create the kv_store
        # table the graph persists through (migrations.py DDL
        # + migration 86's expires_at column)
        self.rt.context.db.execute(
            "CREATE TABLE IF NOT EXISTS kv_store ("
            "key TEXT PRIMARY KEY, value TEXT NOT NULL, "
            "kind TEXT NOT NULL DEFAULT 'json', "
            "updated_at REAL NOT NULL, expires_at REAL)")
        graph = IdentityGraph(self.rt.context)
        # the domain first: the extractor treats a domain
        # preceded by "@" on its line as part of an email
        graph.ingest("example.com @ada_ng", source="test")
        graph.save()
        assert self.rt._lookup_graph_links("ada_ng") == [
            "example.com"]


# ── telegramdb phone route: the leaked-corpus lookup ──────────


class _TelegramDbPhoneTests(unittest.TestCase):
    """Route (4) of the phone chain: the local Telegram user
    database.  The fixture DB is built with the module's own
    ``_write_sqlite`` — synthetic rows, test data only."""

    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.db import Database

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-tdb-")
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt.context = SimpleNamespace(
            db=Database(":memory:"),
            settings=SimpleNamespace(
                community_dir=self._tmp.name,
                home=self._tmp.name),
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
        surface.store.save()
        self.surface = surface
        self.rt._group_surface_obj = surface
        # No live phone, no contacts: only the database can
        # supply the number.
        self.rt._lookup_user_full = lambda uid: {
            "username": "ada_ng", "name": "Ada N."}
        self.rt._group_adapter = lambda platform: None

    def tearDown(self):
        self._tmp.cleanup()

    def _dm(self, sender_id, text):
        chat = SimpleNamespace(key=f"telegram:{sender_id}",
                               platform="telegram", kind=ChatKind.DM,
                               chat_id=sender_id)
        return SimpleNamespace(chat=chat, text=text, meta={},
                               sender="someone", sender_id=sender_id)

    def _lookup(self, sender_id, tail):
        import html

        out = self.rt.handle_control(
            f"/lookup {tail}", f"telegram:{sender_id}",
            self._dm(sender_id, f"/lookup {tail}"))
        return html.unescape(getattr(out, "text", out))

    def _fixture_db(self, rows):
        from nomorals.agents.osint_telegramdb import _write_sqlite

        path = Path(self._tmp.name) / "osint" / "telegramdb.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_sqlite(iter(rows), path, dataset="test-fixture")
        return path

    def test_phone_from_telegramdb_by_id(self):
        self._fixture_db([("7", "ada_ng", "2348077776666")])
        out = self._lookup("5478650254", "7")
        assert "+2348077776666 (Telegram user database" in out

    def test_phone_from_telegramdb_by_username_only(self):
        # No row for the ID — the handle alone finds it.
        self._fixture_db([("424242", "ada_ng", "2348077776667")])
        out = self._lookup("5478650254", "7")
        assert "+2348077776667 (Telegram user database" in out

    def test_phone_hidden_when_db_absent(self):
        out = self._lookup("5478650254", "7")
        assert "not visible" in out.lower()
        assert "Telegram only shows a number" in out

    def test_phone_hidden_when_db_has_no_row(self):
        self._fixture_db([("424242", "someone_else",
                           "2348000000001")])
        out = self._lookup("5478650254", "7")
        assert "not visible" in out.lower()
        assert "+2348000000001" not in out
        assert "Telegram database" in out

    def test_earlier_routes_win_over_telegramdb(self):
        self._fixture_db([("7", "ada_ng", "2348077776666")])
        adapter = SimpleNamespace(
            user_contact_phone=lambda uid: "+2348011112222"
            if str(uid) == "7" else None)
        self.rt._group_adapter = lambda platform: (
            adapter if platform == "telegram" else None)
        out = self._lookup("5478650254", "7")
        assert "+2348011112222 (in the userbot's contacts)" in out
        assert "+2348077776666" not in out

    def test_lookup_provision_status_missing(self):
        out = self._lookup("5478650254", "provision")
        assert "telegram" in out.lower()
        assert "not" in out.lower() and "provision" in out.lower()

    def test_lookup_provision_status_present(self):
        self._fixture_db([("7", "ada_ng", "2348077776666")])
        out = self._lookup("5478650254", "provision")
        assert "1" in out  # one row stocked
        assert "ready" in out.lower() or "stocked" in out.lower()

    def test_lookup_provision_refuses_when_disk_short(self):
        from nomorals.agents import osint_telegramdb as tdb

        old = tdb.NEED_BYTES
        tdb.NEED_BYTES = 10 ** 18
        try:
            out = self._lookup("5478650254", "provision telegramdb")
        finally:
            tdb.NEED_BYTES = old
        assert "free" in out.lower() or "space" in out.lower()
        assert "not" in out.lower() or "cannot" in out.lower()


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
    photos = types.ModuleType("telethon.tl.functions.photos")
    events_mod = types.ModuleType("telethon.events")

    class GetFullUserRequest:
        def __init__(self, user_id):
            self.user_id = user_id

    class GetCommonChatsRequest:
        def __init__(self, user_id, max_id=0, limit=100):
            self.user_id = user_id
            self.max_id = max_id
            self.limit = limit

    class GetUserPhotosRequest:
        def __init__(self, user_id, offset=0, max_id=0, limit=1):
            self.user_id = user_id
            self.offset = offset
            self.max_id = max_id
            self.limit = limit

    users.GetFullUserRequest = GetFullUserRequest
    messages.GetCommonChatsRequest = GetCommonChatsRequest
    photos.GetUserPhotosRequest = GetUserPhotosRequest
    telethon.tl = tl
    telethon.events = events_mod
    sys.modules["telethon"] = telethon
    sys.modules["telethon.tl"] = tl
    sys.modules["telethon.tl.functions"] = functions
    sys.modules["telethon.tl.functions.users"] = users
    sys.modules["telethon.tl.functions.messages"] = messages
    sys.modules["telethon.tl.functions.photos"] = photos
    sys.modules["telethon.events"] = events_mod
    return GetFullUserRequest, GetCommonChatsRequest


class _FakeClient:
    def __init__(self, *, full=None, chats=None, photos=None,
                 photo_path=None, contacts=None):
        self._full = full
        self._chats = chats
        self._photos = photos
        self._photo_path = photo_path
        self._contacts = contacts
        self.calls = []

    async def __call__(self, request):
        self.calls.append(type(request).__name__)
        if type(request).__name__ == "GetFullUserRequest":
            return self._full
        if type(request).__name__ == "GetCommonChatsRequest":
            return self._chats
        if type(request).__name__ == "GetUserPhotosRequest":
            if self._photos is None:
                raise AssertionError(
                    "unexpected request GetUserPhotosRequest")
            return self._photos
        raise AssertionError(f"unexpected request {request!r}")

    async def download_profile_photo(self, peer, file=None):
        self.calls.append("download_profile_photo")
        return self._photo_path

    async def get_contacts(self):
        self.calls.append("get_contacts")
        return self._contacts or []


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
                     "name": "Ada N.", "online": "",
                     "photo_id": ""}


def test_adapter_user_update_facts_none_when_no_user():
    from nomorals.social.chat.telegram import TelegramAdapter

    assert TelegramAdapter._user_update_facts(
        SimpleNamespace()) is None

# ── profile-photo observation ─────────────────────────────────────


def test_photo_change_is_recorded_as_identity_event(tmp_path):
    surface = _surface(tmp_path)
    surface.handle_message(KEY, "7", username="ada", name="Ada",
                           text="hi", ts=1000.0)
    touched = surface.note_user_update(
        "telegram-bot", "7", username="ada", name="Ada",
        photo_id="111")
    assert touched == [KEY]

    def _photo_events():
        history = (surface.member(KEY, "7") or {}).get(
            "identity_history") or []
        return [e for e in history if e["kind"] == "photo_changed"]

    events = _photo_events()
    assert len(events) == 1
    assert events[0]["new"] == "111"
    assert events[0]["old"] == ""
    assert events[0]["ts"]
    # the same photo again: no new event
    surface.note_user_update("telegram-bot", "7", username="ada",
                             name="Ada", photo_id="111")
    assert len(_photo_events()) == 1
    # a new photo: a second event, and the member carries it
    surface.note_user_update("telegram-bot", "7", username="ada",
                             name="Ada", photo_id="222")
    assert len(_photo_events()) == 2
    assert surface.member(KEY, "7").get("photo_id") == "222"


def test_photo_id_seeded_without_event_for_new_member(tmp_path):
    surface = _surface(tmp_path)
    surface.note_join(KEY, "7", username="ada", name="Ada",
                      photo_id="111", ts=1000.0)
    member = surface.member(KEY, "7")
    assert member.get("photo_id") == "111"
    assert not [e for e in (member.get("identity_history") or [])
                if e["kind"] == "photo_changed"]


# ── adapter: enriched user_full + photo + contact phone ──────────


def test_adapter_user_full_enriched_fields():
    from datetime import UTC, datetime

    _install_fake_telethon()

    class UserStatusOffline:
        def __init__(self, was_online):
            self.was_online = was_online

    user = SimpleNamespace(
        id=7, username="ada_ng", first_name="Ada", last_name="N.",
        phone="+2348012345678", bot=False, premium=True,
        verified=True, scam=False, fake=False, restricted=False,
        support=False, contact=True, mutual_contact=True,
        deleted=False, lang_code="en",
        status=UserStatusOffline(datetime.fromtimestamp(
            NOW - 3600, tz=UTC)),
        usernames=[SimpleNamespace(username="ada_old", active=True),
                   SimpleNamespace(username="ada_dead",
                                   active=False)])
    full_user = SimpleNamespace(
        user=user, about="Builder of things.",
        common_chats_count=2, personal_channel_id=123,
        profile_photo=SimpleNamespace(id=999, dc_id=4))
    fake = _FakeClient(
        full=SimpleNamespace(
            full_user=full_user,
            chats=[SimpleNamespace(id=123,
                                   title="Ada's channel")]),
        photos=SimpleNamespace(count=3))
    adapter = _adapter_with_fake_client(fake)
    out = adapter.user_full("7")
    assert out is not None
    assert out["phone"] == "+2348012345678"
    assert out["about"] == "Builder of things."
    assert out["status_kind"] == "offline"
    assert out["last_seen_ts"] == NOW - 3600
    assert out["usernames"] == ["ada_old"]
    assert out["personal_channel_id"] == "123"
    assert out["personal_channel_title"] == "Ada's channel"
    assert out["has_photo"] is True
    assert out["photo_id"] == "999"
    assert out["photo_dc"] == 4
    assert out["photo_count"] == 3
    assert out["verified"] is True
    assert out["mutual_contact"] is True
    assert out["common_chats_count"] == 2


def test_adapter_user_full_photo_count_absent_when_read_fails():
    _install_fake_telethon()
    user = SimpleNamespace(id=7, username="ada_ng",
                           first_name="Ada", last_name="",
                           phone=None, bot=False, premium=False)
    fake = _FakeClient(full=SimpleNamespace(full_user=SimpleNamespace(
        user=user)))
    adapter = _adapter_with_fake_client(fake)
    out = adapter.user_full("7")
    assert out is not None
    # the photos read raised in the fake: the count stays
    # absent, never 0 dressed up as "no photos"
    assert out["photo_count"] is None
    assert out["has_photo"] is False


def test_adapter_user_photo_path_returns_downloaded_file(tmp_path):
    _install_fake_telethon()
    wanted = str(tmp_path / "pfp-7.jpg")
    fake = _FakeClient(photo_path=wanted)
    adapter = _adapter_with_fake_client(fake)
    assert adapter.user_photo_path("7", str(tmp_path)) == wanted
    assert "download_profile_photo" in fake.calls


def test_adapter_user_photo_path_none_when_no_photo():
    _install_fake_telethon()
    fake = _FakeClient(photo_path=None)
    adapter = _adapter_with_fake_client(fake)
    assert adapter.user_photo_path("7", "/tmp") is None


def test_adapter_user_contact_phone_from_own_contacts():
    _install_fake_telethon()
    fake = _FakeClient(contacts=[
        SimpleNamespace(id=7, phone="+2348012345678"),
        SimpleNamespace(id=8, phone="+2348000000001")])
    adapter = _adapter_with_fake_client(fake)
    assert adapter.user_contact_phone("7") == "+2348012345678"
    assert adapter.user_contact_phone("9") is None
    assert "get_contacts" in fake.calls


# ── WhatsApp adapter: contact_info ──────────────────────────────


def _wa_adapter_with_admin(response):
    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = WhatsAppAdapter.__new__(WhatsAppAdapter)
    calls = []

    def _admin_cmd(cmd, payload, *, timeout=30.0):
        calls.append((cmd, payload))
        return dict(response)

    adapter._admin_cmd = _admin_cmd
    adapter._admin_calls = calls
    return adapter


def test_whatsapp_contact_info_normalizes_bare_number():
    adapter = _wa_adapter_with_admin({
        "ok": True, "jid": "2348012345678@s.whatsapp.net",
        "registered": True, "about": "Hey there",
        "has_photo": True,
        "photo_url": "https://example.invalid/p.jpg",
        "verified_name": "Ada", "lid": "123@lid"})
    out = adapter.contact_info("2348012345678")
    assert out["ok"] is True
    assert out["jid"] == "2348012345678@s.whatsapp.net"
    assert out["verified_name"] == "Ada"
    assert len(adapter._admin_calls) == 1
    cmd, payload = adapter._admin_calls[0]
    assert cmd == "contact_info"
    assert payload["jid"] == "2348012345678@s.whatsapp.net"


def test_whatsapp_contact_info_empty_jid_refuses():
    adapter = _wa_adapter_with_admin({})
    out = adapter.contact_info("")
    assert out["ok"] is False
    assert adapter._admin_calls == []
