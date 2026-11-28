"""The /lookup rebuild (owner correction 2026-11-28): the spine is
the BOT ITSELF — Bot API + a bot-token MTProto session + public
sources — and the userbot is a BONUS source only, never a
dependency. A name lookup returns a candidate list, never a
silent first pick; a public @username never returns an empty
dossier while any route is alive; every field gets every
crack-through route, and the dossier says what was tried.

Failing-first: every test here proves the old shape fails
before the rebuild lands.
"""
from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-100123"
KEY_B = "telegram-bot:-100456"
NOW = 1_790_000_000.0


# ── fakes ───────────────────────────────────────────────────────


class FakeBotSession:
    """A stand-in for the bot-token MTProto session: answers
    from the bot's own token, no userbot involved."""

    def __init__(self, *, full=None, resolve=None):
        self._full = full or {}
        self._resolve = resolve or {}

    def healthy(self):
        return True

    def resolve_username(self, username):
        return dict(self._resolve.get(username.lstrip("@")) or {})

    def user_full(self, user_id):
        found = self._full.get(str(user_id))
        return dict(found) if found else None


class FakeUserbotAdapter:
    """A stand-in for the userbot adapter: every rung answers
    only when the test arms it."""

    def __init__(self, **armed):
        self._armed = armed

    def member_by_username(self, chat, token):
        return self._armed.get("member_by_username")

    def user_full(self, user_id):
        return self._armed.get("user_full")

    def user_contact_phone(self, user_id):
        return self._armed.get("user_contact_phone")

    def resolve_phone(self, phone):
        return self._armed.get("resolve_phone")

    def import_phone_contact(self, phone, first_name="", last_name=""):
        return self._armed.get("import_phone_contact")

    def list_contacts(self):
        return self._armed.get("list_contacts") or []

    def search_group_members(self, chat, query, limit=50):
        return self._armed.get("search_group_members") or []

    def search_usernames(self, query, limit=10):
        return self._armed.get("search_usernames") or []


class FakeBotAdapter:
    """A stand-in for the Bot API adapter: getChat only."""

    def __init__(self, **armed):
        self._armed = armed

    def get_chat(self, ref):
        return self._armed.get("get_chat")


class FakeTelethonClient:
    """A duck-type Telethon client for BotTokenSession: the
    session talks to this small interface only, never to TL
    classes, so the session is testable with no telethon
    installed."""

    def __init__(self, *, entities=None, full=None,
                 photo_count=None, start_raises=False):
        self._entities = entities or {}
        self._full = full or {}
        self._photo_count = photo_count
        self._start_raises = start_raises
        self.started = False

    async def start(self, bot_token=None):
        if self._start_raises:
            raise RuntimeError("bad token")
        self.started = True

    def is_connected(self):
        return self.started

    async def get_entity(self, ref):
        key = str(ref).lstrip("@")
        found = self._entities.get(key) or self._entities.get(str(ref))
        if found is None:
            raise ValueError("no such entity")
        return found

    async def get_full_user(self, user_id):
        found = self._full.get(str(user_id))
        if found is None:
            raise ValueError("no such user")
        return found

    async def get_photo_count(self, user_id):
        return self._photo_count

    async def disconnect(self):
        self.started = False


def _fake_user(user_id, username, name, **extra):
    first, _, last = name.partition(" ")
    user = SimpleNamespace(
        id=int(user_id), username=username, first_name=first,
        last_name=last, bot=False, premium=False, verified=False,
        scam=False, fake=False, restricted=False, support=False,
        contact=False, mutual_contact=False, deleted=False,
        lang_code="en", phone=None, usernames=[], status=None)
    for key, value in extra.items():
        setattr(user, key, value)
    return user


def _fake_full(user, *, about="", photo_count=None):
    return SimpleNamespace(
        full_user=SimpleNamespace(
            user=user, about=about, common_chats_count=None,
            profile_photo=None, personal_channel_id=None),
        users=[user], chats=[])


# ── BotTokenSession unit tests ──────────────────────────────────


class BotTokenSessionTests(unittest.TestCase):
    def _session(self, **kw):
        from nomorals.social.chat.bot_session import BotTokenSession

        kw.setdefault("api_id", 12345)
        kw.setdefault("api_hash", "hash")
        kw.setdefault("bot_token", "123:ABC")
        kw.setdefault("session_path", "/tmp/no-such.session")
        kw.setdefault("call_timeout", 5.0)
        return BotTokenSession(**kw)

    def test_resolve_username_returns_entity(self):
        user = _fake_user(77, "stranger_x", "Stranger X")
        session = self._session(client_factory=lambda *a: (
            FakeTelethonClient(entities={"stranger_x": user})))
        found = session.resolve_username("stranger_x")
        assert found is not None
        assert found["id"] == "77"
        assert found["username"] == "stranger_x"
        assert found["name"] == "Stranger X"
        session.close()

    def test_user_full_shapes_like_adapter(self):
        user = _fake_user(77, "stranger_x", "Stranger X")
        full = _fake_full(user, about="Bot session bio")
        session = self._session(client_factory=lambda *a: (
            FakeTelethonClient(full={"77": full}, photo_count=2)))
        shaped = session.user_full("77")
        assert shaped is not None
        assert shaped["username"] == "stranger_x"
        assert shaped["about"] == "Bot session bio"
        assert shaped["photo_count"] == 2
        session.close()

    def test_dead_session_fails_fast_never_hangs(self):
        session = self._session(client_factory=lambda *a: (
            FakeTelethonClient(start_raises=True)))
        assert session.resolve_username("stranger_x") is None
        assert session.user_full("77") is None
        assert session.healthy() is False
        session.close()

    def test_rate_cap_refuses_after_limit(self):
        user = _fake_user(77, "stranger_x", "Stranger X")
        session = self._session(
            calls_per_minute=2,
            client_factory=lambda *a: (
                FakeTelethonClient(entities={"stranger_x": user})))
        assert session.resolve_username("stranger_x") is not None
        assert session.resolve_username("stranger_x") is not None
        assert session.resolve_username("stranger_x") is None
        session.close()

    def test_no_token_fails_closed(self):
        session = self._session(
            bot_token="",
            client_factory=lambda *a: FakeTelethonClient())
        assert session.resolve_username("stranger_x") is None
        assert session.healthy() is False
        session.close()


# ── t.me page rung unit tests ───────────────────────────────────


class TmeTests(unittest.TestCase):
    _HTML = (
        "<html><head>"
        '<meta property="og:title" content="Public Star">'
        '<meta property="og:description" content="Star bio\n\n'
        "12 000 subscribers\">"
        '<meta property="og:image" content="https://cdn.example/x.jpg">'
        "</head><body>"
        '<span class="tgme_page_title"><span dir="auto">Public Star'
        '</span><i class="verified-icon"></i></span>'
        "</body></html>")

    def test_fetch_tme_parses_public_page(self):
        from nomorals.agents.osint_tme import fetch_tme

        class Resp:
            def read(self):
                return TmeTests._HTML.encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        found = fetch_tme("public_star",
                          opener=lambda req: Resp())
        assert found is not None
        assert found["name"] == "Public Star"
        assert found["verified"] is True
        assert "Star bio" in found["bio"]
        assert found["photo_url"] == "https://cdn.example/x.jpg"

    def test_fetch_tme_malformed_never_raises(self):
        from nomorals.agents.osint_tme import fetch_tme

        class Resp:
            def read(self):
                return b"<html>garbage"

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        assert fetch_tme("nobody", opener=lambda req: Resp()) is None


# ── corpus coverage + phone lookup ──────────────────────────────


def _fixture_db(path: Path) -> Path:
    from nomorals.agents.osint_telegramdb import _write_sqlite

    rows = [
        ("100000001", "fixture_alpha", "15551234567"),
        ("100000002", "fixture_beta", "15551234568"),
        ("100000003", "fixture_gamma", "15551234569"),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_sqlite(iter(rows), path, dataset="test-fixture")
    return path


class TelegramDbCoverageTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-tdb2-")
        self.home = Path(self._tmp.name)
        self.db = _fixture_db(self.home / "osint" / "telegramdb.sqlite")

    def tearDown(self):
        self._tmp.cleanup()

    def test_coverage_names_the_window(self):
        from nomorals.agents.osint_telegramdb import coverage

        cov = coverage(home=self.home)
        assert cov["stocked"] is True
        assert cov["rows"] == 3
        assert cov["min_id"] == 100000001
        assert cov["max_id"] == 100000003
        # the fixture IDs are all early-2015 accounts: the window
        # says so plainly, so a 2022 account is honestly out of
        # range instead of silently missing.
        assert "2015" in cov["window"]

    def test_lookup_by_phone(self):
        from nomorals.agents.osint_telegramdb import (
            lookup_telegramdb_by_phone,
        )

        assert lookup_telegramdb_by_phone(
            "15551234568", db_path=self.db) == ("100000002",
                                                "fixture_beta")
        assert lookup_telegramdb_by_phone(
            "+15551234569", db_path=self.db) == ("100000003",
                                                 "fixture_gamma")
        assert lookup_telegramdb_by_phone(
            "15550000000", db_path=self.db) is None


# ── the rebuilt resolution order, end to end ────────────────────


class RebuildRuntimeTests(unittest.TestCase):
    def setUp(self):
        from nomorals.agents.partner_runtime import PartnerRuntime
        from nomorals.storage.db import Database

        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-lookup2-")
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
        surface.record_join(KEY, "7", username="ada_one",
                            name="Ada Lovelace", ts=NOW)
        surface.record_join(KEY_B, "8", username="ada_two",
                            name="Ada Ngozi", ts=NOW)
        surface.record_join(KEY, "9", username="jose_g",
                            name="José García", ts=NOW)
        surface.store.save()
        self.surface = surface
        self.rt._group_surface_obj = surface
        # the userbot is DOWN for every test here unless a test
        # arms it: the spine must answer without it.
        self.rt._group_adapter = lambda platform: None
        self.rt._lookup_bot_session = lambda: None
        self.rt._lookup_tme = lambda username: None

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

    def test_name_lookup_single_candidate_returns_dossier(self):
        out = self._lookup("5478650254", "Ada Lovelace")
        assert "ada_one" in out
        assert "7" in out
        assert "no record" not in out.lower()

    def test_name_lookup_multiple_candidates_lists_them(self):
        out = self._lookup("5478650254", "Ada")
        assert "Ada Lovelace" in out
        assert "Ada Ngozi" in out
        assert "7" in out and "8" in out
        # a list, not a silent first pick: no dossier rows
        assert "Account age" not in out
        assert "1." in out and "2." in out

    def test_name_lookup_accent_and_case_insensitive(self):
        out = self._lookup("5478650254", "JOSE GARCIA")
        assert "jose_g" in out
        assert "9" in out

    def test_username_resolves_via_bot_session_when_userbot_down(self):
        full = {
            "username": "stranger_x", "name": "Stranger X",
            "phone": None, "bot": False, "premium": False,
            "verified": True, "scam": False, "fake": False,
            "restricted": False, "support": False,
            "mutual_contact": False, "deleted": False,
            "lang_code": "en", "about": "Bot session bio",
            "status_kind": "offline", "last_seen_ts": None,
            "usernames": [], "personal_channel_id": "",
            "personal_channel_title": "", "has_photo": False,
            "photo_id": "", "photo_dc": None, "photo_count": 0,
            "common_chats_count": None,
        }
        self.rt._lookup_bot_session = lambda: FakeBotSession(
            full={"77": full},
            resolve={"stranger_x": {"id": "77",
                                    "username": "stranger_x",
                                    "name": "Stranger X"}})
        out = self._lookup("5478650254", "@stranger_x")
        assert "Stranger X" in out
        assert "Bot session bio" in out
        assert "77" in out
        assert "no record of @stranger_x" not in out.lower()

    def test_getchat_rung_answers_when_mtproto_misses(self):
        self.rt._group_adapter = lambda platform: FakeBotAdapter(
            get_chat={"id": "424242", "type": "private",
                      "username": "seen_user",
                      "first_name": "Seen", "last_name": "User"})
        out = self._lookup("5478650254", "424242")
        assert "Seen User" in out
        assert "seen_user" in out
        assert "no record of 424242" not in out.lower()

    def test_tme_rung_answers_for_public_username(self):
        self.rt._lookup_tme = lambda username: {
            "name": "Public Star", "verified": True,
            "bio": "Star bio", "photo_url": "https://cdn.example/x.jpg",
            "subscribers": "12 000"}
        out = self._lookup("5478650254", "@public_star")
        assert "Public Star" in out
        assert "Star bio" in out
        assert "no record" not in out.lower()

    def test_phone_number_resolves_via_resolve_phone(self):
        self.rt._group_adapter = lambda platform: FakeUserbotAdapter(
            resolve_phone={"id": "77", "username": "num_user",
                           "name": "Num User"})
        out = self._lookup("5478650254", "+2348012345678")
        assert "Num User" in out
        assert "77" in out
        assert "no record of +2348012345678" not in out.lower()

    def test_db_phone_verified_by_import(self):
        db = _fixture_db(
            Path(self._tmp.name) / "osint" / "telegramdb.sqlite")
        with sqlite3.connect(db) as con:
            con.execute(
                "INSERT INTO telegram_users VALUES (?, ?, ?)",
                ("7", "ada_ng", "+2348012345678"))
            con.commit()
        self.rt._group_adapter = lambda platform: FakeUserbotAdapter(
            import_phone_contact={"id": "7", "username": "ada_ng",
                                  "name": "Ada N."})
        out = self._lookup("5478650254", "7")
        assert "+2348012345678" in out
        assert "confirmed" in out.lower()

    def test_db_phone_mismatch_is_not_shown(self):
        db = _fixture_db(
            Path(self._tmp.name) / "osint" / "telegramdb.sqlite")
        with sqlite3.connect(db) as con:
            con.execute(
                "INSERT INTO telegram_users VALUES (?, ?, ?)",
                ("7", "ada_ng", "+2348012345678"))
            con.commit()
        self.rt._group_adapter = lambda platform: FakeUserbotAdapter(
            import_phone_contact={"id": "999", "username": "someone_else",
                                  "name": "Someone Else"})
        out = self._lookup("5478650254", "7")
        assert "+2348012345678" not in out
        assert "did not verify" in out.lower()

    def test_dossier_names_missed_rungs(self):
        # ledger-known account, every live rung down, no DB:
        # the dossier names what was tried instead of ending
        # the phone line at a bare "not visible".
        out = self._lookup("5478650254", "7")
        assert "bot session" in out.lower()
        assert "userbot" in out.lower()


if __name__ == "__main__":
    unittest.main()
