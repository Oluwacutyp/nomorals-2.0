"""Wave 85b — Discord as YOUR account: the user-client adapter.

Covers the account-mode surface hermetically (no network, no discord.py):

* preflight rejects a bot token, accepts a personal account token
* person discovery: servers / members / contacts mapping
* ``start_dm`` — the "say hi first" primitive (by id, by seen username)
* new-server-member reporting: at most once per person, persisted
* runtime wiring: a newcomer joins → the brain decides → first DM sent
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from nomorals.core.errors import ValidationError
from nomorals.social.chat import build_adapter
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.discord import DiscordAdapter


def _jwt(payload: dict) -> str:
    def b64(d: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(d).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'HS256', 'typ': 'JWT'})}.{b64(payload)}.c2ln"


class FakeUser:
    def __init__(self, uid: int, name: str, global_name: str = "") -> None:
        self.id = uid
        self.name = name
        self.global_name = global_name
        self.joined_at = None
        self.created_dms: list = []

    def __str__(self) -> str:
        return f"{self.name}#{self.id % 10000}"

    async def create_dm(self):
        dm = FakeDMChannel(900000000 + self.id, recipient=self)
        self.created_dms.append(dm)
        return dm


class FakeDMChannel:
    def __init__(self, cid: int, recipient: FakeUser, last: int = 0) -> None:
        self.id = cid
        self.recipient = recipient
        self.last_message_id = last


class FakeGuild:
    def __init__(self, gid: int, name: str, members: list | None = None) -> None:
        self.id = gid
        self.name = name
        self.members = members or []          # local cache
        self.api_members = list(self.members)  # what fetch_members() serves
        self.member_count = len(self.members)
        self.fetched = False

    async def fetch_members(self, limit: int = 250):
        self.fetched = True
        for m in self.api_members[:limit]:
            yield m


class FakeBot:
    def __init__(self) -> None:
        self.guilds: list = []
        self.users: dict[int, FakeUser] = {}
        self.dm_channels: list = []

    def get_user(self, uid: int):
        return self.users.get(uid)

    async def fetch_user(self, uid: int):
        if uid not in self.users:
            raise ValueError(f"unknown user {uid}")
        return self.users[uid]

    def get_users(self):
        return iter(self.users.values())

    def get_private_channels(self):
        return list(self.dm_channels)


class _LoopMixin:
    def setUp(self) -> None:
        super().setUp()
        self.tmp = tempfile.mkdtemp(prefix="nm-discord-test-")
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()
        time.sleep(0.05)

    def tearDown(self) -> None:
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(5)
        self.loop.close()
        shutil.rmtree(self.tmp, ignore_errors=True)
        super().tearDown()

    def _adapter(self, bot: FakeBot, **kw) -> DiscordAdapter:
        adapter = DiscordAdapter(
            token=_jwt({"username": "owner", "exp": 4102444800}),
            media_dir=os.path.join(self.tmp, "media"),
            **kw,
        )
        adapter._bot = bot
        adapter._loop = self.loop
        adapter._me = FakeUser(1, "owner")
        return adapter


class PreflightTests(_LoopMixin, unittest.TestCase):
    def test_rejects_bot_token(self) -> None:
        adapter = DiscordAdapter(token=_jwt({"user": "1547124778324992121", "bot": True,
                                             "exp": 4102444800}),
                                 media_dir=self.tmp)
        with self.assertRaises(ValidationError):
            adapter.preflight()

    def test_accepts_personal_token(self) -> None:
        adapter = DiscordAdapter(token=_jwt({"username": "owner", "exp": 4102444800}),
                                 media_dir=self.tmp)
        adapter.preflight()  # no raise

    def test_ignores_non_jwt(self) -> None:
        adapter = DiscordAdapter(token="not-a-jwt-at-all", media_dir=self.tmp)
        adapter.preflight()  # no raise — connect() will surface a bad token


class DiscoveryTests(_LoopMixin, unittest.TestCase):
    def _fixture(self) -> tuple[FakeBot, dict]:
        bot = FakeBot()
        newbie = FakeUser(111, "newbie", global_name="Newbie")
        regular = FakeUser(222, "regular")
        leaver = FakeUser(333, "leaver")
        newbie.joined_at = "2026-09-14T20:00:00+00:00"
        g1 = FakeGuild(100, "Game Night", members=[newbie, regular])
        g2 = FakeGuild(200, "Chill Hub", members=[leaver])
        bot.guilds = [g1, g2]
        bot.users.update({111: newbie, 222: regular, 333: leaver})
        bot.dm_channels = [
            FakeDMChannel(500, regular, last=5),
            FakeDMChannel(501, newbie, last=999),
            FakeDMChannel(502, leaver, last=500),
        ]
        return bot, {"newbie": newbie, "regular": regular, "leaver": leaver,
                     "g1": g1, "g2": g2}

    def test_servers(self) -> None:
        bot, _ = self._fixture()
        rows = self._adapter(bot).servers()
        self.assertEqual([r["name"] for r in rows], ["Game Night", "Chill Hub"])
        self.assertEqual(rows[0]["members"], 2)

    def test_members_by_name_and_id(self) -> None:
        bot, refs = self._fixture()
        adapter = self._adapter(bot)
        by_name = adapter.members("game night")
        self.assertEqual([m["id"] for m in by_name], ["111", "222"])
        self.assertEqual(by_name[0]["name"], "Newbie")
        self.assertIn("2026-09-14", by_name[0]["joined"])
        by_id = adapter.members("200")
        self.assertEqual([m["id"] for m in by_id], ["333"])
        self.assertEqual(adapter.members("no such guild"), [])

    def test_members_falls_back_to_fetch(self) -> None:
        bot, refs = self._fixture()
        # big server: nothing in the local cache, only the API knows
        refs["g2"].members = []
        refs["g2"].api_members = [refs["leaver"]]
        bot.guilds = [refs["g2"]]
        rows = self._adapter(bot).members("Chill Hub")
        self.assertTrue(refs["g2"].fetched)
        self.assertEqual([m["id"] for m in rows], ["333"])

    def test_contacts_sorted_by_recency(self) -> None:
        bot, _ = self._fixture()
        rows = self._adapter(bot).contacts(limit=10)
        self.assertEqual([r["id"] for r in rows], ["111", "333", "222"])
        self.assertEqual(rows[0]["dm_id"], "501")
        self.assertEqual([r["id"] for r in self._adapter(bot).contacts(limit=1)], ["111"])

    def test_start_dm_by_id(self) -> None:
        bot, refs = self._fixture()
        chat = self._adapter(bot).start_dm("111")
        self.assertIsNotNone(chat)
        self.assertEqual(chat.kind, ChatKind.DM)
        self.assertEqual(chat.chat_id, str(900000000 + 111))
        self.assertEqual(chat.peer, str(refs["newbie"]))
        self.assertEqual(chat.title, "Newbie")

    def test_start_dm_by_seen_username(self) -> None:
        bot, refs = self._fixture()
        chat = self._adapter(bot).start_dm("regular")
        self.assertIsNotNone(chat)
        self.assertEqual(chat.peer, str(refs["regular"]))

    def test_start_dm_unknown_returns_none(self) -> None:
        bot, _ = self._fixture()
        self.assertIsNone(self._adapter(bot).start_dm("ghost#9999"))
        self.assertIsNone(self._adapter(bot).start_dm("999999"))
        self.assertIsNone(self._adapter(bot).start_dm("   "))

    def test_health(self) -> None:
        bot, _ = self._fixture()
        health = self._adapter(bot).health()
        self.assertEqual(health["me"], "owner#1")
        self.assertEqual(health["servers"], 2)
        self.assertEqual(health["dms"], 3)
        self.assertTrue(health["connected"])


class NewMemberTests(_LoopMixin, unittest.TestCase):
    def _join(self, member_id: int, name: str, guild: str = "Game Night") -> None:
        member = SimpleNamespace(id=member_id, name=name, global_name=name,
                                 joined_at="2026-09-14T20:00:00+00:00",
                                 guild=SimpleNamespace(name=guild))
        return member

    def test_fires_once_and_persists(self) -> None:
        calls: list = []
        def cb(guild: str, member: dict) -> None:
            calls.append((guild, member))

        adapter = self._adapter(FakeBot(), on_new_member=cb)
        adapter._fire_new_member("Game Night", self._join(111, "newbie"))
        adapter._fire_new_member("Game Night", self._join(111, "newbie"))  # repeat
        adapter._fire_new_member("Chill Hub", self._join(222, "regular"))
        deadline = time.time() + 5
        while len(calls) < 2 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(calls), 2)
        guilds, members = zip(*calls)
        self.assertEqual(set(guilds), {"Game Night", "Chill Hub"})
        self.assertEqual({m["id"] for m in members}, {"111", "222"})
        # survives a restart: a fresh adapter must not re-fire for 111
        adapter2 = DiscordAdapter(token=_jwt({"username": "o", "exp": 99}),
                                  media_dir=adapter.media_dir,
                                  on_new_member=calls.append)
        adapter2._fire_new_member("Game Night", self._join(111, "newbie"))
        deadline = time.time() + 2
        while len(calls) < 3 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(calls), 2)  # 111 deduped from disk

    def test_disabled_never_fires(self) -> None:
        calls: list = []
        adapter = self._adapter(FakeBot(), greet_new=False,
                                on_new_member=lambda g, m: calls.append((g, m)))
        adapter._fire_new_member("Game Night", self._join(111, "newbie"))
        time.sleep(0.2)
        self.assertEqual(calls, [])

    def test_no_callback_is_a_noop(self) -> None:
        adapter = self._adapter(FakeBot())
        adapter._fire_new_member("Game Night", self._join(111, "newbie"))  # must not raise


class BuildAdapterTests(_LoopMixin, unittest.TestCase):
    def _settings(self) -> SimpleNamespace:
        settings = SimpleNamespace(
            partner=SimpleNamespace(platforms="discord"),
            chat=SimpleNamespace(
                discord_enabled=True,
                discord_token=_jwt({"username": "owner", "exp": 99}),
                discord_channels="",
                discord_greet_new=True,
                local_enabled=False,
            ),
        )
        settings.resolve = lambda p: os.path.join(self.tmp, p)
        return settings

    def test_wires_new_member_callback(self) -> None:
        cb = lambda *a: None  # noqa: E731
        adapter = build_adapter(self._settings(), "discord", on_new_member=cb)
        self.assertIsInstance(adapter, DiscordAdapter)
        self.assertIs(adapter.on_new_member, cb)
        self.assertTrue(adapter.greet_new)

    def test_disabled_returns_none(self) -> None:
        settings = self._settings()
        settings.chat.discord_enabled = False
        self.assertIsNone(build_adapter(settings, "discord"))


class RuntimeNewcomerTests(_LoopMixin, unittest.TestCase):
    """The brain decides the first DM — the runtime just carries it."""

    def _stub_runtime(self, dm_ref: ChatRef | None, parts: list[str]):
        from nomorals.agents.partner_runtime import PartnerRuntime

        greet = PartnerRuntime._greet_discord_newcomer
        on_new = PartnerRuntime._on_new_discord_member
        fake_adapter = SimpleNamespace(start_dm=lambda target: dm_ref)
        sent: list = []
        stub = SimpleNamespace(
            _pool=ThreadPoolExecutor(max_workers=2),
            _stopped=threading.Event(),
            dry_run=False,
            gateway=SimpleNamespace(adapters={"discord": fake_adapter}),
            brain=SimpleNamespace(
                handle_message=lambda m: SimpleNamespace(parts=list(parts))),
            stats={"replies": 0},
        )
        stub._send_reply = lambda msg, ps: sent.append((msg.chat.peer, list(ps)))
        stub._greet_discord_newcomer = lambda g, m: greet(stub, g, m)
        stub._sent = sent
        return stub, greet, on_new

    def test_says_hi_when_brain_wants_to(self) -> None:
        ref = ChatRef(platform="discord", chat_id="501", kind=ChatKind.DM,
                      title="Newbie", peer="newbie#111")
        stub, greet, on_new = self._stub_runtime(ref, ["hey, welcome to the server!"])
        greet(stub, "Game Night", {"id": "111", "name": "Newbie", "tag": "newbie#111",
                                   "joined": "2026-09-14"})
        self.assertEqual(stub._sent, [("newbie#111", ["hey, welcome to the server!"])])
        self.assertEqual(stub.stats["replies"], 1)

    def test_stays_quiet_when_brain_stays_quiet(self) -> None:
        ref = ChatRef(platform="discord", chat_id="501", kind=ChatKind.DM,
                      title="Newbie", peer="newbie#111")
        stub, greet, _ = self._stub_runtime(ref, [])
        greet(stub, "Game Night", {"id": "111", "name": "Newbie", "tag": "", "joined": ""})
        self.assertEqual(stub._sent, [])
        self.assertEqual(stub.stats["replies"], 0)

    def test_unresolvable_dm_is_skipped(self) -> None:
        stub, greet, _ = self._stub_runtime(None, ["hi?"])
        greet(stub, "Game Night", {"id": "999", "name": "Ghost", "tag": "", "joined": ""})
        self.assertEqual(stub._sent, [])

    def test_pool_handoff_and_dry_run_gate(self) -> None:
        ref = ChatRef(platform="discord", chat_id="501", kind=ChatKind.DM,
                      title="Newbie", peer="newbie#111")
        stub, _, on_new = self._stub_runtime(ref, ["hi!"])
        on_new(stub, "Game Night", {"id": "111", "name": "Newbie", "tag": "", "joined": ""})
        deadline = time.time() + 5
        while not stub._sent and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(len(stub._sent), 1)
        # dry run: nothing is submitted at all
        stub.dry_run = True
        on_new(stub, "Game Night", {"id": "222", "name": "Other", "tag": "", "joined": ""})
        time.sleep(0.2)
        self.assertEqual(len(stub._sent), 1)


if __name__ == "__main__":
    unittest.main()
