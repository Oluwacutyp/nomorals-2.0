"""Owner broadcast + group re-add recovery (DESIGN §20b, §9.8).

§20b: the owner speaks through Devon to all groups/channels at once
with pacing and dry-run (the mass-tools class, §14, under owner
authority). §20e (sealed DM rebind) is already built — /owner rebind,
test-pinned in test_webshot_owner.py — and is NOT re-tested here
beyond the registry fact the recovery DM reads.

Discipline under test (the §14 mass-tools rules, applied to a tool
that speaks as the BOT, not as him): dry-run is the default and
touches NOTHING (the fake gateway asserts zero platform calls), a
live run refuses until the tool is ARMED (and the refusal is
journaled), pacing has a 3s floor the owner can raise but never
lower (Telegram groups: 20 msg/min), one send per room with
per-room failure isolation, every run lands in the journal
complete. Recovery: when the bot itself is re-added to a group it
already served, the owner gets a DM naming the latest backup and
the exact restore command — the restore itself never auto-applies.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    COMMAND_DETAILS,
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database
from tests.frozen_clock import MIDDAY_TS

OWNER_ID = "111"
BOT_ID = "999888777"
MEMBER_ID = "u99"
GROUP_A = "telegram-bot:-100111"
GROUP_B = "telegram-bot:-100222"
CHANNEL = "telegram-bot:-100333"
FOREIGN = "discord:444555"
NOW = MIDDAY_TS


def _bare_runtime(tmp_path: Path) -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(
            community_dir=str(tmp_path),
            workspace_dir=str(tmp_path)),
    )
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _dm_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=f"telegram:{sender_id}", platform="telegram",
                           kind=ChatKind.DM, chat_id=sender_id)
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id, sender_username="owner")


def _join_msg(members, key=GROUP_A):
    chat = SimpleNamespace(key=key, platform="telegram-bot",
                           kind=ChatKind.GROUP, chat_id="-100111",
                           title="Group A")
    return SimpleNamespace(
        chat=chat, text="", meta={"service": "join",
                                  "new_members": members},
        sender="someone", sender_id="1", sender_username="ada",
        message_id="")


def _text(out) -> str:
    return getattr(out, "text", out)


class FakeGateway:
    """The gateway contract the broadcast organ rides: broadcast +
    send, with a call log and scripted per-chat results."""

    def __init__(self, *, fail_chats=()):
        self.broadcast_calls: list = []
        self.sends: list = []
        self.adapters: dict = {}
        self._fail = set(fail_chats)

    def broadcast(self, platform, chats, text, **kw):
        self.broadcast_calls.append((platform, list(chats), text, kw))
        results = []
        for chat in chats:
            key = str(chat)
            ok = key not in self._fail
            results.append({"chat": key, "ok": ok,
                            "error": "" if ok else "chat_write_forbidden"})
        return {"sent": sum(1 for r in results if r["ok"]),
                "failed": sum(1 for r in results if not r["ok"]),
                "aborted": False, "results": results}

    def send(self, platform, chat, text, **kw):
        self.sends.append((platform, chat, text))
        return SimpleNamespace(ok=True, message_id="1")


class FakeBotAdapter:
    """A Telegram bot adapter: its own id + a member-status probe."""

    def __init__(self, bot_id=BOT_ID, status="administrator"):
        self._bot_id = bot_id
        self._status = status
        self.probes: list = []

    def admin_member(self, chat, user_id):
        self.probes.append((str(getattr(chat, "key", chat)), str(user_id)))
        return {"status": self._status}


def _seed_surface(tmp_path: Path) -> GroupSurface:
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    for uid, name in (("1", "Ada"), ("2", "Bola")):
        surface.record_join(GROUP_A, uid, username=name.lower(),
                            name=name, ts=NOW)
    surface.store.group(GROUP_A)["title"] = "Group A"
    surface.record_join(GROUP_B, "1", username="ada", name="Ada", ts=NOW)
    surface.store.group(GROUP_B)["title"] = "Group B"
    centry = surface.store.group(CHANNEL)
    centry.setdefault("extras", {})["channel"] = {"title": "Signals"}
    # A foreign room Devon merely sits in: no designation, no bot
    # standing on a bot platform — the §33 gate must exclude it.
    surface.store.group(FOREIGN)["title"] = "Foreign"
    surface.store.save()
    return surface


class _BroadcastCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-obcast-")
        self.tmp_path = Path(self._tmp.name)
        self.rt = _bare_runtime(self.tmp_path)
        self.surface = _seed_surface(self.tmp_path)
        self.rt._group_surface_obj = self.surface
        self.gateway = FakeGateway()
        self.gateway.adapters["telegram-bot"] = FakeBotAdapter()
        self.rt.gateway = self.gateway
        self.rt._owner_registry().seed(telegram_ids=[OWNER_ID])

    def tearDown(self):
        self._tmp.cleanup()

    def _reply(self, text, sender_id=OWNER_ID):
        return self.rt.handle_control(
            text, f"telegram:{sender_id}",
            message=_dm_msg(sender_id, text))


class TestOwnerBroadcastOrgan(_BroadcastCase):
    """The organ on a bare GroupSurface + fake gateway."""

    def test_targets_only_claimed_rooms(self):
        from nomorals.social.chat.group_broadcast import broadcast_targets

        targets = broadcast_targets(self.surface)
        self.assertEqual([t["key"] for t in targets["groups"]],
                         [GROUP_A, GROUP_B])
        self.assertEqual([t["key"] for t in targets["channels"]],
                         [CHANNEL])
        self.assertEqual(targets["channels"][0]["title"], "Signals")
        self.assertEqual(targets["groups"][0]["title"], "Group A")
        all_keys = [t["key"] for t in
                    targets["groups"] + targets["channels"]]
        self.assertNotIn(FOREIGN, all_keys)

    def test_preview_sends_nothing(self):
        from nomorals.social.chat.group_broadcast import preview_broadcast

        record = preview_broadcast(self.surface, "hello all", scope="all")
        self.assertFalse(record["live"])
        self.assertEqual(record["sent"], 0)
        self.assertEqual(len(record["targets"]), 3)
        self.assertEqual(self.gateway.broadcast_calls, [])
        self.assertEqual(self.gateway.sends, [])

    def test_live_run_fans_out_per_platform(self):
        from nomorals.social.chat.group_broadcast import run_owner_broadcast

        record = run_owner_broadcast(
            surface=self.surface, gateway=self.gateway,
            text="hello all", scope="all", pace_seconds=3.0)
        self.assertTrue(record["live"])
        self.assertEqual(record["sent"], 3)
        self.assertEqual(record["failed"], 0)
        self.assertFalse(record["aborted"])
        self.assertEqual(len(self.gateway.broadcast_calls), 1)
        platform, chats, text, kw = self.gateway.broadcast_calls[0]
        self.assertEqual(platform, "telegram-bot")
        self.assertEqual(text, "hello all")
        self.assertEqual(kw.get("pace_seconds"), 3.0)
        self.assertEqual(len(chats), 3)

    def test_live_run_isolates_a_dead_room(self):
        from nomorals.social.chat.group_broadcast import run_owner_broadcast

        self.gateway._fail = {GROUP_B}
        record = run_owner_broadcast(
            surface=self.surface, gateway=self.gateway,
            text="hello all", scope="all", pace_seconds=3.0)
        self.assertEqual(record["sent"], 2)
        self.assertEqual(record["failed"], 1)
        failed = [r for r in record["results"] if not r["ok"]]
        self.assertEqual(failed[0]["chat"], GROUP_B)
        self.assertIn("chat_write_forbidden", failed[0]["error"])

    def test_scope_groups_only_and_channels_only(self):
        from nomorals.social.chat.group_broadcast import run_owner_broadcast

        groups = run_owner_broadcast(
            surface=self.surface, gateway=self.gateway,
            text="g", scope="groups", pace_seconds=3.0)
        self.assertEqual(groups["sent"], 2)
        channels = run_owner_broadcast(
            surface=self.surface, gateway=self.gateway,
            text="c", scope="channels", pace_seconds=3.0)
        self.assertEqual(channels["sent"], 1)
        self.assertEqual(channels["targets"][0]["key"], CHANNEL)

    def test_pace_below_floor_is_refused(self):
        from nomorals.social.chat.group_broadcast import (
            BroadcastRefused,
            run_owner_broadcast,
        )

        with self.assertRaisesRegex(BroadcastRefused, "3s"):
            run_owner_broadcast(
                surface=self.surface, gateway=self.gateway,
                text="hello", scope="all", pace_seconds=1.0)
        self.assertEqual(self.gateway.broadcast_calls, [])

    def test_empty_text_is_refused(self):
        from nomorals.social.chat.group_broadcast import (
            BroadcastRefused,
            run_owner_broadcast,
        )

        with self.assertRaisesRegex(BroadcastRefused, "empty"):
            run_owner_broadcast(
                surface=self.surface, gateway=self.gateway,
                text="   ", scope="all", pace_seconds=3.0)


class TestReaddRecoveryOffer(_BroadcastCase):
    """The re-add recovery offer, on the organ alone."""

    def test_first_add_group_has_no_offer(self):
        from nomorals.social.chat.group_broadcast import readd_recovery_offer

        fresh = "telegram-bot:-100999"
        self.surface.store.group(fresh)["title"] = "Fresh"
        self.assertIsNone(readd_recovery_offer(
            self.surface, fresh, bot_id=BOT_ID))

    def test_readd_without_backup_says_so_plainly(self):
        from nomorals.social.chat.group_broadcast import readd_recovery_offer

        offer = readd_recovery_offer(self.surface, GROUP_A, bot_id=BOT_ID)
        self.assertIsNotNone(offer)
        self.assertFalse(offer["has_backup"])
        self.assertIn("no backup", offer["text"].lower())
        self.assertIn("/gbackup", offer["text"])
        kinds = [a.get("kind")
                 for a in self.surface.store.group(GROUP_A).get("audit") or []]
        self.assertIn("bot_readded", kinds)

    def test_readd_with_backup_names_it_and_the_restore(self):
        from nomorals.social.chat.group_backup import create_backup
        from nomorals.social.chat.group_broadcast import readd_recovery_offer

        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        offer = readd_recovery_offer(self.surface, GROUP_A, bot_id=BOT_ID)
        self.assertTrue(offer["has_backup"])
        self.assertEqual(offer["backup_id"], info["backup_id"])
        self.assertIn(info["backup_id"], offer["text"])
        self.assertIn(
            f"/gbackup restore {GROUP_A} {info['backup_id']}",
            offer["text"])


class TestOwnerBroadcastCommand(_BroadcastCase):
    """The /obroadcast surface through handle_control."""

    def test_command_is_registered_and_documented(self):
        self.assertIn("obroadcast", CONTROL_COMMANDS)
        self.assertIn("obroadcast", COMMAND_DETAILS)
        self.assertNotIn("obroadcast", PUBLIC_COMMANDS)
        page = COMMAND_DETAILS["obroadcast"]
        self.assertIn("usage", page)
        self.assertIn("live", page["usage"])

    def test_bare_command_shows_usage_and_targets(self):
        out = _text(self._reply("/obroadcast"))
        self.assertIn("/obroadcast", out)
        self.assertIn("Group A", out)

    def test_dry_run_lists_targets_and_sends_nothing(self):
        out = _text(self._reply("/obroadcast hello all"))
        self.assertIn("dry-run", out.lower())
        self.assertIn("Group A", out)
        self.assertIn("Signals", out)
        self.assertEqual(self.gateway.broadcast_calls, [])
        self.assertEqual(self.gateway.sends, [])

    def test_live_run_refused_until_armed(self):
        out = _text(self._reply("/obroadcast hello all | live"))
        self.assertIn("arm", out.lower())
        self.assertEqual(self.gateway.broadcast_calls, [])
        rows = self._journal_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["tool"], "obroadcast")
        self.assertFalse(rows[0]["live"])

    def test_arm_then_live_sends_and_journals(self):
        self.assertIn("ARMED", _text(self._reply("/obroadcast arm")))
        out = _text(self._reply("/obroadcast hello all | live"))
        self.assertIn("sent 3", out)
        self.assertEqual(len(self.gateway.broadcast_calls), 1)
        rows = self._journal_rows()
        live = [r for r in rows if r.get("live")]
        self.assertEqual(len(live), 1)
        self.assertEqual(live[0]["sent"], 3)

    def test_scope_segment_limits_targets(self):
        self._reply("/obroadcast arm")
        out = _text(self._reply("/obroadcast hi | channels | live"))
        self.assertIn("sent 1", out)
        self.assertEqual(len(self.gateway.broadcast_calls), 1)
        self.assertEqual(len(self.gateway.broadcast_calls[0][1]), 1)

    def test_pacing_floor_cannot_go_under(self):
        out = _text(self._reply("/obroadcast pacing 1"))
        self.assertIn("3s", out)
        out = _text(self._reply("/obroadcast pacing 5"))
        self.assertIn("5s", out)

    def test_status_and_runs(self):
        self._reply("/obroadcast arm")
        self._reply("/obroadcast hello | live")
        status = _text(self._reply("/obroadcast status"))
        self.assertIn("ARMED", status)
        self.assertIn("1 live run", status)
        runs = _text(self._reply("/obroadcast runs"))
        self.assertIn("hello", runs)

    def test_disarm_blocks_live_again(self):
        self._reply("/obroadcast arm")
        self._reply("/obroadcast disarm")
        out = _text(self._reply("/obroadcast hello | live"))
        self.assertIn("arm", out.lower())
        self.assertEqual(self.gateway.broadcast_calls, [])

    def _journal_rows(self):
        path = self.tmp_path / "data" / "social" / "mass" / "journal.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestBotReaddJoin(_BroadcastCase):
    """The join handler when the bot itself is (re-)added."""

    def test_readd_with_backup_dms_owner_the_offer(self):
        from nomorals.social.chat.group_backup import create_backup

        info = create_backup(self.surface, GROUP_A, by=OWNER_ID,
                             reason="manual")
        msg = _join_msg([{"id": BOT_ID, "username": "devonbot",
                          "name": "Devon", "is_bot": True}])
        self.assertTrue(self.rt._group_surface_turn(msg))
        dms = [s for s in self.gateway.sends
               if str(s[1].chat_id) == OWNER_ID]
        self.assertEqual(len(dms), 1)
        self.assertIn(info["backup_id"], dms[0][2])
        self.assertIn("/gbackup restore", dms[0][2])
        # The bot itself is never welcomed or ledger-joined.
        member = self.surface.member(GROUP_A, BOT_ID)
        self.assertIsNone(member)

    def test_readd_without_backup_still_tells_owner(self):
        msg = _join_msg([{"id": BOT_ID, "username": "devonbot",
                          "name": "Devon", "is_bot": True}])
        self.assertTrue(self.rt._group_surface_turn(msg))
        dms = [s for s in self.gateway.sends
               if str(s[1].chat_id) == OWNER_ID]
        self.assertEqual(len(dms), 1)
        self.assertIn("no backup", dms[0][2].lower())

    def test_first_add_bot_is_quiet(self):
        fresh = "telegram-bot:-100999"
        self.surface.store.group(fresh)["title"] = "Fresh"
        msg = _join_msg([{"id": BOT_ID, "username": "devonbot",
                          "name": "Devon", "is_bot": True}], key=fresh)
        self.assertTrue(self.rt._group_surface_turn(msg))
        self.assertEqual(self.gateway.sends, [])

    def test_ordinary_member_join_is_unaffected(self):
        msg = _join_msg([{"id": "7", "username": "cara",
                          "name": "Cara", "is_bot": False}])
        self.assertTrue(self.rt._group_surface_turn(msg))
        # The welcome goes to the group (correct); no DM to the owner.
        dms = [s for s in self.gateway.sends
               if str(s[1].chat_id) == OWNER_ID]
        self.assertEqual(dms, [])
        self.assertIsNotNone(self.surface.member(GROUP_A, "7"))


if __name__ == "__main__":
    unittest.main()
