"""Telegram Communities (DESIGN §16): the community-of-groups surface.

Failing-first pins: the Bot API 10.2/10.3 surface (Community {id, name}
via getChat; community_chat_added / _joined / _removed service
messages), the organ that finally writes the Devon-side community_id
tag the §14 boards read, and the /community command over both
platforms.  Nothing here touches the real home or a live adapter:
surfaces sit on temp stores, adapters are fakes, the runtime is the
bare-runtime shape from test_groups_command.py.
"""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind
from nomorals.social.chat.control import (
    CONTROL_COMMANDS,
    PUBLIC_COMMANDS,
    caller_may_run,
)
from nomorals.social.chat.group_community import (
    clear_community_tag,
    communities_index,
    community_of,
    describe_community,
    detect_community,
    note_community_added,
    note_community_joined,
    note_community_removed,
    parse_community,
    set_community_tag,
    sync_all_communities,
    sync_group_community,
)
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.storage.db import Database
from tests.frozen_clock import MIDDAY_TS

OWNER_ID = "5478650254"
ADMIN_ID = "9001"
STRANGER_ID = "9002"
GROUP_A = "telegram-bot:-100111"
GROUP_B = "telegram-bot:-100222"
CHANNEL = "telegram-bot:-100333"
WA_GROUP = "whatsapp:120363111111111111@g.us"
NOW = MIDDAY_TS
COMMUNITY = {"id": "777", "name": "Builders United"}
COMMUNITY_2 = {"id": "888", "name": "Second Circle"}


def _surface(tmp_path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "groups.json").load())


def _seed(tmp_path) -> GroupSurface:
    surface = _surface(tmp_path)
    surface.record_join(GROUP_A, "u1", username="ada", name="Ada",
                        ts=NOW)
    surface.record_join(GROUP_B, "u1", username="ada", name="Ada",
                        ts=NOW)
    surface.store.save()
    return surface


# ── the organ ─────────────────────────────────────────────────────

class OrganParseTests(unittest.TestCase):
    def test_parse_community(self):
        self.assertEqual(parse_community({"id": 777, "name": "X"}),
                         {"id": "777", "name": "X"})
        self.assertEqual(parse_community({"id": "777"}),
                         {"id": "777", "name": ""})
        self.assertIsNone(parse_community({"name": "No id"}))
        self.assertIsNone(parse_community({"id": ""}))
        self.assertIsNone(parse_community(None))
        self.assertIsNone(parse_community("777"))


class OrganTagTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            prefix="nm-test-tgcomm-")
        self.surface = _seed(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_added_sets_tag_and_name_and_source(self):
        out = note_community_added(self.surface, GROUP_A, COMMUNITY)
        self.assertTrue(out["ok"])
        self.assertTrue(out["changed"])
        self.assertEqual(community_of(self.surface, GROUP_A), {
            "id": "777", "name": "Builders United",
            "source": "platform"})
        audit = self.surface.audit(GROUP_A)
        self.assertTrue(any(row["kind"] == "community_added"
                            for row in audit))
        # A repeat add of the same community changes nothing.
        again = note_community_added(self.surface, GROUP_A, COMMUNITY)
        self.assertTrue(again["ok"])
        self.assertFalse(again["changed"])

    def test_added_never_clobbers_a_manual_tag(self):
        set_community_tag(self.surface, GROUP_A, "999", "Hand picked")
        out = note_community_added(self.surface, GROUP_A, COMMUNITY)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "manual tag kept")
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "999")
        audit = self.surface.audit(GROUP_A)
        self.assertTrue(any(row["kind"] == "community_added_conflict"
                            for row in audit))

    def test_added_with_no_payload_reports(self):
        out = note_community_added(self.surface, GROUP_A, None)
        self.assertFalse(out["ok"])
        self.assertIsNone(community_of(self.surface, GROUP_A))

    def test_removed_clears_platform_tag_only(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        out = note_community_removed(self.surface, GROUP_A)
        self.assertTrue(out["changed"])
        self.assertEqual(out["removed"]["id"], "777")
        self.assertIsNone(community_of(self.surface, GROUP_A))
        audit = self.surface.audit(GROUP_A)
        self.assertTrue(any(row["kind"] == "community_removed"
                            for row in audit))

    def test_removed_keeps_a_manual_tag(self):
        set_community_tag(self.surface, GROUP_A, "999", "Hand picked")
        out = note_community_removed(self.surface, GROUP_A)
        self.assertFalse(out["changed"])
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "999")

    def test_removed_with_no_tag_is_quiet(self):
        out = note_community_removed(self.surface, GROUP_A)
        self.assertTrue(out["ok"])
        self.assertFalse(out["changed"])

    def test_joined_audits_without_touching_the_tag(self):
        out = note_community_joined(self.surface, GROUP_A, COMMUNITY,
                                    user_id="42", username="zoe")
        self.assertTrue(out["ok"])
        self.assertIsNone(community_of(self.surface, GROUP_A))
        audit = self.surface.audit(GROUP_A)
        row = next(r for r in audit if r["kind"] == "community_join")
        self.assertEqual(row["community_id"], "777")
        self.assertEqual(row["user_id"], "42")

    def test_manual_link_and_unlink(self):
        out = set_community_tag(self.surface, GROUP_A, "999",
                                "Hand picked")
        self.assertTrue(out["ok"])
        self.assertEqual(community_of(self.surface, GROUP_A), {
            "id": "999", "name": "Hand picked", "source": "manual"})
        self.assertEqual(describe_community(self.surface, GROUP_A),
                         "community Hand picked (manual)")
        out = clear_community_tag(self.surface, GROUP_A)
        self.assertTrue(out["changed"])
        self.assertIsNone(community_of(self.surface, GROUP_A))
        self.assertEqual(describe_community(self.surface, GROUP_A),
                         "no community")
        # Unlinking an untagged group is quiet, not an error.
        again = clear_community_tag(self.surface, GROUP_A)
        self.assertTrue(again["ok"])
        self.assertFalse(again["changed"])

    def test_index_groups_by_community_across_platforms(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        set_community_tag(self.surface, GROUP_B, "777",
                          "Builders United")
        self.surface.record_join(WA_GROUP, "u1", username="ada",
                                 name="Ada", ts=NOW)
        note_community_added(self.surface, WA_GROUP, COMMUNITY)
        index = communities_index(self.surface)
        self.assertEqual(set(index), {"777"})
        row = index["777"]
        self.assertEqual(row["name"], "Builders United")
        self.assertEqual(row["platforms"], ["telegram-bot", "whatsapp"])
        self.assertEqual(row["groups"],
                         sorted([GROUP_A, GROUP_B, WA_GROUP]))


class FakeTGAdapter:
    """Bot API adapter shape: get_chat_community + admin_member.

    ``admins`` maps chat_id -> {user ids} who admin that chat.
    """

    def __init__(self, communities=None, admins=None):
        self.communities = communities or {}
        self.admins = admins or {}
        self.probed = []

    def get_chat_community(self, chat):
        self.probed.append(str(getattr(chat, "chat_id", chat)))
        return self.communities.get(
            str(getattr(chat, "chat_id", chat)),
            {"ok": True, "community": None})

    def admin_member(self, chat, user_id):
        chat_id = str(getattr(chat, "chat_id", chat))
        status = ("administrator"
                  if str(user_id) in self.admins.get(chat_id, set())
                  else "member")
        return {"ok": True, "result": {"status": status}}


class FakeWAAdapter:
    def __init__(self, linked=None, communities=None):
        self.linked = linked or {}
        self._communities = communities or []

    def group_info(self, chat):
        jid = str(getattr(chat, "chat_id", chat))
        return {"linked_parent": self.linked.get(jid, "")}

    def communities(self):
        return self._communities


class DetectTests(unittest.TestCase):
    def test_detect_telegram_bot_via_getchat(self):
        adapter = FakeTGAdapter({
            "-100111": {"ok": True,
                        "community": {"id": 777,
                                      "name": "Builders United"}}})
        found, error = detect_community(adapter, "telegram-bot",
                                        SimpleNamespace(chat_id="-100111"))
        self.assertEqual(error, "")
        self.assertEqual(found, {"id": "777", "name": "Builders United"})

    def test_detect_telegram_bot_failure_is_honest(self):
        adapter = FakeTGAdapter({
            "-100111": {"ok": False, "error": "Forbidden: bot is not "
                        "a member of the channel chat"}})
        found, error = detect_community(adapter, "telegram-bot",
                                        SimpleNamespace(chat_id="-100111"))
        self.assertIsNone(found)
        self.assertIn("Forbidden", error)

    def test_detect_whatsapp_via_linked_parent(self):
        adapter = FakeWAAdapter(
            linked={"120363111111111111@g.us": "120363999999999999@g.us"},
            communities=[{"id": "120363999999999999@g.us",
                          "subject": "The Family"}])
        found, error = detect_community(
            adapter, "whatsapp",
            SimpleNamespace(chat_id="120363111111111111@g.us"))
        self.assertEqual(error, "")
        self.assertEqual(found, {"id": "120363999999999999@g.us",
                                 "name": "The Family"})

    def test_detect_whatsapp_unlinked_group_is_none(self):
        adapter = FakeWAAdapter()
        found, error = detect_community(
            adapter, "whatsapp",
            SimpleNamespace(chat_id="120363111111111111@g.us"))
        self.assertIsNone(found)
        self.assertEqual(error, "")

    def test_detect_userbot_is_an_honest_limit(self):
        found, error = detect_community(FakeTGAdapter(), "telegram",
                                        SimpleNamespace(chat_id="-100111"))
        self.assertIsNone(found)
        self.assertIn("MTProto", error)

    def test_detect_without_adapter_is_an_error(self):
        found, error = detect_community(None, "telegram-bot",
                                        SimpleNamespace(chat_id="-100111"))
        self.assertIsNone(found)
        self.assertIn("no adapter", error)


class SyncTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            prefix="nm-test-tgcomm-")
        self.surface = _seed(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_sync_writes_platform_tag_on_change(self):
        adapter = FakeTGAdapter({
            "-100111": {"ok": True, "community": COMMUNITY}})
        out = sync_group_community(self.surface, GROUP_A, adapter)
        self.assertTrue(out["ok"])
        self.assertTrue(out["changed"])
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "777")
        # A second sync with the same community changes nothing.
        again = sync_group_community(self.surface, GROUP_A, adapter)
        self.assertFalse(again["changed"])

    def test_sync_never_overwrites_a_manual_tag(self):
        set_community_tag(self.surface, GROUP_A, "999", "Hand picked")
        adapter = FakeTGAdapter({
            "-100111": {"ok": True, "community": COMMUNITY}})
        out = sync_group_community(self.surface, GROUP_A, adapter)
        self.assertTrue(out["ok"])
        self.assertFalse(out["changed"])
        self.assertEqual(adapter.probed, [])
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "999")

    def test_sync_replaces_a_platform_tag_and_clears_when_gone(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        adapter = FakeTGAdapter({
            "-100111": {"ok": True, "community": COMMUNITY_2}})
        out = sync_group_community(self.surface, GROUP_A, adapter)
        self.assertTrue(out["changed"])
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "888")
        adapter.communities["-100111"] = {"ok": True,
                                          "community": None}
        out = sync_group_community(self.surface, GROUP_A, adapter)
        self.assertTrue(out["changed"])
        self.assertIsNone(community_of(self.surface, GROUP_A))

    def test_sync_all_survives_one_bad_group(self):
        adapter = FakeTGAdapter({
            "-100111": {"ok": True, "community": COMMUNITY},
            "-100222": {"ok": False, "error": "chat not found"}})
        reports = sync_all_communities(
            self.surface, {"telegram-bot": adapter})
        self.assertEqual(len(reports), 2)
        by_key = {r["key"]: r for r in reports}
        self.assertTrue(by_key[GROUP_A]["ok"])
        self.assertFalse(by_key[GROUP_B]["ok"])
        self.assertIn("chat not found", by_key[GROUP_B]["error"])
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "777")


# ── the Bot API adapter ───────────────────────────────────────────

class AdapterTests(unittest.TestCase):
    def _adapter(self):
        from nomorals.social.chat.telegram import TelegramBotAdapter

        adapter = object.__new__(TelegramBotAdapter)
        adapter.name = "telegram-bot"
        adapter.owner_only = False
        adapter.chat_allow = set()
        adapter._bot_username = "DevonBot"
        adapter._last_inbound_was_command = {}
        adapter._last_command = {}
        adapter.partner_context = ""
        adapter.media_dir = ""
        return adapter

    def _update(self, **msg_fields):
        base = {
            "message_id": 5, "date": int(NOW),
            "chat": {"id": -100111, "type": "supergroup",
                     "title": "Builders"},
            "from": {"id": 7, "first_name": "Ada", "username": "ada"},
        }
        base.update(msg_fields)
        return {"update_id": 1, "message": base}

    def test_convert_maps_community_chat_added(self):
        adapter = self._adapter()
        msg = adapter._convert(self._update(
            community_chat_added={
                "community": {"id": 777,
                              "name": "Builders United"}}))
        self.assertIsNotNone(msg)
        self.assertEqual(msg.meta.get("service"), "community_chat_added")
        self.assertEqual(msg.meta.get("community"),
                         {"id": "777", "name": "Builders United"})

    def test_convert_maps_community_chat_joined(self):
        adapter = self._adapter()
        msg = adapter._convert(self._update(
            community_chat_joined={
                "community": {"id": 888, "name": "Second Circle"}}))
        self.assertEqual(msg.meta.get("service"),
                         "community_chat_joined")
        self.assertEqual(msg.meta.get("community")["id"], "888")

    def test_convert_maps_community_chat_removed(self):
        adapter = self._adapter()
        msg = adapter._convert(self._update(
            community_chat_removed={}))
        self.assertEqual(msg.meta.get("service"),
                         "community_chat_removed")
        # Removed holds no information — no community payload rides it.
        self.assertNotIn("community", msg.meta)

    def test_get_chat_community_reads_getchat(self):
        adapter = self._adapter()
        adapter._api = lambda method, **kw: (
            {"id": -100111, "type": "supergroup",
             "community": {"id": 777, "name": "Builders United"}}
            if method == "getChat" else {})
        from nomorals.social.chat.base import ChatRef

        out = adapter.get_chat_community(ChatRef(
            platform="telegram-bot", chat_id="-100111"))
        self.assertTrue(out["ok"])
        self.assertEqual(out["community"],
                         {"id": "777", "name": "Builders United"})

    def test_get_chat_community_none_when_untagged(self):
        adapter = self._adapter()
        adapter._api = lambda method, **kw: {"id": -100111,
                                             "type": "supergroup"}
        from nomorals.social.chat.base import ChatRef

        out = adapter.get_chat_community(ChatRef(
            platform="telegram-bot", chat_id="-100111"))
        self.assertTrue(out["ok"])
        self.assertIsNone(out["community"])

    def test_get_chat_community_failure_is_honest(self):
        adapter = self._adapter()

        def _boom(method, **kw):
            raise RuntimeError("Forbidden: bot is not a member")

        adapter._api = _boom
        from nomorals.social.chat.base import ChatRef

        out = adapter.get_chat_community(ChatRef(
            platform="telegram-bot", chat_id="-100111"))
        self.assertFalse(out["ok"])
        self.assertIn("Forbidden", out["error"])


# ── the runtime service branch ────────────────────────────────────

class _Chat:
    platform = "telegram-bot"
    chat_id = "-100111"
    title = "Builders"
    key = GROUP_A
    kind = ChatKind.GROUP


class _Msg:
    def __init__(self, **kw):
        self.chat = _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")


class _RuntimeAdapter:
    def __init__(self):
        self.calls = []

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "member"}}


class _Gateway:
    def __init__(self, adapter):
        self.adapters = {"telegram-bot": adapter}


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter, surface):
        self.sent = []
        self.gateway = _Gateway(adapter)
        self.context = SimpleNamespace(
            settings=SimpleNamespace(workspace_dir=str(tmp_path)))
        self._group_surface_obj = surface

    def _send(self, chat, text, **kw):
        self.sent.append(text)

    def _is_operator(self, message):
        return False

    def turn(self, msg):
        return self._group_surface_turn(msg)


class RuntimeServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(
            prefix="nm-test-tgcomm-")
        self.surface = _seed(Path(self._tmp.name))
        self.rt = _Runtime(Path(self._tmp.name), _RuntimeAdapter(),
                           self.surface)

    def tearDown(self):
        self._tmp.cleanup()

    def test_added_service_message_tags_the_group_silently(self):
        msg = _Msg(meta={"service": "community_chat_added",
                         "community": {"id": "777",
                                       "name": "Builders United"}})
        self.assertTrue(self.rt.turn(msg))
        self.assertEqual(self.rt.sent, [])
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "777")

    def test_removed_service_message_clears_the_tag(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        msg = _Msg(meta={"service": "community_chat_removed"})
        self.assertTrue(self.rt.turn(msg))
        self.assertIsNone(community_of(self.surface, GROUP_A))

    def test_joined_service_message_audits_only(self):
        msg = _Msg(meta={"service": "community_chat_joined",
                         "community": {"id": "777",
                                       "name": "Builders United"}})
        self.assertTrue(self.rt.turn(msg))
        self.assertIsNone(community_of(self.surface, GROUP_A))
        audit = self.surface.audit(GROUP_A)
        self.assertTrue(any(row["kind"] == "community_join"
                            for row in audit))


# ── the /community command ────────────────────────────────────────

def _bare_runtime(community_dir: str) -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(
        db=Database(":memory:"),
        settings=SimpleNamespace(community_dir=community_dir),
    )
    rt._owner_chats = {OWNER_ID}
    rt._owner_sender_ids = {OWNER_ID}
    return rt


def _dm_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=f"telegram:{sender_id}", platform="telegram",
                           kind=ChatKind.DM, chat_id=sender_id)
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _group_msg(sender_id: str, text: str):
    chat = SimpleNamespace(key=GROUP_A, platform="telegram-bot",
                           kind=ChatKind.GROUP, chat_id="-100111")
    return SimpleNamespace(chat=chat, text=text, meta={}, sender="someone",
                           sender_id=sender_id)


def _text(out) -> str:
    return getattr(out, "text", out)


class CommandRegistrationTests(unittest.TestCase):
    def test_community_is_a_public_command(self):
        self.assertIn("community", CONTROL_COMMANDS)
        self.assertIn("community", PUBLIC_COMMANDS)

    def test_community_grant_admits_non_owner(self):
        self.assertTrue(caller_may_run("community", False))


class CommandHandleControlTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="nm-test-tgcomm-")
        self.rt = _bare_runtime(self._tmp.name)
        self.surface = _seed(Path(self._tmp.name))
        self.rt._group_surface_obj = self.surface

    def tearDown(self):
        self._tmp.cleanup()

    def _community(self, sender_id: str, text: str,
                   *, in_group: bool = False):
        message = (_group_msg(sender_id, text) if in_group
                   else _dm_msg(sender_id, text))
        out = self.rt.handle_control(
            text, message.chat.key, message=message)
        return _text(out)

    def _attach_adapters(self, tg=None, wa=None):
        adapters = {}
        if tg is not None:
            adapters["telegram-bot"] = tg
        if wa is not None:
            adapters["whatsapp"] = wa
        self.rt.gateway = SimpleNamespace(adapters=adapters)

    def test_owner_dm_empty_list_says_so_plainly(self):
        text = self._community(OWNER_ID, "/community")
        self.assertIn("no community carries", text)

    def test_owner_dm_list_shows_tagged_groups(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        text = self._community(OWNER_ID, "/community")
        self.assertIn("Builders United (777)", text)
        self.assertIn("1 group(s)", text)

    def test_owner_dm_show_names_member_groups(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        set_community_tag(self.surface, GROUP_B, "777",
                          "Builders United")
        text = self._community(OWNER_ID, "/community show 777")
        self.assertIn(GROUP_A, text)
        self.assertIn(GROUP_B, text)
        self.assertIn("claimed", text)

    def test_show_unknown_community_says_so(self):
        text = self._community(OWNER_ID, "/community show nope")
        self.assertIn("no community", text)

    def test_owner_dm_link_and_unlink(self):
        text = self._community(OWNER_ID, "/community link -100111 999 "
                               "Hand picked")
        self.assertIn("linked", text)
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "999")
        text = self._community(OWNER_ID, "/community unlink -100111")
        self.assertIn("unlinked", text)
        self.assertIsNone(community_of(self.surface, GROUP_A))

    def test_link_by_cold_gn_handle(self):
        text = self._community(OWNER_ID, "/community link g1 999")
        self.assertIn("linked", text)
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "999")

    def test_non_admin_dm_sees_no_communities(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        text = self._community(STRANGER_ID, "/community")
        self.assertIn("no community carries", text)

    def test_admin_dm_sees_only_their_groups(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        note_community_added(self.surface, GROUP_B, COMMUNITY_2)
        self._attach_adapters(tg=FakeTGAdapter(
            admins={"-100111": {ADMIN_ID}}))
        text = self._community(ADMIN_ID, "/community")
        self.assertIn("Builders United (777)", text)
        self.assertNotIn("Second Circle", text)

    def test_stranger_cannot_link_a_group(self):
        # Resolution is scoped to groups the sender admins (the
        # /gbackup pattern): a stranger's handle resolves to
        # nothing, so the refusal names the handle, not the group —
        # and, critically, no tag is written.
        text = self._community(STRANGER_ID, "/community link -100111 999")
        self.assertIn("that group handle expired", text)
        self.assertIsNone(community_of(self.surface, GROUP_A))

    def test_stale_session_token_still_checks_admin(self):
        # A session token seated when the sender WAS an admin
        # resolves the key — and the admin proof is re-checked at
        # use time, so the link is refused, not honored.
        self.rt._console_session_map()[f"telegram:{STRANGER_ID}"] = {
            "groups": {"g1": GROUP_A}, "channels": {},
            "at": time.time()}
        text = self._community(STRANGER_ID, "/community link g1 999")
        self.assertIn("don't admin", text)
        self.assertIsNone(community_of(self.surface, GROUP_A))

    def test_owner_dm_sync_one_group(self):
        self._attach_adapters(tg=FakeTGAdapter({
            "-100111": {"ok": True, "community": COMMUNITY}}))
        text = self._community(OWNER_ID, "/community sync g1")
        self.assertIn("Builders United", text)
        self.assertEqual(community_of(self.surface, GROUP_A)["id"],
                         "777")

    def test_owner_dm_sync_all_reports_counts(self):
        self._attach_adapters(tg=FakeTGAdapter({
            "-100111": {"ok": True, "community": COMMUNITY},
            "-100222": {"ok": False, "error": "chat not found"}}))
        text = self._community(OWNER_ID, "/community sync")
        self.assertIn("synced 2 claimed group(s)", text)
        self.assertIn("1 changed", text)
        self.assertIn("1 failed", text)

    def test_non_operator_cannot_sync_all(self):
        text = self._community(STRANGER_ID, "/community sync")
        self.assertIn("operator call", text)

    def test_in_group_bare_shows_only_this_rooms_line(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        text = self._community(OWNER_ID, "/community", in_group=True)
        self.assertIn("community Builders United (platform)", text)

    def test_in_group_management_is_redirected_to_dm(self):
        text = self._community(OWNER_ID, "/community link g1 999",
                               in_group=True)
        self.assertIn("lives in my DM", text)

    def test_groups_board_shows_the_community_line(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        out = self.rt.handle_control(
            "/groups", f"telegram:{OWNER_ID}",
            message=_dm_msg(OWNER_ID, "/groups"))
        self.assertIn("community Builders United (platform)",
                      _text(out))

    def test_console_menu_shows_the_community_line(self):
        note_community_added(self.surface, GROUP_A, COMMUNITY)
        # The console home first — it seats the g1 token in the
        # session map; a cold /console g1 falls back to home.
        self.rt.handle_control(
            "/console", f"telegram:{OWNER_ID}",
            message=_dm_msg(OWNER_ID, "/console"))
        out = self.rt.handle_control(
            "/console g1", f"telegram:{OWNER_ID}",
            message=_dm_msg(OWNER_ID, "/console g1"))
        self.assertIn("community Builders United (platform)",
                      _text(out))


if __name__ == "__main__":
    unittest.main()
