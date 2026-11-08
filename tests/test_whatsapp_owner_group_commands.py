"""WhatsApp census gap #3 — the owner commands from inside a
WhatsApp group, plus the gsettings mute fold-in.

The gap: on the Telegram userbot the owner IS the operator in
groups — his commands dispatch. On WhatsApp his own messages
were persist-only outside the self-chat (the adapter's
own_message discipline), so he could not run a single command
from inside a WhatsApp group — not even designating the room
he was standing in.

The seam, end to end:
- the adapter promotes one of the paired account's OWN group
  messages to an owner turn (incoming, from_owner,
  owner_command) iff it parses as a registered command — the
  runtime's own parse_control verdict. The bridge's own_message
  event is the identity proof: it exists only for the paired
  account's own messages, the same provenance the self-chat
  carve-out already trusts (phone JID + LID identities alike).
  Devon's own send echoes, non-command text and unknown
  slashes are never promoted — they stay persist-only as-him
  conversation, exactly as before.
- the gateway's inbound owner test and the runtime's
  _is_operator honor the owner_command stamp exactly like the
  self_chat stamp, so the promoted turn reaches the SAME
  control dispatch (claimed-once is skipped for the operator;
  the reply is a solicited control reply, never §33 system
  output; the brain gate is untouched — no conversation path
  opens, because non-commands never become turns).
- a stale group command (typed while the surface was down) is
  stamped stale_replay, like the self-chat: observed at
  reconnect, never executed.

Fold-in (the wamenu remainder): /gsettings `filter action
mute` and `mute default <length>` cannot be newly selected
for a WhatsApp group — the same platform-truth rule as the
warn ladder (WhatsApp has no per-person mute to fire them).
Clearing (off) still works; values already stored keep their
data, the settings surfaces state the truth about them, and
their execution keeps refusing honestly. Telegram is
untouched.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef
from nomorals.social.chat.whatsapp import WhatsAppAdapter

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_JID = "2348012345678@s.whatsapp.net"
OWNER_LID = "105382746195847@lid"
GROUP_JID = "1203630123456@g.us"
TARGET_JID = "2347099999999@s.whatsapp.net"


def _now_ms() -> int:
    return int(time.time() * 1000)


# ── adapter: the promotion seam ──────────────────────────────

class OwnerGroupCommandAdapterTests(unittest.TestCase):
    """Bridge own_message events in a group, fed through the
    adapter's real dispatch (the SelfChatLidTests pattern)."""

    def _adapter(self, with_lid=False):
        adapter = WhatsAppAdapter()
        msgs: list = []
        status = {"type": "status", "state": "open",
                  "user": OWNER_JID}
        if with_lid:
            status["lid"] = OWNER_LID
        adapter._dispatch(status, msgs.append)
        return adapter, msgs

    @staticmethod
    def _own(text, *, kind="group", chat_id=GROUP_JID, mid="m1",
             ts=None, reply_to=""):
        return {"type": "own_message", "id": mid,
                "chat": {"id": chat_id, "kind": kind,
                         "title": "Crew"},
                "text": text,
                "ts": _now_ms() if ts is None else ts,
                "reply_to": reply_to}

    def test_owner_group_command_becomes_an_owner_turn(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(self._own("/gsettings"), msgs.append)
        self.assertEqual(len(msgs), 1)
        msg = msgs[0]
        self.assertTrue(msg.incoming)
        self.assertEqual(msg.chat.kind, ChatKind.GROUP)
        self.assertEqual(msg.chat.chat_id, GROUP_JID)
        self.assertEqual(msg.meta.get("from_owner"), True)
        self.assertEqual(msg.meta.get("owner_command"), True)
        self.assertNotIn("own_message", msg.meta)
        self.assertNotIn("self_chat", msg.meta)
        self.assertNotIn("stale_replay", msg.meta)
        # the actor identity is the paired account's own JID,
        # the self-chat turn's construction
        self.assertEqual(msg.sender_id, OWNER_JID)

    def test_owner_group_command_recognized_with_lid_known(self):
        """LID-migrated account: the bridge reports the account
        under its LID as well as its phone JID. The promotion
        keys off the own_message provenance (the paired
        account's own send), so a group command is the owner's
        with either identity known — no second identity check
        is invented for groups."""
        adapter, msgs = self._adapter(with_lid=True)
        self.assertEqual(adapter.user_lid, OWNER_LID)
        adapter._dispatch(self._own("/help"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("owner_command"), True)

    def test_owner_group_plain_text_stays_persist_only(self):
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._own("gm everyone, big day today"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)
        self.assertNotIn("owner_command", msgs[0].meta)

    def test_owner_group_unknown_slash_stays_persist_only(self):
        """An unregistered slash is not a Devon command — it
        must not become a turn (it would fall through to the
        persona and open a conversation path in the group)."""
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._own("/definitelynotacommand"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)

    def test_own_send_echo_in_group_stays_persist_only(self):
        """Devon's own sends echo back as own_message carrying
        the id her send was assigned — even with command-shaped
        text the echo is never promoted, or she would execute
        her own output."""
        adapter, msgs = self._adapter()
        adapter.connected.set()
        adapter._send_cmd = lambda cmd, timeout=25.0: {
            "id": cmd.get("id"), "ok": True, "message_id": "ECHO-G"}
        chat = ChatRef(platform="whatsapp", chat_id=GROUP_JID,
                       kind=ChatKind.GROUP, peer=GROUP_JID)
        result = adapter.send(chat, "/status")
        self.assertTrue(result.ok)
        adapter._dispatch(
            self._own("/status", mid="ECHO-G"), msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertFalse(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("own_message"), True)

    def test_stale_owner_group_command_is_observed_not_executed(self):
        """Typed while the surface was down, delivered at
        reconnect: promoted (so the group surface observes it
        as his) but stamped stale_replay — the runtime holds
        stale replays from executing, the boot-flood law."""
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._own("/gsettings system output on",
                      ts=1_700_000_000_000),
            msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("owner_command"), True)
        self.assertEqual(msgs[0].meta.get("stale_replay"), True)

    def test_owner_dm_command_outside_self_chat_promotes(self):
        """2026-11-08 truth fix: the promotion covers 1:1 DMs
        too.  The groups-only shape was the /viewonce silence:
        the owner's commands typed as-him inside a contact's
        DM (the natural place to answer a view-once photo)
        fell to the persist-only fallback and died without a
        word — EVERY chatops command, not just /viewonce.
        A registered command in a DM now promotes exactly
        like a group one; non-command DM text still stays
        persist-only (see test_truthfix_personal for the
        full DM matrix)."""
        adapter, msgs = self._adapter()
        adapter._dispatch(
            self._own("/status", kind="dm",
                      chat_id="2347099999999@s.whatsapp.net"),
            msgs.append)
        self.assertEqual(len(msgs), 1)
        self.assertTrue(msgs[0].incoming)
        self.assertEqual(msgs[0].meta.get("from_owner"), True)
        self.assertEqual(msgs[0].meta.get("owner_command"), True)
        self.assertNotIn("own_message", msgs[0].meta)


# ── pipeline: the promoted turn reaches control dispatch ────

class _FakeBridge:
    """Recording bridge router (the wamod FakeBridge shape)."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, cmd, timeout: float = 25.0):
        self.calls.append(dict(cmd))
        name = cmd.get("cmd")
        if name == "group_participants":
            return {"ok": True, "participants": [
                {"jid": OWNER_JID, "name": "Boss",
                 "role": "admin"},
                {"jid": TARGET_JID, "name": "Spam", "role": ""}]}
        if name == "group_members_update":
            return {"ok": True, "action": cmd.get("action"),
                    "results": [{"jid": p, "status": "200"}
                                for p in cmd.get("participants")
                                or []]}
        return {"ok": False, "error": f"unknown cmd {name}"}

    def by_cmd(self, name: str) -> list[dict]:
        return [c for c in self.calls if c.get("cmd") == name]


@pytest.fixture()
def rig():
    from nomorals.agents.context import build_context
    from nomorals.agents.partner_runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="nm-wacommand-", dir=_TEST_TMPDIR)
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)

    class _Router:
        def chat(self, *a, **k):
            raise AssertionError(
                "the brain must not be called: the command "
                "path owns this message")

    context.router = _Router()

    sends: list[str] = []
    bridge = _FakeBridge()
    wa = WhatsAppAdapter()
    wa.connected.set()
    wa._send_cmd = bridge  # type: ignore[method-assign]

    class _Gateway:
        dry_run = False

        def __init__(self):
            self.adapters = {"whatsapp": wa}

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
    yield runtime, sends, bridge
    try:
        runtime.stop()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass
    try:
        context.close()
    except Exception:  # noqa: BLE001 - best-effort teardown
        pass


def _owner_turn(text, *, reply_to="", chat_id=GROUP_JID):
    """The turn exactly as the adapter now delivers it."""
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=chat_id,
                     kind=ChatKind.GROUP, title="Crew"),
        incoming=True, text=text, sender="", sender_id=OWNER_JID,
        message_id="own1", reply_to=reply_to, ts=time.time(),
        meta={"from_owner": True, "owner_command": True})


def _member_msg(text, *, chat_id=GROUP_JID):
    return ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=chat_id,
                     kind=ChatKind.GROUP, title="Crew"),
        incoming=True, text=text, sender="member",
        sender_id=TARGET_JID, message_id="mem1", ts=time.time(),
        meta={})


def test_owner_help_in_whatsapp_group_answers_once(rig):
    runtime, sends, _bridge = rig
    runtime._process_inner(_owner_turn("/help"))
    assert len(sends) == 1  # claimed once: one reply, no brain line
    assert "what I do here" in sends[0]  # GROUP help, not the DM menu


def test_owner_ban_in_whatsapp_group_executes(rig):
    runtime, sends, bridge = rig
    key = f"whatsapp:{GROUP_JID}"
    # the target is on the group's ledger with a recorded
    # message — the resolver's reply path lands on it
    runtime._group_surface().handle_message(
        key, TARGET_JID, username="spammy", name="Spam",
        text="buy my course", message_id="T1")
    runtime._process_inner(_owner_turn("/ban", reply_to="T1"))
    updates = bridge.by_cmd("group_members_update")
    assert len(updates) == 1
    assert updates[0]["action"] == "remove"
    assert updates[0]["participants"] == [TARGET_JID]
    assert len(sends) == 1
    assert "banned" in sends[0]
    audit = runtime._group_surface().audit(key, limit=20)
    assert any(e["kind"] == "admin_action"
               and e["action"] == "ban" for e in audit)


def test_owner_noncommand_group_text_is_never_answered(rig):
    """The persist-only shape the adapter delivers for his
    plain group text: observed, never answered, never the
    brain's (the rig's router raises if the brain is called)."""
    runtime, sends, _bridge = rig
    msg = ChatMessage(
        chat=ChatRef(platform="whatsapp", chat_id=GROUP_JID,
                     kind=ChatKind.GROUP, title="Crew"),
        incoming=False, text="gm everyone", sender="",
        sender_id="", message_id="own2", ts=time.time(),
        meta={"own_message": True, "from_owner": True})
    runtime._process_inner(msg)
    assert sends == []


def test_stranger_owner_only_command_in_unclaimed_group_redirects(rig):
    """§33/claimed-once behavior for everyone else is
    unchanged: a member's owner-only kind gets the one honest
    redirect, and nothing executes."""
    runtime, sends, bridge = rig
    runtime._process_inner(_member_msg("/status"))
    assert len(sends) == 1
    assert "owner-only" in sends[0]
    # no moderation verb executed (a roster read by the group
    # surface while observing the room is not an execution)
    assert bridge.by_cmd("group_members_update") == []


def test_stranger_public_command_in_unclaimed_group_still_runs(rig):
    runtime, sends, _bridge = rig
    runtime._process_inner(_member_msg("/ping"))
    assert len(sends) == 1
    assert sends[0].strip()


# ── fold-in: gsettings / console mute selection ─────────────

WA_KEY = f"whatsapp:{GROUP_JID}"
TG_KEY = "telegram:-1001"
FILTER_MUTE_REFUSAL = "no per-person mute for the filter to fire"
MUTE_DEFAULT_REFUSAL = "no per-person mute, so a standing mute length"


def _group_rig(tmp_path, key, platform):
    from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
    from nomorals.social.chat.group_surface import GroupStore, GroupSurface

    class _Gateway:
        def __init__(self):
            self.sent: list = []

        def send(self, platform, chat, text, **kw):
            self.sent.append({"platform": platform, "text": text,
                              **kw})
            return {"ok": True}

    class _Adapter:
        def admin_member(self, chat, user_id):
            return {"status": "administrator"}

    class _RT(RuntimeGroupMixin):
        operator_ids = frozenset()

        def _group_adapter(self, platform):  # type: ignore[no-untyped-def]
            return _Adapter()

    rt = _RT()
    rt.gateway = _Gateway()
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    rt._group_surface_obj = surface
    msg = SimpleNamespace(
        chat=SimpleNamespace(key=key, platform=platform,
                             kind="group", title="Crew"),
        sender_id="m1", sender="Member", sender_username="",
        sender_role="", text="", meta={}, reply_to="",
        message_id="1")
    return rt, surface, msg


def test_gsettings_filter_action_mute_refused_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._control_gsettings("filter action mute", msg)
    assert FILTER_MUTE_REFUSAL in out
    stored = surface.store.settings(WA_KEY)["moderation"]
    assert stored["filter_action"] != "mute"  # nothing was set


def test_gsettings_filter_action_delete_and_warn_fine_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._control_gsettings("filter action warn", msg)
    assert out == "filter action: warn."
    stored = surface.store.settings(WA_KEY)["moderation"]
    assert stored["filter_action"] == "warn"


def test_gsettings_filter_action_mute_fine_on_telegram(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, TG_KEY, "telegram")
    out = rt._control_gsettings("filter action mute", msg)
    assert out == "filter action: mute."
    stored = surface.store.settings(TG_KEY)["moderation"]
    assert stored["filter_action"] == "mute"


def test_gsettings_mute_default_refused_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._control_gsettings("mute default 2h", msg)
    assert MUTE_DEFAULT_REFUSAL in out
    stored = surface.store.settings(WA_KEY)
    assert int(stored.get("mute_default_s") or 0) == 0


def test_gsettings_mute_default_off_still_clears_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    surface.update_settings(WA_KEY, mute_default_s=7200)
    out = rt._control_gsettings("mute default off", msg)
    assert "no standing mute length" in out
    stored = surface.store.settings(WA_KEY)
    assert int(stored.get("mute_default_s") or 0) == 0


def test_gsettings_mute_default_fine_on_telegram(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, TG_KEY, "telegram")
    out = rt._control_gsettings("mute default 2h", msg)
    assert "standing mute length" in out
    stored = surface.store.settings(TG_KEY)
    assert int(stored.get("mute_default_s") or 0) == 7200


def test_console_filter_action_mute_refused_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._console_apply(surface, WA_KEY, msg, "m1", "defense",
                            ["filter", "action", "mute"])
    assert FILTER_MUTE_REFUSAL in out
    stored = surface.store.settings(WA_KEY)["moderation"]
    assert stored["filter_action"] != "mute"


def test_console_mute_default_refused_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._console_apply(surface, WA_KEY, msg, "m1", "defense",
                            ["mute", "default", "2h"])
    assert MUTE_DEFAULT_REFUSAL in out
    stored = surface.store.settings(WA_KEY)
    assert int(stored.get("mute_default_s") or 0) == 0


def test_console_mute_grammar_fine_on_telegram(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, TG_KEY, "telegram")
    out = rt._console_apply(surface, TG_KEY, msg, "m1", "defense",
                            ["filter", "action", "mute"])
    assert out == "filter action: mute."
    out = rt._console_apply(surface, TG_KEY, msg, "m1", "defense",
                            ["mute", "default", "2h"])
    assert "standing mute length" in out


def test_gsettings_card_offers_no_mute_on_whatsapp(tmp_path):
    rt, _surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._control_gsettings("", msg)
    assert "filter action delete|warn|mute" not in out.text
    assert "mute default 2h|off" not in out.text
    assert "no mute action on WhatsApp" in out.text
    assert "no standing mute length on WhatsApp" in out.text


def test_gsettings_card_offers_unchanged_on_telegram(tmp_path):
    rt, _surface, msg = _group_rig(tmp_path, TG_KEY, "telegram")
    out = rt._control_gsettings("", msg)
    assert "filter action delete|warn|mute" in out.text
    assert "mute default 2h|off" in out.text


def test_console_card_offers_no_mute_on_whatsapp(tmp_path):
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    out = rt._console_section(
        surface, WA_KEY, msg, "tok", "defense")
    assert "filter action delete|warn|mute" not in out.text
    assert "mute default 2h|off" not in out.text
    assert "no mute action on WhatsApp" in out.text


def test_stored_mute_settings_show_the_truth_and_keep_data(tmp_path):
    """Stored before the rule (or on another surface): the
    data is NOT rewritten, and both settings surfaces state
    plainly that the values cannot fire on WhatsApp."""
    rt, surface, msg = _group_rig(tmp_path, WA_KEY, "whatsapp")
    surface.update_settings(WA_KEY, mute_default_s=7200)
    mod = dict(surface.store.settings(WA_KEY)["moderation"])
    mod["filter_action"] = "mute"
    surface.update_settings(WA_KEY, moderation=mod)
    out = rt._control_gsettings("", msg)
    assert "mute settings are stored for this group" in out.text
    assert "no per-person mute" in out.text
    out2 = rt._console_section(surface, WA_KEY, msg, "tok",
                               "defense")
    assert "stored mute settings can't fire here" in out2.text
    # the stored data itself is untouched by being displayed
    stored = surface.store.settings(WA_KEY)
    assert int(stored.get("mute_default_s") or 0) == 7200
    assert stored["moderation"]["filter_action"] == "mute"
