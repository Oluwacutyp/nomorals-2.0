"""WhatsApp moderation seam (census gap #1) — the group runtime
moderates through adapter verbs (admin_ban / admin_restrict /
admin_delete / admin_pin …) with the Telegram contract, and
WhatsAppAdapter implemented NONE of them: every ban/mute/delete/
pin in a WhatsApp group audited admin_action_failed.  Pins:

- the adapter verbs mapped onto the bridge's real affordances
  (ban = remove with the per-user status honored, unban vacuous
  BY DESIGN so kick stays a kick, delete/pin via the bridge
  message verbs, LID members resolvable through the roster);
- the honest gaps: NO per-member mute exists on WhatsApp, so
  admin_restrict refuses with its reason and never removes or
  group-locks anyone as a side effect; pin propagates the
  bridge's own answer (the Baileys engine's unsupported
  included); bare unpin names no message and refuses;
- _mod_run end-to-end on a WhatsApp group surface: a ban and a
  purge EXECUTE against the (fake) bridge and audit
  admin_action; a bridge refusal audits admin_action_failed and
  the member is told it didn't go through.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind, ChatRef
from nomorals.social.chat.group_surface import GroupStore, GroupSurface
from nomorals.social.chat.whatsapp import WhatsAppAdapter

GROUP_JID = "1203630123456@g.us"
KEY = f"whatsapp:{GROUP_JID}"
ADMIN_JID = "2347011111111@s.whatsapp.net"
TARGET_JID = "2347099999999@s.whatsapp.net"
LID_JID = "555444333@lid"
CHAT = ChatRef(platform="whatsapp", chat_id=GROUP_JID,
               kind=ChatKind.GROUP, title="Crew")

ROSTER = [
    {"jid": ADMIN_JID, "name": "Boss", "role": "admin"},
    {"jid": TARGET_JID, "name": "Spam", "role": ""},
    {"jid": LID_JID, "name": "Ghost", "role": ""},
]


class FakeBridge:
    """A recording _send_cmd router shaped like both bridges."""

    def __init__(self, *, member_status: str = "200",
                 message_ok: bool = True):
        self.calls: list[dict] = []
        self.member_status = member_status
        self.message_ok = message_ok

    def __call__(self, cmd, timeout: float = 25.0):
        self.calls.append(dict(cmd))
        name = cmd.get("cmd")
        if name == "group_participants":
            return {"ok": True, "participants": ROSTER}
        if name == "group_members_update":
            return {"ok": True, "action": cmd.get("action"),
                    "results": [{"jid": p,
                                 "status": self.member_status}
                                for p in cmd.get("participants")
                                or []]}
        if name in ("message_delete", "message_pin",
                    "message_unpin"):
            if self.message_ok:
                return {"ok": True}
            return {"ok": False,
                    "error": "unsupported: the bridge refused "
                             "the message verb"}
        return {"ok": False, "error": f"unknown cmd {name}"}

    def by_cmd(self, name: str) -> list[dict]:
        return [c for c in self.calls if c.get("cmd") == name]


def _adapter(bridge: FakeBridge) -> WhatsAppAdapter:
    adapter = WhatsAppAdapter()
    adapter.connected.set()
    adapter._send_cmd = bridge  # type: ignore[method-assign]
    return adapter


# ── adapter verb contracts ──────────────────────────────────────

def test_ban_is_a_checked_remove():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_ban(CHAT, TARGET_JID)
    assert out["ok"] is True
    updates = bridge.by_cmd("group_members_update")
    assert len(updates) == 1
    assert updates[0]["action"] == "remove"
    assert updates[0]["participants"] == [TARGET_JID]
    assert updates[0]["chat"] == GROUP_JID


def test_ban_honors_the_per_user_refusal():
    bridge = FakeBridge(member_status="403")
    out = _adapter(bridge).admin_ban(CHAT, TARGET_JID)
    assert out["ok"] is False
    assert "403" in out["error"]


def test_ban_resolves_a_bare_lid_member_via_the_roster():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_ban(CHAT, "555444333")
    assert out["ok"] is True
    updates = bridge.by_cmd("group_members_update")
    assert updates[0]["participants"] == [LID_JID]


def test_unban_is_vacuous_and_never_touches_the_bridge():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_unban(CHAT, TARGET_JID)
    assert out["ok"] is True
    assert bridge.calls == []  # a re-add here would undo every kick


def test_promote_and_demote_map_to_member_updates():
    bridge = FakeBridge()
    adapter = _adapter(bridge)
    assert adapter.admin_promote(CHAT, TARGET_JID)["ok"] is True
    assert adapter.admin_demote(CHAT, TARGET_JID)["ok"] is True
    actions = [c["action"]
               for c in bridge.by_cmd("group_members_update")]
    assert actions == ["promote", "demote"]


def test_restrict_is_an_honest_unsupported_and_applies_nothing():
    bridge = FakeBridge()
    adapter = _adapter(bridge)
    for out in (adapter.admin_restrict(CHAT, TARGET_JID,
                                       minutes=10),
                adapter.admin_restrict(CHAT, TARGET_JID, until=0),
                adapter.admin_unrestrict(CHAT, TARGET_JID)):
        assert out["ok"] is False
        assert "no per-member mute" in out["error"]
    assert bridge.calls == []  # no removal, no group lock, nothing


def test_delete_and_pin_ride_the_bridge_message_verbs():
    bridge = FakeBridge()
    adapter = _adapter(bridge)
    assert adapter.admin_delete(CHAT, "MSG42")["ok"] is True
    assert adapter.admin_pin(CHAT, "MSG42")["ok"] is True
    assert adapter.admin_unpin(CHAT, "MSG42")["ok"] is True
    sent = [(c["cmd"], c.get("message_id")) for c in bridge.calls
            if c["cmd"].startswith("message_")]
    assert sent == [("message_delete", "MSG42"),
                    ("message_pin", "MSG42"),
                    ("message_unpin", "MSG42")]


def test_pin_propagates_the_bridge_refusal_untouched():
    bridge = FakeBridge(message_ok=False)
    out = _adapter(bridge).admin_pin(CHAT, "MSG42")
    assert out["ok"] is False
    assert "unsupported" in out["error"]


def test_bare_unpin_refuses_without_a_bridge_call():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_unpin(CHAT)
    assert out["ok"] is False
    assert "specific message" in out["error"]
    assert bridge.calls == []


def test_admin_member_reads_the_roster():
    adapter = _adapter(FakeBridge())
    assert adapter.admin_member(CHAT, ADMIN_JID)["status"] == \
        "administrator"
    assert adapter.admin_member(CHAT, TARGET_JID)["status"] == \
        "member"
    assert adapter.admin_member(
        CHAT, "2347000000000@s.whatsapp.net")["ok"] is False


def test_disconnected_adapter_refuses_ban():
    adapter = WhatsAppAdapter()  # never connected
    out = adapter.admin_ban(CHAT, TARGET_JID)
    assert out["ok"] is False
    assert "not connected" in out["error"]


# ── _mod_run end-to-end on a WhatsApp group ─────────────────────

def _rig(tmp_path: Path, bridge: FakeBridge):
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    adapter = _adapter(bridge)
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore[method-assign]
    mixin._group_adapter = lambda platform: adapter  # type: ignore[method-assign]
    surface.handle_message(KEY, TARGET_JID, username="spammy",
                           text="buy my course")
    return mixin, surface


def _msg(*, reply_to="", reply_sender=None):
    meta = {}
    if reply_sender:
        meta["reply_to_sender"] = reply_sender
    return SimpleNamespace(
        chat=CHAT, sender_id=ADMIN_JID, sender="Boss",
        sender_username="", text="", meta=meta,
        reply_to=reply_to, message_id="900")


def test_mod_run_ban_executes_on_whatsapp(tmp_path):
    bridge = FakeBridge()
    mixin, surface = _rig(tmp_path, bridge)
    msg = _msg(reply_sender={"id": TARGET_JID, "name": "Spam"})
    out = mixin._control_ban("", msg)
    assert "banned" in out
    updates = bridge.by_cmd("group_members_update")
    assert updates and updates[0]["action"] == "remove"
    assert updates[0]["participants"] == [TARGET_JID]
    audit = surface.audit(KEY, limit=10)
    assert any(e["kind"] == "admin_action" and e["action"] == "ban"
               for e in audit)


def test_mod_run_kick_removes_without_readding(tmp_path):
    bridge = FakeBridge()
    mixin, surface = _rig(tmp_path, bridge)
    msg = _msg(reply_sender={"id": TARGET_JID, "name": "Spam"})
    out = mixin._control_kick("", msg)
    assert "is out" in out
    updates = bridge.by_cmd("group_members_update")
    assert [u["action"] for u in updates] == ["remove"]


def test_mod_run_purge_deletes_on_whatsapp(tmp_path):
    bridge = FakeBridge()
    mixin, surface = _rig(tmp_path, bridge)
    out = mixin._control_purge("", _msg(reply_to="MSG42"))
    assert "gone" in out
    deletes = bridge.by_cmd("message_delete")
    assert deletes and deletes[0]["message_id"] == "MSG42"
    assert deletes[0]["chat"] == GROUP_JID


def test_mod_run_bridge_refusal_audits_failure(tmp_path):
    bridge = FakeBridge(member_status="403")
    mixin, surface = _rig(tmp_path, bridge)
    msg = _msg(reply_sender={"id": TARGET_JID, "name": "Spam"})
    out = mixin._control_ban("", msg)
    assert "didn't go through" in out
    audit = surface.audit(KEY, limit=10)
    assert any(e["kind"] == "admin_action_failed"
               and e["action"] == "ban" for e in audit)


def test_mod_run_mute_is_refused_honestly_never_a_removal(tmp_path):
    bridge = FakeBridge()
    mixin, surface = _rig(tmp_path, bridge)
    msg = _msg(reply_sender={"id": TARGET_JID, "name": "Spam"})
    out = mixin._control_mute("10m", msg)
    assert "didn't go through" in out
    assert bridge.by_cmd("group_members_update") == []
    audit = surface.audit(KEY, limit=10)
    assert any(e["kind"] == "admin_action_failed"
               and e["action"] == "mute" for e in audit)
