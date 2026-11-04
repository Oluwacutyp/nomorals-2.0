"""WhatsApp join-request queue (DESIGN §28, census gap #5).

Telegram delivers join requests as events; WhatsApp holds them
server-side, so §28 had no WhatsApp limb at all: the bridges
exposed only the approval-mode toggle, the adapter had no verb,
and pending applicants were invisible and undecidable.  Pins:

- the bridge verbs' adapter contracts (list parses rows and
  refuses honestly; decide maps approve/reject and honors the
  requester's OWN per-user status, never the batch ok);
- discovery on the group cadence: a claimed request-mode
  WhatsApp group polls its door (throttled), and every new
  requester lands in the SAME §28 queue Telegram uses —
  deduped by (requester, requested_at), closed as "gone" when
  WhatsApp stops listing them, untouched when the bridge
  can't answer;
- the shared verdicts perform: clean auto-approves + welcome,
  a banned-driven decline declines for real;
- WhatsApp's platform truth: no restricted-admit state
  exists, so a restricted verdict QUEUES with the reason said
  out loud, a decline resting only on criteria WhatsApp can't
  report (usernames don't exist; an unknown name can't be
  judged) queues instead of refusing — and an admin approving
  a held request is told the member is in FULLY;
- the gates: approval-mode off at the door, open mode,
  unclaimed room, adapter without the verb — no polling, no
  errors, no state.
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
REQ_JID = "2347099999999@s.whatsapp.net"
REQ2_JID = "2347088888888@s.whatsapp.net"
REQ_TS = 1_800_000_000.0
CHAT = ChatRef(platform="whatsapp", chat_id=GROUP_JID,
               kind=ChatKind.GROUP, title="Crew")


def _criteria(**on):
    crit = {name: False for name in (
        "username", "photo", "firstname", "lastname", "hygiene",
        "banned", "bots")}
    crit.update(on)
    return crit


# ── adapter contracts over a fake bridge ──────────────────────


class FakeBridge:
    """A recording _send_cmd router shaped like both bridges."""

    def __init__(self, *, requests=None, decide_status="200",
                 known_cmds=True):
        self.calls: list[dict] = []
        self.requests = list(requests or [])
        self.decide_status = decide_status
        self.known_cmds = known_cmds

    def __call__(self, cmd, timeout: float = 25.0):
        self.calls.append(dict(cmd))
        name = cmd.get("cmd")
        if name == "group_participants":
            return {"ok": True, "participants": []}
        if not self.known_cmds:
            return {"ok": False, "error": f"unknown cmd {name}"}
        if name == "group_join_requests":
            return {"ok": True, "requests": [
                dict(r) for r in self.requests]}
        if name == "group_join_request_decide":
            return {"ok": True, "action": cmd.get("action"),
                    "results": [{"jid": p,
                                 "status": self.decide_status}
                                for p in cmd.get("participants")
                                or []]}
        return {"ok": False, "error": f"unknown cmd {name}"}

    def by_cmd(self, name: str) -> list[dict]:
        return [c for c in self.calls if c.get("cmd") == name]


def _adapter(bridge: FakeBridge) -> WhatsAppAdapter:
    adapter = WhatsAppAdapter()
    adapter.connected.set()
    adapter._send_cmd = bridge  # type: ignore[method-assign]
    return adapter


def test_list_parses_bridge_rows():
    bridge = FakeBridge(requests=[
        {"jid": REQ_JID, "name": "Ngozi", "requested_at": REQ_TS},
        {"jid": REQ2_JID, "name": "", "requested_at": 0}])
    out = _adapter(bridge).admin_join_requests(CHAT)
    assert out["ok"] is True
    assert out["requests"] == [
        {"jid": REQ_JID, "name": "Ngozi", "requested_at": REQ_TS},
        {"jid": REQ2_JID, "name": "", "requested_at": 0.0}]
    assert bridge.by_cmd("group_join_requests")[0]["chat"] \
        == GROUP_JID


def test_list_empty_when_nobody_waits():
    out = _adapter(FakeBridge()).admin_join_requests(CHAT)
    assert out == {"ok": True, "requests": []}


def test_list_refusal_is_honest_never_empty():
    # An older bridge without the verb: the answer is a named
    # refusal — the runtime must not read it as "door is empty".
    out = _adapter(FakeBridge(known_cmds=False)
                   ).admin_join_requests(CHAT)
    assert out["ok"] is False
    assert "unknown cmd" in out["error"]


def test_list_disconnected_refuses():
    adapter = WhatsAppAdapter()  # never connected
    out = adapter.admin_join_requests(CHAT)
    assert out["ok"] is False


def test_decide_approve_calls_the_bridge_verb():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_join_request(CHAT, REQ_JID, True)
    assert out["ok"] is True
    call = bridge.by_cmd("group_join_request_decide")[0]
    assert call["action"] == "approve"
    assert call["participants"] == [REQ_JID]
    assert call["chat"] == GROUP_JID


def test_decide_decline_maps_to_reject():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_join_request(CHAT, REQ_JID, False)
    assert out["ok"] is True
    assert bridge.by_cmd(
        "group_join_request_decide")[0]["action"] == "reject"


def test_decide_honors_the_requesters_own_status():
    bridge = FakeBridge(decide_status="403")
    out = _adapter(bridge).admin_join_request(CHAT, REQ_JID, True)
    assert out["ok"] is False
    assert "403" in out["error"]


def test_decide_bare_number_resolves_to_the_phone_server():
    bridge = FakeBridge()
    out = _adapter(bridge).admin_join_request(
        CHAT, "2347099999999", True)
    assert out["ok"] is True
    assert bridge.by_cmd("group_join_request_decide")[0][
        "participants"] == ["2347099999999@s.whatsapp.net"]


# ── the discovery drive over a fake adapter ───────────────────


class _WaAdapter:
    """The runtime-side fake: door list + decide + door state."""

    name = "whatsapp"

    def __init__(self, *, requests=None, approval=True,
                 list_ok=True):
        self.calls: list = []
        self.requests = list(requests or [])
        self.approval = approval
        self.list_ok = list_ok

    def group_info(self, chat):
        return {"join_approval_mode": self.approval}

    def admin_join_requests(self, chat):
        self.calls.append(("list",))
        if not self.list_ok:
            return {"ok": False,
                    "error": "unsupported: bridge has no verb"}
        return {"ok": True,
                "requests": [dict(r) for r in self.requests]}

    def admin_join_request(self, chat, user_id, approve):
        self.calls.append(("decide", user_id, approve))
        return {"ok": True}

    def admin_member(self, chat, user_id):
        return {"ok": True, "status": "administrator"}

    def decides(self):
        return [c for c in self.calls if c[0] == "decide"]


def _req(jid=REQ_JID, name="Ngozi", ts=REQ_TS):
    return {"jid": jid, "name": name, "requested_at": ts}


def _surface(tmp_path: Path, *, designated=True, **acceptance):
    surface = GroupSurface(GroupStore(tmp_path / "g.json").load())
    if designated:
        surface.set_designated(KEY, True)
    base: dict = {"mode": "request"}
    base.update(acceptance)
    surface.update_settings(KEY, acceptance=base)
    return surface


def _rig(tmp_path: Path, *, adapter=None, designated=True,
         **acceptance):
    surface = _surface(tmp_path, designated=designated,
                       **acceptance)
    adapter = adapter if adapter is not None else _WaAdapter()
    sent: list = []
    mixin = RuntimeGroupMixin()
    mixin._group_surface = lambda: surface  # type: ignore
    mixin._group_adapter = (  # type: ignore
        lambda platform: {"whatsapp": adapter}.get(platform))
    mixin.gateway = SimpleNamespace(  # type: ignore
        send=lambda *a, **k: sent.append((a, k)),
        adapters={"whatsapp": adapter})
    return mixin, surface, adapter, sent


def _msg(sender_id=ADMIN_JID):
    return SimpleNamespace(
        chat=SimpleNamespace(key=KEY, platform="whatsapp",
                             kind="group", title="Crew",
                             chat_id=GROUP_JID),
        sender_id=sender_id, sender="Boss", sender_username="",
        text="hello", meta={}, reply_to="", message_id="1")


def _texts(sent):
    return [str(a[2]) for a, _k in sent]


def _unstamp(surface):
    entry = surface.store.group(KEY)
    surface._acceptance_state(entry)["wa_req_poll_ts"] = 0.0


def _audit_kinds(surface, kind):
    return [a for a in surface.audit(KEY, limit=500)
            if a.get("kind") == kind]


def test_clean_request_auto_approves_and_welcomes(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.decides() == [("decide", REQ_JID, True)]
    assert surface.acceptance_queue(KEY) == []
    assert surface.member(KEY, REQ_JID) is not None
    assert any("Ngozi" in t for t in _texts(sent))  # the welcome
    executed = [a for a in _audit_kinds(
        surface, "moderation_executed")
        if a.get("action") == "join_request_approve"]
    assert len(executed) == 1


def test_queued_request_notes_and_decides_via_joinq(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.decides() == []
    queue = surface.acceptance_queue(KEY)
    assert len(queue) == 1
    record = queue[0]
    assert record["source"] == "wa_poll"
    assert record["user_id"] == REQ_JID
    texts = _texts(sent)
    assert any("wants in" in t for t in texts)  # the queue note
    buttons = [k.get("buttons") for _a, k in sent
               if k.get("buttons")]
    flat = [pair for rows in buttons for row in rows
            for pair in row]
    assert any(data == f"joinq approve {record['id']}"
               for _label, data in flat)
    # The admin decides in the SAME /joinq surface Telegram uses.
    reply = mixin._control_joinq(
        f"approve {record['id']}", _msg())
    assert "approved — they're in" in reply
    assert adapter.decides() == [("decide", REQ_JID, True)]
    assert surface.acceptance_queue(KEY) == []


def test_joinq_decline_rejects_at_the_door(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    rid = surface.acceptance_queue(KEY)[0]["id"]
    reply = mixin._control_joinq(f"decline {rid}", _msg())
    assert "declined" in reply
    assert adapter.decides() == [("decide", REQ_JID, False)]


def test_restricted_verdict_queues_with_platform_truth(tmp_path):
    # firstname required, the bridge knows no name: the rule's
    # verdict is restricted — WhatsApp has no restricted
    # admission, so the request queues instead of half-landing.
    adapter = _WaAdapter(requests=[_req(name="")])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter,
        criteria=_criteria(firstname=True))
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.decides() == []
    queue = surface.acceptance_queue(KEY)
    assert len(queue) == 1
    assert queue[0]["state"] == "pending"
    assert any("can't admit a member restricted" in t
               for t in _texts(sent))


def test_decline_on_unverifiable_criterion_queues(tmp_path):
    # username required + decline action: WhatsApp has no
    # usernames, so the failure is the platform's shape, not
    # the requester's fault — a human decides.
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, action="decline",
        criteria=_criteria(username=True))
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.decides() == []
    assert len(surface.acceptance_queue(KEY)) == 1


def test_decline_driven_by_a_real_ban_stands(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, action="decline",
        criteria=_criteria(banned=True))
    surface.bans.ban(REQ_JID, reason="raid spam", group_id=KEY)
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.decides() == [("decide", REQ_JID, False)]
    assert surface.acceptance_queue(KEY) == []


def test_admin_approve_of_held_request_says_full_member(tmp_path):
    # lastname required, bridge knows only "Ada": the record
    # queues with a miss.  An admin's approve admits her — and
    # is told plainly she is a FULL member (no restricted
    # state exists to hold her in).
    adapter = _WaAdapter(requests=[_req(name="Ada")])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria(lastname=True))
    mixin._wa_join_request_drive(surface, KEY, _msg())
    queue = surface.acceptance_queue(KEY)
    assert len(queue) == 1
    assert queue[0]["missing"] == ["lastname"]
    reply = mixin._control_joinq(
        f"approve {queue[0]['id']}", _msg())
    assert "WhatsApp has no restricted state" in reply
    assert "full member" in reply
    assert adapter.decides() == [("decide", REQ_JID, True)]


def test_joinq_listing_states_the_decision_set(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    listing = mixin._control_joinq("", _msg())
    text = getattr(listing, "text", listing)
    assert "no restricted state here" in text
    assert "Ngozi" in text


def test_second_poll_never_duplicates_a_request(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    _unstamp(surface)
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert len(surface.acceptance_queue(KEY)) == 1
    assert len(_audit_kinds(surface, "acceptance_eval")) == 1


def test_vanished_request_closes_as_gone(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert len(surface.acceptance_queue(KEY)) == 1
    adapter.requests = []  # withdrawn / decided in the WA app
    _unstamp(surface)
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert surface.acceptance_queue(KEY) == []
    assert _audit_kinds(surface, "acceptance_request_gone")


def test_failed_poll_leaves_the_queue_untouched(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert len(surface.acceptance_queue(KEY)) == 1
    adapter.list_ok = False
    _unstamp(surface)
    mixin._wa_join_request_drive(surface, KEY, _msg())
    # Absence of an answer is not absence of requesters.
    assert len(surface.acceptance_queue(KEY)) == 1


def test_sync_never_closes_event_driven_records(tmp_path):
    # A record the poll did NOT discover (no wa_poll source)
    # is not the poll's to close.
    adapter = _WaAdapter(requests=[])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, auto_approve=False,
        criteria=_criteria())
    out = surface.acceptance_on_request(
        KEY, {"user_id": REQ_JID, "username": "", "name": "Ada",
              "first_name": "Ada", "last_name": "",
              "is_bot": False}, has_photo=None)
    assert out["verdict"] == "queued"
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert len(surface.acceptance_queue(KEY)) == 1


def test_approval_mode_off_never_polls(tmp_path):
    adapter = _WaAdapter(requests=[_req()], approval=False)
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.calls == []  # the door list was never read
    assert surface.acceptance_queue(KEY) == []


def test_open_mode_and_unclaimed_and_verbless_skip(tmp_path):
    adapter = _WaAdapter(requests=[_req()])
    mixin, surface, adapter, sent = _rig(
        tmp_path, adapter=adapter, mode="open",
        criteria=_criteria())
    mixin._wa_join_request_drive(surface, KEY, _msg())
    assert adapter.calls == []

    adapter2 = _WaAdapter(requests=[_req()])
    mixin2, surface2, adapter2, _sent2 = _rig(
        tmp_path / "u", adapter=adapter2, designated=False,
        criteria=_criteria())
    mixin2._wa_join_request_drive(surface2, KEY, _msg())
    assert adapter2.calls == []

    # An adapter without the verb (older build): quiet skip.
    bare = SimpleNamespace(
        group_info=lambda chat: {"join_approval_mode": True})
    mixin3, surface3, _a3, _s3 = _rig(
        tmp_path / "v", adapter=bare, criteria=_criteria())
    mixin3._group_adapter = lambda platform: bare  # type: ignore
    mixin3._wa_join_request_drive(surface3, KEY, _msg())
    assert surface3.acceptance_queue(KEY) == []


def test_poll_throttle_stamps_before_work(tmp_path):
    surface = _surface(tmp_path, criteria=_criteria())
    assert surface.acceptance_poll_due(KEY, now=1000.0)
    surface.note_acceptance_poll(KEY, ts=1000.0)
    assert not surface.acceptance_poll_due(KEY, now=1100.0)
    assert surface.acceptance_poll_due(KEY, now=1000.0 + 121)
