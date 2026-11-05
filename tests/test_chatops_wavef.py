"""FEI gap wave F — the chat & privacy command set.

Network/bridge-free.  Fixtures are the REAL shapes: bridge
verdict dicts and command payloads (Go + Baileys), the pairing
registry's records, the personal vault's journals, MTProto
verdict dicts.  Covered:

* registry presence (CONTROL_COMMANDS / DETAILS / ONELINERS /
  exactly one LIST_GROUPS home), none of the kinds public,
  owner-only via caller_may_run, and the runtime dispatch
  literal set == chatops.CHATOPS_COMMANDS;
* the pure helpers (disappear vocabulary, tag chunking + cap,
  mass-tag test, poll formatting, group-status composition);
* the personal organ's new gates: status_react default-OFF /
  toggle / once-per-post / sticky-stop, anticall rejection +
  stop suppression, antitag journal + DM-flag cooldown with
  the suppressed count, the antiedit cache/edit gates (and
  delete recovery staying antidelete's alone);
* the WhatsApp adapter's bridge command shapes and the
  Telegram adapter's import-free verb shapes (fake client);
* the handler end-to-end over fake adapters: chat-target
  resolution BY NAME, unsend/edit own-vs-other, the window
  notes, the honest absences (Telegram unread/star, WhatsApp
  pollres, bot surfaces), banchecker honesty, gcstatus from
  enumerated data only, tagall chunking + the named cap.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))

from nomorals.social.chat import control  # noqa: E402
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef  # noqa: E402
from nomorals.social.chat.chatops import (  # noqa: E402
    CHATOPS_COMMANDS,
    TAG_CAP,
    chunk_mentions,
    compose_group_status,
    format_poll_results,
    is_mass_tag,
    kind_for_chat_key,
    parse_disappear_duration,
)

WA_GROUP = ChatRef(platform="whatsapp", chat_id="123@g.us",
                   kind=ChatKind.GROUP, title="Family Group")
WA_DM = ChatRef(platform="whatsapp",
                chat_id="2348011111111@s.whatsapp.net",
                kind=ChatKind.DM, title="Ada Lovelace",
                peer="2348011111111@s.whatsapp.net")
TG_GROUP = ChatRef(platform="telegram", chat_id="-100123",
                   kind=ChatKind.GROUP, title="TG Group")
TG_DM = ChatRef(platform="telegram", chat_id="777",
                kind=ChatKind.DM, title="Bob", peer="777")


def _msg(chat, *, text="", reply_to="", sender_id="111",
         mentioned=False, meta=None, message_id="c1"):
    return ChatMessage(chat=chat, incoming=True, text=text,
                       sender="Ada", sender_id=sender_id,
                       message_id=message_id, reply_to=reply_to,
                       mentioned=mentioned, meta=meta or {})


# ── registry ─────────────────────────────────────────────────


def test_registry_presence_and_owner_only():
    for kind in CHATOPS_COMMANDS:
        assert kind in control.CONTROL_COMMANDS, kind
        assert kind in control.COMMAND_DETAILS, kind
        assert kind in control.LIST_ONELINERS, kind
        assert kind not in control.PUBLIC_COMMANDS, kind
        assert kind not in control.GAME_COMMANDS, kind
    homes = [name for name, members in control.LIST_GROUPS
             if set(members) & CHATOPS_COMMANDS]
    assert len(homes) == 1
    members = dict(control.LIST_GROUPS)[homes[0]]
    assert set(members) == CHATOPS_COMMANDS
    # owner-only through the real gate
    for kind in CHATOPS_COMMANDS:
        assert control.caller_may_run(kind, is_owner=True)
        assert not control.caller_may_run(kind, is_owner=False)


def test_dispatch_literal_set_matches():
    src = Path("nomorals/agents/partner/runtime.py").read_text(
        encoding="utf-8")
    tree = ast.parse(src)
    found = None
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.ops[0], ast.In)):
            continue
        right = test.comparators[0]
        if not isinstance(right, (ast.Set, ast.Tuple, ast.List)):
            continue
        values = {elt.value for elt in right.elts
                  if isinstance(elt, ast.Constant)
                  and isinstance(elt.value, str)}
        if "tagall" in values and "antiedit" in values:
            found = values
    assert found is not None
    assert found == set(CHATOPS_COMMANDS)


# ── pure helpers ─────────────────────────────────────────────


def test_disappear_vocabulary():
    assert parse_disappear_duration("off") == 0
    assert parse_disappear_duration("24h") == 86400
    assert parse_disappear_duration("7d") == 604800
    assert parse_disappear_duration("90d") == 7776000
    assert parse_disappear_duration("3600") == 3600
    assert parse_disappear_duration("whenever") is None
    assert parse_disappear_duration("") is None


def test_chunk_mentions_cap_and_chunks():
    members = [(str(i), "") for i in range(120)]
    chunks, tagged, capped = chunk_mentions(members)
    assert capped is True
    assert tagged == TAG_CAP == 100
    assert [len(c) for c in chunks] == [50, 50]
    small, tagged2, capped2 = chunk_mentions(members[:3])
    assert (tagged2, capped2) == (3, False)
    assert [len(c) for c in small] == [3]


def test_mass_tag_threshold():
    assert is_mass_tag(5) and is_mass_tag(9)
    assert not is_mass_tag(4) and not is_mass_tag(0)


def test_poll_format_counts_only():
    out = format_poll_results({
        "ok": True, "total_voters": 10,
        "options": [{"text": "Yes", "voters": 7,
                     "correct": None},
                    {"text": "No", "voters": None,
                     "correct": None}]})
    assert "Yes — 7 votes (70%)" in out
    assert "No — votes not exposed" in out
    assert "Total: 10 voters" in out


def test_group_status_composition_labels_source():
    out = compose_group_status(
        title="Crew", platform="whatsapp", kind="group",
        member_count=None, admins=[], extra=[],
        source="the roster")
    assert "Members: not counted on this surface" in out
    assert "Admins: not enumerated on this surface" in out
    assert "composed from the roster" in out
    out2 = compose_group_status(
        title="Crew", platform="whatsapp", kind="group",
        member_count=12, admins=["ada", "bola"], extra=[],
        source="the roster")
    assert "Members: 12" in out2
    assert "Admins (2): ada, bola" in out2


def test_kind_for_chat_key():
    assert kind_for_chat_key("whatsapp:1@g.us") == "group"
    assert kind_for_chat_key("whatsapp:1@newsletter") == "channel"
    assert kind_for_chat_key("whatsapp:1@s.whatsapp.net") == "dm"
    assert kind_for_chat_key("telegram:-10055") == "group"
    assert kind_for_chat_key("telegram:777") == "dm"


# ── the personal organ ───────────────────────────────────────


def _personal(tmp_path, **seams):
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    registry = PairingRegistry(tmp_path / "paired.json")
    vault = PersonalVault(tmp_path / "personal")
    personal = PersonalSurfaces(registry, vault, **seams)
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked", created_at=1.0))
    return personal, registry, vault, rec


def test_personal_toggles_default_off_and_present(tmp_path):
    from nomorals.social.pairing import PERSONAL_TOGGLES

    for name in ("status_react", "anticall", "antitag",
                 "antiedit"):
        assert name in PERSONAL_TOGGLES
    _obj, _registry, _vault, rec = _personal(tmp_path)
    for name in ("status_react", "anticall", "antitag",
                 "antiedit"):
        assert rec.toggle(name) is False  # every one ships OFF
    assert rec.status_react_emoji == "❤️"


def test_status_react_gates(tmp_path):
    calls = []

    def reactor(rec, item):
        calls.append(item)
        return True

    personal, registry, _vault, rec = _personal(
        tmp_path, status_reactor=reactor)
    post = dict(poster_id="222@s.whatsapp.net",
                poster_name="Bola", text="hi", media=[],
                source_id="st1", ts=10.0)
    personal.observe_status_post("whatsapp", **post)
    assert calls == []  # default OFF
    registry.set_toggles(rec.id, status_react=True)
    personal.observe_status_post("whatsapp", **post)
    assert len(calls) == 1
    assert calls[0]["emoji"] == "❤️"
    personal.observe_status_post("whatsapp", **post)
    assert len(calls) == 1  # once per post per process
    # the owner's own later reply discipline is untouched: a
    # reaction is not a message, and nothing here sends one.
    registry.set_owner_stopped("whatsapp", True)
    personal.observe_status_post(
        "whatsapp", **{**post, "source_id": "st2"})
    assert len(calls) == 1  # sticky stop halts it


def test_anticall_reject_and_stop(tmp_path):
    rejected = []

    def rejector(rec, item):
        rejected.append(item)
        return True

    personal, registry, vault, rec = _personal(
        tmp_path, call_rejector=rejector)
    out = personal.observe_call("whatsapp", from_id="222",
                                call_id="call1", ts=5.0)
    assert out is None and rejected == []  # default OFF
    registry.set_toggles(rec.id, anticall=True)
    out = personal.observe_call("whatsapp", from_id="222",
                                call_id="call1", ts=6.0)
    assert out is not None and out["kind"] == "call_offer"
    assert rejected and rejected[0]["call_id"] == "call1"
    registry.set_owner_stopped("whatsapp", True)
    out = personal.observe_call("whatsapp", from_id="222",
                                call_id="call2", ts=7.0)
    assert out is None and len(rejected) == 1  # fully suppressed


def test_antitag_cooldown_and_suppressed(tmp_path):
    personal, registry, _vault, rec = _personal(tmp_path)
    kw = dict(chat_key="whatsapp:123@g.us", sender_id="9",
              sender_name="Ada", mention_count=8)
    assert personal.observe_mass_tag("whatsapp", ts=100.0,
                                     **kw) is None  # OFF
    registry.set_toggles(rec.id, antitag=True)
    first = personal.observe_mass_tag("whatsapp", ts=100.0,
                                      **kw)
    assert first is not None and first["deliver"] is True
    assert personal.observe_mass_tag("whatsapp", ts=200.0,
                                     **kw) is None
    assert personal.observe_mass_tag("whatsapp", ts=300.0,
                                     **kw) is None
    later = personal.observe_mass_tag("whatsapp", ts=5000.0,
                                      **kw)
    assert later is not None
    assert later["suppressed_before"] == 2


def test_antiedit_recovery_and_silence(tmp_path):
    personal, registry, vault, rec = _personal(tmp_path)
    kw = dict(chat_key="whatsapp:111@s.whatsapp.net",
              counterpart_id="111@s.whatsapp.net")
    # antiedit ON alone arms the cache (antidelete stays off)
    registry.set_toggles(rec.id, antiedit=True)
    personal.observe_message(
        "whatsapp", sender_id="111", sender_name="Ada",
        message_id="m1", text="meet at 7", ts=10.0, **kw)
    entry = personal.observe_edit(
        "whatsapp", message_id="m1", new_text="meet at 8",
        sender_id="111", sender_name="Ada", ts=20.0, **kw)
    assert entry is not None
    assert entry["original_text"] == "meet at 7"
    assert entry["new_text"] == "meet at 8"
    recs = [r for r in vault.recoveries(account_id=rec.id)
            if r.get("kind") == "edit"]
    assert len(recs) == 1
    # an edit of a message never seen recovers nothing
    ghost = personal.observe_edit(
        "whatsapp", message_id="mX", new_text="boo",
        sender_id="111", ts=30.0, **kw)
    assert ghost is None
    # both toggles off: silence — no cache, no recovery
    registry.set_toggles(rec.id, antiedit=False)
    personal.observe_message(
        "whatsapp", sender_id="111", message_id="m2",
        text="secret", ts=40.0, **kw)
    assert personal.observe_edit(
        "whatsapp", message_id="m2", new_text="changed",
        sender_id="111", ts=50.0, **kw) is None


def test_delete_recovery_stays_antidelete_only(tmp_path):
    personal, registry, _vault, rec = _personal(tmp_path)
    kw = dict(chat_key="whatsapp:111@s.whatsapp.net",
              counterpart_id="111@s.whatsapp.net")
    registry.set_toggles(rec.id, antiedit=True)  # NOT antidelete
    personal.observe_message(
        "whatsapp", sender_id="111", message_id="m1",
        text="meet at 7", ts=10.0, **kw)
    out = personal.observe_delete(
        "whatsapp", deleted_ids=["m1"], ts=20.0, **kw)
    assert out == []  # the cache exists; delete recovery is not
    # antiedit's to give


def test_status_react_emoji_roundtrip(tmp_path):
    personal, registry, _vault, rec = _personal(tmp_path)
    personal.set_status_react_emoji(rec.id, "🔥")
    assert registry.get(rec.id).status_react_emoji == "🔥"
    try:
        personal.set_status_react_emoji("whatsapp:nope", "🔥")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown account must raise")


# ── adapter verb shapes ──────────────────────────────────────


def _wa_adapter():
    import threading

    from nomorals.social.chat.whatsapp import WhatsAppAdapter

    adapter = object.__new__(WhatsAppAdapter)
    adapter.connected = threading.Event()
    adapter.connected.set()
    sent = []

    def fake_send_cmd(request, timeout=30.0):
        sent.append(request)
        return {"ok": True, "id": "req1"}

    adapter._send_cmd = fake_send_cmd  # type: ignore[method-assign]
    return adapter, sent


def test_wa_adapter_command_shapes():
    adapter, sent = _wa_adapter()
    ref = ChatRef(platform="whatsapp", chat_id="123@g.us",
                  kind=ChatKind.GROUP)
    assert adapter.chat_archive(ref, True)["ok"]
    assert sent[-1]["cmd"] == "chat_modify"
    assert sent[-1]["archive"] is True
    assert adapter.chat_mute(ref, seconds=-1)["ok"]
    assert sent[-1]["mute_seconds"] == -1
    assert adapter.chat_mark_read(ref, False)["ok"]
    assert sent[-1]["read"] is False
    assert adapter.star_message(ref, "m9", True)["ok"]
    assert sent[-1]["cmd"] == "message_star"
    assert sent[-1]["message_id"] == "m9"
    assert adapter.edit_message(ref, "m9", "new")["ok"]
    assert sent[-1]["cmd"] == "edit"
    assert adapter.react_status("222@s.whatsapp.net", "st1",
                                "❤️")["ok"]
    assert sent[-1]["cmd"] == "status_react"
    assert adapter.reject_call("c1", "222@s.whatsapp.net")["ok"]
    assert sent[-1]["cmd"] == "reject_call"
    assert adapter.block_user("222@s.whatsapp.net",
                              "block")["ok"]
    assert sent[-1]["cmd"] == "blocklist"
    assert adapter.set_disappearing(ref, 604800)["ok"]
    assert sent[-1]["cmd"] == "chat_disappear"
    verdict = adapter.check_number("+234 801 234 5678")
    assert sent[-1]["cmd"] == "check_number"
    assert "ban" not in str(verdict).lower()


def test_tg_adapter_import_free_verb_shapes():
    import asyncio

    from nomorals.social.chat.telegram import TelegramAdapter

    calls = []

    class FakeClient:
        async def send_read_acknowledge(self, ent):
            calls.append(("read", ent))

        async def edit_message(self, ent, mid, text):
            calls.append(("edit", ent, mid, text))

        async def delete_messages(self, ent, ids, revoke=False):
            calls.append(("revoke", ent, ids, revoke))

    adapter = object.__new__(TelegramAdapter)
    adapter._client = FakeClient()
    adapter._run_on_loop = (  # type: ignore[method-assign]
        lambda coro, timeout=30.0: asyncio.run(coro))

    async def fake_resolve(chat):
        return "ENT"

    adapter._resolve = fake_resolve  # type: ignore[method-assign]
    ref = ChatRef(platform="telegram", chat_id="777",
                  kind=ChatKind.DM)
    assert adapter.mark_read(ref)["ok"]
    assert calls[-1] == ("read", "ENT")
    assert adapter.edit_message(ref, "42", "fixed")["ok"]
    assert calls[-1] == ("edit", "ENT", 42, "fixed")
    assert adapter.revoke_message(ref, "42")["ok"]
    assert calls[-1] == ("revoke", "ENT", [42], True)
    bad = adapter.edit_message(ref, "not-a-number", "x")
    assert bad["ok"] is False  # honest, pre-flight


# ── the handler over fake adapters ───────────────────────────


class _FakeWA:
    name = "whatsapp"
    user_jid = "2348012345678@s.whatsapp.net"

    def __init__(self):
        self.calls = []
        self.fail = {}
        self.history_rows = []
        self.roster = []
        self.info = {}

    def _v(self, verb, *args):
        self.calls.append((verb, args))
        return self.fail.get(verb, {"ok": True})

    def chats(self, limit=30):
        return [{"id": "123@g.us", "title": "Family Group"},
                {"id": "2348011111111@s.whatsapp.net",
                 "title": "Ada Lovelace"}]

    def chat_archive(self, ref, archived):
        return self._v("chat_archive", ref, archived)

    def chat_mute(self, ref, *, seconds):
        return self._v("chat_mute", ref, seconds)

    def chat_mark_read(self, ref, read):
        return self._v("chat_mark_read", ref, read)

    def star_message(self, ref, mid, star=True):
        return self._v("star_message", ref, mid, star)

    def edit_message(self, ref, mid, text):
        return self._v("edit_message", ref, mid, text)

    def admin_delete(self, ref, mid):
        return self._v("admin_delete", ref, mid)

    def block_user(self, jid, action):
        return self._v("block_user", jid, action)

    def set_disappearing(self, ref, seconds):
        return self._v("set_disappearing", ref, seconds)

    def check_number(self, phone):
        self.calls.append(("check_number", (phone,)))
        return self.fail.get(
            "check_number",
            {"ok": True, "exists": True,
             "jid": f"{phone}@s.whatsapp.net"})

    def history(self, chat, limit=20):
        return list(self.history_rows)

    def group_info(self, chat):
        return dict(self.info)

    def group_participants(self, chat):
        return list(self.roster)

    def send(self, chat, text, *, mentions=None, **kw):
        self.calls.append(("send", (chat, text, mentions)))
        return SimpleNamespace(ok=True, message_id="s1",
                               error="")


class _FakeTG:
    name = "telegram"

    def __init__(self):
        self.calls = []
        self.fail = {}
        self.history_rows = []
        self.members = []
        self.poll = {"ok": True, "total_voters": 3,
                     "options": [{"text": "A", "voters": 2,
                                  "correct": None},
                                 {"text": "B", "voters": 1,
                                  "correct": None}]}

    def _v(self, verb, *args):
        self.calls.append((verb, args))
        return self.fail.get(verb, {"ok": True})

    def archive_chat(self, ref, archived):
        return self._v("archive_chat", ref, archived)

    def mute_chat(self, ref, *, muted):
        return self._v("mute_chat", ref, muted)

    def mark_read(self, ref):
        return self._v("mark_read", ref)

    def edit_message(self, ref, mid, text):
        return self._v("edit_message", ref, mid, text)

    def revoke_message(self, ref, mid):
        return self._v("revoke_message", ref, mid)

    def block_user(self, target, action):
        return self._v("block_user", target, action)

    def set_disappearing(self, ref, seconds):
        return self._v("set_disappearing", ref, seconds)

    def history(self, chat, limit=20):
        return list(self.history_rows)

    def channel_poll_results(self, chat, mid):
        self.calls.append(("channel_poll_results", (chat, mid)))
        return self.fail.get("channel_poll_results", self.poll)

    def admin_member_count(self, chat):
        self.calls.append(("admin_member_count", (chat,)))
        return {"ok": True, "count": len(self.members)}

    def admin_members_full(self, chat, limit=20000):
        self.calls.append(("admin_members_full", (chat,)))
        return {"ok": True, "members": list(self.members)}

    def send_tagged(self, chat, text, members, *, hidden):
        self.calls.append(
            ("send_tagged", (chat, text, list(members), hidden)))
        return {"ok": True, "mentioned": len(members)}

    def saved_messages_ref(self):
        return ChatRef(platform="telegram", chat_id="self",
                       kind=ChatKind.DM, title="Saved Messages")


def _runtime(tmp_path, wa=None, tg=None):
    from nomorals.agents.partner.runtime_chatops import RuntimeChatOpsMixin
    from nomorals.agents.partner.runtime_provision import RuntimeProvisionMixin
    from nomorals.social.pairing import AccountRecord, PairingRegistry
    from nomorals.social.personal import PersonalSurfaces, PersonalVault

    wa = wa if wa is not None else _FakeWA()
    tg = tg if tg is not None else _FakeTG()
    registry = PairingRegistry(tmp_path / "paired.json")
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked",
        created_at=1.0))
    personal = PersonalSurfaces(
        registry, PersonalVault(tmp_path / "personal"))
    sends = []

    def fake_send(platform, ref, text):
        sends.append((platform, ref, text))
        return SimpleNamespace(ok=True, error="")

    class _RT(RuntimeChatOpsMixin, RuntimeProvisionMixin):
        pass

    rt = object.__new__(_RT)
    rt.context = SimpleNamespace(
        db=None,
        settings=SimpleNamespace(workspace_dir=str(tmp_path)))
    rt.gateway = SimpleNamespace(
        adapters={"whatsapp": wa, "telegram": tg},
        send=fake_send)
    rt._pair_personal_svc = {
        "registry": registry, "personal": personal}
    rt._housekeeping_beat = lambda: None
    return rt, wa, tg, registry, rec, sends


def test_archive_by_chat_name(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("archive", "Family Group",
                              _msg(WA_DM))
    assert out == "archived Family Group."
    verb, args = wa.calls[-1]
    assert verb == "chat_archive"
    assert args[0].chat_id == "123@g.us"  # resolved, not raw
    assert args[1] is True


def test_chat_state_current_chat_default(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("mutechat", "", _msg(WA_DM))
    assert out.startswith("muted Ada Lovelace")
    verb, args = wa.calls[-1]
    assert verb == "chat_mute" and args[1] == -1
    out = rt._control_chatops("unreadchat", "", _msg(WA_DM))
    assert "marked Ada Lovelace unread" in out
    verb, args = wa.calls[-1]
    assert verb == "chat_mark_read" and args[1] is False


def test_telegram_unread_is_the_honest_absence(tmp_path):
    rt, _wa, tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("unreadchat", "", _msg(TG_DM))
    assert "no mark-as-unread" in out
    assert tg.calls == []  # nothing was even attempted
    out = rt._control_chatops("readchat", "", _msg(TG_DM))
    assert "marked Bob read" in out
    assert tg.calls[-1][0] == "mark_read"


def test_star_telegram_absence_and_wa_shape(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("star", "", _msg(
        TG_DM, reply_to="9"))
    assert "no message star" in out
    out = rt._control_chatops("star", "", _msg(
        WA_DM, reply_to="m9"))
    assert out == "starred ⭐"
    verb, args = wa.calls[-1]
    assert verb == "star_message" and args[1] == "m9"
    out = rt._control_chatops("star", "", _msg(WA_DM))
    assert "reply to the message" in out  # no target, no claim


def _row(mid, incoming):
    return ChatMessage(chat=WA_DM, incoming=incoming, text="x",
                       sender="Ada", sender_id="111",
                       message_id=mid)


def test_unsend_own_other_and_window(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    wa.history_rows = [_row("m1", incoming=False),
                       _row("m2", incoming=True)]
    out = rt._control_chatops("unsend", "", _msg(
        WA_DM, reply_to="m1"))
    assert out == "deleted for everyone."
    assert wa.calls[-1][0] == "admin_delete"
    out = rt._control_chatops("unsend", "", _msg(
        WA_DM, reply_to="m2"))
    assert "not your message" in out
    assert [c for c in wa.calls
            if c[0] == "admin_delete"] == [wa.calls[0]]
    wa.fail["admin_delete"] = {
        "ok": False, "error": "message too old to revoke"}
    out = rt._control_chatops("unsend", "", _msg(
        WA_DM, reply_to="m1"))
    assert "too old" in out and "60 hours" in out


def test_edit_own_other_and_window(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    wa.history_rows = [_row("m1", incoming=False),
                       _row("m2", incoming=True)]
    out = rt._control_chatops("editmsg", "fixed text", _msg(
        WA_DM, reply_to="m2"))
    assert "not your message" in out
    assert "impossible on every platform" in out
    out = rt._control_chatops("editmsg", "fixed text", _msg(
        WA_DM, reply_to="m1"))
    assert out == "edited."
    verb, args = wa.calls[-1]
    assert verb == "edit_message" and args[2] == "fixed text"
    wa.fail["edit_message"] = {"ok": False,
                               "error": "edit window closed"}
    out = rt._control_chatops("edit", "again", _msg(
        WA_DM, reply_to="m1"))
    assert "15 minutes" in out


def test_block_by_number_builds_jid(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("block", "+234 809 999 0000",
                              _msg(WA_DM))
    assert out.startswith("blocked")
    verb, args = wa.calls[-1]
    assert verb == "block_user"
    assert args[0] == "2348099990000@s.whatsapp.net"
    assert args[1] == "block"


def test_disappear_parse_and_target(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("disappear", "7d", _msg(WA_DM))
    assert "7d" in out
    verb, args = wa.calls[-1]
    assert verb == "set_disappearing" and args[1] == 604800
    out = rt._control_chatops("disappear", "whenever",
                              _msg(WA_DM))
    assert "isn't a duration" in out


def test_banchecker_number_honesty(tmp_path):
    rt, _wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("banchecker", "+2348012345678",
                              _msg(WA_DM))
    assert "is on WhatsApp" in out
    assert "publishes no ban state" in out
    assert "banned" not in out.split("publishes")[0]


def test_banchecker_own_accounts(tmp_path):
    rt, _wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("banchecker", "", _msg(WA_DM))
    assert "whatsapp" in out and "status: linked" in out
    assert "no ban verdict" in out


def test_gcstatus_from_enumerated_data(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    wa.info = {"subject": "Family Group", "size": 3,
               "admins": [], "announce": False,
               "join_approval_mode": True,
               "ephemeral_hours": 0, "desc": ""}
    wa.roster = [
        {"jid": "1@s.whatsapp.net", "name": "Ada",
         "role": "superadmin"},
        {"jid": "2@s.whatsapp.net", "name": "Bola", "role": ""},
        {"jid": "3@s.whatsapp.net", "name": "Cy", "role": ""}]
    out = rt._control_chatops("gcstatus", "", _msg(WA_GROUP))
    assert "Members: 3" in out
    assert "Admins (1): Ada" in out
    assert "Join approval: on" in out
    assert "no platform offers a group-stats endpoint" in out
    out = rt._control_chatops("statusinfo", "", _msg(WA_DM))
    assert "inside the group" in out


def test_pollres_shapes_and_absences(tmp_path):
    rt, _wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("pollres", "", _msg(
        TG_GROUP, reply_to="p1"))
    assert "A — 2 votes (67%)" in out
    assert "Total: 3 voters" in out
    out = rt._control_chatops("pollres", "", _msg(
        WA_GROUP, reply_to="p1"))
    assert "doesn't expose a poll's tallies" in out


def test_tagall_chunks_and_cap(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    wa.roster = [{"jid": f"{1000 + i}@s.whatsapp.net",
                  "name": f"M{i}", "role": ""}
                 for i in range(120)]
    out = rt._control_chatops("tagall", "standup in 5",
                              _msg(WA_GROUP))
    sends = [c for c in wa.calls if c[0] == "send"]
    assert len(sends) == 2  # 100 capped, in chunks of 50
    assert len(sends[0][1][2]) == 50
    assert "tagged 100 members" in out
    assert "100-member cap" in out
    assert "@1000" in sends[0][1][1]  # names visible for tagall


def test_hidetag_hides_names(tmp_path):
    rt, wa, _tg, _reg, _rec, _s = _runtime(tmp_path)
    wa.roster = [{"jid": "2000@s.whatsapp.net", "name": "Ada",
                  "role": ""}]
    out = rt._control_chatops("hidetag", "quiet ping",
                              _msg(WA_GROUP))
    sends = [c for c in wa.calls if c[0] == "send"]
    assert sends and sends[0][1][1] == "quiet ping"
    assert sends[0][1][2] == ["2000@s.whatsapp.net"]
    assert "hidden mentions" in out


def test_tagall_telegram_chunks_via_send_tagged(tmp_path):
    rt, _wa, tg, _reg, _rec, _s = _runtime(tmp_path)
    tg.members = [{"id": str(5000 + i), "username": f"u{i}",
                   "name": ""} for i in range(75)]
    out = rt._control_chatops("tagall", "yo", _msg(TG_GROUP))
    sends = [c for c in tg.calls if c[0] == "send_tagged"]
    assert [len(c[1][2]) for c in sends] == [50, 25]
    assert "tagged 75 members" in out


def test_toggle_commands_flip_the_pair_store(tmp_path):
    rt, _wa, _tg, registry, rec, _s = _runtime(tmp_path)
    out = rt._control_chatops("antitag", "", _msg(WA_DM))
    assert "is OFF" in out
    out = rt._control_chatops("antitag", "on", _msg(WA_DM))
    assert "ON" in out
    assert registry.get(rec.id).toggle("antitag") is True
    out = rt._control_chatops("statusreact", "emoji 🔥",
                              _msg(WA_DM))
    assert "🔥" in out
    assert registry.get(rec.id).status_react_emoji == "🔥"
    out = rt._control_chatops("anticall", "on", _msg(WA_DM))
    assert registry.get(rec.id).toggle("anticall") is True
    out = rt._control_chatops("antiedit", "", _msg(WA_DM))
    assert "antiedit is OFF" in out


# ── the live turn: call offers + the antitag flag ────────────


def test_turn_consumes_call_offer(tmp_path):
    rt, _wa, _tg, registry, rec, _s = _runtime(tmp_path)
    registry.set_toggles(rec.id, anticall=True)
    msg = _msg(WA_DM, meta={"service": "call_offer",
                            "call_id": "c9"})
    assert rt._personal_surface_turn(msg) is True
    rows = rt._pair_personal_svc["personal"].vault.journal(
        account_id=rec.id, kind="call_offer")
    assert len(rows) == 1  # journaled; no reject verb on the
    # fake adapter → the entry honestly says it did not land


def test_turn_antitag_flags_owner_dm_once(tmp_path):
    rt, _wa, _tg, registry, rec, sends = _runtime(tmp_path)
    registry.set_toggles(rec.id, antitag=True)
    meta = {"envelope": {"mentions": ["1", "2", "3", "4", "5",
                                      "6"],
                         "entities": []}}
    msg = _msg(WA_GROUP, mentioned=True, meta=meta)
    assert rt._personal_surface_turn(msg) is False  # flows on
    assert len(sends) == 1
    platform, ref, text = sends[0]
    assert platform == "whatsapp"
    assert ref.chat_id == "2348012345678@s.whatsapp.net"
    assert "mass tag" in text and "Family Group" in text
    msg2 = _msg(WA_GROUP, mentioned=True, meta=meta)
    rt._personal_surface_turn(msg2)
    assert len(sends) == 1  # cooldown: flagged once
    # a tag that doesn't include the owner never flags
    msg3 = _msg(WA_GROUP, mentioned=False, meta=meta)
    rt._personal_surface_turn(msg3)
    assert len(sends) == 1


def test_turn_status_post_reacts_through_wired_seams(tmp_path):
    # The runtime's own seam wiring: _pairing_personal builds
    # PersonalSurfaces WITH the wave's reactor/rejector seams.
    from nomorals.social.pairing import AccountRecord, PairingRegistry

    rt, wa, _tg, _registry, _rec, _s = _runtime(tmp_path)
    del rt._pair_personal_svc  # force the real builder
    reacted = []

    def react_status(poster, mid, emoji):
        reacted.append((poster, mid, emoji))
        return {"ok": True}

    wa.react_status = react_status
    svc = rt._pairing_personal()
    assert svc["personal"].status_reactor is not None
    assert svc["personal"].call_rejector is not None
    registry = svc["registry"]
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked",
        created_at=1.0))
    assert isinstance(registry, PairingRegistry)
    registry.set_toggles(rec.id, status_react=True)
    msg = _msg(ChatRef(platform="whatsapp",
                       chat_id="status@broadcast",
                       kind=ChatKind.DM,
                       peer="222@s.whatsapp.net"),
               sender_id="222@s.whatsapp.net",
               message_id="st1",
               meta={"service": "status_post"})
    assert rt._personal_surface_turn(msg) is True
    assert reacted == [("222@s.whatsapp.net", "st1", "❤️")]
