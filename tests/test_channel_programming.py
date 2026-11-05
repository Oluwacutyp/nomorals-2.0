"""Channel programming depth (groups/channels census #8 + #4).

Pins:
- series: a named format posts its instalments in order on
  its cadence through the same cycle/claim/cadence gates as
  the queue; the position persists across restarts, advances
  only on a confirmed send, and a finished series says so;
- polls in the queue: a poll entry posts through the
  adapter's real send_poll verb (both Telegram surfaces),
  stays due when the surface has no verb, and its results
  read back from the platform — absence on the Bot API,
  which exposes no read-back;
- cross-post: a published post native-forwards between two
  claimed channels only, driven from the console, journaled
  in the target's ring, the pairing recorded as state;
- published-post management: edit/delete by console number
  through the real verbs; the ring changes only on a
  platform-confirmed result;
- discussion context: the linkage is learned from the
  platform (getFullChannel / getChat), mirrored posts accrue
  no member state and draw no defense, the antichannel
  sensor exempts the group's own linked channel while still
  striking a foreign one, and replies to a mirror count as
  comments on the channel;
- digest (gap #4): a scheduled post compiled from recorded
  facts only — silent when the period is empty (per the
  channel's setting), honest when it is not, never posted
  to an unclaimed channel.

Rigs follow the house shapes: GroupSurface over a tmp_path
GroupStore, fake Telethon modules in sys.modules for the
userbot verbs, RuntimeGroupMixin stubs like the console
suite.  No network, no real Telethon.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.social.chat.base import ChatRef
from nomorals.social.chat.group_surface import GroupStore, \
    GroupSurface

NOW = time.time()
G = "telegram-bot:-100111"          # the discussion group
C = "telegram-bot:-100222"          # a claimed channel
C2 = "telegram-bot:-100444"         # a second claimed channel
UC = "telegram:-100333"             # a userbot-held channel


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


class _Send:
    """A fake channel transport: records posts, can fail."""

    def __init__(self, fail=False):
        self.sent: list = []
        self.fail = fail

    def __call__(self, text, entry):
        if self.fail:
            return False, "boom"
        self.sent.append((text, dict(entry)))
        return True, f"m{len(self.sent)}"


def _claimed(surface, key=C, title="Gold Signals"):
    surface.note_channel_seen(key, title=title)
    surface.note_bot_admin(key, True)
    assert surface.claim_state(key)["claimed"]
    return key


# ── series ──────────────────────────────────────────────────────

def _series(surface, key=C, items=("one", "two"), at="09:00"):
    row = surface.channel_series_create(
        key, "TA Daily", {"kind": "daily", "time": at},
        now=NOW, by="99")
    assert row is not None
    for item in items:
        surface.channel_series_add(key, row["id"], item)
    # white-box: make the first instalment due now
    surface._extras(key)["channel_series"][0]["next_due"] = \
        NOW - 1
    return row


def test_series_posts_in_order_and_completes(tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    _series(surface)
    send = _Send()
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 1 and posts[0]["kind"] == "series"
    text = send.sent[0][0]
    assert "TA Daily" in text and "#1" in text and "one" in text
    row = surface.channel_series(C)[0]
    assert row["position"] == 1 and not row["done"]
    # next cadence: not due again at the same instant
    assert surface.channel_cycle(C, now=NOW + 60, send=send,
                                 force=True) == []
    # due again -> the SECOND instalment, then done
    surface._extras(C)["channel_series"][0]["next_due"] = \
        NOW + 59
    posts = surface.channel_cycle(C, now=NOW + 60, send=send,
                                  force=True)
    assert len(posts) == 1 and "#2" in send.sent[-1][0]
    row = surface.channel_series(C)[0]
    assert row["position"] == 2 and row["done"] is True
    kinds = [a["kind"] for a in
             surface.store.group(C)["audit"]]
    assert "channel_series_complete" in kinds


def test_series_failed_send_holds_position(tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    _series(surface)
    posts = surface.channel_cycle(C, now=NOW, send=_Send(
        fail=True), force=True)
    assert posts == []
    row = surface.channel_series(C)[0]
    assert row["position"] == 0 and not row["done"]


def test_series_position_persists_across_restart(tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    _series(surface)
    surface.channel_cycle(C, now=NOW, send=_Send(), force=True)
    again = _surface(tmp_path)  # reload from the same store
    row = again.channel_series(C)[0]
    assert row["position"] == 1
    assert row["items"] == ["one", "two"]


def test_series_unclaimed_posts_nothing_and_archive_holds(
        tmp_path):
    # a userbot-held channel is unclaimed until designated
    # (bot-platform presence alone never claims it)
    surface = _surface(tmp_path)
    surface.note_channel_seen(UC, title="Gold Signals")
    row = _series(surface, key=UC)
    send = _Send()
    assert surface.channel_cycle(UC, now=NOW, send=send,
                                 force=True) == []
    surface.set_designated(UC, True, by="99")
    assert surface.claim_state(UC)["claimed"]
    surface.channel_series_set_archived(UC, row["id"], True)
    assert surface.channel_cycle(UC, now=NOW, send=send,
                                 force=True) == []
    assert send.sent == []


# ── digest (census gap #4) ──────────────────────────────────────

def _digest_on(surface, key=C, silence=True):
    surface.update_settings(key, channel={
        "digest": {"enabled": True, "kind": "daily",
                   "time": "09:00", "silence": silence}})
    surface._extras(key)["channel_digest_next"] = NOW - 1


def _seed_period(surface, key=C):
    """One published post with a real view counter + a
    two-sample subscriber movement inside the period."""
    send = _Send()
    surface.channel_enqueue(key, {
        "kind": "once", "text": "gold update",
        "due": NOW - 5000, "source": "dm"})
    posts = surface.channel_cycle(key, now=NOW - 4000,
                                  send=send, force=True)
    assert len(posts) == 1
    surface.note_channel_views(key, posts[0]["message_id"],
                               1500, now=NOW - 4000)
    surface.note_channel_count(key, 100, now=NOW - 90000)
    surface.note_channel_count(key, 130, now=NOW - 3000)


def test_digest_posts_real_numbers(tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    _seed_period(surface)
    _digest_on(surface)
    send = _Send()
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 1 and posts[0]["kind"] == "digest"
    text = send.sent[0][0]
    assert "1 post(s) published" in text
    assert "1,500" in text
    assert "+30" in text and "130" in text
    # and it does not repeat on the next beat
    assert surface.channel_cycle(C, now=NOW + 60, send=send,
                                 force=True) == []


def test_digest_empty_period_is_silent_by_default(tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    _digest_on(surface)
    send = _Send()
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert posts == [] and send.sent == []
    kinds = [a["kind"] for a in
             surface.store.group(C)["audit"]]
    assert "channel_digest_skipped" in kinds
    # the schedule still advanced — silence is not a retry
    nxt = surface._extras(C)["channel_digest_next"]
    assert nxt > NOW


def test_digest_empty_period_posts_quiet_line_when_asked(
        tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    _digest_on(surface, silence=False)
    send = _Send()
    posts = surface.channel_cycle(C, now=NOW, send=send,
                                  force=True)
    assert len(posts) == 1
    assert "quiet period" in send.sent[0][0]


def test_digest_unclaimed_posts_nothing(tmp_path):
    # real data on record, digest due — the claim alone
    # holds it (a userbot-held channel, undesignated)
    surface = _surface(tmp_path)
    surface.note_channel_seen(UC, title="Gold Signals")
    surface.note_channel_count(UC, 100, now=NOW - 90000)
    surface.note_channel_count(UC, 130, now=NOW - 3000)
    surface._extras(UC)["channel_posts"] = [{
        "ts": NOW - 4000, "entry": "p1", "kind": "once",
        "source": "dm", "trigger": "", "message_id": "m1",
        "label": "gold update", "poll": False}]
    surface.note_channel_views(UC, "m1", 1500, now=NOW - 4000)
    _digest_on(surface, key=UC)
    send = _Send()
    assert surface.channel_cycle(UC, now=NOW, send=send,
                                 force=True) == []
    assert send.sent == []


# ── discussion context ──────────────────────────────────────────

def _linked(surface, group=G, key=C):
    _claimed(surface, key)
    surface.store.group(group)
    out = surface.note_channel_discussion(key, "-100111",
                                          now=NOW)
    assert out == group
    return key


def test_mirrored_post_accrues_no_member_state(tmp_path):
    surface = _surface(tmp_path)
    _linked(surface)
    result = surface.handle_message(
        G, "-100222", name="Gold Signals", text="the post",
        message_id="900",
        envelope={"forward": {
            "is_forwarded": True, "origin_kind": "channel",
            "origin_name": "Gold Signals",
            "sender_chat_id": "222", "is_automatic": True}},
        ts=NOW)
    assert result.get("discussion_mirror") == C
    members = surface.store.group(G)["members"]
    assert "-100222" not in members
    assert surface.channel_mirror_ids(C) == {"900"}


def test_comment_on_mirror_counts_on_the_channel(tmp_path):
    surface = _surface(tmp_path)
    _linked(surface)
    surface.handle_message(
        G, "-100222", text="the post", message_id="900",
        envelope={"forward": {
            "is_forwarded": True, "origin_kind": "channel",
            "sender_chat_id": "-100222", "is_automatic": True}},
        ts=NOW)
    surface.handle_message(
        G, "7", username="ada", name="Ada", text="nice post",
        message_id="901", reply_to="900", ts=NOW + 10)
    analytics = surface.channel_analytics(C)
    assert analytics["discussion"]["comments_total"] == 1
    assert analytics["discussion"]["mirrors"] == 1


def test_antichannel_exempts_linked_but_strikes_foreign(
        tmp_path):
    surface = _surface(tmp_path)
    _linked(surface)

    def env(chat_id):
        return {"forward": {
            "is_forwarded": False, "origin_kind": "",
            "origin_name": "", "sender_chat_id": chat_id,
            "is_automatic": False}}

    own = surface.handle_message(
        G, "7", username="ada", name="Ada",
        text="posting as the channel", message_id="910",
        envelope=env("-100222"), ts=NOW)
    assert not any(a.get("rule") == "antichannel"
                   for a in own.get("actions") or [])
    foreign = surface.handle_message(
        G, "8", username="mallory", name="Mallory",
        text="posting as another channel", message_id="911",
        envelope=env("-100999"), ts=NOW + 5)
    assert any(a.get("rule") == "antichannel"
               for a in foreign.get("actions") or [])


def test_discussion_unlink_restores_foreign_treatment(tmp_path):
    surface = _surface(tmp_path)
    _linked(surface)
    surface.note_channel_discussion(C, "", now=NOW + 10)
    assert surface.channel_for_discussion(G) == ""
    result = surface.handle_message(
        G, "-100222", text="the post", message_id="900",
        envelope={"forward": {
            "is_forwarded": True, "origin_kind": "channel",
            "sender_chat_id": "222", "is_automatic": True}},
        ts=NOW + 20)
    assert "discussion_mirror" not in result


# ── published-post management (surface half) ────────────────────

def test_post_records_forget_and_edit_stamps(tmp_path):
    surface = _surface(tmp_path)
    _claimed(surface)
    send = _Send()
    surface.channel_enqueue(C, {"kind": "once", "text": "first",
                                "due": NOW - 5, "source": "dm"})
    surface.channel_cycle(C, now=NOW, send=send, force=True)
    record = surface.channel_post_record(C, 1)
    assert record is not None
    assert record["label"] == "first"
    assert surface.channel_post_record(C, 2) is None
    mid = record["message_id"]
    surface.note_channel_views(C, mid, 42, now=NOW)
    assert surface.note_channel_post_edited(
        C, mid, label="first (revised)")
    record = surface.channel_post_record(C, 1)
    assert record["edited_at"] and \
        record["label"] == "first (revised)"
    assert surface.channel_forget_post(C, mid)
    assert surface.channel_posts(C) == []
    assert surface.channel_post_views_map(C) == {}


# ── fake Telethon seam for the userbot verbs ────────────────────

class _Req:
    def __init__(self, kind, **kw):
        self.kind = kind
        self.kw = kw


@pytest.fixture()
def fake_telethon_types(monkeypatch):
    types_mod = SimpleNamespace(
        TextWithEntities=lambda text, entities: _Req(
            "twe", text=text, entities=entities),
        PollAnswer=lambda text, option: _Req(
            "answer", text=text, option=option),
        Poll=lambda **kw: _Req("poll", **kw),
        InputMediaPoll=lambda poll: _Req("media_poll",
                                         poll=poll))
    messages_mod = SimpleNamespace(
        SendMediaRequest=lambda peer, media, message,
        random_id: _Req("send_media", peer=peer, media=media,
                        message=message, random_id=random_id),
        GetMessagesViewsRequest=lambda peer, id, increment:
        _Req("views", peer=peer, id=id, increment=increment))
    channels_mod = SimpleNamespace(
        GetFullChannelRequest=lambda channel: _Req(
            "full_channel", channel=channel))
    functions_mod = SimpleNamespace(channels=channels_mod,
                                    messages=messages_mod)
    tl_mod = SimpleNamespace(functions=functions_mod,
                             types=types_mod)
    telethon_mod = SimpleNamespace(tl=tl_mod)
    for name, mod in (
            ("telethon", telethon_mod),
            ("telethon.tl", tl_mod),
            ("telethon.tl.functions", functions_mod),
            ("telethon.tl.functions.channels", channels_mod),
            ("telethon.tl.functions.messages", messages_mod),
            ("telethon.tl.types", types_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    return types_mod


class _ProgClient:
    def __init__(self, *, linked=None, poll_message=None):
        self.requests: list = []
        self.forwards: list = []
        self.edits: list = []
        self._linked = linked
        self._poll_message = poll_message

    async def __call__(self, request):
        self.requests.append(request)
        if request.kind == "full_channel":
            return SimpleNamespace(
                full_chat=SimpleNamespace(
                    linked_chat_id=self._linked,
                    participants_count=None),
                chats=[])
        if request.kind == "send_media":
            return SimpleNamespace(updates=[SimpleNamespace(
                message=SimpleNamespace(id=777, views=12))])
        raise AssertionError(request.kind)

    async def get_messages(self, entity, ids=None):
        return self._poll_message

    async def forward_messages(self, entity, messages,
                               from_peer=None):
        self.forwards.append((entity, messages, from_peer))
        return SimpleNamespace(id=888, views=None)

    async def edit_message(self, entity, message_id, text):
        self.edits.append((entity, message_id, text))
        return SimpleNamespace(id=message_id)


def _userbot(client):
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter.__new__(TelegramAdapter)
    adapter._client = client
    adapter.stats = {"sent": 0, "send_errors": 0}
    ent = SimpleNamespace(broadcast=True, id=222)

    async def _resolve(chat):
        return ent

    adapter._resolve = _resolve  # type: ignore[method-assign]

    def _run(coro, timeout=60.0):
        return asyncio.new_event_loop().run_until_complete(
            coro)

    adapter._run_on_loop = _run  # type: ignore[method-assign]
    return adapter


UCHAN = ChatRef(platform="telegram", chat_id="-100333",
                kind="channel")


def test_userbot_send_poll_quiz_shape(fake_telethon_types):
    client = _ProgClient()
    adapter = _userbot(client)
    out = adapter.send_poll(
        UCHAN, "Best entry?", ["long", "short"], quiz=True,
        correct_option=1, explanation="because structure")
    assert out.ok and out.message_id == "777"
    req = client.requests[-1]
    poll = req.kw["media"].kw["poll"]
    assert poll.kw["quiz"] is True
    assert poll.kw["public_voters"] is False  # anonymous default
    assert poll.kw["correct_answers"] == [b"\x01"]
    assert [a.kw["option"] for a in poll.kw["answers"]] == \
        [b"\x00", b"\x01"]


def test_userbot_send_poll_validation_is_honest(
        fake_telethon_types):
    adapter = _userbot(_ProgClient())
    out = adapter.send_poll(UCHAN, "Q?", ["only one"])
    assert not out.ok and "2–10" in out.error
    out = adapter.send_poll(UCHAN, "Q?", ["a", "b"], quiz=True)
    assert not out.ok and "correct option" in out.error


def test_userbot_poll_results_readback(fake_telethon_types):
    poll = SimpleNamespace(
        answers=[SimpleNamespace(text="Yes", option=b"\x00"),
                 SimpleNamespace(text="No", option=b"\x01")],
        correct_answers=[b"\x01"])
    results = SimpleNamespace(
        total_voters=130,
        results=[SimpleNamespace(option=b"\x00", voters=90,
                                 chosen=False),
                 SimpleNamespace(option=b"\x01", voters=40,
                                 chosen=True)])
    msg = SimpleNamespace(
        media=SimpleNamespace(poll=poll, results=results))
    adapter = _userbot(_ProgClient(poll_message=msg))
    out = adapter.channel_poll_results(UCHAN, "777")
    assert out["ok"] and out["total_voters"] == 130
    assert out["options"][0] == {
        "text": "Yes", "voters": 90, "chosen": False,
        "correct": False}
    assert out["options"][1]["correct"] is True


def test_userbot_poll_results_absence(fake_telethon_types):
    msg = SimpleNamespace(media=SimpleNamespace(poll=None))
    adapter = _userbot(_ProgClient(poll_message=msg))
    out = adapter.channel_poll_results(UCHAN, "777")
    assert out["ok"] is False and "not a poll" in out["error"]


def test_userbot_forward_and_edit(fake_telethon_types):
    client = _ProgClient()
    adapter = _userbot(client)
    out = adapter.forward_post(UCHAN, "42", ChatRef(
        platform="telegram", chat_id="-100444", kind="channel"))
    assert out.ok and out.message_id == "888"
    assert client.forwards[0][1] == [42]
    out = adapter.edit_post(UCHAN, "42", "revised text")
    assert out == {"ok": True}
    assert client.edits[0][1:] == (42, "revised text")


def test_userbot_discussion_link_marks_id(fake_telethon_types):
    adapter = _userbot(_ProgClient(linked=111))
    out = adapter.channel_discussion_link(UCHAN)
    assert out == {"ok": True, "linked_chat_id": "-100111"}
    adapter = _userbot(_ProgClient(linked=None))
    out = adapter.channel_discussion_link(UCHAN)
    assert out == {"ok": True, "linked_chat_id": ""}


# ── bot adapter verbs (Bot API shapes) ──────────────────────────

def _bot_adapter(canned):
    from nomorals.social.chat.telegram import \
        TelegramBotAdapter

    adapter = TelegramBotAdapter.__new__(TelegramBotAdapter)
    adapter.name = "telegram-bot"
    calls: list = []

    def _api(method, **params):
        calls.append((method, params))
        out = canned(method, params)
        if isinstance(out, Exception):
            raise out
        return out

    adapter._api = _api  # type: ignore[method-assign]
    adapter.calls = calls
    return adapter


def test_bot_send_poll_and_forward():
    def canned(method, params):
        if method == "sendPoll":
            return {"message_id": 555}
        if method == "forwardMessage":
            return {"message_id": 556}
        raise AssertionError(method)

    adapter = _bot_adapter(canned)
    out = adapter.send_poll(
        ChatRef(platform="telegram-bot", chat_id="-100222",
                kind="channel"),
        "Best entry?", ["long", "short"], quiz=True,
        correct_option=0)
    assert out.ok and out.message_id == "555"
    method, params = adapter.calls[0]
    assert method == "sendPoll" and params["type"] == "quiz"
    assert params["correct_option_id"] == 0
    assert params["is_anonymous"] is True
    out = adapter.forward_post(
        ChatRef(platform="telegram-bot", chat_id="-100222",
                kind="channel"), "9",
        ChatRef(platform="telegram-bot", chat_id="-100444",
                kind="channel"))
    assert out.ok and out.message_id == "556"
    method, params = adapter.calls[1]
    assert method == "forwardMessage"
    assert params["from_chat_id"] == -100222


def test_bot_edit_and_discussion_link():
    def canned(method, params):
        if method == "editMessageText":
            return True
        if method == "getChat":
            return {"linked_chat_id": -100111}
        raise AssertionError(method)

    adapter = _bot_adapter(canned)
    ref = ChatRef(platform="telegram-bot", chat_id="-100222",
                  kind="channel")
    assert adapter.edit_post(ref, "9", "new text")["ok"]
    out = adapter.channel_discussion_link(ref)
    assert out == {"ok": True, "linked_chat_id": "-100111"}
    # the Bot API exposes no poll read-back: the verb's
    # ABSENCE is the pinned truth
    assert not hasattr(adapter, "channel_poll_results")


# ── runtime: drive transports + the console ─────────────────────

from nomorals.agents.partner.runtime_group import \
    RuntimeGroupMixin
from nomorals.social.chat.group_surface import default_surface


class _Chat:
    def __init__(self, key=G, platform="telegram-bot",
                 kind="group", title="Builders"):
        self.key = key
        self.platform = platform
        self.kind = kind
        self.title = title


class _Msg:
    def __init__(self, chat=None, **kw):
        self.chat = chat or _Chat()
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "7")
        self.sender = kw.get("sender", "Ada")
        self.sender_username = kw.get("sender_username", "ada")
        self.text = kw.get("text", "")
        self.message_id = kw.get("message_id", "55")
        self.reply_to = kw.get("reply_to", "")
        self.ts = kw.get("ts", 0.0)

    def first_media(self):
        return None


class _ProgAdapter:
    """A bot-shaped adapter carrying the programming verbs."""

    def __init__(self, *, linked=""):
        self.calls: list = []
        self.polls: list = []
        self.forwards: list = []
        self.edits: list = []
        self.deletes: list = []
        self._bot_id = 4242
        self._linked = linked

    def admin_member(self, chat, user_id):
        status = "administrator" if str(user_id) in (
            "99", "4242") else "member"
        return {"ok": True, "result": {"status": status}}

    def admin_member_count(self, chat):
        return {"ok": False, "error": "no counts here"}

    def send_poll(self, chat, question, options, **kw):
        self.polls.append((str(chat.key), question, options, kw))
        return SimpleNamespace(ok=True, message_id="p1",
                               views=None, error="")

    def forward_post(self, from_chat, message_id, to_chat):
        self.forwards.append((str(from_chat.key), message_id,
                              str(to_chat.key)))
        return SimpleNamespace(ok=True, message_id="f1",
                               views=None, error="")

    def edit_post(self, chat, message_id, text):
        self.edits.append((message_id, text))
        return {"ok": True}

    def admin_delete(self, chat, message_id):
        self.deletes.append(message_id)
        return {"ok": True}

    def channel_poll_results(self, chat, message_id):
        return {"ok": True, "total_voters": 12,
                "options": [
                    {"text": "long", "voters": 9,
                     "chosen": False, "correct": None},
                    {"text": "short", "voters": 3,
                     "chosen": False, "correct": None}]}

    def channel_discussion_link(self, chat):
        return {"ok": True, "linked_chat_id": self._linked}


class _BareProgAdapter:
    """No programming verbs at all (the absence shape)."""

    _bot_id = 4242

    def admin_member(self, chat, user_id):
        status = "administrator" if str(user_id) in (
            "99", "4242") else "member"
        return {"ok": True, "result": {"status": status}}


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent: list = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, str(
            getattr(chat, "key", chat)), text))
        return SimpleNamespace(ok=True, message_id="1",
                               error="")

    def send_file(self, platform, chat, path, **kw):
        return SimpleNamespace(ok=True, message_id="2",
                               error="")


class _Runtime(RuntimeGroupMixin):
    def __init__(self, tmp_path, adapter,
                 platform="telegram-bot", operator=True):
        self.sent = []
        self.gateway = _Gateway({platform: adapter})
        self._operator = operator
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))
        self._group_surface_obj = default_surface(str(tmp_path))

    def _send(self, chat, text, **kw):
        self.sent.append((str(text), kw.get("buttons")))
        return SimpleNamespace(ok=True, platform=chat.platform,
                               message_id="4242", error="")

    def _is_operator(self, message):
        return self._operator

    def turn(self, msg):
        return self._group_surface_turn(msg)


def _dm_msg(sender_id="99", sender="Boss"):
    chat = _Chat(key="telegram-bot:99", platform="telegram-bot",
                 kind="dm", title="")
    return _Msg(chat=chat, sender_id=sender_id, sender=sender,
                sender_username="boss")


def _text(out):
    return getattr(out, "text", out)


def _claimed_console(tmp_path, adapter, keys=(C,)):
    rt = _Runtime(tmp_path, adapter)
    surface = rt._group_surface_obj
    for key in keys:
        surface.note_channel_seen(key, title="Gold Signals")
    rt._control_console("", _dm_msg())
    return rt, surface


def test_console_poll_posts_through_adapter(tmp_path):
    adapter = _ProgAdapter()
    rt, surface = _claimed_console(tmp_path, adapter)
    out = _text(rt._control_console(
        "c1 post poll Best entry? | long | short", _dm_msg()))
    assert "posted now" in out
    assert len(adapter.polls) == 1
    _key, question, options, kw = adapter.polls[0]
    assert question == "Best entry?"
    assert options == ["long", "short"]
    assert kw["anonymous"] is True and kw["quiz"] is False
    ring = surface.channel_posts(C)
    assert len(ring) == 1 and ring[0]["poll"] is True
    # the queue card shows the poll for what it is
    rt._control_console("c1 post at 23:59 later note",
                        _dm_msg())
    surface.channel_enqueue(C, {
        "kind": "once", "text": "",
        "poll": {"question": "Later?", "options": ["a", "b"],
                 "anonymous": True},
        "due": time.time() + 90000, "source": "dm"})
    card = _text(rt._control_console("c1 queue", _dm_msg()))
    assert "🗳 poll" in card and "Later?" in card


def test_console_poll_quiz_needs_marked_option(tmp_path):
    adapter = _ProgAdapter()
    rt, _surface_ = _claimed_console(tmp_path, adapter)
    out = _text(rt._control_console(
        "c1 post poll quiz 2+2? | 3 | 4", _dm_msg()))
    assert "marks its correct option" in out
    assert adapter.polls == []
    out = _text(rt._control_console(
        "c1 post poll quiz 2+2? | 3 | *4", _dm_msg()))
    assert "posted now" in out
    assert adapter.polls[0][3]["quiz"] is True
    assert adapter.polls[0][3]["correct_option"] == 1


def test_poll_entry_stays_due_without_the_verb(tmp_path):
    rt, surface = _claimed_console(tmp_path,
                                   _BareProgAdapter())
    surface.channel_enqueue(C, {
        "kind": "once", "text": "Q?",
        "poll": {"question": "Q?", "options": ["a", "b"],
                 "anonymous": True},
        "due": time.time() - 5, "source": "dm"})
    posts = rt._channel_drive(C, force=True)
    assert posts == []
    assert len(surface.channel_queue(C)) == 1


def test_console_series_flow(tmp_path):
    adapter = _ProgAdapter()
    rt, surface = _claimed_console(tmp_path, adapter)
    out = _text(rt._control_console(
        "c1 series new TA Daily daily 09:00", _dm_msg()))
    assert "series open" in out
    out = _text(rt._control_console(
        "c1 series add 1 first instalment", _dm_msg()))
    assert "instalment 1" in out
    card = _text(rt._control_console("c1 series", _dm_msg()))
    assert "TA Daily" in card and "0/1 posted" in card
    out = _text(rt._control_console(
        "c1 series archive 1", _dm_msg()))
    assert "archived" in out
    assert surface.channel_series(C)[0]["archived"] is True


def test_console_xpost_between_claimed_channels(tmp_path):
    adapter = _ProgAdapter()
    rt, surface = _claimed_console(tmp_path, adapter,
                                   keys=(C, C2))
    # publish a post on the source channel first
    out = _text(rt._control_console(
        "c1 post now the source post", _dm_msg()))
    assert "posted now" in out
    assert len(surface.channel_posts(C)) == 1
    out = _text(rt._control_console(
        "c2 post xpost c1 1", _dm_msg()))
    assert "posted now" in out
    assert adapter.forwards == [(C, "1", C2)]
    ring = surface.channel_posts(C2)
    assert len(ring) == 1 and ring[0]["kind"] == "forward"
    assert surface.channel_cfg(C2)["xpost_sources"] == [C]
    # a channel can't cross-post into itself
    out = _text(rt._control_console(
        "c1 post xpost c1 1", _dm_msg()))
    assert "itself" in out


def test_console_posts_edit_delete_results(tmp_path):
    adapter = _ProgAdapter()
    rt, surface = _claimed_console(tmp_path, adapter)
    rt._control_console("c1 post now editable post", _dm_msg())
    rt._control_console(
        "c1 post poll Best entry? | long | short", _dm_msg())
    card = _text(rt._control_console("c1 posts", _dm_msg()))
    assert "editable post" in card and "poll" in card
    out = _text(rt._control_console(
        "c1 posts edit 1 revised words", _dm_msg()))
    assert "edited" in out
    assert adapter.edits and "revised words" in adapter.edits[0][1]
    assert surface.channel_post_record(C, 1)["edited_at"]
    out = _text(rt._control_console(
        "c1 posts results 2", _dm_msg()))
    assert "12 voter(s)" in out and "long: 9" in out
    out = _text(rt._control_console(
        "c1 posts delete 1", _dm_msg()))
    assert "deleted" in out
    assert adapter.deletes == ["1"]
    assert len(surface.channel_posts(C)) == 1


def test_console_digest_settings_and_card(tmp_path):
    adapter = _ProgAdapter()
    rt, surface = _claimed_console(tmp_path, adapter)
    out = _text(rt._control_console(
        "c1 settings digest daily 08:30", _dm_msg()))
    assert "digest: daily at 08:30" in out
    cfg = surface.channel_cfg(C)["digest"]
    assert cfg["enabled"] is True and cfg["kind"] == "daily"
    card = _text(rt._control_console("c1 settings", _dm_msg()))
    assert "digest: daily at 08:30" in card
    out = _text(rt._control_console(
        "c1 settings digest silence off", _dm_msg()))
    assert "silence off" in out


def test_drive_learns_discussion_linkage(tmp_path):
    adapter = _ProgAdapter(linked="-100111")
    rt, surface = _claimed_console(tmp_path, adapter)
    surface.store.group(G)
    rt._channel_drive(C, force=True)
    disc = surface.channel_discussion(C)
    assert disc["group_key"] == G
    # and the group turn that follows treats the channel's
    # own mirrored post as the channel's traffic
    chat = _Chat(key=G, kind="group")
    rt.turn(_Msg(chat=chat, sender_id="-100222",
                 sender="Gold Signals", text="the post",
                 message_id="900"))
    assert surface.channel_mirror_ids(C) == {"900"}
    assert "-100222" not in surface.store.group(G)["members"]
