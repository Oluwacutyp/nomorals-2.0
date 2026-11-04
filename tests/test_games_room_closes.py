"""§34 closes, pushed (games2 addendum, 2026-11-03):

The games-engine pass settled timed rounds on the clock but
delivered the closes as held notices on the member's next /math
(math had no poster). The owner's §34 words: when a 60s round
completes the bot should CLOSE the question and post the answer
with its working. This pass gives Math Class a room drain and
rides the group cycle:

* the runtime stamps the room a member plays math in; rounds and
  duels snapshot it at start (an explicit "" snapshot is a DM
  round — roomless by construction);
* on each group cycle pass the runtime drains this room's
  settled closes and posts them into the room — one message,
  the same board the held-note path renders;
* §33 gate at send time: an unclaimed room gets no post and its
  closes fall back to held notes; §31 quiet (defense autonomy
  off) diverts the same way;
* dedupe is structural: settlement happens once, and a close is
  delivered exactly once — room post XOR held note, never both,
  in either order.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.games.math_class import MathClass
from nomorals.games.math_commands import MathCommandHandler
from nomorals.games.math_economy import MathEconomy
from nomorals.games.math_persist import dump_state, load_state
from nomorals.social.chat.base import ChatKind

T0 = 1_700_000_000.0

ADA = "telegram:ada"
BOB = "discord:bob"
ROOM = "telegram-bot:-100123"
ROOM2 = "discord:777"


def _handler() -> MathCommandHandler:
    eco = MathEconomy()
    for p in (ADA, BOB):
        eco.verify_member(p, fingerprint=f"dev-{p}")
    return MathCommandHandler(MathClass(), eco)


def _note_kinds(h: MathCommandHandler, player: str) -> list[str]:
    return [n.get("kind") for n in h._pending_notes.get(player) or []]


# ── handler: the room drain ──────────────────────────────────────


def test_room_close_drains_to_its_room_only():
    h = _handler()
    h.note_room(ADA, ROOM)
    h.handle("/math easy", ADA, now=T0)
    closes = h.peek_room_closes(ROOM, now=T0 + 61)
    assert [c["kind"] for c in closes] == ["round_close"]
    # the close carries the reveal: answer + the generator's
    # own working, exactly what the room post renders
    assert closes[0]["problem"]["answer"]
    assert closes[0]["problem"]["solution_steps"]
    # another room's drain sees nothing; the member holds no note
    assert h.peek_room_closes(ROOM2, now=T0 + 61) == []
    assert _note_kinds(h, ADA) == []
    h.ack_room_closes(ROOM, closes)
    assert h.peek_room_closes(ROOM, now=T0 + 62) == []


def test_no_duplicate_close_either_order():
    # cycle first: the member's next /math shows no close note
    h = _handler()
    h.note_room(ADA, ROOM)
    h.handle("/math easy", ADA, now=T0)
    closes = h.peek_room_closes(ROOM, now=T0 + 61)
    h.ack_room_closes(ROOM, closes)
    res = h.handle("/math streak", ADA, now=T0 + 62)
    assert "notes" not in res
    # pull first: the /math sweep settles + notes the close, and
    # the room drain must not produce it a second time
    h2 = _handler()
    h2.note_room(ADA, ROOM)
    h2.handle("/math easy", ADA, now=T0)
    swept = h2.sweep_rounds(now=T0 + 61)
    assert [c["kind"] for c in swept] == ["round_close"]
    assert h2.peek_room_closes(ROOM, now=T0 + 61) == []
    res = h2.handle("/math streak", ADA, now=T0 + 62)
    assert [n["kind"] for n in res["notes"]] == ["round_close"]
    res2 = h2.handle("/math streak", ADA, now=T0 + 63)
    assert "notes" not in res2  # delivered exactly once


def test_roomless_round_falls_back_to_held_note():
    h = _handler()  # no room stamp: DM play
    h.handle("/math easy", ADA, now=T0)
    assert h.peek_room_closes(ROOM, now=T0 + 61) == []
    assert _note_kinds(h, ADA) == ["round_close"]


def test_fallback_converts_buffered_closes_to_notes():
    h = _handler()
    h.note_room(ADA, ROOM)
    h.handle("/math easy", ADA, now=T0)
    payloads = h.peek_room_closes(ROOM, now=T0 + 61)
    assert [c["kind"] for c in payloads] == ["round_close"]
    h.fallback_room_closes(ROOM)  # the room refused the post
    assert h.peek_room_closes(ROOM, now=T0 + 61) == []
    res = h.handle("/math streak", ADA, now=T0 + 62)
    assert [n["kind"] for n in res["notes"]] == ["round_close"]


def test_duel_close_rides_the_drain():
    h = _handler()
    h.note_room(ADA, ROOM)
    h.note_room(BOB, ROOM)
    h.handle("/math duel discord:bob", ADA, now=T0)
    closes = h.peek_room_closes(ROOM, now=T0 + 301)
    assert [c["kind"] for c in closes] == ["duel_close"]
    assert _note_kinds(h, ADA) == [] and _note_kinds(h, BOB) == []
    h.ack_room_closes(ROOM, closes)
    assert h.sweep_rounds(now=T0 + 302) == []


def test_duel_close_posts_to_each_players_room():
    h = _handler()
    h.note_room(ADA, ROOM)
    h.note_room(BOB, ROOM2)  # a cross-platform duel: two rooms
    h.handle("/math duel discord:bob", ADA, now=T0)
    a = h.peek_room_closes(ROOM, now=T0 + 301)
    b = h.peek_room_closes(ROOM2, now=T0 + 301)
    assert [c["kind"] for c in a] == ["duel_close"]
    assert [c["kind"] for c in b] == ["duel_close"]
    assert _note_kinds(h, ADA) == [] and _note_kinds(h, BOB) == []
    # one room refusing notes only ITS player; the other still acks
    h.fallback_room_closes(ROOM2)
    assert _note_kinds(h, BOB) == ["duel_close"]
    assert _note_kinds(h, ADA) == []
    h.ack_room_closes(ROOM, a)
    assert h.peek_room_closes(ROOM, now=T0 + 302) == []


def test_buffered_room_close_survives_persistence():
    h = _handler()
    h.note_room(ADA, ROOM)
    h.handle("/math easy", ADA, now=T0)
    h.peek_room_closes(ROOM, now=T0 + 61)  # buffered, never acked
    h2 = load_state(dump_state(h))
    closes = h2.peek_room_closes(ROOM, now=T0 + 62)
    assert [c["kind"] for c in closes] == ["round_close"]
    assert _note_kinds(h2, ADA) == []


# ── runtime rig (after test_group_system_output_gate) ───────────

BOT_KEY = "telegram-bot:-100123"
USER_KEY = "telegram:-100999"
PLAYER = "telegram:7"


class _Chat:
    def __init__(self, platform, key, title="Builders"):
        self.platform = platform
        self.key = key
        self.chat_id = key.split(":", 1)[1]
        self.title = title
        self.kind = ChatKind.GROUP


class _Msg:
    def __init__(self, chat, **kw):
        self.chat = chat
        self.meta = kw.get("meta") or {}
        self.sender_id = kw.get("sender_id", "8")
        self.sender = kw.get("sender", "Bob")
        self.sender_username = kw.get("sender_username", "bob")
        self.text = kw.get("text", "hi")
        self.message_id = kw.get("message_id", "55")


class _Adapter:
    def __init__(self):
        self.calls = []
        self._bot_id = 4242

    def admin_member(self, chat, user_id):
        status = ("administrator"
                  if str(user_id) == str(self._bot_id) else "member")
        return {"ok": True, "result": {"status": status}}


class _Gateway:
    def __init__(self, adapters):
        self.adapters = adapters
        self.sent = []

    def send(self, platform, chat, text, **kw):
        self.sent.append((platform, str(
            getattr(chat, "key", chat)), text))
        return SimpleNamespace(ok=True, message_id="1")


class _Runtime(RuntimeGroupMixin, RuntimeGamesMixin):
    def __init__(self, tmp_path, adapters, operators=()):
        self.sent = []
        self.gateway = _Gateway(adapters)
        self.operators = set(operators)
        self.context = SimpleNamespace(settings=SimpleNamespace(
            workspace_dir=str(tmp_path)))

    def _send(self, chat, text, **kw):
        self.sent.append(text)
        return SimpleNamespace(ok=True, message_id="1")

    def _is_operator(self, message):
        return str(message.sender_id or "") in self.operators

    def turn(self, msg):
        return self._group_surface_turn(msg)


def _rig(tmp_path, platform="telegram-bot", key=BOT_KEY):
    adapter = _Adapter()
    rt = _Runtime(tmp_path, {platform: adapter}, operators={"99"})
    chat = _Chat(platform, key)
    return rt, adapter, chat


def _rt_handler() -> MathCommandHandler:
    eco = MathEconomy()
    eco.verify_member(PLAYER, fingerprint="dev-7")
    return MathCommandHandler(MathClass(), eco)


def _seed_expired_round(rt, key) -> MathCommandHandler:
    """A quickfire round started in `key` two minutes ago: its
    60s clock is already dead in real time."""
    handler = _rt_handler()
    rt._math_handler_obj = handler
    rt._math_save = lambda h: None
    handler.note_room(PLAYER, key)
    handler.handle("/math easy", PLAYER, now=time.time() - 120)
    return handler


def _closes(sent) -> list[str]:
    return [t for t in sent if "Timed round closed" in t]


# ── runtime: the cycle posts the close ───────────────────────────


def test_cycle_posts_close_into_claimed_room(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    handler = _seed_expired_round(rt, BOT_KEY)
    rt.turn(_Msg(chat))  # an ordinary message: a cycle pass
    closes = _closes(rt.sent)
    assert len(closes) == 1
    assert "The answer was" in closes[0]
    assert "Worked solution:" in closes[0]
    # the next pass posts nothing more, and the member's own
    # /math carries no duplicate close note
    rt.turn(_Msg(chat, message_id="56", text="again"))
    assert _closes(rt.sent) == closes
    res = handler.handle("/math streak", PLAYER)
    assert "notes" not in res


def test_cycle_in_unclaimed_room_posts_nothing_note_survives(
        tmp_path):
    rt, adapter, chat = _rig(tmp_path, platform="telegram",
                             key=USER_KEY)
    handler = _seed_expired_round(rt, USER_KEY)
    rt.turn(_Msg(chat))
    assert _closes(rt.sent) == []
    assert rt.sent == []  # no system output abroad at all
    # the member still gets the close — as a held note on their
    # next /math, exactly the pre-drain behavior
    res = handler.handle("/math streak", PLAYER)
    assert [n["kind"] for n in res["notes"]] == ["round_close"]


def test_cycle_in_quiet_room_falls_back_to_held_note(tmp_path):
    rt, adapter, chat = _rig(tmp_path)
    handler = _seed_expired_round(rt, BOT_KEY)
    surface = rt._group_surface()
    rt._defense_update(surface, BOT_KEY, "_root", autonomy=False)
    rt.turn(_Msg(chat))
    assert _closes(rt.sent) == []  # §31 quiet: the room stays silent
    res = handler.handle("/math streak", PLAYER)
    assert [n["kind"] for n in res["notes"]] == ["round_close"]
