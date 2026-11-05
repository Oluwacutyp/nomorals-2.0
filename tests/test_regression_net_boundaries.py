"""Boundary pins for the three 2026-11-04 fix nets.

The primary nets shipped with their fixes and are mutation-proven
(2026-11-04 burn-down pass: reintroducing each bug shape in the
production seam fails the net — strikefix 10/16, wager 21 tests,
slotstart 8 + 2 across its two seams):

* strikefix (0c67784) — tests/test_group_strikefix.py
* wager     (43ae6be) — tests/test_wager_integrity.py
* slotstart (bb1679d) — tests/test_slots_quickstart.py

What those nets did NOT pin are the boundary conditions each fix
introduced at its edges.  This file pins exactly those, one class
per fix:

* strikefix — the manual-ban check stands even for an exempt or
  uncounted sender (the exemption covers AUTOMATED stages only);
  an uncounted message still faces the link stage (uncounted
  skips counting stages, not content stages); the game-input
  probe refuses §33-frozen rooms, defers to the relay matcher in
  relay rooms, and answers False when the probe itself fails.
* wager — the purse boundary itself: AI seats and seat-less
  calls never touch the backend (no balance, no debit, no
  credit — house spectacle never moves money); a human debit is
  all-or-nothing and journals under the composed game/room tag
  and ref; non-positive amounts never reach the backend.
* slotstart — roulette is untouched by the amount grammar: an
  amount tail names no outcome bet, so the grammar kept it at
  0 arguments — "/roulette 500" is refused at parse, and the
  plain "/roulette" start escrows nothing.
"""
from __future__ import annotations

from types import SimpleNamespace

from nomorals.community.welcome_moderation import (
    ModerationConfig,
    Moderator,
)


# ══ strikefix boundaries ═══════════════════════════════════════

class TestStrikefixBoundaries:

    def test_manual_ban_stands_for_exempt_and_uncounted_senders(self):
        # The fix's own contract: exempt skips the AUTOMATED
        # stages; the ban check "enforces a manual decision, it
        # is not an automated stage" and runs BEFORE the verdicts.
        mod = Moderator(config=ModerationConfig(
            flood_window_s=10.0, flood_max=3, repeat_threshold=3,
            repeat_window_s=300.0, word_filters=["badword"],
            filter_action="delete"))
        mod.ban_member("g", "boss", reason="manual decision",
                       banned_by="owner")
        for kwargs in ({"exempt": True}, {"uncounted": True}, {}):
            acts = mod.check_message("g", "boss", "badword spam",
                                     ts=1000.0, **kwargs)
            assert [a.kind for a in acts] == ["ban"], kwargs
            assert acts[0].rule == "banned"
        # control: an un-banned exempt sender is untouched, as
        # the primary net pins.
        assert mod.check_message("g", "other", "badword spam",
                                 ts=1000.0, exempt=True) == []

    def test_uncounted_still_faces_the_link_stage(self):
        # uncounted skips the COUNTING stages (flood, repeats);
        # the link stage is a content stage and still evaluates.
        mod = Moderator(config=ModerationConfig(
            link_allowlist=["example.com"]))
        acts = mod.check_message(
            "g", "u1", "look https://scam-site.com/free now",
            ts=1000.0, uncounted=True)
        assert [a.kind for a in acts] == ["delete"]
        assert acts[0].rule == "link_allowlist"
        # an allowlisted link passes, counted or not.
        assert mod.check_message(
            "g", "u1", "look https://example.com/page",
            ts=1001.0, uncounted=True) == []


# ── the game-input probe's edges (real engine, stub surfaces) ──

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.engine import GameEngine
from nomorals.games.players import Player
from nomorals.storage.db import Database

KEY = "telegram-bot:-1001234567"
ADA = Player.from_sender("telegram", "111", "Ada")


def _make_engine() -> GameEngine:
    db = Database(":memory:")
    db.migrate()
    return GameEngine(
        SimpleNamespace(db=db), send=lambda chat, text: None)


class _Probe(RuntimeGamesMixin):
    """The games mixin over a real engine; the group surface is
    an empty store, so no room reads as §33-blocked by default."""

    def __init__(self, engine):
        self._engine = engine
        self.context = SimpleNamespace(db=None)

    def _game_engine(self):
        return self._engine

    def _group_surface(self):
        return SimpleNamespace(store=SimpleNamespace(data={}))


from nomorals.games.games.base import MultiGame


class _RelayVocabGame(MultiGame):
    """A relayed game with a fixed move vocabulary: casual chat
    is NOT a move (the strikefix net's _VocabGame shape)."""

    name = "relay-vocab"
    description = "relay probe test double"
    min_players = 1
    max_players = 2
    ai_seats = 0

    def is_move_text(self, text: str) -> bool:
        return (text or "").strip().lower() in {"hit", "pass"}

    def on_move(self, room, player, text, mind):
        return []


class TestStrikefixProbeBoundaries:

    def test_probe_refuses_a_system_blocked_room(self):
        engine = _make_engine()
        probe = _Probe(engine)
        engine.start(KEY, "slots", ADA, kind="group")
        # control: unblocked, the seated player's input probes in
        assert probe._is_game_input(KEY, "spin", player=ADA,
                                    kind="group") is True
        # a §33-frozen room earns no exemption, seat or not
        probe._group_system_blocked = lambda key: True
        assert probe._is_game_input(KEY, "spin", player=ADA,
                                    kind="group") is False

    def test_probe_defers_to_the_relay_matcher(self):
        engine = _make_engine()
        engine.register(_RelayVocabGame())
        probe = _Probe(engine)
        relay_room = SimpleNamespace(game_name="relay-vocab")
        probe._game_relay = lambda: SimpleNamespace(
            get_relay_for_chat=lambda key: relay_room
            if key == KEY else None)
        # no engine room exists and ADA holds no seat anywhere:
        # the verdict is the relay matcher's alone (a closed-
        # vocabulary game, so casual chat is decisively not a
        # move — slots' open vocabulary could not show this).
        assert engine.live(KEY) is None
        assert probe._is_game_input(KEY, "hit", player=ADA,
                                    kind="group") is True
        assert probe._is_game_input(KEY, "just chatting",
                                    player=ADA, kind="group") \
            is False
        # slash engine commands ride the relay; /game is the
        # control plane and never does.
        assert probe._is_game_input(KEY, "/status", player=ADA,
                                    kind="group") is True
        assert probe._is_game_input(KEY, "/game quit", player=ADA,
                                    kind="group") is False

    def test_probe_failure_answers_false(self):
        # an exemption is earned, never assumed: a probe whose
        # engine seam blows up must not exempt anyone.
        class _Boom(RuntimeGamesMixin):
            def __init__(self):
                self.context = SimpleNamespace(db=None)

            def _game_engine(self):
                raise RuntimeError("engine seam down")

            def _game_relay(self):
                return SimpleNamespace(
                    get_relay_for_chat=lambda key: None)

            def _group_surface(self):
                return SimpleNamespace(
                    store=SimpleNamespace(data={}))

        probe = _Boom()
        assert probe._is_game_input(KEY, "spin", player=ADA,
                                    kind="group") is False


# ══ wager boundaries ═══════════════════════════════════════════

from nomorals.games.purse import BoundPurse


class _RecordingBackend:
    name = "test purse"

    def __init__(self, debit_ok: bool = True):
        self.calls: list = []
        self._debit_ok = debit_ok

    def balance(self, player):
        self.calls.append(("balance",))
        return 1_000

    def debit(self, player, amount, reason, ref):
        self.calls.append(("debit", amount, reason, ref))
        return self._debit_ok

    def credit(self, player, amount, reason, ref):
        self.calls.append(("credit", amount, reason, ref))
        return amount


class TestWagerPurseBoundaries:

    def _purse(self, backend) -> BoundPurse:
        return BoundPurse(
            backend, SimpleNamespace(game="slots", id="room-7"))

    def test_ai_and_seatless_calls_never_touch_the_backend(self):
        # the fix's invariant: AI seats have NO purse, so
        # house-played spectacle tables never move money.
        backend = _RecordingBackend()
        purse = self._purse(backend)
        ai = SimpleNamespace(is_ai=True)
        assert purse.balance(ai) == 0
        assert purse.debit(ai, 200, "bet") is False
        assert purse.credit(ai, 200, "payout") == 0
        assert purse.balance(None) == 0
        assert purse.debit(None, 200, "bet") is False
        assert purse.credit(None, 200, "payout") == 0
        assert backend.calls == []

    def test_human_debit_is_all_or_nothing_and_tagged(self):
        backend = _RecordingBackend()
        purse = self._purse(backend)
        human = SimpleNamespace(is_ai=False)
        assert purse.debit(human, 200, "bet") is True
        # every movement journals under the game + room identity
        assert backend.calls == [
            ("debit", 200, "game:slots bet", "wager:slots:room-7")]
        # a backend refusal is the whole story: False, and the
        # game layer above treats it as "nothing moved"
        backend2 = _RecordingBackend(debit_ok=False)
        purse2 = self._purse(backend2)
        assert purse2.debit(human, 200, "bet") is False
        assert backend2.calls == [
            ("debit", 200, "game:slots bet", "wager:slots:room-7")]
        assert purse2.credit(human, 350, "payout") == 350
        assert backend2.calls[-1] == (
            "credit", 350, "game:slots payout", "wager:slots:room-7")

    def test_non_positive_amounts_never_reach_the_backend(self):
        backend = _RecordingBackend()
        purse = self._purse(backend)
        human = SimpleNamespace(is_ai=False)
        assert purse.debit(human, 0, "bet") is False
        assert purse.debit(human, -50, "bet") is False
        assert purse.credit(human, 0, "payout") == 0
        assert purse.credit(human, -50, "payout") == 0
        assert backend.calls == []


# ══ slotstart boundaries ═══════════════════════════════════════

from nomorals.social.chat.control import parse_control


def _dispatch_verb(text: str) -> str:
    """The any-chat verb building in PartnerRuntime._process
    (the quickstart net's helper shape)."""
    parsed = parse_control(text)
    assert parsed is not None, f"{text} did not parse"
    return (parsed.tail or parsed.arg) if parsed.kind == "game" \
        else parsed.kind + ((" " + parsed.tail) if parsed.tail else "")


def _call_game(engine, text, chat_key, player, kind):
    rt = RuntimeGamesMixin.__new__(RuntimeGamesMixin)
    rt.context = SimpleNamespace()
    rt._game_engine = lambda: engine
    rt._group_system_blocked = lambda key: False
    return rt._control_game(
        _dispatch_verb(text), chat_key=chat_key, player=player,
        kind=kind)


def _wallet_rows(db, key):
    rows = db.execute(
        "SELECT amount, reason FROM game_wallet WHERE player_key = ?",
        (key,)).fetchall()
    return [(r[0], r[1]) for r in rows]


class TestSlotstartBoundaries:

    def test_roulette_amount_tail_is_inert(self):
        # the fix's stated boundary: only slots and blackjack
        # read a tail word as a stake — "an amount names no
        # outcome bet" in roulette.  The grammar kept roulette
        # at 0 arguments (slots/blackjack grew to (0, 1)), so
        # the amount form is refused at parse and no table or
        # money ever moves; the plain form opens a table whose
        # start escrows nothing.
        parsed = parse_control("/roulette 500")
        assert parsed is not None and parsed.kind == "error"
        assert "at most 0 argument" in parsed.arg
        db = Database(":memory:")
        db.migrate()
        engine = GameEngine(
            SimpleNamespace(db=db), send=lambda chat, text: None)
        try:
            engine.store.add_coins(ADA, 500, reason="seed funding")
            _call_game(engine, "/roulette", "test:rn2", ADA, "dm")
            room = engine.live("test:rn2")
            assert room is not None and room.game == "roulette"
            assert room.state.get("escrow", 0) == 0
            assert engine.store.get_for(ADA).coins == 500
            assert _wallet_rows(db, ADA.key) == [
                (500, "seed funding")]
        finally:
            engine.shutdown()
