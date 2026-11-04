"""The moderation self-strike fix: staff are never auto-moderated,
and a seated player's game inputs are not spam.

The live defect (owner's own group, Devon admin): the owner typed
"Bet 20000" to play slots and the Moderator's repeat stage warned
him ("rule broken: repeated" — even the phrase fell back, the rule
key was misaligned), then DELETED his next game message.  Two
blindnesses caused it: check_message had no sender-role awareness,
and no notion that the text was an input to a game the bot itself
was running — playing slots REQUIRES repeated short inputs.

Pins here:
* the Moderator's new ``exempt`` / ``uncounted`` verdicts (organ
  level, defaults = historical behaviour);
* the surface forwarding them (privileged sender flags, the
  caller's game_input verdict);
* the games seam's probe (_is_game_input) against REAL room
  state — a live slots room, seated vs non-seated, the game's own
  move vocabulary, engine commands vs the /game control plane;
* the runtime turn wiring the probe into handle_message;
* ordinary members are NOT defanged: repeat still warns then
  deletes, flood still mutes, word filters still delete — and the
  warning now renders the house phrase, rule key "repeat".
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nomorals.community.welcome_moderation import (
    ModerationConfig,
    Moderator,
)
from nomorals.social.chat.group_defense import rule_phrase
from nomorals.social.chat.group_surface import GroupStore, GroupSurface

KEY = "telegram-bot:-1001234567"
NOW = 1_700_000_000.0


def _surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


# ── the Moderator organ: exempt / uncounted ─────────────────────

def _tight_moderator() -> Moderator:
    return Moderator(config=ModerationConfig(
        flood_window_s=10.0, flood_max=3, repeat_threshold=3,
        repeat_window_s=300.0, word_filters=["badword"],
        filter_action="delete"))


def test_moderator_exempt_sender_is_never_auto_moderated():
    mod = _tight_moderator()
    # word filter, repeats and a flood — an exempt sender earns
    # no action from any automated stage.
    assert mod.check_message("g", "boss", "badword here",
                             ts=1000.0, exempt=True) == []
    for i in range(6):
        assert mod.check_message("g", "boss", "same",
                                 ts=1001.0 + i, exempt=True) == []
    # …and the exempt traffic was never COUNTED either: the same
    # sender, judged as an ordinary member now, starts from zero.
    assert mod.check_message("g", "boss", "same", ts=1010.0) == []
    assert mod.check_message("g", "boss", "same", ts=1070.0) == []
    third = mod.check_message("g", "boss", "same", ts=1130.0)
    assert third and third[0].kind == "warn"


def test_moderator_uncounted_is_not_counted_by_counting_stages():
    mod = _tight_moderator()
    # a slots session's worth of identical inputs, uncounted:
    # no warn, no delete, no flood mute.
    for i in range(6):
        assert mod.check_message("g", "u1", "spin",
                                 ts=2000.0 + i * 0.5,
                                 uncounted=True) == []
    # the bucket starts from zero afterwards — three COUNTED
    # repeats are needed before the warn, exactly as for a
    # member who never sent the uncounted ones.
    assert mod.check_message("g", "u1", "spin", ts=2100.0) == []
    assert mod.check_message("g", "u1", "spin", ts=2160.0) == []
    third = mod.check_message("g", "u1", "spin", ts=2220.0)
    assert third and third[0].kind == "warn"
    # content stages still evaluate an uncounted message: the
    # word filter is not a counting stage.
    assert mod.check_message("g", "u2", "badword post",
                             ts=2300.0, uncounted=True)[0].kind \
        == "delete"


def test_moderator_repeat_rule_key_is_repeat():
    mod = _tight_moderator()
    assert mod.check_message("g", "u1", "buy now", ts=3000.0) == []
    assert mod.check_message("g", "u1", "buy now", ts=3060.0) == []
    third = mod.check_message("g", "u1", "buy now", ts=3120.0)
    assert third and third[0].kind == "warn"
    assert third[0].rule == "repeat"
    fourth = mod.check_message("g", "u1", "buy now", ts=3180.0)
    assert fourth and fourth[0].kind == "delete"
    assert fourth[0].rule == "repeat"


def test_rule_phrase_alignment_and_legacy_alias():
    assert rule_phrase("repeat") == \
        "repeating the same message isn't allowed"
    # stored records from before the alignment carry "repeated"
    # and must render the same house phrase, never the fallback.
    assert rule_phrase("repeated") == rule_phrase("repeat")


# ── the surface: privileged senders ─────────────────────────────

def test_owner_slots_session_earns_no_warning_no_delete(tmp_path):
    surface = _surface(tmp_path)
    session = ["Bet 20000", "spin", "Bet 20000", "spin",
               "Bet 20000", "spin"]
    for i, text in enumerate(session):
        out = surface.handle_message(
            KEY, "1", name="Peacethefirst", text=text,
            message_id=f"s{i}", ts=NOW + i * 5,
            sender_is_operator=True)
        assert out["actions"] == [], (text, out["actions"])
    # a burst inside the flood window is likewise untouched
    for i in range(8):
        out = surface.handle_message(
            KEY, "1", name="Peacethefirst", text=f"note {i}",
            message_id=f"f{i}", ts=NOW + 100 + i,
            sender_is_operator=True)
        assert out["actions"] == []
    assert surface.strikes_for(KEY, "1", now=NOW + 200)["count"] == 0


def test_admin_repeats_and_flood_are_exempt_too(tmp_path):
    surface = _surface(tmp_path)
    for i in range(4):
        out = surface.handle_message(
            KEY, "2", name="Ada", text="same", message_id=f"a{i}",
            ts=NOW + i * 60, sender_is_admin=True)
        assert out["actions"] == []
    assert surface.strikes_for(KEY, "2", now=NOW + 300)["count"] == 0


def test_word_filter_spares_admin_still_deletes_member(tmp_path):
    surface = _surface(tmp_path)
    surface.update_settings(KEY, moderation={
        "flood_max": 6, "flood_window_s": 10.0,
        "repeat_threshold": 3, "link_allowlist": [],
        "word_filters": ["badword"], "filter_action": "delete"})
    out = surface.handle_message(
        KEY, "2", name="Ada", text="badword in a mod note",
        message_id="w1", ts=NOW, sender_is_admin=True)
    assert out["actions"] == []
    out = surface.handle_message(
        KEY, "3", name="Bo", text="badword post",
        message_id="w2", ts=NOW)
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    assert out["actions"][0]["rule"] == "word_filter"


def test_anticrash_still_deletes_an_admin_payload(tmp_path):
    # the anticrash organ is a different pipeline with its own
    # shipped rule for staff: deleted for safety, never struck.
    # The stage-2 exemption must not leak into it.
    surface = _surface(tmp_path)
    out = surface.handle_message(
        KEY, "2", name="Ada", text="a" * 13000, message_id="x1",
        ts=NOW, sender_is_admin=True)
    assert [a["kind"] for a in out["actions"]] == ["delete"]
    assert surface.strikes_for(KEY, "2", now=NOW)["count"] == 0


# ── the surface: the game-input verdict ─────────────────────────

def test_game_input_flag_exempts_counting_only(tmp_path):
    surface = _surface(tmp_path)
    # the seated player's session: identical inputs, uncounted.
    for i in range(6):
        out = surface.handle_message(
            KEY, "5", name="Cy", text="spin", message_id=f"g{i}",
            ts=NOW + i * 5, game_input=True)
        assert out["actions"] == []
    # nothing was counted: one ordinary "spin" afterwards is a
    # first offence, not a seventh.
    out = surface.handle_message(
        KEY, "5", name="Cy", text="spin", message_id="g6",
        ts=NOW + 600)
    assert out["actions"] == []


def test_same_text_from_a_non_seated_member_still_counts(tmp_path):
    surface = _surface(tmp_path)
    first = surface.handle_message(KEY, "6", name="Di",
                                   text="spin", message_id="n1",
                                   ts=NOW)
    second = surface.handle_message(KEY, "6", name="Di",
                                    text="spin", message_id="n2",
                                    ts=NOW + 60)
    third = surface.handle_message(KEY, "6", name="Di",
                                   text="spin", message_id="n3",
                                   ts=NOW + 120)
    assert first["actions"] == [] and second["actions"] == []
    assert third["actions"][0]["kind"] == "warn"
    assert third["actions"][0]["rule"] == "repeat"
    warning = third["actions"][0].get("warning") or ""
    assert "repeating the same message isn't allowed" in warning
    assert "rule broken" not in warning
    fourth = surface.handle_message(KEY, "6", name="Di",
                                    text="spin", message_id="n4",
                                    ts=NOW + 180)
    assert fourth["actions"][0]["kind"] == "delete"
    assert fourth["actions"][0]["rule"] == "repeat"


# ── the games seam: the probe against real room state ───────────

from nomorals.agents.partner.runtime_games import RuntimeGamesMixin
from nomorals.games.engine import GameEngine
from nomorals.games.games.base import MultiGame
from nomorals.games.players import Player
from nomorals.storage.db import Database


class _Ctx:
    def __init__(self, db):
        self.db = db


def _make_engine() -> GameEngine:
    db = Database(":memory:")
    db.migrate()
    return GameEngine(_Ctx(db), send=lambda chat, text: None)


class _VocabGame(MultiGame):
    """A game with a fixed move vocabulary (the pvp shape):
    casual chat is NOT a move."""

    name = "vocab-duel"
    description = "probe test double"
    min_players = 1
    max_players = 2
    ai_seats = 0

    def is_move_text(self, text: str) -> bool:
        return (text or "").strip().lower() in {"hit", "pass"}

    def on_move(self, room, player, text, mind):
        return []


class _Probe(RuntimeGamesMixin):
    """The games mixin over a real engine; the group surface is
    an empty store, so no room reads as §33-blocked."""

    def __init__(self, engine):
        self._engine = engine
        self.context = SimpleNamespace(db=None)

    def _game_engine(self):
        return self._engine

    def _group_surface(self):
        return SimpleNamespace(store=SimpleNamespace(data={}))


ADA = Player.from_sender("telegram", "111", "Ada")
BOB = Player.from_sender("telegram", "222", "Bob")


def test_probe_seated_slots_player_inputs():
    engine = _make_engine()
    probe = _Probe(engine)
    engine.start(KEY, "slots", ADA, kind="group")
    assert engine.live(KEY) is not None
    assert probe._is_game_input(KEY, "spin", player=ADA,
                                kind="group") is True
    assert probe._is_game_input(KEY, "Bet 20000", player=ADA,
                                kind="group") is True
    # engine slash commands are the room's own; /game is the
    # control plane and other slash commands are not moves.
    assert probe._is_game_input(KEY, "/status", player=ADA,
                                kind="group") is True
    assert probe._is_game_input(KEY, "/game quit", player=ADA,
                                kind="group") is False
    assert probe._is_game_input(KEY, "/mood", player=ADA,
                                kind="group") is False


def test_probe_requires_a_seat_and_a_room():
    engine = _make_engine()
    probe = _Probe(engine)
    # no room anywhere yet
    assert probe._is_game_input(KEY, "spin", player=ADA,
                                kind="group") is False
    engine.start(KEY, "slots", ADA, kind="group")
    # Bob never took a seat: his identical text is ordinary chat.
    assert probe._is_game_input(KEY, "spin", player=BOB,
                                kind="group") is False
    # a different chat has no table at all
    assert probe._is_game_input("telegram-bot:-100999", "spin",
                                player=ADA, kind="group") is False
    # channels never carry player moves; no player, no verdict
    assert probe._is_game_input(KEY, "spin", player=ADA,
                                kind="channel") is False
    assert probe._is_game_input(KEY, "spin", player=None,
                                kind="group") is False


def test_probe_respects_the_games_move_vocabulary():
    engine = _make_engine()
    engine.register(_VocabGame())
    probe = _Probe(engine)
    engine.start(KEY, "vocab-duel", ADA, kind="group")
    assert probe._is_game_input(KEY, "hit", player=ADA,
                                kind="group") is True
    # casual chat at a vocabulary game's table is not an input
    assert probe._is_game_input(KEY, "say that again?",
                                player=ADA, kind="group") is False


def test_probe_verdict_drives_the_surface(tmp_path):
    # the composed path the live bug travelled, with the fix:
    # real room state → probe verdict → surface counting.
    engine = _make_engine()
    probe = _Probe(engine)
    engine.start(KEY, "slots", ADA, kind="group")
    surface = _surface(tmp_path)
    for i in range(4):
        verdict = probe._is_game_input(KEY, "spin", player=ADA,
                                       kind="group")
        out = surface.handle_message(
            KEY, "111", name="Ada", text="spin",
            message_id=f"p{i}", ts=NOW + i * 5,
            game_input=verdict)
        assert out["actions"] == []
    # Bob's identical text probes False (no seat) and counts.
    for i in range(3):
        verdict = probe._is_game_input(KEY, "spin", player=BOB,
                                       kind="group")
        assert verdict is False
        out = surface.handle_message(
            KEY, "222", name="Bob", text="spin",
            message_id=f"q{i}", ts=NOW + 100 + i * 60,
            game_input=verdict)
    assert out["actions"][0]["kind"] == "warn"
    assert out["actions"][0]["rule"] == "repeat"


# ── the runtime turn: wiring ────────────────────────────────────

from nomorals.agents.partner.runtime_group import RuntimeGroupMixin
from nomorals.social.chat.base import ChatKind


class _Chat:
    platform = "telegram-bot"
    chat_id = "-1001234567"
    title = "Builders"
    key = KEY
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


class _Adapter:
    def __init__(self):
        self.calls = []

    def admin_delete(self, chat, message_id):
        self.calls.append(("delete", message_id))
        return {"ok": True}

    def admin_ban(self, chat, user_id):
        self.calls.append(("ban", user_id))
        return {"ok": True}

    def admin_restrict(self, chat, user_id, *, until=0, **kw):
        self.calls.append(("restrict", user_id, until))
        return {"ok": True}

    def admin_member(self, chat, user_id):
        return {"ok": True, "result": {"status": "member"}}


class _Gateway:
    def __init__(self, adapter):
        self.adapters = {"telegram-bot": adapter}


class _TurnRuntime(RuntimeGroupMixin):
    """RuntimeGroupMixin on a stub: send + gateway only (the
    test_group_surface rig shape)."""

    def __init__(self, tmp_path, adapter):
        self.sent = []
        self.gateway = _Gateway(adapter)
        self.context = type("C", (), {
            "settings": type("S", (), {
                "workspace_dir": str(tmp_path)})()})()

    def _send(self, chat, text, **kw):
        self.sent.append(text)

    def _is_operator(self, message):
        return False

    def turn(self, msg):
        return self._group_surface_turn(msg)


class _GameTurnRuntime(_TurnRuntime):
    """The same stub with the games seam attached: every text
    the probe is asked about is a seated player's input."""

    def _game_player(self, message):
        return object()

    def _is_game_input(self, key, text, *, player=None, kind=""):
        return True


def test_turn_without_games_seam_moderates_as_before(tmp_path):
    adapter = _Adapter()
    rt = _TurnRuntime(tmp_path, adapter)
    # warn on the third identical message (not consumed)…
    for i in range(3):
        consumed = rt.turn(_Msg(text="same",
                                message_id=f"t{i}"))
    assert consumed is False
    assert adapter.calls == []
    # …delete on the fourth, exactly as today
    consumed = rt.turn(_Msg(text="same", message_id="t3"))
    assert ("delete", "t3") in adapter.calls
    assert consumed is True


def test_turn_with_games_seam_never_moderates_game_input(tmp_path):
    adapter = _Adapter()
    rt = _GameTurnRuntime(tmp_path, adapter)
    for i in range(4):
        consumed = rt.turn(_Msg(text="spin",
                                message_id=f"g{i}"))
        assert consumed is False
    assert adapter.calls == []
