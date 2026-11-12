"""Dupfix2 — cross-surface duplication in the owner's Telegram DM
(2026-11-11 screen recording + owner correction 11:37).

The recording shows the same core-mind status blocks and wordchain
leaderboards posted twice — once per surface, under two identities —
and the owner reports: "Any command I use user bot replies it too
right there in the DM with personal bot."

Root cause (proved by the probes in dupfix2-verify, not assumed):
the shipped arbitration (``command_claims``) is a *shared-state*
mechanism — a durable claim ledger plus a peer rule fed by
configured companion bot ids. It is exactly-once whenever the
surfaces share a home store, or the userbot was built with both bot
tokens (both shapes probed green). It does NOT hold in the live
shape: the incident record itself documents the live userbot
carrying no companion ids, so a second surface claims in a store
the first never sees while its peer rule has no inputs. And
arbitration only ever covered COMMANDS at two checkpoints — a
bot's posts entering the second surface as turns fed its game and
coremind engines, which is how identical status/leaderboard outputs
were produced by the second surface at all.

The fix pinned here works when the surfaces share nothing: in a
bot's DM, a non-bot surface DEFERS a command and watches the room
for the primary's answer (the MTProto stream the userbot always
has), failing over only when no answer ever comes; a bot's post in
a DM is an observation, never a turn.

The live shape is replayed exactly: two runtimes, SEPARATE homes,
the userbot adapter WITHOUT companion ids (the documented live
shape) but WITH the real adapter's ``peer_is_bot`` stamp on its
echoes — the signal that survives split hosting with no shared
config.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from nomorals.agents.context import build_context
from nomorals.agents.partner.failures import clear_failures
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888


class RecordingAdapter(ChatAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list[tuple[str, Any]] = []

    def run(self, handler) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


class BotAdapter(RecordingAdapter):
    def __init__(self, name: str, bot_id: int, username: str) -> None:
        super().__init__(name)
        self._bot_id = bot_id
        self._bot_username = username


class BareUserbotAdapter(RecordingAdapter):
    """The split-hosting userbot: its own account identity, and
    NO companion bot ids — the documented live shape, whose peer
    rule had no inputs. The peer-is-bot stamp (new) is what the
    runtime reads instead."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


class UserbotWithCompanions(RecordingAdapter):
    """Same userbot, built with both bot tokens (the shape where
    the old peer rule already had its inputs)."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)
    companion_bot_ids = frozenset({PERSONAL_BOT_ID})


def _context(home: str):
    settings = load_settings(overrides={
        "home": home,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    ws = Path(context.settings.workspace_dir)
    for marker in ("pulse_last_run.json",
                   "pulse_last_run_afternoon.json",
                   "pulse_last_run_evening.json",
                   "pulse_last_run_midnight.json"):
        (ws / marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    context.db.execute(
        "INSERT OR REPLACE INTO briefings (id, date, sections_json,"
        " generated_at) VALUES (?, ?, '[]', ?)",
        ("dupfix2", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, peer_is_bot=False,
         service=None, reply_to=""):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=ChatKind.DM,
                   peer=chat_id)
    meta: dict = {"is_owner": True}
    if own:
        meta["own_message"] = True
        meta["from_owner"] = True
    if peer_is_bot:
        meta["peer_is_bot"] = True
    if service:
        meta["service"] = service
        meta["observe_only"] = True
    if reply_to:
        meta["reply_to"] = reply_to
    msg = ChatMessage(chat=chat, incoming=incoming, text=text,
                      sender="death", message_id=message_id, meta=meta,
                      reply_to=reply_to)
    msg.sender_id = sender_id
    return msg


def _bot_copy(text, message_id):
    """The Bot API view of the owner's message in the personal
    bot's DM."""
    return _msg("telegram-personal", chat_id=OWNER_ID, text=text,
                incoming=True, message_id=message_id)


def _echo_copy(text, message_id):
    """The MTProto own-echo view of the same message (the real
    adapter's shape: incoming=False, own_message, peer_is_bot
    stamped from the peer entity)."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=False, message_id=message_id, own=True,
                peer_is_bot=True)


def _peer_post(text, message_id, *, reply_to="", service="peer_post"):
    """The bot's own post in that DM, as the userbot stream sees
    it — delivered as an observation."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=True, message_id=message_id,
                sender_id=str(PERSONAL_BOT_ID), service=service,
                reply_to=reply_to)


@pytest.fixture()
def shared_claims_file(monkeypatch):
    """One temp shared-claims file for the test (groupdup2):
    the runtimes' claims must not leak into the real default
    shared file, or into another test's claims."""
    tmp = tempfile.TemporaryDirectory(prefix="nm-d2-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    monkeypatch.setenv("NM_SHARED_CLAIMS", path)
    yield path
    tmp.cleanup()


@pytest.fixture()
def split_homes(shared_claims_file):
    """Two runtimes, SEPARATE homes — the live shape. Claims
    live in the per-test shared file (groupdup2)."""
    clear_failures()
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d2a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d2b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
    rt_b._peer_answer_window_s = 5.0
    yield (rt_a, ad_a, ctx_a), (rt_b, ad_b, ctx_b)
    for runtime, _adapters, context in ((rt_a, ad_a, ctx_a),
                                        (rt_b, ad_b, ctx_b)):
        try:
            runtime.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        context.close()
    tmp_a.cleanup()
    tmp_b.cleanup()


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def test_split_homes_observation_resolves_deferral(split_homes):
    """The incident, replayed in the live shape: the bot answers,
    the echo defers, the bot's post observed in the room resolves
    the deferral — the userbot sends nothing, ever."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    text = "/chatmode games-only +2348164302080"
    rt_a._process_inner(_bot_copy(text, "100"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    rt_b._process_inner(_echo_copy(text, "100"))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 1
    # The primary's answer, arriving on the userbot's own stream:
    rt_b._process_inner(_peer_post(
        "whatsapp:2348164302080@s.whatsapp.net is now games-only",
        "101", reply_to="100"))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    # And it stays resolved: the sweep has nothing to fail over.
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["telegram"]) == []


def test_split_homes_failover_emits_once(split_homes):
    """The primary never answers (bot down): the deferred echo
    fails over and handles exactly once — silence from both is
    the failure this closes."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    text = "/chatmode games-only +2348164302080"
    rt_b._process_inner(_echo_copy(text, "100"))
    assert _texts(ad_b["telegram"]) == []
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    sent = _texts(ad_b["telegram"])
    assert len(sent) == 1, sent
    assert "games-only" in sent[0]
    # A second sweep never re-emits.
    rt_b._sweep_peer_deferrals(now=time.time() + 120)
    assert len(_texts(ad_b["telegram"])) == 1


def test_split_homes_mind_status_once(split_homes):
    """The recorded status-block duplication: /mind status on
    both copies yields exactly one core-mind block, from the
    primary."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_bot_copy("/mind status", "110"))
    rt_b._process_inner(_echo_copy("/mind status", "110"))
    rt_b._process_inner(_peer_post("🧠 core mind", "111",
                                   reply_to="110"))
    mine = [t for t in _texts(ad_a["telegram-personal"])
            if "core mind" in t]
    assert len(mine) == 1, mine
    assert _texts(ad_b["telegram"]) == []


def test_bot_post_is_never_a_turn(split_homes):
    """The second duplication source: a bot's post arriving on
    the userbot surface is an observation, never a game move /
    coremind turn / reply — via the adapter's observation shape,
    and via the turn law when the runtime knows the peer is a
    bot (companion ids)."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    # A live wordchain room in the bot's DM (bot copy only).
    rt_a._process_inner(_bot_copy("/wordchain", "200"))
    rt_a._process_inner(_bot_copy("yellow", "201"))
    assert len(_texts(ad_a["telegram-personal"])) >= 2
    # (a) The house's word, as the adapter delivers it — a
    # peer_post observation: no move, no reply, no new room.
    before = len(_texts(ad_b["telegram"]))
    rt_b._process_inner(_peer_post("years", "202"))
    assert len(_texts(ad_b["telegram"])) == before
    rooms = _ctx_b.db.query(
        "SELECT chat_key FROM game_rooms")
    assert [r["chat_key"] for r in rooms] == []
    # (b) The same word arriving as an ordinary incoming message
    # on a runtime that KNOWS the peer is a bot (companion ids):
    # the turn law converts it to an observation.
    clear_failures()
    tmp = tempfile.TemporaryDirectory(prefix="nm-d2c-",
                                      dir=_TEST_TMPDIR)
    ctx_c = _context(tmp.name)
    ad_c = {"telegram": UserbotWithCompanions("telegram")}
    rt_c = _runtime(ctx_c, ad_c)
    try:
        rt_c._process_inner(_msg(
            "telegram", chat_id=str(PERSONAL_BOT_ID), text="years",
            incoming=True, message_id="203",
            sender_id=str(PERSONAL_BOT_ID)))
        assert _texts(ad_c["telegram"]) == []
        rooms_c = ctx_c.db.query("SELECT chat_key FROM game_rooms")
        assert [r["chat_key"] for r in rooms_c] == []
    finally:
        try:
            rt_c.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        ctx_c.close()
        tmp.cleanup()


def test_stop_intent_mirrors_on_observation(split_homes):
    """The owner's stop stays sticky across both surfaces: the
    deferred /stop is answered by the primary, and the deferring
    surface mirrors the stop into ITS durable registry, so its
    own reconcile honors it too."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    assert rt_b._pairing_supervisor().registry.owner_stopped(
        "whatsapp") is False
    rt_b._process_inner(_echo_copy("/stop whatsapp", "120"))
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._process_inner(_peer_post("whatsapp: stopped", "121",
                                   reply_to="120"))
    assert rt_b._pairing_supervisor().registry.owner_stopped(
        "whatsapp") is True
    assert _texts(ad_b["telegram"]) == []


def test_superseded_deferral_never_emits(split_homes):
    """A newer owner command supersedes an older pending one: the
    sweep emits only the latest, never re-applying the replaced
    command."""
    (_rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_b._process_inner(_echo_copy(
        "/chatmode games-only +2348164302080", "130"))
    rt_b._process_inner(_echo_copy(
        "/chatmode normal +2348164302080", "131"))
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    sent = _texts(ad_b["telegram"])
    assert len(sent) == 1, sent
    assert "back to normal" in sent[0]


def test_one_runtime_sibling_still_instant(split_homes):
    """The intended shape is untouched: one runtime, both adapters
    on one gateway — the bot copy claims, the echo stands down
    AT ONCE (no deferral registered), exactly one reply."""
    clear_failures()
    tmp = tempfile.TemporaryDirectory(prefix="nm-d2one-",
                                      dir=_TEST_TMPDIR)
    ctx = _context(tmp.name)
    adapters = {
        "telegram-personal": BotAdapter(
            "telegram-personal", PERSONAL_BOT_ID, "De_von_bot"),
        "telegram": UserbotWithCompanions("telegram"),
    }
    rt = _runtime(ctx, adapters)
    try:
        text = "/chatmode games-only +2348164302080"
        rt._process_inner(_bot_copy(text, "140"))
        rt._process_inner(_echo_copy(text, "140"))
        assert len(_texts(adapters["telegram-personal"])) == 1
        assert _texts(adapters["telegram"]) == []
        assert rt._peer_deferral().pending_count() == 0
    finally:
        try:
            rt.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        ctx.close()
        tmp.cleanup()
