"""Telegram DM duplication loop (dupfix3) — 2026-11-12 owner video.

One plain message in the personal-bot DM became dozens of alternating
copies (bot replies + own-account re-sends). The diagnosis
(dupdiag-verify/DIAGNOSIS.md) proved the loop is structural: the
userbot's only defence against answering the bot's replies is a
peer-bot guard fed by its own settings' bot tokens; when that set is
empty both guards fail together, and plain-text turns have no
cross-surface arbitration anywhere. The 'post this' phrase in the
incident's message is an accelerant, not the cause: the override
fires above the claim seam and its canned reply contains the phrase.

Live updates folded in: DM COMMANDS duplicate too, and TELEGRAM
GROUPS duplicate too — both are pinned here as well. The fix must
hold in the Telegram DM and in Telegram groups, for plain text and
for commands, on both surfaces.

The live shape is replayed exactly: two runtimes, SEPARATE homes,
one shared claims file, the userbot WITHOUT companion ids.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
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
from nomorals.social.drafts import detect_explicit_post

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
GROUP_ID = "-100123456789"
CANNED = "what should I post? Send the text first, then say 'post this'."
SEED = "what should i post? Send the text first, then say 'post this'."


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
    """The split-hosting userbot: its own account identity, and NO
    companion bot ids — the documented live shape, whose peer rule
    had no inputs."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


class NoteTakingUserbotAdapter(BareUserbotAdapter):
    """Bare userbot that records registry pushes, like the real
    TelegramAdapter.note_companion_bot_ids."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.pushed: list[set] = []

    def note_companion_bot_ids(self, ids) -> bool:
        self.pushed.append(set(ids))
        return True


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
        ("dupfix3", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    return runtime


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, peer_is_bot=False,
         service=None, reply_to="", kind=ChatKind.DM):
    chat = ChatRef(platform=platform, chat_id=chat_id, kind=kind,
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


def _echo_copy(text, message_id, *, stamped=True):
    """The MTProto own-echo view of the same message. ``stamped``
    is the adapter's peer_is_bot stamp on the echo — True is the
    shape the old tests used; False (unstamped) is the live gap:
    the stamp is missing exactly when the companion set is empty,
    which is the documented live shape."""
    return _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=text,
                incoming=False, message_id=message_id, own=True,
                peer_is_bot=stamped)


def _peer_post(text, message_id, *, reply_to="", service="peer_post",
               chat_id=str(PERSONAL_BOT_ID), kind=ChatKind.DM):
    """The bot's own post in that chat, as the userbot stream sees
    it — delivered as an observation."""
    return _msg("telegram", chat_id=chat_id, text=text,
                incoming=True, message_id=message_id,
                sender_id=str(PERSONAL_BOT_ID), service=service,
                reply_to=reply_to, kind=kind)


def _group_msg(platform, text, message_id, *, sender_id=OWNER_ID):
    """The same group message as one surface sees it (platform
    differs per surface; chat id, sender and message id agree)."""
    return _msg(platform, chat_id=GROUP_ID, text=text,
                incoming=True, message_id=message_id,
                sender_id=sender_id, kind=ChatKind.GROUP)


@pytest.fixture()
def split_homes(monkeypatch):
    """Two runtimes, SEPARATE homes — the live shape. Claims live
    in the per-test shared file (groupdup2). HOME is pointed at a
    temp dir so DraftQueue's Path.home() path stays in tmp."""
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-d3-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    tmp = tempfile.TemporaryDirectory(prefix="nm-d3-shared-",
                                      dir=_TEST_TMPDIR)
    path = str(Path(tmp.name) / "claims.db")
    monkeypatch.setenv("NM_SHARED_CLAIMS", path)
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d3a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d3b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
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
    tmp.cleanup()
    home_tmp.cleanup()


@pytest.fixture()
def split_homes_broken_store(monkeypatch):
    """Same split homes, but the shared claims file cannot open
    (groupdup2's fallback shape): arbitration must fail to defer,
    never to double-handle — and a deferral must resolve by
    observation in the room."""
    clear_failures()
    home_tmp = tempfile.TemporaryDirectory(prefix="nm-d3-home-",
                                           dir=_TEST_TMPDIR)
    monkeypatch.setenv("HOME", home_tmp.name)
    # A path under a REGULAR FILE can never open: Database.open
    # creates missing parent directories, so a merely-missing dir
    # would open fine (caught in the first run). The blocker file
    # makes the store genuinely unreachable.
    blocker = os.path.join(_TEST_TMPDIR, "claims-blocker")
    with open(blocker, "w", encoding="utf-8") as fh:
        fh.write("not a directory\n")
    monkeypatch.setenv("NM_SHARED_CLAIMS",
                       os.path.join(blocker, "claims.db"))
    tmp_a = tempfile.TemporaryDirectory(prefix="nm-d3a-",
                                        dir=_TEST_TMPDIR)
    tmp_b = tempfile.TemporaryDirectory(prefix="nm-d3b-",
                                        dir=_TEST_TMPDIR)
    ctx_a = _context(tmp_a.name)
    ctx_b = _context(tmp_b.name)
    ad_a = {"telegram-personal": BotAdapter(
        "telegram-personal", PERSONAL_BOT_ID, "De_von_bot")}
    ad_b = {"telegram": BareUserbotAdapter("telegram")}
    rt_a = _runtime(ctx_a, ad_a)
    rt_b = _runtime(ctx_b, ad_b)
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
    home_tmp.cleanup()


def _texts(adapter):
    return [t for t, _b in adapter.sent]


# ── (b) the 'post this' override: mention vs instruction ──────────────

def test_detect_mention_does_not_fire():
    """A message that merely MENTIONS 'post this' inside a question
    is not an explicit-post instruction — the incident's message
    itself must not fire the override."""
    assert detect_explicit_post(SEED) is None
    assert detect_explicit_post("what should i post?") is None
    assert detect_explicit_post("please post this") is None
    assert detect_explicit_post("he said 'post this' and left") is None


def test_detect_instruction_fires():
    """A message that IS the instruction still fires — leading
    with the phrase, command-shaped."""
    assert detect_explicit_post("post this")["action"] == "post"
    assert detect_explicit_post("Post This")["action"] == "post"
    assert detect_explicit_post("post this: hello world")["action"] \
        == "post"
    assert detect_explicit_post("send it")["action"] == "post"
    assert detect_explicit_post("post it at 6")["action"] == "schedule"


def test_override_does_not_fire_on_mention(split_homes):
    """Runtime level: the incident's message runs the override and
    gets None — it falls through instead of minting the canned
    reply that re-fires the loop."""
    (rt_a, _ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    msg = _bot_copy(SEED, "10")
    assert rt_a._explicit_post_override(msg) is None
    assert rt_a._explicit_post_override(_bot_copy("post this", "11")) \
        is not None


# ── (a) the userbot step: bot-peer DM never becomes a turn ────────────

def test_adapter_peer_is_bot_dm_by_chat_id(tmp_path):
    """The DM peer is the chat id itself — known before any entity
    resolution, and independent of the .bot flag. Empty companion
    config means no peer is recognised; a pushed id is."""
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=1, api_hash="x", session_path=str(tmp_path / "t.session"),
        media_dir=str(tmp_path / "media"), companion_bot_ids=set())
    assert adapter._peer_is_bot_dm(str(PERSONAL_BOT_ID)) is False
    assert adapter._peer_is_bot_dm(OWNER_ID) is False
    assert adapter._peer_is_bot_dm(GROUP_ID) is False
    assert adapter.note_companion_bot_ids({PERSONAL_BOT_ID}) is True
    assert adapter._peer_is_bot_dm(str(PERSONAL_BOT_ID)) is True
    assert adapter._peer_is_bot_dm(OWNER_ID) is False
    assert adapter._peer_is_bot_dm(GROUP_ID) is False
    # A second push merges instead of replacing.
    assert adapter.note_companion_bot_ids({777}) is True
    assert adapter._peer_is_bot_dm("777") is True
    assert adapter._peer_is_bot_dm(str(PERSONAL_BOT_ID)) is True


def test_adapter_delivers_bot_group_post_as_observation(tmp_path):
    """A bot's post in a Telegram GROUP reaches the runtime as a
    peer_post observation (group kind) instead of being dropped at
    the adapter — the observation is what resolves a pending group
    deferral, and the drop is what left the 8s side answering
    alongside the 3s side."""
    from nomorals.social.chat.telegram import TelegramAdapter

    adapter = TelegramAdapter(
        api_id=1, api_hash="x", session_path=str(tmp_path / "t.session"),
        media_dir=str(tmp_path / "media"),
        companion_bot_ids={PERSONAL_BOT_ID})
    delivered: list[ChatMessage] = []

    def handler(message):
        delivered.append(message)

    event = SimpleNamespace(
        chat_id=GROUP_ID, sender_id=PERSONAL_BOT_ID, sender=None,
        message=SimpleNamespace(id=55, raw_text="pong",
                                sender_id=PERSONAL_BOT_ID))
    asyncio.run(adapter._handle_inbound(
        event, None, handler, allow_check=True))
    assert len(delivered) == 1, delivered
    obs = delivered[0]
    assert (obs.meta or {}).get("service") == "peer_post"
    assert (obs.meta or {}).get("peer_is_bot") is True
    assert obs.chat.kind == ChatKind.GROUP
    assert obs.text == "pong"


def test_bot_peer_dm_never_a_turn_on_bare_userbot(split_homes):
    """(a) The loop's userbot step, in the live shape: the bot's
    reply arrives on the bare userbot (no companion ids, no
    peer_is_bot stamp — an ordinary incoming message) and must
    become an observation, never a turn. The registry sync is
    what gives the runtime the bot id without settings tokens."""
    (rt_a, _ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_a._sync_bot_registry()
    white = _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=CANNED,
                 incoming=True, message_id="300",
                 sender_id=str(PERSONAL_BOT_ID))
    rt_b._process_inner(white)
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0


# ── (c) the override consults the claim seam ──────────────────────────

def test_override_reply_is_claimed(split_homes):
    """(c) An override reply is one conversational turn for one
    message identity: the byte-identical repeat inside the window
    is held, on the same runtime."""
    (rt_a, ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_bot_copy("post this", "400"))
    sent = _texts(ad_a["telegram-personal"])
    assert len(sent) == 1, sent
    assert "what should I post?" in sent[0]
    rt_a._process_inner(_bot_copy("post this", "401"))
    assert len(_texts(ad_a["telegram-personal"])) == 1


# ── (d) cross-surface conversational arbitration ──────────────────────

def test_conversation_claim_arbitrates_across_surfaces(split_homes):
    """(d) The shared conversational claim: the first surface to
    claim a (chat, sender, content-hash) turn owns it; the second
    surface — separate runtime, separate home, same shared file —
    stands down."""
    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_homes
    msg_a = _bot_copy("say hi", "500")
    msg_b = _bot_copy("say hi", "501")
    assert rt_a._conversation_already_claimed(msg_a) is False
    assert rt_b._conversation_already_claimed(msg_b) is True
    assert rt_a._conversation_already_claimed(
        _bot_copy("say something else", "502")) is False


def test_loop_replay_seed_white_green(split_homes):
    """(d) The full loop, replayed: seed → white copy on the
    userbot (held by the peer-bot rule) → green copy back on the
    bot (held by the shared conversational claim). One reply
    total, ever."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_a._sync_bot_registry()
    rt_a._process_inner(_bot_copy("post this", "600"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    white = _msg("telegram", chat_id=str(PERSONAL_BOT_ID), text=CANNED,
                 incoming=True, message_id="601",
                 sender_id=str(PERSONAL_BOT_ID))
    rt_b._process_inner(white)
    assert _texts(ad_b["telegram"]) == []
    green = _bot_copy("post this", "602")
    rt_a._process_inner(green)
    assert len(_texts(ad_a["telegram-personal"])) == 1
    assert _texts(ad_b["telegram"]) == []


# ── follow-up 1: DM commands duplicate too ────────────────────────────

def test_dm_command_answered_once_across_surfaces(split_homes):
    """A DM slash command with an UNSTAMPED echo (the live gap —
    the old test used a stamped echo): the bot answers once, the
    bare userbot defers, the bot's answer observed in the room
    resolves the deferral. Exactly one reply across both
    surfaces."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_a._sync_bot_registry()
    text = "/chatmode games-only +2348164302080"
    rt_a._process_inner(_bot_copy(text, "700"))
    assert len(_texts(ad_a["telegram-personal"])) == 1
    rt_b._process_inner(_echo_copy(text, "700", stamped=False))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._process_inner(_peer_post(
        "whatsapp:2348164302080@s.whatsapp.net is now games-only",
        "701", reply_to="700"))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["telegram"]) == []


# ── follow-up 2: Telegram groups duplicate too ────────────────────────

def test_group_plain_text_answered_once_across_surfaces(split_homes):
    """A plain-text message in a TELEGRAM group, seen by both
    surfaces with the same chat id, sender and words: the shared
    conversational claim lets exactly one surface answer."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_a._process_inner(_group_msg("telegram-personal", "post this",
                                   "800"))
    rt_b._process_inner(_group_msg("telegram", "post this", "800"))
    total = (len(_texts(ad_a["telegram-personal"]))
             + len(_texts(ad_b["telegram"])))
    assert total == 1, (_texts(ad_a["telegram-personal"]),
                        _texts(ad_b["telegram"]))


def test_group_command_answered_once_across_surfaces(split_homes):
    """A slash command in a TELEGRAM group, seen by both surfaces:
    the shared command claim lets exactly one surface answer."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = split_homes
    rt_b._process_inner(_group_msg("telegram", "/ping", "810"))
    rt_a._process_inner(_group_msg("telegram-personal", "/ping", "810"))
    assert len(_texts(ad_b["telegram"])) == 1, _texts(ad_b["telegram"])
    assert "pong" in _texts(ad_b["telegram"])[0]
    assert _texts(ad_a["telegram-personal"]) == []


def test_group_bot_post_observation_resolves_deferral(
        split_homes_broken_store):
    """The adapter fix at runtime level, in the fallback shape
    (shared store unreachable, so the group command DEFERS
    instead of claiming): a bot's group post arriving as a
    peer_post observation resolves a pending group deferral, so
    the 8s side never fails over into a second answer."""
    (rt_a, ad_a, _ctx_a), (rt_b, ad_b, _ctx_b) = \
        split_homes_broken_store
    assert rt_b._claims_are_shared() is False
    text = "/chatmode games-only +2348164302080"
    rt_b._process_inner(_group_msg("telegram", text, "820"))
    assert rt_b._peer_deferral().pending_count() == 1
    rt_b._process_inner(_peer_post(
        "whatsapp:2348164302080@s.whatsapp.net is now games-only",
        "821", reply_to="820", chat_id=GROUP_ID, kind=ChatKind.GROUP))
    assert _texts(ad_b["telegram"]) == []
    assert rt_b._peer_deferral().pending_count() == 0
    rt_b._sweep_peer_deferrals(now=time.time() + 60)
    assert _texts(ad_b["telegram"]) == []


# ── registry wiring ───────────────────────────────────────────────────

def test_registry_sync_registers_and_pushes(split_homes):
    """The shared bot-id registry: the bot runtime registers its
    id, the userbot runtime reads it (no settings tokens needed)
    and pushes it into its adapter's companion set."""
    from nomorals.agents.partner.command_claims import (
        load_bot_ids,
        open_shared_claims_db,
    )

    (rt_a, _ad_a, _ctx_a), (rt_b, _ad_b, _ctx_b) = split_homes
    rt_a._sync_bot_registry()
    db = open_shared_claims_db()
    assert db is not None
    assert str(PERSONAL_BOT_ID) in load_bot_ids(db)
    assert str(PERSONAL_BOT_ID) in rt_b._bot_account_ids()
    # The push reaches an adapter that accepts it.
    ad_c = {"telegram": NoteTakingUserbotAdapter("telegram")}
    ctx_c = _context(tempfile.mkdtemp(dir=_TEST_TMPDIR))
    rt_c = _runtime(ctx_c, ad_c)
    try:
        rt_c._sync_bot_registry()
        assert ad_c["telegram"].pushed, ad_c["telegram"].pushed
        pushed = set().union(*ad_c["telegram"].pushed)
        assert str(PERSONAL_BOT_ID) in {str(x) for x in pushed}
    finally:
        try:
            rt_c.stop()
        except Exception as exc:  # noqa: BLE001 - teardown only
            assert isinstance(exc, Exception)
        ctx_c.close()


def test_conversation_claim_ignores_different_words(split_homes):
    """No over-blocking: a genuinely different message in the same
    chat is a new turn, not a repeat."""
    (rt_a, _ad_a, _ctx_a), (_rt_b, _ad_b, _ctx_b) = split_homes
    assert rt_a._conversation_already_claimed(
        _bot_copy("say hi", "900")) is False
    assert rt_a._conversation_already_claimed(
        _bot_copy("say hello there", "901")) is False
