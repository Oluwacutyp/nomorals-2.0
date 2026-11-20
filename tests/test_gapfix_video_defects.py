"""Video-defects lane pins (owner screen recording + screenshot,
2026-11-19): the /play turn must never emit the money ask, the
open-threads digest must never report a missing tool the registry
actually has, /mock usage renders through the designed card, the
menu total equals the live registry count, group announcements
post on their own clock through the scheduler tool, the owner
surface holds when the bot's tick marker is live, an unkeyable
peer deferral stands down, and a bot post is recognised as a peer
post via a sibling adapter alone.

Each pin names the root it closes; the behaviour, not the words,
is asserted where the words are the LLM's.
"""
from __future__ import annotations

import contextlib
import json
import os
import random
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from nomorals.agents.context import build_context
from nomorals.agents.group_rhythm import (
    TICK_BUCKET_S,
    run_group_rhythm_tick_once,
)
from nomorals.agents.partner.brain import (
    PartnerBrain,
    _loop_claims_missing_tool,
)
from nomorals.agents.partner.runtime_mock_screens import (
    RuntimeMockScreensMixin,
)
from nomorals.agents.partner_runtime import PartnerRuntime
from nomorals.core.config import load_settings
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.control import CONTROL_COMMANDS, menu_sections
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.group_surface import (
    GroupStore,
    GroupSurface,
    default_surface,
)
from nomorals.social.chat.mock_screens.authoring import parse_mock_tail
from nomorals.tools.registry import ToolRegistry

_TEST_TMPDIR = os.environ.get("NM_TEST_TMPDIR") or None

OWNER_ID = "4242"
PERSONAL_BOT_ID = 888
GROUP_ID = "-100123456789"
GROUP_ID2 = "-100999999999"
KEY = "telegram-bot:-1001234567"


# ── shared harnesses (copied patterns, not cross-module imports) ──


def _context(home: str):
    settings = load_settings(overrides={
        "home": home,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
        "web.bot_username": "Nomorals_bot",
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
        ("video-defects", "2000-01-01", time.time()))
    return context


def _runtime(context, adapters):
    gateway = ChatGateway(adapters, db=context.db)
    runtime = PartnerRuntime(context, gateway=gateway)
    runtime.brain.presence_rng = random.Random(34)
    runtime._typing_enabled = lambda chat: False
    runtime._owner_sender_ids = (OWNER_ID,)
    return runtime


def _make_runtime(home, adapters):
    ctx = _context(home)
    return _runtime(ctx, adapters), ctx


def _stop(rt):
    with contextlib.suppress(Exception):  # noqa: BLE001 - teardown
        rt.stop()


class _Send:
    def __init__(self):
        self.calls: list = []

    def __call__(self, ref, caption=""):
        self.calls.append((ref, caption))
        return f"sent:{ref}"


class _MockRuntime(RuntimeMockScreensMixin):
    """The mock mixin over recording stubs for the live seams."""

    def __init__(self, tmp_path, operator=True):
        self._tmp = tmp_path
        self._operator = operator
        self.send = _Send()

    def _is_operator(self, message):
        return self._operator

    def _expression_root(self):
        return self._tmp

    def _makers_audience(self, message):
        return "personal"

    def _expression_send(self, message, ref, caption=""):
        return self.send(ref, caption)


def _mock_message():
    return SimpleNamespace(
        chat=SimpleNamespace(kind="dm"), sender="owner")


class _FakeLedger:
    """Shared claim ledger with the real contract
    (claim + claimed_by), thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._claims: dict[str, tuple[str, float]] = {}

    def claim(self, key: str, surface: str, *,
              now: float | None = None) -> bool:
        with self._lock:
            held = self._claims.get(key)
            if held is not None and held[0] != surface:
                return False
            self._claims[key] = (surface, time.time())
            return True

    def claimed_by(self, key: str, *,
                   now: float | None = None) -> str:
        with self._lock:
            held = self._claims.get(key)
            return held[0] if held else ""


class _FakeSend:
    """Captures tick sends: (platform, chat_key, text, buttons)."""

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    def __call__(self, platform: str, chat: Any, text: str, *,
                 buttons: Any = None, parse_mode: str = "") -> bool:
        self.sent.append({
            "platform": platform,
            "chat_key": getattr(chat, "key", str(chat)),
            "text": str(text),
            "buttons": buttons,
            "parse_mode": parse_mode,
        })
        return True

    def texts(self) -> list[str]:
        return [s["text"] for s in self.sent]


class RecordingAdapter(ChatAdapter):
    def __init__(self, name: str, tmp_path, bot_id=None) -> None:
        super().__init__(media_dir=str(Path(tmp_path) / "media"))
        self.name = name
        self.sent: list[tuple[str, Any]] = []
        if bot_id is not None:
            self._bot_id = bot_id

    def run(self, handler) -> None:
        pass

    def send(self, chat: ChatRef, text: str, *, reply_to: str = "",
             buttons: Any = None, reply_keyboard: Any = None,
             parse_mode: str = "") -> SendResult:
        self.sent.append((text, buttons))
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


class BotAdapter(RecordingAdapter):
    def __init__(self, name: str, tmp_path, bot_id: int,
                 username: str) -> None:
        super().__init__(name, tmp_path, bot_id=bot_id)
        self._bot_username = username


class BareUserbotAdapter(RecordingAdapter):
    """The split-hosting userbot: its own account identity, and NO
    companion bot ids — the documented live shape."""

    _me_username = "peacethefirst"
    _me_id = int(OWNER_ID)


def _msg(platform, *, chat_id, text, incoming, message_id,
         sender_id=OWNER_ID, own=False, peer_is_bot=False,
         service=None, kind=ChatKind.DM):
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
    msg = ChatMessage(chat=chat, incoming=incoming, text=text,
                      sender="death", message_id=message_id, meta=meta)
    msg.sender_id = sender_id
    return msg


def _echo_copy(text, message_id, *, chat_id=PERSONAL_BOT_ID,
               peer_is_bot=True):
    """The MTProto own-echo view of the owner's message in a bot DM
    (incoming=False)."""
    return _msg("telegram", chat_id=str(chat_id), text=text,
                incoming=False, message_id=message_id, own=True,
                peer_is_bot=peer_is_bot)


def _texts(adapter):
    return [t for t, _b in adapter.sent]


def _pending(runtime):
    return runtime._peer_deferral().pending_count()


# ── item 1: the money ask never fires without money intent ────────


def _finance_registry():
    from nomorals.finance.tools import register as register_finance
    reg = ToolRegistry(SimpleNamespace())
    register_finance(reg)
    return reg


def test_play_turn_never_emits_money_ask(monkeypatch, tmp_path):
    """The video's turn: the brain calls finance_send with an empty
    recipient after an unrelated /play. The tool refuses with a
    structured error — the slot-fill ask never fires."""
    monkeypatch.setenv("HOME", str(tmp_path))
    reg = _finance_registry()
    out = reg.call("finance_send",
                   context={"turn_text": "/play cheap thrills"},
                   to="", amount="5k")
    assert out.ok, out
    assert out.value.get("ok") is False, out.value
    assert "ask" not in out.value, out.value
    assert "No money transfer was requested" in str(
        out.value.get("error", "")), out.value


def test_send_5k_still_asks_for_recipient(monkeypatch, tmp_path):
    """The legitimate slot-fill survives: a turn that IS a transfer
    with an amount but no recipient still asks who to send it to."""
    monkeypatch.setenv("HOME", str(tmp_path))
    reg = _finance_registry()
    out = reg.call("finance_send",
                   context={"turn_text": "send 5k"},
                   to="", amount="5k")
    assert out.ok, out
    assert out.value.get("needs") == "recipient", out.value
    assert out.value.get("ask") == "who should I send it to?", out.value


def test_generate_with_tools_threads_turn_text():
    """The intent gate is only as good as its input:
    generate_with_tools must thread the turn's own text into
    loop_ctx as turn_text, so finance_send can tell a real
    transfer from the LLM's hallucinated args. The tool loop is
    stubbed — only the threading is pinned."""
    brain = object.__new__(PartnerBrain)
    captured: dict = {}

    class _FakeLoop:
        def run(self, *args, **kwargs):
            captured.update(kwargs.get("loop_ctx") or {})
            return SimpleNamespace(parts=[], trace=[])

    brain._tool_loop = lambda: _FakeLoop()
    chat = ChatRef(platform="telegram-bot", chat_id="123",
                   kind=ChatKind.DM, peer="123")
    msg = ChatMessage(chat=chat, incoming=True,
                      text="send 5k to ada", sender="4242")
    brain.generate_with_tools(msg)
    assert captured.get("turn_text") == "send 5k to ada", captured


def test_send_money_both_empty_is_error_not_ask(tmp_path):
    """Defence in depth, already true at the send_money layer: both
    recipient and amount empty is an error, never a question."""
    from nomorals.finance.ledger import Ledger
    from nomorals.finance.send import RecipientStore, send_money
    result = send_money(
        "", "",
        ledger=Ledger(tmp_path / "l.jsonl"),
        recipients=RecipientStore(tmp_path / "r.json"))
    assert result.get("ok") is False, result
    assert "ask" not in result, result
    assert "couldn't parse amount" in str(result.get("error", "")), result


# ── item 2: the digest never reports a missing tool that exists ───


def test_open_loop_stale_image_refusal_dropped(tmp_path):
    """The video's digest line claims no image generation tool —
    the live registry has image_generate and artist, so the loop
    is a stale refusal and is dropped at store time. A genuine
    missing-capability loop and a vague one are kept."""
    reg = ToolRegistry(SimpleNamespace())
    reg.register_builtins()
    assert len(reg.names()) >= 700, "the probe must see the real registry"
    lie = ("Image generation: I can't do that — I don't have an image "
           "generation tool, so that request stays unfulfilled on my end.")
    assert _loop_claims_missing_tool(lie, reg) is True
    assert _loop_claims_missing_tool(
        "I don't have a time-travel tool", reg) is False
    assert _loop_claims_missing_tool(
        "I don't have the right tool for this", reg) is False
    assert _loop_claims_missing_tool(lie, None) is False
    ctx = _context(str(tmp_path / "home"))
    try:
        brain = object.__new__(PartnerBrain)
        brain.context = SimpleNamespace(db=ctx.db, tools=reg)
        kept = brain._merge_open_loops(
            [lie, "Waiting on Mama to confirm the transfer"],
            "telegram-bot:123")
        assert not any("image generation tool" in t for t in kept), kept
        assert any("Mama" in t for t in kept), kept
    finally:
        ctx.close()


# ── item 3: /mock renders through the designed renderer ───────────


def test_bare_family_renders_its_demo():
    """/mock WhatsApp (a bare known family, no script) renders that
    family's demo screen — not the usage wall."""
    screen, mode = parse_mock_tail("WhatsApp")
    assert mode == "demo"
    assert screen.platform == "whatsapp"


def test_bare_family_with_style_renders_its_demo():
    screen, mode = parse_mock_tail("whatsapp iphone")
    assert mode == "demo"
    assert screen.platform == "whatsapp"


def test_mock_usage_error_renders_card(tmp_path):
    """A shape the compact form cannot carry (usage-shaped
    MockScreensError — an unknown family) renders the composed menu
    card at the _control_mock boundary — sections, not a dumped
    block. A more specific parse error keeps its own text."""
    runtime = _MockRuntime(tmp_path)
    reply = runtime._control_mock("mock", "myspace", _mock_message())
    assert "Mock screens" in reply
    assert "/mock demo whatsapp" in reply
    assert "/mock data" in reply
    assert not reply.startswith("usage: /mock")
    specific = runtime._control_mock(
        "mock", "whatsapp just some words", _mock_message())
    assert specific.startswith("each message segment"), specific[:80]


def test_mock_empty_tail_renders_card(tmp_path):
    runtime = _MockRuntime(tmp_path)
    reply = runtime._control_mock("mock", "", _mock_message())
    assert "Mock screens" in reply
    assert "/mock demo whatsapp" in reply


# ── item 4: the menu total equals the live registry count ─────────


def test_menu_total_equals_live_registry_count():
    """The menu total is computed live from the sections — it must
    equal the owner-gated command count, never a stale literal."""
    total = sum(len(items)
                for _emoji, _title, items in menu_sections("telegram"))
    assert total == len(CONTROL_COMMANDS), (
        f"menu total {total} != CONTROL_COMMANDS {len(CONTROL_COMMANDS)}")
    assert total > 0


# ── item 5: announcements post on their own clock ──────────────────


def _rhythm_surface(tmp_path: Path) -> GroupSurface:
    return GroupSurface(GroupStore(tmp_path / "g.json").load())


def test_tick_posts_through_scheduler_tool_with_live_gateway(
        monkeypatch, tmp_path):
    """The production wiring, end to end: the rhythm tick invoked
    through the scheduler tool, with a context carrying a live
    gateway, posts a due richest-balance record with NO message
    anywhere in the room."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(tmp_path / "shared-home"))
    ctx = _context(str(home))
    try:
        adapter = BotAdapter("telegram-bot", tmp_path, PERSONAL_BOT_ID,
                             "De_von_bot")
        gateway = ChatGateway({"telegram-bot": adapter}, db=ctx.db)
        ctx.extras["gateway"] = gateway
        ctx.gateway = gateway
        ws = Path(ctx.settings.workspace_dir)
        surface = default_surface(ws)
        surface.update_settings(KEY, rhythm={})
        surface.store.save()
        reg = ToolRegistry(ctx)
        from nomorals.tools import autonomy as autonomy_tools
        autonomy_tools.register(reg)
        # Seed the milestone book with a DIRECT tick (fake ledger):
        # the book baselines silently, and the real shared ledger
        # carries no tick marker yet — the tool tick below is the
        # first real tick, so its marker claim is fresh. (Two real
        # ticks seconds apart would collide on the 30s claim window
        # and the 60s rhythm throttle by design, not by defect.)
        seed = run_group_rhythm_tick_once(
            surface=surface, now=time.time(), send=_FakeSend(),
            claim=_FakeLedger(),
            adapters=[("telegram-bot", _FakeBotAdapter())])
        assert seed.get("groups", 0) >= 1, seed
        assert seed.get("posted", 0) == 0, seed
        # Reload: the seed tick saved the surface (book seeded, room
        # claimed) — the stale pre-seed object would overwrite it.
        surface = default_surface(ws)
        now = time.time()
        surface.economy.treasury_bonus(KEY, 100, "seed", now=now)
        surface.grant(KEY, "7", 100, by="41", now=now + 10)
        # Simulate two minutes passing past the rhythm throttle.
        surface._extras(KEY)["rhythm_cycle_at"] = time.time() - 120
        surface.store.save()
        out = reg.call("__group_rhythm_tick__")
        assert out.ok, out
        assert out.value.get("posted", 0) >= 1, out.value
        assert any("richest balance" in t for t in _texts(adapter)), (
            _texts(adapter))
    finally:
        ctx.close()


def test_tick_without_sender_records_the_failure(tmp_path):
    """A tick with no live sender used to skip every group silently
    — the summary now records the failure with its reason."""
    surface = _rhythm_surface(tmp_path)
    surface.update_settings(KEY, rhythm={})
    out = run_group_rhythm_tick_once(
        surface=surface, now=time.time(), send=None,
        claim=_FakeLedger(),
        adapters=[("telegram-bot", _FakeBotAdapter())])
    assert out.get("skipped", 0) >= 1, out
    assert "sender_error" in out, out
    assert "no live sender" in str(out["sender_error"]), out


class _FakeBotAdapter:
    _bot_id = 888


# ── item 6: the owner surface holds when the bot's tick is live ───


def test_owner_holds_when_bot_tick_marker_live(monkeypatch, tmp_path):
    """The bot's rhythm tick marker is live but no notice claim has
    landed (the bot is slow, not down): the owner surface HOLDS
    instead of sending as the userbot — and it holds at once, not
    after the 5s claim poll. Without the marker the same call
    sends (bot absent)."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("NM_SHARED_CLAIMS", str(tmp_path / "claims.db"))
    home = tmp_path / "rt-home"
    rt, ctx = _make_runtime(str(home), {
        "telegram": BareUserbotAdapter("telegram", tmp_path)})
    try:
        ledger = rt._shared_conversation_claims()
        assert ledger is not None
        chat = ChatRef(platform="telegram-personal", chat_id=GROUP_ID,
                       kind=ChatKind.GROUP)
        bucket = int(time.time() // TICK_BUCKET_S)
        marker = f"system:rhythm-tick:{chat.key}:{bucket}"
        assert ledger.claim(marker, "telegram-bot:888") is True
        started = time.monotonic()
        assert rt._claim_system_send(
            chat, "notice", "hello room", bot_wait_s=0.05) is False
        assert time.monotonic() - started < 1.5, (
            "a live tick marker must short-circuit the wait")
        chat2 = ChatRef(platform="telegram-personal", chat_id=GROUP_ID2,
                        kind=ChatKind.GROUP)
        assert rt._claim_system_send(
            chat2, "notice", "hello other room",
            bot_wait_s=0.05) is True
    finally:
        _stop(rt)
        ctx.close()


# ── item 7: an unkeyable peer deferral stands down ─────────────────


def test_unkeyable_peer_deferral_stands_down(monkeypatch, tmp_path):
    """The video's double, pinned on the shared arbitration branch
    (/ping stands in for /play — the hole is in the branch, not the
    media handler): the owner's /ping in the bot's DM arrives on the
    userbot as an INCOMING slash with NO message id, so the deferral
    cannot be registered. Before the fix the userbot handled it at
    once (the second answer); now it stands down — the peer owns
    the DM and its own intake handles the command. Nothing is
    registered and nothing is sent. (The own-echo shape is NOT the
    pin: the echo branch already stands unkeyable echoes down
    silently — the hole is the slash/operator branch's fall-through
    to handle.)"""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home_a = tmp_path / "a"
    home_a.mkdir()
    ad = {"telegram": BareUserbotAdapter("telegram", tmp_path)}
    rt, ctx = _make_runtime(str(home_a), ad)
    try:
        incoming = _msg("telegram", chat_id=str(PERSONAL_BOT_ID),
                        text="/ping", incoming=True, message_id="",
                        peer_is_bot=True)
        assert incoming.message_id == ""
        rt._process_inner(incoming)
        assert _texts(ad["telegram"]) == [], (
            "the unkeyable peer deferral must stand down, not handle")
        assert _pending(rt) == 0
    finally:
        _stop(rt)
        ctx.close()


# ── item 8: a bot post is recognised via a sibling adapter alone ──


def test_peer_identified_via_sibling_adapter_alone(monkeypatch,
                                                   tmp_path):
    """The registry is stale-empty (synced before the bot adapter
    attached) and the echo carries NO peer_is_bot stamp — the only
    layer that can identify the bot's post is the sibling bot
    adapter attached to this gateway. _peer_bot_dm must still
    return the peer, so the post never becomes a brain turn."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home_a = tmp_path / "a"
    home_a.mkdir()
    ad = {"telegram": BareUserbotAdapter("telegram", tmp_path)}
    rt, ctx = _make_runtime(str(home_a), ad)
    try:
        assert rt._bot_account_ids() == set()
        rt.gateway.adapters["telegram-personal"] = BotAdapter(
            "telegram-personal", tmp_path, PERSONAL_BOT_ID, "De_von_bot")
        echo = _echo_copy("/ping", "m8-1", peer_is_bot=False)
        assert echo.meta.get("peer_is_bot") is not True
        assert rt._peer_bot_dm(echo) == str(PERSONAL_BOT_ID)
        assert rt._peer_dm_owns_command(echo) is True
    finally:
        _stop(rt)
        ctx.close()


def test_bot_post_observed_not_turned_on(monkeypatch, tmp_path):
    """End to end for item 8: the bot's own post in the DM — the
    now-playing line the video shows — arrives as an ordinary
    incoming message with NO peer_is_bot stamp, and the registry
    is stale-empty. The sibling adapter identifies it, so the
    runtime observes it as a peer post and never turns the brain
    on it (the turn that produced the second audio and the
    money-send question)."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    home_a = tmp_path / "a"
    home_a.mkdir()
    ad = {"telegram": BareUserbotAdapter("telegram", tmp_path)}
    rt, ctx = _make_runtime(str(home_a), ad)
    try:
        assert rt._bot_account_ids() == set()
        rt.gateway.adapters["telegram-personal"] = BotAdapter(
            "telegram-personal", tmp_path, PERSONAL_BOT_ID, "De_von_bot")
        observed: list = []
        orig_observed = rt._peer_post_observed
        rt._peer_post_observed = lambda m: (
            observed.append(m), orig_observed(m))
        post = _msg("telegram", chat_id=str(PERSONAL_BOT_ID),
                    text="Cheap Thrills — now playing",
                    incoming=True, message_id="m8-2",
                    sender_id=str(PERSONAL_BOT_ID), peer_is_bot=False)
        rt._process_inner(post)
        assert len(observed) == 1, (
            "the bot's post must be observed as a peer post, "
            "never a turn")
        assert _texts(ad["telegram"]) == []
    finally:
        _stop(rt)
        ctx.close()
