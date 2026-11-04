"""WhatsApp live-attach, round two: the cold-start hole.

The owner's live failure (2026-11-04, after walive shipped): he
pulled, restarted, and WhatsApp STILL never attached — no reply
anywhere.  The walive reconcile existed and worked, but its only
automatic driver was the 60s housekeeping beat inside
``_personal_surface_turn`` — and that turn returns False
immediately unless the inbound message's platform is "whatsapp"
or "telegram" (the userbot).  In his deployment (config platforms
= telegram BOTS only) bot messages carry a bot platform, and
WhatsApp cannot message without an adapter: the beat effectively
fired only on inbound personal-Telegram traffic, which never
came.  Nothing attached at cold start; only /pair and
/start whatsapp (which call the reconcile directly) worked.

The fix, pinned here:

1.  Startup attach — ``PartnerRuntime.start`` reconciles once,
    right after the gateway goes live: a linked, bridge-running
    WhatsApp account raises its conversational surface with ZERO
    inbound messages.  A failing attach never breaks the boot.
2.  Beat reach — the housekeeping beat (supervision sweep +
    live-attach reconcile + story sweep) is one shared,
    60s-throttled implementation (``_housekeeping_beat``), driven
    from ``_process_inner`` so ANY inbound message — bot
    platforms included — can drive it.  The personal-surface
    observation logic (status posts, delete events, DM cache)
    stays gated to personal platforms exactly as before.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.social.chat.base import (
    ChatAdapter,
    ChatKind,
    ChatMessage,
    ChatRef,
    SendResult,
)
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.pairing import BridgeSupervisor, PairingRegistry

BASE_PORT = 18911


# ── fakes (the pairing suites' idiom) ───────────────────────────────

class FakeProc:
    """A bridge process that parks alive until terminated."""

    def __init__(self, lines=()):
        self.terminated = False
        self._poll = None
        self._gate = threading.Event()
        self.stdout = self._stream(list(lines))

    def _stream(self, lines):
        yield from lines
        self._gate.wait(30)

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0
        self._gate.set()


def make_supervisor(tmp_path, base_port=BASE_PORT, lines=()):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc(lines)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        base_port=base_port,
        binary_resolver=lambda: "/staged/whatsapp-bridge")
    return sup, registry, spawns


def make_settings(tmp_path, *, whatsapp_enabled=True):
    chat = SimpleNamespace(
        whatsapp_enabled=whatsapp_enabled,
        whatsapp_host="127.0.0.1",
        whatsapp_port=8787,  # the static config endpoint — NOT the
        whatsapp_outbox_dir="",  # record's port; the adapter must
        telegram_api_id="", telegram_api_hash="")  # follow the pool
    return SimpleNamespace(
        chat=chat, workspace_dir=str(tmp_path),
        resolve=lambda rel: Path(tmp_path) / rel)


class CountingProxy:
    """Delegates everything to the wrapped organ, counting calls
    to the named methods.  Installed INTO the shared service
    bundle (svc["supervision"] / svc["personal"]), so the real
    beat and the real personal turn both run through it."""

    def __init__(self, target, counts, names):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_counts", counts)
        object.__setattr__(self, "_names", frozenset(names))

    def __getattr__(self, name):
        target = object.__getattribute__(self, "_target")
        attr = getattr(target, name)
        if name in object.__getattribute__(self, "_names") \
                and callable(attr):
            counts = object.__getattribute__(self, "_counts")

            def counted(*a, **k):
                counts[name] = counts.get(name, 0) + 1
                return attr(*a, **k)

            return counted
        return attr

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_target"), name, value)


# ── 1. startup attach (full runtime, the E2E idiom) ─────────────────

class _FakeRouter:
    def chat(self, messages, params=None, **kw):
        from nomorals.llm.base import LLMResponse

        return LLMResponse(text="mhm.", model="fake-7b")


class _FakeAdapter(ChatAdapter):
    """A parked local adapter (LocalAdapter would read stdin)."""

    def __init__(self, name="local"):
        super().__init__(media_dir="/tmp/nm-test-media")
        self.name = name
        self.sent: list = []

    def run(self, handler) -> None:
        self._handler = handler
        while not self.stopped:
            time.sleep(0.01)

    def send(self, chat, text, *, reply_to="", buttons=None,
             parse_mode=""):
        self.sent.append(text)
        return SendResult(ok=True, platform=self.name,
                          message_id=f"m{len(self.sent)}")


def _make_context(tmp_path):
    """A real offline context (test_partner_runtime's builder,
    trimmed): partner disabled, local platform only, and the
    pulse/briefing catch-up slots pre-seeded so start() fires
    nothing into the adapter on a fresh workspace."""
    from nomorals.agents.context import build_context
    from nomorals.core.config import load_settings

    settings = load_settings(overrides={
        "home": str(tmp_path),
        "partner.platforms": "local",
        "chat.local_enabled": "true",
        "partner.enabled": "false",
    })
    context = build_context(settings, with_executor=False,
                            with_tools=False)
    context.router = _FakeRouter()
    ws = Path(context.settings.workspace_dir)
    for marker in ("pulse_last_run.json",
                   "pulse_last_run_afternoon.json",
                   "pulse_last_run_evening.json",
                   "pulse_last_run_midnight.json"):
        (ws / marker).write_text(
            json.dumps({"ts": time.time(), "delivered_text": True}),
            encoding="utf-8")
    from nomorals.agents.morning_briefing import _today_str

    context.db.execute(
        "INSERT OR REPLACE INTO briefings"
        " (id, date, sections_json, generated_at)"
        " VALUES (?, ?, '[]', ?)",
        ("test-seed", _today_str(context), time.time()))
    return context


def _full_runtime(tmp_path):
    context = _make_context(tmp_path)
    gateway = ChatGateway({"local": _FakeAdapter()}, db=context.db)
    rt = PartnerRuntime(context, gateway=gateway)
    return rt, context


def test_startup_attach_cold_start_no_inbound(tmp_path):
    """(a) A linked, bridge-running account + runtime.start() +
    ZERO inbound messages → the WhatsApp surface is up, on the
    ACCOUNT's bridge port (not the static config endpoint)."""
    sup, registry, _ = make_supervisor(tmp_path)
    rec = sup.pair_whatsapp("2348012345678")
    registry.set_status(rec.id, "linked")
    assert rec.port == BASE_PORT
    rt, context = _full_runtime(tmp_path)
    rt._pair_supervisor = sup  # the pool from the previous run
    try:
        started = rt.start()
        assert "local" in started
        adapter = rt.gateway.adapters.get("whatsapp")
        assert adapter is not None
        assert adapter.port == BASE_PORT
        assert adapter._thread is not None \
            and adapter._thread.is_alive()
        assert rt._wa_surface_account == rec.id
    finally:
        rt.stop()
        context.close()


def test_startup_no_whatsapp_accounts_no_adapter(tmp_path):
    """(d) Cold start with an empty pool: no adapter, no error —
    start() completes exactly as it did before the attach."""
    rt, context = _full_runtime(tmp_path)
    try:
        started = rt.start()
        assert "local" in started
        assert "whatsapp" not in rt.gateway.adapters
        assert getattr(rt, "_wa_surface_account", "") == ""
    finally:
        rt.stop()
        context.close()


def test_startup_attach_failure_never_breaks_start(tmp_path):
    """(e) The attach raising at startup is recorded in the
    failure ledger under its own site — and the boot completes."""
    from nomorals.agents.partner.failures import recent_failures

    rt, context = _full_runtime(tmp_path)

    def _boom():
        raise RuntimeError("boom-attach")

    rt._sync_whatsapp_surface = _boom
    try:
        started = rt.start()  # must return, not raise
        assert "local" in started
        hits = [f for f in recent_failures()
                if f.site == "runtime.PartnerRuntime.start:wa_attach"
                and "boom-attach" in f.message]
        assert hits, "startup attach failure was not recorded"
    finally:
        rt.stop()
        context.close()


# ── 2. beat reach + personal gating (bare runtime harness) ──────────

def _beat_runtime(tmp_path):
    """PartnerRuntime on a bare harness: real provision mixin,
    real gateway, the deep pipeline past the beat shadowed off —
    a channel message returns from _process_inner right after
    the personal turn, so _process can be driven end to end."""
    rt = PartnerRuntime.__new__(PartnerRuntime)
    settings = make_settings(tmp_path)
    rt.settings = settings
    rt.context = SimpleNamespace(settings=settings)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    rt._bump = lambda *a, **k: None
    rt._harvest_play_downloads = lambda *a, **k: None
    rt._drive_webshot_watches = None
    rt.channel_notes: list = []
    rt._channel_note_and_drive = rt.channel_notes.append
    # The shared §15 bundle, with counting proxies installed so
    # both the beat and the personal turn run through them.
    counts: dict = {}
    svc = rt._pairing_personal()
    svc["personal"].ensure_primary("whatsapp")
    svc["supervision"] = CountingProxy(
        svc["supervision"], counts, {"sweep"})
    svc["personal"] = CountingProxy(
        svc["personal"], counts,
        {"cycle_stories", "observe_message", "observe_status_post",
         "observe_delete", "observe_edit"})
    real_sync = rt._sync_whatsapp_surface

    def counted_sync():
        counts["sync"] = counts.get("sync", 0) + 1
        return real_sync()

    rt._sync_whatsapp_surface = counted_sync
    rt.beat_counts = counts
    rt.beat_svc = svc
    return rt


def _msg(platform, kind, *, text="", sender_id="", peer="",
         message_id="", meta=None):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=peer or "chat1",
                     kind=kind, title="", peer=peer),
        incoming=True, text=text, sender="",
        sender_id=sender_id, message_id=message_id,
        media=[], meta=meta or {}, ts=time.time())


def test_bot_platform_message_drives_housekeeping_beat(tmp_path):
    """(b) An inbound telegram-bot message — a platform the
    personal turn rejects at its gate — drives the beat through
    _process: sweep + reconcile + story cycle, once per 60s
    window, and the throttle is the SHARED one (rewinding it
    lets the very next message drive the beat again)."""
    rt = _beat_runtime(tmp_path)
    counts = rt.beat_counts

    rt._process(_msg("telegram-bot", ChatKind.CHANNEL,
                      peer="chan1"))
    assert counts.get("sweep") == 1
    assert counts.get("sync") == 1
    assert counts.get("cycle_stories") == 1
    assert len(rt.channel_notes) == 1  # the message flowed on

    # Inside the window: processed, but the beat does not re-run.
    rt._process(_msg("telegram-bot", ChatKind.CHANNEL,
                      peer="chan1"))
    assert counts.get("sweep") == 1
    assert counts.get("sync") == 1
    assert counts.get("cycle_stories") == 1
    assert len(rt.channel_notes) == 2

    # Window lapsed: the next bot message drives it again.
    rt.beat_svc["last_cycle"] = 0.0
    rt._process(_msg("telegram-bot", ChatKind.CHANNEL,
                      peer="chan1"))
    assert counts.get("sweep") == 2
    assert counts.get("sync") == 2
    assert counts.get("cycle_stories") == 2
    assert len(rt.channel_notes) == 3


def test_personal_observation_stays_personal_gated(tmp_path):
    """(c) Widening the beat did NOT widen observation: a
    telegram-bot message is never observed (and the personal
    turn's own gate still turns it away before even the beat),
    while personal-platform traffic is observed exactly as
    before — and the personal turn still drives the shared beat
    for direct callers, throttled against the same state."""
    rt = _beat_runtime(tmp_path)
    counts = rt.beat_counts

    # Bot surface: turned away at the gate — no observation,
    # and not even the beat runs from inside the personal turn.
    bot = _msg("telegram-bot", ChatKind.DM, text="hi",
               message_id="b1")
    assert rt._personal_surface_turn(bot) is False
    assert counts.get("observe_message", 0) == 0
    assert counts.get("sweep", 0) == 0

    # WhatsApp DM: observed, flows on — and the turn's own beat
    # call fires (first window), as it did before the move.
    dm = _msg("whatsapp", ChatKind.DM, text="meet at 7",
              sender_id="111", peer="111@s.whatsapp.net",
              message_id="m1")
    assert rt._personal_surface_turn(dm) is False
    assert counts.get("observe_message") == 1
    assert counts.get("sweep") == 1
    assert counts.get("sync") == 1

    # Status post on the personal surface: consumed + observed;
    # the beat is already inside its window — no double sweep.
    post = _msg("whatsapp", ChatKind.DM, sender_id="222",
                peer="222", message_id="s1",
                meta={"service": "status_post"})
    assert rt._personal_surface_turn(post) is True
    assert counts.get("observe_status_post") == 1
    assert counts.get("sweep") == 1

    # DM delete on the personal surface: consumed + observed.
    delete = _msg("whatsapp", ChatKind.DM, sender_id="111",
                  peer="111@s.whatsapp.net",
                  meta={"service": "delete", "deleted_ids": ["m1"]})
    assert rt._personal_surface_turn(delete) is True
    assert counts.get("observe_delete") == 1
    assert counts.get("sweep") == 1
