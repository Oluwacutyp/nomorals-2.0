"""Pass 45 — wire the degradation ladders into production.

The ladders in ``core/degradation.py`` were declared at boot but dead:

1. ``build_error_system`` was called from ``agents/context.py`` WITHOUT
   the context, so every rung-0 handler resolved ``context.router`` /
   ``context.gateway`` to None and raised "no LLM router on context" /
   "no chat gateway on context" — and the llm probe was None.
2. Nothing in production called the ladders: the brain called the
   router directly, sends went through the gateway directly, voice
   tools ran directly. The ladders reported "not degraded" forever —
   true only because they were never exercised.
3. The telegram ladder's last rung queued into a ``tg_retry:`` KV store
   that NOTHING drained — its honesty clause ("will be delivered when
   the connection recovers") was false. It is now backed by the
   gateway's durable outbox (pass #38) whenever a gateway exists, and
   the legacy KV rows are drained into that outbox by the partner
   runtime at boot (:func:`drain_tg_retry_queue`).
4. The voice ladder's rung 0 was a raise-stub ("voice primary rung
   needs the TTS engine"). It now calls the real TTS entry point via an
   injected factory, and the partner voice-reply path routes through
   the ladder so a TTS failure degrades to text WITH the honesty note.

Each test below failed on pristine HEAD for the stated reason.
"""

from __future__ import annotations

import os
import shutil
import tempfile
from types import SimpleNamespace

import pytest

from nomorals.core.degradation import LadderExhausted
from nomorals.core.error_system import (
    build_error_system,
    drain_tg_retry_queue,
    set_error_system,
)
from nomorals.llm.base import LLMResponse, Message
from nomorals.llm.brain import Brain
from nomorals.social.chat.base import ChatAdapter, ChatRef, SendResult
from nomorals.social.chat.gateway import ChatGateway
from nomorals.storage.db import Database
from nomorals.storage.kv import KVStore


@pytest.fixture(autouse=True)
def _reset_global_error_system():
    set_error_system(None)
    yield
    set_error_system(None)


def _tmp_db() -> tuple[str, Database]:
    tmp = tempfile.mkdtemp(prefix="pass45-")
    db = Database(os.path.join(tmp, "test.db"))
    db.migrate()
    return tmp, db


# ── fake routers ────────────────────────────────────────────────────────

class _OkRouter:
    """Router double: chat returns a real LLMResponse."""

    def __init__(self) -> None:
        self.calls: list = []

    def chat(self, messages, params=None, **kw):
        self.calls.append((messages, params, kw))
        return LLMResponse(text="router says hi", model="fake-1")

    def _chain(self):
        return [object()]


class _DownRouter:
    """Router double: the whole provider chain is down."""

    def chat(self, messages, params=None, **kw):
        raise RuntimeError("all providers down: 503 from groq")

    def _chain(self):
        return []


class _ErrorResponseRouter:
    """Router double: total failure reported as a response, not a raise
    (the real LLMRouter's contract — pass #33)."""

    def chat(self, messages, params=None, **kw):
        return LLMResponse(text="", error="every provider failed: 401")

    def _chain(self):
        return []


class _FlakyRouter:
    """Down until healed; the probe reads ``_chain()``."""

    def __init__(self) -> None:
        self.down = True

    def chat(self, messages, params=None, **kw):
        if self.down:
            raise RuntimeError("provider chain unavailable")
        return {"ok": True, "text": "back online"}

    def _chain(self):
        return [] if self.down else [object()]


def _ctx(**kw):
    base = {"router": None, "db": None, "extras": {}}
    base.update(kw)
    return SimpleNamespace(**base)


def _message_factory(prompt):
    return [Message.user(prompt)]


# ── (f) pre-fix reproduction: the unwired shape fails honestly ─────────

def test_unwired_error_system_ladders_cannot_serve():
    """The OLD production call shape — build_error_system with no
    context — produces ladders whose rung 0s can only raise. This is
    the defect pass #45 fixes by passing the context in agents/context.
    """
    es = build_error_system(None)
    try:
        with pytest.raises(LadderExhausted) as llm_exc:
            es.ladders.get("llm").call("hello")
        llm_errors = " ".join(llm_exc.value.details.get("errors", []))
        assert "no LLM router on context" in llm_errors

        with pytest.raises(LadderExhausted) as tg_exc:
            es.ladders.get("telegram").call("telegram:1", "hello")
        tg_errors = " ".join(tg_exc.value.details.get("errors", []))
        assert "no chat gateway on context" in tg_errors

        # Voice: rung 0 has no engine (pre-fix it was a raise-stub;
        # post-fix an honest "no factory injected" failure) — either
        # way the ladder can only serve its silent rung.
        result, meta = es.ladders.get("voice").call("hello")
        assert result["silent"] is True and meta["degraded"] is True
        assert meta["rung"] == 1
    finally:
        es.close()


# ── (a) production context path: late-attached router/gateway work ────

def test_build_context_wires_context_into_ladders():
    """build_context (the production composition root) must hand the
    context to the error system, and the handlers resolve router /
    gateway LAZILY — both attach after the error system is built."""
    from nomorals.agents.context import build_context
    from nomorals.core.config import Settings

    tmp = tempfile.mkdtemp(prefix="pass45-ctx-")
    try:
        settings = Settings(home=tmp)
        context = build_context(
            settings, with_executor=False, with_router=False,
            with_memory=False, with_tools=False)
        try:
            es = context.error_system
            assert es is not None, "build_context must build the error system"

            # Router attached AFTER the error system exists.
            router = _OkRouter()
            context.router = router
            result, meta = es.ladders.get("llm").call(
                "", messages=[Message.user("hi")])
            assert isinstance(result, LLMResponse)
            assert result.text == "router says hi"
            assert router.calls, "rung 0 never reached the late router"
            assert meta["rung"] == 0 and meta["degraded"] is False

            # Gateway attached AFTER too — via context.extras, which is
            # where PartnerRuntime actually stores it.
            sent: list[tuple] = []

            class _Gw:
                outbox = None

                def send(self, platform, chat, text, **kw):
                    sent.append((platform, chat, text, kw))
                    return SendResult(ok=True, platform=platform,
                                      message_id="g1")

            context.extras["gateway"] = _Gw()
            result, meta = es.ladders.get("telegram").call(
                "telegram:777", "hello gateway")
            assert result.ok is True
            assert sent and sent[0][1] == "telegram:777"
            assert meta["rung"] == 0
        finally:
            context.db.close()
    finally:
        set_error_system(None)
        shutil.rmtree(tmp, ignore_errors=True)


# ── (b) llm ladder: total outage steps down, journals, reports ─────────

def test_llm_ladder_steps_down_on_router_raise_and_journals():
    es = build_error_system(
        None, context=_ctx(router=_DownRouter()),
        message_factory=_message_factory)
    try:
        ladder = es.ladders.get("llm")
        with pytest.raises(LadderExhausted) as exc:
            ladder.call("hello")
        errors = " ".join(exc.value.details.get("errors", []))
        assert "all providers down" in errors  # the REAL error is kept
        assert ladder.degraded is True
        assert "llm" in es.ladders.degraded_subsystems()
        assert es.health()["degraded_subsystems"] == ["llm"]
        # The ladder journals its heartbeats: a failed llm/ladder beat.
        fails, total, ratio = es.journal.subsystem_ratio("llm/ladder", 3600)
        assert total >= 1 and fails >= 1 and ratio > 0
    finally:
        es.close()


def test_llm_ladder_steps_down_on_error_response():
    """The real router RETURNS failure (LLMResponse.error) instead of
    raising — rung 0 must still step down, not report success."""
    es = build_error_system(
        None, context=_ctx(router=_ErrorResponseRouter()),
        message_factory=_message_factory)
    try:
        ladder = es.ladders.get("llm")
        with pytest.raises(LadderExhausted) as exc:
            ladder.call("hello")
        errors = " ".join(exc.value.details.get("errors", []))
        assert "every provider failed" in errors
        assert ladder.degraded is True
    finally:
        es.close()


def test_brain_chat_surfaces_ladder_honesty_on_total_outage():
    """Brain.chat (the funnel for the partner reply path AND the tool
    loop) routes through the ladder when the error system is wired:
    a total outage produces the ladder's honesty clause + the real
    error — and Brain still never raises."""
    router = _DownRouter()
    es = build_error_system(
        None, context=_ctx(router=router), message_factory=_message_factory)
    set_error_system(es)
    try:
        brain = Brain(router=router)
        resp = brain.chat([Message.user("hi")], timeout_s=10.0)
        assert not resp.ok
        assert "All language models are unreachable" in resp.error
        assert "all providers down" in resp.error
        assert es.ladders.get("llm").degraded is True
    finally:
        set_error_system(None)
        es.close()


def test_brain_chat_success_through_ladder():
    router = _OkRouter()
    es = build_error_system(
        None, context=_ctx(router=router), message_factory=_message_factory)
    set_error_system(es)
    try:
        brain = Brain(router=router)
        resp = brain.chat([Message.user("hi")])
        assert resp.ok and resp.text == "router says hi"
        assert es.ladders.get("llm").degraded is False
    finally:
        set_error_system(None)
        es.close()


def test_brain_without_error_system_is_unchanged():
    """No wired error system → Brain uses its direct path, exactly as
    before pass #45 (no ladder honesty text, no ladder involvement)."""
    brain = Brain(router=_DownRouter())
    resp = brain.chat([Message.user("hi")])
    assert not resp.ok
    assert "All language models are unreachable" not in resp.error

    brain = Brain(router=_OkRouter())
    resp = brain.chat([Message.user("hi")])
    assert resp.ok and resp.text == "router says hi"


def test_brain_with_private_router_skips_the_ladder():
    """A Brain wrapping a DIFFERENT router than the error system's
    context must not be rerouted onto the context's chain."""
    ctx_router = _DownRouter()
    es = build_error_system(
        None, context=_ctx(router=ctx_router),
        message_factory=_message_factory)
    set_error_system(es)
    try:
        brain = Brain(router=_OkRouter())  # not the context's router
        resp = brain.chat([Message.user("hi")])
        assert resp.ok and resp.text == "router says hi"
        assert es.ladders.get("llm").degraded is False
    finally:
        set_error_system(None)
        es.close()


# ── (c) probe climbs back when the primary heals ───────────────────────

def test_llm_ladder_climbs_back_when_probe_heals():
    router = _FlakyRouter()
    es = build_error_system(
        None, context=_ctx(router=router), message_factory=_message_factory)
    try:
        ladder = es.ladders.get("llm")
        # The probe must exist even though it resolves the router lazily.
        assert ladder.rungs[0].probe is not None
        with pytest.raises(LadderExhausted):
            ladder.call("hello")
        assert ladder.rung_idx == 1 and ladder.degraded is True

        ladder.rungs[0].probe_interval_s = 0.0  # probe on the next call
        router.down = False
        result, meta = ladder.call("hello again")
        assert result == {"ok": True, "text": "back online"}
        assert ladder.rung_idx == 0 and ladder.degraded is False
        assert meta["rung"] == 0
    finally:
        es.close()


# ── (d) telegram: outbox-backed queue rung, delivered end-to-end ──────

class _FlakyAdapter(ChatAdapter):
    name = "telegram"

    def __init__(self) -> None:
        super().__init__()
        self.down = True
        self.texts: list[str] = []

    def run(self, handler):  # pragma: no cover - unused
        pass

    def send(self, chat, text, **kw):
        if self.down:
            return SendResult(ok=False, platform=self.name,
                              error="ConnectionError: connection reset")
        self.texts.append(text)
        return SendResult(ok=True, platform=self.name, message_id="t-ok")

    def send_media(self, chat, media, *, caption=""):
        return SendResult(ok=True, platform=self.name, message_id="m-ok")


def _fast_gateway(adapter, db) -> ChatGateway:
    from nomorals.core.retry import BackoffPolicy

    return ChatGateway(
        {"telegram": adapter}, db=db, durable_outbox=True,
        retry_policy=BackoffPolicy(base=0.0, factor=1.0, cap=0.0,
                                   max_attempts=1))


def test_telegram_ladder_queues_to_outbox_and_delivers_after_heal():
    tmp, db = _tmp_db()
    try:
        adapter = _FlakyAdapter()
        gw = _fast_gateway(adapter, db)
        ctx = _ctx(db=db, extras={"gateway": gw})
        es = build_error_system(
            None, context=ctx, kv_factory=lambda d: KVStore(d),
            message_factory=_message_factory)
        try:
            ladder = es.ladders.get("telegram")
            result, meta = ladder.call("telegram:4242", "hello ladder")
            # Rung 0 (live send) failed; rung 1 queued into the durable
            # outbox — the verdict names the receipt.
            assert result["ok"] is False and result["queued"] is True
            assert result["chat"] == "telegram:4242"
            assert result.get("job_id")
            assert meta["degraded"] is True
            assert "queued" in meta["honesty"]
            assert adapter.texts == []  # nothing delivered while down
            # The row is really in the outbox, not terminal.
            assert gw.outbox.job_result(result["job_id"]) is None
            receipts = gw.receipts(chat_key="telegram:4242")
            queued = [r for r in receipts
                      if r["text_preview"] == "hello ladder"]
            assert queued and queued[0]["state"] == "queued"

            # Gateway heals: the next ordinary send drains the topic in
            # order — the queued message goes out FIRST.
            adapter.down = False
            verdict = gw.send("telegram", "telegram:4242", "second message")
            assert verdict.ok
            assert adapter.texts == ["hello ladder", "second message"]
            assert gw.outbox.job_result(result["job_id"])["ok"] is True
        finally:
            es.close()
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_telegram_ladder_kv_fallback_and_boot_drain():
    """No gateway (CLI-style process): the ladder's last rung keeps the
    legacy KV queue — and drain_tg_retry_queue (called by the partner
    runtime once its gateway exists) moves those rows into the durable
    outbox, where the normal drain delivers them."""
    tmp, db = _tmp_db()
    try:
        ctx = _ctx(db=db)  # no gateway anywhere
        es = build_error_system(
            None, context=ctx, kv_factory=lambda d: KVStore(d),
            message_factory=_message_factory)
        try:
            result, _meta = es.ladders.get("telegram").call(
                "telegram:777", "queued while offline")
            assert result == {"ok": False, "queued": True,
                              "chat": "telegram:777"}
            kv = KVStore(db)
            keys = kv.keys("tg_retry:")
            assert len(keys) == 1

            adapter = _FlakyAdapter()
            adapter.down = False
            gw = _fast_gateway(adapter, db)
            moved = drain_tg_retry_queue(
                ctx, gw, kv_factory=lambda d: KVStore(d))
            assert moved == 1
            assert kv.keys("tg_retry:") == []
            # The row is in the outbox now; a normal send drains it.
            verdict = gw.send("telegram", "telegram:777", "fresh")
            assert verdict.ok
            assert adapter.texts == ["queued while offline", "fresh"]
        finally:
            es.close()
    finally:
        db.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ── (e) voice ladder: real factory rung + honest silent rung ──────────

def test_voice_ladder_uses_injected_factory():
    calls: list[str] = []

    def fake_tts(ctx, text):
        calls.append(text)
        return "/tmp/pass45-voice.ogg"

    es = build_error_system(
        None, context=_ctx(), message_factory=_message_factory,
        tts_factory=fake_tts)
    try:
        result, meta = es.ladders.get("voice").call("say this")
        assert result == {"ok": True, "path": "/tmp/pass45-voice.ogg",
                          "silent": False}
        assert meta["rung"] == 0 and meta["degraded"] is False
        assert calls == ["say this"]
    finally:
        es.close()


def test_voice_ladder_silent_rung_on_engine_failure():
    def broken_tts(ctx, text):
        raise RuntimeError("espeak-ng exploded")

    es = build_error_system(
        None, context=_ctx(), message_factory=_message_factory,
        tts_factory=broken_tts)
    try:
        ladder = es.ladders.get("voice")
        result, meta = ladder.call("say this")
        assert result["ok"] is False and result["silent"] is True
        assert meta["degraded"] is True
        assert "Voice synthesis is unavailable" in meta["honesty"]
        assert ladder.degraded is True
    finally:
        es.close()


# ── partner voice-reply path routes through the ladder ─────────────────

class _VoiceGateway:
    """Minimal gateway double for _send_voice_reply."""

    dry_run = False
    outbox = None

    def __init__(self, adapter) -> None:
        self._adapter = adapter
        self.texts: list[str] = []

    def _adapter_for(self, platform):
        return self._adapter

    def typing(self, platform, chat, seconds=0.0, action="typing"):
        pass

    def send(self, platform, chat, text, **kw):
        self.texts.append(text)
        return SendResult(ok=True, platform="telegram", message_id="txt")


class _MediaAdapter:
    def __init__(self) -> None:
        self.media: list[str] = []

    def send_media(self, chat, media, *, caption=""):
        self.media.append(media.path)
        return SendResult(ok=True, platform="telegram", message_id="med")


def _voice_runtime(tts_factory):
    from nomorals.agents.partner.runtime import PartnerRuntime

    es = build_error_system(
        None, context=_ctx(), message_factory=_message_factory,
        tts_factory=tts_factory)
    adapter = _MediaAdapter()
    gateway = _VoiceGateway(adapter)
    context = SimpleNamespace(error_system=es, settings=SimpleNamespace(),
                              tools=None)
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = context
    rt.gateway = gateway
    rt._typing_enabled = lambda chat: False
    message = SimpleNamespace(
        chat=ChatRef(platform="telegram", chat_id="9"), meta={})
    return rt, es, gateway, adapter, message


def test_partner_voice_reply_sends_audio_via_ladder():
    rt, es, gateway, adapter, message = _voice_runtime(
        lambda ctx, text: "/tmp/pass45-reply.ogg")
    try:
        assert rt._send_voice_reply(message, ["hello there"]) is True
        assert adapter.media == ["/tmp/pass45-reply.ogg"]
        assert gateway.texts == []  # no text fallback needed
    finally:
        es.close()


def test_partner_voice_reply_degrades_to_text_with_honesty_note():
    def broken(ctx, text):
        raise RuntimeError("tts engine died")

    rt, es, gateway, adapter, message = _voice_runtime(broken)
    try:
        assert rt._send_voice_reply(message, ["hello there"]) is True
        assert adapter.media == []
        assert len(gateway.texts) == 1
        assert gateway.texts[0].startswith("Voice synthesis is unavailable")
        assert "hello there" in gateway.texts[0]
        assert es.ladders.get("voice").degraded is True
    finally:
        es.close()


# ── status visibility: /status renders, degradation shows when real ──

def test_control_status_renders_and_shows_degradation():
    """The owner's /status text must render at all (pre-fix it raised
    AttributeError on the gateway status's ``_delivery_mode`` string
    entry — reproduced on pristine HEAD) and must name a degraded
    ladder while one is degraded, and only then."""
    from nomorals.agents.context import build_context
    from nomorals.agents.partner.runtime import PartnerRuntime
    from nomorals.core.config import load_settings

    tmp = tempfile.mkdtemp(prefix="pass45-status-")
    settings = load_settings(overrides={
        "home": tmp,
        "partner.platforms": "local",
        "chat.local_enabled": "true",
    })
    context = build_context(settings, with_executor=False, with_tools=False)
    rt = PartnerRuntime(context)
    try:
        healthy = rt._control_status()
        assert "degradation:" not in healthy
        assert rt.status()["degradation"]["wired"] is True

        # Force a real degradation: the context router is replaced by
        # a dead one and the llm ladder is exercised through it.
        context.router = _DownRouter()
        with pytest.raises(LadderExhausted):
            context.error_system.ladders.get("llm").call(
                "", messages=[Message.user("hi")])
        degraded = rt._control_status()
        assert "degradation:" in degraded
        assert "llm" in degraded
        assert rt.status()["degradation"]["degraded"] == ["llm"]
    finally:
        rt.stop()
        set_error_system(None)
        context.db.close()
        shutil.rmtree(tmp, ignore_errors=True)
