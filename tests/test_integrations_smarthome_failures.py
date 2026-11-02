"""Tail-89 pins: nomorals/integrations/smarthome_integration.py
census sites — all inside HAWebSocket's teardown paths.

The class contract is printed on start()/stop(): "Never
raises". Every site guards the await or close of a CHILD
task / socket during teardown — the standard asyncio idiom,
where the exception being swallowed belongs to the child,
not to the caller — and all are held as exclusions:

1091 — ``stop()``: CancelledError from awaiting the run-loop
task stop() itself just cancelled -> pass. The cancellation
was requested by this very call; propagating it would make
every clean stop raise. stop() then proceeds to close the
socket (the generic sibling guard at 1093 binds +
_log.exception for a task that died with a real error).
1155 — ``_connect_and_serve`` finally: awaiting the
just-cancelled read task raises (CancelledError, _WSClosed,
HAWebSocketError) -> pass. The read task's terminal error is
not the serve loop's outcome: the original error (or clean
exit) propagates unchanged — probe- and pin-proven below,
where a failing command surfaces ITS error, never the read
task's cancellation.
1184 — ``_close()``: writer.close()/wait_closed() crash ->
pass ("close is best-effort"). The writer reference is
dropped BEFORE the close attempt, so the client reports
not-connected either way; a wedged socket close cannot
strand the client in a half-open lie.
1192 — ``_fail_pending``: one future's set_exception crash
-> pass, and the loop still fails every other pending
future before clearing the map. Per-item isolation inside a
batch teardown; the pending map is emptied regardless.
1379 — ``unsubscribe_trigger``: the fire-and-forget
unsubscribe send crash -> pass. The returned True is the
LOCAL removal truth (the subscription was popped before the
send; an unknown id returns False), and no surface claims
the server dropped it — a stale server-side registration
cannot resurrect locally: reconnect re-sends only subs
still registered. The tension, stated plainly: the sibling
send guards (subscribe 1363, resubscribe 1391) bind +
_log.debug(exc_info=True) while 1379 is a bare pass, so
this one failure is traceless — a bind+debug sketch
matching the siblings is shelved as a follow-up, not a
convert (observability only; the return is already true).

NOT-recording is pinned with the snapshot method: the module
calls no failures-ledger record function anywhere, so both
the agency and agents ledgers must be unchanged across each
trigger.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.integrations import smarthome_integration as S


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _client():
    return S.HAWebSocket("ws://ha.local:8123/api/websocket", "tok")


# ── 1091: stop() swallows the cancelled run-loop task ───────────

def test_1091_stop_swallows_cancelled_and_dead_task():
    async def main():
        # a run-loop task that died with a real error: stop()
        # still completes (the 1093 sibling logs it) and clears
        c = _client()
        c._running = True

        async def _die():
            raise RuntimeError("loop died")
        c._task = asyncio.ensure_future(_die())
        await c.stop()
        assert c._task is None and c._running is False
        # a healthy long-running task: our own cancel lands in
        # the 1091 guard, stop() completes cleanly
        c2 = _client()
        c2._running = True

        async def _block():
            await asyncio.sleep(3600)
        c2._task = asyncio.ensure_future(_block())
        await c2.stop()
        assert c2._task is None and c2._running is False
    before = _snap()
    asyncio.run(main())
    assert _snap() == before


# ── 1155: read-task teardown never masks the serve outcome ──────

def test_1155_read_task_teardown_swallowed(monkeypatch):
    async def main():
        c = _client()

        async def _fake_connect():
            writer = SimpleNamespace(
                close=lambda: None,
                wait_closed=lambda: asyncio.sleep(0))
            return SimpleNamespace(), writer
        calls = {"n": 0}

        async def _next():
            calls["n"] += 1
            return {"type": "auth_required"} if calls["n"] == 1 \
                else {"type": "auth_ok"}

        async def _recv_block(reader, writer):
            await asyncio.sleep(3600)

        async def _cmd_boom(payload):
            raise S.HAWebSocketError("send failed")
        c._connect = _fake_connect
        c._next_message = _next
        c._send_command = _cmd_boom
        monkeypatch.setattr(S, "_ws_recv_text", _recv_block)
        monkeypatch.setattr(
            S, "_ws_send_text", lambda w, t: asyncio.sleep(0))
        monkeypatch.setattr(
            S, "_ws_handshake", lambda r, w, h, p, pa: asyncio.sleep(0))
        # the serve loop's OWN error propagates — not the read
        # task's CancelledError from the finally teardown
        with pytest.raises(S.HAWebSocketError, match="send failed"):
            await c._connect_and_serve()
        assert c._read_task is None
    before = _snap()
    asyncio.run(main())
    assert _snap() == before


# ── 1184: socket close is best-effort ───────────────────────────

def test_1184_close_crash_swallowed():
    async def main():
        c = _client()

        class BadWriter:
            def close(self):
                raise RuntimeError("close boom")

            async def wait_closed(self):
                return None
        c._writer = BadWriter()
        c._reader = SimpleNamespace()
        await c._close()
        # references dropped before the close attempt
        assert c._writer is None and c._reader is None
        # control: a healthy writer is really closed
        closed = []

        class GoodWriter:
            def close(self):
                closed.append("close")

            async def wait_closed(self):
                closed.append("wait")
        c._writer = GoodWriter()
        await c._close()
        assert closed == ["close", "wait"]
    before = _snap()
    asyncio.run(main())
    assert _snap() == before


# ── 1192: one bad future cannot stop _fail_pending ──────────────

def test_1192_bad_future_isolated():
    before = _snap()
    c = _client()

    class FakeFut:
        def __init__(self, raises=False):
            self._raises = raises
            self.got = None

        def done(self):
            return False

        def set_exception(self, exc):
            if self._raises:
                raise RuntimeError("fut boom")
            self.got = exc
    bad, good = FakeFut(raises=True), FakeFut()
    c._pending = {1: bad, 2: good}
    err = S._WSClosed("gone")
    c._fail_pending(err)
    assert good.got is err
    assert c._pending == {}
    assert _snap() == before


# ── 1379: unsubscribe send crash — local removal is the truth ───

def test_1379_unsubscribe_send_crash_still_true(monkeypatch):
    async def main():
        c = _client()
        mid = await c.subscribe_trigger(
            {"platform": "state", "entity_id": "light.k"})
        assert mid in c._trigger_subs
        c._writer = SimpleNamespace(is_closing=lambda: False)

        def _send_boom(writer, text):
            raise RuntimeError("send boom")
        monkeypatch.setattr(S, "_ws_send_text", _send_boom)
        assert c.unsubscribe_trigger(mid) is True
        assert mid not in c._trigger_subs
        # control: a healthy send path — the unsubscribe frame
        # is scheduled and True is still the local truth
        sent = []

        async def _send_ok(writer, text):
            sent.append(text)
        monkeypatch.setattr(S, "_ws_send_text", _send_ok)
        mid2 = await c.subscribe_trigger(
            {"platform": "state", "entity_id": "light.k"})
        assert c.unsubscribe_trigger(mid2) is True
        await asyncio.sleep(0)
        assert any("unsubscribe_events" in t for t in sent)
        # control: an unknown id was never there -> False
        assert c.unsubscribe_trigger(424242) is False
    before = _snap()
    asyncio.run(main())
    assert _snap() == before
