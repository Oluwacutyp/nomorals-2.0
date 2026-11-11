"""Computer watch stream — the live tail over a desktop session.

The watch is driven against a fake stream producer emitting
real-shaped stream chunks (StreamFrame) and a recording surface;
the full path producer → manager → delivery thread → surface
send executes. Pins:

* new frames are delivered in order as produced; unchanged
  frames are not re-delivered;
* bursts coalesce through the one-frame pending slot with the
  skip count on the next delivered caption;
* a stop through the ONE cancellation facade lands the job
  cancelled/confirmed on its durable record, with a receipt, and
  delivery ceases;
* a watch whose session ends reads ended (done + reason), on the
  record, its ticket, and the /runs board — not live;
* a stalled producer earns exactly one stall notice per episode
  and the watch recovers when the producer answers again;
* a runtime restart does not resurrect a watch: an orphaned live
  record is reconciled to ended by the next manager;
* the budget ends a forgotten watch;
* takeover flips are surfaced in words while the stream keeps
  flowing (the gate binds hands, never eyes);
* /computer watch + unwatch run end to end through the chat
  handler against a seeded session store.
"""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace

import pytest

from nomorals.computer.sessions import SessionError
from nomorals.computer.watch import (
    ComputerWatchManager,
    StreamFrame,
    WatchRecordStore,
)
from nomorals.storage.db import Database

# ── fakes ─────────────────────────────────────────────────────────


def _frame(hash_: str, title: str = "Editor") -> StreamFrame:
    return StreamFrame(
        session_id="desk-test", ts=time.time(),
        frame_path=f"/frames/{hash_}.png", frame_hash=hash_,
        width=1280, height=1024,
        windows=[{"id": 1, "title": title, "focused": True}],
        focused_id=1, pointer={"x": 3, "y": 4})


class _ScriptedProducer:
    """Emits the script, then holds the last frame (a static
    screen) or ends the session, per ``then``."""

    def __init__(self, script, then: str = "hold"):
        self.script = list(script)
        self.then = then
        self.calls = 0
        self._last: StreamFrame | None = None

    def __call__(self, session_id: str) -> StreamFrame:
        self.calls += 1
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            self._last = item
            return item
        if self.then == "hold" and self._last is not None:
            return self._last
        raise SessionError("the session is closed")


class _LiveProducer:
    """A fresh frame every tick — a busy desktop."""

    def __init__(self):
        self.n = 0
        self._lock = threading.Lock()

    def __call__(self, session_id: str) -> StreamFrame:
        with self._lock:
            self.n += 1
            n = self.n
        return _frame(f"hash-{n}", title=f"Window {n}")


class _ModalProducer:
    """Fails until flipped live — a stalled then recovered driver."""

    def __init__(self):
        self.mode = "fail"
        self.n = 0

    def __call__(self, session_id: str) -> StreamFrame:
        if self.mode == "fail":
            raise RuntimeError("driver not answering")
        self.n += 1
        return _frame(f"live-{self.n}")


class _Surface:
    """The owner's chat, recorded."""

    def __init__(self):
        self.frames: list[tuple[str, str]] = []
        self.notes: list[str] = []
        self._lock = threading.Lock()

    def send_frame(self, frame: StreamFrame, caption: str) -> None:
        with self._lock:
            self.frames.append((frame.frame_path, caption))

    def send_note(self, text: str) -> None:
        with self._lock:
            self.notes.append(text)

    def frame_paths(self) -> list[str]:
        with self._lock:
            return [path for path, _cap in self.frames]

    def all_text(self) -> list[str]:
        with self._lock:
            return [cap for _p, cap in self.frames] + list(self.notes)


class _BlockingSurface(_Surface):
    """The first frame's send blocks until released — a surface
    that cannot keep up with the producer."""

    def __init__(self):
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self._first = True

    def send_frame(self, frame: StreamFrame, caption: str) -> None:
        if self._first:
            self._first = False
            self.entered.set()
            self.release.wait(10)
        super().send_frame(frame, caption)


def _wait_for(pred, timeout: float = 8.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.01)
    return bool(pred())


@pytest.fixture()
def db(tmp_path):
    database = Database(os.path.join(str(tmp_path), "records.db"))
    database.migrate()
    return database


def _manager(db, producer, **kw) -> ComputerWatchManager:
    kw.setdefault("interval_s", 0.05)
    return ComputerWatchManager(capture_fn=producer, db=db, **kw)


def _start(mgr, surface, session_id: str = "desk-test"):
    watch, dup = mgr.start(
        session_id, platform="telegram", chat_key="telegram:1",
        send_frame=surface.send_frame, send_note=surface.send_note)
    assert dup is False
    return watch


# ── delivery ──────────────────────────────────────────────────────


class TestDelivery:
    def test_new_frames_delivered_in_order(self, db):
        frames = [_frame("a"), _frame("b"), _frame("c")]
        producer = _ScriptedProducer(
            [*frames, SessionError("the session is closed")],
            then="end")
        mgr = _manager(db, producer)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert watch.wait(8)
        assert surface.frame_paths() == [f.frame_path
                                         for f in frames]
        record = WatchRecordStore(db).get(watch.watch_id)
        assert record["status"] == "done"
        assert "session ended" in record["reason"]
        assert record["frames_delivered"] == 3
        assert any("computer watch ended" in note
                   for note in surface.notes)

    def test_unchanged_frames_are_not_redelivered(self, db):
        producer = _ScriptedProducer(
            [_frame("a"), _frame("a"), _frame("b"),
             SessionError("the session is closed")], then="end")
        mgr = _manager(db, producer)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert watch.wait(8)
        assert surface.frame_paths() == ["/frames/a.png",
                                         "/frames/b.png"]

    def test_bursts_coalesce_with_skip_note(self, db):
        script = [_frame(f"f{i}") for i in range(1, 7)]
        producer = _ScriptedProducer(script, then="hold")
        mgr = _manager(db, producer, interval_s=0.02)
        surface = _BlockingSurface()
        watch = _start(mgr, surface)
        assert surface.entered.wait(8)
        # The producer pumps f2..f6 into the one-frame slot while
        # the first send is blocked: f2..f5 are overwritten.
        assert _wait_for(lambda: watch.snapshot()["frames_skipped"]
                         >= 4)
        surface.release.set()
        assert _wait_for(lambda: len(surface.frames) >= 2)
        mgr.cancel(watch.watch_id)
        assert watch.wait(8)
        assert surface.frame_paths() == ["/frames/f1.png",
                                         "/frames/f6.png"]
        caption = surface.frames[1][1]
        assert "skipped 4 frame(s) to stay current" in caption
        assert watch.snapshot()["frames_skipped"] == 4

    def test_dedupe_same_session_and_chat(self, db):
        mgr = _manager(db, _LiveProducer())
        surface = _Surface()
        watch = _start(mgr, surface)
        again, dup = mgr.start(
            "desk-test", platform="telegram",
            chat_key="telegram:1", send_frame=surface.send_frame,
            send_note=surface.send_note)
        assert dup is True and again is watch
        fresh, dup2 = mgr.start(
            "desk-test", platform="telegram",
            chat_key="telegram:1", send_frame=surface.send_frame,
            send_note=surface.send_note, force=True)
        assert dup2 is False and fresh is not watch
        mgr.cancel(watch.watch_id)
        mgr.cancel(fresh.watch_id)
        assert watch.wait(8) and fresh.wait(8)


# ── cancellation through the facade ───────────────────────────────


class TestFacadeStop:
    def test_stop_lands_cancelled_with_receipt(self, db):
        from nomorals.agents.run_ticket import ticket_for
        from nomorals.core.cancellation import (
            CancelReceiptStore,
            cancel_with_receipt,
        )

        mgr = _manager(db, _LiveProducer())
        surface = _Surface()
        watch = _start(mgr, surface)
        assert _wait_for(
            lambda: watch.snapshot()["frames_delivered"] >= 2)
        ref = f"watch:{watch.watch_id}"
        result = cancel_with_receipt(
            db, ref, requested_by="owner", reason="test stop",
            watch_manager=mgr)
        assert result["receipt"]["receipt_state"] == "requested"
        assert result["native_state"] == "requested"
        assert watch.wait(8)
        record = WatchRecordStore(db).get(watch.watch_id)
        assert record["status"] == "cancelled"
        assert record["cancel_state"] == "confirmed"
        # The receipt keeps the answer it was minted with.
        receipt = CancelReceiptStore(db).latest_for_ref(ref)
        assert receipt["receipt_state"] == "requested"
        assert receipt["organ"] == "computer_watch"
        # The ticket rides the receipt in its cancel field.
        ticket = ticket_for(ref, db)
        assert ticket["cancel"]["receipt"]["state"] == "requested"
        assert ticket["cancel"]["state"] == "confirmed"
        # Delivery has ceased — the sender thread is done.
        delivered = len(surface.frames)
        time.sleep(0.3)
        assert len(surface.frames) == delivered

    def test_cancel_after_end_is_refused(self, db):
        from nomorals.core.cancellation import cancel_with_receipt

        producer = _ScriptedProducer(
            [SessionError("the session is closed")], then="end")
        mgr = _manager(db, producer)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert watch.wait(8)
        result = cancel_with_receipt(
            db, f"watch:{watch.watch_id}", watch_manager=mgr)
        assert result["native_state"] == "already_terminal:own_terms"
        assert result["receipt"]["receipt_state"] == "refused"

    def test_unknown_watch_is_not_found(self, db):
        from nomorals.core.cancellation import cancel_with_receipt

        mgr = _manager(db, _LiveProducer())
        result = cancel_with_receipt(db, "watch:cw-nope",
                                     watch_manager=mgr)
        assert result["receipt"]["receipt_state"] == "not_found"

    def test_missing_handle_cancels_nothing(self, db):
        from nomorals.core.cancellation import cancel_with_receipt

        result = cancel_with_receipt(db, "watch:cw-x")
        assert result["receipt"] is None
        assert "no computer_watch handle" in result["receipt_error"]


# ── board truth ───────────────────────────────────────────────────


class TestBoardTruth:
    def test_dead_watch_reads_ended_not_live(self, db):
        from nomorals.agents.run_ticket import ticket_for
        from nomorals.agents.ticket_board import board_rows

        producer = _ScriptedProducer(
            [_frame("a"), SessionError("the session is closed")],
            then="end")
        mgr = _manager(db, producer)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert watch.wait(8)
        ref = f"watch:{watch.watch_id}"
        board = board_rows(db)
        rows = [r for r in board["rows"] if r["ref"] == ref]
        assert len(rows) == 1
        row = rows[0]
        assert row["kind"] == "computer_watch"
        assert row["status"] == "done"
        assert row["live"] is False
        assert "cw-" not in row["label"]
        assert "desk-" not in row["label"]
        ticket = ticket_for(ref, db)
        assert ticket["status"] == "done"
        assert "session ended" in ticket["detail"]["reason"]
        assert ticket["liveness"]["frames_delivered"] == 1

    def test_live_watch_is_live_on_the_board(self, db):
        from nomorals.agents.ticket_board import board_rows

        mgr = _manager(db, _LiveProducer())
        surface = _Surface()
        watch = _start(mgr, surface)
        assert _wait_for(
            lambda: watch.snapshot()["frames_delivered"] >= 1)
        board = board_rows(db)
        rows = [r for r in board["rows"]
                if r["ref"] == f"watch:{watch.watch_id}"]
        assert len(rows) == 1
        assert rows[0]["live"] is True
        assert rows[0]["status"] == "running"
        mgr.cancel(watch.watch_id)
        assert watch.wait(8)


# ── stall + recovery ──────────────────────────────────────────────


class TestStall:
    def test_stall_notice_once_then_recovery(self, db):
        from nomorals.agents.run_ticket import ticket_for

        producer = _ModalProducer()
        mgr = _manager(db, producer, stall_after_s=0.3)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert _wait_for(lambda: any("stalled" in note
                                     for note in surface.notes))
        record = WatchRecordStore(db).get(watch.watch_id)
        assert record["status"] == "stalled"
        ticket = ticket_for(f"watch:{watch.watch_id}", db)
        assert ticket["liveness"]["status"] == "stalled"
        # The episode is announced once, however long it lasts.
        time.sleep(0.7)
        stalls = [n for n in surface.notes if "stalled" in n]
        assert len(stalls) == 1
        assert "still on" in stalls[0]
        # Recovery: frames flow again, the status heals, and the
        # recovery is announced on the stream.
        producer.mode = "live"
        assert _wait_for(
            lambda: watch.snapshot()["frames_delivered"] >= 1)
        assert _wait_for(lambda: any(
            "flowing again" in text for text in surface.all_text()))
        record = WatchRecordStore(db).get(watch.watch_id)
        assert record["status"] == "running"
        mgr.cancel(watch.watch_id)
        assert watch.wait(8)


# ── restart truth ─────────────────────────────────────────────────


class TestRestart:
    def test_orphaned_live_record_is_reconciled_to_ended(self, db):
        from nomorals.agents.run_ticket import ticket_for
        from nomorals.agents.ticket_board import board_rows

        store = WatchRecordStore(db)
        now = time.time()
        store.record({
            "watch_id": "cw-orphan", "session_id": "desk-test",
            "label": "", "platform": "telegram",
            "chat_key": "telegram:1", "status": "running",
            "reason": "", "error": "", "frames_delivered": 7,
            "frames_skipped": 0, "frames_dropped": 0,
            "last_frame_at": now - 500, "last_seen_at": now - 500,
            "budget_s": 3600.0, "cancel_state": "",
            "cancel_requested_at": 0, "created_at": now - 600,
            "started_at": now - 600, "finished_at": 0,
            "updated_at": now - 500})
        # A new process: a fresh manager over the same store.
        mgr = ComputerWatchManager(db=db)
        row = store.get("cw-orphan")
        assert row["status"] == "done"
        assert "runtime stopped" in row["reason"]
        assert row["finished_at"] == pytest.approx(now - 500)
        ticket = ticket_for("watch:cw-orphan", db)
        assert ticket["status"] == "done"
        board = board_rows(db)
        rows = [r for r in board["rows"]
                if r["ref"] == "watch:cw-orphan"]
        assert len(rows) == 1 and rows[0]["live"] is False
        # A watch this manager actually holds is never reconciled.
        surface = _Surface()
        live, _dup = mgr.start(
            "desk-test", platform="telegram",
            chat_key="telegram:1",
            capture_fn=_LiveProducer(),
            send_frame=surface.send_frame,
            send_note=surface.send_note)
        assert _wait_for(
            lambda: live.snapshot()["frames_delivered"] >= 1)
        assert mgr.reconcile_stale() == 0
        assert store.get(live.watch_id)["status"] == "running"
        mgr.cancel(live.watch_id)
        assert live.wait(8)


# ── budget + gate flips ───────────────────────────────────────────


class TestBudgetAndGate:
    def test_budget_ends_a_forgotten_watch(self, db):
        mgr = _manager(db, _LiveProducer(), max_seconds=0.3)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert watch.wait(8)
        record = WatchRecordStore(db).get(watch.watch_id)
        assert record["status"] == "done"
        assert "budget" in record["reason"]

    def test_takeover_flips_noted_stream_keeps_flowing(self, db):
        t0 = time.time()

        def gate(session_id):
            elapsed = time.time() - t0
            return {"takeover": 0.2 < elapsed < 0.9,
                    "owner_stop": False, "state": "open"}

        mgr = _manager(db, _LiveProducer(), gate_fn=gate)
        surface = _Surface()
        watch = _start(mgr, surface)
        assert _wait_for(lambda: any("taken over" in note
                                     for note in surface.notes), 5)
        assert _wait_for(lambda: any("handed back" in note
                                     for note in surface.notes), 5)
        takeovers = [n for n in surface.notes if "taken over" in n]
        handbacks = [n for n in surface.notes if "handed back" in n]
        assert len(takeovers) == 1
        assert len(handbacks) == 1
        # The eyes were never gated: frames flowed throughout.
        assert watch.snapshot()["frames_delivered"] >= 3
        assert watch.status == "running"
        mgr.cancel(watch.watch_id)
        assert watch.wait(8)


# ── the chat surface ──────────────────────────────────────────────


class _StubRuntime(SimpleNamespace):
    """The computer handler methods on a plain attribute bag —
    exactly how the real runtime composes them from the mixin
    (the test_computer_surface precedent)."""

    from nomorals.agents.partner.runtime_expression import (
        RuntimeExpressionMixin as _Mixin,
    )
    _control_computer = _Mixin._control_computer
    _computer_send_frame = _Mixin._computer_send_frame
    _computer_watch_manager = _Mixin._computer_watch_manager
    _computer_watch_senders = _Mixin._computer_watch_senders
    _computer_watch_start = _Mixin._computer_watch_start
    _computer_watch_stop = _Mixin._computer_watch_stop


class TestChatSurface:
    @pytest.fixture()
    def rig(self, tmp_path, monkeypatch, db):
        home = tmp_path / "nmhome"
        root = home / "computer"
        root.mkdir(parents=True)
        record = {
            "id": "desk-test", "mode": "virtual", "display": ":99",
            "state": "open", "created": time.time(),
            "last_activity": time.time(), "mission": "watch me",
            "width": 1280, "height": 1024, "xvfb_pid": 0,
            "auth_path": "", "owner_stop": False,
            "takeover": False, "takeover_by": "", "note": "",
        }
        (root / "sessions.json").write_text(
            json.dumps({"sessions": {"desk-test": record}}),
            encoding="utf-8")
        monkeypatch.setenv("NM_HOME", str(home))
        sent_files: list = []
        sent_texts: list = []

        def send_file(*args, **kwargs):
            sent_files.append((args, kwargs))
            return SimpleNamespace(ok=True)

        def send(*args, **kwargs):
            sent_texts.append((args, kwargs))
            return SimpleNamespace(ok=True)

        gateway = SimpleNamespace(send_file=send_file, send=send)
        mgr = ComputerWatchManager(
            capture_fn=_LiveProducer(), db=db, interval_s=0.05)
        stub = _StubRuntime(
            context=SimpleNamespace(db=db), gateway=gateway,
            _computer_watch_manager_inst=mgr)
        message = SimpleNamespace(
            chat=SimpleNamespace(platform="telegram",
                                 key="telegram:1"))
        return stub, mgr, message, sent_files, sent_texts

    def test_watch_then_unwatch_through_the_handler(self, rig, db):
        stub, mgr, message, sent_files, sent_texts = rig
        out = stub._control_computer("watch", message)
        assert "watching desk-test live" in out
        assert _wait_for(lambda: len(sent_files) >= 1)
        out = stub._control_computer("watch", message)
        assert "already watching" in out
        out = stub._control_computer("unwatch", message)
        assert "stop requested" in out
        watch = mgr.list_recent(1)[0]
        assert watch.wait(8)
        record = WatchRecordStore(db).get(watch.watch_id)
        assert record["status"] == "cancelled"
        out = stub._control_computer("unwatch", message)
        assert "no watch is running" in out

    def test_watch_without_a_session_says_so(self, rig):
        stub, _mgr, message, _files, _texts = rig
        out = stub._control_computer("watch desk-ghost", message)
        assert "no desktop session" in out
