"""Owner-priority chat queueing (owner lane, 2026-11-16).

The owner's own chats must never wait behind group chats: when every
main-pool worker is busy answering groups, an owner command on the
personal bot still runs at once on the reserved owner pool. Per-chat
FIFO is unchanged everywhere — two owner messages in one DM still
answer in order — and groups never ride the owner pool, not even when
the owner is the one speaking in them.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.social.chat.base import ChatKind, ChatMessage, ChatRef

OWNER_ID = "123"


def _msg(platform, chat_id, *, kind=ChatKind.DM, sender_id="", text="hi",
         meta=None):
    return ChatMessage(
        chat=ChatRef(platform=platform, chat_id=chat_id, kind=kind),
        incoming=True, text=text, sender_id=sender_id, meta=dict(meta or {}),
    )


class _Harness:
    """A PartnerRuntime with only its chat-queue machinery alive.

    The real on_message / _pump / _is_operator run against stubbed
    pools and a fake _process, so the test proves routing, not the
    brain. The harness mirrors production wiring: the main pool is
    the vps tune (2 workers would be the phone tune; 2 here keeps
    the busy-pool setup to two groups), the owner pool is one worker.
    """

    def __init__(self, main_workers=2):
        rt = PartnerRuntime.__new__(PartnerRuntime)
        rt._pool = ThreadPoolExecutor(
            max_workers=main_workers, thread_name_prefix="partner-chat")
        rt._owner_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="partner-owner")
        rt._queues = {}
        rt._draining = {}
        rt._queue_guard = threading.Lock()
        rt._stopped = threading.Event()
        rt._inflight = set()
        rt._inflight_guard = threading.Lock()
        rt._owner_chats = {"telegram-personal:123"}
        rt._owner_sender_ids = {OWNER_ID}
        rt.gateway = SimpleNamespace(stop=lambda: None)
        # stop() reads these directly (only some via getattr).
        rt._autonomy = None
        rt._arena = None
        rt._research = None
        rt._scheduler = None
        rt._trigger_engine = None
        rt._game_engine_obj = None
        rt._autonomy_db_handle = None
        rt._shared_claims_db_obj = None
        rt._stats_lock = threading.Lock()
        rt.stats = {}
        self.rt = rt
        self.started = {}  # text -> Event set when _process starts
        self.done = {}  # text -> Event set when _process finishes
        self.threads = {}  # text -> thread name that ran _process
        self.completed = []  # texts in _process completion order
        self._guard = threading.Lock()
        self.sleep = {}  # text -> seconds to sleep inside _process
        self.gates = {}  # text -> Event _process waits on after starting
        rt._process = self._fake_process

    def _fake_process(self, message):
        text = message.text
        with self._guard:
            self.threads[text] = threading.current_thread().name
            started = self.started.setdefault(text, threading.Event())
            started.set()
        gate = self.gates.get(text)
        if gate is not None:
            gate.wait(timeout=10.0)
        time.sleep(self.sleep.get(text, 0.0))
        with self._guard:
            self.completed.append(text)
            done = self.done.setdefault(text, threading.Event())
            done.set()

    def wait_started(self, texts, timeout=5.0):
        deadline = time.monotonic() + timeout
        for text in texts:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            event = self.started.setdefault(text, threading.Event())
            if not event.wait(timeout=remaining):
                return False
        return True

    def close(self):
        self.rt._stopped.set()
        for pool in (self.rt._pool, self.rt._owner_pool):
            pool.shutdown(wait=False)


@pytest.fixture
def harness():
    h = _Harness()
    yield h
    h.close()


def test_owner_command_jumps_busy_groups(harness):
    """All main-pool workers busy on groups: the owner command still
    completes at once instead of queueing behind them."""
    release = threading.Event()
    # Block both group pumps inside _process until the test releases them.
    harness.gates["g1"] = release
    harness.gates["g2"] = release
    harness.rt.on_message(_msg(
        "telegram-bot", "g1", kind=ChatKind.GROUP, text="g1"))
    harness.rt.on_message(_msg(
        "telegram-bot", "g2", kind=ChatKind.GROUP, text="g2"))
    assert harness.wait_started({"g1", "g2"}), "group pumps never started"
    harness.rt.on_message(_msg(
        "telegram-personal", "123", sender_id=OWNER_ID, text="owner"))
    # The owner command must finish while both groups are still blocked.
    assert harness.done.setdefault("owner", threading.Event()).wait(
        timeout=3.0), "owner command waited behind the busy groups"
    assert not release.is_set()
    assert harness.threads["owner"].startswith("partner-owner")
    assert harness.threads["g1"].startswith("partner-chat")
    assert harness.threads["g2"].startswith("partner-chat")
    release.set()
    assert harness.done.setdefault("g1", threading.Event()).wait(timeout=5.0)
    assert harness.done.setdefault("g2", threading.Event()).wait(timeout=5.0)


def test_owner_dm_answers_in_order(harness):
    """Two owner messages in one DM still answer in order (per-chat
    FIFO survives the new pool)."""
    harness.sleep["first"] = 0.3
    harness.rt.on_message(_msg(
        "telegram-personal", "123", sender_id=OWNER_ID, text="first"))
    harness.rt.on_message(_msg(
        "telegram-personal", "123", sender_id=OWNER_ID, text="second"))
    assert harness.done.setdefault("first", threading.Event()).wait(
        timeout=5.0)
    assert harness.done.setdefault("second", threading.Event()).wait(
        timeout=5.0)
    assert harness.completed == ["first", "second"]


def test_group_messages_never_use_owner_pool(harness):
    """Groups stay on the main pool even when the owner speaks in
    them; stranger DMs and the owner's gated-bot DM do too."""
    harness.rt.on_message(_msg(
        "telegram", "grp", kind=ChatKind.GROUP, sender_id=OWNER_ID,
        text="owner-in-group"))
    harness.rt.on_message(_msg(
        "telegram-bot", "999", kind=ChatKind.DM, sender_id="999",
        text="stranger"))
    harness.rt.on_message(_msg(
        "telegram-bot", "123", kind=ChatKind.DM, sender_id=OWNER_ID,
        text="gated-owner"))
    for text in ("owner-in-group", "stranger", "gated-owner"):
        assert harness.done.setdefault(text, threading.Event()).wait(
            timeout=5.0), text
        assert harness.threads[text].startswith("partner-chat"), text


def test_group_behind_busy_groups_still_completes(harness):
    """A group message queued behind busy groups still completes once
    a worker frees — the owner lane takes nothing from groups."""
    release = threading.Event()
    harness.gates["g1"] = release
    harness.gates["g2"] = release
    harness.rt.on_message(_msg(
        "telegram-bot", "g1", kind=ChatKind.GROUP, text="g1"))
    harness.rt.on_message(_msg(
        "telegram-bot", "g2", kind=ChatKind.GROUP, text="g2"))
    assert harness.wait_started({"g1", "g2"})
    harness.rt.on_message(_msg(
        "telegram-bot", "g3", kind=ChatKind.GROUP, text="g3"))
    assert not harness.done.setdefault("g3", threading.Event()).wait(
        timeout=1.0), "g3 ran while both workers were busy"
    release.set()
    assert harness.done["g3"].wait(timeout=5.0)
    assert harness.threads["g3"].startswith("partner-chat")


def test_stop_drains_owner_pool_too(harness):
    """stop() waits for an in-flight owner pump, not just group pumps,
    and both pools end shut down."""
    release_owner = threading.Event()
    release_group = threading.Event()
    harness.gates["owner"] = release_owner
    harness.gates["g1"] = release_group
    harness.rt.on_message(_msg(
        "telegram-bot", "g1", kind=ChatKind.GROUP, text="g1"))
    harness.rt.on_message(_msg(
        "telegram-personal", "123", sender_id=OWNER_ID, text="owner"))
    assert harness.wait_started({"g1", "owner"})
    stopped = threading.Event()
    start = time.monotonic()

    def _stop():
        harness.rt.stop()
        stopped.set()

    stopper = threading.Thread(target=_stop, daemon=True)
    stopper.start()
    time.sleep(0.5)
    release_group.set()
    time.sleep(0.5)
    # The owner pump is still blocked: stop() must still be waiting.
    assert not stopped.is_set(), "stop() returned while the owner pump ran"
    release_owner.set()
    assert stopped.wait(timeout=10.0), "stop() never returned"
    assert time.monotonic() - start >= 1.0
    assert harness.done["owner"].is_set()
    assert harness.done["g1"].is_set()
    with pytest.raises(RuntimeError):
        harness.rt._pool.submit(lambda: None)
    with pytest.raises(RuntimeError):
        harness.rt._owner_pool.submit(lambda: None)
