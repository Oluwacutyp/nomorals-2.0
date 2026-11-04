"""Owner stop must STICK: persisted stop intent vs live-attach.

The owner's live bug (2026-11-04, hours after walive2 shipped):
he ran /stop whatsapp to silence Devon on his personal WhatsApp
— she came back online on her own after a while.  The walive
reconcile computed the surface's desired state purely from the
pairing registry (account has a port AND (linked OR bridge
running)) and hot-started the adapter whenever it was down, at
startup and on the 60s housekeeping beat.  A manual /stop
(gateway stop_one) was invisible to it: within a beat or two the
reconcile resurrected the surface.  No record of OWNER INTENT
existed anywhere in the loop.

The fix, pinned here:

1.  Owner intent is durable state: ``owner_stopped`` per
    platform in the pairing registry (the same JSON store,
    additive top-level key — old files load unstopped, older
    code ignores the key).
2.  Every owner stop path records it (/stop <platform> via
    stop_platform, /pair stop); every owner start path clears
    it (/start <platform>, /pair start|restart, a fresh pair);
    removing an account clears its platform's record.
3.  The reconcile honors it absolutely: while the order stands,
    the WhatsApp surface's desired state is DOWN — an adapter
    found up is stopped, nothing is ever attached — and the
    startup attach obeys the same term because it IS the same
    reconcile.
4.  /platforms renders the state truthfully: "stopped (by you)"
    is a different line from "not running".
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.pairing import BridgeSupervisor, PairingRegistry

BASE_PORT = 18921


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


def make_runtime(tmp_path, sup):
    """PartnerRuntime on a bare harness (walive2's idiom): real
    gateway, real provision mixin, the fake-spawn supervisor
    pre-seeded as the pool."""
    rt = PartnerRuntime.__new__(PartnerRuntime)
    settings = make_settings(tmp_path)
    rt.settings = settings
    rt.context = SimpleNamespace(settings=settings)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    rt._pair_supervisor = sup
    return rt


def paired_runtime(tmp_path, lines=()):
    """A runtime whose pool holds one paired WhatsApp account
    with its surface live-attached."""
    sup, registry, spawns = make_supervisor(tmp_path, lines=lines)
    rt = make_runtime(tmp_path, sup)
    rec = sup.pair_whatsapp("2348012345678")
    rt._sync_whatsapp_surface()
    assert "whatsapp" in rt.gateway.adapters
    return rt, sup, registry, rec


# ── 1. the intent record itself ─────────────────────────────────────

def test_registry_owner_stopped_round_trip(tmp_path):
    path = tmp_path / "paired.json"
    registry = PairingRegistry(path)
    assert registry.owner_stopped("whatsapp") is False
    registry.set_owner_stopped("whatsapp", True)
    # Persisted in the same store, under its own top-level key.
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload["owner_stopped"]) == {"whatsapp"}
    # A fresh registry over the same file (a restart) sees it.
    fresh = PairingRegistry(path)
    assert fresh.owner_stopped("whatsapp") is True
    assert fresh.owner_stopped("telegram") is False
    fresh.set_owner_stopped("whatsapp", False)
    assert PairingRegistry(path).owner_stopped("whatsapp") is False


def test_registry_written_before_intent_loads_unstopped(tmp_path):
    path = tmp_path / "paired.json"
    path.write_text(json.dumps({"accounts": []}), encoding="utf-8")
    registry = PairingRegistry(path)
    assert registry.owner_stopped("whatsapp") is False


# ── 2. /stop sticks ─────────────────────────────────────────────────

def test_stop_platform_sticks_across_beats_and_reconciles(tmp_path):
    """The live bug, end to end: surface up, owner stops it, and
    neither the housekeeping beat (×3, throttle rewound) nor a
    direct reconcile may raise it again — while the pool still
    looks perfectly live (bridge running, account registered)."""
    rt, sup, registry, rec = paired_runtime(tmp_path)
    try:
        result = rt.stop_platform("whatsapp")
        assert result.get("ok")
        assert "whatsapp" not in rt.gateway.adapters
        assert registry.owner_stopped("whatsapp") is True
        assert sup.is_running(rec.id)  # the pool never stopped
        svc = rt._pairing_personal()
        for _ in range(3):
            svc["last_cycle"] = 0.0  # rewind the shared throttle
            rt._housekeeping_beat()
            assert "whatsapp" not in rt.gateway.adapters
        assert rt._sync_whatsapp_surface() == ""
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
    finally:
        rt.gateway.stop()


def test_intent_keeps_down_an_adapter_found_up(tmp_path):
    """The order outranks even an adapter that is somehow already
    up when the reconcile runs (a beat that raced the stop):
    the sync puts it down instead of adopting it."""
    rt, sup, registry, rec = paired_runtime(tmp_path)
    try:
        registry.set_owner_stopped("whatsapp", True)
        assert "whatsapp" in rt.gateway.adapters
        assert rt._sync_whatsapp_surface() == ""
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
    finally:
        rt.gateway.stop()


def test_start_platform_clears_intent_and_surface_returns(tmp_path):
    rt, sup, registry, rec = paired_runtime(tmp_path)
    try:
        old = rt.gateway.adapters["whatsapp"]
        rt.stop_platform("whatsapp")
        assert "whatsapp" not in rt.gateway.adapters
        # The stopped adapter's thread exits on its own stop event
        # (its reconnect backoff is seconds); a gateway start while
        # it is still dying honestly reports "already running", so
        # the test waits for the thread to actually be down — the
        # state an owner's later /start meets in production too.
        old._thread.join(timeout=20)
        assert not old._thread.is_alive()
        result = rt.start_platform("whatsapp")
        assert result.get("ok")
        assert registry.owner_stopped("whatsapp") is False
        # The reconcile agrees with the gateway again: adopt, no-op.
        assert rt._sync_whatsapp_surface() == ""
        adapter = rt.gateway.adapters.get("whatsapp")
        assert adapter is not None and adapter.port == BASE_PORT
    finally:
        rt.gateway.stop()


def test_stop_platform_records_intent_for_any_platform(tmp_path):
    """The record is per-platform and general — /stop telegram
    records telegram intent identically (nothing reconciles
    telegram today, but the order is on file and /platforms can
    render it)."""
    sup, registry, _ = make_supervisor(tmp_path)
    rt = make_runtime(tmp_path, sup)
    rt.stop_platform("telegram")  # not running: gateway says so,
    assert registry.owner_stopped("telegram") is True  # intent held
    rt.start_platform("telegram")
    assert registry.owner_stopped("telegram") is False


# ── 3. restart: the order survives, startup obeys ───────────────────

def test_startup_attach_honors_persisted_stop(tmp_path):
    """Simulated restart: fresh registry + supervisor + runtime
    objects over the SAME state dir, the account linked and its
    bridge running — the startup reconcile (the exact call
    PartnerRuntime.start makes) must NOT raise the surface."""
    rt, sup, registry, rec = paired_runtime(tmp_path)
    registry.set_status(rec.id, "linked")  # the owner's real state
    rt.stop_platform("whatsapp")
    rt.gateway.stop()

    registry2 = PairingRegistry(tmp_path / "paired.json")
    assert registry2.owner_stopped("whatsapp") is True
    sup2 = BridgeSupervisor(
        registry2, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions",
        spawn=lambda argv, env: FakeProc(),
        base_port=BASE_PORT,
        binary_resolver=lambda: "/staged/whatsapp-bridge")
    rec2 = registry2.get(rec.id)
    assert rec2 is not None and rec2.status == "linked"
    sup2.start(rec2.id)
    assert sup2.is_running(rec2.id)  # pool fully live again

    rt2 = make_runtime(tmp_path, sup2)
    try:
        assert rt2._sync_whatsapp_surface() == ""
        assert "whatsapp" not in rt2.gateway.adapters
        assert rt2._wa_surface_account == ""
    finally:
        rt2.gateway.stop()
        sup2.stop(rec2.id)


# ── 4. /pair stop/start/remove carry the same intent ────────────────

def test_pair_stop_and_start_paths(tmp_path):
    lines = ["PAIRING CODE for +2348012345678: WXYZ2345"]
    sup, registry, _ = make_supervisor(tmp_path, lines=lines)
    rt = make_runtime(tmp_path, sup)
    try:
        reply = rt._control_pair("whatsapp 2348012345678")
        assert "WXYZ2345" in reply
        rec = registry.all()[0]
        assert "whatsapp" in rt.gateway.adapters
        assert registry.owner_stopped("whatsapp") is False

        reply = rt._control_pair(f"stop {rec.id}")
        assert "bridge stopped" in reply
        assert registry.owner_stopped("whatsapp") is True
        assert "whatsapp" not in rt.gateway.adapters
        # …and it stays down under the beat + reconcile.
        svc = rt._pairing_personal()
        svc["last_cycle"] = 0.0
        rt._housekeeping_beat()
        assert "whatsapp" not in rt.gateway.adapters

        reply = rt._control_pair(f"start {rec.id}")
        assert "bridge starting" in reply
        assert registry.owner_stopped("whatsapp") is False
        assert "whatsapp" in rt.gateway.adapters
    finally:
        rt.gateway.stop()


def test_pair_remove_clears_intent(tmp_path):
    rt, sup, registry, rec = paired_runtime(tmp_path)
    try:
        rt.stop_platform("whatsapp")
        assert registry.owner_stopped("whatsapp") is True
        reply = rt._control_pair(f"remove {rec.id}")
        assert "removed from the pool" in reply
        assert registry.get(rec.id) is None
        assert registry.owner_stopped("whatsapp") is False
    finally:
        rt.gateway.stop()


def test_fresh_pair_clears_intent(tmp_path):
    lines = ["PAIRING CODE for +2348012345678: WXYZ2345"]
    sup, registry, _ = make_supervisor(tmp_path, lines=lines)
    rt = make_runtime(tmp_path, sup)
    try:
        registry.set_owner_stopped("whatsapp", True)
        reply = rt._control_pair("whatsapp 2348012345678")
        assert "WXYZ2345" in reply
        assert registry.owner_stopped("whatsapp") is False
        assert "whatsapp" in rt.gateway.adapters
    finally:
        rt.gateway.stop()


# ── 5. /platforms tells the truth ───────────────────────────────────

def test_platforms_renders_owner_stopped(tmp_path):
    sup, registry, _ = make_supervisor(tmp_path)
    rt = make_runtime(tmp_path, sup)
    plain = rt.handle_control("/platforms", "local:owner")
    assert "whatsapp: not running — start with /start whatsapp" \
        in plain
    registry.set_owner_stopped("whatsapp", True)
    stopped = rt.handle_control("/platforms", "local:owner")
    assert "whatsapp: stopped (by you) — /start whatsapp " \
        "to resume" in stopped
    assert "whatsapp: not running" not in stopped
