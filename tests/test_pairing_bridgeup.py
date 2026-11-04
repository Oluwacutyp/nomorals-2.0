"""Surface start ensures its bridge: never attached to a dead port.

The owner's live failure (2026-11-04, EC2 boot log): the runtime
announced "whatsapp surface live-attached to whatsapp:<acct>
(bridge port 8787)" — and immediately underneath, the adapter
looped forever on "whatsapp bridge unreachable at 127.0.0.1:8787
(WinError 10061 connection refused)", attempts 1–6 with growing
backoff.  No bridge-launch attempt appeared anywhere.  Census of
the old shape:

- The reconcile's desired test was ``status == "linked" or
  is_running`` — a registry status is PERSISTED, so at boot a
  stale "linked" qualified an account whose bridge process did
  not exist in the new session.  The adapter hot-started (no
  preflight on the start_one path) and spun on the dead port
  while the reconcile logged "live-attached" on the strength of
  the adapter thread alone.
- Nothing in the boot/start path ever launched a bridge:
  ``BridgeSupervisor.start`` was reached only from /pair
  start|restart and the supervision sweep — and the sweep runs
  only inside the message-driven housekeeping beat, never at
  boot.  ``/start whatsapp`` (start_platform) was a generic
  gateway start against the static config endpoint.

The fix, pinned here:

1.  ``BridgeSupervisor.ensure_running`` — the automatic ensure:
    alive → serving; listening on the record's port → serving
    (a survivor bridge is adopted, never double-launched);
    otherwise one launch through the same ``start()`` /pair
    uses (both engines, same preflights), with a throttled
    relaunch for a bridge that died right after launch.
2.  The reconcile ensures BEFORE it attaches: no serving bridge
    → no adapter, no "live-attached" claim; an owned adapter is
    put down; the verdict is a "bridge down — <reason>" note on
    the reply AND the account record, with ONE failure-ledger
    entry per episode.
3.  ``/start whatsapp`` routes through the reconcile (a
    pool-stopped account is revived through the supervisor
    first); /platforms renders "whatsapp: bridge down — …"
    whenever the resolved account's bridge is not serving.
4.  Stopfix is absolute: while an owner stop order stands, the
    reconcile launches NEITHER bridge nor adapter.
"""
from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.agents.partner.failures import clear_failures, \
    recent_failures
from nomorals.agents.partner.runtime import PartnerRuntime
from nomorals.social.chat.gateway import ChatGateway
from nomorals.social.chat.local import LocalAdapter
from nomorals.social.pairing import (
    AccountRecord,
    BridgeSupervisor,
    PairingRegistry,
)

BASE_PORT = 18931
ACCT = "whatsapp:2348012345678"


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


class DeadProc:
    """A bridge process that has already exited: no output, the
    exit already reportable by poll()."""

    def __init__(self, lines=()):
        self.terminated = False
        self.stdout = iter(list(lines))

    def poll(self):
        return 0

    def terminate(self):
        self.terminated = True


def make_supervisor(tmp_path, base_port=BASE_PORT, lines=(),
                    spawn_fn=None, **sup_kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = (spawn_fn or FakeProc)(lines)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        base_port=base_port,
        binary_resolver=lambda: "/staged/whatsapp-bridge",
        **sup_kw)
    return sup, registry, spawns


def make_settings(tmp_path, *, whatsapp_enabled=True):
    chat = SimpleNamespace(
        whatsapp_enabled=whatsapp_enabled,
        whatsapp_host="127.0.0.1",
        whatsapp_port=8787,  # the static config endpoint — NOT the
        whatsapp_outbox_dir="",  # record's port; the surface must
        telegram_api_id="", telegram_api_hash="")  # follow the pool
    return SimpleNamespace(
        chat=chat, workspace_dir=str(tmp_path),
        resolve=lambda rel: Path(tmp_path) / rel)


def make_runtime(tmp_path, sup):
    rt = PartnerRuntime.__new__(PartnerRuntime)
    settings = make_settings(tmp_path)
    rt.settings = settings
    rt.context = SimpleNamespace(settings=settings)
    rt.gateway = ChatGateway({"local": LocalAdapter()}, db=None)
    rt._pair_supervisor = sup
    return rt


def linked_record(tmp_path, base_port=BASE_PORT):
    """The boot-time registry truth: an account that linked in a
    PREVIOUS Devon run — record persisted, status "linked", and
    (in the fresh supervisor) no process table entry at all."""
    return AccountRecord(
        id=ACCT, platform="whatsapp", identity="2348012345678",
        status="linked", port=base_port,
        creds_dir=str(tmp_path / "sessions" / "wa-2348012345678"))


def boot_runtime(tmp_path, **sup_kw):
    """A fresh Devon process over a persisted registry: new
    supervisor (empty process table), new runtime, one linked
    account, NO bridge launched yet."""
    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(linked_record(tmp_path))
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc()
        spawns.append((argv, env, proc))
        return proc

    sup_kw = {"binary_resolver":
              lambda: "/staged/whatsapp-bridge", **sup_kw}
    sup = BridgeSupervisor(
        registry, bridge_path="/repo/bridge/whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        base_port=BASE_PORT, **sup_kw)
    rt = make_runtime(tmp_path, sup)
    return rt, sup, registry, spawns


# ── 1. boot: the reconcile launches the bridge, then attaches ───────

def test_boot_reconcile_launches_bridge_then_attaches(tmp_path):
    """The owner's boot, fixed: registry says linked, no bridge
    process exists in this session — the reconcile must LAUNCH
    the account's bridge through the supervisor (its own port,
    the Go program) and only then attach the adapter."""
    rt, sup, registry, spawns = boot_runtime(tmp_path)
    try:
        assert not sup.is_running(ACCT)
        note = rt._sync_whatsapp_surface()
        assert note == ""
        assert len(spawns) == 1
        argv = spawns[0][0]
        assert argv[0] == "/staged/whatsapp-bridge"
        assert argv[argv.index("--port") + 1] == str(BASE_PORT)
        adapter = rt.gateway.adapters.get("whatsapp")
        assert adapter is not None
        assert adapter.port == BASE_PORT
        assert rt._wa_surface_account == ACCT
        # And a second reconcile is a no-op: no second launch.
        assert rt._sync_whatsapp_surface() == ""
        assert len(spawns) == 1
    finally:
        rt.gateway.stop()


def test_boot_survivor_bridge_is_adopted_not_double_launched(
        tmp_path):
    """A bridge Devon did not spawn is already listening on the
    record's port (it survived a Devon restart): the ensure
    adopts it — no second process is launched against the port."""
    rt, sup, registry, spawns = boot_runtime(tmp_path)
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", BASE_PORT))
    listener.listen(5)
    try:
        note = rt._sync_whatsapp_surface()
        assert note == ""
        assert spawns == []
        assert "whatsapp" in rt.gateway.adapters
        assert rt._wa_surface_account == ACCT
    finally:
        rt.gateway.stop()
        listener.close()


def test_boot_bridge_cannot_launch_reports_bridge_down(tmp_path):
    """The bridge program is missing: no adapter, no
    "live-attached" claim — a bridge-down note on the reply AND
    the account record, and ONE failure-ledger entry for the
    whole episode (the next beat must not spam it)."""
    clear_failures()
    rt, sup, registry, spawns = boot_runtime(
        tmp_path, binary_resolver=lambda: None)
    try:
        note = rt._sync_whatsapp_surface()
        assert "bridge down" in note
        assert "whatsapp" not in rt.gateway.adapters
        assert getattr(rt, "_wa_surface_account", "") == ""
        assert spawns == []
        stored = registry.get(ACCT)
        assert stored.last_error.startswith("chat surface: ")
        assert "bridge down" in stored.last_error
        # Same dead bridge, next beat: same note, still one entry.
        assert rt._sync_whatsapp_surface() == note
        hits = [f for f in recent_failures(50)
                if f.site == "runtime_provision.wa_bridge_down"]
        assert len(hits) == 1
        assert ACCT in hits[0].message
    finally:
        rt.gateway.stop()


# ── 2. mid-run death: teardown, one throttled relaunch ──────────────

def test_bridge_death_tears_surface_down_and_relaunch_throttles(
        tmp_path):
    sup, registry, spawns = make_supervisor(tmp_path)
    rt = make_runtime(tmp_path, sup)
    try:
        rec = sup.pair_whatsapp("2348012345678")
        registry.set_status(rec.id, "linked")
        assert rt._sync_whatsapp_surface() == ""
        assert "whatsapp" in rt.gateway.adapters
        assert len(spawns) == 1

        # The bridge process dies mid-run.
        spawns[0][2].terminate()
        assert not sup.is_running(rec.id)

        # Next reconcile: the surface comes DOWN (never left
        # "attached" to the dead port), the verdict is the
        # bridge-down note, and the relaunch is THROTTLED — the
        # bridge died seconds after launch, no fork-loop.
        note = rt._sync_whatsapp_surface()
        assert "bridge down" in note
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
        assert len(spawns) == 1

        # Once the throttle window has passed (the launch is no
        # longer fresh), the reconcile relaunches — exactly once
        # — and the surface returns.
        sup._live[rec.id].launched_at = 0.0
        assert rt._sync_whatsapp_surface() == ""
        assert len(spawns) == 2
        assert "whatsapp" in rt.gateway.adapters
        assert rt._wa_surface_account == rec.id
        # Recovery retired the bridge-down note on the record.
        assert registry.get(rec.id).last_error == ""
    finally:
        rt.gateway.stop()


# ── 3. owner stop is absolute: no bridge launch either ─────────────

def test_owner_stopped_boot_launches_neither_bridge_nor_adapter(
        tmp_path):
    rt, sup, registry, spawns = boot_runtime(tmp_path)
    try:
        registry.set_owner_stopped("whatsapp", True)
        assert rt._sync_whatsapp_surface() == ""
        assert spawns == []  # no bridge launch behind a stop order
        assert "whatsapp" not in rt.gateway.adapters
        assert rt._wa_surface_account == ""
    finally:
        rt.gateway.stop()


# ── 4. /start whatsapp goes through the supervisor ──────────────────

def test_start_platform_whatsapp_ensures_bridge(tmp_path):
    rt, sup, registry, spawns = boot_runtime(tmp_path)
    try:
        result = rt.start_platform("whatsapp")
        assert result.get("ok")
        assert len(spawns) == 1  # the bridge was launched too
        adapter = rt.gateway.adapters.get("whatsapp")
        assert adapter is not None and adapter.port == BASE_PORT
        # The control surface agrees, end to end.
        reply = rt.handle_control("/start whatsapp", "local:owner")
        assert reply == "whatsapp: started"
        assert len(spawns) == 1
    finally:
        rt.gateway.stop()


def test_start_platform_whatsapp_bridge_down_is_an_honest_error(
        tmp_path):
    rt, sup, registry, spawns = boot_runtime(
        tmp_path, binary_resolver=lambda: None)
    try:
        result = rt.start_platform("whatsapp")
        assert not result.get("ok")
        assert "bridge down" in result.get("error", "")
        assert "whatsapp" not in rt.gateway.adapters
    finally:
        rt.gateway.stop()


def test_start_platform_whatsapp_revives_pool_stopped_account(
        tmp_path):
    """An account stopped at the pool level (/pair stop's bridge
    half) is revived by an owner's /start whatsapp: supervisor
    start first, then the reconcile attaches."""
    sup, registry, spawns = make_supervisor(tmp_path)
    rt = make_runtime(tmp_path, sup)
    try:
        rec = sup.pair_whatsapp("2348012345678")
        registry.set_status(rec.id, "linked")
        sup.stop(rec.id)
        assert registry.get(rec.id).status == "stopped"
        assert not sup.is_running(rec.id)
        before = len(spawns)
        result = rt.start_platform("whatsapp")
        assert result.get("ok")
        assert len(spawns) == before + 1  # bridge relaunched
        assert "whatsapp" in rt.gateway.adapters
    finally:
        rt.gateway.stop()


# ── 5. /platforms renders the bridge truth ──────────────────────────

def test_platforms_renders_bridge_down_when_never_attached(tmp_path):
    rt, sup, registry, spawns = boot_runtime(tmp_path)
    try:
        reply = rt.handle_control("/platforms", "local:owner")
        assert ("whatsapp: bridge down — the bridge process is "
                "not running") in reply
        assert "whatsapp: running" not in reply
        assert spawns == []  # listing never launches anything
    finally:
        rt.gateway.stop()


def test_platforms_renders_bridge_down_when_adapter_stranded(
        tmp_path):
    """The owner's exact boot state, rendered honestly: adapter
    thread up (in gateway status) but its bridge dead — the line
    must be "bridge down", not "running"."""
    sup, registry, spawns = make_supervisor(tmp_path)
    rt = make_runtime(tmp_path, sup)
    try:
        rec = sup.pair_whatsapp("2348012345678")
        registry.set_status(rec.id, "linked")
        rt._sync_whatsapp_surface()
        assert "whatsapp" in rt.gateway.adapters
        spawns[0][2].terminate()
        reply = rt.handle_control("/platforms", "local:owner")
        assert "whatsapp: bridge down — " in reply
        assert "whatsapp: running" not in reply
    finally:
        rt.gateway.stop()


def test_platforms_renders_running_when_bridge_serves(tmp_path):
    sup, registry, spawns = make_supervisor(tmp_path)
    rt = make_runtime(tmp_path, sup)
    try:
        rec = sup.pair_whatsapp("2348012345678")
        registry.set_status(rec.id, "linked")
        rt._sync_whatsapp_surface()
        reply = rt.handle_control("/platforms", "local:owner")
        assert "whatsapp: running" in reply
        assert "bridge down" not in reply
    finally:
        rt.gateway.stop()


# ── 6. ensure_running, at the supervisor ────────────────────────────

def test_ensure_running_unknown_and_stopped_accounts(tmp_path):
    sup, registry, spawns = make_supervisor(tmp_path)
    ok, reason = sup.ensure_running("whatsapp:2348000000000")
    assert not ok and "no paired account" in reason
    registry.upsert(AccountRecord(
        id=ACCT, platform="whatsapp", identity="2348012345678",
        status="stopped", port=BASE_PORT,
        creds_dir=str(tmp_path / "sessions" / "wa-2348012345678")))
    ok, reason = sup.ensure_running(ACCT)
    assert not ok and "stopped" in reason
    assert spawns == []  # a stopped account is never auto-launched


def test_ensure_running_dead_on_launch_throttles_then_retries(
        tmp_path):
    sup, registry, spawns = make_supervisor(
        tmp_path, spawn_fn=lambda lines: DeadProc(lines))
    registry.upsert(linked_record(tmp_path))
    # Never launched this session: the launch happens at once.
    ok, reason = sup.ensure_running(ACCT)
    assert ok and reason == ""
    assert len(spawns) == 1
    # It died on the spot: inside the throttle window the ensure
    # reports the death instead of fork-looping a second launch.
    ok, reason = sup.ensure_running(ACCT)
    assert not ok and reason
    assert len(spawns) == 1
    # With the throttle lifted, the retry launches.
    ok, reason = sup.ensure_running(ACCT, relaunch_interval=0)
    assert ok and reason == ""
    assert len(spawns) == 2


def test_ensure_running_node_engine_launches_through_node(tmp_path):
    """Both engines ride the same ensure: a node-engine account
    launches as node + bridge script on the record's port."""
    registry = PairingRegistry(tmp_path / "paired.json")
    registry.upsert(AccountRecord(
        id=ACCT, platform="whatsapp", identity="2348012345678",
        status="linked", port=BASE_PORT,
        creds_dir=str(tmp_path / "sessions" / "wa-2348012345678")))
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc()
        spawns.append((argv, env, proc))
        return proc

    bridge = tmp_path / "bridge" / "whatsapp-bridge.mjs"
    sup = BridgeSupervisor(
        registry, bridge_path=bridge,
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        base_port=BASE_PORT, engine="node",
        node_resolver=lambda: "/staged/node")
    ok, reason = sup.ensure_running(ACCT)
    assert ok and reason == ""
    argv = spawns[0][0]
    assert argv[0] == "/staged/node"
    assert argv[1] == str(bridge)
    assert argv[argv.index("--port") + 1] == str(BASE_PORT)
