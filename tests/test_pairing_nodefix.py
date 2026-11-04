"""Node resolution + honest failure for the WhatsApp bridge (§15).

The live bug this pins: on a machine where the Devon process
cannot see node (a service account whose PATH never had it),
``/pair whatsapp`` died inside subprocess with a FileNotFoundError
that the surface rendered as a traceback digest — and the account
record had already been written as "pairing" with no process
behind it.  Now: node is RESOLVED (PATH, then the platform
install ladder), a launch that cannot happen raises
PairingUnavailable with plain install guidance BEFORE any record
exists, the surface renders that guidance as a normal reply, the
supervision sweep records the named reason instead of
crash-looping, and Telegram pairing — which never needed node —
is untouched by all of it.

The node rung above is now the SECOND engine: the sections at
the end of this file pin the Go rung (the default) the same
way — the bridge program is present-and-probed, never
installed, and its absence fails with the same honesty.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from nomorals.social.pairing import (
    AccountRecord,
    BridgeSupervisor,
    PairingRegistry,
    PairingUnavailable,
    PoolSupervision,
    resolve_node,
)
from nomorals.social.telegram_pairing import TelegramPairer

NODE_NAME = "node.exe" if os.name == "nt" else "node"


def make_node_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    if os.name != "nt":
        path.chmod(0o755)
    return path


def empty_path_env(tmp_path: Path) -> dict:
    """An env whose PATH holds no node at all."""
    empty = tmp_path / "emptybin"
    empty.mkdir(exist_ok=True)
    return {"PATH": str(empty)}


# ── resolve_node: PATH, ladder, explicit, none ─────────────────────

def test_resolve_node_finds_path_node(tmp_path):
    node = make_node_file(tmp_path / "bin" / NODE_NAME)
    found = resolve_node(env={"PATH": str(node.parent)})
    assert found == str(node)


def test_resolve_node_finds_posix_ladder_location(tmp_path):
    early = tmp_path / "ladder-early" / "node"
    late = make_node_file(tmp_path / "ladder-late" / "node")
    found = resolve_node(env=empty_path_env(tmp_path),
                         ladder=[early, late])
    assert found == str(late)
    # first existing candidate wins once the earlier one exists too
    make_node_file(early)
    assert resolve_node(env=empty_path_env(tmp_path),
                        ladder=[early, late]) == str(early)


def test_resolve_node_finds_windows_program_files(tmp_path):
    node = make_node_file(
        tmp_path / "Program Files" / "nodejs" / "node.exe")
    env = empty_path_env(tmp_path)
    env["ProgramFiles"] = str(tmp_path / "Program Files")
    found = resolve_node(env=env, platform="windows",
                         home=tmp_path / "home")
    assert found == str(node)


def test_resolve_node_finds_nvm_symlink_target(tmp_path):
    node = make_node_file(tmp_path / "nvm-link" / "node.exe")
    env = empty_path_env(tmp_path)
    env["NVM_SYMLINK"] = str(tmp_path / "nvm-link")
    found = resolve_node(env=env, platform="windows",
                         home=tmp_path / "home")
    assert found == str(node)


def test_resolve_node_finds_nvm_windows_current(tmp_path):
    node = make_node_file(
        tmp_path / "Roaming" / "nvm" / "current" / "node.exe")
    env = empty_path_env(tmp_path)
    env["APPDATA"] = str(tmp_path / "Roaming")
    found = resolve_node(env=env, platform="windows",
                         home=tmp_path / "home")
    assert found == str(node)


def test_resolve_node_returns_none_when_nothing_exists(tmp_path):
    env = empty_path_env(tmp_path)
    assert resolve_node(env=env, platform="windows",
                        home=tmp_path / "home") is None
    assert resolve_node(env=env, platform="posix",
                        home=tmp_path / "home",
                        ladder=[]) is None


def test_resolve_node_explicit_override_wins_and_missing_fails(tmp_path):
    node = make_node_file(tmp_path / "custom" / NODE_NAME)
    # an explicit real file wins even with an empty PATH
    assert resolve_node(str(node),
                        env=empty_path_env(tmp_path)) == str(node)
    # an explicit name resolves on PATH
    assert resolve_node(NODE_NAME,
                        env={"PATH": str(node.parent)}) == str(node)
    # an explicit node that does not exist fails the resolution —
    # no silent fallback to some other node
    assert resolve_node(str(tmp_path / "nope" / NODE_NAME),
                        env=empty_path_env(tmp_path)) is None


# ── supervisor: resolution cache contract ──────────────────────────

def test_resolution_rechecked_while_missing_cached_once_found(tmp_path):
    calls = []

    def resolver():
        calls.append(1)
        return None if len(calls) == 1 else "/resolved/node"

    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions",
        spawn=lambda argv, env: None, node_resolver=resolver)
    assert sup.resolve_node_path() is None      # miss: not cached
    assert sup.resolve_node_path() == "/resolved/node"
    assert sup.resolve_node_path() == "/resolved/node"  # cached hit
    assert len(calls) == 2


# ── the live bug: no node → guidance, no zombie ────────────────────

def nodeless_supervisor(tmp_path):
    """Production-shaped supervisor (real default spawn) on a
    machine with no node anywhere."""
    registry = PairingRegistry(tmp_path / "paired.json")
    bridge = tmp_path / "whatsapp-bridge.mjs"
    bridge.write_text("// bridge\n", encoding="utf-8")
    sup = BridgeSupervisor(
        registry, bridge_path=bridge,
        sessions_root=tmp_path / "sessions",
        engine="node",
        node_resolver=lambda: None)
    return sup, registry


def test_pair_whatsapp_no_node_raises_guidance_no_zombie(tmp_path):
    sup, registry = nodeless_supervisor(tmp_path)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678", "personal")
    message = str(err.value)
    assert "Node.js" in message
    assert "/pair again" in message
    # no phantom record claims to be pairing
    assert registry.all() == []
    assert sup.counts(probe=False) == {
        "registered": 0, "connected": 0, "dropped": 0}
    # and the failure is repeatable, not state-poisoning
    with pytest.raises(PairingUnavailable):
        sup.pair_whatsapp("2348012345678", "personal")
    assert registry.all() == []


def test_pair_start_no_node_raises_guidance(tmp_path):
    sup, registry = nodeless_supervisor(tmp_path)
    rec = registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="stopped",
        creds_dir=str(tmp_path / "creds"), port=8799,
        created_at=1.0))
    with pytest.raises(PairingUnavailable, match="Node.js"):
        sup.start(rec.id)
    assert registry.get(rec.id).status == "stopped"  # untouched


def test_missing_bridge_names_the_bridge_not_node(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions",
        engine="node",
        node_resolver=lambda: sys.executable)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678")
    message = str(err.value)
    assert "bridge script is missing" in message
    assert "whatsapp-bridge.mjs" in message
    assert registry.all() == []


def test_spawn_oserror_becomes_unavailable_and_rolls_back(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")

    def bad_spawn(argv, env):
        raise FileNotFoundError(2, "No such file or directory")

    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=bad_spawn,
        engine="node",
        node_resolver=lambda: "/resolved/node")
    with pytest.raises(PairingUnavailable):
        sup.pair_whatsapp("2348012345678")
    assert registry.all() == []  # created for the attempt, rolled back


def test_launch_argv_carries_resolved_absolute_node(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    fake_node = make_node_file(tmp_path / "nodejs" / NODE_NAME)
    spawns = []

    class _Proc:
        stdout = iter(())
        def poll(self):
            return None
        def terminate(self):
            pass

    def fake_spawn(argv, env):
        spawns.append(argv)
        return _Proc()

    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        engine="node",
        node_resolver=lambda: str(fake_node))
    sup.pair_whatsapp("2348012345678")
    assert spawns and spawns[0][0] == str(fake_node)
    assert Path(spawns[0][0]).is_absolute()


# ── the /pair surface renders guidance, not a digest ──────────────

def bare_runtime(sup):
    from nomorals.agents.partner.runtime_provision import \
        RuntimeProvisionMixin

    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = sup
    return rt


def test_pair_command_whatsapp_guidance_not_digest(tmp_path):
    sup, _registry = nodeless_supervisor(tmp_path)
    reply = bare_runtime(sup)._control_pair(
        "whatsapp 2348012345678 personal", chat_key="owner")
    assert "Node.js" in reply
    assert "blew up" not in reply
    assert "FileNotFoundError" not in reply
    assert "Traceback" not in reply


def test_pair_command_start_guidance_not_digest(tmp_path):
    sup, registry = nodeless_supervisor(tmp_path)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="stopped",
        creds_dir=str(tmp_path / "creds"), port=8799,
        created_at=1.0))
    for tail in ("start whatsapp:2348012345678",
                 "restart whatsapp:2348012345678"):
        reply = bare_runtime(sup)._control_pair(tail,
                                                chat_key="owner")
        assert "Node.js" in reply, tail
        assert "blew up" not in reply, tail


# ── telegram pairing never needed node ─────────────────────────────

class _FakeTGClient:
    def __init__(self):
        self.code_sent_to = ""

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def send_code(self, phone):
        self.code_sent_to = phone
        return "hash-abc"

    async def sign_in_code(self, phone, code, code_hash):
        pass

    async def sign_in_password(self, password):
        pass

    async def is_authorized(self):
        return True

    async def me_id(self):
        return "777001"


def test_telegram_pairing_works_with_node_absent(tmp_path):
    sup, registry = nodeless_supervisor(tmp_path)
    client = _FakeTGClient()
    pairer = TelegramPairer(
        registry, sessions_root=tmp_path / "tg",
        api_id=12345, api_hash="hash",
        client_factory=lambda stem, i, h: client)
    rec = pairer.start("2348012345678")
    assert registry.get(rec.id).status == "pairing"
    assert client.code_sent_to == "2348012345678"
    # the WhatsApp side is honestly blocked on the same registry…
    with pytest.raises(PairingUnavailable):
        sup.pair_whatsapp("2349098765432")
    # …and the Telegram flow is exactly where it was
    assert registry.get(rec.id).status == "pairing"
    assert registry.get("whatsapp:2349098765432") is None


# ── supervision records the named reason, never crash-loops ───────

def test_sweep_records_blocked_reason_when_node_missing(tmp_path):
    sup, registry = nodeless_supervisor(tmp_path)
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="linked",
        creds_dir=str(tmp_path / "creds"), port=8799,
        created_at=1.0))
    clock = [1000.0]
    sweep = PoolSupervision(registry, sup, clock=lambda: clock[0])
    changes = sweep.sweep()
    assert registry.get("whatsapp:2348012345678").status == "down"
    blocked = [c for c in changes if "blocked" in c["change"]]
    assert blocked and "Node.js" in blocked[0]["change"]
    # throttled: a second pass at the same beat stays quiet rather
    # than re-recording (or raising) on every sweep
    assert sweep.sweep() == []


# ── the Go rung: the default engine needs no node at all ───────────

from nomorals.social.pairing import (  # noqa: E402
    bridge_binary_name, bridge_binary_path)


def test_bridge_binary_name_per_platform():
    assert bridge_binary_name("nt") == "whatsapp-bridge.exe"
    assert bridge_binary_name("posix") == "whatsapp-bridge-linux"
    path = bridge_binary_path(repo_root="/repo", os_name="nt")
    assert path == Path("/repo/bridge/bin/whatsapp-bridge.exe")


def test_default_engine_is_go_and_env_selects_node(tmp_path,
                                                    monkeypatch):
    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions")
    assert sup.engine == "go"
    monkeypatch.setenv("NM_WHATSAPP_BRIDGE_ENGINE", "node")
    sup2 = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions")
    assert sup2.engine == "node"
    sup3 = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions", engine="node")
    assert sup3.engine == "node"  # explicit arg beats the default


def binaryless_supervisor(tmp_path):
    """Production-shaped supervisor (real default spawn) on a
    machine where the bridge program is absent."""
    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "whatsapp-bridge.mjs",
        sessions_root=tmp_path / "sessions",
        binary_resolver=lambda: None)
    return sup, registry


def test_pair_whatsapp_no_binary_raises_guidance_no_zombie(tmp_path):
    sup, registry = binaryless_supervisor(tmp_path)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678", "personal")
    message = str(err.value)
    assert "bridge program is missing" in message
    assert "/pair again" in message
    assert registry.all() == []
    assert sup.counts(probe=False) == {
        "registered": 0, "connected": 0, "dropped": 0}
    with pytest.raises(PairingUnavailable):
        sup.pair_whatsapp("2348012345678", "personal")
    assert registry.all() == []


def test_pair_command_binary_guidance_not_digest(tmp_path):
    sup, _registry = binaryless_supervisor(tmp_path)
    reply = bare_runtime(sup)._control_pair(
        "whatsapp 2348012345678 personal", chat_key="owner")
    assert "bridge program is missing" in reply
    assert "blew up" not in reply
    assert "FileNotFoundError" not in reply
    assert "Traceback" not in reply


def test_binary_resolution_rechecked_while_missing_then_cached(
        tmp_path):
    calls = []

    def resolver():
        calls.append(1)
        return None if len(calls) == 1 else "/resolved/bridge"

    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions",
        spawn=lambda argv, env: None, binary_resolver=resolver)
    assert sup.resolve_bridge_binary() is None      # miss: open
    assert sup.resolve_bridge_binary() == "/resolved/bridge"
    assert sup.resolve_bridge_binary() == "/resolved/bridge"
    assert len(calls) == 2                          # hit: cached


def test_go_launch_argv_carries_the_binary(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    class _Proc:
        stdout = iter(())
        def poll(self):
            return None
        def terminate(self):
            pass

    def fake_spawn(argv, env):
        spawns.append(argv)
        return _Proc()

    staged = tmp_path / "bridge-bin" / "whatsapp-bridge"
    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        binary_resolver=lambda: str(staged))
    sup.pair_whatsapp("2348012345678")
    assert spawns and spawns[0][0] == str(staged)
    assert Path(spawns[0][0]).is_absolute()
    assert not any(a.endswith(".mjs") for a in spawns[0])


def test_per_pair_engine_override_uses_node_rung(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    fake_node = make_node_file(tmp_path / "nodejs" / NODE_NAME)
    spawns = []

    class _Proc:
        stdout = iter(())
        def poll(self):
            return None
        def terminate(self):
            pass

    def fake_spawn(argv, env):
        spawns.append(argv)
        return _Proc()

    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        node_resolver=lambda: str(fake_node),
        binary_resolver=lambda: "/staged/whatsapp-bridge")
    assert sup.engine == "go"
    sup.pair_whatsapp("2348012345678", engine="node")
    assert spawns[0][0] == str(fake_node)
    assert spawns[0][1].endswith("bridge.mjs")


def test_live_entry_engine_sticks_for_relaunch(tmp_path):
    registry = PairingRegistry(tmp_path / "paired.json")
    fake_node = make_node_file(tmp_path / "nodejs" / NODE_NAME)
    spawns = []

    class _DeadProc:
        stdout = iter(())
        def poll(self):
            return 0
        def terminate(self):
            pass

    def fake_spawn(argv, env):
        spawns.append(argv)
        return _DeadProc()

    sup = BridgeSupervisor(
        registry, bridge_path=tmp_path / "bridge.mjs",
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        node_resolver=lambda: str(fake_node),
        binary_resolver=lambda: "/staged/whatsapp-bridge")
    sup.pair_whatsapp("2348012345678", engine="node")
    # the process died but its live entry remains — a relaunch
    # through start() keeps the account's engine, not the default
    sup.start("whatsapp:2348012345678")
    assert len(spawns) == 2
    assert spawns[-1][0] == str(fake_node)
