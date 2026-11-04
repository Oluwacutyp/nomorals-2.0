"""Bridge self-provisioning + death/refusal truth (bridgefix).

The live bug this pins: on the owner's EC2, ``/pair whatsapp``
said "bridge started" and no code ever arrived.  The bridge's
npm packages are gitignored, so a fresh checkout has none; the
bridge process died on its first import, the supervisor's drain
thread ended without a word, the record sat at "pairing"
forever, and ``/pair code`` said "no live code" — every layer
swallowed the one line that named the problem.  And one layer
up: the machine had no Node.js at all, and pairing answered
with install chores for the owner.

Now: Devon provisions the environment herself — Node.js when
it is missing (winget / official MSI / package manager, one
attempt, attempts reported when every route fails), the
bridge's npm packages when node_modules is absent (one install
per missing state) — and the bridge's output is never
swallowed: a death before linking flips the record to "down"
with the bridge's own last line as the reason, and a WhatsApp
refusal of the code request is what ``/pair code`` reports.
All installs run through injected seams here — no real install
ever runs in a test.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import pytest

import nomorals.social.pairing as pairing_mod
from nomorals.social.pairing import (
    NODE_MSI_URL,
    AccountRecord,
    BridgeSupervisor,
    NodeProvisionFailed,
    PairingRegistry,
    PairingUnavailable,
    provision_node,
    resolve_npm,
)


# ── fakes ──────────────────────────────────────────────────────────

class FakeProc:
    """A bridge process: yields its lines, then either dies
    (poll() → 0, like a process whose stdout closed) or parks
    alive on a gate when block=True (a running bridge)."""

    def __init__(self, lines=(), die=True, block=False):
        self.terminated = False
        self._poll = None
        self._gate = threading.Event()
        self.stdout = self._stream(list(lines), die, block)

    def _stream(self, lines, die, block):
        for line in lines:
            yield line
        if block:
            self._gate.wait(30)
        elif die:
            self._poll = 0

    def poll(self):
        return self._poll

    def terminate(self):
        self.terminated = True
        self._poll = 0
        self._gate.set()


def wait_for(pred, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(0.02)
    return bool(pred())


def bare_bridge(tmp_path) -> Path:
    path = tmp_path / "whatsapp-bridge.mjs"
    path.write_text("// bridge\n", encoding="utf-8")
    return path


def make_sup(tmp_path, proc_factory, **kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = proc_factory()
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path=bare_bridge(tmp_path),
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        engine="node",
        node_resolver=lambda: "/staged/node", **kw)
    return sup, registry, spawns


def bare_runtime(sup):
    from nomorals.agents.partner.runtime_provision import \
        RuntimeProvisionMixin

    rt = object.__new__(RuntimeProvisionMixin)
    rt._pair_supervisor = sup
    return rt


DEATH_LINES = [
    "node:internal/modules/esm/resolve:873",
    "  throw new ERR_MODULE_NOT_FOUND(packageName, "
    "fileURLToPath(base), null);",
    "        ^",
    "",
    "Error [ERR_MODULE_NOT_FOUND]: Cannot find package "
    "'@whiskeysockets/baileys' imported from "
    "/srv/devon/bridge/whatsapp-bridge.mjs",
    "    at packageResolve (node:internal/modules/esm/resolve:873:9)",
    "    at moduleResolve (node:internal/modules/esm/resolve:946:18)",
    "Node.js v22.14.0",
]

CODE_LINE = ("2026-11-04T01:00:00.000Z [bridge] PAIRING CODE for "
             "+2348012345678: WXYZ-1234 (type into WhatsApp > "
             "Linked Devices)")


# ── bridge death speaks ────────────────────────────────────────────

def test_bridge_death_speaks_with_its_own_line(tmp_path):
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(DEATH_LINES, die=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: registry.get(rec.id).status == "down")
    reason = sup.pairing_error(rec.id)
    assert reason.startswith("bridge exited:")
    assert "Cannot find package '@whiskeysockets/baileys'" in reason
    # the stored record carries it too (survives restarts)
    assert registry.get(rec.id).last_error == reason
    assert sup.pairing_code(rec.id) is None
    row = [r for r in sup.status_all(probe=False)
           if r["id"] == rec.id][0]
    assert row["status"] == "down"
    assert "bridge exited" in row["last_error"]


def test_pair_code_surface_reports_death_not_no_live_code(tmp_path):
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(DEATH_LINES, die=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: registry.get(rec.id).status == "down")
    reply = bare_runtime(sup)._control_pair(
        "code whatsapp:2348012345678", chat_key="owner")
    assert "bridge exited" in reply
    assert "Cannot find package" in reply
    assert "no live code" not in reply


def test_pair_whatsapp_surface_reports_death_during_wait(tmp_path):
    sup, _registry, _ = make_sup(
        tmp_path, lambda: FakeProc(DEATH_LINES, die=True))
    reply = bare_runtime(sup)._control_pair(
        "whatsapp 2348012345678", chat_key="owner")
    assert "bridge exited" in reply
    assert "Cannot find package" in reply
    assert "hasn't issued" not in reply


def test_start_surface_reports_instant_death(tmp_path):
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(DEATH_LINES, die=True))
    registry.upsert(AccountRecord(
        id="whatsapp:2348012345678", platform="whatsapp",
        identity="2348012345678", status="stopped",
        creds_dir=str(tmp_path / "creds"), port=8799,
        created_at=1.0))
    reply = bare_runtime(sup)._control_pair(
        "start whatsapp:2348012345678", chat_key="owner")
    assert "bridge exited" in reply
    assert "bridge starting" not in reply


def test_summary_row_carries_death_reason(tmp_path):
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(DEATH_LINES, die=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: registry.get(rec.id).status == "down")
    reply = bare_runtime(sup)._control_pair("", chat_key="owner")
    assert "bridge exited" in reply
    assert "Cannot find package" in reply


# ── pairing refusals speak ─────────────────────────────────────────

def test_pairing_refusal_speaks_and_bridge_lives(tmp_path):
    lines = ["2026-11-04T01:00:00.000Z [bridge] pairing code "
             "request failed: too many attempts"]
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(lines, block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: bool(sup.pairing_error(rec.id)))
    assert sup.pairing_error(rec.id) == (
        "WhatsApp refused the code request: too many attempts")
    # a refusal is not a death: the bridge lives, still pairing
    assert registry.get(rec.id).status == "pairing"
    assert sup.pairing_code(rec.id) is None
    reply = bare_runtime(sup)._control_pair(
        "code whatsapp:2348012345678", chat_key="owner")
    assert "WhatsApp refused the code request: too many attempts" \
        in reply
    assert "no live code" not in reply


# ── happy path + linked death are untouched ────────────────────────

def test_happy_path_code_captured_no_error(tmp_path):
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc([CODE_LINE], block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: sup.pairing_code(rec.id) is not None)
    code, _age = sup.pairing_code(rec.id)
    assert code == "WXYZ-1234"
    assert sup.pairing_error(rec.id) == ""
    assert registry.get(rec.id).status == "pairing"
    assert registry.get(rec.id).last_error == ""


def test_linked_account_bridge_death_not_rewritten(tmp_path):
    lines = [CODE_LINE,
             "2026-11-04T01:00:05.000Z [bridge] connected as "
             "2348012345678@s.whatsapp.net"]
    sup, registry, _ = make_sup(
        tmp_path, lambda: FakeProc(lines, die=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(
        lambda: registry.get(rec.id).status == "linked")
    # let the drain observe the exit and run its death handling
    assert wait_for(lambda: sup._live[rec.id].exited)
    # supervision owns a linked account's dead bridge — the
    # pairing layer must not rewrite it as a pairing failure
    assert registry.get(rec.id).status == "linked"
    assert registry.get(rec.id).last_error == ""
    assert sup.pairing_error(rec.id) == ""


# ── bridge dependencies: Devon installs them herself ───────────────

def stage_bridge_pkg(tmp_path, *, lock=True, modules=False):
    bdir = tmp_path / "bridge"
    bdir.mkdir(exist_ok=True)
    script = bdir / "whatsapp-bridge.mjs"
    script.write_text("// bridge\n", encoding="utf-8")
    (bdir / "package.json").write_text(
        '{"name": "devon-whatsapp-bridge", "dependencies": '
        '{"@whiskeysockets/baileys": "^6.7.18"}}',
        encoding="utf-8")
    if lock:
        (bdir / "package-lock.json").write_text("{}", encoding="utf-8")
    if modules:
        (bdir / "node_modules").mkdir()
    return script


def deps_sup(tmp_path, runner, *, lock=True, modules=False,
             npm="/staged/npm"):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = FakeProc(block=True)
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry,
        bridge_path=stage_bridge_pkg(tmp_path, lock=lock,
                                     modules=modules),
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        engine="node",
        node_resolver=lambda: "/staged/node",
        npm_resolver=lambda node: npm,
        deps_runner=runner)
    return sup, registry, spawns


def ok_runner(calls):
    def run(argv, cwd):
        calls.append((list(argv), cwd))
        (Path(cwd) / "node_modules").mkdir(exist_ok=True)
        return 0, "added 87 packages in 4s"
    return run


def test_missing_deps_installs_once_with_ci(tmp_path):
    calls = []
    sup, registry, spawns = deps_sup(tmp_path, ok_runner(calls))
    sup.pair_whatsapp("2348012345678")
    sup.pair_whatsapp("2349098765432")
    assert len(calls) == 1
    argv, cwd = calls[0]
    assert argv == ["/staged/npm", "ci"]
    assert cwd == tmp_path / "bridge"
    assert len(spawns) == 2  # both launches proceeded


def test_missing_deps_without_lockfile_uses_install(tmp_path):
    calls = []
    sup, _registry, _spawns = deps_sup(
        tmp_path, ok_runner(calls), lock=False)
    sup.pair_whatsapp("2348012345678")
    assert calls and calls[0][0][1] == "install"


def test_deps_already_present_never_installs(tmp_path):
    calls = []
    sup, _registry, spawns = deps_sup(
        tmp_path, ok_runner(calls), modules=True)
    sup.pair_whatsapp("2348012345678")
    assert calls == []
    assert len(spawns) == 1


def test_deps_install_failure_unavailable_no_zombie(tmp_path):
    def failing(argv, cwd):
        return 1, ("npm ERR! network read ECONNRESET\n"
                   "npm ERR! network aborted")

    sup, registry, spawns = deps_sup(tmp_path, failing)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678")
    message = str(err.value)
    assert "npm ci" in message
    assert "ECONNRESET" in message  # npm's own tail, named
    assert registry.all() == []     # no record left claiming pair
    assert spawns == []             # and no bridge was launched


def test_deps_npm_unresolvable_unavailable(tmp_path):
    calls = []
    sup, registry, _spawns = deps_sup(
        tmp_path, ok_runner(calls), npm=None)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678")
    assert "npm was not found" in str(err.value)
    assert calls == []
    assert registry.all() == []


def test_deps_install_once_under_concurrent_launches(tmp_path):
    calls = []

    def slow_ok(argv, cwd):
        time.sleep(0.3)
        calls.append(list(argv))
        (Path(cwd) / "node_modules").mkdir(exist_ok=True)
        return 0, "added 87 packages"

    sup, registry, spawns = deps_sup(tmp_path, slow_ok)
    threads = [
        threading.Thread(target=sup.pair_whatsapp, args=(number,))
        for number in ("2348012345678", "2349098765432")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert len(calls) == 1          # serialized: exactly one npm
    assert len(spawns) == 2
    assert len(registry.all()) == 2


def test_resolve_npm_beside_node_then_path(tmp_path):
    bindir = tmp_path / "nodejs"
    bindir.mkdir()
    node = bindir / "node"
    node.write_text("x", encoding="utf-8")
    npm_cmd = bindir / "npm.cmd"
    npm_cmd.write_text("x", encoding="utf-8")
    assert resolve_npm(str(node), env={"PATH": ""}) == str(npm_cmd)
    other = tmp_path / "elsewhere"
    other.mkdir()
    npm = other / "npm"
    npm.write_text("x", encoding="utf-8")
    if os.name != "nt":
        npm.chmod(0o755)
    assert resolve_npm("", env={"PATH": str(other)}) == str(npm)
    assert resolve_npm("", env={"PATH": str(tmp_path / "nope")}) \
        is None


# ── Node.js itself: Devon installs it when missing ─────────────────

def staged_tool(tmp_path, name):
    p = tmp_path / "bin" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    p.chmod(0o755)
    return p


def test_provision_windows_winget_success(tmp_path):
    staged_tool(tmp_path, "winget")
    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        return 0, ""

    found = provision_node(
        env={"PATH": str(tmp_path / "bin")}, platform="windows",
        home=tmp_path / "home", runner=runner,
        resolver=lambda *a, **kw: "C:\\Program Files\\nodejs\\node.exe")
    assert found == "C:\\Program Files\\nodejs\\node.exe"
    assert calls and calls[0][1:] == [
        "install", "-e", "--id", "OpenJS.NodeJS.LTS",
        "--accept-source-agreements",
        "--accept-package-agreements"]


def test_provision_windows_msi_fallback(tmp_path):
    empty = tmp_path / "bin"
    empty.mkdir()
    downloads = []

    def downloader(url, dest):
        downloads.append(url)
        Path(dest).write_bytes(b"msi")

    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        return 0, ""

    found = provision_node(
        env={"PATH": str(empty)}, platform="windows",
        home=tmp_path / "home", runner=runner,
        downloader=downloader,
        resolver=lambda *a, **kw: "C:\\Program Files\\nodejs\\node.exe")
    assert found == "C:\\Program Files\\nodejs\\node.exe"
    assert downloads == [NODE_MSI_URL]
    assert calls and calls[0][0] == "msiexec"
    assert "/quiet" in calls[0] and "/norestart" in calls[0]


def test_provision_windows_all_routes_fail_with_attempts(tmp_path):
    empty = tmp_path / "bin"
    empty.mkdir()

    def downloader(url, dest):
        raise OSError("network unreachable")

    with pytest.raises(NodeProvisionFailed) as err:
        provision_node(
            env={"PATH": str(empty)}, platform="windows",
            home=tmp_path / "home",
            runner=lambda argv, cwd: (0, ""),
            downloader=downloader,
            resolver=lambda *a, **kw: None)
    attempts = err.value.attempts
    assert any("winget is not available" in a for a in attempts)
    assert any("could not be downloaded" in a for a in attempts)


def test_provision_posix_no_privilege_skips_install(tmp_path):
    staged_tool(tmp_path, "apt-get")
    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        return 0, ""

    with pytest.raises(NodeProvisionFailed) as err:
        provision_node(
            env={"PATH": str(tmp_path / "bin")}, platform="posix",
            home=tmp_path / "home", runner=runner,
            resolver=lambda *a, **kw: None, euid=1000)
    assert calls == []  # never attempted, never a sudo prompt
    assert any("root" in a for a in err.value.attempts)


def test_provision_posix_root_apt_success(tmp_path):
    staged_tool(tmp_path, "apt-get")
    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        return 0, ""

    found = provision_node(
        env={"PATH": str(tmp_path / "bin")}, platform="posix",
        home=tmp_path / "home", runner=runner,
        resolver=lambda *a, **kw: "/usr/bin/node", euid=0)
    assert found == "/usr/bin/node"
    assert calls and calls[0][0].endswith("apt-get")
    assert "nodejs" in calls[0]


def test_provision_posix_termux_pkg_needs_no_root(tmp_path):
    staged_tool(tmp_path, "pkg")
    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        return 0, ""

    env = {"PATH": str(tmp_path / "bin"),
           "PREFIX": "/data/data/com.termux/files/usr"}
    found = provision_node(
        env=env, platform="posix", home=tmp_path / "home",
        runner=runner, resolver=lambda *a, **kw: "/data/data/"
        "com.termux/files/usr/bin/node", euid=1000)
    assert found.endswith("/bin/node")
    assert calls and calls[0][0].endswith("pkg")


# ── supervisor: provisioning wired into resolution ──────────────────

def force_node_missing(monkeypatch, staged_node):
    state = {"installed": False}

    def fake_resolve(explicit="", **kw):
        return str(staged_node) if state["installed"] else None

    monkeypatch.setattr(pairing_mod, "resolve_node", fake_resolve)
    return state


def test_provisioned_node_is_used_for_the_launch(tmp_path,
                                                  monkeypatch):
    staged_node = tmp_path / "installed" / "node"
    state = force_node_missing(monkeypatch, staged_node)
    calls = []

    def provisioner():
        calls.append(1)
        state["installed"] = True
        return pairing_mod.resolve_node()

    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        spawns.append(argv)
        return FakeProc(block=True)

    sup = BridgeSupervisor(
        registry, bridge_path=bare_bridge(tmp_path),
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        engine="node",
        node_provisioner=provisioner)
    sup.pair_whatsapp("2348012345678")
    assert calls == [1]
    # the launch ran with the node the install produced
    assert spawns and spawns[0][0] == str(staged_node)


def test_provision_failure_lists_attempts_attempted_once(
        tmp_path, monkeypatch):
    force_node_missing(monkeypatch, tmp_path / "installed" / "node")
    calls = []

    def provisioner():
        calls.append(1)
        raise NodeProvisionFailed([
            "winget install of Node.js LTS (OpenJS.NodeJS.LTS) "
            "failed (exit 1): boom",
            "the official Node.js installer "
            "(node-v22.14.0-x64.msi) failed (exit 1603): fatal"])

    registry = PairingRegistry(tmp_path / "paired.json")
    sup = BridgeSupervisor(
        registry, bridge_path=bare_bridge(tmp_path),
        sessions_root=tmp_path / "sessions",
        spawn=lambda argv, env: FakeProc(),
        engine="node",
        node_provisioner=provisioner)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678")
    message = str(err.value)
    assert "tried to install it herself" in message
    assert "winget" in message
    assert "official Node.js installer" in message
    assert "/pair again" in message
    assert registry.all() == []
    # one attempt per missing state: the second /pair does not
    # reinstall-loop — it re-raises with the recorded attempts
    with pytest.raises(PairingUnavailable) as err2:
        sup.pair_whatsapp("2348012345678")
    assert str(err2.value) == message
    assert calls == [1]
    assert registry.all() == []


# ── the Go rung: same drain truth, its own failure shapes ──────────

GO_DEATH_LINES = [
    "panic: runtime error: invalid memory address or nil "
    "pointer dereference",
    "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0]",
    "",
    "goroutine 1 [running]:",
    "main.(*bridge).onQR(0xc0000)",
    "\tbridge/gobridge/main.go:560 +0x1f3",
    "exit status 2",
]


def make_sup_go(tmp_path, proc_factory, **kw):
    registry = PairingRegistry(tmp_path / "paired.json")
    spawns = []

    def fake_spawn(argv, env):
        proc = proc_factory()
        spawns.append((argv, env, proc))
        return proc

    sup = BridgeSupervisor(
        registry, bridge_path=bare_bridge(tmp_path),
        sessions_root=tmp_path / "sessions", spawn=fake_spawn,
        binary_resolver=kw.pop(
            "binary_resolver",
            lambda: "/staged/whatsapp-bridge"),
        **kw)
    return sup, registry, spawns


def test_go_rung_launches_the_binary_not_a_script(tmp_path):
    sup, _registry, spawns = make_sup_go(
        tmp_path, lambda: FakeProc(block=True))
    assert sup.engine == "go"  # the binary is the default rung
    sup.pair_whatsapp("2348012345678")
    argv, env, _proc = spawns[0]
    assert argv[0] == "/staged/whatsapp-bridge"
    assert not any(a.endswith(".mjs") for a in argv)
    assert "--creds" in argv and "--port" in argv
    assert env["WA_PAIRING_CODE"] == "1"
    assert env["WA_PAIRING_NUMBER"] == "2348012345678"


def test_node_rung_still_launches_node_and_script(tmp_path):
    sup, _registry, spawns = make_sup(
        tmp_path, lambda: FakeProc(block=True))
    sup.pair_whatsapp("2348012345678")
    argv, _env, _proc = spawns[0]
    assert argv[0] == "/staged/node"
    assert argv[1].endswith("whatsapp-bridge.mjs")


def test_go_bridge_death_speaks_with_its_panic_line(tmp_path):
    sup, registry, _ = make_sup_go(
        tmp_path, lambda: FakeProc(GO_DEATH_LINES, die=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: registry.get(rec.id).status == "down")
    reason = sup.pairing_error(rec.id)
    assert reason.startswith("bridge exited:")
    assert "panic: runtime error" in reason
    assert registry.get(rec.id).last_error == reason
    assert sup.pairing_code(rec.id) is None


def test_go_pairing_refused_line_speaks(tmp_path):
    lines = ["2026-11-04T01:00:00.000Z [bridge] PAIRING REFUSED: "
             "too many attempts"]
    sup, registry, _ = make_sup_go(
        tmp_path, lambda: FakeProc(lines, block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: bool(sup.pairing_error(rec.id)))
    assert sup.pairing_error(rec.id) == (
        "WhatsApp refused the code request: too many attempts")
    assert registry.get(rec.id).status == "pairing"
    assert sup.pairing_code(rec.id) is None


def test_go_pairing_link_failed_line_speaks(tmp_path):
    lines = [CODE_LINE,
             "2026-11-04T01:00:09.000Z [bridge] PAIRING LINK "
             "FAILED: disconnected while pairing"]
    sup, registry, _ = make_sup_go(
        tmp_path, lambda: FakeProc(lines, block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: bool(sup.pairing_error(rec.id)))
    assert sup.pairing_error(rec.id) == (
        "WhatsApp closed the pairing: disconnected while pairing")


def test_no_code_ever_comes_from_chatter_or_refusal(tmp_path):
    # Phantom-code pin: bridge output mentioning a code outside
    # the confirmed line shape must never become the displayed
    # code — a code exists only when the PAIRING CODE line (the
    # one a bridge prints after the server confirmed) lands.
    lines = [
        "2026-11-04T01:00:00.000Z [bridge] your code is 483920",
        "2026-11-04T01:00:01.000Z [bridge] code: ABCD-EFGH",
        "2026-11-04T01:00:02.000Z [bridge] PAIRING REFUSED: "
        "rate limited",
    ]
    sup, registry, _ = make_sup_go(
        tmp_path, lambda: FakeProc(lines, block=True))
    rec = sup.pair_whatsapp("2348012345678")
    assert wait_for(lambda: bool(sup.pairing_error(rec.id)))
    assert sup.pairing_code(rec.id) is None
    assert registry.get(rec.id).last_error.startswith(
        "WhatsApp refused the code request:")


# ── the Go rung's provisioning: present + runs, nothing installed ──

def test_probe_runs_once_across_pairs(tmp_path):
    probes = []

    def probe(binary):
        probes.append(binary)
        return 0, "devon-whatsapp-bridge gobridge/1.0.0"

    sup, _registry, spawns = make_sup_go(
        tmp_path, lambda: FakeProc(block=True), probe_runner=probe)
    sup.pair_whatsapp("2348012345678")
    sup.pair_whatsapp("2349098765432")
    assert probes == ["/staged/whatsapp-bridge"]  # cached verdict
    assert len(spawns) == 2


def test_probe_failure_unavailable_no_zombie(tmp_path):
    def probe(binary):
        return 3, "exec format error"

    sup, registry, spawns = make_sup_go(
        tmp_path, lambda: FakeProc(block=True), probe_runner=probe)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678")
    message = str(err.value)
    assert "does not run" in message
    assert "exec format error" in message  # the probe's own tail
    assert registry.all() == []
    assert spawns == []


def test_missing_binary_unavailable_no_zombie(tmp_path):
    sup, registry, spawns = make_sup_go(
        tmp_path, lambda: FakeProc(block=True),
        binary_resolver=lambda: None)
    with pytest.raises(PairingUnavailable) as err:
        sup.pair_whatsapp("2348012345678")
    assert "bridge program is missing" in str(err.value)
    assert registry.all() == []
    assert spawns == []
