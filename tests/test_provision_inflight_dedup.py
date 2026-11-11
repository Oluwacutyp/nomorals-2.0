"""§3.9 — provision runs deduplicate in flight.

While an install of one tool is in flight, a second caller for
the SAME tool attaches to that run and receives its outcome
instead of starting a parallel install. These tests drive REAL
concurrent calls against counting fakes that block and fail on
cue — the concurrency actually happens, on threads, at:

* the shared registry (nomorals.core.inflight) directly;
* the bridge supervisor's Node install (two supervisor
  instances, one machine);
* the supervisor's bridge-deps npm install (failure used to
  retry once per waiter — the storm this closes);
* the PHP pairing page's PHP install;
* the computer organ's bridge build and Xvfb install.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from nomorals.computer import provision as comp_provision
from nomorals.core.inflight import (
    InflightRegistry,
    default_registry,
)
from nomorals.social import pairing as pairing_mod
from nomorals.social import pairing_web
from nomorals.social.pairing import (
    BridgeSupervisor,
    NodeProvisionFailed,
    PairingRegistry,
    PairingUnavailable,
)


def wait_for(cond, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def attached_count(key: str) -> int:
    flight = default_registry._flights.get(key)
    return flight.attached if flight is not None else 0


def run_threads(fns):
    """Start one thread per callable; return (threads, results,
    errors) — results/errors land by index."""
    results: list = [None] * len(fns)
    errors: list = [None] * len(fns)

    def wrap(i, fn):
        try:
            results[i] = fn()
        except BaseException as exc:  # noqa: BLE001 - test capture
            errors[i] = exc

    threads = [threading.Thread(target=wrap, args=(i, fn))
               for i, fn in enumerate(fns)]
    for t in threads:
        t.start()
    return threads, results, errors


def join_all(threads, timeout=15.0):
    for t in threads:
        t.join(timeout)
    assert all(not t.is_alive() for t in threads), \
        "a caller hung — the dedup deadlocked or lost a wakeup"


# ── the registry itself ────────────────────────────────────────

def test_concurrent_same_key_one_invocation_shared_outcome():
    reg = InflightRegistry()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    token = object()

    def install():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return token

    threads, results, errors = run_threads([
        lambda: reg.run("tool", install),
        lambda: reg.run("tool", install),
    ])
    assert entered.wait(10)
    assert wait_for(lambda: reg._flights["tool"].attached >= 1)
    assert calls == [1]  # the attacher started no second install
    release.set()
    join_all(threads)
    assert errors == [None, None]
    assert results[0] is token and results[1] is token
    assert calls == [1]


def test_failure_is_shared_by_all_attachers_not_retried():
    reg = InflightRegistry()
    entered = threading.Event()
    release = threading.Event()
    calls = []
    boom = ValueError("the installer died")

    def install():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        raise boom

    threads, results, errors = run_threads([
        lambda: reg.run("tool", install) for _ in range(4)
    ])
    assert entered.wait(10)
    assert wait_for(lambda: reg._flights["tool"].attached >= 3)
    release.set()
    join_all(threads)
    # one run, one failure, handed to every waiter as-is — never
    # four sequential fresh installs
    assert calls == [1]
    assert all(err is boom for err in errors)


def test_different_keys_run_in_parallel():
    reg = InflightRegistry()
    barrier = threading.Barrier(2)
    calls = []

    def install(name):
        def run_one():
            calls.append(name)
            # passes only if BOTH installs are in flight at once;
            # a global lock would break the barrier and fail here
            barrier.wait(10)
            return name
        return run_one

    threads, results, errors = run_threads([
        lambda: reg.run("tool-a", install("tool-a")),
        lambda: reg.run("tool-b", install("tool-b")),
    ])
    join_all(threads)
    assert errors == [None, None]
    assert sorted(results) == ["tool-a", "tool-b"]
    assert sorted(calls) == ["tool-a", "tool-b"]


def test_attacher_arriving_as_the_run_completes_gets_its_outcome():
    reg = InflightRegistry()
    finishing = threading.Event()
    go = threading.Event()
    calls = []

    def install():
        calls.append(1)
        finishing.set()          # the value is already computed;
        assert go.wait(10)       # the run is in its final stretch
        return "landed"

    threads, results, errors = run_threads([
        lambda: reg.run("tool", install),
    ])
    assert finishing.wait(10)
    # arrive at the boundary: the flight is still registered, so
    # this attaches rather than starting a second install
    threads2, results2, errors2 = run_threads([
        lambda: reg.run("tool", install),
    ])
    assert wait_for(lambda: reg._flights["tool"].attached >= 1)
    assert calls == [1]
    go.set()
    join_all(threads)
    join_all(threads2)
    assert errors == [None] and errors2 == [None]
    assert results == ["landed"] and results2 == ["landed"]
    assert calls == [1]
    # and a completed flight is forgotten: a later call runs the
    # callable again (the registry caches no results — organs
    # take their already-installed path before ever reaching it)
    assert reg.run("tool", lambda: "fresh") == "fresh"


def test_orphaned_flight_falls_back_to_a_fresh_run():
    reg = InflightRegistry()
    staged = []

    def abandon():
        staged.append(reg.begin("tool"))  # claim, never publish

    ghost = threading.Thread(target=abandon)
    ghost.start()
    ghost.join(10)
    assert staged and staged[0] is not None
    assert not ghost.is_alive()

    orphans = []
    calls = []

    def install():
        calls.append(1)
        return "self-installed"

    out = reg.run("tool", install, on_orphan=orphans.append)
    assert out == "self-installed"
    assert calls == [1]               # the fallback ran it, once
    assert orphans == ["tool"]       # ...and the takeover is on
    assert any(e["event"] == "orphaned" and e["key"] == "tool"
               for e in reg.events)   # the record


def test_reentrant_run_for_own_key_fails_fast():
    reg = InflightRegistry()

    def install():
        return reg.run("tool", lambda: "nested")

    with pytest.raises(RuntimeError):
        reg.run("tool", install)


# ── the supervisor's Node install ──────────────────────────────

def _node_supervisor(tmp_path, name, provisioner):
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    bridge = root / "whatsapp-bridge.mjs"
    bridge.write_text("// bridge\n", encoding="utf-8")
    return BridgeSupervisor(
        PairingRegistry(root / "paired.json"),
        bridge_path=bridge, sessions_root=root / "sessions",
        engine="node", node_provisioner=provisioner)


def test_concurrent_supervisors_share_one_node_install(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pairing_mod, "resolve_node",
                        lambda explicit="", **kw: None)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def provisioner():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return "/staged/node"

    sup_a = _node_supervisor(tmp_path, "a", provisioner)
    sup_b = _node_supervisor(tmp_path, "b", provisioner)
    threads, results, errors = run_threads([
        sup_a.resolve_node_path, sup_b.resolve_node_path])
    assert entered.wait(10)
    assert wait_for(
        lambda: attached_count("provision:node") >= 1)
    assert calls == [1]
    release.set()
    join_all(threads)
    assert errors == [None, None]
    # both supervisors walk away with the install's outcome
    assert results == ["/staged/node", "/staged/node"]
    assert sup_a._node_cache == "/staged/node"
    assert sup_b._node_cache == "/staged/node"
    assert calls == [1]


def test_concurrent_node_failure_is_one_attempt_shared(
        tmp_path, monkeypatch):
    monkeypatch.setattr(pairing_mod, "resolve_node",
                        lambda explicit="", **kw: None)
    entered = threading.Event()
    release = threading.Event()
    calls = []
    attempts = ["winget install of Node.js LTS failed (exit 1): "
                "boom"]

    def provisioner():
        calls.append(1)
        entered.set()
        assert release.wait(10)
        raise NodeProvisionFailed(attempts)

    sup_a = _node_supervisor(tmp_path, "a", provisioner)
    sup_b = _node_supervisor(tmp_path, "b", provisioner)
    threads, results, errors = run_threads([
        sup_a.resolve_node_path, sup_b.resolve_node_path])
    assert entered.wait(10)
    assert wait_for(
        lambda: attached_count("provision:node") >= 1)
    release.set()
    join_all(threads)
    assert errors == [None, None]
    assert results == [None, None]
    assert calls == [1]  # one attempt, shared — not one per waiter
    assert sup_a._provision_attempts == attempts
    assert sup_b._provision_attempts == attempts
    # and it stays one attempt: later resolutions re-read the
    # recorded outcome instead of reinstalling (shipped contract)
    assert sup_a.resolve_node_path() is None
    assert sup_b.resolve_node_path() is None
    assert calls == [1]


# ── the supervisor's bridge-deps install ───────────────────────

def _deps_supervisor(tmp_path, runner):
    bdir = tmp_path / "bridge"
    bdir.mkdir(exist_ok=True)
    script = bdir / "whatsapp-bridge.mjs"
    script.write_text("// bridge\n", encoding="utf-8")
    (bdir / "package.json").write_text(
        '{"name": "devon-whatsapp-bridge"}', encoding="utf-8")
    (bdir / "package-lock.json").write_text("{}", encoding="utf-8")
    return BridgeSupervisor(
        PairingRegistry(tmp_path / "paired.json"),
        bridge_path=script, sessions_root=tmp_path / "sessions",
        engine="node", node_resolver=lambda: "/staged/node",
        npm_resolver=lambda node: "/staged/npm",
        deps_runner=runner)


def test_concurrent_deps_install_runs_npm_once(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        entered.set()
        assert release.wait(10)
        (Path(cwd) / "node_modules").mkdir(exist_ok=True)
        return 0, "added 87 packages in 4s"

    sup = _deps_supervisor(tmp_path, runner)
    key = f"provision:bridge-deps:{tmp_path / 'bridge'}"
    threads, results, errors = run_threads([
        lambda: sup._ensure_bridge_deps("/staged/node"),
        lambda: sup._ensure_bridge_deps("/staged/node"),
    ])
    assert entered.wait(10)
    assert wait_for(lambda: attached_count(key) >= 1)
    assert len(calls) == 1
    release.set()
    join_all(threads)
    assert errors == [None, None]
    assert calls == [["/staged/npm", "ci"]]


def test_concurrent_deps_failure_is_shared_not_a_storm(tmp_path):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def runner(argv, cwd):
        calls.append(list(argv))
        entered.set()
        assert release.wait(10)
        return 1, "npm exploded"  # node_modules never lands

    sup = _deps_supervisor(tmp_path, runner)
    key = f"provision:bridge-deps:{tmp_path / 'bridge'}"
    threads, results, errors = run_threads([
        lambda: sup._ensure_bridge_deps("/staged/node"),
        lambda: sup._ensure_bridge_deps("/staged/node"),
    ])
    assert entered.wait(10)
    assert wait_for(lambda: attached_count(key) >= 1)
    release.set()
    join_all(threads)
    # the old lock-and-recheck shape retried the install once per
    # waiter; the flight hands every waiter the SAME failure
    assert len(calls) == 1
    assert all(isinstance(e, PairingUnavailable) for e in errors)
    assert errors[0] is errors[1]
    assert "npm exploded" in str(errors[0])


# ── the pairing page's PHP install ─────────────────────────────

def test_concurrent_page_launches_share_one_php_install(
        tmp_path, monkeypatch):
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_provision_php(**kw):
        calls.append(1)
        entered.set()
        assert release.wait(10)
        return "/staged/php"

    monkeypatch.setattr(pairing_web, "provision_php",
                        fake_provision_php)

    class FakeServerProc:
        pid = 4242

        def poll(self):
            return None

    ctx = SimpleNamespace(settings=SimpleNamespace(
        workspace_dir=str(tmp_path),
        chat=SimpleNamespace(telegram_api_id="",
                             telegram_api_hash=""),
        api=SimpleNamespace(port=9999, token="owner-api-token")))
    key = f"provision:php:{tmp_path}"
    threads, results, errors = run_threads([
        lambda: pairing_web.launch_pairing_page(
            ctx, spawn=lambda argv, **kw: FakeServerProc(),
            env={"PATH": ""}, platform="posix",
            home=tmp_path / "nohome")
        for _ in range(2)
    ])
    assert entered.wait(10)
    assert wait_for(lambda: attached_count(key) >= 1)
    assert calls == [1]
    release.set()
    join_all(threads)
    assert errors == [None, None]
    assert all(r["php"] == "/staged/php" for r in results)
    assert calls == [1]


# ── the computer organ ─────────────────────────────────────────

def test_concurrent_bridge_builds_run_once(tmp_path, monkeypatch):
    monkeypatch.delenv("NM_COMPUTER_BRIDGE", raising=False)
    target = tmp_path / "bin" / "desktop-bridge-linux"
    monkeypatch.setattr(comp_provision, "bridge_binary_path",
                        lambda: target)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    def fake_build(into=None):
        calls.append(1)
        entered.set()
        assert release.wait(10)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"bridge")
        return target

    monkeypatch.setattr(comp_provision, "build_bridge", fake_build)
    threads, results, errors = run_threads([
        comp_provision.ensure_bridge, comp_provision.ensure_bridge])
    assert entered.wait(10)
    assert wait_for(
        lambda: attached_count("provision:desktop-bridge") >= 1)
    assert calls == [1]
    release.set()
    join_all(threads)
    assert errors == [None, None]
    assert all(r.how == "built" and r.path == target
               for r in results)
    assert calls == [1]


def test_concurrent_xvfb_install_runs_apt_once(
        tmp_path, monkeypatch):
    state = {"installed": False}
    monkeypatch.setattr(
        comp_provision, "find_xvfb",
        lambda: "/usr/bin/Xvfb" if state["installed"] else None)
    real_which = shutil.which
    monkeypatch.setattr(
        shutil, "which",
        lambda name, **kw: "/usr/bin/apt-get"
        if name == "apt-get" else real_which(name, **kw))
    monkeypatch.setattr(os, "geteuid", lambda: 0)
    entered = threading.Event()
    release = threading.Event()
    calls = []

    class Done:
        returncode = 0

    def fake_run(argv, **kw):
        calls.append(list(argv))
        entered.set()
        assert release.wait(10)
        state["installed"] = True
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    threads, results, errors = run_threads([
        comp_provision.ensure_xvfb_binary,
        comp_provision.ensure_xvfb_binary])
    assert entered.wait(10)
    assert wait_for(lambda: attached_count("provision:xvfb") >= 1)
    assert len(calls) == 1
    release.set()
    join_all(threads)
    assert errors == [None, None]
    assert results == ["/usr/bin/Xvfb", "/usr/bin/Xvfb"]
    assert len(calls) == 1
    assert calls[0][1:] == ["install", "-y",
                            "--no-install-recommends", "xvfb"]
