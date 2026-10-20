"""Pass #40 — core infrastructure defects.

1. Idempotency records were documented as bound to the tool version,
   but replay never compared it: after a tool was re-registered under
   a new version, the same key returned the OLD implementation's
   recorded result without executing.
2. The oversight classifier matched paths case-sensitively, so on a
   case-insensitive filesystem (the Windows deployment) a write to
   ``NOMORALS/CORE/POLICY.PY`` — the same file — escaped the gate,
   as did ``ID_RSA`` / ``.PEM`` secret names and exec text.
3. ``ToolRegistry.call_many`` ran parallel calls on pool threads
   without propagating the caller's contextvars, so ``tool.call``
   events for those calls carried empty correlation ids and could
   not be joined to their request/run.
4. ``run_isolated`` SIGKILLed only the forked child on timeout; a
   grandchild the callable spawned kept running and producing side
   effects after the caller was told the outcome was known-dead.
"""

from __future__ import annotations

import os
import time

import pytest

from nomorals.core.correlation import correlation_context
from nomorals.core.events import global_bus
from nomorals.core.oversight import is_oversight_target
from nomorals.core.process_isolation import fork_supported, run_isolated
from nomorals.tools.registry import ToolRegistry


# ── 1. idempotency: tool version binding ──────────────────────────────

class TestIdempotencyToolVersion:
    def _registry(self, calls: list[str], version: str) -> ToolRegistry:
        reg = ToolRegistry(None)

        def effect(x: int = 0) -> str:
            calls.append(version)
            return f"{version}-{x}"

        reg.register("effect", effect, capability="fs.write",
                     version=version)
        return reg

    def test_same_version_replays(self) -> None:
        calls: list[str] = []
        reg = self._registry(calls, "1")
        first = reg.call("effect", actor="alice",
                         idempotency_key="k1", x=5)
        second = reg.call("effect", actor="alice",
                          idempotency_key="k1", x=5)
        assert first.ok and second.ok
        assert second.value == "1-5"
        assert calls == ["1"]  # executed once, replayed once

    def test_version_bump_is_a_conflict_not_a_stale_replay(self) -> None:
        calls: list[str] = []
        reg = self._registry(calls, "1")
        first = reg.call("effect", actor="alice",
                         idempotency_key="k1", x=5)
        assert first.ok and first.value == "1-5"
        # The tool implementation is replaced under a new version.
        reg2_calls = calls
        def effect_v2(x: int = 0) -> str:
            reg2_calls.append("2")
            return f"2-{x}"
        reg.register("effect", effect_v2, capability="fs.write",
                     version="2")
        second = reg.call("effect", actor="alice",
                          idempotency_key="k1", x=5)
        assert not second.ok, (
            "stale v1 result must not replay after the tool changed")
        assert "version" in str(second.error).lower()
        assert calls == ["1"]  # the v2 body never executed under the old key


# ── 2. oversight classifier: case-insensitive targets ─────────────────

class TestOversightCaseInsensitivity:
    @pytest.mark.parametrize("path", [
        "NOMORALS/CORE/POLICY.PY",
        "Nomorals\\Core\\Policy.py",
        "nomorals/core/POLICY.py",
        "home/user/.SSH/config",
        "secrets/ID_RSA",
        "certs/server.PEM",
        "nomorals/accounts/VAULT.PY",
    ])
    def test_case_variants_are_oversight(self, path: str) -> None:
        assert is_oversight_target("fs_write", "fs.write", {"path": path})

    def test_exec_text_case_variant_is_oversight(self) -> None:
        assert is_oversight_target(
            "shell_run", "exec.shell",
            {"command": "cp x.py NOMORALS/CORE/POLICY.PY"})

    @pytest.mark.parametrize("path", [
        "src/app/main.py",
        "SRC/APP/Main.PY",
        "core/policies.py",
        "docs/notes.txt",
    ])
    def test_ordinary_paths_stay_unflagged(self, path: str) -> None:
        assert not is_oversight_target(
            "fs_write", "fs.write", {"path": path})


# ── 3. call_many: correlation propagation ─────────────────────────────

class TestCallManyCorrelation:
    def test_parallel_calls_carry_correlation(self) -> None:
        events: list[dict] = []
        handler = lambda e: events.append(e.data)  # noqa: E731
        sub_id = global_bus.subscribe("tool.call", handler, sync=True)
        try:
            reg = ToolRegistry(None)
            reg.register("echo", lambda x=0: f"v{x}",
                         capability="mem.read")
            with correlation_context(request_id="req-P40",
                                     run_id="run-P40"):
                outcomes = reg.call_many(
                    [("echo", {"x": 1}), ("echo", {"x": 2})],
                    max_workers=2)
            assert all(o.ok for o in outcomes)
            finishes = [e for e in events if e.get("phase") == "finish"]
            assert len(finishes) == 2
            for e in finishes:
                assert e.get("request_id") == "req-P40", e
                assert e.get("run_id") == "run-P40", e
        finally:
            global_bus.unsubscribe(sub_id)


# ── 4. process isolation: the kill covers the process tree ────────────

@pytest.mark.skipif(not fork_supported(), reason="fork unavailable")
class TestRunIsolatedTreeKill:
    def test_timeout_kill_reaches_grandchildren(self, tmp_path) -> None:
        marker = tmp_path / "grandchild.txt"

        def spawner() -> str:
            import subprocess
            import sys
            subprocess.Popen(
                [sys.executable, "-c",
                 "import time; time.sleep(1.2); "
                 f"open({str(marker)!r}, 'w').write('alive')"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(30)
            return "never"

        result = run_isolated(spawner, timeout_s=0.4)
        assert result.status == "timeout-killed"
        time.sleep(2.0)
        assert not marker.exists(), (
            "grandchild survived the timeout kill and wrote its marker")

    def test_normal_completion_unaffected(self, tmp_path) -> None:
        def quick() -> dict:
            return {"ok": True}

        result = run_isolated(quick, timeout_s=10.0)
        assert result.status == "ok"
        assert result.value == {"ok": True}

    def test_child_exception_still_reported_as_error(self) -> None:
        def boom() -> None:
            raise ValueError("kaboom")

        result = run_isolated(boom, timeout_s=10.0)
        assert result.status == "error"
        assert "kaboom" in result.error
