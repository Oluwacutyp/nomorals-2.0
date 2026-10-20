"""Pass #47 — security tooling + redteam surface.

Pins five reproduced defects:

1. sandbox_code session names "." / ".." escaped the per-session
   directory (code ran in the sandbox root / shared work dir).
2. tools/shell results never said which sandbox guarantees actually
   held — under the rlimit fallback ``network=False`` neither blocks
   the network nor confines the filesystem, yet results claimed the
   requested posture with no signal otherwise.
3. execbox ``run_project_until_green`` ran project test suites with a
   bare ``subprocess.run`` — full parent environment, full filesystem —
   despite CodeRunner's "extends the sandbox" contract.
4. redteam: a scenario with no detector was counted as PASSED
   (fail-open), and the per-scenario "hard timeout" was not enforced
   (the executor's ``with`` block waited for the wedged worker).
5. dnsleak strict-ASN mode passed vacuously (``ok=True``) when the
   probe service was unreachable and zero resolvers had been verified;
   the dns/net leak report headlines also claimed "NO LEAK" for
   advisory/unmeasured verdicts.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


# ── 1. sandbox_code session containment ──────────────────────────────────

def test_code_session_dotdot_cannot_escape(tmp_path):
    from nomorals.tools.sandbox_code import CodeInterpreter

    ci = CodeInterpreter(root=tmp_path)
    r = ci.run("open('pwned.txt', 'w').write('x')", session="..")
    workdir = Path(r["workdir"])
    # the run must land inside <root>/work/<some session dir> …
    assert workdir.parent == (tmp_path / "work").resolve() or \
        (tmp_path / "work") in workdir.parents
    # … and the file the code wrote must NOT be at the sandbox root
    assert not (tmp_path / "pwned.txt").exists()
    assert (workdir / "pwned.txt").exists()


def test_code_session_dot_is_not_the_shared_workdir(tmp_path):
    from nomorals.tools.sandbox_code import CodeInterpreter

    ci = CodeInterpreter(root=tmp_path)
    r = ci.run("x = 1", session=".")
    assert Path(r["workdir"]) != (tmp_path / "work").resolve()


def test_session_dir_helper_contained(tmp_path, monkeypatch):
    import nomorals.tools.sandbox_code as sc

    monkeypatch.setattr(sc, "SANDBOX_ROOT", tmp_path)
    for name in ("..", ".", "../..", "..\\.."):
        d = sc._session_dir(name)
        assert d.parent == (tmp_path / "work").resolve(), name


def test_named_session_state_still_persists(tmp_path):
    from nomorals.tools.sandbox_code import CodeInterpreter

    ci = CodeInterpreter(root=tmp_path)
    ci.run("token = 41", session="alpha")
    r = ci.run("token + 1", session="alpha")
    assert r["ok"] and r["result"] == "42"


# ── 2. shell sandbox honesty fields ──────────────────────────────────────

def test_rlimit_backend_reports_guarantees_not_held():
    from nomorals.tools.shell import run_sandboxed

    res = run_sandboxed("true", cwd="/tmp", network=False,
                        backend="rlimit", timeout=10)
    assert res["backend"] == "rlimit"
    # the rlimit fallback cannot cut the network or confine the FS —
    # the result must say so instead of implying the requested posture
    assert res["network_enforced"] is False
    assert res["fs_confined"] is False


def test_namespace_backend_reports_guarantees_held():
    from nomorals.tools.shell import _probe_backend, run_sandboxed

    if _probe_backend("bwrap"):
        res = run_sandboxed("true", cwd="/tmp", network=False,
                            backend="bwrap", timeout=10)
        assert res["network_enforced"] is True
        assert res["fs_confined"] is True
    elif _probe_backend("unshare"):
        res = run_sandboxed("true", cwd="/tmp", network=False,
                            backend="unshare", timeout=10)
        assert res["network_enforced"] is True
        assert res["fs_confined"] is False  # unshare binds nothing
    else:
        pytest.skip("no namespace backend on this host")


def test_code_tool_result_carries_guarantee_fields(tmp_path):
    from nomorals.tools.sandbox_code import CodeInterpreter

    r = CodeInterpreter(root=tmp_path).run("1 + 1", backend="rlimit")
    assert r["network_enforced"] is False
    assert r["fs_confined"] is False


# ── 3. execbox project runs are sandboxed ───────────────────────────────

def _make_project(ws: Path, marker: Path) -> None:
    proj = ws / "proj"
    proj.mkdir(parents=True)
    (proj / "test_leak.py").write_text(
        "import os, unittest\n"
        "class T(unittest.TestCase):\n"
        "    def test_env(self):\n"
        "        print('SECRET-SEEN:', os.environ.get("
        "'NM47_PARENT_SECRET'))\n"
        f"        open({str(marker)!r}, 'w').write('escaped')\n"
        "        self.assertTrue(True)\n")


def test_run_project_until_green_is_sandboxed(tmp_path, monkeypatch):
    from nomorals.execbox import CodeRunner

    ws = tmp_path / "ws"
    marker = tmp_path / "outside_marker.txt"
    _make_project(ws, marker)
    monkeypatch.setenv("NM47_PARENT_SECRET", "topsecret-value")
    ctx = SimpleNamespace(
        router=None, settings=SimpleNamespace(workspace_dir=str(ws)))
    out = CodeRunner(ctx).run_project_until_green(
        "proj", max_rounds=1, timeout=60)
    assert out["green"] is True
    final = out["final_run"] or {}
    # the suite ran under the sandbox machinery: the parent env secret
    # is scrubbed on EVERY backend (pre-fix the bare subprocess.run
    # inherited it and the test printed it)
    assert "topsecret-value" not in (final.get("stdout") or "")
    assert final.get("backend") in ("rlimit", "unshare", "bwrap")
    # filesystem confinement is backend-dependent: where the result
    # claims it, the outside write must have failed; where it doesn't
    # (rlimit floor), the field must honestly say so
    if final.get("fs_confined"):
        assert not marker.exists()
    else:
        assert final.get("fs_confined") is False


# ── 4. redteam fail-closed verdicts + real timeout ──────────────────────

def test_redteam_scenario_without_detector_is_inconclusive():
    from nomorals.agents.redteam import (
        AttackScenario, RedTeam, RedTeamReport)

    sc = AttackScenario(
        id="XNODETECT", name="no detector", category="injection",
        description="", severity="low", payload="hi", target="",
        detect=None, make_harness=lambda: {})
    finding = RedTeam().run_scenario(sc)
    assert finding.inconclusive is True
    rep = RedTeamReport(findings=[finding])
    assert rep.passed == []
    assert len(rep.inconclusive) == 1


def test_redteam_timeout_is_enforced():
    from nomorals.agents.redteam import (
        AttackScenario, RedTeam, _FakeRegistry)

    def slow_fn(**kw):
        time.sleep(6)
        return "slept"

    def harness():
        reg = _FakeRegistry()
        reg.add("slowtool", "slow", slow_fn)
        return {
            "tools": reg,
            "script": [
                {"action": "tool", "thought": "x", "tool": "slowtool",
                 "args": {}},
                {"action": "respond", "thought": "d", "response": "done"},
            ],
            "user_message": "go",
        }

    sc = AttackScenario(
        id="XSLOW", name="slow tool", category="injection",
        description="", severity="low", payload="go", target="",
        detect=lambda ctx: (False, "held"), make_harness=harness)
    started = time.time()
    finding = RedTeam(per_scenario_timeout_s=1.0).run_scenario(sc)
    elapsed = time.time() - started
    assert finding.inconclusive is True
    assert "exceeded" in finding.evidence
    # the 1s budget must bound the call — pre-fix this took the tool's
    # full 6s because the executor waited for the worker to finish
    assert elapsed < 4.0


# ── 5. leak-check honesty ────────────────────────────────────────────────

def _mock_dns_fallback(monkeypatch):
    import nomorals.security.dnsleak as D

    monkeypatch.setattr(D, "_bash_ws_id", lambda: "")
    monkeypatch.setattr(D, "configured_resolvers", lambda: ["8.8.8.8"])
    monkeypatch.setattr(D, "_public_ip", lambda: "")
    monkeypatch.setattr(D, "_public_ipv6", lambda: "")
    monkeypatch.setattr(D, "detect_smhnr_risk",
                        lambda: {"applicable": False, "risk": False,
                                 "detail": ""})
    return D


def test_dnsleak_strict_mode_never_passes_unverified(monkeypatch):
    D = _mock_dns_fallback(monkeypatch)
    rep = D.check_dns_leak(expected_asn="999")
    assert rep.method == "local-heuristic"
    assert rep.resolver_info == []  # nothing was ASN-verified
    assert rep.ok is False
    assert "strict" in rep.note.lower()


def test_dnsleak_fallback_headline_is_advisory(monkeypatch):
    D = _mock_dns_fallback(monkeypatch)
    rep = D.check_dns_leak()
    txt = D.format_report(rep)
    assert "✅ NO LEAK" not in txt


def test_dnsleak_verified_headline_still_no_leak():
    from nomorals.security.dnsleak import DnsLeakReport, format_report

    rep = DnsLeakReport(
        ok=True, method="bash.ws", public_ip="1.2.3.4",
        public_asn="9009", resolvers=["9.9.9.9"],
        resolver_info=[{"ip": "9.9.9.9", "asn": "19281",
                        "country": "US"}],
        leak_detail="all good")
    assert "✅ NO LEAK" in format_report(rep)


def test_netleak_headline_inconclusive_without_stun():
    from nomorals.security.netleak import NetLeakReport, format_report

    rep = NetLeakReport(ok=True, http_public_ip="1.2.3.4", stun_ip="")
    txt = format_report(rep)
    assert "✅ NO LEAK" not in txt
    assert "INCONCLUSIVE" in txt


def test_strip_exif_verifies_the_written_file(tmp_path):
    import struct

    from nomorals.security import exif as E

    tiff = b"II*\x00\x08\x00\x00\x00\x00\x00\x00\x00"
    app1 = b"Exif\x00\x00" + tiff
    seg = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    sos_data = (b"\x00\x11\x08\x00\x01\x00\x01\x01\x01\x11\x00\x3f\x00"
                b"\xfb\xd2\x8a\x28\x03\xff\xd9")
    jpeg = (b"\xff\xd8" + seg + b"\xff\xda"
            + struct.pack(">H", len(sos_data)) + sos_data)
    img = tmp_path / "shot.jpg"
    img.write_bytes(jpeg)
    out = E.strip_exif(img, backup=False)
    assert out["had_exif"] is True
    # the summary must be backed by a re-scan of what was written —
    # pre-fix strip_exif reported removals from the BEFORE scan only
    assert out["verified_clean"] is True
    assert out["residual"] == {}
    assert E.scan_metadata(img.read_bytes())["has_exif"] is False


# ── gating pins (pass #47 Q2) ────────────────────────────────────────────

def test_redteam_command_is_owner_only():
    from nomorals.social.chat.control import (
        GAME_COMMANDS, PUBLIC_COMMANDS, caller_may_run)

    assert "redteam" not in PUBLIC_COMMANDS
    assert "redteam" not in GAME_COMMANDS
    assert caller_may_run("redteam", is_owner=False) is False
    assert caller_may_run("redteam", is_owner=True) is True


class _CapFake:
    """Minimal registry stand-in capturing register() capabilities."""

    def __init__(self):
        self.context = None
        self.caps: dict[str, object] = {}

    def register(self, name, description="", capability=None,
                 parameters=None):
        def deco(fn):
            self.caps[name] = capability
            return fn
        return deco


def test_security_tools_carry_capabilities():
    from nomorals.core.policy import Capability
    from nomorals.tools import error_scan as error_scan_mod
    from nomorals.tools import sandbox_code as sandbox_code_mod
    from nomorals.tools import security as security_mod
    from nomorals.tools import shell as shell_mod

    for mod in (security_mod, error_scan_mod, sandbox_code_mod, shell_mod):
        fake = _CapFake()
        mod.register(fake)
        assert fake.caps, mod.__name__
        assert all(c is not None for c in fake.caps.values()), mod.__name__

    fake = _CapFake()
    security_mod.register(fake)
    assert fake.caps["dns_leak_check"] == Capability.NET_OUT
    assert fake.caps["strip_exif"] == Capability.FS_WRITE
    assert fake.caps["exif_report"] == Capability.FS_READ
    fake = _CapFake()
    sandbox_code_mod.register(fake)
    assert fake.caps["code"] == Capability.EXEC_CODE
    fake = _CapFake()
    shell_mod.register(fake)
    assert fake.caps["shell_run"] == Capability.EXEC_SHELL
    assert fake.caps["python_run"] == Capability.EXEC_CODE
