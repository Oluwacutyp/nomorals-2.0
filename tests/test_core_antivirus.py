"""Antivirus / malware scanning (nomorals/core/antivirus.py): one
detection point for the engines on this box, wired into the profile
runtime (register §11(1), owner scope: everything Devon can reach on
every machine, profile runtime sorts per-machine reach).

Every probe test patches the probe seams (``_run`` / ``shutil.which`` /
``glob``) or pins via env — nothing here needs a real engine; this box
has none. Tests that patch the probes restore the process-wide
antivirus cache to the real box state before returning, so a faked
engine never leaks into a later test file (the cache is process-global).

The EICAR test file is built from parts in every test (never the
contiguous string in source) so the test file itself is not flagged
by a scanner run over the repo.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import types

import pytest

from nomorals.core import antivirus as av_mod
from nomorals.core.antivirus import (
    AntivirusInfo,
    detect_antivirus,
    scan_file,
    scan_path,
)

# The EICAR standard test string, assembled from parts so this file
# never contains the contiguous signature.
_EICAR_A = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_B = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def eicar_bytes() -> bytes:
    return (_EICAR_A + _EICAR_B).encode("ascii")


def _restore_real_av():
    """Re-probe with the real box (no patches active) so the cache
    holds the true state, never a faked engine."""
    detect_antivirus(refresh=True)


@pytest.fixture(autouse=True)
def _clean_av_env(monkeypatch):
    monkeypatch.delenv("NM_AV", raising=False)
    monkeypatch.delenv("NM_AV_QUARANTINE", raising=False)
    yield
    monkeypatch.undo()
    _restore_real_av()


def _run_result(returncode=0, stdout="", stderr=""):
    return lambda args, timeout=None, cwd=None: types.SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr)


# ── detection ────────────────────────────────────────────────────

def test_no_engine_when_pinned_none():
    got = detect_antivirus(pin="none", refresh=True)
    assert got.engines == ()
    assert got.primary == ""
    assert got.has_engine is False


def test_builtin_is_the_floor_rung_on_this_box():
    got = detect_antivirus(refresh=True)
    assert got.engines == ("builtin",)
    assert got.primary == "builtin"
    assert got.has_engine is True
    assert "EICAR" in got.detail


def test_defender_detected_when_mpcmdrun_found(monkeypatch):
    monkeypatch.setattr(av_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        av_mod, "_find_mpcmdrun",
        lambda: "C:\\ProgramData\\Microsoft\\Windows Defender\\"
                "Platform\\4.18.0\\MpCmdRun.exe")
    got = detect_antivirus(refresh=True)
    assert got.engines[0] == "defender"
    assert got.primary == "defender"
    assert got.has_engine is True
    assert "MpCmdRun" in got.defender_path


def test_clamav_detected_when_on_path(monkeypatch):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/clamscan"
                        if name == "clamscan" else None)
    got = detect_antivirus(refresh=True)
    assert "clamav" in got.engines
    assert got.clamav_path == "/usr/bin/clamscan"


def test_env_pin_respected(monkeypatch):
    monkeypatch.setenv("NM_AV", "builtin")
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/clamscan"
                        if name == "clamscan" else None)
    got = detect_antivirus(refresh=True)
    assert got.engines == ("builtin",)
    assert got.primary == "builtin"


def test_pin_to_missing_engine_fails_closed(monkeypatch):
    got = detect_antivirus(pin="defender", refresh=True)
    assert got.engines == ()
    assert got.has_engine is False
    assert "defender" in got.detail.lower()


def test_detection_never_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("probe exploded")
    monkeypatch.setattr(av_mod, "_find_mpcmdrun", _boom)
    monkeypatch.setattr(shutil, "which", _boom)
    got = detect_antivirus(refresh=True)
    assert isinstance(got, AntivirusInfo)
    assert got.has_engine is True  # builtin floor survives a broken probe
    assert got.primary == "builtin"


# ── builtin engine: scan_file ────────────────────────────────────

def test_clean_file_is_clean(tmp_path):
    target = tmp_path / "hello.txt"
    target.write_text("hello, world\n", encoding="utf-8")
    got = scan_file(target)
    assert got.verdict == "clean"
    assert got.engine == "builtin"
    assert got.sha256 == hashlib.sha256(b"hello, world\n").hexdigest()
    assert got.threat == ""
    assert "EICAR" in got.detail


def test_eicar_file_is_infected(tmp_path):
    target = tmp_path / "eicar.com"
    target.write_bytes(eicar_bytes())
    got = scan_file(target)
    assert got.verdict == "infected"
    assert got.engine == "builtin"
    assert got.threat == "EICAR-Test-File"
    assert got.quarantined_to == ""  # no quarantine dir given: report only


def test_missing_file_is_unknown_never_clean(tmp_path):
    got = scan_file(tmp_path / "nope.bin")
    assert got.verdict == "unknown"
    assert "not found" in got.detail.lower()


def test_scan_path_directory_aggregates(tmp_path):
    (tmp_path / "clean.txt").write_text("clean\n", encoding="utf-8")
    (tmp_path / "eicar.com").write_bytes(eicar_bytes())
    got = scan_path(tmp_path)
    assert got.verdict == "infected"
    assert len(got.results) == 2
    by_name = {r["path"]: r["verdict"] for r in got.results}
    assert by_name[str(tmp_path / "clean.txt")] == "clean"
    assert by_name[str(tmp_path / "eicar.com")] == "infected"


def test_scan_path_missing_is_unknown(tmp_path):
    got = scan_path(tmp_path / "ghost")
    assert got.verdict == "unknown"


# ── quarantine ───────────────────────────────────────────────────

def test_quarantine_moves_infected_file(tmp_path):
    target = tmp_path / "eicar.com"
    target.write_bytes(eicar_bytes())
    qdir = tmp_path / "quarantine"
    got = scan_file(target, quarantine_dir=qdir)
    assert got.verdict == "infected"
    assert got.quarantined_to != ""
    assert not target.exists()  # moved out of reach
    moved = qdir / got.quarantined_to.split("/")[-1]
    assert moved.exists()
    assert moved.read_bytes() == eicar_bytes()
    sidecar = moved.with_suffix(moved.suffix + ".json")
    assert sidecar.exists()
    record = json.loads(sidecar.read_text(encoding="utf-8"))
    assert record["verdict"] == "infected"
    assert record["threat"] == "EICAR-Test-File"
    journal = qdir / "scan-journal.jsonl"
    assert journal.exists()
    lines = [json.loads(ln) for ln in
             journal.read_text(encoding="utf-8").splitlines() if ln]
    assert lines[-1]["verdict"] == "infected"
    assert lines[-1]["threat"] == "EICAR-Test-File"


def test_clean_file_is_never_quarantined(tmp_path):
    target = tmp_path / "clean.txt"
    target.write_text("clean\n", encoding="utf-8")
    qdir = tmp_path / "quarantine"
    got = scan_file(target, quarantine_dir=qdir)
    assert got.verdict == "clean"
    assert got.quarantined_to == ""
    assert target.exists()
    # The scan journals (audit trail), so the dir may exist — but no
    # file was moved into it: only the journal lives there.
    moved = [p for p in qdir.iterdir()
             if p.name != "scan-journal.jsonl"] if qdir.exists() else []
    assert moved == []


def test_quarantine_failure_is_honest(tmp_path, monkeypatch):
    target = tmp_path / "eicar.com"
    target.write_bytes(eicar_bytes())
    monkeypatch.setattr(shutil, "move",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError("file is locked by another process")))
    got = scan_file(target, quarantine_dir=tmp_path / "quarantine")
    assert got.verdict == "infected"
    assert got.quarantined_to == ""
    assert "quarantine failed" in got.detail.lower()
    assert target.exists()  # never deleted without a successful move


def test_scan_skips_the_quarantine_dir_itself(tmp_path):
    qdir = tmp_path / "quarantine"
    qdir.mkdir()
    (qdir / "old-eicar.com").write_bytes(eicar_bytes())
    (tmp_path / "clean.txt").write_text("clean\n", encoding="utf-8")
    got = scan_path(tmp_path, quarantine_dir=qdir)
    assert got.verdict == "clean"
    assert len(got.results) == 1


# ── Defender runner (fake _run seam) ─────────────────────────────

def _defender_env(monkeypatch, run):
    monkeypatch.setattr(av_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        av_mod, "_find_mpcmdrun", lambda: "C:\\MpCmdRun.exe")
    monkeypatch.setattr(av_mod, "_run", run)
    # Re-probe with the fakes active so the detection cache (which
    # scan_file reads) holds the faked Defender path.
    detect_antivirus(refresh=True)


def test_defender_clean_rc0(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    _defender_env(monkeypatch, _run_result(
        0, "Scan finished. No threats found.\n"))
    got = scan_file(target, engine="defender")
    assert got.verdict == "clean"
    assert got.engine == "defender"


def test_defender_infected_rc2_with_threat(tmp_path, monkeypatch):
    target = tmp_path / "a.exe"
    target.write_text("x", encoding="utf-8")
    _defender_env(monkeypatch, _run_result(
        2, "Threat detected: Trojan:Win32/Wacatac!\n"))
    got = scan_file(target, engine="defender")
    assert got.verdict == "infected"
    assert "Wacatac" in got.threat


def test_defender_rc2_without_threat_is_unknown(tmp_path, monkeypatch):
    target = tmp_path / "a.exe"
    target.write_text("x", encoding="utf-8")
    _defender_env(monkeypatch, _run_result(2, "Scanning error.\n"))
    got = scan_file(target, engine="defender")
    # rc 2 conflates "malware, not remediated" with "scanning error":
    # with no threat named there is no determination.
    assert got.verdict == "unknown"


def test_defender_rc0_with_threat_is_infected_remediated(
        tmp_path, monkeypatch):
    target = tmp_path / "a.exe"
    target.write_text("x", encoding="utf-8")
    _defender_env(monkeypatch, _run_result(
        0, "Threat detected: Trojan:Win32/Foo — remediated.\n"))
    got = scan_file(target, engine="defender")
    assert got.verdict == "infected"
    assert "remediat" in got.detail.lower()


# ── ClamAV runner (fake _run seam) ───────────────────────────────

def _clamav_env(monkeypatch, run):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/clamscan"
                        if name == "clamscan" else None)
    monkeypatch.setattr(av_mod, "_run", run)
    # Re-probe with the fakes active so the detection cache (which
    # scan_file reads) holds the faked clamscan path.
    detect_antivirus(refresh=True)


def test_clamscan_clean_rc0(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    _clamav_env(monkeypatch, _run_result(
        0, f"{target}: OK\n"))
    got = scan_file(target, engine="clamav")
    assert got.verdict == "clean"
    assert got.engine == "clamav"


def test_clamscan_infected_rc1(tmp_path, monkeypatch):
    target = tmp_path / "a.exe"
    target.write_text("x", encoding="utf-8")
    _clamav_env(monkeypatch, _run_result(
        1, f"{target}: Eicar-Test-Signature FOUND\n"))
    got = scan_file(target, engine="clamav")
    assert got.verdict == "infected"
    assert got.threat == "Eicar-Test-Signature"


def test_clamscan_error_rc2_is_unknown(tmp_path, monkeypatch):
    target = tmp_path / "a.exe"
    target.write_text("x", encoding="utf-8")
    _clamav_env(monkeypatch, _run_result(
        2, f"{target}: Can't open file. ERROR\n"))
    got = scan_file(target, engine="clamav")
    # rc 2 is "I don't know" — never clean, never infected.
    assert got.verdict == "unknown"


def test_unknown_engine_is_unknown(tmp_path):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    got = scan_file(target, engine="no-such-engine")
    assert got.verdict == "unknown"


# ── profile wiring ───────────────────────────────────────────────

def test_detect_profile_carries_antivirus():
    from nomorals.core.profile import detect_profile
    prof = detect_profile()
    assert isinstance(prof.antivirus, AntivirusInfo)
    assert prof.antivirus.has_engine is True
    assert "antivirus" in prof.to_dict()
    assert prof.to_dict()["antivirus"]["primary"] == "builtin"


def test_profiles_antivirus_info():
    from nomorals.core.profiles import antivirus_info
    info = antivirus_info()
    assert info["primary"] == "builtin"
    assert info["has_engine"] is True


def test_format_profile_mentions_antivirus():
    from nomorals.core.profiles import format_profile
    out = format_profile()
    assert "antivirus" in out.lower()


def test_profile_visible_carries_antivirus():
    from nomorals.core.profile import detect_profile
    from nomorals.core.profile_visible import profile_summary
    summary = profile_summary(detect_profile())
    assert "antivirus" in summary
    assert "av builtin" in summary["line"]


def test_re_detect_notices_antivirus_change():
    from nomorals.agents.runtime_profile import RuntimeProfileManager
    from nomorals.core.profile import detect_profile
    mgr = RuntimeProfileManager()
    before = detect_profile()
    changed = type(before)(
        **{**{f: getattr(before, f) for f in before.__dataclass_fields__},
           "antivirus": AntivirusInfo(engines=(), primary="",
                                      has_engine=False,
                                      detail="pinned none")})
    out = mgr.re_detect(profile=changed)
    assert out["changed"] is True
    assert "antivirus" in out["changes"]


# ── tool surface ─────────────────────────────────────────────────

def test_av_scan_registered():
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry().register_builtins()
    assert "av_scan" in reg._tools


def test_av_scan_clean_file(tmp_path):
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry().register_builtins()
    target = tmp_path / "clean.txt"
    target.write_text("clean\n", encoding="utf-8")
    out = reg._tools["av_scan"].fn(str(target), quarantine=False)
    assert out["ok"] is True
    assert out["verdict"] == "clean"
    assert out["engine"] == "builtin"


def test_av_scan_eicar_quarantines(tmp_path):
    from types import SimpleNamespace

    from nomorals.core.config import Settings
    from nomorals.tools.registry import ToolRegistry
    settings = Settings(home=str(tmp_path / "home"))
    ctx = SimpleNamespace(settings=settings,
                          workspace=str(tmp_path / "ws"))
    reg = ToolRegistry(context=ctx).register_builtins()
    target = tmp_path / "eicar.com"
    target.write_bytes(eicar_bytes())
    out = reg._tools["av_scan"].fn(str(target))
    assert out["ok"] is True
    assert out["verdict"] == "infected"
    assert out["threat"] == "EICAR-Test-File"
    assert out["quarantined_to"] != ""
    assert not target.exists()
    assert (tmp_path / "home" / "quarantine").exists()


def test_av_scan_missing_file_is_unknown(tmp_path):
    from nomorals.tools.registry import ToolRegistry
    reg = ToolRegistry().register_builtins()
    out = reg._tools["av_scan"].fn(str(tmp_path / "ghost.bin"),
                                   quarantine=False)
    assert out["ok"] is True
    assert out["verdict"] == "unknown"
