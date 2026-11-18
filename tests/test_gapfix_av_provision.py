"""Lane 66 — ClamAV self-provision + per-engine antivirus timeouts.

Failing-first: on the pristine base there is no ClamAV provision
path anywhere (no ``clamav`` in the /models backends, no ``nm av
provision`` verb) and one global 300s timeout shared by Defender and
ClamAV alike. Every test here fails there (``AttributeError`` /
``KeyError`` / usage error / wrong timeout).

1. Per-engine timeouts — ``NM_AV_TIMEOUT_DEFENDER_S`` /
   ``NM_AV_TIMEOUT_CLAMAV_S`` (env/config, 300s default preserved),
   resolved per engine inside ``scan_file``/``scan_path`` (an explicit
   ``timeout_s`` still wins), surfaced in ``nm av status``.
2. ClamAV self-provision — ``clamav`` as a /models backend
   (measure box, floor check, install via the platform package
   manager, verify by a real ``clamscan --version`` + an EICAR probe
   through Devon's own scanner, honest refusal elsewhere) and
   ``nm av provision`` as the console verb for it. Running it on EC2
   stays the owner's hand; this lane builds the path, never claims
   the EC2 run.
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tempfile
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

import nomorals.core.antivirus as av_mod
from nomorals.core.antivirus import (
    detect_antivirus,
    scan_file,
    scan_path,
)

# The EICAR standard test string, assembled from parts so this file
# never contains the contiguous signature.
_EICAR_A = "X5O!P%@AP[4\\PZX54(P^)7CC)7}$"
_EICAR_B = "EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
EICAR_STRING = _EICAR_A + _EICAR_B

BIG_BOX = {"ok": True, "total_ram_gb": 16.0, "free_ram_gb": 10.0,
           "free_disk_gb": 120.0, "cpus": 8, "platform": "linux",
           "profile": "workstation", "detail": "big box"}


@pytest.fixture(autouse=True)
def _clean_av_state(monkeypatch):
    for key in ("NM_AV", "NM_AV_QUARANTINE",
                "NM_AV_TIMEOUT_DEFENDER_S", "NM_AV_TIMEOUT_CLAMAV_S"):
        monkeypatch.delenv(key, raising=False)
    yield
    monkeypatch.undo()
    # Re-probe with the real box (no patches active) so the detection
    # cache holds the true state, never a faked engine.
    detect_antivirus(refresh=True)


def _engine_timeouts():
    """engine_timeouts() via getattr — AttributeError on the base,
    where the function does not exist yet."""
    return av_mod.engine_timeouts()


def _run_result(returncode=0, stdout="", stderr=""):
    return lambda args, timeout=None, cwd=None: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=stderr)


def _capturing_run(stdout="", returncode=0):
    calls: list[dict] = []

    def run(args, timeout=None, cwd=None):
        calls.append({"args": list(args), "timeout": timeout})
        return SimpleNamespace(
            returncode=returncode, stdout=stdout, stderr="")
    return calls, run


class _WhichFake:
    """shutil.which fake: answers from a dict, mutable mid-test."""

    def __init__(self, mapping):
        self.mapping = dict(mapping)

    def __call__(self, name):
        return self.mapping.get(name)


def _ctx(home: Path):
    settings = SimpleNamespace(
        home=str(home), home_path=home,
        workspace_dir=home / "workspace")

    def _resolve(relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (home / p).resolve()

    settings.resolve = _resolve
    return SimpleNamespace(db=None, settings=settings)


def _ns(**kwargs):
    args = argparse.Namespace(
        av_action="status", av_rest=[], json=False,
        no_quarantine=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    return args


# ── per-engine timeouts ──────────────────────────────────────────

class TestEngineTimeouts:
    def test_defaults_are_300(self):
        assert _engine_timeouts() == {"defender": 300.0, "clamav": 300.0}

    def test_env_override_per_engine(self, monkeypatch):
        monkeypatch.setenv("NM_AV_TIMEOUT_DEFENDER_S", "60")
        monkeypatch.setenv("NM_AV_TIMEOUT_CLAMAV_S", "45")
        assert _engine_timeouts() == {"defender": 60.0, "clamav": 45.0}

    def test_bad_env_falls_back_to_300(self, monkeypatch):
        for raw in ("banana", "0", "-5", "", "  "):
            monkeypatch.setenv("NM_AV_TIMEOUT_DEFENDER_S", raw)
            monkeypatch.setenv("NM_AV_TIMEOUT_CLAMAV_S", raw)
            assert _engine_timeouts() == {"defender": 300.0,
                                          "clamav": 300.0}


def _defender_env(monkeypatch, run):
    monkeypatch.setattr(av_mod, "_is_windows", lambda: True)
    monkeypatch.setattr(
        av_mod, "_find_mpcmdrun", lambda: "C:\\MpCmdRun.exe")
    monkeypatch.setattr(av_mod, "_run", run)
    detect_antivirus(refresh=True)


def _clamav_env(monkeypatch, run):
    monkeypatch.setattr(shutil, "which",
                        lambda name: "/usr/bin/clamscan"
                        if name == "clamscan" else None)
    monkeypatch.setattr(av_mod, "_run", run)
    detect_antivirus(refresh=True)


def test_scan_file_defender_uses_env_timeout(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    calls, run = _capturing_run(
        "Scan finished. No threats found.\n", returncode=0)
    _defender_env(monkeypatch, run)
    monkeypatch.setenv("NM_AV_TIMEOUT_DEFENDER_S", "60")
    got = scan_file(target, engine="defender")
    assert got.verdict == "clean"
    assert got.engine == "defender"
    assert len(calls) == 1
    assert calls[0]["timeout"] == 60


def test_scan_file_clamav_uses_env_timeout(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    calls, run = _capturing_run(f"{target}: OK\n", returncode=0)
    _clamav_env(monkeypatch, run)
    monkeypatch.setenv("NM_AV_TIMEOUT_CLAMAV_S", "45")
    got = scan_file(target, engine="clamav")
    assert got.verdict == "clean"
    assert got.engine == "clamav"
    assert len(calls) == 1
    assert calls[0]["timeout"] == 45


def test_scan_file_explicit_timeout_wins(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    calls, run = _capturing_run(f"{target}: OK\n", returncode=0)
    _clamav_env(monkeypatch, run)
    monkeypatch.setenv("NM_AV_TIMEOUT_CLAMAV_S", "45")
    got = scan_file(target, engine="clamav", timeout_s=120)
    assert got.verdict == "clean"
    assert calls[0]["timeout"] == 120


def test_scan_file_no_env_uses_300(tmp_path, monkeypatch):
    target = tmp_path / "a.txt"
    target.write_text("x", encoding="utf-8")
    calls, run = _capturing_run(f"{target}: OK\n", returncode=0)
    _clamav_env(monkeypatch, run)
    got = scan_file(target, engine="clamav")
    assert got.verdict == "clean"
    assert calls[0]["timeout"] == 300


def test_scan_path_threads_clamav_timeout(tmp_path, monkeypatch):
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    calls, run = _capturing_run("OK\n", returncode=0)
    _clamav_env(monkeypatch, run)
    monkeypatch.setenv("NM_AV_TIMEOUT_CLAMAV_S", "45")
    got = scan_path(tmp_path, engine="clamav")
    assert got.verdict == "clean"
    assert len(calls) == 2
    assert all(c["timeout"] == 45 for c in calls)


# ── ClamAV as a /models backend ──────────────────────────────────

class TestClamavBackend:
    def test_clamav_is_a_models_backend(self):
        import nomorals.tools.models_provision as mp

        assert "clamav" in mp.BACKENDS
        info = mp._BACKEND_INFO["clamav"]
        assert info["family"] == "security"
        assert callable(mp._detector("clamav"))

    def test_already_installed_reports_without_running(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        monkeypatch.setattr(mp, "_detect_clamav", lambda: True)
        calls = []
        monkeypatch.setattr(
            mp, "_run",
            lambda *a, **k: calls.append(a) or _run_result()(*a, **k))
        result = mp.provision_models(["clamav"], box=dict(BIG_BOX))
        rep = result["backends"]["clamav"]
        assert rep["status"] == "already-installed"
        assert rep["ok"]
        assert calls == []

    def test_apt_install_then_eicar_probe(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        which = _WhichFake({"apt-get": "/usr/bin/apt-get"})
        monkeypatch.setattr(shutil, "which", which)
        seen: dict = {"eicar_in_scanned_file": False}

        def fake_mp_run(args, timeout=None):
            argv = list(args)
            if argv[:2] == ["apt-get", "install"]:
                # clamscan appears on PATH once the package lands.
                which.mapping["clamscan"] = "/usr/bin/clamscan"
                return SimpleNamespace(
                    returncode=0, stdout="Setting up clamav\n",
                    stderr="")
            if argv[:1] == ["freshclam"]:
                return SimpleNamespace(
                    returncode=0, stdout="Database updated\n",
                    stderr="")
            if argv[:2] == ["clamscan", "--version"]:
                return SimpleNamespace(
                    returncode=0,
                    stdout="ClamAV 1.4.2/27200/Thu Jan 1 00:00:00 2026\n",
                    stderr="")
            raise AssertionError(f"unexpected provision argv: {argv}")

        monkeypatch.setattr(mp, "_run", fake_mp_run)

        def fake_av_run(args, timeout=None, cwd=None):
            argv = list(args)
            assert argv[:2] == ["clamscan", "--no-summary"]
            scanned = Path(argv[-1])
            seen["eicar_in_scanned_file"] = (
                EICAR_STRING.encode("ascii") in scanned.read_bytes())
            return SimpleNamespace(
                returncode=1,
                stdout=f"{scanned}: Eicar-Test-Signature FOUND\n",
                stderr="")

        monkeypatch.setattr(av_mod, "_run", fake_av_run)
        detect_antivirus(refresh=True)
        result = mp.provision_models(["clamav"], box=dict(BIG_BOX))
        rep = result["backends"]["clamav"]
        assert rep["ok"], rep
        assert rep["status"] == "installed"
        assert seen["eicar_in_scanned_file"]
        steps = " ".join(rep["steps"])
        assert "apt-get" in steps
        assert "freshclam" in steps
        assert "ClamAV 1.4.2" in steps

    def test_no_package_manager_is_unavailable(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        monkeypatch.setattr(shutil, "which", lambda name: None)
        calls = []
        monkeypatch.setattr(
            mp, "_run",
            lambda *a, **k: calls.append(a) or _run_result()(*a, **k))
        result = mp.provision_models(["clamav"], box=dict(BIG_BOX))
        rep = result["backends"]["clamav"]
        assert not rep["ok"]
        assert rep["status"] == "unavailable"
        assert "apt-get" in rep["detail"]
        assert calls == []

    def test_install_failure_is_failed_with_stderr(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        monkeypatch.setattr(shutil, "which",
                            lambda name: "/usr/bin/apt-get"
                            if name == "apt-get" else None)
        monkeypatch.setattr(mp, "_run", _run_result(
            1, "", "E: Unable to locate package clamav\n"))
        result = mp.provision_models(["clamav"], box=dict(BIG_BOX))
        rep = result["backends"]["clamav"]
        assert not rep["ok"]
        assert rep["status"] == "failed"
        assert "Unable to locate package" in rep["detail"]

    def test_version_check_failure_is_failed(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        which = _WhichFake({"apt-get": "/usr/bin/apt-get"})
        monkeypatch.setattr(shutil, "which", which)

        def fake_mp_run(args, timeout=None):
            argv = list(args)
            if argv[:2] == ["apt-get", "install"]:
                which.mapping["clamscan"] = "/usr/bin/clamscan"
                return SimpleNamespace(
                    returncode=0, stdout="Setting up clamav\n",
                    stderr="")
            if argv[:1] == ["freshclam"]:
                return SimpleNamespace(
                    returncode=0, stdout="Database updated\n",
                    stderr="")
            if argv[:2] == ["clamscan", "--version"]:
                return SimpleNamespace(
                    returncode=1, stdout="", stderr="clamscan: boom\n")
            raise AssertionError(f"unexpected provision argv: {argv}")

        monkeypatch.setattr(mp, "_run", fake_mp_run)
        result = mp.provision_models(["clamav"], box=dict(BIG_BOX))
        rep = result["backends"]["clamav"]
        assert not rep["ok"]
        assert rep["status"] == "failed"

    def test_eicar_probe_miss_is_failed(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        which = _WhichFake({"apt-get": "/usr/bin/apt-get"})
        monkeypatch.setattr(shutil, "which", which)

        def fake_mp_run(args, timeout=None):
            argv = list(args)
            if argv[:2] == ["apt-get", "install"]:
                which.mapping["clamscan"] = "/usr/bin/clamscan"
                return SimpleNamespace(
                    returncode=0, stdout="Setting up clamav\n",
                    stderr="")
            if argv[:1] == ["freshclam"]:
                return SimpleNamespace(
                    returncode=0, stdout="Database updated\n",
                    stderr="")
            if argv[:2] == ["clamscan", "--version"]:
                return SimpleNamespace(
                    returncode=0, stdout="ClamAV 1.4.2/27200\n",
                    stderr="")
            raise AssertionError(f"unexpected provision argv: {argv}")

        monkeypatch.setattr(mp, "_run", fake_mp_run)
        # The engine runs but does NOT flag the EICAR test file —
        # a broken install must never report "installed".
        monkeypatch.setattr(av_mod, "_run", _run_result(
            0, "clean.txt: OK\n"))
        detect_antivirus(refresh=True)
        result = mp.provision_models(["clamav"], box=dict(BIG_BOX))
        rep = result["backends"]["clamav"]
        assert not rep["ok"]
        assert rep["status"] == "failed"
        assert "EICAR" in rep["detail"]

    def test_windows_uses_winget(self, monkeypatch):
        import nomorals.tools.models_provision as mp

        which = _WhichFake({"winget": r"C:\winget.exe"})
        monkeypatch.setattr(shutil, "which", which)
        seen: list = []

        def fake_mp_run(args, timeout=None):
            argv = list(args)
            seen.append(argv)
            if argv[:1] == ["winget"]:
                which.mapping["clamscan"] = r"C:\ClamAV\clamscan.exe"
                return SimpleNamespace(
                    returncode=0, stdout="Installed ClamAV\n",
                    stderr="")
            if argv[:1] == ["freshclam"]:
                return SimpleNamespace(
                    returncode=0, stdout="Database updated\n",
                    stderr="")
            if argv[:2] == ["clamscan", "--version"]:
                return SimpleNamespace(
                    returncode=0, stdout="ClamAV 1.4.2/27200\n",
                    stderr="")
            raise AssertionError(f"unexpected provision argv: {argv}")

        monkeypatch.setattr(mp, "_run", fake_mp_run)
        monkeypatch.setattr(av_mod, "_run", _run_result(
            1, "eicar.com: Eicar-Test-Signature FOUND\n"))
        detect_antivirus(refresh=True)
        box = dict(BIG_BOX, platform="win32")
        result = mp.provision_models(["clamav"], box=box)
        rep = result["backends"]["clamav"]
        assert rep["ok"], rep
        assert rep["status"] == "installed"
        assert any(a[0] == "winget" for a in seen)


# ── nm av provision ──────────────────────────────────────────────

class TestNmAvProvision:
    def _fake_provision(self, monkeypatch, result):
        import nomorals.tools.models_provision as mp

        monkeypatch.setattr(mp, "provision_models",
                            lambda backends, **kw: result)

    def test_provision_reports_installed(self, monkeypatch):
        from nomorals.cmdline.commands.av import _cmd_av

        self._fake_provision(monkeypatch, {
            "ok": True, "box": dict(BIG_BOX),
            "backends": {"clamav": {
                "ok": True, "status": "installed",
                "detail": "ClamAV installed — clamscan 1.4.2, "
                          "EICAR probe detected",
                "steps": ["apt-get install clamav: rc=0"]}}})
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="provision"), _ctx(Path(td)))
            assert rc == 0
            text = buf.getvalue()
            assert "clamav" in text
            assert "installed" in text

    def test_provision_failure_returns_1(self, monkeypatch):
        from nomorals.cmdline.commands.av import _cmd_av

        self._fake_provision(monkeypatch, {
            "ok": False, "box": dict(BIG_BOX),
            "backends": {"clamav": {
                "ok": False, "status": "unavailable",
                "detail": "no package manager found on this box",
                "steps": []}}})
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="provision"), _ctx(Path(td)))
            assert rc == 1
            assert "unavailable" in buf.getvalue()

    def test_provision_json(self, monkeypatch):
        from nomorals.cmdline.commands.av import _cmd_av

        self._fake_provision(monkeypatch, {
            "ok": True, "box": dict(BIG_BOX),
            "backends": {"clamav": {
                "ok": True, "status": "already-installed",
                "detail": "clamscan already installed",
                "steps": []}}})
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="provision", json=True),
                             _ctx(Path(td)))
            assert rc == 0
            payload = json.loads(buf.getvalue())
            assert payload["backends"]["clamav"]["status"] == \
                "already-installed"

    def test_parser_accepts_av_provision(self):
        from nomorals.cmdline.parser import _parser

        parsed = _parser().parse_args(["av", "provision"])
        assert parsed.av_action == "provision"

    def test_status_shows_timeouts(self, monkeypatch):
        from nomorals.cmdline.commands.av import _cmd_av

        monkeypatch.setenv("NM_AV_TIMEOUT_DEFENDER_S", "60")
        monkeypatch.setenv("NM_AV_TIMEOUT_CLAMAV_S", "45")
        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="status"), _ctx(Path(td)))
            assert rc == 0
            text = buf.getvalue()
            assert "timeout" in text.lower()
            assert "60" in text
            assert "45" in text

    def test_status_json_carries_timeouts(self, monkeypatch):
        from nomorals.cmdline.commands.av import _cmd_av

        with tempfile.TemporaryDirectory() as td:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = _cmd_av(_ns(av_action="status", json=True),
                             _ctx(Path(td)))
            assert rc == 0
            payload = json.loads(buf.getvalue())
            assert payload["timeouts"] == {"defender": 300.0,
                                           "clamav": 300.0}


# ── config mapping ───────────────────────────────────────────────

class TestTimeoutConfigMapping:
    def test_timeout_vars_are_mapped_to_antivirus(self):
        from nomorals.core.config import _ENV_MAP

        assert _ENV_MAP.get("NM_AV_TIMEOUT_DEFENDER_S") == \
            "direct:nomorals/core/antivirus.py"
        assert _ENV_MAP.get("NM_AV_TIMEOUT_CLAMAV_S") == \
            "direct:nomorals/core/antivirus.py"
