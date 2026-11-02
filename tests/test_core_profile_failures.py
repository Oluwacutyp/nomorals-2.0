"""Tail-104 pins for nomorals/core/profile.py — 0 converts / 4 exclusions.

Non-recording is structural: the module has no logger and no failure
ledger (grep-proven); it is stdlib-only best-effort environment
detection whose defaults are documented values, not claims.

Sites:
98  — ``detect_cloud``: any Exception inside the DMI/hypervisor
heuristics -> return ``"none"``. ``"none"`` is the module's defined
bucket for bare metal / unknown (see KNOWN_CLOUDS and the field
comment "none|aws|gcp|azure|other"); detection "never raises" is the
stated contract, and consumers use the value as a label, not a gate.
116 — ``memory_mb``: /proc/meminfo probe (OSError, ValueError,
IndexError) -> ``pass`` and fall through to the sysctl probe.
Multi-method probe chain; the docstring contract is a value "or 0
when unreadable".
125 — ``memory_mb``: the sysctl probe's Exception -> ``pass``, then
return the documented unreadable value 0.
195 — ``is_termux``: any Exception from the heuristics -> ``False``.
The docstring contract is "Stdlib-only, never raises"; False is the
conservative default (standard non-Termux paths).
"""

from __future__ import annotations

import os
import subprocess
from unittest import mock

from nomorals.core import profile
from nomorals.core.profile import KNOWN_CLOUDS, detect_cloud, is_termux, memory_mb


# ── 98: detection crash -> the documented "none" bucket ──────────

def test_98_detect_cloud_crash_returns_none():
    env = {k: v for k, v in os.environ.items() if k != "NM_CLOUD"}
    with mock.patch.dict(os.environ, env, clear=True), \
            mock.patch.object(profile, "_read_sys_file",
                              side_effect=RuntimeError("dmi boom")):
        assert detect_cloud() == "none"
    # control: unpatched detection returns a known bucket on this host
    with mock.patch.dict(os.environ, env, clear=True):
        assert detect_cloud() in KNOWN_CLOUDS


# ── 116: /proc probe fails -> sysctl probe answers ───────────────

def _meminfo_blocking_open(real_open):
    def _open(path, *a, **k):
        if str(path) == "/proc/meminfo":
            raise OSError("meminfo gone")
        return real_open(path, *a, **k)
    return _open


def test_116_meminfo_failure_falls_through_to_sysctl():
    real_open = open
    done = subprocess.CompletedProcess(
        args=["sysctl"], returncode=0, stdout=str(8 * 1024 ** 3), stderr="")
    with mock.patch("builtins.open", _meminfo_blocking_open(real_open)), \
            mock.patch("subprocess.run", return_value=done):
        assert memory_mb() == 8192


# ── 125: sysctl probe fails too -> documented 0 ──────────────────

def test_125_sysctl_failure_returns_zero():
    real_open = open
    with mock.patch("builtins.open", _meminfo_blocking_open(real_open)), \
            mock.patch("subprocess.run",
                       side_effect=FileNotFoundError("no sysctl")):
        assert memory_mb() == 0


# ── 195: heuristic crash -> False (never raises) ─────────────────

def test_195_is_termux_crash_returns_false():
    with mock.patch.object(profile, "_looks_like_termux",
                           side_effect=RuntimeError("heuristics boom")):
        assert is_termux() is False
    # control: the real heuristic answers a plain bool on this host
    assert isinstance(is_termux(), bool)
