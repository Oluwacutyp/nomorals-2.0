"""Tail-104 pins for nomorals/agents/runtime_profile.py — 0 converts /
3 exclusions.

Non-recording: the module's only reporting channel is ``_log``
(grep-proven — no failure-ledger import); pins capture ``_log`` and
assert no ERROR record from the exercised swallow paths.

Sites:
105 — ``shed_level``: the PowerAwareGate telemetry probe raises
Exception -> ``pass``, return ``"none"``. The gate is the THIRD
fallback (after a string status and status attributes); the module
header defines the semantics: "Unknown values behave as ``none``".
``"none"`` here never exceeds the profile envelope — ``_shed_cap``
with "none" returns the envelope maximum, the same ceiling a healthy
"no shedding" reading produces; the failure cannot oversubscribe
beyond the sanctioned envelope.
127 — ``free_memory_mb``: the /proc/meminfo probe (OSError,
ValueError, IndexError) -> ``pass`` and fall through to
``core.profile.memory_mb``. Probe chain; docstring contract is a
value "or 0 when unreadable".
133 — ``free_memory_mb``: the ``memory_mb`` fallback itself raises
Exception -> return the documented unreadable value 0.
"""

from __future__ import annotations

import logging
from unittest import mock

from nomorals.agents import runtime_profile
from nomorals.agents.runtime_profile import free_memory_mb, shed_level
from nomorals.autonomy import power_aware


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=1)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def errors(self) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno >= logging.ERROR]


def _capture_log():
    cap = _Capture()
    old_level = runtime_profile._log.level
    runtime_profile._log.addHandler(cap)
    runtime_profile._log.setLevel(1)
    return cap, old_level


def _release_log(cap: _Capture, old_level: int) -> None:
    runtime_profile._log.removeHandler(cap)
    runtime_profile._log.setLevel(old_level)


def _meminfo_blocking_open(real_open):
    def _open(path, *a, **k):
        if str(path) == "/proc/meminfo":
            raise OSError("meminfo gone")
        return real_open(path, *a, **k)
    return _open


# ── 105: gate telemetry crash -> documented "none" ───────────────

class _ExplodingGate:
    def __init__(self, monitor) -> None:
        raise RuntimeError("telemetry boom")


def test_105_shed_level_gate_crash_returns_none():
    cap, old_level = _capture_log()
    try:
        with mock.patch.object(power_aware, "PowerAwareGate",
                               _ExplodingGate):
            # a status object with no shed attributes reaches the
            # gate probe, which explodes -> documented default
            assert shed_level(None, object()) == "none"
    finally:
        _release_log(cap, old_level)
    # control: the primary string path is unaffected by the probe
    assert shed_level(None, "heavy") == "heavy"
    assert cap.errors() == []


# ── 127: /proc probe fails -> memory_mb fallback answers ─────────

def test_127_free_memory_falls_through_to_memory_mb():
    real_open = open
    with mock.patch("builtins.open", _meminfo_blocking_open(real_open)), \
            mock.patch("nomorals.core.profile.memory_mb",
                       return_value=2048):
        assert free_memory_mb() == 2048


# ── 133: fallback fails too -> documented 0 ──────────────────────

def test_133_free_memory_fallback_failure_returns_zero():
    cap, old_level = _capture_log()
    try:
        real_open = open
        with mock.patch("builtins.open", _meminfo_blocking_open(real_open)), \
                mock.patch("nomorals.core.profile.memory_mb",
                           side_effect=RuntimeError("probe boom")):
            assert free_memory_mb() == 0
    finally:
        _release_log(cap, old_level)
    assert cap.errors() == []
