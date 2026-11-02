"""Failure pins for nomorals/core/profile_visible.py (tail-112).

One pin per census site. All sites adjudicated EXCLUSION:

- 53 free_memory_mb rung 1 (FLAGGED-ACCEPTED — no noqa, stays in the
  flagged scan): typed (OSError, ValueError, IndexError) guard on the
  /proc/meminfo parse -> falls through to the memory_mb() rung. The
  docstring prints the chain: "Free (available) RAM in MB, or 0 when
  unreadable." Same probe-chain family as runtime_profile (tail-104).
- 59 free_memory_mb rung 2: memory_mb() crash -> 0, the documented
  unknown value.
- 93 power_shed: a monitor whose shed_level() raises -> "none". The
  module docstring prints the contract ("fails open to \\"none\\"")
  and the only consumers are display surfaces (the console/dashboard
  one-liner and runtime_profile's summary methods); the real power
  gate computes its own shed level and never reads this value.

The module has no logger: non-recording is structural (pinned below).
"""

from unittest import mock

from nomorals.core import profile as prof_mod
from nomorals.core import profile_visible as pv


def _meminfo_boom(real_open):
    def _open(path, *args, **kwargs):
        if str(path) == "/proc/meminfo":
            raise OSError("meminfo gone")
        return real_open(path, *args, **kwargs)
    return _open


# ------------------------------------------------------------------ 53/59
def test_free_memory_mb_healthy_control():
    assert pv.free_memory_mb() > 0


def test_free_memory_mb_meminfo_failure_falls_through():
    import builtins
    with mock.patch("builtins.open",
                    side_effect=_meminfo_boom(builtins.open)), \
            mock.patch.object(prof_mod, "memory_mb", return_value=8192):
        assert pv.free_memory_mb() == 8192


def test_free_memory_mb_double_failure_returns_zero():
    import builtins
    with mock.patch("builtins.open",
                    side_effect=_meminfo_boom(builtins.open)), \
            mock.patch.object(prof_mod, "memory_mb",
                              side_effect=RuntimeError("probe boom")):
        assert pv.free_memory_mb() == 0


# --------------------------------------------------------------------- 93
class _BadMonitor:
    def shed_level(self, status=None):
        raise RuntimeError("telemetry boom")


class _GoodMonitor:
    def shed_level(self, status=None):
        return "HEAVY"


class _StatusObj:
    level = 2


def test_power_shed_monitor_crash_fails_open_to_none():
    assert pv.power_shed(_BadMonitor()) == "none"


def test_power_shed_controls():
    assert pv.power_shed(_GoodMonitor()) == "heavy"
    assert pv.power_shed(status=" Light ") == "light"
    assert pv.power_shed(status=_StatusObj()) == "medium"
    assert pv.power_shed() == "none"


# ------------------------------------------------------- non-recording
def test_module_has_no_logger():
    assert not hasattr(pv, "_log")
