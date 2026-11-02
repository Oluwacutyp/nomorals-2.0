"""Tail-103 pins: nomorals/media/stems.py census sites.

All four sites are capability probes or best-effort
cleanup in the Demucs stem separator, held as exclusions.
``probe_demucs``' contract is printed on it — "Never
raises" — and these guards are how the contract is kept:

90 — ``_importable``: the import IS the probe; any
exception means "not importable here" -> False. The
failure value is the probe's honest negative answer.

99 — ``_cuda_available``: torch missing or its CUDA
check raising -> False, so the separator plans for CPU.
A probe must not crash the planner that called it.

107 — ``_detect_profile_kind``: profile detection
raising -> "" (unknown). ``probe_demucs`` treats "" as
"no profile claim" and still runs its import probe; no
profile is ever fabricated.

191 — ``_RealSeparator.unload``: clearing the CUDA cache
is best-effort cleanup after ``self._sep`` is already
dropped. A cleanup failure changes no result — the
separator is unloaded either way — and ``separate()``
results are computed long before unload runs.

NOT-recording is pinned with the snapshot method:
stems.py records nothing to either failures ledger at
these sites, so both ledgers must be unchanged across
each trigger.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.core.profile as profile_mod
from nomorals.media import stems as S


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 90: import probe answers False ──────────────────────────────

def test_90_importable_missing_module_is_false():
    before = _snap()
    assert S._importable("no_such_module_xyz_103") is False
    # control: a stdlib module imports fine
    assert S._importable("json") is True
    assert _snap() == before


# ── 99: no torch -> no CUDA ─────────────────────────────────────

def test_99_cuda_without_torch_is_false():
    before = _snap()
    with mock.patch.dict(sys.modules, {"torch": None}):
        assert S._cuda_available() is False
    # control: a torch reporting CUDA -> True
    fake = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True))
    with mock.patch.dict(sys.modules, {"torch": fake}):
        assert S._cuda_available() is True
    assert _snap() == before


# ── 107: profile detection failure -> unknown ───────────────────

def test_107_profile_failure_is_empty():
    before = _snap()
    with mock.patch.object(profile_mod, "detect_profile", _raise):
        assert S._detect_profile_kind() == ""
    # control: a detected kind is reported lower-cased
    with mock.patch.object(
            profile_mod, "detect_profile",
            lambda: SimpleNamespace(kind="Workstation")):
        assert S._detect_profile_kind() == "workstation"
    assert _snap() == before


# ── 191: unload cleanup never raises ────────────────────────────

def test_191_unload_swallows_cache_failure():
    before = _snap()
    sep = S._RealSeparator()
    sep._sep = object()
    bad_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=_raise, empty_cache=lambda: None))
    with mock.patch.dict(sys.modules, {"torch": bad_torch}):
        sep.unload()  # must not raise
    assert sep._sep is None
    # control: healthy torch -> the cache really is emptied
    calls = []
    ok_torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: calls.append(1)))
    sep2 = S._RealSeparator()
    sep2._sep = object()
    with mock.patch.dict(sys.modules, {"torch": ok_torch}):
        sep2.unload()
    assert calls == [1] and sep2._sep is None
    assert _snap() == before
