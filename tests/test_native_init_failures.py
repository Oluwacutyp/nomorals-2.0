"""Tail-108 pins: nomorals/native/__init__.py census sites.

The native package's posture: every probe degrades to the honest
absence value — "unknown", a missing key, the pure-python backend —
never a fabricated version or capability. All four sites are
exclusions:

* 248 — ``load()``: a v1 .so predates ``nm_vecsim_simd_level``, so
  setting its restype raises AttributeError; the guard lets the
  load finish and ``simd_level()`` reports "unknown" for the old
  library. FLAGGED-ACCEPTED: the guard carries no noqa and stays in
  the flagged scan.
* 340 — ``simd_level()`` crash -> "unknown", byte-identical to the
  not-built and v1 answers (in-code: "never break a status probe").
* 755 — ``_kernel_block``: a version symbol that raises is skipped
  and the loop tries the next kernel's symbol; if all raise, the
  block simply has no "version" key.
* 780 — ``info()``: the vecsim version call raising leaves
  "version" out of the doctor dict; every other field is computed
  truthfully.
"""

from pathlib import Path

import pytest

from nomorals import native as nat


class _Fn:
    def __init__(self):
        self.restype = None
        self.argtypes = None


class _FakeLibV1:
    """A loaded library whose nm_vecsim_simd_level symbol is absent."""

    def __getattr__(self, name):
        if name == "nm_vecsim_simd_level":
            raise AttributeError(name)
        return _Fn()


class _FakeLibV2:
    def __init__(self):
        self.simd_fn = _Fn()

    def __getattr__(self, name):
        if name == "nm_vecsim_simd_level":
            return self.simd_fn
        return _Fn()


# ── 248: load() ──────────────────────────────────────────────────

def test_load_survives_v1_library_without_simd_symbol(monkeypatch):
    monkeypatch.setattr(nat, "_load_one", lambda slot: _FakeLibV1())
    assert isinstance(nat.load(), _FakeLibV1)


def test_load_v2_library_gets_simd_restype(monkeypatch):
    lib = _FakeLibV2()
    monkeypatch.setattr(nat, "_load_one", lambda slot: lib)
    assert nat.load() is lib
    assert lib.simd_fn.restype is not None


# ── 340: simd_level() ────────────────────────────────────────────

class _LibBoom:
    def nm_vecsim_simd_level(self):
        raise RuntimeError("ffi boom")


class _LibOk:
    def nm_vecsim_simd_level(self):
        return b"avx2"


def test_simd_level_crash_reports_unknown(monkeypatch):
    monkeypatch.setattr(nat, "load", lambda: _LibBoom())
    assert nat.simd_level() == "unknown"


def test_simd_level_decodes_and_not_built_is_unknown(monkeypatch):
    monkeypatch.setattr(nat, "load", lambda: _LibOk())
    assert nat.simd_level() == "avx2"
    monkeypatch.setattr(nat, "load", lambda: None)
    assert nat.simd_level() == "unknown"


# ── 755: _kernel_block ───────────────────────────────────────────

def _boom():
    raise RuntimeError("version boom")


class _LibVersions:
    def nm_vecsim_version(self):
        raise RuntimeError("version boom")

    def nm_mlp_version(self):
        return b"2.0"


class _LibAllBoom:
    def __getattr__(self, name):
        return _boom


def test_kernel_block_continues_past_crashing_version_symbol():
    block = nat._kernel_block(_LibVersions(), Path("/nonexistent/libnm.so"))
    assert block["version"] == "2.0"
    assert block["backend"] == "native-cpp"


def test_kernel_block_all_versions_crash_means_no_version_key():
    block = nat._kernel_block(_LibAllBoom(), Path("/nonexistent/libnm.so"))
    assert "version" not in block
    block_none = nat._kernel_block(None, Path("/nonexistent/libnm.so"))
    assert block_none["backend"] == "pure-python"
    assert "version" not in block_none


# ── 780: info() ──────────────────────────────────────────────────

class _LibInfoBoom:
    def nm_vecsim_version(self):
        raise RuntimeError("version boom")


class _LibInfoOk:
    def nm_vecsim_version(self):
        return b"3.1"


@pytest.fixture()
def _no_kernels(monkeypatch):
    monkeypatch.setattr(nat, "load_mlp", lambda: None)
    monkeypatch.setattr(nat, "load_bpe", lambda: None)
    monkeypatch.setattr(nat, "load_mem", lambda: None)


def test_info_version_crash_omits_key_rest_truthful(monkeypatch, _no_kernels):
    monkeypatch.setattr(nat, "load", lambda: _LibInfoBoom())
    info = nat.info()
    assert "version" not in info
    assert info["loaded"] is True
    assert info["backend"] == "native-cpp"


def test_info_version_present_when_call_works(monkeypatch, _no_kernels):
    monkeypatch.setattr(nat, "load", lambda: _LibInfoOk())
    assert nat.info()["version"] == "3.1"
