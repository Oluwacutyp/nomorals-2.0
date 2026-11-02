"""Pass tail-117 pins: nomorals/media/vocals.py — 3 silent handlers.

- 131 E104 ``_importable``: the import IS the probe; a missing/broken
  module is the honest negative (backend rung skipped, next rung runs).
- 140 E104 ``_detect_profile_kind``: detection crash -> "" (unknown),
  the same shape as media/stems' profile probe; callers treat "" as
  no-profile and still select a backend.
- 630 E104 ``parse_full_request``: the ace_step delegate crash ->
  bed=None and the module's own regex rung answers; the delegated
  result is used verbatim when the delegate works (control).

All exclusions: probes answer honestly, degradation lands on the
module's own next rung. Nothing is recorded by any of the three
guards (no ledger call exists on these paths — behavior pinned).
"""

from types import SimpleNamespace


import nomorals.core.profile as core_profile
import nomorals.media.ace_step as ace_step
from nomorals.media import vocals


# ── 131: _importable ─────────────────────────────────────────────────

def test_importable_real_module_is_true():
    assert vocals._importable("json") is True


def test_importable_missing_module_is_false():
    assert vocals._importable("no_such_module_xyz_117") is False


# ── 140: _detect_profile_kind ────────────────────────────────────────

def test_profile_kind_healthy_returns_string():
    assert isinstance(vocals._detect_profile_kind(), str)


def test_profile_kind_crash_degrades_to_unknown(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("profile boom")

    monkeypatch.setattr(core_profile, "detect_profile", boom)
    assert vocals._detect_profile_kind() == ""


# ── 630: parse_full_request delegate ladder ──────────────────────────

def test_parse_full_request_uses_delegate_when_it_works(monkeypatch):
    monkeypatch.setattr(
        ace_step, "parse_bed_request",
        lambda text, styles=None: SimpleNamespace(
            topic="delegated topic", style="delegatedstyle"))
    assert vocals.parse_full_request(
        "make me an afrobeats song about Lagos") == {
            "topic": "delegated topic", "style": "delegatedstyle"}


def test_parse_full_request_delegate_crash_falls_to_regex(monkeypatch):
    def boom(text, styles=None):
        raise RuntimeError("ace down")

    monkeypatch.setattr(ace_step, "parse_bed_request", boom)
    # the module's own regex rung parses the same text honestly
    assert vocals.parse_full_request(
        "make me an afrobeats song about Lagos") == {
            "topic": "Lagos", "style": "pop"}


def test_parse_full_request_garbage_is_none(monkeypatch):
    def boom(text, styles=None):
        raise RuntimeError("ace down")

    monkeypatch.setattr(ace_step, "parse_bed_request", boom)
    assert vocals.parse_full_request("hello there") is None
