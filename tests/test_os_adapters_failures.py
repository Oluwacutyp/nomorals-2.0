"""Tail-106 pins: nomorals/os/adapters.py census sites (96, 104, 167, 237).

The module's contract is printed in its own header: "Defensive by
contract: an adapter never raises. Missing optional pieces degrade
to a minimal Session whose state records what was unavailable."
That contract is implemented in two layers:

* the OUTER ``to_os_session`` handlers catch a whole-projection
  failure, log a warning, and return ``_safe_view`` — a Session
  whose state carries ``degraded: True`` plus the error string;
* the INNER per-attribute ``_opt`` guards (the census sites) catch
  a single attribute whose property raises and degrade just that
  attribute to its default, so one broken property on a
  half-initialized wrapped object does not nuke the whole view.

All four sites are held as EXCLUSIONS:

96 / 167 / 237 — the ``_opt`` guards in the account / voice /
generic adapters. A raising property yields the attribute's
default (None / "" / the caller's default); every other attribute
still projects. The defaults read as "absent", never as a
fabricated value, and the wrapped objects are duck-typed session
concepts whose properties may legitimately raise (the docstrings
say exactly that). Non-recording is pinned: a per-attribute
failure emits no "degraded" warning and no ledger entry — that
signal belongs to the outer layer and fires only when the whole
projection fails (control test at the bottom proves the split).

104 — the account validator guard: ``is_valid()`` raising yields
``valid: False``. Fail-closed: an unverifiable session is
reported invalid, never valid. The healthy controls pin that
True and False verdicts pass through untouched.
"""
from __future__ import annotations

import logging

import pytest

from nomorals.os.adapters import (
    AccountSessionAdapter,
    GenericSessionAdapter,
    VoiceSessionAdapter,
)


class _EvilAccount:
    """Account whose ``metadata`` property and validator both raise."""

    service = "github"
    username = "u"
    scope = "s"
    oauth_token = None
    created_at = "c"
    last_used = "l"

    @property
    def metadata(self):
        raise RuntimeError("metadata boom")

    def is_valid(self):
        raise RuntimeError("validator boom")


class _GoodAccount:
    service = "github"
    username = "u"
    scope = "s"
    oauth_token = "tok"
    metadata = {"a": 1}
    created_at = "c"
    last_used = "l"

    def is_valid(self):
        return True


class _Manager:
    def __init__(self, acct):
        self._acct = acct

    def get_session(self, service, username):
        return self._acct


# ── 96: account _opt — raising property degrades to the default ──

def test_96_raising_property_degrades_attribute_only(caplog):
    adapter = AccountSessionAdapter(_Manager(_EvilAccount()))
    with caplog.at_level(logging.WARNING, logger="nomorals.os.adapters"):
        sess = adapter.to_os_session("github", "u")
    # the broken attribute takes its default; the projection survives
    assert sess.state["metadata"] == {}
    assert sess.state["kind"] == "account"
    assert sess.state["service"] == "github"
    # NOT the outer degraded view, and nothing was logged/recorded:
    # per-attribute degradation is silent by design
    assert "degraded" not in sess.state
    assert not [r for r in caplog.records if "degraded" in r.getMessage()]


def test_96_healthy_account_projects_fully():
    adapter = AccountSessionAdapter(_Manager(_GoodAccount()))
    sess = adapter.to_os_session("github", "u")
    assert sess.state["metadata"] == {"a": 1}
    assert sess.state["has_oauth"] is True


# ── 104: validator crash -> valid False (fail-closed) ────────────

def test_104_validator_crash_reports_invalid():
    adapter = AccountSessionAdapter(_Manager(_EvilAccount()))
    sess = adapter.to_os_session("github", "u")
    assert sess.state["valid"] is False


def test_104_healthy_validator_verdicts_pass_through():
    class _Invalid(_GoodAccount):
        def is_valid(self):
            return False

    good = AccountSessionAdapter(_Manager(_GoodAccount()))
    assert good.to_os_session("github", "u").state["valid"] is True
    bad = AccountSessionAdapter(_Manager(_Invalid()))
    assert bad.to_os_session("github", "u").state["valid"] is False


# ── 167: voice _opt — raising property degrades to the default ───

class _EvilVoice:
    session_id = "vs1"
    device_id = "d1"
    profile = "p"
    consent = object()
    stats = None

    @property
    def state(self):
        raise RuntimeError("state boom")


class _GoodVoice:
    session_id = "vs2"
    device_id = "d2"
    profile = "p"
    consent = None
    stats = None
    state = "listening"


def test_167_raising_property_degrades_attribute_only():
    sess = VoiceSessionAdapter().to_os_session(_EvilVoice())
    assert sess.state["voice_state"] == ""  # the default, read as absent
    assert sess.state["device_id"] == "d1"  # siblings still project
    assert sess.state["kind"] == "voice"
    assert "degraded" not in sess.state


def test_167_healthy_voice_projects_fully():
    sess = VoiceSessionAdapter().to_os_session(_GoodVoice())
    assert sess.state["voice_state"] == "listening"
    assert sess.state["consented"] is False


# ── 237: generic _opt — raising property omits the field ─────────

class _EvilPayload:
    session_id = "s1"
    status = "live"

    @property
    def metadata(self):
        raise RuntimeError("metadata boom")


def test_237_raising_property_omits_field():
    sess = GenericSessionAdapter().to_os_session(_EvilPayload())
    assert sess.state["session_id"] == "s1"
    assert sess.state["status"] == "live"
    # default None -> the field is omitted, never fabricated
    assert "metadata" not in sess.state
    assert "degraded" not in sess.state


def test_237_dict_payload_control():
    sess = GenericSessionAdapter().to_os_session(
        {"session_id": "s2", "status": "x"})
    assert sess.state["session_id"] == "s2"
    assert sess.state["status"] == "x"


# ── control (not a census site): the OUTER layer is the one that
#    marks degraded + logs, proving the inner guards' silence is a
#    layering decision, not a missing signal ──────────────────────

def test_outer_layer_marks_degraded_and_logs(caplog):
    adapter = AccountSessionAdapter(None)  # no manager: projection raises
    with caplog.at_level(logging.WARNING, logger="nomorals.os.adapters"):
        sess = adapter.to_os_session("github", "u")
    assert sess.state["degraded"] is True
    assert "error" in sess.state
    assert [r for r in caplog.records if "degraded" in r.getMessage()]
