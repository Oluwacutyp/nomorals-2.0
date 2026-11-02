"""Tail-97 pins: nomorals/accounts/temp_mail.py census sites.

All three sites are the same shape in three providers: the
``probe()`` availability check. Each carries the stated contract
"Never raises — a failed probe means 'skip me'", logs the failure
at debug, and returns False. False is the REFUSAL value in this
system: ``grab_address_cascade`` (the consumer) skips any provider
whose probe() returns False and only ever returns an address from
a provider that probed True and then actually grabbed one. A probe
crash can never produce an address, a token, or a success claim —
it can only remove a provider from the cascade, which is the safe
direction for disposable-mail signup flows. All three are held as
exclusions:

178 — OneSecMailProvider.probe: ``_get`` crash -> False.
247 — GuerrillaMailProvider.probe: ``_call`` crash -> False.
337 — MailTmProvider.probe: ``list_domains`` crash -> False.

NOT-recording is pinned with the snapshot method: temp_mail.py
calls no failures-ledger record function anywhere, so both the
agency and agents ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
from nomorals.accounts import temp_mail as T


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise(*args, **kwargs):
    raise RuntimeError("boom")


# ── 178: 1secmail probe crash -> unavailable ────────────────────

def test_178_onesecmail_probe_failure_is_unavailable():
    before = _snap()
    prov = T.OneSecMailProvider()
    with mock.patch.object(prov, "_get", _raise):
        assert prov.probe() is False
    # control halves: a parsed domain list probes True, an empty
    # one probes False by value (not by exception).
    with mock.patch.object(prov, "_get", lambda **kw: ["x.example"]):
        assert prov.probe() is True
    with mock.patch.object(prov, "_get", lambda **kw: []):
        assert prov.probe() is False
    assert _snap() == before


# ── 247: guerrillamail probe crash -> unavailable ───────────────

def test_247_guerrillamail_probe_failure_is_unavailable():
    before = _snap()
    prov = T.GuerrillaMailProvider()
    with mock.patch.object(prov, "_call", _raise):
        assert prov.probe() is False
    # control: a session address in the reply probes True.
    with mock.patch.object(prov, "_call",
                           lambda **kw: {"email_addr": "a@x.example"}):
        assert prov.probe() is True
    assert _snap() == before


# ── 337: mail.tm probe crash -> unavailable ─────────────────────

def test_337_mailtm_probe_failure_is_unavailable():
    before = _snap()
    prov = T.MailTmProvider()
    with mock.patch.object(prov, "list_domains", _raise):
        assert prov.probe() is False
    # control: a parsed domain list probes True.
    with mock.patch.object(prov, "list_domains", lambda: ["x.example"]):
        assert prov.probe() is True
    assert _snap() == before
