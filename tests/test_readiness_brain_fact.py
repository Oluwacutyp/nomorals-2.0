"""Pin: the startup readiness card's brain fact reads BrainGate's
``degraded_now`` PROPERTY (brain_gate.py declares it @property).

Regression: runtime._readiness_facts called ``gate.degraded_now()``
— invoking the returned bool raised TypeError on every boot, the
method's own try swallowed it into facts["brain"] = None, and the
readiness card could never report the brain ready OR degraded.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import nomorals.agents.notifier as notifier
from nomorals.agents.partner.runtime import PartnerRuntime


class _Gate:
    """Mirrors BrainGate's real shape: degraded_now is a property."""

    def __init__(self, degraded: bool):
        self._degraded = degraded

    @property
    def degraded_now(self) -> bool:
        return self._degraded


class _Gateway:
    def known_platforms(self):
        return []


def _runtime() -> PartnerRuntime:
    rt = PartnerRuntime.__new__(PartnerRuntime)
    rt.context = SimpleNamespace(db=object())
    rt._boot_intent_receipt = None
    rt._boot_update_receipt = None
    rt.gateway = _Gateway()
    return rt


def test_brain_fact_degraded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        notifier, "resolve_brain_gate", lambda ctx: _Gate(True))
    facts = _runtime()._readiness_facts([])
    assert facts["brain"] is not None
    assert facts["brain"][0] == "degraded"


def test_brain_fact_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        notifier, "resolve_brain_gate", lambda ctx: _Gate(False))
    facts = _runtime()._readiness_facts([])
    assert facts["brain"] == ("ready", "")


def test_brain_fact_no_gate_stays_ready(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(notifier, "resolve_brain_gate", lambda ctx: None)
    facts = _runtime()._readiness_facts([])
    assert facts["brain"] == ("ready", "")
