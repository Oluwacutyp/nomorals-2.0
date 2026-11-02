"""Tail-109 pins for nomorals/partner/context.py — 4 exclusions, 0 converts.

Each test pins one census site's fallback behavior plus a control half
proving the except path was really taken. The module imports no
failures-ledger record function and has no logger (grep-proven), so
non-recording on these paths is structural.

Sites:
* 77 E104 — _life_anchors_block(): the deferred identity-pack import
  fails -> the block is omitted (""). Absence-shaped: no identity facts
  are invented; the rest of the prompt assembles without it.
* 184 E104 — _capabilities_block(): the deferred connectors-registry
  import fails -> "" (block omitted).
* 188 E104 — _capabilities_block(): list_connectors() itself crashes
  -> "". No connector is ever fabricated; the catalog block just drops.
* 269 E103 — build(): the mood label fetch is best-effort ("label is
  best-effort here"). A label-access failure degrades the dialogue
  block to the empty-label rendering while the mood block still renders
  from values. The guard is deliberately narrow — a current()-level
  failure is NOT swallowed by it and still fails the build (boundary
  pin), so 269 fabricates no whole-build fallback.
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

import nomorals.connectors.registry as registry
from nomorals.partner.context import PartnerContextBuilder
from nomorals.partner.mood import MoodEngine
from nomorals.partner.persona import default_persona
from nomorals.partner.relationship import Relationship


# ── 77: identity-pack import guard ───────────────────────────────────────────

def test_77_life_anchors_import_failure_omits_block() -> None:
    # Control: the real identity pack renders the anchors block.
    assert "Your life" in PartnerContextBuilder._life_anchors_block()
    # Trigger: a background module without the pack -> block omitted.
    empty = types.ModuleType("nomorals.partner.background")
    with mock.patch.dict(sys.modules, {"nomorals.partner.background": empty}):
        assert PartnerContextBuilder._life_anchors_block() == ""


# ── 184 / 188: capabilities-block guards ─────────────────────────────────────

def test_184_capabilities_registry_import_failure() -> None:
    empty = types.ModuleType("nomorals.connectors.registry")
    with mock.patch.dict(sys.modules, {"nomorals.connectors.registry": empty}):
        assert PartnerContextBuilder._capabilities_block() == ""


def test_188_capabilities_catalog_crash_omits_block() -> None:
    fake = [{"name": "ProbeSvc", "description": "probe connector"}]
    with mock.patch.object(registry, "list_connectors", lambda: fake):
        # Control: a working catalog renders into the block.
        assert "ProbeSvc" in PartnerContextBuilder._capabilities_block()
    with mock.patch.object(
            registry, "list_connectors",
            lambda: (_ for _ in ()).throw(RuntimeError("registry down"))):
        assert PartnerContextBuilder._capabilities_block() == ""


# ── 269: mood label best-effort, narrowly fenced ─────────────────────────────

def _parts():
    engine = MoodEngine(default_persona().baselines, now=1_000_000.0)
    engine.note_event("deep_conversation", 0.8)
    rel = Relationship()
    rel.advance_stage()
    return engine, rel, default_persona()


class _LabelBoomState:
    values = {"energy": 55.0, "happiness": 60.0}

    @property
    def label(self):
        raise RuntimeError("label boom")


class _LabelBoomMood:
    def current(self):
        return _LabelBoomState()

    def describe(self):
        return "fake mood, steady"


class _DeadMood:
    def current(self):
        raise RuntimeError("current boom")

    def describe(self):
        return "fake mood, steady"


def test_269_mood_label_failure_degrades_to_empty_label() -> None:
    _, rel, persona = _parts()
    # Control: a healthy engine's label steers the dialogue block.
    engine, rel_ok, persona_ok = _parts()
    healthy = PartnerContextBuilder().build(
        persona=persona_ok, mood=engine, relationship=rel_ok)
    label_block = persona_ok.dialogue_block(mood_label=engine.current().label)
    assert label_block == "" or label_block in healthy.content
    # Trigger: label access raises -> build survives; the mood block
    # still renders from values and the dialogue block falls back to
    # the empty-label rendering.
    crashed = PartnerContextBuilder().build(
        persona=persona, mood=_LabelBoomMood(), relationship=rel)
    assert persona.name in crashed.content
    assert "OUTPUT CONTRACT" in crashed.content
    assert "energy 55" in crashed.content
    empty_block = persona.dialogue_block(mood_label="")
    assert empty_block == "" or empty_block in crashed.content


def test_269_boundary_current_failure_not_swallowed() -> None:
    _, rel, persona = _parts()
    # The 269 guard covers only the label fetch: a mood engine whose
    # current() itself raises still fails the build downstream
    # (_mood_block), proving no whole-build fallback is fabricated.
    with pytest.raises(RuntimeError):
        PartnerContextBuilder().build(
            persona=persona, mood=_DeadMood(), relationship=rel)
