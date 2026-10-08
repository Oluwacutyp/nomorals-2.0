"""Tests for nomorals/media/distribute.py — build-map #60.

All offline. Covers: split validation (must sum to 100), AI disclosure
presence, packet completeness, Phase 1 never uploads (no network calls),
split ledger recording, conversational draft flow, tool registration.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from nomorals.media import distribute as dist
from nomorals.media.distribute import (
    AI_DISCLOSURE,
    DistributionError,
    ReleaseDraft,
    advance_draft,
    clear_draft,
    draft_prompt,
    format_packet,
    legal_notes,
    parse_distribute_request,
    parse_splits,
    pending_draft,
    phase2_hooks,
    prepare,
    start_draft,
    submission_checklist,
    validate_splits,
)
from nomorals.finance.ledger import Ledger


@pytest.fixture()
def wav():
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture()
def ledger():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    yield Ledger(path)
    os.unlink(path)


# ── splits ──────────────────────────────────────────────────────────────────

def test_splits_valid():
    assert validate_splits({"artist": 70, "producer": 30}) == {
        "artist": 70.0, "producer": 30.0}


def test_splits_reject_90():
    with pytest.raises(DistributionError, match="sum to 100"):
        validate_splits({"artist": 60, "producer": 30})


def test_splits_reject_empty():
    with pytest.raises(DistributionError, match="mandatory"):
        validate_splits({})


def test_splits_reject_negative():
    with pytest.raises(DistributionError, match="negative"):
        validate_splits({"artist": 110, "producer": -10})


def test_splits_reject_nonnumeric():
    with pytest.raises(DistributionError, match="isn't a number"):
        validate_splits({"artist": "lots"})


def test_parse_splits():
    assert parse_splits("artist=70, producer=30") == {
        "artist": 70.0, "producer": 30.0}


def test_parse_splits_percent_sign():
    assert parse_splits("artist=70%, producer=30%") == {
        "artist": 70.0, "producer": 30.0}


def test_parse_splits_garbage():
    assert parse_splits("just some words") is None


# ── AI disclosure + legal ───────────────────────────────────────────────────

def test_ai_disclosure_constant():
    assert AI_DISCLOSURE == "AI-generated instrumental + AI vocals"


def test_legal_notes_cover_all():
    notes = legal_notes()
    joined = " ".join(notes)
    assert "AI disclosure" in joined
    assert "LANDR" in joined and "Content ID" in joined
    assert "UMG" in joined or "DistroKid" in joined


# ── packet ──────────────────────────────────────────────────────────────────

def test_prepare_packet(wav, ledger):
    packet = prepare(wav, title="Lagos Nights", artist="Devon",
                     splits={"artist": 70, "producer": 30}, ledger=ledger)
    assert packet.title == "Lagos Nights"
    assert packet.artist == "Devon"
    assert packet.ai_disclosure == AI_DISCLOSURE
    assert packet.splits == {"artist": 70.0, "producer": 30.0}
    assert packet.phase == 1
    assert packet.packet_id


def test_prepare_rejects_bad_splits(wav):
    with pytest.raises(DistributionError):
        prepare(wav, title="X", artist="Y",
                splits={"artist": 50, "producer": 30})


def test_prepare_rejects_missing_audio():
    with pytest.raises(DistributionError, match="not found"):
        prepare("/nonexistent/song.wav", title="X", artist="Y",
                splits={"artist": 100})


def test_prepare_default_platforms(wav):
    packet = prepare(wav, title="X", artist="Y", splits={"artist": 100})
    assert "spotify" in packet.platforms
    assert "boomplay" in packet.platforms  # Nigerian platforms matter
    assert "audiomack" in packet.platforms


def test_checklist_complete(wav):
    packet = prepare(wav, title="Lagos Nights", artist="Devon",
                     splits={"artist": 70, "producer": 30})
    items = submission_checklist(packet)
    joined = " ".join(items)
    assert "Lagos Nights" in joined
    assert AI_DISCLOSURE in joined
    assert "artist: 70%" in joined and "producer: 30%" in joined
    assert "3000" in joined  # cover art spec


def test_format_packet_has_everything(wav):
    packet = prepare(wav, title="Lagos Nights", artist="Devon",
                     splits={"artist": 100})
    out = format_packet(packet)
    assert AI_DISCLOSURE in out
    assert "Phase 1" in out
    assert "YOU click submit" in out or "you submit" in out.lower()
    assert "Legal weather" in out


def test_phase1_never_uploads(wav, monkeypatch):
    """No network calls anywhere in the prepare path."""
    import socket
    def _nope(*a, **k):
        raise AssertionError("network call attempted")
    monkeypatch.setattr(socket, "socket", _nope)
    monkeypatch.setattr(socket, "create_connection", _nope)
    packet = prepare(wav, title="X", artist="Y", splits={"artist": 100})
    assert packet.phase == 1


def test_split_ledger_recording(wav, ledger):
    prepare(wav, title="Lagos Nights", artist="Devon",
            splits={"artist": 70, "producer": 30}, ledger=ledger)
    txns = ledger.transactions(category="royalty_split")
    assert len(txns) == 1
    assert "Lagos Nights" in txns[0].note
    assert "artist=70%" in txns[0].note


# ── conversational draft ────────────────────────────────────────────────────

def test_draft_flow():
    key = "test-chat-distribute"
    clear_draft(key)
    draft = start_draft(key)
    assert draft.step == "title"
    assert "title" in draft_prompt(draft).lower()

    nxt = advance_draft(draft, "Lagos Nights")
    assert draft.step == "artist"
    assert "artist" in nxt.lower()

    nxt = advance_draft(draft, "Devon")
    assert draft.step == "platforms"

    nxt = advance_draft(draft, "all")
    assert draft.step == "splits"
    assert "boomplay" in str(draft.platforms)

    nxt = advance_draft(draft, "artist=70, producer=30")
    assert draft.step == "confirm"
    assert draft.splits == {"artist": 70.0, "producer": 30.0}

    nxt = advance_draft(draft, "yes")
    assert nxt == "done"
    clear_draft(key)


def test_draft_rejects_bad_splits():
    draft = ReleaseDraft(step="splits")
    nxt = advance_draft(draft, "artist=50, producer=30")
    assert draft.step == "splits"  # didn't advance
    assert "100" in nxt


def test_draft_cancel():
    key = "test-chat-distribute-cancel"
    start_draft(key)
    draft = pending_draft(key)
    assert advance_draft(draft, "cancel") == "cancel"


def test_parse_distribute_request():
    assert parse_distribute_request("/distribute") == {"song_path": ""}
    assert parse_distribute_request("/distribute /tmp/x.wav") == {
        "song_path": "/tmp/x.wav"}
    assert parse_distribute_request("hello") is None


# ── phase 2 ─────────────────────────────────────────────────────────────────

def test_phase2_gated():
    hooks = phase2_hooks()
    assert "gate" in hooks
    assert "legal review" in hooks["gate"].lower()


# ── tool registration ───────────────────────────────────────────────────────

def test_tool_registers():
    from unittest.mock import MagicMock
    registry = MagicMock()
    registry.context = MagicMock()
    dist.register(registry)
    names = [c.args[0] for c in registry.register.call_args_list]
    assert "distribute" in names
