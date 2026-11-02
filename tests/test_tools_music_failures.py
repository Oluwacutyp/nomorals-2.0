"""Tail-95 pins: nomorals/tools/music.py census sites.

Four exclusions and one convert.

148 / 168 / 347 — save_draft persistence swallows in _draft,
_revise and _collab (``except Exception: pass``). Persistence is
best-effort BY DESIGN across the whole module: the draft lives in
_ACTIVE_DRAFTS and every flow (revise / perform / collab) works
from the in-memory copy; the reply each function returns is the
rendered notebook of that live draft and makes no persistence
claim. A failed save loses only the cross-restart reload, never
the session's work, and the failure value (continue with the
in-memory draft) is the contract's own degraded mode. Held as
exclusions; the trio is the batch's WEAKEST exclusion — the
counter-argument is real: the loss is silent, so a restart
discovers it late. Shelved sketch: log the save failure at
warning via the module's ``log`` (observability only).

248 — _taste_hint degrades to "" on any profile read failure.
The hint is a prompt nudge; "" is the function's designed
no-hint value (also what a genuinely empty profile yields), so
the failure value and the no-data value are the same by
contract. Exclusion.

374 — CONVERTED. The _dj_mix genre path swallowed a
fetch_trending crash into ``queries = []``, which then returned
"no trending tracks found for '<genre>'" — a failure reported
as a factual empty chart, the false-clean shape. The module's
own idiom everywhere else is "<op> failed: {exc}" (draft /
revise / taste / collab / video). The except now returns
"trending lookup failed: {exc} — name songs explicitly"; the
genuine-empty paths (ok=False, ok with no usable titles) still
return the "no trending tracks" line, pinned as controls.

NOT-recording is pinned with the snapshot method: music.py
calls no failures-ledger record function anywhere, so both the
agency and agents ledgers must be unchanged across each trigger.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import nomorals.agency.failures as ag_fail
import nomorals.agents.failures as agents_fail
import nomorals.characters as characters_mod
import nomorals.characters.dialogue as dialogue_mod
import nomorals.media.dj as dj_mod
import nomorals.media.dj_mixdrop as mixdrop_mod
import nomorals.media.resolver as resolver_mod
import nomorals.media.song_draft as song_draft_mod
import nomorals.media.taste as taste_mod
from nomorals.tools import music as M


def _snap():
    return (list(ag_fail.recent_failures(limit=200)),
            list(agents_fail.recent_failures(limit=200)))


def _raise_save(*args, **kwargs):
    raise OSError("disk full")


def _fake_taste_store():
    return SimpleNamespace(
        profile=SimpleNamespace(liked_styles=[], disliked_styles=[]))


# ── 148: _draft save failure -> notebook still returned ─────────

def test_148_draft_save_failure_keeps_session_draft():
    key = "tail95-draft"
    draft = SimpleNamespace(topic="lagos nights")
    before = _snap()
    try:
        with mock.patch.object(taste_mod, "TasteStore",
                               lambda: _fake_taste_store()), \
             mock.patch.object(song_draft_mod, "draft_song",
                               lambda *a, **k: draft), \
             mock.patch.object(song_draft_mod, "save_draft", _raise_save), \
             mock.patch.object(song_draft_mod, "render_notebook",
                               lambda d: "NOTEBOOK"):
            out = M._draft("lagos nights", "pop", key, None)
        assert out == "NOTEBOOK"
        assert M._ACTIVE_DRAFTS[key] is draft
    finally:
        M._ACTIVE_DRAFTS.pop(key, None)
    assert _snap() == before


# ── 168: _revise save failure -> revised notebook returned ──────

def test_168_revise_save_failure_keeps_session_draft():
    key = "tail95-revise"
    draft0 = SimpleNamespace(topic="rain")
    draft1 = SimpleNamespace(topic="rain, darker")
    M._ACTIVE_DRAFTS[key] = draft0
    before = _snap()
    try:
        with mock.patch.object(song_draft_mod, "revise_draft",
                               lambda *a, **k: draft1), \
             mock.patch.object(song_draft_mod, "save_draft", _raise_save), \
             mock.patch.object(song_draft_mod, "render_notebook",
                               lambda d: "NOTEBOOK2"):
            out = M._revise("make it darker", key, None)
        assert out == "NOTEBOOK2"
        assert M._ACTIVE_DRAFTS[key] is draft1
    finally:
        M._ACTIVE_DRAFTS.pop(key, None)
    assert _snap() == before


# ── 248: _taste_hint read failure -> "" (the no-hint value) ─────

class _BrokenTaste:
    @property
    def liked_styles(self):
        raise RuntimeError("profile store boom")

    @property
    def disliked_styles(self):
        raise RuntimeError("profile store boom")


def test_248_taste_hint_failure_returns_empty_hint():
    before = _snap()
    assert M._taste_hint(_BrokenTaste()) == ""
    # control: a readable profile yields its hint
    ok = SimpleNamespace(liked_styles=["afrobeats"], disliked_styles=[])
    assert M._taste_hint(ok) == "owner leans toward: afrobeats"
    assert _snap() == before


# ── 347: _collab save failure -> verse still returned ───────────

def test_347_collab_save_failure_keeps_verse():
    key = "tail95-collab"
    draft = SimpleNamespace(topic="rain", style="pop", performance_notes="")
    M._ACTIVE_DRAFTS[key] = draft
    char = SimpleNamespace(name="Zara")
    store = SimpleNamespace(get=lambda name: char)
    before = _snap()
    try:
        with mock.patch.object(characters_mod, "CharacterStore",
                               lambda: store), \
             mock.patch.object(dialogue_mod, "converse",
                               lambda *a, **k: "sixteen bars of rain"), \
             mock.patch.object(song_draft_mod, "save_draft", _raise_save):
            out = M._collab("Zara verse", key, None)
        assert "Zara wrote the verse" in out
        assert "sixteen bars of rain" in out
        assert "[Zara — verse]" in draft.performance_notes
        assert "sixteen bars of rain" in draft.performance_notes
    finally:
        M._ACTIVE_DRAFTS.pop(key, None)
    assert _snap() == before


# ── 374: CONVERT — trending crash is a failure, not an empty chart

def test_374_trending_crash_reports_failure_not_empty_chart():
    before = _snap()

    def _boom(text):
        raise RuntimeError("source down")

    with mock.patch.object(dj_mod, "fetch_trending", _boom):
        out = M._dj_mix("rock")
    assert out.startswith("trending lookup failed:")
    assert "source down" in out
    assert "no trending tracks found" not in out

    # control 1: a genuine empty result keeps the original line
    with mock.patch.object(dj_mod, "fetch_trending",
                           lambda text: {"ok": False, "reason": "nope"}):
        out = M._dj_mix("rock")
    assert out == ("no trending tracks found for 'rock' — "
                   "name songs explicitly")

    # control 2: a working lookup still builds and drops the mix
    with mock.patch.object(
            dj_mod, "fetch_trending",
            lambda text: {"ok": True, "tracks": [{"title": "Song A"}]}), \
         mock.patch.object(resolver_mod, "SourceResolver",
                           lambda: object()), \
         mock.patch.object(mixdrop_mod, "fetch_track_audio",
                           lambda q, r: {"query": q}), \
         mock.patch.object(mixdrop_mod, "render_mixdrop",
                           lambda fetches, workdir, mix_name: {
                               "ok": True, "path": "mix.mp3",
                               "duration_s": 12.0, "tracklist": [],
                               "skipped": []}):
        out = M._dj_mix("rock")
    assert out.startswith("🎧 mix dropped")
    assert _snap() == before
