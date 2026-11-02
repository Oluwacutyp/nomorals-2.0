"""Tail-94 pins: nomorals/social/voice_notes.py census sites.

The module's design law is printed in its own header: "Never
raises: a broken note is a missing note, not a crash." Every read
path declares its down-value in its docstring (``get``: "None when
missing/down"; ``list``/``search``: the empty list is the no-notes
value; ``stats``: ``{}`` is the same value the db-None path
returns). All five sites are E104 — broad handlers whose fallback
is one of those declared values, consumed as such. Held as
exclusions:

155 — ``from .profiles import AUDIO_EXTENSIONS`` crash -> the
hardcoded fallback tuple. The fallback is byte-identical to the
canonical list (profiles.py:53-54, verified side by side), so the
extension gate at the next line enforces exactly the same set;
the failure value is the real list, not an approximation.
226 — ``stats`` DB error -> ``{}``. The declared empty-store
value; no in-repo consumer branches on stats (it powers a
human-facing activity answer), and a down store answers the
same ``{}`` as a never-connected one (the ``_db is None`` path).
238 — ``get`` DB error -> None, the docstring's "missing/down"
value; indistinguishable from a genuinely absent note BY DESIGN.
254 — ``list`` DB error -> ``[]``, the no-notes value.
275 — ``search_voice_notes`` DB error -> ``[]``, the no-matches
value; the chat surface honestly reports "no matches" because,
to every consumer the module offers, a down store has none.

None of the five records: the module imports no ledger and the
social failures ledger (social/failures.py — the package norm
leads.py follows) must be unchanged across each trigger; the
pins assert exactly that. The gap is observability only — a
shelved sketch (bind + record at these five sites), not a
convert: no failure here is success-shaped on a decision-
consumed surface, every fallback is a value the function's
contract already names.

Each test pins the degraded behavior AND the NOT-recording
property. No bare ``except: pass`` anywhere; no sys.path hacks.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import nomorals.social.failures as social_fail
from nomorals.social import voice_notes as V


class _BoomDB:
    """A connection whose every execute fails (store went down)."""

    def execute(self, *args, **kwargs):
        raise RuntimeError("db down")


def _store(tmp_path, **kw):
    return V.VoiceNoteStore(db_path=str(tmp_path / "vn.db"),
                            vault_dir=str(tmp_path / "vault"), **kw)


def _audio(tmp_path, name="note.wav"):
    p = tmp_path / name
    p.write_bytes(b"RIFF\x24\x00\x00\x00WAVE")
    return str(p)


def _count():
    return social_fail.failure_count()


# ── 155: profiles import crash -> identical fallback list ──────

def test_155_profiles_import_crash_uses_identical_fallback(tmp_path):
    before = _count()
    store = _store(tmp_path)
    broken = ModuleType("nomorals.social.profiles")  # no AUDIO_EXTENSIONS
    with mock.patch.dict(sys.modules,
                         {"nomorals.social.profiles": broken}):
        note = store.store_voice_note("chat1", "ada", _audio(tmp_path))
        assert note is not None  # .wav admitted by the fallback list
        exe = tmp_path / "tool.exe"
        exe.write_bytes(b"MZ")
        assert store.store_voice_note("chat1", "ada", str(exe)) is None
    # control: the canonical import path admits/rejects identically
    note2 = store.store_voice_note("chat1", "ada",
                                   _audio(tmp_path, "n2.wav"))
    assert note2 is not None
    assert _count() == before


# ── 226: stats DB error -> {} (the empty-store value) ──────────

def test_226_stats_db_error_returns_empty_dict(tmp_path):
    before = _count()
    store = _store(tmp_path,
                   stt=lambda p, l: {"text": "hi", "backend": "fake"})
    assert store.store_voice_note("chat1", "ada",
                                  _audio(tmp_path)) is not None
    healthy = store.stats("chat1")
    assert healthy["total"] == 1 and healthy["transcribed"] == 1
    store._db = _BoomDB()
    assert store.stats("chat1") == {}
    assert store.stats() == {}
    assert _count() == before


# ── 238: get DB error -> None (missing/down are one value) ─────

def test_238_get_db_error_returns_none(tmp_path):
    before = _count()
    store = _store(tmp_path)
    note = store.store_voice_note("chat1", "ada", _audio(tmp_path))
    assert note is not None
    assert store.get(note.note_id) is not None      # control: present
    assert store.get("vn_missing") is None          # control: absent
    store._db = _BoomDB()
    assert store.get(note.note_id) is None          # down == absent
    assert _count() == before


# ── 254: list DB error -> [] (the no-notes value) ──────────────

def test_254_list_db_error_returns_empty(tmp_path):
    before = _count()
    store = _store(tmp_path)
    assert store.store_voice_note("chat1", "ada",
                                  _audio(tmp_path)) is not None
    assert len(store.list("chat1")) == 1            # control
    store._db = _BoomDB()
    assert store.list("chat1") == []
    assert store.list() == []
    assert _count() == before


# ── 275: search DB error -> [] (the no-matches value) ──────────

def test_275_search_db_error_returns_empty(tmp_path):
    before = _count()
    store = _store(
        tmp_path,
        stt=lambda p, l: {"text": "the price is five",
                          "backend": "fake"})
    assert store.store_voice_note("chat1", "ada",
                                  _audio(tmp_path)) is not None
    hits = store.search_voice_notes("price")        # control
    assert len(hits) == 1 and hits[0].transcript == "the price is five"
    store._db = _BoomDB()
    assert store.search_voice_notes("price") == []
    assert _count() == before
