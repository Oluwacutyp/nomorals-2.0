"""Tail-19 pins: ProfileStore failure values + NOT-recording.

Every method in nomorals/social/profiles.py is documented never-raises;
each pin drives the handler into its except path and pins the exact
fallback value plus the fact that the social failures ledger is
unchanged (snapshot before/after, tail-8 method). No handler here calls
record_failure, and none of these failures may surface as a crash or a
success claim.
"""

import os
import sqlite3
import sys
import tempfile
import types
import wave

import pytest

from nomorals.social import failures as soc_fail
from nomorals.social.profiles import ProfileStore


def _store():
    return ProfileStore(db_path=os.path.join(tempfile.mkdtemp(), "p.db"))


def _ledger():
    return list(soc_fail.recent_failures())


class _FailDB:
    """Wraps a real connection; raises only on SQL with the given prefix."""

    def __init__(self, conn, prefix):
        self._c = conn
        self._p = prefix

    def execute(self, sql, params=()):
        if sql.strip().upper().startswith(self._p):
            raise sqlite3.DatabaseError("boom")
        return self._c.execute(sql, params)


def _good_wav(directory):
    path = os.path.join(directory, "good.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(b"\x00\x00" * 800)
    return path


def _profile_with_element(store):
    p = store.create("owner1", "gig", "Ann")
    el = store.add_element(p.profile_id, "project", "T")
    return p, el


def test_list_broken_db_is_empty():
    s = _store()
    s.create("owner1", "gig", "Ann")
    before = _ledger()
    s._db.close()
    assert s.list() == []
    assert _ledger() == before


def test_answer_prompt_update_failure_is_false():
    s = _store()
    p = s.create("owner1", "gig", "Ann")
    s._db = _FailDB(s._db, "UPDATE PROFILES SET PROMPTS_JSON")
    before = _ledger()
    assert s.answer_prompt(p.profile_id, "work", "x") is False
    assert _ledger() == before


def test_wav_duration_hint_corrupt_is_none():
    d = tempfile.mkdtemp()
    bad = os.path.join(d, "bad.wav")
    with open(bad, "wb") as fh:
        fh.write(b"not a wav at all")
    assert ProfileStore._wav_duration_hint(bad) is None
    assert isinstance(ProfileStore._wav_duration_hint(_good_wav(d)), float)


def test_voice_intro_import_failure_preserves_requested_language():
    # Parent convert (tail-19, site 411): on a voice_notes import
    # failure the handler used to substitute the "en" default for an
    # explicitly requested language, storing a wrong-language STT
    # transcript under an "en" label. It now preserves the request;
    # STT is called with the requested language, and no-arg callers
    # still get "en".
    s = _store()
    p = s.create("owner1", "gig", "Ann")
    vault = tempfile.mkdtemp()
    orig_vault = ProfileStore.__dict__["voice_vault_dir"]
    ProfileStore.voice_vault_dir = staticmethod(lambda: vault)
    real_mod = sys.modules.get("nomorals.social.voice_notes")
    sys.modules["nomorals.social.voice_notes"] = types.ModuleType("voice_notes")
    seen = []

    def _fake_stt(path, lang):
        seen.append(lang)
        return {"text": "hello", "backend": "fake"}

    before = _ledger()
    try:
        vi = s.set_voice_intro(
            p.profile_id, _good_wav(tempfile.mkdtemp()), language="yoruba",
            stt=_fake_stt)
        vi_default = s.set_voice_intro(
            p.profile_id, _good_wav(tempfile.mkdtemp()), stt=_fake_stt)
    finally:
        if real_mod is not None:
            sys.modules["nomorals.social.voice_notes"] = real_mod
        else:
            del sys.modules["nomorals.social.voice_notes"]
        ProfileStore.voice_vault_dir = orig_vault
    assert vi is not None
    assert vi.language == "yoruba"
    assert vi.transcript == "hello"
    assert os.path.isfile(vi.audio_path)
    assert vi_default is not None
    assert vi_default.language == "en"
    assert seen == ["yoruba", "en"]
    assert _ledger() == before


def test_get_voice_intro_broken_db_is_none():
    s = _store()
    p = s.create("owner1", "gig", "Ann")
    before = _ledger()
    s._db.close()
    assert s.get_voice_intro(p.profile_id) is None
    assert _ledger() == before


def test_delete_voice_intro_update_failure_is_false():
    s = _store()
    p, _el = _profile_with_element(s)
    vault = tempfile.mkdtemp()
    orig_vault = ProfileStore.__dict__["voice_vault_dir"]
    ProfileStore.voice_vault_dir = staticmethod(lambda: vault)
    try:
        s.set_voice_intro(p.profile_id, _good_wav(tempfile.mkdtemp()))
    finally:
        ProfileStore.voice_vault_dir = orig_vault
    s._db = _FailDB(s._db, "UPDATE PROFILES")
    before = _ledger()
    assert s.delete_voice_intro(p.profile_id) is False
    assert _ledger() == before


def test_add_element_insert_failure_is_none():
    s = _store()
    p = s.create("owner1", "gig", "Ann")
    s._db = _FailDB(s._db, "INSERT INTO ELEMENTS")
    before = _ledger()
    assert s.add_element(p.profile_id, "project", "T") is None
    assert _ledger() == before


def test_like_insert_failure_is_none():
    s = _store()
    p, el = _profile_with_element(s)
    s._db = _FailDB(s._db, "INSERT OR IGNORE INTO LIKES")
    before = _ledger()
    assert s.like(p.profile_id, el.element_id, "bob") is None
    assert _ledger() == before


def test_likes_for_broken_db_is_empty():
    s = _store()
    p, el = _profile_with_element(s)
    s.like(p.profile_id, el.element_id, "bob")
    before = _ledger()
    s._db.close()
    assert s.likes_for(el.element_id) == []
    assert _ledger() == before


def test_matches_broken_db_is_empty():
    s = _store()
    p, el = _profile_with_element(s)
    s.like(p.profile_id, el.element_id, "bob")
    before = _ledger()
    s._db.close()
    assert s.matches("owner1") == []
    assert _ledger() == before
