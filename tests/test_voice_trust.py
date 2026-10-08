"""Offline tests for build-map #82: the voice trust layer.

STT/TTS are injectable seams — every test uses mocks, so this suite
never needs a microphone, a model, or a network connection.
"""

import os
import tempfile

import pytest

from nomorals.social.profiles import ProfileStore, VoiceIntro, format_profile
from nomorals.social.voice_notes import (
    VoiceNoteStore,
    control_vnote,
    normalize_voice_language,
    VOICE_LANGUAGES,
)


# ── fixtures ──────────────────────────────────────────────────────────

def _audio(suffix=".wav") -> str:
    """A fake-but-plausibly-shaped audio file (never actually decoded)."""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.write(fd, b"RIFF....fake audio bytes")
    os.close(fd)
    return path


def _mock_stt(text="mock transcript here"):
    return lambda path, lang: {"text": text, "backend": "mock-stt",
                               "language": lang}


def _mock_tts(out_path):
    return lambda text, lang, stack="chatterbox": out_path


@pytest.fixture()
def store():
    return ProfileStore(db_path=tempfile.mktemp(suffix=".db"))


@pytest.fixture()
def vstore():
    base = tempfile.mkdtemp()
    return VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                          vault_dir=os.path.join(base, "vault"))


# ── voice intro (#81 extension) ─────────────────────────────────────────

def test_set_and_get_voice_intro(store):
    p = store.create("Ada", "gig")
    vi = store.set_voice_intro(p.profile_id, _audio(),
                               language="yo-ekiti", stt=_mock_stt())
    assert isinstance(vi, VoiceIntro)
    assert vi.language == "yo-ekiti"
    assert vi.transcript == "mock transcript here"
    assert vi.stt_backend == "mock-stt"
    assert os.path.isfile(vi.audio_path)
    got = store.get_voice_intro(p.profile_id)
    assert got is not None and got.audio_path == vi.audio_path
    # get() loads it too
    assert store.get(p.profile_id).voice_intro is not None


def test_voice_intro_no_stt_honest(store):
    p = store.create("Ada", "gig")
    vi = store.set_voice_intro(p.profile_id, _audio())
    assert vi is not None
    assert vi.transcript == "" and vi.stt_backend == ""


def test_voice_intro_missing_profile(store):
    assert store.set_voice_intro("prof_nope", _audio()) is None


def test_voice_intro_bad_audio(store):
    p = store.create("Ada", "gig")
    assert store.set_voice_intro(p.profile_id, "/nope/missing.wav") is None
    txt = tempfile.mktemp(suffix=".txt")
    open(txt, "w").write("not audio")
    assert store.set_voice_intro(p.profile_id, txt) is None


def test_voice_intro_language_normalization(store):
    p = store.create("Ada", "gig")
    vi = store.set_voice_intro(p.profile_id, _audio(), language="pidgin")
    assert vi.language == "pcm"
    vi2 = store.set_voice_intro(p.profile_id, _audio(), language="klingon")
    assert vi2.language == "en"


def test_delete_voice_intro(store):
    p = store.create("Ada", "gig")
    vi = store.set_voice_intro(p.profile_id, _audio())
    assert store.delete_voice_intro(p.profile_id) is True
    assert store.get_voice_intro(p.profile_id) is None
    assert not os.path.isfile(vi.audio_path)
    assert store.delete_voice_intro(p.profile_id) is False


def test_format_profile_shows_voice_intro(store):
    p = store.create("Ada", "gig")
    store.set_voice_intro(p.profile_id, _audio(), stt=_mock_stt("i love design"))
    out = format_profile(store.get(p.profile_id))
    assert "🎙️" in out and "i love design" in out


# ── voice notes ─────────────────────────────────────────────────────────

def test_send_voice_note_defaults_to_no_stt(vstore):
    # The fixture store has no STT seam: audio is stored, transcript stays
    # empty — honest, never fabricated.
    n = vstore.send_voice_note("chat1", "Ada", _audio(), language="pcm")
    assert n is not None and n.transcript == "" and n.language == "pcm"


def test_send_voice_note_with_stt_seam():
    base = tempfile.mkdtemp()
    s = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                       vault_dir=os.path.join(base, "v"),
                       stt=_mock_stt("the price is negotiable"))
    n = s.send_voice_note("chat1", "Ada", _audio(), language="pcm")
    assert n is not None
    assert n.transcript == "the price is negotiable"
    assert n.language == "pcm" and n.stt_backend == "mock-stt"
    assert os.path.isfile(n.audio_path)


def test_send_voice_note_no_stt_honest(vstore):
    n = vstore.send_voice_note("chat1", "Ada", _audio())
    assert n is not None
    assert n.transcript == "" and n.stt_backend == ""


def test_send_voice_note_bad_input(vstore):
    assert vstore.send_voice_note("", "Ada", _audio()) is None
    assert vstore.send_voice_note("chat1", "Ada", "/nope.wav") is None
    txt = tempfile.mktemp(suffix=".txt")
    open(txt, "w").write("x")
    assert vstore.send_voice_note("chat1", "Ada", txt) is None


def test_search_voice_notes():
    base = tempfile.mkdtemp()
    s = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                       vault_dir=os.path.join(base, "v"),
                       stt=_mock_stt("the price is negotiable"))
    s.send_voice_note("chat1", "Ada", _audio(), language="en")
    s2 = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                        vault_dir=os.path.join(base, "v"),
                        stt=_mock_stt("see you tomorrow"))
    s2.send_voice_note("chat2", "Emeka", _audio(), language="yo")
    hits = s.search_voice_notes("price")
    assert len(hits) == 1 and hits[0].chat_id == "chat1"
    # chat-scoped search
    assert s.search_voice_notes("price", "chat2") == []
    assert s.search_voice_notes("") == []


def test_get_and_list():
    base = tempfile.mkdtemp()
    s = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                       vault_dir=os.path.join(base, "v"),
                       stt=_mock_stt("hi"))
    n = s.send_voice_note("chat1", "Ada", _audio())
    assert s.get(n.note_id).sender == "Ada"
    assert s.get("vn_nope") is None
    assert len(s.list("chat1")) == 1
    assert s.list("chat9") == []


# ── stack routing (#30) ─────────────────────────────────────────────────

def test_stack_routing():
    assert VoiceNoteStore.stack_for(True) == "chatterbox"
    assert VoiceNoteStore.stack_for(False) == "xtts"


def test_synthesize_preview_routed():
    base = tempfile.mkdtemp()
    out = _audio(".mp3")
    s = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                       vault_dir=os.path.join(base, "v"),
                       tts=_mock_tts(out))
    assert s.synthesize_preview("hello", public=True) == out
    assert s.synthesize_preview("hello", public=False) == out
    assert s.synthesize_preview("   ") is None


def test_synthesize_preview_no_tts_honest(vstore):
    assert vstore.synthesize_preview("hello") is None


# ── language scope ─────────────────────────────────────────────────────

def test_voice_languages_scope():
    for code in ("en", "pcm", "yo", "yo-ekiti", "ha", "ig"):
        assert code in VOICE_LANGUAGES
    assert normalize_voice_language("yo-ekiti") == "yo-ekiti"
    assert normalize_voice_language("pidgin") == "pcm"
    assert normalize_voice_language("ekiti") == "yo-ekiti"
    assert normalize_voice_language("klingon") == "en"
    assert normalize_voice_language(None) == "en"


# ── chat: /vnote ────────────────────────────────────────────────────────

def test_vnote_send_and_search_chat():
    base = tempfile.mkdtemp()
    s = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                       vault_dir=os.path.join(base, "v"),
                       stt=_mock_stt("lagos traffic is wild"))
    ctx = type("C", (), {"voice_note_store": s})()
    out = control_vnote(f"send grp1 {_audio()} pcm", context=ctx,
                        sender="Ada")
    assert "voice note sent" in out and "pcm" in out
    out2 = control_vnote("search traffic", context=ctx)
    assert "lagos traffic" in out2


def test_vnote_play_and_preview():
    base = tempfile.mkdtemp()
    out_audio = _audio(".mp3")
    s = VoiceNoteStore(db_path=os.path.join(base, "vn.db"),
                       vault_dir=os.path.join(base, "v"),
                       stt=_mock_stt("hi"), tts=_mock_tts(out_audio))
    ctx = type("C", (), {"voice_note_store": s})()
    n = s.send_voice_note("grp1", "Ada", _audio())
    out = control_vnote(f"play {n.note_id}", context=ctx)
    assert n.note_id in out
    out2 = control_vnote("preview public en hello there", context=ctx)
    assert "chatterbox" in out2 and out_audio in out2
    out3 = control_vnote("preview private yo kaabo", context=ctx)
    assert "xtts" in out3


def test_vnote_preview_no_tts_honest(vstore):
    ctx = type("C", (), {"voice_note_store": vstore})()
    out = control_vnote("preview public en hello", context=ctx)
    assert "no TTS wired" in out and "chatterbox" in out


def test_vnote_usage_and_never_raises():
    assert "send" in control_vnote("")
    assert "usage" in control_vnote("send only")
    assert "no voice note" in control_vnote("play vn_nope")
    # garbage never raises
    assert isinstance(control_vnote(None), str)
    assert isinstance(control_vnote("blow up \x00", context=object()), str)


# ── chat: /uprofile voice ───────────────────────────────────────────────

def test_uprofile_voice_chat():
    from nomorals.social.profiles import control_uprofile
    base = tempfile.mkdtemp()
    s = ProfileStore(db_path=os.path.join(base, "p.db"))
    ctx = type("C", (), {"profile_store": s})()
    # instance attribute (not a class attribute) — no method binding
    ctx.stt = _mock_stt("i build apps")
    out = control_uprofile("create gig Ada", context=ctx, sender="Ada")
    pid = out.split("(")[1].split(")")[0]
    out2 = control_uprofile(f"voice {pid} {_audio()} yo", context=ctx,
                            sender="Ada")
    assert "voice intro set" in out2 and "yo" in out2
    out3 = control_uprofile(f"voice {pid}", context=ctx, sender="Ada")
    assert "i build apps" in out3
    out4 = control_uprofile(f"show {pid}", context=ctx, sender="Ada")
    assert "🎙️" in out4
