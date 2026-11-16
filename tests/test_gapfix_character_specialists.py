"""Gapfix 17 (audit R1-44) — character generative specialists are invoked.

The four specialists in ``nomorals/characters/specialists.py``
(``character_voice``, ``swap_voice``, ``lora_for``, ``swap_face``) were
implemented and unit-pinned but had ZERO production callers: the only
live character-speech path, ``/charagent talk``, answered in text and
never rendered a line in the character's own voice.

These tests drive the real handler (``RuntimeGamesMixin._control_charagent``)
against a scratch character store (via ``NOMORALS_DATA_DIR``), with the
specialists mocked — no real model is needed to prove the invocation.

Pins:
- a provisioned character's talk renders the line via ``character_voice``
  and delivers it as a voice note through ``_deliver_voice_note``;
- when an RVC model is named, the render is swapped via ``swap_voice``
  and the SWAPPED path is delivered; when the swap fails, the unswapped
  render is delivered as-is (the ladder's ``character_voice_rvc`` code rung);
- on ``ok: False`` the path falls back to the text reply and names the
  wall (never a crash, never an empty result);
- an unvoiced character never calls the specialist (falls back directly);
- a missing model refuses honestly at the seam, naming the model.
"""
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

from nomorals.agents.partner.runtime_games import _character_voice_audio
from nomorals.characters import Character, CharacterStore


def _suggest(prompt: str) -> str:
    first = (prompt or "").splitlines()[0]
    if first.startswith("You are Mara"):
        return "MARA-LINE"
    return "BRAIN-LINE"


def _handler():
    from nomorals.agents.partner.runtime_games import RuntimeGamesMixin

    class _RT(RuntimeGamesMixin):
        def __init__(self, ctx):
            self.context = ctx

    return _RT(SimpleNamespace(suggest=_suggest))


def _scratch_store():
    root = Path(tempfile.mkdtemp())
    os.environ["NOMORALS_DATA_DIR"] = str(root)
    store = CharacterStore()  # honors NOMORALS_DATA_DIR
    assert store.dir == root / "characters"
    return root, store


class _Deliver:
    """Fake _deliver_voice_note: records the call, returns '' (landed)."""

    def __init__(self):
        self.calls = []

    def __call__(self, chat_key, path, text, backend):
        self.calls.append((chat_key, path, text, backend))
        return ""


def _make_char(store, **kw):
    c = Character(name="Mara", persona={"witty": 0.9}, **kw)
    store.save(c)
    return c


def test_voiced_talk_renders_voice_and_delivers_voice_note(monkeypatch):
    """A voiced character's talk calls character_voice and delivers audio."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry")
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    calls = []

    def fake_voice(char, line, **kw):
        calls.append((char.name, line))
        return {"ok": True, "path": "/tmp/mara.wav", "backend": "xtts"}

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice", fake_voice)
    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_voice",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no swap")))

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert calls == [("Mara", "MARA-LINE")]
    assert deliver.calls == [(
        "local:test", "/tmp/mara.wav", "🎭 Mara: MARA-LINE", "xtts")]
    assert out == ""  # the voice note IS the reply


def test_voiced_talk_with_rvc_delivers_swapped_path(monkeypatch):
    """An RVC model swaps the render; the swapped path is delivered."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry", rvc_model="mara-rvc")
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    def fake_voice(char, line, **kw):
        return {"ok": True, "path": "/tmp/mara-raw.wav", "backend": "xtts"}

    swaps = []

    def fake_swap(char, wav_path, **kw):
        swaps.append((char.name, wav_path))
        return {"ok": True, "path": "/tmp/mara-swapped.wav"}

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice", fake_voice)
    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_voice", fake_swap)

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert swaps == [("Mara", "/tmp/mara-raw.wav")]
    assert deliver.calls == [(
        "local:test", "/tmp/mara-swapped.wav", "🎭 Mara: MARA-LINE", "xtts")]
    assert out == ""


def test_rvc_failure_delivers_unswapped_render(monkeypatch):
    """A failed RVC swap falls back to the unswapped render (code rung)."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry", rvc_model="mara-rvc")
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda char, line, **kw: {
            "ok": True, "path": "/tmp/mara-raw.wav", "backend": "xtts"})
    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_voice",
        lambda char, wav_path, **kw: {
            "ok": False, "reason": "rvc unavailable on this box"})

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert deliver.calls == [(
        "local:test", "/tmp/mara-raw.wav", "🎭 Mara: MARA-LINE", "xtts")]
    assert out == ""


def test_voice_render_failure_falls_back_to_text_naming_wall(monkeypatch):
    """ok: False falls back to the text reply and names the wall."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry")
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda char, line, **kw: {
            "ok": False,
            "reason": "no TTS backend installed on this box"})

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert deliver.calls == []
    assert out == ("🎭 Mara: MARA-LINE\n"
                   "🔇 voice: no TTS backend installed on this box")


def test_voice_render_with_no_path_falls_back_to_text(monkeypatch):
    """ok: True with no audio path is still a failure, never an empty result."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry")
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda char, line, **kw: {"ok": True, "path": "", "backend": "xtts"})

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert deliver.calls == []
    assert out.startswith("🎭 Mara: MARA-LINE\n🔇 voice: ")


def test_unvoiced_talk_never_calls_specialist(monkeypatch):
    """A character with no voice named falls back directly, no render."""
    _, store = _scratch_store()
    _make_char(store)
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    def boom(*a, **k):
        raise AssertionError("specialist must not be called")

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice", boom)
    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_voice", boom)

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert deliver.calls == []
    assert out == "🎭 Mara: MARA-LINE"


def test_missing_model_refuses_honestly_at_the_seam(monkeypatch):
    """The REAL specialist, no TTS backend: text + wall, never a crash."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry")
    rt = _handler()
    deliver = _Deliver()
    rt._deliver_voice_note = deliver

    monkeypatch.setattr(
        "nomorals.voice.tts.available_backends", lambda: [])

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert deliver.calls == []
    assert out.startswith("🎭 Mara: MARA-LINE\n🔇 voice: ")
    assert "no TTS backend installed" in out


def test_voice_note_without_delivery_method_names_the_audio(monkeypatch):
    """A runtime with no voice transport still names the rendered audio."""
    _, store = _scratch_store()
    _make_char(store, voice_name="mara-dry")
    rt = _handler()  # bare mixin: no _deliver_voice_note

    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda char, line, **kw: {
            "ok": True, "path": "/tmp/mara.wav", "backend": "xtts"})

    out = rt._control_charagent("talk Mara hello there", "local:test")

    assert out == "🎭 Mara: MARA-LINE\n🔊 /tmp/mara.wav"


def test_character_voice_audio_chains_rvc(monkeypatch):
    """Helper: XTTS render then RVC swap, swapped path wins."""
    char = Character(name="Mara", voice_name="mara-dry",
                     rvc_model="mara-rvc")
    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda c, line, **kw: {
            "ok": True, "path": "/tmp/raw.wav", "backend": "xtts"})
    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_voice",
        lambda c, path, **kw: {"ok": True, "path": "/tmp/swapped.wav"})

    assert _character_voice_audio(char, "hello") == (
        "/tmp/swapped.wav", "xtts", "")


def test_character_voice_audio_swap_failure_keeps_render(monkeypatch):
    """Helper: a failed swap keeps the unswapped render (code rung)."""
    char = Character(name="Mara", voice_name="mara-dry",
                     rvc_model="mara-rvc")
    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda c, line, **kw: {
            "ok": True, "path": "/tmp/raw.wav", "backend": "xtts"})
    monkeypatch.setattr(
        "nomorals.characters.specialists.swap_voice",
        lambda c, path, **kw: {"ok": False, "reason": "rvc down"})

    assert _character_voice_audio(char, "hello") == (
        "/tmp/raw.wav", "xtts", "")


def test_character_voice_audio_failure_names_the_wall(monkeypatch):
    """Helper: ok: False passes the specialist's reason straight through."""
    char = Character(name="Mara", voice_name="mara-dry")
    monkeypatch.setattr(
        "nomorals.characters.specialists.character_voice",
        lambda c, line, **kw: {"ok": False, "reason": "no voice named"})

    assert _character_voice_audio(char, "hello") == ("", "", "no voice named")
