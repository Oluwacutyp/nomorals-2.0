"""Tests for /caption, /vision, /voices chat wiring.

Thin wiring layers — test command parsing, engine-not-available paths,
and success paths (mocked). Never hits real STT/vision/TTS.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


def _mixin():
    """Minimal mixin instance with mocked context/gateway."""
    from nomorals.agents.partner.runtime_media import RuntimeMediaMixin
    from nomorals.agents.partner.runtime_voice import RuntimeVoiceMixin

    class M(RuntimeMediaMixin, RuntimeVoiceMixin):
        def __init__(self):
            self.context = SimpleNamespace(
                tools=MagicMock(),
                extras={},
            )
            self.gateway = MagicMock()

        def _ref_from_key(self, chat_key):
            return SimpleNamespace(platform="tg", chat_id="1")

        def _send_long_checked(self, platform, chat, text):
            return text

    return M()


def _msg(media_kinds=()):
    """Fake chat message with media attachments."""
    media = [SimpleNamespace(kind=k, path=f"/tmp/fake.{k}.mp4", mime="")
             for k in media_kinds]
    return SimpleNamespace(media=media)


# ── /caption ──────────────────────────────────────────────────────────

def test_caption_no_video_returns_usage():
    m = _mixin()
    out = m._control_caption("", chat_key="", message=_msg())
    assert "usage" in out.lower()
    assert "hormozi" in out


def test_caption_no_message_returns_usage():
    m = _mixin()
    out = m._control_caption("", chat_key="")
    assert "usage" in out.lower()


def test_caption_style_parsing():
    m = _mixin()
    msg = _msg(media_kinds=("video",))
    with patch("nomorals.media_edit.captions.caption_video",
               return_value={"output": "/tmp/out.mp4"}) as cv:
        m.gateway.send_file.return_value = SimpleNamespace(ok=True)
        out = m._control_caption("karaoke", chat_key="tg:1", message=msg)
        assert cv.call_args[1]["style"] == "karaoke"
        assert "karaoke" in out


def test_caption_default_style_is_hormozi():
    m = _mixin()
    msg = _msg(media_kinds=("video",))
    with patch("nomorals.media_edit.captions.caption_video",
               return_value={"output": "/tmp/out.mp4"}) as cv:
        m.gateway.send_file.return_value = SimpleNamespace(ok=True)
        m._control_caption("", chat_key="tg:1", message=msg)
        assert cv.call_args[1]["style"] == "hormozi"


def test_caption_engine_error_is_honest():
    m = _mixin()
    msg = _msg(media_kinds=("video",))
    from nomorals.media_edit.videos import MediaEditError
    with patch("nomorals.media_edit.captions.caption_video",
               side_effect=MediaEditError(
                   "captioning needs faster-whisper: pip install faster-whisper")):
        out = m._control_caption("", chat_key="tg:1", message=msg)
        assert "caption failed" in out
        assert "faster-whisper" in out


def test_caption_never_raises_on_broken_message():
    m = _mixin()
    out = m._control_caption("", chat_key="", message=None)
    assert isinstance(out, str)
    out = m._control_caption("", chat_key="",
                             message=SimpleNamespace())
    assert isinstance(out, str)


# ── /vision ───────────────────────────────────────────────────────────

def test_vision_no_image_returns_usage():
    m = _mixin()
    out = m._control_vision("", chat_key="", message=_msg())
    assert "usage" in out.lower()


def test_vision_attached_image():
    m = _mixin()
    msg = _msg(media_kinds=("image",))
    outcome = SimpleNamespace(
        ok=True, value={"description": "a cat", "provider": "groq"})
    m.context.tools.call.return_value = outcome
    out = m._control_vision("what is this?", chat_key="tg:1", message=msg)
    assert "a cat" in out
    # The image path from the attachment was used.
    call_kwargs = m.context.tools.call.call_args[1]
    assert call_kwargs["path"] == "/tmp/fake.image.mp4"
    assert call_kwargs["prompt"] == "what is this?"


def test_vision_tool_failure_is_honest():
    m = _mixin()
    msg = _msg(media_kinds=("image",))
    outcome = SimpleNamespace(
        ok=False, error=SimpleNamespace(message="no vision provider"))
    m.context.tools.call.return_value = outcome
    out = m._control_vision("", chat_key="tg:1", message=msg)
    assert "vision failed" in out
    assert "no vision provider" in out


def test_vision_never_raises():
    m = _mixin()
    m.context.tools.call.side_effect = RuntimeError("boom")
    out = m._control_vision("", chat_key="",
                             message=_msg(media_kinds=("image",)))
    assert isinstance(out, str)
    assert "vision failed" in out
    assert "boom" in out


def test_vision_path_first_then_question():
    m = _mixin()
    outcome = SimpleNamespace(
        ok=True, value={"description": "a dog", "provider": "groq"})
    m.context.tools.call.return_value = outcome
    out = m._control_vision("/tmp/pic.png what animal is this?",
                            chat_key="tg:1", message=_msg())
    call_kwargs = m.context.tools.call.call_args[1]
    assert call_kwargs["path"] == "/tmp/pic.png"
    assert call_kwargs["prompt"] == "what animal is this?"
    assert "a dog" in out


def test_vision_question_then_path_last():
    m = _mixin()
    outcome = SimpleNamespace(
        ok=True, value={"description": "a dog", "provider": "groq"})
    m.context.tools.call.return_value = outcome
    out = m._control_vision("what animal is this? /tmp/pic.png",
                            chat_key="tg:1", message=_msg())
    call_kwargs = m.context.tools.call.call_args[1]
    assert call_kwargs["path"] == "/tmp/pic.png"
    assert call_kwargs["prompt"] == "what animal is this?"


def test_vision_url_target():
    m = _mixin()
    outcome = SimpleNamespace(
        ok=True, value={"description": "a sunset", "provider": "groq"})
    m.context.tools.call.return_value = outcome
    m._control_vision("https://example.com/img.jpg describe this",
                      chat_key="tg:1", message=_msg())
    call_kwargs = m.context.tools.call.call_args[1]
    assert call_kwargs["path"] == "https://example.com/img.jpg"
    assert call_kwargs["prompt"] == "describe this"


def test_vision_attached_image_beats_path_arg():
    m = _mixin()
    msg = _msg(media_kinds=("image",))
    outcome = SimpleNamespace(
        ok=True, value={"description": "attached wins", "provider": "groq"})
    m.context.tools.call.return_value = outcome
    m._control_vision("/tmp/other.png", chat_key="tg:1", message=msg)
    call_kwargs = m.context.tools.call.call_args[1]
    # Attached media takes priority over the path argument.
    assert call_kwargs["path"] == "/tmp/fake.image.mp4"


# ── /voices ───────────────────────────────────────────────────────────

def test_voices_lists_sections():
    m = _mixin()
    out = m._control_voices("", chat_key="tg:1")
    assert "🎙️ voices" in out
    assert "system" in out.lower()
    assert "piper" in out.lower()
    assert "catalogue" in out.lower()


def test_voices_never_raises_without_backends():
    m = _mixin()
    with patch("nomorals.voice.tts.SystemTTSBackend",
               side_effect=ImportError("no tts")), \
         patch("nomorals.voice.catalogue.default_catalogue",
               side_effect=ImportError("no catalogue")):
        out = m._control_voices("", chat_key="tg:1")
        assert isinstance(out, str)
        assert "🎙️ voices" in out


def test_voices_shows_piper_downloads(tmp_path):
    m = _mixin()
    # Fake a Piper voices dir with .onnx files.
    (tmp_path / "en_US-lessac-medium.onnx").write_text("x")
    (tmp_path / "en_US-lessac-medium.onnx.json").write_text("{}")
    (tmp_path / "en_GB-alan-low.onnx").write_text("x")
    with patch("nomorals.voice.tts.PiperBackend._search_dirs",
               return_value=[str(tmp_path)]):
        out = m._control_voices("", chat_key="tg:1")
        assert "en_US-lessac-medium" in out
        assert "en_GB-alan-low" in out
        # .onnx.json sidecar must not appear as a voice.
        assert "onnx.json" not in out.replace(
            "en_US-lessac-medium.onnx.json", "")


def test_voices_shows_catalogue_voices():
    m = _mixin()
    fake_cat = SimpleNamespace(
        list=lambda: [
            {"name": "devon", "backend": "xtts", "active": True,
             "description": "default"},
            {"name": "narrator", "backend": "piper", "active": False,
             "description": ""},
        ],
        chat_overrides={},
    )
    with patch("nomorals.voice.catalogue.default_catalogue",
               return_value=fake_cat):
        out = m._control_voices("", chat_key="tg:1")
        assert "devon" in out
        assert "narrator" in out
