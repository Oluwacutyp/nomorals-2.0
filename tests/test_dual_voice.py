"""Dual voice stack: XTTS private / Chatterbox public (build-map #30).

License-correct by audience, enforced structurally.
"""
import pytest
from unittest.mock import patch

from nomorals.voice.tts import UniversalTTS

FAKE_ORDER = ["chatterbox", "xtts", "piper", "kokoro"]


def _patched():
    return patch("nomorals.voice.tts.available_backends",
                 return_value=list(FAKE_ORDER))


def test_public_excludes_xtts_structurally():
    with _patched():
        got = UniversalTTS(audience="public")._audience_backends("public")
    assert "xtts" not in got
    assert got == ["chatterbox", "piper", "kokoro"]


def test_private_prefers_xtts():
    with _patched():
        got = UniversalTTS(audience="private")._audience_backends("private")
    assert got[0] == "xtts"


def test_private_without_xtts_installed():
    with patch("nomorals.voice.tts.available_backends",
               return_value=["chatterbox", "piper"]):
        got = UniversalTTS(audience="private")._audience_backends("private")
    assert got == ["chatterbox", "piper"]


def test_explicit_xtts_public_blocked():
    with _patched():
        eng = UniversalTTS(backend="xtts", audience="public")
        with pytest.raises(RuntimeError, match="non-commercial"):
            eng._load_backend()


def test_explicit_xtts_private_allowed():
    with _patched(), \
         patch("nomorals.voice.tts._spec", return_value=True), \
         patch.dict("nomorals.voice.tts._BACKENDS", {"xtts": lambda: object()}):
        eng = UniversalTTS(backend="xtts", audience="private")
        # _load_backend would construct the backend; just check no license error
        try:
            eng._load_backend()
        except RuntimeError as exc:
            assert "non-commercial" not in str(exc)


def test_invalid_audience_rejected():
    with pytest.raises(ValueError, match="audience"):
        UniversalTTS(audience="everyone")


def test_default_audience_private_backward_compat():
    assert UniversalTTS().audience == "private"


def test_speak_accepts_audience_override():
    with _patched():
        eng = UniversalTTS(audience="private")
        assert eng._audience_backends("public") != \
            eng._audience_backends("private")
