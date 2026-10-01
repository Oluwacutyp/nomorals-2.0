"""Devon's voice catalogue — named voices, switchable at runtime.

A catalogue entry binds a :class:`VoiceLibrary` profile to a preferred
backend plus a human description. The catalogue keeps:

- an **active** voice (the global default), and
- **per-chat overrides** (``chat_key → voice name``), so ``/voice use``
  in one Telegram chat doesn't change the voice in another — and
  switching works live from Telegram, WhatsApp, console, anywhere the
  runtime routes control commands.

Persisted as ``catalogue.json`` next to the voice profiles. Stdlib only.

Storage: the catalogue is the canonical voice store, shared by the CLI
and the runtime (Telegram/WhatsApp/console all use
:func:`default_catalogue`). It lives at ``~/.config/nomorals/voices`` —
user-level and transport-independent — or wherever
``NOMORALS_VOICES_DIR`` points. The older tool-layer
``<workspace>/audio/voices`` dir (``tts_register_voice``) is a legacy
profile store; the catalogue's ``VoiceLibrary`` is the same class, so
copying a legacy ``*.wav`` + index entry into the catalogue dir adopts
it. One catalogue, no parallel indexes.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.logging_setup import get_logger
from .tts import UniversalTTS, VoiceLibrary

__all__ = ["CatalogueVoice", "VoiceCatalogue", "default_catalogue"]

_log = get_logger(__name__)

_VALID_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,47}$")


@dataclass
class CatalogueVoice:
    """One named voice in the catalogue."""

    name: str
    backend: str = "auto"            # preferred TTS backend
    profile: str = ""                # VoiceLibrary profile name ("" = none)
    description: str = ""
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "backend": self.backend,
                "profile": self.profile, "description": self.description,
                "tags": list(self.tags)}

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CatalogueVoice":
        return cls(name=raw.get("name", ""),
                   backend=raw.get("backend", "auto") or "auto",
                   profile=raw.get("profile", "") or "",
                   description=raw.get("description", "") or "",
                   tags=tuple(raw.get("tags", []) or []))


def _default_voices_dir() -> str:
    """Canonical catalogue store: env override, else user-level default."""
    override = os.environ.get("NOMORALS_VOICES_DIR", "").strip()
    if override:
        return os.path.expanduser(override)
    return os.path.join(os.path.expanduser("~"), ".config",
                        "nomorals", "voices")


class VoiceCatalogue:
    """Named voices with a runtime-switchable active voice."""

    def __init__(self, voices_dir: str = "") -> None:
        self.voices_dir = voices_dir or _default_voices_dir()
        self.library = VoiceLibrary(self.voices_dir)
        self.voices: dict[str, CatalogueVoice] = {}
        self.active: str = ""
        self.chat_overrides: dict[str, str] = {}
        self._load()

    # -- persistence -------------------------------------------------

    def _path(self) -> str:
        return os.path.join(self.voices_dir, "catalogue.json")

    def _load(self) -> None:
        try:
            with open(self._path(), encoding="utf-8") as fh:
                raw = json.load(fh) or {}
        except (OSError, ValueError):
            raw = {}
        self.voices = {v["name"]: CatalogueVoice.from_dict(v)
                       for v in raw.get("voices", []) if v.get("name")}
        self.active = raw.get("active", "") or ""
        self.chat_overrides = dict(raw.get("chat_overrides", {}) or {})

    def _save(self) -> None:
        try:
            with open(self._path(), "w", encoding="utf-8") as fh:
                json.dump({"voices": [v.to_dict()
                                       for v in self.voices.values()],
                           "active": self.active,
                           "chat_overrides": self.chat_overrides},
                          fh, indent=2)
        except OSError as e:
            _log.warning("voice catalogue save failed (%s): %s", self._path(), e)

    # -- catalogue management ----------------------------------------

    @staticmethod
    def _check_name(name: str) -> str:
        name = (name or "").strip()
        if not _VALID_NAME_RE.match(name):
            raise ValueError(
                "voice name must be 1–48 chars: letters, digits, space, _ -")
        return name

    def add(self, name: str, *, backend: str = "auto", profile: str = "",
            description: str = "", tags: tuple[str, ...] = ()) -> CatalogueVoice:
        """Register (or replace) a catalogue voice."""
        name = self._check_name(name)
        voice = CatalogueVoice(name=name, backend=backend or "auto",
                               profile=profile or "",
                               description=description or "",
                               tags=tuple(tags or ()))
        self.voices[name] = voice
        if not self.active:
            self.active = name
        self._save()
        return voice

    def remove(self, name: str) -> bool:
        if name not in self.voices:
            return False
        del self.voices[name]
        if self.active == name:
            self.active = next(iter(self.voices), "")
        self.chat_overrides = {k: v for k, v in self.chat_overrides.items()
                               if v != name}
        self._save()
        return True

    def get(self, name: str) -> Optional[CatalogueVoice]:
        return self.voices.get(name)

    def list(self) -> list[dict[str, Any]]:
        return [{"name": v.name, "backend": v.backend,
                 "profile": v.profile, "description": v.description,
                 "tags": list(v.tags),
                 "active": v.name == self.active}
                for v in self.voices.values()]

    # -- runtime switching --------------------------------------------

    def set_active(self, name: str) -> CatalogueVoice:
        """Switch the global active voice. Returns the voice."""
        if name not in self.voices:
            raise KeyError(f"unknown voice {name!r}")
        self.active = name
        self._save()
        return self.voices[name]

    def set_chat_voice(self, chat_key: str, name: str) -> CatalogueVoice:
        """Switch the voice for one chat only (runtime override)."""
        if name not in self.voices:
            raise KeyError(f"unknown voice {name!r}")
        self.chat_overrides[chat_key] = name
        self._save()
        return self.voices[name]

    def clear_chat_voice(self, chat_key: str) -> bool:
        if chat_key in self.chat_overrides:
            del self.chat_overrides[chat_key]
            self._save()
            return True
        return False

    def active_for_chat(self, chat_key: str = "") -> Optional[CatalogueVoice]:
        """Chat override wins, then the global active voice."""
        name = (self.chat_overrides.get(chat_key or "")
                or self.active)
        return self.voices.get(name)

    def resolve(self, chat_key: str = "") -> tuple[str, str]:
        """``(profile_name_or_empty, backend)`` for a chat."""
        voice = self.active_for_chat(chat_key)
        if voice is None:
            return "", "auto"
        return voice.profile, voice.backend

    # -- speaking ------------------------------------------------------

    def speak(self, text: str, *, chat_key: str = "",
              out_path: str = "", mood: str = "neutral", intensity: int = 3,
              seed: Optional[int] = None, effect: Optional[str] = None,
              perform: bool = True) -> dict:
        """Speak ``text`` in the chat's active catalogue voice.

        Resolves the voice (chat override → global active), builds a
        ``UniversalTTS`` on its preferred backend, and runs the full
        director performance. Returns the engine result dict.
        """
        profile_name, backend = self.resolve(chat_key)
        engine = UniversalTTS(backend=backend, voices_dir=self.voices_dir)
        if perform:
            return engine.perform(text, voice_name=profile_name or None,
                                  out_path=out_path, mood=mood,
                                  intensity=intensity, seed=seed,
                                  effect=effect)
        return engine.speak(text, voice_name=profile_name or None,
                            out_path=out_path)

    # -- cloning ---------------------------------------------------------

    def clone(self, name: str, audio_path: str, *,
              transcript: str = "", language: str = "en",
              backend: str = "auto", description: str = "",
              consent_confirmed: bool = False) -> CatalogueVoice:
        """Clone a voice from a reference clip into the catalogue.

        Registers the clip in the VoiceLibrary (copied into managed
        storage) and adds a catalogue entry pointing at it. The consent
        gate lives in :class:`VoiceProfile.validate_for_cloning` —
        cloning backends refuse to run without
        ``consent_confirmed=True``; set it True only for your own voice
        or with explicit permission.
        """
        name = self._check_name(name)
        profile = self.library.upload_voice(
            name, audio_path, language=language,
            consent_confirmed=consent_confirmed, description=description)
        if transcript.strip():
            profile.prompt_text = transcript.strip()
            self.library._save_index()
        return self.add(name, backend=backend, profile=name,
                        description=description or f"cloned voice '{name}'",
                        tags=("cloned",))


def default_catalogue(voices_dir: str = "") -> VoiceCatalogue:
    """The shared catalogue (same voices dir the CLI uses)."""
    return VoiceCatalogue(voices_dir or _default_voices_dir())
