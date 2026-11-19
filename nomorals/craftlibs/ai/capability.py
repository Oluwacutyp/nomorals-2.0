"""Capabilities (Craft-Libs port of ai-core ``capability.rs``).

A capability is ``<modality>.<task>`` — ``image.inpaint``,
``text.complete`` — where the modality is an open set: the six
known ones below, plus anything else a provider reports, so a new
modality is data, never a code change.
"""

from __future__ import annotations

import contextlib

from .ids import validate_token

__all__ = ["CapabilityId", "Modality", "WELL_KNOWN", "well_known"]

_KNOWN_MODALITIES = ("text", "image", "audio", "video", "document",
                     "mesh")

WELL_KNOWN = [
    "text.complete",
    "text.embed",
    "image.generate",
    "image.detect",
    "image.edit",
    "image.inpaint",
    "image.outpaint",
    "image.segment",
    "image.matte",
    "image.upscale",
    "image.vectorize",
    "audio.transcribe",
    "audio.generate",
    "audio.separate",
    "audio.enhance",
    "video.generate",
    "video.segment",
    "video.upscale",
    "document.ocr",
    "document.extract",
    "mesh.generate",
]


class Modality:
    """A capability's modality: one of the known ones, or
    ``Other`` carrying whatever a provider reported."""

    _KNOWN = frozenset(_KNOWN_MODALITIES)

    def __init__(self, name: str) -> None:
        self._name = name

    @classmethod
    def parse(cls, name: str) -> Modality:
        return cls(name)

    def as_str(self) -> str:
        return self._name

    @property
    def value(self) -> str:
        return self._name

    def is_known(self) -> bool:
        return self._name in self._KNOWN

    def __eq__(self, other) -> bool:
        return isinstance(other, Modality) and self._name == other._name

    def __hash__(self):
        return hash(self._name)

    def __repr__(self) -> str:
        return f"Modality({self._name!r})"


class CapabilityId:
    """``<modality>.<task>``. Both halves are tokens (no dot)."""

    def __init__(self, value: str) -> None:
        self._value = value

    @classmethod
    def parse(cls, text: str) -> CapabilityId:
        parts = str(text).split(".")
        if len(parts) != 2:
            raise ValueError(
                f"a capability is '<modality>.<task>', not {text!r}")
        validate_token(parts[0])
        validate_token(parts[1])
        return cls(str(text))

    def modality(self) -> Modality:
        return Modality(self._value.split(".", 1)[0])

    def task(self) -> str:
        return self._value.split(".", 1)[1]

    def as_str(self) -> str:
        return self._value

    def __str__(self) -> str:
        return self._value

    def __eq__(self, other) -> bool:
        return isinstance(other, CapabilityId) and self._value == other._value

    def __hash__(self):
        return hash(self._value)

    def __repr__(self) -> str:
        return f"CapabilityId({self._value!r})"


def well_known() -> list[CapabilityId]:
    """The well-known capabilities, as parsed ids."""
    out = []
    for text in WELL_KNOWN:
        with contextlib.suppress(ValueError):
            out.append(CapabilityId.parse(text))
    return out
