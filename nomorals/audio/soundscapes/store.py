"""Durable soundscape state — named mixes, hearing, last render.

One atomic JSON file, ``<home>/data/soundscapes.json``. Mix names
are case-insensitive ("Rainy Night" loads as "rainy night"); the
hearing profile is keyed by band centre in Hz as floats. A corrupt
file is treated as empty — the owner's mixes are never invented
from a broken read, and a bad write never leaves a half-written
file behind.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

__all__ = ["SoundscapeStore", "open_store"]

_FILE_NAME = "soundscapes.json"
_DEFAULT_DURATION_S = 60.0


def open_store(home: str | os.PathLike[str]) -> SoundscapeStore:
    """Open (creating if needed) the soundscape store for a home."""
    return SoundscapeStore(home)


class SoundscapeStore:
    """Atomic JSON store for mixes, hearing profile and last render."""

    def __init__(self, home: str | os.PathLike[str]) -> None:
        self.home = Path(home)
        self.path = self.home / "data" / _FILE_NAME
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                data.setdefault("mixes", {})
                data.setdefault("hearing", {})
                data.setdefault("last_render", None)
                return data
        except Exception:  # noqa: BLE001
            pass
        return {"mixes": {}, "hearing": {}, "last_render": None}

    def _save(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                       prefix=".soundscapes-", suffix=".tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(self._data, fh, indent=2, sort_keys=True)
                    fh.write("\n")
                os.replace(tmp, self.path)
            except Exception:  # noqa: BLE001
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
                raise
            return True
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _key(name: str) -> str:
        return (name or "").strip().lower()

    # -- mixes ---------------------------------------------------------

    def save_mix(self, name: str, generator: str,
                 sliders: Any, *, animate: bool = False,
                 duration_s: float = _DEFAULT_DURATION_S) -> bool:
        """Save (or overwrite) a named mix. Returns False on a bad name."""
        key = self._key(name)
        if not key:
            return False
        self._data.setdefault("mixes", {})[key] = {
            "name": (name or "").strip(),
            "generator": str(generator),
            "sliders": [float(v) for v in sliders],
            "animate": bool(animate),
            "duration_s": float(duration_s),
        }
        return self._save()

    def load_mix(self, name: str) -> dict[str, Any] | None:
        """Load a named mix (case-insensitive) or None."""
        mix = self._data.get("mixes", {}).get(self._key(name))
        return dict(mix) if isinstance(mix, dict) else None

    def delete_mix(self, name: str) -> bool:
        mixes = self._data.get("mixes", {})
        if self._key(name) not in mixes:
            return False
        del mixes[self._key(name)]
        return self._save()

    def list_mixes(self) -> list[str]:
        return sorted(m["name"] for m in self._data.get("mixes", {}).values()
                      if isinstance(m, dict) and m.get("name"))

    # -- hearing profile ------------------------------------------------

    def set_hearing(self, profile: dict[float, float]) -> bool:
        """Store the hearing profile, keyed by band centre in Hz."""
        self._data["hearing"] = {str(float(k)): float(v)
                                 for k, v in profile.items()}
        return self._save()

    def hearing(self) -> dict[float, float]:
        """The hearing profile, keyed by band centre in Hz."""
        hearing = self._data.get("hearing", {})
        if not isinstance(hearing, dict):
            return {}
        out: dict[float, float] = {}
        for key, value in hearing.items():
            try:
                out[float(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return out

    def set_hearing_band(self, band: float, gain_db: float) -> bool:
        profile = self.hearing()
        profile[float(band)] = float(gain_db)
        return self.set_hearing(profile)

    # -- last render ----------------------------------------------------

    def set_last(self, receipt: dict[str, Any]) -> bool:
        self._data["last_render"] = dict(receipt)
        return self._save()

    def last(self) -> dict[str, Any] | None:
        render = self._data.get("last_render")
        return dict(render) if isinstance(render, dict) else None
