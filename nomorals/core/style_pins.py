"""Owner style pins (Living Chat P3, 2026-11-08).

The owner can PIN a style — a typography intent, a font family,
a theme pack / preset, palette traits — globally and per
surface family. A pin is the owner's standing answer to "what
should this look like": it outranks the semantic defaults at
selection time, it persists across restarts, and unsetting it
restores the default exactly.

One store, one resolver — the ONE seam every selection consumer
consults:

* :meth:`nomorals.media.typography.TypographyService.select`
  applies the ``intent`` / ``font_family`` fields;
* :func:`nomorals.social.chat.typography.resolve_style`
  applies the ``theme`` / ``preset`` fields;
* living cards name the pin in force for their family on the
  rendered card itself.

Scopes are consumer-named strings: ``""`` (also spelled
``"global"``) is the global pin; any other scope is a surface
family — a living-card family (``"creation"``, ``"progress"``),
a theme surface class (``"chat"``, ``"briefing"``), or a
``surface_family`` a typography caller declares (``"posters"``).
Resolution merges field-by-field: the family pin wins the
fields it sets, the global pin fills the rest. The merged
result names every scope that contributed, so "a pin did this"
is always attributable.

Persistence is a JSON file under the Devon home
(``style_pins.json``) written atomically; a store bound to
``path=None`` is in-memory only (tests, ephemeral contexts).
This module never raises on the read path: a corrupt pin file
reads as no pins, never as a broken selection.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "GLOBAL_SCOPE",
    "PIN_FIELDS",
    "StylePin",
    "PinStore",
    "effective_pin",
    "get_pin_store",
    "set_pin_store",
]

#: The scope key for the global pin ("" and "global" both name it).
GLOBAL_SCOPE = ""

#: The structured fields a pin may carry (``traits`` is the
#: free-form palette/theme-trait bag beside them).
PIN_FIELDS = ("intent", "font_family", "theme", "preset")


def _norm_scope(scope: Any) -> str:
    scope = str(scope or "").strip().lower()
    return "" if scope in ("", "global") else scope


@dataclass
class StylePin:
    """One scope's pinned style fields ("" = unset)."""

    intent: str = ""
    font_family: str = ""
    theme: str = ""
    preset: str = ""
    traits: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"intent": self.intent,
                "font_family": self.font_family,
                "theme": self.theme, "preset": self.preset,
                "traits": dict(self.traits)}

    @classmethod
    def from_dict(cls, data: Any) -> StylePin:
        if not isinstance(data, dict):
            return cls()
        traits = data.get("traits") or {}
        return cls(
            intent=str(data.get("intent", "") or ""),
            font_family=str(data.get("font_family", "") or ""),
            theme=str(data.get("theme", "") or ""),
            preset=str(data.get("preset", "") or ""),
            traits={str(k): str(v) for k, v in traits.items()
                    if str(k).strip()} if isinstance(traits, dict)
            else {})

    def empty(self) -> bool:
        return not (self.intent or self.font_family or self.theme
                    or self.preset or self.traits)


@dataclass(frozen=True)
class EffectivePin:
    """A merged pin plus its attribution: the scopes that
    contributed, in force order (global first, family last)."""

    intent: str = ""
    font_family: str = ""
    theme: str = ""
    preset: str = ""
    traits: dict[str, str] = field(default_factory=dict)
    scopes: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """The human attribution: "global + family:creation"."""
        parts = ["global" if s == "" else f"family:{s}"
                 for s in self.scopes]
        return " + ".join(parts)

    def describe(self) -> str:
        """The pinned fields as one line, for cards and replies."""
        bits = []
        if self.intent:
            bits.append(f"intent {self.intent}")
        if self.font_family:
            bits.append(f"font {self.font_family}")
        if self.theme:
            bits.append(f"theme {self.theme}")
        if self.preset:
            bits.append(f"preset {self.preset}")
        for key in sorted(self.traits):
            bits.append(f"{key} {self.traits[key]}")
        return ", ".join(bits)


class PinStore:
    """Durable style pins: a JSON file, atomically rewritten.

    ``path=None`` keeps the store in memory only. Every write
    persists the whole pin set; reads after a write in another
    process see the file's state at construction (the store is
    a settings surface, not a coordination primitive).
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = Path(path) if path else None
        self._lock = threading.Lock()
        self._pins: dict[str, StylePin] = {}
        if self._path is not None:
            self._pins = self._load(self._path)

    @staticmethod
    def _load(path: Path) -> dict[str, StylePin]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - corrupt reads as no pins
            return {}
        if not isinstance(raw, dict):
            return {}
        return {_norm_scope(scope): StylePin.from_dict(pin)
                for scope, pin in raw.items()
                if not StylePin.from_dict(pin).empty()}

    def _save(self) -> None:
        if self._path is None:
            return
        data = {scope or "global": pin.to_dict()
                for scope, pin in self._pins.items()
                if not pin.empty()}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=1, sort_keys=True),
                           encoding="utf-8")
            os.replace(tmp, self._path)
        except Exception:  # noqa: BLE001 - a failed save keeps memory
            pass

    # ── reads ───────────────────────────────────────────────

    def get(self, scope: Any) -> StylePin:
        """One scope's own pin (unmerged); empty when unset."""
        with self._lock:
            pin = self._pins.get(_norm_scope(scope))
            return StylePin.from_dict(pin.to_dict()) if pin \
                else StylePin()

    def scopes(self) -> list[str]:
        """Every scope carrying a pin, global ("") first."""
        with self._lock:
            return sorted(self._pins, key=lambda s: (s != "", s))

    # ── writes ──────────────────────────────────────────────

    def set(self, scope: Any, *, intent: str | None = None,
            font_family: str | None = None, theme: str | None = None,
            preset: str | None = None,
            trait: dict[str, str] | None = None) -> StylePin:
        """Set fields on one scope's pin (None leaves a field)."""
        key = _norm_scope(scope)
        with self._lock:
            pin = self._pins.get(key) or StylePin()
            if intent is not None:
                pin.intent = str(intent).strip()
            if font_family is not None:
                pin.font_family = str(font_family).strip()
            if theme is not None:
                pin.theme = str(theme).strip().lower()
            if preset is not None:
                pin.preset = str(preset).strip().lower()
            if trait:
                pin.traits.update({str(k).strip(): str(v).strip()
                                   for k, v in trait.items()
                                   if str(k).strip()})
            if pin.empty():
                self._pins.pop(key, None)
            else:
                self._pins[key] = pin
            self._save()
            return StylePin.from_dict(pin.to_dict())

    def clear(self, scope: Any, field: str | None = None) -> None:
        """Unset one scope's pin — or one field of it (a trait
        name clears that trait). Unsetting restores the default:
        resolution simply stops seeing the field."""
        key = _norm_scope(scope)
        with self._lock:
            pin = self._pins.get(key)
            if pin is None:
                return
            if field is None:
                self._pins.pop(key, None)
            elif field in PIN_FIELDS:
                setattr(pin, field, "")
                if pin.empty():
                    self._pins.pop(key, None)
            else:
                pin.traits.pop(str(field), None)
                if pin.empty():
                    self._pins.pop(key, None)
            self._save()


def effective_pin(store: PinStore, family: Any) -> EffectivePin | None:
    """The pin in force for ``family``: global merged under the
    family's own pin, family winning field-by-field. None when
    no scope contributes any field."""
    fam = _norm_scope(family)
    merged = StylePin()
    scopes: list[str] = []
    for scope in ("", fam) if fam else ("",):
        pin = store.get(scope)
        if pin.empty():
            continue
        scopes.append(scope)
        for name in PIN_FIELDS:
            value = getattr(pin, name)
            if value:
                setattr(merged, name, value)
        merged.traits.update(pin.traits)
    if merged.empty():
        return None
    return EffectivePin(
        intent=merged.intent, font_family=merged.font_family,
        theme=merged.theme, preset=merged.preset,
        traits=dict(merged.traits), scopes=tuple(scopes))


# ── the process-wide store ──────────────────────────────────────


def _default_path() -> Path:
    home = os.environ.get("NM_HOME", "") or str(Path.home() / ".nomorals")
    return Path(home).expanduser() / "style_pins.json"


_STORE: PinStore | None = None
_STORE_LOCK = threading.Lock()


def get_pin_store() -> PinStore:
    """The process-wide pin store (the runtime and the tests
    may replace it via :func:`set_pin_store`)."""
    global _STORE
    with _STORE_LOCK:
        if _STORE is None:
            _STORE = PinStore(path=_default_path())
        return _STORE


def set_pin_store(store: PinStore | None) -> PinStore | None:
    """Replace the process-wide store; returns the previous one."""
    global _STORE
    with _STORE_LOCK:
        previous = _STORE
        _STORE = store
        return previous
