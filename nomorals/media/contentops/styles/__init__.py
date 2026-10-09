"""Edit style presets — each composes engine primitives into a look.

A preset is a module with ``build_spec(**content) -> EditSpec`` where
``content`` is plain data (clips/scenes, beats, voiceover, music,
captions, …). No preset may reach past the engine's public API.

* ``phonk`` — beat-synced cuts, motion effects, karaoke captions,
  ducked music bed (the short-form edit grammar).
* ``documentary`` — slow ken-burns, dissolves, lower-thirds, calm mix.
* ``vlog`` — jump cuts, pop zooms, bold center captions.
* ``minimal`` — clean cuts, simple captions, no effects.
"""

from __future__ import annotations

from typing import Any

from ....media_edit.videos import MediaEditError

_PRESETS: dict[str, str] = {
    "phonk": ".phonk",
    "documentary": ".documentary",
    "vlog": ".vlog",
    "minimal": ".minimal",
}


def list_styles() -> list[str]:
    """All registered style preset names."""
    return sorted(_PRESETS)


def get_style(name: str) -> Any:
    """The preset module for ``name`` (raises on unknown)."""
    key = str(name or "").strip().lower()
    mod_path = _PRESETS.get(key)
    if mod_path is None:
        raise MediaEditError(
            f"unknown edit style {name!r}; use: {list_styles()}")
    import importlib
    return importlib.import_module(mod_path, package=__name__)


def build_spec(style: str, **content: Any) -> Any:
    """``get_style(style).build_spec(**content)`` → engine EditSpec."""
    return get_style(style).build_spec(**content)


def register_style(name: str, module_path: str) -> None:
    """Register an additional preset (dotted path, relative ok)."""
    key = str(name).strip().lower()
    if not key or not module_path:
        raise MediaEditError("register_style needs a name and module path")
    _PRESETS[key] = module_path
