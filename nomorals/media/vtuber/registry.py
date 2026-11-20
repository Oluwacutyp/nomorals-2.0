"""VTuber sessions — the loaded characters and acts, shared by the
tool, the CLI, and chat.

One process-level registry, the same shape as the show and rig
registries: a character or an act is loaded once under a name and
performed many times. Loading the same name twice replaces it;
driving an unknown name is refused by name.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..rig import default_rig
from .act import Act
from .character import Character
from .errors import VtuberError

__all__ = [
    "drop_act",
    "drop_character",
    "default_name",
    "get_act",
    "get_character",
    "list_acts",
    "list_characters",
    "load_act",
    "load_character",
    "reset_all",
]

_CHARACTERS: dict[str, Character] = {}
_ACTS: dict[str, Act] = {}


def default_name(path: str | Path) -> str:
    """The load name for a file: ``opener.vtact.json`` → ``opener``,
    ``devon.json`` → ``devon`` — the format suffix is not the
    name."""
    stem = Path(path).expanduser().name
    for suffix in (".vtact.json", ".json"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return Path(stem).stem


def load_character(name: str, path: str | Path) -> Character:
    """Load (or replace) a named character from a JSON file."""
    name = (name or "").strip() or default_name(path)
    if not name:
        raise VtuberError("character name must not be empty",
                          kind="bad_character")
    character = Character.from_file(path, default_rig())
    character.name = name
    _CHARACTERS[name] = character
    return character


def get_character(name: str = "default") -> Character:
    """The named character. An unknown name is refused by name."""
    name = (name or "").strip() or "default"
    character = _CHARACTERS.get(name)
    if character is None:
        raise VtuberError(
            f"no character {name!r} — load one first "
            f"(vtuber character <path>)",
            kind="unknown_character")
    return character


def list_characters() -> list[dict[str, Any]]:
    """Every loaded character, in load order."""
    return [character.to_dict()
            for character in _CHARACTERS.values()]


def drop_character(name: str) -> dict[str, Any]:
    """Drop a character."""
    character = get_character(name)
    del _CHARACTERS[character.name]
    return {"ok": True, "character": character.name}


def load_act(name: str, path: str | Path) -> Act:
    """Load (or replace) a named act from a .vtact.json file."""
    name = (name or "").strip() or default_name(path)
    if not name:
        raise VtuberError("act name must not be empty",
                          kind="bad_act")
    act = Act.from_file(path)
    act.name = name
    _ACTS[name] = act
    return act


def get_act(name: str = "default") -> Act:
    """The named act. An unknown name is refused by name."""
    name = (name or "").strip() or "default"
    act = _ACTS.get(name)
    if act is None:
        raise VtuberError(
            f"no act {name!r} — load one first "
            f"(vtuber act <path>)",
            kind="unknown_act")
    return act


def list_acts() -> list[dict[str, Any]]:
    """Every loaded act, in load order."""
    return [{"name": act.name, "character": act.character,
             "fps": act.fps, "beats": len(act.beats)}
            for act in _ACTS.values()]


def drop_act(name: str) -> dict[str, Any]:
    """Drop an act."""
    act = get_act(name)
    del _ACTS[act.name]
    return {"ok": True, "act": act.name}


def reset_all() -> None:
    """Drop every loaded character and act (tests)."""
    _CHARACTERS.clear()
    _ACTS.clear()
