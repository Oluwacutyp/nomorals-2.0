"""Project config — the ``Config/*.ini`` analogue.

StreamTwo's ``Config/`` holds ``DefaultEngine.ini`` (hardware
targeting, engine redirects, GameMapsSettings with the startup
and default maps), ``DefaultGame.ini`` (ProjectID),
``DefaultEditor.ini`` and ``DefaultInput.ini`` (axis/action
mappings). Devon's project config is one file,
``Config/DefaultProject.ini``:

* ``[Render]`` — width / height / fps, the render hints a scene
  carries in its ``output``.
* ``[Input]`` — action/axis mappings, name → key, the
  ``DefaultInput.ini`` analogue.

The manifest is the identity source of truth (name, modules,
plugins, entries, default scene); the INI holds runtime
settings only, so the two can never disagree about what the
project IS. A missing INI is a validation problem, never a
silent default.
"""

from __future__ import annotations

import configparser
from pathlib import Path
from typing import Any

from .errors import ProjectError

__all__ = ["CONFIG_FILENAME", "ProjectConfig", "read_config",
           "write_config"]

CONFIG_FILENAME = "DefaultProject.ini"

_DEFAULTS: dict[str, dict[str, str]] = {
    "Render": {"width": "1920", "height": "1080", "fps": "30"},
    "Input": {},
}


class ProjectConfig:
    """The runtime settings of a project, from its INI file."""

    def __init__(self, render: dict[str, Any] | None = None,
                 inputs: dict[str, str] | None = None):
        self.render = {
            "width": int((render or {}).get("width", 1920)),
            "height": int((render or {}).get("height", 1080)),
            "fps": int((render or {}).get("fps", 30)),
        }
        if self.render["width"] <= 0 or self.render["height"] <= 0 \
                or self.render["fps"] <= 0:
            raise ProjectError(
                "render width/height/fps must all be positive",
                kind="bad_config")
        self.inputs = dict(inputs or {})

    def to_dict(self) -> dict[str, Any]:
        """The config as plain data."""
        return {"render": dict(self.render),
                "inputs": dict(self.inputs)}

    def set_input(self, action: str, key: str) -> None:
        """Bind an action to a key. An empty action or key is
        refused by name."""
        action = (action or "").strip()
        key = (key or "").strip()
        if not action:
            raise ProjectError(
                "an input mapping needs a non-empty action",
                kind="bad_config")
        if not key:
            raise ProjectError(
                f"input action {action!r} needs a non-empty key",
                kind="bad_config")
        self.inputs[action] = key


def read_config(path: str | Path) -> ProjectConfig:
    """Read a ``DefaultProject.ini``. A missing file or a
    malformed value is refused by name."""
    target = Path(path)
    if not target.exists():
        raise ProjectError(
            f"no project config at {target}",
            kind="missing_file")
    parser = configparser.ConfigParser()
    parser.optionxform = str  # keep action names case-sensitive
    try:
        parser.read(target, encoding="utf-8")
    except (configparser.Error, OSError) as exc:
        raise ProjectError(
            f"project config at {target} is unreadable: {exc}",
            kind="bad_config") from exc
    render: dict[str, Any] = {}
    if parser.has_section("Render"):
        for key in ("width", "height", "fps"):
            if parser.has_option("Render", key):
                raw = parser.get("Render", key)
                try:
                    render[key] = int(raw)
                except ValueError as exc:
                    raise ProjectError(
                        f"project config Render.{key} must be an "
                        f"integer, got {raw!r}",
                        kind="bad_config") from exc
    inputs: dict[str, str] = {}
    if parser.has_section("Input"):
        inputs = dict(parser.items("Input"))
    try:
        return ProjectConfig(render=render, inputs=inputs)
    except ProjectError:
        raise
    except (TypeError, ValueError) as exc:
        raise ProjectError(
            f"project config at {target} has a bad value: {exc}",
            kind="bad_config") from exc


def write_config(path: str | Path, config: ProjectConfig) -> str:
    """Write a ``DefaultProject.ini``. The parent directory is
    created; the write is atomic."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser["Render"] = {k: str(v)
                        for k, v in config.render.items()}
    parser["Input"] = dict(config.inputs)
    text = ("; Devon project config — runtime settings only.\n"
            "; Identity (name, modules, plugins, entries, default\n"
            "; scene) lives in the .dvproject manifest.\n"
            + _render_ini(parser))
    tmp = target.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(target)
    return str(target)


def _render_ini(parser: configparser.ConfigParser) -> str:
    """configparser output, without the module docstring header."""
    import io

    buf = io.StringIO()
    parser.write(buf)
    return buf.getvalue()
