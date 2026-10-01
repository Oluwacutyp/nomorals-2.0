"""Project scaffolding: render a template into a runnable project directory.

Rendering is plain stdlib: copy the template tree, then substitute
``$PROJECT_NAME`` in text files with :class:`string.Template`.  No jinja,
no new dependency.  Every rendered project passes its own test suite
unmodified.
"""

from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["KINDS", "ScaffoldResult", "scaffold", "template_dir"]

#: The template kinds this package ships.
KINDS = ("webapp", "bot", "cli_tool")

_TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"

#: Files that are never name-substituted (binary or already-final).
_SKIP_SUBSTITUTE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".gz"}


@dataclass
class ScaffoldResult:
    """What :func:`scaffold` produced."""

    kind: str
    name: str
    project_dir: Path
    files: list[str] = field(default_factory=list)
    #: Exact command that runs the rendered project's own test suite.
    test_cmd: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "project_dir": str(self.project_dir),
            "files": list(self.files),
            "test_cmd": list(self.test_cmd),
        }


def template_dir(kind: str) -> Path:
    """Absolute path of the template tree for ``kind``."""
    if kind not in KINDS:
        raise ToolError(f"unknown template kind {kind!r}; choose from {', '.join(KINDS)}")
    path = _TEMPLATES_ROOT / kind
    if not path.is_dir():
        raise ToolError(f"template {kind!r} is missing from the package ({path})")
    return path


def _render_text(path: Path, name: str) -> None:
    raw = path.read_text(encoding="utf-8")
    rendered = Template(raw).safe_substitute(PROJECT_NAME=name)
    if rendered != raw:
        path.write_text(rendered, encoding="utf-8")


def scaffold(kind: str, name: str, dest: str | Path) -> ScaffoldResult:
    """Render template ``kind`` into ``dest/name`` and return a :class:`ScaffoldResult`.

    Raises :class:`ToolError` on an unknown kind, a bad name, or an
    existing non-empty destination.
    """
    name = (name or "").strip()
    if not name:
        raise ToolError("project name must not be empty")
    if any(c in name for c in "/\\"):
        raise ToolError(f"project name must not contain path separators: {name!r}")
    src = template_dir(kind)

    project_dir = Path(dest).expanduser().resolve() / name
    if project_dir.exists() and any(project_dir.iterdir()):
        raise ToolError(f"destination {project_dir} already exists and is not empty")

    shutil.copytree(src, project_dir, dirs_exist_ok=True)
    manifest_path = project_dir / ".builders.json"
    manifest_path.write_text(
        json.dumps({"kind": kind, "name": name, "template": kind}, indent=2) + "\n",
        encoding="utf-8",
    )
    files: list[str] = []
    for path in sorted(project_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(project_dir).as_posix()
        files.append(rel)
        if path.suffix.lower() not in _SKIP_SUBSTITUTE_SUFFIXES:
            try:
                _render_text(path, name)
            except UnicodeDecodeError:
                _log.warning("skipping substitution in non-UTF8 file %s", rel)

    test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    _log.info("scaffolded %s %r -> %s (%d files)", kind, name, project_dir, len(files))
    return ScaffoldResult(kind=kind, name=name, project_dir=project_dir,
                          files=files, test_cmd=test_cmd)
