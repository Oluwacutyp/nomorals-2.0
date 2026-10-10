"""Project scaffolding: render a template into a runnable project directory.

Rendering is plain stdlib: copy the template tree, then substitute
template variables in text files with :class:`string.Template`
(Cookiecutter-style multi-variable prompts, no jinja, no new
dependency).  Every rendered project passes its own test suite
unmodified.

Templates form a browsable catalog (:func:`list_templates`,
:func:`describe`), support ``dry_run`` rehearsal, ``overwrite``, and an
optional ``git_init`` post-hook (``git init`` + initial commit —
cookiecutter's ``post_gen_project`` gold).
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

from ..core.errors import ToolError
from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "KINDS", "ScaffoldResult", "TemplateInfo",
    "scaffold", "template_dir", "list_templates", "describe",
]

#: The template kinds this package ships.
KINDS = ("webapp", "bot", "cli_tool", "rest_api", "telegram_bot", "dashboard")

#: Human catalog metadata per template kind (Backstage-catalog style).
_TEMPLATE_META: dict[str, dict[str, str]] = {
    "webapp": {
        "title": "Web app",
        "description": "Stdlib-only HTTP app: HTML landing page, JSON "
                       "/api/health, POST /api/echo. No third-party deps.",
    },
    "bot": {
        "title": "Chat bot",
        "description": "Command-dispatch chat bot skeleton with a test suite.",
    },
    "cli_tool": {
        "title": "CLI tool",
        "description": "Argparse CLI with subcommands and a JSON data file.",
    },
    "rest_api": {
        "title": "REST API",
        "description": "JSON REST API service template with tests.",
    },
    "telegram_bot": {
        "title": "Telegram bot",
        "description": "Telegram bot skeleton (python-telegram-bot style).",
    },
    "dashboard": {
        "title": "Dashboard",
        "description": "Data dashboard app rendering data.json.",
    },
}

_TEMPLATES_ROOT = Path(__file__).resolve().parent / "templates"

#: Files that are never name-substituted (binary or already-final).
_SKIP_SUBSTITUTE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".zip", ".gz"}


@dataclass
class TemplateInfo:
    """Catalog entry for one template kind."""

    kind: str
    title: str
    description: str
    path: Path
    files: list[str] = field(default_factory=list)
    #: Variables the template files reference (detected, $-style).
    variables: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "title": self.title,
            "description": self.description, "path": str(self.path),
            "files": list(self.files), "variables": list(self.variables),
            "file_count": len(self.files),
        }


@dataclass
class ScaffoldResult:
    """What :func:`scaffold` produced."""

    kind: str
    name: str
    project_dir: Path
    files: list[str] = field(default_factory=list)
    #: Exact command that runs the rendered project's own test suite.
    test_cmd: list[str] = field(default_factory=list)
    #: Variables the render used (builtins merged over user-supplied).
    variables: dict[str, str] = field(default_factory=dict)
    #: True when dry_run rehearsed the render without writing.
    dry_run: bool = False
    #: True when git_init ran `git init` + an initial commit.
    git_initialized: bool = False
    #: Non-fatal git hook output / error (None when git_init not requested).
    git_note: str = ""

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "name": self.name,
            "project_dir": str(self.project_dir),
            "files": list(self.files),
            "test_cmd": list(self.test_cmd),
            "variables": dict(self.variables),
            "dry_run": self.dry_run,
            "git_initialized": self.git_initialized,
            "git_note": self.git_note,
        }


def template_dir(kind: str) -> Path:
    """Absolute path of the template tree for ``kind``."""
    if kind not in KINDS:
        raise ToolError(f"unknown template kind {kind!r}; choose from {', '.join(KINDS)}")
    path = _TEMPLATES_ROOT / kind
    if not path.is_dir():
        raise ToolError(f"template {kind!r} is missing from the package ({path})")
    return path


def _builtin_variables(name: str, variables: dict[str, str] | None) -> dict[str, str]:
    """Builtin template variables, merged under user-supplied ones.

    Cookiecutter-style: ``PROJECT_NAME`` plus the derived values real
    templates reach for — slug, author, year, date, description.
    """
    from ..core.text import slugify

    now = datetime.datetime.now()
    builtins = {
        "PROJECT_NAME": name,
        "PROJECT_SLUG": slugify(name, fallback="app"),
        "AUTHOR": os.environ.get("USER") or os.environ.get("USERNAME") or "devon",
        "YEAR": now.strftime("%Y"),
        "DATE": now.strftime("%Y-%m-%d"),
        "DESCRIPTION": f"{name} — scaffolded by the No-Morals builder.",
    }
    merged = dict(builtins)
    for key, value in (variables or {}).items():
        merged[str(key).upper()] = str(value)
    return merged


def _render_text(path: Path, variables: dict[str, str]) -> None:
    raw = path.read_text(encoding="utf-8")
    rendered = Template(raw).safe_substitute(variables)
    if rendered != raw:
        path.write_text(rendered, encoding="utf-8")


def _detect_variables(path: Path) -> set[str]:
    """$-style identifiers referenced in a template tree (catalog use)."""
    found: set[str] = set()
    for file in sorted(path.rglob("*")):
        if not file.is_file() or file.suffix.lower() in _SKIP_SUBSTITUTE_SUFFIXES:
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for match in Template.pattern.finditer(text):
            name = match.group("named") or match.group("braced")
            if name:
                found.add(name)
    return found


def list_templates() -> list[TemplateInfo]:
    """Catalog of every bundled template (Backstage-catalog style)."""
    return [describe(kind) for kind in KINDS]


def describe(kind: str) -> TemplateInfo:
    """Rich metadata for one template kind: files, variables, description."""
    path = template_dir(kind)
    files = sorted(
        p.relative_to(path).as_posix()
        for p in path.rglob("*") if p.is_file()
    )
    meta = _TEMPLATE_META.get(kind, {})
    return TemplateInfo(
        kind=kind,
        title=meta.get("title", kind),
        description=meta.get("description", ""),
        path=path,
        files=files,
        variables=sorted(_detect_variables(path)),
    )


def _git_init_hook(project_dir: Path, name: str) -> tuple[bool, str]:
    """post_gen_project gold: git init + initial commit. Never raises."""
    git = shutil.which("git")
    if not git:
        return False, "git not found on PATH — skipping git init"
    try:
        subprocess.run([git, "init", "-q"], cwd=str(project_dir),
                       capture_output=True, timeout=30, check=True)
        subprocess.run([git, "add", "-A"], cwd=str(project_dir),
                       capture_output=True, timeout=30, check=True)
        subprocess.run(
            [git, "-c", "user.name=devon", "-c",
             "user.email=devon@localhost", "commit", "-qm",
             f"scaffold {name}"],
            cwd=str(project_dir), capture_output=True, timeout=60,
            check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            OSError) as exc:
        return False, f"git init hook failed (non-fatal): {exc}"
    return True, "git init + initial commit ok"


def scaffold(kind: str, name: str, dest: str | Path, *,
             variables: dict[str, str] | None = None,
             overwrite: bool = False,
             dry_run: bool = False,
             git_init: bool = False) -> ScaffoldResult:
    """Render template ``kind`` into ``dest/name`` and return a :class:`ScaffoldResult`.

    * ``variables`` — extra template variables merged over the builtins
      (``PROJECT_NAME``, ``PROJECT_SLUG``, ``AUTHOR``, ``YEAR``,
      ``DATE``, ``DESCRIPTION``).
    * ``overwrite`` — replace a non-empty destination instead of raising.
    * ``dry_run`` — render into a temp dir and report the file list;
      the destination is never touched.
    * ``git_init`` — run the post-gen hook: ``git init`` + initial
      commit (non-fatal; recorded in the result).

    Raises :class:`ToolError` on an unknown kind, a bad name, or an
    existing non-empty destination (without ``overwrite=True``).
    """
    name = (name or "").strip()
    if not name:
        raise ToolError("project name must not be empty")
    if any(c in name for c in "/\\"):
        raise ToolError(f"project name must not contain path separators: {name!r}")
    src = template_dir(kind)
    varmap = _builtin_variables(name, variables)

    project_dir = Path(dest).expanduser().resolve() / name
    if project_dir.exists() and any(project_dir.iterdir()):
        if not overwrite:
            raise ToolError(f"destination {project_dir} already exists and is not empty "
                            "(pass overwrite=True to replace it)")
        shutil.rmtree(project_dir)

    render_root = Path(tempfile.mkdtemp(prefix="nm-scaffold-dryrun-")) \
        if dry_run else project_dir
    target = render_root if dry_run else project_dir
    try:
        shutil.copytree(src, target, dirs_exist_ok=True)
        manifest_path = target / ".builders.json"
        manifest_path.write_text(
            json.dumps({"kind": kind, "name": name, "template": kind,
                        "variables": varmap}, indent=2) + "\n",
            encoding="utf-8",
        )
        files: list[str] = []
        for path in sorted(target.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(target).as_posix()
            files.append(rel)
            if path.suffix.lower() not in _SKIP_SUBSTITUTE_SUFFIXES:
                try:
                    _render_text(path, varmap)
                except UnicodeDecodeError:
                    _log.warning("skipping substitution in non-UTF8 file %s", rel)
    finally:
        if dry_run:
            shutil.rmtree(render_root, ignore_errors=True)

    test_cmd = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."]
    result = ScaffoldResult(kind=kind, name=name, project_dir=project_dir,
                            files=files, test_cmd=test_cmd,
                            variables=varmap, dry_run=dry_run)
    if git_init and not dry_run:
        ok, note = _git_init_hook(project_dir, name)
        result.git_initialized = ok
        result.git_note = note
    _log.info("scaffolded %s %r -> %s (%d files)%s%s", kind, name,
              project_dir, len(files),
              " [dry-run]" if dry_run else "",
              " [git init]" if result.git_initialized else "")
    return result
