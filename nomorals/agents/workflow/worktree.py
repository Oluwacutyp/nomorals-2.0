"""Isolated git workspaces for workflow agents.

A thin adapter over the proven commands in
``nomorals/agents/bench/staging.py`` (same calls, workflow-owned
dirs): ``git worktree add --detach`` first, ``git clone
--no-hardlinks`` + checkout as the fallback when worktrees fail
(which the bench docstring notes is unproven on the EC2 Windows
host). No new git machinery is invented here.

The ``git`` parameter is the command runner (a callable taking the
full argv list and returning an object with ``returncode``) —
``subprocess.run`` by default, injectable in tests.
"""

from __future__ import annotations

import contextlib
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import WorkflowError

__all__ = ["create_worktree", "remove_worktree"]


def _run(argv: list[str],
         git: Callable[..., Any] | None) -> Any:
    if git is not None:
        return git(argv)
    return subprocess.run(argv, capture_output=True, text=True,
                          check=False)


def create_worktree(repo: Path | str, dest: Path | str,
                    base_revision: str,
                    git: Callable[..., Any] | None = None) -> str:
    """Create an isolated workspace at ``base_revision``.

    Returns ``"worktree"`` or ``"copy"`` (the mechanism used).
    Raises :exc:`WorkflowError` naming the repo + revision when
    both mechanisms fail — never a silent inherit.
    """
    repo = Path(repo)
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = _run(["git", "-C", str(repo), "worktree", "add", "--detach",
                 str(dest), base_revision], git)
    if proc.returncode == 0 and dest.is_dir():
        return "worktree"
    shutil.rmtree(dest, ignore_errors=True)
    proc = _run(["git", "-C", str(repo), "clone", "--quiet",
                 "--no-hardlinks", str(repo), str(dest)], git)
    if proc.returncode == 0:
        proc = _run(["git", "-C", str(dest), "checkout", "--quiet",
                     base_revision], git)
        if proc.returncode == 0 and dest.is_dir():
            return "copy"
    shutil.rmtree(dest, ignore_errors=True)
    raise WorkflowError(
        f"could not create a workflow workspace at {base_revision} "
        f"in {repo} (worktree and copy both failed)")


def remove_worktree(repo: Path | str, path: Path | str,
                    git: Callable[..., Any] | None = None) -> bool:
    """Remove an isolated workspace. True when it is gone.

    Tries ``worktree remove --force`` first (with a prune after);
    falls back to deleting the directory. Never raises.
    """
    repo = Path(repo)
    path = Path(path)
    if not path.exists():
        return True
    with contextlib.suppress(Exception):  # noqa: BLE001 - removal falls back to rmtree
        proc = _run(["git", "-C", str(repo), "worktree", "remove",
                     "--force", str(path)], git)
        if proc.returncode == 0 and not path.exists():
            _run(["git", "-C", str(repo), "worktree", "prune"], git)
            return True
    shutil.rmtree(path, ignore_errors=True)
    with contextlib.suppress(Exception):  # noqa: BLE001 - prune is best-effort
        _run(["git", "-C", str(repo), "worktree", "prune"], git)
    return not path.exists()
