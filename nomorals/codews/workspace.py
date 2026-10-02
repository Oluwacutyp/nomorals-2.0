"""L5 code workspace: a git repository as an agent-operable object.

:class:`CodeWorkspace` wraps the repo-local operations an agent needs —
status, branches, worktrees, diffs, and history — with fail-fast
:class:`WorkspaceError` semantics.  All git traffic runs through
``subprocess`` with list arguments (never ``shell=True``) and a timeout.

This module intentionally keeps its own small git runner instead of
reusing :mod:`nomorals.tools.git`: the tools there raise ``ToolError``
and parse ``porcelain=v1`` (no ahead/behind), while the workspace needs
``WorkspaceError`` and ``porcelain=v2`` branch tracking.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

__all__ = ["WorkspaceError", "CodeWorkspace"]

_DEFAULT_TIMEOUT = 60


class WorkspaceError(Exception):
    """A code workspace operation failed."""


class CodeWorkspace:
    """An agent-operable handle on one git repository."""

    def __init__(self, root: str | Path, *, artifact_store: Any = None,
                 mission_id: str = "") -> None:
        self.root = Path(root).expanduser().resolve()
        self.artifact_store = artifact_store
        self.mission_id = mission_id

    # ── plumbing ──────────────────────────────────────────────────────────
    def _git_bin(self) -> str:
        git = shutil.which("git")
        if not git:
            raise WorkspaceError("git binary not found on PATH")
        return git

    def _run(self, args: list[str], *, timeout: int = _DEFAULT_TIMEOUT) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [self._git_bin(), *args],
                cwd=str(self.root),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise WorkspaceError(
                f"git {' '.join(args)} timed out after {timeout}s") from exc

    def _ensure_git(self) -> None:
        """Fail fast unless ``root`` is a git repository."""
        if not self.root.is_dir():
            raise WorkspaceError(f"{self.root} is not a git repository")
        proc = self._run(["rev-parse", "--git-dir"])
        if proc.returncode != 0:
            raise WorkspaceError(f"{self.root} is not a git repository")

    def _checked(self, args: list[str], what: str) -> subprocess.CompletedProcess[str]:
        self._ensure_git()
        proc = self._run(args)
        if proc.returncode != 0:
            raise WorkspaceError(f"{what} failed: {proc.stderr.strip()}")
        return proc

    # ── status ──────────────────────────────────────────────────────────
    def status(self) -> dict[str, Any]:
        """Working-tree status parsed from ``git status --porcelain=v2 --branch``."""
        proc = self._checked(["status", "--porcelain=v2", "--branch"], "git status")
        branch = ""
        ahead = behind = 0
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        for line in proc.stdout.splitlines():
            if line.startswith("# branch.head "):
                head = line[len("# branch.head "):]
                branch = "" if head == "(detached)" else head
            elif line.startswith("# branch.ab "):
                for token in line[len("# branch.ab "):].split():
                    if token.startswith("+"):
                        ahead = int(token[1:])
                    elif token.startswith("-"):
                        behind = int(token[1:])
            elif line.startswith("? "):
                untracked.append(line[2:])
            elif line.startswith(("1 ", "2 ", "u ")):
                parts = line.split(" ")
                # porcelain v2: "<n> <XY> <subm> <mH> <mI> <mW> <hH> <hI> <path>"
                # — 8 fixed fields, then the path (quoted when it has spaces);
                # renames/copies are "<to>\t<from>".
                path = " ".join(parts[8:]).split("\t")[0] if len(parts) > 8 else ""
                if not path:
                    continue
                x, y = parts[1][0], parts[1][1]
                if x != ".":
                    staged.append(path)
                if y != ".":
                    unstaged.append(path)
            # "!" (ignored) lines are skipped.
        return {
            "branch": branch,
            "staged": sorted(staged),
            "unstaged": sorted(unstaged),
            "untracked": sorted(untracked),
            "ahead": ahead,
            "behind": behind,
        }

    # ── branches ────────────────────────────────────────────────────────
    def branches(self) -> list[dict[str, Any]]:
        """Local branches with a ``current`` flag."""
        proc = self._checked(["branch", "--format=%(refname:short)%00%(HEAD)"],
                             "git branch")
        out = []
        for line in proc.stdout.splitlines():
            if "\x00" not in line:
                continue
            name, head = line.split("\x00", 1)
            if name:
                out.append({"name": name, "current": head == "*"})
        return sorted(out, key=lambda b: b["name"])

    def current_branch(self) -> str:
        """Name of the checked-out branch ("" when HEAD is detached)."""
        proc = self._checked(["rev-parse", "--abbrev-ref", "HEAD"],
                             "git rev-parse")
        branch = proc.stdout.strip()
        return "" if branch == "HEAD" else branch

    def create_branch(self, name: str, start: str = "HEAD") -> dict[str, Any]:
        """Create branch ``name`` at ``start`` (default HEAD)."""
        if not name or not name.strip():
            raise WorkspaceError("branch name must not be empty")
        self._checked(["branch", name.strip(), start], "git branch")
        return {"name": name.strip(), "start": start, "created": True}

    def switch_branch(self, name: str) -> dict[str, Any]:
        """Check out branch ``name``."""
        if not name or not name.strip():
            raise WorkspaceError("branch name must not be empty")
        self._checked(["switch", name.strip()], "git switch")
        return {"name": name.strip(), "current": True}

    # ── worktrees ───────────────────────────────────────────────────────
    def worktree_add(self, path: str | Path, branch: str) -> dict[str, Any]:
        """Add a linked worktree at ``path``.

        Checks out ``branch`` when it already exists, otherwise creates it
        (``git worktree add -b``).
        """
        if not branch or not branch.strip():
            raise WorkspaceError("branch name must not be empty")
        self._ensure_git()
        dest = str(Path(path).expanduser())
        exists = self._run(
            ["rev-parse", "--verify", "--quiet", f"refs/heads/{branch.strip()}"])
        args = (["worktree", "add", dest, branch.strip()] if exists.returncode == 0
                else ["worktree", "add", "-b", branch.strip(), dest])
        proc = self._run(args)
        if proc.returncode != 0:
            raise WorkspaceError(
                f"git worktree add failed: {proc.stderr.strip()}")
        return {"path": dest, "branch": branch.strip(), "added": True}

    def worktree_list(self) -> list[dict[str, str]]:
        """Linked worktrees parsed from ``git worktree list --porcelain``."""
        proc = self._checked(["worktree", "list", "--porcelain"],
                             "git worktree list")
        out: list[dict[str, str]] = []
        cur: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if line.startswith("worktree "):
                if cur:
                    out.append(cur)
                cur = {"path": line[len("worktree "):], "branch": "", "sha": ""}
            elif line.startswith("HEAD ") and cur:
                cur["sha"] = line[len("HEAD "):].strip()
            elif line.startswith("branch ") and cur:
                ref = line[len("branch "):].strip()
                cur["branch"] = (ref[len("refs/heads/"):]
                                 if ref.startswith("refs/heads/") else ref)
        if cur:
            out.append(cur)
        return out

    def worktree_remove(self, path: str | Path, force: bool = False) -> dict[str, Any]:
        """Remove the worktree at ``path`` (``force`` discards dirty state)."""
        self._ensure_git()
        args = ["worktree", "remove"] + (["--force"] if force else []) + [str(path)]
        proc = self._run(args)
        if proc.returncode != 0:
            raise WorkspaceError(
                f"git worktree remove failed: {proc.stderr.strip()}")
        return {"path": str(path), "removed": True}

    # ── diff / log ──────────────────────────────────────────────────────
    def diff(self, ref: str = "") -> str:
        """Unified diff of the working tree, or against ``ref`` when given."""
        args = ["diff"]
        if ref:
            args.append(ref)
        args.append("--")
        proc = self._checked(args, "git diff")
        return proc.stdout

    def log(self, n: int = 10) -> list[dict[str, str]]:
        """Recent commits: sha, author, date, message."""
        proc = self._checked(
            ["log", f"-{max(1, n)}", "--format=%H%x1f%an%x1f%ad%x1f%s",
             "--date=iso"],
            "git log",
        )
        commits = []
        for line in proc.stdout.splitlines():
            parts = line.split("\x1f")
            if len(parts) == 4:
                commits.append({
                    "sha": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                })
        return commits
