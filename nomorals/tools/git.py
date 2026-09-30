"""First-class git operations for agents and the ``nm code`` CLI.

Read operations (status, diff, log, branch) are always available.
Write operations (commit, push, stash, restore) go through the registry's
existing approval surface — ``confirm=True`` on the tool spec — and never
auto-commit: :func:`git_commit` refuses when the tree has dirty changes
outside the explicitly scoped ``paths``.

All commands run with a timeout and return structured results; failures are
:class:`ToolError`\\ s, never tracebacks.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from ..core.errors import ToolError, ValidationError
from ..core.logging_setup import get_logger
from ..core.policy import Capability

__all__ = [
    "git_status",
    "git_diff",
    "git_log",
    "git_branch",
    "git_commit",
    "git_push",
    "git_stash",
    "restore",
]

_log = get_logger(__name__)

_DEFAULT_TIMEOUT = 60
_DIFF_CAP = 50_000  # characters; larger diffs are truncated with a note


def _resolve_repo(repo: str | None) -> Path:
    """Resolve the target repo directory. Defaults to the current directory."""
    root = Path(repo).expanduser().resolve() if repo else Path.cwd().resolve()
    if not root.is_dir():
        raise ValidationError(f"not a directory: {root}", field="repo")
    return root


def _run_git(args: list[str], repo: Path, *, timeout: int = _DEFAULT_TIMEOUT) -> subprocess.CompletedProcess[str]:
    git = shutil.which("git")
    if not git:
        raise ToolError("git binary not found on PATH")
    try:
        return subprocess.run(
            [git, *args],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise ToolError(f"git {' '.join(args)} timed out after {timeout}s") from exc


def _ensure_repo(repo: Path) -> None:
    proc = _run_git(["rev-parse", "--git-dir"], repo)
    if proc.returncode != 0:
        raise ToolError(f"not a git repository: {repo}")


def git_status(repo: str | None = None) -> dict[str, Any]:
    """Working-tree status: branch, staged/unstaged/untracked files."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], root).stdout.strip()
    proc = _run_git(["status", "--porcelain=v1"], root)
    if proc.returncode != 0:
        raise ToolError(f"git status failed: {proc.stderr.strip()}")
    staged, unstaged, untracked = [], [], []
    for line in proc.stdout.splitlines():
        if not line.strip():
            continue
        x, y, path = line[0], line[1], line[3:]
        if x == "?" and y == "?":
            untracked.append(path)
            continue
        if x != " ":
            staged.append(path)
        if y != " ":
            unstaged.append(path)
    return {
        "repo": str(root),
        "branch": branch,
        "staged": sorted(staged),
        "unstaged": sorted(unstaged),
        "untracked": sorted(untracked),
        "dirty": bool(staged or unstaged or untracked),
    }


def git_diff(ref: str | None = None, paths: list[str] | None = None, repo: str | None = None) -> dict[str, Any]:
    """Unified diff of the working tree (or ``ref``). Output is capped."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    args = ["diff"]
    if ref:
        args.append(ref)
    args += ["--", *(paths or [])]
    proc = _run_git(args, root)
    if proc.returncode != 0:
        raise ToolError(f"git diff failed: {proc.stderr.strip()}")
    diff = proc.stdout
    truncated = False
    if len(diff) > _DIFF_CAP:
        diff = diff[:_DIFF_CAP]
        truncated = True
    return {"repo": str(root), "ref": ref, "diff": diff, "truncated": truncated,
            "bytes": len(proc.stdout)}


def git_log(n: int = 10, repo: str | None = None) -> dict[str, Any]:
    """Recent commits: hash, subject, author, date."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    proc = _run_git(
        ["log", f"-{max(1, n)}", "--format=%H%x1f%an%x1f%ad%x1f%s", "--date=short"],
        root,
    )
    if proc.returncode != 0:
        raise ToolError(f"git log failed: {proc.stderr.strip()}")
    commits = []
    for line in proc.stdout.splitlines():
        parts = line.split("\x1f")
        if len(parts) == 4:
            commits.append({"hash": parts[0][:12], "author": parts[1],
                            "date": parts[2], "subject": parts[3]})
    return {"repo": str(root), "commits": commits}


def git_branch(repo: str | None = None) -> dict[str, Any]:
    """Current branch and the local branch list."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    proc = _run_git(["branch", "--format=%(refname:short)%00%(HEAD)"], root)
    if proc.returncode != 0:
        raise ToolError(f"git branch failed: {proc.stderr.strip()}")
    branches, current = [], ""
    for line in proc.stdout.splitlines():
        name, head = line.split("\x00")
        branches.append(name)
        if head == "*":
            current = name
    return {"repo": str(root), "current": current, "branches": sorted(branches)}


def _dirty_files(root: Path) -> set[str]:
    """All dirty paths (staged, unstaged, untracked), relative to repo root."""
    proc = _run_git(["status", "--porcelain=v1"], root)
    dirty = set()
    for line in proc.stdout.splitlines():
        if line.strip():
            dirty.add(line[3:].strip().strip('"'))
    return dirty


def git_commit(message: str, paths: list[str] | None = None, repo: str | None = None) -> dict[str, Any]:
    """Commit. Refuses when the tree has dirty changes outside ``paths``.

    With ``paths``: stages exactly those paths, then refuses if anything else
    is dirty. Without ``paths``: commits only already-staged changes, and
    refuses if there are unstaged or untracked changes.
    """
    root = _resolve_repo(repo)
    _ensure_repo(root)
    if not message or not message.strip():
        raise ValidationError("commit message must not be empty", field="message")

    if paths:
        proc = _run_git(["add", "--", *paths], root)
        if proc.returncode != 0:
            raise ToolError(f"git add failed: {proc.stderr.strip()}")
        allowed = {p.strip().strip('"') for p in paths}
        # Also allow files under the given directories.
        dirty = _dirty_files(root)
        unrelated = {d for d in dirty
                     if d not in allowed
                     and not any(d.startswith(a.rstrip("/") + "/") for a in allowed)}
        if unrelated:
            raise ToolError(
                "refusing to commit: unrelated dirty changes present: "
                + ", ".join(sorted(unrelated)[:10])
            )
    else:
        status = git_status(str(root))
        if status["unstaged"] or status["untracked"]:
            raise ToolError(
                "refusing to commit: unstaged/untracked changes present "
                "(pass paths= to scope the commit)"
            )
        if not status["staged"]:
            raise ToolError("nothing staged to commit")

    proc = _run_git(["commit", "-m", message.strip()], root)
    if proc.returncode != 0:
        raise ToolError(f"git commit failed: {proc.stderr.strip() or proc.stdout.strip()}")
    _log.info("committed in %s: %s", root, message.strip()[:60])
    return {"repo": str(root), "committed": True,
            "summary": proc.stdout.strip().splitlines()[:3]}


def git_push(remote: str = "origin", branch: str | None = None, repo: str | None = None) -> dict[str, Any]:
    """Push the current branch. Requires confirmation via the registry."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    target = branch or git_branch(str(root))["current"]
    proc = _run_git(["push", remote, target], root, timeout=120)
    if proc.returncode != 0:
        raise ToolError(f"git push failed: {proc.stderr.strip()}")
    return {"repo": str(root), "pushed": True, "remote": remote, "branch": target,
            "output": proc.stderr.strip().splitlines()[:5]}


def git_stash(message: str = "", repo: str | None = None) -> dict[str, Any]:
    """Stash working-tree changes."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    args = ["stash", "push"]
    if message.strip():
        args += ["-m", message.strip()]
    proc = _run_git(args, root)
    if proc.returncode != 0:
        raise ToolError(f"git stash failed: {proc.stderr.strip()}")
    return {"repo": str(root), "stashed": True, "output": proc.stdout.strip()}


def restore(path: str, repo: str | None = None) -> dict[str, Any]:
    """Restore ``path`` to its committed state (``git checkout --``)."""
    root = _resolve_repo(repo)
    _ensure_repo(root)
    proc = _run_git(["checkout", "--", path], root)
    if proc.returncode != 0:
        raise ToolError(f"git restore failed: {proc.stderr.strip()}")
    return {"repo": str(root), "restored": path}


def register(registry: Any) -> None:
    """Attach the git tools to a registry."""
    context = registry.context

    def _repo_kw(repo: str | None) -> str | None:
        # Agent-called tools default to the workspace; the CLI passes --root.
        if repo:
            return repo
        settings = getattr(context, "settings", None) if context is not None else None
        if settings is not None:
            try:
                return str(settings.workspace_dir)
            except Exception:  # noqa: BLE001 — fall back to cwd
                pass
        return None

    @registry.register(
        "git_status",
        description="Show git working-tree status: branch, staged/unstaged/untracked files.",
        capability=Capability.FS_READ,
    )
    def _git_status(repo: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            return git_status(_repo_kw(repo))
        finally:
            _log.debug("git_status took %.2fs", time.perf_counter() - t0)

    @registry.register(
        "git_diff",
        description="Unified diff of the working tree (or a ref). Output is capped at 50KB.",
        capability=Capability.FS_READ,
    )
    def _git_diff(ref: str | None = None, paths: list[str] | None = None,
                  repo: str | None = None) -> dict[str, Any]:
        return git_diff(ref, paths, _repo_kw(repo))

    @registry.register(
        "git_log",
        description="Recent commits: hash, subject, author, date.",
        capability=Capability.FS_READ,
    )
    def _git_log(n: int = 10, repo: str | None = None) -> dict[str, Any]:
        return git_log(n, _repo_kw(repo))

    @registry.register(
        "git_branch",
        description="Current branch and the local branch list.",
        capability=Capability.FS_READ,
    )
    def _git_branch(repo: str | None = None) -> dict[str, Any]:
        return git_branch(_repo_kw(repo))

    @registry.register(
        "git_commit",
        description=("Commit. Refuses when the tree has dirty changes outside the "
                     "scoped paths. Requires confirmation."),
        capability=Capability.FS_WRITE,
        confirm=True,
    )
    def _git_commit(message: str, paths: list[str] | None = None,
                    repo: str | None = None) -> dict[str, Any]:
        return git_commit(message, paths, _repo_kw(repo))

    @registry.register(
        "git_push",
        description="Push the current branch to a remote. Requires confirmation.",
        capability=Capability.NET_OUT,
        confirm=True,
    )
    def _git_push(remote: str = "origin", branch: str | None = None,
                  repo: str | None = None) -> dict[str, Any]:
        return git_push(remote, branch, _repo_kw(repo))

    @registry.register(
        "git_stash",
        description="Stash working-tree changes. Requires confirmation.",
        capability=Capability.FS_WRITE,
        confirm=True,
    )
    def _git_stash(message: str = "", repo: str | None = None) -> dict[str, Any]:
        return git_stash(message, _repo_kw(repo))
