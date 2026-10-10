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

import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger

__all__ = ["WorkspaceError", "CodeWorkspace"]

_log = get_logger(__name__)

_DEFAULT_TIMEOUT = 60


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break workspace operations (fail-open telemetry,
    fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)


class WorkspaceError(Exception):
    """A code workspace operation failed."""


class CodeWorkspace:
    """An agent-operable handle on one git repository."""

    def __init__(self, root: str | Path, *, artifact_store: Any = None,
                 mission_id: str = "") -> None:
        self.root = Path(root).expanduser().resolve()
        self.artifact_store = artifact_store
        self.mission_id = mission_id
        _emit("codews.workspace.opened", {
            "root": str(self.root),
            "mission_id": mission_id,
        })

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
        """Working-tree status parsed from ``git status --porcelain=v2 --branch``.

        Adds ``renames`` (``[{"from", "to"}]`` from ``2`` lines) and
        ``conflicts`` (unmerged ``u`` lines) on top of the staged /
        unstaged / untracked lists.
        """
        proc = self._checked(["status", "--porcelain=v2", "--branch"], "git status")
        branch = ""
        ahead = behind = 0
        staged: list[str] = []
        unstaged: list[str] = []
        untracked: list[str] = []
        renames: list[dict[str, str]] = []
        conflicts: list[str] = []
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
                # porcelain v2 field counts before <path>:
                #   "1": 8 fields — <n> <XY> <subm> <mH> <mI> <mW> <hH> <hI>
                #   "2": 9 fields — ... plus <score> (renames/copies)
                #   "u": 10 fields — <n> <XY> <subm> <m1> <m2> <m3> <mW>
                #        <h1> <h2> <h3>  (paths may be quoted when spaced)
                kind = line[0]
                first = {"1": 8, "2": 9, "u": 10}[kind]
                rest = " ".join(parts[first:]) if len(parts) > first else ""
                path = rest.split("\t")[0]
                if not path:
                    continue
                if kind == "u":
                    conflicts.append(path)
                    continue
                if kind == "2":
                    sides = rest.split("\t")
                    if len(sides) == 2:
                        renames.append({"from": sides[1], "to": sides[0]})
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
            "renames": renames,
            "conflicts": sorted(conflicts),
            "ahead": ahead,
            "behind": behind,
        }

    def conflicts(self) -> list[str]:
        """Paths with unresolved merge conflicts (porcelain v2 ``u`` lines)."""
        return self.status()["conflicts"]

    def is_dirty(self) -> bool:
        """True when tracked files differ from HEAD (staged, unstaged, or
        conflicted).  Untracked files don't count — same default as
        GitPython's ``is_dirty()``."""
        st = self.status()
        return bool(st["staged"] or st["unstaged"] or st["conflicts"])

    # ── branches ────────────────────────────────────────────────────────
    def branches(self) -> list[dict[str, Any]]:
        """Local branches with a ``current`` flag.

        Each entry also carries ``upstream`` (``""`` when unset) and the
        ``ahead``/``behind`` counts against it, parsed from
        ``%(upstream:track)`` — the same data ``git branch -vv`` shows.
        """
        proc = self._checked(
            ["for-each-ref", "--format=%(refname:short)%00%(HEAD)%00"
             "%(upstream:short)%00%(upstream:track)", "refs/heads"],
            "git for-each-ref")
        out = []
        for line in proc.stdout.splitlines():
            parts = line.split("\x00")
            if len(parts) != 4 or not parts[0]:
                continue
            name, head, upstream, track = parts
            ahead = behind = 0
            m = re.search(r"ahead (\d+)", track)
            if m:
                ahead = int(m.group(1))
            m = re.search(r"behind (\d+)", track)
            if m:
                behind = int(m.group(1))
            out.append({
                "name": name,
                "current": head == "*",
                "upstream": upstream,
                "ahead": ahead,
                "behind": behind,
            })
        return sorted(out, key=lambda b: b["name"])

    def delete_branch(self, name: str, force: bool = False) -> dict[str, Any]:
        """Delete branch ``name`` (``force`` = ``-D``)."""
        if not name or not name.strip():
            raise WorkspaceError("branch name must not be empty")
        flag = "-D" if force else "-d"
        self._checked(["branch", flag, name.strip()], "git branch delete")
        return {"name": name.strip(), "deleted": True}

    def rename_branch(self, old: str, new: str) -> dict[str, Any]:
        """Rename branch ``old`` to ``new``."""
        if not old or not old.strip() or not new or not new.strip():
            raise WorkspaceError("branch names must not be empty")
        self._checked(["branch", "-m", old.strip(), new.strip()],
                      "git branch rename")
        return {"old": old.strip(), "new": new.strip(), "renamed": True}

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
        """Linked worktrees parsed from ``git worktree list --porcelain``.

        Entries carry ``path``, ``branch``, ``sha``, plus ``locked``
        (lock reason or ``""``) and ``prunable`` (``"prunable"``/``""``).
        """
        proc = self._checked(["worktree", "list", "--porcelain"],
                             "git worktree list")
        out: list[dict[str, str]] = []
        cur: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            if line.startswith("worktree "):
                if cur:
                    out.append(cur)
                cur = {"path": line[len("worktree "):], "branch": "",
                       "sha": "", "locked": "", "prunable": ""}
            elif line.startswith("HEAD ") and cur:
                cur["sha"] = line[len("HEAD "):].strip()
            elif line.startswith("branch ") and cur:
                ref = line[len("branch "):].strip()
                cur["branch"] = (ref[len("refs/heads/"):]
                                 if ref.startswith("refs/heads/") else ref)
            elif line.startswith("locked") and cur:
                cur["locked"] = line[len("locked"):].strip()
            elif line == "prunable" and cur:
                cur["prunable"] = "prunable"
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

    # ── commit / sync ─────────────────────────────────────────────────
    def commit(self, message: str, paths: list[str] | None = None) -> dict[str, Any]:
        """Stage and commit. ``paths`` limits the commit; default stages
        everything (``git add -A``).  Fail fast on an empty message or a
        git error (e.g. nothing to commit)."""
        if not message or not message.strip():
            raise WorkspaceError("commit message must not be empty")
        self._ensure_git()
        if paths:
            proc = self._run(["add", "--", *paths])
            if proc.returncode != 0:
                raise WorkspaceError(f"git add failed: {proc.stderr.strip()}")
        else:
            proc = self._run(["add", "-A"])
            if proc.returncode != 0:
                raise WorkspaceError(f"git add -A failed: {proc.stderr.strip()}")
        proc = self._run(["commit", "-m", message.strip()])
        if proc.returncode != 0:
            raise WorkspaceError(f"git commit failed: {proc.stderr.strip()}")
        sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
        return {"sha": sha, "message": message.strip(), "committed": True}

    def push(self, remote: str = "origin", branch: str = "") -> dict[str, Any]:
        """Push to ``remote`` (default ``origin``); ``branch`` pins the
        refspec when given."""
        if not remote or not remote.strip():
            raise WorkspaceError("remote name must not be empty")
        args = ["push", remote.strip()]
        if branch and branch.strip():
            args.append(branch.strip())
        proc = self._checked(args, f"git push {remote.strip()}")
        return {"remote": remote.strip(), "branch": branch.strip(),
                "pushed": True, "output": proc.stderr.strip()}

    def pull(self, remote: str = "origin", branch: str = "") -> dict[str, Any]:
        """Pull from ``remote`` (default ``origin``).  Merge conflicts
        surface as a WorkspaceError with git's own message."""
        if not remote or not remote.strip():
            raise WorkspaceError("remote name must not be empty")
        args = ["pull", remote.strip()]
        if branch and branch.strip():
            args.append(branch.strip())
        proc = self._checked(args, f"git pull {remote.strip()}")
        return {"remote": remote.strip(), "branch": branch.strip(),
                "pulled": True, "output": proc.stdout.strip()}

    def fetch(self, remote: str = "origin") -> dict[str, Any]:
        """Fetch from ``remote`` (default ``origin``) without merging."""
        if not remote or not remote.strip():
            raise WorkspaceError("remote name must not be empty")
        self._checked(["fetch", remote.strip()], f"git fetch {remote.strip()}")
        return {"remote": remote.strip(), "fetched": True}

    # ── stash ─────────────────────────────────────────────────────────
    def stash_push(self, message: str = "") -> dict[str, Any]:
        """Stash working-tree changes (including untracked files)."""
        args = ["stash", "push", "--include-untracked"]
        if message and message.strip():
            args += ["-m", message.strip()]
        proc = self._checked(args, "git stash push")
        return {"stashed": True, "message": message.strip(),
                "output": proc.stdout.strip()}

    def stash_pop(self) -> dict[str, Any]:
        """Restore the most recent stash entry."""
        proc = self._checked(["stash", "pop"], "git stash pop")
        return {"popped": True, "output": proc.stdout.strip()}

    def stash_list(self) -> list[dict[str, str]]:
        """Stash entries: index, message."""
        proc = self._checked(["stash", "list"], "git stash list")
        out = []
        for line in proc.stdout.splitlines():
            m = re.match(r"^stash@\{(\d+)\}:\s*(.*)$", line)
            if m:
                out.append({"index": m.group(1), "message": m.group(2)})
        return out

    # ── staging ─────────────────────────────────────────────────────────
    def add(self, paths: list[str]) -> dict[str, Any]:
        """Stage ``paths`` (``git add --``)."""
        if not paths:
            raise WorkspaceError("no paths given to stage")
        self._checked(["add", "--", *paths], "git add")
        return {"staged": list(paths)}

    def restore(self, paths: list[str], *, staged: bool = False) -> dict[str, Any]:
        """Restore ``paths``: unstage them (``staged=True``) or discard
        working-tree changes (``staged=False``)."""
        if not paths:
            raise WorkspaceError("no paths given to restore")
        args = ["restore"] + (["--staged"] if staged else []) + ["--", *paths]
        self._checked(args, "git restore")
        return {"restored": list(paths), "staged": staged}

    def clean(self, *, force: bool = False) -> dict[str, Any]:
        """Remove untracked files.  ``force=False`` (default) is a dry run
        (``git clean -nd``) reporting what *would* be removed; ``force=True``
        actually removes (``git clean -fd``)."""
        args = ["clean", "-fd" if force else "-nd"]
        proc = self._checked(args, "git clean")
        removed = [l[13:] for l in proc.stdout.splitlines()
                   if l.startswith("Would remove ")]
        return {"removed": removed, "dry_run": not force}

    # ── commit / sync (extended) ────────────────────────────────────────
    def amend(self, message: str = "") -> dict[str, Any]:
        """Amend the previous commit, optionally replacing its message."""
        args = ["commit", "--amend"]
        if message and message.strip():
            args += ["-m", message.strip()]
        else:
            args.append("--no-edit")
        self._checked(args, "git commit --amend")
        sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
        return {"sha": sha, "amended": True}

    def merge_branch(self, branch: str, *, no_ff: bool = False) -> dict[str, Any]:
        """Merge ``branch`` into the current one.

        On conflict, raises :class:`WorkspaceError` listing the conflicted
        paths (parsed from porcelain v2 ``u`` lines) so the caller can
        resolve and commit, or call :meth:`abort_merge`.
        """
        if not branch or not branch.strip():
            raise WorkspaceError("branch name must not be empty")
        self._ensure_git()
        args = ["merge"] + (["--no-ff"] if no_ff else []) + [branch.strip()]
        proc = self._run(args)
        if proc.returncode != 0:
            conflicts = self.conflicts()
            detail = (f"conflicted files: {', '.join(conflicts)}"
                      if conflicts else proc.stderr.strip())
            raise WorkspaceError(f"git merge {branch.strip()} failed: {detail}")
        sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
        _emit("codews.workspace.merged", {"branch": branch.strip(),
                                          "sha": sha})
        return {"branch": branch.strip(), "sha": sha, "merged": True}

    def abort_merge(self) -> dict[str, Any]:
        """Abort an in-progress merge."""
        self._checked(["merge", "--abort"], "git merge --abort")
        return {"aborted": True}

    def cherry_pick(self, commit: str) -> dict[str, Any]:
        """Cherry-pick ``commit`` onto the current branch."""
        if not commit or not commit.strip():
            raise WorkspaceError("commit must not be empty")
        self._ensure_git()
        proc = self._run(["cherry-pick", commit.strip()])
        if proc.returncode != 0:
            conflicts = self.conflicts()
            detail = (f"conflicted files: {', '.join(conflicts)}"
                      if conflicts else proc.stderr.strip())
            raise WorkspaceError(
                f"git cherry-pick {commit.strip()} failed: {detail}")
        sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
        return {"commit": commit.strip(), "sha": sha, "cherry_picked": True}

    def revert(self, commit: str) -> dict[str, Any]:
        """Revert ``commit`` (no editor, auto-commit)."""
        if not commit or not commit.strip():
            raise WorkspaceError("commit must not be empty")
        self._ensure_git()
        proc = self._run(["revert", "--no-edit", commit.strip()])
        if proc.returncode != 0:
            raise WorkspaceError(
                f"git revert {commit.strip()} failed: {proc.stderr.strip()}")
        sha = self._run(["rev-parse", "HEAD"]).stdout.strip()
        return {"commit": commit.strip(), "sha": sha, "reverted": True}

    def push(self, remote: str = "origin", branch: str = "",
             *, set_upstream: bool = False) -> dict[str, Any]:
        """Push to ``remote`` (default ``origin``); ``branch`` pins the
        refspec when given.  ``set_upstream=True`` adds ``-u``."""
        if not remote or not remote.strip():
            raise WorkspaceError("remote name must not be empty")
        args = ["push"]
        if set_upstream:
            args.append("-u")
        args.append(remote.strip())
        if branch and branch.strip():
            args.append(branch.strip())
        proc = self._checked(args, f"git push {remote.strip()}")
        return {"remote": remote.strip(), "branch": branch.strip(),
                "pushed": True, "output": proc.stderr.strip()}

    # ── remotes ─────────────────────────────────────────────────────────
    def remotes(self) -> list[dict[str, str]]:
        """Configured remotes: name and URL."""
        proc = self._checked(["remote", "-v"], "git remote")
        seen: dict[str, str] = {}
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] not in seen:
                seen[parts[0]] = parts[1]
        return [{"name": n, "url": u} for n, u in sorted(seen.items())]

    def remote_add(self, name: str, url: str) -> dict[str, Any]:
        """Add remote ``name`` pointing at ``url``."""
        if not name or not name.strip():
            raise WorkspaceError("remote name must not be empty")
        if not url or not url.strip():
            raise WorkspaceError("remote URL must not be empty")
        self._checked(["remote", "add", name.strip(), url.strip()],
                      "git remote add")
        _emit("codews.workspace.remote_added",
              {"name": name.strip(), "url": url.strip()})
        return {"name": name.strip(), "url": url.strip(), "added": True}

    def remote_remove(self, name: str) -> dict[str, Any]:
        """Remove remote ``name``."""
        if not name or not name.strip():
            raise WorkspaceError("remote name must not be empty")
        self._checked(["remote", "remove", name.strip()], "git remote remove")
        return {"name": name.strip(), "removed": True}

    def remote_set_url(self, name: str, url: str) -> dict[str, Any]:
        """Point remote ``name`` at a new ``url``."""
        if not name or not name.strip():
            raise WorkspaceError("remote name must not be empty")
        if not url or not url.strip():
            raise WorkspaceError("remote URL must not be empty")
        self._checked(["remote", "set-url", name.strip(), url.strip()],
                      "git remote set-url")
        return {"name": name.strip(), "url": url.strip(), "updated": True}

    # ── tags ────────────────────────────────────────────────────────────
    def tags(self) -> list[dict[str, str]]:
        """Tags: name, sha, date."""
        proc = self._checked(
            ["for-each-ref", "--format=%(refname:short)%00%(*objectname:short)"
             "%00%(objectname:short)%00%(creatordate:iso)",
             "refs/tags"],
            "git for-each-ref tags")
        out = []
        for line in proc.stdout.splitlines():
            parts = line.split("\x00")
            if len(parts) != 4 or not parts[0]:
                continue
            name, deref, sha, date = parts
            out.append({"name": name, "sha": deref or sha, "date": date})
        return sorted(out, key=lambda t: t["name"])

    def create_tag(self, name: str, ref: str = "HEAD",
                   message: str = "") -> dict[str, Any]:
        """Create tag ``name`` at ``ref``; annotated when ``message`` is
        given, lightweight otherwise."""
        if not name or not name.strip():
            raise WorkspaceError("tag name must not be empty")
        args = ["tag"]
        if message and message.strip():
            args += ["-a", name.strip(), ref, "-m", message.strip()]
        else:
            args += [name.strip(), ref]
        self._checked(args, "git tag")
        return {"name": name.strip(), "ref": ref, "created": True,
                "annotated": bool(message and message.strip())}

    def delete_tag(self, name: str) -> dict[str, Any]:
        """Delete tag ``name``."""
        if not name or not name.strip():
            raise WorkspaceError("tag name must not be empty")
        self._checked(["tag", "-d", name.strip()], "git tag delete")
        return {"name": name.strip(), "deleted": True}

    # ── stash (extended) ────────────────────────────────────────────────
    def stash_apply(self, index: int = 0) -> dict[str, Any]:
        """Apply stash entry ``index`` without dropping it."""
        proc = self._checked(["stash", "apply", f"stash@{{{max(0, index)}}}"],
                             "git stash apply")
        return {"applied": True, "index": max(0, index),
                "output": proc.stdout.strip()}

    def stash_drop(self, index: int = 0) -> dict[str, Any]:
        """Drop stash entry ``index``."""
        self._checked(["stash", "drop", f"stash@{{{max(0, index)}}}"],
                      "git stash drop")
        return {"dropped": True, "index": max(0, index)}

    # ── worktrees (extended) ────────────────────────────────────────────
    def worktree_prune(self) -> dict[str, Any]:
        """Prune worktree metadata for removed directories."""
        self._checked(["worktree", "prune"], "git worktree prune")
        return {"pruned": True}

    # ── diff / log (extended) ───────────────────────────────────────────
    def diff(self, ref_a: str = "", ref_b: str = "",
             paths: list[str] | None = None) -> str:
        """Unified diff.

        ``diff()`` → working tree vs index/HEAD; ``diff(ref_a)`` →
        working tree vs ``ref_a``; ``diff(ref_a, ref_b)`` → between two
        refs; ``paths`` limits the diff.
        """
        args = ["diff"]
        if ref_a:
            args.append(ref_a)
        if ref_b:
            args.append(ref_b)
        args.append("--")
        if paths:
            args.extend(paths)
        proc = self._checked(args, "git diff")
        return proc.stdout

    def diff_stat(self, ref_a: str = "", ref_b: str = "") -> dict[str, Any]:
        """Per-file ``--numstat`` between refs (or working tree): additions
        and deletions per path, plus totals."""
        args = ["diff", "--numstat"]
        if ref_a:
            args.append(ref_a)
        if ref_b:
            args.append(ref_b)
        args.append("--")
        proc = self._checked(args, "git diff --numstat")
        files: list[dict[str, Any]] = []
        total_add = total_del = 0
        for line in proc.stdout.splitlines():
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            try:
                added = int(parts[0]) if parts[0] != "-" else 0
                deleted = int(parts[1]) if parts[1] != "-" else 0
            except ValueError:
                continue
            files.append({"path": parts[2], "additions": added,
                          "deletions": deleted})
            total_add += added
            total_del += deleted
        return {"files": files,
                "totals": {"files": len(files), "additions": total_add,
                           "deletions": total_del}}

    def log(self, n: int = 10, paths: list[str] | None = None,
            *, stats: bool = False, follow: bool = False
            ) -> list[dict[str, Any]]:
        """Recent commits: sha, author, date, message.

        ``paths`` limits to commits touching those paths; ``stats=True``
        adds per-commit ``files`` with ``--numstat`` additions/deletions;
        ``follow=True`` follows renames (single path only).
        """
        args = ["log", f"-{max(1, n)}", "--format=%H%x1f%an%x1f%ad%x1f%s",
                "--date=iso"]
        if stats:
            args.append("--numstat")
        if follow:
            args.append("--follow")
        if paths:
            args += ["--", *paths]
        proc = self._checked(args, "git log")
        commits: list[dict[str, Any]] = []
        cur: dict[str, Any] | None = None
        for line in proc.stdout.splitlines():
            if "\x1f" in line:
                parts = line.split("\x1f")
                if len(parts) == 4:
                    cur = {"sha": parts[0], "author": parts[1],
                           "date": parts[2], "message": parts[3],
                           "files": []}
                    commits.append(cur)
                else:
                    cur = None
            elif stats and cur is not None and "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    try:
                        added = int(parts[0]) if parts[0] != "-" else 0
                        deleted = int(parts[1]) if parts[1] != "-" else 0
                    except ValueError:
                        continue
                    cur["files"].append({"path": parts[2],
                                        "additions": added,
                                        "deletions": deleted})
        return commits

    def file_history(self, path: str, n: int = 10) -> list[dict[str, Any]]:
        """Commit history for one path, following renames
        (``git log --follow``)."""
        if not path or not path.strip():
            raise WorkspaceError("path must not be empty")
        return self.log(max(1, n), paths=[path.strip()], follow=True)

    def blame(self, path: str) -> list[dict[str, Any]]:
        """Per-line blame from ``git blame --porcelain``: line number,
        sha, author, date, and content."""
        if not path or not path.strip():
            raise WorkspaceError("path must not be empty")
        proc = self._checked(["blame", "--porcelain", "--", path.strip()],
                             "git blame")
        out: list[dict[str, Any]] = []
        sha = author = date = ""
        for line in proc.stdout.splitlines():
            if line.startswith("\t"):
                out.append({"line": len(out) + 1, "sha": sha,
                            "author": author, "date": date,
                            "content": line[1:]})
            elif re.match(r"^[0-9a-f]{40} \d+ \d+", line):
                sha = line.split(" ")[0]
            elif line.startswith("author "):
                author = line[len("author "):]
            elif line.startswith("author-time "):
                try:
                    import datetime
                    date = datetime.datetime.fromtimestamp(
                        int(line[len("author-time "):]),
                        tz=datetime.timezone.utc).isoformat()
                except ValueError:
                    date = ""
        return out

    def show(self, ref: str, path: str) -> str:
        """File content at ``ref`` (``git show ref:path``)."""
        if not ref or not ref.strip():
            raise WorkspaceError("ref must not be empty")
        if not path or not path.strip():
            raise WorkspaceError("path must not be empty")
        proc = self._checked(["show", f"{ref.strip()}:{path.strip()}"],
                             "git show")
        return proc.stdout
