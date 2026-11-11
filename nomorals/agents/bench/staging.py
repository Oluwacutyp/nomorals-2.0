"""Per-cursor staging + candidate capture (Lane 2).

The dossier's isolation rule (§2.3): a cursor's unfinished work
is *unreadable* — not discouraged, unreadable. This module is
where that becomes filesystem fact:

- At claim time the monitor creates the task's **staging area**:
  an isolated checkout of the integration repo pinned to the
  task's ``base_revision`` (the current promoted HEAD), under the
  bench dir (``<bench root>/<run_id>/<task_id>/``). The mechanism
  is a **git worktree** where worktrees hold (proven on this box
  by the lane's tests: worktree add/capture/remove against a real
  repo), with a **plain local clone** as the fallback where they
  don't — the semantics (isolation, pinned base, capturable diff)
  are identical either way, and the mechanism used is recorded on
  the area. Whether worktrees hold on the EC2 *Windows* host is
  NOT proven by anything here and stays an open verdict; the
  fallback exists for exactly that case.
- A cursor resolves exactly **three read views**, exposed by
  :class:`BenchViews`: (a) promoted revisions of the integration
  repo, (b) its *own* staging — holder-checked against the store,
  so a peer asking for another cursor's staging path is refused
  by construction, there is no API that returns it — and
  (c) promoted artifacts of completed dependencies, read at the
  revision the journal recorded for their promotion. There is no
  fourth view.
- **Submit = capture, not copy.** Capture follows the
  CheckpointStore idiom (a git commit object built from the
  working tree without moving any ref and without touching the
  stash list) in its commit-tree form — ``add -A`` +
  ``write-tree`` + ``commit-tree`` — which additionally carries
  untracked files, as a candidate must. The candidate commit is
  immutable, pinned under ``refs/bench/candidates/<task_id>``,
  and stays invisible to peers (``BenchStore.peer_view``) until
  the integration queue promotes it.
- A dead claim's staging is **quarantined** (moved aside, kept
  for inspection, never promoted and never handed to the next
  cursor); a promoted or voluntarily released task's staging is
  discarded — its content already lives in git objects.

The integration repo itself is the promoted tree: its HEAD is
the journal's ``promoted_revision``, and the integration queue
(:mod:`.integration`) is its only writer.

Layer: agents (L5) — stdlib + the bench model only.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import TASK_PROMOTED, BenchError

__all__ = [
    "BenchViews",
    "StagingArea",
    "StagingManager",
    "MECHANISM_WORKTREE",
    "MECHANISM_COPY",
]

MECHANISM_WORKTREE = "worktree"
MECHANISM_COPY = "copy"

_STATE_ACTIVE = "active"
_STATE_QUARANTINED = "quarantined"
_STATE_DISCARDED = "discarded"

#: Identity stamped on bench-created commits (candidates and
#: promotions). Explicit on every commit-creating git call so a
#: bench commit never depends on the host's git config.
_GIT_IDENTITY = (
    "-c", "user.name=Devon Bench",
    "-c", "user.email=bench@nomorals.invalid",
    "-c", "commit.gpgsign=false",
)


def _git(cwd: Path, args: list[str], *,
         identity: bool = False,
         stdin: str | None = None,
         timeout: int = 120) -> subprocess.CompletedProcess[str]:
    argv = ["git"]
    if identity:
        argv.extend(_GIT_IDENTITY)
    argv.extend(args)
    return subprocess.run(
        argv, cwd=str(cwd), input=stdin, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        timeout=timeout)


def _git_bytes(cwd: Path, args: list[str],
               timeout: int = 60) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True,
        timeout=timeout)


@dataclass
class StagingArea:
    """One task's isolated checkout, as recorded on disk."""

    run_id: str
    task_id: str
    key: str
    path: Path
    mechanism: str
    base_revision: str
    state: str = _STATE_ACTIVE

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id, "task_id": self.task_id,
            "key": self.key, "path": str(self.path),
            "mechanism": self.mechanism,
            "base_revision": self.base_revision,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StagingArea:
        return cls(
            run_id=str(data.get("run_id", "")),
            task_id=str(data.get("task_id", "")),
            key=str(data.get("key", "")),
            path=Path(str(data.get("path", ""))),
            mechanism=str(data.get("mechanism", "")),
            base_revision=str(data.get("base_revision", "")),
            state=str(data.get("state", _STATE_ACTIVE)))


class StagingManager:
    """Owns staging areas and candidate capture for one
    integration repo. ``repo`` is the integration repo's working
    dir (the promoted tree); ``root`` is the bench dir under
    which per-run staging lives."""

    def __init__(self, repo: str | Path, root: str | Path) -> None:
        self.repo = Path(repo)
        self.root = Path(root)

    # -- paths / records --------------------------------------------

    def _run_dir(self, run_id: str) -> Path:
        return self.root / str(run_id)

    def _area_dir(self, run_id: str, task_id: str) -> Path:
        return self._run_dir(run_id) / str(task_id)

    def _sidecar(self, run_id: str, task_id: str) -> Path:
        return self._run_dir(run_id) / (str(task_id) + ".staging.json")

    def _save(self, area: StagingArea, *,
              signature: dict[str, Any] | None = None,
              signature_at: float | None = None) -> None:
        path = self._sidecar(area.run_id, area.task_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = area.to_dict()
        prior = self._read_record(area.run_id, area.task_id) or {}
        record["signature"] = (signature if signature is not None
                               else prior.get("signature"))
        record["signature_at"] = (
            signature_at if signature_at is not None
            else prior.get("signature_at", 0.0))
        record["last_change_at"] = prior.get("last_change_at", 0.0)
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")

    def _read_record(self, run_id: str,
                     task_id: str) -> dict[str, Any] | None:
        try:
            data = json.loads(
                self._sidecar(run_id, task_id).read_text(
                    encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def area(self, run_id: str,
             task_id: str) -> StagingArea | None:
        """The recorded area for a task, any state — the
        monitor-side read (never handed to a cursor directly;
        cursor access goes through :class:`BenchViews`)."""
        record = self._read_record(run_id, task_id)
        if record is None:
            return None
        return StagingArea.from_dict(record)

    # -- lifecycle ----------------------------------------------------

    def create(self, run_id: str, task: dict[str, Any], *,
               now: float | None = None) -> tuple[StagingArea, bool]:
        """Create the staging area for a freshly claimed task,
        pinned to the task's ``base_revision``. Idempotent: an
        already-active area is returned as-is. Returns
        ``(area, created)``."""
        now = time.time() if now is None else float(now)
        task_id = str(task["id"])
        existing = self.area(run_id, task_id)
        if existing is not None and existing.state == _STATE_ACTIVE \
                and existing.path.is_dir():
            return existing, False
        base_revision = str(task.get("base_revision") or "")
        if not base_revision:
            raise BenchError(
                "staging needs the task's pinned base_revision")
        dest = self._area_dir(run_id, task_id)
        dest.parent.mkdir(parents=True, exist_ok=True)
        mechanism = self._create_worktree(dest, base_revision)
        if mechanism is None:
            mechanism = self._create_copy(dest, base_revision)
        if mechanism is None:
            raise BenchError(
                "could not create a staging area at "
                + base_revision + " (worktree and copy both failed)")
        area = StagingArea(run_id=str(run_id), task_id=task_id,
                           key=str(task.get("key") or ""), path=dest,
                           mechanism=mechanism,
                           base_revision=base_revision)
        sig = self.signature(area)
        self._save(area, signature=sig, signature_at=now)
        return area, True

    def _create_worktree(self, dest: Path,
                         base_revision: str) -> str | None:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        proc = _git(self.repo, ["worktree", "add", "--detach",
                                str(dest), base_revision])
        if proc.returncode == 0 and dest.is_dir():
            return MECHANISM_WORKTREE
        shutil.rmtree(dest, ignore_errors=True)
        return None

    def _create_copy(self, dest: Path,
                     base_revision: str) -> str | None:
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        proc = _git(self.repo, ["clone", "--quiet", "--no-hardlinks",
                                str(self.repo), str(dest)])
        if proc.returncode != 0 or not dest.is_dir():
            shutil.rmtree(dest, ignore_errors=True)
            return None
        proc = _git(dest, ["checkout", "--quiet", base_revision])
        if proc.returncode != 0:
            shutil.rmtree(dest, ignore_errors=True)
            return None
        return MECHANISM_COPY

    def discard(self, run_id: str, task_id: str) -> bool:
        """Remove an area outright (promotion done / claim
        voluntarily released — the content already lives in git
        objects or was abandoned by its owner)."""
        area = self.area(run_id, task_id)
        if area is None or area.state == _STATE_DISCARDED:
            return False
        self._remove_tree(area)
        area.state = _STATE_DISCARDED
        self._save(area)
        return True

    def quarantine(self, run_id: str,
                   task_id: str) -> StagingArea | None:
        """Move a dead claim's area aside, contents intact for
        inspection. It is never promoted and never becomes
        another cursor's starting point (dossier §2.5)."""
        area = self.area(run_id, task_id)
        if area is None or area.state != _STATE_ACTIVE:
            return area
        dest = self._area_dir(run_id, task_id + ".quarantine")
        n = 2
        while dest.exists():
            dest = self._area_dir(
                run_id, task_id + ".quarantine-" + str(n))
            n += 1
        moved = False
        if area.mechanism == MECHANISM_WORKTREE:
            proc = _git(self.repo, ["worktree", "move",
                                    str(area.path), str(dest)])
            moved = proc.returncode == 0 and dest.is_dir()
        if not moved:
            try:
                os.rename(area.path, dest)
                moved = dest.is_dir()
            except OSError:
                moved = False
            if area.mechanism == MECHANISM_WORKTREE:
                _git(self.repo, ["worktree", "prune"])
        if not moved:
            return area
        area.path = dest
        area.state = _STATE_QUARANTINED
        self._save(area)
        return area

    def _remove_tree(self, area: StagingArea) -> None:
        if area.mechanism == MECHANISM_WORKTREE:
            proc = _git(self.repo, ["worktree", "remove", "--force",
                                    str(area.path)])
            if proc.returncode != 0:
                shutil.rmtree(area.path, ignore_errors=True)
                _git(self.repo, ["worktree", "prune"])
        else:
            shutil.rmtree(area.path, ignore_errors=True)

    # -- artifact signature (the watchdog's progress signal) ---------

    def signature(self, area: StagingArea) -> dict[str, Any]:
        """A cheap content-shape fingerprint of the staging tree:
        every file's (path, size, mtime) folded into a digest,
        plus the file count and newest mtime. Any edit a cursor
        makes moves it; nothing else does."""
        digest = hashlib.sha256()
        files = 0
        max_mtime = 0.0
        if area.path.is_dir():
            for dirpath, dirnames, filenames in os.walk(area.path):
                dirnames[:] = sorted(
                    d for d in dirnames if d != ".git")
                for name in sorted(filenames):
                    if name == ".git":
                        continue
                    full = Path(dirpath) / name
                    try:
                        stat = full.stat()
                    except OSError:
                        continue
                    rel = full.relative_to(area.path).as_posix()
                    digest.update(rel.encode("utf-8"))
                    digest.update(str(stat.st_size).encode())
                    digest.update(str(stat.st_mtime_ns).encode())
                    files += 1
                    max_mtime = max(max_mtime, stat.st_mtime)
        return {"digest": digest.hexdigest(), "files": files,
                "max_mtime": max_mtime}

    def observe_progress(self, run_id: str,
                         task_id: str) -> dict[str, Any]:
        """Compare the area's current signature with the recorded
        baseline. A change IS progress (the dossier's artifact
        signal): the baseline advances and the change is reported
        with its evidence. No area → no signal, never an error."""
        area = self.area(run_id, task_id)
        if area is None or area.state != _STATE_ACTIVE:
            return {"changed": False, "digest": "", "files": 0,
                    "last_change_at": 0.0}
        record = self._read_record(run_id, task_id) or {}
        baseline = record.get("signature") or {}
        current = self.signature(area)
        changed = current["digest"] != baseline.get("digest")
        last_change_at = float(record.get("last_change_at") or 0)
        if changed:
            last_change_at = time.time()
            path = self._sidecar(run_id, task_id)
            record["signature"] = current
            record["signature_at"] = last_change_at
            record["last_change_at"] = last_change_at
            with contextlib.suppress(OSError):
                path.write_text(json.dumps(record, indent=1),
                                encoding="utf-8")
        return {"changed": changed, "digest": current["digest"],
                "files": current["files"],
                "last_change_at": last_change_at}

    # -- candidate capture ---------------------------------------------

    def capture(self, run_id: str,
                task_id: str) -> dict[str, Any]:
        """Capture the staging diff as an immutable candidate
        commit (see the module docstring for the idiom). The
        candidate's parent is the staging checkout's HEAD — the
        task's pinned base revision — so the integration queue
        can rebase-apply it onto whatever HEAD has become.

        Returns ``{"candidate_ref", "files", "stat", "empty",
        "base_revision"}``; an untouched staging captures as
        ``empty`` with no commit created."""
        area = self.area(run_id, task_id)
        if area is None or area.state != _STATE_ACTIVE:
            raise BenchError(
                "no active staging area to capture for task "
                + str(task_id))
        work = area.path
        head = _git(work, ["rev-parse", "HEAD"])
        if head.returncode != 0:
            raise BenchError("staging HEAD unreadable: "
                             + head.stderr.strip())
        base_revision = head.stdout.strip()
        add = _git(work, ["add", "-A"])
        if add.returncode != 0:
            raise BenchError("candidate capture failed at add: "
                             + add.stderr.strip())
        diff = _git(work, ["diff", "--cached", "--name-status"])
        files: list[str] = []
        if diff.returncode == 0:
            for line in diff.stdout.splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    files.append(parts[-1])
        tree = _git(work, ["write-tree"])
        if tree.returncode != 0:
            raise BenchError("candidate capture failed at "
                             "write-tree: " + tree.stderr.strip())
        tree_sha = tree.stdout.strip()
        head_tree = _git(work, ["rev-parse", "HEAD^{tree}"])
        if head_tree.returncode == 0 \
                and head_tree.stdout.strip() == tree_sha:
            return {"candidate_ref": "", "files": [], "stat": "",
                    "empty": True, "base_revision": base_revision}
        message = (
            "bench candidate\n\n"
            + "Bench-Run: " + str(run_id) + "\n"
            + "Bench-Task-Id: " + str(task_id) + "\n"
            + "Bench-Task: " + str(area.key) + "\n")
        commit = _git(work, ["commit-tree", tree_sha, "-p", "HEAD"],
                      identity=True, stdin=message)
        if commit.returncode != 0:
            raise BenchError("candidate capture failed at "
                             "commit-tree: " + commit.stderr.strip())
        candidate_ref = commit.stdout.strip()
        _git(work, ["update-ref",
                    "refs/bench/candidates/" + str(task_id),
                    candidate_ref])
        if area.mechanism == MECHANISM_COPY:
            # A copy staging has its OWN object store: the
            # candidate must be fetched into the integration
            # repo before the queue can apply it (a worktree
            # shares the store, so there is nothing to move).
            fetched = _git(
                self.repo,
                ["fetch", "--quiet", str(work),
                 "+refs/bench/candidates/" + str(task_id)
                 + ":refs/bench/candidates/" + str(task_id)])
            if fetched.returncode != 0:
                raise BenchError(
                    "candidate transfer into the integration "
                    "repo failed: " + fetched.stderr.strip())
        stat = _git(self.repo, ["diff", "--stat", base_revision,
                                candidate_ref])
        return {"candidate_ref": candidate_ref,
                "files": sorted(files),
                "stat": stat.stdout.strip() if stat.returncode == 0
                else "",
                "empty": False, "base_revision": base_revision}

    # -- the integration repo (promoted tree) ---------------------------

    def head(self) -> str:
        proc = _git(self.repo, ["rev-parse", "HEAD"])
        if proc.returncode != 0:
            raise BenchError("integration repo HEAD unreadable: "
                             + proc.stderr.strip())
        return proc.stdout.strip()

    def tree_of(self, revision: str) -> str:
        proc = _git(self.repo, ["rev-parse",
                                str(revision) + "^{tree}"])
        return proc.stdout.strip() if proc.returncode == 0 else ""

    def commit_message(self, revision: str) -> str:
        proc = _git(self.repo, ["log", "-1", "--format=%B",
                                str(revision)])
        return proc.stdout if proc.returncode == 0 else ""

    def commit_parent(self, revision: str) -> str:
        proc = _git(self.repo, ["rev-list", "--parents", "-n", "1",
                                str(revision)])
        if proc.returncode != 0:
            return ""
        parts = proc.stdout.strip().split()
        return parts[1] if len(parts) > 1 else ""

    def read_at_revision(self, revision: str,
                         relpath: str) -> bytes:
        """Bytes of one file at one revision of the integration
        repo — the only way bench code reads promoted content."""
        proc = _git_bytes(self.repo, ["show",
                                      str(revision) + ":"
                                      + str(relpath)])
        if proc.returncode != 0:
            raise BenchError(
                "not present at revision " + str(revision)[:12]
                + ": " + str(relpath))
        return proc.stdout

    def apply_candidate(self, candidate_ref: str) -> dict[str, Any]:
        """Stage a candidate's diff onto the promoted tree
        (``cherry-pick --no-commit`` — nothing is committed until
        the queue's smoke has passed and it says so).

        Outcomes: ``{"landed": True, "staged": True}`` (diff is
        staged, awaiting commit), ``{"landed": True,
        "already_present": True}`` (the candidate's content is
        already the tree's content — nothing to stage), or
        ``{"conflicted": True, "files", "output", "reason"}``
        with the tree restored to HEAD untouched."""
        head = self.head()
        proc = _git(self.repo, ["cherry-pick", "--no-commit",
                                str(candidate_ref)])
        unmerged = _git(self.repo, ["diff", "--name-only",
                                    "--diff-filter=U"])
        unmerged_files = [line.strip()
                          for line in unmerged.stdout.splitlines()
                          if line.strip()]
        if unmerged_files:
            output = (proc.stderr or proc.stdout or "").strip()
            self._abort_pick(head)
            return {"conflicted": True, "files": unmerged_files,
                    "output": output[-2000:],
                    "reason": "apply_conflict"}
        staged = _git(self.repo, ["diff", "--cached", "--name-only"])
        staged_files = [line.strip()
                        for line in staged.stdout.splitlines()
                        if line.strip()]
        if not staged_files:
            self._abort_pick(head)
            if self.tree_of(candidate_ref) \
                    and self.tree_of(candidate_ref) \
                    == self.tree_of(head):
                return {"landed": True, "already_present": True,
                        "head": head}
            return {"conflicted": True, "files": [],
                    "output": (proc.stderr
                               or proc.stdout or "").strip()[-2000:],
                    "reason": "apply_failed"}
        return {"landed": True, "staged": True, "head": head,
                "files": staged_files}

    def commit_staged(self, message: str) -> str:
        """Commit the staged candidate as the new promoted
        revision and return its hash."""
        proc = _git(self.repo, ["commit", "-q", "-F", "-"],
                    identity=True, stdin=message)
        if proc.returncode != 0:
            raise BenchError("promotion commit failed: "
                             + proc.stderr.strip())
        return self.head()

    def _abort_pick(self, head: str) -> None:
        _git(self.repo, ["cherry-pick", "--abort"])
        _git(self.repo, ["reset", "--hard", "-q", head])

    def reset_tree(self, revision: str) -> None:
        """Restore the promoted tree to a journaled revision —
        the recovery move when the tree and the journal disagree
        (an unjournaled partial apply is discarded, never
        adopted silently)."""
        _git(self.repo, ["cherry-pick", "--abort"])
        _git(self.repo, ["reset", "--hard", "-q", str(revision)])


class BenchViews:
    """The three read views a cursor may resolve (dossier §2.3).
    Constructed over the store + staging manager; every method
    either returns promoted content, the requester's own staging,
    or a promoted dependency's artifact — a peer's staging has no
    route here at all."""

    def __init__(self, store: Any, staging: StagingManager) -> None:
        self.store = store
        self.staging = staging

    def promoted_revision(self, run_id: str) -> str:
        run = self.store.get_run(run_id)
        if run is None:
            raise BenchError("unknown bench run: " + str(run_id))
        return str(run["promoted_revision"] or "")

    def promoted_tree_path(self) -> Path:
        """Where the promoted tree lives (read at the promoted
        revision; its HEAD is the journal's word)."""
        return self.staging.repo

    def read_promoted(self, run_id: str, relpath: str, *,
                      revision: str = "") -> bytes:
        rev = str(revision or "") or self.promoted_revision(run_id)
        return self.staging.read_at_revision(rev, relpath)

    def own_staging_path(self, run_id: str, key: str,
                         cursor_id: str) -> Path:
        """The requester's OWN staging dir — and only theirs.
        The holder check is against the store's task row, so a
        peer's staging cannot be reached through this view by
        any argument."""
        task = self.store.get_task(run_id, key)
        if task is None:
            raise BenchError("unknown bench task: " + str(key))
        if task["holder"] != str(cursor_id):
            raise BenchError(
                "staging for task " + str(key)
                + " belongs to its claimant, not to "
                + str(cursor_id))
        area = self.staging.area(run_id, task["id"])
        if area is None or area.state != _STATE_ACTIVE:
            raise BenchError(
                "no active staging area for task " + str(key))
        return area.path

    def dependency_artifact(self, run_id: str, dep_key: str,
                            relpath: str) -> bytes:
        """One file from a completed dependency, read at the
        exact revision its promotion landed — the journal's
        ``task_promoted`` event is the record of that revision.
        An unpromoted dep has no artifact to read, by rule."""
        task = self.store.get_task(run_id, dep_key)
        if task is None:
            raise BenchError("unknown bench task: " + str(dep_key))
        if task["state"] != TASK_PROMOTED:
            raise BenchError(
                "dependency " + str(dep_key) + " is not promoted"
                " (state: " + str(task["state"]) + ")")
        revision = ""
        for event in self.store.events(run_id, task["id"],
                                       "task_promoted"):
            revision = str(event["detail"].get("revision") or revision)
        if not revision:
            raise BenchError(
                "no promotion record for dependency " + str(dep_key))
        return self.staging.read_at_revision(revision, relpath)
