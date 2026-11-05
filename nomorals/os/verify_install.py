"""Verify-then-install (Echelon Wave 8).

The push lesson, generalized: a tool that moves code must verify what
actually LANDED before claiming success — a transport's "VERIFIED" /
"pull succeeded" line is not proof.  ``UpdateManager`` uses this module
after a pull moves HEAD: it greps the resulting state (HEAD identity,
ancestry, object presence, working-tree content, blob sizes) and only
then lets the update proceed.  On any failure the caller rolls the
tree back; a failed verification is never papered over.

The checks, each reported as data:

``head_matches``
    ``rev-parse HEAD`` equals the sha the pull claims it landed.
``descendant``
    The landed sha descends from the pre-update sha (no silent
    sideways/rewrite move).  Skipped with a named reason on a first
    install (no previous sha).
``objects_present``
    Every object in the landed tree exists in the object store
    (``git cat-file --batch-check`` over ``ls-tree -r``) — a partial
    clone/fetch leaves exactly this shape of hole.
``worktree_matches``
    Every file the update changed hashes to the blob HEAD records for
    it, and every file the update deleted is actually gone — the
    working tree is what will run, so it is what gets verified.
``no_oversized_new_blobs``
    No blob NEW in the landed tree exceeds ``max_new_blob_bytes``
    (the multi-MB blob that once rode a push unnoticed).  Skipped with
    a named reason when there is no previous tree to diff against.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = [
    "DEFAULT_MAX_NEW_BLOB_BYTES",
    "verify_install",
]

#: Default cap for a single new blob introduced by an update (64 MiB).
DEFAULT_MAX_NEW_BLOB_BYTES = 64 * 1024 * 1024


def _git(repo: Path, git_bin: str, args: list[str],
         input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [git_bin, "-C", str(repo), *args],
        input=input_text, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=600)


def _resolve(repo: Path, git_bin: str, ref: str) -> str:
    proc = _git(repo, git_bin, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _tree_blobs(repo: Path, git_bin: str, sha: str) -> dict[str, tuple[str, int]]:
    """path → (blob sha, size) for one tree."""
    proc = _git(repo, git_bin, ["ls-tree", "-r", "-l", "-z", sha])
    blobs: dict[str, tuple[str, int]] = {}
    if proc.returncode != 0:
        return blobs
    for entry in proc.stdout.split("\0"):
        if not entry:
            continue
        meta, _, path = entry.partition("\t")
        parts = meta.split()
        if len(parts) >= 4 and parts[1] == "blob":
            try:
                size = int(parts[3])
            except ValueError:
                size = -1
            blobs[path] = (parts[2], size)
    return blobs


def verify_install(
    repo: str | Path,
    expected_sha: str,
    previous_sha: str = "",
    *,
    git_bin: str = "git",
    max_new_blob_bytes: int = DEFAULT_MAX_NEW_BLOB_BYTES,
) -> dict[str, Any]:
    """Verify that ``expected_sha`` really landed in ``repo``.

    Returns ``{ok, repo, expected_sha, previous_sha, checks, detail}``
    where each check is ``{name, ok, skipped, detail}``.  ``ok`` is
    True only when every non-skipped check passes.
    """
    root = Path(repo).resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str,
              skipped: bool = False) -> None:
        checks.append({"name": name, "ok": bool(ok) or skipped,
                       "skipped": skipped, "detail": detail})

    # 1. HEAD is the landed sha.
    head = _resolve(root, git_bin, "HEAD")
    expected = _resolve(root, git_bin, expected_sha) or expected_sha
    check("head_matches", bool(head) and head == expected,
          f"HEAD={head or '<unresolvable>'} expected={expected}")

    # 2. Landed descends from previous.
    if previous_sha:
        previous = _resolve(root, git_bin, previous_sha) or previous_sha
        proc = _git(root, git_bin,
                    ["merge-base", "--is-ancestor", previous, expected])
        check("descendant", proc.returncode == 0,
              f"{previous[:12]} is "
              f"{'' if proc.returncode == 0 else 'NOT '}an ancestor of "
              f"{expected[:12]}")
    else:
        check("descendant", True, "no previous sha (first install)",
              skipped=True)

    # 3. Every object in the landed tree is present.
    landed_blobs = _tree_blobs(root, git_bin, expected)
    if not landed_blobs:
        check("objects_present", False,
              f"landed tree {expected[:12]} lists no objects")
    else:
        want = sorted({sha for sha, _size in landed_blobs.values()})
        proc = _git(root, git_bin, ["cat-file", "--batch-check"],
                    input_text="\n".join(want) + "\n")
        missing: list[str] = []
        for line in (proc.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == "missing":
                missing.append(parts[0])
        check("objects_present", not missing,
              f"{len(want)} blobs in tree; "
              f"{len(missing)} object(s) missing from the store"
              if missing else
              f"all {len(want)} objects of tree {expected[:12]} present")

    # 4. Working tree matches HEAD for everything the update changed.
    if previous_sha and previous_sha != expected_sha:
        proc = _git(root, git_bin,
                    ["diff", "--name-status", "-z",
                     f"{previous_sha}..{expected}"])
        entries = [e for e in proc.stdout.split("\0") if e]
        mismatches: list[str] = []
        i = 0
        while i < len(entries):
            status = entries[i]
            i += 1
            if status.startswith(("R", "C")):
                i += 2  # old path, new path — verify the new path below
                path = entries[i - 1]
                status = "A"
            else:
                path = entries[i]
                i += 1
            target = root / path
            if status.startswith("D"):
                if target.exists():
                    mismatches.append(f"{path}: deleted in HEAD, "
                                      "still on disk")
                continue
            blob = landed_blobs.get(path)
            if blob is None or not target.is_file():
                if blob is not None:
                    mismatches.append(f"{path}: in HEAD, missing on disk")
                continue
            hashed = _git(root, git_bin, ["hash-object", "--", path])
            actual = (hashed.stdout or "").strip()
            if actual != blob[0]:
                mismatches.append(
                    f"{path}: worktree hash {actual[:12]} != "
                    f"HEAD blob {blob[0][:12]}")
        check("worktree_matches", not mismatches,
              "; ".join(mismatches[:5]) if mismatches else
              "every changed file on disk hashes to its HEAD blob")
    else:
        check("worktree_matches", True,
              "no previous sha to diff the worktree against",
              skipped=True)

    # 5. No oversized NEW blobs.
    if previous_sha:
        previous_blobs = _tree_blobs(
            root, git_bin, _resolve(root, git_bin, previous_sha)
            or previous_sha)
        oversized = [
            (path, size) for path, (sha, size) in landed_blobs.items()
            if previous_blobs.get(path, ("", -1))[0] != sha
            and size > max_new_blob_bytes
        ]
        check("no_oversized_new_blobs", not oversized,
              "; ".join(f"{p} is {s} bytes" for p, s in oversized[:5])
              if oversized else
              f"no new blob exceeds {max_new_blob_bytes} bytes")
    else:
        check("no_oversized_new_blobs", True,
              "no previous tree to compare new blobs against",
              skipped=True)

    ok = all(c["ok"] for c in checks)
    failed = [c["name"] for c in checks if not c["ok"]]
    return {
        "ok": ok,
        "repo": str(root),
        "expected_sha": expected,
        "previous_sha": previous_sha,
        "checks": checks,
        "detail": ("install verified: HEAD, ancestry, objects, worktree "
                   "and blob sizes all match what landed"
                   if ok else
                   f"install verification FAILED: {', '.join(failed)}"),
    }
