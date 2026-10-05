"""Automated git-bisect regression hunting (L5 agent helper).

``bisect_regression`` drives ``git bisect start/bad/good`` itself and runs
a caller-supplied test command at every step, so the coding agent can find
the exact commit that introduced a regression without hand-holding.

Exit-code contract for ``test_cmd`` (mirrors ``git bisect run``):

* ``0``        -> good
* ``125``      -> skip (cannot judge this commit)
* anything else -> bad
* timeout      -> skip (a hang is not evidence either way)
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger

_log = get_logger(__name__)

__all__ = ["bisect_regression", "BisectError"]

#: "git bisect" prints this on the step that concludes the search.
_FIRST_BAD = re.compile(r"^([0-9a-f]{5,40}) is the first bad commit", re.MULTILINE)

#: Safety bound so a confused bisect can never loop forever.
_MAX_STEPS = 128


class BisectError(RuntimeError):
    """The bisect itself could not run (not a repo, dirty tree, git error)."""


def _git(repo: Path, *args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True,
        timeout=timeout)


def _run_test(repo: Path, test_cmd: str | list[str], timeout: float) -> str:
    """Run the test command at the current checkout; return good/bad/skip."""
    try:
        proc = subprocess.run(
            test_cmd, cwd=repo, capture_output=True, text=True,
            timeout=timeout, shell=isinstance(test_cmd, str))
    except subprocess.TimeoutExpired:
        _log.warning("bisect: test command timed out — skipping commit")
        return "skip"
    except OSError as exc:
        raise BisectError(f"cannot run test command: {exc}") from exc
    if proc.returncode == 0:
        return "good"
    if proc.returncode == 125:
        return "skip"
    return "bad"


def bisect_regression(
    repo: str | Path,
    good_ref: str,
    bad_ref: str,
    test_cmd: str | list[str],
    *,
    timeout: float = 300.0,
) -> dict[str, Any] | None:
    """Find the commit that introduced a regression.

    ``good_ref`` is a ref known to pass ``test_cmd``; ``bad_ref`` is a ref
    known to fail it.  Returns ``{"hash", "subject", "author", "date"}`` for
    the culprit commit, or ``None`` when there is no regression to find
    (``bad_ref`` already passes, or every candidate had to be skipped).

    Raises :class:`BisectError` when the repo cannot be bisected (not a
    git repo, dirty working tree, bisect already in progress, git failure).
    The working tree is always restored: ``git bisect reset`` runs in a
    ``finally`` block and the original HEAD is checked back out.
    """
    root = Path(repo).expanduser().resolve()
    proc = _git(root, "rev-parse", "--is-inside-work-tree")
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        raise BisectError(f"not a git repository: {root}")
    if (root / ".git" / "BISECT_LOG").exists():
        raise BisectError("git bisect already in progress in "
                          f"{root} — reset it first")
    if _git(root, "status", "--porcelain").stdout.strip():
        raise BisectError("working tree is dirty — commit or stash first; "
                          "bisect refuses to run on a dirty tree")

    orig = _git(root, "rev-parse", "HEAD").stdout.strip()
    if not orig:
        raise BisectError("cannot resolve HEAD")

    def _restore() -> None:
        try:
            _git(root, "checkout", "-q", orig)
        except Exception as exc:  # noqa: BLE001 — restore is best-effort
            _log.error("bisect: failed to restore %s: %s", orig, exc)

    # Sanity check first: if bad_ref already passes, there is no
    # regression and starting a bisect would be nonsense.
    _git(root, "checkout", "-q", bad_ref)
    verdict = _run_test(root, test_cmd, timeout)
    if verdict == "good":
        _log.info("bisect: %s already passes — no regression", bad_ref)
        _restore()
        return None
    if verdict == "skip":
        _restore()
        raise BisectError(f"test command cannot judge {bad_ref} "
                          "(exit 125 / timeout)")

    culprit: str | None = None
    try:
        for cmd in (("bisect", "start"),
                    ("bisect", "bad", bad_ref),
                    ("bisect", "good", good_ref)):
            proc = _git(root, *cmd)
            if proc.returncode != 0:
                raise BisectError(
                    f"'git {' '.join(cmd)}' failed: "
                    f"{proc.stderr.strip() or proc.stdout.strip()}")
        for _ in range(_MAX_STEPS):
            cur = _git(root, "rev-parse", "HEAD").stdout.strip()
            verdict = _run_test(root, test_cmd, timeout)
            proc = _git(root, "bisect", verdict)
            out = proc.stdout + "\n" + proc.stderr
            match = _FIRST_BAD.search(out)
            if match:
                culprit = match.group(1)
                break
            if "only 'skip'ped commits left" in out:
                _log.warning("bisect: only skipped commits remain — "
                             "no culprit identifiable")
                break
            if proc.returncode != 0 and "Bisecting:" not in out:
                raise BisectError(f"'git bisect {verdict}' failed at {cur}: "
                                  f"{proc.stderr.strip()}")
        else:
            raise BisectError(f"bisect did not converge in {_MAX_STEPS} steps")
    finally:
        # Always leave the repo exactly as we found it: no bisect state,
        # original HEAD checked out.
        reset = _git(root, "bisect", "reset")
        if reset.returncode != 0:
            _log.error("bisect: 'git bisect reset' failed: %s",
                       reset.stderr.strip())
        _restore()

    if culprit is None:
        return None
    show = _git(root, "show", "-s", "--format=%H%n%an%n%ad%n%s",
                "--date=iso-strict", culprit)
    if show.returncode != 0:
        raise BisectError(f"cannot describe culprit {culprit}: "
                          f"{show.stderr.strip()}")
    lines = show.stdout.splitlines()
    return {
        "hash": lines[0] if len(lines) > 0 else culprit,
        "author": lines[1] if len(lines) > 1 else "",
        "date": lines[2] if len(lines) > 2 else "",
        "subject": lines[3] if len(lines) > 3 else "",
    }


def register(registry: Any) -> None:
    """Expose git-bisect regression hunting as an agent tool."""

    @registry.register(
        "bisect_regression",
        description=(
            "Find the commit that introduced a regression via git bisect. "
            "good_ref is a ref known to pass test_cmd; bad_ref is a ref known "
            "to fail it. Returns the culprit commit (hash, author, date, subject)."
        ),
        capability="code.bisect",
        parameters={
            "repo": "str — path to the git repository",
            "good_ref": "str — git ref known to pass (e.g. commit hash or tag)",
            "bad_ref": "str — git ref known to fail",
            "test_cmd": "str — test command; exit 0=good, 125=skip, else=bad",
            "timeout": "float — per-step timeout seconds (default 300)",
        },
    )
    def _bisect_regression(
        repo: str,
        good_ref: str,
        bad_ref: str,
        test_cmd: str,
        timeout: float = 300.0,
    ) -> dict[str, Any]:
        try:
            result = bisect_regression(
                repo, good_ref, bad_ref, test_cmd, timeout=timeout
            )
            return {"ok": True, "culprit": result}
        except BisectError as exc:
            return {"ok": False, "error": str(exc)}
