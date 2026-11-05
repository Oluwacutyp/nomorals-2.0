"""Landability scoring (Echelon Wave 8).

Given a diff (a git range, a branch, or a patch file), score whether
the change can land — as comparable data, not a vibe.  The score starts
at 100 and loses points for the things that have actually burned this
repo; three classes of finding are hard blocks that no score survives:

* **frozen-file touch** — ``nomorals/storage/migrations.py``: its
  migration functions are frozen by the AST tamper guard; editing one
  bricks every install that already applied it.
* **persona-file touch** — ``nomorals/memory/persona.py``: the persona
  is owner-owned content; a diff that changes it is never a routine
  landing.
* **secret-shaped addition** — an added line matching the security
  battery (:mod:`nomorals.security.audit_patterns`): secrets land in
  history forever.

Deductions, each named and measured:

* no test file in the diff (−20) — untested change;
* silent-failure shapes in added lines (−10 each, capped −30): bare
  ``except:`` and ``except …: pass`` — the swallow-empty defect class;
* oversized diff (>800 added+deleted lines, −10) — review load.

Status: ``blocked`` (any hard block), ``needs_work`` (score < 80),
else ``landable``.  Callers may attach free-text ANNOTATIONS (a
reviewer's notes); they are echoed in the report and NEVER change the
score or the status — the tool measures the diff, it does not grade
the commentary.  That boundary is pinned by tests.

Git invocations that fail return ``ok=False`` with the git error —
never a fabricated clean score.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from ..core.logging_setup import get_logger
from ..security.audit_patterns import scan_text_for_secrets

_log = get_logger(__name__)

__all__ = [
    "FROZEN_FILES",
    "OVERSIZED_DIFF_LINES",
    "score_patch",
    "score_range",
    "register",
    "main",
]

#: Files whose touch is a hard block, with the reason.
FROZEN_FILES: dict[str, str] = {
    "nomorals/storage/migrations.py": (
        "migration functions are FROZEN by the AST tamper guard — "
        "editing an applied migration bricks existing installs"),
    "nomorals/memory/persona.py": (
        "the persona file is owner-owned content; changing it is never "
        "a routine landing"),
}

#: Diff size above which a review-load deduction applies.
OVERSIZED_DIFF_LINES = 800

_NO_TESTS_DEDUCTION = 25
_SILENT_FAILURE_DEDUCTION = 10
_SILENT_FAILURE_CAP = 30
_OVERSIZED_DEDUCTION = 10
_LANDABLE_FLOOR = 80


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parents[2]


def _parse_unified_diff(text: str) -> dict[str, Any]:
    """Parse a unified diff into per-file added-line data.

    Returns {files: {path: {added: int, deleted: int, binary: bool}},
    added_lines: {path: [str]}}.  Rename/copy headers attribute the
    change to the destination path.
    """
    files: dict[str, dict[str, Any]] = {}
    added_lines: dict[str, list[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" ")
            current = parts[3][2:] if len(parts) >= 4 else None
            if current:
                files.setdefault(current, {"added": 0, "deleted": 0,
                                           "binary": False})
                added_lines.setdefault(current, [])
            continue
        if current is None:
            continue
        if line.startswith("Binary files"):
            files[current]["binary"] = True
            continue
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            files[current]["added"] += 1
            added_lines[current].append(line[1:])
        elif line.startswith("-"):
            files[current]["deleted"] += 1
    return {"files": files, "added_lines": added_lines}


def _silent_failure_shape(line: str) -> str | None:
    """Name the silent-failure shape on one added line, if any."""
    stripped = line.strip()
    if stripped in ("except:", "except :"):
        return "bare except clause"
    if stripped.startswith("except") and stripped.endswith(": pass"):
        return "except clause with pass body (swallowed failure)"
    return None


def score_patch(patch_text: str, *,
                annotations: list[str] | None = None) -> dict[str, Any]:
    """Score a unified-diff patch text (pure function of the diff).

    ``annotations`` are echoed verbatim in the report and have no
    effect on score or status — by design, and pinned by tests.
    """
    parsed = _parse_unified_diff(patch_text)
    files: dict[str, dict[str, Any]] = parsed["files"]
    added_lines: dict[str, list[str]] = parsed["added_lines"]
    hard_blocks: list[dict[str, Any]] = []
    deductions: list[dict[str, Any]] = []

    for path, reason in FROZEN_FILES.items():
        if path in files:
            hard_blocks.append({"kind": "frozen_file", "path": path,
                                "reason": reason})

    secret_hits = 0
    for path, lines in added_lines.items():
        text = "\n".join(lines)
        for hit in scan_text_for_secrets(text):
            secret_hits += 1
            hard_blocks.append({
                "kind": "secret_in_diff", "path": path,
                "pattern": hit.pattern,
                "value_redacted": hit.redacted,
                "reason": "added lines carry a secret-shaped value",
            })

    has_tests = any(
        path.startswith("tests/") and path.endswith(".py")
        for path in files)
    if files and not has_tests:
        deductions.append({
            "kind": "no_tests", "points": _NO_TESTS_DEDUCTION,
            "reason": "no test file in the diff"})

    silent_shapes = 0
    for path, lines in added_lines.items():
        if not path.endswith(".py"):
            continue
        for line in lines:
            shape = _silent_failure_shape(line)
            if shape:
                silent_shapes += 1
                if silent_shapes * _SILENT_FAILURE_DEDUCTION \
                        <= _SILENT_FAILURE_CAP:
                    deductions.append({
                        "kind": "silent_failure_shape", "path": path,
                        "shape": shape,
                        "points": _SILENT_FAILURE_DEDUCTION,
                        "reason": f"{shape} in added lines"})

    total_lines = sum(f["added"] + f["deleted"] for f in files.values())
    if total_lines > OVERSIZED_DIFF_LINES:
        deductions.append({
            "kind": "oversized_diff", "points": _OVERSIZED_DEDUCTION,
            "reason": (f"{total_lines} changed lines exceeds the "
                       f"{OVERSIZED_DIFF_LINES}-line review-load bar")})

    score = max(0, 100 - sum(d["points"] for d in deductions))
    if hard_blocks:
        status = "blocked"
    elif score < _LANDABLE_FLOOR:
        status = "needs_work"
    else:
        status = "landable"
    return {
        "ok": True,
        "status": status,
        "score": score,
        "hard_blocks": hard_blocks,
        "deductions": deductions,
        "files_changed": len(files),
        "files": {p: files[p] for p in sorted(files)},
        "lines_added": sum(f["added"] for f in files.values()),
        "lines_deleted": sum(f["deleted"] for f in files.values()),
        "has_tests": has_tests,
        "secret_hits": secret_hits,
        "silent_failure_shapes": silent_shapes,
        "annotations": list(annotations or []),
    }


def _git(repo: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300)


def score_range(repo: str | Path | None = None, *,
                base: str = "HEAD~1", head: str = "HEAD",
                annotations: list[str] | None = None) -> dict[str, Any]:
    """Score the diff ``base..head`` of a checkout (or the working
    branch against ``base`` when ``head`` is a branch name)."""
    root = Path(repo).resolve() if repo else repo_root()
    proc = _git(root, ["diff", f"{base}...{head}"])
    if proc.returncode != 0:
        return {"ok": False,
                "detail": (f"git diff {base}...{head} failed: "
                           f"{(proc.stderr or '').strip()[:400]}")}
    result = score_patch(proc.stdout, annotations=annotations)
    result["base"] = base
    result["head"] = head
    result["repo"] = str(root)
    return result


def register(registry: Any) -> None:
    """Attach landability scoring to a registry."""
    from ..core.policy import Capability

    @registry.register(
        "landability",
        description=("Score whether a diff can land. Params: base, "
                     "head (git range in this checkout) or patch (raw "
                     "unified diff text); annotations are echoed, never "
                     "scored. Returns {status: landable|needs_work|"
                     "blocked, score, hard_blocks, deductions, "
                     "has_tests}."),
        capability=Capability.FS_READ,
    )
    def _landability(base: str = "HEAD~1", head: str = "HEAD",
                     patch: str | None = None,
                     annotations: list[str] | None = None,
                     ) -> dict[str, Any]:
        if patch is not None:
            return score_patch(patch, annotations=annotations)
        return score_range(base=base, head=head,
                           annotations=annotations)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="nomorals.tools.landability",
        description="Landability scoring for a diff.")
    parser.add_argument("--repo", default=None)
    parser.add_argument("--base", default="HEAD~1")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--patch-file", default=None,
                        help="score a unified diff from this file instead")
    parser.add_argument("--annotate", action="append", default=[],
                        help="reviewer annotation (echoed, never scored)")
    args = parser.parse_args(argv)
    if args.patch_file:
        result = score_patch(
            Path(args.patch_file).read_text(encoding="utf-8"),
            annotations=args.annotate)
    else:
        result = score_range(repo=args.repo, base=args.base,
                             head=args.head, annotations=args.annotate)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("ok"):
        return 2
    return 0 if result["status"] == "landable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
