"""Unified-diff application for coding missions (L5 agent helper).

On-disk, per-file all-or-nothing application with per-hunk failure
reports.  The parse/apply ENGINE is canonical in
``nomorals.tools.edit_loop`` (differential-fuzzed 500/500 against GNU
``patch``: count-driven hunk parsing, patch-style offset tolerance,
``\\ No newline`` handling); this module is the mission-facing adapter
on top of it: path-escape rejection, existence checks, and structured
``{"file", "hunk", "reason"}`` failure reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.diff import (
    _DEV_NULL,
    _apply_parsed_diff,
    _parse_unified_diff,
    DiffApplyError,
)

__all__ = ["apply_unified_diff", "parse_unified_diff", "FilePatch", "Hunk"]


@dataclass
class Hunk:
    """One ``@@`` hunk: raw body lines including their `` /-/+`` markers."""
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str] = field(default_factory=list)
    no_trailing_newline: bool = False


@dataclass
class FilePatch:
    """All hunks for one file in the patch."""
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)

    @property
    def is_new(self) -> bool:
        return self.old_path == _DEV_NULL

    @property
    def is_deleted(self) -> bool:
        return self.new_path == _DEV_NULL

    @property
    def target_rel(self) -> str:
        """Repo-relative path this patch writes to."""
        return self.new_path if not self.is_deleted else self.old_path


def parse_unified_diff(patch_text: str) -> list[FilePatch]:
    """Parse standard unified-diff text into per-file patches.

    Raises :class:`ValueError` on structurally broken patches.
    Returns an empty list when the text contains no file headers at all.
    """
    out: list[FilePatch] = []
    for fp in _parse_unified_diff(patch_text or ""):
        hunks = [
            Hunk(
                old_start=h.old_start,
                old_len=h.old_count,
                new_start=h.new_start,
                new_len=h.new_count,
                lines=[f"{kind}{text}" for kind, text, _term in h.body],
                no_trailing_newline=bool(h.body) and not h.body[-1][2],
            )
            for h in fp.hunks
        ]
        out.append(FilePatch(old_path=fp.old_path, new_path=fp.new_path,
                             hunks=hunks))
    return out


def _resolve_under(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``; reject escapes (fail fast)."""
    if not rel or rel == _DEV_NULL:
        raise ValueError(f"patch: bad target path {rel!r}")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"patch: path {rel!r} escapes root {root}") from exc
    return candidate


def apply_unified_diff(patch_text: str, root: str | Path) -> dict[str, Any]:
    """Apply a unified diff under ``root``.

    Returns ``{"applied_files": [...], "failed_hunks": [...]}`` where each
    failed hunk is ``{"file", "hunk" (1-based within the file), "reason"}``.
    Files are all-or-nothing: a file whose hunks all match is written;
    a file with any failing hunk is left untouched.

    Raises :class:`ValueError` on a structurally broken patch or a target
    path that escapes ``root``.
    """
    root = Path(root).expanduser().resolve()
    result: dict[str, Any] = {"applied_files": [], "failed_hunks": []}

    def _fail(rel: str, hunk: int, reason: str) -> None:
        result["failed_hunks"].append(
            {"file": rel, "hunk": hunk, "reason": reason})

    try:
        patches = _parse_unified_diff(patch_text or "")
    except DiffApplyError as exc:
        raise ValueError(str(exc)) from exc

    for fp in patches:
        rel = fp.new_path if fp.new_path != _DEV_NULL else fp.old_path
        try:
            target = _resolve_under(root, rel)
        except ValueError as exc:
            _fail(rel, 0, str(exc))
            continue
        if fp.old_path == _DEV_NULL:
            if target.exists():
                _fail(rel, 0, "new file already exists")
                continue
            current: str | None = None
        else:
            if not target.is_file():
                _fail(rel, 0, "target file does not exist")
                continue
            current = target.read_text(encoding="utf-8")
        try:
            new_map = _apply_parsed_diff([fp], {rel: current})
        except DiffApplyError as exc:
            _fail(rel, exc.hunk or 0, str(exc))
            continue
        new_text = new_map[rel]
        if new_text is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(new_text, encoding="utf-8")
        result["applied_files"].append(rel)
    return result
