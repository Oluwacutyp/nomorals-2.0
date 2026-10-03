"""Patch review/apply/preview/record for the code workspace.

Parsing and in-memory application reuse the canonical engine in
:mod:`nomorals.core.diff` — nothing here reimplements diff application.
What this module adds is the file-level view on top of it: per-file
statistics, disk application with path-escape refusal, before/after
snippets, and artifact recording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core.diff import DiffApplyError, apply_unified_diff
from ..core.events import Event, global_bus
from ..core.logging_setup import get_logger
from ..storage.artifacts import Provenance
from .workspace import WorkspaceError

__all__ = ["review_patch", "apply_patch", "preview_patch", "record_patch"]

_log = get_logger(__name__)


def _emit(topic: str, data: dict[str, Any]) -> None:
    """Publish a telemetry event. Best-effort: a broken bus or subscriber
    must never break patching (fail-open telemetry, fail-closed function)."""
    try:
        global_bus.publish(Event(topic=topic, data=data, source=__name__))
    except Exception:  # noqa: BLE001 - telemetry is fail-open
        _log.debug("event %s failed", topic, exc_info=True)

_DEV_NULL = "/dev/null"
_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_CONTEXT_LINES = 3


def _clean_diff_path(raw: str) -> str:
    p = raw.split("\t", 1)[0].strip().strip('"')
    if p.startswith(("a/", "b/")) and p != _DEV_NULL:
        p = p[2:]
    return p


@dataclass
class _FileSection:
    """One ``---``/``+++`` file section of a unified diff."""
    old: str
    new: str
    text: str


def _split_sections(diff_text: str) -> list[_FileSection]:
    """Split a unified diff into per-file sections.

    Only real ``---``/``+++`` header pairs start a section: hunk body lines
    always carry their `` ``/``-``/``+`` prefix, so content can never be
    mistaken for a header.
    """
    sections: list[_FileSection] = []
    lines = diff_text.split("\n")
    start: int | None = None
    old = new = ""
    for i, line in enumerate(lines):
        if (line.startswith("--- ") and i + 1 < len(lines)
                and lines[i + 1].startswith("+++ ")):
            if start is not None:
                sections.append(_FileSection(old, new, "\n".join(lines[start:i])))
            old = _clean_diff_path(line[4:])
            new = _clean_diff_path(lines[i + 1][4:])
            start = i
    if start is not None:
        sections.append(_FileSection(old, new, "\n".join(lines[start:])))
    return sections


def _target_of(section: _FileSection) -> tuple[str, bool]:
    """Return (target path, is_deletion) for a file section."""
    if section.new == _DEV_NULL:
        return section.old, True
    return section.new, False


def _refuse_escape(root: Path, relpath: str) -> Path:
    """Resolve ``relpath`` under ``root``; raise unless it stays inside."""
    root = root.resolve()
    resolved = (root / relpath).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise WorkspaceError(
            f"refusing to write outside workspace root: {relpath!r}") from None
    return resolved


def review_patch(diff_text: str) -> dict[str, Any]:
    """Per-file statistics for a unified diff.

    Parses ``---``/``+++``/``@@`` headers itself and counts added/removed
    lines per hunk.  Binary sections report 0/0.
    """
    sections = _split_sections(diff_text)
    files: list[dict[str, Any]] = []
    total_add = total_del = 0
    for section in sections:
        if section.old == _DEV_NULL:
            status = "added"
        elif section.new == _DEV_NULL:
            status = "deleted"
        else:
            status = "modified"
        additions = deletions = hunks = 0
        for line in section.text.split("\n"):
            if _HUNK_RE.match(line):
                hunks += 1
            elif line.startswith(("+++", "---")):
                continue  # the section's own headers
            elif line.startswith("+"):
                additions += 1
            elif line.startswith("-"):
                deletions += 1
        path, _ = _target_of(section)
        files.append({"path": path, "status": status, "additions": additions,
                      "deletions": deletions, "hunks": hunks})
        total_add += additions
        total_del += deletions
    return {
        "files": files,
        "totals": {"files": len(files), "additions": total_add,
                   "deletions": total_del},
    }


def _read_targets(root: Path, sections: list[_FileSection]) -> dict[str, str | None]:
    """Current on-disk text per section target (None when absent)."""
    texts: dict[str, str | None] = {}
    for section in sections:
        target, _ = _target_of(section)
        if target in texts:
            continue
        _refuse_escape(root, target)
        p = root / target
        texts[target] = p.read_text(encoding="utf-8") if p.is_file() else None
    return texts


def apply_patch(diff_text: str, *, dry_run: bool = True,
                root: str | Path = ".") -> list[dict[str, Any]]:
    """Apply a unified diff per file via :mod:`nomorals.core.diff`.

    ``dry_run=True`` applies to in-memory copies and never touches disk;
    ``dry_run=False`` writes the results, creating parent directories.
    Any target escaping ``root`` raises :class:`WorkspaceError` immediately.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise WorkspaceError(f"not a directory: {base}")
    sections = _split_sections(diff_text)
    current = _read_targets(base, sections)  # raises WorkspaceError on escape
    results: list[dict[str, Any]] = []
    for section in sections:
        target, _is_deletion = _target_of(section)
        try:
            patched = apply_unified_diff(section.text, {target: current[target]})
        except DiffApplyError as exc:
            results.append({"path": target, "ok": False, "error": str(exc)})
            continue
        new_text = patched[target]
        if not dry_run:
            try:
                dest = _refuse_escape(base, target)
                if new_text is None:
                    if dest.is_file():
                        dest.unlink()
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(new_text, encoding="utf-8")
            except (OSError, WorkspaceError) as exc:
                results.append({"path": target, "ok": False,
                                "error": f"write failed: {exc}"})
                continue
        results.append({"path": target, "ok": True, "error": ""})
    if not dry_run:
        patched = sum(1 for r in results if r["ok"])
        failed = len(results) - patched
        _emit("codews.patch.applied", {
            "root": str(base),
            "dry_run": False,
            "files": [r["path"] for r in results],
            "patched": patched,
            "failed": failed,
        })
    return results


def _changed_lines(section: _FileSection) -> tuple[set[int], set[int]]:
    """Old-side and new-side (1-based) line numbers the section changes."""
    old_changed: set[int] = set()
    new_changed: set[int] = set()
    old_ln = new_ln = 0
    in_hunk = False
    for line in section.text.split("\n"):
        m = _HUNK_RE.match(line)
        if m:
            old_ln, new_ln = int(m.group(1)), int(m.group(3))
            in_hunk = True
            continue
        if not in_hunk or line.startswith("\\"):
            continue
        if line.startswith(" "):
            old_ln += 1
            new_ln += 1
        elif line.startswith("-"):
            old_changed.add(old_ln)
            old_ln += 1
        elif line.startswith("+"):
            new_changed.add(new_ln)
            new_ln += 1
    return old_changed, new_changed


def _windowed(lines: list[str], changed: set[int],
              mark: set[int], marker: str) -> str:
    """Render ``lines`` around ``changed`` line numbers with a gutter.

    ``mark`` is the subset of shown lines that changed (``marker`` = the
    per-line prefix for changed lines, e.g. ``-`` or ``+``).
    """
    if not changed:
        return ""
    keep: set[int] = set()
    for ln in changed:
        for i in range(max(1, ln - _CONTEXT_LINES), min(len(lines), ln + _CONTEXT_LINES) + 1):
            keep.add(i)
    shown = sorted(keep)
    out: list[str] = []
    prev = None
    for ln in shown:
        if prev is not None and ln > prev + 1:
            out.append("...")
        prefix = marker if ln in mark else " "
        out.append(f"{prefix} {ln:>4}  {lines[ln - 1]}")
        prev = ln
    return "\n".join(out)


def preview_patch(diff_text: str, path: str,
                  *, root: str | Path = ".") -> dict[str, str]:
    """Before/after snippets around the changed lines for one file."""
    base = Path(root).expanduser().resolve()
    section = next((s for s in _split_sections(diff_text)
                    if _target_of(s)[0] == path), None)
    if section is None:
        raise WorkspaceError(f"path {path!r} is not present in the diff")
    target, _ = _target_of(section)
    dest = _refuse_escape(base, target)
    exists = dest.is_file()
    before_text = dest.read_text(encoding="utf-8") if exists else ""
    try:
        patched = apply_unified_diff(
            section.text, {target: before_text if exists else None})
    except DiffApplyError as exc:
        raise WorkspaceError(f"cannot preview {path!r}: {exc}") from exc
    after_text = patched[target] or ""
    old_changed, new_changed = _changed_lines(section)
    before_lines = before_text.split("\n")
    after_lines = after_text.split("\n")
    return {
        "before": _windowed(before_lines, old_changed, old_changed, "-"),
        "after": _windowed(after_lines, new_changed, new_changed, "+"),
    }


def record_patch(store: Any, diff_text: str, *, mission_id: str = "",
                 creator: str = "codews") -> str:
    """Store a diff as a ``patch`` artifact; return its ``artifact://`` URI."""
    stats = review_patch(diff_text)
    art = store.put_text(
        diff_text,
        type="patch",
        creator=creator,
        mission_id=mission_id,
        metadata={"files": [f["path"] for f in stats["files"]],
                  "totals": stats["totals"]},
        provenance=Provenance(source_type="codews"),
    )
    return art.uri
