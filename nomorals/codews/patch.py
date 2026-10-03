"""Patch review/apply/preview/record for the code workspace.

Parsing and in-memory application reuse the canonical engine in
:mod:`nomorals.core.diff` — nothing here reimplements diff application.
What this module adds is the file-level view on top of it: per-file
statistics, disk application with path-escape refusal, before/after
snippets, and artifact recording.

Coverage beyond plain unified diffs:

- file creation (``--- /dev/null``) and deletion (``+++ /dev/null``);
- git renames (``rename from``/``rename to``): applied as read-old,
  patch, write-new, delete-old;
- binary sections (``GIT binary patch`` / ``Binary files ... differ``):
  reported as ``status="binary"`` in reviews and refused in
  :func:`apply_patch` with a clear error — binary content cannot be
  reconstructed textually;
- mode-only sections (``diff --git`` with no hunks) are reported with a
  clear per-file error instead of silently doing nothing.
"""

from __future__ import annotations

import re
import shlex
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


def _parse_diff_git_paths(line: str) -> tuple[str, str] | None:
    """Parse ``diff --git a/old b/new``; None when it cannot be parsed."""
    try:
        parts = shlex.split(line)
    except ValueError:
        return None
    if len(parts) != 4 or parts[0] != "diff" or parts[1] != "--git":
        return None
    return _clean_diff_path(parts[2]), _clean_diff_path(parts[3])


_BINARY_FILES_RE = re.compile(r"^Binary files (\S+) and (\S+) differ$")


def _parse_binary_files_paths(line: str) -> tuple[str, str] | None:
    m = _BINARY_FILES_RE.match(line)
    if not m:
        return None
    return _clean_diff_path(m.group(1)), _clean_diff_path(m.group(2))


@dataclass
class _FileSection:
    """One file section of a diff: git extended headers plus the
    ``---``/``+++`` file header pair (absent for binary/mode-only
    sections)."""
    old: str
    new: str
    text: str
    has_headers: bool = False
    rename_from: str = ""
    rename_to: str = ""
    binary: bool = False


def _split_sections(diff_text: str) -> list[_FileSection]:
    """Split a diff into per-file sections.

    Sections are anchored on ``diff --git`` lines when present (git-style
    diffs) and on ``---``/``+++`` header pairs otherwise (plain unified
    diffs).  Extended headers — ``rename from``/``rename to``,
    ``GIT binary patch`` / ``Binary files ... differ`` markers — are
    captured so callers can handle renames and binary content.  Hunk body
    lines always carry their `` ``/``-``/``+`` prefix, so content can
    never be mistaken for a header.
    """
    sections: list[_FileSection] = []
    lines = diff_text.split("\n")
    n = len(lines)

    cur_old = ""
    cur_new = ""
    cur_start: int | None = None
    cur_headers = False
    cur_rename_from = ""
    cur_rename_to = ""
    cur_binary = False

    def _flush(end: int) -> None:
        nonlocal cur_old, cur_new, cur_start, cur_headers
        nonlocal cur_rename_from, cur_rename_to, cur_binary
        if cur_start is None:
            return
        if not cur_old and not cur_new and not cur_binary:
            # Junk or a mode-only section with no identifiable paths:
            # nothing actionable.
            cur_start = None
            return
        sections.append(_FileSection(
            old=cur_old, new=cur_new,
            text="\n".join(lines[cur_start:end]),
            has_headers=cur_headers,
            rename_from=cur_rename_from, rename_to=cur_rename_to,
            binary=cur_binary))
        cur_old = cur_new = ""
        cur_start = None
        cur_headers = False
        cur_rename_from = cur_rename_to = ""
        cur_binary = False

    def _open(i: int) -> None:
        nonlocal cur_start
        if cur_start is None:
            cur_start = i

    i = 0
    while i < n:
        line = lines[i]
        if line.startswith("diff --git "):
            _flush(i)
            parsed = _parse_diff_git_paths(line)
            if parsed:
                cur_old, cur_new = parsed
            _open(i)
        elif line.startswith("rename from "):
            cur_rename_from = _clean_diff_path(line[len("rename from "):])
            _open(i)
        elif line.startswith("rename to "):
            cur_rename_to = _clean_diff_path(line[len("rename to "):])
            _open(i)
        elif (line.startswith("GIT binary patch")
              or line.startswith("Binary files ")):
            cur_binary = True
            parsed = _parse_binary_files_paths(line)
            if parsed:
                cur_old, cur_new = parsed
            _open(i)
        elif (line.startswith("--- ") and i + 1 < n
                and lines[i + 1].startswith("+++ ")):
            if cur_headers:
                # A second ---/+++ pair with no intervening `diff --git`:
                # the previous file section is complete.
                _flush(i)
            old = _clean_diff_path(line[4:])
            new = _clean_diff_path(lines[i + 1][4:])
            if old == "empty file" or new == "empty file":
                # e.g. "--- empty file" — a tool's marker for binary content.
                cur_binary = True
            if not cur_old and not cur_new:
                cur_old, cur_new = old, new
            elif old != cur_old or new != cur_new:
                # Headers disagree with the diff --git line: trust the
                # explicit file headers.
                cur_old, cur_new = old, new
            cur_headers = True
            _open(i)
            i += 1
        i += 1
    _flush(n)
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


def _status_of(section: _FileSection) -> str:
    if section.binary:
        return "binary"
    if section.rename_from and section.rename_to:
        return "renamed"
    if section.old == _DEV_NULL:
        return "added"
    if section.new == _DEV_NULL:
        return "deleted"
    return "modified"


def review_patch(diff_text: str) -> dict[str, Any]:
    """Per-file statistics for a diff.

    Parses ``---``/``+++``/``@@`` headers and counts added/removed lines
    per hunk.  Binary sections report status ``"binary"`` with 0/0
    counts; renamed files report status ``"renamed"`` with the
    ``rename_from`` path included.
    """
    sections = _split_sections(diff_text)
    files: list[dict[str, Any]] = []
    total_add = total_del = 0
    for section in sections:
        status = _status_of(section)
        additions = deletions = hunks = 0
        if not section.binary:
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
        entry: dict[str, Any] = {
            "path": path, "status": status, "additions": additions,
            "deletions": deletions, "hunks": hunks,
        }
        if status == "renamed":
            entry["rename_from"] = section.rename_from
        files.append(entry)
        total_add += additions
        total_del += deletions
    return {
        "files": files,
        "totals": {"files": len(files), "additions": total_add,
                   "deletions": total_del},
    }


def _read_targets(root: Path, sections: list[_FileSection]) -> dict[str, str | None]:
    """Current on-disk text per section read path (None when absent).

    Renames read the *old* path; every path is escape-checked, including
    rename destinations (which must also stay inside the root).
    """
    texts: dict[str, str | None] = {}
    for section in sections:
        if section.binary:
            _refuse_escape(root, _target_of(section)[0])
            continue
        if section.rename_from and section.rename_to:
            read_paths = [section.rename_from]
            for p in (section.rename_from, section.rename_to):
                _refuse_escape(root, p)
        else:
            target, _ = _target_of(section)
            read_paths = [target]
            _refuse_escape(root, target)
        for rp in read_paths:
            if rp in texts:
                continue
            p = root / rp
            texts[rp] = p.read_text(encoding="utf-8") if p.is_file() else None
    return texts


def _apply_rename(section: _FileSection, old_text: str | None,
                  new_exists: bool) -> tuple[str | None, str]:
    """In-memory rename application.  Returns (new text, error)."""
    old = section.rename_from
    new = section.rename_to
    if old_text is None:
        return None, f"cannot rename {old!r}: no such file"
    if new_exists:
        return None, f"cannot rename {old!r} to {new!r}: target already exists"
    try:
        patched = apply_unified_diff(section.text, {new: old_text})
    except DiffApplyError as exc:
        return None, str(exc)
    return patched[new], ""


def apply_patch(diff_text: str, *, dry_run: bool = True,
                root: str | Path = ".") -> list[dict[str, Any]]:
    """Apply a diff per file via :mod:`nomorals.core.diff`.

    Handles creation (``--- /dev/null``), deletion (``+++ /dev/null``),
    and git renames (read old path, patch, write new path, delete old).
    Binary sections cannot be applied textually: they come back as
    ``ok=False`` with a clear error.

    ``dry_run=True`` applies to in-memory copies and never touches disk;
    ``dry_run=False`` writes the results, creating parent directories.
    Any target escaping ``root`` raises :class:`WorkspaceError`
    immediately.  A diff with no file sections at all raises
    :class:`WorkspaceError` — it never silently succeeds.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise WorkspaceError(f"not a directory: {base}")
    sections = _split_sections(diff_text)
    if not sections:
        raise WorkspaceError("no file sections found in diff")
    current = _read_targets(base, sections)  # raises WorkspaceError on escape
    results: list[dict[str, Any]] = []
    for section in sections:
        target, is_deletion = _target_of(section)
        status = _status_of(section)
        if section.binary:
            results.append({"path": target, "status": status, "ok": False,
                            "error": "binary diffs cannot be applied "
                                     "textually; handle the file out of band"})
            continue
        if not section.has_headers:
            # e.g. a mode-only `diff --git` section with no hunks.
            results.append({"path": target, "status": status, "ok": False,
                            "error": "section contains no patchable content "
                                     "(mode change only?)"})
            continue
        if status == "renamed":
            new_text, error = _apply_rename(
                section, current.get(section.rename_from),
                (base / section.rename_to).is_file())
            if error:
                results.append({"path": target, "status": status, "ok": False,
                                "error": error})
                continue
            if not dry_run:
                try:
                    dest = _refuse_escape(base, section.rename_to)
                    src = _refuse_escape(base, section.rename_from)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(new_text or "", encoding="utf-8")
                    if src != dest and src.is_file():
                        src.unlink()
                except (OSError, WorkspaceError) as exc:
                    results.append({"path": target, "status": status,
                                    "ok": False,
                                    "error": f"write failed: {exc}"})
                    continue
            results.append({"path": target, "status": status, "ok": True,
                            "error": "", "rename_from": section.rename_from})
            continue
        try:
            patched = apply_unified_diff(section.text, {target: current[target]})
        except DiffApplyError as exc:
            results.append({"path": target, "status": status, "ok": False,
                            "error": str(exc)})
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
                results.append({"path": target, "status": status, "ok": False,
                                "error": f"write failed: {exc}"})
                continue
        results.append({"path": target, "status": status, "ok": True, "error": ""})
    if not dry_run:
        patched_count = sum(1 for r in results if r["ok"])
        failed = len(results) - patched_count
        _emit("codews.patch.applied", {
            "root": str(base),
            "dry_run": False,
            "files": [r["path"] for r in results],
            "patched": patched_count,
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
    """Before/after snippets around the changed lines for one file.

    For renames, ``path`` is the new path and the "before" side is read
    from the old path.  Binary sections raise :class:`WorkspaceError`:
    there is no textual preview.
    """
    base = Path(root).expanduser().resolve()
    section = next((s for s in _split_sections(diff_text)
                    if _target_of(s)[0] == path), None)
    if section is None:
        raise WorkspaceError(f"path {path!r} is not present in the diff")
    if section.binary:
        raise WorkspaceError(f"path {path!r} is a binary section: "
                             "no textual preview available")
    target, _ = _target_of(section)
    if section.rename_from and section.rename_to:
        before_path = section.rename_from
        dest = _refuse_escape(base, before_path)
        exists = dest.is_file()
        before_text = dest.read_text(encoding="utf-8") if exists else ""
    else:
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
