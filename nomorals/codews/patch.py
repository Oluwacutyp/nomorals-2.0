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

Beyond unified diffs, this module also applies aider-style
``<<<<<<< SEARCH`` / ``=======`` / ``>>>>>>> REPLACE`` edit blocks
(:func:`apply_edit_blocks`) with aider's fallback ladder
(exact → blank-line → trailing-whitespace → indent-drift → fuzzy),
and exposes GNU-``patch --fuzz`` semantics on :func:`apply_patch`.
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

__all__ = [
    "review_patch",
    "apply_patch",
    "preview_patch",
    "record_patch",
    "split_patch",
    "check_patch",
    "apply_edit_blocks",
]

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
    old_mode: str = ""
    new_mode: str = ""
    similarity: int = -1  # -1 = not a similarity-scored section


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
    cur_old_mode = ""
    cur_new_mode = ""
    cur_similarity = -1

    def _flush(end: int) -> None:
        nonlocal cur_old, cur_new, cur_start, cur_headers
        nonlocal cur_rename_from, cur_rename_to, cur_binary
        nonlocal cur_old_mode, cur_new_mode, cur_similarity
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
            binary=cur_binary,
            old_mode=cur_old_mode, new_mode=cur_new_mode,
            similarity=cur_similarity))
        cur_old = cur_new = ""
        cur_start = None
        cur_headers = False
        cur_rename_from = cur_rename_to = ""
        cur_binary = False
        cur_old_mode = cur_new_mode = ""
        cur_similarity = -1

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
        elif line.startswith("old mode "):
            cur_old_mode = line[len("old mode "):].strip()
            _open(i)
        elif line.startswith("new mode "):
            cur_new_mode = line[len("new mode "):].strip()
            _open(i)
        elif line.startswith("new file mode "):
            cur_new_mode = line[len("new file mode "):].strip()
            _open(i)
        elif line.startswith("deleted file mode "):
            cur_old_mode = line[len("deleted file mode "):].strip()
            _open(i)
        elif line.startswith("similarity index "):
            try:
                cur_similarity = int(
                    line[len("similarity index "):].strip().rstrip("%"))
            except ValueError:
                pass
            _open(i)
        elif line.startswith("dissimilarity index "):
            try:
                cur_similarity = 100 - int(
                    line[len("dissimilarity index "):].strip().rstrip("%"))
            except ValueError:
                pass
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


def _hunk_headers(section: _FileSection) -> list[dict[str, Any]]:
    """Parsed ``@@`` headers for a section: old/new start+length and the
    section heading (the function/context text after the second ``@@``)."""
    out: list[dict[str, Any]] = []
    for line in section.text.split("\n"):
        m = _HUNK_RE.match(line)
        if not m:
            continue
        os_, oc, ns, nc = m.groups()
        out.append({
            "old_start": int(os_), "old_lines": int(oc) if oc else 1,
            "new_start": int(ns), "new_lines": int(nc) if nc else 1,
            "section": line[m.end():].strip(),
        })
    return out


def review_patch(diff_text: str) -> dict[str, Any]:
    """Per-file statistics for a diff.

    Parses ``---``/``+++``/``@@`` headers and counts added/removed lines
    per hunk.  Binary sections report status ``"binary"`` with 0/0
    counts; renamed files report status ``"renamed"`` with the
    ``rename_from`` path included.  Extended headers surface as
    ``old_mode``/``new_mode`` (``""`` when absent), ``similarity``
    (``-1`` when not a scored rename/copy), and ``is_symlink``.
    """
    sections = _split_sections(diff_text)
    files: list[dict[str, Any]] = []
    total_add = total_del = 0
    for section in sections:
        status = _status_of(section)
        additions = deletions = 0
        hunk_list: list[dict[str, Any]] = []
        if not section.binary:
            hunk_list = _hunk_headers(section)
            for line in section.text.split("\n"):
                if _HUNK_RE.match(line):
                    continue
                elif line.startswith(("+++", "---")):
                    continue  # the section's own headers
                elif line.startswith("+"):
                    additions += 1
                elif line.startswith("-"):
                    deletions += 1
        path, _ = _target_of(section)
        entry: dict[str, Any] = {
            "path": path, "status": status, "additions": additions,
            "deletions": deletions, "hunks": len(hunk_list),
            "hunk_headers": hunk_list,
            "is_added": section.old == _DEV_NULL,
            "is_removed": section.new == _DEV_NULL,
            "is_binary": section.binary,
            "old_mode": section.old_mode, "new_mode": section.new_mode,
            "is_symlink": section.new_mode == "120000"
                          or section.old_mode == "120000",
            "similarity": section.similarity,
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


def split_patch(diff_text: str) -> dict[str, str]:
    """Split a multi-file diff into per-file diff texts keyed by target path.

    Each value is a self-contained diff for one file (extended headers
    included), suitable for selective application via :func:`apply_patch`.
    """
    out: dict[str, str] = {}
    for section in _split_sections(diff_text):
        path, _ = _target_of(section)
        text = section.text
        if not text.endswith("\n"):
            text += "\n"
        out[path] = out.get(path, "") + text
    return out


def check_patch(diff_text: str, *, root: str | Path = ".") -> dict[str, Any]:
    """``git apply --check`` equivalent: dry-run the whole diff and report.

    Returns ``{"ok", "files", "totals"}`` — ``ok`` is True only when every
    file section applies cleanly.  Never touches disk.
    """
    results = apply_patch(diff_text, dry_run=True, root=root)
    ok = all(r["ok"] for r in results)
    return {
        "ok": ok,
        "files": results,
        "totals": {
            "files": len(results),
            "ok": sum(1 for r in results if r["ok"]),
            "failed": sum(1 for r in results if not r["ok"]),
        },
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


def _apply_rename(section_text: str, section: _FileSection,
                  old_text: str | None, new_exists: bool, *,
                  fuzz: int = 0) -> tuple[str | None, bool, str]:
    """In-memory rename application.  Returns (new text, fuzzy, error)."""
    old = section.rename_from
    new = section.rename_to
    if old_text is None:
        return None, False, f"cannot rename {old!r}: no such file"
    if new_exists:
        return None, False, (f"cannot rename {old!r} to {new!r}: "
                             "target already exists")
    new_text, was_fuzzy, error = _apply_section_text(
        section_text, new, old_text, fuzz=fuzz, label=new)
    if error:
        return None, False, error
    return new_text, was_fuzzy, ""


def _fuzz_section_text(section_text: str, level: int) -> str:
    """Rewrite a section's hunks per GNU ``patch --fuzz`` semantics.

    Drops up to ``level`` leading and ``level`` trailing context (`` ``)
    lines per hunk and recomputes the ``@@`` line counts to match.  Added
    and removed lines are never dropped — only context.  A trailing
    context line glued to a ``\\ No newline`` marker is kept so the
    marker still describes the right line.
    """
    lines = section_text.split("\n")
    out: list[str] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        m = _HUNK_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue
        body: list[str] = []
        i += 1
        while i < n and not _HUNK_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        while body and body[-1] == "":
            body.pop()  # the section's trailing newline, not hunk content
        # Pair "\ No newline" markers with the line they describe.
        entries: list[tuple[str, bool]] = []
        for bl in body:
            if bl.startswith("\\"):
                if entries:
                    prev, _ = entries[-1]
                    entries[-1] = (prev, True)
                continue
            entries.append((bl, False))
        drop_head = 0
        while (drop_head < level and drop_head < len(entries)
               and entries[drop_head][0].startswith(" ")):
            drop_head += 1
        drop_tail = 0
        while (drop_tail < level
               and drop_tail < len(entries) - drop_head
               and entries[len(entries) - 1 - drop_tail][0].startswith(" ")
               and not entries[len(entries) - 1 - drop_tail][1]):
            drop_tail += 1
        kept = entries[drop_head:len(entries) - drop_tail if drop_tail else len(entries)]
        old_count = sum(1 for bl, _ in kept if bl[:1] in (" ", "-"))
        new_count = sum(1 for bl, _ in kept if bl[:1] in (" ", "+"))
        os_, ns = m.group(1), m.group(3)
        out.append(f"@@ -{os_},{old_count} +{ns},{new_count} @@{line[m.end():]}")
        for bl, nonl in kept:
            out.append(bl)
            if nonl:
                out.append("\\ No newline at end of file")
    return "\n".join(out)


def _apply_section_text(section_text: str, target: str,
                        current_text: str | None, *, fuzz: int = 0,
                        label: str) -> tuple[str | None, bool, str]:
    """Apply one section's text; return (new text, fuzzy, error).

    With ``fuzz > 0``, a context-mismatch failure is retried with the
    section rewritten per :func:`_fuzz_section_text`.  ``fuzzy`` reports
    whether the fuzz fallback was what succeeded — a fuzzy success is
    never presented as a clean apply.
    """
    try:
        patched = apply_unified_diff(section_text, {target: current_text})
        return patched[target], False, ""
    except DiffApplyError as exc:
        if fuzz <= 0:
            return None, False, str(exc)
    try:
        patched = apply_unified_diff(
            _fuzz_section_text(section_text, fuzz), {target: current_text})
    except DiffApplyError as exc:
        return None, False, f"{exc} (even with fuzz={fuzz})"
    return patched[target], True, ""


def apply_patch(diff_text: str, *, dry_run: bool = True,
                root: str | Path = ".", fuzz: int = 0) -> list[dict[str, Any]]:
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

    ``fuzz`` mirrors GNU ``patch --fuzz``: with ``fuzz > 0``, hunks whose
    context drifted are retried ignoring up to ``fuzz`` leading/trailing
    context lines per hunk.  ``fuzz=0`` (default) requires all context to
    match, like ``git apply``.  Results carry ``"fuzzy": bool`` so a
    fuzzy success is always visible.
    """
    if fuzz < 0:
        raise WorkspaceError("fuzz must be >= 0")
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
                                     "textually; handle the file out of band",
                            "fuzzy": False})
            continue
        if not section.has_headers:
            # e.g. a mode-only `diff --git` section with no hunks.
            results.append({"path": target, "status": status, "ok": False,
                            "error": "section contains no patchable content "
                                     "(mode change only?)",
                            "fuzzy": False})
            continue
        if status == "renamed":
            new_text, was_fuzzy, error = _apply_rename(
                section.text, section, current.get(section.rename_from),
                (base / section.rename_to).is_file(), fuzz=fuzz)
            if error:
                results.append({"path": target, "status": status, "ok": False,
                                "error": error, "fuzzy": False})
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
                                    "error": f"write failed: {exc}",
                                    "fuzzy": False})
                    continue
            results.append({"path": target, "status": status, "ok": True,
                            "error": "", "fuzzy": was_fuzzy,
                            "rename_from": section.rename_from})
            continue
        new_text, was_fuzzy, error = _apply_section_text(
            section.text, target, current[target], fuzz=fuzz, label=target)
        if error:
            results.append({"path": target, "status": status, "ok": False,
                            "error": error, "fuzzy": False})
            continue
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
                                "error": f"write failed: {exc}",
                                "fuzzy": False})
                continue
        results.append({"path": target, "status": status, "ok": True,
                        "error": "", "fuzzy": was_fuzzy})
    if not dry_run:
        patched_count = sum(1 for r in results if r["ok"])
        failed = len(results) - patched_count
        _emit("codews.patch.applied", {
            "root": str(base),
            "dry_run": False,
            "files": [r["path"] for r in results],
            "patched": patched_count,
            "failed": failed,
            "fuzzy": [r["path"] for r in results if r.get("fuzzy")],
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
              mark: set[int], marker: str,
              context_lines: int = _CONTEXT_LINES) -> str:
    """Render ``lines`` around ``changed`` line numbers with a gutter.

    ``mark`` is the subset of shown lines that changed (``marker`` = the
    per-line prefix for changed lines, e.g. ``-`` or ``+``).
    """
    if not changed:
        return ""
    keep: set[int] = set()
    for ln in changed:
        for i in range(max(1, ln - context_lines),
                       min(len(lines), ln + context_lines) + 1):
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
                  *, root: str | Path = ".", fuzz: int = 0,
                  context_lines: int = _CONTEXT_LINES) -> dict[str, str]:
    """Before/after snippets around the changed lines for one file.

    For renames, ``path`` is the new path and the "before" side is read
    from the old path.  Binary sections raise :class:`WorkspaceError`:
    there is no textual preview.  ``fuzz`` mirrors
    :func:`apply_patch`'s fuzz fallback; ``context_lines`` sizes the
    window shown around each change.
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
    new_text, was_fuzzy, error = _apply_section_text(
        section.text, target, before_text if exists else None,
        fuzz=fuzz, label=path)
    if error:
        raise WorkspaceError(f"cannot preview {path!r}: {error}")
    after_text = new_text or ""
    old_changed, new_changed = _changed_lines(section)
    before_lines = before_text.split("\n")
    after_lines = after_text.split("\n")
    return {
        "before": _windowed(before_lines, old_changed, old_changed, "-",
                            context_lines),
        "after": _windowed(after_lines, new_changed, new_changed, "+",
                           context_lines),
        "fuzzy": was_fuzzy,
    }


def record_patch(store: Any, diff_text: str, *, mission_id: str = "",
                 creator: str = "codews") -> str:
    """Store a diff as a ``patch`` artifact; return its ``artifact://`` URI."""
    import hashlib

    stats = review_patch(diff_text)
    art = store.put_text(
        diff_text,
        type="patch",
        creator=creator,
        mission_id=mission_id,
        metadata={"files": [f["path"] for f in stats["files"]],
                  "totals": stats["totals"],
                  "sha256": hashlib.sha256(
                      diff_text.encode("utf-8")).hexdigest()},
        provenance=Provenance(source_type="codews"),
    )
    return art.uri


# ── SEARCH/REPLACE edit blocks (aider format) ─────────────────────────────
# Models emit unified-diff arithmetic wrong far more often than they quote
# text wrong, so this module also applies the edit-block format:
#
#     path/to/file.py
#     ```python
#     <<<<<<< SEARCH
#     old text, quoted exactly
#     =======
#     new text
#     >>>>>>> REPLACE
#     ```
#
# The fence is optional.  Matching follows aider's ladder: exact →
# blank-line → trailing-whitespace → indent-drift → difflib fuzzy
# (threshold-gated, strategy always reported; ambiguous matches refused).

_EDIT_BLOCK_RE = re.compile(
    r"<<<<<<< SEARCH\r?\n(.*?)\r?\n=======\r?\n(.*?)\r?\n>>>>>>> REPLACE",
    re.DOTALL,
)
_FUZZY_THRESHOLD = 0.9


def _parse_edit_blocks(text: str) -> list[dict[str, Any]]:
    """Split ``text`` into ``{"path", "search", "replace"}`` blocks.

    The path is the nearest preceding non-empty, non-fence line — the
    aider convention of a filename line above the block.
    """
    blocks: list[dict[str, Any]] = []
    for m in _EDIT_BLOCK_RE.finditer(text):
        path = ""
        for bl in reversed(text[:m.start()].splitlines()):
            s = bl.strip()
            if not s or s.startswith("```"):
                continue
            path = s.strip("`").strip()
            break
        blocks.append({"path": path, "search": m.group(1),
                       "replace": m.group(2)})
    return blocks


def _find_spans(lines: list[str], want: list[str]) -> list[int]:
    """Start indices where ``want`` matches exactly."""
    n = len(want)
    if not n:
        return []
    return [i for i in range(len(lines) - n + 1) if lines[i:i + n] == want]


def _find_spans_rstrip(lines: list[str], want: list[str]) -> list[int]:
    """Start indices where ``want`` matches ignoring trailing whitespace."""
    w = [l.rstrip() for l in want]
    n = len(w)
    if not n:
        return []
    return [i for i in range(len(lines) - n + 1)
            if [l.rstrip() for l in lines[i:i + n]] == w]


def _indent_of(line: str) -> str:
    return line[:len(line) - len(line.lstrip())]


def _find_spans_indent_drift(lines: list[str],
                             want: list[str]) -> list[tuple[int, str, str]]:
    """Start indices where ``want`` matches up to a *uniform* leading-
    whitespace drift.  Returns ``(index, file_base, want_base)`` so the
    replacement can be re-indented to the file's indentation."""
    n = len(want)
    if not n:
        return []
    hits: list[tuple[int, str, str]] = []
    for i in range(len(lines) - n + 1):
        seg = lines[i:i + n]
        deltas: set[int] = set()
        ok = True
        for f, s in zip(seg, want):
            if not s.strip():
                if f.strip():
                    ok = False
                    break
                continue
            if f.strip() != s.strip():
                ok = False
                break
            deltas.add(len(_indent_of(f)) - len(_indent_of(s)))
        if ok and len(deltas) == 1:
            f_base = _indent_of(next(f for f in seg if f.strip()))
            w_base = _indent_of(next(s for s in want if s.strip()))
            hits.append((i, f_base, w_base))
    return hits


def _find_spans_fuzzy(lines: list[str], want: list[str],
                      threshold: float = _FUZZY_THRESHOLD
                      ) -> list[tuple[float, int]]:
    """``(score, index)`` for windows scoring >= ``threshold``."""
    import difflib

    target = "\n".join(want)
    n = len(want)
    if not n:
        return []
    return sorted(
        ((difflib.SequenceMatcher(None, "\n".join(lines[i:i + n]),
                                  target).ratio(), i)
         for i in range(len(lines) - n + 1)),
        reverse=True,
    )


def _reindent(replace_lines: list[str], file_base: str,
              want_base: str) -> list[str]:
    """Re-indent replacement lines from the block's base to the file's."""
    out: list[str] = []
    for r in replace_lines:
        if not r.strip():
            out.append("")
        elif want_base and r.startswith(want_base):
            out.append(file_base + r[len(want_base):])
        else:
            out.append(file_base + r.lstrip())
    return out


def _apply_one_block(path: str, search: str, replace: str,
                     current: str | None) -> tuple[str | None, str, str]:
    """Apply a single SEARCH/REPLACE block.  Returns (new text, strategy,
    error); strategy is "" on failure."""
    search_lines = search.split("\n")
    replace_lines = replace.split("\n")
    if not search.strip():
        # aider convention: an empty SEARCH creates the file.
        if current is not None:
            return None, "", (f"cannot create {path!r}: file already "
                               "exists (empty SEARCH block)")
        return "\n".join(replace_lines), "create", ""
    if current is None:
        return None, "", f"cannot edit {path!r}: no such file"
    lines = current.split("\n")

    def _splice(at: int, n: int, new_lines: list[str]) -> str:
        return "\n".join(lines[:at] + new_lines + lines[at + n:])

    hits = _find_spans(lines, search_lines)
    if len(hits) == 1:
        return _splice(hits[0], len(search_lines), replace_lines), "exact", ""
    if len(hits) > 1:
        return None, "", (f"ambiguous SEARCH block for {path!r}: "
                          f"{len(hits)} matches, refusing to guess")
    # Drop one spurious leading blank line (aider does this).
    if search_lines and not search_lines[0].strip():
        trimmed = search_lines[1:]
        hits = _find_spans(lines, trimmed)
        if len(hits) == 1:
            return _splice(hits[0], len(trimmed), replace_lines), \
                "blank-line", ""
        if len(hits) > 1:
            return None, "", (f"ambiguous SEARCH block for {path!r}: "
                              f"{len(hits)} matches, refusing to guess")
    hits = _find_spans_rstrip(lines, search_lines)
    if len(hits) == 1:
        return _splice(hits[0], len(search_lines), replace_lines), \
            "trailing-ws", ""
    if len(hits) > 1:
        return None, "", (f"ambiguous SEARCH block for {path!r}: "
                          f"{len(hits)} matches, refusing to guess")
    drift = _find_spans_indent_drift(lines, search_lines)
    if len(drift) == 1:
        at, f_base, w_base = drift[0]
        return _splice(at, len(search_lines),
                       _reindent(replace_lines, f_base, w_base)), \
            "indent-drift", ""
    if len(drift) > 1:
        return None, "", (f"ambiguous SEARCH block for {path!r}: "
                          f"{len(drift)} matches, refusing to guess")
    scored = [s for s in _find_spans_fuzzy(lines, search_lines)
              if s[0] >= _FUZZY_THRESHOLD]
    if scored:
        top = [s for s in scored if s[0] == scored[0][0]]
        if len(top) > 1:
            return None, "", (f"ambiguous SEARCH block for {path!r}: "
                              f"{len(top)} equally fuzzy matches")
        return _splice(top[0][1], len(search_lines), replace_lines), \
            "fuzzy", ""
    return None, "", (f"SEARCH block for {path!r} does not match the file "
                      "(no strategy reached the similarity threshold)")


def apply_edit_blocks(text: str, *, dry_run: bool = True,
                      root: str | Path = ".") -> list[dict[str, Any]]:
    """Apply aider-style SEARCH/REPLACE edit blocks to files under ``root``.

    Each block names its file on the line above it.  Matching uses the
    fallback ladder exact → blank-line → trailing-ws → indent-drift →
    fuzzy (``difflib``, threshold 0.9); the winning ``"strategy"`` is
    reported per block so a loose match is always visible.  Ambiguous
    (multi-match) blocks are refused outright.

    ``dry_run=True`` validates without touching disk; ``dry_run=False``
    writes, creating parent directories.  Raises :class:`WorkspaceError`
    when no blocks are found or a path escapes ``root``.
    """
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise WorkspaceError(f"not a directory: {base}")
    blocks = _parse_edit_blocks(text)
    if not blocks:
        raise WorkspaceError("no SEARCH/REPLACE blocks found in text")
    results: list[dict[str, Any]] = []
    for block in blocks:
        path = block["path"]
        if not path:
            results.append({"path": "", "ok": False, "strategy": "",
                            "error": "block has no filename line above it"})
            continue
        try:
            dest = _refuse_escape(base, path)
        except WorkspaceError as exc:
            results.append({"path": path, "ok": False, "strategy": "",
                            "error": str(exc)})
            continue
        current = dest.read_text(encoding="utf-8") if dest.is_file() else None
        new_text, strategy, error = _apply_one_block(
            path, block["search"], block["replace"], current)
        if error:
            results.append({"path": path, "ok": False, "strategy": "",
                            "error": error})
            continue
        if not dry_run:
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(new_text or "", encoding="utf-8")
            except OSError as exc:
                results.append({"path": path, "ok": False, "strategy": "",
                                "error": f"write failed: {exc}"})
                continue
        results.append({"path": path, "ok": True, "strategy": strategy,
                        "error": ""})
    if not dry_run:
        _emit("codews.edit_blocks.applied", {
            "root": str(base),
            "blocks": [{"path": r["path"], "ok": r["ok"],
                        "strategy": r["strategy"]} for r in results],
        })
    return results
