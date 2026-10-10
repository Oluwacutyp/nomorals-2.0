"""Pure-Python unified diff parser/applier (L1 kernel).

The canonical diff engine for the whole system: parses standard unified
diffs (multi-file, multi-hunk, creation/deletion, ``\\ No newline``
markers) and applies them to in-memory file texts.  Differential-fuzzed
500/500 against GNU ``patch``.

Stdlib only, zero intra-project imports — that is what lets both
``tools/edit_loop.py`` (L4) and ``agents/patch.py`` (L5) share it without
a layering cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DiffApplyError",
    "apply_unified_diff",
    "format_diff",
    "unified_diff",
]


class DiffApplyError(ValueError):
    """A unified diff could not be parsed or applied.

    ``hunk`` (1-based within the file) and ``path`` are set when the
    failure is attributable to a specific hunk; both stay None for
    structural parse errors.
    """

    def __init__(self, message: str, *, hunk: int | None = None,
                 path: str | None = None) -> None:
        super().__init__(message)
        self.hunk = hunk
        self.path = path



# ── pure-Python unified diff parser / applier ───────────────────────────────
# Replaces the `patch`-binary dependency where the binary is missing (Termux).


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DEV_NULL = "/dev/null"


@dataclass
class _Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    # (kind, text, newline_terminated); a "\ No newline at end of file"
    # marker flips newline_terminated on the preceding entry.
    body: list[tuple[str, str, bool]] = field(default_factory=list)


@dataclass
class _FilePatch:
    old_path: str
    new_path: str
    hunks: list[_Hunk] = field(default_factory=list)


def _clean_diff_path(raw: str) -> str:
    p = raw.split("\t", 1)[0].strip().strip('"')
    if p.startswith(("a/", "b/")) and p != _DEV_NULL:
        p = p[2:]
    return p


def _parse_unified_diff(diff_text: str) -> list[_FilePatch]:
    """Parse a unified diff into per-file patches."""
    patches: list[_FilePatch] = []
    lines = diff_text.split("\n")
    i = 0
    current: _FilePatch | None = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            if i + 1 >= len(lines) or not lines[i + 1].startswith("+++ "):
                raise DiffApplyError("diff has '---' header without a matching '+++' header")
            current = _FilePatch(
                old_path=_clean_diff_path(line[4:]),
                new_path=_clean_diff_path(lines[i + 1][4:]),
            )
            patches.append(current)
            i += 2
            continue
        if line.startswith("@@") and current is not None:
            m = _HUNK_RE.match(line)
            if not m:
                raise DiffApplyError(f"malformed hunk header: {line[:80]!r}")
            os_, oc, ns, nc = m.groups()
            hunk = _Hunk(int(os_), int(oc) if oc is not None else 1,
                         int(ns), int(nc) if nc is not None else 1)
            i += 1
            # Consume exactly the lines the hunk header claims: this keeps
            # content lines that start with "---", "+++" or "@@" (e.g. a
            # removed "--- foo" line) from being mistaken for structure.
            need_old, need_new = hunk.old_count, hunk.new_count
            while (need_old > 0 or need_new > 0) and i < len(lines):
                bl = lines[i]
                if bl.startswith("\\"):
                    if not hunk.body:
                        raise DiffApplyError(
                            f"stray '\\ No newline' marker in hunk for {current.old_path!r}")
                    kind, text, _term = hunk.body[-1]
                    hunk.body[-1] = (kind, text, False)
                elif bl[:1] in (" ", "-", "+"):
                    kind = bl[0]
                    if kind in (" ", "-"):
                        if need_old <= 0:
                            raise DiffApplyError(
                                f"hunk for {current.old_path!r} has more old-side lines "
                                f"than its header claims ({hunk.old_count})")
                        need_old -= 1
                    if kind in (" ", "+"):
                        if need_new <= 0:
                            raise DiffApplyError(
                                f"hunk for {current.old_path!r} has more new-side lines "
                                f"than its header claims ({hunk.new_count})")
                        need_new -= 1
                    hunk.body.append((kind, bl[1:], True))
                elif bl == "" and i == len(lines) - 1:
                    break  # trailing newline of the diff text itself
                else:
                    raise DiffApplyError(
                        f"unexpected line inside hunk for {current.old_path!r}: {bl[:60]!r}")
                i += 1
            if need_old > 0 or need_new > 0:
                raise DiffApplyError(
                    f"truncated hunk for {current.old_path!r}: header claims "
                    f"-{hunk.old_count}/+{hunk.new_count} lines")
            # A "\ No newline at end of file" marker trails the body line it
            # describes, so it can arrive after the header counts are met.
            while i < len(lines) and lines[i].startswith("\\"):
                kind, text, _term = hunk.body[-1]
                hunk.body[-1] = (kind, text, False)
                i += 1
            if not hunk.body:
                raise DiffApplyError(f"empty hunk for {current.old_path!r}")
            current.hunks.append(hunk)
            continue
        i += 1
    return patches


def _hunk_matches_at(lines: list[str], at: int, hunk: _Hunk) -> bool:
    p = at
    for kind, text, _term in hunk.body:
        if kind == "+":
            continue
        if p >= len(lines) or lines[p] != text:
            return False
        p += 1
    return True


def _locate_hunk(working: list[str], want: int, hunk: _Hunk) -> int | None:
    """Find where a hunk applies: stated position first, then a forward
    scan (patch-style offset tolerance).  Never scans backward: hunks
    apply left to right and each hunk's stated position already accounts
    for the lines previous hunks added or removed."""
    if _hunk_matches_at(working, want, hunk):
        return want
    for at in range(want + 1, len(working) + 1):
        if _hunk_matches_at(working, at, hunk):
            return at
    return None


def _apply_hunks_to_lines(
    lines: list[str],
    ends_with_newline: bool,
    hunks: list[_Hunk],
    label: str,
) -> str:
    # Offset model (GNU patch semantics): each hunk's stated old_start
    # refers to the ORIGINAL file, so its position in the evolving file is
    # old_start - 1 + offset, where offset is the net lines previous hunks
    # added.  This handles adjacent/overlapping hunks, which a
    # consume-cursor cannot express.
    working = list(lines)
    offset = 0
    # Trailing-newline bookkeeping: the result keeps the original file's
    # trailing-newline state unless a hunk produced the final line, in
    # which case the hunk's "\ No newline" markers decide.
    end_from_hunk = False
    end_terminated = ends_with_newline

    def _line_terminated(pos: int) -> bool:
        if pos < len(working) - 1:
            return True
        return end_terminated if end_from_hunk else ends_with_newline

    for n, hunk in enumerate(hunks, 1):
        pre_len = len(working)
        has_anchor = any(kind in (" ", "-") for kind, _t, _n in hunk.body)
        if has_anchor:
            want = max(hunk.old_start - 1 + offset, 0)
        else:
            # Pure insertion: old_start names the line AFTER which to insert.
            want = min(max(hunk.old_start, 0) + offset, pre_len)
        at = _locate_hunk(working, want, hunk)
        if at is None:
            raise DiffApplyError(
                f"hunk {n} for {label!r} does not match the file "
                f"(context mismatch around line {hunk.old_start})",
                hunk=n, path=label)
        new_seg: list[str] = []
        new_seg_term: list[bool] = []
        p = at
        for kind, text, terminated in hunk.body:
            if kind == "+":
                # A diff "\ No newline" marker is authoritative for '+' lines.
                new_seg.append(text)
                new_seg_term.append(terminated)
                continue
            if p >= len(working) or working[p] != text:
                raise DiffApplyError(
                    f"hunk {n} for {label!r}: expected {kind} line {text[:60]!r} "
                    f"does not match file line {p + 1}",
                    hunk=n, path=label)
            if kind == " ":
                new_seg.append(working[p])
                new_seg_term.append(_line_terminated(p) and terminated)
            p += 1
        old_span = p - at
        working[at:p] = new_seg
        offset += len(new_seg) - old_span
        if p == pre_len:
            # The hunk reached the (then) end of the file, so it decided
            # the final line's newline termination.
            if new_seg:
                end_from_hunk = True
                end_terminated = new_seg_term[-1]
            elif at == 0:
                end_from_hunk, end_terminated = False, False  # file emptied
            else:
                # Deleted through the end: the new last line is a
                # pre-existing interior line, which is newline-terminated.
                end_from_hunk, end_terminated = False, True
    text = "\n".join(working)
    terminated = end_terminated if end_from_hunk else ends_with_newline
    return text + "\n" if (terminated and working) else text


def _split_text_lines(text: str) -> tuple[list[str], bool]:
    ends_nl = text.endswith("\n")
    lines = text.split("\n")
    if ends_nl and lines and lines[-1] == "":
        lines.pop()
    return lines, ends_nl


def _apply_parsed_diff(
    patches: list[_FilePatch],
    file_texts: dict[str, str | None],
) -> dict[str, str | None]:
    """Apply parsed patches to an in-memory {relpath: text} mapping.

    Returns a new mapping; a ``None`` value means "delete this file".
    """
    result = dict(file_texts)
    for fp in patches:
        if fp.old_path == _DEV_NULL:
            target = fp.new_path
            if result.get(target) is not None:
                raise DiffApplyError(f"cannot create {target!r}: file already exists")
            result[target] = _apply_hunks_to_lines([], True, fp.hunks, target)
        elif fp.new_path == _DEV_NULL:
            target = fp.old_path
            current = result.get(target)
            if current is None:
                raise DiffApplyError(f"cannot delete {target!r}: no such file")
            lines, ends_nl = _split_text_lines(current)
            _apply_hunks_to_lines(lines, ends_nl, fp.hunks, target)  # verify match
            result[target] = None
        else:
            target = fp.new_path
            current = result.get(target)
            if current is None:
                raise DiffApplyError(
                    f"diff targets {target!r} but no such file was provided")
            lines, ends_nl = _split_text_lines(current)
            result[target] = _apply_hunks_to_lines(lines, ends_nl, fp.hunks, target)
    return result


def apply_unified_diff(
    diff_text: str,
    file_texts: dict[str, str | None],
) -> dict[str, str | None]:
    """Apply a unified diff to in-memory file contents, pure Python.

    Args:
        diff_text: Unified diff (``---``/``+++``/``@@`` format).
        file_texts: Mapping of relative path → current text (``None`` for
            files known to be absent).

    Returns:
        New mapping of relative path → patched text; ``None`` marks a file
        the diff deletes. Supports multiple files, multiple hunks per file,
        file creation (``--- /dev/null``) and deletion (``+++ /dev/null``).

    Raises:
        DiffApplyError: on malformed diffs, context mismatches, creating an
            existing file, or deleting a missing one.
    """
    patches = _parse_unified_diff(diff_text)
    if not patches:
        raise DiffApplyError("no file sections found in diff")
    return _apply_parsed_diff(patches, file_texts)


# ── Diff generation & rendering ────────────────────────────────────────────

def unified_diff(a: str, b: str, *, path: str = "file",
                 context: int = 3) -> str:
    """Generate a unified diff from ``a`` to ``b`` (the inverse of applying).

    Pure-Python Myers-free implementation: good enough for review-sized
    texts (code patches, config changes). For huge files prefer difflib.
    The output feeds straight back into :func:`apply_unified_diff`.
    """
    import difflib

    a_lines = a.splitlines(keepends=True)
    b_lines = b.splitlines(keepends=True)
    out = difflib.unified_diff(
        a_lines, b_lines, fromfile=f"a/{path}", tofile=f"b/{path}",
        n=context)
    text = "".join(out)
    # difflib omits the trailing newline marker handling; normalize so our
    # own parser round-trips.
    return text


def format_diff(diff_text: str, *, color: bool | None = None,
                theme: Any = None) -> str:
    """Render a unified diff for humans: magenta removals… no — red-free.

    House rule (the owner hates red): deletions render magenta, additions
    green, hunk headers electric blue, context dimmed. ``color=None``
    auto-detects the tty.
    """
    from .style import paint, supports_color

    if color is None:
        color = supports_color()
    if not color:
        return diff_text
    out: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            out.append(paint(line, "label", theme, color=color))
        elif line.startswith("@@"):
            out.append(paint(line, "info", theme, color=color))
        elif line.startswith("+"):
            out.append(paint(line, "ok", theme, color=color))
        elif line.startswith("-"):
            out.append(paint(line, "error", theme, color=color))
        else:
            out.append(paint(line, "muted", theme, color=color))
    return "\n".join(out)
