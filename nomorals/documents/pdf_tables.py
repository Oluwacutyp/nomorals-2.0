"""Table recovery from plain extracted text (PDF pages, pasted reports).

The pure-Python PDF reader in :mod:`nomorals.core.pdf` returns text lines
with column spacing preserved.  :func:`extract_text_tables` finds runs of
consecutive lines that share a consistent whitespace-column layout and
turns each run into a :class:`~nomorals.documents.model.Table`.

Two strategies (Camelot's ``lattice``/``stream`` distinction, stdlib
edition):

* ``ruling`` — explicit grid lines (box-drawing ``─│┌┐`` or ASCII
  ``+---+`` art).  High precision: the grid *is* the structure.
* ``stream`` — whitespace-aligned columns.  Multi-pass over gap widths
  (2..5 spaces, pdfplumber's tolerance-tuning idea); each candidate is
  scored and only the best non-overlapping tables survive.

Every table gets a ``confidence`` score (0..1, Camelot's accuracy idea):
column-start alignment consistency + cell fill + size.  Heuristics stay
conservative — a missed table beats a hallucinated one — but low scores
let pipelines flag tables for human review instead of guessing.
"""

from __future__ import annotations

import re
import statistics
from typing import NamedTuple

from .model import Table

__all__ = ["extract_text_tables", "score_table"]

#: Box-drawing + ASCII-art ruling characters for the lattice-style pass.
_RULING_CHARS = frozenset("─│┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬+-|")
_RULING_CORNERS = frozenset("+┌┐└┘├┤┬┴┼╔╗╚╝╠╣╦╩╬")


def _gap_re(gap: int) -> re.Pattern:
    return re.compile(r"(?: {%d,}|\t)+" % gap)


def _split_columns(line: str, gap: int) -> list[str]:
    """Split a line on wide whitespace gaps into stripped cells."""
    return [cell.strip() for cell in _gap_re(gap).split(line.strip())]


def _is_columnar(line: str, min_cols: int, gap: int) -> bool:
    cells = _split_columns(line, gap)
    return len(cells) >= min_cols and all(cells)


def _column_starts(line: str, gap: int) -> list[int]:
    """Start offset of each cell in the stripped line (alignment signal)."""
    stripped = line.strip()
    starts = [0]
    for match in _gap_re(gap).finditer(stripped):
        starts.append(match.end())
    return starts


_NUMERIC_RE = re.compile(r"^[+-]?[\d,]*\.?\d+%?$")


def _is_numeric(cell: str) -> bool:
    return bool(_NUMERIC_RE.match(cell.strip().replace(" ", "")))


def score_table(grid: list[list[str]],
                starts: list[list[int]] | None = None) -> float:
    """Confidence score (0..1) for a recovered table grid.

    Combines column-start alignment consistency (the strongest signal —
    real tables line their columns up), cell fill ratio, and a small size
    bonus.  Used to pick the best gap width per block and to let callers
    threshold junk (``confidence < 0.6`` → human review).
    """
    if not grid or not grid[0]:
        return 0.0
    n_cols = len(grid[0])
    n_rows = len(grid)
    if starts:
        stds = []
        for col in range(n_cols):
            offsets = [row[col] for row in starts if col < len(row)]
            if len(offsets) > 1:
                stds.append(statistics.pstdev(offsets))
        mean_std = sum(stds) / len(stds) if stds else 0.0
    else:
        mean_std = 0.0
    align = 1.0 / (1.0 + mean_std / 6.0)
    total = n_rows * n_cols
    filled = sum(1 for row in grid for cell in row if cell.strip())
    fill = filled / total if total else 0.0
    size = min(1.0, max(0, n_rows - 1) / 6.0) * 0.15 + \
        min(1.0, n_cols / 5.0) * 0.05
    return round(max(0.0, min(1.0, 0.6 * align + 0.25 * fill + 0.15 + size)), 3)


_CAPTION_RE = re.compile(r"(?i)^\s*(table|tab\.?)\s*(\d+)\s*[:.\-]?\s*(.*)$")


def _caption_above(lines: list[str], start: int) -> tuple[str, str]:
    """Look ≤2 lines above ``start`` for a ``Table N: …`` caption.

    Returns (name, caption); ("", "") when nothing matches.
    """
    for look in range(start - 1, max(start - 4, -1), -1):
        line = lines[look]
        if not line.strip():
            continue
        match = _CAPTION_RE.match(line)
        if match:
            rest = match.group(3).strip()
            caption = f"Table {match.group(2)}" + (f": {rest}" if rest else "")
            return f"Table {match.group(2)}", caption[:120]
        return "", ""  # nearest non-blank line is not a caption: stop
    return "", ""


class _Candidate(NamedTuple):
    start: int
    end: int  # exclusive
    grid: list[list[str]]
    starts: list[list[int]]
    name: str
    caption: str
    confidence: float


def _stream_candidates(lines: list[str], min_rows: int, min_cols: int,
                       gap: int) -> list[_Candidate]:
    """Whitespace-column table candidates at one gap width."""
    candidates: list[_Candidate] = []
    i, n = 0, len(lines)
    while i < n:
        if not _is_columnar(lines[i], min_cols, gap):
            i += 1
            continue
        width = len(_split_columns(lines[i], gap))
        block: list[str] = []
        block_starts: list[list[int]] = []
        j = i
        while j < n and _is_columnar(lines[j], min_cols, gap):
            if len(_split_columns(lines[j], gap)) != width:
                break
            block.append(lines[j])
            block_starts.append(_column_starts(lines[j], gap))
            j += 1
        if len(block) >= min_rows:
            grid = [_split_columns(line, gap) for line in block]
            if all(all(cell for cell in row) for row in grid):
                name, caption = _caption_above(lines, i)
                candidates.append(_Candidate(
                    start=i, end=j, grid=grid, starts=block_starts,
                    name=name, caption=caption,
                    confidence=score_table(grid, block_starts)))
            i = j
        else:
            i += 1
    return candidates


def _is_ruling_line(line: str) -> bool:
    """A line that is (almost) entirely grid-drawing characters."""
    stripped = line.strip()
    if len(stripped) < 3:
        return False
    ruling = sum(1 for ch in stripped if ch in _RULING_CHARS)
    return ruling / len(stripped) >= 0.6 and \
        any(ch in _RULING_CORNERS for ch in stripped)


def _split_ruling_row(line: str) -> list[str]:
    """Split a ruling-table body row on │ or | into stripped cells."""
    return [cell.strip() for cell in re.split(r"[│|]", line.strip())]


def _ruling_candidates(lines: list[str], min_rows: int,
                       min_cols: int) -> list[_Candidate]:
    """Lattice-style pass: explicit box/ASCII-art grids."""
    candidates: list[_Candidate] = []
    i, n = 0, len(lines)
    while i < n:
        stripped = lines[i].strip()
        has_cell_sep = "│" in stripped or "|" in stripped
        if not (has_cell_sep or _is_ruling_line(stripped)):
            i += 1
            continue
        start = i
        grid: list[list[str]] = []
        j = i
        while j < n:
            s = lines[j].strip()
            if _is_ruling_line(s):
                j += 1
                continue
            if "│" in s or "|" in s:
                cells = [c for c in _split_ruling_row(s)]
                # Drop empty edge cells from leading/trailing separators.
                if cells and not cells[0]:
                    cells = cells[1:]
                if cells and not cells[-1]:
                    cells = cells[:-1]
                if cells and any(cells):
                    grid.append(cells)
                j += 1
                continue
            break
        width_ok = grid and all(len(r) == len(grid[0]) for r in grid)
        if grid and len(grid) >= min_rows and len(grid[0]) >= min_cols \
                and width_ok:
            name, caption = _caption_above(lines, start)
            candidates.append(_Candidate(
                start=start, end=j, grid=grid, starts=[],
                name=name, caption=caption, confidence=0.95))
            i = j
        else:
            i += 1
    return candidates


def _apply_header(grid: list[list[str]],
                  has_header: bool | str) -> tuple[list[str], list[list[str]]]:
    """Split header row from data rows per ``has_header``.

    ``True``: first row is the header (legacy).  ``False``: generated
    ``col_1…`` headers, all rows are data.  ``"auto"``: first row is the
    header only when most of its cells are non-numeric.
    """
    if has_header is True or (
            has_header == "auto"
            and sum(1 for c in grid[0] if not _is_numeric(c))
            >= (len(grid[0]) + 1) // 2):
        return grid[0], grid[1:]
    headers = [f"col_{i + 1}" for i in range(len(grid[0]))]
    return headers, grid


def extract_text_tables(text: str, *, min_rows: int = 3,
                        min_cols: int = 2, strategy: str = "auto",
                        has_header: bool | str = True) -> list[Table]:
    """Detect tables in plain text.

    ``strategy``: ``"auto"`` (ruling pass, then whitespace stream),
    ``"stream"``, or ``"ruling"``.  The stream pass tries gap widths
    5→2 and keeps the best-scoring non-overlapping tables.  ``has_header``
    controls header detection: ``True`` (first row, the legacy default),
    ``False`` (generated ``col_N`` headers), or ``"auto"`` (non-numeric
    first row → header).

    Returns a list of :class:`Table` (possibly empty) with ``confidence``
    scores and recovered captions.  Never raises on odd input — returns []
    when nothing table-like is found.
    """
    if min_rows < 2:
        raise ValueError(f"min_rows must be >= 2, got {min_rows}")
    if min_cols < 2:
        raise ValueError(f"min_cols must be >= 2, got {min_cols}")
    if strategy not in ("auto", "stream", "ruling"):
        raise ValueError(f"strategy must be auto/stream/ruling, "
                         f"got {strategy!r}")
    if has_header not in (True, False, "auto"):
        raise ValueError(f"has_header must be True/False/'auto', "
                         f"got {has_header!r}")
    lines = text.splitlines()
    candidates: list[_Candidate] = []
    if strategy in ("auto", "ruling"):
        candidates.extend(_ruling_candidates(lines, min_rows, min_cols))
    if strategy in ("auto", "stream"):
        # Collect candidates at every gap width, then keep the
        # best-scoring non-overlapping set: a wide gap can miss the true
        # header row (claiming a shorter block), so gap passes must not
        # claim line ranges before narrower gaps are even considered.
        stream_cands: list[_Candidate] = []
        for gap in (5, 4, 3, 2):
            stream_cands.extend(_stream_candidates(lines, min_rows, min_cols, gap))
        candidates.extend(stream_cands)
    # Prefer higher confidence, then earlier position, then longer
    # coverage (deterministic).
    candidates.sort(key=lambda c: (-c.confidence, c.start, -(c.end - c.start)))
    accepted: list[_Candidate] = []
    for cand in candidates:
        if any(not (cand.end <= taken.start or cand.start >= taken.end)
               for taken in accepted):
            continue
        accepted.append(cand)
    accepted.sort(key=lambda c: c.start)

    tables: list[Table] = []
    for cand in accepted:
        headers, rows = _apply_header(cand.grid, has_header)
        name = cand.name or f"Table {len(tables) + 1}"
        tables.append(Table(name=name, headers=headers, rows=rows,
                            caption=cand.caption, confidence=cand.confidence))
    return tables
