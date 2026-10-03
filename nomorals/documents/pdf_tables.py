"""Table recovery from plain extracted text (PDF pages, pasted reports).

The pure-Python PDF reader in :mod:`nomorals.core.pdf` returns text lines
with column spacing preserved.  :func:`extract_text_tables` finds runs of
consecutive lines that share a consistent whitespace-column layout and
turns each run into a :class:`~nomorals.documents.model.Table`.

Heuristics are deliberately conservative: a candidate block needs at
least ``min_rows`` consecutive lines that all split into the *same* number
of cells on wide (3+ spaces / tab) gaps, at least ``min_cols`` columns,
and no blank lines inside.  Anything ambiguous is left as prose —
a missed table is better than a hallucinated one.
"""

from __future__ import annotations

import re

from .model import Table

__all__ = ["extract_text_tables"]

_COLUMN_GAP_RE = re.compile(r"(?: {3,}|\t)+")


def _split_columns(line: str) -> list[str]:
    """Split a line on wide whitespace gaps into stripped cells."""
    return [cell.strip() for cell in _COLUMN_GAP_RE.split(line.strip())]


def _is_columnar(line: str, min_cols: int) -> bool:
    cells = _split_columns(line)
    return len(cells) >= min_cols and all(cells)


def extract_text_tables(text: str, *, min_rows: int = 3,
                        min_cols: int = 2) -> list[Table]:
    """Detect whitespace-aligned tables in plain text.

    Returns a list of :class:`Table` (possibly empty).  The first row of
    each detected block becomes the header row.  Never raises on odd
    input — it returns [] when nothing table-like is found.
    """
    if min_rows < 2:
        raise ValueError(f"min_rows must be >= 2, got {min_rows}")
    if min_cols < 2:
        raise ValueError(f"min_cols must be >= 2, got {min_cols}")
    lines = text.splitlines()
    tables: list[Table] = []
    i = 0
    while i < len(lines):
        if not _is_columnar(lines[i], min_cols):
            i += 1
            continue
        width = len(_split_columns(lines[i]))
        block: list[str] = []
        j = i
        while j < len(lines) and _is_columnar(lines[j], min_cols):
            if len(_split_columns(lines[j])) != width:
                break
            block.append(lines[j])
            j += 1
        if len(block) >= min_rows:
            grid = [_split_columns(line) for line in block]
            # All cells non-empty was checked by _is_columnar; guard anyway.
            if all(all(cell for cell in row) for row in grid):
                tables.append(Table(
                    name=f"Table {len(tables) + 1}",
                    headers=grid[0],
                    rows=grid[1:],
                ))
            i = j
        else:
            i += 1
    return tables
