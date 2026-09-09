#!/usr/bin/env python3
"""Count real lines of code, so the number in the README is reproducible.

Excludes blanks and whole-line comments. Does not try to strip trailing comments
or docstrings — that would need a real parser and the difference is small enough
that the simpler number is the more honest one.
"""

from __future__ import annotations

import sys
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "build", "dist", ".venv", "data", "backups", "models", "artifacts"}


def count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    by_area: dict[str, list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        relative = path.relative_to(root)
        area = relative.parts[0] if len(relative.parts) > 1 else "(root)"
        by_area.setdefault(area, []).append(count(path))

    total_files = sum(len(v) for v in by_area.values())
    total_lines = sum(sum(v) for v in by_area.values())
    width = max(len(k) for k in by_area)

    for area, counts in sorted(by_area.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {area:<{width}}  {len(counts):>3} files  {sum(counts):>6} lines")
    print(f"  {'':<{width}}  {'-' * 3}        {'-' * 6}")
    print(f"  {'TOTAL':<{width}}  {total_files:>3} files  {total_lines:>6} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
