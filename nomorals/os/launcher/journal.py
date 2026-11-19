"""The update journal: every launcher attempt, one JSON line each.

An entry records the source, the package hash, the verification
result, the install result, any rollback, and health before and
after — so the owner can read what updated, when, and why, without
re-running anything. The journal never records package contents or
file bytes, only hashes and outcomes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

__all__ = ["Journal"]


class Journal:
    """Append-only JSONL journal at ``path``."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        """Append one entry (stamped with the current time) and return it."""
        entry = {"ts": time.time(), **event}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> list[dict[str, Any]]:
        """All entries, oldest first; a corrupt line is reported, not hidden."""
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"ts": 0.0, "event": "corrupt_line",
                            "line": line[:200]})
        return out
