"""Durable JSON store for arena-researcher runs.

One file (``arena_researcher.json``) under the runtime home's ``data/``
directory, mirroring :class:`agents.swarm_board.SwarmBoardStore`:
atomic tmp+replace writes, quarantine on corrupt data, and an in-memory
fallback when the runtime has no home. A run record is a plain dict;
``last_run`` is what the status surface reads.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

#: Default store filename under ``<home>/data/``.
STORE_FILENAME = "arena_researcher.json"

#: Runs kept in the store (newest first on read).
MAX_RUNS = 50


def default_arena_researcher_path(home: str | Path | None) -> Path | None:
    """Default store path for *home*, or None when there is no home."""
    if not home:
        return None
    return Path(home) / "data" / STORE_FILENAME


class ArenaResearcherStore:
    """Atomic JSON store for arena-researcher run records."""

    def __init__(self, *, path: str | Path | None = None):
        self.path = Path(path) if path else None
        self._degraded_reason = ""
        self._memory: list[dict[str, Any]] = []
        self._loaded = False

    @property
    def degraded_reason(self) -> str:
        """Why the store degraded (corrupt file), or \"\" when healthy."""
        return self._degraded_reason

    def _load(self) -> list[dict[str, Any]]:
        if self._loaded:
            return self._memory
        self._loaded = True
        if self.path is None:
            return self._memory
        if not self.path.exists():
            return self._memory
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._degraded_reason = "corrupt store file quarantined"
            try:
                stamp = time.strftime("%Y%m%d%H%M%S", time.localtime())
                self.path.rename(
                    self.path.with_suffix(f".json.{stamp}.corrupt"))
            except OSError:
                pass
            return self._memory
        runs = raw.get("runs") if isinstance(raw, dict) else None
        if not isinstance(runs, list):
            return self._memory
        self._memory = [r for r in runs if isinstance(r, dict)]
        return self._memory

    def _save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps({"runs": self._memory[:MAX_RUNS]},
                           indent=2, sort_keys=True),
                encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            self._degraded_reason = "store write failed"

    def record_run(self, record: dict[str, Any]) -> str:
        """Append one run record; returns its id."""
        runs = self._load()
        rid = str(record.get("id") or f"ar-run-{int(time.time())}")
        record = dict(record)
        record["id"] = rid
        runs.insert(0, record)
        del runs[MAX_RUNS:]
        self._save()
        return rid

    def last_run(self) -> dict[str, Any] | None:
        """Newest run record, or None when no run is recorded."""
        runs = self._load()
        return runs[0] if runs else None

    def runs(self, limit: int = 10) -> list[dict[str, Any]]:
        """Newest *limit* run records."""
        return self._load()[:max(0, int(limit))]
