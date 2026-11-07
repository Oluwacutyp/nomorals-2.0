"""The durable eval store: every run is journaled, runs are comparable.

File-backed by design — no DB migration: a run is one JSON document
(``runs/<run_id>.json``, the full per-case record) plus one summary row
appended to ``runs.jsonl`` (the index ``last``/``compare`` read). The
shape mirrors the router's own cost log (a JSONL journal under the
Devon home). Root: ``$NM_HOME/braineval`` when NM_HOME is set, else
``~/.nomorals/braineval``; tests and operators may pass any root.

A run record carries: run_id, started/finished stamps, pack_version,
rung + provider/model that actually answered, status
(complete | partial | failed), the aggregate scores, contract info
(template hash, listing source), and the per-case rows. A failed run is
stored exactly as proudly as a complete one — an unreachable provider
is a finding, and the record says why.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from ...core.ids import new_short_id
from ...core.logging_setup import get_logger

_log = get_logger(__name__)


def default_root() -> Path:
    home = os.environ.get("NM_HOME", "").strip()
    base = Path(home) if home else Path.home() / ".nomorals"
    return base / "braineval"


def new_run_id(rung: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    return f"{stamp}-{rung}-{new_short_id(length=6)}"


class EvalStore:
    """Append-only journal of eval runs under one root directory."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root is not None else default_root()
        self.runs_dir = self.root / "runs"
        self.index_path = self.root / "runs.jsonl"

    # ------------------------------------------------------------ writing
    def save_run(self, run: dict[str, Any]) -> str:
        """Persist one run record; returns its run_id."""
        run_id = run["run_id"]
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        detail_path = self.runs_dir / f"{run_id}.json"
        detail_path.write_text(
            json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
        summary = {k: v for k, v in run.items() if k != "cases"}
        summary["detail_file"] = detail_path.name
        with self.index_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary, ensure_ascii=False) + "\n")
        _log.info("braineval: run %s journaled (%s, %d cases)",
                  run_id, run.get("status", "?"), len(run.get("cases", [])))
        return run_id

    # ------------------------------------------------------------ reading
    def summaries(self) -> list[dict[str, Any]]:
        """All run summaries, oldest first. A torn final line (a crash
        mid-append) is skipped, never fatal."""
        if not self.index_path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                _log.warning("braineval: skipping torn runs.jsonl line")
                continue
        return out

    def get(self, run_id: str) -> dict[str, Any] | None:
        """One full run record by id or unique id prefix."""
        if not run_id:
            return None
        exact = self.runs_dir / f"{run_id}.json"
        if exact.exists():
            return json.loads(exact.read_text(encoding="utf-8"))
        matches = sorted(self.runs_dir.glob(f"{run_id}*.json"))
        if len(matches) == 1:
            return json.loads(matches[0].read_text(encoding="utf-8"))
        return None

    def last(self) -> dict[str, Any] | None:
        """The most recent full run record, or None if no run exists."""
        summaries = self.summaries()
        if not summaries:
            return None
        return self.get(summaries[-1]["run_id"])
